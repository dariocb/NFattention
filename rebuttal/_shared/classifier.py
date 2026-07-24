"""Common encoder/classifier used by all rebuttal models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .baselines import build_attention
from .fska import AttentionResult, FSKAConfig, FSKAAttention


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    vocab_size: int
    num_classes: int
    max_length: int
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    feature_width: int = 128
    num_spectral_pairs: int = 64
    density_mode: str = "learned_flow"
    feature_map: str = "elu_plus_one"
    qk_mode: str = "identity"
    kl_weight: float = 1e-3
    pooling: str = "mean"
    pad_idx: int = 0
    sample_seed: int = 1729
    gradient_checkpointing: bool = False


@dataclass
class ModelOutput:
    logits: Tensor
    kl_raw: Tensor
    diagnostics: Dict[str, Tensor]


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(max_length, hidden_dim)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, hidden_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, mask: Tensor, return_diagnostics: bool = False
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        result: AttentionResult = self.attention(
            self.norm1(x), mask, return_diagnostics=return_diagnostics
        )
        x = x + self.dropout(result.output)
        x = x + self.dropout(self.ff(self.norm2(x)))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x, result.kl_raw, result.diagnostics


class SequenceClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        extra = 1 if config.pooling == "cls" else 0
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_dim, padding_idx=config.pad_idx
        )
        self.position = SinusoidalPositionEncoding(
            config.hidden_dim, config.max_length + extra
        )
        self.input_dropout = nn.Dropout(config.dropout)
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        else:
            self.cls_token = None

        layers = []
        for layer_index in range(config.n_layers):
            seed = config.sample_seed + layer_index * 10007
            if config.model_name == "fska":
                attention = FSKAAttention(
                    FSKAConfig(
                        hidden_dim=config.hidden_dim,
                        n_heads=config.n_heads,
                        num_spectral_pairs=config.num_spectral_pairs,
                        density_mode=config.density_mode,
                        feature_map=config.feature_map,
                        qk_mode=config.qk_mode,
                        kl_weight=config.kl_weight,
                        dropout=config.dropout,
                        sample_seed=seed,
                    )
                )
            else:
                attention = build_attention(
                    config.model_name,
                    config.hidden_dim,
                    config.n_heads,
                    config.feature_width,
                    config.dropout,
                    seed,
                )
            layers.append(
                EncoderLayer(attention, config.hidden_dim, config.ff_dim, config.dropout)
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    @property
    def kl_weight(self) -> float:
        return self.config.kl_weight if self.config.model_name == "fska" else 0.0

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor,
        return_diagnostics: bool = False
    ) -> ModelOutput:
        mask = attention_mask.bool()
        x = self.embedding(input_ids)
        if self.cls_token is not None:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
            mask = torch.cat(
                [
                    torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device),
                    mask,
                ],
                dim=1,
            )
        x = self.input_dropout(self.position(x))
        x = x * mask.unsqueeze(-1).to(x.dtype)

        kl_raw = x.new_zeros(())
        diagnostics: Dict[str, Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            if self.config.gradient_checkpointing and self.training and not return_diagnostics:
                def layer_fn(
                    states: Tensor, layer_mask: Tensor, current_layer=layer
                ):
                    next_states, layer_kl, _ = current_layer(
                        states, layer_mask, False
                    )
                    return next_states, layer_kl
                x, layer_kl = checkpoint(layer_fn, x, mask)
                layer_diagnostics = {}
            else:
                x, layer_kl, layer_diagnostics = layer(
                    x, mask, return_diagnostics
                )
            kl_raw = kl_raw + layer_kl
            for key, value in layer_diagnostics.items():
                diagnostics[f"layer_{layer_index}.{key}"] = value

        x = self.final_norm(x)
        if self.cls_token is not None:
            pooled = x[:, 0]
        else:
            weights = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return ModelOutput(self.classifier(pooled), kl_raw, diagnostics)

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad or not trainable_only
        )

    def set_sampling_step(self, step: int) -> None:
        for layer in self.layers:
            if hasattr(layer.attention, "density"):
                layer.attention.density.set_training_step(step)


def build_model(config: ModelConfig) -> SequenceClassifier:
    return SequenceClassifier(config)
