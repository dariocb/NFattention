"""Shared rebuttal-local attention baselines with correct padding semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .fska import AttentionResult


def _split_heads(x: Tensor, n_heads: int) -> Tensor:
    batch, length, hidden = x.shape
    head_dim = hidden // n_heads
    return x.view(batch, length, n_heads, head_dim).permute(0, 2, 1, 3)


def _merge_heads(x: Tensor) -> Tensor:
    batch, heads, length, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(
        batch, length, heads * head_dim
    )


def _linear_context(phi_q: Tensor, phi_k: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    key_mask = mask[:, None, :, None].to(phi_k.dtype)
    phi_k = phi_k * key_mask
    values = values * key_mask
    kv = torch.einsum("bhlm,bhld->bhmd", phi_k, values)
    ks = phi_k.sum(dim=2)
    numerator = torch.einsum("bhlm,bhmd->bhld", phi_q, kv)
    denominator = torch.einsum("bhlm,bhm->bhl", phi_q, ks).clamp_min(1e-6)
    return numerator / denominator.unsqueeze(-1)


class TransformerAttention(nn.Module):
    def __init__(
        self, hidden_dim: int, n_heads: int, dropout: float = 0.1,
        attention_bias: bool = True, **_: object
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
            bias=attention_bias,
        )

    def forward(
        self, hidden_states: Tensor, padding_mask: Optional[Tensor] = None,
        return_diagnostics: bool = False
    ) -> AttentionResult:
        if padding_mask is None:
            padding_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        output, _ = self.attention(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~padding_mask.bool(),
            need_weights=False,
        )
        output = output * padding_mask.unsqueeze(-1).to(output.dtype)
        return AttentionResult(output, output.new_zeros(()), {})


class PerformerAttention(nn.Module):
    """Positive orthogonal random-feature softmax approximation."""

    def __init__(
        self, hidden_dim: int, n_heads: int, feature_width: int = 128,
        dropout: float = 0.1, sample_seed: int = 1234, **_: object
    ):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.feature_width = feature_width
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed)
        projection = torch.randn(
            n_heads, feature_width, self.head_dim, generator=generator
        )
        projection = F.normalize(projection, dim=-1)
        self.register_buffer("projection", projection)
        self.dropout = nn.Dropout(dropout)

    def _features(self, x: Tensor) -> Tensor:
        x = x / math.sqrt(math.sqrt(self.head_dim))
        projected = torch.einsum("bhld,hmd->bhlm", x, self.projection)
        squared_norm = 0.5 * x.square().sum(dim=-1, keepdim=True)
        projected = projected - squared_norm
        projected = projected - projected.amax(dim=-1, keepdim=True).detach()
        return projected.exp() / math.sqrt(self.feature_width)

    def forward(
        self, hidden_states: Tensor, padding_mask: Optional[Tensor] = None,
        return_diagnostics: bool = False
    ) -> AttentionResult:
        batch, length, _ = hidden_states.shape
        if padding_mask is None:
            padding_mask = torch.ones(
                batch, length, dtype=torch.bool, device=hidden_states.device
            )
        q = _split_heads(self.q_proj(hidden_states), self.n_heads)
        k = _split_heads(self.k_proj(hidden_states), self.n_heads)
        v = _split_heads(self.v_proj(hidden_states), self.n_heads)
        context = _linear_context(self._features(q), self._features(k), v, padding_mask)
        output = self.out_proj(_merge_heads(self.dropout(context)))
        output = output * padding_mask.unsqueeze(-1).to(output.dtype)
        return AttentionResult(output, output.new_zeros(()), {})


class RKAAttention(nn.Module):
    """Learned RFF generator with a genuinely linear attention contraction."""

    def __init__(
        self, hidden_dim: int, n_heads: int, feature_width: int = 128,
        dropout: float = 0.1, sample_seed: int = 5678, **_: object
    ):
        super().__init__()
        if feature_width % 2:
            raise ValueError("RKA feature_width must be even")
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.feature_width = feature_width
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed)
        noise = torch.randn(feature_width // 2, self.head_dim, generator=generator)
        self.register_buffer("noise", noise)
        self.generator = nn.Sequential(
            nn.Linear(self.head_dim, 2 * self.head_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(2 * self.head_dim, self.head_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(dropout)

    def _features(self, x: Tensor) -> Tensor:
        omega = self.generator(self.noise)
        projected = torch.einsum("bhld,md->bhlm", x, omega)
        rff = math.sqrt(2.0 / self.feature_width) * torch.cat(
            [projected.cos(), projected.sin()], dim=-1
        )
        return F.elu(rff) + 1.0

    def forward(
        self, hidden_states: Tensor, padding_mask: Optional[Tensor] = None,
        return_diagnostics: bool = False
    ) -> AttentionResult:
        batch, length, _ = hidden_states.shape
        if padding_mask is None:
            padding_mask = torch.ones(
                batch, length, dtype=torch.bool, device=hidden_states.device
            )
        q = _split_heads(self.q_proj(hidden_states), self.n_heads)
        k = _split_heads(self.k_proj(hidden_states), self.n_heads)
        v = _split_heads(self.v_proj(hidden_states), self.n_heads)
        context = _linear_context(self._features(q), self._features(k), v, padding_mask)
        output = self.out_proj(_merge_heads(self.dropout(context)))
        output = output * padding_mask.unsqueeze(-1).to(output.dtype)
        return AttentionResult(output, output.new_zeros(()), {})


def build_attention(
    model_name: str,
    hidden_dim: int,
    n_heads: int,
    feature_width: int,
    dropout: float,
    sample_seed: int,
    attention_bias: bool = True,
) -> nn.Module:
    common = dict(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        feature_width=feature_width,
        dropout=dropout,
        sample_seed=sample_seed,
        attention_bias=attention_bias,
    )
    if model_name == "transformer":
        return TransformerAttention(**common)
    if model_name == "performer":
        return PerformerAttention(**common)
    if model_name == "rka":
        return RKAAttention(**common)
    raise ValueError(f"Unknown baseline: {model_name}")
