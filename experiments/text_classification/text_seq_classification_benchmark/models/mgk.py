"""
MGK (Mixture of Gaussian Keys) classifier.

Implements key-conditioned Gaussian-mixture attention logits as described in:
Nguyen et al., "Improving Transformers with Probabilistic Attention Keys".
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseClassifier, PositionwiseFeedforward


class MGKAttention(nn.Module):
    """Multi-head attention using Mixture-of-Gaussian Keys logits."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        mgk_components: int = 4,
        mgk_proj_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.mgk_components = max(1, int(mgk_components))
        self.mgk_proj_dim = int(mgk_proj_dim) if mgk_proj_dim is not None else max(16, hidden_dim // 4)

        self.log2pi = math.log(2.0 * math.pi)

        # MGK logits projections.
        self.q_proj = nn.Linear(hidden_dim, self.n_heads * self.mgk_proj_dim)
        self.k_mu_proj = nn.Linear(hidden_dim, self.n_heads * self.mgk_components * self.mgk_proj_dim)
        self.k_logvar_proj = nn.Linear(hidden_dim, self.n_heads * self.mgk_components * self.mgk_proj_dim)
        self.k_mix_proj = nn.Linear(hidden_dim, self.n_heads * self.mgk_components)

        # Standard value/output projections to keep Transformer layer contract.
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def _mgk_logits(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        bsz, lq, _ = query.shape
        _, lk, _ = key.shape

        # qz: (B, Lq, H, Dp)
        qz = self.q_proj(query).view(bsz, lq, self.n_heads, self.mgk_proj_dim)

        # key-conditioned GMM parameters.
        # mu/logvar: (B, Lk, H, M, Dp), mix_logits: (B, Lk, H, M)
        mu = self.k_mu_proj(key).view(
            bsz,
            lk,
            self.n_heads,
            self.mgk_components,
            self.mgk_proj_dim,
        )
        logvar = self.k_logvar_proj(key).view(
            bsz,
            lk,
            self.n_heads,
            self.mgk_components,
            self.mgk_proj_dim,
        ).clamp(min=-8.0, max=6.0)
        mix_logits = self.k_mix_proj(key).view(
            bsz,
            lk,
            self.n_heads,
            self.mgk_components,
        )
        log_pi = F.log_softmax(mix_logits, dim=-1)

        # Broadcast to (B, Lq, Lk, H, M, Dp)
        qz_e = qz.unsqueeze(2).unsqueeze(4)
        mu_e = mu.unsqueeze(1)
        logvar_e = logvar.unsqueeze(1)

        diff = qz_e - mu_e
        inv_var = torch.exp(-logvar_e)
        quad = (diff * diff * inv_var).sum(dim=-1)
        log_det = logvar_e.sum(dim=-1)

        # Gaussian log-likelihood per component: (B, Lq, Lk, H, M)
        ll = -0.5 * (quad + log_det + self.mgk_proj_dim * self.log2pi)

        # Marginalize components -> logits per key: (B, H, Lq, Lk)
        logits = torch.logsumexp(log_pi.unsqueeze(1) + ll, dim=-1)
        return logits.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_len, _ = query.shape
        value_len = value.shape[1]

        energy = self._mgk_logits(query, key)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        V = self.fc_v(value).view(batch_size, value_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.hidden_dim)
        x = self.fc_o(x)

        kl_div = torch.tensor(0.0, device=x.device)
        return x, attention, kl_div


class MGKEncoderLayer(nn.Module):
    """MGK encoder layer."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        mgk_components: int = 4,
        mgk_proj_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.self_attention = MGKAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            mgk_components=mgk_components,
            mgk_proj_dim=mgk_proj_dim,
            dropout=dropout,
            device=device,
        )
        self.feedforward = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _src, attention, kl_div = self.self_attention(src, src, src, mask)
        src = self.ln1(src + self.dropout(_src))

        _src = self.feedforward(src)
        src = self.ln2(src + self.dropout(_src))

        return src, attention, kl_div


class MGKClassifier(BaseClassifier):
    """Text classifier using MGK attention."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        n_dims: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
        device: str = "cpu",
        mgk_components: Optional[int] = None,
        mgk_proj_dim: Optional[int] = None,
        num_mixtures: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            pf_dim=pf_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            device=device,
            **kwargs,
        )

        # Keep compatibility with existing configs:
        # - num_mixtures from existing model_params can drive components.
        # - n_dims can drive projection dim unless explicitly set.
        components = mgk_components if mgk_components is not None else (num_mixtures if num_mixtures is not None else 4)
        proj_dim = mgk_proj_dim if mgk_proj_dim is not None else max(16, hidden_dim // 4)

        self.layers = nn.ModuleList(
            [
                MGKEncoderLayer(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    pf_dim=pf_dim,
                    mgk_components=components,
                    mgk_proj_dim=proj_dim,
                    dropout=dropout,
                    device=device,
                )
                for _ in range(n_layers)
            ]
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.input_projection(x)

        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        total_kl_div = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl_div = layer(x, mask)
            total_kl_div = total_kl_div + kl_div

        return x, total_kl_div
