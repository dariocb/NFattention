"""Causal GMM-RKS language model."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseCausalLM, build_dense_causal_mask


class GMMRKSFeatureMap(nn.Module):
    def __init__(self, n_dims: int, head_dim: int, orthogonal: bool = False):
        super().__init__()
        self.n_dims = n_dims
        self.head_dim = head_dim
        self.orthogonal = orthogonal
        self.register_buffer("mu", torch.randn(n_dims, head_dim))
        self.register_buffer("sigma", torch.ones(n_dims, head_dim))
        self.register_buffer("pi", torch.ones(n_dims) / n_dims)
        self.eps = 1e-6

    def forward(self, x):
        # x: [B,H,L,D]
        x_exp = x.unsqueeze(-2)
        mu = self.mu.view(1, 1, 1, self.n_dims, self.head_dim)
        sigma = self.sigma.view(1, 1, 1, self.n_dims, self.head_dim)
        logits = -0.5 * (((x_exp - mu) / sigma) ** 2).sum(dim=-1)
        logits = logits + torch.log(self.pi.clamp_min(self.eps)).view(1, 1, 1, self.n_dims)
        weights = torch.softmax(logits, dim=-1)
        proj = torch.matmul(weights, self.mu)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class GMMRKSAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, n_dims: int = 64, dropout: float = 0.1, orthogonal: bool = False):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_dims = n_dims
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.feature_map = GMMRKSFeatureMap(n_dims, self.head_dim, orthogonal=orthogonal)
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        phi_q = self.feature_map(q)
        phi_k = self.feature_map(k)
        energy = torch.matmul(phi_q, phi_k.transpose(-2, -1))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.fc_o(x), attention, torch.tensor(0.0, device=x.device)


class GMMRKSBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pf_dim: int, n_dims: int = 64, dropout: float = 0.1, orthogonal: bool = False):
        super().__init__()
        self.attn = GMMRKSAttention(hidden_dim, n_heads, n_dims, dropout, orthogonal=orthogonal)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, pf_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(pf_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y, attn, kl = self.attn(x, x, x, mask)
        x = self.ln1(x + self.dropout(y))
        y = self.ff(x)
        x = self.ln2(x + self.dropout(y))
        return x, attn, kl


class GMMRKSLM(BaseCausalLM):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 4, pf_dim: int = 512, n_dims: int = 64, dropout: float = 0.1, max_seq_len: int = 512, pad_idx: int = 0, orthogonal: bool = False, **kwargs):
        super().__init__(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, max_seq_len=max_seq_len, pad_idx=pad_idx, **kwargs)
        self.layers = nn.ModuleList([GMMRKSBlock(hidden_dim, n_heads, pf_dim, n_dims, dropout, orthogonal=orthogonal) for _ in range(n_layers)])

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(self.pos_encoder(self.embedding(input_ids)))
        mask = build_dense_causal_mask(attention_mask)
        if mask is None:
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        kl_total = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl = layer(x, mask)
            kl_total = kl_total + kl
        return x, kl_total


