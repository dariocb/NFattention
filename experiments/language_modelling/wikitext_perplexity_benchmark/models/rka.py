"""Causal RKA language model."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseCausalLM, build_dense_causal_mask


class GeneratorLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, batch_norm: bool = True, activation: str = "leaky"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.activation = {"relu": F.relu, "leaky": F.leaky_relu, "gelu": F.gelu, "tanh": torch.tanh, "linear": lambda x: x}[activation]
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.linear(x)
        if len(x.shape) == 3:
            orig_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            x = self.batch_norm(x)
            x = x.reshape(orig_shape)
        else:
            x = self.batch_norm(x)
        return self.activation(x)


class GeneratorBlock(nn.Module):
    def __init__(self, layer_dims: list, output_dim: int, hidden_act: str = "leaky", output_act: str = "tanh"):
        super().__init__()
        layers = []
        for in_size, out_size in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(GeneratorLayer(in_size, out_size, activation=hidden_act))
        layers.append(GeneratorLayer(layer_dims[-1], output_dim, batch_norm=False, activation=output_act))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RKAAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, n_features: int = 64, noise_dims: list = None, dropout: float = 0.1, redraw_interval: int = 1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_features = n_features
        self.redraw_interval = redraw_interval
        self.noise_dims = noise_dims or [self.head_dim, 2 * self.head_dim, self.head_dim]
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.generator = GeneratorBlock(self.noise_dims, output_dim=self.head_dim, hidden_act="leaky", output_act="tanh")
        self.dropout = nn.Dropout(dropout)
        self.softmax_temp = 1.0 / math.sqrt(self.head_dim)
        self.register_buffer("omega", torch.zeros(n_features // 2, self.noise_dims[0]))
        self.register_buffer("_calls", torch.tensor(-1, dtype=torch.int))

    def new_feature_map(self):
        self._calls += 1
        if (self._calls % self.redraw_interval) != 0 and not self.training:
            return
        omega = torch.randn(self.n_features // 2, self.noise_dims[0], dtype=self.omega.dtype, device=self.omega.device)
        self.omega.copy_(omega)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        self.new_feature_map()
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        omega = self.generator(self.omega)
        q_scaled = q * math.sqrt(self.softmax_temp)
        k_scaled = k * math.sqrt(self.softmax_temp)
        q_proj = torch.einsum("bhnd,md->bhnm", q_scaled, omega)
        k_proj = torch.einsum("bhnd,md->bhnm", k_scaled, omega)
        phi_q = torch.cat([q_proj.cos(), q_proj.sin()], dim=-1) * math.sqrt(2.0 / self.n_features)
        phi_k = torch.cat([k_proj.cos(), k_proj.sin()], dim=-1) * math.sqrt(2.0 / self.n_features)
        energy = torch.matmul(phi_q, phi_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        attention = torch.nan_to_num(attention, nan=0.0)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.fc_o(x), attention, torch.tensor(0.0, device=x.device)


class RKABlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pf_dim: int, n_features: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn = RKAAttention(hidden_dim, n_heads, n_features, dropout=dropout)
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


class RKALM(BaseCausalLM):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 4, pf_dim: int = 512, n_features: int = 64, dropout: float = 0.1, max_seq_len: int = 512, pad_idx: int = 0, **kwargs):
        super().__init__(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, max_seq_len=max_seq_len, pad_idx=pad_idx, **kwargs)
        self.layers = nn.ModuleList([RKABlock(hidden_dim, n_heads, pf_dim, n_features, dropout) for _ in range(n_layers)])

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


