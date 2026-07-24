"""Causal Performer language model."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .common import BaseCausalLM, build_dense_causal_mask


def gaussian_orthogonal_random_matrix(nb_rows: int, nb_columns: int, device=None):
    nb_full_blocks = nb_rows // nb_columns
    block_list = []
    for _ in range(nb_full_blocks):
        q = torch.randn(nb_columns, nb_columns, device=device)
        q, _ = torch.linalg.qr(q)
        block_list.append(q.T)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = torch.randn(nb_columns, nb_columns, device=device)
        q, _ = torch.linalg.qr(q)
        block_list.append(q.T[:remaining_rows])
    final_matrix = torch.cat(block_list, dim=0)
    multiplier = torch.randn(nb_rows, nb_columns, device=device).norm(dim=1)
    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(data, projection_matrix, is_query, eps=1e-4):
    data_normalizer = data.shape[-1] ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5
    data_dash = torch.einsum("bhnd,md->bhnm", data_normalizer * data, projection_matrix)
    diag_data = (data ** 2).sum(dim=-1, keepdim=True) * (data_normalizer ** 2) / 2.0
    if is_query:
        max_val = data_dash.max(dim=-1, keepdim=True)[0].detach()
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + eps)
    else:
        max_val = data_dash.amax(dim=(-1, -2), keepdim=True).detach()
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + eps)
    return data_dash


class PerformerAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, n_features: int = None, dropout: float = 0.1, redraw_interval: int = 1000):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_features = n_features or int(self.head_dim * math.log(self.head_dim))
        self.redraw_interval = redraw_interval
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("projection_matrix", gaussian_orthogonal_random_matrix(self.n_features, self.head_dim))
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.projection_matrix.copy_(gaussian_orthogonal_random_matrix(self.n_features, self.head_dim, device=self.projection_matrix.device))

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        if self.training:
            self.calls_since_last_redraw += 1
            if self.calls_since_last_redraw >= self.redraw_interval:
                self.redraw_projection_matrix()
                self.calls_since_last_redraw.zero_()
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q_prime = softmax_kernel(q, self.projection_matrix, True)
        k_prime = softmax_kernel(k, self.projection_matrix, False)
        attention = torch.matmul(q_prime, k_prime.transpose(-2, -1))
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.fc_o(x), attention, torch.tensor(0.0, device=x.device)


class PerformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pf_dim: int, n_features: int = None, dropout: float = 0.1):
        super().__init__()
        self.attn = PerformerAttention(hidden_dim, n_heads, n_features, dropout)
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


class PerformerLM(BaseCausalLM):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 4, pf_dim: int = 512, n_features: int = 64, dropout: float = 0.1, max_seq_len: int = 512, pad_idx: int = 0, **kwargs):
        super().__init__(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, max_seq_len=max_seq_len, pad_idx=pad_idx, **kwargs)
        self.layers = nn.ModuleList([PerformerBlock(hidden_dim, n_heads, pf_dim, n_features, dropout) for _ in range(n_layers)])

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


