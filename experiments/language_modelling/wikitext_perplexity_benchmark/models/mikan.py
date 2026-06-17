"""Causal MIKAN language model."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseCausalLM, build_dense_causal_mask


def ika_kernel(Q, K, w1, w2, scale, M):
    if len(w1.size()) == 3:
        w1_t = w1.permute(1, 0, 2)
        w2_t = w2.permute(1, 0, 2)
        phi_q1 = torch.einsum("bhnd,hdm->bhnm", Q, w1_t * scale)
        phi_q2 = torch.einsum("bhnd,hdm->bhnm", Q, w2_t * scale)
        phi_k1 = torch.einsum("bhnd,hdm->bhnm", K, w1_t * scale)
        phi_k2 = torch.einsum("bhnd,hdm->bhnm", K, w2_t * scale)
    else:
        phi_q1 = torch.einsum("bhnd,bhnm->bhnm", Q, w1 * scale)
        phi_q2 = torch.einsum("bhnd,bhnm->bhnm", Q, w2 * scale)
        phi_k1 = torch.einsum("bhnd,bhnm->bhnm", K, w1 * scale)
        phi_k2 = torch.einsum("bhnd,bhnm->bhnm", K, w2 * scale)

    phi_q = torch.cat([torch.cos(phi_q1) + torch.cos(phi_q2), torch.sin(phi_q1) + torch.sin(phi_q2)], dim=-1)
    phi_k = torch.cat([torch.cos(phi_k1) + torch.cos(phi_k2), torch.sin(phi_k1) + torch.sin(phi_k2)], dim=-1)
    scores = torch.matmul(phi_q, phi_k.transpose(-2, -1))
    scores = scores / (4.0 * M)
    scores = scores * scores
    d_k = Q.size(-1)
    norm = torch.pow(torch.norm(Q, dim=-1, keepdim=True, p=2), 2) + torch.pow(torch.norm(K, dim=-1, keepdim=True, p=2), 2).transpose(-2, -1)
    norm = norm / (2 * math.sqrt(d_k))
    return scores, norm


class MIKANAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, M: int = 64, dropout: float = 0.1, freeze_qk: bool = False):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.M = M
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.w1 = nn.Parameter(torch.empty(self.head_dim, n_heads, M))
        self.w2 = nn.Parameter(torch.empty(self.head_dim, n_heads, M))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        if freeze_qk:
            nn.init.orthogonal_(self.fc_q.weight)
            nn.init.orthogonal_(self.fc_k.weight)
            self.fc_q.weight.requires_grad = False
            self.fc_q.bias.requires_grad = False
            self.fc_k.weight.requires_grad = False
            self.fc_k.bias.requires_grad = False

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        scores, norm = ika_kernel(q, k, self.w1, self.w2, 2 * np.pi, self.M)
        energy = torch.log(scores + 1e-5) + norm
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.fc_o(x), attention, torch.tensor(0.0, device=x.device)


class MIKANBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pf_dim: int, M: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn = MIKANAttention(hidden_dim, n_heads, M, dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y, attn, kl = self.attn(x, x, x, mask)
        x = self.ln1(x + self.dropout(y))
        y = self.ff(x)
        x = self.ln2(x + self.dropout(y))
        return x, attn, kl


class MIKANLM(BaseCausalLM):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 256, n_heads: int = 8, n_layers: int = 4, pf_dim: int = 512, M: int = 64, dropout: float = 0.1, max_seq_len: int = 512, pad_idx: int = 0, **kwargs):
        super().__init__(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, max_seq_len=max_seq_len, pad_idx=pad_idx, **kwargs)
        self.layers = nn.ModuleList([MIKANBlock(hidden_dim, n_heads, pf_dim, M, dropout) for _ in range(n_layers)])

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


