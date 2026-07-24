"""Shared helpers for causal language models."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def build_dense_causal_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.float32))
    causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
    key_mask = attention_mask[:, None, None, :].float()
    query_mask = attention_mask[:, None, :, None].float()
    return causal * key_mask * query_mask


class BaseCausalLM(nn.Module, ABC):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
        tie_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        self.input_projection = nn.Linear(embed_dim, hidden_dim) if embed_dim != hidden_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_embeddings and embed_dim == hidden_dim:
            self.lm_head.weight = self.embedding.weight

    @abstractmethod
    def encode(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, kl_div = self.encode(input_ids, attention_mask)
        logits = self.lm_head(encoded)
        return logits, kl_div


