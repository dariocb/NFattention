"""
KPCA / Scaled-Attention 风格分类器

复现自 Teo & Nguyen (NeurIPS 2024) 官方代码仓库中的 scaled self-attention 变体：
https://github.com/rachtsy/KPCA_code/blob/master/Scaled_Attention/softmax.py

在标准 softmax(QK^T) 注意力下，先用 (I - s·softmax(KK^T)) 对 V 做一度变换（s 为可标量），
再与注意力权重相乘。与 ViT 原版相比增加了对 value 的 K-gram / 对称核调制，便于作为同一
benchmark 下的对照。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


class KPCAScaledMultiHeadAttention(nn.Module):
    """Scaled_Attention/softmax.py 中 Attention 的多头、带 padding mask 版本（s_scalar=True）。"""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.s = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, Hid]
            mask: [B, 1, 1, L]，1 表示保留（与 TransformerClassifier 一致）
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        sym_logits = (k @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            sym_logits = sym_logits.masked_fill(mask == 0, float("-inf"))
        sym_attn = sym_logits.softmax(dim=-1)

        eye = torch.eye(n, device=x.device, dtype=x.dtype).view(1, 1, n, n)
        # 与官方 (I - sym_attn * s) @ v 等价（s 为标量）
        v_mod = torch.matmul(eye - sym_attn * self.s, v)
        out = torch.matmul(attn, v_mod)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj_drop(self.proj(out))
        return out, attn


class KPCAScaledEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pf_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = KPCAScaledMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.ff = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None):
        y, attn = self.self_attn(self.ln1(src), mask)
        src = src + self.dropout(y)
        src = src + self.dropout(self.ff(self.ln2(src)))
        return src, attn


class KPCAScaledClassifier(BaseClassifier):
    """KPCA 仓库 Scaled Attention ViT-block 结构的文本编码 + 池化分类。"""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
        device: str = "cpu",
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
        self.layers = nn.ModuleList(
            [
                KPCAScaledEncoderLayer(hidden_dim, n_heads, pf_dim, dropout)
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

        for layer in self.layers:
            x, _ = layer(x, mask)

        return x, torch.tensor(0.0, device=x.device)
