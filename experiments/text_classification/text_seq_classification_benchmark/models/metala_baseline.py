"""
MetaLA 风格编码器分类器（因果 token mixer + 标准 FFN）

结构对齐 BICLab/MetaLA 的 QVMetaLA_self_aug（NeurIPS 2024 Oral）：
https://github.com/BICLab/MetaLA/blob/main/metala/modeling_metala.py

- depthwise 1D 卷积（非因果、same padding，便于双向语境下的序列分类）
- q 投影到 dk、门控 k_gate → gk，k = 1 - exp(gk)
- GLA 核：使用与 fla naive_recurrent_simple_gla 相同的递推（纯 PyTorch，免 CUDA/Triton 依赖）
- self-augmentation 与 silu(g)·output 与官方一致

说明：完整训练管线依赖 GPT-NeoX + flash-linear-attention；此处为文本分类 benchmark 的轻量可运行复现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


def _naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    q,k,g: [B, L, H, K], v: [B, L, H, V]
    与 fla-org/flash-linear-attention 中 naive_recurrent_simple_gla 一致（输入 layout 为 B T H D）。
    """
    dtype = q.dtype
    q, k, v, g = (x.transpose(1, 2).float() for x in (q, k, v, g))
    b, h, t, kdim = q.shape
    vdim = v.shape[-1]
    if scale is None:
        scale = kdim ** -0.5
    q = q * scale
    o = v.new_zeros(b, h, t, vdim)
    s = q.new_zeros(b, h, kdim, vdim)
    for i in range(t):
        gate = g[:, :, i].exp()
        ki = k[:, :, i]
        vi = v[:, :, i]
        kv = ki.unsqueeze(-1) * vi.unsqueeze(-2)
        s = s * gate.unsqueeze(-1).unsqueeze(-1) + kv
        qi = q[:, :, i, :]
        o[:, :, i] = (qi.unsqueeze(-1) * s).sum(-2)
    return o.transpose(1, 2).to(dtype)


class MetaLATokenMixer(nn.Module):
    """QVMetaLA_self_aug 的编码器化简版（无 KV cache）。"""

    def __init__(self, embed_dim: int, num_heads: int, d_conv: int = 3):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        half = embed_dim // 2
        if half % num_heads != 0:
            dk = embed_dim
        else:
            dk = half

        self.dk = dk
        self.head_dim = embed_dim // num_heads
        self.key_dim = dk // num_heads

        self.q_proj = nn.Linear(embed_dim, dk, bias=False)
        self.k_gate = nn.Linear(embed_dim, dk, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.aug_balance = nn.Parameter(torch.zeros(dk))
        self.group_norm = nn.LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False)

        # 奇数核 + padding=(k-1)/2 保持序列长度（官方为因果卷积；此处为 same 近似）
        assert d_conv % 2 == 1, "d_conv 应为奇数以便保持长度"
        pad = (d_conv - 1) // 2
        self.conv1d = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=d_conv, groups=embed_dim, padding=pad, bias=False,
        )
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        x = x.transpose(1, 2)
        x = F.silu(self.conv1d(x))
        x = x.transpose(1, 2)

        q = self.q_proj(x)
        k_gate_in = self.k_gate(x)
        v = self.v_proj(x)
        g_out = self.g_proj(x)

        gk = F.logsigmoid(k_gate_in) / 16.0
        k_coeff = 1.0 - torch.exp(gk)

        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            q = q * m
            k_coeff = k_coeff * m
            v = v * m
            gk = gk * m

        q_h = q.view(q.shape[0], q.shape[1], self.num_heads, self.key_dim).transpose(1, 2)
        k_h = k_coeff.view(k_coeff.shape[0], k_coeff.shape[1], self.num_heads, self.key_dim).transpose(1, 2)
        v_h = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        g_h = gk.view(gk.shape[0], gk.shape[1], self.num_heads, self.key_dim).transpose(1, 2)

        o = _naive_recurrent_gla(q_h, k_h, v_h, g_h)
        aug_b = self.aug_balance.view(self.num_heads, self.key_dim).to(o.dtype)
        augk = k_h * aug_b.unsqueeze(0).unsqueeze(2)
        aug_w = (q_h * augk).sum(-1)
        o = o + torch.sigmoid(aug_w.unsqueeze(-1)) * v_h

        o = self.group_norm(o)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.embed_dim)
        o = F.silu(g_out) * o
        return self.out_proj(o)


class MetaLAEncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, pf_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.mixer = MetaLATokenMixer(hidden_dim, n_heads)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        y = self.mixer(self.token_norm(x), mask)
        x = x + self.dropout(y)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class MetaLABaselineClassifier(BaseClassifier):
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
            [MetaLAEncoderLayer(hidden_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x, torch.tensor(0.0, device=x.device)
