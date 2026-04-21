"""
MIKAN (Multi-head Implicit Kernel Attention Network) 分类器
基于论文中的 IKAN-direct 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torch.distributions.normal import Normal
import numpy as np
import math
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


def ika_kernel(
    Q: torch.Tensor,
    K: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    scale: float,
    M: int,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implicit Kernel Attention 核计算
    
    Args:
        Q: [batch, n_heads, seq_len, head_dim]
        K: [batch, n_heads, seq_len, head_dim]
        w1, w2: [head_dim, n_heads, M] - 共享的频率权重
        scale: 缩放因子 (2π)
        M: 随机特征数量
        training: 是否训练模式
        
    Returns:
        scores: [batch, n_heads, seq_len, seq_len]
        norm: [batch, n_heads, seq_len, seq_len]
    """
    # 计算 Random Fourier Features
    if len(w1.size()) == 3:
        # w1: [head_dim, n_heads, M]
        # Q: [batch, n_heads, seq_len, head_dim]
        # 需要: [batch, n_heads, seq_len, M]
        # 重塑 w1 为 [n_heads, head_dim, M] 以便计算
        w1_t = w1.permute(1, 0, 2)  # [n_heads, head_dim, M]
        w2_t = w2.permute(1, 0, 2)  # [n_heads, head_dim, M]
        
        # [batch, n_heads, seq_len, head_dim] @ [n_heads, head_dim, M] -> [batch, n_heads, seq_len, M]
        phi_q1 = torch.einsum('bhnd,hdm->bhnm', Q, w1_t * scale)
        phi_q2 = torch.einsum('bhnd,hdm->bhnm', Q, w2_t * scale)
        phi_k1 = torch.einsum('bhnd,hdm->bhnm', K, w1_t * scale)
        phi_k2 = torch.einsum('bhnd,hdm->bhnm', K, w2_t * scale)
    else:
        phi_q1 = torch.einsum('bhnd,bhnm->bhnm', Q, w1 * scale)
        phi_q2 = torch.einsum('bhnd,bhnm->bhnm', Q, w2 * scale)
        phi_k1 = torch.einsum('bhnd,bhnm->bhnm', K, w1 * scale)
        phi_k2 = torch.einsum('bhnd,bhnm->bhnm', K, w2 * scale)
    
    # RFF 特征
    phi_q = torch.cat([
        torch.cos(phi_q1) + torch.cos(phi_q2),
        torch.sin(phi_q1) + torch.sin(phi_q2)
    ], dim=-1)  # [batch, n_heads, seq_len, 2M]
    
    phi_k = torch.cat([
        torch.cos(phi_k1) + torch.cos(phi_k2),
        torch.sin(phi_k1) + torch.sin(phi_k2)
    ], dim=-1)  # [batch, n_heads, seq_len, 2M]
    
    # 计算核矩阵
    scores = torch.matmul(phi_q, phi_k.transpose(-2, -1))  # [batch, n_heads, seq_len, seq_len]
    scores = scores / (4.0 * M)
    scores = scores * scores  # 平方
    
    # 计算 norm 项
    d_k = Q.size(-1)
    norm = torch.pow(torch.norm(Q, dim=-1, keepdim=True, p=2), 2) + \
           torch.pow(torch.norm(K, dim=-1, keepdim=True, p=2), 2).transpose(-2, -1)
    norm = norm / (2 * math.sqrt(d_k))
    
    if training:
        norm = F.dropout(norm, p=0.1, training=training)
    
    return scores, norm


class MIKANMultiHeadAttention(nn.Module):
    """MIKAN 多头注意力层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        M: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu',
        freeze_qk: bool = False
    ):
        super().__init__()
        
        assert hidden_dim % n_heads == 0, "hidden_dim 必须能被 n_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.M = M
        self.device = device
        
        # Q, K, V 投影
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        # 共享的频率权重 w1, w2
        self.w1 = nn.Parameter(torch.empty(self.head_dim, n_heads, M))
        self.w2 = nn.Parameter(torch.empty(self.head_dim, n_heads, M))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 如果 freeze_qk=True，冻结 Q 和 K 的投影矩阵
        if freeze_qk:
            self._freeze_qk_parameters()
            
    def _freeze_qk_parameters(self):
        """冻结 Q 和 K 的投影矩阵参数"""
        nn.init.orthogonal_(self.fc_q.weight)
        nn.init.orthogonal_(self.fc_k.weight)
        
        self.fc_q.weight.requires_grad = False
        self.fc_q.bias.requires_grad = False
        self.fc_k.weight.requires_grad = False
        self.fc_k.bias.requires_grad = False
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, hidden_dim]
            key: [batch_size, seq_len, hidden_dim]
            value: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, 1, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention: [batch_size, n_heads, seq_len, seq_len]
            kl_div: KL 散度
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 线性投影
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算 IKA 核
        scores, norm = ika_kernel(Q, K, self.w1, self.w2, 2 * np.pi, self.M, self.training)
        
        # 计算能量
        energy = torch.log(scores + 1e-5) + norm
        
        # 应用 mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力到 V
        x = torch.matmul(attention, V)
        
        # 重塑回原始形状
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        x = self.fc_o(x)
        
        # MIKAN-direct 无 KL 散度
        kl_div = torch.tensor(0.0, device=x.device)
        
        return x, attention, kl_div


class MIKANEncoderLayer(nn.Module):
    """MIKAN 编码器层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        M: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu',
        freeze_qk: bool = False
    ):
        super().__init__()
        
        self.self_attention = MIKANMultiHeadAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            M=M,
            dropout=dropout,
            device=device,
            freeze_qk=freeze_qk
        )
        self.feedforward = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, 1, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention: [batch_size, n_heads, seq_len, seq_len]
            kl_div: KL 散度
        """
        # Self-attention with residual and layer norm
        _src, attention, kl_div = self.self_attention(src, src, src, mask)
        src = self.ln1(src + self.dropout(_src))
        
        # Feedforward with residual and layer norm
        _src = self.feedforward(src)
        src = self.ln2(src + self.dropout(_src))
        
        return src, attention, kl_div


class MIKANClassifier(BaseClassifier):
    """MIKAN 文本分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        M: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
        device: str = 'cpu',
        freeze_qk: bool = False,
        **kwargs
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
            **kwargs
        )
        
        self.M = M
        self.freeze_qk = freeze_qk
        
        # MIKAN 编码器层
        self.layers = nn.ModuleList([
            MIKANEncoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                pf_dim=pf_dim,
                M=M,
                dropout=dropout,
                device=device,
                freeze_qk=freeze_qk
            )
            for _ in range(n_layers)
        ])
        
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码输入序列
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            encoded: [batch_size, seq_len, hidden_dim]
            kl_div: KL 散度损失
        """
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 准备 attention mask
        mask = None
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 通过 MIKAN 层
        total_kl_div = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl_div = layer(x, mask)
            total_kl_div = total_kl_div + kl_div
        
        return x, total_kl_div

