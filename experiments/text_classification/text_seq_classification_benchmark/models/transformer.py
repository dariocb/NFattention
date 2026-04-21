"""
标准 Transformer 分类器（Dot-product attention）
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


class MultiHeadAttention(nn.Module):
    """标准多头注意力"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        
        assert hidden_dim % n_heads == 0, "hidden_dim 必须能被 n_heads 整除"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.device = device
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, hidden_dim]
            key: [batch_size, seq_len, hidden_dim]
            value: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, 1, seq_len] (attention mask, 1=keep, 0=mask)
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention: [batch_size, n_heads, seq_len, seq_len]
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
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 计算注意力分数
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch_size, n_heads, seq_len, seq_len]
        
        # 应用 mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力到 V
        x = torch.matmul(attention, V)
        # [batch_size, n_heads, seq_len, head_dim]
        
        # 重塑回原始形状
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        x = self.fc_o(x)
        
        return x, attention


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(hidden_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforward(hidden_dim, pf_dim, dropout)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, 1, 1, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention: [batch_size, n_heads, seq_len, seq_len]
        """
        # Self-attention with residual and layer norm
        _src, attention = self.self_attention(src, src, src, mask)
        src = self.ln1(src + self.dropout(_src))
        
        # Feedforward with residual and layer norm
        _src = self.feedforward(src)
        src = self.ln2(src + self.dropout(_src))
        
        return src, attention


class TransformerClassifier(BaseClassifier):
    """标准 Transformer 文本分类器"""
    
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
        device: str = 'cpu',
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
        
        # Transformer 编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                pf_dim=pf_dim,
                dropout=dropout,
                device=device
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
            kl_div: KL 散度损失（Transformer 无 KL 散度）
        """
        # 词嵌入
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # 准备 attention mask
        mask = None
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 通过 Transformer 层
        for layer in self.layers:
            x, _ = layer(x, mask)
        
        # Transformer 无 KL 散度
        kl_div = torch.tensor(0.0, device=x.device)
        
        return x, kl_div

