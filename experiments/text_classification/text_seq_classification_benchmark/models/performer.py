"""
Performer 分类器
基于 FAVOR+ (Fast Attention Via positive Orthogonal Random features)
参考论文: "Rethinking Attention with Performers" (Choromanski et al., 2020)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


def softmax_kernel(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    is_query: bool,
    eps: float = 1e-4
) -> torch.Tensor:
    """
    FAVOR+ softmax kernel feature map
    
    Args:
        data: [batch, n_heads, seq_len, head_dim]
        projection_matrix: [n_features, head_dim]
        is_query: 是否为 query
        eps: 数值稳定性
        
    Returns:
        feature_map: [batch, n_heads, seq_len, n_features]
    """
    # 数据归一化
    data_normalizer = data.shape[-1] ** -0.25
    ratio = projection_matrix.shape[0] ** -0.5
    
    # 投影 [batch, n_heads, seq_len, head_dim] @ [head_dim, n_features]
    # -> [batch, n_heads, seq_len, n_features]
    data_dash = torch.einsum('bhnd,md->bhnm', data_normalizer * data, projection_matrix)
    
    # 计算 ||x||^2 / 2
    diag_data = (data ** 2).sum(dim=-1, keepdim=True) * (data_normalizer ** 2) / 2.0
    
    # 计算 exp(x^T w - ||x||^2 / 2)
    if is_query:
        # Query: 减去每行最大值以保持数值稳定
        max_val = data_dash.max(dim=-1, keepdim=True)[0].detach()
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + eps)
    else:
        # Key: 减去全局最大值
        max_val = data_dash.amax(dim=(-1, -2), keepdim=True).detach()
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + eps)
    
    return data_dash


def gaussian_orthogonal_random_matrix(
    nb_rows: int,
    nb_columns: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    生成正交随机高斯矩阵
    
    Args:
        nb_rows: 行数 (n_features)
        nb_columns: 列数 (head_dim)
        device: 设备
        
    Returns:
        matrix: [nb_rows, nb_columns]
    """
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
    
    # 乘以范数
    multiplier = torch.randn(nb_rows, nb_columns, device=device).norm(dim=1)
    
    return torch.diag(multiplier) @ final_matrix


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    线性注意力计算: O(N D^2) 复杂度
    
    Args:
        q: [batch, n_heads, seq_len, n_features] - transformed queries
        k: [batch, n_heads, seq_len, n_features] - transformed keys
        v: [batch, n_heads, seq_len, head_dim] - values
        
    Returns:
        output: [batch, n_heads, seq_len, head_dim]
    """
    # K^T V: [batch, n_heads, n_features, head_dim]
    kv = torch.einsum('bhnf,bhnd->bhfd', k, v)
    
    # Q @ (K^T V): [batch, n_heads, seq_len, head_dim]
    qkv = torch.einsum('bhnf,bhfd->bhnd', q, kv)
    
    # 归一化因子: Q @ sum(K)
    k_sum = k.sum(dim=2)  # [batch, n_heads, n_features]
    normalizer = torch.einsum('bhnf,bhf->bhn', q, k_sum)  # [batch, n_heads, seq_len]
    normalizer = normalizer.unsqueeze(-1).clamp(min=1e-6)
    
    return qkv / normalizer


class PerformerAttention(nn.Module):
    """Performer 注意力层 (FAVOR+)"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_features: int = None,
        dropout: float = 0.1,
        device: str = 'cpu',
        ortho_scaling: int = 0,
        redraw_interval: int = 1000
    ):
        super().__init__()
        
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # 默认 n_features = head_dim * log(head_dim)
        self.n_features = n_features or int(self.head_dim * math.log(self.head_dim))
        self.ortho_scaling = ortho_scaling
        self.redraw_interval = redraw_interval
        self.device = device
        
        # Q, K, V 投影
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 注册投影矩阵 buffer
        self.register_buffer(
            'projection_matrix',
            gaussian_orthogonal_random_matrix(self.n_features, self.head_dim)
        )
        
        # 计数器（用于重新绘制投影矩阵）
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))
        
    @torch.no_grad()
    def redraw_projection_matrix(self):
        """重新绘制投影矩阵"""
        self.projection_matrix.copy_(
            gaussian_orthogonal_random_matrix(
                self.n_features, 
                self.head_dim, 
                device=self.projection_matrix.device
            )
        )
        
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
            mask: [batch_size, 1, 1, seq_len] (当前未使用)
            
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            attention: None (线性注意力不显式计算注意力矩阵)
            kl_div: 0 (Performer 无 KL 散度)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 检查是否需要重绘投影矩阵
        if self.training:
            self.calls_since_last_redraw += 1
            if self.calls_since_last_redraw >= self.redraw_interval:
                self.redraw_projection_matrix()
                self.calls_since_last_redraw.zero_()
        
        # 线性投影
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # 重塑为多头格式 [batch, n_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 应用 mask 到 V (如果有)
        if mask is not None:
            # mask: [batch, 1, 1, seq_len] -> [batch, 1, seq_len, 1]
            mask_v = mask.squeeze(2).unsqueeze(-1).float()
            V = V * mask_v
        
        # 应用 FAVOR+ kernel
        Q_prime = softmax_kernel(Q, self.projection_matrix, is_query=True)
        K_prime = softmax_kernel(K, self.projection_matrix, is_query=False)
        
        # 线性注意力
        x = linear_attention(Q_prime, K_prime, V)
        
        # 重塑回原始形状
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        x = self.fc_o(x)
        x = self.dropout(x)
        
        # Performer 无 KL 散度
        kl_div = torch.tensor(0.0, device=x.device)
        
        return x, None, kl_div


class PerformerEncoderLayer(nn.Module):
    """Performer 编码器层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        n_features: int = None,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.self_attention = PerformerAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_features=n_features,
            dropout=dropout,
            device=device
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
        # Self-attention with residual and layer norm
        _src, attention, kl_div = self.self_attention(src, src, src, mask)
        src = self.ln1(src + self.dropout(_src))
        
        # Feedforward with residual and layer norm
        _src = self.feedforward(src)
        src = self.ln2(src + self.dropout(_src))
        
        return src, attention, kl_div


class PerformerClassifier(BaseClassifier):
    """Performer 文本分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        n_features: int = None,
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
        
        self.n_features = n_features
        
        # Performer 编码器层
        self.layers = nn.ModuleList([
            PerformerEncoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                pf_dim=pf_dim,
                n_features=n_features,
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
        
        # 通过 Performer 层
        total_kl_div = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl_div = layer(x, mask)
            total_kl_div = total_kl_div + kl_div
        
        return x, total_kl_div

