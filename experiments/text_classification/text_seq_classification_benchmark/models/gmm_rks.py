"""
GMM-RKS (Gaussian Mixture Model - Random Kitchen Sinks) 分类器
使用 Gaussian Mixture 来学习频谱分布
参考论文: "On Learning the Kernel for Random Feature Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt, log
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


def orthogonal_random_matrix_(w: torch.Tensor):
    """
    正交化随机矩阵 (in-place)
    
    Args:
        w: [rows, columns] 矩阵
    """
    rows, columns = w.shape
    
    if rows == columns:
        block = torch.randn(rows, rows, device=w.device)
        norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
        Q, _ = torch.linalg.qr(block)
        w.copy_(Q * norms[None])
    else:
        start = 0
        while start < columns:
            end = min(start + rows, columns)
            block = torch.randn(rows, rows, device=w.device)
            norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
            Q, _ = torch.linalg.qr(block)
            w[:, start:end] = Q[:, :end-start] * norms[None, :end-start]
            start += rows


class GaussianMixtureFourierFeatures(nn.Module):
    """
    Gaussian Mixture Fourier Features
    每个 head 有独立的 mean 和 covariance 参数
    """
    
    def __init__(
        self,
        query_dimensions: int,
        n_heads: int,
        n_dims: int = None,
        softmax_temp: float = None,
        orthogonal: bool = False,
        redraw: int = 1,
        deterministic_eval: bool = False
    ):
        super().__init__()
        
        self.query_dims = query_dimensions
        self.n_heads = n_heads
        self.n_dims = n_dims or query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = softmax_temp if softmax_temp else 1.0 / sqrt(query_dimensions)
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval
        
        # 存储采样的 omega
        self.register_buffer(
            'omega',
            torch.zeros(query_dimensions, self.n_dims // 4)
        )
        
        # 计数器
        self.register_buffer(
            '_calls',
            torch.tensor(-1, dtype=torch.int)
        )
        
        # 可学习参数：每个 head 有独立的 mean 和 covariance
        # mean: [n_heads, query_dimensions]
        self.mean = nn.Parameter(torch.Tensor(n_heads, query_dimensions))
        # sigma: [n_heads, query_dimensions, query_dimensions] - full covariance
        self.sigma = nn.Parameter(torch.Tensor(n_heads, query_dimensions, query_dimensions))
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # 初始化 covariance 矩阵
        nn.init.xavier_uniform_(self.sigma)
        
        # 初始化 mean 向量
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma)
        bound = 1 / sqrt(fan_in) if fan_in > 0 else 0.1
        nn.init.uniform_(self.mean, -bound, bound)
        
    def new_feature_map(self, device, dtype):
        """重新采样基础随机矩阵"""
        # 如果不是训练模式且使用确定性评估，则跳过
        if self.deterministic_eval and not self.training:
            return
            
        # 只在每 redraw 次调用时重新采样
        self._calls += 1
        if (self._calls % self.redraw) != 0:
            return
            
        omega = torch.zeros(
            self.query_dims,
            self.n_dims // 4,
            dtype=dtype,
            device=device
        )
        
        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()
            
        self.omega.copy_(omega)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 feature map
        
        Args:
            x: [batch, seq_len, n_heads, head_dim] 或 [N, L, H, D]
            
        Returns:
            phi: [batch, seq_len, n_heads, n_dims] - feature map
        """
        # 缩放输入
        x = x * sqrt(self.softmax_temp)
        
        # Covariance 变换: omega' = omega @ sigma
        # omega: [query_dims, n_dims//4]
        # sigma: [n_heads, query_dims, query_dims]
        # omega_gauss: [n_heads, query_dims, n_dims//4]
        omega_gauss = torch.einsum('dm,hsd->hdm', self.omega, self.sigma)
        
        # 加/减 mean 向量
        # mean: [n_heads, query_dims]
        omega_p = omega_gauss + self.mean.unsqueeze(-1)  # [n_heads, query_dims, n_dims//4]
        omega_m = omega_gauss - self.mean.unsqueeze(-1)
        
        # 投影输入
        # x: [N, L, H, D] -> u_p/u_m: [N, L, H, n_dims//4]
        u_p = torch.einsum('nlhd,hdm->nlhm', x, omega_p)
        u_m = torch.einsum('nlhd,hdm->nlhm', x, omega_m)
        
        # Feature map: [cos(u+), sin(u+), cos(u-), sin(u-)]
        phi = torch.cat([
            torch.cos(u_p), torch.sin(u_p),
            torch.cos(u_m), torch.sin(u_m)
        ], dim=-1)  # [N, L, H, n_dims]
        
        return phi * sqrt(4.0 / self.n_dims)


class GMMRKSAttention(nn.Module):
    """GMM-RKS (Gaussian Mixture Model Random Kitchen Sinks) 多头注意力层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_dims: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu',
        orthogonal: bool = False
    ):
        super().__init__()
        
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_dims = n_dims
        self.device = device
        
        # Q, K, V 投影
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        # GMM Feature Map
        self.feature_map = GaussianMixtureFourierFeatures(
            query_dimensions=self.head_dim,
            n_heads=n_heads,
            n_dims=n_dims,
            orthogonal=orthogonal
        )
        
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6
        
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
            kl_div: KL 散度 (0)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # 重新采样 feature map
        self.feature_map.new_feature_map(query.device, query.dtype)
        
        # 线性投影
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # 重塑为多头格式 [batch, seq_len, n_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 应用 feature map: [batch, seq_len, n_heads, n_dims]
        phi_Q = self.feature_map(Q)
        phi_K = self.feature_map(K)
        
        # 转换为 [batch, n_heads, seq_len, dim] 格式
        phi_Q = phi_Q.permute(0, 2, 1, 3)  # [batch, n_heads, seq_len, n_dims]
        phi_K = phi_K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)  # [batch, n_heads, seq_len, head_dim]
        
        # 应用 mask (如果有)
        if mask is not None:
            # mask: [batch, 1, 1, seq_len]
            mask_expanded = mask.float()  # [batch, 1, 1, seq_len]
            phi_K = phi_K * mask_expanded.transpose(-1, -2)  # [batch, n_heads, seq_len, n_dims]
        
        # 线性注意力计算 (O(N D^2))
        # K^T @ V: [batch, n_heads, n_dims, head_dim]
        KV = torch.einsum("bhnd,bhne->bhde", phi_K, V)
        
        # 归一化因子: Q @ sum(K)
        Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", phi_Q, phi_K.sum(dim=2)) + self.eps)
        
        # Q @ (K^T @ V): [batch, n_heads, seq_len, head_dim]
        output = torch.einsum("bhnd,bhde,bhn->bhne", phi_Q, KV, Z)
        
        # 重塑回原始形状 [batch, seq_len, hidden_dim]
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        output = self.fc_o(output)
        output = self.dropout(output)
        
        # 计算近似的注意力矩阵 (用于可视化，可选)
        attention = torch.matmul(phi_Q, phi_K.transpose(-2, -1))  # [batch, n_heads, seq_len, seq_len]
        attention = attention / (attention.sum(dim=-1, keepdim=True) + self.eps)
        
        # GMM-RKS 无 KL 散度
        kl_div = torch.tensor(0.0, device=output.device)
        
        return output, attention, kl_div


class GMMRKSEncoderLayer(nn.Module):
    """GMM-RKS 编码器层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        n_dims: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu',
        orthogonal: bool = False
    ):
        super().__init__()
        
        self.self_attention = GMMRKSAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_dims=n_dims,
            dropout=dropout,
            device=device,
            orthogonal=orthogonal
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


class GMMRKSClassifier(BaseClassifier):
    """GMM-RKS (Gaussian Mixture Model Random Kitchen Sinks) 文本分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        n_dims: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
        device: str = 'cpu',
        orthogonal: bool = False,
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
        
        self.n_dims = n_dims
        self.orthogonal = orthogonal
        
        # GMM-RKS 编码器层
        self.layers = nn.ModuleList([
            GMMRKSEncoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                pf_dim=pf_dim,
                n_dims=n_dims,
                dropout=dropout,
                device=device,
                orthogonal=orthogonal
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
        
        # 通过 GMM-RKS 层
        total_kl_div = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl_div = layer(x, mask)
            total_kl_div = total_kl_div + kl_div
        
        return x, total_kl_div

