"""
RKA (Random Kernel Attention) 分类器
使用生成器网络学习 Random Fourier Features
参考论文: "On Learning the Kernel for Random Feature Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward


class GeneratorLayer(nn.Module):
    """生成器层"""
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        batch_norm: bool = True, 
        activation: str = 'leaky'
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        # BatchNorm1d 需要 [N, C] 或 [N, C, L] 格式
        if len(x.shape) == 3:
            # [batch, seq, dim] -> [batch * seq, dim]
            orig_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            x = self.batch_norm(x)
            x = x.reshape(orig_shape)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x


class GeneratorBlock(nn.Module):
    """生成器网络块"""
    
    def __init__(
        self, 
        layer_dims: list, 
        output_dim: int, 
        hidden_act: str = 'leaky', 
        output_act: str = 'tanh'
    ):
        super().__init__()
        
        generator_layers = []
        # 隐藏层
        for in_size, out_size in zip(layer_dims[:-1], layer_dims[1:]):
            generator_layers.append(
                GeneratorLayer(in_size, out_size, activation=hidden_act)
            )
        # 最后一层
        generator_layers.append(
            GeneratorLayer(
                layer_dims[-1], 
                output_dim, 
                batch_norm=False, 
                activation=output_act
            )
        )
        self.generator_layers = nn.ModuleList(generator_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.generator_layers:
            x = layer(x)
        return x


class RKAAttention(nn.Module):
    """RKA (Random Kernel Attention) 多头注意力层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_features: int = 64,
        noise_dims: list = None,
        dropout: float = 0.1,
        device: str = 'cpu',
        redraw_interval: int = 1
    ):
        super().__init__()
        
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.n_features = n_features
        self.device = device
        self.redraw_interval = redraw_interval
        
        # 默认噪声维度
        if noise_dims is None:
            noise_dims = [self.head_dim, 2 * self.head_dim, self.head_dim]
        self.noise_dims = noise_dims
        
        # Q, K, V 投影
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        # 生成器网络：将噪声转换为 random features
        self.generator = GeneratorBlock(
            layer_dims=noise_dims,
            output_dim=self.head_dim,
            hidden_act='leaky',
            output_act='tanh'
        )
        
        self.dropout = nn.Dropout(dropout)
        self.softmax_temp = 1.0 / math.sqrt(self.head_dim)
        
        # 存储采样的 omega
        self.register_buffer(
            'omega',
            torch.zeros(n_features // 2, noise_dims[0])
        )
        
        # 计数器
        self.register_buffer('_calls', torch.tensor(-1, dtype=torch.int))
        
    def new_feature_map(self):
        """重新采样噪声"""
        self._calls += 1
        if (self._calls % self.redraw_interval) != 0 and not self.training:
            return
            
        omega = torch.randn(
            self.n_features // 2,
            self.noise_dims[0],
            dtype=self.omega.dtype,
            device=self.omega.device
        )
        self.omega.copy_(omega)
        
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
        
        # 重新采样噪声
        self.new_feature_map()
        
        # 线性投影
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        # 重塑为多头格式 [batch, n_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 通过生成器网络生成 omega [n_features//2, head_dim]
        omega = self.generator(self.omega)  # [n_features//2, head_dim]
        
        # 缩放
        Q_scaled = Q * math.sqrt(self.softmax_temp)
        K_scaled = K * math.sqrt(self.softmax_temp)
        
        # 计算 RFF: x @ omega^T -> [batch, n_heads, seq_len, n_features//2]
        # 需要广播 omega 到所有 heads
        Q_proj = torch.einsum('bhnd,md->bhnm', Q_scaled, omega)
        K_proj = torch.einsum('bhnd,md->bhnm', K_scaled, omega)
        
        # RFF 特征: [cos(x @ omega), sin(x @ omega)]
        phi_Q = torch.cat([Q_proj.cos(), Q_proj.sin()], dim=-1)  # [batch, n_heads, seq_len, n_features]
        phi_K = torch.cat([K_proj.cos(), K_proj.sin()], dim=-1)
        
        # 归一化
        phi_Q = phi_Q * math.sqrt(2.0 / self.n_features)
        phi_K = phi_K * math.sqrt(2.0 / self.n_features)
        
        # 计算注意力分数 (kernel approximation)
        # phi_Q @ phi_K^T 近似 exp(-||q-k||^2 / 2)
        energy = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
        
        # 缩放
        energy = energy / math.sqrt(self.head_dim)
        
        # 应用 mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 处理 NaN (当整行都被 mask 时)
        attention = torch.nan_to_num(attention, nan=0.0)
        
        # 应用注意力到 V
        x = torch.matmul(attention, V)
        
        # 重塑回原始形状
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        x = self.fc_o(x)
        
        # RKA 无 KL 散度
        kl_div = torch.tensor(0.0, device=x.device)
        
        return x, attention, kl_div


class RKAEncoderLayer(nn.Module):
    """RKA 编码器层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        n_features: int = 64,
        noise_dims: list = None,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.self_attention = RKAAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_features=n_features,
            noise_dims=noise_dims,
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


class RKAClassifier(BaseClassifier):
    """RKA (Random Kernel Attention) 文本分类器"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        n_features: int = 64,
        noise_dims: list = None,
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
        self.noise_dims = noise_dims
        
        # RKA 编码器层
        self.layers = nn.ModuleList([
            RKAEncoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                pf_dim=pf_dim,
                n_features=n_features,
                noise_dims=noise_dims,
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
        
        # 通过 RKA 层
        total_kl_div = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl_div = layer(x, mask)
            total_kl_div = total_kl_div + kl_div
        
        return x, total_kl_div

