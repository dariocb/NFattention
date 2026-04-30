"""
我们的方法（Ours）分类器
基于 FRSKA (Flow-Regularized Spectral Kernel Attention)

支持两种配置：
- ours_fixed_qk: W_Q, W_K 固定
- ours_trainable_qk: W_Q, W_K 可训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

from .base import BaseClassifier, PositionwiseFeedforward

# 尝试导入 normflows
try:
    import normflows as nf
    HAS_NORMFLOWS = True
except ImportError:
    HAS_NORMFLOWS = False
    print("警告: normflows 未安装，将使用简化版本的 Flow")


class SpectralFlow(nn.Module):
    """使用 Normalizing Flow 学习谱密度"""
    
    def __init__(
        self,
        dim: int,
        num_flows: int = 3,
        hidden_dim: int = 64,
        num_mixtures: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        self.dim = dim
        self.device = device
        self.num_mixtures = num_mixtures
        
        if HAS_NORMFLOWS:
            torch.manual_seed(0)
            latent_size = dim
            
            # 定义 mask
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
            
            flows = []
            for i in range(num_flows):
                s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                
                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                flows += [nf.flows.ActNorm(latent_size)]
            
            q0 = nf.distributions.DiagGaussian(latent_size)
            
            target = nf.distributions.GaussianMixture(
                n_modes=num_mixtures,
                dim=latent_size,
                loc=None,
                scale=None,
                weights=None,
                trainable=True
            )
            
            self.nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
            self.nfm.to(device)
        else:
            self.nfm = None
            layers = []
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, dim * 2))
            self.net = nn.Sequential(*layers)
            
    def sample(self, num_samples: int) -> torch.Tensor:
        """从 flow 中采样"""
        if HAS_NORMFLOWS and self.nfm is not None:
            samples, _ = self.nfm.sample(num_samples=num_samples)
            return samples
        else:
            z = torch.randn(num_samples, self.dim, device=self.device)
            params = self.net(z)
            scale = torch.sigmoid(params[:, :self.dim]) * 2.0 + 0.5
            shift = params[:, self.dim:]
            x = scale * z + shift
            return x
            
    def reverse_kld(self, num_samples: int) -> torch.Tensor:
        """计算 reverse KL 散度"""
        if HAS_NORMFLOWS and self.nfm is not None:
            return self.nfm.reverse_kld(num_samples)
        return torch.tensor(0.0, device=self.device)


class OursMultiHeadAttention(nn.Module):
    """我们的多头注意力层 (FRSKA-based)"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        M: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu',
        qk_mode: str = 'no_qk',
        v_mode: str = 'fixed_orth',
        shared_flow: bool = True,
        use_shared_kernel: bool = False,
        freeze_qk: bool = False,
        num_flows: int = 3,
        flow_hidden_dim: int = 64,
        num_mixtures: int = 10,
        kl_lambda: float = 0.001
    ):
        super().__init__()
        
        assert hidden_dim % n_heads == 0
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.M = M
        self.device = device
        self.kl_lambda = kl_lambda
        self.qk_mode = qk_mode
        self.v_mode = v_mode
        self.shared_flow = shared_flow
        self.use_shared_kernel = use_shared_kernel
        
        # Q, K, V 投影
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        # Normalizing Flow：支持共享或每头独立
        if shared_flow:
            self.flow = SpectralFlow(
                dim=self.head_dim * 2,
                num_flows=num_flows,
                hidden_dim=flow_hidden_dim,
                num_mixtures=num_mixtures,
                device=device
            )
            self.flows = None
        else:
            self.flow = None
            self.flows = nn.ModuleList([
                SpectralFlow(
                    dim=self.head_dim * 2,
                    num_flows=num_flows,
                    hidden_dim=flow_hidden_dim,
                    num_mixtures=num_mixtures,
                    device=device
                )
                for _ in range(n_heads)
            ])

        if self.v_mode == 'fixed_orth':
            g = torch.Generator(device='cpu')
            g.manual_seed(42)
            rand_mat = torch.randn(hidden_dim, hidden_dim, generator=g)
            q, _ = torch.linalg.qr(rand_mat)
            fixed_v = torch.zeros(n_heads, hidden_dim, self.head_dim)
            for h in range(n_heads):
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                fixed_v[h] = q[:, start:end]
            self.register_buffer('fixed_v_weights', fixed_v)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        if self.use_shared_kernel:
            self.head_embedding_q = nn.Embedding(n_heads, self.head_dim)
            self.head_embedding_k = nn.Embedding(n_heads, self.head_dim)
            self.head_q_norm = nn.LayerNorm(self.head_dim)
            self.head_k_norm = nn.LayerNorm(self.head_dim)
        
        # 如果 freeze_qk=True，冻结 Q 和 K
        if freeze_qk and self.qk_mode == 'normal':
            self._freeze_qk_parameters()
            
    def _freeze_qk_parameters(self):
        """冻结 Q 和 K 的投影矩阵"""
        nn.init.orthogonal_(self.fc_q.weight)
        nn.init.orthogonal_(self.fc_k.weight)
        
        self.fc_q.weight.requires_grad = False
        self.fc_q.bias.requires_grad = False
        self.fc_k.weight.requires_grad = False
        self.fc_k.bias.requires_grad = False
        
    def compute_rff_features(
        self,
        x: torch.Tensor,
        omega1: torch.Tensor,
        omega2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Random Fourier Features
        
        Args:
            x: [batch, n_heads, seq_len, head_dim]
            omega1, omega2: [batch, n_heads, M, head_dim]
            
        Returns:
            phi: [batch, n_heads, seq_len, 2*M]
        """
        # 计算 x^T @ omega
        x_spectral1 = (2 * np.pi) * torch.einsum('bhnd,bhmd->bhnm', x, omega1)
        x_spectral2 = (2 * np.pi) * torch.einsum('bhnd,bhmd->bhnm', x, omega2)
        
        # Clamp 防止溢出
        x_spectral1 = torch.clamp(x_spectral1, min=-10.0, max=10.0)
        x_spectral2 = torch.clamp(x_spectral2, min=-10.0, max=10.0)
        
        # RFF 特征
        scale_factor = math.sqrt(1.0 / (4.0 * self.M))
        phi = scale_factor * torch.cat([
            x_spectral1.cos() + x_spectral2.cos(),
            x_spectral1.sin() + x_spectral2.sin()
        ], dim=-1)
        
        return phi

    def sample_spectral_density(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.shared_flow:
            num_samples = self.n_heads * batch_size * self.M
            if HAS_NORMFLOWS and self.flow.nfm is not None:
                omega_samples, _ = self.flow.nfm.sample(num_samples=num_samples)
            else:
                omega_samples = self.flow.sample(num_samples)
            omega_samples = torch.clamp(omega_samples, min=-10.0, max=10.0)
            if torch.isnan(omega_samples).any():
                omega_samples = torch.randn_like(omega_samples) * 0.5
            omega = omega_samples.view(batch_size, self.n_heads, self.M, 2 * self.head_dim)
            omega1, omega2 = torch.chunk(omega, chunks=2, dim=-1)
            return omega1, omega2

        omega1_list = []
        omega2_list = []
        for h in range(self.n_heads):
            num_samples = batch_size * self.M
            if HAS_NORMFLOWS and self.flows[h].nfm is not None:
                omega_samples, _ = self.flows[h].nfm.sample(num_samples=num_samples)
            else:
                omega_samples = self.flows[h].sample(num_samples)
            omega_samples = torch.clamp(omega_samples, min=-10.0, max=10.0)
            if torch.isnan(omega_samples).any():
                omega_samples = torch.randn_like(omega_samples) * 0.5
            omega = omega_samples.view(batch_size, self.M, 2 * self.head_dim)
            omega1_head, omega2_head = torch.chunk(omega, chunks=2, dim=-1)
            omega1_list.append(omega1_head)
            omega2_list.append(omega2_head)
        omega1 = torch.stack(omega1_list, dim=1)
        omega2 = torch.stack(omega2_list, dim=1)
        return omega1, omega2
        
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
        
        # Q/K：支持 normal 与 no_qk（满血设定使用 no_qk）
        if self.qk_mode == 'no_qk':
            Q = query
            K = key
        else:
            Q = self.fc_q(query)
            K = self.fc_k(key)

        # V：支持 learnable 与 fixed_orth（满血设定使用 fixed_orth）
        if self.v_mode == 'fixed_orth':
            V = torch.einsum('bsd,hdk->bshk', value, self.fixed_v_weights)
            V = V.permute(0, 2, 1, 3).contiguous()
        else:
            V = self.fc_v(value)
            V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 从 Flow 采样 omega
        omega1, omega2 = self.sample_spectral_density(batch_size)
        
        # Shared-kernel mode: condition all heads with head-id embedding.
        if self.use_shared_kernel:
            head_ids = torch.arange(self.n_heads, device=Q.device)
            head_emb_q = self.head_embedding_q(head_ids).view(1, self.n_heads, 1, self.head_dim)
            head_emb_k = self.head_embedding_k(head_ids).view(1, self.n_heads, 1, self.head_dim)
            Q = self.head_q_norm(Q + head_emb_q)
            K = self.head_k_norm(K + head_emb_k)

        # 计算 RFF 特征
        phi_Q = self.compute_rff_features(Q, omega1, omega2)
        phi_K = self.compute_rff_features(K, omega1, omega2)
        
        # 数值稳定修复：将特征映射到正域，避免分母符号翻转
        phi_Q_pos = torch.nn.functional.elu(phi_Q) + 1.0
        phi_K_pos = torch.nn.functional.elu(phi_K) + 1.0

        if mask is not None:
            # mask: [b,1,1,l] -> [b,1,l,1]
            key_mask = mask.squeeze(1).transpose(1, 2).unsqueeze(1)
            phi_K_pos = phi_K_pos * key_mask
            V = V * key_mask

        # 论文 Eq.(9c) 线性重排
        kv_summary = torch.einsum('bhlm,bhld->bhmd', phi_K_pos, V)      # [b,h,2M,d]
        k_summary = torch.sum(phi_K_pos, dim=2)                          # [b,h,2M]
        numerator = torch.einsum('bhlm,bhmd->bhld', phi_Q_pos, kv_summary)
        denominator = torch.einsum('bhlm,bhm->bhl', phi_Q_pos, k_summary)
        denominator = torch.clamp(denominator, min=1e-6).unsqueeze(-1)
        x = numerator / denominator
        x = self.dropout(x)
        
        # 仅用于返回可视化attention（不参与主计算路径）
        attention = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
        attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
        attn_denom = torch.clamp(torch.sum(attention, dim=-1, keepdim=True), min=1e-6)
        attention = attention / attn_denom
        
        # 重塑回原始形状
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        x = self.fc_o(x)
        
        # 计算 KL 散度
        kl_div = torch.tensor(0.0, device=x.device)
        if HAS_NORMFLOWS:
            if self.shared_flow and self.flow is not None and self.flow.nfm is not None:
                try:
                    kl_div_raw = self.flow.reverse_kld(self.M)
                    if not (torch.isnan(kl_div_raw) or torch.isinf(kl_div_raw)):
                        kl_div = torch.clamp(kl_div_raw, min=0.0, max=100.0) * self.n_heads * self.kl_lambda
                except:
                    pass
            elif not self.shared_flow and self.flows is not None:
                for flow in self.flows:
                    if flow.nfm is None:
                        continue
                    try:
                        kl_div_head = flow.reverse_kld(self.M)
                        if not (torch.isnan(kl_div_head) or torch.isinf(kl_div_head)):
                            kl_div = kl_div + torch.clamp(kl_div_head, min=0.0, max=100.0) * self.kl_lambda
                    except:
                        continue
        
        return x, attention, kl_div


class OursEncoderLayer(nn.Module):
    """我们的编码器层"""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        M: int = 64,
        dropout: float = 0.1,
        device: str = 'cpu',
        qk_mode: str = 'no_qk',
        v_mode: str = 'fixed_orth',
        shared_flow: bool = True,
        use_shared_kernel: bool = False,
        freeze_qk: bool = False,
        num_flows: int = 3,
        flow_hidden_dim: int = 64,
        num_mixtures: int = 10,
        kl_lambda: float = 0.001
    ):
        super().__init__()
        
        self.self_attention = OursMultiHeadAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            M=M,
            dropout=dropout,
            device=device,
            qk_mode=qk_mode,
            v_mode=v_mode,
            shared_flow=shared_flow,
            use_shared_kernel=use_shared_kernel,
            freeze_qk=freeze_qk,
            num_flows=num_flows,
            flow_hidden_dim=flow_hidden_dim,
            num_mixtures=num_mixtures,
            kl_lambda=kl_lambda
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
        # Self-attention
        _src, attention, kl_div = self.self_attention(src, src, src, mask)
        src = self.ln1(src + self.dropout(_src))
        
        # Feedforward
        _src = self.feedforward(src)
        src = self.ln2(src + self.dropout(_src))
        
        return src, attention, kl_div


class OursClassifier(BaseClassifier):
    """我们的方法分类器 (FRSKA-based)"""
    
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
        qk_mode: str = 'no_qk',
        v_mode: str = 'fixed_orth',
        shared_flow: bool = True,
        use_shared_kernel: bool = False,
        freeze_qk: bool = False,
        num_flows: int = 3,
        flow_hidden_dim: int = 64,
        num_mixtures: int = 10,
        kl_lambda: float = 0.001,
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
        self.qk_mode = qk_mode
        self.v_mode = v_mode
        self.shared_flow = shared_flow
        self.use_shared_kernel = use_shared_kernel
        self.freeze_qk = freeze_qk
        self.num_flows = num_flows
        self.flow_hidden_dim = flow_hidden_dim
        self.num_mixtures = num_mixtures
        self.kl_lambda = kl_lambda

        # 编码器层
        self.layers = nn.ModuleList([
            OursEncoderLayer(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                pf_dim=pf_dim,
                M=M,
                dropout=dropout,
                device=device,
                qk_mode=qk_mode,
                v_mode=v_mode,
                shared_flow=shared_flow,
                use_shared_kernel=use_shared_kernel,
                freeze_qk=freeze_qk,
                num_flows=num_flows,
                flow_hidden_dim=flow_hidden_dim,
                num_mixtures=num_mixtures,
                kl_lambda=kl_lambda,
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
        
        # 通过编码器层
        total_kl_div = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl_div = layer(x, mask)
            total_kl_div = total_kl_div + kl_div
        
        return x, total_kl_div
