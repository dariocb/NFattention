"""
LRA 实验模型定义

与 seq_classification 中的 baseline 保持一致:
- transformer: 标准 Softmax Transformer
- mikan: MIKAN (Implicit Kernel Attention)
- performer: Performer (FAVOR+)
- rka: RKA (Random Kernel Attention)
- gmm_rks: GMM-RKS
- kpca_scaled: KPCA 仓库 Scaled Attention (Teo & Nguyen, NeurIPS 2024)
- metala: MetaLA 风格 GLA mixer (Chou et al., NeurIPS 2024)
- ours_latest: 我们最新版本 (no_qk + fixed_orth_v + shared spectral)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


# ============================================================
# 基础组件
# ============================================================

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 16384, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """前馈网络 (与 seq_classification 一致)"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ============================================================
# 1. Softmax Transformer (标准)
# ============================================================

class SoftmaxAttention(nn.Module):
    """标准 Softmax 注意力 - O(n²)"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
# 2. MIKAN (Implicit Kernel Attention)
# ============================================================

class MIKANAttention(nn.Module):
    """MIKAN 注意力 (基于 IKA kernel)"""
    def __init__(self, d_model: int, n_heads: int, M: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.M = M
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 共享的频率权重
        self.w1 = nn.Parameter(torch.empty(self.d_k, n_heads, M))
        self.w2 = nn.Parameter(torch.empty(self.d_k, n_heads, M))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        scale = 2 * np.pi
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # IKA kernel
        w1_t = self.w1.permute(1, 0, 2)  # [n_heads, d_k, M]
        w2_t = self.w2.permute(1, 0, 2)
        
        phi_q1 = torch.einsum('bhnd,hdm->bhnm', Q, w1_t * scale)
        phi_q2 = torch.einsum('bhnd,hdm->bhnm', Q, w2_t * scale)
        phi_k1 = torch.einsum('bhnd,hdm->bhnm', K, w1_t * scale)
        phi_k2 = torch.einsum('bhnd,hdm->bhnm', K, w2_t * scale)
        
        phi_q = torch.cat([torch.cos(phi_q1) + torch.cos(phi_q2),
                          torch.sin(phi_q1) + torch.sin(phi_q2)], dim=-1)
        phi_k = torch.cat([torch.cos(phi_k1) + torch.cos(phi_k2),
                          torch.sin(phi_k1) + torch.sin(phi_k2)], dim=-1)
        
        scores = torch.matmul(phi_q, phi_k.transpose(-2, -1)) / (4.0 * self.M)
        scores = scores * scores
        
        # norm 项
        norm = (torch.norm(Q, dim=-1, keepdim=True, p=2) ** 2 + 
                torch.norm(K, dim=-1, keepdim=True, p=2).transpose(-2, -1) ** 2) / (2 * math.sqrt(self.d_k))
        
        energy = torch.log(scores + 1e-5) + norm
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(energy, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
# 3. Performer (FAVOR+)
# ============================================================

class PerformerAttention(nn.Module):
    """Performer (FAVOR+) 注意力 - O(n)"""
    def __init__(self, d_model: int, n_heads: int, M: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.M = M
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 正交随机投影矩阵
        self.register_buffer('projection_matrix', self._create_projection_matrix())
        
    def _create_projection_matrix(self) -> torch.Tensor:
        """创建正交随机高斯矩阵"""
        nb_rows = self.M
        nb_columns = self.d_k
        
        block_list = []
        nb_full_blocks = nb_rows // nb_columns
        
        for _ in range(nb_full_blocks):
            q = torch.randn(nb_columns, nb_columns)
            q, _ = torch.linalg.qr(q)
            block_list.append(q.T)
        
        remaining_rows = nb_rows - nb_full_blocks * nb_columns
        if remaining_rows > 0:
            q = torch.randn(nb_columns, nb_columns)
            q, _ = torch.linalg.qr(q)
            block_list.append(q.T[:remaining_rows])
        
        final_matrix = torch.cat(block_list, dim=0)
        multiplier = torch.randn(nb_rows, nb_columns).norm(dim=1)
        
        return torch.diag(multiplier) @ final_matrix
    
    def _softmax_kernel(self, x: torch.Tensor, is_query: bool) -> torch.Tensor:
        """FAVOR+ softmax kernel"""
        data_normalizer = x.shape[-1] ** -0.25
        ratio = self.projection_matrix.shape[0] ** -0.5
        
        data_dash = torch.einsum('bhnd,md->bhnm', data_normalizer * x, self.projection_matrix)
        diag_data = (x ** 2).sum(dim=-1, keepdim=True) * (data_normalizer ** 2) / 2.0
        
        if is_query:
            max_val = data_dash.max(dim=-1, keepdim=True)[0].detach()
            data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + 1e-4)
        else:
            max_val = data_dash.amax(dim=(-1, -2), keepdim=True).detach()
            data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + 1e-4)
        
        return data_dash
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        Q_prime = self._softmax_kernel(Q, is_query=True)
        K_prime = self._softmax_kernel(K, is_query=False)
        
        # 线性注意力: O(n)
        KV = torch.einsum('bhnf,bhnd->bhfd', K_prime, V)
        QKV = torch.einsum('bhnf,bhfd->bhnd', Q_prime, KV)
        
        # 归一化
        k_sum = K_prime.sum(dim=2)
        normalizer = torch.einsum('bhnf,bhf->bhn', Q_prime, k_sum).unsqueeze(-1).clamp(min=1e-6)
        
        out = QKV / normalizer
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
# 4. RKA (Random Kernel Attention with Generator)
# ============================================================

class GeneratorBlock(nn.Module):
    """RKA 生成器网络"""
    def __init__(self, layer_dims: list, output_dim: int):
        super().__init__()
        layers = []
        for in_size, out_size in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(layer_dims[-1], output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RKAAttention(nn.Module):
    """RKA 注意力 (Random Kernel with Learnable Features)"""
    def __init__(self, d_model: int, n_heads: int, M: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.M = M
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 生成器网络
        noise_dims = [self.d_k, 2 * self.d_k, self.d_k]
        self.generator = GeneratorBlock(noise_dims, self.d_k)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('omega', torch.zeros(M // 2, self.d_k))
        
    def _new_feature_map(self):
        """采样新的噪声"""
        self.omega.copy_(torch.randn_like(self.omega))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        if self.training:
            self._new_feature_map()
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # 通过生成器
        omega = self.generator(self.omega)  # [M//2, d_k]
        
        softmax_temp = 1.0 / math.sqrt(self.d_k)
        Q_scaled = Q * math.sqrt(softmax_temp)
        K_scaled = K * math.sqrt(softmax_temp)
        
        Q_proj = torch.einsum('bhnd,md->bhnm', Q_scaled, omega)
        K_proj = torch.einsum('bhnd,md->bhnm', K_scaled, omega)
        
        phi_Q = torch.cat([Q_proj.cos(), Q_proj.sin()], dim=-1) * math.sqrt(2.0 / self.M)
        phi_K = torch.cat([K_proj.cos(), K_proj.sin()], dim=-1) * math.sqrt(2.0 / self.M)
        
        energy = torch.matmul(phi_Q, phi_K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(energy, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
# 5. GMM-RKS (Gaussian Mixture RKS)
# ============================================================

class GMMRKSAttention(nn.Module):
    """GMM-RKS 注意力 (Gaussian Mixture Fourier Features)"""
    def __init__(self, d_model: int, n_heads: int, M: int = 64, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.M = M
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # GMM 参数 (每个 head 独立)
        self.mean = nn.Parameter(torch.randn(n_heads, self.d_k) * 0.1)
        self.sigma = nn.Parameter(torch.eye(self.d_k).unsqueeze(0).repeat(n_heads, 1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('omega', torch.zeros(self.d_k, M // 4))
        self.softmax_temp = 1.0 / math.sqrt(self.d_k)
        
    def _new_feature_map(self):
        """采样新的基础随机矩阵"""
        self.omega.copy_(torch.randn_like(self.omega))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        if self.training:
            self._new_feature_map()
        
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k)
        
        # 缩放输入
        Q = Q * math.sqrt(self.softmax_temp)
        K = K * math.sqrt(self.softmax_temp)
        
        # Covariance 变换
        omega_gauss = torch.einsum('dm,hsd->hdm', self.omega, self.sigma)
        omega_p = omega_gauss + self.mean.unsqueeze(-1)
        omega_m = omega_gauss - self.mean.unsqueeze(-1)
        
        # 投影
        u_q_p = torch.einsum('blhd,hdm->blhm', Q, omega_p)
        u_q_m = torch.einsum('blhd,hdm->blhm', Q, omega_m)
        u_k_p = torch.einsum('blhd,hdm->blhm', K, omega_p)
        u_k_m = torch.einsum('blhd,hdm->blhm', K, omega_m)
        
        # Feature map
        phi_Q = torch.cat([torch.cos(u_q_p), torch.sin(u_q_p), 
                          torch.cos(u_q_m), torch.sin(u_q_m)], dim=-1) * math.sqrt(4.0 / self.M)
        phi_K = torch.cat([torch.cos(u_k_p), torch.sin(u_k_p),
                          torch.cos(u_k_m), torch.sin(u_k_m)], dim=-1) * math.sqrt(4.0 / self.M)
        
        # 转换维度顺序 [B, L, H, M] -> [B, H, L, M]
        phi_Q = phi_Q.permute(0, 2, 1, 3)
        phi_K = phi_K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # 线性注意力
        KV = torch.einsum('bhnd,bhne->bhde', phi_K, V)
        Z = 1.0 / (torch.einsum('bhnd,bhd->bhn', phi_Q, phi_K.sum(dim=2)) + 1e-6)
        out = torch.einsum('bhnd,bhde,bhn->bhne', phi_Q, KV, Z)
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
# 5b. KPCA Scaled Attention（与 text_classification.models.kpca_scaled 一致）
# ============================================================


class KPCAScaledAttention(nn.Module):
    """
    与 KPCA_code/Scaled_Attention/softmax.py 多头版一致；
    LRA 接口: forward(x, mask) -> [B,L,d]，与 SoftmaxAttention 对齐。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.s = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        v_mod = torch.matmul(eye - sym_attn * self.s, v)
        out = torch.matmul(attn, v_mod)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj_drop(self.proj(out))
        return out


# ============================================================
# 5c. MetaLA / GLA（与 text_classification.models.metala_baseline 一致）
# ============================================================


def _naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """与 flash-linear-attention naive_recurrent_simple_gla 一致（B T H D layout）。"""
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


class MetaLATokenMixerLRA(nn.Module):
    """与 MetaLABaselineClassifier 中 MetaLATokenMixer 相同实现。"""

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


class MetaLAAAttention(nn.Module):
    """LRA 用：包装 MetaLATokenMixer，mask 与 Transformer 块兼容。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mixer = MetaLATokenMixerLRA(d_model, n_heads)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        am = None
        if mask is not None:
            if mask.dim() == 4:
                am = mask.squeeze(1).squeeze(1).float()
            elif mask.dim() == 2:
                am = mask.float()
        return self.mixer(x, am)


# ============================================================
# 6. Ours (FRSKA-style with Spectral Learning)
# ============================================================

class OursAttention(nn.Module):
    """我们的方法 - 支持 latest/no_qk/fixed_orth_v 等配置 - O(n)"""
    def __init__(self, d_model: int, n_heads: int, M: int = 64, 
                 fixed_qk: bool = True, n_components: int = 4, dropout: float = 0.1,
                 qk_mode: str = "normal", v_mode: str = "learnable"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.M = M
        self.fixed_qk = fixed_qk
        self.n_components = n_components
        self.qk_mode = qk_mode
        self.v_mode = v_mode
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # latest 设定中 Q/K 直接使用输入，不参与学习
        if self.qk_mode == "no_qk":
            self.W_q.weight.requires_grad = False
            self.W_q.bias.requires_grad = False
            self.W_k.weight.requires_grad = False
            self.W_k.bias.requires_grad = False

        # latest 设定中 V 使用固定正交投影
        if self.v_mode == "fixed_orth":
            self.W_v.weight.requires_grad = False
            self.W_v.bias.requires_grad = False
            g = torch.Generator(device='cpu')
            g.manual_seed(42)
            rand_mat = torch.randn(d_model, d_model, generator=g)
            q, _ = torch.linalg.qr(rand_mat)
            fixed_v = torch.zeros(n_heads, d_model, self.d_k)
            for h in range(n_heads):
                start = h * self.d_k
                end = (h + 1) * self.d_k
                fixed_v[h] = q[:, start:end]
            self.register_buffer('fixed_v_weights', fixed_v)

        # 兼容旧设定：固定 QK
        if fixed_qk:
            nn.init.orthogonal_(self.W_q.weight)
            nn.init.orthogonal_(self.W_k.weight)
            self.W_q.weight.requires_grad = False
            self.W_q.bias.requires_grad = False
            self.W_k.weight.requires_grad = False
            self.W_k.bias.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的谱分布参数 (Normalizing Flow 风格)
        self.spectral_means = nn.Parameter(torch.randn(n_heads, n_components, self.d_k) * 0.1)
        self.spectral_log_vars = nn.Parameter(torch.zeros(n_heads, n_components, self.d_k))
        self.spectral_weights = nn.Parameter(torch.ones(n_heads, n_components) / n_components)
        
        # 额外的学习变换
        self.spectral_transform = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.Tanh(),
            nn.Linear(self.d_k, self.d_k)
        )
        
    def _sample_omega(self) -> torch.Tensor:
        """从学习的谱分布采样"""
        samples_per_comp = self.M // self.n_components
        omegas = []
        
        for i in range(self.n_components):
            mean = self.spectral_means[:, i, :]
            std = torch.exp(0.5 * self.spectral_log_vars[:, i, :])
            eps = torch.randn(self.n_heads, samples_per_comp, self.d_k, device=mean.device)
            omega = mean.unsqueeze(1) + std.unsqueeze(1) * eps
            omegas.append(omega)
        
        omega = torch.cat(omegas, dim=1)  # [H, M, d_k]
        omega = omega + self.spectral_transform(omega)
        return omega
        
    def _feature_map(self, x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """RKS 特征映射"""
        proj = torch.einsum('bhld,hmd->bhlm', x, omega)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(self.M)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape

        # Q/K: latest 使用 no_qk
        if self.qk_mode == "no_qk":
            Q = x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            K = x.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        else:
            # 如果固定 QK，使用 no_grad 减少内存
            if self.fixed_qk:
                with torch.no_grad():
                    Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                    K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
            else:
                Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
                K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # V: latest 使用 fixed orthogonal projection
        if self.v_mode == "fixed_orth":
            V = torch.einsum('bld,hdk->bhlk', x, self.fixed_v_weights)
        else:
            V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        omega = self._sample_omega()
        
        Q_prime = self._feature_map(Q, omega)
        K_prime = self._feature_map(K, omega)
        
        # 线性注意力
        KV = torch.matmul(K_prime.transpose(-2, -1), V)
        out = torch.matmul(Q_prime, KV)
        
        K_sum = K_prime.sum(dim=2, keepdim=True)
        normalizer = torch.matmul(Q_prime, K_sum.transpose(-2, -1)) + 1e-6
        out = out / normalizer
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# ============================================================
# Transformer 块和完整模型
# ============================================================

class TransformerBlock(nn.Module):
    """Transformer 块 (Pre-norm)"""
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = attention
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class LRAModel(nn.Module):
    """LRA 实验用 Transformer 模型"""
    def __init__(
        self,
        attention_type: str,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        max_seq_len: int = 4096,
        M: int = 64,
        dropout: float = 0.1,
        fixed_qk: bool = False
    ):
        super().__init__()
        self.attention_type = attention_type
        self.d_model = d_model
        
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.input_proj = nn.Linear(d_model, d_model)
        
        def create_attention():
            if attention_type == "transformer":
                return SoftmaxAttention(d_model, n_heads, dropout)
            elif attention_type == "mikan":
                return MIKANAttention(d_model, n_heads, M, dropout)
            elif attention_type == "performer":
                return PerformerAttention(d_model, n_heads, M, dropout)
            elif attention_type == "rka":
                return RKAAttention(d_model, n_heads, M, dropout)
            elif attention_type == "gmm_rks":
                return GMMRKSAttention(d_model, n_heads, M, dropout)
            elif attention_type == "kpca_scaled":
                return KPCAScaledAttention(d_model, n_heads, dropout)
            elif attention_type == "metala":
                return MetaLAAAttention(d_model, n_heads, dropout)
            elif attention_type == "ours_latest":
                return OursAttention(
                    d_model, n_heads, M,
                    fixed_qk=False,
                    qk_mode="no_qk",
                    v_mode="fixed_orth",
                    dropout=dropout
                )
            elif attention_type == "ours_fixed_qk":
                return OursAttention(d_model, n_heads, M, fixed_qk=True, dropout=dropout)
            elif attention_type == "ours_trainable_qk":
                return OursAttention(d_model, n_heads, M, fixed_qk=False, dropout=dropout)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.layers = nn.ModuleList([
            TransformerBlock(create_attention(), d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)
    
    def count_parameters(self) -> int:
        """统计可训练参数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_parameters(self) -> int:
        """统计总参数"""
        return sum(p.numel() for p in self.parameters())


def get_model(attention_type: str, **kwargs) -> LRAModel:
    """获取模型"""
    return LRAModel(attention_type=attention_type, **kwargs)


# ============================================================
# 模型名称映射 (与 seq_classification 一致)
# ============================================================

MODEL_TYPES = [
    "transformer",      # 标准 Softmax Transformer
    "mikan",           # MIKAN (Implicit Kernel Attention)
    "performer",       # Performer (FAVOR+)
    "rka",             # RKA (Random Kernel Attention)
    "gmm_rks",         # GMM-RKS
    "kpca_scaled",     # KPCA Scaled Attention
    "metala",          # MetaLA-style GLA
    "ours_latest",     # 我们的方法 (latest full setting)
    # Legacy aliases for compatibility:
    "ours_fixed_qk",
    "ours_trainable_qk"
]


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    batch_size = 2
    seq_len = 256
    d_model = 64
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    print(f"\nConfig: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print("-" * 70)
    
    for attn_type in MODEL_TYPES:
        model = get_model(
            attention_type=attn_type,
            d_model=d_model,
            n_heads=2,
            n_layers=1,
            max_seq_len=seq_len,
            M=32
        ).to(device)
        
        trainable = model.count_parameters()
        total = model.count_total_parameters()
        
        output = model(x)
        print(f"{attn_type:20s}: output={output.shape}, trainable={trainable:>8,}, total={total:>8,}, ratio={trainable/total*100:.1f}%")
