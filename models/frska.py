"""
FRSKA (Flow-Regularized Spectral Kernel Attention) Implementation
基于论文: NF_for_attention_kernel.pdf

使用Normalizing Flow学习显式谱密度，通过KL散度正则化到NGSM先验
"""

import torch
import torch.nn as nn
import numpy as np

# 尝试导入normflows
try:
    import normflows as nf
    HAS_NORMFLOWS = True
except ImportError:
    HAS_NORMFLOWS = False
    print("警告: normflows未安装，将使用简化版本的Flow")


class SpectralFlow(nn.Module):
    """使用normflows实现的Normalizing Flow用于学习双变量谱密度"""
    def __init__(self, dim, num_flows=3, hidden_dim=128, device='cpu', num_mixtures=10):
        super().__init__()
        self.dim = dim  # 2*d (for bivariate: ω₁, ω₂)
        self.device = device
        self.num_mixtures = num_mixtures
        
        if HAS_NORMFLOWS:
            # 使用normflows构建真正的可逆Flow
            torch.manual_seed(0)
            latent_size = dim
            
            # 定义mask（用于MaskedAffineFlow）
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
            
            flows = []
            for i in range(num_flows):
                # 定义s和t网络（用于仿射耦合层）
                s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                
                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                flows += [nf.flows.ActNorm(latent_size)]
            
            # 基础分布：标准高斯
            q0 = nf.distributions.DiagGaussian(latent_size)
            
            # 目标分布：可学习的混合高斯（NGSM先验）
            target = nf.distributions.GaussianMixture(
                n_modes=num_mixtures, 
                dim=latent_size, 
                loc=None,  # 可学习
                scale=None,  # 可学习
                weights=None,  # 可学习
                trainable=True
            )
            
            # 创建NormalizingFlow
            self.nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
            self.nfm.to(device)
        else:
            # 回退到简化版本
            self.nfm = None
            layers = []
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, dim * 2))
            self.net = nn.Sequential(*layers)
    
    def sample(self, num_samples):
        """从flow中采样"""
        if HAS_NORMFLOWS and self.nfm is not None:
            samples, _ = self.nfm.sample(num_samples=num_samples)
            return samples
        else:
            # 简化版本
            z = torch.randn(num_samples, self.dim, device=self.device)
            params = self.net(z)
            scale = torch.sigmoid(params[:, :self.dim]) * 2.0 + 0.5
            shift = params[:, self.dim:]
            x = scale * z + shift
            return x
    
    def log_prob(self, x):
        """计算log密度 q_φ(ω₁, ω₂)"""
        if HAS_NORMFLOWS and self.nfm is not None:
            return self.nfm.log_prob(x)
        else:
            # 简化版本
            try:
                z = torch.randn(x.shape[0], self.dim, device=x.device)
                params = self.net(z)
                scale = torch.sigmoid(params[:, :self.dim]) * 2.0 + 0.5
                base_log_prob = -0.5 * torch.sum(z ** 2, dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
                log_det = torch.sum(torch.log(scale + 1e-6), dim=-1)
                log_prob = base_log_prob + log_det
                return torch.clamp(log_prob, min=-100, max=100)
            except:
                return torch.full((x.shape[0],), -10.0, device=x.device)
    
    def reverse_kld(self, num_samples):
        """计算reverse KL散度: KL(q || p)"""
        if HAS_NORMFLOWS and self.nfm is not None:
            return self.nfm.reverse_kld(num_samples)
        else:
            # 简化版本：返回0
            return torch.tensor(0.0, device=self.device)


# NGSM先验现在集成在SpectralFlow中（使用normflows的GaussianMixture）


class FRSKAAttention(nn.Module):
    """Flow-Regularized Spectral Kernel Attention"""
    def __init__(self, args, head_dim, n_heads, device, shared_flow=False, positive_feature_map=True):
        super().__init__()
        self.args = args
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.M = args.M  # 随机特征数量
        self.device = device
        self.shared_flow = shared_flow  # 是否共享Flow
        self.positive_feature_map = positive_feature_map  # 是否使用ELU+1正特征映射稳定化
        
        if shared_flow:
            # 共享一个Normalizing Flow（所有头共用）
            self.flow = SpectralFlow(
                dim=head_dim * 2, 
                num_flows=3,  # 3层
                hidden_dim=128,  # 增加隐藏层维度
                device=device,
                num_mixtures=10  # 10个混合成分
            )
            self.flows = None
        else:
            # 每个头有独立的Normalizing Flow（完全独立的头）
            # 使用nn.ModuleList为每个头创建独立的Flow
            self.flow = None
            self.flows = nn.ModuleList([
                SpectralFlow(
                    dim=head_dim * 2, 
                    num_flows=3,  # 3层
                    hidden_dim=128,  # 增加隐藏层维度
                    device=device,
                    num_mixtures=10  # 10个混合成分
                )
                for _ in range(n_heads)
            ])
        
        # 备用NGSM prior（如果normflows不可用）
        if not HAS_NORMFLOWS:
            self.ngsm_prior = None
        else:
            self.ngsm_prior = None  # 使用flow中的GaussianMixture
        
        # 注意：不需要spectral_encoder，flow直接采样，不依赖于Q、K
        
    def sample_spectral_density(self, Q, K):
        """
        从flow中采样谱密度 ω₁, ω₂
        如果shared_flow=True，所有头共享同一个Flow
        如果shared_flow=False，每个头使用自己独立的Flow
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        
        if self.shared_flow:
            # 共享Flow：所有头使用同一个Flow采样
            # 采样 n_heads * batch_size * M 个样本
            num_samples = n_heads * batch_size * self.M
            
            if HAS_NORMFLOWS and self.flow.nfm is not None:
                w, _ = self.flow.nfm.sample(num_samples=num_samples)
                omega_flat = w
            else:
                omega_flat = self.flow.sample(num_samples)  # 简化版本
            
            # 数值稳定性：clamp采样值，防止异常大的值
            omega_flat = torch.clamp(omega_flat, min=-10.0, max=10.0)
            
            # 检查NaN
            if torch.isnan(omega_flat).any():
                # 如果出现NaN，用标准高斯采样替代
                omega_flat = torch.randn_like(omega_flat) * 0.5
            
            # 重塑为 [batch, n_heads, M, 2*head_dim]
            omega = omega_flat.view(batch_size, n_heads, self.M, 2 * head_dim)
            
            # 分离 ω₁ 和 ω₂
            omega1, omega2 = torch.chunk(omega, chunks=2, dim=-1)
            # omega1: [batch, n_heads, M, head_dim]
            # omega2: [batch, n_heads, M, head_dim]
            
            omega_flat_all = omega_flat
            
        else:
            # 独立Flow：为每个头独立采样
            omega1_list = []
            omega2_list = []
            omega_flat_list = []
            
            for head_idx in range(n_heads):
                # 每个头使用自己独立的Flow
                num_samples = batch_size * self.M
                
                if HAS_NORMFLOWS and self.flows[head_idx].nfm is not None:
                    w, _ = self.flows[head_idx].nfm.sample(num_samples=num_samples)
                    omega_flat_head = w
                else:
                    omega_flat_head = self.flows[head_idx].sample(num_samples)  # 简化版本
                
                # 数值稳定性：clamp采样值
                omega_flat_head = torch.clamp(omega_flat_head, min=-10.0, max=10.0)
                
                # 检查NaN
                if torch.isnan(omega_flat_head).any():
                    omega_flat_head = torch.randn_like(omega_flat_head) * 0.5
                
                # 重塑为 [batch, M, 2*head_dim]
                omega_head = omega_flat_head.view(batch_size, self.M, 2 * head_dim)
                
                # 分离 ω₁ 和 ω₂
                omega1_head, omega2_head = torch.chunk(omega_head, chunks=2, dim=-1)
                # omega1_head: [batch, M, head_dim]
                # omega2_head: [batch, M, head_dim]
                
                omega1_list.append(omega1_head)
                omega2_list.append(omega2_head)
                omega_flat_list.append(omega_flat_head)
            
            # 拼接所有头的结果
            omega1 = torch.stack(omega1_list, dim=1)  # [batch, n_heads, M, head_dim]
            omega2 = torch.stack(omega2_list, dim=1)  # [batch, n_heads, M, head_dim]
            omega_flat_all = torch.cat(omega_flat_list, dim=0)  # [batch*n_heads*M, 2*head_dim]
        
        return omega1, omega2, omega_flat_all
    
    def compute_rff_features(self, x, omega1, omega2):
        """
        计算Random Fourier Features (RFF)
        x: [batch, n_heads, seq_len, head_dim]
        omega1, omega2: [batch, n_heads, M, head_dim]
        
        返回: [batch, n_heads, seq_len, 2*M]
        """
        # 数值稳定性：对输入进行归一化
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        omega1_norm = omega1 / (torch.norm(omega1, dim=-1, keepdim=True) + 1e-8)
        omega2_norm = omega2 / (torch.norm(omega2, dim=-1, keepdim=True) + 1e-8)
        
        # 计算 2π * x^T * ω，添加clamp防止溢出
        x_spectral1 = (2 * np.pi) * torch.einsum('bhnd,bhmd->bhnm', x_norm, omega1_norm)
        x_spectral2 = (2 * np.pi) * torch.einsum('bhnd,bhmd->bhnm', x_norm, omega2_norm)
        
        # Clamp防止cos/sin输入过大
        x_spectral1 = torch.clamp(x_spectral1, min=-10.0, max=10.0)
        x_spectral2 = torch.clamp(x_spectral2, min=-10.0, max=10.0)
        
        # RFF公式: sqrt(1/(4*M)) * [cos(2πx^Tω₁) + cos(2πx^Tω₂), sin(2πx^Tω₁) + sin(2πx^Tω₂)]
        scale_factor = torch.sqrt(torch.tensor(1.0 / (4.0 * self.M), device=x.device, dtype=x.dtype))
        
        phi = scale_factor * torch.cat([
            x_spectral1.cos() + x_spectral2.cos(),
            x_spectral1.sin() + x_spectral2.sin()
        ], dim=-1)
        
        # 检查NaN并替换
        if torch.isnan(phi).any():
            phi = torch.where(torch.isnan(phi), torch.zeros_like(phi), phi)
        
        # 最终clamp确保数值范围合理
        phi = torch.clamp(phi, min=-1.0, max=1.0)
        # [batch, n_heads, seq_len, 2*M]
        
        return phi
    
    def forward(self, Q, K, V, return_dense_attention=False):
        """
        Q, K, V: [batch, n_heads, seq_len, head_dim]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        
        # 数值稳定性：对Q和K进行归一化
        Q = Q / (torch.norm(Q, dim=-1, keepdim=True) + 1e-8)
        K = K / (torch.norm(K, dim=-1, keepdim=True) + 1e-8)
        
        # 采样谱密度
        omega1, omega2, omega_samples = self.sample_spectral_density(Q, K)
        
        # 计算RFF特征
        phi_Q = self.compute_rff_features(Q, omega1, omega2)  # [batch, n_heads, seq_len, 2*M]
        phi_K = self.compute_rff_features(K, omega1, omega2)  # [batch, n_heads, seq_len, 2*M]
        
        # 可切换的稳定化路径：
        # - True:  ELU+1 正特征映射（推荐，分母更稳定）
        # - False: 直接RFF（用于与原始实现做消融对照）
        if self.positive_feature_map:
            phi_Q_pos = torch.nn.functional.elu(phi_Q) + 1.0
            phi_K_pos = torch.nn.functional.elu(phi_K) + 1.0
        else:
            phi_Q_pos = phi_Q
            phi_K_pos = phi_K

        # 论文 Eq.(9c) 线性注意力重排：
        # numerator_i = phi(q_i)^T * sum_j(phi(k_j) v_j^T)
        # denominator_i = phi(q_i)^T * sum_j phi(k_j)
        kv_summary = torch.einsum('bhlm,bhld->bhmd', phi_K_pos, V)   # [b, h, 2M, d]
        k_summary = torch.sum(phi_K_pos, dim=2)                      # [b, h, 2M]
        numerator = torch.einsum('bhlm,bhmd->bhld', phi_Q_pos, kv_summary)   # [b, h, l, d]
        denominator = torch.einsum('bhlm,bhm->bhl', phi_Q_pos, k_summary)    # [b, h, l]
        denominator = torch.clamp(denominator, min=1e-6).unsqueeze(-1)   # [b, h, l, 1]
        context = numerator / denominator
        
        # 计算KL散度（参考代码方式）
        kl_div = torch.tensor(0.0, device=Q.device)
        
        if HAS_NORMFLOWS:
            if self.shared_flow:
                # 共享Flow：只计算一次KL，然后乘以头数（因为所有头共用）
                if self.flow.nfm is not None:
                    try:
                        kl_div_raw = self.flow.reverse_kld(self.M)
                        # 检查并处理NaN/Inf
                        if torch.isnan(kl_div_raw) or torch.isinf(kl_div_raw):
                            kl_div_raw = torch.tensor(0.0, device=Q.device)
                        kl_div = torch.clamp(kl_div_raw, min=0.0, max=100.0) * self.n_heads
                    except:
                        kl_div = torch.tensor(0.0, device=Q.device)
            else:
                # 独立Flow：每个头独立计算KL，然后求和
                for head_idx in range(self.n_heads):
                    if self.flows[head_idx].nfm is not None:
                        try:
                            # 每个头独立计算KL
                            kl_div_head = self.flows[head_idx].reverse_kld(self.M)
                            # 检查并处理NaN/Inf
                            if torch.isnan(kl_div_head) or torch.isinf(kl_div_head):
                                kl_div_head = torch.tensor(0.0, device=Q.device)
                            kl_div_head = torch.clamp(kl_div_head, min=0.0, max=100.0)
                            kl_div = kl_div + kl_div_head
                        except:
                            # 如果计算失败，跳过这个头的KL
                            continue
        # 如果normflows不可用，kl_div保持为0
        
        # 最终检查
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            kl_div = torch.tensor(0.0, device=Q.device)
        
        attention = None
        if return_dense_attention:
            # 仅用于可视化/调试，不参与主路径计算复杂度
            dense_scores = torch.matmul(phi_Q, phi_K.transpose(-2, -1))
            dense_scores = torch.nan_to_num(dense_scores, nan=0.0, posinf=0.0, neginf=0.0)
            dense_denom = torch.clamp(torch.sum(dense_scores, dim=-1, keepdim=True), min=1e-6)
            attention = dense_scores / dense_denom
        
        return context, attention, kl_div


class FRSKAMultiHeadAttention(nn.Module):
    """FRSKA Multi-Head Attention Layer
    
    支持的配置模式:
    - qk_mode: 
        'normal': 使用可学习的Q/K投影 (默认)
        'no_qk': 跳过Q/K投影，直接Q=X, K=X
    - v_mode:
        'learnable': 使用可学习的V投影 (默认)
        'fixed_orth': 每个头使用固定的正交V矩阵 (buffer, 不参与训练)
    """
    def __init__(self, args, hid_dim, n_heads, dropout, device, w1, w2, shared_flow=False,
                 freeze_qk=False, freeze_v=False, qk_mode='normal', v_mode='learnable',
                 positive_feature_map=True):
        super().__init__()
        self.args = args
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.key_dim = args.KEY_DIM
        self.head_dim = args.KEY_DIM // n_heads
        self.device = device
        self.qk_mode = qk_mode
        self.v_mode = v_mode
        self.positive_feature_map = positive_feature_map
        self.return_dense_attention = getattr(args, 'frska_return_dense_attention', False)
        
        # Q, K 投影 (如果 qk_mode='no_qk' 则不使用)
        if qk_mode == 'normal':
            self.fc_q = nn.Linear(args.KEY_DIM, args.KEY_DIM)
            self.fc_k = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        else:
            self.fc_q = None
            self.fc_k = None
        
        # V 投影 (根据 v_mode 选择不同实现)
        if v_mode == 'learnable':
            self.fc_v = nn.Linear(args.KEY_DIM, args.KEY_DIM)
            self.fixed_v_weights = None
        elif v_mode == 'fixed_orth':
            self.fc_v = None
            # 论文 Eq.(18) 对齐：在全维输入空间构造跨头互正交子空间
            # fixed_v_weights: [n_heads, key_dim, head_dim]
            g = torch.Generator(device='cpu')
            g.manual_seed(42)
            rand_mat = torch.randn(self.key_dim, self.key_dim, generator=g)
            q, _ = torch.linalg.qr(rand_mat)
            fixed_v = torch.zeros(n_heads, self.key_dim, self.head_dim)
            for h in range(n_heads):
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                fixed_v[h] = q[:, start:end]
            self.register_buffer('fixed_v_weights', fixed_v)
        else:
            raise ValueError(f"Unknown v_mode: {v_mode}. Use 'learnable' or 'fixed_orth'")
        
        self.fc_o = nn.Linear(args.KEY_DIM, args.KEY_DIM)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
        # FRSKA注意力
        self.frska_attention = FRSKAAttention(
            args,
            self.head_dim,
            n_heads,
            device,
            shared_flow=shared_flow,
            positive_feature_map=positive_feature_map,
        )
        
        # 如果freeze_qk=True且使用normal qk_mode，冻结Q和K的投影矩阵
        if freeze_qk and qk_mode == 'normal':
            self._freeze_qk_parameters()
        
        # 如果freeze_v=True且使用learnable v_mode，冻结V的投影矩阵
        if freeze_v and v_mode == 'learnable':
            self._freeze_v_parameters()
    
    def _freeze_qk_parameters(self):
        """
        冻结Q和K的投影矩阵参数
        注意：在冻结前对权重矩阵应用正交初始化，确保固定QK时多头仍然有意义
        """
        if self.fc_q is None or self.fc_k is None:
            return
        # 对权重矩阵应用正交初始化
        torch.nn.init.orthogonal_(self.fc_q.weight)
        torch.nn.init.orthogonal_(self.fc_k.weight)
        
        # 冻结参数
        self.fc_q.weight.requires_grad = False
        self.fc_q.bias.requires_grad = False
        self.fc_k.weight.requires_grad = False
        self.fc_k.bias.requires_grad = False
    
    def _freeze_v_parameters(self):
        """
        冻结V的投影矩阵参数
        注意：在冻结前对权重矩阵应用正交初始化
        """
        if self.fc_v is None:
            return
        # 对权重矩阵应用正交初始化
        torch.nn.init.orthogonal_(self.fc_v.weight)
        
        # 冻结参数
        self.fc_v.weight.requires_grad = False
        self.fc_v.bias.requires_grad = False
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        maxlen = query.shape[1]
        
        # Q, K 投影 (根据 qk_mode)
        if self.qk_mode == 'normal':
            Q = self.fc_q(query)
            K = self.fc_k(key)
        else:
            # no_qk: 直接使用输入
            Q = query
            K = key
        
        # V 投影 (根据 v_mode)
        if self.v_mode == 'learnable':
            V = self.fc_v(value)
        else:
            # fixed_orth: 使用全维互正交子空间投影
            # value: [batch, seq_len, key_dim]
            # fixed_v_weights: [n_heads, key_dim, head_dim]
            # V_fixed: [batch, seq_len, n_heads, head_dim]
            V_fixed = torch.einsum('bsd,hdk->bshk', value, self.fixed_v_weights)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.v_mode == 'learnable':
            V = V.view(batch_size, maxlen, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            V = V_fixed.permute(0, 2, 1, 3).contiguous()
        
        # FRSKA注意力（论文 Eq.(9c) 线性重排）
        context, attention, kl_div = self.frska_attention(
            Q, K, V, return_dense_attention=self.return_dense_attention
        )
        
        # 应用KL权重（如果指定了kl_lambda_frska，使用它；否则使用kl_lambda）
        if hasattr(self.args, 'kl_lambda_frska'):
            kl_div = kl_div * self.args.kl_lambda_frska
        else:
            kl_div = kl_div * self.args.kl_lambda
        
        x = self.dropout(context)
        
        # 重塑回原始形状
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.args.KEY_DIM)
        
        x = self.fc_o(x)
        
        return x, attention, kl_div

