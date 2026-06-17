"""Causal language model for the FRSKA-style attention family."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import normflows as nf
    HAS_NORMFLOWS = True
except ImportError:
    HAS_NORMFLOWS = False

from .common import BaseCausalLM, build_dense_causal_mask


class SpectralFlow(nn.Module):
    def __init__(self, dim: int, num_flows: int = 3, hidden_dim: int = 64, num_mixtures: int = 10, device: str = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = device
        self.num_mixtures = num_mixtures
        if HAS_NORMFLOWS:
            torch.manual_seed(0)
            latent_size = dim
            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
            flows = []
            for i in range(num_flows):
                s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
                flows.append(nf.flows.MaskedAffineFlow(b if i % 2 == 0 else 1 - b, t, s))
                flows.append(nf.flows.ActNorm(latent_size))
            q0 = nf.distributions.DiagGaussian(latent_size)
            target = nf.distributions.GaussianMixture(n_modes=num_mixtures, dim=latent_size, trainable=True)
            self.nfm = nf.NormalizingFlow(q0=q0, flows=flows, p=target)
            self.nfm.to(device)
        else:
            self.nfm = None
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim * 2),
            )

    def sample(self, num_samples: int) -> torch.Tensor:
        if HAS_NORMFLOWS and self.nfm is not None:
            samples, _ = self.nfm.sample(num_samples=num_samples)
            return samples
        z = torch.randn(num_samples, self.dim, device=self.device)
        params = self.net(z)
        scale = torch.sigmoid(params[:, : self.dim]) * 2.0 + 0.5
        shift = params[:, self.dim :]
        return scale * z + shift

    def reverse_kld(self, num_samples: int) -> torch.Tensor:
        if HAS_NORMFLOWS and self.nfm is not None:
            return self.nfm.reverse_kld(num_samples)
        return torch.tensor(0.0, device=self.device)


class HybridSpectralFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        num_flows: int = 3,
        trunk_layers: int = 2,
        mlp_hidden_dim: int = 16,
        num_mixtures: int = 10,
        prior_type: str = "none",
        device: str = "cpu",
    ):
        super().__init__()
        if not HAS_NORMFLOWS:
            raise RuntimeError("HybridSpectralFlow requires normflows")
        self.dim = dim
        self.n_heads = n_heads
        self.num_flows = num_flows
        self.trunk_layers = trunk_layers
        self.tail_layers = num_flows - trunk_layers
        self.num_mixtures = num_mixtures
        self.prior_type = prior_type
        self.device = device
        self.q0 = nf.distributions.DiagGaussian(dim)
        for p in self.q0.parameters():
            p.requires_grad = False
        b = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(dim)])
        self.register_buffer("_mask", b)
        trunk = []
        for i in range(trunk_layers):
            mask_i = b if i % 2 == 0 else 1 - b
            s = nf.nets.MLP([dim, mlp_hidden_dim, dim], init_zeros=True)
            t = nf.nets.MLP([dim, mlp_hidden_dim, dim], init_zeros=True)
            trunk.append(nf.flows.MaskedAffineFlow(mask_i, t, s))
            trunk.append(nf.flows.ActNorm(dim))
        self.trunk_flows = nn.ModuleList(trunk)
        self.head_tails = nn.ModuleList()
        for _ in range(n_heads):
            tail = []
            for j in range(self.tail_layers):
                idx = trunk_layers + j
                mask_j = b if idx % 2 == 0 else 1 - b
                s = nf.nets.MLP([dim, mlp_hidden_dim, dim], init_zeros=True)
                t = nf.nets.MLP([dim, mlp_hidden_dim, dim], init_zeros=True)
                tail.append(nf.flows.MaskedAffineFlow(mask_j, t, s))
                tail.append(nf.flows.ActNorm(dim))
            self.head_tails.append(nn.ModuleList(tail))
        if prior_type == "ngsm":
            self.head_priors = nn.ModuleList([
                nf.distributions.GaussianMixture(n_modes=num_mixtures, dim=dim, trainable=True)
                for _ in range(n_heads)
            ])
        elif prior_type == "rbf":
            self.head_priors = nn.ModuleList([nf.distributions.DiagGaussian(dim) for _ in range(n_heads)])
        elif prior_type == "none":
            self.head_priors = None
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")
        self.to(device)

    def _forward_chain(self, z, flows):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for flow in flows:
            z, ld = flow(z)
            if ld.dim() == 0:
                ld = ld.expand(z.shape[0])
            log_det = log_det + ld
        return z, log_det

    def sample_omegas(self, batch_size: int, M: int):
        device = next(self.parameters()).device
        n = self.n_heads * batch_size * M
        z, _ = self.q0(n)
        z = z.to(device)
        z_trunk, _ = self._forward_chain(z, self.trunk_flows)
        z_trunk = z_trunk.view(self.n_heads, batch_size * M, self.dim)
        omega_per_head = []
        for h in range(self.n_heads):
            z_h, _ = self._forward_chain(z_trunk[h], self.head_tails[h])
            omega_per_head.append(z_h)
        omega = torch.stack(omega_per_head, dim=0)
        omega = omega.view(self.n_heads, batch_size, M, self.dim)
        omega = omega.permute(1, 0, 2, 3).contiguous()
        omega = torch.clamp(omega, min=-10.0, max=10.0)
        if torch.isnan(omega).any():
            omega = torch.randn_like(omega) * 0.5
        omega1, omega2 = torch.chunk(omega, chunks=2, dim=-1)
        return omega1, omega2

    def reverse_kld(self, M: int):
        if self.prior_type == "none" or self.head_priors is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        device = next(self.parameters()).device
        total_kl = torch.tensor(0.0, device=device)
        for h in range(self.n_heads):
            z, log_q0 = self.q0(M)
            z = z.to(device)
            log_q0 = log_q0.to(device)
            z_trunk, ld_trunk = self._forward_chain(z, self.trunk_flows)
            z_head, ld_head = self._forward_chain(z_trunk, self.head_tails[h])
            log_p_theta = log_q0 - ld_trunk - ld_head
            log_prior = self.head_priors[h].log_prob(z_head)
            kl_h = (log_p_theta - log_prior).mean()
            if torch.isnan(kl_h) or torch.isinf(kl_h):
                continue
            total_kl = total_kl + kl_h
        return total_kl


class OursAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        M: int = 64,
        dropout: float = 0.1,
        qk_mode: str = "no_qk",
        v_mode: str = "fixed_orth",
        shared_flow: bool = True,
        freeze_qk: bool = False,
        num_flows: int = 3,
        flow_hidden_dim: int = 64,
        num_mixtures: int = 10,
        kl_lambda: float = 0.001,
        architecture: str = "shared_full",
        prior_type: str = "ngsm",
        trunk_layers: int = 2,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.M = M
        self.qk_mode = qk_mode
        self.v_mode = v_mode
        self.shared_flow = shared_flow
        self.kl_lambda = kl_lambda
        self.architecture = architecture
        self.prior_type = prior_type
        if self.architecture == "hybrid" and not HAS_NORMFLOWS:
            self.architecture = "shared_full"
            self.prior_type = "none"
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        if architecture == "hybrid":
            self.flow = None
            self.flows = None
            self.hybrid_flow = HybridSpectralFlow(
                dim=self.head_dim * 2,
                n_heads=n_heads,
                num_flows=num_flows,
                trunk_layers=trunk_layers,
                mlp_hidden_dim=mlp_hidden_dim if mlp_hidden_dim is not None else 16,
                num_mixtures=num_mixtures,
                prior_type=prior_type,
            )
        elif architecture == "shared_full":
            self.flow = SpectralFlow(self.head_dim * 2, num_flows, flow_hidden_dim, num_mixtures)
            self.flows = None
            self.hybrid_flow = None
        else:
            self.flow = None
            self.flows = nn.ModuleList([
                SpectralFlow(self.head_dim * 2, num_flows, flow_hidden_dim, num_mixtures)
                for _ in range(n_heads)
            ])
            self.hybrid_flow = None
        if v_mode == "fixed_orth":
            g = torch.Generator(device="cpu")
            g.manual_seed(42)
            rand_mat = torch.randn(hidden_dim, hidden_dim, generator=g)
            q, _ = torch.linalg.qr(rand_mat)
            fixed_v = torch.zeros(n_heads, hidden_dim, self.head_dim)
            for h in range(n_heads):
                start = h * self.head_dim
                end = (h + 1) * self.head_dim
                fixed_v[h] = q[:, start:end]
            self.register_buffer("fixed_v_weights", fixed_v)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        if freeze_qk and qk_mode == "normal":
            nn.init.orthogonal_(self.fc_q.weight)
            nn.init.orthogonal_(self.fc_k.weight)
            self.fc_q.weight.requires_grad = False
            self.fc_q.bias.requires_grad = False
            self.fc_k.weight.requires_grad = False
            self.fc_k.bias.requires_grad = False

    def compute_rff_features(self, x: torch.Tensor, omega1: torch.Tensor, omega2: torch.Tensor) -> torch.Tensor:
        x_spectral1 = (2 * np.pi) * torch.einsum("bhnd,bhmd->bhnm", x, omega1)
        x_spectral2 = (2 * np.pi) * torch.einsum("bhnd,bhmd->bhnm", x, omega2)
        x_spectral1 = torch.clamp(x_spectral1, min=-10.0, max=10.0)
        x_spectral2 = torch.clamp(x_spectral2, min=-10.0, max=10.0)
        scale_factor = math.sqrt(1.0 / (4.0 * self.M))
        return scale_factor * torch.cat(
            [x_spectral1.cos() + x_spectral2.cos(), x_spectral1.sin() + x_spectral2.sin()],
            dim=-1,
        )

    def sample_spectral_density(self, batch_size: int):
        if self.architecture == "hybrid":
            return self.hybrid_flow.sample_omegas(batch_size, self.M)
        if self.shared_flow:
            num_samples = self.n_heads * batch_size * self.M
            omega_samples = self.flow.sample(num_samples)
            omega_samples = torch.clamp(omega_samples, min=-10.0, max=10.0)
            if torch.isnan(omega_samples).any():
                omega_samples = torch.randn_like(omega_samples) * 0.5
            omega = omega_samples.view(batch_size, self.n_heads, self.M, 2 * self.head_dim)
            return torch.chunk(omega, chunks=2, dim=-1)
        omega1_list, omega2_list = [], []
        for h in range(self.n_heads):
            omega_samples = self.flows[h].sample(batch_size * self.M)
            omega_samples = torch.clamp(omega_samples, min=-10.0, max=10.0)
            if torch.isnan(omega_samples).any():
                omega_samples = torch.randn_like(omega_samples) * 0.5
            omega = omega_samples.view(batch_size, self.M, 2 * self.head_dim)
            omega1_head, omega2_head = torch.chunk(omega, chunks=2, dim=-1)
            omega1_list.append(omega1_head)
            omega2_list.append(omega2_head)
        return torch.stack(omega1_list, dim=1), torch.stack(omega2_list, dim=1)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        if self.qk_mode == "no_qk":
            q = query
            k = key
        else:
            q = self.fc_q(query)
            k = self.fc_k(key)
        if self.v_mode == "fixed_orth":
            v = torch.einsum("bsd,hdk->bshk", value, self.fixed_v_weights).permute(0, 2, 1, 3).contiguous()
        else:
            v = self.fc_v(value).view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        omega1, omega2 = self.sample_spectral_density(batch_size)
        phi_q = self.compute_rff_features(q, omega1, omega2)
        phi_k = self.compute_rff_features(k, omega1, omega2)
        phi_q = torch.nn.functional.elu(phi_q) + 1.0
        phi_k = torch.nn.functional.elu(phi_k) + 1.0
        kv_summary = torch.einsum("bhlm,bhld->bhmd", phi_k, v)
        k_summary = torch.sum(phi_k, dim=2)
        output = torch.einsum("bhlm,bhmd->bhld", phi_q, kv_summary)
        denominator = torch.einsum("bhlm,bhm->bhl", phi_q, k_summary).clamp(min=1e-6).unsqueeze(-1)
        x = output / denominator
        x = self.dropout(x)
        attention_logits = torch.matmul(phi_q, phi_k.transpose(-2, -1))
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(attention_logits, dim=-1)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.hidden_dim)
        x = self.fc_o(x)
        kl_div = torch.tensor(0.0, device=x.device)
        if self.architecture == "hybrid":
            try:
                kl_raw = self.hybrid_flow.reverse_kld(self.M)
                if not (torch.isnan(kl_raw) or torch.isinf(kl_raw)):
                    kl_div = torch.clamp(kl_raw, min=0.0, max=100.0 * self.n_heads) * self.kl_lambda
            except Exception:
                pass
        elif self.shared_flow and self.flow is not None:
            try:
                kl_raw = self.flow.reverse_kld(self.M)
                if not (torch.isnan(kl_raw) or torch.isinf(kl_raw)):
                    kl_div = torch.clamp(kl_raw, min=0.0, max=100.0) * self.n_heads * self.kl_lambda
            except Exception:
                pass
        elif self.flows is not None:
            for flow in self.flows:
                try:
                    kl_raw = flow.reverse_kld(self.M)
                    if not (torch.isnan(kl_raw) or torch.isinf(kl_raw)):
                        kl_div = kl_div + torch.clamp(kl_raw, min=0.0, max=100.0) * self.kl_lambda
                except Exception:
                    continue
        return x, attention, kl_div


class OursBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        pf_dim: int,
        M: int = 64,
        dropout: float = 0.1,
        qk_mode: str = "no_qk",
        v_mode: str = "fixed_orth",
        shared_flow: bool = True,
        freeze_qk: bool = False,
        num_flows: int = 3,
        flow_hidden_dim: int = 64,
        num_mixtures: int = 10,
        kl_lambda: float = 0.001,
        architecture: str = "shared_full",
        prior_type: str = "ngsm",
        trunk_layers: int = 2,
        mlp_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.attn = OursAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            M=M,
            dropout=dropout,
            qk_mode=qk_mode,
            v_mode=v_mode,
            shared_flow=shared_flow,
            freeze_qk=freeze_qk,
            num_flows=num_flows,
            flow_hidden_dim=flow_hidden_dim,
            num_mixtures=num_mixtures,
            kl_lambda=kl_lambda,
            architecture=architecture,
            prior_type=prior_type,
            trunk_layers=trunk_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.ff = nn.Sequential(nn.Linear(hidden_dim, pf_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(pf_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        y, attn, kl = self.attn(x, x, x, mask)
        x = self.ln1(x + self.dropout(y))
        y = self.ff(x)
        x = self.ln2(x + self.dropout(y))
        return x, attn, kl


class OursLM(BaseCausalLM):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        pf_dim: int = 512,
        M: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
        qk_mode: str = "no_qk",
        v_mode: str = "fixed_orth",
        shared_flow: bool = True,
        use_shared_kernel: bool = False,
        freeze_qk: bool = False,
        num_flows: int = 3,
        flow_hidden_dim: int = 64,
        num_mixtures: int = 10,
        kl_lambda: float = 0.001,
        architecture: str = "shared_full",
        prior_type: str = "ngsm",
        trunk_layers: int = 2,
        mlp_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            **kwargs,
        )
        self.layers = nn.ModuleList(
            [
                OursBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    pf_dim=pf_dim,
                    M=M,
                    dropout=dropout,
                    qk_mode=qk_mode,
                    v_mode=v_mode,
                    shared_flow=shared_flow,
                    freeze_qk=freeze_qk,
                    num_flows=num_flows,
                    flow_hidden_dim=flow_hidden_dim,
                    num_mixtures=num_mixtures,
                    kl_lambda=kl_lambda,
                    architecture=architecture,
                    prior_type=prior_type,
                    trunk_layers=trunk_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(n_layers)
            ]
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(self.pos_encoder(self.embedding(input_ids)))
        mask = build_dense_causal_mask(attention_mask)
        if mask is None:
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).view(1, 1, seq_len, seq_len)
        kl_total = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, kl = layer(x, mask)
            kl_total = kl_total + kl
        return x, kl_total

