"""Minimal, corrected FSKA implementation used only by rebuttal experiments.

The original project implementations are intentionally not imported or modified.
This module retains the paper configuration while making masking, sampling, KL
regularisation, and feature-map variants explicit.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    import normflows as nf
except ImportError:  # Paper runs fail at construction time with a clear message.
    nf = None


def require_normflows() -> None:
    if nf is None:
        raise RuntimeError(
            "The rebuttal FSKA experiments require normflows. "
            "Install rebuttal/requirements.txt before running them."
        )


@dataclass(frozen=True)
class FSKAConfig:
    hidden_dim: int = 128
    n_heads: int = 4
    num_spectral_pairs: int = 64
    density_mode: str = "learned_flow"  # learned_flow | frozen_flow | fixed_bivariate | fixed_single_gaussian | learned_two_component_gmm
    feature_map: str = "elu_plus_one"  # elu_plus_one | raw_rff
    qk_mode: str = "identity"  # identity | learned
    value_mode: str = "fixed_orthogonal"
    num_flows: int = 3
    flow_hidden_dim: int = 64
    num_mixtures: int = 10
    kl_weight: float = 1e-3
    dropout: float = 0.1
    sample_seed: int = 1729
    orthogonal_seed: int = 42
    denominator_epsilon: float = 1e-6

    def validate(self) -> None:
        if self.hidden_dim % self.n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.density_mode not in {
            "learned_flow",
            "frozen_flow",
            "fixed_bivariate",
            "fixed_single_gaussian",
            "learned_two_component_gmm",
        }:
            raise ValueError(f"Unsupported density_mode: {self.density_mode}")
        if self.feature_map not in {"elu_plus_one", "raw_rff"}:
            raise ValueError(f"Unsupported feature_map: {self.feature_map}")
        if self.qk_mode not in {"identity", "learned"}:
            raise ValueError(f"Unsupported qk_mode: {self.qk_mode}")
        if self.value_mode != "fixed_orthogonal":
            raise ValueError("The rebuttal experiments only use fixed_orthogonal V")


@dataclass
class AttentionResult:
    output: Tensor
    kl_raw: Tensor
    diagnostics: Dict[str, Tensor]


def _forked_rng(seed: int, device: torch.device):
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    return torch.random.fork_rng(devices=devices, enabled=True)


class SpectralDensity(nn.Module):
    """Bivariate NGSM prior with an optional RealNVP variational density."""

    def __init__(self, dim: int, config: FSKAConfig):
        super().__init__()
        require_normflows()
        self.dim = dim
        self.config = config

        # Construct deterministically without leaking a manual_seed call globally.
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(config.sample_seed)
            self.prior = nf.distributions.GaussianMixture(
                n_modes=config.num_mixtures,
                dim=dim,
                loc=None,
                scale=None,
                weights=None,
                trainable=False,
            )
            for parameter in self.prior.parameters():
                parameter.requires_grad_(False)

            # This is an additional conventional stationary baseline, not the
            # matched fixed-prior arm: it replaces the 10-component target
            # mixture by one diagonal Gaussian over the joint (omega1,omega2).
            self.fixed_distribution = (
                nf.distributions.DiagGaussian(dim)
                if config.density_mode == "fixed_single_gaussian"
                else None
            )
            if self.fixed_distribution is not None:
                for parameter in self.fixed_distribution.parameters():
                    parameter.requires_grad_(False)

            # Low-capacity learned-density baseline. Equal component weights
            # avoid a non-reparameterised discrete-weight estimator; means and
            # diagonal log-scales are still learned by pathwise gradients.
            if config.density_mode == "learned_two_component_gmm":
                self.learned_gmm_means = nn.Parameter(torch.zeros(2, dim))
                self.learned_gmm_log_scales = nn.Parameter(torch.zeros(2, dim))
            else:
                self.register_parameter("learned_gmm_means", None)
                self.register_parameter("learned_gmm_log_scales", None)

            if config.density_mode in {"learned_flow", "frozen_flow"}:
                mask = torch.tensor(
                    [1.0 if i % 2 == 0 else 0.0 for i in range(dim)]
                )
                flows = []
                for index in range(config.num_flows):
                    scale = nf.nets.MLP(
                        [dim, config.flow_hidden_dim, dim], init_zeros=True
                    )
                    translate = nf.nets.MLP(
                        [dim, config.flow_hidden_dim, dim], init_zeros=True
                    )
                    active_mask = mask if index % 2 == 0 else 1.0 - mask
                    flows.append(
                        nf.flows.MaskedAffineFlow(active_mask, translate, scale)
                    )
                    flows.append(nf.flows.ActNorm(dim))
                base = nf.distributions.DiagGaussian(dim)
                self.flow = nf.NormalizingFlow(q0=base, flows=flows, p=self.prior)
                if config.density_mode == "frozen_flow":
                    for parameter in self.flow.parameters():
                        parameter.requires_grad_(False)
            else:
                self.flow = None

        self._training_draw = 0
        self._training_step: Optional[int] = None

    def set_training_step(self, step: int) -> None:
        self._training_step = int(step)

    def train(self, mode: bool = True):
        """Keep the parameter-matched frozen-flow control completely fixed."""
        super().train(mode)
        if self.config.density_mode == "frozen_flow" and self.flow is not None:
            self.flow.eval()
        return self

    def _device(self) -> torch.device:
        for parameter in self.parameters():
            return parameter.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

    def _seed(self, evaluation: bool, purpose_offset: int = 0) -> int:
        if evaluation:
            return self.config.sample_seed + purpose_offset
        if self._training_step is not None:
            return (
                self.config.sample_seed
                + 104729 * self._training_step
                + purpose_offset
            )
        seed = self.config.sample_seed + 104729 * self._training_draw + purpose_offset
        self._training_draw += 1
        return seed

    def sample(self, count: int, evaluation: bool) -> Tensor:
        device = self._device()
        seed = self._seed(evaluation)
        with _forked_rng(seed, device):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            if self.flow is not None:
                sampled = self.flow.sample(num_samples=count)
                samples = sampled[0] if isinstance(sampled, tuple) else sampled
            elif self.learned_gmm_means is not None:
                component = torch.randint(0, 2, (count,), device=device)
                noise = torch.randn(count, self.dim, device=device)
                samples = self.learned_gmm_means[component] + (
                    self.learned_gmm_log_scales[component].exp() * noise
                )
            else:
                distribution = self.fixed_distribution or self.prior
                sampled = distribution.sample(num_samples=count)
                samples = sampled[0] if isinstance(sampled, tuple) else sampled
        if not torch.isfinite(samples).all():
            raise FloatingPointError("Non-finite spectral sample")
        return samples.clamp(-10.0, 10.0)

    def reverse_kl(self, count: int, evaluation: bool) -> Tensor:
        if (
            self.flow is None
            or self.config.density_mode == "frozen_flow"
        ) and self.learned_gmm_means is None:
            return torch.zeros((), device=self._device())
        device = self._device()
        seed = self._seed(evaluation, purpose_offset=7919)
        with _forked_rng(seed, device):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            if self.flow is not None:
                value = self.flow.reverse_kld(count)
            else:
                component = torch.randint(0, 2, (count,), device=device)
                noise = torch.randn(count, self.dim, device=device)
                scales = self.learned_gmm_log_scales.exp()
                samples = self.learned_gmm_means[component] + scales[component] * noise
                normalized = (
                    samples[:, None, :] - self.learned_gmm_means[None]
                ) / scales[None]
                component_log_prob = -0.5 * (
                    normalized.square()
                    + 2.0 * self.learned_gmm_log_scales[None]
                    + math.log(2.0 * math.pi)
                ).sum(dim=-1)
                log_q = torch.logsumexp(component_log_prob, dim=1) - math.log(2.0)
                value = (log_q - self.prior.log_prob(samples)).mean()
        if not torch.isfinite(value):
            raise FloatingPointError("Non-finite reverse KL")
        return value

    def checksum(self, include_flow: bool = True) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            if not include_flow and (
                name.startswith("flow.") or name.startswith("learned_gmm_")
            ):
                continue
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


class FSKAAttention(nn.Module):
    """Linear bivariate spectral attention with an explicit practical map."""

    def __init__(self, config: FSKAConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.M = config.num_spectral_pairs

        if config.qk_mode == "learned":
            self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        else:
            self.q_proj = None
            self.k_proj = None

        # A seeded signed permutation is exactly orthogonal and avoids a QR/LAPACK
        # dependency in small CPU smoke tests.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.orthogonal_seed)
        permutation = torch.randperm(config.hidden_dim, generator=generator)
        signs = torch.randint(
            0, 2, (config.hidden_dim,), generator=generator, dtype=torch.float32
        ).mul_(2).sub_(1)
        matrix = torch.eye(config.hidden_dim)[:, permutation] * signs
        fixed_v = torch.stack(
            [
                matrix[:, h * self.head_dim : (h + 1) * self.head_dim]
                for h in range(config.n_heads)
            ],
            dim=0,
        )
        self.register_buffer("fixed_v", fixed_v)

        self.density = SpectralDensity(2 * self.head_dim, config)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

    def sample_omegas(
        self, batch_size: int, dtype: Optional[torch.dtype] = None
    ) -> Tuple[Tensor, Tensor]:
        samples = self.density.sample(
            batch_size * self.n_heads * self.M,
            evaluation=not self.training,
        )
        if dtype is not None:
            samples = samples.to(dtype=dtype)
        samples = samples.view(
            batch_size, self.n_heads, self.M, 2 * self.head_dim
        )
        return samples.chunk(2, dim=-1)

    def rff_features(self, x: Tensor, omega1: Tensor, omega2: Tensor) -> Tensor:
        projection1 = (2.0 * math.pi) * torch.einsum(
            "bhld,bhmd->bhlm", x, omega1
        )
        projection2 = (2.0 * math.pi) * torch.einsum(
            "bhld,bhmd->bhlm", x, omega2
        )
        scale = math.sqrt(1.0 / (4.0 * self.M))
        return scale * torch.cat(
            [
                projection1.cos() + projection2.cos(),
                projection1.sin() + projection2.sin(),
            ],
            dim=-1,
        )

    def _map_features(self, features: Tensor) -> Tensor:
        if self.config.feature_map == "elu_plus_one":
            return F.elu(features) + 1.0
        return features

    def linear_context(
        self, phi_q: Tensor, phi_k: Tensor, values: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        key_mask = mask[:, None, :, None].to(dtype=phi_k.dtype)
        phi_k = phi_k * key_mask
        values = values * key_mask
        kv_summary = torch.einsum("bhlm,bhld->bhmd", phi_k, values)
        k_summary = phi_k.sum(dim=2)
        numerator = torch.einsum("bhlm,bhmd->bhld", phi_q, kv_summary)
        denominator = torch.einsum("bhlm,bhm->bhl", phi_q, k_summary)

        epsilon = self.config.denominator_epsilon
        if self.config.feature_map == "raw_rff":
            signs = torch.where(denominator < 0, -1.0, 1.0)
            clamped = denominator.abs() < epsilon
            stable_denominator = signs * denominator.abs().clamp_min(epsilon)
        else:
            clamped = denominator < epsilon
            stable_denominator = denominator.clamp_min(epsilon)
        return numerator / stable_denominator.unsqueeze(-1), denominator, clamped

    def forward(
        self,
        hidden_states: Tensor,
        padding_mask: Optional[Tensor] = None,
        return_diagnostics: bool = False,
    ) -> AttentionResult:
        batch, length, _ = hidden_states.shape
        if padding_mask is None:
            padding_mask = torch.ones(
                batch, length, dtype=torch.bool, device=hidden_states.device
            )
        else:
            padding_mask = padding_mask.to(dtype=torch.bool)

        q_source = (
            self.q_proj(hidden_states) if self.q_proj is not None else hidden_states
        )
        k_source = (
            self.k_proj(hidden_states) if self.k_proj is not None else hidden_states
        )
        q = self._split_heads(q_source)
        k = self._split_heads(k_source)
        # Spectral trigonometric features and their contractions stay in FP32
        # even when the surrounding encoder uses AMP.
        values = torch.einsum(
            "bld,hde->bhle", hidden_states.float(), self.fixed_v.float()
        )

        omega1, omega2 = self.sample_omegas(batch, dtype=torch.float32)
        raw_q = self.rff_features(q.float(), omega1, omega2)
        raw_k = self.rff_features(k.float(), omega1, omega2)
        phi_q = self._map_features(raw_q)
        phi_k = self._map_features(raw_k)
        context, denominator, clamped = self.linear_context(
            phi_q, phi_k, values, padding_mask
        )
        context = self.dropout(context).to(dtype=hidden_states.dtype)
        context = context.permute(0, 2, 1, 3).contiguous().view(
            batch, length, self.hidden_dim
        )
        output = self.out_proj(context)
        output = output * padding_mask.unsqueeze(-1).to(output.dtype)

        kl_raw = self.density.reverse_kl(
            self.M, evaluation=not self.training
        )
        diagnostics: Dict[str, Tensor] = {}
        if return_diagnostics:
            diagnostics = {
                "denominator": denominator.detach(),
                "denominator_clamped": clamped.detach(),
                "omega1": omega1.detach(),
                "omega2": omega2.detach(),
                "raw_phi_q": raw_q.detach(),
                "raw_phi_k": raw_k.detach(),
                "mapped_phi_q": phi_q.detach(),
                "mapped_phi_k": phi_k.detach(),
            }
        return AttentionResult(output=output, kl_raw=kl_raw, diagnostics=diagnostics)
