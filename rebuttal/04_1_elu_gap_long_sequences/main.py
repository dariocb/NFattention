#!/usr/bin/env python
"""Length-scaling diagnostic for the raw-RFF versus ELU+1 kernel gap.

This is a supplemental numerical diagnostic, not a new downstream-quality
benchmark. It keeps the paper FSKA feature configuration fixed, uses nested
spectral samples, and measures sampled kernel and linear-context error without
ever allocating an L x L attention matrix.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.stats import spearmanr

from rebuttal._shared.fska import FSKAConfig, FSKAAttention
from rebuttal._shared.results import ResultWriter, write_json


def _parse_lengths(value: str) -> List[int]:
    lengths = [int(item) for item in value.replace(",", " ").split()]
    if not lengths or min(lengths) < 2:
        raise ValueError("--lengths must contain integers >= 2")
    return lengths


def _features(x: torch.Tensor, omega1: torch.Tensor, omega2: torch.Tensor, pairs: int) -> torch.Tensor:
    first = 2.0 * math.pi * torch.einsum("bhld,bhmd->bhlm", x, omega1)
    second = 2.0 * math.pi * torch.einsum("bhld,bhmd->bhlm", x, omega2)
    return math.sqrt(1.0 / (4.0 * pairs)) * torch.cat(
        [first.cos() + second.cos(), first.sin() + second.sin()], dim=-1
    )


def _linear_context(
    phi_q: torch.Tensor, phi_k: torch.Tensor, values: torch.Tensor, signed: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    kv = torch.einsum("bhlm,bhld->bhmd", phi_k, values)
    ks = phi_k.sum(dim=2)
    numerator = torch.einsum("bhlm,bhmd->bhld", phi_q, kv)
    denominator = torch.einsum("bhlm,bhm->bhl", phi_q, ks)
    if signed:
        sign = torch.where(denominator < 0, -1.0, 1.0)
        stable = sign * denominator.abs().clamp_min(1e-6)
    else:
        stable = denominator.clamp_min(1e-6)
    return numerator / stable.unsqueeze(-1), denominator


def _sampled_kernel_metrics(
    reference_q: torch.Tensor,
    reference_k: torch.Tensor,
    raw_q: torch.Tensor,
    raw_k: torch.Tensor,
    elu_q: torch.Tensor,
    elu_k: torch.Tensor,
    count: int,
    generator: torch.Generator,
) -> Dict[str, float]:
    length = reference_q.shape[2]
    query = torch.randint(length, (count,), generator=generator, device=reference_q.device)
    key = torch.randint(length, (count,), generator=generator, device=reference_q.device)
    heads = torch.randint(reference_q.shape[1], (count,), generator=generator, device=reference_q.device)
    reference = (reference_q[0, heads, query] * reference_k[0, heads, key]).sum(dim=-1)
    raw = (raw_q[0, heads, query] * raw_k[0, heads, key]).sum(dim=-1)
    elu = (elu_q[0, heads, query] * elu_k[0, heads, key]).sum(dim=-1)
    reference_np, raw_np, elu_np = (value.detach().cpu().numpy() for value in (reference, raw, elu))
    norm = max(float(np.linalg.norm(reference_np)), 1e-12)
    return {
        "raw_kernel_relative_error": float(np.linalg.norm(raw_np - reference_np) / norm),
        "elu_kernel_relative_error": float(np.linalg.norm(elu_np - reference_np) / norm),
        "raw_kernel_spearman": float(spearmanr(reference_np, raw_np).statistic),
        "elu_kernel_spearman": float(spearmanr(reference_np, elu_np).statistic),
    }


@torch.no_grad()
def run_length(
    attention: FSKAAttention, length: int, reference_pairs: int, sampled_pairs: int,
    seed: int, device: torch.device,
) -> Dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + length)
    states = torch.randn(1, length, attention.hidden_dim, generator=generator, device=device)
    states = torch.nn.functional.layer_norm(states, (attention.hidden_dim,))
    q = attention._split_heads(states)
    k = attention._split_heads(states)
    values = torch.einsum("bld,hde->bhle", states, attention.fixed_v)

    samples = attention.density.sample(attention.n_heads * reference_pairs, evaluation=True)
    samples = samples.to(device=device, dtype=torch.float32).view(
        1, attention.n_heads, reference_pairs, 2 * attention.head_dim
    )
    omega1, omega2 = samples.chunk(2, dim=-1)
    reference_q = _features(q, omega1, omega2, reference_pairs)
    reference_k = _features(k, omega1, omega2, reference_pairs)
    raw_q = _features(q, omega1[:, :, : attention.M], omega2[:, :, : attention.M], attention.M)
    raw_k = _features(k, omega1[:, :, : attention.M], omega2[:, :, : attention.M], attention.M)
    elu_q, elu_k = torch.nn.functional.elu(raw_q) + 1.0, torch.nn.functional.elu(raw_k) + 1.0

    reference_context, reference_denominator = _linear_context(reference_q, reference_k, values, signed=True)
    raw_context, raw_denominator = _linear_context(raw_q, raw_k, values, signed=True)
    elu_context, elu_denominator = _linear_context(elu_q, elu_k, values, signed=False)
    context_norm = reference_context.norm().clamp_min(1e-12)
    metrics = _sampled_kernel_metrics(
        reference_q, reference_k, raw_q, raw_k, elu_q, elu_k, sampled_pairs, generator
    )
    metrics.update(
        {
            "length": float(length),
            "raw_context_relative_error": float((raw_context - reference_context).norm() / context_norm),
            "elu_context_relative_error": float((elu_context - reference_context).norm() / context_norm),
            "raw_negative_denominator_rate": float((raw_denominator < 0).float().mean()),
            "raw_near_zero_denominator_rate": float((raw_denominator.abs() < 1e-6).float().mean()),
            "elu_near_zero_denominator_rate": float((elu_denominator < 1e-6).float().mean()),
            "reference_negative_denominator_rate": float((reference_denominator < 0).float().mean()),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", default="128,256,512,1024,2000,4096")
    parser.add_argument("--reference-pairs", type=int, default=2048)
    parser.add_argument("--sampled-pairs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    lengths = [32, 64] if args.smoke else _parse_lengths(args.lengths)
    reference_pairs = 64 if args.smoke else args.reference_pairs
    if reference_pairs < 64:
        raise ValueError("--reference-pairs must be at least the 64 candidate pairs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    torch.manual_seed(args.seed)
    attention = FSKAAttention(
        FSKAConfig(
            hidden_dim=128, n_heads=4, num_spectral_pairs=64,
            density_mode="fixed_bivariate", dropout=0.0,
        )
    ).to(device).eval()
    config = {
        "experiment": "elu_gap_long_sequences",
        "purpose": "supplemental numerical length-scaling diagnostic; not a downstream benchmark",
        "lengths": lengths,
        "reference_pairs": reference_pairs,
        "candidate_pairs": attention.M,
        "density": "frozen 10-component bivariate prior",
        "sampled_kernel_pairs": args.sampled_pairs,
        "seed": args.seed,
        "device": str(device),
        "dense_attention_allocated": False,
    }
    writer = ResultWriter(args.output_dir, config)
    rows: List[Dict[str, float]] = []
    try:
        for length in lengths:
            row = run_length(attention, length, reference_pairs, args.sampled_pairs, args.seed, device)
            rows.append(row)
            writer.add_run(row)
    except Exception as error:
        writer.add_failure({"length": length if "length" in locals() else None}, error)
    writer.write_summary(rows)
    writer.write_latex(rows, ["length", "raw_kernel_relative_error", "elu_kernel_relative_error", "raw_context_relative_error", "elu_context_relative_error"])
    write_json(args.output_dir / "length_records.json", rows)
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        x = [row["length"] for row in rows]
        axes[0].plot(x, [row["raw_kernel_relative_error"] for row in rows], "o-", label="64-pair raw RFF")
        axes[0].plot(x, [row["elu_kernel_relative_error"] for row in rows], "o-", label="64-pair ELU+1")
        axes[0].set_xscale("log", base=2); axes[0].set_xlabel("Sequence length"); axes[0].set_ylabel("Sampled-kernel relative error"); axes[0].legend(); axes[0].grid(alpha=.25)
        axes[1].plot(x, [row["raw_negative_denominator_rate"] for row in rows], "o-", label="raw negative rate")
        axes[1].plot(x, [row["raw_near_zero_denominator_rate"] for row in rows], "o-", label="raw near-zero rate")
        axes[1].plot(x, [row["elu_near_zero_denominator_rate"] for row in rows], "o-", label="ELU+1 near-zero rate")
        axes[1].set_xscale("log", base=2); axes[1].set_xlabel("Sequence length"); axes[1].set_ylabel("Denominator rate"); axes[1].legend(); axes[1].grid(alpha=.25)
        figure.tight_layout(); figure.savefig(args.output_dir / "length_scaling.png", dpi=180); figure.savefig(args.output_dir / "length_scaling.pdf"); plt.close(figure)
    except ImportError:
        pass
    writer.finish()
    return 0 if not writer.failures and len(rows) == len(lengths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
