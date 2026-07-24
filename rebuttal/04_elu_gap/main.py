#!/usr/bin/env python
"""Train raw/ELU variants and quantify the practical kernel distortion."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from rebuttal._shared.classifier import ModelConfig, build_model
from rebuttal._shared.datasets import (
    EncodedTextDataset,
    Vocabulary,
    load_sst5,
)
from rebuttal._shared.grid import (
    add_common_arguments,
    parse_seeds,
    run_sst_grid,
)
from rebuttal._shared.results import write_json


def _centered_alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean(dim=-1, keepdim=True) - left.mean(
        dim=-2, keepdim=True
    ) + left.mean(dim=(-2, -1), keepdim=True)
    right = right - right.mean(dim=-1, keepdim=True) - right.mean(
        dim=-2, keepdim=True
    ) + right.mean(dim=(-2, -1), keepdim=True)
    numerator = (left * right).sum()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).item())


def _features(x, omega1, omega2, pairs: int):
    projection1 = (2.0 * math.pi) * torch.einsum(
        "bhld,bhmd->bhlm", x, omega1
    )


def _diagnostic_context(phi_q, phi_k, values, mask, signed: bool):
    key_mask = mask[:, None, :, None].to(phi_k.dtype)
    phi_k = phi_k * key_mask
    values = values * key_mask
    kv = torch.einsum("bhlm,bhld->bhmd", phi_k, values)
    ks = phi_k.sum(dim=2)
    numerator = torch.einsum("bhlm,bhmd->bhld", phi_q, kv)
    denominator = torch.einsum("bhlm,bhm->bhl", phi_q, ks)
    epsilon = 1e-6
    if signed:
        signs = torch.where(denominator < 0, -1.0, 1.0)
        clamped = denominator.abs() < epsilon
        stable = signs * denominator.abs().clamp_min(epsilon)
    else:
        clamped = denominator < epsilon
        stable = denominator.clamp_min(epsilon)
    return numerator / stable.unsqueeze(-1), denominator, clamped
    projection2 = (2.0 * math.pi) * torch.einsum(
        "bhld,bhmd->bhlm", x, omega2
    )
    return math.sqrt(1.0 / (4.0 * pairs)) * torch.cat(
        [
            projection1.cos() + projection2.cos(),
            projection1.sin() + projection2.sin(),
        ],
        dim=-1,
    )


@torch.no_grad()
def kernel_diagnostics(
    output_dir: Path,
    data_dir: Path,
    seeds: List[int],
    device: torch.device,
    max_examples: int,
    revision: str | None,
) -> None:
    bundle = load_sst5(data_dir, revision=revision)
    records: List[Dict[str, float]] = []
    for seed in seeds:
        checkpoint_path = output_dir / "checkpoints" / f"elu_plus_one__seed{seed}.pt"
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = ModelConfig(**payload["model_config"])
        model = build_model(config)
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        vocabulary = Vocabulary.from_dict(payload["vocabulary"])
        dataset = EncodedTextDataset(bundle.test, vocabulary, config.max_length)
        per_class = max(1, math.ceil(max_examples / bundle.num_classes))
        selected_indices: List[int] = []
        selected_counts = {label: 0 for label in range(bundle.num_classes)}
        for index, label in enumerate(bundle.test.labels):
            if selected_counts[label] < per_class:
                selected_indices.append(index)
                selected_counts[label] += 1
            if len(selected_indices) >= max_examples:
                break
        loader = DataLoader(
            Subset(dataset, selected_indices[:max_examples]),
            batch_size=1,
            shuffle=False,
        )
        attention = model.layers[0].attention

        for example_index, (
            input_ids,
            mask,
            _label,
            _length,
            _index,
        ) in enumerate(loader):
            if example_index >= max_examples:
                break
            input_ids, mask = input_ids.to(device), mask.to(device)
            states = model.input_dropout(model.position(model.embedding(input_ids)))
            states = states * mask.unsqueeze(-1).to(states.dtype)
            valid_length = max(1, int(mask.sum().item()))
            states = states[:, :valid_length]
            mask = mask[:, :valid_length]
            states = model.layers[0].norm1(states)
            q_source = (
                attention.q_proj(states)
                if attention.q_proj is not None
                else states
            )
            k_source = (
                attention.k_proj(states)
                if attention.k_proj is not None
                else states
            )
            q = attention._split_heads(q_source)
            k = attention._split_heads(k_source)
            values = torch.einsum("bld,hde->bhle", states, attention.fixed_v)

            reference_pairs = 2048
            samples = attention.density.sample(
                attention.n_heads * reference_pairs, evaluation=True
            ).to(dtype=q.dtype).view(
                1, attention.n_heads, reference_pairs, 2 * attention.head_dim
            )
            omega1, omega2 = samples.chunk(2, dim=-1)
            raw_reference_q = _features(q, omega1, omega2, reference_pairs)
            raw_reference_k = _features(k, omega1, omega2, reference_pairs)
            raw_q = _features(q, omega1[:, :, :64], omega2[:, :, :64], 64)
            raw_k = _features(k, omega1[:, :, :64], omega2[:, :, :64], 64)
            elu_q = torch.nn.functional.elu(raw_q) + 1.0
            elu_k = torch.nn.functional.elu(raw_k) + 1.0

            reference_kernel = torch.matmul(
                raw_reference_q, raw_reference_k.transpose(-2, -1)
            )
            raw_kernel = torch.matmul(raw_q, raw_k.transpose(-2, -1))
            elu_kernel = torch.matmul(elu_q, elu_k.transpose(-2, -1))
            reference_norm = reference_kernel.norm().clamp_min(1e-12)

            raw_context, raw_denominator, raw_clamped = _diagnostic_context(
                raw_q, raw_k, values, mask, signed=True
            )
            elu_context, elu_denominator, elu_clamped = _diagnostic_context(
                elu_q, elu_k, values, mask, signed=False
            )
            reference_context, _, _ = _diagnostic_context(
                raw_reference_q, raw_reference_k, values, mask, signed=True
            )
            flat_reference = reference_kernel.flatten().cpu().numpy()
            flat_raw = raw_kernel.flatten().cpu().numpy()
            flat_elu = elu_kernel.flatten().cpu().numpy()
            records.append(
                {
                    "seed": seed,
                    "example": example_index,
                    "raw_relative_frobenius": float(
                        ((raw_kernel - reference_kernel).norm() / reference_norm).item()
                    ),
                    "elu_relative_frobenius": float(
                        ((elu_kernel - reference_kernel).norm() / reference_norm).item()
                    ),
                    "raw_centered_alignment": _centered_alignment(
                        raw_kernel, reference_kernel
                    ),
                    "elu_centered_alignment": _centered_alignment(
                        elu_kernel, reference_kernel
                    ),
                    "raw_spearman": float(
                        spearmanr(flat_raw, flat_reference).statistic
                    ),
                    "elu_spearman": float(
                        spearmanr(flat_elu, flat_reference).statistic
                    ),
                    "raw_negative_denominator_rate": float(
                        (raw_denominator < 0).float().mean().item()
                    ),
                    "raw_clamp_rate": float(raw_clamped.float().mean().item()),
                    "elu_clamp_rate": float(elu_clamped.float().mean().item()),
                    "raw_context_relative_error": float(
                        (
                            (raw_context - reference_context).norm()
                            / reference_context.norm().clamp_min(1e-12)
                        ).item()
                    ),
                    "elu_context_relative_error": float(
                        (
                            (elu_context - reference_context).norm()
                            / reference_context.norm().clamp_min(1e-12)
                        ).item()
                    ),
                    "raw_denominator_min": float(raw_denominator.min().item()),
                    "raw_denominator_median": float(raw_denominator.median().item()),
                    "elu_denominator_min": float(elu_denominator.min().item()),
                    "elu_denominator_median": float(elu_denominator.median().item()),
                }
            )

    summary = {
        key: {
            "mean": float(np.nanmean([record[key] for record in records])),
            "std": float(np.nanstd([record[key] for record in records], ddof=1)),
        }
        for key in records[0]
        if key not in {"seed", "example"}
    }
    write_json(output_dir / "kernel_gap_records.json", records)
    write_json(output_dir / "kernel_gap_summary.json", summary)
    try:
        import matplotlib.pyplot as plt

        names = [
            "raw_relative_frobenius",
            "elu_relative_frobenius",
            "raw_context_relative_error",
            "elu_context_relative_error",
        ]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(
            range(len(names)),
            [summary[name]["mean"] for name in names],
            yerr=[summary[name]["std"] for name in names],
            capsize=3,
        )
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "kernel_gap.png", dpi=180)
        figure.savefig(output_dir / "kernel_gap.pdf")
        plt.close(figure)
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--diagnostic-examples", type=int, default=100)
    args = parser.parse_args()
    variants = [
        {
            "name": "raw_rff",
            "model_name": "fska",
            "feature_map": "raw_rff",
            "density_mode": "learned_flow",
            "qk_mode": "identity",
            "kl_weight": 1e-3,
        },
        {
            "name": "elu_plus_one",
            "model_name": "fska",
            "feature_map": "elu_plus_one",
            "density_mode": "learned_flow",
            "qk_mode": "identity",
            "kl_weight": 1e-3,
        },
    ]
    status = run_sst_grid(args, "elu_gap", variants)
    if status == 0 and not args.smoke:
        kernel_diagnostics(
            args.output_dir,
            args.data_dir,
            parse_seeds(args.seeds),
            torch.device(args.device),
            args.diagnostic_examples,
            args.revision,
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
