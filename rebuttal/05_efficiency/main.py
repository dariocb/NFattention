#!/usr/bin/env python
"""Attention-only latency, throughput, memory, and FSKA component benchmark."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from rebuttal._shared.baselines import build_attention
from rebuttal._shared.fska import FSKAConfig, FSKAAttention
from rebuttal._shared.grid import resolve_device
from rebuttal._shared.results import ResultWriter


def _measure(operation, device: torch.device) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    started = time.perf_counter()
    operation()
    return (time.perf_counter() - started) * 1000.0


def _summary(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "q1_ms": float(np.quantile(array, 0.25)),
        "q3_ms": float(np.quantile(array, 0.75)),
        "iqr_ms": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
    }


@torch.no_grad()
def _fska_components(
    attention: FSKAAttention, x: torch.Tensor, mask: torch.Tensor,
    device: torch.device, repeats: int
) -> Dict[str, float]:
    results = {"sampling": [], "features": [], "contraction": [], "projection": []}
    for _ in range(repeats):
        holder: Dict[str, object] = {}

        def sampling():
            holder["omega"] = attention.sample_omegas(
                x.shape[0], dtype=x.dtype
            )

        results["sampling"].append(_measure(sampling, device))
        q = attention._split_heads(x)
        values = torch.einsum("bld,hde->bhle", x, attention.fixed_v)
        omega1, omega2 = holder["omega"]

        def features():
            raw_q = attention.rff_features(q, omega1, omega2)
            raw_k = attention.rff_features(q, omega1, omega2)
            holder["phi_q"] = attention._map_features(raw_q)
            holder["phi_k"] = attention._map_features(raw_k)

        results["features"].append(_measure(features, device))

        def contraction():
            holder["context"] = attention.linear_context(
                holder["phi_q"], holder["phi_k"], values, mask
            )[0]

        results["contraction"].append(_measure(contraction, device))

        def projection():
            context = holder["context"].permute(0, 2, 1, 3).contiguous().view(
                x.shape[0], x.shape[1], attention.hidden_dim
            )
            attention.out_proj(context)

        results["projection"].append(_measure(projection, device))
    return {
        f"{name}_median_ms": float(np.median(values))
        for name, values in results.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lengths", default="256,512,1024,2000,4096")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    lengths = [int(value) for value in args.lengths.split(",")]
    if args.smoke:
        lengths, args.warmups, args.blocks, args.iterations = [32], 1, 1, 2

    config = {
        "experiment": "efficiency",
        "device": str(device),
        "batch_size": 1,
        "hidden_dim": 512,
        "n_heads": 8,
        "feature_width": 128,
        "lengths": lengths,
        "warmups": args.warmups,
        "blocks": args.blocks,
        "iterations": args.iterations,
        "dtype": "float32",
    }
    writer = ResultWriter(args.output_dir, config)
    rows: List[Dict[str, object]] = []
    models = ["transformer", "performer", "rka", "fska"]
    for model_name in models:
        for length in lengths:
            context = {"model": model_name, "length": length}
            try:
                if model_name == "fska":
                    attention = FSKAAttention(
                        FSKAConfig(
                            hidden_dim=512,
                            n_heads=8,
                            num_spectral_pairs=64,
                            dropout=0.0,
                            kl_weight=1e-3,
                        )
                    )
                else:
                    attention = build_attention(
                        model_name, 512, 8, 128, 0.0, 1729
                    )
                attention.to(device)
                x = torch.randn(1, length, 512, device=device)
                mask = torch.ones(1, length, dtype=torch.bool, device=device)

                attention.eval()
                for _ in range(args.warmups):
                    with torch.no_grad():
                        attention(x, mask)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                inference_blocks = []
                def inference_step():
                    with torch.no_grad():
                        attention(x, mask)
                for _ in range(args.blocks):
                    timings = []
                    for _ in range(args.iterations):
                        timings.append(
                            _measure(
                                inference_step,
                                device,
                            )
                        )
                    inference_blocks.append(float(np.median(timings)))
                inference = _summary(inference_blocks)
                peak_allocated = (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                )
                peak_reserved = (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                )

                attention.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for parameter in attention.parameters():
                    parameter.grad = None

                def training_step():
                    result = attention(x, mask)
                    loss = result.output.square().mean() + 1e-3 * result.kl_raw
                    loss.backward()
                    for parameter in attention.parameters():
                        parameter.grad = None

                for _ in range(max(1, args.warmups // 4)):
                    training_step()
                training_blocks = []
                for _ in range(args.blocks):
                    timings = [
                        _measure(training_step, device)
                        for _ in range(args.iterations)
                    ]
                    training_blocks.append(float(np.median(timings)))
                training = _summary(training_blocks)
                training_peak_allocated = (
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                )
                training_peak_reserved = (
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                )
                row: Dict[str, object] = {
                    **context,
                    **{f"inference_{key}": value for key, value in inference.items()},
                    **{f"training_{key}": value for key, value in training.items()},
                    "inference_tokens_per_second": 1000.0
                    * length
                    / inference["median_ms"],
                    "training_tokens_per_second": 1000.0
                    * length
                    / training["median_ms"],
                    "peak_allocated_bytes": int(peak_allocated),
                    "peak_reserved_bytes": int(peak_reserved),
                    "training_peak_allocated_bytes": int(training_peak_allocated),
                    "training_peak_reserved_bytes": int(training_peak_reserved),
                    "trainable_parameters": sum(
                        p.numel() for p in attention.parameters() if p.requires_grad
                    ),
                    "total_parameters": sum(p.numel() for p in attention.parameters()),
                }
                if isinstance(attention, FSKAAttention):
                    attention.eval()
                    row.update(
                        _fska_components(
                            attention,
                            x,
                            mask,
                            device,
                            max(3, min(20, args.iterations)),
                        )
                    )
                rows.append(row)
                writer.add_run(row)
            except Exception as error:
                writer.add_failure(context, error)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    writer.write_summary(rows)
    writer.write_latex(
        rows,
        [
            "model",
            "length",
            "inference_median_ms",
            "training_median_ms",
            "inference_tokens_per_second",
            "peak_allocated_bytes",
        ],
    )
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 4.5))
        for model_name in models:
            selected = [row for row in rows if row["model"] == model_name]
            axis.plot(
                [row["length"] for row in selected],
                [row["inference_tokens_per_second"] for row in selected],
                marker="o",
                label=model_name,
            )
        axis.set_xlabel("Sequence length")
        axis.set_ylabel("Inference tokens/s")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / "throughput.png", dpi=180)
        figure.savefig(args.output_dir / "throughput.pdf")
        plt.close(figure)
    except ImportError:
        pass
    writer.finish()
    return 0 if len(rows) == len(models) * len(lengths) and not writer.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
