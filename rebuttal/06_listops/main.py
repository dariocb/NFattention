#!/usr/bin/env python
"""Official-protocol ListOps quality and FSKA capacity/QK ablations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from rebuttal._shared.classifier import ModelConfig
from rebuttal._shared.datasets import (
    DatasetBundle,
    load_listops,
    make_loaders,
    synthetic_bundle,
)
from rebuttal._shared.grid import parse_seeds, resolve_device
from rebuttal._shared.metrics import aggregate_runs
from rebuttal._shared.results import ResultWriter, write_json
from rebuttal._shared.training import TrainConfig, run_training


def _length_metrics(test: Mapping[str, object]) -> Dict[str, object]:
    bins = [
        ("0-512", 0, 512),
        ("513-1024", 513, 1024),
        ("1025-1536", 1025, 1536),
        ("1537-2000", 1537, 2000),
    ]
    lengths = np.asarray(test["lengths"])
    labels = np.asarray(test["labels"])
    predictions = np.asarray(test["predictions"])
    output: Dict[str, object] = {}
    for name, lower, upper in bins:
        selected = (lengths >= lower) & (lengths <= upper)
        output[name] = {
            "count": int(selected.sum()),
            "accuracy": (
                float((labels[selected] == predictions[selected]).mean())
                if selected.any()
                else None
            ),
        }
    return output


def _dataset_length_stats(bundle: DatasetBundle, max_length: int) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for name, split in (
        ("train", bundle.train),
        ("validation", bundle.validation),
        ("test", bundle.test),
    ):
        lengths = np.asarray([len(text.split()) for text in split.texts])
        output[name] = {
            "p50": float(np.quantile(lengths, 0.5)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": int(lengths.max()),
            "truncation_rate": float((lengths > max_length).mean()),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("rebuttal/data/listops"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-seeds", default="0,1,2,3,4")
    parser.add_argument("--ablation-seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    primary_seeds = parse_seeds(args.primary_seeds)
    ablation_seeds = parse_seeds(args.ablation_seeds)
    if args.smoke:
        bundle = synthetic_bundle(num_classes=10, train_size=40)
        primary_seeds = primary_seeds[:1] or [0]
        ablation_seeds = ablation_seeds[:1] or [0]
        max_length, hidden_dim, n_layers, ff_dim = 32, 32, 1, 64
        args.steps, args.microbatch, args.gradient_accumulation = 2, 4, 1
        args.eval_every = 1
    else:
        bundle = load_listops(args.data_dir, download=args.download)
        max_length, hidden_dim, n_layers, ff_dim = 2000, 512, 4, 1024

    device = resolve_device(args.device)
    primary_variants = [
        {"name": "transformer", "model_name": "transformer"},
        {"name": "performer", "model_name": "performer"},
        {"name": "rka", "model_name": "rka"},
        {"name": "fska_main", "model_name": "fska"},
    ]
    ablation_variants = [
        {
            "name": "fska_fixed_bivariate",
            "model_name": "fska",
            "density_mode": "fixed_bivariate",
            "kl_weight": 0.0,
        },
        {
            "name": "fska_raw_rff",
            "model_name": "fska",
            "feature_map": "raw_rff",
        },
        {
            "name": "fska_learned_qk",
            "model_name": "fska",
            "qk_mode": "learned",
        },
        {
            "name": "fska_m32",
            "model_name": "fska",
            "num_spectral_pairs": 32,
            "feature_width": 64,
        },
        {
            "name": "fska_m128",
            "model_name": "fska",
            "num_spectral_pairs": 128,
            "feature_width": 256,
        },
    ]
    config = {
        "experiment": "listops",
        "dataset": bundle.manifest(),
        "length_statistics": _dataset_length_stats(bundle, max_length),
        "official_protocol": {
            "max_length": max_length,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "n_heads": 8 if not args.smoke else 4,
            "ff_dim": ff_dim,
            "effective_batch": args.microbatch * args.gradient_accumulation,
            "optimizer_steps": args.steps,
            "learning_rate": 0.05,
            "warmup_steps": 1000,
            "weight_decay": 0.1,
        },
        "primary_seeds": primary_seeds,
        "ablation_seeds": ablation_seeds,
        "primary_variants": primary_variants,
        "ablation_variants": ablation_variants,
        "smoke": args.smoke,
    }
    writer = ResultWriter(args.output_dir, config)
    runs: List[Dict[str, object]] = []
    (args.output_dir / "histories").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    grids = [
        ("primary", primary_variants, primary_seeds),
        ("ablation", ablation_variants, ablation_seeds),
    ]
    for grid_name, variants, seeds in grids:
        for variant in variants:
            for seed in seeds:
                context = {
                    "grid": grid_name,
                    "variant": variant["name"],
                    "seed": seed,
                }
                try:
                    loaders = make_loaders(
                        bundle,
                        max_length=max_length,
                        batch_size=args.microbatch,
                        seed=seed,
                        min_frequency=1,
                    )
                    model_config = ModelConfig(
                        model_name=str(variant["model_name"]),
                        vocab_size=len(loaders.vocabulary.token_to_id),
                        num_classes=bundle.num_classes,
                        max_length=max_length,
                        hidden_dim=hidden_dim,
                        n_heads=4 if args.smoke else 8,
                        n_layers=n_layers,
                        ff_dim=ff_dim,
                        dropout=0.0 if args.smoke else 0.1,
                        feature_width=int(variant.get("feature_width", 128)),
                        num_spectral_pairs=int(
                            variant.get("num_spectral_pairs", 64)
                        ),
                        density_mode=str(
                            variant.get("density_mode", "learned_flow")
                        ),
                        feature_map=str(
                            variant.get("feature_map", "elu_plus_one")
                        ),
                        qk_mode=str(variant.get("qk_mode", "identity")),
                        kl_weight=float(variant.get("kl_weight", 1e-3)),
                        pooling="cls",
                        sample_seed=1729 + seed,
                        gradient_checkpointing=not args.smoke,
                    )
                    train_config = TrainConfig(
                        learning_rate=0.05,
                        weight_decay=0.1,
                        gradient_accumulation=args.gradient_accumulation,
                        max_steps=args.steps,
                        eval_every=args.eval_every,
                        warmup_steps=1000,
                        mixed_precision=not args.smoke,
                        optimizer="listops_adam",
                        selection_metrics=("accuracy",),
                    )
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    model, result = run_training(
                        model_config,
                        train_config,
                        loaders,
                        device,
                        seed,
                        bundle.num_classes,
                    )
                    test = result["selections"]["accuracy"]["test"]
                    peak_allocated = (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    )
                    record: Dict[str, object] = {
                        **context,
                        "accuracy": test["accuracy"],
                        "macro_f1": test["macro_f1"],
                        "balanced_accuracy": test["balanced_accuracy"],
                        "train_time": result["train_time"],
                        "training_tokens_per_second": (
                            args.steps
                            * args.microbatch
                            * args.gradient_accumulation
                            * max_length
                            / max(float(result["train_time"]), 1e-12)
                        ),
                        "best_step": result["selections"]["accuracy"]["best_step"],
                        "length_bin_metrics": _length_metrics(test),
                        "peak_allocated_bytes": int(peak_allocated),
                        "num_trainable_parameters": model.count_parameters(True),
                        "num_total_parameters": model.count_parameters(False),
                        "model_config": asdict(model_config),
                        "density_checksums_initial": result[
                            "density_checksums_initial"
                        ],
                        "density_checksums_final": result[
                            "density_checksums_final"
                        ],
                        "prior_checksums_initial": result[
                            "prior_checksums_initial"
                        ],
                        "prior_checksums_final": result[
                            "prior_checksums_final"
                        ],
                    }
                    runs.append(record)
                    writer.add_run(record)
                    write_json(
                        args.output_dir
                        / "histories"
                        / f"{variant['name']}__seed{seed}.json",
                        result["history"],
                    )
                    write_json(
                        args.output_dir
                        / "predictions"
                        / f"{variant['name']}__seed{seed}.json",
                        {
                            key: test[key]
                            for key in (
                                "indices",
                                "labels",
                                "predictions",
                                "probabilities",
                                "lengths",
                            )
                        },
                    )
                    torch.save(
                        {
                            "state_dict": result["best_state"],
                            "model_config": asdict(model_config),
                            "vocabulary": loaders.vocabulary.to_dict(),
                            "seed": seed,
                        },
                        args.output_dir
                        / "checkpoints"
                        / f"{variant['name']}__seed{seed}.pt",
                    )
                except Exception as error:
                    writer.add_failure(context, error)
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    summary = aggregate_runs(runs, ["grid", "variant"])
    writer.write_summary(summary)
    writer.write_latex(
        summary,
        [
            "grid",
            "variant",
            "accuracy_mean",
            "accuracy_std",
            "train_time_mean",
        ],
    )
    try:
        import matplotlib.pyplot as plt

        capacity_rows = [
            (32, [run["accuracy"] for run in runs if run["variant"] == "fska_m32"]),
            (64, [run["accuracy"] for run in runs if run["variant"] == "fska_main"]),
            (128, [run["accuracy"] for run in runs if run["variant"] == "fska_m128"]),
        ]
        figure, axis = plt.subplots(figsize=(5.5, 4))
        axis.errorbar(
            [row[0] for row in capacity_rows],
            [np.mean(row[1]) if row[1] else np.nan for row in capacity_rows],
            yerr=[
                np.std(row[1], ddof=1) if len(row[1]) > 1 else 0
                for row in capacity_rows
            ],
            marker="o",
            capsize=3,
        )
        axis.set_xlabel("Spectral pairs (M)")
        axis.set_ylabel("ListOps accuracy")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output_dir / "capacity.png", dpi=180)
        figure.savefig(args.output_dir / "capacity.pdf")
        plt.close(figure)
    except ImportError:
        pass
    writer.finish()
    expected = len(primary_variants) * len(primary_seeds) + len(
        ablation_variants
    ) * len(ablation_seeds)
    return 0 if len(runs) == expected and not writer.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
