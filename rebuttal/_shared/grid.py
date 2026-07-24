"""Reusable CLI and grid execution for SST-5 rebuttal experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch

from .classifier import ModelConfig
from .datasets import DatasetBundle, load_sst5, make_loaders, synthetic_bundle
from .metrics import aggregate_runs
from .results import ResultWriter, write_json
from .training import TrainConfig, run_training


def parse_seeds(value: str) -> List[int]:
    return [
        int(item)
        for item in value.replace(",", " ").split()
        if item.strip()
    ]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=Path("rebuttal/data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--smoke", action="store_true")
    # Locked paper-text architecture. These values are never selected on test data.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)


def resolve_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {value}, but torch.cuda.is_available() is False"
        )
    return torch.device(value)


def _plot_summary(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    group_field: str = "variant",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [str(row[group_field]) for row in rows]
    accuracy = [float(row.get("accuracy_mean", float("nan"))) for row in rows]
    accuracy_error = [float(row.get("accuracy_std", 0.0)) for row in rows]
    macro_f1 = [float(row.get("macro_f1_mean", float("nan"))) for row in rows]
    macro_error = [float(row.get("macro_f1_std", 0.0)) for row in rows]
    positions = list(range(len(labels)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.3), 4.5))
    axis.bar(
        [position - width / 2 for position in positions],
        accuracy,
        width,
        yerr=accuracy_error,
        label="Accuracy",
        capsize=3,
    )
    axis.bar(
        [position + width / 2 for position in positions],
        macro_f1,
        width,
        yerr=macro_error,
        label="Macro-F1",
        capsize=3,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "metrics.png", dpi=180)
    figure.savefig(output_dir / "metrics.pdf")
    plt.close(figure)


def run_sst_grid(
    args: argparse.Namespace,
    experiment_name: str,
    variants: Sequence[Mapping[str, object]],
) -> int:
    seeds = parse_seeds(args.seeds)
    if args.smoke:
        seeds = seeds[:1] or [0]
        bundle = synthetic_bundle()
        max_length = 16
        epochs = 1
        patience = 1
    else:
        bundle = load_sst5(args.data_dir, revision=args.revision)
        max_length = 256
        epochs = args.epochs
        patience = args.patience

    device = resolve_device(args.device)
    config = {
        "experiment": experiment_name,
        "seeds": seeds,
        "smoke": args.smoke,
        "dataset": bundle.manifest(),
        "variants": [dict(variant) for variant in variants],
        "training": {
            "epochs": epochs,
            "patience": patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "architecture": {"hidden_dim": args.hidden_dim, "n_heads": args.n_heads, "n_layers": args.n_layers, "ff_dim": args.ff_dim, "dropout": args.dropout, "max_length": max_length},
        },
    }
    writer = ResultWriter(args.output_dir, config)
    runs: List[Dict[str, object]] = []
    predictions_dir = args.output_dir / "predictions"
    histories_dir = args.output_dir / "histories"
    checkpoints_dir = args.output_dir / "checkpoints"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        variant_name = str(variant["name"])
        model_name = str(variant.get("model_name", "fska"))
        for seed in seeds:
            context = {"variant": variant_name, "model": model_name, "seed": seed}
            try:
                loaders = make_loaders(
                    bundle,
                    max_length=max_length,
                    batch_size=args.batch_size,
                    seed=seed,
                    min_frequency=1 if args.smoke else 2,
                )
                model_config = ModelConfig(
                    model_name=model_name,
                    vocab_size=len(loaders.vocabulary.token_to_id),
                    num_classes=bundle.num_classes,
                    max_length=max_length,
                    hidden_dim=32 if args.smoke else args.hidden_dim,
                    n_heads=4 if args.smoke else args.n_heads,
                    n_layers=1 if args.smoke else args.n_layers,
                    ff_dim=64 if args.smoke else args.ff_dim,
                    dropout=0.0 if args.smoke else args.dropout,
                    feature_width=int(variant.get("feature_width", 128)),
                    num_spectral_pairs=int(variant.get("num_spectral_pairs", 64)),
                    density_mode=str(variant.get("density_mode", "learned_flow")),
                    feature_map=str(variant.get("feature_map", "elu_plus_one")),
                    qk_mode=str(variant.get("qk_mode", "identity")),
                    kl_weight=float(variant.get("kl_weight", 1e-3)),
                    pooling="mean",
                    sample_seed=1729 + seed,
                )
                train_config = TrainConfig(
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    max_epochs=epochs,
                    patience=patience,
                    selection_metrics=("loss", "accuracy", "macro_f1"),
                )
                model, result = run_training(
                    model_config,
                    train_config,
                    loaders,
                    device,
                    seed,
                    bundle.num_classes,
                )
                primary = result["selections"]["loss"]["test"]
                record: Dict[str, object] = {
                    **context,
                    "accuracy": primary["accuracy"],
                    "macro_f1": primary["macro_f1"],
                    "balanced_accuracy": primary["balanced_accuracy"],
                    "train_time": result["train_time"],
                    "best_step": result["selections"]["loss"]["best_step"],
                    "num_trainable_parameters": model.count_parameters(True),
                    "num_total_parameters": model.count_parameters(False),
                    "model_config": asdict(model_config),
                    "selection_audit": {
                        metric: {
                            "best_step": selection["best_step"],
                            "validation_value": selection["validation_value"],
                            "test_accuracy": selection["test"]["accuracy"],
                            "test_macro_f1": selection["test"]["macro_f1"],
                        }
                        for metric, selection in result["selections"].items()
                    },
                    "per_class": primary["per_class"],
                    "confusion_matrix": primary["confusion_matrix"],
                    "predicted_class_counts": primary["predicted_class_counts"],
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
                    histories_dir / f"{variant_name}__seed{seed}.json",
                    result["history"],
                )
                prediction_rows = [
                    {
                        "index": index,
                        "label": label,
                        "prediction": prediction,
                        "length": length,
                        "probabilities": probabilities,
                    }
                    for index, label, prediction, length, probabilities in zip(
                        primary["indices"],
                        primary["labels"],
                        primary["predictions"],
                        primary["lengths"],
                        primary["probabilities"],
                    )
                ]
                write_json(
                    predictions_dir / f"{variant_name}__seed{seed}.json",
                    prediction_rows,
                )
                torch.save(
                    {
                        "state_dict": result["best_state"],
                        "model_config": asdict(model_config),
                        "vocabulary": loaders.vocabulary.to_dict(),
                        "seed": seed,
                    },
                    checkpoints_dir / f"{variant_name}__seed{seed}.pt",
                )
            except Exception as error:
                writer.add_failure(context, error)

    summary = aggregate_runs(runs, ["variant", "model"])
    writer.write_summary(summary)
    writer.write_latex(
        summary,
        [
            "variant",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )
    _plot_summary(args.output_dir, summary)
    writer.finish()
    expected = len(variants) * len(seeds)
    return 0 if len(runs) == expected and not writer.failures else 1
