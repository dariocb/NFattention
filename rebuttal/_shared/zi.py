"""Protocol-reconstruction helpers for the colleague PDF (``*_zi`` suite).

The document gives result tables but omits dataset revisions/splits, tokenizer,
and training hyperparameters.  These helpers therefore make every missing
choice CLI-visible and write it to the manifest; they must not be described as
an exact reproduction until those choices are supplied by the colleague.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch

from .classifier import ModelConfig
from .datasets import load_hf_text_classification, make_loaders, synthetic_bundle
from .grid import _plot_summary, parse_seeds, resolve_device
from .metrics import aggregate_runs
from .results import ResultWriter, write_json
from .training import TrainConfig, run_training


def add_zi_arguments(parser: argparse.ArgumentParser, *, default_dataset: str = "SetFit/20_newsgroups") -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("rebuttal/data"))
    parser.add_argument("--dataset-id", default=default_dataset)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    # Explicit reconstruction assumptions; the PDF itself does not state them.
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)


def run_zi_grid(args: argparse.Namespace, experiment: str, variants: Sequence[Mapping[str, object]]) -> int:
    seeds = parse_seeds(args.seeds)
    if args.smoke:
        seeds, bundle, max_length, epochs, patience = (seeds[:1] or [0], synthetic_bundle(), 16, 1, 1)
    else:
        bundle = load_hf_text_classification(
            args.dataset_id, args.data_dir, config_name=args.dataset_config,
            revision=args.dataset_revision, text_column=args.text_column,
            label_column=args.label_column, validation_split=args.validation_split,
            test_split=args.test_split,
        )
        max_length, epochs, patience = args.max_length, args.epochs, args.patience
    device = resolve_device(args.device)
    config = {"experiment": experiment, "protocol_status": "reconstruction; PDF omits dataset revision/splits/tokenizer/training details", "dataset": bundle.manifest(), "seeds": seeds, "variants": [dict(v) for v in variants], "training": {"max_length": max_length, "epochs": epochs, "patience": patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "hidden_dim": args.hidden_dim, "n_heads": args.n_heads, "n_layers": args.n_layers, "ff_dim": args.ff_dim, "dropout": args.dropout}}
    writer, runs = ResultWriter(args.output_dir, config), []
    (args.output_dir / "histories").mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for seed in seeds:
            context = {"variant": str(variant["name"]), "model": str(variant.get("model_name", "fska")), "seed": seed}
            try:
                loaders = make_loaders(bundle, max_length, args.batch_size, seed, min_frequency=1 if args.smoke else 2)
                mc = ModelConfig(model_name=context["model"], vocab_size=len(loaders.vocabulary.token_to_id), num_classes=bundle.num_classes, max_length=max_length, hidden_dim=32 if args.smoke else args.hidden_dim, n_heads=4 if args.smoke else args.n_heads, n_layers=1 if args.smoke else args.n_layers, ff_dim=64 if args.smoke else args.ff_dim, dropout=0.0 if args.smoke else args.dropout, feature_width=int(variant.get("feature_width", 128)), num_spectral_pairs=int(variant.get("num_spectral_pairs", 64)), num_flows=int(variant.get("num_flows", 3)), density_mode=str(variant.get("density_mode", "learned_flow")), feature_map=str(variant.get("feature_map", "elu_plus_one")), qk_mode=str(variant.get("qk_mode", "identity")), kl_weight=float(variant.get("kl_weight", 1e-3)), pooling="mean", sample_seed=1729 + seed)
                tc = TrainConfig(learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_epochs=epochs, patience=patience, selection_metrics=("loss", "accuracy", "macro_f1"))
                model, result = run_training(mc, tc, loaders, device, seed, bundle.num_classes)
                primary = result["selections"]["loss"]["test"]
                record: Dict[str, object] = {**context, "accuracy": primary["accuracy"], "macro_f1": primary["macro_f1"], "balanced_accuracy": primary["balanced_accuracy"], "train_time": result["train_time"], "best_step": result["selections"]["loss"]["best_step"], "num_trainable_parameters": model.count_parameters(True), "num_total_parameters": model.count_parameters(False), "model_config": asdict(mc), "per_class": primary["per_class"], "density_checksums_initial": result["density_checksums_initial"], "density_checksums_final": result["density_checksums_final"]}
                runs.append(record); writer.add_run(record)
                write_json(args.output_dir / "histories" / f"{context['variant']}__seed{seed}.json", result["history"])
            except Exception as error:
                writer.add_failure(context, error)
    summary = aggregate_runs(runs, ["variant", "model"])
    writer.write_summary(summary); writer.write_latex(summary, ["variant", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]); _plot_summary(args.output_dir, summary); writer.finish()
    return 0 if len(runs) == len(variants) * len(seeds) and not writer.failures else 1
