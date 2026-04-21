#!/usr/bin/env python3
"""
Hugging Face Transformer benchmark for text sequence classification.

This script lives in a separate experiments folder but reuses:
- text benchmark dataset loader
- shared result logger/util helpers
"""

import os
import re
import sys
import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
TEXT_BENCH_DIR = os.path.join(
    REPO_ROOT,
    "experiments",
    "text_classification",
    "text_seq_classification_benchmark",
)
if TEXT_BENCH_DIR not in sys.path:
    sys.path.insert(0, TEXT_BENCH_DIR)

from data import load_dataset_by_name, get_available_datasets  # type: ignore
from utils import (  # type: ignore
    set_seed,
    get_device,
    save_config,
    ResultLogger,
    ExperimentResult,
    format_time,
    setup_logging,
    write_repro_manifest,
)


MODEL_PRESETS: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}


@dataclass
class HFRunSpec:
    key: str
    model_id: str


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def resolve_hf_models(model_args: List[str]) -> List[HFRunSpec]:
    specs: List[HFRunSpec] = []
    for item in model_args:
        if item in MODEL_PRESETS:
            model_id = MODEL_PRESETS[item]
            key = f"hf_{item}"
        else:
            model_id = item
            key = f"hf_{_sanitize_name(item)}"
        specs.append(HFRunSpec(key=key, model_id=model_id))
    return specs


def _make_compute_metrics(num_classes: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        if num_classes > 2:
            f1 = f1_score(labels, preds, average="macro")
        else:
            f1 = f1_score(labels, preds, average="binary")
        return {"accuracy": acc, "f1_macro": f1}

    return compute_metrics


def run_single_hf_experiment(
    dataset_name: str,
    model_spec: HFRunSpec,
    seed: int,
    data_dir: str,
    output_dir: str,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    gradient_accumulation_steps: int,
    early_stopping_patience: int,
    fp16: bool,
    logger=None,
) -> ExperimentResult:
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    set_seed(seed)

    if logger:
        logger.info(
            f"[HF] dataset={dataset_name}, model={model_spec.key} ({model_spec.model_id}), seed={seed}"
        )

    ds = load_dataset_by_name(dataset_name, data_dir)
    train_texts, train_labels = ds["train"]
    val_texts, val_labels = ds["val"]
    test_texts, test_labels = ds["test"]
    num_classes = ds["info"].num_classes

    tokenizer = AutoTokenizer.from_pretrained(model_spec.model_id, use_fast=True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_hf = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    val_hf = Dataset.from_dict({"text": val_texts, "labels": val_labels})
    test_hf = Dataset.from_dict({"text": test_texts, "labels": test_labels})

    train_hf = train_hf.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_hf = val_hf.map(tokenize_batch, batched=True, remove_columns=["text"])
    test_hf = test_hf.map(tokenize_batch, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_spec.model_id,
        num_labels=num_classes,
    )

    run_name = f"{dataset_name}__{model_spec.key}__seed{seed}"
    run_output = os.path.join(output_dir, "hf_checkpoints", run_name)
    os.makedirs(run_output, exist_ok=True)

    args = TrainingArguments(
        output_dir=run_output,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        seed=seed,
        report_to=[],
        fp16=fp16,
        save_total_limit=1,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        compute_metrics=_make_compute_metrics(num_classes),
        callbacks=callbacks,
    )

    start = time.time()
    train_output = trainer.train()
    train_time = time.time() - start

    test_metrics = trainer.predict(test_hf).metrics

    best_ckpt = trainer.state.best_model_checkpoint
    best_epoch = 0
    if best_ckpt:
        m = re.search(r"checkpoint-(\d+)$", best_ckpt)
        if m:
            best_epoch = int(m.group(1))

    num_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

    return ExperimentResult(
        dataset=dataset_name,
        model=model_spec.key,
        seed=seed,
        fold=None,
        accuracy=float(test_metrics.get("test_accuracy", 0.0)),
        f1_macro=float(test_metrics.get("test_f1_macro", 0.0)),
        train_loss=float(train_output.training_loss if train_output.training_loss is not None else 0.0),
        val_loss=None,
        test_loss=float(test_metrics.get("test_loss", 0.0)),
        train_time=train_time,
        num_params=num_params,
        best_epoch=best_epoch,
        config={
            "hf_model_id": model_spec.model_id,
            "max_length": max_length,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "early_stopping_patience": early_stopping_patience,
            "fp16": fp16,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run Hugging Face Transformer benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["trec", "ag_news", "dbpedia_14", "yelp_review_full"],
        help="Datasets to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert", "bert", "roberta"],
        help="Model presets or HF model ids",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "experiments", "text_classification", "data"),
        help="Data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "benchmark_results_hf"),
        help="Output root directory",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max sequence length")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Train batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Eval batch size per device")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Early stopping patience (0 disables)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (trec + distilbert + seed=0)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 even on CUDA")
    args = parser.parse_args()

    available = set(get_available_datasets())
    for d in args.datasets:
        if d not in available:
            raise ValueError(f"Unsupported dataset: {d}. Available: {sorted(available)}")

    hf_models = resolve_hf_models(args.models)
    device = get_device(args.gpu)
    fp16 = torch.cuda.is_available() and (device.type == "cuda") and (not args.no_fp16)

    if args.quick:
        args.datasets = ["trec"]
        hf_models = resolve_hf_models(["distilbert"])
        args.seeds = [0]
        args.epochs = min(args.epochs, 1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(run_dir, "hf_benchmark")

    config_snapshot: Dict[str, Any] = {
        "datasets": args.datasets,
        "models": [m.key for m in hf_models],
        "model_ids": {m.key: m.model_id for m in hf_models},
        "seeds": args.seeds,
        "data_dir": args.data_dir,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "fp16": fp16,
        "device": str(device),
    }
    save_config(config_snapshot, os.path.join(run_dir, "config.yaml"))

    write_repro_manifest(
        output_dir=run_dir,
        config=config_snapshot,
        datasets=args.datasets,
        models=[m.key for m in hf_models],
        seeds=args.seeds,
        data_dir=args.data_dir,
        argv=list(sys.argv),
        benchmark_dir=CURRENT_DIR,
    )

    result_logger = ResultLogger(run_dir)
    total = len(args.datasets) * len(hf_models) * len(args.seeds)
    logger.info("=" * 72)
    logger.info("Hugging Face Transformer Benchmark")
    logger.info(f"Total experiments: {total}")
    logger.info(f"Device: {device}, fp16={fp16}")
    logger.info("=" * 72)

    idx = 0
    global_start = time.time()
    for dataset_name in args.datasets:
        for model_spec in hf_models:
            for seed in args.seeds:
                idx += 1
                logger.info(
                    f"[{idx}/{total}] dataset={dataset_name}, model={model_spec.key}, seed={seed}"
                )
                try:
                    result = run_single_hf_experiment(
                        dataset_name=dataset_name,
                        model_spec=model_spec,
                        seed=seed,
                        data_dir=args.data_dir,
                        output_dir=run_dir,
                        max_length=args.max_length,
                        train_batch_size=args.train_batch_size,
                        eval_batch_size=args.eval_batch_size,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        warmup_ratio=args.warmup_ratio,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        early_stopping_patience=args.early_stopping_patience,
                        fp16=fp16,
                        logger=logger,
                    )
                    result_logger.add_result(result)
                    logger.info(
                        f"done: acc={result.accuracy*100:.2f}%, f1={result.f1_macro*100:.2f}%, "
                        f"time={format_time(result.train_time)}"
                    )
                except Exception as e:
                    logger.exception(
                        f"failed: dataset={dataset_name}, model={model_spec.key}, seed={seed}, error={e}"
                    )

    elapsed = time.time() - global_start
    result_logger.save_csv("all_results.csv")
    result_logger.save_results("all_results.json")
    result_logger.save_summary_csv("summary.csv")
    result_logger.print_summary()

    logger.info("=" * 72)
    logger.info(f"Benchmark complete in {format_time(elapsed)}")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
