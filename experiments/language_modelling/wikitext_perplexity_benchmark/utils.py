"""Utilities for the WikiText benchmark."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(gpu_id: Optional[int] = None, allow_cpu_fallback: bool = True) -> torch.device:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU device {device} for training and evaluation.")
            if cuda_visible:
                print(f"CUDA_VISIBLE_DEVICES={cuda_visible} (requested index is relative to visible devices).")
            return device
        message = (
            f"GPU requested (gpu_id={gpu_id}) but CUDA is unavailable in this environment."
        )
        if cuda_visible:
            message += f" CUDA_VISIBLE_DEVICES={cuda_visible}."
        if allow_cpu_fallback:
            print(message + " Falling back to CPU.")
            return torch.device("cpu")
        raise RuntimeError(message)

    if torch.cuda.is_available():
        print("Using GPU device cuda:0 for training and evaluation.")
        if cuda_visible:
            print(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
        return torch.device("cuda")

    if allow_cpu_fallback:
        print("Using CPU for training and evaluation.")
        return torch.device("cpu")

    raise RuntimeError("CUDA is unavailable and CPU fallback is disabled.")


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ExperimentResult:
    dataset: str
    model: str
    seed: int
    train_loss: float
    val_loss: float
    test_loss: float
    train_ppl: float
    val_ppl: float
    test_ppl: float
    train_time: float
    num_params: int
    best_epoch: int
    config: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[ExperimentResult] = []

    def add_result(self, result: ExperimentResult) -> None:
        self.results.append(result)

    def save_results(self, filename: str = "results.json") -> str:
        path = os.path.join(self.output_dir, filename)
        save_json(path, [r.to_dict() for r in self.results])
        return path

    def save_csv(self, filename: str = "results.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        if not self.results:
            return path
        fieldnames = [k for k in self.results[0].to_dict().keys() if k != "config"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                row = result.to_dict()
                row.pop("config", None)
                writer.writerow(row)
        return path

    def save_summary_csv(self, filename: str = "summary.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        if not self.results:
            return path
        from collections import defaultdict

        grouped = defaultdict(list)
        for result in self.results:
            grouped[(result.dataset, result.model)].append(result)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "runs",
                    "train_loss_mean",
                    "train_loss_std",
                    "val_loss_mean",
                    "val_loss_std",
                    "test_loss_mean",
                    "test_loss_std",
                    "train_ppl_mean",
                    "train_ppl_std",
                    "val_ppl_mean",
                    "val_ppl_std",
                    "test_ppl_mean",
                    "test_ppl_std",
                    "train_time_mean",
                    "train_time_std",
                    "num_params",
                    "best_epoch_mean",
                ]
            )
            for (dataset, model), runs in grouped.items():
                def _vals(attr):
                    return np.array([getattr(r, attr) for r in runs], dtype=float)

                writer.writerow(
                    [
                        dataset,
                        model,
                        len(runs),
                        _vals("train_loss").mean(),
                        _vals("train_loss").std(),
                        _vals("val_loss").mean(),
                        _vals("val_loss").std(),
                        _vals("test_loss").mean(),
                        _vals("test_loss").std(),
                        _vals("train_ppl").mean(),
                        _vals("train_ppl").std(),
                        _vals("val_ppl").mean(),
                        _vals("val_ppl").std(),
                        _vals("test_ppl").mean(),
                        _vals("test_ppl").std(),
                        _vals("train_time").mean(),
                        _vals("train_time").std(),
                        runs[0].num_params,
                        _vals("best_epoch").mean(),
                    ]
                )
        return path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
