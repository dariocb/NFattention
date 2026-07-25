"""Training/evaluation loops shared by the standalone rebuttal experiments."""

from __future__ import annotations

import copy
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .classifier import ModelConfig, SequenceClassifier, build_model
from .datasets import LoaderBundle
from .metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 10
    gradient_clip: float = 1.0
    gradient_accumulation: int = 1
    max_steps: Optional[int] = None
    eval_every: int = 50
    warmup_steps: int = 1000
    mixed_precision: bool = False
    optimizer: str = "adam"
    selection_metrics: Tuple[str, ...] = ("loss", "accuracy", "macro_f1")
    # When set, fail at the first non-finite module boundary and persist an
    # actionable trace. This is deliberately opt-in: hooks add overhead and
    # are used for the ListOps numerical-stability audit.
    nonfinite_trace_path: Optional[str] = None


class NonFiniteTensorError(FloatingPointError):
    """Raised as soon as a traced training tensor becomes non-finite."""

    def __init__(self, record: Mapping[str, object]):
        self.record = dict(record)
        super().__init__(
            "Non-finite tensor at "
            f"{self.record['location']} ({self.record['kind']}, "
            f"shape={self.record['shape']}, dtype={self.record['dtype']})"
        )


def _tensor_leaves(value: object) -> Iterable[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensor_leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)
    elif hasattr(value, "__dict__"):
        yield from _tensor_leaves(vars(value))


class NonFiniteTensorTracker:
    """Forward-boundary tracer used to localise AMP/long-sequence failures."""

    def __init__(self, model: nn.Module, trace_path: str, step_getter):
        self.trace_path = trace_path
        self.step_getter = step_getter
        self.handles = []
        for name, module in model.named_modules():
            location = name or model.__class__.__name__
            self.handles.append(
                module.register_forward_pre_hook(
                    lambda _module, inputs, where=location: self._check(
                        inputs, where, "input"
                    )
                )
            )
            self.handles.append(
                module.register_forward_hook(
                    lambda _module, inputs, output, where=location: self._check(
                        output, where, "output"
                    )
                )
            )

    def _check(self, value: object, location: str, kind: str) -> None:
        for tensor in _tensor_leaves(value):
            if not tensor.is_floating_point() and not tensor.is_complex():
                continue
            if torch.isfinite(tensor).all():
                continue
            finite = tensor[torch.isfinite(tensor)]
            record = {
                "step": int(self.step_getter()),
                "location": location,
                "kind": kind,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
                "nonfinite_count": int((~torch.isfinite(tensor)).sum().item()),
                "finite_min": float(finite.min().item()) if finite.numel() else None,
                "finite_max": float(finite.max().item()) if finite.numel() else None,
            }
            path = Path(self.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            raise NonFiniteTensorError(record)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _clone_state(model: nn.Module) -> Dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _make_optimizer(model: nn.Module, config: TrainConfig):
    if config.optimizer == "listops_adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _listops_lr(step: int, config: TrainConfig) -> float:
    actual = max(step, 1)
    warmup = max(config.warmup_steps, 1)
    return config.learning_rate * min(1.0, actual / warmup) / math.sqrt(
        max(actual, warmup)
    )


@torch.no_grad()
def evaluate(
    model: SequenceClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels: List[int] = []
    predictions: List[int] = []
    probabilities: List[List[float]] = []
    indices: List[int] = []
    lengths: List[int] = []
    for input_ids, attention_mask, batch_labels, batch_lengths, batch_indices in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        output = model(input_ids, attention_mask)
        loss = F.cross_entropy(output.logits, batch_labels, reduction="sum")
        total_loss += float(loss.item())
        total_examples += len(batch_labels)
        probs = output.logits.float().softmax(dim=-1)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(probs.argmax(dim=-1).cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        indices.extend(batch_indices.tolist())
        lengths.extend(batch_lengths.tolist())
    metrics = classification_metrics(labels, predictions, num_classes)
    metrics.update(
        {
            "loss": total_loss / max(total_examples, 1),
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
            "indices": indices,
            "lengths": lengths,
        }
    )
    return metrics


def _is_better(metric: str, value: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    return value < best if metric == "loss" else value > best


def train_model(
    model: SequenceClassifier,
    loaders: LoaderBundle,
    train_config: TrainConfig,
    device: torch.device,
    num_classes: int,
    on_validation: Optional[Callable[[Dict[str, float], SequenceClassifier, bool], None]] = None,
) -> Dict[str, object]:
    model.to(device)
    optimizer = _make_optimizer(model, train_config)
    use_amp = train_config.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_values: Dict[str, Optional[float]] = {
        metric: None for metric in train_config.selection_metrics
    }
    best_states: Dict[str, Dict[str, Tensor]] = {}
    best_steps: Dict[str, int] = {}
    history: List[Dict[str, float]] = []
    global_step = 0
    stale_evaluations = 0
    started = time.perf_counter()
    stop = False
    tracker = (
        NonFiniteTensorTracker(
            model, train_config.nonfinite_trace_path, lambda: global_step
        )
        if train_config.nonfinite_trace_path
        else None
    )

    def record_validation(epoch_number: int) -> bool:
        nonlocal stale_evaluations
        validation = evaluate(model, loaders.validation, device, num_classes)
        row = {
            "epoch": float(epoch_number),
            "step": float(global_step),
            "train_cross_entropy": epoch_ce / max(epoch_examples, 1),
            "train_kl_raw": epoch_kl_raw / max(epoch_examples, 1),
            "train_weighted_kl": epoch_kl / max(epoch_examples, 1),
            "train_kl_ce_ratio": epoch_kl / max(epoch_ce, 1e-12),
            "val_loss": float(validation["loss"]),
            "val_accuracy": float(validation["accuracy"]),
            "val_macro_f1": float(validation["macro_f1"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        improved_primary = False
        for metric in train_config.selection_metrics:
            value = float(validation[metric])
            if _is_better(metric, value, best_values[metric]):
                best_values[metric] = value
                best_states[metric] = _clone_state(model)
                best_steps[metric] = global_step
                if metric == train_config.selection_metrics[0]:
                    improved_primary = True
        if on_validation is not None:
            on_validation(row, model, improved_primary)
        stale_evaluations = 0 if improved_primary else stale_evaluations + 1
        model.train()
        return improved_primary

    try:
        while not stop:
            epoch = int(history[-1]["epoch"]) + 1 if history and train_config.max_steps is None else 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_ce = 0.0
            epoch_kl_raw = 0.0
            epoch_kl = 0.0
            epoch_examples = 0
            for batch_index, (
            input_ids,
            attention_mask,
            labels,
            _lengths,
            _indices,
            ) in enumerate(loaders.train):
                model.set_sampling_step(global_step)
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                autocast_context = (
                    torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
                )
                with autocast_context:
                    output = model(input_ids, attention_mask)
                    ce_loss = F.cross_entropy(output.logits, labels)
                    weighted_kl = output.kl_raw * model.kl_weight
                    loss = (ce_loss + weighted_kl) / train_config.gradient_accumulation
                if not torch.isfinite(loss):
                    location = "training.loss"
                    record = {
                        "step": int(global_step), "location": location,
                        "kind": "loss", "shape": list(loss.shape),
                        "dtype": str(loss.dtype), "device": str(loss.device),
                        "numel": int(loss.numel()), "nonfinite_count": 1,
                        "finite_min": None, "finite_max": None,
                    }
                    if train_config.nonfinite_trace_path:
                        path = Path(train_config.nonfinite_trace_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    raise NonFiniteTensorError(record)
                scaler.scale(loss).backward()
                epoch_ce += float(ce_loss.detach().item()) * len(labels)
                epoch_kl_raw += float(output.kl_raw.detach().item()) * len(labels)
                epoch_kl += float(weighted_kl.detach().item()) * len(labels)
                epoch_examples += len(labels)

                boundary = (batch_index + 1) % train_config.gradient_accumulation == 0
                last_batch = batch_index + 1 == len(loaders.train)
                if boundary or last_batch:
                    if train_config.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), train_config.gradient_clip
                        )
                    if train_config.optimizer == "listops_adam":
                        next_step = global_step + 1
                        lr = _listops_lr(next_step, train_config)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    should_evaluate = train_config.max_steps is not None and (
                        global_step % train_config.eval_every == 0
                        or global_step >= train_config.max_steps
                    )
                    if should_evaluate:
                        record_validation(epoch)

                    if train_config.max_steps is not None and global_step >= train_config.max_steps:
                        stop = True
                        break

            if train_config.max_steps is None:
                record_validation(epoch)
                if stale_evaluations >= train_config.patience or epoch >= train_config.max_epochs:
                    stop = True
            elif not stop and len(loaders.train) == 0:
                raise RuntimeError("Empty training loader")
    finally:
        if tracker is not None:
            tracker.close()

    train_time = time.perf_counter() - started
    selections: Dict[str, object] = {}
    for metric, state in best_states.items():
        model.load_state_dict(state)
        test = evaluate(model, loaders.test, device, num_classes)
        selections[metric] = {
            "best_step": best_steps[metric],
            "validation_value": best_values[metric],
            "test": test,
        }
    primary_metric = train_config.selection_metrics[0]
    model.load_state_dict(best_states[primary_metric])
    return {
        "history": history,
        "selections": selections,
        "primary_selection": primary_metric,
        "train_time": train_time,
        "best_state": best_states[primary_metric],
        "global_steps": global_step,
    }


def run_training(
    model_config: ModelConfig,
    train_config: TrainConfig,
    loaders: LoaderBundle,
    device: torch.device,
    seed: int,
    num_classes: int,
    on_validation: Optional[Callable[[Dict[str, float], SequenceClassifier, bool], None]] = None,
) -> Tuple[SequenceClassifier, Dict[str, object]]:
    set_seed(seed)
    model = build_model(model_config)
    initial_density_checksums = [
        layer.attention.density.checksum()
        for layer in model.layers
        if hasattr(layer.attention, "density")
    ]
    initial_prior_checksums = [
        layer.attention.density.checksum(include_flow=False)
        for layer in model.layers
        if hasattr(layer.attention, "density")
    ]
    result = train_model(model, loaders, train_config, device, num_classes, on_validation)
    result["density_checksums_initial"] = initial_density_checksums
    result["density_checksums_final"] = [
        layer.attention.density.checksum()
        for layer in model.layers
        if hasattr(layer.attention, "density")
    ]
    result["prior_checksums_initial"] = initial_prior_checksums
    result["prior_checksums_final"] = [
        layer.attention.density.checksum(include_flow=False)
        for layer in model.layers
        if hasattr(layer.attention, "density")
    ]
    return model, result
