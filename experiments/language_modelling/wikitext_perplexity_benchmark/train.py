"""Training and evaluation for WikiText perplexity."""

from __future__ import annotations

import argparse
import math
import os
import time
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from .data import WikitextDataModule
    from .models import get_model
    from .utils import (
        ExperimentResult,
        ResultLogger,
        count_parameters,
        format_time,
        get_device,
        load_config,
        save_yaml,
        set_seed,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from data import WikitextDataModule  # type: ignore
    from models import get_model  # type: ignore
    from utils import ExperimentResult, ResultLogger, count_parameters, format_time, get_device, load_config, save_yaml, set_seed  # type: ignore


def _batch_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=pad_token_id,
        reduction="sum",
    )
    valid = (shift_labels != pad_token_id).sum().item()
    token_loss = loss / max(valid, 1)
    return token_loss, valid, shift_logits


@torch.no_grad()
def evaluate(model, dataloader, device, pad_token_id: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_tokens = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits, kl = model(input_ids, attention_mask)
        loss, valid, _ = _batch_loss(logits, input_ids, pad_token_id)
        total_loss += float(loss.item()) * max(valid, 1)
        total_kl += float(kl.item()) * max(valid, 1)
        total_tokens += max(valid, 1)
    avg_loss = total_loss / max(total_tokens, 1)
    avg_kl = total_kl / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss + avg_kl, 20.0))
    return {"loss": avg_loss, "kl": avg_kl, "ppl": ppl}


def _limited_loader(loader, max_batches: Optional[int]):
    if max_batches is None:
        yield from loader
        return
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        yield batch


def train_one_run(
    config: Dict[str, Any],
    model_name: str,
    seed: int,
    output_dir: str,
    gpu_id: Optional[int] = None,
    allow_cpu_fallback: bool = True,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    max_test_batches: Optional[int] = None,
) -> ExperimentResult:
    set_seed(seed)
    device = get_device(gpu_id, allow_cpu_fallback=allow_cpu_fallback)
    data_module = WikitextDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    pad_token_id = int(config["evaluation"].get("pad_token_id", data_module.tokenizer.pad_token_id or 0))
    model_params = dict(config.get("model_params", {}))
    model_params["vocab_size"] = int(data_module.tokenizer.vocab_size)
    model_params.setdefault("pad_idx", pad_token_id)
    model = get_model(model_name, **model_params).to(device)
    num_params = count_parameters(model)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    n_epochs = int(config["training"]["n_epochs"])
    grad_clip = float(config["training"].get("gradient_clip", 1.0))
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    start = time.time()
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in _limited_loader(train_loader, max_train_batches):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, kl = model(input_ids, attention_mask)
            ce_loss, valid, _ = _batch_loss(logits, input_ids, pad_token_id)
            loss = ce_loss + kl
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(ce_loss.item()) * max(valid, 1)
            total_tokens += max(valid, 1)
        train_loss = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20.0))
        val_metrics = evaluate(model, _limited_loader(val_loader, max_eval_batches), device, pad_token_id)
        if val_metrics["ppl"] < best_val:
            best_val = val_metrics["ppl"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch+1:02d}/{n_epochs} | "
            f"train ppl {train_ppl:.3f} | val ppl {val_metrics['ppl']:.3f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate(model, _limited_loader(train_loader, max_train_batches), device, pad_token_id)
    val_metrics = evaluate(model, _limited_loader(val_loader, max_eval_batches), device, pad_token_id)
    test_metrics = evaluate(model, _limited_loader(test_loader, max_test_batches), device, pad_token_id)
    train_time = time.time() - start
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_seed{seed}.pt"))
    return ExperimentResult(
        dataset=config["dataset"]["name"],
        model=model_name,
        seed=seed,
        train_loss=train_metrics["loss"],
        val_loss=val_metrics["loss"],
        test_loss=test_metrics["loss"],
        train_ppl=train_metrics["ppl"],
        val_ppl=val_metrics["ppl"],
        test_ppl=test_metrics["ppl"],
        train_time=train_time,
        num_params=num_params,
        best_epoch=best_epoch,
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(description="WikiText perplexity benchmark")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config", "default_config.yaml"))
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--no_cpu_fallback", action="store_true", help="Fail if CUDA is unavailable instead of falling back to CPU")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--max_test_batches", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    result = train_one_run(
        config,
        args.model,
        args.seed,
        config["experiment"]["output_dir"],
        args.gpu,
        allow_cpu_fallback=not args.no_cpu_fallback,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        max_test_batches=args.max_test_batches,
    )
    print(
        f"done | train ppl {result.train_ppl:.3f} | val ppl {result.val_ppl:.3f} | test ppl {result.test_ppl:.3f}"
    )


if __name__ == "__main__":
    main()
