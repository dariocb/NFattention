"""WikiText data loading and tokenization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

try:
    from .utils import repo_root
except ImportError:  # pragma: no cover - script execution fallback
    from utils import repo_root  # type: ignore


def _require_datasets():
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for the WikiText benchmark."
        ) from exc
    return load_dataset, load_from_disk


def _require_tokenizer(tokenizer_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The `transformers` package is required to build the WikiText tokenizer."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


@dataclass
class WikitextInfo:
    vocab_size: int
    pad_token_id: int
    block_size: int


class WikitextDataModule:
    """Small, Lightning-like data module used by the benchmark scripts."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config["dataset"]["name"]
        self.dataset_config = config["dataset"].get("config_name")
        self.block_size = int(config["dataset"].get("block_size", 512))
        self.tokenizer_id = config["dataset"].get("tokenizer_embeddings_model", "gpt2")
        self.cache_root = config["dataset"].get("cache_dir", "data")
        self.batch_size = int(config["training"]["batch_size"])
        self.num_workers = int(config["training"].get("num_workers", 4))
        self.max_length = self.block_size
        self.data = None
        self.num_observations = {"train": 0, "validation": 0, "test": 0}
        self.tokenizer = _require_tokenizer(self.tokenizer_id)

        cache_name = (
            f"wikitext_clm_blocks_tok-"
            f"{self.tokenizer_id.replace('/', '--')}_blocklen{self.block_size}"
        )
        self.cache_dir = os.path.join(repo_root(), self.cache_root, cache_name)

        config.setdefault("model_params", {})
        config["model_params"]["vocab_size"] = int(self.tokenizer.vocab_size)
        config["model_params"]["pad_idx"] = int(self.tokenizer.pad_token_id or 0)
        config["model_params"]["max_seq_len"] = self.block_size

    def prepare_data(self) -> None:
        load_dataset, _ = _require_datasets()
        if not os.path.exists(self.cache_dir):
            self._prep_wikitext_data(load_dataset)

    def _tokenize_batch(self, batch):
        return self.tokenizer(batch["text"], add_special_tokens=True, truncation=False)

    def _group_texts(self, examples):
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // self.block_size) * self.block_size
        return {
            key: [
                tokens[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for key, tokens in concatenated.items()
        }

    def _prep_wikitext_data(self, load_dataset) -> None:
        if self.dataset_config:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                cache_dir=os.path.join(repo_root(), ".cache"),
            )
        else:
            dataset = load_dataset(self.dataset_name, cache_dir=os.path.join(repo_root(), ".cache"))
        column_names = dataset["train"].column_names
        dataset = dataset.map(self._tokenize_batch, batched=True, remove_columns=column_names)
        dataset = dataset.map(self._group_texts, batched=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        dataset.save_to_disk(self.cache_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data is None:
            _, load_from_disk = _require_datasets()
            self.data = load_from_disk(self.cache_dir)
            self.num_observations = {
                "train": len(self.data["train"]),
                "validation": len(self.data["validation"]),
                "test": len(self.data["test"]),
            }

    def collate_fn(self, batch):
        keys = batch[0].keys()
        return {
            key: torch.stack([torch.tensor(sample[key], dtype=torch.long) for sample in batch])
            for key in keys
        }

    def train_dataloader(self):
        return DataLoader(
            self.data["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data["validation"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=self.collate_fn,
        )
