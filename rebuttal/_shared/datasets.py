"""Dataset acquisition, checksums, vocabulary, and PyTorch loaders."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
import shutil
import tarfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

LRA_RELEASE_URL = "https://storage.googleapis.com/long-range-arena/lra_release.gz"
LRA_HF_MIRROR = "fengyang0317/listops-1000"
LRA_HF_MIRROR_REVISION = "7e315de6abf1ec966d04b68a29e83a3ae23ca646"
LRA_SPLIT_SIZES = {"train": 96000, "validation": 2000, "test": 2000}


@dataclass
class TextSplit:
    texts: List[str]
    labels: List[int]
    source: str
    checksum: str


@dataclass
class DatasetBundle:
    train: TextSplit
    validation: TextSplit
    test: TextSplit
    num_classes: int
    name: str

    def manifest(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "splits": {
                name: {
                    "size": len(split.labels),
                    "source": split.source,
                    "checksum": split.checksum,
                }
                for name, split in (
                    ("train", self.train),
                    ("validation", self.validation),
                    ("test", self.test),
                )
            },
        }


def _records_checksum(texts: Sequence[str], labels: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\t")
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_sst5(cache_dir: Path, revision: Optional[str] = None) -> DatasetBundle:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "SST-5 loading requires the 'datasets' package from "
            "rebuttal/requirements.txt."
        ) from exc

    kwargs: Dict[str, object] = {"cache_dir": str(cache_dir)}
    if revision:
        kwargs["revision"] = revision
    dataset = load_dataset("SetFit/sst5", **kwargs)
    if "validation" not in dataset:
        raise RuntimeError("SetFit/sst5 did not expose the expected validation split")

    splits: Dict[str, TextSplit] = {}
    for name in ("train", "validation", "test"):
        texts = [str(value) for value in dataset[name]["text"]]
        labels = [int(value) for value in dataset[name]["label"]]
        splits[name] = TextSplit(
            texts,
            labels,
            source=f"SetFit/sst5:{name}:{revision or 'dataset-default'}",
            checksum=_records_checksum(texts, labels),
        )
    return DatasetBundle(
        splits["train"], splits["validation"], splits["test"], 5, "sst5"
    )


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    archive.extractall(destination)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; NFattention-rebuttal/1.0; "
                    "+https://github.com/google-research/long-range-arena)"
                )
            },
        )
        with urllib.request.urlopen(request) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def _prepare_listops_hf_mirror(data_dir: Path) -> Path:
    """Materialize a revision-pinned mirror of the official 96k/2k/2k splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The official LRA archive was unavailable and the pinned ListOps "
            "mirror requires the 'datasets' package."
        ) from exc

    mirror_root = data_dir / "hf_listops_1000"
    expected_paths = {
        "train": mirror_root / "basic_train.tsv",
        "validation": mirror_root / "basic_val.tsv",
        "test": mirror_root / "basic_test.tsv",
    }
    if all(path.exists() for path in expected_paths.values()):
        return mirror_root

    dataset = load_dataset(
        LRA_HF_MIRROR,
        revision=LRA_HF_MIRROR_REVISION,
        cache_dir=str(data_dir / "hf_cache"),
    )
    for split_name, expected_size in LRA_SPLIT_SIZES.items():
        if split_name not in dataset:
            raise RuntimeError(
                f"{LRA_HF_MIRROR} is missing the {split_name!r} split"
            )
        if len(dataset[split_name]) != expected_size:
            raise RuntimeError(
                f"{LRA_HF_MIRROR}:{split_name} has {len(dataset[split_name])} "
                f"rows; expected {expected_size}"
            )

    mirror_root.mkdir(parents=True, exist_ok=True)
    for split_name, destination in expected_paths.items():
        split = dataset[split_name]
        columns = {name.lower(): name for name in split.column_names}
        if "source" not in columns or "target" not in columns:
            raise RuntimeError(
                f"{LRA_HF_MIRROR}:{split_name} must contain Source and Target"
            )
        source_column = columns["source"]
        target_column = columns["target"]
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with partial.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["Source", "Target"])
                for row in split:
                    target = int(row[target_column])
                    if target < 0 or target > 9:
                        raise RuntimeError(
                            f"Invalid ListOps target {target} in {split_name}"
                        )
                    writer.writerow([str(row[source_column]), target])
            partial.replace(destination)
        finally:
            if partial.exists():
                partial.unlink()

    provenance = {
        "source": "Hugging Face mirror of LRA ListOps",
        "dataset": LRA_HF_MIRROR,
        "revision": LRA_HF_MIRROR_REVISION,
        "expected_split_sizes": LRA_SPLIT_SIZES,
        "official_archive": LRA_RELEASE_URL,
    }
    with (mirror_root / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return mirror_root


def prepare_listops(data_dir: Path, download: bool = True) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(data_dir.rglob("basic_train.tsv"))
    if candidates:
        return candidates[0].parent

    archive_path = data_dir / "lra_release.gz"
    if not archive_path.exists():
        if not download:
            raise FileNotFoundError(
                f"ListOps data not found under {data_dir}. "
                "Set DOWNLOAD_DATA=1 or download the official LRA release."
            )
        try:
            _download(LRA_RELEASE_URL, archive_path)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            print(
                f"Official LRA archive download failed ({exc}); "
                f"using pinned mirror {LRA_HF_MIRROR}@"
                f"{LRA_HF_MIRROR_REVISION}."
            )
            return _prepare_listops_hf_mirror(data_dir)

    extract_dir = data_dir / "lra_release"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            _safe_extract(archive, extract_dir)
    else:
        # Some mirrors expose a gzip-compressed tar without a conventional suffix.
        tar_path = data_dir / "lra_release.tar"
        with gzip.open(archive_path, "rb") as source, tar_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        if not tarfile.is_tarfile(tar_path):
            raise RuntimeError("The downloaded LRA release is not a tar archive")
        with tarfile.open(tar_path, "r:") as archive:
            _safe_extract(archive, extract_dir)

    candidates = list(extract_dir.rglob("basic_train.tsv"))
    if not candidates:
        raise RuntimeError("Could not locate basic_train.tsv in the LRA release")
    return candidates[0].parent


def _parse_listops(path: Path) -> TextSplit:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        rows = iter(reader)
        first = next(rows, None)
        if first is None:
            raise RuntimeError(f"Empty ListOps split: {path}")
        lowered = [cell.strip().lower() for cell in first]
        has_header = any(value in {"source", "target", "label"} for value in lowered)
        if not has_header:
            rows = iter([first] + list(rows))
        for row in rows:
            if len(row) < 2:
                continue
            source, target = row[0], row[1]
            # Official preprocessing replaces the closing bracket and removes
            # parentheses before whitespace tokenisation.
            source = source.replace("]", "X").replace("(", "").replace(")", "")
            texts.append(" ".join(source.split()))
            labels.append(int(target))
    return TextSplit(
        texts, labels, source=str(path.resolve()), checksum=_records_checksum(texts, labels)
    )


def load_listops(data_dir: Path, download: bool = True) -> DatasetBundle:
    root = prepare_listops(data_dir, download=download)
    split_paths = {
        "train": root / "basic_train.tsv",
        "validation": root / "basic_val.tsv",
        "test": root / "basic_test.tsv",
    }
    missing = [str(path) for path in split_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ListOps splits: {missing}")
    splits = {name: _parse_listops(path) for name, path in split_paths.items()}
    return DatasetBundle(
        splits["train"], splits["validation"], splits["test"], 10, "listops"
    )


def synthetic_bundle(
    num_classes: int = 5, train_size: int = 40, validation_size: int = 15,
    test_size: int = 15, seed: int = 0
) -> DatasetBundle:
    rng = random.Random(seed)

    def make(size: int, name: str) -> TextSplit:
        texts, labels = [], []
        for index in range(size):
            label = index % num_classes
            noise = rng.randrange(7)
            texts.append(f"class_{label} signal_{label} noise_{noise} item_{index % 3}")
            labels.append(label)
        return TextSplit(
            texts, labels, f"synthetic:{name}", _records_checksum(texts, labels)
        )

    return DatasetBundle(
        make(train_size, "train"),
        make(validation_size, "validation"),
        make(test_size, "test"),
        num_classes,
        "synthetic",
    )


class Vocabulary:
    def __init__(self, min_frequency: int = 2, max_size: int = 50000):
        self.min_frequency = min_frequency
        self.max_size = max_size
        self.token_to_id = {"<pad>": 0, "<unk>": 1}

    def fit(self, texts: Sequence[str]) -> None:
        counts = Counter(token for text in texts for token in text.lower().split())
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for token, count in ordered:
            if count < self.min_frequency or len(self.token_to_id) >= self.max_size:
                continue
            self.token_to_id[token] = len(self.token_to_id)

    def encode(self, text: str, max_length: int) -> Tuple[List[int], List[int]]:
        tokens = text.lower().split()[:max_length]
        ids = [self.token_to_id.get(token, 1) for token in tokens]
        mask = [1] * len(ids)
        padding = max_length - len(ids)
        return ids + [0] * padding, mask + [0] * padding

    def to_dict(self) -> Dict[str, object]:
        return {
            "min_frequency": self.min_frequency,
            "max_size": self.max_size,
            "token_to_id": self.token_to_id,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "Vocabulary":
        vocabulary = cls(
            min_frequency=int(payload["min_frequency"]),
            max_size=int(payload["max_size"]),
        )
        vocabulary.token_to_id = {
            str(token): int(index)
            for token, index in dict(payload["token_to_id"]).items()
        }
        return vocabulary


class EncodedTextDataset(Dataset):
    def __init__(
        self, split: TextSplit, vocabulary: Vocabulary, max_length: int
    ):
        encoded = [vocabulary.encode(text, max_length) for text in split.texts]
        self.input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.attention_mask = torch.tensor(
            [item[1] for item in encoded], dtype=torch.bool
        )
        self.labels = torch.tensor(split.labels, dtype=torch.long)
        self.lengths = self.attention_mask.sum(dim=1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.input_ids[index],
            self.attention_mask[index],
            self.labels[index],
            self.lengths[index],
            torch.tensor(index, dtype=torch.long),
        )


@dataclass
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    vocabulary: Vocabulary


def make_loaders(
    bundle: DatasetBundle,
    max_length: int,
    batch_size: int,
    seed: int,
    min_frequency: int = 2,
    max_vocab_size: int = 50000,
    num_workers: int = 0,
) -> LoaderBundle:
    vocabulary = Vocabulary(min_frequency=min_frequency, max_size=max_vocab_size)
    vocabulary.fit(bundle.train.texts)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    datasets = {
        "train": EncodedTextDataset(bundle.train, vocabulary, max_length),
        "validation": EncodedTextDataset(bundle.validation, vocabulary, max_length),
        "test": EncodedTextDataset(bundle.test, vocabulary, max_length),
    }
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return LoaderBundle(
        train=DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            generator=generator, **common
        ),
        validation=DataLoader(
            datasets["validation"], batch_size=batch_size, shuffle=False, **common
        ),
        test=DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, **common
        ),
        vocabulary=vocabulary,
    )
