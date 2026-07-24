"""Immutable-ish, append-only result output for standalone experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialise {type(value)!r}")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def config_hash(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def environment_manifest() -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        manifest.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        manifest["git_commit"] = None
    return manifest


class ResultWriter:
    def __init__(self, output_dir: Path, config: Mapping[str, object]):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self.run_id = config_hash(self.config)
        self.runs_path = self.output_dir / "runs.jsonl"
        self.failures_path = self.output_dir / "failures.json"
        self.failures: List[Dict[str, object]] = []
        write_json(self.output_dir / "config.json", self.config)
        write_json(
            self.output_dir / "manifest.json",
            {**environment_manifest(), "config_hash": self.run_id},
        )

    def add_run(self, run: Mapping[str, object]) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(run), default=_json_default, sort_keys=True))
            handle.write("\n")

    def add_failure(self, context: Mapping[str, object], error: BaseException) -> None:
        self.failures.append(
            {
                **dict(context),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(self.failures_path, self.failures)

    def write_summary(self, rows: Sequence[Mapping[str, object]]) -> None:
        write_json(self.output_dir / "summary.json", list(rows))
        keys = sorted({key for row in rows for key in row})
        with (self.output_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def write_latex(
        self, rows: Sequence[Mapping[str, object]], columns: Sequence[str]
    ) -> None:
        lines = [
            "\\begin{tabular}{" + "l" * len(columns) + "}",
            " & ".join(columns) + " \\\\",
            "\\hline",
        ]
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value).replace("_", "\\_"))
            lines.append(" & ".join(values) + " \\\\")
        lines.append("\\end{tabular}")
        (self.output_dir / "table.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def finish(self) -> None:
        if not self.failures_path.exists():
            write_json(self.failures_path, [])

