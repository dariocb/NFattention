from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "04_elu_gap"
        / "main.py"
    )
    spec = importlib.util.spec_from_file_location("rebuttal_elu_gap_main", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnostic_features_and_context_are_tensors():
    module = _load_module()
    x = torch.randn(1, 2, 5, 4)
    omega1 = torch.randn(1, 2, 8, 4)
    omega2 = torch.randn(1, 2, 8, 4)
    features = module._features(x, omega1, omega2, pairs=8)
    assert features.shape == (1, 2, 5, 16)

    values = torch.randn(1, 2, 5, 4)
    mask = torch.ones(1, 5, dtype=torch.bool)
    context, denominator, clamped = module._diagnostic_context(
        features, features, values, mask, signed=True
    )
    assert context.shape == values.shape
    assert denominator.shape == (1, 2, 5)
    assert clamped.dtype == torch.bool


def test_raw_divergence_is_not_an_unexpected_failure(tmp_path: Path):
    module = _load_module()
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "elu_plus_one__seed0.pt").touch()
    (tmp_path / "failures.json").write_text(
        json.dumps(
            [
                {
                    "variant": "raw_rff",
                    "seed": 0,
                    "error_type": "FloatingPointError",
                    "error": "Non-finite spectral sample",
                }
            ]
        ),
        encoding="utf-8",
    )

    divergences, unexpected, missing = module._elu_gap_completion_state(
        tmp_path, [0], grid_status=1
    )

    assert len(divergences) == 1
    assert unexpected == []
    assert missing == []
