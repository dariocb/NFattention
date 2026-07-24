from __future__ import annotations

import importlib.util
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

