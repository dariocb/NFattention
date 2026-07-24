from __future__ import annotations

import pytest
import torch

from rebuttal._shared.baselines import build_attention


@pytest.mark.parametrize("model_name", ["transformer", "performer", "rka"])
def test_padding_invariance(model_name: str):
    torch.manual_seed(0)
    attention = build_attention(model_name, 16, 4, 16, 0.0, 123)
    attention.eval()
    base = torch.randn(2, 6, 16)
    changed = base.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 2, dtype=torch.bool)
    with torch.no_grad():
        first = attention(base, mask).output
        second = attention(changed, mask).output
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(first[:, 4:]) == 0


@pytest.mark.parametrize("model_name", ["performer", "rka"])
def test_linear_baselines_do_not_return_dense_attention(model_name: str):
    attention = build_attention(model_name, 16, 4, 16, 0.0, 123)
    output = attention(
        torch.randn(2, 7, 16), torch.ones(2, 7, dtype=torch.bool)
    )
    assert output.output.shape == (2, 7, 16)
    assert output.diagnostics == {}

