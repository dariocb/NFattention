from __future__ import annotations

import pytest
import torch

pytest.importorskip("normflows")

from rebuttal._shared.fska import FSKAConfig, FSKAAttention


def _attention(**overrides):
    values = dict(
        hidden_dim=16,
        n_heads=4,
        num_spectral_pairs=4,
        density_mode="fixed_bivariate",
        dropout=0.0,
        sample_seed=7,
    )
    values.update(overrides)
    return FSKAAttention(FSKAConfig(**values))


def test_masking_and_deterministic_evaluation():
    torch.manual_seed(0)
    attention = _attention()
    attention.eval()
    base = torch.randn(2, 6, 16)
    changed = base.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 2, dtype=torch.bool)
    first = attention(base, mask).output
    repeated = attention(base, mask).output
    second = attention(changed, mask).output
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(first[:, 4:]) == 0


def test_identity_qk_has_no_projection_parameters():
    attention = _attention(qk_mode="identity")
    names = [name for name, _ in attention.named_parameters()]
    assert not any(name.startswith("q_proj") for name in names)
    assert not any(name.startswith("k_proj") for name in names)


def test_fixed_density_is_immutable_and_kl_is_zero():
    attention = _attention()
    before = attention.density.checksum()
    optimizer = torch.optim.Adam(attention.parameters(), lr=1e-3)
    result = attention(
        torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool)
    )
    (result.output.square().mean() + result.kl_raw).backward()
    optimizer.step()
    assert attention.density.checksum() == before
    assert result.kl_raw.item() == 0.0


def test_learned_flow_has_gradients_and_finite_kl():
    attention = _attention(density_mode="learned_flow")
    attention.train()
    result = attention(
        torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.bool)
    )
    loss = result.output.square().mean() + 1e-3 * result.kl_raw
    loss.backward()
    flow_gradients = [
        parameter.grad
        for name, parameter in attention.named_parameters()
        if "density.flow" in name and parameter.requires_grad
    ]
    assert torch.isfinite(result.kl_raw)
    assert result.kl_raw.abs().item() > 0
    assert any(gradient is not None for gradient in flow_gradients)


def test_linear_contraction_matches_explicit_kernel():
    attention = _attention()
    phi_q = torch.rand(2, 4, 5, 8) + 0.5
    phi_k = torch.rand(2, 4, 5, 8) + 0.5
    values = torch.randn(2, 4, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    actual, _, _ = attention.linear_context(phi_q, phi_k, values, mask)
    key_mask = mask[:, None, :, None].to(phi_k.dtype)
    kernel = torch.matmul(phi_q, (phi_k * key_mask).transpose(-2, -1))
    expected = torch.matmul(kernel, values * key_mask)
    expected = expected / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_diagnostics_are_feature_sized_not_sequence_square():
    attention = _attention()
    result = attention(
        torch.randn(2, 11, 16),
        torch.ones(2, 11, dtype=torch.bool),
        return_diagnostics=True,
    )
    for tensor in result.diagnostics.values():
        assert tensor.ndim < 2 or tensor.shape[-2:] != (11, 11)

