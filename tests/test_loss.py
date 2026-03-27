"""Tests for MultiScaleSpectrogramLoss."""

import torch
import torch.nn as nn
import pytest

from tokamak_foundation_model.models.loss import MultiScaleSpectrogramLoss


def test_scalar_output():
    """Loss returns a scalar for realistic input shapes."""
    loss_fn = MultiScaleSpectrogramLoss()
    pred = torch.randn(2, 4, 128, 64)
    target = torch.randn(2, 4, 128, 64)
    result = loss_fn(pred, target)
    assert result.shape == () or result.shape == (1,)
    assert result.item() > 0


def test_single_scale_matches_l1():
    """Single scale=1.0 should match nn.L1Loss."""
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)

    ms_loss = MultiScaleSpectrogramLoss(scales=(1.0,))
    l1_loss = nn.L1Loss()

    assert torch.allclose(ms_loss(pred, target), l1_loss(pred, target), atol=1e-6)


def test_multi_scale_differs_from_single():
    """Multi-scale loss should differ from single-scale L1."""
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 64, 64, requires_grad=True)
    target = torch.randn(2, 4, 64, 64)

    single = MultiScaleSpectrogramLoss(scales=(1.0,))
    multi = MultiScaleSpectrogramLoss(scales=(1.0, 0.5, 0.25))

    loss_single = single(pred, target)
    loss_multi = multi(pred, target)

    assert not torch.allclose(loss_single, loss_multi, atol=1e-6)


def test_zero_loss_for_identical_inputs():
    """Loss should be zero when pred == target."""
    loss_fn = MultiScaleSpectrogramLoss()
    x = torch.randn(1, 2, 16, 16)
    result = loss_fn(x, x)
    assert result.item() < 1e-7


def test_weighted_multi_scale():
    """multi_scale_l1 with weight should return scalar and differ from unweighted."""
    loss_fn = MultiScaleSpectrogramLoss()
    pred = torch.randn(2, 4, 32, 32)
    target = torch.randn(2, 4, 32, 32)
    w = torch.rand(1, 4, 32, 32) + 0.5  # non-uniform weight

    unweighted = loss_fn(pred, target)
    weighted = loss_fn.multi_scale_l1(pred, target, weight=w)

    assert weighted.shape == () or weighted.shape == (1,)
    assert not torch.allclose(unweighted, weighted, atol=1e-6)


def test_small_scale():
    """Very small scale factor should not crash."""
    loss_fn = MultiScaleSpectrogramLoss(scales=(1.0, 0.1))
    pred = torch.randn(1, 2, 128, 64)
    target = torch.randn(1, 2, 128, 64)
    result = loss_fn(pred, target)
    assert result.item() > 0


def test_custom_weights():
    """Custom weights should be normalized and applied."""
    # Heavy weight on full-res, light on coarse
    loss_heavy_fine = MultiScaleSpectrogramLoss(
        scales=(1.0, 0.25), weights=(10.0, 1.0))
    # Heavy weight on coarse, light on full-res
    loss_heavy_coarse = MultiScaleSpectrogramLoss(
        scales=(1.0, 0.25), weights=(1.0, 10.0))

    pred = torch.randn(1, 2, 64, 64)
    target = torch.randn(1, 2, 64, 64)

    result_fine = loss_heavy_fine(pred, target)
    result_coarse = loss_heavy_coarse(pred, target)

    # They should differ since weights are different
    assert not torch.allclose(result_fine, result_coarse, atol=1e-6)


def test_gradients_flow():
    """Verify gradients flow through the multi-scale loss."""
    loss_fn = MultiScaleSpectrogramLoss()
    pred = torch.randn(1, 2, 32, 32, requires_grad=True)
    target = torch.randn(1, 2, 32, 32)

    loss = loss_fn(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert pred.grad.abs().sum() > 0
