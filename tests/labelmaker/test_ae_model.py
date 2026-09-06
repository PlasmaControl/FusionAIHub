"""`labelmaker.ae.model` - shapes, pool arithmetic and loss behaviour.

Torch lives only in the phase-3 venv, so this module is skipped in the pixi
test env. It is run once against the venv that trains:

    /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \
        -m pytest tests/labelmaker/test_ae_model.py -q -W error -p no:cacheprovider
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from labelmaker.ae.model import (
    DEFAULT_POOL_SIZES,
    AeLossConfig,
    AeSeldNet,
    AeSeldNetConfig,
    ae_loss,
    binary_bce_loss,
    binary_sce_loss,
    denormalise_freq,
    masked_huber_loss,
    normalise_freq,
)


def test_default_pool_sizes_divide_the_band_exactly() -> None:
    product = 1
    for pool in DEFAULT_POOL_SIZES:
        product *= pool
    assert product == 348
    assert AeSeldNetConfig().freq_after_pooling == 1


def test_forward_shape_is_one_row_per_frame() -> None:
    model = AeSeldNet().eval()
    x = torch.randn(2, 4, 710, 348)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 710, 2)
    assert torch.isfinite(out).all()


def test_forward_runs_on_any_record_length() -> None:
    """Nothing pools time, so inference length is free."""
    model = AeSeldNet().eval()
    with torch.no_grad():
        short = model(torch.randn(1, 4, 37, 348))
        long = model(torch.randn(1, 4, 512, 348))
    assert short.shape == (1, 37, 2)
    assert long.shape == (1, 512, 2)


def test_pool_sizes_that_do_not_divide_the_axis_are_rejected() -> None:
    with pytest.raises(ValueError, match="do not divide"):
        AeSeldNet(AeSeldNetConfig(pool_sizes=(9, 8, 2)))


def test_alternative_pool_sizes_keep_more_frequency_rows() -> None:
    cfg = AeSeldNetConfig(pool_sizes=(6, 2), n_freq=348)
    assert cfg.freq_after_pooling == 29
    model = AeSeldNet(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 64, 348))
    assert out.shape == (1, 64, 2)


def test_config_round_trips_through_a_dict() -> None:
    cfg = AeSeldNetConfig(conv_channels=32, dropout=0.1)
    assert AeSeldNetConfig.from_dict(cfg.as_dict()) == cfg


def test_wrong_rank_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"expected \(B, C, T, F\)"):
        AeSeldNet()(torch.randn(4, 710, 348))


def test_normalised_frequency_maps_the_band_to_the_unit_interval() -> None:
    f = torch.tensor([80.0, 165.0, 250.0])
    n = normalise_freq(f)
    assert torch.allclose(n, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(denormalise_freq(n), f)


def test_loss_is_finite_with_an_all_inactive_label() -> None:
    model = AeSeldNet().eval()
    x = torch.randn(2, 4, 710, 348)
    out = model(x)
    zeros = torch.zeros(2, 710)
    parts = ae_loss(out, zeros, torch.ones(2, 710), torch.full((2, 710), float("nan")), zeros)
    for name, value in parts.items():
        assert torch.isfinite(value), name
    assert parts["freq"].item() == 0.0


def test_loss_is_finite_when_every_frame_is_ignored() -> None:
    """The three-way target can leave a whole window unweighted."""
    out = torch.randn(1, 16, 2, requires_grad=True)
    zeros = torch.zeros(1, 16)
    parts = ae_loss(out, zeros, zeros, zeros, zeros)
    assert torch.isfinite(parts["loss"])
    assert parts["loss"].item() == 0.0
    parts["loss"].backward()
    assert torch.isfinite(out.grad).all()


def test_weighted_losses_ignore_zero_weight_frames() -> None:
    logits = torch.tensor([[10.0, -10.0]])
    target = torch.tensor([[1.0, 1.0]])
    weight = torch.tensor([[1.0, 0.0]])
    both = binary_bce_loss(logits, target, torch.ones_like(weight))
    first_only = binary_bce_loss(logits, target, weight)
    assert first_only.item() < 1e-3
    assert both.item() > first_only.item()


def test_sce_reduces_to_scaled_bce_plus_a_bounded_reverse_term() -> None:
    logits = torch.randn(3, 8)
    target = (torch.rand(3, 8) > 0.5).float()
    weight = torch.ones_like(target)
    bce = binary_bce_loss(logits, target, weight)
    sce = binary_sce_loss(logits, target, weight, alpha=1.0, beta=0.5)
    # RCE is non-negative and capped by -log(label_eps) = 9.21.
    rce = (sce - bce) / 0.5
    assert rce.item() >= 0.0
    assert rce.item() <= -torch.log(torch.tensor(1e-4)).item() + 1e-3


def test_sce_with_beta_zero_is_bce() -> None:
    logits = torch.randn(2, 5)
    target = (torch.rand(2, 5) > 0.5).float()
    weight = torch.ones_like(target)
    assert torch.allclose(
        binary_sce_loss(logits, target, weight, beta=0.0),
        binary_bce_loss(logits, target, weight),
        atol=1e-6,
    )


def test_sce_penalises_a_confidently_wrong_prediction_less_sharply_than_bce() -> None:
    """The point of the symmetric term: a flipped label cannot dominate."""
    target = torch.ones(1, 1)
    weight = torch.ones(1, 1)
    mild = torch.tensor([[-2.0]])
    extreme = torch.tensor([[-20.0]])
    bce_ratio = (
        binary_bce_loss(extreme, target, weight) / binary_bce_loss(mild, target, weight)
    ).item()
    sce_ratio = (
        binary_sce_loss(extreme, target, weight) / binary_sce_loss(mild, target, weight)
    ).item()
    assert sce_ratio < bce_ratio


def test_masked_huber_ignores_nan_targets_outside_the_mask() -> None:
    pred = torch.zeros(1, 4)
    target = torch.tensor([[0.5, float("nan"), float("nan"), 0.25]])
    mask = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    value = masked_huber_loss(pred, target, mask)
    expected = (0.5 * 0.5**2 + 0.5 * 0.25**2) / 2
    assert torch.isfinite(value)
    assert value.item() == pytest.approx(expected, rel=1e-6)


def test_unknown_loss_name_is_rejected() -> None:
    out = torch.zeros(1, 4, 2)
    zeros = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="unknown activity loss"):
        ae_loss(out, zeros, zeros, zeros, zeros, AeLossConfig(loss="focal"))


def test_loss_config_round_trips_its_recorded_fields() -> None:
    cfg = AeLossConfig(loss="bce", lambda_f=1.0)
    assert cfg.as_dict()["lambda_f"] == 1.0
    assert cfg.as_dict()["loss"] == "bce"


def test_gradients_flow_to_both_heads() -> None:
    model = AeSeldNet(AeSeldNetConfig(conv_channels=8, rnn_sizes=(16, 16), fnn_size=16))
    out = model(torch.randn(1, 4, 32, 348))
    target = torch.ones(1, 32)
    parts = ae_loss(out, target, target, torch.full((1, 32), 0.3), target)
    parts["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)
