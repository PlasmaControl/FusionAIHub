"""Tests for IGNITE Phase-A losses (shift_consistency, recon_objective, discriminator_loss).

Small synthetic CPU tensors only. Written test-first (TDD).
"""
from __future__ import annotations

import torch

from tokamak_foundation_model.ignite import losses
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN


def _cfg() -> SpectroCodecConfig:
    # small config to keep tensors tiny on CPU
    return SpectroCodecConfig(
        channels=1,
        freq_bins=64,
        time_frames=32,
        patch_f=32,
        patch_t=16,
        d_model=16,
        enc_depth=1,
        dec_depth=1,
        heads=2,
    )


# --------------------------------------------------------------------------- #
# shift_consistency
# --------------------------------------------------------------------------- #
def test_shift_consistency_zero_when_equal():
    feats = torch.randn(3, 8, 16)
    loss = losses.shift_consistency(feats, feats)
    assert loss.dim() == 0, "expected a scalar"
    assert torch.allclose(loss, torch.zeros(())), "loss must be 0 when a == b"


def test_shift_consistency_positive_when_perturbed():
    a = torch.randn(3, 8, 16)
    b = a + 0.5 * torch.randn_like(a)
    loss = losses.shift_consistency(a, b)
    assert loss.item() > 0.0, "loss must be > 0 when b is perturbed"


def test_shift_consistency_differentiable():
    a = torch.randn(2, 8, 16, requires_grad=True)
    b = torch.randn(2, 8, 16)
    loss = losses.shift_consistency(a, b)
    loss.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert a.grad.abs().sum() > 0.0


# --------------------------------------------------------------------------- #
# recon_objective
# --------------------------------------------------------------------------- #
def test_recon_objective_finite_and_differentiable():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    recon = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, requires_grad=True)
    target = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)

    out = losses.recon_objective(recon, target, disc, cfg)
    assert isinstance(out, dict)
    for key in ("total", "adversarial", "pixel"):
        assert key in out, f"missing component '{key}'"
        assert out[key].dim() == 0
        assert torch.isfinite(out[key]).all()

    out["total"].backward()
    assert recon.grad is not None
    assert torch.isfinite(recon.grad).all()
    assert recon.grad.abs().sum() > 0.0


def test_recon_objective_pixel_matches_mae():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    recon = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    target = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = losses.recon_objective(recon, target, disc, cfg)
    expected_pixel = (recon - target).abs().mean()
    assert torch.allclose(out["pixel"], expected_pixel, atol=1e-6)


def test_recon_objective_respects_pixel_weight():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    recon = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    target = recon.clone()  # zero pixel MAE -> total == adversarial term
    out = losses.recon_objective(recon, target, disc, cfg)
    assert torch.allclose(out["pixel"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(out["total"], out["adversarial"], atol=1e-5)


# --------------------------------------------------------------------------- #
# discriminator_loss (hinge)
# --------------------------------------------------------------------------- #
def test_discriminator_loss_finite_and_scalar():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    real = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    fake = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    loss = losses.discriminator_loss(disc, real, fake, cfg)
    assert loss.dim() == 0
    assert torch.isfinite(loss).all()


def test_discriminator_loss_decreases_when_scores_correct():
    """Hinge D-loss should be lower when real scores are high & fake scores are low.

    Use a stub disc so we control the raw scores directly, isolating the hinge math.
    """
    cfg = _cfg()

    dummy = torch.zeros(2, cfg.channels, cfg.freq_bins, cfg.time_frames)

    # Feed real & fake through a stub that returns real_v on the 1st call, fake_v on the 2nd.
    class TwoValueDisc(torch.nn.Module):
        def __init__(self, real_v, fake_v):
            super().__init__()
            self.real_v = real_v
            self.fake_v = fake_v
            self._first = True

        def forward(self, x):
            b = x.shape[0]
            v = self.real_v if self._first else self.fake_v
            self._first = not self._first
            return [torch.full((b, 1, 4, 4), v)]

    good_loss = losses.discriminator_loss(TwoValueDisc(2.0, -2.0), dummy, dummy, cfg)
    bad_loss = losses.discriminator_loss(TwoValueDisc(-2.0, 2.0), dummy, dummy, cfg)
    assert good_loss.item() < bad_loss.item(), (good_loss.item(), bad_loss.item())
    # correctly-classified beyond margin -> ~0
    assert good_loss.item() < 1e-4


def test_discriminator_loss_differentiable():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    real = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    fake = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    loss = losses.discriminator_loss(disc, real, fake, cfg)
    loss.backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
