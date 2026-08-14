"""Tests for the frequency-aware multi-scale PatchGAN discriminator (Phase A).

Small synthetic CPU tensors only. Written test-first (TDD).
"""
from __future__ import annotations

import torch

from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN


def _cfg() -> SpectroCodecConfig:
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


def test_forward_returns_list_of_maps():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = disc(x)
    assert isinstance(out, list)
    assert len(out) >= 2, "expected multi-scale output (>= 2 scales)"
    for m in out:
        assert isinstance(m, torch.Tensor)
        assert m.dim() == 4, f"expected (B,C,H,W) score map, got {m.shape}"
        assert m.shape[0] == x.shape[0]
        assert torch.isfinite(m).all()


def test_scales_have_different_resolution():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = disc(x)
    spatial_sizes = {(m.shape[-2], m.shape[-1]) for m in out}
    assert len(spatial_sizes) >= 2, "multi-scale maps should differ in resolution"


def test_no_final_sigmoid_raw_scores():
    """Raw hinge-ready scores: with random weights, maps should span negative & positive."""
    cfg = _cfg()
    torch.manual_seed(0)
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(8, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = disc(x)
    cat = torch.cat([m.reshape(-1) for m in out])
    assert (cat < 0).any() and (cat > 0).any(), "scores look bounded (sigmoid?) — must be raw"


def test_consumes_freq_pe():
    """A row of constant input across freq should still yield freq-varying features:
    the freq-PE breaks the freq-translation symmetry. Compare against a disc whose PE
    contribution we zero out by feeding identical freq rows -> outputs must NOT be
    freq-uniform because PE was concatenated.
    """
    cfg = _cfg()
    torch.manual_seed(0)
    disc = FreqAwarePatchGAN(cfg)
    # input constant along the frequency axis
    x = torch.ones(1, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = disc(x)
    # first-scale map: variance along the freq (height) axis should be > 0 because the
    # freq-PE injected freq-dependent signal even though the raw input is freq-constant.
    m0 = out[0]
    freq_var = m0.var(dim=-2).mean()
    assert freq_var.item() > 1e-8, "freq-PE does not appear to influence the output"


def test_freq_pe_channel_is_appended():
    """Expose the internal freq-PE builder and check it has the right shape & varies over freq."""
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    pe = disc.freq_pe(x)
    assert pe.shape[0] == x.shape[0]
    assert pe.shape[-2] == cfg.freq_bins and pe.shape[-1] == cfg.time_frames
    assert pe.shape[1] >= 1, "expected >= 1 freq-PE channel"
    # PE must vary across freq axis but be constant across time
    assert pe.var(dim=-2).mean().item() > 1e-8
    assert torch.allclose(pe.var(dim=-1).mean(), torch.zeros(()), atol=1e-6)


def test_differentiable():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, requires_grad=True)
    out = disc(x)
    loss = sum(m.mean() for m in out)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    param_grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert len(param_grads) > 0


def test_multichannel_input():
    cfg = SpectroCodecConfig(
        channels=2, freq_bins=64, time_frames=32, patch_f=32, patch_t=16,
        d_model=16, enc_depth=1, dec_depth=1, heads=2,
    )
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = disc(x)
    assert all(torch.isfinite(m).all() for m in out)
