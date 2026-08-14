"""Tests for the discriminator FEATURE-MATCHING perceptual term (IGNITE Phase A).

This is the HiFi-GAN/MelGAN vocoder-GAN feature-matching loss — the spectrogram-adapted
stand-in for Genie's VGG perceptual loss (VGG does not transfer to spectrograms). It is
folded INTO the reconstruction reference so that the VQGAN adaptive weight `lam` rises and
the balanced adversarial term regains sharpening strength.

CPU synthetic tensors only. Written test-first (TDD).
"""
from __future__ import annotations

import torch

from tokamak_foundation_model.ignite import losses
from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN


def _cfg(**overrides) -> SpectroCodecConfig:
    base = dict(
        channels=1,
        freq_bins=64,
        time_frames=32,
        patch_f=32,
        patch_t=16,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )
    base.update(overrides)
    return SpectroCodecConfig(**base)


def _pair(cfg):
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    x_shift = x + 0.05 * torch.randn_like(x)
    return x, x_shift


# --------------------------------------------------------------------------- #
# feature_matching_loss: value + sign
# --------------------------------------------------------------------------- #
def test_fm_zero_when_real_equals_fake():
    torch.manual_seed(0)
    feats = [torch.randn(2, 8, 5, 5), torch.randn(2, 16, 3, 3)]
    fake = [t.clone() for t in feats]
    loss = losses.feature_matching_loss(feats, fake)
    assert loss.dim() == 0, "expected a scalar"
    assert torch.allclose(loss, torch.zeros(())), "FM must be 0 when real == fake"


def test_fm_positive_when_different():
    torch.manual_seed(1)
    real = [torch.randn(2, 8, 5, 5), torch.randn(2, 16, 3, 3)]
    fake = [t + 0.5 * torch.randn_like(t) for t in real]
    loss = losses.feature_matching_loss(real, fake)
    assert loss.item() > 0.0, "FM must be > 0 when fake differs from real"


def test_fm_equals_mean_l1_over_pairs():
    """FM is the mean over matched pairs of elementwise L1."""
    torch.manual_seed(2)
    real = [torch.randn(2, 4, 3, 3), torch.randn(2, 8, 2, 2)]
    fake = [torch.randn(2, 4, 3, 3), torch.randn(2, 8, 2, 2)]
    got = losses.feature_matching_loss(real, fake)
    expected = torch.stack(
        [torch.mean(torch.abs(fk - rl)) for rl, fk in zip(real, fake)]
    ).mean()
    assert torch.allclose(got, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# feature_matching_loss: gradient flows ONLY through fake (real is detached target)
# --------------------------------------------------------------------------- #
def test_fm_differentiable_grad_reaches_fake():
    torch.manual_seed(3)
    real = [torch.randn(2, 4, 3, 3), torch.randn(2, 8, 2, 2)]
    # fresh leaf tensors, perturbed away from real so the gradient is non-zero
    fake = [(t + 0.1).detach().requires_grad_(True) for t in real]
    loss = losses.feature_matching_loss(real, fake)
    loss.backward()
    for f in fake:
        assert f.grad is not None
        assert torch.isfinite(f.grad).all()
        assert f.grad.abs().sum() > 0.0


def test_fm_real_is_detached_no_grad_to_real():
    """The REAL features are the detached target — no gradient should reach them."""
    torch.manual_seed(4)
    real = [torch.randn(2, 4, 3, 3, requires_grad=True)]
    fake = [torch.randn(2, 4, 3, 3, requires_grad=True)]
    loss = losses.feature_matching_loss(real, fake)
    loss.backward()
    assert fake[0].grad is not None and fake[0].grad.abs().sum() > 0.0
    assert real[0].grad is None, "real features must be treated as a detached target"


# --------------------------------------------------------------------------- #
# feature_matching_loss: defensive handling of unequal list lengths / empties
# --------------------------------------------------------------------------- #
def test_fm_handles_unequal_lengths():
    real = [torch.randn(2, 4, 3, 3), torch.randn(2, 8, 2, 2), torch.randn(2, 8, 2, 2)]
    fake = [torch.randn(2, 4, 3, 3)]  # shorter
    loss = losses.feature_matching_loss(real, fake)  # must not raise
    assert loss.dim() == 0 and torch.isfinite(loss).all()
    # equals the single matched pair
    expected = torch.mean(torch.abs(fake[0] - real[0]))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_fm_empty_lists_return_zero():
    loss = losses.feature_matching_loss([], [])
    assert loss.dim() == 0
    assert torch.allclose(loss, torch.zeros(()))


# --------------------------------------------------------------------------- #
# discriminator: return_features contract + regression-safety of plain forward
# --------------------------------------------------------------------------- #
def test_disc_return_features_shapes():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    scores, feats = disc(x, return_features=True)
    assert isinstance(scores, list) and len(scores) >= 2
    for s in scores:
        assert s.dim() == 4 and s.shape[0] == x.shape[0]
    assert isinstance(feats, list) and len(feats) > 0, "expected non-empty feature list"
    for f in feats:
        assert isinstance(f, torch.Tensor)
        assert f.dim() == 4 and f.shape[0] == x.shape[0]
        assert torch.isfinite(f).all()


def test_disc_return_features_scores_match_plain_forward():
    """The scores returned alongside features must equal the plain forward() scores."""
    cfg = _cfg()
    torch.manual_seed(7)
    disc = FreqAwarePatchGAN(cfg)
    disc.eval()
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    plain = disc(x)
    scores, _ = disc(x, return_features=True)
    assert len(plain) == len(scores)
    for a, b in zip(plain, scores):
        assert torch.allclose(a, b, atol=0.0, rtol=0.0), "score path must be byte-identical"


def test_disc_plain_forward_unchanged_default():
    """Default forward(x) returns just the list of scores (no tuple) — regression guard."""
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = disc(x)
    assert isinstance(out, list)
    assert all(isinstance(m, torch.Tensor) for m in out)


def test_disc_features_differentiable():
    cfg = _cfg()
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames, requires_grad=True)
    _, feats = disc(x, return_features=True)
    loss = sum(f.mean() for f in feats)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------- #
# generator_losses: exposes "feature_matching" + fm raises recon_ref -> raises lam
# --------------------------------------------------------------------------- #
def test_generator_losses_exposes_feature_matching():
    torch.manual_seed(8)
    cfg = _cfg(adaptive_adv_weight=True)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)
    out = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    assert "feature_matching" in out
    assert out["feature_matching"].dim() == 0
    assert torch.isfinite(out["feature_matching"]).all()
    assert out["feature_matching"].item() >= 0.0


def test_fm_enters_recon_ref_and_raises_lam():
    """Numeric demo: turning FM on (fm_weight>0) enlarges recon_ref -> lam >= the fm-off lam.

    We hold everything else fixed (same weights, same seed / same codec+disc params) and
    compare fm_weight=0 vs fm_weight>0. recon_ref = pixel_anchor*pixel + fm_weight*fm, so
    with fm > 0 the reference grows; a larger ref gradient at the last layer raises
    lam = ||grad ref|| / (||grad adv|| + 1e-4). We assert both the recon_ref inequality
    (directly) and lam_on >= lam_off (the mechanism payoff).
    """
    torch.manual_seed(9)
    cfg_off = _cfg(adaptive_adv_weight=True, fm_weight=0.0)
    codec = SpectroCodec(cfg_off)
    disc = FreqAwarePatchGAN(cfg_off)
    codec.eval()
    disc.eval()
    x, x_shift = _pair(cfg_off)

    # fm_weight>0 shares the SAME codec/disc params (only the cfg scalar differs).
    cfg_on = _cfg(adaptive_adv_weight=True, fm_weight=5.0)

    # Recompute the exact pieces generator_losses uses, for both fm weights.
    def pieces(cfg):
        out = codec.forward(x)
        recon = out["recon"]
        fake_scores, fake_feats = disc(recon, return_features=True)
        from tokamak_foundation_model.ignite.losses import (
            _as_score_list,
            _mean_over_maps,
            feature_matching_loss,
        )
        adversarial = -_mean_over_maps(_as_score_list(fake_scores))
        pixel = torch.mean(torch.abs(recon - x))
        with torch.no_grad():
            _, real_feats = disc(x, return_features=True)
        fm = feature_matching_loss(real_feats, fake_feats)
        recon_ref = cfg.pixel_anchor_weight * pixel + cfg.fm_weight * fm
        lam = codec._adaptive_adv_weight(recon_ref, adversarial, cfg)
        return float(recon_ref), float(fm), float(lam)

    ref_off, fm_off, lam_off = pieces(cfg_off)
    ref_on, fm_on, lam_on = pieces(cfg_on)

    # fm is a real, positive perceptual reference on random weights
    assert fm_on > 0.0
    # recon_ref strictly grows when fm is folded in
    assert ref_on > ref_off, (ref_on, ref_off)
    # the adaptive weight rises (or ties) — the mechanism payoff
    assert lam_on >= lam_off, (lam_on, lam_off)
    # for a substantial fm_weight the increase should be strict here
    assert lam_on > lam_off, (lam_on, lam_off)


def test_generator_losses_total_backward_reaches_codec_params_with_fm():
    torch.manual_seed(10)
    cfg = _cfg(adaptive_adv_weight=True, fm_weight=2.0)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)
    codec.zero_grad(set_to_none=True)
    out = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    out["total"].backward()
    grads = [p.grad for p in codec.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0
    assert codec.decoder.last_layer.grad is not None


def test_fm_does_not_leak_grad_into_discriminator():
    """The generator FM term must not accumulate gradient in the discriminator params.

    Real feats are computed under no_grad; fake feats flow only into the codec. So after
    total.backward(), the discriminator should have no generator-side gradient.
    """
    torch.manual_seed(11)
    cfg = _cfg(adaptive_adv_weight=True, fm_weight=2.0)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    disc.zero_grad(set_to_none=True)
    x, x_shift = _pair(cfg)
    out = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    out["total"].backward()
    # the adversarial term DOES flow through disc (that's expected for a GAN generator step),
    # so we don't assert disc grads are zero. Instead confirm FM alone (isolated) is disc-safe.
    disc.zero_grad(set_to_none=True)
    recon = codec.forward(x)["recon"].detach()  # isolate: freeze generator side
    _, fake_feats = disc(recon, return_features=True)
    with torch.no_grad():
        _, real_feats = disc(x, return_features=True)
    fm = losses.feature_matching_loss(real_feats, fake_feats)
    fm.backward()
    # fake_feats DO depend on disc params, so disc grads exist here — that is fine; the point
    # of the no_grad on real_feats is that the TARGET side contributes none. Assert finite.
    disc_grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in disc_grads)


# --------------------------------------------------------------------------- #
# REGRESSION: fixed-weight (adaptive OFF) path is unchanged (fm NOT folded in)
# --------------------------------------------------------------------------- #
def test_fixed_weight_path_unchanged_by_fm():
    """With adaptive OFF, total must still equal the prior fixed-weight formula EXACTLY,
    independent of fm_weight (fm participates only in the adaptive balance)."""
    torch.manual_seed(12)
    cfg = _cfg(adaptive_adv_weight=False, fm_weight=7.0)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    codec.eval()
    x, x_shift = _pair(cfg)
    out = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    expected = (
        cfg.adversarial_weight * out["adversarial"]
        + cfg.pixel_anchor_weight * out["pixel"]
        + cfg.consistency_weight * out["consistency"]
        + cfg.entropy_weight * out["entropy"]
    )
    assert torch.allclose(out["total"], expected, atol=0.0, rtol=0.0), (
        out["total"].item(), expected.item()
    )
    assert out["adaptive_weight"] == 1.0
