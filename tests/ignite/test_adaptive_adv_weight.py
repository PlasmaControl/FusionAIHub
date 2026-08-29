"""Tests for the VQGAN/MagViT-style ADAPTIVE ADVERSARIAL WEIGHT (Taming Transformers §3.3).

Only this lever is under test (no LeCAM / EMA / spectral-norm). CPU synthetic tensors.

Contract (config.py + codec.SpectroCodec.generator_losses):

    lam = ‖∇ref‖ / (‖∇adv‖ + 1e-4)  clamped to [0, cfg.adaptive_adv_clamp], DETACHED,
    where ref = pixel_anchor_weight*pixel + consistency_weight*consistency
                + entropy_weight*entropy   (the NON-adversarial total)
    and   adv = -mean(D(recon))            (raw generator adversarial term),
    both grads taken at ``decoder.last_layer`` (= to_pixels.weight).

    adaptive ON : total = ref + adv_coeff * adv   (adv_coeff = adversarial_weight * lam)
    adaptive OFF: total = adversarial_weight*adv + ref   (== prior fixed-weight behavior)
"""
from __future__ import annotations

import math

import torch

from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN
from tokamak_foundation_model.ignite.nets import SpectroDecoder


def _small_cfg(**overrides) -> SpectroCodecConfig:
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
# SpectroDecoder.last_layer
# --------------------------------------------------------------------------- #
def test_decoder_last_layer_is_final_weight():
    cfg = _small_cfg()
    dec = SpectroDecoder(cfg)
    assert dec.last_layer is dec.to_pixels.weight
    assert isinstance(dec.last_layer, torch.nn.Parameter)
    # it is the weight of the layer that produces the (B,C,F,T) output (patch_dim rows)
    patch_dim = cfg.channels * cfg.patch_f * cfg.patch_t
    assert dec.last_layer.shape == (patch_dim, cfg.d_model)


def test_last_layer_receives_recon_gradient():
    """The exposed last_layer actually carries gradient of the reconstruction output."""
    cfg = _small_cfg()
    dec = SpectroDecoder(cfg)
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    recon = dec(quant)
    recon.sum().backward()
    assert dec.last_layer.grad is not None
    assert dec.last_layer.grad.abs().sum() > 0.0


# --------------------------------------------------------------------------- #
# lam: finite, >= 0, detached, python float
# --------------------------------------------------------------------------- #
def test_adaptive_weight_finite_nonneg_detached_float():
    torch.manual_seed(0)
    cfg = _small_cfg(adaptive_adv_weight=True)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)

    out = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    lam = out["adaptive_weight"]
    assert isinstance(lam, float), type(lam)
    assert math.isfinite(lam)
    assert lam >= 0.0
    # detached: the reported scalar carries no autograd history (it is a python float),
    # and the returned dict's tensor terms are unaffected. Confirm total still backprops.
    assert out["total"].requires_grad


def test_lam_is_detached_no_grad_leak():
    """The internal lam must be detached so it acts as a constant scale, not a grad path."""
    torch.manual_seed(0)
    cfg = _small_cfg(adaptive_adv_weight=True)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)

    # reach into the helper directly to inspect the tensor lam
    out = codec.forward(x)
    from tokamak_foundation_model.ignite.losses import recon_objective, shift_consistency

    rt = recon_objective(out["recon"], x, disc, cfg)
    consistency = shift_consistency(out["feats"], codec.encode(x_shift))
    entropy = codec.quantizer.entropy_loss(out["feats"])
    ref = (
        cfg.pixel_anchor_weight * rt["pixel"]
        + cfg.consistency_weight * consistency
        + cfg.entropy_weight * entropy
    )
    lam_t = codec._adaptive_adv_weight(ref, rt["adversarial"], cfg)
    assert isinstance(lam_t, torch.Tensor)
    assert lam_t.dim() == 0
    assert not lam_t.requires_grad, "lam must be detached"
    assert torch.isfinite(lam_t).all()
    assert float(lam_t) >= 0.0


# --------------------------------------------------------------------------- #
# lam scales DOWN a dominant adversarial gradient
# --------------------------------------------------------------------------- #
def test_lam_below_one_when_adv_grad_dominates():
    """If ∇adv at last_layer >> ∇ref, lam < 1 (adv is scaled DOWN)."""
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    last_layer = codec.decoder.last_layer

    # Construct two scalar losses with controlled gradient magnitudes at last_layer.
    # adv depends STRONGLY on last_layer, ref only WEAKLY.
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    recon = codec.decoder(quant)          # depends on last_layer
    adv = 100.0 * recon.pow(2).mean()     # large-gradient term
    ref = 1e-3 * recon.mean()             # small-gradient term

    lam = codec._adaptive_adv_weight(ref, adv, cfg)
    assert float(lam) < 1.0, float(lam)
    assert float(lam) > 0.0


def test_lam_above_one_when_ref_grad_dominates():
    """Symmetric check: if ∇ref >> ∇adv, lam > 1 (adv is scaled UP)."""
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    recon = codec.decoder(quant)
    ref = 100.0 * recon.pow(2).mean()
    adv = 1e-3 * recon.mean()
    lam = codec._adaptive_adv_weight(ref, adv, cfg)
    assert float(lam) > 1.0, float(lam)


# --------------------------------------------------------------------------- #
# balancing property: with adaptive ON the effective adv-grad-norm ≈ ref-grad-norm
# --------------------------------------------------------------------------- #
def test_adaptive_balances_grad_norms_at_last_layer():
    """After applying lam, ‖∇(lam*adv)‖ ≈ ‖∇ref‖ at last_layer (the balancing property).

    (Ignoring the extra adversarial_weight factor, which is 1.0 by default.)
    """
    torch.manual_seed(1)
    cfg = _small_cfg(adaptive_adv_weight=True, adversarial_weight=1.0)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)

    out = codec.forward(x)
    from tokamak_foundation_model.ignite.losses import recon_objective, shift_consistency

    rt = recon_objective(out["recon"], x, disc, cfg)
    consistency = shift_consistency(out["feats"], codec.encode(x_shift))
    entropy = codec.quantizer.entropy_loss(out["feats"])
    ref = (
        cfg.pixel_anchor_weight * rt["pixel"]
        + cfg.consistency_weight * consistency
        + cfg.entropy_weight * entropy
    )
    adv = rt["adversarial"]

    ll = codec.decoder.last_layer
    g_ref = torch.autograd.grad(ref, ll, retain_graph=True)[0].norm()
    g_adv = torch.autograd.grad(adv, ll, retain_graph=True)[0].norm()
    lam = codec._adaptive_adv_weight(ref, adv, cfg)

    eff_adv_grad = float(lam) * float(g_adv)   # ‖∇(lam*adv)‖ = lam * ‖∇adv‖
    # balancing: eff_adv_grad ≈ g_ref  (the ratio removes the imbalance)
    #   lam*‖∇adv‖ = (‖∇ref‖/(‖∇adv‖+1e-4)) * ‖∇adv‖ ≈ ‖∇ref‖ when ‖∇adv‖ >> 1e-4
    ratio = eff_adv_grad / (float(g_ref) + 1e-12)
    assert 0.9 <= ratio <= 1.1, (ratio, float(g_ref), float(g_adv), float(lam))


# --------------------------------------------------------------------------- #
# total.backward() reaches codec params (adaptive ON)
# --------------------------------------------------------------------------- #
def test_adaptive_total_backward_reaches_codec_params():
    torch.manual_seed(2)
    cfg = _small_cfg(adaptive_adv_weight=True)
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
    # the balanced layer receives gradient
    assert codec.decoder.last_layer.grad is not None


def test_adaptive_generator_step_runs_and_is_finite():
    """A full Adam step under adaptive weighting stays finite (no NaN blowup)."""
    torch.manual_seed(3)
    cfg = _small_cfg(adaptive_adv_weight=True)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-3)
    x, x_shift = _pair(cfg)

    opt.zero_grad(set_to_none=True)
    out = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    out["total"].backward()
    opt.step()
    for p in codec.parameters():
        assert torch.isfinite(p).all()


# --------------------------------------------------------------------------- #
# adv_warmup_steps: adv term is OFF (coeff 0) before warmup completes
# --------------------------------------------------------------------------- #
def test_adv_warmup_zeroes_adversarial_contribution():
    torch.manual_seed(4)
    cfg = _small_cfg(adaptive_adv_weight=True, adv_warmup_steps=10)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)

    # during warmup: total == non_adv_total (adv contributes 0). The reconstruction
    # reference now also carries the disc feature-matching term (cfg.fm_weight * fm), so it
    # is part of non_adv_total too.
    codec.eval()  # deterministic between the two forwards
    out_warm = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    rt_ref = (
        cfg.pixel_anchor_weight * out_warm["pixel"]
        + cfg.fm_weight * out_warm["feature_matching"]
        + cfg.consistency_weight * out_warm["consistency"]
        + cfg.entropy_weight * out_warm["entropy"]
    )
    assert torch.allclose(out_warm["total"], rt_ref, atol=1e-6), (
        out_warm["total"].item(), rt_ref.item()
    )

    # after warmup: total != ref (adv contributes) — unless lam happens to be 0.
    out_hot = codec.generator_losses(x, x_shift, disc, cfg, step=10)
    if out_hot["adaptive_weight"] > 0.0 and abs(out_hot["adversarial"].item()) > 1e-8:
        assert not torch.allclose(out_hot["total"], rt_ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# REGRESSION GUARD: adaptive OFF reproduces the prior fixed-weight total EXACTLY
# --------------------------------------------------------------------------- #
def test_fixed_weight_path_matches_prior_formula_exactly():
    """cfg.adaptive_adv_weight=False must be numerically identical to the old total.

    Old total = adversarial_weight*adv + pixel_anchor_weight*pixel
                + consistency_weight*consistency + entropy_weight*entropy.
    """
    torch.manual_seed(5)
    cfg = _small_cfg(adaptive_adv_weight=False)
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


def test_fixed_weight_reference_recomputes_old_recon_objective_total():
    """The fixed path's total also equals recon_objective.total + consistency + entropy.

    This mirrors the EXACT expression the code used before this change, guarding against
    a silent restructuring drift.
    """
    torch.manual_seed(6)
    cfg = _small_cfg(adaptive_adv_weight=False)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    codec.eval()
    x, x_shift = _pair(cfg)

    from tokamak_foundation_model.ignite.losses import recon_objective, shift_consistency

    out = codec.forward(x)
    rt = recon_objective(out["recon"], x, disc, cfg)
    consistency = shift_consistency(out["feats"], codec.encode(x_shift))
    entropy = codec.quantizer.entropy_loss(out["feats"])
    old_total = (
        rt["total"]
        + cfg.consistency_weight * consistency
        + cfg.entropy_weight * entropy
    )

    got = codec.generator_losses(x, x_shift, disc, cfg, step=0)["total"]
    assert torch.allclose(got, old_total, atol=1e-6)


def test_fixed_weight_default_arg_is_backward_compatible():
    """generator_losses must still be callable WITHOUT the new step arg (default 0)."""
    cfg = _small_cfg(adaptive_adv_weight=False)
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x, x_shift = _pair(cfg)
    out = codec.generator_losses(x, x_shift, disc, cfg)  # no step kwarg
    assert "total" in out and torch.isfinite(out["total"]).all()
    assert "adaptive_weight" in out


# --------------------------------------------------------------------------- #
# STABILIZATION FIX 1: clamp lam to cfg.adaptive_adv_clamp (default 50, not 1e4)
# --------------------------------------------------------------------------- #
def test_default_clamp_is_fifty():
    """The stabilized default clamp is 50.0 across all three adversarial codecs."""
    from tokamak_foundation_model.ignite.config import (
        FastTSCodecConfig,
        SpectroCodecConfig,
        VideoCodecConfig,
    )

    assert SpectroCodecConfig().adaptive_adv_clamp == 50.0
    assert VideoCodecConfig().adaptive_adv_clamp == 50.0
    assert FastTSCodecConfig().adaptive_adv_clamp == 50.0


def test_lam_clamps_to_fifty_when_ratio_would_explode():
    """A collapsing codec drives ‖∇adv‖ -> ~0 so the raw ratio would run away (the crash
    was lam -> ~729). With the default clamp=50, lam must saturate at EXACTLY 50, not 729."""
    cfg = _small_cfg()  # default adaptive_adv_clamp == 50.0
    assert cfg.adaptive_adv_clamp == 50.0
    codec = SpectroCodec(cfg)

    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    recon = codec.decoder(quant)                 # depends on last_layer
    # HUGE ref grad, NEGLIGIBLE adv grad at last_layer -> raw ratio >> 50 (would be ~1e6+).
    ref = 1.0e6 * recon.pow(2).mean()
    adv = 1.0e-9 * recon.mean()

    lam = codec._adaptive_adv_weight(ref, adv, cfg)
    assert float(lam) == 50.0, float(lam)


def test_lam_clamp_tracks_cfg_value():
    """The clamp is READ FROM cfg (not hardcoded): shrinking cfg.adaptive_adv_clamp shrinks
    the saturated lam identically for the same exploding ratio."""
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    quant = torch.randn(2, cfg.n_tok, cfg.d_model)
    recon = codec.decoder(quant)
    ref = 1.0e6 * recon.pow(2).mean()
    adv = 1.0e-9 * recon.mean()

    for clamp in (5.0, 50.0, 123.0):
        cfg.adaptive_adv_clamp = clamp
        lam = codec._adaptive_adv_weight(ref, adv, cfg)
        assert float(lam) == clamp, (clamp, float(lam))
