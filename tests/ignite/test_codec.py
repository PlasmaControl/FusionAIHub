"""CPU smoke tests for the composed Phase-A codec + the gate spike.

Contract (docs/IGNITE_DESIGN.md §4, and the component signatures):

    codec.SpectroCodec(cfg)
        .encode(x: (B,C,F,T)) -> feats (B, n_tok, d_model)
        .forward(x)           -> dict(recon (B,C,F,T), feats, quant, codes)
        .generator_losses(x, x_shift, disc, cfg) -> dict(total, adversarial, pixel,
                                                          consistency, recon, codes)

    spike.synthetic_batches(cfg, n_batches, batch_size, device) -> [(spec_a, spec_b), ...]
    spike.run_spike(cfg, batches, steps, device, eval_every) -> gate dict

Small synthetic tensors only, CPU. No real HDF5. No SLURM / GPU.
"""
from __future__ import annotations

import pytest
import torch

from tokamak_foundation_model.ignite import spike
from tokamak_foundation_model.ignite.codec import SpectroCodec
from tokamak_foundation_model.ignite.config import SpectroCodecConfig
from tokamak_foundation_model.ignite.discriminator import FreqAwarePatchGAN


def _small_cfg() -> SpectroCodecConfig:
    """Tiny transformer + small patch grid, but full-STFT-grid freq/time so the
    data-glue crop/pad path exercises realistic sizes."""
    return SpectroCodecConfig(
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


# --------------------------------------------------------------------------- #
# SpectroCodec forward shapes
# --------------------------------------------------------------------------- #
def test_codec_forward_shapes():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    out = codec(x)
    assert set(out) == {"recon", "feats", "quant", "codes"}
    assert out["recon"].shape == (B, cfg.channels, cfg.freq_bins, cfg.time_frames)
    assert out["feats"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["quant"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["codes"].shape == (B, cfg.n_tok, cfg.fsq_dim)
    assert out["codes"].dtype == torch.long
    assert torch.isfinite(out["recon"]).all()


def test_codec_encode_matches_forward_feats():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    codec.eval()  # no dropout/randomness between the two calls
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    feats = codec.encode(x)
    out = codec(x)
    assert torch.allclose(feats, out["feats"], atol=1e-5)


def test_codec_codes_in_range():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    x = torch.randn(4, cfg.channels, cfg.freq_bins, cfg.time_frames) * 5.0
    codes = codec(x)["codes"]
    levels = torch.tensor(cfg.fsq_levels)
    assert (codes >= 0).all()
    assert (codes < levels).all()
    assert codec.codebook_size == cfg.codebook_size


def test_codec_roundtrip_at_default_fsq_size():
    """The full encode -> quantize -> decode chain works at the right-sized default FSQ
    config ([8, 5, 5, 5] = 1000 codes, 4 dims). The nets project via d_model; the quantizer
    handles d_model <-> fsq_dim, so this proves the whole chain is config-driven at 4 dims."""
    cfg = SpectroCodecConfig(
        channels=1, freq_bins=64, time_frames=32, patch_f=32, patch_t=16,
        d_model=32, enc_depth=1, dec_depth=1, heads=2,
    )  # default fsq_levels [8, 5, 5, 5]
    assert cfg.fsq_dim == 4 and cfg.codebook_size == 1000
    codec = SpectroCodec(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames) * 5.0
    out = codec(x)
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (2, cfg.n_tok, 4)
    levels = torch.tensor(cfg.fsq_levels)
    assert (out["codes"] >= 0).all() and (out["codes"] < levels).all()
    assert codec.codebook_size == 1000
    assert torch.isfinite(out["recon"]).all()


# --------------------------------------------------------------------------- #
# generator_losses: one step reduces the total generator loss
# --------------------------------------------------------------------------- #
def test_generator_losses_keys_and_finite():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.freq_bins, cfg.time_frames)
    x_shift = x + 0.01 * torch.randn_like(x)
    out = codec.generator_losses(x, x_shift, disc, cfg)
    for k in ("total", "adversarial", "pixel", "consistency"):
        assert k in out and out[k].dim() == 0 and torch.isfinite(out[k]).all()
    assert out["recon"].shape == x.shape
    assert out["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)


def test_generator_step_reduces_total_loss():
    """One Adam step on the generator loss should lower it on the SAME batch."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    disc = FreqAwarePatchGAN(cfg)
    opt = torch.optim.Adam(codec.parameters(), lr=1e-2)

    batches = spike.synthetic_batches(cfg, n_batches=1, batch_size=2, seed=1)
    x, x_shift = batches[0]

    codec.train()
    loss_before = codec.generator_losses(x, x_shift, disc, cfg)["total"]
    opt.zero_grad(set_to_none=True)
    loss_before.backward()
    # gradient actually reaches codec params
    grads = [p.grad for p in codec.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
    opt.step()

    loss_after = codec.generator_losses(x, x_shift, disc, cfg)["total"]
    assert loss_after.item() < loss_before.item(), (
        loss_before.item(), loss_after.item()
    )


# --------------------------------------------------------------------------- #
# gate functions run on codec outputs and return finite values
# --------------------------------------------------------------------------- #
def test_gate_runs_on_codec_outputs():
    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    eval_pairs = spike.synthetic_batches(cfg, n_batches=1, batch_size=2, seed=5)
    frame_seq = spike._consecutive_frame_sequence(cfg, batch_size=2, n_frames=4, seed=6)

    g = spike.compute_gate(codec, eval_pairs, frame_seq, cfg)
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    assert isinstance(g["pass_stability"], bool)
    assert isinstance(g["pass_persistence"], bool)

    fc = g["forecastability"]
    for key in ("margin_transition", "margin_stable", "margin_overall", "beats_persistence"):
        assert key in fc
    assert isinstance(fc["beats_persistence"], bool)

    dec = g["decode"]
    for key in ("envelope_corr", "peak_f1", "sharpness"):
        assert key in dec
        assert isinstance(dec[key], float)
    # correlation / f1 are bounded; sharpness is a finite ratio
    assert -1.0 <= dec["envelope_corr"] <= 1.0
    assert 0.0 <= dec["peak_f1"] <= 1.0
    import math as _math
    assert _math.isfinite(dec["sharpness"])


def test_compute_gate_carries_full_spectrogram_metrics_without_moving_the_score():
    """Every gate dict now carries the FULL-spectrogram recon metrics + trivial baselines,
    and best-checkpoint selection is unchanged by them.

    The keys are informational: ``gate_score`` reads envelope_corr / peak_f1 / margin /
    utilization only, so dropping every new key must give a BIT-identical score.
    """
    import math as _math

    cfg = _small_cfg()
    codec = SpectroCodec(cfg)
    eval_pairs = spike.synthetic_batches(cfg, n_batches=1, batch_size=2, seed=5)
    frame_seq = spike._consecutive_frame_sequence(cfg, batch_size=2, n_frames=4, seed=6)

    g = spike.compute_gate(codec, eval_pairs, frame_seq, cfg)
    dec = g["decode"]
    for key in ("spec_nrmse", "spec_corr2d", "spec_valid_frac",
                "spec_nrmse_band", "spec_corr2d_band",
                "base_self_spec_nrmse", "base_self_spec_corr2d",
                "base_tmean_spec_nrmse", "base_tmean_spec_corr2d",
                "base_cfmean_spec_nrmse", "base_wcmean_spec_nrmse"):
        assert key in dec and isinstance(dec[key], float), key
    assert dec["spec_nrmse"] >= 0.0 and -1.0 <= dec["spec_corr2d"] <= 1.0
    # the metric self-check survives the whole pipeline
    assert dec["base_self_spec_nrmse"] == pytest.approx(0.0, abs=1e-9)
    assert dec["base_self_spec_corr2d"] == pytest.approx(1.0, abs=1e-9)
    assert dec["base_wcmean_spec_nrmse"] == pytest.approx(1.0, abs=1e-9)

    stripped = dict(g)
    stripped["decode"] = {k: v for k, v in dec.items()
                          if not (k.startswith("spec_") or k.startswith("base_"))}
    a, b = spike.gate_score(g), spike.gate_score(stripped)
    assert a == b or (_math.isnan(a) and _math.isnan(b))


# --------------------------------------------------------------------------- #
# synthetic_batches shape contract
# --------------------------------------------------------------------------- #
def test_synthetic_batches_shapes():
    cfg = _small_cfg()
    batches = spike.synthetic_batches(cfg, n_batches=3, batch_size=2, seed=0)
    assert len(batches) == 3
    for spec_a, spec_b in batches:
        assert spec_a.shape == (2, cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert spec_b.shape == (2, cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert torch.isfinite(spec_a).all() and torch.isfinite(spec_b).all()
        # δ-shift pair shares modes but is not identical
        assert not torch.allclose(spec_a, spec_b, atol=1e-4)


# NOTE: real_ece_batches is now fully implemented (loads real ece HDF5). Its behaviour
# is verified against real data in scratch/verify_real_ece_path.py, not a hermetic CPU
# unit test — so the former "must remain a stub" assertion has been removed.


# --------------------------------------------------------------------------- #
# run_spike end-to-end (~3 steps) on synthetic data
# --------------------------------------------------------------------------- #
def test_run_spike_returns_gate_dict():
    cfg = _small_cfg()
    batches = spike.synthetic_batches(cfg, n_batches=3, batch_size=2, seed=2)
    g = spike.run_spike(
        cfg, batches, steps=3, device="cpu", eval_every=1, log_fn=None, seed=3
    )
    for key in ("stability", "persistence", "forecastability", "decode",
                "pass_stability", "pass_persistence", "steps"):
        assert key in g, f"missing gate key '{key}'"
    assert g["steps"] == 3
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    import math as _math
    assert _math.isfinite(g["forecastability"]["margin_transition"])
    assert _math.isfinite(g["decode"]["envelope_corr"])


# --------------------------------------------------------------------------- #
# MISSING-DATA MASKING (2026-09-03) — the spectro half of the audit that found the
# video codec had no validity mask at all. The spectro path was worse: the dataset
# discarded the loader's nan_mask entirely, so no term, no discriminator step and no
# gate statistic ever excluded a dead channel or an eps-floor last-resort window.
#
# The no-op proof is the load-bearing test. `mask=None`, an all-ones (B, T) mask and an
# all-ones (B, C, T) mask must give BIT-IDENTICAL loss terms, because that is what makes
# the default path a literal no-change rather than "numerically close".
# --------------------------------------------------------------------------- #
def _mask_cfg() -> SpectroCodecConfig:
    cfg = SpectroCodecConfig(
        channels=3, freq_bins=64, time_frames=32, patch_f=16, patch_t=16,
        d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
    )
    # switch ON every optional recon term so the no-op proof covers all of them.
    cfg.multiscale_recon_weight = 0.3
    cfg.freq_grad_weight = 0.2
    cfg.ms_ssim_weight = 1.0
    cfg.fm_weight = 0.5
    cfg.consistency_weight = 0.1
    cfg.entropy_weight = 0.1
    return cfg


_MASK_TERMS = ("total", "adversarial", "pixel", "multiscale", "freq_grad", "ms_ssim",
               "feature_matching", "consistency", "entropy")


def _gen_terms(codec, disc, cfg, x, xs, mask):
    torch.manual_seed(7)
    out = codec.generator_losses(x, xs, disc, cfg, step=5, frame_mask=mask)
    return {k: out[k].detach().clone() for k in _MASK_TERMS}


def test_spectro_mask_is_bit_identical_when_everything_is_valid():
    torch.manual_seed(0)
    cfg = _mask_cfg()
    codec, disc = SpectroCodec(cfg), FreqAwarePatchGAN(cfg)
    B, C, T = 4, cfg.channels, cfg.time_frames
    x = torch.randn(B, C, cfg.freq_bins, T)
    xs = torch.randn(B, C, cfg.freq_bins, T)

    base = _gen_terms(codec, disc, cfg, x, xs, None)
    for tag, m in (("(B,T)", torch.ones(B, T)), ("(B,C,T)", torch.ones(B, C, T))):
        got = _gen_terms(codec, disc, cfg, x, xs, m)
        for k in _MASK_TERMS:
            assert torch.equal(base[k], got[k]), (
                f"ones{tag} changed '{k}': {base[k].item()!r} vs {got[k].item()!r}"
            )


def test_spectro_discriminator_step_mask_is_bit_identical_when_all_valid():
    from tokamak_foundation_model.ignite import train_codec as tc

    torch.manual_seed(0)
    cfg = _mask_cfg()
    codec, disc = SpectroCodec(cfg), FreqAwarePatchGAN(cfg)
    B, C, T = 4, cfg.channels, cfg.time_frames
    x = torch.randn(B, C, cfg.freq_bins, T)
    with torch.no_grad():
        recon = codec.forward(x)["recon"]
        vals = [
            float(tc._spectro_discriminator_loss(disc, x, recon, cfg, m))
            for m in (None, torch.ones(B, T), torch.ones(B, C, T))
        ]
    assert vals[0] == vals[1] == vals[2], f"D-step mask is not a no-op: {vals}"


def test_spectro_mask_actually_excludes_a_dead_channel():
    """A real mask MUST move the terms — otherwise the no-op test above is vacuous."""
    torch.manual_seed(0)
    cfg = _mask_cfg()
    codec, disc = SpectroCodec(cfg), FreqAwarePatchGAN(cfg)
    B, C, T = 4, cfg.channels, cfg.time_frames
    x = torch.randn(B, C, cfg.freq_bins, T)
    xs = torch.randn(B, C, cfg.freq_bins, T)
    m = torch.ones(B, C, T)
    m[0, 1, :] = 0.0                      # one dead channel in window 0
    base = _gen_terms(codec, disc, cfg, x, xs, None)
    got = _gen_terms(codec, disc, cfg, x, xs, m)
    assert any(not torch.equal(base[k], got[k]) for k in _MASK_TERMS)
    assert not torch.equal(base["pixel"], got["pixel"])      # (b,c,t)-granular
    assert not torch.equal(base["entropy"], got["entropy"])  # window-granular


def test_spectro_frame_mask_flags_dead_channels_nans_and_zero_fill():
    from tokamak_foundation_model.ignite.data import spectro_frame_mask

    cfg = _mask_cfg()
    W = cfg.window_samples
    torch.manual_seed(0)
    raw = torch.randn(3, W)
    nan = torch.zeros(3, W)
    assert float(spectro_frame_mask(raw, nan, cfg).min()) == 1.0     # all live

    dead = raw.clone()
    dead[1] = 0.0                                                    # zero-slab channel
    m = spectro_frame_mask(dead, nan, cfg)
    assert float(m[1].max()) == 0.0 and float(m[0].min()) == 1.0

    nan2 = torch.zeros(3, W)
    nan2[2] = 1.0                                                    # all-NaN channel
    m2 = spectro_frame_mask(raw, nan2, cfg)
    assert float(m2[2].max()) == 0.0 and float(m2[0].min()) == 1.0

    # Window straddling the record edge: the loader zero-fills the part outside
    # [xdata[0], xdata[-1]]. Only the first cfg.time_frames * stft_hop samples survive the
    # time crop, so zero a slice INSIDE that span to get a partially-valid mask.
    edge = raw.clone()
    edge[:, : 8 * cfg.stft_hop] = 0.0
    m3 = spectro_frame_mask(edge, nan, cfg)
    assert 0.0 < float(m3.mean()) < 1.0, float(m3.mean())
    assert float(m3[:, 0].max()) == 0.0 and float(m3[:, -1].min()) == 1.0


def test_spectro_gate_mask_is_a_noop_and_keeps_the_wcmean_anchor_at_one():
    import numpy as np

    from tokamak_foundation_model.ignite import gate

    rng = np.random.default_rng(0)
    B, C, F, T = 4, 3, 64, 32
    t = rng.standard_normal((B, C, F, T))
    r = 0.8 * t + 0.2 * rng.standard_normal((B, C, F, T))
    keys = ("spec_nrmse", "spec_corr2d", "base_wcmean_spec_nrmse",
            "base_tmean_spec_nrmse", "envelope_corr", "sharpness", "patch_lattice_ratio")

    def run(m):
        d = gate.decode_fidelity(r, t, patch_f=16, patch_t=16, full_spec=True, mask=m)
        return {k: d[k] for k in keys}

    base = run(None)
    for m in (np.ones((B, T)), np.ones((B, C, T))):
        assert run(m) == base
    # base_wcmean IS the definition of the 1.0 normalisation anchor: it must stay exactly
    # 1.0 under masking, which is only true if the baseline's OWN mean is masked too.
    assert base["base_wcmean_spec_nrmse"] == pytest.approx(1.0, abs=1e-9)
    dead = np.ones((B, C, T))
    dead[0, 1, :] = 0.0
    assert run(dead)["base_wcmean_spec_nrmse"] == pytest.approx(1.0, abs=1e-9)
    part = np.ones((B, C, T))
    part[:, :, :8] = 0.0
    assert run(part)["base_wcmean_spec_nrmse"] == pytest.approx(1.0, abs=1e-9)


def test_spectro_gate_mask_equals_deleting_the_masked_data():
    """The real correctness proof for the masked gate statistics.

    The no-op test shows masking changes nothing when nothing is missing; this shows the
    masked statistics are genuinely computed over VALID SAMPLES ONLY -- masking a channel is
    bit-identical to deleting that channel and scoring the remainder, and masking a tail of
    time frames is bit-identical to cropping them. Without this, a mask that merely produced
    *different* numbers would pass the no-op test while silently mis-weighting the average.
    """
    import numpy as np

    from tokamak_foundation_model.ignite import gate

    rng = np.random.default_rng(3)
    B, C, F, T = 6, 4, 32, 16
    t = rng.standard_normal((B, C, F, T))
    r = 0.7 * t + 0.3 * rng.standard_normal((B, C, F, T))

    m = np.ones((B, C, T))
    m[:, 2, :] = 0.0                                    # channel 2 dead everywhere
    masked = gate.full_spectro_metrics(r, t, band_bins=None, mask=m)
    kept = [c for c in range(C) if c != 2]
    deleted = gate.full_spectro_metrics(r[:, kept], t[:, kept], band_bins=None)
    assert masked["spec_nrmse"] == pytest.approx(deleted["spec_nrmse"], abs=1e-12)
    assert masked["spec_corr2d"] == pytest.approx(deleted["spec_corr2d"], abs=1e-12)

    m2 = np.ones((B, C, T))
    m2[:, :, 10:] = 0.0                                 # window straddles the record edge
    mask_t = gate.full_spectro_metrics(r, t, band_bins=None, mask=m2)
    crop_t = gate.full_spectro_metrics(r[..., :10], t[..., :10], band_bins=None)
    assert mask_t["spec_nrmse"] == pytest.approx(crop_t["spec_nrmse"], abs=1e-12)
    assert mask_t["spec_corr2d"] == pytest.approx(crop_t["spec_corr2d"], abs=1e-12)
