"""CPU TDD spec for the IGNITE Phase-A fast-TS (filterscopes) RAW-SAMPLE codec family.

2026-09-03 REDESIGN: the codec target is the RAW 10 kHz waveform, sample by sample — the
ELM-envelope (RMS / pooling / log1p) layer this file used to spec is GONE.

    data.fastts_raw_window(raw (B,C,W), cfg) -> raw (B, C, window)   [the codec target]
    data.fastts_shift_pair_windows(raw_shot, t0, cfg, δ) -> (win_a, win_b)  [δ-shift pair]
    fastts_nets.FastTSEncoder(cfg)(x (B,C,W)) -> feats (B, n_tok, d_model)
    fastts_nets.FastTSDecoder(cfg)(quant (B,n_tok,d_model)) -> recon (B, C, W)
    fastts_codec.FastTSCodec(cfg).forward(x) -> dict(recon, feats, quant, codes)
    fastts_codec.FastTSCodec.generator_losses(x, x_shift, disc, cfg, step) -> dict(total,...)
    fastts_discriminator.Env1DPatchGAN(cfg)(x) -> list of 1-D patch-score maps
    gate.fastts_decode_fidelity(recon, target) -> dict(envelope_corr, peak_f1, sharpness)
    gate.full_fastts_metrics / gate.trivial_fastts_baselines -> sample-wise nRMSE + baselines

    fastts_train.FastTSCodecPairDataset  (subclass of TokamakMultiFileDataset)
    fastts_train.fastts_compute_gate / fastts_train.train_fastts_codec

Small synthetic tensors + tiny SYNTHETIC filterscopes HDF5 shots (spiky ELM-like signals)
only. CPU. No SLURM / GPU / real data.

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_fastts_codec.py -q
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.ignite import data as idata
from tokamak_foundation_model.ignite import fastts_train as ft
from tokamak_foundation_model.ignite import gate, spike
from tokamak_foundation_model.ignite.config import (
    FASTTS_PATCH_W,
    fastts_raw_config,
    FASTTS_FS,
    FASTTS_WINDOW,
    FastTSCodecConfig,
)
from tokamak_foundation_model.ignite.fastts_codec import FastTSCodec
from tokamak_foundation_model.ignite.fastts_discriminator import Env1DPatchGAN
from tokamak_foundation_model.ignite.fastts_nets import FastTSDecoder, FastTSEncoder


# --------------------------------------------------------------------------------------- #
# tiny configs
# --------------------------------------------------------------------------------------- #
def _small_cfg(channels: int = 8) -> FastTSCodecConfig:
    """Small transformer + short raw window (divisible patching): 200 samples / 4 tokens."""
    return fastts_raw_config(
        channels=channels,
        window=200,
        patch_w=50,         # -> n_tok = 4
        stem_channels=8,
        stem_kernel=5,
        stem_layers=1,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


def _nostem_cfg(channels: int = 8) -> FastTSCodecConfig:
    """The stem-free control arm (``stem_layers=0``): plain linear patches, no conv."""
    cfg = _small_cfg(channels)
    return fastts_raw_config(
        channels=cfg.channels, window=cfg.window, patch_w=cfg.patch_w,
        stem_layers=0, d_model=cfg.d_model, enc_depth=1, dec_depth=1, heads=2,
        fsq_levels=list(cfg.fsq_levels),
    )


def _prod_geom_cfg(channels: int = 8) -> FastTSCodecConfig:
    """Small transformer at the PRODUCTION raw geometry (500 samples, patch_w=20)."""
    return fastts_raw_config(
        channels=channels, window=FASTTS_WINDOW, patch_w=FASTTS_PATCH_W,
        stem_channels=8, stem_kernel=5, stem_layers=1,
        d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
    )


# --------------------------------------------------------------------------------------- #
# config geometry
# --------------------------------------------------------------------------------------- #
def test_config_geometry_and_tokens():
    cfg = _small_cfg()
    assert cfg.n_tok == 4
    assert cfg.fsq_dim == 3 and cfg.codebook_size == 4 * 4 * 3

    # the RAW production config (opt-in): 500-sample window, patch_w 20 -> 25 tok, cb 1000.
    prod = fastts_raw_config()
    assert prod.window == FASTTS_WINDOW == round(0.05 * FASTTS_FS) == 500
    assert prod.patch_w == FASTTS_PATCH_W == 20
    assert prod.channels == 8
    assert prod.n_tok == 25
    assert prod.codebook_size == 1000 and prod.fsq_dim == 4
    assert prod.window_samples == FASTTS_WINDOW
    # the RATE arithmetic the frame budget is argued from: 25 tokens x log2(1000) bits over
    # 8 x 500 = 4000 raw values.
    assert prod.bits_per_value == pytest.approx(25 * math.log2(1000) / 4000, rel=1e-9)
    assert prod.bits_per_value == pytest.approx(0.06229, abs=1e-5)
    # the 2026-09-03 objective: reconstruction-dominant, GAN and consistency OFF.
    assert prod.pixel_anchor_weight == 1.0
    assert prod.adversarial_weight == 0.0 and prod.fm_weight == 0.0
    assert prod.consistency_weight == 0.0
    # the loss must be the METRIC's own form, not plain MSE (see FastTSCodec._recon_loss).
    assert prod.recon_loss == "nrmse"


def test_config_rejects_indivisible_patch():
    with pytest.raises(AssertionError):
        fastts_raw_config(window=500, patch_w=32)  # 500 % 32 != 0
    with pytest.raises(AssertionError):
        fastts_raw_config(stem_kernel=8)           # must be odd
    with pytest.raises(AssertionError):
        FastTSCodecConfig(env_bins=5, patch_e=2)   # envelope mode still validated


def test_config_patch_w_sets_the_token_count_and_frame_size():
    """patch_w is the RATE knob: it alone decides tokens/frame, hence the world-model frame."""
    for patch_w, n_tok in ((100, 5), (50, 10), (20, 25), (10, 50), (5, 100)):
        cfg = fastts_raw_config(patch_w=patch_w)
        assert cfg.n_tok == n_tok
        assert cfg.bits_per_value == pytest.approx(n_tok * math.log2(1000) / 4000, rel=1e-9)


# --------------------------------------------------------------------------------------- #
# RAW-window transform — correctness + sanitization (replaces the ELM-envelope block)
# --------------------------------------------------------------------------------------- #
def _elm_raw(n_ch: int, W: int, burst_bins, pool: int, seed: int = 0) -> torch.Tensor:
    """Synthetic ELM-like raw signal: quiet baseline + high-frequency spike BURSTS.

    ``burst_bins`` is an iterable of (``pool``-sample bin index) locations that get an ELM
    burst — a dense, high-amplitude oscillation filling a ~2-bin span. Returns (n_ch, W).
    """
    gen = torch.Generator().manual_seed(seed)
    x = 0.02 * torch.randn(n_ch, W, generator=gen)
    t = torch.arange(W, dtype=torch.float32)
    for bin_idx in burst_bins:
        center = (bin_idx + 0.5) * pool
        win = torch.exp(-0.5 * ((t - center) / pool) ** 2)
        carrier = torch.sin(2.0 * math.pi * (t / 2.5) + float(torch.rand((), generator=gen)))
        x = x + (5.0 * win * carrier).unsqueeze(0)
    return x


def test_fastts_raw_window_is_sample_preserving():
    """The transform keeps EVERY sample — nothing is pooled, rectified or log-compressed."""
    cfg = _small_cfg()
    raw = _elm_raw(cfg.channels, cfg.window, [1, 3], pool=50, seed=0).unsqueeze(0)
    out = idata.fastts_raw_window(raw, cfg)
    assert out.shape == (1, cfg.channels, cfg.window)
    # channel_mean/std are None here => the transform is the IDENTITY on finite input.
    assert torch.allclose(out, raw.float(), atol=1e-6)


def test_fastts_raw_window_sanitizes_nonfinite_and_absurd():
    cfg = _small_cfg()
    raw = _elm_raw(cfg.channels, cfg.window, [1], pool=50, seed=1)
    raw[0, 5] = float("nan")
    raw[1, 6] = float("inf")
    raw[2, 7] = 1e25          # absurd sentinel magnitude
    out = idata.fastts_raw_window(raw.unsqueeze(0), cfg)
    assert torch.isfinite(out).all()
    assert float(out[0, 0, 5]) == 0.0 and float(out[0, 1, 6]) == 0.0
    assert float(out[0, 2, 7]) == 0.0


def test_fastts_raw_window_crops_pads_to_window():
    cfg = _small_cfg()
    long_raw = torch.randn(1, cfg.channels, cfg.window + 37)
    short_raw = torch.randn(1, cfg.channels, cfg.window - 40)
    assert idata.fastts_raw_window(long_raw, cfg).shape[-1] == cfg.window
    padded = idata.fastts_raw_window(short_raw, cfg)
    assert padded.shape[-1] == cfg.window
    assert torch.allclose(padded[..., cfg.window - 40:], torch.zeros(1))


def test_fastts_raw_window_rejects_bad_ndim():
    cfg = _small_cfg()
    with pytest.raises(ValueError):
        idata.fastts_raw_window(torch.randn(cfg.channels, cfg.window), cfg)  # (C,W) not (B,C,W)


# --------------------------------------------------------------------------------------- #
# per-channel standardization (the SCALE FIX) on the RAW window
# --------------------------------------------------------------------------------------- #
def _highmag_cfg(std_scale: float, channels: int = 8) -> FastTSCodecConfig:
    """_small_cfg but with per-channel stats matching a ``std_scale``-magnitude raw signal."""
    cfg = _small_cfg(channels)
    return fastts_raw_config(
        channels=cfg.channels, window=cfg.window, patch_w=cfg.patch_w,
        stem_channels=8, stem_kernel=5, stem_layers=1,
        d_model=cfg.d_model, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
        channel_mean=[0.0] * channels, channel_std=[std_scale] * channels,
    )


def test_raw_standardization_rescales_high_magnitude_input():
    """A ~1e15-magnitude raw window is put on the ~O(1) scale the network is sized for."""
    cfg = _highmag_cfg(1e15)
    raw = 1e15 * torch.randn(1, cfg.channels, cfg.window)
    out = idata.fastts_raw_window(raw, cfg)
    assert torch.isfinite(out).all()
    assert 0.2 < float(out.std()) < 5.0, float(out.std())


def test_raw_standardization_preserves_activity_level():
    """GLOBAL (not per-window) standardization: a quiet window stays quieter than an active one."""
    cfg = _highmag_cfg(1.0)
    quiet = 0.02 * torch.randn(1, cfg.channels, cfg.window)
    active = _elm_raw(cfg.channels, cfg.window, [1, 2, 3], pool=50, seed=2).unsqueeze(0)
    q = idata.fastts_raw_window(quiet, cfg)
    a = idata.fastts_raw_window(active, cfg)
    assert float(a.std()) > float(q.std()) * 3.0


def test_raw_standardization_is_nrmse_invariant():
    """The per-(window, channel) nRMSE cannot see the standardization (it is affine per ch)."""
    cfg_none = _small_cfg()
    cfg_std = _highmag_cfg(3.0)
    raw = _elm_raw(cfg_none.channels, cfg_none.window, [1, 3], pool=50, seed=4).unsqueeze(0)
    x0 = idata.fastts_raw_window(raw, cfg_none).numpy()
    x1 = idata.fastts_raw_window(raw, cfg_std).numpy()
    # a fixed 'recon' rule (the window mean) scores identically in both spaces.
    import numpy as _np
    m0 = _np.broadcast_to(x0.mean(-1, keepdims=True), x0.shape)
    m1 = _np.broadcast_to(x1.mean(-1, keepdims=True), x1.shape)
    n0 = gate.full_fastts_metrics(m0, x0)["fastts_nrmse"]
    n1 = gate.full_fastts_metrics(m1, x1)["fastts_nrmse"]
    assert n0 == pytest.approx(n1, abs=1e-9) == pytest.approx(1.0, abs=1e-9)


def test_raw_standardization_still_sanitizes_nonfinite():
    cfg = _highmag_cfg(1e15)
    raw = 1e15 * torch.randn(cfg.channels, cfg.window)
    raw[0, 3] = float("nan")
    raw[1, 4] = 1e25
    out = idata.fastts_raw_window(raw.unsqueeze(0), cfg)
    assert torch.isfinite(out).all()


def test_raw_standardization_none_is_a_noop():
    cfg_none = _small_cfg()
    cfg_id = _highmag_cfg(1.0)   # mean 0 / std 1 => identity
    raw = torch.randn(2, cfg_none.channels, cfg_none.window)
    assert torch.allclose(idata.fastts_raw_window(raw, cfg_none),
                          idata.fastts_raw_window(raw, cfg_id), atol=1e-6)


def test_raw_window_rejects_wrong_length_channel_stats():
    cfg = _highmag_cfg(1.0, channels=8)
    cfg.channel_mean = [0.0] * 3          # wrong length
    with pytest.raises(ValueError):
        idata.fastts_raw_window(torch.randn(1, 8, cfg.window), cfg)


# --------------------------------------------------------------------------------------- #
# gate.full_fastts_metrics / trivial_fastts_baselines — the nRMSE contract
# --------------------------------------------------------------------------------------- #
def test_fastts_nrmse_self_and_wcmean_anchors():
    """The two MANDATORY self-checks: self -> 0.0000, wcmean -> EXACTLY 1.0000."""
    import numpy as _np
    rng = _np.random.default_rng(0)
    t = rng.normal(size=(5, 4, 200)) + 7.0 * rng.normal(size=(5, 4, 1))
    b = gate.trivial_fastts_baselines(t)
    assert b["base_self_fastts_nrmse"] == pytest.approx(0.0, abs=1e-12)
    assert b["base_self_fastts_corr"] == pytest.approx(1.0, abs=1e-12)
    assert b["base_wcmean_fastts_nrmse"] == pytest.approx(1.0, abs=1e-12)
    # for a 1-D window tmean IS wcmean (no structure axis to keep) -> also exactly 1.0000.
    assert b["base_tmean_fastts_nrmse"] == pytest.approx(1.0, abs=1e-12)
    assert b["base_tmean_fastts_nrmse"] == pytest.approx(b["base_wcmean_fastts_nrmse"])
    # cmean (one dataset-level constant per channel) is far WORSE than 1 when the window
    # level varies -- the reason the level is not free.
    assert b["base_cmean_fastts_nrmse"] > 1.0


def test_fastts_nrmse_rejects_wrong_ndim_and_mismatch():
    import numpy as _np
    with pytest.raises(ValueError):
        gate.full_fastts_metrics(_np.zeros((2, 3, 4, 5)), _np.zeros((2, 3, 4, 5)))
    with pytest.raises(ValueError):
        gate.full_fastts_metrics(_np.zeros((2, 3, 4)), _np.zeros((2, 3, 5)))


def test_fastts_nrmse_excludes_dead_channels():
    """A constant (absent/dead) channel has no structure and must not enter the mean."""
    import numpy as _np
    rng = _np.random.default_rng(1)
    t = rng.normal(size=(3, 2, 50))
    t[:, 1, :] = 4.0                      # channel 1 dead (constant)
    m = gate.full_fastts_metrics(t, t)
    assert m["fastts_valid_frac"] == pytest.approx(0.5)
    assert m["fastts_nrmse"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------- #
# δ-shift pair — REPORTING-ONLY now (a shift MOVES the target; consistency defaults OFF)
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_ms", [0.2, 1.0, 3.0, 5.0])
def test_fastts_shift_pair_shapes_and_moves_the_samples(delta_ms):
    """For a RAW-SAMPLE codec a δ-shift is NOT a nuisance — it changes every sample.

    The envelope codec asserted the two windows carried the SAME statistic (corr > 0.5 on the
    5-bin envelope) and trained a consistency loss on that. The raw codec must CARRY the
    timing, so the only contract left is: shapes are right, values are finite, and the shift
    genuinely moved the waveform.
    """
    cfg = _prod_geom_cfg()
    W_full = cfg.window_samples * 4
    shot = _elm_raw(cfg.channels, W_full, [1, 4, 7], pool=100, seed=3)
    t0 = 0.05  # start one 50 ms window in
    win_a, win_b = idata.fastts_shift_pair_windows(shot, t0=t0, cfg=cfg, delta_ms=delta_ms)
    assert win_a.shape == (cfg.channels, cfg.window)
    assert win_b.shape == (cfg.channels, cfg.window)
    assert torch.isfinite(win_a).all() and torch.isfinite(win_b).all()
    assert not torch.allclose(win_a, win_b), "a δ-shift must move the raw waveform"
    # and the shifted window really is the same signal, δ samples later.
    shift = round(delta_ms * FASTTS_FS / 1000.0)
    assert torch.allclose(win_b[:, : cfg.window - shift], win_a[:, shift:], atol=1e-5)


def test_fastts_shift_pair_overrun_raises():
    cfg = _prod_geom_cfg()
    shot = torch.randn(cfg.channels, cfg.window_samples + 10)  # barely one window
    with pytest.raises(ValueError):
        idata.fastts_shift_pair_windows(shot, t0=0.049, cfg=cfg, delta_ms=5.0)


# --------------------------------------------------------------------------------------- #
# nets round-trip
# --------------------------------------------------------------------------------------- #
def test_encoder_decoder_shapes():
    cfg = _small_cfg()
    enc, dec = FastTSEncoder(cfg), FastTSDecoder(cfg)
    B = 3
    x = torch.randn(B, cfg.channels, cfg.window)
    feats = enc(x)
    assert feats.shape == (B, cfg.n_tok, cfg.d_model)
    recon = dec(feats)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_nets_gradient_flows_end_to_end():
    cfg = _small_cfg()
    enc, dec = FastTSEncoder(cfg), FastTSDecoder(cfg)
    x = torch.randn(2, cfg.channels, cfg.window, requires_grad=True)
    dec(enc(x)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_decoder_last_layer_exposed_for_adaptive_weight():
    """With the conv head the last layer is its final Conv1d; without a stem it is the linear."""
    cfg = _small_cfg()
    dec = FastTSDecoder(cfg)
    assert isinstance(dec.last_layer, torch.nn.Parameter)
    last_conv = [m for m in dec.head.modules() if isinstance(m, torch.nn.Conv1d)][-1]
    assert dec.last_layer is last_conv.weight
    dec0 = FastTSDecoder(_nostem_cfg())
    assert dec0.last_layer is dec0.to_features.weight


def test_conv_stem_gives_patches_a_cross_boundary_receptive_field():
    """The recorded fast-TS recommendation: a PRE-PATCH conv stem / overlapping patches.

    Measured at the PATCHIFY input (``enc.stem``), before the transformer — the transformer
    mixes every token with every other by construction, so it cannot show this. With a
    stride-1 odd-kernel stem, perturbing ONE raw sample must move the stem features that fall
    inside a NEIGHBOURING patch (the patches OVERLAP in receptive field); with
    ``stem_layers=0`` the stem is the identity and the patch boundary is a hard seam.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 8, 200)
    for cfg, expect_leak in ((_small_cfg(), True), (_nostem_cfg(), False)):
        enc = FastTSEncoder(cfg).eval()
        xp = x.clone()
        xp[0, :, cfg.patch_w - 1] += 10.0             # last sample of patch 0
        with torch.no_grad():
            h0, h1 = enc.stem(x), enc.stem(xp)
        # how far the perturbation reached INTO patch 1 (samples patch_w .. 2*patch_w).
        leak = float((h1 - h0)[..., cfg.patch_w:2 * cfg.patch_w].abs().max())
        if expect_leak:
            assert leak > 1e-4, "conv stem must leak across the patch seam"
            reach = int((h1 - h0).abs().max(dim=1).values[0].nonzero().max()) - (cfg.patch_w - 1)
            assert reach >= cfg.stem_kernel // 2, (reach, cfg.stem_kernel)
        else:
            assert leak == 0.0, "no stem => the patch boundary is a hard seam"


# --------------------------------------------------------------------------------------- #
# FastTSCodec forward + codes at the DEFAULT FSQ size
# --------------------------------------------------------------------------------------- #
def test_codec_forward_shapes_and_codes_in_range():
    cfg = fastts_raw_config()  # RAW production config (1000-code FSQ, 500 samples, 25 tok)
    codec = FastTSCodec(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.window)
    out = codec.forward(x)
    assert out["recon"].shape == x.shape
    assert out["feats"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["quant"].shape == (B, cfg.n_tok, cfg.d_model)
    assert out["codes"].shape == (B, cfg.n_tok, cfg.fsq_dim)
    assert out["codes"].dtype == torch.long
    # each per-dim code within its FSQ level range.
    for i, lvl in enumerate(cfg.fsq_levels):
        assert int(out["codes"][..., i].min()) >= 0
        assert int(out["codes"][..., i].max()) < lvl
    assert codec.codebook_size == cfg.codebook_size == 1000


def test_codec_generator_losses_finite_and_consistency_off_by_default():
    cfg = _small_cfg()
    codec = FastTSCodec(cfg)
    disc = Env1DPatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.window)
    x_shift = x.roll(3, dims=-1)
    g = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    for k in ("total", "adversarial", "pixel", "feature_matching", "consistency", "entropy"):
        assert torch.isfinite(g[k]).all(), f"{k} must be finite"
    # consistency defaults OFF (weight 0) -> the term is an exact zero and x_shift is UNUSED:
    # a sample-wise codec must carry spike timing, not be invariant to it.
    assert cfg.consistency_weight == 0.0
    assert float(g["consistency"]) == 0.0
    g_other = codec.generator_losses(x, torch.randn_like(x) * 100.0, disc, cfg, step=0)
    assert float(g_other["total"]) == pytest.approx(float(g["total"]), abs=1e-6)
    assert g["recon"].shape == x.shape
    assert g["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)
    codec.generator_losses(x, x_shift, disc, cfg, step=0)["total"].backward()


def test_codec_generator_losses_consistency_on_when_weighted():
    cfg = _small_cfg()
    cfg.consistency_weight = 1.0
    codec = FastTSCodec(cfg)
    disc = Env1DPatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.window)
    g_same = codec.generator_losses(x, x.clone(), disc, cfg, step=0)
    assert float(g_same["consistency"]) < 1e-6
    g_diff = codec.generator_losses(x, x.roll(7, dims=-1), disc, cfg, step=0)
    assert float(g_diff["consistency"]) > 1e-6


def test_recon_loss_is_reconstruction_dominant_and_selectable():
    """pixel_anchor 1.0 vs adversarial 0.0: the objective is reconstruction-led by default."""
    cfg = _small_cfg()
    codec = FastTSCodec(cfg)
    disc = Env1DPatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.window)
    assert cfg.pixel_anchor_weight == 1.0 and cfg.adversarial_weight == 0.0
    g = codec.generator_losses(x, x, disc, cfg, step=0)
    # nrmse (the default): the pixel term IS the reported metric, per (window, channel).
    r = g["recon"].detach()
    d2 = ((r - x) ** 2).mean(dim=-1)
    v = x.var(dim=-1, unbiased=False)
    expected = float((d2 / v).clamp_min(0.0).sqrt().mean())
    assert float(g["pixel"]) == pytest.approx(expected, rel=1e-4)
    for kind in ("nmse", "mse", "l1", "huber"):
        cfg.recon_loss = kind
        assert torch.isfinite(codec.generator_losses(x, x, disc, cfg, step=0)["pixel"]), kind
    cfg.recon_loss = "not-a-loss"
    with pytest.raises(ValueError):
        codec.generator_losses(x, x, disc, cfg, step=0)


def test_plain_mse_cannot_see_the_flat_line_failure_that_nrmse_reports():
    """The reason the default is 'nrmse': on the REAL geometry, plain MSE is blind.

    A flat-line prediction at each (window, channel)'s own mean is what a plain-MSE optimum
    converges to when the window-mean offset dwarfs the within-window fluctuation (measured
    ~25x on held-out shots). It scores the trivial-baseline nRMSE of EXACTLY 1.0000, but
    plain MSE reads ~5e-05 on it -- i.e. 'near perfect'. This is the loss-side twin of the
    blind reconstruction metric that hid the same failure on the spectro side.
    """
    torch.manual_seed(0)
    within, offset = 0.0073, 0.0073 * 25.0     # the measured ratio
    x = torch.randn(6, 8, 500) * within + torch.randn(6, 8, 1) * offset
    flat = x.mean(dim=-1, keepdim=True).expand_as(x).contiguous()

    cfg_n = _small_cfg(); cfg_n.window = 500; cfg_n.recon_loss = "nrmse"
    cfg_m = _small_cfg(); cfg_m.window = 500; cfg_m.recon_loss = "mse"
    nr = float(FastTSCodec._recon_loss(flat, x, cfg_n))
    ms = float(FastTSCodec._recon_loss(flat, x, cfg_m))
    assert nr == pytest.approx(1.0, abs=1e-6), nr
    assert ms < 1e-3, ms                       # MSE calls the trivial baseline 'near perfect'
    # and the loss agrees with what the gate reports, to the metric's own precision.
    assert nr == pytest.approx(
        gate.full_fastts_metrics(flat.numpy(), x.numpy())["fastts_nrmse"], abs=1e-5)


def test_nrmse_loss_masks_dead_channels_like_the_metric():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 500) * 0.01
    x[:, 2, :] = 3.0                           # dead / absent channel: constant target
    cfg = _small_cfg(); cfg.window = 500; cfg.recon_loss = "nrmse"
    recon = torch.zeros_like(x)                # maximally wrong on the dead channel too
    loss = FastTSCodec._recon_loss(recon, x, cfg)
    assert torch.isfinite(loss), "a zero-variance channel must not divide by zero"
    # excluding it is what the metric does: valid_frac drops to 7/8.
    assert gate.full_fastts_metrics(recon.numpy(), x.numpy())["fastts_valid_frac"] == \
        pytest.approx(7 / 8)


# --------------------------------------------------------------------------------------- #
# discriminator
# --------------------------------------------------------------------------------------- #
def test_discriminator_multiscale_score_maps_and_features():
    cfg = _small_cfg()
    disc = Env1DPatchGAN(cfg, scales=2)
    x = torch.randn(2, cfg.channels, cfg.window)
    maps = disc(x)
    assert isinstance(maps, list) and len(maps) == 2
    for m in maps:
        assert m.dim() == 3 and m.shape[0] == 2 and m.shape[1] == 1  # (B, 1, e)
    scores, feats = disc(x, return_features=True)
    assert len(scores) == 2 and len(feats) > 0
    for f in feats:
        assert torch.isfinite(f).all()


def test_discriminator_rejects_wrong_ndim():
    cfg = _small_cfg()
    disc = Env1DPatchGAN(cfg)
    with pytest.raises(ValueError):
        disc(torch.randn(2, cfg.channels, cfg.window, 3))  # 4-D, not (B,C,W)


# --------------------------------------------------------------------------------------- #
# gate.fastts_decode_fidelity
# --------------------------------------------------------------------------------------- #
def test_fastts_decode_fidelity_perfect_recon():
    # recon == target -> perfect envelope_corr + peak_f1, sharpness ~1.
    env = torch.rand(2, 4, 20)
    dm = gate.fastts_decode_fidelity(env, env)
    assert dm["envelope_corr"] > 0.999
    assert dm["peak_f1"] > 0.999
    assert abs(dm["sharpness"] - 1.0) < 1e-6


def test_fastts_decode_fidelity_mean_collapse_low_sharpness():
    target = torch.rand(2, 4, 20)
    recon = target.mean(dim=-1, keepdim=True).expand_as(target).contiguous()  # flat envelope
    dm = gate.fastts_decode_fidelity(recon, target)
    assert dm["sharpness"] < 1.0, "a mean-collapsed envelope must have less HF gradient energy"


def test_fastts_decode_fidelity_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        gate.fastts_decode_fidelity(torch.randn(2, 4, 8, 10), torch.randn(2, 4, 8, 10))


# --------------------------------------------------------------------------------------- #
# synthetic filterscopes HDF5 shots + FastTSCodecPairDataset
# --------------------------------------------------------------------------------------- #
def _write_filterscopes_shot(path: Path, duration_s: float, seed: int) -> None:
    """Write a tiny filterscopes/{xdata,ydata} HDF5 shot in the loader's format.

    ydata is (104, T) raw filterscope samples with sharp ELM-like bursts so windows are
    non-degenerate; only the first 8 channels are used by the SignalConfig. xdata is seconds.
    """
    rng = np.random.default_rng(seed)
    C = 104
    native_fs = 20_000.0  # native rate; loader resamples to target_fs=10 kHz
    T = int(round(duration_s * native_fs))
    t = np.linspace(0.0, duration_s, T).astype(np.float32)
    y = (0.02 * rng.standard_normal((C, T))).astype(np.float32)
    # sprinkle sharp bursts (ELM-like) across time so the envelope has structure.
    n_bursts = max(4, T // 400)
    for _ in range(n_bursts):
        pos = int(rng.integers(0, T))
        y[:, pos] += (5.0 * rng.standard_normal((C,))).astype(np.float32)
    with h5py.File(path, "w") as f:
        g = f.create_group("filterscopes")
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)


@pytest.fixture
def filterscopes_shots(tmp_path):
    shots = []
    duration_s = 1.0 + 8 * 0.05 + 0.1  # room for several 50 ms windows past t0_start=1.0
    for i in range(4):
        sid = f"90000{i}"
        _write_filterscopes_shot(tmp_path / f"{sid}_processed.h5", duration_s, seed=i)
        shots.append(sid)
    return {"dir": tmp_path, "shots": shots}


def _tiny_fastts_cfg() -> FastTSCodecConfig:
    # production RAW geometry (500-sample / 50 ms window, patch_w 100 -> n_tok 5 for speed),
    # tiny transformer + a 1-layer conv stem.
    return fastts_raw_config(
        channels=ft.fastts_channels(),
        window=FASTTS_WINDOW, patch_w=100,
        stem_channels=8, stem_kernel=5, stem_layers=1,
        d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
    )


def test_fastts_channels():
    assert ft.fastts_channels() == 8  # channels_to_use=slice(0,8) on filterscopes
    assert ft.fastts_channels("filterscopes") == 8
    with pytest.raises(ValueError):
        ft.fastts_channels("ece")


def test_load_fastts_channel_stats_slices_raw_stats(tmp_path):
    """load_fastts_channel_stats reads the RAW per-channel mean/std, sliced by channels_to_use.

    Mirrors what the data_loader does for method="standardize" (uses the 'raw' sub-dict), and
    returns exactly C=8 values (filterscopes channels_to_use=slice(0,8)) so the codec's per-
    channel standardization length matches its channels.
    """
    C_full = 104  # filterscopes SignalConfig num_channels
    raw_mean = np.arange(C_full, dtype=np.float64) * 1e13
    raw_std = (np.arange(C_full, dtype=np.float64) + 1.0) * 1e15
    stats = {"filterscopes": {
        "raw": {"mean": raw_mean, "std": raw_std},
        "log": {"mean": np.zeros(C_full), "std": np.ones(C_full)},  # must be IGNORED
    }}
    p = tmp_path / "preprocessing_stats.pt"
    torch.save(stats, p)

    mean, std = ft.load_fastts_channel_stats(p, "filterscopes")
    assert len(mean) == 8 and len(std) == 8   # sliced to channels_to_use=slice(0,8)
    # raw (NOT log) stats, first 8 channels.
    assert np.allclose(mean, raw_mean[:8].astype(np.float32))
    assert np.allclose(std, raw_std[:8].astype(np.float32))


def test_load_fastts_channel_stats_nan_handling(tmp_path):
    C_full = 104
    raw_mean = np.full(C_full, np.nan)
    raw_std = np.full(C_full, np.nan)
    stats = {"filterscopes": {"raw": {"mean": raw_mean, "std": raw_std}}}
    p = tmp_path / "preprocessing_stats.pt"
    torch.save(stats, p)
    mean, std = ft.load_fastts_channel_stats(p, "filterscopes")
    # NaN mean -> 0, NaN std -> 1 (matches _update_preprocessing_stats).
    assert all(m == 0.0 for m in mean)
    assert all(s == 1.0 for s in std)


def test_load_fastts_channel_stats_missing_modality_raises(tmp_path):
    p = tmp_path / "preprocessing_stats.pt"
    torch.save({"ece": {"raw": {"mean": [0.0], "std": [1.0]}}}, p)
    with pytest.raises(KeyError):
        ft.load_fastts_channel_stats(p, "filterscopes")


def test_fastts_dataset_reuses_parent_index_machinery(filterscopes_shots):
    cfg = _tiny_fastts_cfg()
    ds = ft.FastTSCodecPairDataset(
        filterscopes_shots["shots"], cfg, data_dir=filterscopes_shots["dir"], seed=0
    )
    # IS a TokamakMultiFileDataset (reuses idx-map + LRU handles + length cache + pickling).
    assert isinstance(ds, TokamakMultiFileDataset)
    assert len(ds) > 1
    # overrides ONLY the transform hook — NOT __getitem__ / the searchsorted index map.
    assert "_getitem_standard" in vars(ft.FastTSCodecPairDataset)
    assert "__getitem__" not in vars(ft.FastTSCodecPairDataset)
    src = inspect.getsource(ft.FastTSCodecPairDataset)
    assert "searchsorted" not in src, "must reuse the parent's binary-search index map"
    assert int(ds._cumulative_lengths[-1]) == len(ds)


def test_fastts_dataset_yields_pair_shapes(filterscopes_shots):
    cfg = _tiny_fastts_cfg()
    ds = ft.FastTSCodecPairDataset(
        filterscopes_shots["shots"], cfg, data_dir=filterscopes_shots["dir"], seed=0
    )
    for i in range(min(6, len(ds))):
        win_a, win_b = ds[i]
        assert win_a.shape == (cfg.channels, cfg.window)
        assert win_b.shape == (cfg.channels, cfg.window)
        assert torch.isfinite(win_a).all() and torch.isfinite(win_b).all()


def test_fastts_loader_num_workers_0_and_2(filterscopes_shots):
    cfg = _tiny_fastts_cfg()
    ds = ft.FastTSCodecPairDataset(
        filterscopes_shots["shots"], cfg, data_dir=filterscopes_shots["dir"], seed=1
    )
    loader0 = ft.make_fastts_loader(ds, batch_size=2, num_workers=0, seed=1)
    a, b = next(iter(loader0))
    assert a.shape == (2, cfg.channels, cfg.window)
    assert b.shape == (2, cfg.channels, cfg.window)

    loader2 = ft.make_fastts_loader(ds, batch_size=2, num_workers=2, seed=2)
    seen = 0
    for a, b in loader2:
        assert a.shape[1:] == (cfg.channels, cfg.window)
        assert torch.isfinite(a).all()
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1


def test_fastts_loader_ddp_sampler_shards(filterscopes_shots):
    cfg = _tiny_fastts_cfg()
    ds = ft.FastTSCodecPairDataset(
        filterscopes_shots["shots"], cfg, data_dir=filterscopes_shots["dir"], seed=0
    )
    l0 = ft.make_fastts_loader(ds, batch_size=1, num_workers=0, rank=0, world_size=2, seed=0)
    l1 = ft.make_fastts_loader(ds, batch_size=1, num_workers=0, rank=1, world_size=2, seed=0)
    assert isinstance(l0.sampler, ft.DistributedTwoLevelSampler)
    assert set(iter(l0.sampler)).isdisjoint(set(iter(l1.sampler)))


# --------------------------------------------------------------------------------------- #
# fastts_compute_gate — finite for a reconstructing input, -inf score on NaN
# --------------------------------------------------------------------------------------- #
def test_fastts_compute_gate_finite_for_reconstructing_input():
    cfg = _small_cfg()
    cfg.gate_recon_floor = -1.0  # never disqualify: prove finiteness of the plumbing
    codec = FastTSCodec(cfg)
    B, n_win = 2, 4
    pairs = [
        (torch.randn(B, cfg.channels, cfg.window),
         torch.randn(B, cfg.channels, cfg.window))
        for _ in range(2)
    ]
    frame_seq = torch.randn(B, n_win, cfg.channels, cfg.window)
    g = ft.fastts_compute_gate(codec, pairs, frame_seq, cfg)
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    assert isinstance(g["pass_stability"], bool) and isinstance(g["pass_persistence"], bool)
    for k in ("envelope_corr", "peak_f1", "sharpness"):
        assert math.isfinite(g["decode"][k])
    # the 2026-09-03 additions: the PRIMARY sample-wise nRMSE + the mandatory baselines.
    dec = g["decode"]
    for k in ("fastts_nrmse", "fastts_corr", "fastts_global_nrmse", "fastts_valid_frac"):
        assert math.isfinite(dec[k]), k
    assert dec["base_self_fastts_nrmse"] == pytest.approx(0.0, abs=1e-10)
    assert dec["base_wcmean_fastts_nrmse"] == pytest.approx(1.0, abs=1e-10)
    assert dec["base_tmean_fastts_nrmse"] == pytest.approx(1.0, abs=1e-10)
    # envelope_corr (what gate_score reads) IS the sample-wise correlation now.
    assert dec["envelope_corr"] == pytest.approx(dec["fastts_corr"], abs=1e-9)
    score = spike.gate_score(g, recon_floor=cfg.gate_recon_floor)
    assert math.isfinite(score)


def test_fastts_gate_score_minus_inf_on_nan_recon():
    g = {
        "forecastability": {"margin_transition": 0.0},
        "decode": {"envelope_corr": float("nan"), "peak_f1": 0.0, "sharpness": 1.0},
        "utilization": {"min_dim_entropy": 0.0, "frac_of_observable": 0.0},
    }
    assert spike.gate_score(g, recon_floor=0.2) == float("-inf")


# --------------------------------------------------------------------------------------- #
# per-step + full train_fastts_codec loop (single-process CPU)
# --------------------------------------------------------------------------------------- #
def test_fastts_train_step_runs_and_returns_terms():
    torch.manual_seed(0)
    cfg = _small_cfg()
    codec = FastTSCodec(cfg)
    disc = Env1DPatchGAN(cfg)
    opt_g = torch.optim.Adam(codec.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    win_a = torch.randn(3, cfg.channels, cfg.window)
    win_b = win_a.roll(2, dims=-1)
    g_terms, d_loss = ft.fastts_codec_train_step(
        codec, disc, opt_g, opt_d, win_a, win_b, cfg, step=0
    )
    assert torch.isfinite(g_terms["total"]).all()
    assert torch.isfinite(d_loss).all()


def test_train_fastts_codec_full_loop_cpu(filterscopes_shots):
    torch.manual_seed(0)
    cfg = _tiny_fastts_cfg()
    train = filterscopes_shots["shots"][:2]
    ev = filterscopes_shots["shots"][2:]
    out = ft.train_fastts_codec(
        cfg, train, ev,
        steps=3, eval_every=2, batch_size=2, num_workers=0,
        data_dir=filterscopes_shots["dir"],
        eval_batches=1, eval_batch_size=2, eval_frames=3,
        seed=0, log_fn=None,
    )
    assert out["steps"] == 3
    assert out["global_step"] == 3
    # gate ran + produced a (finite or -inf) score and the mandate booleans exist.
    assert "stability" in out and "persistence" in out
    assert "best_score" in out


def test_train_fastts_codec_writes_checkpoints(tmp_path, filterscopes_shots):
    torch.manual_seed(0)
    cfg = _tiny_fastts_cfg()
    cfg.gate_recon_floor = -1.0  # ensure a best-ckpt gets written (finite score)
    cfg.gate_hard_min_codes = 0  # 2-step synthetic smoke uses few codes; floor tested in test_spike_best_ema
    out_dir = tmp_path / "fastts_out"
    train = filterscopes_shots["shots"][:2]
    ev = filterscopes_shots["shots"][2:]
    ft.train_fastts_codec(
        cfg, train, ev,
        steps=2, eval_every=1, batch_size=2, num_workers=0,
        data_dir=filterscopes_shots["dir"],
        eval_batches=1, eval_batch_size=2, eval_frames=3,
        out_dir=out_dir, seed=0, log_fn=None,
    )
    assert (out_dir / "codec_last.pt").exists()
    assert (out_dir / "codec_best.pt").exists()
    # gate json written for at least one step.
    assert any(p.name.startswith("gate_") for p in out_dir.iterdir())


def test_fastts_generator_losses_pure_recipe_gan_free():
    """adv 0 + fm 0 (the 2026-09-03 DEFAULT) => the fast-TS generator objective is
    discriminator-free: finite, backprop-able, and invariant to the D parameters."""
    import torch

    from tokamak_foundation_model.ignite.config import FastTSCodecConfig
    from tokamak_foundation_model.ignite.fastts_codec import FastTSCodec
    from tokamak_foundation_model.ignite.fastts_discriminator import Env1DPatchGAN

    torch.manual_seed(0)
    cfg = fastts_raw_config(d_model=32, enc_depth=1, dec_depth=1, heads=2,
                            stem_channels=8, stem_kernel=5, stem_layers=1)
    # these ARE the 2026-09-03 defaults; asserted so a default change trips this test.
    assert cfg.adversarial_weight == 0.0 and cfg.fm_weight == 0.0
    assert cfg.pixel_anchor_weight == 1.0
    codec = FastTSCodec(cfg)
    disc = Env1DPatchGAN(cfg)
    env = torch.randn(2, cfg.channels, cfg.window)
    env_b = env.roll(3, dims=-1)

    out1 = codec.generator_losses(env, env_b, disc, cfg, step=0)
    assert torch.isfinite(out1["total"])
    out1["total"].backward()

    with torch.no_grad():
        for p in disc.parameters():
            p.add_(torch.randn_like(p))
    out2 = codec.generator_losses(env, env_b, disc, cfg, step=0)
    assert torch.allclose(out1["total"], out2["total"], atol=1e-5), (
        float(out1["total"]), float(out2["total"]))


# --------------------------------------------------------------------------------------- #
# BACKWARD COMPATIBILITY — the 2026-09-03 raw-sample codec must be OPT-IN
# --------------------------------------------------------------------------------------- #
# The sample-wise redesign changed the encoder/decoder module names AND tensor shapes. Left
# as the default it broke a STRICT load_state_dict for every fast-TS checkpoint ever written
# — and because train_dynamics._load_codec_set builds the FULL 14-modality set, that killed
# consumers with nothing to do with filterscopes (a figure render for shot 190735 among
# them). These tests pin the contract: DEFAULT == the old envelope codec, byte-identically.
PROD_FASTTS_CKPT = Path(
    "/lustre/orion/fus187/proj-shared/models/ignite_codecs_prod_noinorm/filterscopes/"
    "codec_best.pt"
)


def test_default_config_is_the_old_envelope_codec():
    """A bare FastTSCodecConfig() must reproduce the pre-2026-09-03 codec exactly."""
    cfg = FastTSCodecConfig()
    assert cfg.target == "envelope"
    assert not cfg.is_raw
    # geometry
    assert cfg.env_bins == 5 and cfg.patch_e == 1 and cfg.n_tok == 5
    assert cfg.pool == 100 and cfg.baseline_win == 20 and cfg.env_eps == 1e-3
    # objective defaults — the OLD ones, not the raw codec's
    assert cfg.pixel_anchor_weight == 0.05
    assert cfg.adversarial_weight == 1.0 and cfg.fm_weight == 1.0
    assert cfg.consistency_weight == 1.0
    assert cfg.gate_recon_floor == 0.2


def test_default_module_names_match_the_old_checkpoint_layout():
    """The exact state_dict keys/shapes every existing fast-TS checkpoint stores."""
    sd = FastTSCodec(FastTSCodecConfig()).state_dict()
    for k in ("encoder.pos_emb.env_pe", "decoder.pos_emb.env_pe",
              "encoder.to_tokens.weight", "decoder.to_pixels.weight"):
        assert k in sd, f"missing legacy key {k}"
    for k in ("encoder.stem.0.weight", "decoder.to_features.weight", "decoder.head.0.weight",
              "encoder.pos_emb.pe", "decoder.pos_emb.pe"):
        assert k not in sd, f"raw-only key {k} leaked into the default (envelope) codec"
    assert tuple(sd["encoder.to_tokens.weight"].shape) == (256, 8)
    assert tuple(sd["decoder.to_pixels.weight"].shape) == (8, 256)
    assert tuple(sd["encoder.pos_emb.env_pe"].shape) == (5, 256)


def test_config_pickled_before_the_redesign_takes_the_envelope_path():
    """Old pickles carry no ``target`` field; they must NOT be read as raw codecs."""
    cfg = FastTSCodecConfig()
    del cfg.__dict__["target"]                      # exactly what an old pickle looks like
    assert getattr(cfg, "target", "envelope") == "envelope"
    assert not cfg.is_raw and cfg.n_tok == 5
    out = FastTSCodec(cfg).forward(torch.randn(2, cfg.channels, cfg.env_bins))
    assert out["recon"].shape == (2, cfg.channels, cfg.env_bins)


@pytest.mark.skipif(not PROD_FASTTS_CKPT.exists(), reason="production checkpoint not present")
def test_production_fastts_checkpoint_loads_strictly_and_runs():
    """THE regression guard: load the REAL shipped checkpoint strictly, then forward."""
    ck = torch.load(PROD_FASTTS_CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert not cfg.is_raw, "the shipped checkpoint is an ENVELOPE codec"
    codec = FastTSCodec(cfg)
    codec.load_state_dict(ck["codec"])              # strict=True: the thing that regressed
    codec.eval()
    x = torch.randn(2, cfg.channels, cfg.env_bins)
    with torch.no_grad():
        out = codec.forward(x)
    assert out["recon"].shape == x.shape
    assert torch.isfinite(out["recon"]).all()
    assert out["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)


def test_raw_config_is_opt_in_and_flips_the_objective():
    from tokamak_foundation_model.ignite.config import fastts_raw_config
    cfg = fastts_raw_config(patch_w=20)
    assert cfg.is_raw and cfg.target == "raw" and cfg.n_tok == 25
    assert cfg.pixel_anchor_weight == 1.0
    assert cfg.adversarial_weight == 0.0 and cfg.fm_weight == 0.0
    assert cfg.consistency_weight == 0.0 and cfg.gate_recon_floor == 0.02
    sd = FastTSCodec(cfg).state_dict()
    assert "decoder.to_features.weight" in sd and "decoder.to_pixels.weight" not in sd


# ========================================================================================= #
# GAIN-SHAPE decomposition (2026-09-03) — the fix for "a trivial encoder beats the codec".
#
# Measured on 4 held-out shots / 483 windows: 96.2% of the ELM-envelope variance is the
# per-(window, channel) DC LEVEL, and an encoder that transmits ONLY that level at the codec's
# own 49.83-bit budget scores pooled nRMSE 0.1935 against the shipped codec's 0.5252. With
# gain_shape ON the level is coded by dedicated tokens through a dedicated head and the shape
# branch is mean-removed, which turns that number from a bar the codec fails into a FLOOR it
# cannot fall below. These tests pin the two STRUCTURAL guarantees that make that true, plus
# the bit-identical no-op.
# ========================================================================================= #


def _gs_cfg(**kw):
    from tokamak_foundation_model.ignite.config import FastTSCodecConfig
    base = dict(channels=8, gain_shape=True, gain_tokens=4)
    base.update(kw)
    return FastTSCodecConfig(**base)


def test_gain_shape_off_is_bit_identical_on_the_shipped_checkpoint():
    """THE no-op proof: defaults OFF => same params, same forward, bit for bit.

    Runs the REAL shipped envelope checkpoint, which was pickled BEFORE the gain_shape fields
    existed, so it also proves the new fields default correctly on an old config.
    """
    ck = torch.load(PROD_FASTTS_CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert getattr(cfg, "gain_shape", False) is False, "old pickles must default to OFF"
    assert cfg.n_gain_tok == 0 and cfg.n_shape_tok == cfg.n_tok
    codec = FastTSCodec(cfg)
    codec.load_state_dict(ck["codec"])             # strict: no new params may exist
    codec.eval()
    g = torch.Generator().manual_seed(1234)
    x = torch.rand((6, cfg.channels, cfg.env_bins), generator=g) * 12.0
    with torch.no_grad():
        out = codec.forward(x)
    # the pre-change reference values, recorded from this exact input on the shipped weights
    assert float(out["recon"].sum()) == pytest.approx(1037.7734375, abs=0.0)
    assert int(out["codes"].sum()) == 187
    assert "gain_pred" not in out, "the no-op path must not add output keys"
    assert sum(p.numel() for p in codec.parameters()) == 12614156


@pytest.mark.parametrize("gain_tokens", [1, 2, 3, 4, 5])
def test_gain_shape_preserves_the_world_model_token_contract(gain_tokens):
    """5 tokens of a 1000-code codebook, whatever the gain/shape split.

    dynamics_config.ModalitySpec("filterscopes", "fastts", 5, 1000) must keep holding — the
    decomposition re-partitions the SAME tokens, it does not add any.
    """
    cfg = _gs_cfg(gain_tokens=gain_tokens)
    assert cfg.n_tok == 5 and cfg.codebook_size == 1000
    assert cfg.n_gain_tok == gain_tokens and cfg.n_shape_tok == 5 - gain_tokens
    codec = FastTSCodec(cfg)
    codec.eval()
    x = torch.rand(4, cfg.channels, cfg.env_bins) * 12.0
    with torch.no_grad():
        out = codec.forward(x)
    assert out["codes"].shape == (4, 5, cfg.fsq_dim)
    assert out["recon"].shape == x.shape
    assert torch.isfinite(out["recon"]).all()


@pytest.mark.parametrize("gain_tokens,gain_scale",
                         [(1, True), (3, True), (4, True), (4, False), (5, True)])
def test_shape_branch_carries_no_dc_level(gain_tokens, gain_scale):
    """GUARANTEE 1 — the reconstruction's per-(window, channel) mean IS the gain head's level.

    This is what makes the per-window-mean anchor a FLOOR: zero the shape path and the codec
    degenerates exactly to the quantised-window-mean encoder.
    """
    cfg = _gs_cfg(gain_tokens=gain_tokens, gain_scale=gain_scale)
    codec = FastTSCodec(cfg)
    codec.eval()
    x = torch.rand(5, cfg.channels, cfg.env_bins) * 12.0
    with torch.no_grad():
        out = codec.forward(x)
    level_hat = out["gain_pred"][:, : cfg.channels]
    assert torch.allclose(out["recon"].mean(dim=-1), level_hat, atol=1e-5)


def test_gain_scale_sets_the_within_window_amplitude_exactly():
    """GUARANTEE 2 — with gain_scale the recon's std IS the transmitted sigma_hat.

    The structural attack on the measured 36% burst-height retention: the amplitude is a
    decoded NUMBER, not an nRMSE-minimising conditional mean that is free to shrink.
    """
    cfg = _gs_cfg(gain_tokens=3, gain_scale=True)
    assert cfg.uses_gain_scale and cfg.gain_values == 2 * cfg.channels
    codec = FastTSCodec(cfg)
    codec.eval()
    x = torch.rand(5, cfg.channels, cfg.env_bins) * 12.0
    with torch.no_grad():
        out = codec.forward(x)
    sigma_hat = torch.expm1(out["gain_pred"][:, cfg.channels:].clamp(0.0, 30.0))
    got = out["recon"].std(dim=-1, unbiased=False)
    assert torch.allclose(got, sigma_hat, rtol=1e-4, atol=1e-6)


def test_gain_scale_is_forced_off_when_there_is_no_shape_path():
    cfg = _gs_cfg(gain_tokens=5, gain_scale=True)
    assert cfg.n_shape_tok == 0
    assert cfg.uses_gain_scale is False and cfg.gain_values == cfg.channels


def test_gain_shape_split_is_the_exact_inverse_of_the_recombination():
    from tokamak_foundation_model.ignite.fastts_nets import gain_shape_split
    x = torch.rand(7, 8, 5) * 12.0
    gain, shape, level, sigma = gain_shape_split(x, use_scale=True)
    assert torch.allclose(shape.mean(-1), torch.zeros_like(level), atol=1e-5)
    assert torch.allclose(shape.std(-1, unbiased=False), torch.ones_like(sigma), atol=1e-4)
    assert torch.allclose(level.unsqueeze(-1) + sigma.unsqueeze(-1) * shape, x, atol=1e-4)
    assert torch.allclose(gain[:, :8], level) and torch.allclose(gain[:, 8:],
                                                                 torch.log1p(sigma))


def test_gain_shape_generator_losses_are_finite_and_train_every_parameter():
    cfg = _gs_cfg(gain_tokens=3)
    codec, disc = FastTSCodec(cfg), Env1DPatchGAN(cfg)
    x = torch.rand(4, cfg.channels, cfg.env_bins) * 12.0
    terms = codec.generator_losses(x, x.clone(), disc, cfg, step=0)
    assert "gain" in terms and torch.isfinite(terms["gain"])
    assert torch.isfinite(terms["total"])
    terms["total"].backward()
    assert all(p.grad is not None for p in codec.parameters()), "a dead branch would not train"


def test_gain_weight_zero_removes_the_auxiliary_term():
    cfg = _gs_cfg(gain_tokens=3, gain_weight=0.0)
    codec, disc = FastTSCodec(cfg), Env1DPatchGAN(cfg)
    x = torch.rand(3, cfg.channels, cfg.env_bins) * 12.0
    terms = codec.generator_losses(x, x.clone(), disc, cfg, step=0)
    assert float(terms["gain"]) == 0.0


def test_gain_shape_rejected_in_raw_mode():
    from tokamak_foundation_model.ignite.config import fastts_raw_config
    with pytest.raises(ValueError, match="ENVELOPE-mode"):
        fastts_raw_config(channels=8, gain_shape=True)


def test_gain_tokens_must_fit_the_token_budget():
    with pytest.raises(AssertionError, match="gain_tokens"):
        _gs_cfg(gain_tokens=6)


def test_fastts_envelope_metrics_anchors_and_amplitude_reads():
    """VALIDATE the new metric against references of KNOWN ordering before ranking with it."""
    from tokamak_foundation_model.ignite import gate as _g
    rng = np.random.default_rng(0)
    # A level-dominated envelope WITH REAL BURSTS, like the real one: ~96% of the variance is
    # the between-window level, and a few percent of windows carry a large single-bin ELM
    # burst. Both features are needed for the ladder to be meaningful -- burst_keep is
    # measured on the top-1% most active windows, so a synthetic with no bursts cannot rank
    # anything (it reads ~1.0 for every prediction, which is how this ladder was calibrated).
    lev = rng.normal(6.0, 2.0, size=(300, 8, 1))
    t = lev + rng.normal(0.0, 0.3, size=(300, 8, 5))
    burst_win = rng.random((300, 8)) < 0.05
    burst_bin = rng.integers(0, 5, size=(300, 8))
    t[burst_win, burst_bin[burst_win]] += 8.0
    bits = 5 * math.log2(1000.0)
    m = _g.fastts_envelope_metrics(t, t, bits=bits)
    assert m["env_nrmse"] == pytest.approx(0.0, abs=1e-9)
    assert m["env_std_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["env_level_corr"] == pytest.approx(1.0, abs=1e-6)
    # global constant is EXACTLY the 1.0000 anchor
    const = np.broadcast_to(t.mean(axis=(0, -1))[None, :, None], t.shape)
    mc = _g.fastts_envelope_metrics(const, t, bits=bits)
    assert mc["env_nrmse"] == pytest.approx(1.0, abs=1e-6)
    assert mc["env_base_gmean"] == pytest.approx(1.0, abs=1e-6)
    # the rate-matched BAR must sit just above the unquantised window-mean FLOOR
    assert m["env_base_wmean"] <= m["env_base_ratematched"] + 1e-9
    assert m["env_base_ratematched"] < 0.5 * mc["env_nrmse"]
    # -- the VALIDATION LADDER: references of KNOWN ordering, scored with the same function.
    mu = t.mean(axis=-1, keepdims=True)
    shrunk = mu + 0.4 * (t - mu)
    ms = _g.fastts_envelope_metrics(shrunk, t, bits=bits)
    # THE TRAP this ladder exists to catch: on a signal whose variance is 96% between-window
    # LEVEL, the pooled std_ratio is nearly blind to a within-window amplitude collapse. It
    # reads ~0.99 on a prediction whose bursts are at 40% height. Only the AC ratio sees it.
    assert ms["env_std_ratio"] > 0.9, "pooled std_ratio is level-dominated (documented)"
    assert ms["env_ac_std_ratio"] == pytest.approx(0.4, abs=0.02)
    assert ms["env_burst_keep"] < 0.85
    # flat (global constant): AC amplitude exactly 0, burst entirely lost
    mf = _g.fastts_envelope_metrics(const, t, bits=bits)
    assert mf["env_ac_std_ratio"] == pytest.approx(0.0, abs=1e-6)
    # self: every amplitude read exactly 1.0
    assert m["env_ac_std_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["env_burst_keep"] == pytest.approx(1.0, abs=1e-6)
    # ordering across the ladder is strict on the AC ratio
    assert mf["env_ac_std_ratio"] < ms["env_ac_std_ratio"] < m["env_ac_std_ratio"]
