"""CPU TDD spec for the IGNITE Phase-A fast-TS (filterscopes) ELM-envelope codec family.

Mirrors the spectro / video codec tests but for the fast-TS pieces (docs/IGNITE_DESIGN.md
§4.3 — "hardest; statistic = ELM ACTIVITY ENVELOPE, NOT spike timing"):

    data.elm_envelope(raw (B,C,W), cfg) -> envelope (B, C, E)        [the codec target]
    data.fastts_shift_pair_windows(raw_shot, t0, cfg, δ) -> (env_a, env_b)  [δ-shift pair]
    fastts_nets.FastTSEncoder(cfg)(x (B,C,E)) -> feats (B, n_tok, d_model)
    fastts_nets.FastTSDecoder(cfg)(quant (B,n_tok,d_model)) -> recon (B, C, E)
    fastts_codec.FastTSCodec(cfg).forward(x) -> dict(recon, feats, quant, codes)
    fastts_codec.FastTSCodec.generator_losses(x, x_shift, disc, cfg, step) -> dict(total,...)
    fastts_discriminator.Env1DPatchGAN(cfg)(x) -> list of 1-D patch-score maps
    gate.fastts_decode_fidelity(recon, target) -> dict(envelope_corr, peak_f1, sharpness)

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
    FASTTS_ENV_BINS,
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
    """Small transformer + small envelope grid (divisible patching)."""
    return FastTSCodecConfig(
        channels=channels,
        env_bins=20,
        patch_e=5,          # -> 4 env patches => n_tok = 4
        pool=10,
        baseline_win=8,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


def _coarse_cfg(channels: int = 8) -> FastTSCodecConfig:
    """Small transformer at the PRODUCTION COARSE geometry: 10 ms pool bins (pool=100).

    Used for the shift-invariance test — the whole point of the coarsening is that a δ up to
    ~5 ms (half a 10 ms bin) stays within a single bin, so it must be exercised at pool=100.
    """
    return FastTSCodecConfig(
        channels=channels,
        env_bins=8,          # 8 bins * 100 samples = 800 samples = 80 ms
        patch_e=1,           # -> n_tok = 8
        pool=100,            # 10 ms bins (production coarse pool)
        baseline_win=8,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


# --------------------------------------------------------------------------------------- #
# config geometry
# --------------------------------------------------------------------------------------- #
def test_config_geometry_and_tokens():
    cfg = _small_cfg()
    assert cfg.n_env_patch == 4
    assert cfg.n_tok == 4
    assert cfg.fsq_dim == 3 and cfg.codebook_size == 4 * 4 * 3

    # production default: COARSE 5 env bins (10 ms each over a 50 ms window), 5 tokens
    # (patch_e=1), 1000-code FSQ. Coarsened from the old 1 ms / 50-bin grid (detector-free
    # coarse RMS envelope; ELM bursts recur ~1.6 ms so a 10 ms bin averages several bursts).
    prod = FastTSCodecConfig()
    assert prod.env_bins == FASTTS_ENV_BINS == 5
    assert prod.pool == round(0.010 * FASTTS_FS) == 100  # 10 ms bins
    assert prod.channels == 8
    assert prod.patch_e == 1
    assert prod.n_tok == 5
    assert prod.codebook_size == 1000 and prod.fsq_dim == 4
    # window_samples matches the filterscopes 50 ms window at 10 kHz.
    assert prod.window_samples == FASTTS_WINDOW == round(0.05 * FASTTS_FS) == 500
    assert prod.env_bins * prod.pool == prod.window_samples
    # δ cap raised to (0.1, 5.0) ms — a δ up to ~5 ms (half a 10 ms bin) stays within a bin.
    assert prod.consistency_delta_ms == (0.1, 5.0)


def test_config_rejects_indivisible_patch():
    with pytest.raises(AssertionError):
        FastTSCodecConfig(env_bins=5, patch_e=2)  # 5 % 2 != 0


# --------------------------------------------------------------------------------------- #
# ELM-envelope transform — correctness + sanitization
# --------------------------------------------------------------------------------------- #
def _elm_raw(n_ch: int, W: int, burst_bins, pool: int, seed: int = 0) -> torch.Tensor:
    """Synthetic ELM-like raw signal: quiet baseline + high-frequency spike BURSTS in bins.

    ``burst_bins`` is an iterable of (pool-bin index) locations that get an ELM burst — a
    dense, high-amplitude oscillation filling a ~2-bin span around the bin (mimicking a real
    ELM: many fast spikes over a short time, not a single delta). The burst amplitude ENVELOPE
    is therefore stable under a sub-bin (few-sample) time shift — the statistic — while the
    exact sample-level waveform is a realization that a δ-shift moves. Returns (n_ch, W).
    """
    gen = torch.Generator().manual_seed(seed)
    x = 0.02 * torch.randn(n_ch, W, generator=gen)
    t = torch.arange(W, dtype=torch.float32)
    for bin_idx in burst_bins:
        center = (bin_idx + 0.5) * pool
        # Gaussian activity window ~2 bins wide, filled with a fast oscillation (dense spikes).
        win = torch.exp(-0.5 * ((t - center) / pool) ** 2)          # (W,) burst envelope
        carrier = torch.sin(2.0 * math.pi * (t / 2.5) + float(torch.rand((), generator=gen)))
        burst = 5.0 * win * carrier
        x = x + burst.unsqueeze(0)  # same burst timing across channels; noise differs
    return x


def test_elm_envelope_shape_and_burst_localization():
    cfg = _small_cfg()
    W = cfg.env_bins * cfg.pool  # 200
    burst_bins = [3, 12]
    raw = _elm_raw(cfg.channels, W, burst_bins, cfg.pool, seed=1).unsqueeze(0)  # (1,C,W)
    env = idata.elm_envelope(raw, cfg)
    assert env.shape == (1, cfg.channels, cfg.env_bins)
    assert torch.isfinite(env).all()
    assert (env >= 0).all()  # log1p of a non-negative RMS is non-negative
    # the burst bins must carry the largest activity envelope (spikes localized there).
    mean_over_ch = env[0].mean(dim=0)  # (E,)
    topk = set(torch.topk(mean_over_ch, k=len(burst_bins)).indices.tolist())
    assert set(burst_bins).issubset(topk), (
        f"burst bins {burst_bins} should be the peaks; got top {topk}"
    )


def test_elm_envelope_sanitizes_nonfinite_and_absurd():
    cfg = _small_cfg()
    W = cfg.env_bins * cfg.pool
    raw = _elm_raw(cfg.channels, W, [5], cfg.pool, seed=2)
    # inject NaN / inf / float32-max sentinel garbage
    raw[0, 10] = float("nan")
    raw[1, 20] = float("inf")
    raw[2, 30] = -float("inf")
    raw[3, 40] = 3.0e38
    env = idata.elm_envelope(raw.unsqueeze(0), cfg)
    assert torch.isfinite(env).all(), "garbage samples must be sanitized to finite envelope"
    assert (env <= 30.0 + 1e-4).all()  # clamped to the sane ceiling


def test_elm_envelope_crops_pads_to_env_bins():
    cfg = _small_cfg()
    pool = cfg.pool
    # a window LONGER than env_bins*pool -> cropped; SHORTER -> silence-floor padded.
    long_raw = torch.randn(1, cfg.channels, (cfg.env_bins + 5) * pool)
    short_raw = torch.randn(1, cfg.channels, (cfg.env_bins - 4) * pool)
    assert idata.elm_envelope(long_raw, cfg).shape[-1] == cfg.env_bins
    env_short = idata.elm_envelope(short_raw, cfg)
    assert env_short.shape[-1] == cfg.env_bins
    # the padded tail is the silence floor (0.0).
    assert torch.allclose(env_short[..., -4:], torch.zeros_like(env_short[..., -4:]))


def test_elm_envelope_rejects_bad_ndim():
    cfg = _small_cfg()
    with pytest.raises(ValueError):
        idata.elm_envelope(torch.randn(cfg.channels, 200), cfg)  # (C,W) not (B,C,W)


# --------------------------------------------------------------------------------------- #
# ELM-envelope SCALE FIX — per-channel standardization of the UNSTANDARDIZED raw filterscopes.
#
# The raw filterscopes signal has magnitudes ~1e13-1e15 (per-channel std ~1e16). Fed straight
# into detrend->rectify->RMS->log1p(rms/env_eps) it drives log1p to its clamp ceiling for ~all
# windows -> a CONSTANT envelope -> codec collapse. The data_loader consumes STANDARDIZED
# filterscopes ((x-mean)/std, per-channel raw stats). The codec must standardize the SAME way,
# GLOBALLY/per-channel (NOT per-window, which would erase the ELM activity LEVEL).
# --------------------------------------------------------------------------------------- #
def _highmag_cfg(std_scale: float, channels: int = 8) -> FastTSCodecConfig:
    """Small envelope grid + per-channel std ~ std_scale (mimics the ~1e15 raw magnitude)."""
    return FastTSCodecConfig(
        channels=channels,
        env_bins=20, patch_e=5, pool=10, baseline_win=8,
        d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3],
        channel_mean=[0.0] * channels,
        channel_std=[std_scale] * channels,
    )


def test_elm_envelope_highmag_raw_saturates_without_fix():
    """1e14-scale spiky raw with NO standardization pins the envelope at the clamp ceiling."""
    cfg = _small_cfg()  # channel_mean/std None -> no standardization (pre-fix path)
    W = cfg.env_bins * cfg.pool
    raw = 1e14 * _elm_raw(cfg.channels, W, [3, 12], cfg.pool, seed=7).unsqueeze(0)
    env = idata.elm_envelope(raw, cfg)
    ceil = 30.0  # _FASTTS_ENV_CEIL
    at_ceiling = (env > ceil - 1e-3).float().mean()
    assert float(at_ceiling) > 0.9, (
        "unstandardized ~1e14 raw must saturate the log1p envelope ceiling (the bug)"
    )
    # the whole envelope collapses to a near-constant (std ~ 0) -> nothing to encode.
    assert float(env.std()) < 1e-3


def test_elm_envelope_highmag_raw_does_not_saturate_with_fix():
    """SAME 1e14-scale spiky raw, standardized per-channel, no longer saturates."""
    C = 8
    cfg_raw = _small_cfg(C)
    W = cfg_raw.env_bins * cfg_raw.pool
    raw = 1e14 * _elm_raw(C, W, [3, 12], cfg_raw.pool, seed=7).unsqueeze(0)
    # measure per-channel std of THIS signal and standardize by it (as the loader stats would).
    std = raw[0].reshape(C, -1).std(dim=-1)          # (C,)
    cfg_fix = _highmag_cfg(std_scale=1.0, channels=C)
    cfg_fix.channel_std = std.tolist()               # real per-channel std
    cfg_fix.channel_mean = raw[0].reshape(C, -1).mean(dim=-1).tolist()
    env = idata.elm_envelope(raw, cfg_fix)
    ceil = 30.0
    at_ceiling = (env > ceil - 1e-3).float().mean()
    assert float(at_ceiling) < 0.1, f"standardized envelope must not saturate; {float(at_ceiling)}"
    assert float(env.std()) > 0.1, "standardized envelope must carry spread (not a constant)"
    # burst bins still the peaks (the statistic survives standardization).
    mean_over_ch = env[0].mean(dim=0)
    topk = set(torch.topk(mean_over_ch, k=2).indices.tolist())
    assert {3, 12}.issubset(topk)


def test_elm_envelope_standardization_preserves_activity_level():
    """A high-activity window yields a LARGER envelope than a low-activity one (level kept).

    Critical: standardization must be GLOBAL/per-channel, NOT per-window — a per-window norm
    would erase the ELM activity level (quiet ~= active). Both windows share the SAME channel
    std used to standardize, so their relative amplitudes are preserved.
    """
    C = 8
    cfg = _highmag_cfg(std_scale=1e14, channels=C)
    W = cfg.env_bins * cfg.pool
    # quiet window: small bursts; active window: large bursts, at the 1e14 raw scale.
    quiet = 1e14 * (0.05 * _elm_raw(C, W, [5], cfg.pool, seed=1))
    active = 1e14 * (1.0 * _elm_raw(C, W, [5, 8, 11], cfg.pool, seed=1))
    env_q = idata.elm_envelope(quiet.unsqueeze(0), cfg)
    env_a = idata.elm_envelope(active.unsqueeze(0), cfg)
    assert float(env_a.mean()) > float(env_q.mean()), (
        "active window envelope must exceed the quiet one (activity LEVEL preserved)"
    )
    # both finite + on a sane O(1)-ish envelope scale (not pinned at the ceiling).
    assert torch.isfinite(env_q).all() and torch.isfinite(env_a).all()
    assert float(env_a.max()) < 30.0 - 1e-3


def test_elm_envelope_standardization_still_sanitizes_nonfinite():
    """NaN/inf/absurd samples are still mapped to a finite envelope WITH standardization on."""
    C = 8
    cfg = _highmag_cfg(std_scale=1e14, channels=C)
    W = cfg.env_bins * cfg.pool
    raw = 1e14 * _elm_raw(C, W, [5], cfg.pool, seed=2)
    raw[0, 10] = float("nan")
    raw[1, 20] = float("inf")
    raw[2, 30] = -float("inf")
    raw[3, 40] = 3.0e38
    env = idata.elm_envelope(raw.unsqueeze(0), cfg)
    assert torch.isfinite(env).all()
    assert (env <= 30.0 + 1e-4).all()


def test_elm_envelope_standardization_preserves_shape_and_none_is_noop():
    """Standardization keeps (B, C, env_bins); channel_mean/std None == the pre-fix envelope."""
    C = 8
    W = 20 * 10
    raw = _elm_raw(C, W, [3, 12], 10, seed=4).unsqueeze(0)
    cfg_none = _small_cfg(C)                                     # None stats -> no-op
    cfg_id = _highmag_cfg(std_scale=1.0, channels=C)            # mean 0 / std 1 -> identity
    env_none = idata.elm_envelope(raw, cfg_none)
    env_id = idata.elm_envelope(raw, cfg_id)
    assert env_none.shape == (1, C, 20)
    assert env_id.shape == (1, C, 20)
    # standardizing by mean 0 / std 1 is the identity -> byte-identical to the None path.
    assert torch.allclose(env_none, env_id, atol=1e-5)


def test_elm_envelope_rejects_wrong_length_channel_stats():
    cfg = _highmag_cfg(std_scale=1.0, channels=8)
    cfg.channel_std = [1.0] * 4  # wrong length (!= C=8)
    with pytest.raises(ValueError):
        idata.elm_envelope(torch.randn(1, 8, 200), cfg)


# --------------------------------------------------------------------------------------- #
# δ-shift pair — same statistic, different realization
# --------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta_ms", [0.2, 1.0, 3.0, 5.0])
def test_fastts_shift_pair_shapes_and_similarity(delta_ms):
    # COARSE 10 ms bins (pool=100): with the coarsening, a δ up to ~5 ms (half a bin) stays
    # WITHIN a bin, so the per-bin RMS activity statistic is preserved and the two envelope
    # curves stay strongly correlated — that is the restored shift-invariance the raised δ cap
    # relies on. (At the OLD 1 ms bin a 5 ms shift was 5 whole bins and would have collapsed
    # the correlation; here it does not.)
    cfg = _coarse_cfg()
    # a raw shot long enough to slice [t0, t0+CHUNK_S+δ]. Use several windows worth.
    W_full = cfg.window_samples * 4
    burst_bins = [b for b in range(0, cfg.env_bins, 3)]
    shot = _elm_raw(cfg.channels, W_full, burst_bins, cfg.pool, seed=3)
    t0 = 0.08  # start 1 window (env_bins*pool = 800 samples = 80 ms) in
    env_a, env_b = idata.fastts_shift_pair_windows(shot, t0=t0, cfg=cfg, delta_ms=delta_ms)
    assert env_a.shape == (cfg.channels, cfg.env_bins)
    assert env_b.shape == (cfg.channels, cfg.env_bins)
    assert torch.isfinite(env_a).all() and torch.isfinite(env_b).all()
    a = env_a.flatten() - env_a.mean()
    b = env_b.flatten() - env_b.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm() + 1e-9))
    assert corr > 0.5, (
        f"δ={delta_ms} ms (≤ half a 10 ms bin) should keep the envelope statistic "
        f"(corr={corr:.3f})"
    )
    assert not torch.allclose(env_a, env_b), "a δ-shift must move the realization"


def test_fastts_shift_pair_overrun_raises():
    cfg = _coarse_cfg()
    shot = torch.randn(cfg.channels, cfg.window_samples + 10)  # barely one window
    with pytest.raises(ValueError):
        # a full-window overrun still raises regardless of the (now larger) δ cap.
        idata.fastts_shift_pair_windows(shot, t0=0.079, cfg=cfg, delta_ms=5.0)


# --------------------------------------------------------------------------------------- #
# nets round-trip
# --------------------------------------------------------------------------------------- #
def test_encoder_decoder_shapes():
    cfg = _small_cfg()
    enc, dec = FastTSEncoder(cfg), FastTSDecoder(cfg)
    B = 3
    x = torch.randn(B, cfg.channels, cfg.env_bins)
    feats = enc(x)
    assert feats.shape == (B, cfg.n_tok, cfg.d_model)
    recon = dec(feats)
    assert recon.shape == x.shape
    assert torch.isfinite(recon).all()


def test_nets_gradient_flows_end_to_end():
    cfg = _small_cfg()
    enc, dec = FastTSEncoder(cfg), FastTSDecoder(cfg)
    x = torch.randn(2, cfg.channels, cfg.env_bins, requires_grad=True)
    dec(enc(x)).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_decoder_last_layer_exposed_for_adaptive_weight():
    cfg = _small_cfg()
    dec = FastTSDecoder(cfg)
    assert dec.last_layer is dec.to_pixels.weight
    assert isinstance(dec.last_layer, torch.nn.Parameter)


# --------------------------------------------------------------------------------------- #
# FastTSCodec forward + codes at the DEFAULT FSQ size
# --------------------------------------------------------------------------------------- #
def test_codec_forward_shapes_and_codes_in_range():
    cfg = FastTSCodecConfig()  # production default (1000-code FSQ, 50 env bins)
    codec = FastTSCodec(cfg)
    B = 2
    x = torch.randn(B, cfg.channels, cfg.env_bins)
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


def test_codec_generator_losses_finite_and_has_consistency():
    cfg = _small_cfg()
    codec = FastTSCodec(cfg)
    disc = Env1DPatchGAN(cfg)
    x = torch.randn(2, cfg.channels, cfg.env_bins)
    x_shift = x + 0.01 * torch.randn_like(x)
    g = codec.generator_losses(x, x_shift, disc, cfg, step=0)
    for k in ("total", "adversarial", "pixel", "feature_matching", "consistency", "entropy"):
        assert torch.isfinite(g[k]).all(), f"{k} must be finite"
    # fast-TS KEEPS the consistency term (unlike video): identical inputs -> ~0 consistency.
    g_same = codec.generator_losses(x, x.clone(), disc, cfg, step=0)
    assert float(g_same["consistency"]) < 1e-6
    assert g["recon"].shape == x.shape
    assert g["codes"].shape == (2, cfg.n_tok, cfg.fsq_dim)
    # backward runs.
    codec.generator_losses(x, x_shift, disc, cfg, step=0)["total"].backward()


# --------------------------------------------------------------------------------------- #
# discriminator
# --------------------------------------------------------------------------------------- #
def test_discriminator_multiscale_score_maps_and_features():
    cfg = _small_cfg()
    disc = Env1DPatchGAN(cfg, scales=2)
    x = torch.randn(2, cfg.channels, cfg.env_bins)
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
        disc(torch.randn(2, cfg.channels, cfg.env_bins, 3))  # 4-D, not (B,C,E)


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
    # production COARSE geometry (5 bins * 100-sample / 10 ms pool = 500-sample / 50 ms window),
    # tiny transformer. patch_e=1 -> n_tok=5.
    return FastTSCodecConfig(
        channels=ft.fastts_channels(),
        env_bins=5, patch_e=1, pool=100, baseline_win=8,
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
        env_a, env_b = ds[i]
        assert env_a.shape == (cfg.channels, cfg.env_bins)
        assert env_b.shape == (cfg.channels, cfg.env_bins)
        assert torch.isfinite(env_a).all() and torch.isfinite(env_b).all()


def test_fastts_loader_num_workers_0_and_2(filterscopes_shots):
    cfg = _tiny_fastts_cfg()
    ds = ft.FastTSCodecPairDataset(
        filterscopes_shots["shots"], cfg, data_dir=filterscopes_shots["dir"], seed=1
    )
    loader0 = ft.make_fastts_loader(ds, batch_size=2, num_workers=0, seed=1)
    a, b = next(iter(loader0))
    assert a.shape == (2, cfg.channels, cfg.env_bins)
    assert b.shape == (2, cfg.channels, cfg.env_bins)

    loader2 = ft.make_fastts_loader(ds, batch_size=2, num_workers=2, seed=2)
    seen = 0
    for a, b in loader2:
        assert a.shape[1:] == (cfg.channels, cfg.env_bins)
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
        (torch.randn(B, cfg.channels, cfg.env_bins),
         torch.randn(B, cfg.channels, cfg.env_bins))
        for _ in range(2)
    ]
    frame_seq = torch.randn(B, n_win, cfg.channels, cfg.env_bins)
    g = ft.fastts_compute_gate(codec, pairs, frame_seq, cfg)
    assert 0.0 <= g["stability"] <= 1.0
    assert 0.0 <= g["persistence"] <= 1.0
    assert isinstance(g["pass_stability"], bool) and isinstance(g["pass_persistence"], bool)
    for k in ("envelope_corr", "peak_f1", "sharpness"):
        assert math.isfinite(g["decode"][k])
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
    env_a = torch.randn(3, cfg.channels, cfg.env_bins)
    env_b = env_a + 0.01 * torch.randn_like(env_a)
    g_terms, d_loss = ft.fastts_codec_train_step(
        codec, disc, opt_g, opt_d, env_a, env_b, cfg, step=0
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
