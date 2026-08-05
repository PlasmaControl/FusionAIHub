"""TDD spec for the activity-stratified re-draw that fixes the 4 collapsing IGNITE codecs
(filterscopes / ts_core_density / tangtv_lower / co2).

These codecs collapse to 1 code because their training batch is swamped by near-constant
(quiet / missing / floored) windows — the codec encodes the dominant degenerate value and
never learns the minority active signal. The fix biases the per-item draw toward ACTIVE windows
WITHOUT dropping quiet windows (a below-threshold draw is still accepted as the fallback, so the
codec keeps a "quiet" code for Phase-B generalization).

Coverage (all CPU; NO SLURM / GPU / production HDF5):

  * ``train_codec._stratified_draw`` unit tests: OFF (active_bias=0) is byte-identical + does NO
    extra RNG; ON lifts a mostly-degenerate stream to >= the target active fraction; quiet windows
    are down-weighted, NOT dropped (a fully-quiet stream still returns something).
  * The 4 dataset classes expose the knobs and, with active_bias=0, are byte-identical to before
    on real-format synthetic shots.
  * ``apply_activity_overrides`` turns the knobs ON only for the 4 failing modalities and leaves
    every working modality (ece/bes/mhr/cer*/ts_core_temp/ts_tangential*) byte-identical; the
    adv-warmup / adversarial-weight change touches ONLY the two adversarially-unstable codecs.
  * On a synthetic co2 shot engineered to be mostly-flat with a structured minority, turning the
    bias ON raises the fraction of active windows in a batch vs OFF.

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_activity_stratification.py -q
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite import fastts_train as ft
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import (
    CHUNK_S,
    STFT_FS,
    FastTSCodecConfig,
    SlowTSCodecConfig,
    SpectroCodecConfig,
    VideoCodecConfig,
)


# ===================================================================================== #
# _stratified_draw — the shared mechanism, unit-tested with pure callables (no HDF5).
# ===================================================================================== #
def _draw_from(stream):
    """A deterministic draw_valid over a fixed list of (activity) floats -> item is the float."""
    def draw_valid(i):
        return float(stream[int(i) % len(stream)])
    return draw_valid


def test_stratified_off_is_byte_identical_and_no_rng():
    """active_bias=0 returns exactly draw_valid(idx) and performs no extra draws.

    We prove "no extra RNG" by making draw_valid record its call sequence: with the bias OFF it
    must be called EXACTLY once (on idx), never on a re-draw index.
    """
    calls = []

    def draw_valid(i):
        calls.append(int(i))
        return 0.0  # always "quiet" — would trigger a re-draw if the bias were on

    out = tc._stratified_draw(
        7, draw_valid, activity_of=lambda x: x, n_chunks=lambda: 100,
        min_activity=1.0, active_bias=0.0, max_tries=8, seed=0,
    )
    assert out == 0.0
    assert calls == [7], "OFF path must call draw_valid once on idx and never re-draw"


def test_stratified_keeps_already_active_window():
    """An already-active base draw is returned unchanged (no re-draw) even with the bias ON."""
    calls = []

    def draw_valid(i):
        calls.append(int(i))
        return 5.0  # active

    out = tc._stratified_draw(
        3, draw_valid, activity_of=lambda x: x, n_chunks=lambda: 100,
        min_activity=1.0, active_bias=1.0, max_tries=8, seed=0,
    )
    assert out == 5.0
    assert calls == [3], "active base draw must not trigger a re-draw"


def test_stratified_lifts_active_fraction_on_mostly_degenerate_stream():
    """A mostly-degenerate stream (20% active) reaches ~>=50% active after the biased re-draw.

    Deterministic per-idx RNG, so we sweep many idx and measure the empirical active fraction.
    """
    rng = np.random.default_rng(0)
    # 20% of windows are active (activity 5.0), 80% are quiet (0.0).
    stream = [(5.0 if rng.random() < 0.20 else 0.0) for _ in range(400)]
    natural = float(np.mean([1.0 if s >= 1.0 else 0.0 for s in stream]))
    assert natural < 0.30  # genuinely degenerate-dominated

    active = 0
    N = 2000
    for idx in range(N):
        out = tc._stratified_draw(
            idx, _draw_from(stream), activity_of=lambda x: x, n_chunks=lambda: len(stream),
            min_activity=1.0, active_bias=0.9, max_tries=8, seed=0,
        )
        active += int(out >= 1.0)
    frac = active / N
    # target: ~50% active (up from ~20%). With active_bias 0.9 + 8 tries at 20% base rate the
    # per-item active probability is ~0.2 + 0.9*0.8*(1-0.8**8) ~= 0.75; assert a clear lift.
    assert frac >= 0.50, frac
    assert frac > natural + 0.20, (frac, natural)


def test_stratified_never_drops_quiet_windows():
    """A FULLY-quiet stream still returns a (quiet) item — quiet windows are down-weighted, not
    dropped (the codec still needs a 'quiet' code)."""
    stream = [0.0] * 50  # nothing is ever active
    for idx in range(20):
        out = tc._stratified_draw(
            idx, _draw_from(stream), activity_of=lambda x: x, n_chunks=lambda: len(stream),
            min_activity=1.0, active_bias=1.0, max_tries=8, seed=0,
        )
        assert out == 0.0  # falls back to the base draw when no active window exists


def test_stratified_bias_monotone_in_active_fraction():
    """Higher active_bias -> higher realized active fraction (monotone lever)."""
    rng = np.random.default_rng(1)
    stream = [(3.0 if rng.random() < 0.25 else 0.0) for _ in range(300)]

    def realized(bias):
        a = 0
        for idx in range(1500):
            out = tc._stratified_draw(
                idx, _draw_from(stream), activity_of=lambda x: x, n_chunks=lambda: len(stream),
                min_activity=1.0, active_bias=bias, max_tries=8, seed=0,
            )
            a += int(out >= 1.0)
        return a / 1500

    f0, f5, f9 = realized(0.0), realized(0.5), realized(0.9)
    assert f0 < f5 < f9, (f0, f5, f9)
    assert abs(f0 - 0.25) < 0.06  # OFF ~= the natural rate


# ===================================================================================== #
# apply_activity_overrides — ON for the 4 failures, OFF (byte-identical) for the rest.
# ===================================================================================== #
def test_overrides_on_for_the_four_failing_codecs():
    co2 = SpectroCodecConfig(channels=4)
    tc.apply_activity_overrides(co2, "co2")
    assert co2.active_bias > 0 and co2.min_activity > 0
    assert co2.adv_warmup_steps > 0 and co2.adversarial_weight < 1.0  # adversarial co-cause

    # mhr (spectro): adversarial-instability anti-collapse ONLY (like co2's adv knobs), NO
    # activity bias — mhr is rich / naturally O(1), not degeneracy-dominated.
    mhr = SpectroCodecConfig(channels=6)
    tc.apply_activity_overrides(mhr, "mhr")
    assert mhr.active_bias == 0.0 and mhr.min_activity == 0.0
    assert mhr.adv_warmup_steps > 0 and mhr.adversarial_weight < 1.0

    vid = VideoCodecConfig(channels=2)
    tc.apply_activity_overrides(vid, "tangtv_lower")
    # video is NOT degeneracy-dominated -> NO activity bias, only the adversarial fix.
    assert vid.active_bias == 0.0 and vid.min_activity == 0.0
    assert vid.adv_warmup_steps > 0 and vid.adversarial_weight < 1.0

    ts = tc.slowts_codec_cfg("ts_core_density", 44)
    tc.apply_activity_overrides(ts, "ts_core_density")
    assert ts.active_bias > 0 and 0.0 < ts.min_activity <= 1.0  # present-fraction threshold

    # mse: neutral-beam-dependent (~75% of windows <10% present) -> same present-fraction
    # stratification as ts_core_density (added 2026-07-27 after the missingness-collapse verdict).
    mse = tc.slowts_codec_cfg("mse", 69)
    tc.apply_activity_overrides(mse, "mse")
    assert mse.active_bias > 0 and 0.0 < mse.min_activity <= 1.0

    fts = FastTSCodecConfig(channels=8)
    tc.apply_activity_overrides(fts, "filterscopes")
    assert fts.active_bias > 0 and fts.min_activity > 0
    # fast-TS adversarial-driven collapse (2026-07-27): rich data (91% structured) still pinned to
    # 1 code under full adversarial_weight=1.0 + zero warmup -> gets the SAME adv-warmup + halved
    # adversarial_weight its collapse-prone siblings (co2 / tangtv_lower) already have.
    assert fts.adv_warmup_steps > 0 and fts.adversarial_weight < 1.0


@pytest.mark.parametrize(
    "modality,ctor",
    [
        ("ece", lambda: SpectroCodecConfig(channels=40)),
        ("bes", lambda: SpectroCodecConfig(channels=16)),
        # NOTE: mhr moved OUT of this no-op list — it now gets the adversarial-instability
        # anti-collapse knobs (adv warmup + halved adversarial_weight); asserted ON above.
        ("tangtv_upper", lambda: VideoCodecConfig(channels=2, divertor="upper")),
        ("ts_core_temp", lambda: tc.slowts_codec_cfg("ts_core_temp", 44)),
        ("cer_ti", lambda: tc.slowts_codec_cfg("cer_ti", 48)),
        ("cer_rot", lambda: tc.slowts_codec_cfg("cer_rot", 48)),
        # NOTE: mse moved OUT of this no-op list 2026-07-27 — it now gets present-fraction
        # activity-stratification (neutral-beam missingness); asserted ON in the test above.
        ("ts_tangential_density", lambda: tc.slowts_codec_cfg("ts_tangential_density", 10)),
    ],
)
def test_overrides_are_noop_for_working_modalities(modality, ctor):
    """Working modalities keep every anti-collapse field at its config default (byte-identical)."""
    cfg = ctor()
    before = {k: getattr(cfg, k) for k in
              ("min_activity", "active_bias") if hasattr(cfg, k)}
    before_adv = {k: getattr(cfg, k) for k in
                  ("adv_warmup_steps", "adversarial_weight") if hasattr(cfg, k)}
    tc.apply_activity_overrides(cfg, modality)
    after = {k: getattr(cfg, k) for k in before}
    after_adv = {k: getattr(cfg, k) for k in before_adv}
    assert after == before == {"min_activity": 0.0, "active_bias": 0.0}, modality
    if before_adv:  # spectro / video have adversarial knobs; slow-TS doesn't
        assert after_adv == before_adv == {"adv_warmup_steps": 0, "adversarial_weight": 1.0}, modality


def test_overrides_do_not_touch_adv_knobs_absent_on_slowts():
    """Slow-TS has no adversarial knobs; the override must not crash / inject them."""
    ts = tc.slowts_codec_cfg("ts_core_density", 44)
    tc.apply_activity_overrides(ts, "ts_core_density")
    assert not hasattr(ts, "adv_warmup_steps")
    assert not hasattr(ts, "adversarial_weight")


# ===================================================================================== #
# dataset integration: knobs exposed + active_bias=0 byte-identical on synthetic shots.
# ===================================================================================== #
# Activity threshold that cleanly splits the two synthetic window modes below. QUIET (broadband-
# noise) windows have a log-power std of ~0.57; ACTIVE (sharp-tone) windows ~2.3 — a wide gap.
_CO2_ACTIVITY_THR = 1.0


def _write_spectro_shot(path, modality, channels, duration_s, seed):
    """Write a co2-format shot with alternating QUIET vs ACTIVE 50 ms segments.

    The two modes mirror the real co2 collapse (a mostly-floored modality with a structured
    minority) while both surviving the degenerate-window guard (raw std >= min_std), so this test
    exercises the ACTIVITY stratification — not the pre-existing degenerate re-draw:
      * QUIET  = low broadband white noise -> a near-FLAT log-power spectrum -> LOW window std.
      * ACTIVE = strong sharp tones        -> sharp spectral PEAKS            -> HIGH window std.
    Windows start at t0=1.0 s and are 50 ms (one segment), so a sampled window is one mode or the
    other. Both survive `_build_pair`'s raw-std guard (the noise floor is 0.5, well above min_std).
    """
    rng = np.random.default_rng(seed)
    fs = STFT_FS
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    ydata = np.zeros((channels, n), dtype=np.float32)
    seg = int(round(CHUNK_S * fs))  # one 50 ms window in samples
    for c in range(channels):
        sig = np.zeros(n, dtype=np.float64)
        nseg = n // seg + 1
        for s in range(nseg):
            lo, hi = s * seg, min((s + 1) * seg, n)
            if (s % 2) == 0:  # ACTIVE: sharp tones (high log-power std)
                for f in (40e3, 90e3, 150e3):
                    ph = 2 * math.pi * rng.random()
                    sig[lo:hi] += np.sin(2 * math.pi * (f + c * 1e3) * t[lo:hi] + ph)
            else:  # QUIET: broadband noise (flat spectrum, LOW log-power std) — survives min_std.
                sig[lo:hi] += 0.5 * rng.standard_normal(hi - lo)
        ydata[c] = sig.astype(np.float32)
    with h5py.File(path, "w") as h5:
        g = h5.create_group(modality)
        g.create_dataset("xdata", data=t.astype(np.float64))
        g.create_dataset("ydata", data=ydata)


@pytest.fixture
def mixed_co2_shots(tmp_path):
    """co2 shots with alternating flat/toned 50 ms segments (mostly-degenerate + active minority)."""
    modality = "co2"
    channels = tc.modality_channels(modality)
    duration_s = 1.0 + 40 * CHUNK_S + 0.3   # ~40 windows/shot so the split is measurable
    shots = []
    for i in range(3):
        sid = f"92000{i}"
        _write_spectro_shot(tmp_path / f"{sid}_processed.h5", modality, channels,
                            duration_s, seed=i)
        shots.append(sid)
    return {"dir": tmp_path, "modality": modality, "channels": channels, "shots": shots}


def _tiny_spectro_cfg(channels, **kw):
    base = dict(channels=channels, freq_bins=64, time_frames=32, patch_f=32, patch_t=16,
                d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3])
    base.update(kw)
    return SpectroCodecConfig(**base)


def test_dataset_exposes_knobs_from_cfg(mixed_co2_shots):
    cfg = _tiny_spectro_cfg(mixed_co2_shots["channels"], min_activity=0.1, active_bias=0.5)
    ds = tc.CodecPairDataset(cfg=cfg, modality=mixed_co2_shots["modality"],
                             shots=mixed_co2_shots["shots"], data_dir=mixed_co2_shots["dir"], seed=0)
    assert ds.min_activity == pytest.approx(0.1) and ds.active_bias == pytest.approx(0.5)


def test_dataset_active_bias_zero_is_byte_identical(mixed_co2_shots):
    """active_bias=0 (the working-modality default) yields byte-identical items to no-stratification.

    Two datasets, one with the field default (0) and one explicitly 0, must return identical items.
    """
    m, d = mixed_co2_shots["modality"], mixed_co2_shots["dir"]
    ch = mixed_co2_shots["channels"]
    cfg_default = _tiny_spectro_cfg(ch)                       # min_activity/active_bias default 0
    cfg_explicit = _tiny_spectro_cfg(ch, min_activity=_CO2_ACTIVITY_THR, active_bias=0.0)  # bias OFF
    ds0 = tc.CodecPairDataset(cfg=cfg_default, modality=m, shots=mixed_co2_shots["shots"],
                              data_dir=d, seed=0)
    ds1 = tc.CodecPairDataset(cfg=cfg_explicit, modality=m, shots=mixed_co2_shots["shots"],
                              data_dir=d, seed=0)
    assert len(ds0) == len(ds1) and len(ds0) > 3
    for i in range(len(ds0)):
        a0, b0 = ds0[i]
        a1, b1 = ds1[i]
        assert torch.equal(a0, a1) and torch.equal(b0, b1), i


def test_dataset_active_bias_raises_batch_active_fraction(mixed_co2_shots):
    """Turning the bias ON raises the fraction of ACTIVE windows in a full pass vs OFF."""
    m, d = mixed_co2_shots["modality"], mixed_co2_shots["dir"]
    ch = mixed_co2_shots["channels"]

    def active_frac(active_bias):
        cfg = _tiny_spectro_cfg(ch, min_activity=_CO2_ACTIVITY_THR, active_bias=active_bias)
        ds = tc.CodecPairDataset(cfg=cfg, modality=m, shots=mixed_co2_shots["shots"],
                                 data_dir=d, seed=0)
        acts = [float(ds[i][0].std()) for i in range(len(ds))]
        return float(np.mean([a >= _CO2_ACTIVITY_THR for a in acts])), len(ds)

    f_off, n = active_frac(0.0)
    f_on, _ = active_frac(0.9)
    assert n > 3
    # the synthetic shots are ~half quiet (broadband) / half active (tones); OFF sits near the
    # natural ~0.5 rate, and the bias must lift it clearly toward all-active.
    assert f_off < 0.7, f_off               # OFF ~= natural rate (~0.5)
    assert f_on > f_off + 0.15, (f_on, f_off)
    assert f_on >= 0.75, f_on               # target: strongly active-dominated batch


# --- the other three dataset families: active_bias=0 byte-identical (regression guard) --------- #
def _write_tangtv_shot(path, duration_s, seed, raw_hw=(30, 40), off=(1, 3, 5)):
    rng = np.random.default_rng(seed)
    fps = 200.0
    T = int(round(duration_s * fps))
    H, W = raw_hw
    t = np.linspace(0.0, duration_s, T)
    y = rng.random((7, T, H, W)).astype(np.float32)
    for ch in off:
        y[ch] = np.nan
    with h5py.File(path, "w") as f:
        g = f.create_group("tangtv")
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)


def test_video_dataset_active_bias_zero_byte_identical(tmp_path):
    shots = []
    duration_s = 1.0 + 6 * CHUNK_S + 0.1
    for i in range(3):
        sid = f"83000{i}"
        _write_tangtv_shot(tmp_path / f"{sid}_processed.h5", duration_s, seed=i)
        shots.append(sid)
    base = dict(channels=tc.modality_channels("tangtv_lower"), frames=5, height=20, width=30,
                patch_t=5, patch_h=10, patch_w=10, d_model=32, enc_depth=1, dec_depth=1,
                heads=2, fsq_levels=[4, 4, 3])
    cfg0 = VideoCodecConfig(**base)                                   # default 0
    cfg1 = VideoCodecConfig(**base, min_activity=1.0, active_bias=0.0)  # threshold set, bias OFF
    ds0 = tc.VideoCodecPairDataset("tangtv_lower", shots, cfg0, data_dir=tmp_path, seed=0)
    ds1 = tc.VideoCodecPairDataset("tangtv_lower", shots, cfg1, data_dir=tmp_path, seed=0)
    assert len(ds0) == len(ds1) and len(ds0) > 3
    for i in range(len(ds0)):
        f0, m0 = ds0[i]
        f1, m1 = ds1[i]
        assert torch.equal(f0, f1) and torch.equal(m0, m1), i


def _write_slowts_shot(path, signal, channels, duration_s, seed, zero_is_missing):
    rng = np.random.default_rng(seed)
    fps = 100.0
    T = int(round(duration_s * fps))
    t = np.linspace(0.0, duration_s, T)
    pos = np.linspace(1.0, 5.0, channels)[:, None]
    drift = 1.0 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, T))[None, :]
    y = (pos * drift + 0.05 * np.abs(rng.standard_normal((channels, T)))).astype(np.float32)
    miss_val = 0.0 if zero_is_missing else np.nan
    i0 = int(round(1.05 * fps))
    y[0, i0:i0 + 2] = miss_val
    y[channels // 2, i0 + 3:i0 + 5] = miss_val
    with h5py.File(path, "w") as f:
        g = f.create_group(signal)
        g.create_dataset("xdata", data=t)
        g.create_dataset("ydata", data=y)


def test_slowts_dataset_active_bias_zero_byte_identical(tmp_path):
    signal, channels = "ts_core_density", 12
    duration_s = 1.0 + 8 * CHUNK_S + 0.1
    shots = []
    for i in range(4):
        sid = f"73000{i}"
        _write_slowts_shot(tmp_path / f"{sid}_processed.h5", signal, channels,
                          duration_s, seed=i, zero_is_missing=True)
        shots.append(sid)
    base = dict(signal=signal, channels=channels, time_steps=5, n_zones=2, patch_c=6, patch_t=5,
                d_model=32, enc_depth=1, dec_depth=1, heads=2, fsq_levels=[4, 4, 3])
    cfg0 = SlowTSCodecConfig(**base)                                    # default 0
    cfg1 = SlowTSCodecConfig(**base, min_activity=0.5, active_bias=0.0)  # threshold set, bias OFF
    ds0 = tc.SlowTSCodecPairDataset(signal, shots, cfg0, data_dir=tmp_path, seed=0)
    ds1 = tc.SlowTSCodecPairDataset(signal, shots, cfg1, data_dir=tmp_path, seed=0)
    assert len(ds0) == len(ds1) and len(ds0) > 3
    for i in range(len(ds0)):
        s0, m0 = ds0[i]
        s1, m1 = ds1[i]
        assert torch.equal(s0, s1) and torch.equal(m0, m1), i


def _write_fastts_shot(path, channels, duration_s, seed):
    """filterscopes-format shot: O(1)-scaled raw so the envelope is NOT ceiling-saturated (the
    byte-identical test only needs a real-format shot; scale realism is exercised separately)."""
    rng = np.random.default_rng(seed)
    fs = 10_000.0
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    y = np.zeros((channels, n), dtype=np.float32)
    for c in range(channels):
        # bursty ELM-like envelope so windows are non-degenerate + varied.
        base = 0.2 * rng.standard_normal(n)
        burst = (np.sin(2 * math.pi * 600 * t) ** 2) * (rng.random(n) < 0.3)
        y[c] = (base + 3.0 * burst).astype(np.float32)
    with h5py.File(path, "w") as f:
        g = f.create_group("filterscopes")
        g.create_dataset("xdata", data=t.astype(np.float64))
        g.create_dataset("ydata", data=y)


def test_fastts_dataset_active_bias_zero_byte_identical(tmp_path):
    channels = ft.fastts_channels()
    duration_s = 1.0 + 8 * CHUNK_S + 0.05
    shots = []
    for i in range(3):
        sid = f"74000{i}"
        _write_fastts_shot(tmp_path / f"{sid}_processed.h5", channels, duration_s, seed=i)
        shots.append(sid)
    cfg0 = FastTSCodecConfig(channels=channels)                                    # default 0
    cfg1 = FastTSCodecConfig(channels=channels, min_activity=0.5, active_bias=0.0)  # bias OFF
    ds0 = ft.FastTSCodecPairDataset(shots, cfg0, data_dir=tmp_path, seed=0)
    ds1 = ft.FastTSCodecPairDataset(shots, cfg1, data_dir=tmp_path, seed=0)
    assert len(ds0) == len(ds1) and len(ds0) > 3
    for i in range(len(ds0)):
        a0, b0 = ds0[i]
        a1, b1 = ds1[i]
        assert torch.equal(a0, a1) and torch.equal(b0, b1), i
