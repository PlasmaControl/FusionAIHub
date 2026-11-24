"""TokEye spectrogram quality scores (`port.specquality`) on synthetic spectrograms.

The segmenter is replaced by a stand-in that lights a pixel where its input is bright, so these
tests pin the bookkeeping (pickup rows, frames, contrast, validity), not TokEye's weights.
"""

from __future__ import annotations

import numpy as np
import torch

from tokamak_foundation_model.ignite.mast.port import specquality as q

N_BINS, T = 256, 400
DT = 1.024e-3
SIG_T0 = -0.1


class _Bright(torch.nn.Module):
    """Coherent logit high where the z-scored input is above 2, transient never lit."""

    def forward(self, x):
        coh = 20.0 * (x - 2.0)
        tra = torch.full_like(x, -20.0)
        return (torch.cat([coh, tra], dim=1),)


def _times():
    return SIG_T0 + (np.arange(T) + 0.5) * DT


def _spectrogram(rng, line_bin=40, chirp=True, plasma=(0.0, 0.25)):
    """Noise floor 1; a stationary pickup line at ``line_bin`` everywhere; louder broadband and
    a chirp (60 -> 30 kHz bins) inside the plasma window."""
    t = _times()
    p = rng.exponential(1.0, (N_BINS, T))
    p[line_bin - 1:line_bin + 2] *= 400.0
    on = (t >= plasma[0]) & (t < plasma[1])
    p[:, on] *= 10.0
    if chirp:
        cols = np.flatnonzero(on)
        f = np.linspace(60, 30, cols.size).round().astype(int)
        p[f, cols] *= 3000.0
        p[f + 1, cols] *= 3000.0
    return p, on


def test_tokeye_input_shape_and_invalid_columns():
    rng = np.random.default_rng(0)
    p, _ = _spectrogram(rng)
    p[:, :30] = 0.0                                   # missing stretch, stored as zeros
    valid = q.valid_columns(p)
    assert not valid[:30].any() and valid[30:].all()
    x = q.tokeye_input(p, valid)
    assert x.shape == (q.F_UP * (N_BINS - 1), T)
    assert np.all(x[:, :30] == 0.0)
    assert abs(float(x[:, valid].mean())) < 1e-3 and abs(float(x[:, valid].std()) - 1.0) < 1e-2


def test_pickup_line_is_excluded_and_chirp_is_kept():
    rng = np.random.default_rng(1)
    p, on = _spectrogram(rng)
    ps = q.score_probe(_Bright(), p, on)
    # the stationary line (bin 40 -> rows ~78-80 after the x2 interpolation, DC dropped) is pickup
    rows = np.flatnonzero(ps.pickup)
    assert rows.size > 0 and rows.min() <= 2 * 39 <= rows.max()
    # chirp rows (60 -> 30 kHz) are not all pickup: activity remains inside the plasma window
    assert np.nanmean(ps.coh[on]) > 0.003
    assert np.nanmean(ps.coh[~on]) < 1e-3              # outside the plasma only the pickup is lit
    assert 0.0 < ps.pickup_frac < 1.0
    assert ps.contrast_db > 5.0                         # plasma window is 10x louder


def test_pickup_only_probe_has_no_activity():
    rng = np.random.default_rng(2)
    p, on = _spectrogram(rng, chirp=False)
    ps = q.score_probe(_Bright(), p, on)
    # the stand-in lights single speckle pixels of the exponential noise; no track survives
    assert np.nanmean(ps.coh[on]) < 1e-3
    assert ps.pickup_frac > 0.9


def test_score_shot_frames_and_keys():
    rng = np.random.default_rng(3)
    p, _ = _spectrogram(rng)
    power = np.stack([p, p, np.zeros_like(p)])          # third probe dead
    n_frames, frame_t0, frame_dt = 50, 0.0, 0.005
    res = q.score_shot(_Bright(), {q.SIGNALS[1]: (power, SIG_T0, DT)}, frame_t0, n_frames, frame_dt)
    s = q.SHORT[q.SIGNALS[1]]
    assert res[f"{s}_coh"].shape == (3, n_frames)
    assert np.all(res[f"{s}_valid"][:2] == 1.0)
    assert np.all(res[f"{s}_valid"][2] == 0.0)
    assert np.isnan(res[f"{s}_coh"][2]).all()
    assert np.nanmean(res[f"{s}_coh"][0]) > 0.003
    assert res[f"{s}_n_pickup_rows"][0] > 0


def test_column_frames_and_to_frames():
    fidx = q.column_frames(T, SIG_T0, DT, 0.0, 0.005)
    tc = SIG_T0 + (np.arange(T) + 0.5) * DT
    np.testing.assert_array_equal(fidx, np.floor(tc / 0.005).astype(int))
    vals = np.ones(T, dtype=np.float32)
    vals[fidx == 3] = np.nan
    fr = q.to_frames(vals, fidx, 10)
    assert np.isnan(fr[3]) and np.all(fr[[0, 1, 2, 4]] == 1.0)


def test_ac1_coh_separates_smooth_from_speckle():
    rng = np.random.default_rng(4)
    valid = np.ones(T, dtype=bool)
    speckle = rng.normal(size=(20, T))
    smooth = np.cumsum(rng.normal(size=(20, T)), axis=1) * 0.1
    a_s, c_s = q.band_ac1_coh(speckle, valid)
    a_m, c_m = q.band_ac1_coh(smooth, valid)
    assert abs(a_s) < 0.1 and c_s < 0.3
    assert a_m > 0.9 and c_m > 0.9


def test_probe_class_and_signal_verdict():
    assert q.probe_class(0.0, float("nan"), float("nan")) == "absent"
    assert q.probe_class(0.1, 0.7, 0.7) == "absent"
    assert q.probe_class(0.85, -0.02, 0.2) == "noise"           # white in time
    assert q.probe_class(0.85, 0.05, 0.5) == "ok"               # smooth enough despite low ac1
    assert q.probe_class(0.85, 0.65, 0.7) == "ok"
    assert q.signal_verdict(["ok", "ok", "ok"])
    assert q.signal_verdict(["absent", "ok", "absent"])       # constant probes cost nothing
    assert q.signal_verdict(["ok", "noise", "ok"])
    assert not q.signal_verdict(["absent", "noise", "absent"])  # early-MAST CC: noise and zeros
    assert not q.signal_verdict(["noise", "ok", "noise"])
    assert not q.signal_verdict(["absent", "absent", "absent"])


def test_mode_frames_average_ok_probes_only():
    n = 6
    scores = {"n_frames": np.int64(n), "omv_coh": np.array([
        [0.04] * n, [0.0] * n, [np.nan] * 3 + [0.02] * 3], dtype=np.float32)}
    mf = q.mode_frames(scores, "omv", ["ok", "noise", "ok"])
    np.testing.assert_allclose(mf[:3], 0.04, rtol=1e-6)
    np.testing.assert_allclose(mf[3:], 0.03, rtol=1e-6)
    assert np.isnan(q.mode_frames(scores, "omv", ["noise", "noise", "absent"])).all()


def test_per_band_stats_find_a_coherent_low_band_under_white_bins():
    # 255 bins: the lowest band carries a slow random walk, every other bin is white noise.
    # Whole-band ac1 is pulled toward 0; the per-band maximum sees the coherent band.
    rng = np.random.default_rng(5)
    valid = np.ones(T, dtype=bool)
    x = rng.normal(size=(255, T))
    x[:32] = np.cumsum(rng.normal(size=(32, T)), axis=1) * 0.05 + rng.normal(size=(32, T)) * 0.05
    a_all, _ = q.band_ac1_coh(x, valid)
    a_b, c_b = q.band_ac1_coh_per_band(x, valid)
    assert a_b.shape == (q.N_BANDS,) and c_b.shape == (q.N_BANDS,)
    assert a_b[0] > 0.8 and np.all(np.abs(a_b[1:]) < 0.1)
    assert q.probe_class(0.9, float(np.nanmax(a_b)), float(np.nanmax(c_b))) == "ok"
    assert a_all < a_b[0]


def test_mode_agreement_finds_a_lost_chirp():
    rng = np.random.default_rng(3)
    truth, on = _spectrogram(rng)
    truth, same = truth[:, on], truth[:, on].copy()
    flat, _ = _spectrogram(np.random.default_rng(4), chirp=False)
    flat = flat[:, on]
    m = q.mode_agreement(_Bright(), truth, same)
    assert m["f1"] == 1.0 and m["recall"] == 1.0 and m["truth_lit"] > 0
    lost = q.mode_agreement(_Bright(), truth, flat)       # the pickup line is not a mode
    assert lost["recall"] < 0.05 and lost["f1"] < 0.1       # the stand-in lights some speckle
    none = q.mode_agreement(_Bright(), flat, flat)
    assert none["truth_lit"] < 1e-3
    dead = q.mode_agreement(_Bright(), truth, np.zeros_like(truth))
    assert dead["n_cols"] == 0 and np.isnan(dead["f1"])


def test_tokeye_rescue_only_for_intermittent_activity_on_rescue_signals():
    burst = np.r_[np.zeros(80), np.full(20, 0.05)]          # a mode for the last fifth of the shot
    steady = np.full(100, 0.05)                              # lit all along: a line, not a mode
    assert q.tokeye_rescues(burst) and not q.tokeye_rescues(steady)
    assert not q.tokeye_rescues(np.full(100, 0.01))          # too little activity
    assert q.probe_class(1.0, 0.05, 0.22, burst, "omv") == "ok"
    assert q.probe_class(1.0, 0.05, 0.22, burst, "cc") == "noise"
    assert q.probe_class(1.0, 0.05, 0.22) == "noise"
    assert q.probe_class(0.1, 0.05, 0.22, burst, "omv") == "absent"
