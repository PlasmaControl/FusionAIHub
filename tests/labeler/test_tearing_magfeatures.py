"""Mirnov-array features of the tearing-mode detector, on synthetic rotating modes."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.tearing import magfeatures as mf

RATE = 100_000.0


def probes(n=None, f_khz=8.0, amp=1.0, seconds=1.0, noise=0.05, seed=0):
    """Six probe traces with a rotating mode of toroidal number `n` and white noise."""
    t = np.arange(int(seconds * RATE)) / RATE
    phi = np.deg2rad(mf.PROBE_PHI)
    rng = np.random.default_rng(seed)
    y = noise * rng.normal(size=(len(phi), t.size))
    if n is not None:
        y += amp * np.cos(2 * np.pi * f_khz * 1000 * t[None, :] - n * phi[:, None])
    return t * 1000.0, y


def features(n=None, **kw):
    t, y = probes(n, **kw)
    window = (100.0, float(t[-1]) - 100.0)
    centres, feats = mf.shot_features(t, y, window)
    return centres, dict(zip(mf.FEATURE_NAMES, feats.T, strict=True))


def test_there_is_one_name_per_feature_column():
    t, y = probes(1)
    _, feats = mf.shot_features(t, y, (100.0, 900.0))
    assert feats.shape[1] == len(mf.FEATURE_NAMES)
    assert len(set(mf.FEATURE_NAMES)) == len(mf.FEATURE_NAMES)


def test_a_rotating_mode_shows_in_its_own_toroidal_harmonic_and_band():
    for n in (1, 2, 3):
        _, f = features(n)
        mid = slice(20, -20)
        own = f[f"a{n}_5_10"][mid].mean()
        others = [f[f"a{m}_5_10"][mid].mean() for m in (1, 2, 3) if m != n]
        assert own > max(others) + 2.0, (n, own, others)
        assert f[f"fit{n}"][mid].mean() > 0.95


def test_the_strongest_line_is_found_at_the_modes_frequency():
    _, f = features(2, f_khz=12.0)
    assert np.median(f["line_khz"][20:-20]) == pytest.approx(12.0, abs=0.2)
    assert f["line_prominence_db"][20:-20].mean() > 20.0


def test_a_mode_is_louder_than_noise_in_its_band_and_not_in_the_others():
    _, quiet = features(None)
    _, mode = features(2, f_khz=8.0)
    mid = slice(20, -20)
    assert mode["logp_4_8"][mid].mean() + mode["logp_8_12"][mid].mean() > (
        quiet["logp_4_8"][mid].mean() + quiet["logp_8_12"][mid].mean() + 3.0
    )
    assert mode["logp_16_22"][mid].mean() == pytest.approx(
        quiet["logp_16_22"][mid].mean(), abs=0.3
    )


def test_the_level_above_the_quiet_floor_is_zero_for_a_featureless_shot():
    _, quiet = features(None)
    assert abs(np.median(quiet["rel_8_12"])) < 0.5


def test_bins_average_their_columns_and_an_empty_bin_is_nan():
    t = np.array([2.0, 4.0, 12.0, 14.0, 16.0])
    v = np.array([[1.0], [3.0], [5.0], [7.0], [9.0]])
    centres = np.array([5.0, 15.0, 25.0])
    out = mf.bin_columns(t, v, centres)
    assert out[0, 0] == pytest.approx(2.0)
    assert out[1, 0] == pytest.approx(7.0)
    assert np.isnan(out[2, 0])


def test_rms_features_take_the_log_of_the_bin_maximum():
    t = np.arange(0.0, 200.0)
    n1 = np.full(t.size, 0.1)
    n1[100:140] = 30.0
    n2 = np.full(t.size, 0.1)
    centres = np.arange(5.0, 200.0, 10.0)
    out = mf.rms_bin_features(t, n1, n2, centres)
    assert out[5, 0] == pytest.approx(np.log10(0.1), abs=1e-6)
    assert out[12, 0] == pytest.approx(np.log10(30.0), abs=1e-6)
    assert out[12, 1] == pytest.approx(np.log10(0.1), abs=1e-6)
