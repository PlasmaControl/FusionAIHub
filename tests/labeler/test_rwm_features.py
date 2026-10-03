"""Causal time-slice features of the RWM baseline."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.rwm import features as f


def test_hold_takes_the_last_value_and_drops_a_stale_one():
    t = np.array([0.0, 20.0, 40.0, 200.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    grid = np.array([-5.0, 0.0, 19.0, 20.0, 60.0, 100.0, 200.0])
    out = f.hold(t, y, grid, max_age_ms=50.0)
    assert np.isnan(out[0])  # before the first sample
    assert out[1:4].tolist() == [1.0, 1.0, 2.0]
    assert out[4] == 3.0  # 20 ms old
    assert np.isnan(out[5])  # 60 ms old, over the cap
    assert out[6] == 4.0


def test_trailing_mean_uses_only_past_samples_and_needs_enough_of_them():
    t = np.arange(0.0, 100.0)  # 1 kHz
    y = np.where(t < 50, 1.0, 5.0)
    grid = np.array([49.0, 59.0, 99.0])
    out = f.trailing_mean(t, y, grid, width_ms=5.0)
    assert out[0] == 1.0  # window (44, 49]
    assert out[1] == 5.0  # window (54, 59]
    assert out[2] == 5.0
    # A hole in the record gives NaN instead of the mean of a stray sample.
    keep = (t < 40) | (t > 58)
    held = f.trailing_mean(t[keep], y[keep], np.array([50.0, 59.0, 63.0]), 5.0)
    assert np.isnan(held[0]) and np.isnan(held[1])  # 0 and 1 samples of 5
    assert held[2] == 5.0  # window (58, 63] is whole again


def test_a_future_sample_never_moves_an_earlier_feature():
    t = np.arange(0.0, 200.0)
    y = np.random.default_rng(0).uniform(0.5, 2.0, t.size)
    grid = np.arange(20.0, 100.0, 10.0)
    base = f.trailing_log_slope(t, y, grid, 20.0, 0.05)
    changed = y.copy()
    changed[120:] *= 50.0
    assert np.array_equal(base, f.trailing_log_slope(t, changed, grid, 20.0, 0.05))
    for fn in (f.trailing_mean, f.trailing_max):
        args = (t, y, grid, 5.0)
        assert np.array_equal(fn(*args), fn(t, changed, grid, 5.0))


def test_log_slope_recovers_an_exponential_growth_rate():
    t = np.arange(0.0, 300.0)  # ms, 1 kHz
    rate = 200.0  # per second, a wall time of 5 ms
    y = 0.5 * np.exp(rate * t / 1000.0)
    slope = f.trailing_log_slope(t, y, np.array([100.0, 250.0]), 20.0, 1e-6)
    assert slope == pytest.approx([rate, rate], rel=1e-6)
    flat = f.trailing_log_slope(t, np.ones_like(t), np.array([100.0]), 20.0, 1e-6)
    assert flat[0] == pytest.approx(0.0, abs=1e-9)


def test_log_slope_stays_finite_when_the_rms_is_exactly_zero():
    t = np.arange(0.0, 100.0)
    out = f.trailing_log_slope(t, np.zeros_like(t), np.array([60.0]), 20.0, 0.05)
    assert np.isfinite(out[0])


def test_trailing_max_and_empty_windows():
    t = np.arange(0.0, 50.0)
    y = np.zeros_like(t)
    y[30] = 7.0
    out = f.trailing_max(t, y, np.array([29.0, 30.0, 49.0, 80.0]), 10.0)
    assert out[:2].tolist() == [0.0, 7.0]
    assert out[2] == 0.0
    assert np.isnan(out[3])


def test_profile_at_interpolates_between_radial_points_and_holds_in_time():
    rho = np.linspace(0, 1, 5)  # 0, .25, .5, .75, 1
    profile = np.stack([100 * rho, 100 * rho + 10], axis=1)  # (5, 2 times)
    out = f.profile_at(
        np.array([0.0, 20.0]), profile, rho, 0.375, np.array([5.0, 25.0, 200.0]), 50.0
    )
    assert out[:2] == pytest.approx([37.5, 47.5])
    assert np.isnan(out[2])


def test_flattop_window_is_the_longest_run_above_the_fraction():
    t = np.arange(0.0, 100.0)
    ip = np.where((t >= 10) & (t < 20), 1.0, 0.0) + np.where(
        (t >= 40) & (t < 90), 1.0, 0.0
    )
    assert f.flattop_window(t, ip, 0.5) == (40.0, 89.0)
    assert f.flattop_window(t, np.full_like(t, np.nan), 0.5) is None


def _signals(n=1000):
    t = np.arange(0.0, n)
    efit_t = np.arange(0.0, n, 20.0)
    return {
        "ip": (t, np.where((t > 100) & (t < 900), -1.2e6, 1e4)),
        "betan": (efit_t, np.full_like(efit_t, 2.0)),
        "li": (efit_t, np.full_like(efit_t, 0.8)),
        "q95": (efit_t, np.full_like(efit_t, 5.0)),
        "wmhd": (efit_t, np.full_like(efit_t, 4e5)),
        "n1rms": (t, np.full_like(t, 1.5)),
        "n2rms": (t, np.full_like(t, 0.5)),
    }


def test_slice_table_has_the_features_in_order_and_the_derived_ones():
    table = f.slice_table(_signals())
    assert list(table.columns) == ["t_ms", *f.FEATURES, "time_since_flattop_ms"]
    assert table.t_ms.min() >= 100 and table.t_ms.max() <= 900
    row = table.iloc[20]
    assert row.betan_over_li == pytest.approx(2.5)
    assert row.betan_minus_4li == pytest.approx(2.0 - 3.2)
    assert row.ip_ma == pytest.approx(1.2)  # the sign of Ip is dropped
    assert row.wmhd_mj == pytest.approx(0.4)
    assert row.n1rms_g == pytest.approx(1.5)
    assert row.n1rms_growth_per_s == pytest.approx(0.0, abs=1e-9)
    # No rotation or locked-mode record: the columns exist and are empty.
    assert table[["rot_core_khz", "rot_mid_khz", "lock_v"]].isna().all().all()


def test_slice_table_reads_rotation_and_the_locked_mode_when_present():
    signals = _signals()
    rho = np.linspace(0, 1, 33)
    zip_t = np.arange(0.0, 1000.0, 25.0)
    signals["rot_zipfit"] = (zip_t, 100.0 * np.outer(1 - rho, np.ones_like(zip_t)))
    signals["rho_grid"] = rho
    signals["dusbradial"] = (np.arange(0.0, 1000.0), np.full(1000, -3.0))
    table = f.slice_table(signals)
    assert table.rot_core_khz.iloc[10] == pytest.approx(75.0)
    assert table.rot_mid_khz.iloc[10] == pytest.approx(37.5)
    assert table.lock_v.iloc[10] == pytest.approx(3.0)


def test_slice_table_without_a_current_is_empty():
    t = np.arange(0.0, 100.0)
    table = f.slice_table({"ip": (t, np.full_like(t, np.nan))})
    assert table.empty and list(table.columns) == [
        "t_ms",
        *f.FEATURES,
        "time_since_flattop_ms",
    ]


def test_elapsed_time_uses_a_fixed_current_crossing_without_future_peak():
    t = np.arange(0.0, 100.0)
    ip = np.where(t < 20, 1e4, 0.6e6)
    grid = np.array([10.0, 20.0, 30.0, 40.0])
    base = f.time_since_flattop(t, ip, grid)
    ip[60:] = 3e6
    assert np.isnan(base[0])
    assert base[1:].tolist() == [0.0, 10.0, 20.0]
    assert np.array_equal(base, f.time_since_flattop(t, ip, grid), equal_nan=True)
