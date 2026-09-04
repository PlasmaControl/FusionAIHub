"""The time-base conventions labelmaker shares with IGNITE."""
import numpy as np
import pytest

from labelmaker.timebase import (
    decimate_to_step,
    index_at,
    sample_at,
    sample_rate,
    window_mean,
)


def test_sample_rate_is_exact_on_float32_records():
    # A real actuator group: 10 kHz, starting 10 s before the shot, stored
    # float32. Quantization puts up to 0.82% of a step of error on any single
    # diff - the figure train_dynamics.py:190 quotes, reproduced here as
    # 8.18e-7 s. See train_dynamics.py:189-191.
    x64 = np.arange(-10.0, 1.0, 1e-4)
    x32 = x64.astype(np.float32)
    fs = sample_rate(x32)
    assert abs(fs - 10_000.0) / 10_000.0 < 1e-6          # measured 1.5e-9
    worst_step = np.abs(np.diff(x32.astype(np.float64)) - 1e-4).max()
    assert worst_step > 0.005 * 1e-4                     # measured 0.82% of a step

    # The median is a *robust* estimator, so a rate taken from diff() is only
    # ~1.7e-4 off - small enough to look fine and still wrong. The damage is
    # cumulative: that rate misplaces a sample 18 deep into an 11 s record,
    # which is precisely what index_at must never do.
    naive = 1.0 / float(np.median(np.diff(x32.astype(np.float64))))
    assert abs(naive - 10_000.0) / 10_000.0 < 1e-3       # NOT a big rate error
    elapsed = 0.9 - float(x32[0])
    assert abs(round(elapsed * naive) - round(elapsed * fs)) >= 10


def test_sample_rate_rejects_degenerate_axes():
    with pytest.raises(ValueError):
        sample_rate(np.array([1.0]))
    with pytest.raises(ValueError):
        sample_rate(np.array([1.0, 1.0, 1.0]))


def test_index_at_counts_from_record_start_and_clamps():
    x = np.arange(-10.0, 1.0, 1e-4)  # 110000 samples
    assert index_at(x, 0.0)[0] == 100_000
    assert index_at(x, -10.0)[0] == 0
    assert index_at(x, -50.0)[0] == 0        # clamped, not wrapped
    assert index_at(x, 99.0)[0] == x.size - 1
    np.testing.assert_array_equal(index_at(x, [0.0, 0.5]), [100_000, 105_000])


def test_sample_at_handles_1d_and_2d_and_gaps():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y1 = np.array([10.0, 11.0, 12.0, 13.0])
    y2 = np.stack([y1, y1 * 2])
    np.testing.assert_allclose(sample_at(x, y1, [0.1, 2.9]), [10.0, 13.0])
    np.testing.assert_allclose(sample_at(x, y2, [0.1])[:, 0], [10.0, 20.0])
    out = sample_at(x, y1, [10.0], max_gap=0.5)
    assert np.isnan(out[0])


def test_window_mean_averages_the_following_window():
    x = np.arange(0.0, 1.0, 0.1)
    y = np.arange(10.0)
    # window [0.0, 0.5) covers samples 0..4 -> mean 2.0
    np.testing.assert_allclose(window_mean(x, y, [0.0], 0.5), [2.0])
    np.testing.assert_allclose(window_mean(x, y, [0.5], 0.5), [7.0])
    assert np.isnan(window_mean(x, y, [5.0], 0.5)[0])  # window outside record


def test_decimate_to_step_bin_averages_and_marks_empty_bins():
    x = np.array([0.0, 0.001, 0.002, 0.05, 0.051])
    y = np.array([1.0, 3.0, 5.0, 10.0, 20.0])
    xg, yg = decimate_to_step(x, y, 0.01)
    assert xg[0] == 0.0 and yg.shape == (1, 6)
    np.testing.assert_allclose(yg[0, 0], 3.0)     # (1+3+5)/3
    np.testing.assert_allclose(yg[0, 5], 15.0)    # (10+20)/2
    assert np.isnan(yg[0, 1:5]).all()             # nothing sampled there


def test_decimate_to_step_ignores_nans_within_a_bin():
    x = np.array([0.0, 0.001, 0.002])
    y = np.array([1.0, np.nan, 5.0])
    _, yg = decimate_to_step(x, y, 0.01)
    np.testing.assert_allclose(yg[0, 0], 3.0)
