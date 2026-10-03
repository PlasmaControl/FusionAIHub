"""The 1 ms grid of a shot's 0D signals and the interval labels on it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeler.confinement import zerod


def test_bin_stats_mean_max_std_and_empty_bins():
    t = np.array([0.1, 0.5, 0.9, 2.2, 2.4])
    y = np.array([1.0, 2.0, 3.0, 4.0, 8.0])
    mean, peak, std = zerod.bin_stats(t, y, 4)
    assert mean[0] == 2.0 and peak[0] == 3.0
    assert np.isclose(std[0], np.sqrt(2 / 3), atol=1e-6)
    assert np.isnan(mean[1]) and np.isnan(peak[1])
    assert mean[2] == 6.0 and peak[2] == 8.0
    assert np.isnan(mean[3])


def test_interpolate_sparse_refuses_long_gaps_and_extrapolation():
    t = np.array([10.0, 20.0, 30.0, 500.0, 510.0])
    y = np.array([1.0, 2.0, 3.0, 9.0, 9.0])
    out = zerod.interpolate_sparse(t, y, 600)
    assert np.isnan(out[:9]).all()  # before the first sample
    assert np.isclose(out[14], 1.45, atol=1e-5)  # bin centre 14.5 ms
    assert np.isnan(out[100])  # the 470 ms gap
    assert np.isnan(out[550])  # after the last sample


def test_interval_labels_mark_whole_bins_only():
    iv = pd.DataFrame({"t_start": [10.5, 40.0], "t_end": [20.0, 45.0], "label": [1, 3]})
    y = zerod.interval_labels(iv, 60)
    assert (y[:11] == -1).all() and (y[11:20] == 1).all() and (y[20:40] == -1).all()
    assert (y[40:45] == 3).all() and (y[45:] == -1).all()


def test_to_input_zero_fills_and_adds_one_mask_per_sparse_channel():
    grid = np.full((len(zerod.CHANNELS), 5), 0.5, dtype=np.float32)
    grid[zerod.CHANNELS.index("betan"), 2:] = np.nan
    out = zerod.to_input(grid)
    assert out.shape == (zerod.N_INPUT, 5)
    assert np.isfinite(out).all()
    assert out[zerod.CHANNELS.index("betan"), 2:].tolist() == [0.0, 0.0, 0.0]
    mask = out[len(zerod.CHANNELS) + zerod.SPARSE.index("betan")]
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_assemble_without_d_alpha_returns_none(tmp_path):
    assert zerod.assemble(1, tmp_path, tmp_path, corpus=tmp_path) is None
