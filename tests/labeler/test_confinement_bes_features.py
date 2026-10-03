"""The BES window features: the paper's recipe at any sampling rate."""

from __future__ import annotations

import numpy as np

from labeler.confinement import bes_features as bf


def test_window_starts_stay_inside_the_interval():
    starts = bf.window_starts(100, 100 + 1024 + 2048 * 3 + 5, 2048)
    assert starts.tolist() == [100, 2148, 4196, 6244]
    assert starts[-1] + bf.WINDOW <= 100 + 1024 + 2048 * 3 + 5
    assert bf.window_starts(0, 1023, 2048).size == 0


def test_features_peak_at_the_driven_bin():
    fs = 1e6
    t = np.arange(bf.WINDOW) / fs
    freq = 40e3  # bin 40e3 / (fs / 256) = 10.24 of the 256-point segments
    window = np.sin(2 * np.pi * freq * t)[None, None, :]
    feats = bf.spectral_features(window).astype(np.float64)
    assert feats.shape == (1, 2, 1, bf.FREQS)
    assert abs(int(np.argmax(feats[0, 0, 0])) - 10) <= 1
    assert feats[0, 0, 0, 10] > feats[0, 0, 0, 60] + 3


def test_filtered_windows_remove_the_dc_level_and_keep_the_band():
    fs = 1e6
    n = 20_000
    t = np.arange(n) / fs
    signal = (5.0 + np.sin(2 * np.pi * 50e3 * t))[None, :]
    starts = bf.window_starts(8000, n, 2048)
    out = bf.filtered_windows(signal, starts, bf.band_sos(fs))
    assert out.shape == (starts.size, 1, bf.WINDOW)
    assert abs(out.mean()) < 0.05
    assert 0.6 < np.sqrt((out**2).mean()) < 0.8


def test_standardising_offset_gives_unit_variance():
    rng = np.random.default_rng(0)
    power = rng.uniform(2.0, 4.0, size=(50, 3))
    offset = bf.standardising_offset(power)
    assert np.allclose(power.mean(axis=0) * 10.0**offset, 1.0)
