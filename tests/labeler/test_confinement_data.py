"""Regression tests for conservative native-sample confinement features."""

import h5py
import numpy as np
import pandas as pd

from labeler.confinement import data


def test_missing_and_singleton_diagnostics_are_nan_not_physical_zero(tmp_path):
    p = tmp_path / "7_processed.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("co2")
        g["xdata"] = [0.0]
        g["ydata"] = np.zeros((4, 1))
    targets = pd.DataFrame({"shot": [7], "t_start": [0.0], "t_end": [50.0]})
    features, _ = data.extract_shot(7, targets, corpus=tmp_path)
    assert features.loc[0, "co2__present"] == 0
    assert np.isnan(features.loc[0, "co2__mean"])
    assert not data.physical_eligibility(features, data.feature_columns()).any()


def test_window_crossing_gap_is_unavailable():
    t = np.r_[np.arange(0.0, 0.02, 0.001), np.arange(0.04, 0.05, 0.001)]
    result = data.summarize_window(t, np.ones((2, len(t))), 0.0, 0.05)
    assert result["present"] == 0
    assert np.isnan(result["mean"])


def test_native_spectrum_keeps_frequency_above_decimated_nyquist():
    t = np.arange(5000) / 100000.0
    y = np.sin(2 * np.pi * 20000 * t)[None, :]
    result = data.summarize_window(t, y, 0.0, 0.05, fluctuation=True)
    assert result["present"] == 1
    assert result["power_10_50khz"] > 0.99
    assert abs(result["mean"]) < 1e-10


def test_thomson_never_uses_stale_profile():
    t = np.array([0.0, 0.1, 0.2])
    result = data.summarize_window(t, np.ones((3, 3)), 0.02, 0.07)
    assert result["present"] == 0


def test_subtype_requires_exact_one_regime_for_entire_bin():
    targets = pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "t_start": [0.0, 50.0, 100.0],
            "t_end": [50.0, 100.0, 150.0],
        }
    )
    merged = pd.DataFrame(
        {
            "shot": [1, 1, 1, 1],
            "t_start": [0.0, 50.0, 75.0, 110.0],
            "t_end": [50.0, 75.0, 100.0, 150.0],
            "regimes": ["QH", "H", "WP", "H"],
        }
    )
    assert data.derive_regimes(targets, merged).tolist() == [2, -1, -1]


def test_selected_channels_ignore_unselected_noise(tmp_path):
    t = np.arange(501) / 10000.0
    p = tmp_path / "9_processed.h5"
    with h5py.File(p, "w") as f:
        g = f.create_group("filterscopes")
        g["xdata"] = t
        g["ydata"] = np.r_[np.ones((8, len(t))), np.full((4, len(t)), 1000.0)]
    targets = pd.DataFrame({"shot": [9], "t_start": [0.0], "t_end": [50.0]})
    features, _ = data.extract_shot(9, targets, corpus=tmp_path)
    assert features.loc[0, "dalpha__mean"] == 1
