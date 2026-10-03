import h5py
import numpy as np
import pytest

from labeler.sawtooth.preprocessing import sample_native, state_spans


def test_native_filter_suppresses_alias_and_chunk_boundaries(tmp_path):
    t = np.arange(50000) / 500000
    y = np.sin(2 * np.pi * 49400 * t)[None, :]
    with h5py.File(tmp_path / "signal.h5", "w") as h:
        h["xdata"], h["ydata"] = t, y
        tx, filtered = sample_native(h, fs=10000, chunk_samples=10000)
        _, other = sample_native(h, fs=10000, chunk_samples=20000)
    assert np.std(filtered[:, 20:-20]) < 1e-3
    assert np.allclose(filtered, other, atol=1e-6)
    assert np.allclose(np.diff(tx), 1 / 10000)


def test_native_rms_retains_high_frequency_magnetic_energy(tmp_path):
    t = np.arange(50000) / 500000
    y = np.sin(2 * np.pi * 49400 * t)[None, :]
    with h5py.File(tmp_path / "signal.h5", "w") as h:
        h["xdata"], h["ydata"] = t, y
        _, rms = sample_native(h, rms=True)
    assert np.allclose(rms[:, 20:-20], 1 / np.sqrt(2), atol=0.001)


def test_native_filter_preserves_nan_support_and_constant(tmp_path):
    t = np.arange(20000) / 100000
    y = np.ones((2, len(t))) * 2
    y[0, 10000:10020] = np.nan
    with h5py.File(tmp_path / "signal.h5", "w") as h:
        h["xdata"], h["ydata"] = t, y
        tx, filtered = sample_native(h, chunk_samples=3000)
    assert np.allclose(filtered[1], 2)
    assert np.isnan(filtered[0, np.abs(tx - 0.1) < 0.001]).all()
    assert np.isfinite(filtered[0, tx < 0.09]).all()


def test_float32_native_clock_roundoff_is_not_a_sampling_gap(tmp_path):
    t = (3 + np.arange(50000) / 500000).astype(np.float32)
    with h5py.File(tmp_path / "signal.h5", "w") as h:
        h["xdata"], h["ydata"] = t, np.ones((1, len(t)))
        tx, filtered = sample_native(h)
    assert np.isfinite(filtered).all()
    assert np.max(np.abs(np.diff(tx) - np.median(np.diff(tx)))) < 1e-10


def test_duplicate_native_timestamp_is_rejected_separately_from_roundoff(tmp_path):
    t = (6 + np.arange(50000) / 500000).astype(np.float32)
    t[101] = t[100]
    with h5py.File(tmp_path / "signal.h5", "w") as h:
        h["xdata"], h["ydata"] = t, np.ones((1, len(t)))
        with pytest.raises(ValueError, match="increasing"):
            sample_native(h)


def test_four_states_do_not_turn_missing_or_uncertain_into_absence():
    t = np.arange(10) * 0.001
    observable = np.ones(10, dtype=bool)
    observable[4:6] = False
    states, assessed = state_spans(t, observable, [(0.001, 0.004)], [(0.003, 0.007)])
    expanded = {r["state"] for r in states}
    assert expanded == {"present", "absent", "uncertain", "unassessed"}
    assert not assessed[3:7].any()
    assert assessed[:3].all()
