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
    absent = np.zeros(10, dtype=bool)
    absent[7:] = True
    states, assessed = state_spans(
        t, observable, [(0.001, 0.004)], [(0.003, 0.007)], absent=absent
    )
    expanded = {r["state"] for r in states}
    assert expanded == {"present", "absent", "uncertain", "unassessed"}
    assert not assessed[3:7].any()
    assert not assessed[0]
    assert assessed[1:3].all()
    assert assessed[7:].all()


def test_observable_support_without_absence_evidence_stays_uncertain():
    t = np.arange(10) * 0.001
    spans, assessed = state_spans(t, np.ones(10, dtype=bool), [], [])
    assert len(spans) == 1
    assert spans[0]["state"] == "uncertain"
    assert spans[0]["start_s"] == 0.0
    assert spans[0]["end_s"] == pytest.approx(0.01)
    assert not assessed.any()


def test_uncertainty_and_missing_support_override_explicit_absence():
    t = np.arange(10) * 0.001
    observable = np.ones(10, dtype=bool)
    observable[4] = False
    spans, assessed = state_spans(
        t, observable, [], [(0.002, 0.006)], absent=np.ones(10, dtype=bool)
    )
    assert [row["state"] for row in spans] == [
        "absent",
        "uncertain",
        "unassessed",
        "uncertain",
        "absent",
    ]
    assert not assessed[2:6].any()


def test_q_prior_time_is_its_own_unassessed_state_and_never_a_negative():
    t = np.arange(10) * 0.001
    q_prior = np.zeros(10, dtype=bool)
    q_prior[2:8] = True
    absent = np.zeros(10, dtype=bool)
    absent[6:8] = True
    spans, assessed = state_spans(
        t, np.ones(10, dtype=bool), [], [(0.004, 0.005)], absent=absent, q_prior=q_prior
    )
    states = {(r["start_s"], r["state"]) for r in spans}
    assert (0.002, "absent_q_prior") in states
    assert (0.006, "absent") in states
    assert (0.004, "uncertain") in states  # uncertain overrides the prior
    assert assessed.tolist() == [False] * 6 + [True] * 2 + [False] * 2
