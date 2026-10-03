"""Independent Smith targets, event matching and leakage guards."""

import numpy as np
import pandas as pd
import pytest

from labeler.elm import smith


def test_target_union_never_double_counts_overlapping_windows():
    rows = pd.DataFrame(
        {
            "t0_ms": [0.2, 3.2],
            "end_ms": [6.2, 9.2],
            "label_t0_ms": [2.1, 7.1],
            "label_t1_ms": [2.4, 7.5],
        }
    )
    state, target, mask = smith.targets(rows, 12, grid0=0.0)
    assert np.flatnonzero(mask).tolist() == list(range(1, 9))
    assert np.flatnonzero(state == 1).tolist() == [2, 7]
    assert state[0] == -1 and state[9] == -1
    assert target[2] > 0.9 and target[7] > 0.9


def test_matching_maximizes_matches_and_keeps_signed_error():
    # Greedy nearest matching would consume the event needed by the second truth.
    result = smith.match_events(np.array([0.0, 2.0]), np.array([1.1, 3.0]), 1.5)
    assert result["tp"] == 2
    assert result["fp"] == result["fn"] == 0
    assert result["errors_ms"] == pytest.approx([-1.1, -1.0])


def test_matching_one_detection_cannot_match_two_truths():
    result = smith.match_events(np.array([1.0]), np.array([0.0, 2.0]), 2.0)
    assert result["tp"] == 1 and result["fn"] == 1


def test_fast_validation_counts_equal_optimal_assignment():
    rng = np.random.default_rng(13)
    for _ in range(30):
        found, truth = rng.uniform(0, 20, (2, 12))
        result = smith.match_events(found, truth, 2.0)
        assert smith.event_counts(found, truth, 2.0) == tuple(
            result[k] for k in ("tp", "fp", "fn")
        )


def test_prohibit_review_and_blind_test_in_smith_training():
    cohort = pd.DataFrame({"shot": [1, 2, 3], "split": ["train", "val", "test"]})
    with pytest.raises(ValueError, match="review"):
        smith.check_disjoint([1, 4], [1, 2], cohort)
    with pytest.raises(ValueError, match="test"):
        smith.check_disjoint([3, 4], [1, 2], cohort)
    smith.check_disjoint([4, 5], [1, 2], cohort)


def test_detection_dedup_preserves_close_distinct_events():
    assert smith.deduplicate_events([1.0, 1.01, 3.0], 0.1) == [1.0, 3.0]


def test_first_monotone_segment_does_not_mix_repeated_acquisitions():
    time = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
    signal = np.array([[10.0, 11.0, 12.0, 100.0, 101.0]])
    t, y, audit = smith.first_monotone_segment(time, signal)
    np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(y, [[10.0, 11.0, 12.0]])
    assert audit["discarded_samples"] == 2
    t, y, audit = smith.first_monotone_segment(t, y)
    assert audit["discarded_samples"] == 0


def test_event_bootstrap_records_undefined_draws():
    parts = [
        {"tp": 1, "fp": 0, "fn": 0, "errors_ms": [0.1]},
        {"tp": 0, "fp": 0, "fn": 0, "errors_ms": []},
    ]
    result = smith.event_summary(parts, np.array([[0, 0], [1, 1], [0, 1]]))
    assert result["bootstrap"]["precision"] == {"valid": 2, "undefined": 1}
    assert "ci95" not in result  # fewer than five positive-bearing shots
    assert result["descriptive_only"]
