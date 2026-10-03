"""Undefined resamples and tiny shot sets cannot imply population precision."""

import numpy as np
import pandas as pd
import pytest

from labeler.elm import labels, score, swap


def _part(shot, truth):
    truth = np.asarray(truth, np.int8)
    return score.ShotScore(
        shot,
        truth,
        np.where(truth, "crowd", "absent").astype(object),
        truth.astype(bool),
        truth.astype(float),
    )


def test_all_undefined_and_finite_draws_are_counted_without_tiny_subset_cis():
    parts = [_part(1, [1, 0]), _part(2, [0, 0])]
    boot = np.array([[0, 0], [0, 1], [1, 1]])
    result = score.summarise(parts, boot)

    assert result["point"]["auroc"] == 1.0
    assert result["bootstrap_draw_counts"]["auroc"] == {"valid": 2, "undefined": 1}
    assert result["bootstrap_draw_counts"]["non_crowd_span_touch_recall"] == {
        "valid": 0,
        "undefined": 3,
    }
    assert result["positive_shots"] == 1 and result["descriptive_only"]
    assert all(value is None for value in result["ci95"].values())
    paired = score.paired_difference(parts, parts, boot, "auroc")
    assert paired["ci95"] is None
    assert paired["bootstrap_draw_counts"] == {"valid": 2, "undefined": 1}


def test_positive_shot_count_controls_intervals_even_in_a_large_negative_set():
    parts = [_part(shot, [int(shot < 4), 0]) for shot in range(8)]
    result = score.summarise(parts, score.draws(8, 30))
    assert result["shots"] == 8 and result["positive_shots"] == 4
    assert result["ci95"]["auroc"] is None

    parts[4] = _part(4, [1, 0])
    result = score.summarise(parts, score.draws(8, 30))
    assert not result["descriptive_only"] and result["positive_shots"] == 5
    assert result["ci95"]["auroc"] == [1.0, 1.0]
    assert sum(result["bootstrap_draw_counts"]["auroc"].values()) == 30


def test_review_tau_merge_preserves_unknown_gaps_and_crowd_barriers():
    spans = pd.DataFrame(
        {
            "t_start": [0, 50, 100, 150, 200, 250, 300, 400],
            "t_end": [50, 100, 150, 200, 250, 300, 350, 450],
            "kind": [
                "non_crowd",
                "absent",
                "non_crowd",
                "uncertain",
                "non_crowd",
                "crowd",
                "non_crowd",
                "non_crowd",
            ],
        }
    )
    merged = swap.review_positive_intervals(spans, 100)
    assert list(zip(merged.t_start, merged.t_end)) == [
        (0, 150),
        (200, 250),
        (300, 350),
        (400, 450),
    ]
    bins = labels.Bins(
        np.array([0, 50, 100, 150, 200, 250], float),
        np.array([1, 0, 1, -1, 1, 1], np.int8),
        np.array(
            ["non_crowd", "absent", "non_crowd", "uncertain", "non_crowd", "crowd"]
        ),
        np.zeros(6, int),
    )
    assert swap.review_merged_truth(spans, bins, 100).tolist() == [1, 1, 1, -1, 1, 1]
    assert swap.review_merged_truth(spans, bins, 49).tolist() == [1, 0, 1, -1, 1, 1]
    with pytest.raises(ValueError, match="nonnegative"):
        swap.review_positive_intervals(spans, -1)
