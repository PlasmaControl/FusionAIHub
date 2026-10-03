"""Span metrics only credit detections in the panel's analysed intervals."""

import numpy as np
import pandas as pd
import pytest

from labeler.elm import methods, score


def _review(start, stop, kind):
    return pd.DataFrame({"t_start": [start], "t_end": [stop], "kind": [kind]})


def test_detection_after_common_coverage_cannot_hit_shot_203941_span():
    review = _review(5586.0, 6012.0, "non_crowd")
    cover = methods.cover_frame([5500.0], [5975.0])
    found = methods.span_frame([5990.0], [6012.0])

    counts = methods.span_counts(found, cover, review)

    assert counts["non_crowd_spans"] == 1
    assert counts["non_crowd_span_hit"] == 0


def test_raw_and_guarded_alarms_ignore_detection_inside_coverage_gap():
    review = _review(0.0, 100.0, "absent")
    cover = methods.cover_frame([0.0, 75.0], [25.0, 100.0])
    found = methods.span_frame([30.0], [70.0])

    counts = methods.span_counts(found, cover, review)

    assert counts["absent_spans"] == 1  # exactly half the span is covered
    assert counts["absent_span_alarm"] == 0
    assert counts["absent_spans_guard25_eligible"] == 1
    assert counts["absent_span_alarm_guard25"] == 0


def test_clipping_preserves_covered_raw_touch_but_removes_guarded_touch():
    review = _review(0.0, 100.0, "absent")
    cover = methods.cover_frame([0.0, 75.0], [25.0, 100.0])
    found = methods.span_frame([20.0], [80.0])

    counts = methods.span_counts(found, cover, review)

    assert counts["absent_span_alarm"] == 1
    assert counts["absent_span_alarm_guard25"] == 0


def test_empty_coverage_has_no_eligible_spans_or_alarms():
    counts = methods.span_counts(
        methods.span_frame([10.0], [90.0]),
        methods.cover_frame([], []),
        _review(0.0, 100.0, "absent"),
    )

    assert all(value == 0 for value in counts.values())


def test_annotation_strata_use_full_review_and_keep_shots_in_one_group():
    reviews = {
        1: _review(0.0, 100.0, "crowd"),
        2: _review(0.0, 100.0, "non_crowd"),
        3: pd.concat(
            [_review(0.0, 100.0, "crowd"), _review(100.0, 200.0, "non_crowd")]
        ),
        4: _review(0.0, 100.0, "absent"),
    }
    parts = []
    for shot, truth, kind, call in (
        (1, [1, 1, 0], ["crowd", "crowd", "absent"], [1, 0, 0]),
        (2, [1, 0], ["non_crowd", "absent"], [1, 1]),
        # The non-crowd annotation has no scored bin, but this is still mixed.
        (3, [1, 0], ["crowd", "absent"], [1, 0]),
        (4, [0, 0], ["absent", "absent"], [1, 0]),
    ):
        parts.append(
            score.ShotScore(
                shot,
                np.array(truth, np.int8),
                np.array(kind, object),
                np.array(call, bool),
                np.array(call, float),
            )
        )

    result = methods.annotation_summary({"elm-ours": parts}, reviews, replicates=20)

    assert {key: group["shots"] for key, group in result.items()} == {
        "crowd_only": [1],
        "non_crowd_only": [2],
        "mixed": [3],
        "no_present": [4],
    }
    absent = result["no_present"]["methods"]["elm-ours"]
    assert absent["point"]["no_present_false_positive_fraction"] == 0.5
    assert absent["ci95"]["no_present_false_positive_fraction"] == [0.5, 0.5]
    assert result["crowd_only"]["methods"]["elm-ours"]["replicates"] == 20


def test_per_kind_summary_has_bin_and_span_recall_with_shot_intervals():
    part = score.ShotScore(
        1,
        np.array([1, 1, 1, 0], np.int8),
        np.array(["crowd", "crowd", "non_crowd", "absent"], object),
        np.array([1, 0, 1, 0], bool),
        spans={"non_crowd_spans": 4, "non_crowd_span_hit": 3},
    )

    out = methods.kind_summary([part], score.draws(1, replicates=20))

    assert out["crowd_bin_recall"] == {
        "point": 0.5,
        "ci95": [0.5, 0.5],
        "numerator": 1,
        "denominator": 2,
    }
    assert out["non_crowd_bin_recall"]["point"] == 1.0
    assert out["non_crowd_span_touch_recall"]["point"] == pytest.approx(3 / 4)
    assert out["non_crowd_span_touch_recall"]["ci95"] == [0.75, 0.75]
