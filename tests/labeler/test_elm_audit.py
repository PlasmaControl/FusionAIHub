"""Coverage audit tests: boundaries, uncertain time, and span starts stay distinct."""

import numpy as np
import pandas as pd
import pytest

from labeler.elm import labels, score, swap


def test_majority_coverage_includes_boundary_and_uncertain_bins():
    table = pd.DataFrame(
        {"shot": [1, 1], "t_start": [0, 100], "t_end": [100, 200], "category": [1, 0]}
    )
    review = pd.DataFrame(
        {
            "shot": [1] * 4,
            "t_start": [10, 40, 100, 150],
            "t_end": [40, 100, 150, 190],
            "kind": ["non_crowd", "absent", "uncertain", "crowd"],
        }
    )
    bins, legacy, status = swap.coverage_bins(table, 1, review)
    assert bins.t0.tolist() == [0, 50, 100, 150]
    assert bins.truth.tolist() == [1, 0, -1, 1]
    assert legacy.tolist() == [1, 1, 0, 0]
    assert status.tolist() == ["present", "absent", "uncertain", "present"]
    known = bins.truth >= 0
    c = swap.agreement_counts(bins.truth[known], bins.kind[known], legacy[known])
    result = swap.agreement_summary(c[None], score.draws(1, 2))
    assert result["M"] == 1 and result["P"] == 1 and result["bins"] == 3


def test_legacy_majority_and_gaps_are_not_midpoint_labels():
    table = pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "t_start": [0, 30, 100],
            "t_end": [30, 50, 140],
            "category": [1, 0, 0],
        }
    )
    review = pd.DataFrame(
        {"shot": [1], "t_start": [0], "t_end": [150], "kind": ["absent"]}
    )
    bins, legacy, _ = swap.coverage_bins(table, 1, review)
    assert bins.t0.tolist() == [0, 100]  # uncovered middle bin excluded
    assert legacy.tolist() == [1, 0]


def test_onset_agreement_is_per_span_start_and_has_explicit_tolerance():
    review = pd.DataFrame(
        {
            "t_start": [101, 400, 600],
            "t_end": [350, 450, 800],
            "kind": ["non_crowd", "non_crowd", "crowd"],
        }
    )
    result = swap.start_agreement(review, np.array([100, 350, 650]), 50)
    assert result["non_crowd"]["spans"] == 2
    assert result["non_crowd"]["matched"] == 2  # inclusive 50 ms
    assert result["crowd"]["matched"] == 1
    assert (
        swap.start_agreement(review, np.array([100, 350]), 5)["non_crowd"]["matched"]
        == 1
    )


def test_non_crowd_length_summary_counts_spans_not_bins():
    review = pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "t_start": [0, 150, 600],
            "t_end": [50, 450, 900],
            "kind": ["non_crowd", "non_crowd", "crowd"],
        }
    )
    result = labels.span_length_summary(review)
    assert result["spans"] == 2
    assert result["length_ms"]["median"] == 175
    assert result["longer_than_100ms"] == 1
    assert result["possible_bins_from_longer_than_100ms"] == 6


def test_sweep_bin_scores_preserve_rank_auc_and_empty_setting():
    bins = labels.Bins(
        np.arange(4) * 50.0,
        np.array([1, 0, 1, 0]),
        np.array(["non_crowd", "absent", "crowd", "absent"], object),
        np.arange(4),
    )
    sweep = pd.DataFrame(
        {"eta": [0.9, 0.5, 0.5], "t_start_ms": [0, 0, 100], "t_end_ms": [1, 51, 101]}
    )
    ranked = swap.sweep_bin_scores(sweep, bins)
    assert ranked.tolist() == pytest.approx([0.9, 0.5, 0.5, -1])
    assert score.roc_auc(bins.truth, ranked) == pytest.approx(0.875)


def test_table_marks_high_recall_f1_as_degenerate():
    from labeler.elm import swap_tex

    result = {"point": {"recall": 0.99, "f1": 0.4}, "ci95": {"f1": [0.2, 0.5]}}
    assert swap_tex.metric_cell(result, "f1") == r"\textit{degenerate}"
    result["point"]["recall"] = 0.989
    assert "0.400" in swap_tex.metric_cell(result, "f1")
