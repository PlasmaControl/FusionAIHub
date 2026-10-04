"""Segment scores (MS-TCN), rank metrics and confident learning on small cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeler.confinement import confident, scoring, segments


def test_segments_split_at_unlabelled_bins():
    labels = np.array([0, 0, 1, 1, 1, -1, 1, 1])
    assert segments.segments(labels) == [(0, 0, 2), (1, 2, 5), (1, 6, 8)]


def test_edit_distance_and_score():
    assert segments.edit_distance([0, 1, 2], [0, 2]) == 1
    truth = [(0, 0, 5), (1, 5, 10)]
    assert segments.edit_score(truth, truth) == 100.0
    assert segments.edit_score(
        truth, [(0, 0, 3), (1, 3, 5), (0, 5, 7), (1, 7, 10)]
    ) == pytest.approx(50.0)


def test_perfect_prediction_matches_every_segment():
    truth = np.array([0] * 50 + [1] * 50 + [3] * 20)
    counts = segments.shot_counts(truth, truth)
    assert (counts["tp50"], counts["fp50"], counts["fn50"]) == (3, 0, 0)
    assert counts["edit"] == 100.0


def test_flicker_is_penalised_though_bins_are_mostly_right():
    truth = np.array([0] * 100)
    pred = truth.copy()
    pred[50::5] = 1
    counts = segments.shot_counts(truth, pred)
    assert counts["fp50"] >= 9
    assert counts["tp50"] == 1  # the first half, IoU 0.5, still matches


def test_overlap_threshold_decides_a_match():
    truth = np.array([0] * 100)
    pred = np.array([1] * 60 + [0] * 40)
    assert segments.shot_counts(truth, pred)["tp25"] == 1
    assert segments.shot_counts(truth, pred)["tp50"] == 0


def test_summarise_counts_f1_over_shots():
    shots = [
        segments.shot_counts(np.array([0] * 20), np.array([0] * 20)) for _ in range(4)
    ]
    out = segments.summarise(shots, replicates=20)
    assert out["f1_50"]["value"] == 1.0
    assert out["edit"]["value"] == 100.0


def test_auroc_and_average_precision_on_known_scores():
    score = np.array([0.1, 0.4, 0.35, 0.8])
    positive = np.array([0, 0, 1, 1])
    assert scoring.auroc(score, positive) == pytest.approx(0.75)
    assert scoring.average_precision(score, positive) == pytest.approx(
        0.8333333, abs=1e-6
    )
    assert scoring.auroc(np.array([1.0, 1.0]), np.array([1, 0])) == 0.5
    assert np.isnan(scoring.auroc(score, np.zeros(4)))


def test_one_vs_rest_macro_skips_missing_classes():
    probs = np.eye(4)[[0, 1, 0, 1]] * 0.9 + 0.025
    out = scoring.one_vs_rest(probs, np.array([0, 1, 0, 1]))
    assert out["auroc"]["macro"] == 1.0
    assert np.isnan(out["auroc"]["QH"])


def test_confident_joint_finds_a_flipped_label():
    labels = np.repeat([0, 1], 100)
    probs = np.where(labels[:, None] == np.arange(4), 0.75, 0.0625)  # exact in binary
    labels = labels.copy()
    labels[:5] = 1  # five L windows labelled H
    joint, guess = confident.confident_joint(probs, labels)
    assert joint[1, 0] == 5
    assert joint[0, 0] == 95 and joint[1, 1] == 100
    assert (guess[:5] == 0).all()


def test_interval_issues_flags_the_interval_that_disagrees():
    labels = np.repeat([0, 1, 1], 20)
    interval = np.repeat([0, 1, 2], 20)
    probs = np.where(labels[:, None] == np.arange(4), 0.75, 0.0625)
    probs[40:] = [0.75, 0.0625, 0.0625, 0.0625]  # interval 2 is labelled H but looks L
    frame = pd.DataFrame(
        {"shot": 1, "interval": interval, "label": labels, "keep": True}
    )
    out = confident.interval_issues(frame, probs)
    assert out.set_index("interval").flagged.to_dict() == {0: False, 1: False, 2: True}
    assert out.set_index("interval").other[2] == "L"


def test_calibrated_joint_keeps_label_counts_and_total():
    joint = np.array([[8, 2], [0, 5]])
    counts = np.array([20, 5])
    cal = confident.calibrate_joint(joint, counts)
    assert cal.sum() == pytest.approx(25.0)
    assert cal.sum(axis=1) == pytest.approx([20.0, 5.0])
    assert cal[0, 1] / cal[0].sum() == pytest.approx(0.2)
    assert confident.calibrate_joint(np.zeros((2, 2)), counts).sum() == 0.0


def test_rank_with_ci_brackets_the_point_estimate_and_resamples_whole_shots():
    rng = np.random.default_rng(3)
    shots = np.repeat(np.arange(12), 40)
    truth = np.tile(np.arange(4), 120)
    probs = rng.dirichlet(np.ones(4), size=480) * 0.4
    probs[np.arange(480), truth] += 0.6 * rng.random(480)
    probs /= probs.sum(1, keepdims=True)
    out = scoring.rank_with_ci(probs, truth, shots, replicates=60)
    lo, hi = out["ci95"]["auroc"]
    assert lo <= out["auroc"]["macro"] <= hi
    assert 0.5 < lo and hi <= 1.0
    assert out["replicates"] == 60
