"""Regression checks for leakage, label conversion and shot-level paired scores."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.ae.supervision import (
    ShotMetric,
    activity_targets,
    clean_split,
    dense_states,
    frame_index,
    frame_mean,
    paired_scores,
    selection_threshold,
    validate_split,
)
from labeler.events.review.labels import Label
from labeler.scoring.frames import Assessment, frame_states


def test_split_is_fixed_by_shot_and_refuses_gold():
    train, valid = list(range(120)), list(range(120, 180))
    split = clean_split(train, valid, {999})
    assert split == clean_split(train[::-1], valid[::-1], {999})
    assert [len(split[k]) for k in ("train", "selection", "evaluation")] == [
        100,
        20,
        60,
    ]
    assert split["evaluation"] == valid
    validate_split(split, {999})
    with pytest.raises(ValueError, match="blind gold"):
        clean_split(train, valid, {119})
    split["selection"][0] = split["train"][0]
    with pytest.raises(ValueError, match="overlapping"):
        validate_split(split, {999})


def test_legacy_conversion_matches_audit_even_at_frame_edges():
    native = np.zeros(7820)
    index = frame_index(7820)
    cols = np.flatnonzero(index == 3)
    native[cols[: len(cols) // 2]] = 1
    assert frame_mean(native)[3] < 0.5
    native[cols[len(cols) // 2]] = 1
    assert frame_mean(native)[3] >= 0.5
    y, w, _, _ = activity_targets(native, native, native * 100, "legacy")
    np.testing.assert_array_equal(y, (frame_mean(native) >= 0.5)[index])
    assert w.all()


def test_dense_any_touch_matches_catalog_and_unions_crowds():
    label = Label((0, 2000), ((15, 19, 1), (41, 42, 2)))
    np.testing.assert_array_equal(
        dense_states(label), frame_states(Assessment.from_label(label), 0, 200)
    )
    overlapping = Label((0, 2000), ((15, 50, 1), (30, 60, 1)), (1, 0))
    assert dense_states(overlapping)[1:6].tolist() == [1] * 5


def test_dense_unknowns_masked_and_frequency_supervision_identical():
    ann = np.tile([0, 1], 400)
    active = np.tile([1, 1, 0, 0], 200)
    freq = np.full(800, 100.0)
    freq[1] = np.nan
    dense = np.zeros(200, dtype=int)
    dense[10:20] = 1
    dense[20:30] = 2
    dense[30:40] = -1
    arms = {
        name: activity_targets(active, ann, freq, name, dense)
        for name in ("legacy", "dense", "threeway")
    }
    for name in ("legacy", "dense"):
        np.testing.assert_array_equal(arms[name][2], arms["threeway"][2])
        np.testing.assert_array_equal(arms[name][3], arms["threeway"][3])
    assert not arms["dense"][1][80:160].any()
    np.testing.assert_array_equal(arms["threeway"][1], ann == active)


def test_threshold_uses_only_given_selection_and_handles_score_ties():
    got = selection_threshold(np.array([0.9, 0.9, 0.1]), np.array([1, 0, 1]))
    assert got["threshold"] == 0.1
    assert got["f1"] == 0.8
    with pytest.raises(ValueError, match="positive"):
        selection_threshold(np.array([0.1]), np.array([0]))


def _brute(score, truth, threshold):
    pos, neg = score[truth], score[~truth]
    auc = np.mean((pos[:, None] > neg) + 0.5 * (pos[:, None] == neg))
    ap = (
        sum(
            np.sum(truth[score >= s]) / np.sum(score >= s) * np.sum(truth[score == s])
            for s in np.unique(score)
        )
        / truth.sum()
    )
    hard = score >= threshold
    f1 = 2 * (hard & truth).sum() / (hard.sum() + truth.sum())
    return np.array([auc, ap, f1])


def test_weighted_bootstrap_equals_literal_whole_shot_resampling():
    parts = [
        (np.array([0.9, 0.5, 0.5]), np.array([1, 1, 0], bool)),
        (np.array([0.5, 0.2]), np.array([0, 1], bool)),
    ]
    metric = ShotMetric(parts, 0.5)
    for weights in ([1, 1], [2, 1], [0, 2]):
        chosen = [parts[i] for i, count in enumerate(weights) for _ in range(count)]
        score = np.concatenate([p[0] for p in chosen])
        truth = np.concatenate([p[1] for p in chosen])
        np.testing.assert_allclose(
            metric.values(np.array(weights, float)), _brute(score, truth, 0.5)
        )


def test_paired_bootstrap_identical_methods_have_exact_zero_difference():
    parts = [(np.array([0.2, 0.8]), np.array([0, 1], bool))] * 3
    result = paired_scores({"a": (parts, 0.5), "b": (parts, 0.5)}, 1000)
    for score in result["paired_differences"]["a minus b"].values():
        assert score["value"] == 0
        assert score["ci95"] == [0, 0]
        assert score["valid_replicates"] == 1000
    bad = [(parts[0][0], ~parts[0][1])] * 3
    with pytest.raises(ValueError, match="identical"):
        paired_scores({"a": (parts, 0.5), "b": (bad, 0.5)}, 10)


def test_single_class_auc_and_ap_are_explicitly_unavailable():
    parts = [(np.array([0.1, 0.2]), np.zeros(2, bool))] * 2
    result = paired_scores({"a": (parts, 0.5)}, 10)
    assert result["methods"]["a"]["auroc"]["value"] is None
    assert result["methods"]["a"]["auprc"]["ci95"] is None


def test_seed_groups_pool_paired_shot_and_seed_draws_with_sample_sd():
    methods = {}
    for seed in range(3):
        parts = [
            (np.array([0.2 + 0.3 * seed, 0.8 - 0.2 * seed]), np.array([0, 1], bool)),
            (np.array([0.1, 0.9 - 0.35 * seed]), np.array([0, 1], bool)),
        ]
        methods[f"a{seed}"] = (parts, 0.5)
        methods[f"b{seed}"] = (parts, 0.5)
    groups = {"a": ["a0", "a1", "a2"], "b": ["b0", "b1", "b2"]}
    result = paired_scores(methods, 100, seed=23, groups=groups)
    points = np.array(
        [
            _brute(
                np.concatenate([p[0] for p in methods[name][0]]),
                np.concatenate([p[1] for p in methods[name][0]]),
                0.5,
            )
            for name in groups["a"]
        ]
    )
    rng = np.random.default_rng(23)
    shots = rng.integers(2, size=(100, 2))
    seeds = rng.integers(3, size=(100, 3))
    samples = []
    for selected_shots, selected_seeds in zip(shots, seeds, strict=True):
        values = []
        for seed in selected_seeds:
            parts = [methods[f"a{seed}"][0][i] for i in selected_shots]
            values.append(
                _brute(
                    np.concatenate([p[0] for p in parts]),
                    np.concatenate([p[1] for p in parts]),
                    0.5,
                )
            )
        samples.append(np.mean(values, axis=0))
    for i, metric in enumerate(("auroc", "auprc", "f1")):
        got = result["seed_summary"]["methods"]["a"][metric]
        assert got["mean"] == pytest.approx(points[:, i].mean())
        assert got["sd"] == pytest.approx(points[:, i].std(ddof=1))
        np.testing.assert_allclose(
            got["ci95"], np.percentile(np.array(samples)[:, i], [2.5, 97.5])
        )
        diff = result["seed_summary"]["paired_differences"]["a minus b"][metric]
        assert diff["mean"] == 0
        assert diff["sd"] == 0
        assert diff["ci95"] == [0, 0]


def test_seed_groups_refuse_mismatched_seed_counts():
    parts = [(np.array([0.2, 0.8]), np.array([0, 1], bool))]
    with pytest.raises(ValueError, match="seed counts"):
        paired_scores(
            {k: (parts, 0.5) for k in "abcde"},
            10,
            groups={"x": ["a", "b"], "y": ["c", "d", "e"]},
        )
