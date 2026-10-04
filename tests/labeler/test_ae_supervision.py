"""Regression checks for leakage, label conversion and shot-level paired scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from labeler.ae.supervision import (
    ShotMetric,
    activity_targets,
    any_touch_prevalence,
    clean_split,
    dense_states,
    frame_index,
    frame_mean,
    grid_shift,
    interval_changing_saves,
    paired_scores,
    selection_threshold,
    snapshot_difference,
    validate_split,
)
from labeler.events.catalog.states import ABSENT, PRESENT
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


def test_convergence_screen_uses_selection_variation_and_discrimination():
    from labeler.ae.supervision import convergence_screen

    truth = np.tile([False, True], 100)
    good = [(np.where(truth, 0.8, 0.2), truth)] * 2
    assert convergence_screen(good)["passed"]
    constant = [(np.where(truth, 0.081, 0.080), truth)] * 2
    assert not convergence_screen(constant)["passed"]
    assert not convergence_screen([(1 - s, y) for s, y in good])["passed"]


def test_clock_prior_masks_unknown_and_refuses_unobserved_bins():
    from labeler.ae.supervision import clock_prior

    states = np.zeros((2, 200), dtype=int)
    states[0, :100] = PRESENT
    states[1, :100] = 2  # unknown is not a negative
    prior = clock_prior(states)
    np.testing.assert_array_equal(prior[:100], 1)
    np.testing.assert_array_equal(prior[100:], 0)
    with pytest.raises(ValueError, match="observed"):
        clock_prior(np.full((2, 200), 2))


def test_within_shot_scores_exclude_one_class_and_bootstrap_shots():
    from labeler.ae.supervision import within_shot_scores

    parts = [
        (np.array([0.1, 0.9]), np.array([False, True])),
        (np.array([0.9, 0.1]), np.array([False, True])),
        (np.array([0.2, 0.3]), np.array([False, False])),
    ]
    result = within_shot_scores(parts, replicates=100)
    assert result["n_two_class_shots"] == 2
    assert result["auroc"]["value"] == 0.5
    assert result["auroc"]["ci95"] == [0.0, 1.0]


LEGACY_SEED2 = [2.211, 1.675, 1.645, 1.655, 1.657, 1.647, 1.662, 1.66]
DENSE_SEED1 = [2.856, 2.25, 2.163, 1.76, 1.552, 1.382, 1.357, 1.366, 1.398, 1.371]
DENSE_SEED1 += [1.367, 1.37]


def test_declared_patience_rule_flags_runs_that_stopped_inside_the_plateau():
    from labeler.ae.supervision import training_conformance

    first = training_conformance(LEGACY_SEED2)
    assert not first["conforms"] and not first["stopped"]
    assert first["best_epoch"] == 2
    second = training_conformance(DENSE_SEED1)
    assert not second["conforms"]
    # the same trace continued until five post-epoch-9 misses conforms
    longer = DENSE_SEED1 + [1.37, 1.372, 1.4]
    third = training_conformance(longer)
    assert third["conforms"] and third["final_epoch"] == 14
    # a best epoch of 9 or later behaves exactly as the plain patience-5 rule
    late = [2.0, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7] + [0.71] * 5
    assert training_conformance(late)["conforms"]
    assert training_conformance(late[:-1])["conforms"] is False


def test_training_loop_counter_matches_replayed_rule():
    import importlib

    from labeler.ae.supervision import replay_early_stop

    scripts = Path(__file__).resolve().parents[2] / "scripts/labeler"
    sys.path.insert(0, str(scripts))
    try:
        trainer = importlib.import_module("ae_train")
    finally:
        sys.path.remove(str(scripts))
    rng = np.random.default_rng(7)
    for _ in range(200):
        losses = (2.0 - np.cumsum(rng.uniform(-0.05, 0.12, 30))).tolist()
        best, bad, stop = float("inf"), 0, None
        for epoch, loss in enumerate(losses):
            improved = loss < best - 1e-6
            if improved:
                best = loss
            bad = trainer.next_bad_count(bad, improved, epoch, 10)
            if not improved and bad >= 5:
                stop = epoch
                break
        replay = replay_early_stop(losses)
        assert (replay["final_epoch"] if replay["stopped"] else None) == stop


def test_seed_group_shot_interval_ignores_seed_resampling():
    parts = [
        (np.array([0.2, 0.8]), np.array([0, 1], bool)),
        (np.array([0.4, 0.3, 0.9]), np.array([0, 1, 1], bool)),
        (np.array([0.1, 0.6]), np.array([0, 1], bool)),
    ]
    methods = {f"a{i}": (parts, 0.5) for i in range(2)}
    result = paired_scores(methods, 200, groups={"a": ["a0", "a1"]})
    for metric in ("auroc", "auprc", "f1"):
        got = result["seed_summary"]["methods"]["a"][metric]
        # identical seeds: seed resampling adds nothing, so both intervals agree
        np.testing.assert_allclose(got["ci95"], got["ci95_shot"])
        assert got["sd"] == 0


def test_snapshot_difference_counts_frames_and_net_present():
    old = {1: np.full(200, ABSENT, np.int8), 2: np.full(200, ABSENT, np.int8)}
    new = {1: old[1].copy(), 2: old[2].copy()}
    new[2][10:15] = PRESENT
    old[2][40:42] = PRESENT
    diff = snapshot_difference(old, new, {"a": [1], "b": [2]})
    assert diff["a"]["differing_shots"] == [] and diff["a"]["frames"] == 0
    assert diff["b"] == {
        "shots": 1,
        "differing_shots": [2],
        "frames": 7,
        "net_present": 3,
    }
    assert any_touch_prevalence(new, [1, 2]) == 5 / 400


def test_interval_changing_saves_ignores_first_saves_and_repeats():
    def save(shot, when, name, intervals):
        return {
            "shot": shot,
            "saved_at": when,
            "name": name,
            "window": [0, 2000],
            "intervals": intervals,
        }

    entries = [
        save(1, "2026-01-01T00:00", None, [[0, 10, 1]]),
        save(1, "2026-01-02T00:00", "A", [[0, 10, 1]]),  # confirmation only
        save(1, "2026-01-03T00:00", "B", [[0, 20, 1]]),  # a change
        save(2, "2026-01-01T00:00", None, []),
        save(2, "2026-01-04T00:00", "A", [[5, 6, 1]]),  # a change
    ]
    got = interval_changing_saves(entries)
    assert [(c["shot"], c["name"]) for c in got] == [(1, "B"), (2, "A")]
    later = interval_changing_saves(entries, after="2026-01-03T12:00")
    assert [(c["shot"], c["name"]) for c in later] == [(2, "A")]


def test_grid_shift_counts_columns_moved_by_the_recorded_times():
    n = 7820
    centres = (np.arange(n) + 0.5) * 2000 / n
    assert grid_shift(centres)["columns_in_another_frame"] == 0
    shifted = grid_shift(centres - 0.9)
    assert shifted["columns_in_another_frame"] > 0
    assert shifted["max_abs_offset_ms"] == pytest.approx(0.9)
