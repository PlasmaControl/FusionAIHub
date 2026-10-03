"""Nested shot-grouped scoring of the RWM baselines on synthetic slice tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeler.rwm import evaluate as ev
from labeler.rwm import features, labels


def _table(n_hanson=12, n_comparison=24, seed=0):
    """Shots with one onset at 600 ms; betan/li climbs towards it in Hanson shots."""
    rng = np.random.default_rng(seed)
    frames, onsets = [], {}
    for shot in range(n_hanson + n_comparison):
        hanson = shot < n_hanson
        t = np.arange(100.0, 900.0, 10.0)
        ratio = 2.0 + 0.2 * rng.standard_normal(len(t))
        if hanson:
            # It climbs towards the onset, then collapses with the mode.
            ratio = ratio + np.where(t < 600, np.clip((t - 450.0) / 100.0, 0, 1.5), 0.0)
            onsets[shot] = [600.0]
        frame = pd.DataFrame({"t_ms": t, **dict.fromkeys(features.FEATURES, np.nan)})
        frame["betan_over_li"] = ratio
        frame["betan"] = ratio * 0.8
        frame["li"] = 0.8
        frame["n1rms_g"] = np.abs(rng.standard_normal(len(t)))
        frame["shot"] = shot
        frame["role"] = "hanson" if hanson else "comparison"
        frame["campaign"] = 2014 + 4 * (shot % 2)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    return ev.relabel(table, onsets, {}, 100.0), onsets


def test_relabel_marks_comparison_unlabelled_and_hanson_by_the_horizon():
    table, _ = _table()
    comparison = table[table.role == "comparison"]
    assert (comparison.label == labels.UNLABELLED).all()
    hanson = table[table.shot == 0]
    positive = hanson.t_ms[hanson.label == labels.POSITIVE]
    assert positive.min() == 500.0 and positive.max() == 590.0
    wide = ev.relabel(table, {0: [600.0]}, {}, 200.0)
    assert wide[wide.shot == 0].label.eq(labels.POSITIVE).sum() == 20


def test_imputer_fills_with_training_medians_and_zero_for_an_empty_column():
    frame = pd.DataFrame({"a": [1.0, 3.0, np.nan], "b": [np.nan] * 3})
    imputer = ev.Imputer().fit(frame, ["a", "b"])
    out = imputer.transform(pd.DataFrame({"a": [np.nan, 5.0], "b": [np.nan, 2.0]}))
    assert out.tolist() == [[2.0, 0.0], [5.0, 2.0]]


def test_make_folds_partitions_the_shots_and_spreads_each_stratum():
    shots = np.arange(20)
    strata = np.array(["a"] * 8 + ["b"] * 12)
    folds = ev.make_folds(shots, strata, 4, seed=1)
    assert sorted(np.concatenate(folds).tolist()) == list(range(20))
    for fold in folds:
        assert 1 <= (fold < 8).sum() <= 3  # every fold holds some of the small stratum
    assert ev.make_folds(shots, strata, 4, seed=1)[0].tolist() == folds[0].tolist()


def test_choose_rule_picks_levels_that_detect_without_false_alarms():
    t = np.arange(0.0, 600.0, 10.0)
    rng = np.random.default_rng(3)
    quiet = 0.1 + 0.02 * rng.standard_normal((2, len(t)))
    rising = 0.1 + 0.02 * rng.standard_normal(len(t))
    rising[(t >= 400) & (t < 500)] = 0.9  # a warning 100 ms before the onset at 500
    traces = {1: (t, rising), 2: (t, quiet[0]), 3: (t, quiet[1])}
    negatives = np.concatenate([quiet[0], quiet[1], rising[t < 400]])
    k_low, k_high, hold = ev.choose_rule(traces, {1: [500.0], 2: [], 3: []}, negatives)
    outcome = ev.score_alarms(traces, {1: [500.0], 2: [], 3: []}, (k_low, k_high, hold))
    assert outcome[1]["warning_ms"][0] is not None
    assert not outcome[2]["false"] and not outcome[3]["false"]


def test_cross_validation_scores_every_shot_once_and_finds_the_signal():
    table, onsets = _table()
    oof, alarms, rules = ev.cross_validate(
        table,
        lambda seed: ev.Brf(["betan_over_li", "n1rms_g"], seed=seed, n_estimators=20),
        onsets,
        outer=3,
        inner=2,
    )
    assert len(oof) == len(table) and oof.shot.nunique() == table.shot.nunique()
    assert len(rules) == 3 and set(alarms) == set(table.shot)
    groups = ev.shot_records(oof, alarms, onsets, onsets)
    stats = ev.statistic(groups)
    assert stats["slice_auroc"] > 0.9
    assert 0.0 <= stats["onset_detection_rate"] <= 1.0
    sizes = ev.counts(groups)
    assert sizes["hanson_shots"] == 12 and sizes["comparison_shots"] == 24
    assert sizes["target_onsets"] == 12
    assert sizes["positive_slices"] == 12 * 10


def test_a_held_out_shot_never_trains_its_own_model():
    table, onsets = _table()
    seen = []

    class Spy(ev.Brf):
        def fit(self, train):
            seen.append(set(train.shot))
            return super().fit(train)

        def score(self, frame):
            seen.append(("score", set(frame.shot)))
            return super().score(frame)

    ev.cross_validate(
        table,
        lambda seed: Spy(["betan_over_li"], seed=seed, n_estimators=5),
        onsets,
        outer=3,
        inner=2,
    )
    scored = [s for s in seen if isinstance(s, tuple)]
    trained = [s for s in seen if not isinstance(s, tuple)]
    assert scored and trained
    # Each fit is followed by a score on shots it has not seen.
    for fit_shots, (_, score_shots) in zip(trained, scored):
        assert not fit_shots & score_shots


def test_a_binary_rule_is_scored_without_folds():
    table, onsets = _table()
    table["call"] = (table.t_ms >= 520).astype(float)
    oof, alarms, rules = ev.cross_validate(
        table, lambda seed: ev.Rule("call", binary=True), onsets
    )
    assert rules == [{"fold": 0, "cutoff": 0.5, "rule": [0.5, 0.5, 0.0]}]
    hanson = alarms[0]
    assert hanson["warning_ms"] == [pytest.approx(80.0)]  # alarm at 520, onset 600
    assert alarms[20]["false"] == [520.0]  # a comparison shot with no onset
    stats = ev.statistic(ev.shot_records(oof, alarms, onsets, onsets))
    assert stats["onset_detection_rate"] == 1.0
    assert stats["comparison_false_alarm_shot_rate"] == 1.0


def test_single_feature_auroc_reads_the_direction():
    table, _ = _table()
    out = ev.single_feature_auroc(table, ["betan_over_li", "n1rms_g", "q95"])
    assert (
        out["betan_over_li"]["auroc"] > 0.9
        and out["betan_over_li"]["higher_means_unstable"]
    )
    assert out["n1rms_g"]["auroc"] < 0.7
    assert out["q95"]["present"] == 0.0
