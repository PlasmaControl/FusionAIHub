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
    assert stats["comparison_alarm_incidence"] == 1.0


def test_single_feature_auroc_reads_the_direction():
    table, _ = _table()
    out = ev.single_feature_auroc(table, ["betan_over_li", "n1rms_g", "q95"])
    assert (
        out["betan_over_li"]["auroc"] > 0.9
        and out["betan_over_li"]["higher_means_unstable"]
    )
    assert out["n1rms_g"]["auroc"] < 0.7
    assert out["q95"]["present"] == 0.0


def test_chance_detection_is_the_hit_probability_of_randomly_placed_alarms():
    records = [
        {"warning_ms": [100.0, None], "alarms": 2, "span_ms": 3900.0},
        {"warning_ms": [None], "alarms": 0, "span_ms": 3900.0},
        {"warning_ms": [], "alarms": 5, "span_ms": 3900.0},
    ]
    # Window 390 ms of a 3900 ms span: one alarm hits with probability 0.1.
    one_alarm = 1.0 - 0.9**2
    expected = (2 * one_alarm + 1 * 0.0) / 3
    assert ev.chance_detection(records) == pytest.approx(expected)
    assert np.isnan(
        ev.chance_detection([{"warning_ms": [], "alarms": 1, "span_ms": 5}])
    )
    # An alarm every few ms over a span shorter than the window is a sure hit.
    short = [{"warning_ms": [None], "alarms": 1, "span_ms": 100.0}]
    assert ev.chance_detection(short) == 1.0


def test_shot_records_carry_the_alarm_count_and_scored_span():
    table, onsets = _table()
    traces = ev.shot_traces(table, np.arange(len(table), dtype=float))
    table = table.assign(
        score=np.arange(len(table), dtype=float), called=np.zeros(len(table), bool)
    )
    alarms = {
        shot: ev.score_alarms({shot: traces[shot]}, onsets, (1e9, 1e9, 0.0))[shot]
        for shot in traces
    }
    groups = ev.shot_records(table, alarms, onsets, onsets)
    record = groups["hanson"][0]
    assert record["alarms"] == 0
    assert record["span_ms"] == pytest.approx(600.0)  # last onset +100 minus start


def test_primary_fit_ignores_comparison_slices_and_their_feature_values():
    table, _ = _table()
    kwargs = {"seed": 3, "n_estimators": 10}
    first = ev.Brf(["betan_over_li"], **kwargs).fit(table).score(table)
    changed = table.copy()
    changed.loc[changed.role == "comparison", "betan_over_li"] = 1e6
    second = ev.Brf(["betan_over_li"], **kwargs).fit(changed).score(table)
    assert np.array_equal(first, second)


def test_comparison_scores_cannot_tune_primary_thresholds_or_alarm_rules():
    table, onsets = _table()
    changed = table.copy()
    changed.loc[changed.role == "comparison", "betan_over_li"] = 1e6

    def factory(seed):
        return ev.Rule("betan_over_li")

    _, _, original = ev.cross_validate(table, factory, onsets, outer=3, inner=2)
    _, _, modified = ev.cross_validate(changed, factory, onsets, outer=3, inner=2)
    assert original == modified


def test_alarm_targets_and_explanations_are_separate():
    t = np.arange(0.0, 310.0, 10.0)
    traces = {7: (t, (t == 100.0).astype(float))}
    targets = {7: [1000.0]}
    explanations = {7: [200.0, 1000.0]}  # n=2 can explain but not reward
    outcome = ev.score_alarms(
        traces, targets, (0.5, 0.5, 0.0), explanation_onsets=explanations
    )
    assert outcome[7]["alarms"] == [100.0]
    assert outcome[7]["warning_ms"] == [None]
    assert outcome[7]["false"] == []
    assert outcome[7]["early"] == []
    assert outcome[7]["category"] == "Missed"
    chosen = ev.choose_rule(
        traces, targets, np.array([0.1, 0.8]), explanation_onsets=explanations
    )
    assert chosen[2] == 0.0  # no spurious detection reward from n=2


def test_conditional_metrics_and_broad_sensitivity_use_their_own_masks():
    frame = pd.DataFrame(
        {
            "shot": [7] * 5,
            "role": ["hanson"] * 5,
            "campaign": [2014] * 5,
            "t_ms": [0.0, 100.0, 200.0, 300.0, 400.0],
            "label": [0, 0, 1, -1, -1],
            "label_broad": [0, 0, 1, -1, 0],
            "betan": [1.0, 4.0, 4.0, 0.0, 0.0],
            "betan_over_li": [1.0, 5.0, 6.0, 0.0, 0.0],
            "high_beta": [False, True, True, False, False],
            "above_proxy": [False, True, True, False, False],
            "score": [0.9, 0.8, 0.7, 0.1, 0.2],
            "called": [True] * 5,
        }
    )
    alarms = {7: {"warning_ms": [None], "false": [], "alarms": []}}
    groups = ev.shot_records(frame, alarms, {7: [250.0]}, {7: [250.0]})
    stats = ev.statistic(groups)
    assert stats["slice_auroc"] == 0.0
    assert stats["broad_auroc"] == pytest.approx(1 / 3)
    assert stats["high_beta_auroc"] == 0.0
    assert stats["above_proxy_auprc"] == 0.5
    sizes = ev.counts(groups)
    assert sizes["high_beta_positive_slices"] == 1
    assert sizes["high_beta_negative_slices"] == 1


def test_primary_alarms_stop_after_last_explanation_but_comparison_keeps_trace():
    t = np.arange(0.0, 1210.0, 10.0)
    score = np.isin(t, [50.0, 200.0, 700.0, 1000.0, 1100.0]).astype(float)
    traces = {7: (t, score), 8: (t, score)}
    targets = {7: [500.0]}
    explanations = {7: [500.0, 900.0]}
    roles = {7: "hanson", 8: "comparison"}
    primary = ev.score_alarms(
        traces,
        targets,
        (0.5, 0.5, 0.0),
        explanation_onsets=explanations,
        shot_roles=roles,
    )
    assert primary[7]["alarms"] == [50.0, 200.0, 700.0, 1000.0]
    assert primary[7]["ignored"] == [1100.0]
    assert primary[7]["early"] == [50.0]
    assert primary[7]["false"] == [50.0]
    assert primary[7]["category"] == "Detected"
    assert primary[7]["span_ms"] == 1000.0
    assert primary[8]["alarms"] == [50.0, 200.0, 700.0, 1000.0, 1100.0]
    assert primary[8]["ignored"] == [] and primary[8]["span_ms"] == 1200.0
    full = ev.score_alarms(
        traces,
        targets,
        (0.5, 0.5, 0.0),
        explanation_onsets=explanations,
        shot_roles=roles,
        alarm_scope="full",
    )
    assert full[7]["false"] == [50.0, 1100.0]
    assert full[7]["ignored"] == [] and full[7]["span_ms"] == 1200.0


def test_rule_tuning_ignores_primary_aftermath_and_retunes_full_trace():
    t = np.arange(0.0, 1210.0, 10.0)
    score = np.zeros(len(t))
    score[(t >= 500.0) & (t <= 510.0)] = 0.85
    score[(t >= 1100.0) & (t <= 1170.0)] = 0.85
    traces = {7: (t, score)}
    negatives = np.array([0.1] * 90 + [0.8] * 8 + [0.9] * 2)
    primary = ev.choose_rule(traces, {7: [600.0]}, negatives)
    full = ev.choose_rule(traces, {7: [600.0]}, negatives, alarm_scope="full")
    outcome = ev.score_alarms(traces, {7: [600.0]}, primary)
    assert outcome[7]["warning_ms"] == [100.0]
    assert outcome[7]["ignored"] == [1100.0]
    assert primary != full


def test_shot_categories_carry_early_ignored_and_multiple_target_counts():
    table, _ = _table(n_hanson=4, n_comparison=1)
    table["score"] = 0.0
    table["called"] = False
    targets = {0: [600.0, 800.0], 1: [600.0], 2: [600.0], 3: []}
    for shot, times in {
        0: [100.0, 400.0, 700.0],
        1: [100.0, 800.0],
        2: [600.0],
        3: [300.0],
        4: [800.0],
    }.items():
        table.loc[(table.shot == shot) & table.t_ms.isin(times), "score"] = 1.0
    explanations = {**targets, 3: [500.0]}
    outcomes = ev.score_alarms(
        ev.shot_traces(table, table.score.to_numpy()),
        targets,
        (0.5, 0.5, 0.0),
        explanation_onsets=explanations,
        shot_roles=table.drop_duplicates("shot").set_index("shot").role.to_dict(),
    )
    groups = ev.shot_records(table, outcomes, targets, explanations)
    assert [r["category"] for r in groups["hanson"]] == [
        "Detected",
        "Early",
        "Missed",
        "No target",
    ]
    assert groups["hanson"][0]["warning_ms"] == [200.0, 400.0]
    assert groups["hanson"][0]["early_alarms"] == 1
    assert groups["hanson"][1]["ignored_alarms"] == 1
    assert groups["hanson"][1]["span_ms"] == 600.0
    sizes = ev.counts(groups)
    assert sizes["hanson_target_shots"] == 3
    assert sizes["hanson_detected_shots"] == 1
    assert sizes["hanson_early_shots"] == 1
    assert sizes["hanson_missed_shots"] == 1
    assert sizes["hanson_no_target_shots"] == 1
    assert sizes["hanson_early_alarms"] == 2
    assert sizes["hanson_ignored_alarms"] == 1
    stats = ev.statistic(groups)
    assert stats["hanson_shot_detection_rate"] == pytest.approx(1 / 3)
    assert stats["hanson_shot_early_rate"] == pytest.approx(1 / 3)
    assert stats["hanson_shot_miss_rate"] == pytest.approx(1 / 3)
    assert stats["onset_detection_rate"] == 0.5
    assert stats["comparison_alarm_incidence"] == 1.0


def test_run_record_outer_holdout_keeps_siblings_out_of_training():
    table, onsets = _table(n_hanson=12, n_comparison=0)
    table["run_record"] = table.shot // 3
    seen = []

    class TracedRule(ev.Rule):
        def fit(self, train):
            self.train_shots = set(train.shot)
            return super().fit(train)

        def score(self, frame):
            seen.append((self.train_shots, set(frame.shot)))
            return super().score(frame)

    oof, _, rules = ev.cross_validate(
        table,
        lambda seed: TracedRule("betan_over_li"),
        onsets,
        outer_groups="run_record",
        inner=2,
    )
    assert len(rules) == 4 and len(oof) == len(table)
    assert oof.groupby("run_record").fold.nunique().eq(1).all()
    assert oof.groupby("fold").run_record.nunique().eq(1).all()
    for train, test in seen:
        assert not train & test
    # Each outer fit is the one scoring a complete three-shot run record.
    for train, test in seen:
        if len(test) == 3:
            assert not {s // 3 for s in train} & {s // 3 for s in test}
    mapped, _, _ = ev.cross_validate(
        table,
        lambda seed: ev.Rule("betan_over_li"),
        onsets,
        outer_groups={shot: shot // 3 for shot in range(12)},
        inner=2,
    )
    assert mapped.set_index(["shot", "t_ms"]).fold.equals(
        oof.set_index(["shot", "t_ms"]).fold
    )


@pytest.mark.parametrize("scope", ["primary", "full_trace"])
def test_saved_alarm_replay_matches_live_cv_without_changing_predictions(scope):
    table, onsets = _table(n_hanson=6, n_comparison=6)
    explanations = {shot: times + [650.0] for shot, times in onsets.items()}
    oof, alarms, rules = ev.cross_validate(
        table,
        lambda seed: ev.Rule("betan_over_li"),
        onsets,
        explanation_onsets=explanations,
        outer=3,
        inner=2,
        alarm_scope=scope,
    )
    original = oof.copy(deep=True)
    replay = ev.replay_alarms(
        oof,
        rules,
        onsets,
        explanation_onsets=explanations,
        alarm_scope=scope,
    )
    assert replay == alarms
    pd.testing.assert_frame_equal(oof, original)


def test_saved_alarm_replay_rejects_incomplete_rules_and_split_shots():
    table, onsets = _table(n_hanson=6, n_comparison=0)
    oof, _, rules = ev.cross_validate(
        table, lambda seed: ev.Rule("betan_over_li"), onsets, outer=3, inner=2
    )
    with pytest.raises(ValueError, match="exactly once"):
        ev.replay_alarms(oof, rules[:-1], onsets)
    with pytest.raises(ValueError, match="exactly once"):
        ev.replay_alarms(oof, rules + rules[:1], onsets)
    changed = oof.copy()
    changed.loc[changed.index[0], "fold"] = (int(changed.fold.iloc[0]) + 1) % 3
    with pytest.raises(ValueError, match="one fold per shot"):
        ev.replay_alarms(changed, rules, onsets)


def test_explicit_outer_shot_folds_cover_every_shot_once():
    table, onsets = _table(n_hanson=6, n_comparison=0)
    folds = [[0, 1], [2, 3], [4, 5]]
    oof, _, _ = ev.cross_validate(
        table,
        lambda seed: ev.Rule("betan_over_li"),
        onsets,
        outer_shot_folds=folds,
        inner=2,
    )
    assert oof.groupby("fold").shot.unique().apply(list).tolist() == folds
    with pytest.raises(ValueError, match="exactly once"):
        ev.cross_validate(
            table,
            lambda seed: ev.Rule("betan_over_li"),
            onsets,
            outer_shot_folds=[[0, 1], [1, 2, 3, 4, 5]],
        )
