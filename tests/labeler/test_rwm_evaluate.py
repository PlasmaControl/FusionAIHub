"""Nested shot-grouped scoring of the RWM baselines on synthetic slice tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeler.rwm import alarm, features, labels
from labeler.rwm import evaluate as ev


@pytest.mark.parametrize(
    ("alarms", "targets", "explanations", "category", "any_category"),
    [
        ([100.0, 400.0], [600.0], [600.0], "Early", "Detected"),
        ([600.0, 900.0], [1000.0], [600.0, 1000.0], "Detected", "Detected"),
        ([300.0, 900.0], [1000.0], [300.0, 1000.0], "Missed", "Detected"),
        ([600.0, 900.0], [600.0, 1000.0], [600.0, 1000.0], "Detected", "Detected"),
        ([590.0], [600.0], [600.0], "Detected", "Detected"),
        ([591.0], [600.0], [600.0], "Missed", "Missed"),
        ([701.0], [600.0], [600.0], "Missed", "Missed"),
        ([100.0], [], [600.0], "No target", "No target"),
        ([], [600.0], [600.0], "Missed", "Missed"),
    ],
)
def test_first_considered_alarm_decides_shot_category(
    alarms, targets, explanations, category, any_category
):
    outcome = alarm.shot_outcome(
        alarms[::-1], targets, explanations, ignore_after_ms=max(explanations) + 100
    )
    assert outcome["category"] == category
    assert outcome["any_alarm_category"] == any_category
    if alarms == [100.0, 400.0]:
        assert outcome["warning_ms"] == [200.0]
        assert outcome["early"] == [100.0]


def test_within_shot_auroc_separates_masks_and_weights_shots_equally():
    table = pd.DataFrame(
        {
            "shot": [1] * 4 + [2] * 2 + [3] + [4] * 2,
            "role": ["hanson"] * 7 + ["comparison"] * 2,
            "label": [0, 0, 1, -1, 0, 1, 0, 0, 1],
            "label_broad": [0, 0, 1, 0, 0, 1, 0, 0, 1],
            "score": [0, 1, 2, 3, 1, 0, 0, 0, 1],
        }
    )
    result = ev.within_shot_auroc(table)
    primary, broad = result["primary"], result["broad"]
    assert primary["n_shots"] == 2
    assert primary["mean"] == primary["median"] == 0.5
    assert [r["auroc"] for r in primary["per_shot"]] == [1.0, 0.0, None]
    assert primary["per_shot"][0]["n_negative"] == 2
    assert broad["per_shot"][0]["auroc"] == pytest.approx(2 / 3)
    assert broad["mean"] == pytest.approx(1 / 3)
    empty = ev.within_shot_auroc(table[table.shot == 3])["primary"]
    assert empty["n_shots"] == 0
    assert empty["mean"] is None and empty["median"] is None


def test_phase_controlled_auroc_excludes_cross_bin_and_campaign_pairs():
    groups = {
        "hanson": [
            {
                "shot": 1,
                "campaign": 2014,
                "score": [0, 0, 0, 4, 3, 2, 1, 100],
                "label": [1, 1, 0, 1, 0, 0, 0, -1],
                "elapsed_time_ms": [0, 1, 199, 200, 201, 202, 399, 0],
            },
            {
                "shot": 2,
                "campaign": 2018,
                "score": [1000],
                "label": [1],
                "elapsed_time_ms": [0],
            },
            {
                "shot": 3,
                "campaign": 2014,
                "score": [-1000, -1000],
                "label": [0, 0],
                "elapsed_time_ms": [400, np.nan],
            },
        ],
        "comparison": [
            {
                "shot": 4,
                "campaign": 2014,
                "score": [1000],
                "label": [1],
                "elapsed_time_ms": [0],
            }
        ],
    }
    # Bin 0 has two tied pairs; bin 1 has three concordant pairs: (1+3)/5.
    assert ev.phase_controlled_auroc(groups, bin_ms=200, min_slices=1) == 0.8
    assert np.isnan(ev.phase_controlled_auroc({"hanson": groups["hanson"][1:]}))
    interval = ev.phase_controlled_bootstrap(
        groups, bin_ms=200, min_slices=1, replicates=20
    )
    assert interval["estimate"] == interval["low"] == interval["high"] == 0.8
    paired = ev.phase_controlled_bootstrap(
        groups, groups, bin_ms=200, min_slices=1, replicates=20
    )
    assert paired == {"estimate": 0.0, "low": 0.0, "high": 0.0}


def test_phase_default_uses_100_ms_and_bootstrap_keeps_the_same_bins():
    groups = {
        "hanson": [
            {
                "shot": 1,
                "campaign": 2014,
                "score": list(range(10)),
                "label": [1, 0, 0, 0, 0] * 2,
                "elapsed_time_ms": [0, 10, 20, 30, 40, 100, 110, 120, 130, 140],
            }
        ]
    }
    assert ev.phase_controlled_auroc(groups) == 0.0
    assert ev.phase_controlled_auroc(groups, bin_ms=200) == 0.25
    assert ev.phase_controlled_auroc(groups, bin_ms=None) == 0.25
    for bin_ms, expected in ((100, 0.0), (200, 0.25)):
        result = ev.phase_controlled_bootstrap(groups, bin_ms=bin_ms, replicates=20)
        assert result == {"estimate": expected, "low": expected, "high": expected}


@pytest.mark.parametrize(("scores", "expected"), [([2, 1], 2 / 7), ([2, 2], 1 / 7)])
def test_phase_resamples_exclude_pairs_between_copies_of_the_same_shot(
    scores, expected
):
    first = {
        "shot": 1,
        "campaign": 2014,
        "score": scores,
        "label": [1, 0],
        "elapsed_time_ms": [10, 20],
    }
    second = {**first, "shot": 2, "score": [0, 3]}
    # Two copies of shot 1: two within-copy and five other eligible pairs.
    assert ev.phase_controlled_auroc(
        {"hanson": [first, first, second]}, min_slices=1
    ) == pytest.approx(expected)
    assert ev.phase_controlled_auroc({"hanson": [first, first]}, min_slices=1) == (
        1.0 if scores == [2, 1] else 0.5
    )


def test_first_onset_mask_excludes_inter_onset_negatives_and_repeat_positives():
    table = pd.DataFrame(
        {
            "shot": [1] * 7 + [2],
            "role": ["hanson"] * 7 + ["comparison"],
            "t_ms": [0, 90, 100, 150, 200, 290, 300, 0],
            "label": [0, 1, -1, -1, 0, 1, -1, -2],
        }
    )
    result = ev.first_onset_mask(table, {1: [300, 100]})
    assert result.tolist() == [True, True, False, False, False, False, False, False]


def test_phase_control_drops_bins_with_fewer_than_five_eligible_slices():
    groups = {
        "hanson": [
            {
                "shot": 1,
                "campaign": 2014,
                "score": [0, 1, 2, 3, 4, 5],
                "label": [1, 0, 0, 0, -1, 0],
                "elapsed_time_ms": [0, 10, 20, 30, 40, np.nan],
            }
        ]
    }
    assert np.isnan(ev.phase_controlled_auroc(groups))
    groups["hanson"][0]["label"][4] = 0
    assert ev.phase_controlled_auroc(groups) == 0.0


def test_phase_paired_bootstrap_rejects_misaligned_shots_and_bins():
    first = {
        "hanson": [
            {
                "shot": 1,
                "campaign": 2014,
                "score": [1, 0],
                "label": [1, 0],
                "elapsed_time_ms": [0, 199],
            }
        ]
    }
    r = first["hanson"][0]
    for replacement in ({"shot": 2}, {"elapsed_time_ms": [0, 200]}, {"label": [0, 1]}):
        second = {"hanson": [{**r, **replacement}]}
        with pytest.raises(ValueError, match="aligned"):
            ev.phase_controlled_bootstrap(first, second, replicates=20)


def test_campaign_pairs_keep_shot_draws_inside_each_campaign():
    table, onsets = _table(n_hanson=4, n_comparison=0)
    # Give the two campaigns opposite ranking errors; pooling hides the failure.
    table["score"] = np.where(table.campaign == 2014, -table.label, table.label)
    table["called"] = False
    alarms = ev.score_alarms(
        ev.shot_traces(table, table.score.to_numpy()), onsets, (100, 100, 0)
    )
    first = ev.shot_records(table, alarms, onsets)
    second = ev.shot_records(table.assign(score=table.label), alarms, onsets)
    result = ev.paired_time_by_campaign(first, second, replicates=20)
    assert result["2014"]["slice_auroc"]["estimate"] == -1.0
    assert result["2018"]["slice_auroc"]["estimate"] == 0.0
    second["hanson"].reverse()
    with pytest.raises(ValueError, match="shot order"):
        ev.paired_time_by_campaign(first, second, replicates=20)


def test_paired_within_shot_intervals_use_shared_shots_and_separate_masks():
    first = {
        "hanson": [
            {
                "shot": 1,
                "score": [0, 1, 2],
                "label": [0, 1, -1],
                "label_broad": [0, 1, 0],
            },
            {"shot": 2, "score": [0, 1], "label": [0, 1], "label_broad": [0, 1]},
            {"shot": 3, "score": [0, 1], "label": [0, -1], "label_broad": [0, 1]},
        ]
    }
    second = {
        "hanson": [{**r, "score": [0] * len(r["score"])} for r in first["hanson"]]
    }
    result = ev.paired_within_shot_auroc(first, second, replicates=1000, seed=0)
    assert result["primary"] == {
        "n_shots": 2,
        "estimate": 0.5,
        "low": 0.5,
        "high": 0.5,
    }
    assert result["broad"]["n_shots"] == 3
    assert result["broad"]["estimate"] == pytest.approx(1 / 3)
    assert result["broad"]["low"] == pytest.approx(1 / 6)
    assert result["broad"]["high"] == pytest.approx(2 / 3)
    identical = ev.paired_within_shot_auroc(first, first, replicates=1000)
    for row in identical.values():
        assert row["estimate"] == row["low"] == row["high"] == 0.0
    second["hanson"].reverse()
    with pytest.raises(ValueError, match="shot order"):
        ev.paired_within_shot_auroc(first, second)


def test_onset_physics_separates_window_snapshot_from_actual_onset():
    table = pd.DataFrame(
        {
            "shot": [1, 1, 1],
            "campaign": [2014] * 3,
            "t_ms": [80.0, 90.0, 100.0],
            "betan": [3.0, 5.0, 5.0],
            "li": [1.0] * 3,
            "betan_over_li": [3.0, 5.0, 5.0],
            features.TIME_COLUMN: [30.0, 40.0, 50.0],
            "high_beta": [False] * 3,
        }
    )
    signals = {
        1: {
            "betan": (np.array([80.0, 95.0]), np.array([3.0, 5.0])),
            "li": (np.array([80.0, 95.0]), np.array([1.0, 1.0])),
            "ip": (np.array([50.0, 100.0]), np.array([0.6e6, 0.6e6])),
        }
    }
    result = ev.onset_physics(table, {1: [100.0]}, signals)
    row = result["rows"][0]
    assert row["sample_ms"] == 80.0
    assert row["betan_over_li"] == 3.0 and row["below_proxy"]
    assert row["onset_betan_over_li"] == 5.0 and not row["onset_below_proxy"]
    assert row["onset_elapsed_time_ms"] == 50.0
    assert row["n_high_beta_pre_onset_slices"] == 0
    assert result["by_campaign"]["2014"]["no_high_beta_pre_onset_fraction"] == 1.0
    signals[1]["betan"] = (np.array([0.0]), np.array([5.0]))
    missing = ev.onset_physics(table, {1: [100.0]}, signals)["rows"][0]
    assert missing["onset_efit_missing"] and not missing["onset_below_proxy"]


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
    assert record["span_start_ms"] == 100.0
    assert record["span_end_ms"] == 700.0
    assert record["target_onsets_ms"] == [600.0]


def test_random_alarm_reference_clips_each_warning_window_to_scored_span():
    records = [
        {
            "warning_ms": [None, None],
            "target_onsets_ms": [1598.05, 1990.0],
            "alarms": 1,
            "span_ms": 800.0,
            "span_start_ms": 1210.0,
            "span_end_ms": 2010.0,
        }
    ]
    # First warning interval [1198.05, 1588.05] has only 378.05 ms in coverage.
    assert ev.chance_detection(records) == pytest.approx((378.05 + 390) / 1600)


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
    assert primary[7]["category"] == "Early"
    assert primary[7]["any_alarm_category"] == "Detected"
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
        "Early",
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
    assert sizes["hanson_detected_shots"] == 0
    assert sizes["hanson_early_shots"] == 2
    assert sizes["hanson_any_alarm_detected_shots"] == 1
    assert sizes["hanson_any_alarm_early_shots"] == 1
    assert sizes["hanson_missed_shots"] == 1
    assert sizes["hanson_no_target_shots"] == 1
    assert sizes["hanson_early_alarms"] == 2
    assert sizes["hanson_ignored_alarms"] == 1
    stats = ev.statistic(groups)
    assert stats["hanson_shot_detection_rate"] == 0.0
    assert stats["hanson_shot_early_rate"] == pytest.approx(2 / 3)
    assert stats["hanson_any_alarm_shot_detection_rate"] == pytest.approx(1 / 3)
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
