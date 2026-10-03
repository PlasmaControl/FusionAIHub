"""The numpy balanced forest, the nnPU risk, the slice metrics and the alarm rule."""

from __future__ import annotations

import numpy as np
import pytest

from labeler.rwm import alarm, metrics
from labeler.rwm.forest import BalancedForest
from labeler.rwm.nnpu import NnPU


def _blobs(n_pos, n_neg, seed=0, separation=2.0):
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [rng.normal(separation, 1.0, (n_pos, 4)), rng.normal(0.0, 1.0, (n_neg, 4))]
    )
    return x, np.r_[np.ones(n_pos), np.zeros(n_neg)]


def test_the_forest_separates_two_blobs_at_a_hundred_to_one():
    x, y = _blobs(30, 3000)
    model = BalancedForest(n_estimators=60, max_depth=5, seed=1).fit(x, y)
    xt, yt = _blobs(200, 2000, seed=5)
    assert metrics.auroc(model.predict_proba(xt), yt) > 0.9
    p = model.predict_proba(xt)
    assert p.min() >= 0 and p.max() <= 1


def test_the_forest_is_reproducible_and_seeds_differ():
    x, y = _blobs(20, 400)
    a = BalancedForest(n_estimators=10, seed=3).fit(x, y).predict_proba(x)
    b = BalancedForest(n_estimators=10, seed=3).fit(x, y).predict_proba(x)
    c = BalancedForest(n_estimators=10, seed=4).fit(x, y).predict_proba(x)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_each_tree_trains_on_a_balanced_sample():
    x, y = _blobs(10, 500)
    model = BalancedForest(n_estimators=5, max_depth=0, seed=0).fit(x, y)
    # A depth-zero tree is one leaf holding the sample's positive fraction: 0.5.
    assert [t.value[0] for t in model.trees] == [0.5] * 5


def test_the_forest_needs_both_classes():
    with pytest.raises(ValueError):
        BalancedForest().fit(np.ones((4, 2)), np.ones(4))


def test_nnpu_ranks_positives_above_the_unlabelled_mixture():
    rng = np.random.default_rng(0)
    pos = rng.normal(2.0, 1.0, (200, 3))
    unlabelled = np.vstack(
        [rng.normal(0.0, 1.0, (1900, 3)), rng.normal(2.0, 1.0, (100, 3))]
    )
    x = np.vstack([pos, unlabelled])
    labelled = np.r_[np.ones(len(pos)), np.zeros(len(unlabelled))]
    model = NnPU(prior=0.05, hidden=(16,), epochs=120, batch_u=256, seed=0).fit(
        x, labelled
    )
    xt = np.vstack([rng.normal(2.0, 1.0, (300, 3)), rng.normal(0.0, 1.0, (300, 3))])
    yt = np.r_[np.ones(300), np.zeros(300)]
    assert metrics.auroc(model.predict_proba(xt), yt) > 0.85


def test_nnpu_validates_its_inputs():
    with pytest.raises(ValueError):
        NnPU(prior=0.0)
    with pytest.raises(ValueError):
        NnPU(prior=0.1).fit(np.ones((5, 2)), np.zeros(5))


def test_auroc_counts_ties_as_half():
    assert metrics.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert metrics.auroc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert np.isnan(metrics.auroc([1, 2], [1, 1]))


def test_auprc_matches_a_hand_computation():
    # Ranked: pos, neg, pos -> precision 1 at recall 1/2, 2/3 at recall 1.
    assert metrics.auprc([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx(
        0.5 * 1 + 0.5 * 2 / 3
    )
    assert metrics.auprc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_roc_cutoff_and_confusion():
    score = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    label = np.array([0, 0, 0, 0, 1, 1])
    cutoff = metrics.roc_cutoff(score, label)
    assert cutoff == 0.7
    tpr, fpr, precision, f1 = metrics.confusion(score, label, cutoff)
    assert (tpr, fpr, precision, f1) == (1.0, 0.0, 1.0, 1.0)


def test_shot_bootstrap_is_stratified_and_brackets_the_estimate():
    groups = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0]}

    def mean_of_means(sample):
        return np.mean([np.mean(sample["a"]), np.mean(sample["b"])])

    out = metrics.shot_bootstrap(groups, mean_of_means, replicates=200, seed=1)
    assert out["estimate"] == pytest.approx(8.75)
    assert out["low"] <= out["estimate"] <= out["high"]
    sizes = metrics.shot_bootstrap(
        groups, lambda s: {"na": len(s["a"]), "nb": len(s["b"])}, replicates=20, seed=0
    )
    assert sizes["na"]["low"] == sizes["na"]["high"] == 4


def test_hysteresis_needs_the_high_crossing_and_the_hold():
    t = np.arange(0.0, 200.0, 10.0)
    score = np.zeros_like(t)
    score[5:12] = 0.5  # above k_low only: no alarm
    assert alarm.hysteresis_alarms(t, score, 0.4, 0.8, 30.0) == []
    score[8] = 0.9  # crosses k_high at t=80, stays above k_low
    assert alarm.hysteresis_alarms(t, score, 0.4, 0.8, 30.0) == [110.0]
    # Dropping below k_low before the hold ends cancels it, and an alarm latches.
    score[10] = 0.0
    assert alarm.hysteresis_alarms(t, score, 0.4, 0.8, 30.0) == []
    score[:] = 0.9
    assert alarm.hysteresis_alarms(t, score, 0.4, 0.8, 30.0) == [30.0]


def test_hysteresis_resets_across_a_gap_and_a_nan():
    t = np.array([0.0, 10.0, 20.0, 200.0, 210.0, 220.0, 230.0])
    score = np.array([0.9, 0.9, 0.9, 0.9, 0.9, np.nan, 0.9])
    out = alarm.hysteresis_alarms(t, score, 0.4, 0.8, 20.0, max_gap_ms=50.0)
    assert out == [20.0]  # the second run is cut by the gap and then by the NaN


def test_match_alarms_applies_the_papers_400_and_10_ms_limits():
    per_onset, unmatched = alarm.match_alarms([50.0, 700.0, 995.0, 1500.0], [1000.0])
    assert per_onset == [pytest.approx(300.0)]  # 700 -> a 300 ms warning
    assert unmatched == [50.0, 1500.0]  # 995 is too late but belongs to the onset
    per_onset, unmatched = alarm.match_alarms([500.0], [1000.0])
    assert per_onset == [None] and unmatched == [500.0]  # 500 ms early: too early
    per_onset, _ = alarm.match_alarms([], [1000.0, 2000.0])
    assert per_onset == [None, None]
