"""Per-bin scoring of tearing-mode detectors on cases with one answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labeler.tearing import scoring


def rows(*items):
    return pd.DataFrame(items, columns=["category", "t_start", "t_end"])


def test_folds_deal_every_shot_once_and_evenly_and_the_same_way_each_time():
    shots = list(range(1000, 1050))
    a = scoring.shot_folds(shots, 5, seed=3)
    assert a == scoring.shot_folds(shots[::-1], 5, seed=3)
    assert sorted(a) == shots
    assert sorted(np.bincount(list(a.values()))) == [10] * 5
    assert a != scoring.shot_folds(shots, 5, seed=4)


def test_bin_centres_lie_wholly_inside_the_window_on_an_absolute_grid():
    c = scoring.bin_centres((1003.0, 1052.0))
    assert c.tolist() == [1015.0, 1025.0, 1035.0, 1045.0]
    assert scoring.bin_centres((1000.0, 1004.0)).size == 0


def test_bins_in_a_present_span_are_positive_and_ignored_ones_left_out():
    table = rows(
        (1, 100.0, 200.0),
        (1, 150.0, 150.0),
        (2, 300.0, 330.0),
        (3, 500.0, 600.0),
        (0, 0.0, 100.0),
    )
    c = scoring.bin_centres((0.0, 700.0))
    y, valid = scoring.label_bins(table, c)
    assert y[(c > 100) & (c < 200)].all() and not y[c < 100].any()
    assert not y[(c > 300) & (c < 330)].any()
    assert not valid[(c > 300) & (c < 330)].any()
    assert not valid[(c > 500) & (c < 600)].any()
    assert valid[(c < 300) | ((c > 340) & (c < 500))].all()


def test_scores_are_interpolated_with_the_shift_and_absent_across_a_hole():
    t = np.array([0.0, 25.0, 50.0, 75.0, 200.0, 225.0])
    v = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    c = np.array([5.0, 15.0, 30.0, 100.0, 210.0, 300.0])
    out = scoring.align_scores(t, v, c)
    assert out[0] == pytest.approx(0.2) and out[1] == pytest.approx(0.6)
    assert np.isnan(out[3]) and np.isnan(out[5])
    assert out[4] == pytest.approx(0.4)
    shifted = scoring.align_scores(t, v, c, shift_ms=25.0)
    assert np.isnan(shifted[0]) and np.isnan(shifted[1])
    assert shifted[2] == pytest.approx(0.2)


def edges_of(values):
    u = np.unique(values)
    return np.concatenate(([u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]))


def test_auroc_and_auprc_from_histograms_match_a_direct_count():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.2
    s = rng.normal(size=400) + 1.2 * y
    edges = edges_of(s)
    pos, neg = np.histogram(s[y], edges)[0], np.histogram(s[~y], edges)[0]
    auroc, auprc = scoring.auroc_auprc(pos, neg)
    sp, sn = s[y], s[~y]
    pairs = (sp[:, None] > sn[None, :]).mean() + 0.5 * (
        sp[:, None] == sn[None, :]
    ).mean()
    assert auroc == pytest.approx(pairs, abs=1e-12)
    order = np.argsort(-s)
    hits = np.cumsum(y[order])
    precision_at_hits = hits[y[order]] / (np.flatnonzero(y[order]) + 1)
    assert auprc == pytest.approx(precision_at_hits.mean(), abs=1e-12)


def test_ties_count_half_in_the_auroc():
    pos, neg = np.array([0, 4]), np.array([0, 4])
    assert scoring.auroc_auprc(pos, neg)[0] == pytest.approx(0.5)
    assert np.isnan(scoring.auroc_auprc([0, 0], [1, 1])[0])


def test_segments_merge_short_gaps_and_drop_short_runs():
    m = np.zeros(60, dtype=bool)
    m[5:15] = True  # 100 ms
    m[18:25] = True  # a 30 ms gap from the run before: merged
    m[40:43] = True  # 30 ms: dropped
    assert scoring.segments(m) == [(5, 25)]


def test_segments_never_close_an_unavailable_bin():
    mask = np.ones(20, dtype=bool)
    valid = np.ones(20, dtype=bool)
    valid[9:11] = False
    assert scoring.segments(mask, valid=valid) == [(0, 9), (11, 20)]


def test_segment_iou_excludes_unavailable_time_and_uses_hard_boundaries():
    true = np.ones(20, dtype=bool)
    pred = true.copy()
    valid = true.copy()
    valid[9:11] = False
    # Missing time is neither a negative gap nor part of either segment's IoU.
    assert scoring.segment_counts(pred, true, 1.0, valid=valid) == (2, 0, 0)
    pred[11:20] = False
    assert scoring.segment_counts(pred, true, 1.0, valid=valid) == (1, 0, 1)


def test_shot_stats_preserves_missing_score_and_label_barriers():
    y = np.ones(20, dtype=np.int8)
    valid = np.ones(20, dtype=bool)
    score = np.full(20, 0.9)
    score[9] = np.nan
    valid[10] = False
    stats = scoring.shot_stats(1, y, valid, score, [0, 0.5, 1], 0.5)
    assert stats.seg[0.5] == (2, 0, 0)


def test_segment_counts_by_temporal_iou():
    true = np.zeros(100, dtype=bool)
    true[10:30] = True
    true[60:80] = True
    pred = np.zeros(100, dtype=bool)
    pred[10:30] = True  # exact
    pred[70:90] = True  # IoU 1/3 with the second true segment
    assert scoring.segment_counts(pred, true, 0.3) == (2, 0, 0)
    assert scoring.segment_counts(pred, true, 0.5) == (1, 1, 1)
    assert scoring.segment_counts(np.zeros(100, dtype=bool), true, 0.5) == (0, 0, 2)


def stats_for(shot, y, s, edges, thr=0.5):
    y = np.asarray(y)
    return scoring.shot_stats(
        shot, y, np.ones(len(y), bool), np.asarray(s, float), edges, thr
    )


def test_a_perfect_detector_scores_one_everywhere_with_a_degenerate_interval():
    y = np.r_[np.zeros(40), np.ones(30), np.zeros(30)]
    s = y * 0.8 + 0.1
    edges = np.array([0.0, 0.5, 1.0])
    out = scoring.bootstrap([stats_for(k, y, s, edges) for k in range(8)], n=50)
    for key in ("auroc", "auprc", "f1", "segf1_0.5"):
        assert out[key]["value"] == out[key]["lo"] == out[key]["hi"] == 1.0
    assert out["n_shots"] == 8 and out["replicates"] == 50


def test_ignored_and_unscored_bins_count_nowhere():
    y = np.array([1, 1, 0, 0])
    valid = np.array([True, False, True, True])
    s = np.array([0.9, 0.9, np.nan, 0.1])
    st = scoring.shot_stats(1, y, valid, s, np.array([0.0, 0.5, 1.0]), 0.5)
    assert (st.n_pos, st.n_neg, st.tp, st.fp, st.fn) == (1, 1, 1, 0, 0)


def test_the_interval_around_a_noisy_detector_covers_its_point_and_is_seeded():
    rng = np.random.default_rng(1)
    edges = np.linspace(0, 1, 21)
    shots = []
    for k in range(30):
        y = (rng.random(200) < 0.3).astype(int)
        s = np.clip(0.5 * y + rng.random(200) * 0.7, 0, 1)
        shots.append(stats_for(k, y, s, edges, 0.6))
    a = scoring.bootstrap(shots, n=200, seed=2)
    b = scoring.bootstrap(shots, n=200, seed=2)
    assert a == b
    for key in ("auroc", "auprc", "f1"):
        assert a[key]["lo"] <= a[key]["value"] <= a[key]["hi"]
        assert a[key]["hi"] - a[key]["lo"] > 0


def test_the_best_threshold_separates_two_clean_classes():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.r_[np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)]
    edges = np.linspace(0, 1, 101)
    st = scoring.shot_stats(0, y, np.ones(100, bool), s, edges)
    thr = scoring.best_threshold([st], edges)
    assert 0.4 < thr <= 0.6


def test_evaluate_leaves_out_a_shot_with_no_scored_bin_and_takes_per_shot_thresholds():
    rng = np.random.default_rng(3)
    y, valid, score = {}, {}, {}
    for k in range(6):
        y[k] = (rng.random(100) < 0.3).astype(int)
        valid[k] = np.ones(100, bool)
        score[k] = 0.2 + 0.6 * y[k] + 0.05 * rng.random(100)
    score[6], y[6], valid[6] = np.full(10, np.nan), np.zeros(10, int), np.ones(10, bool)
    out = scoring.evaluate(
        list(range(7)), y, valid, score, {k: 0.5 for k in range(7)}, n=20
    )
    assert out["shots_without_a_scored_bin"] == [6]
    assert out["n_shots"] == 6 and out["bins_scored"] == 600
    assert out["auroc"]["value"] == 1.0 and out["f1"]["value"] == 1.0


def test_evaluate_without_tious_has_no_segmental_f1_and_records_the_bin_width():
    rng = np.random.default_rng(4)
    y = {k: (rng.random(80) < 0.3).astype(int) for k in range(4)}
    valid = {k: np.ones(80, bool) for k in range(4)}
    score = {k: 0.2 + 0.6 * y[k] for k in range(4)}
    full = scoring.evaluate(list(range(4)), y, valid, score, 0.5, n=10)
    rows = scoring.evaluate(
        list(range(4)), y, valid, score, 0.5, n=10, tious=(), bin_ms=25.0
    )
    assert "segf1_0.5" in full and full["bin_ms"] == scoring.BIN_MS
    assert not any(key.startswith("segf1") for key in rows)
    assert rows["bin_ms"] == 25.0
    assert rows["f1"]["value"] == full["f1"]["value"] == 1.0


def test_inner_split_holds_back_a_seeded_tenth_and_loses_no_shot():
    shots = list(range(100, 150))
    train, val = scoring.inner_split(shots, 7)
    assert len(val) == 5 and sorted(train + val) == shots
    assert (train, val) == scoring.inner_split(shots, 7)
    assert val != scoring.inner_split(shots, 8)[1]
    assert len(scoring.inner_split([1, 2], 0)[1]) == 1


def test_shared_cv_uses_complete_cohort_before_model_availability():
    shots = list(range(100, 150))
    folds, splits = scoring.shared_cv(shots)
    assert folds == scoring.shot_folds(shots)
    for split in splits:
        held = [s for s in shots if folds[s] == split["fold"]]
        pool = [s for s in shots if folds[s] != split["fold"]]
        train, val = scoring.inner_split(pool, 100 + split["fold"])
        assert split == {
            "fold": split["fold"],
            "train": train,
            "validation": val,
            "held": held,
        }
        assert not set(held) & (set(train) | set(val))


def test_cv_thresholds_never_use_held_labels_and_intersect_availability():
    shots = list(range(100, 150))
    folds, splits = scoring.shared_cv(shots)
    score = {s: np.array([0.1, 0.8]) for s in shots[:-1]}
    y = {s: np.array([0, 1]) for s in score}
    valid = {s: np.ones(2, dtype=bool) for s in score}
    thresholds, info = scoring.cv_thresholds(shots, y, valid, score)
    assert set(thresholds) == set(score)
    for split, record in zip(splits, info, strict=True):
        assert record["validation"] == split["validation"]
        assert record["validation_scored"] == [
            s for s in split["validation"] if s in score
        ]
        assert 0.1 < record["threshold"] <= 0.8
    # Change one held fold's targets. Its own threshold must stay unchanged.
    changed = dict(y)
    for s in shots:
        if folds[s] == 0 and s in changed:
            changed[s] = 1 - changed[s]
    other, _ = scoring.cv_thresholds(shots, changed, valid, score)
    for s in thresholds:
        if folds[s] == 0:
            assert other[s] == thresholds[s]
