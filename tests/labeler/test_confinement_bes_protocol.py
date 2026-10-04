"""The protocol pieces of the BES confinement benchmark: gating, margins, splits,
scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labeler.confinement import bes_protocol as bp


def _table(**cols):
    base = {
        "shot": [1, 1, 1, 1],
        "label": [0, 1, 1, 3],
        "start_ms": [100.0, 500.0, 510.0, 900.0],
        "dt_ms": [0.002] * 4,
        "t_start": [0.0, 400.0, 400.0, 800.0],
        "t_end": [300.0, 700.0, 700.0, 1000.0],
        "prev_label": [np.nan, 0.0, 0.0, np.nan],
        "gap_ms": [np.nan, 0.0, 0.0, np.nan],
        "p15L_min": [2e6, 2e6, 5e5, 5e5],
        "p15R_max": [0.0, 0.0, 0.0, 0.0],
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_gate_keeps_wp_windows_at_400_kw_in_training_only():
    train, score = bp.window_masks(_table(), gate=True)
    assert train.tolist() == [True, True, False, True]
    assert score.tolist() == [True, True, False, False]


def test_right_beam_blocks_a_window():
    train, score = bp.window_masks(_table(p15R_max=[0.0, 3e5, 0.0, 0.0]), gate=True)
    assert not train[1] and not score[1]


def test_transition_margin_and_buildup_after_l_mode():
    _, score = bp.window_masks(_table(), gate=False, transition_ms=20.0)
    # window 0 starts 100 ms into its interval and ends 190 ms before its end: kept
    assert score.tolist() == [True, True, True, True]
    table = _table(start_ms=[5.0, 410.0, 590.0, 900.0])
    _, score = bp.window_masks(table, gate=False, transition_ms=20.0, buildup_ms=100.0)
    # 5 ms from the start, 10 ms into an interval that follows L-mode, 5.9 ms from the
    # end
    assert score.tolist() == [False, False, True, True]


def test_deal_spreads_a_small_stratum_over_the_groups():
    signature = {s: "a" if s < 10 else "b" for s in range(14)}
    groups = bp.deal(list(range(14)), signature, 5, np.random.default_rng(0))
    assert set(groups.values()) == {0, 1, 2, 3, 4}
    assert len({groups[s] for s in range(10, 14)}) == 4


def test_cv_roles_never_split_a_shot():
    rng = np.random.default_rng(1)
    table = pd.DataFrame(
        {"shot": np.repeat(np.arange(40), 5), "label": rng.integers(0, 4, 200)}
    )
    seen = np.zeros(40, dtype=int)
    for fold in range(5):
        roles = bp.cv_roles(table, fold)
        per_shot = pd.Series(roles).groupby(table.shot).nunique()
        assert (per_shot == 1).all()
        seen += (pd.Series(roles).groupby(table.shot).first() == 2).to_numpy()
    assert (seen == 1).all()


def test_paper_roles_proportions_and_exclusivity():
    shots = np.arange(80)
    table = pd.DataFrame({"shot": np.repeat(shots, 3), "label": 0})
    roles = bp.paper_roles(table, {int(s): int(s % 2) for s in shots}, seed=3)
    by_shot = pd.Series(roles).groupby(table.shot).first()
    assert abs((by_shot == 2).mean() - 0.125) < 0.04
    assert abs((by_shot == 1).mean() - 0.15) < 0.05


def test_perfect_prediction_scores_one_with_a_tight_interval():
    truth = np.tile(np.arange(4), 25)
    shots = np.repeat(np.arange(10), 10)
    out = bp.summarise(truth, truth, shots, replicates=50)
    assert out["macro_f1"] == 1.0
    assert out["ci95"]["macro_f1"] == [1.0, 1.0]
    assert out["shots"] == 10


def test_macro_f1_ignores_a_class_with_no_windows():
    conf = np.diag([5, 5, 0, 0])
    assert bp.macro_f1(conf) == 1.0


def test_live_shots_flags_a_dead_block():
    power = np.ones((8, 4), dtype=np.float32)
    power[4:, :3] = 0.0
    table = pd.DataFrame({"shot": [1] * 4 + [2] * 4})
    assert bp.live_shots(table, power, slice(0, 4)) == {1}


def test_paired_difference_is_positive_when_a_beats_b_on_every_shot():
    good = np.tile(np.diag([9, 9, 9, 9]), (12, 1, 1))
    bad = np.tile(
        np.array([[5, 4, 0, 0], [4, 5, 0, 0], [0, 0, 5, 4], [0, 0, 4, 5]]), (12, 1, 1)
    )
    out = bp.paired_difference(good, bad, replicates=50)
    assert out["difference"] > 0.3
    assert out["ci95"][0] > 0
    assert out["share_not_above_zero"] == 0.0


def test_block_roles_split_inside_a_shot_in_runs():
    table = pd.DataFrame({"shot": np.repeat([1, 2], 400)})
    roles = bp.block_roles(table, seed=1, block=50)
    runs = roles.reshape(2, 8, 50)
    assert (runs == runs[:, :, :1]).all()  # one role per block of 50 windows
    assert set(np.unique(roles)) <= {0, 1, 2}
    # at least one of the two shots has windows in more than one role
    assert any(len(set(roles[table.shot == s])) > 1 for s in (1, 2))


def test_reweighting_to_a_class_mix_keeps_the_per_class_recall():
    conf = np.array(
        [[90, 10, 0, 0], [5, 95, 0, 0], [0, 0, 50, 50], [0, 0, 0, 100]], dtype=float
    )
    # 100 windows per true class; ask for a mix with 3x as many class-0 windows
    re = bp.reweight_to_mix(conf, mix=(3.0, 1.0, 1.0, 1.0))
    assert np.isclose(re.sum(), conf.sum())
    assert np.isclose(re[0].sum() / re.sum(), 0.5)
    np.testing.assert_allclose(re[0] / re[0].sum(), conf[0] / conf[0].sum())
    # the same mix as the support leaves the score unchanged
    same = bp.reweight_to_mix(conf, mix=(1.0, 1.0, 1.0, 1.0))
    assert np.isclose(bp.macro_f1(same), bp.macro_f1(conf))


def test_run_day_groups_follow_the_local_day_and_inherit_for_undated_shots():
    dates = pd.DataFrame(
        {
            "shot": [10, 11, 12, 13, 14, 15],
            "inserted_local": [
                "2021-03-01T15:00:00",
                "2021-03-01T16:30:00",
                "",
                "2021-03-02T14:00:00",
                "2021-09-09T09:00:00",  # EFIT run months later: untrusted
                "2021-03-02T18:00:00",
            ],
            "consistent": [True, True, False, True, False, True],
        }
    )
    got = bp.run_day_groups(dates)
    assert got[10] == got[11] == got[12]  # 12 has no date: its predecessor's day
    assert got[13] == got[14] == got[15]
    assert got[10] != got[13]


def test_deal_groups_keeps_a_group_whole_and_spreads_the_rare_class():
    rng = np.random.default_rng(0)
    shots = list(range(60))
    group_of = {s: s // 3 for s in shots}  # 20 run days of 3 shots
    # WP on one shot of every fifth day; QH on every second day; L and H everywhere
    signature = {
        s: "L,H,"
        + ("QH," if (s // 3) % 2 == 0 else "")
        + ("WP," if (s // 3) % 5 == 0 and s % 3 == 0 else "")
        for s in shots
    }
    fold_of = bp.deal_groups(shots, group_of, signature, 4, rng)
    for g in set(group_of.values()):
        assert len({fold_of[s] for s in shots if group_of[s] == g}) == 1
    for cls in ("WP", "QH"):
        folds = {fold_of[s] for s in shots if cls in signature[s]}
        assert folds == {0, 1, 2, 3}, cls
    sizes = np.bincount([fold_of[s] for s in shots], minlength=4)
    assert sizes.max() - sizes.min() <= 6
    again = bp.deal_groups(shots, group_of, signature, 4, np.random.default_rng(0))
    assert again == fold_of


def _repeats():
    """Shot 1 is tested in repeats 0 and 2, shot 2 in repeat 1 and shot 3 in 0 and 1;
    each shot has two windows (class 0 and 1), and the probabilities differ by repeat."""
    shots = np.array([1, 1, 3, 3, 1, 1, 2, 2, 3, 3])
    split = np.array(["split0"] * 4 + ["split2"] * 2 + ["split1"] * 4)
    start = np.array([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0])
    truth = np.array([0, 1] * 5)
    probs = np.array(
        [
            [0.9, 0.1, 0, 0],
            [0.2, 0.8, 0, 0],
            [0.6, 0.4, 0, 0],
            [0.1, 0.9, 0, 0],
            [0.5, 0.5, 0, 0],
            [0.0, 1.0, 0, 0],
            [0.7, 0.3, 0, 0],
            [0.3, 0.7, 0, 0],
            [0.2, 0.8, 0, 0],
            [0.4, 0.6, 0, 0],
        ]
    )
    return shots, start, truth, split, probs


def test_repeat_overlap_counts_tests_shots_and_distinct_windows():
    shots, start, truth, split, _ = _repeats()
    out = bp.repeat_overlap(shots, start, truth, split)
    assert out["tests"] == 5 and out["shots"] == 3  # (1,0) (1,2) (3,0) (3,1) (2,1)
    assert out["shots_by_times_tested"] == {1: 1, 2: 2}
    assert out["windows"] == 10 and out["distinct_windows"] == 6
    assert out["classes"]["L"] == {"windows": 5, "distinct_windows": 3, "shots": 3}
    assert out["classes"]["WP"]["windows"] == 0


def test_first_occurrence_keeps_each_shot_from_the_first_repeat_that_tested_it():
    shots, _, _, split, _ = _repeats()
    keep = bp.first_occurrence(shots, split)
    # shot 1: repeat 0 (rows 0, 1), not 2; shot 3: repeat 0 (rows 2, 3), not 1
    assert keep.tolist() == [True] * 4 + [False] * 2 + [True] * 2 + [False] * 2
    assert len(np.unique(shots[keep])) == 3


def test_average_repeats_gives_one_row_per_window_with_the_mean_probability():
    shots, start, truth, _, probs = _repeats()
    s, t, p, y = bp.average_repeats(shots, start, probs, truth)
    assert len(s) == 6
    row = {(a, b): (q, c) for a, b, q, c in zip(s, t, p, y, strict=True)}
    q, c = row[(1, 0.0)]
    np.testing.assert_allclose(q, [0.7, 0.3, 0, 0])  # mean of 0.9 and 0.5
    assert c == 0
    np.testing.assert_allclose(row[(2, 2.0)][0], [0.3, 0.7, 0, 0])  # tested once


def test_average_repeats_refuses_a_window_with_two_labels():
    shots, start, truth, _, probs = _repeats()
    truth = truth.copy()
    truth[4] = 3  # shot 1's first window labelled differently in repeat 2
    with np.testing.assert_raises(ValueError):
        bp.average_repeats(shots, start, probs, truth)


def test_failure_blocks_lists_a_class_of_a_shot_that_is_almost_all_wrong():
    # shot 7: 200 L windows, 5 called right; 200 H windows, all right.
    # shot 8: 200 L windows all right; 100 QH windows, none right (too short a block)
    truth = np.array([0] * 200 + [1] * 200 + [0] * 200 + [2] * 100)
    shots = np.array([7] * 400 + [8] * 300)
    guess = truth.copy()
    guess[:195] = 2
    guess[600:] = 1
    out = bp.failure_blocks(guess, truth, shots)
    assert out.shot.tolist() == [7] and out.label.tolist() == [0]
    assert out.windows.tolist() == [200] and out.correct.tolist() == [5]
    assert out.called.tolist() == [2] and np.isclose(out.called_share[0], 0.975)
    both = bp.failure_blocks(guess, truth, shots, min_windows=100)
    assert sorted(zip(both.shot, both.label, strict=True)) == [(7, 0), (8, 2)]
    # a mask drops the windows it excludes
    none = bp.failure_blocks(guess, truth, shots, shots == 8)
    assert none.empty
