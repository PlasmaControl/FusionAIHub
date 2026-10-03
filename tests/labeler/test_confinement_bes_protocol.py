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
