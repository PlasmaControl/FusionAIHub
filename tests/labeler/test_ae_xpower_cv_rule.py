"""Deviation 11's rule on hand-made tables: its three branches and its ties."""

import pytest

from labeler.ae.xpower import cv

WEIGHTS = {"band80-mhd3": 3.0, "band80-mhd10": 10.0, "band80-mhd30": 30.0}


def _row(candidate, threshold, f1, mhd):
    return {
        "candidate": candidate,
        "mhd_weight": WEIGHTS[candidate],
        "threshold": threshold,
        "f1": f1,
        "fp_rate_mhd": mhd,
    }


def _picked(rows):
    row, branch = cv.choose_row(rows)
    return row["candidate"], row["threshold"], branch


def test_branch_one_the_best_f1_among_those_with_mhd_fp_at_most_005():
    rows = [
        _row("band80-mhd3", 0.5, 0.97, 0.06),  # the best F1, but over 0.05
        _row("band80-mhd10", 0.5, 0.91, 0.05),  # at the bound: eligible
        _row("band80-mhd30", 0.6, 0.93, 0.01),
        _row("band80-mhd30", 0.7, 0.88, 0.0),
    ]
    assert _picked(rows) == ("band80-mhd30", 0.6, 1)


def test_branch_two_the_lowest_mhd_fp_among_those_with_f1_at_least_090():
    rows = [
        _row("band80-mhd3", 0.5, 0.95, 0.20),
        _row("band80-mhd10", 0.5, 0.90, 0.08),  # at the bound: eligible
        _row("band80-mhd30", 0.5, 0.89, 0.06),  # lower MHD FP, but F1 under 0.90
        _row("band80-mhd30", 0.8, 0.92, 0.09),
    ]
    assert _picked(rows) == ("band80-mhd10", 0.5, 2)


def test_branch_three_the_highest_f1():
    rows = [
        _row("band80-mhd3", 0.5, 0.85, 0.20),
        _row("band80-mhd10", 0.4, 0.88, 0.30),
        _row("band80-mhd30", 0.5, 0.80, 0.06),
    ]
    assert _picked(rows) == ("band80-mhd10", 0.4, 3)


@pytest.mark.parametrize("branch", [1, 2, 3])
def test_ties_go_to_the_lower_mhd_weight_before_the_threshold(branch):
    f1, mhd = {1: (0.92, 0.04), 2: (0.93, 0.07), 3: (0.85, 0.2)}[branch]
    rows = [
        _row("band80-mhd30", 0.5, f1, mhd),
        _row("band80-mhd10", 0.5, f1, mhd),
        _row("band80-mhd10", 0.85, f1, mhd),  # the lower weight wins first
        _row("band80-mhd3", 0.85, f1 - 0.01 if branch != 2 else f1, mhd + 0.001),
    ]
    assert _picked(rows) == ("band80-mhd10", 0.5, branch)
    rows.append(_row("band80-mhd3", 0.9, f1, mhd))
    assert _picked(rows) == ("band80-mhd3", 0.9, branch)


def test_then_the_threshold_nearest_05_and_then_the_lower_one():
    rows = [_row("band80-mhd3", t, 0.95, 0.01) for t in (0.1, 0.3, 0.45, 0.55, 0.9)]
    # 0.45 and 0.55 are equally near 0.5 (in hundredths, not in floating point);
    # the rule does not say, so the lower threshold, as `train.pick_threshold`.
    assert _picked(rows) == ("band80-mhd3", 0.45, 1)
    assert _picked(rows[3:]) == ("band80-mhd3", 0.55, 1)
    assert _picked([*rows, _row("band80-mhd3", 0.5, 0.95, 0.01)])[1] == 0.5


def test_undefined_scores_never_win():
    rows = [
        _row("band80-mhd3", 0.5, None, None),
        _row("band80-mhd10", 0.5, 0.99, None),  # no MHD frame: not branch 1 or 2
        _row("band80-mhd30", 0.5, 0.91, 0.30),
    ]
    assert _picked(rows) == ("band80-mhd30", 0.5, 2)
    assert _picked(rows[:2]) == ("band80-mhd10", 0.5, 3)
    with pytest.raises(ValueError, match="defined F1"):
        cv.choose_row(rows[:1])


def test_the_rule_is_stated_as_the_ledger_states_it():
    assert cv.MHD_FP_MAX == 0.05 and cv.F1_MIN == 0.90
    assert [round(float(t), 2) for t in cv.THRESHOLDS] == [
        round(0.10 + 0.05 * k, 2) for k in range(17)
    ]
    assert set(cv.BRANCHES) == {1, 2, 3}
