"""RWM presence per bin: the bin a slice scores, its target, the cut and folds."""

import numpy as np
import pandas as pd
import pytest

from labeler.events import grid
from labeler.rwm import classify


def table(*rows):
    return pd.DataFrame(rows, columns=["category", "t_start", "t_end"])


def test_slice_at_t_scores_the_frame_before_it():
    rows = table(
        (3, 5, 100), (4, 100, 856), (2, 836, 856), (1, 856, 879), (4, 879, 2000)
    )
    t = np.array([10, 100, 110, 850, 860, 870, 880, 890, 2000, 2010, 2500])
    got = classify.bin_categories(rows, t)
    # [0, 10) is before the window's snapped start (10): unassessed; [840, 860)
    # uncertain; the present span snaps to [860, 880): the slices at 870 and 880
    assert got.tolist() == [-1, 3, 4, 2, 2, 1, 1, 4, 4, -1, -1]
    assert classify.bin_categories(table(), t).tolist() == [grid.UNASSESSED] * len(t)


def test_targets_leave_out_uncertain_unless_the_sensitivity_asks():
    c = np.array([1, 2, 3, 4, 0, -1])
    assert classify.targets(c).tolist() == [1, -1, -1, 0, 0, -1]
    unc = classify.targets(c, classify.UNCERTAIN_NEGATIVE)
    assert unc.tolist() == [1, 0, -1, 0, 0, -1]
    with pytest.raises(ValueError):
        classify.targets(c, "other")


def test_calendar_day_joins_same_day_records():
    assert classify.calendar_day("20180314A") == classify.calendar_day("20180314")
    assert classify.calendar_day("20140620A") == "20140620"


def test_f1_cut_and_statistic():
    score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    label = np.array([1, 0, 1, 0, 0])
    assert classify.f1_cut(score, label) == pytest.approx(0.7)
    record = {
        "shot": 1,
        "day": "d",
        "score": score,
        "call": score >= 0.7,
        "y": label.astype(bool),
    }
    got = classify.statistic({"hanson": [record]})
    assert got["f1"] == pytest.approx(0.8)
    assert got["auroc"] == pytest.approx(5 / 6)
    assert got["prevalence"] == pytest.approx(0.4)
    assert classify.baseline(0.4)["f1"] == pytest.approx(0.8 / 1.4)


def test_cross_validation_scores_every_day_at_one_cut():
    rng = np.random.default_rng(0)
    frames = []
    for shot, day in enumerate(["a", "a", "b", "b", "c", "c"]):
        y = (np.arange(60) % 10 == 0).astype(np.int8)
        frames.append(
            pd.DataFrame(
                {
                    "shot": shot,
                    "t_ms": np.arange(60) * 10.0,
                    "day": day,
                    "y": y,
                    "x": y + 0.1 * rng.standard_normal(60),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    oof = classify.cross_validate(
        frame, ["x"], forest={"n_estimators": 5, "max_depth": 2, "min_leaf": 1}
    )
    assert oof.score.notna().all() and oof.cut.groupby(oof.day).nunique().eq(1).all()
    got = classify.statistic({"h": classify.shot_records(oof)})
    assert got["auroc"] > 0.95 and got["f1"] > 0.9
