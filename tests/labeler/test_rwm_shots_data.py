"""Choosing the comparison shots and building a shot's labelled slice table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from labeler.events import raw
from labeler.events.verify import NoDataError
from labeler.features.store import FeatureArray, write_features
from labeler.rwm import data, labels, shots

from . import editor_tree as tree


def _runs():
    rows = [
        # run day A (2014) holds Hanson shot 100; 101 is a plasma shot of the same day
        (100, "A", "20140101", "Testing Kinetic  RWM Stabilization", "plasma"),
        (101, "A", "20140101", "Testing Kinetic RWM Stabilization", "plasma"),
        (102, "A", "20140101", "Testing Kinetic RWM Stabilization", "no_plasma"),
        # a different 2014 day of an RWM experiment
        (110, "B", "20140301", "Explore access to BN~5", "plasma"),
        # an unrelated 2014 day, and a day of the right title in another year
        (120, "C", "20140401", "Edge physics", "plasma"),
        (130, "D", "20180101", "Explore access to BN~5", "plasma"),
    ]
    return pd.DataFrame(
        rows, columns=["shot", "run", "shot_entered", "run_title", "shot_type"]
    ).assign(mpid="", configuration="")


def test_choose_takes_the_hanson_day_and_same_year_experiment_days():
    out = shots.choose(_runs(), [100]).set_index("shot")
    assert out.role.to_dict() == {
        100: "hanson",
        101: "same_day",
        110: "same_experiment",
    }
    assert out.loc[100, "run_title"] == "testing kinetic rwm stabilization"


def test_choose_keeps_a_hanson_shot_the_logbook_lacks():
    out = shots.choose(_runs(), [100, 999]).set_index("shot")
    assert out.loc[999, "role"] == "hanson" and out.loc[999, "run"] == ""


def test_read_runs_decodes_only_the_shots_in_range(tmp_path):
    path = tmp_path / "logs.jsonl"
    lines = [
        {"shot": 5, "run": "X"},
        {"shot": 156786, "run": "A", "run_title": "t", "shot_type": "plasma"},
        {"shot": 156786, "run": "B"},  # a second entry of the same shot is ignored
        {"shot": 176068, "run": "C", "run_title": "u"},
        {"shot": 170000, "run": "Y"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    out = shots.read_runs(path)
    assert out.shot.tolist() == [156786, 176068]
    assert out.run.tolist() == ["A", "C"]


def _write_shot(p, shot, *, n1=1.0, with_optional=True):
    t_efit = np.arange(0.0, 1.0, 0.02)
    t_fast = np.arange(0.0, 1.0, 0.001)
    arrays = {
        "ip": FeatureArray(
            t_fast, np.atleast_2d(np.full_like(t_fast, 1.0e6)), {"resolver": "fdp"}
        ),
        "betan": FeatureArray(
            t_efit, np.atleast_2d(np.full_like(t_efit, 2.5)), {"resolver": "fdp"}
        ),
        "li": FeatureArray(
            t_efit, np.atleast_2d(np.full_like(t_efit, 0.7)), {"resolver": "fdp"}
        ),
        "n1rms": FeatureArray(
            t_fast, np.atleast_2d(np.full_like(t_fast, n1)), {"resolver": "fdp"}
        ),
    }
    if with_optional:
        arrays["q95"] = FeatureArray(
            t_efit, np.atleast_2d(np.full_like(t_efit, 5.0)), {"resolver": "fdp"}
        )
    write_features(raw.cache_path(shot, paths=p), shot, arrays, {})


def test_load_signals_reads_the_raw_cache_in_milliseconds(tmp_path):
    p = tree.paths(tmp_path)
    _write_shot(p, 7)
    signals = data.load_signals(7, p)
    assert {"ip", "betan", "li", "n1rms", "q95"} <= set(signals)
    assert "wmhd" not in signals  # optional and absent
    assert signals["betan"][0][1] == pytest.approx(20.0)


def test_a_shot_without_a_required_signal_raises(tmp_path):
    p = tree.paths(tmp_path)
    with pytest.raises(NoDataError):
        data.load_signals(8, p)


def test_shot_table_labels_an_examined_shot_and_leaves_the_rest_unlabelled(tmp_path):
    p = tree.paths(tmp_path)
    _write_shot(p, 7)
    examined = data.shot_table(7, p, [600.0], examined=True)
    assert set(examined.label) == {labels.NEGATIVE, labels.POSITIVE, labels.EXCLUDED}
    positive = examined.t_ms[examined.label == labels.POSITIVE]
    assert positive.min() >= 500.0 and positive.max() < 600.0
    assert (examined.shot == 7).all()
    other = data.shot_table(7, p, [600.0], examined=False)
    assert (other.label == labels.UNLABELLED).all()
    assert len(other) == len(examined)
