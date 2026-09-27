"""Scoring the AE segmentation: frame calls, coverage, the bar, a run on the tree."""

from __future__ import annotations

import json

import numpy as np
import pytest

from labeler.ae.seg import evaluate, model_dir, pseudo, train
from labeler.events.review.rows import Grid

from . import ae_tree
from .test_ae_seg_train import Small


def test_a_frame_is_present_when_half_its_columns_hold_a_pixel():
    grid = Grid(0.0, 2.0, 10)  # frame 0 holds columns 0-4, frame 1 columns 5-9
    on = np.zeros((20, 10), dtype=bool)
    on[3, [1, 2, 4]] = True
    on[:, [6, 8]] = True
    assert evaluate.frame_calls(on, grid, 0, 2).tolist() == [True, False]
    assert evaluate.frame_calls(on, grid, 1, 1).tolist() == [False]


def test_a_frame_is_covered_only_when_all_its_columns_are():
    grid = Grid(0.0, 2.0, 10)
    covered = np.ones(10, dtype=bool)
    covered[7] = False
    assert evaluate.covered_frames(covered, grid, 0, 3).tolist() == [True, False, False]


def test_each_column_takes_its_frames_flag():
    grid = Grid(5.0, 2.0, 6)  # centres 6, 8, 10, 12, 14, 16 ms
    flags = np.array([True, False])  # frames 0 and 1
    assert evaluate.column_frames(flags, 0, grid).tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def _scores(dice, low, precision, mhd):
    ours = {
        "dice": {"value": dice, "low": low},
        "frame_precision": {"value": precision},
        "fp_rate_mhd": {"value": mhd},
    }
    return {"methods": {"ae_seg": ours}}


@pytest.mark.parametrize(
    ("scores", "gates"),
    [
        (_scores(0.80, 0.70, 0.95, 0.01), (True, True, True)),
        (_scores(0.80, 0.60, 0.90, 0.01), (False, True, True)),
        (_scores(0.74, 0.70, 0.90, 0.01), (False, True, True)),
        (_scores(0.80, 0.70, 0.89, 0.01), (True, False, True)),
        (_scores(0.80, 0.70, 0.90, 0.06), (True, True, False)),
        (_scores(0.80, 0.70, 0.90, None), (True, True, False)),
    ],
)
def test_the_bar(scores, gates):
    bar = evaluate.verdict(scores)
    assert (bar["G1"], bar["G2"], bar["G3"]) == gates
    assert bar["all"] == all(gates)


def test_the_command_scores_the_test_shots_against_the_baselines(tmp_path, monkeypatch):
    splits = {s: "train" for s in (101, 102, 103, 104)}
    paths = ae_tree.build(tmp_path, splits, tokeye_dt=0.256)
    ae_tree.chosen(paths, {101: "train", 102: "train", 103: "val", 104: "test"})
    ae_tree.env(monkeypatch, paths)
    assert pseudo.main([]) == 0
    monkeypatch.setattr(train, "TrainConfig", Small)
    assert train.main(["--epochs", "1"]) == 0
    assert evaluate.main([]) == 0
    record = json.loads((model_dir(paths) / "evaluation.json").read_text())
    assert set(record["methods"]) == {"ae_seg", "recipe", "tokeye"}
    assert record["counts"]["shots"] == 1 and record["counts"]["frames"] >= 190
    assert record["counts"]["mhd_absent_frames"] >= 25
    tokeye = record["methods"]["tokeye"]
    assert tokeye["frame_recall"]["value"] == 1.0, "TokEye lights every AE frame"
    assert tokeye["fp_rate_mhd"]["value"] > 0.5, "and the MHD harmonics too"
    assert tokeye["dice"]["value"] < 1.0
    assert record["meta"]["tier"] == "suggestions"
    assert set(record["bar"]) == {"G1", "G2", "G3", "all"}
    report = (model_dir(paths) / "evaluation.md").read_text()
    assert report.startswith("# AE segmentation on the test shots")
    assert "Tier: suggestions." in report
