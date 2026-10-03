"""Regression coverage for missing-truth masking and inner shot selection."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/labeler"))
import sawtooth_benchmark as benchmark

from labeler.sawtooth.metrics import (
    aggregate,
    bootstrap_cells,
    point_metrics,
)


def test_undefined_metrics_remain_null_through_bootstrap():
    assert point_metrics([0, 0, 0]) == {
        "precision": None,
        "recall": None,
        "f1": None,
    }
    result = bootstrap_cells([{"cells": [0, 0, 0]}], replicates=10)
    assert result["ci95"] == {
        "precision": None,
        "recall": None,
        "f1": None,
    }


def test_inner_selection_contains_only_outer_training_shots():
    outer_training = list(range(40))
    training, selection = benchmark.inner_split(outer_training, seed=23)
    assert len(selection) == 8
    assert set(training).isdisjoint(selection)
    assert set(training) | set(selection) == set(outer_training)
    assert benchmark.inner_split(outer_training, seed=23) == (training, selection)


def test_dense_loss_and_gradients_ignore_unassessed_bins():
    target = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    actual = benchmark.loss("saw-ours", logits, target)
    actual.backward()
    assert torch.equal(logits.grad[:, :, 1], torch.zeros(1, 3))
    expected = 2 * np.log(2)
    assert actual.item() == pytest.approx(expected)


def test_targets_exclude_uncertain_crashes_and_intervals():
    t = np.arange(10) * 0.001
    rec = {
        "intervals": [
            {"start_s": 0.0, "end_s": 0.003, "attrs": {"period_ms": 60}},
            {
                "start_s": 0.005,
                "end_s": 0.009,
                "attrs": {"period_ms": 60, "state": "uncertain"},
            },
        ],
        "crashes": [
            {"time_s": 0.001, "attrs": {"state": "present"}},
            {"time_s": 0.007, "attrs": {"state": "uncertain"}},
        ],
    }
    pick, presence, classes = benchmark.targets(t, rec, 50)
    assert pick[1] == 1
    assert pick[7] == 0
    assert presence.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert classes[6] == 0


def test_masked_points_exclude_missing_uncertain_and_out_of_coverage():
    signal = {
        "t": np.arange(5) * 0.001,
        "observable": np.array([True, False, True, True, True]),
        "assessed": np.array([True, False, False, True, True]),
    }
    points = [-0.001, 0.0, 0.001, 0.002, 0.003, 0.006]
    assert benchmark.assessed_points(signal, points).tolist() == [0.0, 0.003]


def test_aggregate_uses_each_shots_calibrated_presence_confusion():
    rows = [
        {
            "shot": 1,
            "cells": [0, 0, 0],
            "histogram": np.zeros((2, 512)),
            "presence_cells": [2, 1, 0, 3],
        },
        {
            "shot": 2,
            "cells": [0, 0, 0],
            "histogram": np.zeros((2, 512)),
            "presence_cells": [1, 0, 2, 4],
        },
    ]
    result = aggregate(rows, replicates=10)
    assert result["presence"]["precision"] == 0.75
    assert result["presence"]["recall"] == 0.6
    assert result["presence"]["accuracy"] == pytest.approx(10 / 13)
