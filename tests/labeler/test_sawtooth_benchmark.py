"""Regression coverage for missing-truth masking and inner shot selection."""

import json
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

SYNTHETIC_SPLIT = {
    "training_cohort": [1],
    "fixed_validation_excluded": [98],
    "blind_test_excluded": [99],
    "all_reviewed_excluded": [100],
}


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
        "observable_spans": [[0.0, 0.01]],
        "assessed_spans": [[0.0, 0.005], [0.009, 0.01]],
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


def test_uncertain_crash_center_cannot_leak_gaussian_into_assessed_bins():
    t = np.arange(200) * 0.0001
    rec = {
        "observable_spans": [[0.0, 0.02]],
        "assessed_spans": [[0.0, 0.004], [0.005, 0.02]],
        "intervals": [],
        "crashes": [{"time_s": 0.0042, "attrs": {"state": "present"}}],
    }
    pick, _, _ = benchmark.targets(t, rec, 50)
    assessed = benchmark.spans_at(t, rec["assessed_spans"])
    # The old Gaussian reaches t=5 ms, outside the uncertain interval, even
    # though its center is unknown. A masked loss alone does not remove it.
    assert not np.any(pick[assessed] > 0)


def test_target_center_eligibility_uses_nearest_native_bin():
    t = np.arange(200) * 0.0001
    rec = {
        "observable_spans": [[0.0, 0.02]],
        "assessed_spans": [[0.0, 0.004], [0.005, 0.02]],
        "intervals": [],
        # This time lies before assessed support, but its nearest native bin
        # is t=5 ms, which is assessed. It must match assessed_points semantics.
        "crashes": [{"time_s": 0.00496, "attrs": {"state": "present"}}],
    }
    pick, _, _ = benchmark.targets(t, rec, 50)
    assert pick[50] == pytest.approx(np.exp(-0.5 * (0.00004 / 0.0005) ** 2))


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


@pytest.mark.parametrize("name,channels", [("saw-hl3", 4), ("saw-ours", 48)])
def test_inputs_retain_observable_uncertain_measurements(name, channels):
    signal = {
        "t": np.arange(5) * 0.001,
        "observable": np.array([True, False, True, True, True]),
        "assessed": np.array([True, False, False, True, True]),
        "baseline": np.full((4, 5), 3.0),
        "y": np.full((48, 5), 3.0),
    }
    values = benchmark.input_values(signal, name)
    assert values.shape == (channels, 5)
    assert np.isnan(values[:, 1]).all()
    assert np.equal(values[:, 2], 3).all()
    signal["assessed"][:] = False
    np.testing.assert_equal(benchmark.input_values(signal, name), values)
    del signal["assessed"]
    np.testing.assert_equal(benchmark.input_values(signal, name), values)


@pytest.mark.parametrize("name,width", [("saw-hl3", 200), ("saw-ours", 1000)])
def test_sampled_windows_retain_uncertain_inputs_and_mask_missing_inputs(
    tmp_path, name, width
):
    (tmp_path / "signals").mkdir()
    (tmp_path / "shots").mkdir()
    t = np.arange(width) * 0.0001
    observable = np.ones(width, dtype=bool)
    observable[20] = False
    assessed = observable.copy()
    assessed[30:40] = False
    np.savez(
        tmp_path / "signals/1.npz",
        t=t,
        observable=observable,
        assessed=assessed,
        baseline=np.full((4, width), 3.0),
        y=np.full((48, width), 3.0),
    )
    rec = {
        "observable_spans": [[0.0, width * 0.0001]],
        "assessed_spans": [[0.0, 0.003], [0.004, width * 0.0001]],
        "intervals": [],
        "crashes": [],
    }
    (tmp_path / "shots/1.json").write_text(json.dumps(rec))
    x, _, _ = benchmark.windows(tmp_path, [1], name, 50, per_shot=1)
    assert np.equal(x[:, :, 30:40], 3).all()
    assert np.isnan(x[:, :, 20]).all()


def test_crash_centred_sampling_balances_both_period_classes():
    # One rare small-period crash must receive the same half of positive
    # windows as three long-period crashes; random windows remain half.
    t = np.arange(3000) * 0.0001
    classes = np.zeros(len(t), dtype=int)
    classes[490:510] = 1
    classes[1490:1510] = classes[1990:2010] = classes[2490:2510] = 2
    centers = np.array([0.05, 0.15, 0.2, 0.25])
    starts = benchmark.sample_window_starts(
        t, classes, centers, 200, np.random.default_rng(21), count=20
    )
    assert len(starts) == 20
    labels = classes[starts[10:] + 100]
    assert np.bincount(labels, minlength=3).tolist() == [0, 5, 5]


@pytest.mark.parametrize("forbidden", [90, 91, 92])
def test_fit_guard_rejects_fixed_val_test_and_expert_shots(forbidden):
    split = {
        "training_cohort": [1, 2, 3],
        "fixed_validation_excluded": [90],
        "blind_test_excluded": [91],
        "all_reviewed_excluded": [92],
    }
    with pytest.raises(ValueError, match="fixed train"):
        benchmark.guard_partition(split, [1, forbidden], [2], [3])


def test_fit_guard_rejects_shot_overlap():
    split = {
        "training_cohort": [1, 2, 3],
        "fixed_validation_excluded": [],
        "blind_test_excluded": [],
        "all_reviewed_excluded": [],
    }
    with pytest.raises(ValueError, match="overlap"):
        benchmark.guard_partition(split, [1], [1, 2], [3])


def test_calibration_refuses_shots_outside_inner_selection(tmp_path):
    with pytest.raises(ValueError, match="selection"):
        benchmark.calibrate(
            "saw-hl3",
            {99: {}},
            tmp_path,
            selection_shots=[1],
            split=SYNTHETIC_SPLIT,
        )


@pytest.mark.parametrize("forbidden", [98, 99, 100])
def test_calibration_refuses_nontrain_selection_even_if_predictions_match(
    tmp_path, forbidden
):
    with pytest.raises(ValueError, match="fixed train"):
        benchmark.calibrate(
            "saw-hl3",
            {forbidden: {}},
            tmp_path,
            selection_shots=[forbidden],
            split=SYNTHETIC_SPLIT,
        )


def test_derivative_calibration_can_select_above_previous_grid_limit(tmp_path):
    (tmp_path / "signals").mkdir()
    (tmp_path / "shots").mkdir()
    t = np.arange(10000) * 0.0001
    core = 0.001 * np.sin(2 * np.pi * 1000 * t)
    core -= 0.03 * (t >= 0.25) + (t >= 0.5) + 0.03 * (t >= 0.75)
    signal = {
        "t": t,
        "baseline": np.tile(core, (4, 1)),
        "observable": np.ones(len(t), dtype=bool),
        "assessed": np.ones(len(t), dtype=bool),
    }
    np.savez(tmp_path / "signals/1.npz", **signal)
    rec = {
        "intervals": [{"start_s": 0.4, "end_s": 0.6}],
        "crashes": [{"time_s": 0.5}],
    }
    (tmp_path / "shots/1.json").write_text(json.dumps(rec))
    prediction = {"t": t, "presence": np.ones(len(t))}
    selected, _ = benchmark.calibrate(
        "saw-hl3",
        {1: prediction},
        tmp_path,
        selection_shots=[1],
        split=SYNTHETIC_SPLIT,
    )
    assert selected["z"] > 10
    assert selected["cells"] == [1, 0, 0]


def test_queue_prediction_guard_accepts_only_fixed_validation_shots():
    split = {"fixed_validation_excluded": [90, 91], "blind_test_excluded": [99]}
    assert benchmark.guard_prediction_shots(split, [91, 90, 90]) == [90, 91]
    with pytest.raises(ValueError, match="fixed validation"):
        benchmark.guard_prediction_shots(split, [99])


def test_ensemble_averages_frozen_probabilities_and_retains_native_support():
    support = {
        "t": np.array([0.0, 0.001, 0.002]),
        "observable": np.array([True, True, False]),
        "assessed": np.array([True, False, False]),
    }
    predictions = [
        {**support, "presence": np.array([0.2, 0.4, 0.1]), "crash": np.ones(3)},
        {**support, "presence": np.array([0.4, 0.8, 0.3]), "crash": np.zeros(3)},
    ]
    result = benchmark.ensemble_predictions(predictions)
    np.testing.assert_allclose(result["presence"], [0.3, 0.6, 0.2])
    np.testing.assert_allclose(result["crash"], [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(result["observable"], [True, True, False])
    np.testing.assert_array_equal(result["assessed"], [True, False, False])
