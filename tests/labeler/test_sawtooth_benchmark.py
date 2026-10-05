"""Regression coverage for missing-truth masking and inner shot selection."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_manifest_cannot_admit_stale_signal_after_failed_refresh(tmp_path):
    (tmp_path / "signals").mkdir()
    (tmp_path / "shots").mkdir()
    for shot, record in ((1, {}), (2, {"error": "unsupported physical core"})):
        (tmp_path / "signals" / f"{shot}.npz").touch()
        (tmp_path / "shots" / f"{shot}.json").write_text(json.dumps(record))
    (tmp_path / "signals/3.npz").touch()
    assert benchmark.usable_signal_shots(tmp_path) == {1}


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
        "y": np.tile(core, (48, 1)),
        "central_channel": np.array(12),
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


def test_derivative_uses_axis_selected_single_channel_not_core_mean():
    t = np.arange(10000) * 0.0001
    core = 0.001 * np.sin(2 * np.pi * 1000 * t) - (t >= 0.5)
    y = np.zeros((48, len(t)))
    y[12] = core
    signal = {
        "t": t,
        "y": y,
        "baseline": np.zeros((4, len(t))),
        "central_channel": np.array(12),
        "observable": np.ones(len(t), dtype=bool),
        "assessed": np.zeros(len(t), dtype=bool),
    }
    peaks, amplitude = benchmark.derivative_candidates(signal)
    assert any(abs(t[peak] - 0.5) < 0.002 for peak in peaks[amplitude >= 20])
    signal["baseline"][:] = 100
    np.testing.assert_array_equal(benchmark.derivative_candidates(signal)[0], peaks)


def test_derivative_refuses_missing_axis_selected_channel():
    signal = {
        "t": np.arange(100) * 0.0001,
        "baseline": np.zeros((4, 100)),
        "observable": np.ones(100, bool),
    }
    with pytest.raises(ValueError, match="central_channel"):
        benchmark.derivative_candidates(signal)


def test_derivative_gap_cannot_manufacture_edge_at_missing_boundary():
    t = np.arange(10000) * 0.0001
    core = np.full(len(t), 2.0)
    core[t >= 0.7] = 1.0
    core[(t >= 0.3) & (t < 0.7)] = np.nan
    signal = {
        "t": t,
        "y": np.tile(core, (48, 1)),
        "central_channel": 12,
        "observable": np.ones(len(t), bool),
        "assessed": np.ones(len(t), bool),
    }
    peaks, amplitude = benchmark.derivative_candidates(signal)
    assert not len(peaks[amplitude >= 20])


def test_derivative_period_classes_keep_single_edge_abstentions_and_gap_support():
    t = np.arange(8001) * 0.0001
    observable = np.ones(len(t), bool)
    observable[(t >= 0.25) & (t < 0.35)] = False
    signal = {"t": t, "observable": observable, "assessed": observable}
    classes = benchmark.derivative_classes(
        signal,
        np.array([0.01, 0.1, 0.16, 0.4, 0.5, 0.75]),
        20,
        80,
        derivative=(np.array([1000, 1600, 4000, 5000]), np.full(4, 25.0)),
    )
    assert classes.tolist() == [1, 1, 1, 2, 2, 0]
    # A separate observable run with one edge cannot fabricate a period class.
    classes = benchmark.derivative_classes(
        signal,
        np.array([0.4, 0.5]),
        20,
        80,
        derivative=(np.array([4000]), np.array([25.0])),
    )
    assert classes.tolist() == [-1, -1]


def test_derivative_period_cannot_cross_missing_selected_channel_measurements():
    t = np.arange(8001) * 0.0001
    y = np.full((48, len(t)), 2.0)
    y[12, (t >= 0.25) & (t < 0.35)] = np.nan
    # Other core measurements keep rule observability true across this gap.
    signal = {
        "t": t,
        "y": y,
        "central_channel": 12,
        "observable": np.ones(len(t), bool),
        "assessed": np.ones(len(t), bool),
    }
    result = benchmark.derivative_classes(
        signal,
        np.array([0.16, 0.4]),
        20,
        80,
        derivative=(np.array([1000, 1600, 4000, 5000]), np.full(4, 25.0)),
    )
    assert result.tolist() == [1, 2]


def test_derivative_presence_has_exact_125ms_support_and_pick_counts():
    t = np.array([0.0, 0.3749, 0.375, 0.5, 0.625, 0.6251, 1.0])
    signal = {
        "t": t,
        "observable": np.array([True, True, True, True, True, True, False]),
        "assessed": np.array([True, True, False, False, True, True, False]),
    }
    prediction = benchmark.derivative_prediction(
        signal, 20, derivative=(np.array([3, 6]), np.array([20.0, 30.0]))
    )
    assert prediction["presence"].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert prediction["picks"].tolist() == [0.5]
    assert benchmark.pick_counts(signal, prediction["picks"]) == {
        "observable_picks": 1,
        "assessed_picks": 0,
        "excluded_picks": 1,
    }


def test_phase_picker_keeps_observable_uncertain_picks_for_reporting():
    t = np.arange(200) * 0.0001
    crash = np.zeros(len(t))
    crash[[40, 140]] = 1
    signal = {
        "t": t,
        "observable": np.ones(len(t), dtype=bool),
        "assessed": t < 0.01,
    }
    prediction = {"t": t, "crash": crash}
    actual = benchmark.picks("saw-ours", prediction, signal, 0.99)
    assert actual.tolist() == [0.004, 0.014]


def test_derivative_calibration_refuses_fixed_validation(tmp_path):
    with pytest.raises(ValueError, match="fixed train"):
        benchmark.calibrate_derivative(tmp_path, [98], split=SYNTHETIC_SPLIT)


def test_derivative_crash_and_presence_select_independent_inner_z(tmp_path):
    (tmp_path / "signals").mkdir()
    (tmp_path / "shots").mkdir()
    t = np.arange(10000) * 0.0001
    core = 0.005 * np.sin(2 * np.pi * 1000 * t)
    core -= 0.3 * (t >= 0.2) + 0.18 * (t >= 0.5) + 0.6 * (t >= 0.58)
    assessed = ~((t >= 0.198) & (t < 0.202))
    np.savez(
        tmp_path / "signals/1.npz",
        t=t,
        y=np.tile(core, (48, 1)),
        central_channel=12,
        observable=np.ones(len(t), bool),
        assessed=assessed,
    )
    (tmp_path / "shots/1.json").write_text(
        json.dumps(
            {
                "intervals": [{"start_s": 0.455, "end_s": 0.705}],
                "crashes": [{"time_s": 0.5}, {"time_s": 0.58}],
            }
        )
    )
    selected = benchmark.calibrate_derivative(tmp_path, [1], split=SYNTHETIC_SPLIT)
    assert selected["cells"] == [2, 0, 0]
    assert selected["presence_z"] > selected["z"]


def test_dense_picker_calibration_can_reject_peaks_above_point95(tmp_path):
    (tmp_path / "signals").mkdir()
    (tmp_path / "shots").mkdir()
    t = np.arange(10000) * 0.0001
    signal = {
        "t": t,
        "observable": np.ones(len(t), bool),
        "assessed": np.ones(len(t), bool),
    }
    np.savez(tmp_path / "signals/1.npz", **signal)
    (tmp_path / "shots/1.json").write_text(
        json.dumps(
            {
                "intervals": [{"start_s": 0.4, "end_s": 0.6}],
                "crashes": [{"time_s": 0.5}],
            }
        )
    )
    crash = np.zeros(len(t))
    crash[[2500, 5000]] = [0.98, 1.0]
    selected, _ = benchmark.calibrate(
        "saw-ours",
        {1: {"t": t, "presence": np.ones(len(t)), "crash": crash}},
        tmp_path,
        selection_shots=[1],
        split=SYNTHETIC_SPLIT,
    )
    assert selected["threshold"] > 0.98
    assert selected["cells"] == [1, 0, 0]


def test_evaluation_exports_peer_baselines_fixed_holdout_and_excluded_picks(
    tmp_path, monkeypatch
):
    import pandas as pd

    cohort = tmp_path / "data/events/catalog/cohort.csv"
    cohort.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "shot": [1, 2, 3, 4, 5, 6, 7, 99],
            "split": ["train", "train", "train", "val", "val", "val", "train", "test"],
        }
    ).to_csv(cohort, index=False)
    review = tmp_path / "review.csv"
    pd.DataFrame(
        {
            "shot": [5, 5, 5],
            "t_start": [0, 40, 80],
            "t_end": [40, 80, 200],
            "category": [0, 1, 0],
        }
    ).to_csv(review, index=False)
    monkeypatch.setattr(benchmark, "REPO", tmp_path)
    monkeypatch.setattr(benchmark, "REVIEW", review)
    monkeypatch.setattr(benchmark, "OUTPUT", tmp_path / "output")
    work = tmp_path / "work"
    (work / "signals").mkdir(parents=True)
    (work / "shots").mkdir()
    t = np.arange(2000) * 0.0001
    core = 0.001 * np.sin(2 * np.pi * 1000 * t) - (t >= 0.05)
    observable, assessed = np.ones(len(t), bool), t < 0.1
    for shot in (1, 2, 3, 4, 5):
        np.savez(
            work / "signals" / f"{shot}.npz",
            t=t,
            y=np.tile(core, (48, 1)),
            baseline=np.tile(core, (4, 1)),
            central_channel=12,
            observable=observable,
            assessed=assessed,
            q_prior=(t >= 0.1) & (t < 0.15),
            absent_holdoff=(t < 0.1) & ~((t >= 0.04) & (t < 0.08)),
        )
        (work / "shots" / f"{shot}.json").write_text(
            json.dumps(
                {
                    "states": [
                        {"start_s": 0, "end_s": 0.04, "state": "absent"},
                        {"start_s": 0.04, "end_s": 0.08, "state": "present"},
                        {"start_s": 0.08, "end_s": 0.1, "state": "absent"},
                    ],
                    "uncertain_intervals": [],
                    "absence_diagnostics": {
                        "core_relaxation_test": {"ambiguous_edge_times_s": []}
                    },
                    "observable_spans": [[0, 0.2]],
                    "assessed_spans": [[0, 0.1]],
                    "intervals": [
                        {"start_s": 0.04, "end_s": 0.08, "attrs": {"period_ms": 60}}
                    ],
                    "crashes": [{"time_s": 0.05}],
                }
            )
        )
    for shot in (6, 7):
        (work / "shots" / f"{shot}.json").write_text(
            json.dumps({"error": "unusable ECE core", "error_kind": "core_geometry"})
        )
    split = benchmark.manifest(work)
    assert split["fixed_validation_nonexpert"] == [4, 6]
    assert split["fixed_validation_supported"] == [4]
    assert split["requested_training_cohort"] == [1, 2, 3, 7]
    requested, supported, unavailable = benchmark.prediction_population(
        split, work, [4, 5, 6]
    )
    assert requested == [4, 5, 6]
    assert supported == [4, 5]
    assert unavailable[0]["shot"] == 6
    assert unavailable[0]["exclusion_reason"] == "unusable ECE core"
    for name in benchmark.MODELS:
        for fold in range(3):
            directory = work / "models" / name / f"fold_{fold}"
            directory.mkdir(parents=True)
            (directory / "complete.json").write_text(
                json.dumps(
                    {
                        "input_policy": benchmark.INPUT_POLICY,
                        "training_shots": [],
                        "selection_shots": [],
                        "heldout_shots": [
                            s for s in (1, 2, 3) if split["folds"][s] == fold
                        ],
                        "fold": fold,
                        "presence_threshold": 0.5,
                        "period_boundary_ms": 50,
                        "selected_crash_threshold": {"threshold": 0.99, "z": 20},
                        "derivative_baseline": {"z": 20, "presence_z": 20},
                        "majority_class": 0,
                        "best_epoch": 3,
                        "epochs_completed": 8,
                        "seconds": 1.0,
                    }
                )
            )
            destination = work / "predictions" / name / f"fold_{fold}"
            destination.mkdir(parents=True)
            for shot in (1, 2, 3, 4, 5):
                prediction = {
                    "t": t,
                    "observable": observable,
                    "assessed": assessed,
                    "presence": np.ones(len(t)),
                    "picks": np.array([0.05, 0.15]),
                }
                if name == "saw-ours":
                    prediction["crash"] = np.zeros(len(t))
                    prediction["crash"][[500, 1500]] = 1
                else:
                    prediction["class_t"] = t[100::20]
                    prediction["class_prob"] = np.tile([0, 0, 1], (95, 1))
                np.savez(destination / f"{shot}.npz", **prediction)
    benchmark.evaluate(SimpleNamespace(work=work, models=list(benchmark.MODELS)))
    result = json.loads((work / "benchmark.json").read_text())
    scores = result["Tokamak-SI"]
    assert set(scores) == {
        "saw-hl3",
        "saw-ours",
        "saw-derivative",
        "saw-always-present",
    }
    assert scores["saw-always-present"]["crash_tolerance_2ms"]["crash"] is None
    assert scores["saw-hl3"]["crash_metric_label"] == "derivative picker gated by HL-3"
    assert scores["saw-ours"]["assessment_totals"]["observable_picks"] == 6
    assert scores["saw-ours"]["assessment_totals"]["excluded_picks"] == 3
    sensitivity = scores["saw-ours"]["old_negatives_sensitivity"]
    assert (
        sensitivity["assessment_totals"]["assessed_bins"]
        > scores["saw-ours"]["assessment_totals"]["assessed_bins"]
    )
    assert "crash_tolerance_2ms" in sensitivity
    context = scores["saw-ours"]["edge_context_sensitivity"]
    assert context["contexts_ms"] == [5.15, 50.0, 375.0]
    # No qualified edges in this fixture: every veto reproduces the assessed set.
    for key in ("5.15_ms", "50_ms", "375_ms"):
        assert (
            context[key]["assessment_totals"]["assessed_bins"]
            == scores["saw-ours"]["assessment_totals"]["assessed_bins"]
        )
    fixed_presence = scores["saw-ours"]["presence_fixed_threshold"]
    assert fixed_presence["threshold"] == 0.5
    assert [row["fold"] for row in fixed_presence["by_fold"]] == [0, 1, 2]
    assert fixed_presence["pooled"]["f1"] is not None
    points = scores["saw-ours"]["operating_points"]
    assert [row["fold"] for row in points["by_fold"]] == [0, 1, 2]
    assert points["presence_threshold_range"] == 0.0
    derivative_classes = scores["saw-derivative"]["three_class"]
    assert derivative_classes["windows"] == scores["saw-hl3"]["three_class"]["windows"]
    assert derivative_classes["unclassified_windows"] > 0
    assert scores["saw-always-present"]["three_class"]["window_accuracy"] is None
    for score in scores.values():
        assert score["fixed_validation"]["crash_tolerance_2ms"]["shot_ids"] == [4]
        assert score["coverage"]["requested_shots"] == 4
        assert score["coverage"]["supported_shots"] == 3
        assert score["crash_tolerance_2ms"]["shot_ids"] == [1, 2, 3]
        fixed = score["fixed_validation"]
        assert fixed["coverage"]["requested_shots"] == 2
        assert fixed["coverage"]["supported_shots"] == 1
        assert fixed["coverage"]["scored_shots"] == 1
        missing = next(row for row in fixed["by_shot"] if row["shot"] == 6)
        assert missing["outcome_status"] == "unknown"
        assert missing["observable_bins"] == missing["assessed_bins"] == 0
        assert missing["presence"] is missing["observable_picks"] is None
        assert missing["exclusion_reason"] == "unusable ECE core"
        assert score["expert"]["by_shot"][0]["shot"] == 5
    paired = scores["saw-ours"]["fixed_validation"]["paired_vs_derivative"]
    assert paired["crash_tolerance_2ms"]["bootstrap_replicates"] == 1000


def test_edge_context_sensitivity_vetoes_absence_only_near_qualified_edges():
    t = np.arange(2000) * 0.0001
    present = (t >= 0.04) & (t < 0.08)
    holdoff = (t < 0.1) & ~present
    signal = {
        "t": t,
        "observable": np.ones(len(t), bool),
        "absent_holdoff": holdoff,
        "assessed": present | holdoff,
    }
    rec = {
        "states": [{"start_s": 0.04, "end_s": 0.08, "state": "present"}],
        "uncertain_intervals": [],
        "absence_diagnostics": {
            "core_relaxation_test": {"ambiguous_edge_times_s": [0.09]}
        },
    }
    narrow = benchmark.with_edge_context(signal, rec, 5.15)["assessed"]
    wide = benchmark.with_edge_context(signal, rec, 50.0)["assessed"]
    widest = benchmark.with_edge_context(signal, rec, 375.0)["assessed"]
    assert narrow[present].all() and wide[present].all() and widest[present].all()
    assert narrow.sum() > wide.sum() > widest.sum() == present.sum()
    assert not narrow[np.abs(t - 0.09) < 0.005].any()
    assert narrow[t < 0.04].all() and not wide[(t >= 0.08) & (t < 0.1)].any()
    # An uncertain interval and unobservable time override tested absence.
    rec["uncertain_intervals"] = [{"start_s": 0.0, "end_s": 0.02}]
    signal["observable"] = signal["observable"] & ~((t >= 0.03) & (t < 0.035))
    both = benchmark.with_edge_context(signal, rec, 5.15)["assessed"]
    assert not both[t < 0.02].any() and not both[(t >= 0.03) & (t < 0.035)].any()
    with pytest.raises(ValueError, match="absent_holdoff"):
        benchmark.with_edge_context({"t": t}, rec, 5.15)
