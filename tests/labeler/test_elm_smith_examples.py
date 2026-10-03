"""Smith metric conditioning and transparent example selection/scored-time audits."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.elm import inputs, labels


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts/labeler" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smith_evaluate = load_script("elm_smith_evaluate")
examples = load_script("elm_example_figure")


@pytest.mark.parametrize("with_interval", [True, False])
def test_window_precision_is_retained_but_population_precision_is_unknown(
    with_interval,
):
    summary = {"point": {"precision": 0.8, "recall": 0.9, "f1": 0.847}}
    if with_interval:
        summary["ci95"] = {"precision": [0.7, 0.9], "f1": [0.8, 0.9]}
    result = smith_evaluate.condition_selected_window_metrics(summary)
    assert result["point"]["precision"] == 0.8
    assert result["point"]["f1"] == 0.847
    assert result["selected_window_precision_f1"]["precision"] == {
        "point": 0.8,
        "ci95": [0.7, 0.9] if with_interval else None,
    }
    assert "conditional" in result["metric_scope"]
    for key in ("precision", "f1"):
        assert result["continuous_discharge_precision_f1"][key] == {
            "point": None,
            "ci95": None,
        }


def test_calendar_day_sharing_uses_outer_test_membership_and_records_missing_dates():
    audit = smith_evaluate.onset_run_day_audit(
        {"1": "20250101", "2": "20250101", "3": "20250102", "4": None},
        [{"fold": 0, "test": [1, 3]}, {"fold": 1, "test": [2, 4]}],
    )
    assert audit["run_days"] == 2
    assert audit["run_days_spanning_multiple_folds"] == 1
    assert audit["shared_run_days"] == ["20250101"]
    assert audit["shots_missing_run_day"] == [4]
    assert audit["memberships"][0] == {
        "run_day": "20250101",
        "shots": [1, 2],
        "folds": [0, 1],
    }


def test_example_replacement_preserves_original_selection_history(monkeypatch):
    shots = list(range(100, 138))
    shots[9], shots[10], shots[28] = 200427, 203941, 195111
    ranks = {shot: rank for rank, shot in enumerate(shots)}

    def part(spans, shot, bins, cover, trace, threshold):
        truth = np.r_[np.ones(40, bool), np.zeros(40, bool)]
        call = np.arange(80) < ranks[shot] + 1
        return SimpleNamespace(truth=truth, call=call)

    monkeypatch.setattr(examples.methods, "trace_part", part)
    data = {shot: SimpleNamespace(spans=None) for shot in shots}
    bes = SimpleNamespace(
        shots=shots,
        bins=dict.fromkeys(shots),
        cover=dict.fromkeys(shots),
    )
    oof = SimpleNamespace(trace=lambda shot: [None], threshold=dict.fromkeys(shots))
    selected, audit = examples.pick_shots(data, {"bes73": bes}, oof)
    assert selected == [195111, 203941]
    assert audit["replaced_panel_b"]["shot"] == 200427
    assert audit["replaced_panel_b"]["rank"] == 9
    assert audit["replacement_rank"] == 10
    assert "25th percentile" in audit["original_rule"]
    assert "reviewer" in audit["rule"]


def test_displayed_disagreement_can_be_inside_scored_time_at_review_boundary():
    spans = pd.DataFrame(
        {"t_start": [21.0, 328.0], "t_end": [328.0, 750.0], "kind": ["absent", "crowd"]}
    )
    bins = labels.scored_bins(spans, np.array([200.0]), np.array([800.0]))
    event = np.zeros(850)
    start = int(650 - inputs.GRID0_MS)
    event[start : start + 50] = 0.5
    elmo = {
        1: pd.DataFrame(
            {
                "t_start_ms": np.arange(375.0, 700.0, 50.0),
                "t_end_ms": np.arange(375.0, 700.0, 50.0) + 0.1,
            }
        )
    }
    audit = examples.scored_window_audit(1, spans, bins, event, 0.3, elmo, 320, 700)
    assert audit["ours_false_negative_bins"] == 6
    assert audit["elmo_positive_bins"] == 7
    assert audit["excluded_display_grid_bins_ms"] == [[300.0, 350.0]]
    assert [row["span_ms"][0] for row in audit["scored_bins"]] == list(
        np.arange(350.0, 700.0, 50.0)
    )
    assert audit["scored_bins"][-1]["ours_call"]
