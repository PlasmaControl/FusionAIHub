"""Coverage for the TRAIN-only edge-context derivation and its provenance."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/labeler"))
import sawtooth_edge_context as edge_context
import sawtooth_fix5_setup as setup
import sawtooth_radial_drop as radial
import sawtooth_reference_rule as reference


def test_choose_context_is_the_longest_candidate_keeping_half_the_time():
    rows = [
        {"context_ms": 375.0, "tested_absent_s": 19.0},
        {"context_ms": 5.15, "tested_absent_s": 100.0},
        {"context_ms": 50.0, "tested_absent_s": 74.0},
    ]
    assert edge_context.choose_context(rows) == 50.0
    by_context = {row["context_ms"]: row for row in rows}
    assert by_context[50.0]["retained_fraction"] == pytest.approx(0.74)
    assert [by_context[c]["passes"] for c in (5.15, 50.0, 375.0)] == [
        True,
        True,
        False,
    ]


def test_choose_context_refuses_when_no_tested_absence_remains():
    rows = [
        {"context_ms": 5.15, "tested_absent_s": 0.0},
        {"context_ms": 50.0, "tested_absent_s": 0.0},
    ]
    with pytest.raises(ValueError, match="no candidate"):
        edge_context.choose_context(rows)


def test_tested_absence_vetoes_only_near_edges_and_never_present_or_doubtful_time():
    t = np.arange(0, 2.0, 1e-4)
    signal = {
        "t": t,
        "observable": np.ones(len(t), dtype=bool),
        "absent_holdoff": np.ones(len(t), dtype=bool),
    }
    record = {
        "absence_diagnostics": {
            "core_relaxation_test": {"ambiguous_edge_times_s": [1.0]}
        },
        "states": [{"start_s": 0.0, "end_s": 0.5, "state": "present"}],
        "uncertain_intervals": [{"start_s": 1.5, "end_s": 1.6}],
    }
    short = edge_context.tested_absence(signal, record, 5.15)
    long = edge_context.tested_absence(signal, record, 375.0)
    assert not short[t < 0.5].any()
    assert not short[(t >= 1.5) & (t < 1.6)].any()
    assert not short[np.abs(t - 1.0) <= 0.005].any()
    assert short[(np.abs(t - 1.0) > 0.0054) & (np.abs(t - 1.0) < 0.1)].all()
    assert not long[np.abs(t - 1.0) <= 0.375].any()
    assert long.sum() < short.sum()
    assert edge_context.edges_near(t, [1.0], short, 0.375) == 1
    assert edge_context.edges_near(t, [1.0], long, 0.375) == 0


def test_freeze_refuses_a_value_the_derivation_did_not_choose():
    derivation = {
        "chosen_ms": 50.0,
        "derivation_shots": [1, 2],
        "criterion": "criterion text",
        "rows": [],
        "previous_exploration": {
            "shots": [{"shot": 1, "split": "train"}, {"shot": 9, "split": "val"}],
            "split_counts": {"train": 1, "val": 1},
        },
    }
    guard = setup.edge_context_guard(derivation, {"isolated_edge_context_ms": 50.0})
    assert guard["value"] == 50.0
    assert guard["shots"] == [1, 2]
    assert "val" in guard["evidence"] and "not carried forward" in guard["evidence"]
    with pytest.raises(ValueError, match="differs"):
        setup.edge_context_guard(derivation, {"isolated_edge_context_ms": 5.15})


def test_packaged_rule_and_derivation_record_agree_and_read_only_train():
    path = REPO / "outputs/labeler/sawtooth/fix5/edge_context_derivation.json"
    if not path.exists():
        pytest.skip("derivation record not generated yet")
    derivation = json.loads(path.read_text())
    packaged = json.loads(setup.PACKAGED.read_text())
    assert derivation["chosen_ms"] == packaged["rule"]["isolated_edge_context_ms"]
    assert derivation["val_or_test_shots_read"] == 0
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    train = set(cohort.loc[cohort.split == "train", "shot"].astype(int))
    assert set(derivation["derivation_shots"]) <= train
    splits = derivation["previous_exploration"]["split_counts"]
    assert sum(splits.values()) == 23 and splits["val"] == 3


def test_gate_attribution_names_each_reference_crashs_gate():
    found = SimpleNamespace(
        crashes=[SimpleNamespace(t0_s=2.80)],
        accepted_times_s=[2.80, 2.90],
        rejected_events=[(2.95, "central_relative_drop"), (2.75, "coincidence")],
    )
    picker = [2.5, 2.7501, 2.8, 2.9005, 2.9501, 2.99, 3.5]
    rows = reference.gate_attribution(found, picker, windows=((2.7, 3.0),))
    assert {row["time_s"]: row["gate"] for row in rows["crashes"]} == {
        2.7501: "coincidence",
        2.8: "found",
        2.9005: "train_test",
        2.9501: "central_relative_drop",
        2.99: "no_edge_cluster",
    }
    assert rows["counts"]["found"] == 1


def _dalpha(spikes):
    tx = np.arange(0, 2.0, 1e-4)
    values = 1.0 + np.random.default_rng(3).normal(0, 0.01, len(tx))
    for stamp in spikes:
        values[np.searchsorted(tx, stamp)] += 5.0
    return tx, values


def test_spike_within_needs_a_robust_excess_inside_one_millisecond():
    trace = _dalpha([1.0])
    assert radial.spike_within(trace, 1.0005) is True
    assert radial.spike_within(trace, 1.01) is False
    assert radial.spike_within((trace[0][:5], trace[1][:5]), 1.0) is None


def test_dalpha_coincidence_separates_elm_locked_events_from_chance(monkeypatch):
    events = [0.3, 0.6, 0.9, 1.2, 1.5]
    monkeypatch.setattr(radial, "dalpha_trace", lambda shot: _dalpha(events))
    locked = radial.dalpha_coincidence(1, events)
    assert locked["coincident_fraction"] == 1.0
    assert locked["chance_fraction"] == 0.0
    assert locked["marks_elms"] is True
    monkeypatch.setattr(radial, "dalpha_trace", lambda shot: _dalpha([]))
    quiet = radial.dalpha_coincidence(1, events)
    assert quiet["coincident_fraction"] == 0.0
    assert quiet["marks_elms"] is False
    monkeypatch.setattr(radial, "dalpha_trace", lambda shot: None)
    assert radial.dalpha_coincidence(1, events)["status"].startswith("no_filterscope")
