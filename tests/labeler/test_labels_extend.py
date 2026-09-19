"""Export only listed shots and the chosen category/producer, with bounded CSV size."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from labeler.events import schema
from labeler.events.interval_tables import (
    INTERVAL_COLUMNS,
    read_label_grid,
    validate_intervals,
)

REPO = Path(__file__).resolve().parents[2]


def extend():
    return runpy.run_path(str(REPO / "scripts/labeler/labels_extend.py"))["main"]


def setup_export(tmp_path, producer="elm_clock"):
    events = tmp_path / "products/events"
    shot_list = tmp_path / "recommender_v1.yaml"
    shot_list.write_text(
        yaml.safe_dump(
            {
                "name": "recommender_v1",
                "n": 3,
                "shots": [{"shot": s} for s in (3, 1, 2)],
            }
        )
    )
    out = tmp_path / "edge_localized_mode" / f"extend_{producer}/recommender_v1.csv"
    argv = [
        "--category",
        "edge_localized_mode",
        "--producer",
        producer,
        "--shot-list",
        str(shot_list),
        "--events-root",
        str(events),
        "--out",
        str(out),
    ]
    return events, out, argv


def event(shot, t, *, source="elm_clock", phenomenon="elm", **over):
    return schema.Event(
        shot,
        source,
        phenomenon,
        t,
        t,
        evidence_kind="heuristic",
        t_cov0_s=0,
        t_cov1_s=6,
        **over,
    )


def test_extend_selects_source_and_category_and_preserves_provenance(tmp_path):
    events, out, argv = setup_export(tmp_path)
    schema.write_events(
        events / "1_events.parquet",
        1,
        [
            event(1, 2, attrs={"peak": 2}),
            event(1, 1),
            event(1, 3, source="other"),
            event(1, 4, phenomenon="tearing"),
        ],
        run_id="producer-run",
    )
    schema.write_events(
        events / "999_events.parquet", 999, [event(999, 2)], run_id="outside-list"
    )
    before = {p: p.read_bytes() for p in events.iterdir()}
    assert extend()(argv) == 0
    frame = pd.read_csv(out)
    validate_intervals(frame)
    assert frame.shot.tolist() == [1, 1]
    assert frame.t_start.tolist() == [1000, 2000]
    grid = read_label_grid(out.with_suffix("") / "1.npz")
    assert grid["label"].shape[1] == 20
    assert (grid["label"][grid["time_ms"] == 1000] == 1).all()
    assert (grid["label"][grid["time_ms"] == 1500] == 0).all()
    absent = read_label_grid(out.with_suffix("") / "2.npz")
    assert np.isnan(absent["label"]).all()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert (meta["n_rows"], meta["n_shots"], meta["n_requested_shots"]) == (2, 1, 3)
    assert meta["made_from"][0]["producer"] == "elm_clock"
    assert meta["made_from"][0]["run_id"] == "producer-run"
    assert meta["made_from"][0]["git_sha"] != "unknown"
    assert meta["missing_event_shots"]["count"] == 2
    assert meta["missing_event_shots"]["first_20"] == [2, 3]
    assert {p: p.read_bytes() for p in events.iterdir()} == before


def test_extend_refuses_a_phenomenon_spanning_sources_before_writing(tmp_path):
    events, out, argv = setup_export(tmp_path, producer="elm")
    schema.write_events(events / "1_events.parquet", 1, [event(1, 1)], run_id="first")
    schema.write_events(
        events / "2_events.parquet",
        2,
        [schema.Event(2, "text", "elm", 2, 2, evidence_kind="text")],
        run_id="second",
    )
    with pytest.raises(ValueError, match="spans sources.*elm_clock.*text"):
        extend()(argv)
    assert not out.parent.exists()


def test_single_source_phenomenon_requires_the_actual_source_directory(tmp_path):
    events, out, argv = setup_export(tmp_path, producer="elm")
    schema.write_events(
        events / "1_events.parquet", 1, [event(1, 1)], run_id="only-source"
    )
    with pytest.raises(ValueError, match="extend_elm_clock"):
        extend()(argv)
    assert not out.parent.exists()
    out = out.parent.parent / "extend_elm_clock/recommender_v1.csv"
    argv[-1] = str(out)
    assert extend()(argv) == 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert {r["producer"] for r in meta["made_from"]} == {"elm_clock"}


def test_missing_event_shots_are_bounded_with_a_complete_list_sidecar(tmp_path):
    _, out, argv = setup_export(tmp_path)
    shot_list = tmp_path / "recommender_v1.yaml"
    shot_list.write_text(yaml.safe_dump({"shots": list(range(1, 26))}))
    assert extend()(argv) == 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    missing = meta["missing_event_shots"]
    assert missing["count"] == 25
    assert missing["first_20"] == list(range(1, 21))
    full_list = out.parent / missing["full_list"]
    assert full_list.parent == out.parent
    assert json.loads(full_list.read_text()) == list(range(1, 26))


def test_zero_events_has_header_metadata_and_successful_zero_source_provenance(
    tmp_path,
):
    events, out, argv = setup_export(tmp_path)
    schema.write_sources(
        events / "1_sources.parquet",
        1,
        [
            {
                "source": "elm_clock",
                "status": "ok",
                "n_events": 0,
                "t_cov0_s": 0,
                "t_cov1_s": 5,
            }
        ],
        run_id="zero-run",
    )
    assert extend()(argv) == 0
    assert pd.read_csv(out).empty
    assert tuple(pd.read_csv(out).columns) == INTERVAL_COLUMNS
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_rows"] == meta["n_shots"] == 0
    assert meta["made_from"][0]["run_id"] == "zero-run"
    assert meta["source_status_counts"] == {"ok": 1}


@pytest.mark.parametrize("n", [50_000, 50_001])
def test_extend_keeps_all_intervals_and_sampled_values_at_any_size(tmp_path, n):
    events, out, argv = setup_export(tmp_path)
    schema.write_events(
        events / "1_events.parquet",
        1,
        [event(1, i / 10_000) for i in range(n)],
        run_id="many",
    )
    schema.write_sources(
        events / "2_sources.parquet",
        2,
        [
            {
                "source": "elm_clock",
                "status": "ok",
                "n_events": 0,
                "t_cov0_s": 0.5,
                "t_cov1_s": 5.5,
            }
        ],
        run_id="zero",
    )
    assert extend()(argv) == 0
    frame = pd.read_csv(out)
    assert len(frame) == n
    assert tuple(frame.columns) == INTERVAL_COLUMNS
    assert not out.with_suffix(".summary.csv").exists()
    grid = read_label_grid(out.with_suffix("") / "2.npz")
    known = (grid["time_ms"] >= 500) & (grid["time_ms"] <= 5500)
    assert (grid["label"][known] == 0).all()
    assert np.isnan(grid["label"][~known]).all()
    # A fresh export replaces prior positives and removes stale grid files.
    schema.write_events(
        events / "1_events.parquet", 1, [], sources=["elm_clock"], run_id="cleared"
    )
    assert extend()(argv) == 0
    assert pd.read_csv(out).empty
    assert np.isnan(read_label_grid(out.with_suffix("") / "1.npz")["label"]).all()


def test_extend_rejects_mismatched_output_directory_without_writes(tmp_path):
    _, out, argv = setup_export(tmp_path)
    argv[-1] = str(out.parent.parent / "extend_other/recommender_v1.csv")
    with pytest.raises(SystemExit):
        extend()(argv)
    assert not out.parent.parent.exists()


@pytest.mark.parametrize("status", ["ok", "error"])
def test_phenomenon_selection_preserves_sources_with_no_positive_events(
    tmp_path, status
):
    events, out, argv = setup_export(tmp_path, producer="elm")
    out = out.parent.parent / "extend_elm_clock/recommender_v1.csv"
    argv[-1] = str(out)
    schema.write_sources(
        events / "1_sources.parquet",
        1,
        [
            {
                "source": "elm_clock",
                "status": status,
                "n_events": 0,
                "reason": "failed to read" if status == "error" else "",
                "t_cov0_s": 0,
                "t_cov1_s": 5,
            }
        ],
        run_id="empty-producer",
    )
    assert extend()(argv) == 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["made_from"][0]["producer"] == "elm_clock"
    assert meta["made_from"][0]["run_id"] == "empty-producer"
    assert meta["source_status_counts"] == {status: 1}


def test_an_unrelated_successful_run_cannot_certify_a_zero_result(tmp_path):
    events, out, argv = setup_export(tmp_path)
    run_path = events.parent / "runs/events/unrelated.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps(
            {
                "run_id": "unrelated",
                "git_sha": "abc123",
                "settings": {},
                "shots": [{"shot": s, "status": "ok"} for s in (1, 2, 3)],
                "totals": {"tables": ["database:rwm_onsets_2017"], "n_events": 0},
            }
        )
    )
    with pytest.raises(ValueError, match="databases-only"):
        extend()([*argv, "--root", str(events.parent), "--run-id", "unrelated"])
    assert not out.exists()


def test_qmin_extension_uses_integer_classes_broadcast_over_rho(tmp_path):
    events = tmp_path / "products/events"
    shots = tmp_path / "shots.yaml"
    shots.write_text("shots: [1]\n")
    out = tmp_path / "minimum_safety_factor/extend_qmin_rule/shots.csv"
    schema.write_events(
        events / "1_events.parquet",
        1,
        [
            schema.Event(
                1, "qmin_rule", "qmin_hybrid", 1, 2, evidence_kind="heuristic"
            ),
            schema.Event(1, "qmin_rule", "qmin_high", 3, 4, evidence_kind="heuristic"),
        ],
        run_id="classes",
    )
    assert (
        extend()(
            [
                "--category",
                "minimum_safety_factor",
                "--producer",
                "qmin_rule",
                "--shot-list",
                str(shots),
                "--events-root",
                str(events),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    grid = read_label_grid(out.with_suffix("") / "1.npz")
    assert (grid["label"][grid["time_ms"] == 1500] == 2).all()
    assert (grid["label"][grid["time_ms"] == 3500] == 4).all()
    assert np.isnan(grid["label"][grid["time_ms"] == 2500]).all()
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["per_shot_files"]["classes"]["4"] == "high"
