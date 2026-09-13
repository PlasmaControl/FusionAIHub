"""Export only listed shots and the chosen category/producer, with bounded CSV size."""
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from labelmaker.events import databases as db
from labelmaker.events import schema

REPO = Path(__file__).resolve().parents[2]


def extend():
    return runpy.run_path(str(REPO / "scripts/labelmaker/labels_extend.py"))["main"]


def setup_export(tmp_path, producer="elm_clock"):
    events = tmp_path / "products/events"
    shot_list = tmp_path / "recommender_v1.yaml"
    shot_list.write_text(yaml.safe_dump({"name": "recommender_v1", "n": 3,
                                       "shots": [{"shot": s} for s in (3, 1, 2)]}))
    out = tmp_path / "edge_localized_mode" / f"extend_{producer}/recommender_v1.csv"
    argv = ["--category", "edge_localized_mode", "--producer", producer,
            "--shot-list", str(shot_list), "--events-root", str(events),
            "--out", str(out)]
    return events, out, argv


def event(shot, t, *, source="elm_clock", phenomenon="elm", **over):
    return schema.Event(shot, source, phenomenon, t, t, evidence_kind="heuristic",
                        t_cov0_s=0, t_cov1_s=6, **over)


def test_extend_selects_source_and_category_and_preserves_provenance(tmp_path):
    events, out, argv = setup_export(tmp_path)
    schema.write_events(events / "1_events.parquet", 1, [
        event(1, 2, attrs={"peak": 2}), event(1, 1),
        event(1, 3, source="other"), event(1, 4, phenomenon="tearing"),
    ], run_id="producer-run")
    schema.write_events(events / "999_events.parquet", 999, [event(999, 2)],
                        run_id="outside-list")
    before = {p: p.read_bytes() for p in events.iterdir()}
    assert extend()(argv) == 0
    frame = pd.read_csv(out)
    db.validate_format(frame)
    assert frame.shot.tolist() == [1, 1]
    assert frame.t0_s.tolist() == [1, 2]
    assert frame.evidence_kind.tolist() == ["heuristic", "heuristic"]
    assert json.loads(frame["attrs"].iloc[1]) == {"peak": 2}
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert (meta["n_rows"], meta["n_shots"], meta["n_requested_shots"]) == (2, 1, 3)
    assert meta["made_from"][0]["producer"] == "elm_clock"
    assert meta["made_from"][0]["run_id"] == "producer-run"
    assert meta["made_from"][0]["git_sha"] != "unknown"
    assert meta["missing_event_shots"] == [2, 3]
    assert {p: p.read_bytes() for p in events.iterdir()} == before


def test_extend_accepts_a_phenomenon_and_keeps_each_actual_source(tmp_path):
    events, out, argv = setup_export(tmp_path, producer="elm")
    schema.write_events(events / "1_events.parquet", 1,
                        [event(1, 1), event(1, 2, source="other")], run_id="first")
    schema.write_events(events / "2_events.parquet", 2,
                        [event(2, 1)], run_id="second")
    assert extend()(argv) == 0
    assert set(pd.read_csv(out).source) == {"elm_clock", "other"}
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert {row["run_id"] for row in meta["made_from"]} == {"first", "second"}


def test_zero_events_has_header_metadata_and_successful_zero_source_provenance(tmp_path):
    events, out, argv = setup_export(tmp_path)
    schema.write_sources(events / "1_sources.parquet", 1, [{
        "source": "elm_clock", "status": "ok", "n_events": 0,
        "t_cov0_s": 0, "t_cov1_s": 5,
    }], run_id="zero-run")
    assert extend()(argv) == 0
    assert pd.read_csv(out).empty
    assert tuple(pd.read_csv(out).columns) == db.FORMAT_COLUMNS
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_rows"] == meta["n_shots"] == 0
    assert meta["made_from"][0]["run_id"] == "zero-run"
    assert meta["source_status_counts"] == {"ok": 1}


@pytest.mark.parametrize("n, summary", [(50_000, False), (50_001, True)])
def test_extend_summary_threshold_counts_events_and_uses_source_coverage(
    tmp_path, n, summary,
):
    events, out, argv = setup_export(tmp_path)
    schema.write_events(events / "1_events.parquet", 1,
                        [event(1, i / 10_000) for i in range(n)], run_id="many")
    schema.write_sources(events / "1_sources.parquet", 1, [{
        "source": "elm_clock", "status": "ok", "n_events": n,
        "t_cov0_s": 0, "t_cov1_s": 6,
    }], run_id="many")
    schema.write_sources(events / "2_sources.parquet", 2, [{
        "source": "elm_clock", "status": "ok", "n_events": 0,
        "t_cov0_s": 0.5, "t_cov1_s": 5.5,
    }, {"source": "unrelated", "status": "ok", "n_events": 0,
        "t_cov0_s": -10, "t_cov1_s": 100}], run_id="zero")
    assert extend()(argv) == 0
    actual = out.with_suffix(".summary.csv") if summary else out
    assert actual.exists()
    assert out.exists() is (not summary)
    frame = pd.read_csv(actual)
    meta = json.loads(actual.with_suffix(".meta.json").read_text())
    assert meta["n_events"] == n
    if summary:
        assert list(frame.columns) == ["shot", "n_events", "t_first_s", "t_last_s",
                                       "t_cov0_s", "t_cov1_s"]
        assert frame.shot.tolist() == [1, 2, 3]
        assert frame.n_events.tolist() == [50_001, 0, 0]
        assert frame.t_first_s.iloc[0] == 0
        assert frame.t_last_s.iloc[0] == 5
        assert frame.t_cov1_s.iloc[1] == 5.5
        assert frame.t_first_s.iloc[1:].isna().all()
        assert pd.isna(frame.t_cov1_s.iloc[2])
        assert meta["table_kind"] == "per_shot_summary"
        assert meta["full_events_root"] == str(events)
        assert "$LABELMAKER_ROOT/events" in meta["full_events"]
        # Re-running below the threshold removes the stale summary and sidecar.
        schema.write_events(events / "1_events.parquet", 1, [],
                            sources=["elm_clock"], run_id="cleared")
        assert extend()(argv) == 0
        assert out.exists() and not actual.exists()
        assert not actual.with_suffix(".meta.json").exists()
    else:
        assert len(frame) == 50_000


def test_extend_rejects_mismatched_output_directory_without_writes(tmp_path):
    _, out, argv = setup_export(tmp_path)
    argv[-1] = str(out.parent.parent / "extend_other/recommender_v1.csv")
    with pytest.raises(SystemExit):
        extend()(argv)
    assert not out.parent.parent.exists()


@pytest.mark.parametrize("status", ["ok", "error"])
def test_phenomenon_selection_preserves_sources_with_no_positive_events(tmp_path, status):
    events, out, argv = setup_export(tmp_path, producer="elm")
    schema.write_sources(events / "1_sources.parquet", 1, [{
        "source": "elm_clock", "status": status, "n_events": 0,
        "reason": "failed to read" if status == "error" else "",
        "t_cov0_s": 0, "t_cov1_s": 5,
    }], run_id="empty-producer")
    assert extend()(argv) == 0
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["made_from"][0]["producer"] == "elm_clock"
    assert meta["made_from"][0]["run_id"] == "empty-producer"
    assert meta["source_status_counts"] == {status: 1}


def test_an_unrelated_successful_run_cannot_certify_a_zero_result(tmp_path):
    events, out, argv = setup_export(tmp_path)
    run_path = events.parent / "runs/events/unrelated.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(json.dumps({
        "run_id": "unrelated", "git_sha": "abc123", "settings": {},
        "shots": [{"shot": s, "status": "ok"} for s in (1, 2, 3)],
        "totals": {"tables": ["database:rwm_onsets_2017"], "n_events": 0},
    }))
    with pytest.raises(ValueError, match="databases-only"):
        extend()([*argv, "--root", str(events.parent), "--run-id", "unrelated"])
    assert not out.exists()
