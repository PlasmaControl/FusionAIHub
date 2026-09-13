"""The committed format inputs retain the RWM CLI contract on both shot sets."""
import json
import runpy
from pathlib import Path

import pandas as pd
import yaml

from labelmaker import run
from labelmaker.events import databases as db
from labelmaker.events import schema

REPO = Path(__file__).resolve().parents[2]


def test_databases_only_on_all_committed_rwm_shots_writes_56_events_33_sources(
    tmp_path, monkeypatch,
):
    labels = REPO / "data/labels"
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(labels))
    shots = sorted(set().union(*(db.shots(s, labels) for s in db.load_manifest(labels))))
    assert run.main(["events", "--databases-only", "--root", str(tmp_path),
                     "--run-id", "rwm-integration", "--shots", *map(str, shots)]) == 0
    payload = json.loads((tmp_path / "runs/events/rwm-integration.json").read_text())
    assert payload["totals"]["n_events"] == 56
    assert payload["totals"]["n_source_records"] == 33
    assert payload["totals"]["n_shots_named"] == 33
    events = pd.concat([schema.read_events(tmp_path / f"events/{s}_events.parquet")
                        for s in shots])
    sources = pd.concat([schema.read_sources(tmp_path / f"events/{s}_sources.parquet")
                         for s in shots])
    assert len(events) == 56 and len(sources) == 33
    assert set(events.evidence_kind) == {"database"}
    assert events.confidence.isna().all()
    assert events.t_cov0_s.isna().all() and sources.t_cov1_s.isna().all()


def test_rwm_500_scan_exports_empty_table_with_completed_run_provenance(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(REPO / "data/labels"))
    shot_list = REPO / "configs/ideate/shot_lists/recommender_v1.yaml"
    shots = [r["shot"] for r in yaml.safe_load(shot_list.read_text())["shots"]]
    assert len(shots) == 500
    assert run.main(["events", "--databases-only", "--root", str(tmp_path),
                     "--run-id", "rwm-zero", "--shots", *map(str, shots)]) == 0
    main = runpy.run_path(str(REPO / "scripts/labelmaker/labels_extend.py"))["main"]
    out = tmp_path / "resistive_wall_mode/extend_rwm/recommender_v1.csv"
    assert main(["--category", "resistive_wall_mode", "--producer", "rwm",
                 "--shot-list", str(shot_list), "--root", str(tmp_path),
                 "--run-id", "rwm-zero", "--out", str(out)]) == 0
    assert pd.read_csv(out).empty
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_requested_shots"] == 500 and meta["n_rows"] == 0
    assert meta["made_from"][0]["run_id"] == "rwm-zero"
    assert meta["made_from"][0]["producer"] == "rwm"
    assert not list((tmp_path / "events").glob("*.parquet"))
