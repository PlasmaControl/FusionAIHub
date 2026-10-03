"""Physics/source and blind-split regressions for the detachment review queue."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.events import rosters
from labeler.events.review import detachment, geometry


def roster_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/labeler/detachment_review_roster.py"
    )
    spec = importlib.util.spec_from_file_location("detachment_review_roster", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shelf_gate_matches_producer_bounds_and_sentinels():
    rv = [1.4, 1.36, 1.4, 1.4, -0.89, 1.4, 1.4, 1.37]
    zv = [-1.25, -1.363, -1.25, -1.1, -1.25, -1.25, -1.25, -1.25]
    rx = [1.3] * 8
    zx = [-1.0, -1.0, 1.0, -1.0, -1.0, -9.99, -0.5, -1.0]
    valid, reasons = geometry.shelf_gate(rv, zv, rx, zx)
    assert valid.tolist() == [True, False, False, False, False, False, False, True]
    assert reasons.tolist() == [
        "",
        "strike_on_floor",
        "not_lower_null",
        "strike_not_on_shelf",
        "efit_missing",
        "efit_missing",
        "not_lower_null",
        "",
    ]


def test_topology_does_not_guess_double_null_or_limited():
    config = geometry.configurations(
        [1.4, 1.4, 1.4, 1.4, 1.4, -9.99, 1.4],
        [-1.0, 1.0, -1.0, -1.0, -1.0, -9.99, -1.0],
        [-9.99, -9.99, 1.4, 1.4, 1.4, -9.99, 1.4],
        [-9.99, -9.99, 1.0, 1.0, 1.0, -9.99, 1.0],
        [np.nan, np.nan, np.nan, 0.005, 0.03, np.nan, 0.4],
    )
    assert config.tolist() == [
        "LSN",
        "USN",
        "unknown",
        "DN",
        "LSN",
        "unknown",
        "unknown",
    ]


def test_nearest_enforces_gap_and_does_not_fill_missing_values():
    actual = geometry.nearest([100, 0, 200], [1, 0, np.nan], [50, 60, 140, 200, 241])
    np.testing.assert_allclose(actual, [np.nan, 1, 1, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(geometry.nearest([0, 80], [1, 2], [40]), [1])
    assert np.isnan(geometry.nearest([0], [1, 2], [0])).all()


def test_geometry_load_and_camera_gate_use_local_clock(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    root = tmp_path / "cached_efit"
    root.mkdir()
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(root))
    arrays = {}
    for key, values in {
        "rvsod": [1.4, 1.36],
        "zvsod": [-1.25, -1.363],
        "rxpt1": [1.3, 1.3],
        "zxpt1": [-1.0, -1.0],
    }.items():
        arrays[key + "__t"] = [100, 200]
        arrays[key + "__y"] = values
    np.savez(root / "190109.npz", **arrays)
    meta = geometry.load(190109, paths, [50, 250])
    assert meta["total_samples"] == 2
    assert meta["shelf_gate_samples"] == 1
    assert meta["counts"]["LSN"] == 2
    assert meta["counts"]["limited"] == 0
    configurations, valid = geometry.at_times(meta, [100, 140, 150, 200, 250])
    assert valid.tolist() == [True, True, False, False, False]
    assert configurations.tolist() == ["LSN", "LSN", "unknown", "LSN", "unknown"]
    assert geometry.load(190110, paths)["reason"] == "efit_missing"
    paths.corpus.mkdir()
    with h5py.File(paths.corpus_file(190109), "w") as source:
        camera = source.create_group("tangtv")
        camera.create_dataset("xdata", data=[0.1, 0.2, 0.3])
        camera.create_dataset("ydata", data=np.arange(144).reshape(3, 3, 4, 4))
    cohort = pd.DataFrame(
        {
            "shot": [190109],
            "split": ["train"],
            "queue_rank": [1],
            "window_start_ms": [50],
            "window_end_ms": [250],
        }
    )
    previous = [{"shot": 190109, "lower_channels": [], "reason": "old reader stub"}]
    refreshed = roster_module().scan(paths, cohort, previous)
    assert refreshed[0]["lower_channels"]


def test_queue_union_excludes_cohort_and_explicit_producer_test():
    module = roster_module()
    cohort = pd.DataFrame(
        {
            "shot": [190001, 190002, 190003],
            "split": ["train", "val", "test"],
            "window_start_ms": [0, 0, 0],
            "window_end_ms": [1000, 1000, 1000],
        }
    )
    records = [
        {"shot": shot, "split": "train", "lower_channels": [0]}
        for shot in [190001, 190003]
    ]
    producer = {
        "shots": [190001, 190002, 190003, 190004, 190005],
        "explicit_test_shots": [190005],
    }
    queue, excluded = module.queue_records(records, producer, cohort)
    assert [r["shot"] for r in queue] == [190001, 190002, 190004]
    assert queue[-1]["split"] == "producer_external"
    assert len(queue[0]["queue_sources"]) == 2
    assert excluded == [190003, 190005]


def test_producer_discovery_includes_draft_votes_and_ignores_rosters(tmp_path):
    module = roster_module()
    (tmp_path / "bins").mkdir()
    np.savez(tmp_path / "bins/190001.npz", tangtv_vote=[1, 2])
    np.savez(tmp_path / "bins/190009.npz", ze=[1, 2])
    pd.DataFrame(
        {"shot": [190002, 190003], "category": [1, 2], "split": ["val", "test"]}
    ).to_csv(tmp_path / "labels.csv", index=False)
    pd.DataFrame({"shot": [1, 2, 3], "tier": ["gold"] * 3}).to_csv(
        tmp_path / "shots.csv", index=False
    )
    snapshot = module.producer_snapshot([tmp_path])
    assert snapshot["shots"] == [190001, 190002, 190003]
    assert snapshot["explicit_test_shots"] == [190003]
    assert len(snapshot["sources"]) == 2
    assert not snapshot["errors"]


def test_bulk_producer_predictions_preserve_explicit_test_split(tmp_path):
    module = roster_module()
    np.savez(
        tmp_path / "predictions.npz",
        prob=[[0.8, 0.2], [0.1, 0.9]],
        shot=[190001, 190002],
        split=["train", "test"],
    )
    snapshot = module.producer_snapshot([tmp_path])
    assert snapshot["shots"] == [190001, 190002]
    assert snapshot["explicit_test_shots"] == [190002]
    assert json.loads(json.dumps(snapshot))["shots"] == [190001, 190002]


def test_partial_producer_npz_is_recorded_and_geometry_stays_unknown(
    tmp_path, monkeypatch
):
    module = roster_module()
    path = tmp_path / "190001.npz"
    path.write_bytes(b"PK\x03\x04partial ZIP output")
    (tmp_path / "labels.csv.gz").write_bytes(
        gzip.compress(b"shot,category\n190001,1\n")[:-5]
    )
    snapshot = module.producer_snapshot([tmp_path])
    assert snapshot["shots"] == []
    assert len(snapshot["errors"]) == 2
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(tmp_path))
    assert geometry.load(190001, Paths(root=tmp_path))["reason"].startswith(
        "efit_unreadable"
    )


def test_context_sources_fingerprint_existing_inputs(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path)
    root = tmp_path / "indicators"
    root.mkdir()
    monkeypatch.setenv("LABELER_DETACHMENT_INDICATORS", str(root))
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(root))
    assert all(
        value["sha256"] is None
        for value in detachment.context_sources(190001, paths).values()
    )
    np.savez(root / "190001.npz", start_ms=[0, 50])
    initial = detachment.context_sources(190001, paths)
    assert initial["geometry"]["sha256"] == initial["indicators"]["sha256"]
    np.savez(root / "190001.npz", start_ms=[0, 100])
    assert detachment.context_sources(190001, paths) != initial


def test_cached_plasma_window_uses_producer_current_rule(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path)
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(tmp_path))
    shot = 190001
    assert detachment.plasma_window(shot, paths) is None
    np.savez(
        tmp_path / f"{shot}.npz",
        ipmeas__t=[-100, 0, 100, 200, 300, 400, 500],
        ipmeas__y=[np.nan, 299999, 300000, -600000, 600000, 300000, 200000],
    )
    assert detachment.plasma_window(shot, paths) == (100.0, 400.0)
    fingerprint = detachment.context_sources(shot, paths)["plasma_window"]
    assert fingerprint["window_ms"] == [100.0, 400.0]
    assert fingerprint["min_abs_ip_a"] == 300000
    # The catalog's established window remains authoritative when present.
    monkeypatch.setattr(
        detachment.panels.detachment, "plasma_window", lambda *_: (0, 500)
    )
    assert detachment.plasma_window(shot, paths) == (0, 500)
    assert "plasma_window" not in detachment.context_sources(shot, paths)


def test_cached_plasma_window_rejects_incomplete_or_nonphysical_cache(
    tmp_path, monkeypatch
):
    paths = Paths(root=tmp_path)
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(tmp_path))
    path = tmp_path / "190001.npz"
    for arrays in (
        {"ipmeas__t": [0, 100]},
        {"ipmeas__t": [0, 100], "ipmeas__y": [200000, 200000]},
        {"ipmeas__t": [np.nan, 100], "ipmeas__y": [600000, 600000]},
        {"ipmeas__t": [0, 100], "ipmeas__y": [600000]},
    ):
        np.savez(path, **arrays)
        assert detachment.plasma_window(190001, paths) is None
    path.write_bytes(b"PK\x03\x04partial cache")
    assert detachment.plasma_window(190001, paths) is None


def test_roster_rerun_preserves_curation_and_does_not_accumulate_notes(tmp_path):
    module = roster_module()
    path = tmp_path / "shots.csv"
    pd.DataFrame(
        [[190001, "silver", "false", "expert", "2026-10-03", "checked"]],
        columns=rosters.ROSTER_COLUMNS,
    ).to_csv(path, index=False)
    queue = [
        {
            "shot": 190001,
            "split": "train",
            "queue_sources": ["producer_labels_or_votes"],
        },
        {"shot": 190002, "split": "val", "queue_sources": ["cohort_camera_geometry"]},
    ]
    first = module.roster_frame(queue, path)
    rosters.write_roster(first, path)
    second = module.roster_frame(queue, path)
    pd.testing.assert_frame_equal(first, second)
    assert second.iloc[0].reviewers == "expert"
    assert second.iloc[0].tier == "silver"
