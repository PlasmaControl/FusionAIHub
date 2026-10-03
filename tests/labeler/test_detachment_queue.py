"""Review delivery excludes blind shots and keeps producer inputs read-only."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events import rosters
from labeler.events.review import labels


def roster_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/labeler/detachment_review_roster.py"
    )
    spec = importlib.util.spec_from_file_location("detachment_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_prioritizes_camera_shelf_coverage_over_shot_number():
    records = [
        {"shot": 190001, "lower_channels": [{}], "camera_geometry_eligible": False},
        {"shot": 200977, "lower_channels": [{}], "camera_geometry_eligible": True},
    ]
    cohort = pd.DataFrame({"shot": [190001, 200977], "split": ["train", "val"]})
    queue, excluded = roster_module().queue_records(
        records, {"shots": [], "explicit_test_shots": []}, cohort
    )
    assert [r["shot"] for r in queue] == [200977, 190001]
    assert not excluded


def test_recipe_snapshot_is_written_under_output(tmp_path, monkeypatch):
    module = roster_module()
    method = tmp_path / "method.md"
    method.write_text("Producer definitions")
    out = tmp_path / "out"
    monkeypatch.setenv("LABELER_DETACHMENT_METHOD", str(method))
    module.write_recipe_help(tmp_path / "labels.csv", out)
    assert "Producer definitions" in (out / "producer_recipe.md").read_text()


def test_reordering_frozen_queue_preserves_curation_and_does_not_read_producer(
    tmp_path, monkeypatch
):
    from labeler.config import sha256_of

    module = roster_module()
    repo = tmp_path / "repo"
    cohort = repo / "data/events/catalog/cohort.csv"
    cohort.parent.mkdir(parents=True)
    cohort.write_text("shot,split\n190001,train\n200977,val\n")
    monkeypatch.setattr(module, "REPO", repo)
    event = tmp_path / "out/tables/detachment"
    event.mkdir(parents=True)
    (event / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n"
        "190001,unverified,false,Alice,2026-10-03,keep this note\n"
        "200977,gold,false,Bob,2026-10-03,reviewed\n"
    )
    (tmp_path / "out/corpus_scan.json").write_text(
        json.dumps(
            {
                "cohort_sha256": sha256_of(cohort),
                "records": [
                    {
                        "shot": 190001,
                        "lower_channels": [{}],
                        "camera_geometry_eligible": False,
                    },
                    {
                        "shot": 200977,
                        "lower_channels": [{}],
                        "camera_geometry_eligible": True,
                    },
                ],
            }
        )
    )
    record = module.reorder_frozen_queue(tmp_path / "out", tmp_path / "order.json")
    result = rosters.read_roster(event / "shots.csv")
    assert result.shot.tolist() == [200977, 190001]
    assert result.iloc[0].reviewers == "Bob"
    assert result.iloc[1].notes == "keep this note"
    assert record["shelf_covered_first"] and record["store_rebuilds"] == 0


def test_delivered_source_clips_to_window_and_retains_original_bin_bounds(tmp_path):
    source = tmp_path / "labels.csv"
    pd.DataFrame(
        {"shot": [202205, 202205], "start_ms": [3100, 3150], "state_lm": [2, 2]}
    ).to_csv(source, index=False)
    event = tmp_path / "delivery/tables/detachment"
    result = roster_module().snapshot_suggestions(
        source, [202205], event, windows={202205: [120, 3134]}
    )
    frame = pd.read_csv(result["table"])
    assert frame.t_end.tolist() == [3134]
    attrs = json.loads(frame["attrs"].iloc[0])
    assert attrs["producer_bin_start_ms"] == 3100
    assert attrs["producer_bin_end_ms"] == 3150


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("flag,value", [("split", "test"), ("holdout", "true")])
def test_per_shot_csv_preserves_blind_flags(tmp_path, compressed, flag, value):
    payload = f"category,{flag}\n2,{value}\n".encode()
    suffix = ".csv.gz" if compressed else ".csv"
    path = tmp_path / f"190001{suffix}"
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    result = roster_module().producer_snapshot([tmp_path])
    assert result["shots"] == [190001]
    assert result["explicit_test_shots"] == [190001]


def test_bulk_csv_and_npz_holdout_flags_reserve_whole_shot(tmp_path):
    pd.DataFrame(
        {"shot": [190001, 190002], "state_lm": [1, 2], "holdout": [True, False]}
    ).to_csv(tmp_path / "labels.csv", index=False)
    np.savez(
        tmp_path / "predictions.npz",
        shot=[190003, 190004],
        state=[1, 2],
        holdout=[False, True],
    )
    result = roster_module().producer_snapshot([tmp_path])
    assert result["explicit_test_shots"] == [190001, 190004]


def test_ambiguous_npz_split_lengths_are_recorded_instead_of_admitted(tmp_path):
    np.savez(
        tmp_path / "predictions.npz",
        shot=[190001, 190002, 190003],
        state=[1, 2, 3],
        split=["train", "test"],
    )
    result = roster_module().producer_snapshot([tmp_path])
    assert result["shots"] == []
    assert len(result["errors"]) == 1


@pytest.mark.parametrize(
    "density,want",
    [
        ({"quantity": "aux_ne"}, "aux_ne"),
        ({"node": "DENR0UF"}, "cached_DENR0UF"),
        ({"corpus_group": "co2"}, "corpus_CO2"),
        ({"corpus_group": "ts_core_density"}, "Thomson"),
        ({}, "unavailable"),
    ],
)
def test_build_evidence_reports_density_hierarchy_and_valid_front_sources(
    density, want
):
    result = roster_module().context_summary(
        {
            "p0": density,
            "p1": {"quantity": "aux_te_div"},
            "p2": {
                "indicator": "tangtv",
                "tangtv_source": ["inversion", "surrogate", "surrogate"],
                "valid": [True, True, False],
            },
        },
        {"bin_start_ms": [0, 50, 100], "state_lm": [1, 4, 0]},
    )
    assert result["density_source"] == want
    assert result["divertor_te_available"] is True
    assert result["tangtv_valid_source_bins"] == {"inversion": 1, "surrogate": 1}
    assert result["producer_strip_bins"] == 3


def test_context_evidence_records_why_a_producer_strip_is_unavailable():
    result = roster_module().context_summary(
        {}, {"reason": "Producer snapshot disagrees on afrac_vote"}
    )
    assert result["producer_strip_bins"] == 0
    assert (
        result["producer_strip_reason"] == "Producer snapshot disagrees on afrac_vote"
    )


def camera(paths, shot, live=True):
    paths.corpus.mkdir(exist_ok=True)
    with h5py.File(paths.corpus_file(shot), "w") as source:
        group = source.create_group("tangtv")
        group.create_dataset("xdata", data=[0.1, 0.2] if live else [0.1])
        frames = np.arange(96).reshape(3, 2, 4, 4)
        group.create_dataset("ydata", data=frames if live else frames[:, :1])


def test_resume_refreshes_only_recipe_metadata_without_reencoding_frames(tmp_path):
    paths = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus")
    camera(paths, 190001)
    record = {"shot": 190001, "split": "train"}
    module = roster_module()
    first = module.build_store(record, paths, tmp_path / "bins", resume=True)
    with h5py.File(first["store"]) as source:
        before = source["videos/tangtv/2/frames"][:]
    recipe_path = paths.root / "round4/detach/review_recipe.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps({"method": "updated producer gates"}))
    second = module.build_store(record, paths, tmp_path / "bins", resume=True)
    assert second["action"] == "refreshed recipe"
    with h5py.File(second["store"]) as source:
        np.testing.assert_array_equal(source["videos/tangtv/2/frames"][:], before)
        params = json.loads(source.attrs["params"])
        assert params["detachment_producer"]["recipe"]["record"] == {
            "method": "updated producer gates"
        }
    third = module.build_store(record, paths, tmp_path / "bins", resume=True)
    assert third["action"] == "kept current store"


def test_camera_scan_does_not_hide_missing_efit(tmp_path, monkeypatch):
    module = roster_module()
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(tmp_path / "cache"))
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    camera(paths, 190001)
    camera(paths, 190002, live=False)
    cohort = pd.DataFrame(
        {
            "shot": [190001, 190002],
            "split": ["train", "val"],
            "queue_rank": [1, 2],
            "window_start_ms": [0, 0],
            "window_end_ms": [300, 300],
        }
    )
    first, second = module.scan(paths, cohort)
    assert first["camera_available"] is True
    assert first["geometry_reason"] == "efit_missing"
    assert first["camera_geometry_eligible"] is False
    assert second["camera_available"] is False
    assert "stub" in second["reason"]


def test_queue_excludes_no_video_by_default_and_orders_flagged_fallback_last():
    module = roster_module()
    records = [
        {"shot": 190003, "split": "train", "lower_channels": [2]},
        {"shot": 190002, "split": "val", "lower_channels": []},
        {"shot": 190004, "split": "test", "lower_channels": [2]},
    ]
    producer = {
        "shots": [190002, 190003, 190004, 190005],
        "roster_shots": [190001],
        "explicit_test_shots": [190005],
    }
    cohort = pd.DataFrame(
        {
            "shot": [190002, 190003, 190004],
            "split": ["val", "train", "test"],
            "window_start_ms": [0, 0, 0],
            "window_end_ms": [300, 300, 300],
        }
    )
    queue, excluded = module.queue_records(records, producer, cohort)
    assert [row["shot"] for row in queue] == [190003]
    assert excluded == [190004, 190005]
    queue, _ = module.queue_records(records, producer, cohort, include_no_video=True)
    assert [row["shot"] for row in queue] == [190003, 190001, 190002]
    assert queue[-1]["camera_available"] is False


def test_suggestion_snapshot_uses_primary_states_and_excludes_all_blind_flags(
    tmp_path,
):
    source = tmp_path / "labels_bins.csv.gz"
    pd.DataFrame(
        {
            "shot": [190001, 190001, 190001, 190002, 190003],
            "start_ms": [100, 150, 250, 0, 0],
            "state_lm": [1, 2, 0, 3, 2],
            "state_rule": [4, 4, 0, 1, 1],
            "split": ["train", "train", "train", "test", "train"],
            "holdout": [False, False, False, False, True],
            "confidence": [0.8, 0.9, np.nan, 0.9, 0.9],
        }
    ).to_csv(source, index=False)
    event = tmp_path / "delivery/tables/detachment"
    result = roster_module().snapshot_suggestions(
        source, [190001, 190002, 190003], event
    )
    assert result["shots"] == [190001]
    assert result["explicit_test_shots"] == [190002, 190003]
    assert labels.source_path(event) == event / "review/suggestions.csv"
    assert labels.read_source(event)[190001].intervals == (
        (100, 150, 1),
        (150, 200, 2),
    )
    assert labels.read_source(event)[190001].window == (100, 300)
    pointer = json.loads((event / "review/source.json").read_text())
    assert pointer["producer_sha256"] == result["producer_sha256"]


def test_source_snapshot_excludes_stale_votes_and_keeps_matching_or_legacy_labels(
    tmp_path,
):
    data = {
        "start_ms": np.array([100, 150]),
        "tangtv_source": np.array(["none", "none"]),
    }
    for name in ("afrac", "prad", "tangtv"):
        live = name != "tangtv"
        data[f"{name}_valid"] = np.array([live, live])
        data[f"{name}_vote"] = np.array([2, 1] if live else [-1, -1])
        data[f"{name}_reason"] = np.array(["", ""] if live else ["no_frames"] * 2)
        data[f"{name}_value"] = np.array([0.4, 0.8] if live else [np.nan, np.nan])
    published = []
    for shot in (190001, 190002, 190004):
        shot_frame = pd.DataFrame(data)
        shot_frame["shot"] = shot
        shot_frame["state_lm"] = [2, 1]
        shot_frame["state_rule"] = [2, 1]
        shot_frame["confidence"] = [0.8, 0.9]
        published.append(shot_frame)
    source = tmp_path / "labels_bins.csv.gz"
    pd.concat(published, ignore_index=True).to_csv(source, index=False)
    indicators = tmp_path / "bins"
    indicators.mkdir()
    np.savez(indicators / "190001.npz", **{**data, "afrac_vote": [2, 2]})
    np.savez(indicators / "190002.npz", **data)
    # No published label exists for this votes-only shot. It must remain empty
    # in Source rather than being treated as a stale published label.
    np.savez(indicators / "190003.npz", **data)
    inputs = [source, *indicators.glob("*.npz")]
    before = {path: path.read_bytes() for path in inputs}
    event = tmp_path / "delivery/tables/detachment"
    result = roster_module().snapshot_suggestions(
        source,
        [190001, 190002, 190003, 190004],
        event,
        indicator_root=indicators,
    )
    assert result["shots"] == [190002, 190004]
    assert result["excluded_inconsistent_shots"] == [190001]
    assert "afrac_vote" in result["inconsistencies"][0]["reason"]
    assert set(labels.read_source(event)) == {190002, 190004}
    assert labels.read_source(event)[190002].intervals == (
        (100, 150, 2),
        (150, 200, 1),
    )
    assert result["indicator_fingerprints"]["190001"]["sha256"]
    assert result["indicator_fingerprints"]["190002"]["sha256"]
    pointer = json.loads((event / "review/source.json").read_text())
    assert pointer["excluded_inconsistent_shots"] == [190001]
    assert pointer["inconsistencies"] == result["inconsistencies"]
    assert pointer["indicator_fingerprints"] == result["indicator_fingerprints"]
    assert all(path.read_bytes() == before[path] for path in inputs)


@pytest.mark.parametrize("manual_holdout", [False, True])
def test_main_writes_overlay_without_changing_producer_roster(
    tmp_path, monkeypatch, manual_holdout
):
    module = roster_module()
    repo = tmp_path / "repo"
    cohort_path = repo / "data/events/catalog/cohort.csv"
    cohort_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "shot": [190001, 190002, 190003],
            "split": ["train", "val", "test"],
            "queue_rank": [1, 2, 3],
            "window_start_ms": [0, 0, 0],
            "window_end_ms": [300, 300, 300],
        }
    ).to_csv(cohort_path, index=False)
    root = tmp_path / "producer"
    root.mkdir()
    roster = root / "shots.csv"
    roster.write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n"
        "190001,silver,false,expert,2026-10-02,producer note\n"
        "190002,unverified,false,,,no frames\n"
        "190003,unverified,true,,,blind\n"
    )
    before = roster.read_bytes()
    bins = root / "labels_bins.csv.gz"
    pd.DataFrame(
        {
            "shot": [190001, 190002, 190003],
            "start_ms": [100, 100, 100],
            "state_lm": [2, 1, 2],
            "split": ["train", "val", "test"],
            "confidence": [0.8, 0.8, 0.8],
        }
    ).to_csv(bins, index=False)
    before_bins = bins.read_bytes()
    monkeypatch.setenv("LABELER_DETACHMENT_LABELS", str(bins))
    monkeypatch.setenv("LABELER_DETACHMENT_INDICATORS", str(root / "bins"))
    monkeypatch.setenv("LABELER_DETACHMENT_GEOMETRY_ROOT", str(root / "cache"))
    paths = Paths(root=root, corpus=tmp_path / "corpus")
    camera(paths, 190001)
    camera(paths, 190002, live=manual_holdout)
    monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(module.Paths, "from_env", lambda: paths)
    monkeypatch.setattr(module, "original_candidates", lambda *_: {"shots": []})
    out = tmp_path / "delivery"
    if manual_holdout:
        overlay = out / "tables/detachment/shots.csv"
        overlay.parent.mkdir(parents=True)
        overlay.write_text(
            "shot,tier,holdout,reviewers,verified_on,notes\n"
            "190001,gold,false,expert,2026-10-03,UI curation\n"
            "190002,silver,true,expert,2026-10-03,manual reservation\n"
        )
    record = out / "queue.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "queue",
            "--out",
            str(out),
            "--producer-root",
            str(root),
            "--producer-roster",
            str(roster),
            "--producer-labels",
            str(bins),
            "--record",
            str(record),
        ],
    )
    module.main()
    assert roster.read_bytes() == before
    assert bins.read_bytes() == before_bins
    overlay = rosters.read_roster(out / "tables/detachment/shots.csv")
    assert overlay.shot.tolist() == [190001]
    assert overlay.iloc[0].tier == ("gold" if manual_holdout else "silver")
    if manual_holdout:
        assert "UI curation" in overlay.iloc[0].notes
    assert "camera available" in overlay.iloc[0].notes
    assert "EFIT missing" in overlay.iloc[0].notes
    assert not (repo / "data/events/detachment/shots.csv").exists()
    evidence = json.loads(record.read_text())
    assert evidence["summary"]["missing_efit"] == (1 if manual_holdout else 2)
    assert evidence["summary"]["no_video_excluded"] == (0 if manual_holdout else 1)
    assert evidence["suggestions"]["shots"] == [190001]
    if manual_holdout:
        scan = json.loads((out / "corpus_scan.json").read_text())
        assert [row["shot"] for row in scan["records"]] == [190001]
        assert evidence["delivery_holdouts"]["shots"] == [190002]
        # The excluded row is no longer in the queue. Its reservation must
        # survive the following regeneration, rather than be silently erased.
        module.main()
        overlay = rosters.read_roster(out / "tables/detachment/shots.csv")
        assert overlay.shot.tolist() == [190001]
        assert roster.read_bytes() == before
        assert bins.read_bytes() == before_bins


def test_build_targets_exclude_reserved_shots_from_queue_and_existing_stores(tmp_path):
    for shot in (190001, 190002, 190003, 190004):
        (tmp_path / f"{shot}.h5").touch()
    cohort = pd.DataFrame({"shot": [190003, 190004], "split": ["test", "val"]})
    targets = roster_module().build_targets(
        [{"shot": 190001, "split": "train"}, {"shot": 190002, "split": "train"}],
        cohort,
        {190002, 190003},
        tmp_path,
        rebuild_existing=True,
    )
    assert sorted(targets) == [190001, 190004]
    assert targets[190004]["split"] == "val"
