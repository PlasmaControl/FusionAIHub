"""Producer labels and votes remain suggestions with their original semantics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.events.review import producer, recipe, video


def test_method_record_resolves_within_repository(monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.delenv("LABELER_DETACHMENT_METHOD", raising=False)
    path = recipe.sources(tmp_path / "labels.csv")["method_record"]
    assert path == Path(__file__).resolve().parents[2] / "docs/labeler/detachment.md"
    assert path.is_file()


def test_missing_method_override_fails_loudly(monkeypatch, tmp_path):
    import pytest

    missing = tmp_path / "missing.md"
    monkeypatch.setenv("LABELER_DETACHMENT_METHOD", str(missing))
    with pytest.raises(FileNotFoundError, match="method record"):
        recipe.load(tmp_path / "labels.csv")


def test_method_override_is_preserved(monkeypatch, tmp_path):
    method = tmp_path / "method.md"
    method.write_text("Producer-owned definitions")
    monkeypatch.setenv("LABELER_DETACHMENT_METHOD", str(method))
    result = recipe.load(tmp_path / "labels.csv")
    assert result["documentation"] == method.read_text()
    assert result["sources"]["method_record"]["path"] == str(method)


def test_recipe_metadata_is_frozen_and_invalidates_resume(tmp_path):
    from labeler.events.review import detachment

    root = tmp_path / "round4/detach"
    root.mkdir(parents=True)
    table(root / "labels_bins.csv.gz")
    path = root / "review_recipe.json"
    recipe = {
        "method": "compatible consensus",
        "thresholds": {"tangtv": {"marfe_min": 1.23}},
        "evidence_gates": {"marfe": ["sustained bins", "inside separatrix"]},
        "definitions": {"uncertain": "evidence gates fail"},
    }
    path.write_text(json.dumps(recipe))
    paths = Paths(root=tmp_path)
    result = producer.load(170815, paths)
    assert result["recipe"]["record"] == recipe
    before = detachment.context_sources(170815, paths)
    recipe["thresholds"]["tangtv"]["marfe_min"] = 1.34
    path.write_text(json.dumps(recipe))
    assert detachment.context_sources(170815, paths) != before
    assert producer.load(170815, paths)["recipe"]["record"] == recipe


def test_suppression_is_structured_for_unreadable_source(tmp_path):
    path = tmp_path / "round4/detach/labels_bins.csv.gz"
    path.parent.mkdir(parents=True)
    path.write_text("broken gzip")
    assert producer.load(170815, Paths(root=tmp_path))["source_suppressed"] is True


def table(path, *, split="train"):
    frame = pd.DataFrame(
        {
            "shot": [170815] * 5,
            "start_ms": [100, 150, 200, 250, 300],
            "state_lm": [1, 2, 3, 4, 0],
            "state_rule": [1, 4, 3, 4, 0],
            "split": [split] * 5,
            "confidence": [0.8, 0.9, 0.95, 0.5, 0],
            "tangtv_source": ["inversion", "surrogate", "surrogate", "none", "none"],
            **{
                f"{name}_{suffix}": values
                for name in ("afrac", "prad", "tangtv")
                for suffix, values in (
                    ("valid", [True, True, True, True, False]),
                    ("vote", [1, 2, 3 if name == "tangtv" else 2, -1, -1]),
                    ("reason", ["", "", "", "", "no_samples"]),
                )
            },
        }
    )
    frame.to_csv(path, index=False)
    return frame


def test_labels_and_votes_are_clipped_by_overlap_without_reclassification(tmp_path):
    path = tmp_path / "round4/detach/labels_bins.csv.gz"
    path.parent.mkdir(parents=True)
    table(path)
    result = producer.load(170815, Paths(root=tmp_path), (120, 330))
    assert result["bin_start_ms"] == [120, 150, 200, 250, 300]
    assert result["bin_end_ms"] == [150, 200, 250, 300, 330]
    assert result["state_lm"] == [1, 2, 3, 4, 0]
    assert result["state_rule"] == [1, 4, 3, 4, 0]
    assert result["votes"]["afrac"]["vote"][-2:] == [-1, -1]
    assert result["votes"]["afrac"]["valid"][-2:] == [True, False]
    assert result["votes"]["afrac"]["reason"][-1] == "no_samples"
    assert result["tangtv_source"][:3] == ["inversion", "surrogate", "surrogate"]
    assert "recipe" in result


def test_producer_labels_default_and_override_never_offer_test_rows(
    tmp_path, monkeypatch
):
    path = tmp_path / "custom.csv.gz"
    table(path, split="test")
    monkeypatch.setenv("LABELER_DETACHMENT_LABELS", str(path))
    result = producer.load(170815, Paths(root=tmp_path))
    assert result["state_lm"] == []
    assert "blind test" in result["reason"]
    assert producer.source_path(Paths(root=tmp_path)) == path


def test_changed_producer_file_is_not_served_from_stale_cache(tmp_path, monkeypatch):
    path = tmp_path / "bins.csv.gz"
    frame = table(path)
    monkeypatch.setenv("LABELER_DETACHMENT_LABELS", str(path))
    paths = Paths(root=tmp_path)
    assert producer.load(170815, paths)["state_lm"][0] == 1
    frame.loc[0, "state_lm"] = 4
    frame.to_csv(path, index=False)
    assert producer.load(170815, paths)["state_lm"][0] == 4


def test_producer_holdout_flags_exclude_whole_shot(tmp_path, monkeypatch):
    path = tmp_path / "bins.csv.gz"
    frame = table(path)
    frame["holdout"] = [False, False, True, False, False]
    frame.to_csv(path, index=False)
    monkeypatch.setenv("LABELER_DETACHMENT_LABELS", str(path))
    result = producer.load(170815, Paths(root=tmp_path))
    assert result["state_lm"] == []
    assert "blind test" in result["reason"]


def test_unassessed_bins_keep_their_votes_from_producer_npz(tmp_path):
    root = tmp_path / "round4/detach"
    (root / "bins").mkdir(parents=True)
    frame = table(root / "labels_bins.csv.gz")
    frame.loc[frame.state_lm > 0].to_csv(root / "labels_bins.csv.gz", index=False)
    fields = {
        column: (
            np.asarray(frame[column], dtype=str)
            if column.endswith("_reason") or column == "tangtv_source"
            else frame[column].to_numpy()
        )
        for column in frame.columns
        if column not in ("shot", "split", "confidence", "state_lm", "state_rule")
    }
    np.savez(root / "bins/170815.npz", **fields)
    result = producer.load(170815, Paths(root=tmp_path))
    assert result["bin_start_ms"] == [100, 150, 200, 250, 300]
    assert result["state_lm"][-1] == 0
    assert result["votes"]["afrac"]["valid"][-1] is False
    assert result["votes"]["afrac"]["reason"][-1] == "no_samples"


def test_ui_rejects_missing_assessed_label_or_stale_vote_snapshot(tmp_path):
    root = tmp_path / "round4/detach"
    (root / "bins").mkdir(parents=True)
    frame = table(root / "labels_bins.csv.gz")
    fields = {
        column: (
            np.asarray(frame[column], dtype=str)
            if column.endswith("_reason") or column == "tangtv_source"
            else frame[column].to_numpy()
        )
        for column in frame.columns
        if column not in ("shot", "split", "confidence", "state_lm", "state_rule")
    }
    np.savez(root / "bins/170815.npz", **fields)
    missing = frame[frame.state_lm > 0].iloc[1:]
    missing.to_csv(root / "labels_bins.csv.gz", index=False)
    result = producer.load(170815, Paths(root=tmp_path))
    assert result["state_lm"] == []
    assert "assessed" in result["reason"]
    stale = frame[frame.state_lm > 0].copy()
    stale.loc[stale.index[0], "tangtv_vote"] = 2
    stale.to_csv(root / "labels_bins.csv.gz", index=False)
    result = producer.load(170815, Paths(root=tmp_path))
    assert result["state_lm"] == []
    assert "tangtv_vote" in result["reason"]


def test_unpublished_shot_keeps_votes_without_inventing_an_assessment(tmp_path):
    root = tmp_path / "round4/detach"
    (root / "bins").mkdir(parents=True)
    frame = table(root / "labels_bins.csv.gz")
    frame.iloc[:0].to_csv(root / "labels_bins.csv.gz", index=False)
    fields = {
        column: (
            np.asarray(frame[column], dtype=str)
            if column.endswith("_reason") or column == "tangtv_source"
            else frame[column].to_numpy()
        )
        for column in frame.columns
        if column not in ("shot", "split", "confidence", "state_lm", "state_rule")
    }
    np.savez(root / "bins/170815.npz", **fields)
    result = producer.load(170815, Paths(root=tmp_path))
    assert result["label_available"] is False
    assert "votes only" in result["note"]
    assert len(result["bin_start_ms"]) == 5
    assert result["state_lm"] == [0] * 5
    assert result["votes"]["tangtv"]["vote"] == [1, 2, 3, -1, -1]
    assert "reason" not in result


def test_malformed_producer_clock_and_state_fail_explicitly(tmp_path, monkeypatch):
    path = tmp_path / "bins.csv"
    frame = table(path)
    monkeypatch.setenv("LABELER_DETACHMENT_LABELS", str(path))
    frame.loc[1, "start_ms"] = 100
    frame.to_csv(path, index=False)
    result = producer.load(170815, Paths(root=tmp_path))
    assert not result["state_lm"] and "increasing" in result["reason"]
    frame.loc[1, "start_ms"] = 150
    frame.loc[1, "state_lm"] = 9
    frame.to_csv(path, index=False)
    result = producer.load(170815, Paths(root=tmp_path))
    assert not result["state_lm"] and "state" in result["reason"]


def test_default_tangtv_prefers_the_producers_perpendicular_channel(tmp_path):
    import h5py

    from labeler.events.review import rows
    from labeler.events.review.rows import Grid

    source = tmp_path / "corpus.h5"
    with h5py.File(source, "w") as store:
        group = store.create_group("tangtv")
        group["xdata"] = [0, 0.1]
        group["ydata"] = np.ones((3, 2, 4, 6))
    path = tmp_path / "review.h5"
    rows.write(path, Grid(0, 50, 2), [], video_corpus=source)
    assert video.meta(path)["cameras"][1]["default_channel"] == 2


def test_shot_manifest_exposes_producer_strips_and_trace_caveats(tmp_path):
    from labeler.events.review import rows
    from labeler.events.review.rows import Grid

    path = tmp_path / "review.h5"
    params = {
        "detachment_producer": {"state_lm": [2]},
        "panel_metadata": {"p0": {"caveat": "uncalibrated"}},
    }
    rows.write(path, Grid(0, 50, 1), [], params=params)
    assert rows.meta(path)["params"] == params
