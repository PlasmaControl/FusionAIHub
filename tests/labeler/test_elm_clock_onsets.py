"""Recover clock point timestamps while preserving the original period gate."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.elm import clock_onsets

SHOT = 200427


@pytest.fixture
def clock_tree(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path / "labeler", corpus=tmp_path / "corpus")
    paths.corpus.mkdir(parents=True)
    csv = paths.root / clock_onsets.CLOCK_CSV
    csv.parent.mkdir(parents=True)
    pd.DataFrame(
        {"shot": [SHOT], "category": [1], "t_start": [195.0], "t_end": [250.0]}
    ).to_csv(csv, index=False)
    csv.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "git_sha": "original",
                "rule": {"min_run": 3},
                "per_shot": {str(SHOT): {"channel": "FS01"}},
            }
        )
    )
    t_s = np.arange(10000) * 0.0001
    y = np.ones((8, len(t_s)), dtype=np.float32)
    for at in (0.2, 0.22, 0.24, 0.7):
        y[0] += 3 * np.exp(-(((t_s - at) / 0.0004) ** 2))
    with h5py.File(paths.corpus / f"{SHOT}_processed.h5", "w") as handle:
        group = handle.create_group("filterscopes")
        group["xdata"] = t_s
        group["ydata"] = y
    monkeypatch.setattr(
        clock_onsets,
        "_picker_provenance",
        lambda revision: {"original_git": revision, "verified_identical": True},
    )
    given = {SHOT: pd.DataFrame({"t_start_ms": [195.0], "t_end_ms": [250.0]})}
    return paths, given


def test_recompute_points_keeps_original_spans_and_never_regates(
    clock_tree, monkeypatch
):
    paths, given = clock_tree

    def forbidden(*args, **kwargs):
        raise AssertionError("must not recompute period or H-mode gates")

    monkeypatch.setattr(clock_onsets.spans, "elm_onsets", forbidden)
    monkeypatch.setattr(clock_onsets.spans, "detect_elm", forbidden)
    monkeypatch.setattr(clock_onsets.raw, "fetch", forbidden, raising=False)
    got, metadata = clock_onsets.load_clock_onsets(paths, [SHOT], given)
    np.testing.assert_allclose(got[SHOT], [200.0, 220.0, 240.0], atol=0.1)
    assert 195.0 not in got[SHOT], "period starts are not detected points"
    record = metadata["per_shot"][str(SHOT)]
    assert record["points_before_original_span_gate"] == 4
    assert record["points_retained"] == 3
    assert record["native_median_step_ms"] == pytest.approx(0.1)
    assert record["route"] == "recomputed_original_picker_points"
    assert metadata["review_independent"] is False
    assert metadata["unavailable"] == {}


def test_changed_span_gate_is_refused(clock_tree):
    paths, given = clock_tree
    given[SHOT].loc[0, "t_end_ms"] = 750.0
    with pytest.raises(ValueError, match="differ from original clock"):
        clock_onsets.load_clock_onsets(paths, [SHOT], given)


def test_changed_channel_is_unavailable_not_empty_detections(clock_tree):
    paths, given = clock_tree
    meta_file = (paths.root / clock_onsets.CLOCK_CSV).with_suffix(".meta.json")
    meta = json.loads(meta_file.read_text())
    meta["per_shot"][str(SHOT)]["channel"] = "FS02"
    meta_file.write_text(json.dumps(meta))
    got, metadata = clock_onsets.load_clock_onsets(paths, [SHOT], given)
    assert SHOT not in got
    assert "differs from original FS02" in metadata["unavailable"][str(SHOT)]


def test_changed_picker_is_unavailable(clock_tree, monkeypatch):
    paths, given = clock_tree
    monkeypatch.setattr(
        clock_onsets,
        "_picker_provenance",
        lambda revision: {"verified_identical": False, "reason": "changed picker"},
    )
    got, metadata = clock_onsets.load_clock_onsets(paths, [SHOT], given)
    assert got == {}
    assert metadata["unavailable"] == {str(SHOT): "changed picker"}


def test_saved_original_points_take_precedence_and_use_half_open_gate(
    clock_tree, monkeypatch
):
    paths, given = clock_tree
    paths.events.mkdir()
    common = {
        "source": "elm_clock",
        "git_sha": "original",
        "run_id": "original-clock-run",
        "channel": 0,
        "diag": "filterscopes",
    }
    pd.DataFrame(
        [
            {**common, "phenomenon": "elm", "t0_s": at, "t1_s": at}
            for at in (0.2, 0.25, 0.7)
        ]
    ).to_parquet(paths.events_file(SHOT), index=False)
    pd.DataFrame([{**common, "status": "ran", "n_events": 3}]).to_parquet(
        paths.sources_file(SHOT), index=False
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("original saved point events take precedence")

    monkeypatch.setattr(clock_onsets, "_recomputed_points", forbidden)
    got, metadata = clock_onsets.load_clock_onsets(paths, [SHOT], given)
    np.testing.assert_allclose(got[SHOT], [200.0])
    assert metadata["per_shot"][str(SHOT)]["route"] == (
        "saved_original_revision_points"
    )


def test_saved_other_revision_does_not_override_original_picker(clock_tree):
    paths, given = clock_tree
    paths.events.mkdir()
    pd.DataFrame(
        [
            {
                "source": "elm_clock",
                "git_sha": "other",
                "status": "ran",
                "channel": 0,
                "diag": "filterscopes",
            }
        ]
    ).to_parquet(paths.sources_file(SHOT), index=False)
    pd.DataFrame({"source": ["elm_clock"]}).to_parquet(
        paths.events_file(SHOT), index=False
    )
    got, metadata = clock_onsets.load_clock_onsets(paths, [SHOT], given)
    np.testing.assert_allclose(got[SHOT], [200.0, 220.0, 240.0], atol=0.1)
    assert metadata["per_shot"][str(SHOT)]["route"] == (
        "recomputed_original_picker_points"
    )
