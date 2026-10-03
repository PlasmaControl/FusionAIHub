"""Regressions for fetched DSM inputs, cache identity and serving diagnostics."""

import json
from dataclasses import replace

import numpy as np
import pytest

from labeler.config import Paths
from labeler.elm import compare, dsm
from labeler.features.store import FeatureArray


def _rows(shot=7, source_signature=""):
    return dsm.Rows(
        shot,
        np.zeros((4, 60), np.float32),
        np.array([False, True, True, False]),
        np.array([False, True, False, False]),
        ("bt",),
        {"ece": "corpus"},
        tuple(dsm.spec.ALWAYS_MEAN_FILLED),
        source_signature,
    )


def _fetched(paths, shot=7):
    folder = dsm.fetched_features_dir(paths)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / f"{shot}.npz",
        ip_x=np.array([0.0, 1.0]),
        ip_y=np.array([[1e6, 2e6]]),
        ip_attrs=json.dumps({"resolver": "fdp", "units_from_source": "a"}),
        bt_x=np.array([0.0, 1.0]),
        bt_y=np.array([[-2.0, -2.1]]),
        bt_attrs=json.dumps({"resolver": "fdp", "units_from_source": "t"}),
    )


def test_new_fetch_or_normalization_invalidates_rows(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path, corpus=tmp_path / "corpus")
    monkeypatch.setattr(dsm.resolve_archive, "ARCHIVE_FILES", ())
    norm = {"mean": [0.0], "std": [1.0]}
    calls = []

    def build(paths, shot, norm):
        calls.append(shot)
        return _rows(shot, dsm.source_signature(paths, shot, norm))

    monkeypatch.setattr(dsm, "shot_rows", build)
    cache = tmp_path / "rows"
    first = dsm.cached_rows(paths, 7, norm, cache)
    again = dsm.cached_rows(paths, 7, norm, cache)
    assert first.source_signature == again.source_signature and calls == [7]
    _fetched(paths)
    fetched = dsm.cached_rows(paths, 7, norm, cache)
    assert fetched.source_signature != first.source_signature and calls == [7, 7]
    changed = dsm.cached_rows(paths, 7, {"mean": [2.0], "std": [1.0]}, cache)
    assert changed.source_signature != fetched.source_signature and calls == [7, 7, 7]


def test_old_rows_are_rebuilt_when_signature_is_required(tmp_path):
    path = tmp_path / "7.npz"
    dsm.save_rows(_rows(), path)
    assert dsm.load_rows(7, path) is not None
    assert dsm.load_rows(7, path, signature="new sources") is None


def test_fetched_inputs_fill_gaps_and_keep_fdp_sampling(tmp_path, monkeypatch):
    paths = Paths(root=tmp_path)
    _fetched(paths)
    archived_ip = FeatureArray(
        x=np.array([0.0, 1.0]),
        y=np.array([[3e6, 3e6]]),
        attrs={"resolver": "archive"},
    )
    monkeypatch.setattr(
        dsm.resolve_archive, "resolve", lambda shot, want: ({"ip": archived_ip}, {})
    )
    monkeypatch.setattr(
        dsm.resolve_corpus, "resolve", lambda shot, want, **kwargs: ({}, {})
    )
    got = dsm.shot_features(paths, 7)
    assert got["ip"] is archived_ip
    assert got["bt"].attrs["resolver"] == "fdp"
    assert got["bt"].attrs["units_from_source"] == "t"
    np.testing.assert_allclose(got["bt"].y, [[-2.0, -2.1]])


def test_scores_cannot_be_reused_with_changed_rows(tmp_path):
    rows = {7: _rows()}
    saved = compare.DsmScores(rows, {7: np.ones((4, 4))})
    saved.save(tmp_path)
    loaded = compare.DsmScores.load(tmp_path, rows)
    np.testing.assert_array_equal(loaded.risk[7], saved.risk[7])
    changed = {7: replace(rows[7], x=rows[7].x + 1)}
    with pytest.raises(ValueError, match="without --rescore"):
        compare.DsmScores.load(tmp_path, changed)
    (tmp_path / "row_cache_manifest.json").unlink()
    with pytest.raises(ValueError, match="unrecorded rows"):
        compare.DsmScores.load(tmp_path, rows)


def test_filter_share_and_risk_quantiles_use_only_usable_rows():
    rows = {7: _rows()}
    risk = {7: np.array([[99.0] * 4, [0.1] * 4, [0.3] * 4, [99.0] * 4])}
    out = dsm.row_diagnostics(rows, risk)
    assert out["usable_rows"] == 2
    assert out["outside_training_filter_usable_rows"] == 1
    assert out["outside_training_filter_usable_row_share"] == 0.5
    assert out["missing_features"]["bt"] == {"n_shots": 1, "shots": [7]}
    assert out["always_mean_filled_columns"] == list(dsm.spec.ALWAYS_MEAN_FILLED)
    assert out["risk_quantiles_usable_rows"]["h50ms"]["quantiles"]["0.5"] == 0.2


def test_risk_quantiles_record_nonfinite_values():
    out = dsm.risk_quantiles(np.array([np.nan, 0.1, 0.3, np.inf]))
    assert out["n"] == 4 and out["nonfinite"] == 2
    assert out["quantiles"]["0.5"] == 0.2
