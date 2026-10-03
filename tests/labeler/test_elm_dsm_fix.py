"""Regressions for fetched DSM inputs, cache identity and serving diagnostics."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
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
    assert out["display_name"] == "elm-dsm refit"
    assert out["serving"]["nbi_lookahead_ms"] == 25.0
    exposure = out["preprocessing_exposure"]
    assert exposure["applies_to"] == [
        "elm-dsm refit",
        "elm-dsm detection exposed",
        "elm-dsm detection init",
    ]
    assert exposure["blind_cohort_shots"] == [190532, 190646]


def test_risk_quantiles_record_nonfinite_values():
    out = dsm.risk_quantiles(np.array([np.nan, 0.1, 0.3, np.inf]))
    assert out["n"] == 4 and out["nonfinite"] == 2
    assert out["quantiles"]["0.5"] == 0.2


def test_historical_detector_migration_never_relabels_isolated_scores():
    first = {"elm-dsm-detect": {}, "elm-dsm-detect-init": {}}
    assert dsm.historical_detector_sources(first) == {
        "elm-dsm-detect-exposed": "elm-dsm-detect",
        "elm-dsm-detect-init": "elm-dsm-detect-init",
    }
    rerun = {**first, "elm-dsm-detect-exposed": {}}
    assert dsm.historical_detector_sources(rerun) == {
        "elm-dsm-detect-exposed": "elm-dsm-detect-exposed",
        "elm-dsm-detect-init": "elm-dsm-detect-init",
    }
    assert dsm.historical_detector_sources({}) == {}
    clean_only = {"elm-dsm-detect": {"source_normalization_reused": False}}
    assert dsm.historical_detector_sources(clean_only) == {}


def test_detector_normalization_excludes_validation_and_missing_columns():
    """Held-out levels and mean-filled columns must not affect fitted statistics."""
    rows = {}
    spans = {}
    for shot, levels in ((1, [2.0, 4.0]), (2, [1000.0, 2000.0])):
        x = np.zeros((240, 60), np.float64)
        x[2:4, 0] = levels
        x[2:4, 1] = [8.0, 12.0]
        usable = np.zeros(240, bool)
        usable[2:4] = True
        rows[shot] = dsm.Rows(
            shot, x, usable, usable.copy(), (), {}, (dsm.spec.COLUMNS[1],)
        )
        spans[shot] = pd.DataFrame(
            {"t_start": [0.0], "t_end": [100.0], "kind": ["absent"]}
        )
    norm = dsm.fit_detection_normalization(rows, spans, [1])
    assert norm["fit_shots"] == [1]
    assert norm["mean"][0] == 3.0 and norm["std"][0] == 1.0
    assert norm["measured_rows_per_column"][0] == 2
    assert norm["measured_rows_per_column"][1] == 0
    normalized = dsm.normalize_detection_rows(rows[1], norm)
    np.testing.assert_array_equal(normalized.x[2:4, 0], [-1.0, 1.0])
    np.testing.assert_array_equal(normalized.x[2:4, 1], [0.0, 0.0])
    heldout = dsm.normalize_detection_rows(rows[2], norm)
    np.testing.assert_array_equal(heldout.x[2:4, 0], [10.0, 10.0])
    assert not heldout.in_filter[2:4].any()


def test_raw_detector_rows_do_not_use_source_normalization_or_clip(
    tmp_path, monkeypatch
):
    """Raw extraction must preserve levels larger than the source z clipping limit."""
    paths = Paths(root=tmp_path)
    arrays = {
        "ip": FeatureArray(
            x=np.arange(6001) / 1000.0,
            y=np.full((1, 6001), 20e6),
            attrs={"resolver": "archive"},
        ),
        "ece": FeatureArray(
            x=np.arange(6001) / 1000.0,
            y=np.ones((48, 6001)),
            attrs={"resolver": "corpus"},
        ),
    }
    monkeypatch.setattr(dsm, "shot_features", lambda paths, shot: arrays)
    monkeypatch.setattr(dsm.resolve_archive, "ARCHIVE_FILES", ())
    rows = dsm.shot_rows(paths, 7, None)
    assert rows.x[10, 0] == 20.0
    assert rows.in_filter[10]


def test_phase_ids_decode_before_review_and_blind_cohort_overlap():
    raw = ["190643_0", "190643_1", "192721_0"]
    np.testing.assert_array_equal(dsm.physical_shot_ids(raw), [190643, 190643, 192721])
    np.testing.assert_array_equal(
        dsm.physical_shot_ids([1906430, 1906431, 1927210]), [190643, 190643, 192721]
    )
    assert dsm.physical_shot_ids([190643]).tolist() == [190643]
    identity = dsm.split_identity(["190643_1", "192721_0"], ["190643_0", "196541_0"])
    assert identity["split_shots"] == {
        "train": [190643, 192721],
        "test": [190643, 196541],
    }
    assert identity["physical_shots_in_both_split_sides"] == [190643]
    assert identity["selection_role"].startswith("early-stopping validation")
    cohort = pd.DataFrame(
        {"shot": [190643, 192721, 196541], "split": ["val", "train", "test"]}
    )
    overlap = dsm.split_overlap(identity, [190643, 192721], cohort)
    assert overlap["reviewed_shot_ids_in_published_split"] == {
        "train": [190643, 192721],
        "test": [190643],
    }
    assert overlap["cohort_physical_shot_overlap"]["test"]["test"] == [196541]
    json.dumps(overlap)  # integers remain JSON-safe after pandas set intersection


def test_physical_shot_bootstrap_groups_multiple_phases():
    phase = np.array(["190643_0", "190643_1", "192721_0", "192721_0"])
    physical = dsm.physical_shot_ids(phase)
    shots, pos, neg = dsm.auroc_by_shot(
        np.array([0.9, 0.1, 0.8, 0.2]),
        np.array([True, False, True, False]),
        np.ones(4, bool),
        physical,
        n_bins=8,
    )
    assert shots.tolist() == [190643, 192721]
    assert pos.sum(axis=1).tolist() == [1, 1]
    assert neg.sum(axis=1).tolist() == [1, 1]


def test_native_forecast_target_looks_forward_and_masks_unreviewed_time():
    spans = pd.DataFrame(
        {
            "t_start": [0.0, 10.0, 12.0, 30.0],
            "t_end": [10.0, 12.0, 20.0, 40.0],
            "kind": ["absent", "non_crowd", "absent", "absent"],
        }
    )
    times = np.array([4.0, 5.0, 9.0, 10.0, 11.0, 12.0, 16.0, 25.0, 32.0])
    np.testing.assert_array_equal(
        dsm.future_review_targets(spans, times, 5.0),
        [0, 1, 1, 1, 1, 0, -1, -1, 0],
    )
    np.testing.assert_array_equal(
        dsm.future_review_targets(spans, times, 5.0, onsets=True),
        [0, 1, 1, 0, 0, 0, -1, -1, 0],
    )


def test_native_onset_target_does_not_treat_crowds_as_negative():
    spans = pd.DataFrame(
        {
            "t_start": [0.0, 10.0, 20.0],
            "t_end": [10.0, 20.0, 30.0],
            "kind": ["absent", "crowd", "non_crowd"],
        }
    )
    assert dsm.future_review_targets(spans, [8.0], 5, onsets=True)[0] == -1


def test_repaired_detector_uses_photodiodes_and_density_not_source_statistics(
    tmp_path, monkeypatch
):
    paths = Paths(root=tmp_path)
    folder = tmp_path / "benchmarks/elm/elmo/signals"
    folder.mkdir(parents=True)
    t = np.arange(-100, 6001, dtype=float)
    np.savez_compressed(
        folder / "7.npz",
        t_fs_ms=t,
        t_int_ms=t,
        filterscopes=np.stack([np.full(len(t), x) for x in (3, 4, 5)]),
        interferometer=np.stack([np.full(len(t), x) for x in (6e13, 7e13)]),
    )
    photo = tmp_path / "round4/elm/dsm/native_photodiodes"
    photo.mkdir(parents=True)
    np.savez_compressed(photo / "7_pcphd02.npz", x=t, y=np.full((1, len(t)), 8))
    monkeypatch.setattr(dsm, "upstream_photodiodes", dict)
    raw = dsm.Rows(
        7,
        np.zeros((240, 60)),
        np.ones(240, bool),
        np.ones(240, bool),
        ("co2_v2", "co2_v3"),
        {},
        tuple(dsm.spec.ALWAYS_MEAN_FILLED)
        + ("co2_density_slow_v2_downsampled", "co2_density_slow_v3_downsampled"),
    )
    got = dsm.repair_detection_inputs(paths, raw)
    cols = {n: i for i, n in enumerate(dsm.spec.COLUMNS)}
    for name, value in (
        ("pcphd02_downsampled", 8),
        ("pcphd03_downsampled", 4),
        ("co2_density_slow_v2_downsampled", 6e13),
        ("co2_density_slow_v3_downsampled", 7e13),
    ):
        assert got.x[10, cols[name]] == value
        assert name not in got.filled
    assert got.resolvers["pcphd02"].startswith("PCPHD02")
    assert got.resolvers["pcphd03"].startswith("FS03 substitute")
    assert not got.missing
    np.testing.assert_array_equal(raw.x, 0)  # legacy rows remain unchanged
    np.savez_compressed(
        folder / "7.npz",
        t_fs_ms=t,
        t_int_ms=t,
        filterscopes=np.stack([np.full(len(t), value) for value in (3, 4, 5)]),
        interferometer=np.stack([np.full(len(t), value) for value in (1e18, 7e13)]),
    )
    bad = dsm.repair_detection_inputs(paths, raw)
    assert "co2_v2" in bad.missing
    assert "co2_density_slow_v2_downsampled" in bad.filled
    assert bad.resolvers["co2_v2"] == "DENV2F rejected: failed digitiser"
    assert bad.x[10, cols["co2_density_slow_v2_downsampled"]] == 0
