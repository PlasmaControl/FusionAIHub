"""Build pipeline + in-memory store.

Numbers in the assertions below are measured against the synthetic fixtures, not copied from the
task plan: the plan asserted `betan_mean == 1.9` for staged_shot_a, which is what the fixture's
betan *step* is worth (1.9 inside 800-4800 ms) but not what the flat top actually spans. Ip's flat
top is [680, 4905] ms -- the 85 % crossings of a trapezoid that is only level between 800 and
4800 -- so 8 of the 169 EFIT slices in the window sit on the 0.5 shoulder and the mean is
1.8337275981903076. The fixture is right and the plan's expectation was wrong.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import inspect
import json
import time

import numpy as np
import pandas as pd
import pytest

from ideate.schema import Range
from ideate.shotdb import build, corpus_signals, store, text

from .conftest import CORPUS_BARE_SHOT


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Stand in for MiniLM: a unit vector for any non-empty text, a zero row for an empty one.

    Loading the real sentence-transformer costs seconds and downloads a checkpoint; nothing in
    this module is testing the encoder, only that the right texts reach it and that the rows come
    back aligned with shots.parquet.
    """

    def fake(texts: list[str]) -> np.ndarray:
        return np.stack(
            [
                np.full(384, 1 / np.sqrt(384), np.float32)
                if t and t.strip()
                else np.zeros(384, np.float32)
                for t in texts
            ]
        )

    monkeypatch.setattr(text, "embed_texts", fake)


def test_build_record_staged(paths, staged_shot_a, text_fixtures, stub_embeddings):
    text.build_logs_subset(paths, {staged_shot_a})
    rec, shapes = build.build_record(staged_shot_a, paths, build.load_build_cfg())
    flat = rec.segment("flat_top")
    assert flat is not None and abs(flat.raw["ip_mean"] - 1.2e6) < 1.0e4
    assert abs(flat.raw["pnbi_total_peak"] - 4.5e6) < 1.0
    # statistics are float64 since features._finite casts (was float32: 1.8337275981903076)
    assert abs(flat.derived["betan_mean"] - 1.8337277879376384) < 1e-9
    assert rec.coverage["ip"] == "present"
    assert rec.coverage["pech_LEIA"] == "pending"  # no ech group, but the ech system is fetchable
    assert rec.coverage["ne0"] == "unavailable"  # profile fits only exist in staged files
    assert rec.derived_provenance["betan"].assumed is True
    assert rec.raw_sources["ip"] == "staged"
    assert rec.labels.regime == "QH" and rec.labels.regime_source == "text"
    assert "disruption_free" in rec.labels.operational
    assert rec.outcome.ip_target_hit is True and rec.human.mpid == "2014-21-20"
    assert shapes["flat_top"].shape == (7 * 20,)  # 7 waveforms x 20 points
    assert rec.campaign == "unknown"  # 900001 is outside every configured shot-number band


def test_build_record_disrupted(paths, our_shot_c, text_fixtures, stub_embeddings):
    text.build_logs_subset(paths, {our_shot_c})
    rec, _ = build.build_record(our_shot_c, paths, build.load_build_cfg())
    assert rec.outcome.fast_quench is True and rec.outcome.end_reason == "fast_current_quench"
    assert {"fast_current_quench", "early_termination", "dud"} <= rec.labels.operational
    assert rec.human.verdict == "bad" and rec.raw_sources["ip"] == "fetched"
    assert rec.outcome.fault_strings == ["dud trip", "disrupted"]


def test_build_record_without_ip_has_no_segments(paths, text_fixtures, stub_embeddings):
    """A shot with no raw file at all is a record with zero segments, not a build failure."""
    rec, shapes = build.build_record(999999, paths, build.load_build_cfg())
    assert rec.segments == [] and shapes == {}
    assert rec.outcome.end_reason == "no_ip_signal"
    assert set(rec.coverage.values()) <= {"unavailable", "pending"}


def test_build_add_and_store(
    paths, staged_shot_a, staged_shot_b, our_shot_c, text_fixtures, stub_embeddings
):
    cfg = build.load_build_cfg()
    report = build.build([staged_shot_a, staged_shot_b], paths, cfg, workers=1, encode=False)
    assert report.shots == [staged_shot_a, staged_shot_b]
    assert (paths.db_dir / "manifest.json").exists()
    db = store.ShotDB.load(paths.db_dir)
    assert len(db.shots) == 2 and db.emb["scalar"].shape[0] == len(db.segments)
    assert db.get(staged_shot_a).human.mpid == "2014-21-20"

    build.add([our_shot_c, staged_shot_a], paths, cfg)  # one new, one replaced
    db = store.ShotDB.load(paths.db_dir)
    assert sorted(db.shots["shot"]) == sorted([staged_shot_a, staged_shot_b, our_shot_c])
    assert db.emb["scalar"].shape[0] == len(db.segments)
    assert db.emb["text_mp"].shape[0] == len(db.shots)
    assert json.loads((paths.db_dir / "manifest.json").read_text())["adds_since_fit"] == 2

    m = db.mask("flat_top", constraints={"ip_mean": Range(lo=1.0e6, hi=1.4e6)})
    assert m.sum() == 2  # A and C have 1.2 MA flat tops; B is 0.9 MA
    ids = [i for i, keep in zip(db.segments.index, m, strict=True) if keep]
    hits = db.knn("scalar", db.emb["scalar"][db.segments.index.get_loc(ids[0])], k=2, mask=m)
    assert hits[0][0] == ids[0] and hits[0][1] > 0.99
    assert db.mask("flat_top", exclude_shots={staged_shot_a}).sum() == 2
    assert db.mask("flat_top", require_labels={"dud"}).sum() == 1
    assert db.mask("flat_top", avoid_labels={"dud"}).sum() == 2
    assert db.mask("flat_top", require_labels={"QH"}).sum() == 3  # regime joins the label set
    assert db.mask("flat_top", exclude_runs={"20150120"}).sum() == 0
    with pytest.raises(KeyError):
        db.mask("flat_top", constraints={"no_such_column": Range(lo=0.0)})


def test_add_keeps_embeddings_aligned_with_their_rows(
    paths, staged_shot_a, staged_shot_b, our_shot_c, text_fixtures, stub_embeddings
):
    """The rebuilt matrices must line up row-for-row with the rebuilt frames after an upsert.

    A zero row is the stub encoder's mark for "this shot has no text"; staged_shot_b is the only
    fixture shot without a logbook record of its own, so it is the one that must still carry a
    zero text_log row after A and C are added in front of nothing and behind everything.
    """
    cfg = build.load_build_cfg()
    build.build([staged_shot_a, staged_shot_b], paths, cfg, workers=1, encode=False)
    build.add([our_shot_c, staged_shot_a], paths, cfg)
    db = store.ShotDB.load(paths.db_dir)
    pos = {int(s): i for i, s in enumerate(db.shots["shot"])}
    assert np.linalg.norm(db.emb["text_log"][pos[staged_shot_b]]) == 0.0
    assert np.linalg.norm(db.emb["text_log"][pos[our_shot_c]]) > 0.0
    # every segment row's embedding is the projection of that row's own features
    X, _ = build._scalar_matrix(db.segments_base, db.shape_matrix(), db.pca["feature_cols"])
    assert np.allclose(build.project_scalar(X, db.pca), db.emb["scalar"], atol=1e-5)


def test_nan_survives_the_flatten_and_the_parquet_round_trip(
    paths, staged_shot_b, text_fixtures, stub_embeddings
):
    """ "Not recorded" must stay NaN in segments.parquet, never become 0.0.

    staged_shot_b holds Ip and betan/betap only, so every beam, gyrotron, valve and RMP coil
    column is unrecorded for it -- and an idle actuator that *was* recorded reads 0.0, which is a
    different fact. features.py spent four fix waves on that distinction; flattening must not
    undo it.
    """
    build.build([staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False)
    seg = pd.read_parquet(paths.db_dir / "segments.parquet")
    row = seg.loc[f"{staged_shot_b}:flat_top"]
    assert np.isnan(row["pnbi_15L_mean"]) and np.isnan(row["pnbi_total_mean"])
    assert np.isnan(row["pech_LEIA_on_frac"]) and np.isnan(row["gas_GASA_peak"])
    assert row["ip_mean"] > 0.0 and np.isfinite(row["betan_mean"])
    assert seg["pnbi_15L_mean"].dtype == np.float64  # an all-missing column stays a float column


def test_idle_actuator_is_zero_not_missing(paths, staged_shot_a, text_fixtures, stub_embeddings):
    """The other half of the same distinction: 21L never fired on staged_shot_a but was recorded."""
    build.build([staged_shot_a], paths, build.load_build_cfg(), workers=1, encode=False)
    row = pd.read_parquet(paths.db_dir / "segments.parquet").loc[f"{staged_shot_a}:flat_top"]
    assert row["pnbi_21L_on_frac"] == 0.0 and np.isnan(row["pnbi_21L_mean"])
    assert row["pnbi_15L_mean"] == pytest.approx(2.0e6, abs=1.0)


def test_encoder_absent_is_the_default_path(paths, staged_shot_a, text_fixtures, stub_embeddings):
    """`encode=True` (the default) with no shotdb.ignite module: scalar-only build, said out loud."""
    assert inspect.signature(build.build).parameters["encode"].default is True
    report = build.build([staged_shot_a], paths, build.load_build_cfg(), workers=1)
    assert report.encoded is False
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["ignite"]["status"] == "not_installed"
    assert "ignite" in manifest["ignite"]["reason"]
    assert not list(paths.db_dir.glob("emb_ignite_*.npy"))
    assert not (paths.db_dir / "windows.parquet").exists()
    db = store.ShotDB.load(paths.db_dir)
    assert set(db.emb) == {"scalar", "text_mp", "text_log"}


def test_shot_cols_is_the_one_list_of_shots_parquet_columns(
    paths, staged_shot_a, text_fixtures, stub_embeddings
):
    """_empty_shots used to carry its own hand-copied 17-name list beside the row dict in
    records_to_tables. SHOT_COLS is now the only list: a built table, an empty table and the row
    itself all follow it, and TEXT_KEYS names the text columns the embeddings are built from."""
    text.build_logs_subset(paths, {staged_shot_a})
    rec, shapes = build.build_record(staged_shot_a, paths, build.load_build_cfg())
    shots_df, _, _ = build.records_to_tables([rec], {staged_shot_a: shapes})
    assert list(shots_df.columns) == list(build.SHOT_COLS)
    assert list(build._empty_shots().columns) == list(build.SHOT_COLS)
    assert list(build.records_to_tables([], {})[0].columns) == list(build.SHOT_COLS)
    assert tuple(build._shot_row(rec)) == build.SHOT_COLS
    assert set(build.TEXT_KEYS) <= set(build.SHOT_COLS)
    assert set(build._embeddings(shots_df, np.zeros((0, 0)), build.empty_pca())) == {
        "scalar",
        *build.TEXT_KEYS,
    }


def test_build_records_a_failed_shot_without_losing_the_others(
    paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings, monkeypatch
):
    real = build.build_record

    def boom(shot, paths_, cfg, reader=None):
        if shot == staged_shot_b:
            raise ValueError("synthetic explosion")
        return real(shot, paths_, cfg, reader)

    monkeypatch.setattr(build, "build_record", boom)
    report = build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    assert report.shots == [staged_shot_a]
    assert report.failed[staged_shot_b] == "ValueError: synthetic explosion"
    assert json.loads((paths.db_dir / "manifest.json").read_text())["failed"] == {
        str(staged_shot_b): "ValueError: synthetic explosion"
    }


def test_coverage_report_counts_every_status(
    paths, staged_shot_a, our_shot_c, text_fixtures, stub_embeddings
):
    text.build_logs_subset(paths, {staged_shot_a, our_shot_c})
    cfg = build.load_build_cfg()
    recs = [build.build_record(s, paths, cfg)[0] for s in (staged_shot_a, our_shot_c)]
    df = build.coverage_report(recs)
    assert list(df.columns[:4]) == ["present", "unavailable", "pending", "not_installed"]
    assert df.loc["ip", "present"] == 2
    assert df.loc["pech_LEIA", "pending"] == 1 and df.loc["pech_LEIA", "present"] == 1
    assert df.loc["pech_LUKE", "unavailable"] == 1  # our_shot_c records LUKE as missing
    assert (df[["present", "unavailable", "pending", "not_installed"]].sum(axis=1) == 2).all()
    assert "present_frac" in df.columns
    rendered = build.format_coverage(df)
    assert "pech_LEIA" in rendered and "present" in rendered


def test_timing_targets(paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings):
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    db = store.ShotDB.load(paths.db_dir)
    db.get(staged_shot_a)  # warm pydantic's validator cache; the target is per steady-state call
    t0 = time.perf_counter()
    db.get(staged_shot_a)
    assert time.perf_counter() - t0 < 0.05
    big = np.random.default_rng(0).standard_normal((5000, 24)).astype(np.float32)
    big /= np.linalg.norm(big, axis=1, keepdims=True)
    db.emb["big"] = big
    t0 = time.perf_counter()
    hits = db.knn("big", big[0], k=10)
    assert time.perf_counter() - t0 < 0.2
    assert len(hits) == 10 and hits[0][1] > 0.999


def test_scalar_embedding_round_trip():
    X = np.random.default_rng(1).standard_normal((50, 10)).astype(np.float32)
    X[3, 2] = np.nan
    pca = build.fit_scalar_embedding(X, n_components=4)
    Z = build.project_scalar(X, pca)
    assert Z.shape == (50, 4) and np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)
    assert pca["impute_frac"][2] == pytest.approx(1 / 50)
    assert sum(pca["explained_variance_ratio"]) <= 1.0 + 1e-9


def test_scalar_embedding_survives_a_dead_column():
    """An all-NaN feature (a channel nobody in the build recorded) must not poison the fit."""
    X = np.random.default_rng(2).standard_normal((30, 5)).astype(np.float32)
    X[:, 1] = np.nan
    X[:, 3] = 7.0  # zero variance
    pca = build.fit_scalar_embedding(X, n_components=3)
    Z = build.project_scalar(X, pca)
    assert np.isfinite(Z).all() and pca["impute_frac"][1] == 1.0


def test_scalar_scale_is_robust_to_one_outlier_and_z_is_clipped():
    """The first 200-shot build: ne_line_mean median 1.07e14, one shot at 1.23e19, std-scale
    8.7e17 -- every ordinary shot at z ~ 1e-4 while the outlier owned a component."""
    rng = np.random.default_rng(0)
    ordinary = 1.0e14 * (1 + 0.1 * rng.standard_normal(199))
    col = np.concatenate([ordinary, [1.23e19]])
    X = np.column_stack([col, rng.standard_normal(200)])
    pca = build.fit_scalar_embedding(X, 2)
    p5, p95 = np.percentile(col, [5, 95])
    assert pca["scale"][0] == pytest.approx((p95 - p5) / 3.29, rel=1e-9)
    assert pca["scale"][0] < 1e14  # not the ~8.7e17 std
    assert pca["mean"][0] == pytest.approx(np.median(col))
    assert pca["z_clip"] == build.Z_CLIP
    # An ordinary shot's density now moves the embedding; the outlier is merely "far"
    E = build.project_scalar(X, pca)
    assert np.isfinite(E).all()
    z_out = (col[-1] - pca["mean"][0]) / pca["scale"][0]
    assert z_out > build.Z_CLIP  # would be clipped -- the point of the test


def test_constant_and_degenerate_columns_still_get_a_usable_scale():
    two_valued = np.r_[np.zeros(20), 1.0]  # p5 == p95 == 0 (one row in 21) -> MAD 0 -> std
    X = np.column_stack([np.zeros(21), two_valued, np.arange(21.0)])
    pca = build.fit_scalar_embedding(X, 2)
    assert pca["scale"][0] == 1.0  # constant -> 1.0
    assert pca["scale"][1] == pytest.approx(np.std(two_valued))
    p5, p95 = np.percentile(np.arange(21.0), [5, 95])
    assert pca["scale"][2] == pytest.approx((p95 - p5) / 3.29)


def test_project_scalar_without_z_clip_key_still_works_for_old_pca_json():
    X = np.column_stack([np.arange(6.0), np.ones(6)])
    pca = build.fit_scalar_embedding(X, 1)
    pca.pop("z_clip")
    assert np.isfinite(build.project_scalar(X, pca)).all()


def test_a_rebuild_has_nothing_to_reuse_without_a_previous_encoded_database(paths, tmp_path):
    """No database, or one whose ignite block is not `ok`, means `build` encodes everything (or
    reports not_installed) rather than trying to carry rows over."""
    assert build._reuse_encodings(tmp_path, [], pd.DataFrame(), paths, 1) is None
    paths.db_dir.mkdir(parents=True, exist_ok=True)
    (paths.db_dir / "manifest.json").write_text(json.dumps({"ignite": {"status": "not_installed"}}))
    assert build._reuse_encodings(tmp_path, [], pd.DataFrame(), paths, 1) is None


# ------------------------------------------------------------------ building from the corpus


def test_build_record_from_the_corpus_reader(
    paths, signal_corpus, labelmaker_features, stub_embeddings
):
    """The whole record, built through `CorpusSignalReader`: segments off a labelmaker `ip`,
    actuator totals off the corpus's channel arrays, and the provenance of both written down."""
    rec, _ = build.build_record(
        signal_corpus,
        paths,
        build.load_build_cfg(),
        reader=corpus_signals.CorpusSignalReader(paths),
    )
    flat = rec.segment("flat_top")
    assert flat is not None and flat.raw["pnbi_total_peak"] == pytest.approx(3.6e6)
    assert rec.reader == "corpus"
    assert rec.feature_resolvers["ip"] == "labelmaker:fdp"
    assert rec.feature_resolvers["bt"] == "labelmaker:archive"
    assert rec.feature_resolvers["pnbi_15L"] == "corpus"
    assert rec.raw_sources["dalpha"] == "corpus" and rec.raw_sources["bt"] == "labelmaker"
    assert "pinj" in rec.raw_groups and "co2" in rec.raw_groups
    assert rec.has_frame_codes is False


def test_an_efit_scalar_from_the_archive_is_flagged_assumed(
    paths, signal_corpus, labelmaker_features, stub_embeddings
):
    """A labelmaker feature resolved from the archive store carries no EFIT run id either -- the
    same claim the staged files' `assumed_for_staged` makes, and the same answer."""
    rec, _ = build.build_record(
        signal_corpus,
        paths,
        build.load_build_cfg(),
        reader=corpus_signals.CorpusSignalReader(paths),
    )
    assert rec.derived_provenance["betan"].assumed is True


def test_build_many_constructs_the_corpus_reader_inside_each_worker(
    paths, signal_corpus, labelmaker_features, stub_embeddings
):
    """The reader is not pickled and shipped -- an open HDF5 handle must never be forked -- so
    what crosses the process boundary is `reader_kind` plus `paths`."""
    report = build.BuildReport(shots=[])
    records, _ = build._build_many(
        [signal_corpus, CORPUS_BARE_SHOT],
        paths,
        build.load_build_cfg(),
        workers=2,
        report=report,
        reader_kind="corpus",
    )
    assert sorted(report.shots) == sorted([signal_corpus, CORPUS_BARE_SHOT])
    assert {r.reader for r in records} == {"corpus"}
    bare = next(r for r in records if r.shot == CORPUS_BARE_SHOT)
    assert bare.coverage["ip"] == "pending"  # no feature file: fdp can still fetch it
    assert bare.coverage["dalpha"] == "unavailable"  # the corpus did not record it


def test_shots_parquet_carries_the_corpus_coverage_columns(
    paths, signal_corpus, labelmaker_features, text_fixtures, stub_embeddings
):
    report = build.build(
        [signal_corpus],
        paths,
        build.load_build_cfg(),
        workers=1,
        encode=False,
        reader_kind="corpus",
        list_name="unit_test_list",
    )
    assert report.shots == [signal_corpus]
    row = pd.read_parquet(paths.db_dir / "shots.parquet").iloc[0]
    assert row["reader"] == "corpus"
    assert row["has_filterscopes"] and row["has_co2"] and not row["has_ece"]
    assert row["groups_filled"] == 11  # every group `signal_corpus` wrote, counted once
    assert json.loads(row["feature_resolvers"])["bt"] == "labelmaker:archive"
    assert row["has_frame_codes"] is False or row["has_frame_codes"] == 0
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["reader"] == "corpus" and manifest["list"] == "unit_test_list"
    assert manifest["git_sha"] and manifest["config_sha"]


def test_has_frame_codes_is_the_file_on_disk(
    paths, signal_corpus, labelmaker_features, stub_embeddings
):
    (paths.data_root / "frame_codes").mkdir(parents=True, exist_ok=True)
    (paths.data_root / "frame_codes" / f"{signal_corpus}.pt").write_bytes(b"")
    rec, _ = build.build_record(
        signal_corpus,
        paths,
        build.load_build_cfg(),
        reader=corpus_signals.CorpusSignalReader(paths),
    )
    assert rec.has_frame_codes is True


def test_no_encode_skips_the_ignite_channel_and_nothing_else(
    paths, staged_shot_a, text_fixtures, stub_embeddings
):
    """`--no-encode` is about the IGNITE channel only: the scalar and text embeddings are part of
    the database itself and are still written, and the manifest says exactly that rather than a
    bare "encode=False" a reader has to interpret."""
    build.build([staged_shot_a], paths, build.load_build_cfg(), workers=1, encode=False)
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["ignite"]["status"] == "disabled"
    assert "--no-encode" in manifest["ignite"]["reason"]
    assert "scalar and text embeddings" in manifest["ignite"]["reason"]
    for name in ("emb_scalar", "emb_text_mp", "emb_text_log"):
        assert (paths.db_dir / f"{name}.npy").exists(), name
    assert not list(paths.db_dir.glob("emb_ignite_*.npy"))
