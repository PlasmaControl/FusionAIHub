"""`ideate labels join`: labelmaker's per-time probabilities and events -> the ideate DB.

Two rules of the plan are what these tests exist to hold (plan section 2 "Label semantics",
Appendix C item 7):

* a probability is **summarised**, never thresholded into a stored truth -- `labels_wide` keeps
  `max/mean/p95` beside `thr`/`frac_above` so a consumer can see what the threshold did, and every
  statistic is over the VALID samples only, because an invalid sample is not a measurement of a
  low probability;
* a DSM risk is a **forecast**: it becomes an event with `evidence_kind="forecast"` and a finite
  `horizon_s`, and never one with `evidence_kind="detector"`.

Everything here is synthetic and written through labelmaker's own writers
(`labelmaker.labels.store.write_labels`, `labelmaker.events.schema.write_events`), so a change to
either layout breaks these tests rather than silently giving ideate a file the real one is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ideate import cli
from ideate.labels import join
from labelmaker.events import schema as ev
from labelmaker.labels import store as label_store
from labelmaker.labels.schema import LabelSpec
from labelmaker.models.base import Decoded

# One label file's worth of hand-built numbers. dt is 0.1 s so every duration below is exact in
# binary and can be asserted without a tolerance on the time axis.
T = np.arange(0.0, 0.6, 0.1)
DT = 0.1


def _spec(
    slug: str, name: str, *, sha: str = "deadbeef", task: str = "binary",
    attrs: tuple[tuple[str, str], ...] = (),
) -> LabelSpec:
    return LabelSpec(
        name=name,
        task=task,
        attrs=attrs,
        activation="none",
        units="",
        classes=("quiet", "active"),
        slug=slug,
        card_id=f"plasmacontrol/{slug.replace('_', '-')}",
        time_step_ms=100.0,
        ensemble_n=1,
        artifact_sha256=sha,
    )


def write_labels(
    root: Path,
    shot: int,
    slug: str,
    series: dict[str, np.ndarray],
    valid: np.ndarray,
    *,
    t: np.ndarray = T,
    sha: str = "deadbeef",
    attrs: tuple[tuple[str, str], ...] = (),
) -> Path:
    """One model's labels for one shot, through labelmaker's own writer."""
    path = Path(root) / "labels" / f"{shot}_labels.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = [_spec(slug, name, sha=sha, attrs=attrs) for name in series]
    decoded = {
        name: Decoded(mean=np.asarray(y, float), lo=np.asarray(y, float), hi=np.asarray(y, float))
        for name, y in series.items()
    }
    label_store.write_labels(
        path, shot, t, decoded, specs, np.asarray(valid, bool),
        run_id="test-run", features_sha256="0" * 64,
    )
    return path


def rule(**over) -> join.ForecastRule:
    got = {
        "slug": "d3d_tearing_time_to_event_dsm",
        "label": "tm_risk_250ms",
        "thr": 0.5,
        "horizon_s": 0.25,
        "phenomenon": "tearing",
    }
    got.update(over)
    return join.ForecastRule(**got)


# ------------------------------------------------------------------------------- labels_wide


def test_statistics_are_over_the_valid_samples_only(tmp_path):
    """Hand-computed, sample by sample. The invalid sample at t=0.4 carries the largest value
    above the threshold and the second-largest overall: if any of these numbers moved, it leaked."""
    y = np.array([0.1, 0.9, 0.4, 0.8, 0.85, 0.2])
    v = np.array([1, 1, 1, 1, 0, 1], bool)
    write_labels(tmp_path, 900001, "slug_a", {"p": y}, v)

    df = join.labels_wide(
        [900001], labelmaker_root=tmp_path, thresholds={"slug_a/p": (0.5, "card")}
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["shot"] == 900001
    assert (row["slug"], row["label"]) == ("slug_a", "p")
    assert row["n_valid"] == 5
    assert row["valid_frac"] == pytest.approx(5 / 6, rel=1e-6)
    assert row["max_valid"] == pytest.approx(0.9, rel=1e-6)      # not the invalid 0.85's neighbour
    assert row["mean_valid"] == pytest.approx(2.4 / 5, rel=1e-6)
    assert row["p95_valid"] == pytest.approx(0.88, rel=1e-6)     # linear between 0.8 and 0.9
    assert row["thr"] == pytest.approx(0.5)
    assert row["frac_above"] == pytest.approx(2 / 5, rel=1e-6)   # 0.85 is invalid, so 2 of 5
    assert row["first_above_t_s"] == pytest.approx(0.1)
    assert row["n_intervals"] == 2                               # the invalid sample breaks the run
    assert row["longest_interval_s"] == pytest.approx(DT)
    assert row["artifact_sha256"] == "deadbeef"


def test_a_threshold_free_label_leaves_the_alarm_columns_nan(tmp_path):
    """No operating threshold is not a threshold of zero: `frac_above` must be NaN, not 0.0, and
    `n_intervals` must be null, not 0 -- a reader that sees 0 concludes the alarm never fired."""
    write_labels(tmp_path, 900001, "slug_a", {"p": np.full(T.size, 0.9)}, np.ones(T.size, bool))

    row = join.labels_wide([900001], labelmaker_root=tmp_path, thresholds={}).iloc[0]

    assert np.isnan(row["thr"])
    assert np.isnan(row["frac_above"])
    assert np.isnan(row["first_above_t_s"])
    assert np.isnan(row["longest_interval_s"])
    assert pd.isna(row["n_intervals"])
    assert row["max_valid"] == pytest.approx(0.9, rel=1e-6)  # the summary is still there


def test_a_label_with_no_valid_sample_summarises_to_nan_not_zero(tmp_path):
    write_labels(tmp_path, 900001, "slug_a", {"p": np.full(T.size, 0.9)}, np.zeros(T.size, bool))

    row = join.labels_wide(
        [900001], labelmaker_root=tmp_path, thresholds={"slug_a/p": (0.5, "card")}
    ).iloc[0]

    assert row["n_valid"] == 0
    assert row["valid_frac"] == pytest.approx(0.0)
    for col in ("max_valid", "mean_valid", "p95_valid", "frac_above", "first_above_t_s"):
        assert np.isnan(row[col]), col


def test_a_non_finite_valid_sample_nulls_the_alarm_columns(tmp_path):
    """A model that emits NaN where its own validity mask says the inputs were fine has not
    measured a low probability there. `max/mean/p95_valid` already come out NaN by propagation;
    `frac_above` and its companions must too, or the row would report an alarm rate computed over
    a population its own summary refuses to describe."""
    y = np.array([0.9, np.nan, 0.9, 0.9, 0.9, 0.9])
    write_labels(tmp_path, 900001, "slug_a", {"p": y}, np.ones(T.size, bool))

    row = join.labels_wide(
        [900001], labelmaker_root=tmp_path, thresholds={"slug_a/p": (0.5, "card")}
    ).iloc[0]

    assert row["n_valid"] == 6
    for col in ("max_valid", "mean_valid", "p95_valid", "frac_above", "first_above_t_s",
                "longest_interval_s"):
        assert np.isnan(row[col]), col
    assert pd.isna(row["n_intervals"])


def test_a_shot_with_no_labels_file_produces_no_rows_and_is_named(tmp_path):
    write_labels(tmp_path, 900001, "slug_a", {"p": np.full(T.size, 0.5)}, np.ones(T.size, bool))

    result = join.join([900001, 900002], labelmaker_root=tmp_path)

    assert sorted(set(result.labels_wide["shot"])) == [900001]
    assert result.manifest["labels_missing"] == [900002]
    assert result.manifest["n_shots_with_labels"] == 1


def test_columns_and_dtypes_are_the_contract(tmp_path):
    write_labels(tmp_path, 900001, "slug_a", {"p": np.full(T.size, 0.5)}, np.ones(T.size, bool))
    df = join.labels_wide([900001], labelmaker_root=tmp_path)
    empty = join.labels_wide([], labelmaker_root=tmp_path)

    assert list(df.columns) == list(join.LABELS_WIDE_COLUMNS)
    assert list(empty.columns) == list(join.LABELS_WIDE_COLUMNS)
    for name, dtype in join.LABELS_WIDE_DTYPES.items():
        assert df[name].dtype == dtype, name
        assert empty[name].dtype == dtype, name


def test_every_label_of_every_model_in_the_file_is_summarised(tmp_path):
    write_labels(tmp_path, 900001, "slug_a", {"p": T, "q": T}, np.ones(T.size, bool))
    write_labels(tmp_path, 900001, "slug_b", {"r": T}, np.ones(T.size, bool))

    df = join.labels_wide([900001], labelmaker_root=tmp_path)

    assert sorted(zip(df["slug"], df["label"], strict=True)) == [
        ("slug_a", "p"), ("slug_a", "q"), ("slug_b", "r")
    ]


def test_the_valid_companion_is_not_summarised_as_a_label(tmp_path):
    write_labels(tmp_path, 900001, "slug_a", {"p": T}, np.ones(T.size, bool))
    df = join.labels_wide([900001], labelmaker_root=tmp_path)
    assert set(df["label"]) == {"p"}


# ------------------------------------------------------------------------- forecast events


def test_a_run_above_the_threshold_becomes_one_forecast_event(tmp_path):
    y = np.array([0.1, 0.6, 0.7, 0.2, 0.9, 0.95])
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm", {"tm_risk_250ms": y},
                 np.ones(T.size, bool))

    df = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])

    assert len(df) == 2
    first, second = df.iloc[0], df.iloc[1]
    assert first["t0_s"] == pytest.approx(0.1) and first["t1_s"] == pytest.approx(0.3)
    assert first["confidence"] == pytest.approx(0.7, rel=1e-6)
    assert second["t0_s"] == pytest.approx(0.4) and second["t1_s"] == pytest.approx(0.6)
    assert second["confidence"] == pytest.approx(0.95, rel=1e-6)
    assert list(df["t_cov0_s"]) == [pytest.approx(0.0)] * 2
    assert list(df["t_cov1_s"]) == [pytest.approx(0.6)] * 2


def test_a_forecast_is_never_a_detection(tmp_path):
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool))

    df = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])

    assert set(df["evidence_kind"]) == {"forecast"}
    assert "detector" not in set(df["evidence_kind"])
    assert set(df["source"]) == {"label_forecast"}
    assert set(df["phenomenon"]) == {"tearing"}
    assert np.isfinite(df["horizon_s"]).all()
    assert df["horizon_s"].iloc[0] == pytest.approx(0.25)


def test_an_invalid_sample_breaks_a_forecast_run(tmp_path):
    """A run is a run of *evidence*. Spanning the gap would claim an alarm over a stretch where
    the model was not asked -- exactly the claim the validity mask exists to prevent."""
    y = np.full(T.size, 0.9)
    v = np.array([1, 1, 0, 1, 1, 1], bool)
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm", {"tm_risk_250ms": y}, v)

    df = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])

    assert len(df) == 2
    assert df["t0_s"].tolist() == [pytest.approx(0.0), pytest.approx(0.3)]
    assert df["t_cov0_s"].iloc[0] == pytest.approx(0.0)
    assert df["t_cov1_s"].iloc[0] == pytest.approx(0.6)


def test_a_rule_naming_a_label_the_file_does_not_carry_is_silent(tmp_path):
    write_labels(tmp_path, 900001, "slug_a", {"p": np.full(T.size, 0.9)}, np.ones(T.size, bool))
    df = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])
    assert df.empty
    assert list(df.columns) == list(ev.COLUMNS)


def test_forecast_rows_carry_the_labels_provenance(tmp_path):
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool), sha="c0ffee")

    df = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])

    attrs = json.loads(df["attrs"].iloc[0])
    assert attrs["slug"] == "d3d_tearing_time_to_event_dsm"
    assert attrs["label"] == "tm_risk_250ms"
    assert attrs["thr"] == pytest.approx(0.5)
    assert attrs["artifact_sha256"] == "c0ffee"
    assert attrs["n_samples"] == 6


def test_forecast_attrs_carry_the_labels_queried_at_ms_when_it_has_one(tmp_path):
    """The DSM heads are queried one step past their horizon and the label records at which
    millisecond. A forecast event that dropped it would be a risk with no answer to "as of when"."""
    y = np.full(T.size, 0.9)
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm", {"tm_risk_250ms": y},
                 np.ones(T.size, bool), attrs=(("queried_at_ms", "251"),))
    write_labels(tmp_path, 900002, "d3d_tearing_time_to_event_dsm", {"tm_risk_250ms": y},
                 np.ones(T.size, bool))

    df = join.label_forecast_events([900001, 900002], labelmaker_root=tmp_path, rules=[rule()])

    assert json.loads(df["attrs"].iloc[0])["queried_at_ms"] == "251"
    assert "queried_at_ms" not in json.loads(df["attrs"].iloc[1])


# --------------------------------------------------------------------------- the events union


def _detector_event(shot: int, t0: float) -> ev.Event:
    return ev.Event(
        shot=shot, source="tokeye_track", phenomenon="eho", t0_s=t0, t1_s=t0 + 0.1,
        confidence=0.8, diag="mhr", channel=1, pass_name="zoom", t_cov0_s=0.0, t_cov1_s=5.0,
    )


def test_union_of_labelmaker_events_and_forecasts_keeps_labelmakers_dtypes(tmp_path):
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool))
    (tmp_path / "events").mkdir()
    ev.write_events(tmp_path / "events" / "900001_events.parquet", 900001,
                    [_detector_event(900001, 1.0)], run_id="test-run")

    forecasts = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])
    df = join.events_union([900001], labelmaker_root=tmp_path, forecasts=forecasts)

    assert list(df.columns) == list(ev.COLUMNS)
    assert {c: str(d) for c, d in df.dtypes.items()} == dict(ev.DTYPES)
    assert set(df["source"]) == {"tokeye_track", "label_forecast"}
    assert df["t0_s"].is_monotonic_increasing  # one shot, so (shot, t0_s) reduces to t0_s


def test_union_sorts_by_shot_then_time(tmp_path):
    (tmp_path / "events").mkdir()
    for shot in (900002, 900001):
        ev.write_events(tmp_path / "events" / f"{shot}_events.parquet", shot,
                        [_detector_event(shot, 2.0), _detector_event(shot, 1.0)],
                        run_id="test-run")

    df = join.events_union([900001, 900002], labelmaker_root=tmp_path,
                           forecasts=join.empty_events())

    assert list(zip(df["shot"], df["t0_s"], strict=True)) == [
        (900001, 1.0), (900001, 2.0), (900002, 1.0), (900002, 2.0)
    ]
    assert list(df.index) == [0, 1, 2, 3]


def test_a_forecast_row_on_disk_is_replaced_by_this_joins_own(tmp_path):
    """`label_forecast` rows are ideate's to compute: they come out of the labels and this join's
    threshold map. If labelmaker ever writes some of its own into `events/`, keeping both would
    put two rows with different thresholds over the same stretch of the same shot into one table."""
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool))
    (tmp_path / "events").mkdir()
    stale = ev.Event(
        shot=900001, source="label_forecast", evidence_kind="forecast", phenomenon="tearing",
        t0_s=0.0, t1_s=0.6, confidence=0.9, horizon_s=0.25, t_cov0_s=0.0, t_cov1_s=0.6,
    )
    ev.write_events(tmp_path / "events" / "900001_events.parquet", 900001,
                    [stale, _detector_event(900001, 1.0)], run_id="test-run")

    forecasts = join.label_forecast_events([900001], labelmaker_root=tmp_path, rules=[rule()])
    df = join.events_union([900001], labelmaker_root=tmp_path, forecasts=forecasts)

    assert len(df) == 2
    assert sorted(df["source"]) == ["label_forecast", "tokeye_track"]
    assert list(df.loc[df["source"] == "label_forecast", "event_id"]) == list(forecasts["event_id"])


def test_no_events_directory_at_all_is_an_empty_typed_frame(tmp_path):
    """`events/` does not exist until the mask job has run. Everything must still work."""
    assert not (tmp_path / "events").exists()

    result = join.join([900001, 900002], labelmaker_root=tmp_path)

    assert result.events.empty
    assert list(result.events.columns) == list(ev.COLUMNS)
    assert {c: str(d) for c, d in result.events.dtypes.items()} == dict(ev.DTYPES)
    assert result.manifest["events_missing"] == [900001, 900002]
    assert result.manifest["n_shots_with_events"] == 0
    assert result.manifest["n_forecast_events"] == 0


def test_a_curated_table_row_reaches_the_union_with_its_coverage_still_null(tmp_path):
    """`events_union` reads each shot's file wholesale, so a curated label table
    (`labelmaker.events.databases`) needs no wiring here - but it needs a guard. The two things
    that must survive the join are the NaN coverage, which is what says nobody declared an
    examined interval, and the NaN confidence, which is what stops a ranker treating a human's
    list as a perfectly confident detector. A join that filled either with a default would turn
    a listing into a measurement."""
    (tmp_path / "events").mkdir()
    curated = ev.Event(
        shot=900001, source="database:rwm_onsets_2017", evidence_kind="database",
        phenomenon="rwm", t0_s=2.613, t1_s=2.613, diag="", channel=-1,
        attrs={"NTOR": 1, "MODE_TYPE": "rwm", "table": "rwm_onsets_2017"},
        confidence=float("nan"), t_cov0_s=float("nan"), t_cov1_s=float("nan"),
    )
    ev.write_events(tmp_path / "events" / "900001_events.parquet", 900001,
                    [curated, _detector_event(900001, 1.0)], run_id="test-run")

    df = join.events_union([900001], labelmaker_root=tmp_path, forecasts=join.empty_events())

    row = df[df["evidence_kind"] == "database"]
    assert len(row) == 1
    assert row["event_id"].item() == "900001-database:rwm_onsets_2017-00000"
    assert row["t_cov0_s"].isna().all() and row["t_cov1_s"].isna().all()
    assert row["confidence"].isna().all()
    # And it did not displace the detector row it shares the shot with.
    assert sorted(df["evidence_kind"]) == ["database", "detector"]


# ------------------------------------------------------------------- the event-source table


def _sources(root: Path, shot: int, rows):
    from ideate.labels import event_sources as es

    return es.write_sources(es.sources_file(Path(root) / "events", shot), rows)


def test_the_join_ingests_every_source_file_that_exists_and_counts_the_rest(tmp_path):
    """`events.parquet` says what was FOUND. This table says who LOOKED, so a shot with no rows
    can be reported as unprocessed rather than as a quiet shot."""
    from ideate.labels import event_sources as es

    _sources(tmp_path, 900001, [
        es.source_row(900001, "tokeye_track", t_cov0_s=0.0, t_cov1_s=6.0, n_events=0, diag="mhr"),
        es.source_row(900001, "dalpha_lh", status="skipped", reason="no d_alpha"),
    ])
    _sources(tmp_path, 900002, [
        es.source_row(900002, "tokeye_track", status="error", reason="mhr read failed"),
    ])

    result = join.join([900001, 900002, 900003], labelmaker_root=tmp_path)

    assert list(result.sources.columns) == list(es.SOURCES_COLUMNS)
    assert len(result.sources) == 3
    m = result.manifest
    assert m["n_event_source_rows"] == 3
    assert m["n_shots_with_source_rows"] == 2
    assert (m["n_sources_ok"], m["n_sources_skipped"], m["n_sources_error"]) == (1, 1, 1)
    # 900001 completed a source; 900002's only source crashed; 900003 has no file at all.
    assert m["n_shots_with_observed_products"] == 1
    assert m["n_shots_unprocessed"] == 2
    assert m["sources_by_shot"][900001]["has_observed_products"] is True
    assert m["sources_by_shot"][900003] == {
        "n_sources": 0, "n_sources_ok": 0, "n_sources_skipped": 0, "n_sources_error": 0,
        "has_observed_products": False,
    }


def test_a_sources_file_labelmaker_itself_wrote_is_ingested_as_is(tmp_path):
    """The other tests here build the table through ideate's fixture writer. This one writes it
    through labelmaker's REAL `schema.write_sources` - the producer - and reads it back through
    the join, so the two sides' definitions of the contract cannot drift apart unnoticed: the
    columns, their order, their dtypes, and the file's name and place."""
    from ideate.labels import event_sources as es
    from labelmaker.config import Paths as LabelmakerPaths

    assert tuple(ev.SOURCE_COLUMNS) == es.SOURCES_COLUMNS
    assert dict(ev.SOURCE_DTYPES) == es.SOURCES_DTYPES
    lm = LabelmakerPaths(root=tmp_path, corpus=tmp_path / "corpus",
                         text_root=tmp_path / "bundles", logs_jsonl=tmp_path / "logs.jsonl")
    assert lm.sources_file(900001) == es.sources_file(tmp_path / "events", 900001)

    lm.events.mkdir(parents=True, exist_ok=True)
    ev.write_sources(lm.sources_file(900001), 900001, [
        {"source": "tokeye_track", "diag": "mhr", "channel": 4, "pass_name": "wide",
         "status": "ok", "reason": "", "t_cov0_s": 0.0, "t_cov1_s": 6.0, "n_events": 0},
        {"source": "dalpha_lh", "diag": "filterscopes", "channel": -1, "pass_name": "",
         "status": "skipped", "reason": "KeyError: no group 'co2'", "n_events": 0},
    ], run_id="test-run")

    result = join.join([900001, 900002], labelmaker_root=tmp_path)
    src = result.sources
    assert list(src.columns) == list(es.SOURCES_COLUMNS)
    assert src.dtypes.astype(str).to_dict() == es.SOURCES_DTYPES
    assert len(src) == 2 and set(src["run_id"]) == {"test-run"}
    assert set(src["status"]) == {"ok", "skipped"}
    assert np.isnan(src.loc[src["source"] == "dalpha_lh", "t_cov0_s"]).all()
    m = result.manifest
    assert m["sources_by_shot"][900001]["has_observed_products"] is True
    assert (m["n_shots_with_observed_products"], m["n_shots_unprocessed"]) == (1, 1)


def test_a_join_with_no_source_files_at_all_still_writes_a_typed_empty_table(tmp_path):
    from ideate.labels import event_sources as es

    db = tmp_path / "db"
    db.mkdir()
    result = join.join([900001], labelmaker_root=tmp_path)
    join.write_tables(db, result.labels_wide, result.events, result.claims, result.manifest,
                      sources_df=result.sources)
    back = pd.read_parquet(db / "event_sources.parquet")
    assert back.empty and list(back.columns) == list(es.SOURCES_COLUMNS)
    assert json.loads((db / "manifest.json").read_text())["labels"]["n_shots_unprocessed"] == 1


# ------------------------------------------------------------- refreshing has_frame_codes


def test_the_join_refreshes_has_frame_codes_from_the_directory(tmp_path, ideate_db, monkeypatch):
    """The column is set at BUILD time and the encode is a later job, so a database built before
    the encode said 13 true while all 500 caches existed. The join is the step that runs after
    the long jobs, so it is where the flag is brought back in line."""
    from ideate import config
    from ideate.schema import ShotRecord
    from ideate.shotdb import build as build_mod

    db_dir = ideate_db / "db"
    paths = config.load_paths()
    codes = build_mod.frame_codes_dirs(paths)[0]
    codes.mkdir(parents=True, exist_ok=True)
    for shot in (100, 200):
        (codes / f"{shot}.pt").write_bytes(b"")

    before = pd.read_parquet(db_dir / "shots.parquet")
    assert not before["has_frame_codes"].any()

    got = build_mod.refresh_frame_codes(db_dir, paths)
    assert got["n_shots"] == 4
    assert got["n_has_frame_codes"] == 2
    assert got["n_changed"] == 2 and got["refreshed"] is True

    after = pd.read_parquet(db_dir / "shots.parquet")
    assert dict(zip(after.index, after["has_frame_codes"], strict=True)) == {
        100: True, 101: False, 200: True, 201: False
    }
    # ... and inside record_json too, which is what `describe_shot` hands back.
    for shot in (100, 200):
        assert ShotRecord.model_validate_json(after.loc[shot, "record_json"]).has_frame_codes
    assert not ShotRecord.model_validate_json(after.loc[101, "record_json"]).has_frame_codes


def test_refreshing_twice_changes_nothing_and_a_missing_table_is_not_an_error(tmp_path, ideate_db):
    from ideate import config
    from ideate.shotdb import build as build_mod

    db_dir = ideate_db / "db"
    paths = config.load_paths()
    assert build_mod.refresh_frame_codes(db_dir, paths)["n_changed"] == 0
    before = (db_dir / "shots.parquet").read_bytes()
    assert build_mod.refresh_frame_codes(db_dir, paths)["refreshed"] is False
    assert (db_dir / "shots.parquet").read_bytes() == before

    absent = build_mod.refresh_frame_codes(tmp_path / "nowhere", paths)
    assert absent["refreshed"] is False and absent["n_shots"] == 0


def test_write_tables_records_the_refreshed_count_in_its_own_manifest_block(tmp_path):
    db = tmp_path / "db"
    db.mkdir()
    (db / "manifest.json").write_text(json.dumps({"n_shots": 2}))
    result = join.join([], labelmaker_root=tmp_path)
    join.write_tables(
        db, result.labels_wide, result.events, result.claims, result.manifest,
        sources_df=result.sources,
        join_block={"frame_codes": {"n_shots": 500, "n_has_frame_codes": 500, "n_changed": 487}},
    )
    manifest = json.loads((db / "manifest.json").read_text())
    assert manifest["labels_join"]["frame_codes"]["n_has_frame_codes"] == 500
    assert manifest["labels_join"]["written_at"] == manifest["labels"]["written_at"]
    assert manifest["n_shots"] == 2  # still merged, never replaced


# ------------------------------------------------------------------------------ write_tables


def test_write_tables_writes_three_tables_and_merges_the_counts(tmp_path):
    db = tmp_path / "db"
    db.mkdir()
    (db / "manifest.json").write_text(json.dumps({"n_shots": 2, "built_at": "yesterday"}))
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool))
    result = join.join([900001], labelmaker_root=tmp_path, rules=[rule()])

    join.write_tables(db, result.labels_wide, result.events, result.claims, result.manifest)

    assert not list(db.glob("*.tmp")) and not list(db.glob("*.part"))
    back = pd.read_parquet(db / "labels_wide.parquet")
    assert list(back.columns) == list(join.LABELS_WIDE_COLUMNS)
    assert back["n_intervals"].dtype == "Int32"  # the null survives the round trip
    # Read back through labelmaker's own reader, which is how a consumer reads a table in this
    # schema and is what restores its dtypes: pyarrow hands a string column back as pandas' `str`,
    # and `read_events`'s `.astype(DTYPES)` is the cast that makes a thousand shots concatenate.
    assert {c: str(d) for c, d in ev.read_events(db / "events.parquet").dtypes.items()} == dict(
        ev.DTYPES
    )
    assert list(pd.read_parquet(db / "text_claims.parquet").columns) == list(join.CLAIMS_COLUMNS)
    manifest = json.loads((db / "manifest.json").read_text())
    assert manifest["n_shots"] == 2 and manifest["built_at"] == "yesterday"  # merged, not replaced
    assert manifest["labels"]["n_shots_with_labels"] == 1
    assert manifest["labels"]["n_shots_with_events"] == 0
    assert manifest["labels"]["n_forecast_events"] == 1
    assert manifest["labels"]["written_at"]


def test_a_failed_write_leaves_the_old_tables_in_place(tmp_path, monkeypatch):
    db = tmp_path / "db"
    db.mkdir()
    empty = join.join([], labelmaker_root=tmp_path)
    join.write_tables(db, empty.labels_wide, empty.events, empty.claims, empty.manifest)
    before = (db / "labels_wide.parquet").read_bytes()

    real = pd.DataFrame.to_parquet

    def boom(self, path, *a, **kw):
        if "events" in str(path):
            raise OSError("disk full")
        return real(self, path, *a, **kw)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        join.write_tables(db, empty.labels_wide, empty.events, empty.claims, empty.manifest)

    assert (db / "labels_wide.parquet").read_bytes() == before
    assert not list(db.glob("*.tmp"))


# --------------------------------------------------------------- thresholds and the rules file


def test_a_threshold_comes_from_the_cards_own_outputs_block():
    card = {"labelmaker": {"slug": "d3d_x", "outputs": [
        {"name": "p", "task": "binary", "threshold": 0.7},
        {"name": "q", "task": "regression"},
    ]}}
    assert join.thresholds_from_card(card) == {"d3d_x/p": 0.7}


def test_the_real_cards_thresholds_are_all_finite_and_named():
    """Whatever the cards record, `thr` may only ever be a real operating point of a real label."""
    for key, thr in join.card_thresholds().items():
        slug, label = key.split("/", 1)
        assert slug and label
        assert np.isfinite(thr)


def test_the_configured_forecast_rules_name_real_models_and_real_labels():
    """`configs/ideate/labels.yaml`'s `forecasts:` block against the model cards it points at."""
    from labelmaker.models import registry

    rules = join.forecast_rules()
    assert rules, "labels.yaml records no forecast rules"
    for r in rules:
        card = registry.read_card(r.slug)["labelmaker"]
        assert card["status"] == "implemented"
        assert r.label in {o["name"] for o in card["outputs"]}
        assert 0.0 < r.thr < 1.0
        assert np.isfinite(r.horizon_s) and r.horizon_s > 0.0
        assert r.phenomenon
    assert {r.phenomenon for r in rules} == {"tearing", "elm"}


TM = "d3d_tearing_time_to_event_dsm/tm_risk_250ms"


def test_a_card_threshold_and_a_config_threshold_are_told_apart(tmp_path):
    """`thr` alone cannot say whose number it is. A card's operating point is the model authors'
    word about their own model; the `forecasts:` block's level is ideate's alarm choice, and a
    consumer weighing an alarm has to be able to tell one from the other row by row."""
    y = np.full(T.size, 0.9)
    write_labels(tmp_path, 900001, "slug_a", {"p": y, "q": y, "r": y}, np.ones(T.size, bool))

    df = join.labels_wide(
        [900001], labelmaker_root=tmp_path,
        thresholds={"slug_a/p": (0.7, "card"), "slug_a/q": (0.2, "config")},
    ).set_index("label")

    assert (df.loc["p", "thr"], df.loc["p", "thr_source"]) == (pytest.approx(0.7), "card")
    assert (df.loc["q", "thr"], df.loc["q", "thr_source"]) == (pytest.approx(0.2), "config")
    assert np.isnan(df.loc["r", "thr"]) and df.loc["r", "thr_source"] == ""


def test_a_threshold_must_name_its_source(tmp_path):
    """A bare number would be recorded as a card's operating point whatever it really was, which
    is the confusion `thr_source` exists to end."""
    write_labels(tmp_path, 900001, "slug_a", {"p": T}, np.ones(T.size, bool))
    with pytest.raises(ValueError, match="source"):
        join.labels_wide([900001], labelmaker_root=tmp_path, thresholds={"slug_a/p": 0.5})


def test_a_card_threshold_beats_the_config_in_both_tables_at_once(tmp_path, monkeypatch):
    """One map feeds `labels_wide` and the forecast events, so the two tables cannot disagree
    about the number an alarm was raised at. The card's 0.25 and the rule's 0.2 pick different
    runs out of this series, so a table still reading the rule's would show it."""
    y = np.array([0.1, 0.22, 0.3, 0.1, 0.1, 0.1])
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm", {"tm_risk_250ms": y},
                 np.ones(T.size, bool))
    monkeypatch.setattr(join, "card_thresholds", lambda: {TM: 0.25})

    result = join.join([900001], labelmaker_root=tmp_path, rules=[rule(thr=0.2)])

    row = result.labels_wide.iloc[0]
    assert (row["thr"], row["thr_source"]) == (pytest.approx(0.25), "card")
    assert row["frac_above"] == pytest.approx(1 / 6, rel=1e-6)   # the rule's 0.2 would say 2/6
    assert row["n_intervals"] == 1
    assert len(result.events) == 1
    event = result.events.iloc[0]
    assert event["t0_s"] == pytest.approx(0.2)                   # the rule's 0.2 would start at 0.1
    attrs = json.loads(event["attrs"])
    assert attrs["thr"] == pytest.approx(0.25)
    assert attrs["thr_source"] == "card"
    assert attrs["n_samples"] == 1


def test_the_manifest_records_every_threshold_with_its_source(tmp_path, monkeypatch):
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool))
    monkeypatch.setattr(join, "card_thresholds", lambda: {"slug_a/p": 0.7})

    result = join.join([900001], labelmaker_root=tmp_path, rules=[rule(thr=0.2)])

    assert result.manifest["thresholds"] == {
        "slug_a/p": [pytest.approx(0.7), "card"],
        TM: [pytest.approx(0.2), "config"],
    }
    assert result.manifest["thresholds_from_card"] == 1
    assert result.manifest["thresholds_from_config"] == 1

    db = tmp_path / "db"
    join.write_tables(db, result.labels_wide, result.events, result.claims, result.manifest)
    block = json.loads((db / "manifest.json").read_text())["labels"]
    assert block["thresholds"][TM] == [pytest.approx(0.2), "config"]


def test_the_manifest_records_which_forecast_rules_were_applied(tmp_path):
    """The rule set is a file that will be edited. A table built from it has to say which version
    of it it was built from, or a `frac_above` from last week is unattributable."""
    from_config = join.join([], labelmaker_root=tmp_path).manifest["forecast_rules"]
    assert from_config["path"].endswith("labels.yaml")
    assert len(from_config["sha256"]) == 64
    assert from_config["n_rules"] == len(join.forecast_rules())

    explicit = join.join([], labelmaker_root=tmp_path, rules=[rule()]).manifest["forecast_rules"]
    assert explicit["path"] is None                 # no file said so
    assert explicit["n_rules"] == 1
    assert explicit["sha256"] != from_config["sha256"]
    changed = join.join([], labelmaker_root=tmp_path, rules=[rule(thr=0.3)])
    assert changed.manifest["forecast_rules"]["sha256"] != explicit["sha256"]


# -------------------------------------------------------------------------------------- CLI


def test_cli_labels_join(tmp_path, capsys, paths):
    write_labels(tmp_path, 900001, "d3d_tearing_time_to_event_dsm",
                 {"tm_risk_250ms": np.full(T.size, 0.9)}, np.ones(T.size, bool))

    code = cli.main([
        "labels", "join", "--shots", "900001", "900002",
        "--labelmaker-root", str(tmp_path), "--db", str(paths.db_dir), "--no-text",
    ])

    assert code == 0
    out = capsys.readouterr().out
    assert "labels_wide" in out
    assert (paths.db_dir / "labels_wide.parquet").exists()
    assert (paths.db_dir / "events.parquet").exists()
    assert (paths.db_dir / "text_claims.parquet").exists()
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["labels"]["n_shots_with_labels"] == 1
    assert manifest["labels"]["n_shots_with_events"] == 0
