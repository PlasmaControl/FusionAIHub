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


def _spec(slug: str, name: str, *, sha: str = "deadbeef", task: str = "binary") -> LabelSpec:
    return LabelSpec(
        name=name,
        task=task,
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
) -> Path:
    """One model's labels for one shot, through labelmaker's own writer."""
    path = Path(root) / "labels" / f"{shot}_labels.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = [_spec(slug, name, sha=sha) for name in series]
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

    df = join.labels_wide([900001], labelmaker_root=tmp_path, thresholds={"slug_a/p": 0.5})

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

    row = join.labels_wide([900001], labelmaker_root=tmp_path, thresholds={"slug_a/p": 0.5}).iloc[0]

    assert row["n_valid"] == 0
    assert row["valid_frac"] == pytest.approx(0.0)
    for col in ("max_valid", "mean_valid", "p95_valid", "frac_above", "first_above_t_s"):
        assert np.isnan(row[col]), col


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
