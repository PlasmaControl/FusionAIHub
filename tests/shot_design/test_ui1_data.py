"""Stored blurbs and structured descriptions; all tables live in tmp_path."""

import json

import pandas as pd
import pytest

from shot_design import schema
from shot_design.mcp import tools
from shot_design.retrieval import describe, phenomena, rank
from shot_design.shotdb.store import ShotDB

from .test_describe import entries, record
from .test_phenomena import _db_with, _event
from .test_ui import client, local_auxiliary_paths  # noqa: F401

BLURB = "Reach H-mode. Target met. An EHO appeared."


def write_blurb(db_dir, value=BLURB, source="llm"):
    table = pd.read_parquet(db_dir / "shots.parquet")
    table["record_json"] = table["record_json"].map(lambda raw: json.dumps({
        **json.loads(raw), "blurb": "Stale build-time text.", "blurb_source": "template",
    }))
    table["blurb"] = pd.Series(index=table.index, dtype="string")
    table["blurb_source"] = pd.Series(index=table.index, dtype="string")
    table["blurb_model"] = "private-model-provenance"
    table["blurb_prompt_version"] = 5
    table.loc[100, ["blurb", "blurb_source"]] = [value, source]
    table.to_parquet(db_dir / "shots.parquet")


def assert_blurb_fields(row, value, source):
    fields = row if isinstance(row, dict) else row.model_dump(mode="json")
    assert fields["blurb"] == value
    assert fields["blurb_source"] == source
    assert not {"summary", "summaries", "blurb_model", "blurb_prompt_version"} & fields.keys()
    if not isinstance(row, dict):
        assert not hasattr(row, "summary") and not hasattr(row, "summaries")


def test_absent_blurb_is_none_on_records_search_and_locate(shot_design_db):
    _db_with(shot_design_db, [_event(100, "observed")])
    write_blurb(shot_design_db / "db")
    pd.read_parquet(shot_design_db / "db/shots.parquet").drop(
        columns=["blurb", "blurb_source"],
    ).to_parquet(
        shot_design_db / "db/shots.parquet"
    )
    db = ShotDB.load(shot_design_db / "db")
    for row in (db.get(100), rank.search(schema.QueryState(ref_shot=101), db).items[0],
                phenomena.locate("tearing", db)[0],
                schema.SearchHit(shot=100, phenomenon="tearing", score=1)):
        assert_blurb_fields(row, None, None)


@pytest.mark.parametrize("source", ["llm", "template", None])
def test_blurb_columns_reach_records_search_and_locate_without_writes(shot_design_db, source):
    db_dir = shot_design_db / "db"
    _db_with(shot_design_db, [_event(100, "observed")])
    write_blurb(db_dir, source=source)
    before = (db_dir / "shots.parquet").read_bytes()
    db = ShotDB.load(db_dir)
    hit = next(h for h in rank.search(schema.QueryState(ref_shot=101), db).items
               if h.shot == 100)
    for row in (db.get(100), hit, phenomena.locate("tearing", db)[0]):
        assert_blurb_fields(row, BLURB, source)
    assert_blurb_fields(db.get(101), None, None)
    assert not hasattr(db, "summary") and not hasattr(db, "summaries")
    assert "summaries" not in db.load_errors
    assert (db_dir / "shots.parquet").read_bytes() == before
    tools.reset_cache()
    reply = tools.describe_shot(100)
    assert_blurb_fields(reply, BLURB, source)
    assert_blurb_fields(reply["record"], BLURB, source)
    assert schema.SearchHit is schema.PhenomenonHit


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, "", "  \t\n"])
def test_null_or_empty_blurb_fields_are_none(shot_design_db, value):
    write_blurb(shot_design_db / "db", value=value, source=value)
    assert_blurb_fields(ShotDB.load(shot_design_db / "db").get(100), None, None)


@pytest.mark.parametrize("column", ["blurb", "blurb_source"])
def test_missing_blurb_column_never_uses_record_json(shot_design_db, column):
    write_blurb(shot_design_db / "db")
    path = shot_design_db / "db/shots.parquet"
    pd.read_parquet(path).drop(columns=[column]).to_parquet(path)
    rec = ShotDB.load(shot_design_db / "db").get(100)
    assert_blurb_fields(rec, None if column == "blurb" else BLURB,
                        None if column == "blurb_source" else "llm")


def test_record_rejects_unknown_blurb_source():
    data = record().model_dump(mode="json")
    with pytest.raises(ValueError, match="blurb_source"):
        schema.ShotRecord.model_validate({**data, "blurb_source": "unknown"})


@pytest.mark.parametrize("column,unit", [
    ("ip_mean", "A"), ("ip_slope", "A/s"), ("bt_mean", "T"),
    ("aminor_mean", "m"), ("vloop_mean", "V"), ("n1rms_peak", "G"),
    ("wmhd_peak", "J"), ("volume_mean", "m^3"), ("ne_line_mean", "m/cm3"),
    ("dalpha_std", "ph/cm2/sr/s"), ("pnbi_total_mean", "W"),
    ("pech_total_on_frac", ""), ("betan_mean", ""), ("unknown_slope", ""),
])
def test_scalar_units_follow_registry_and_statistic(column, unit):
    assert describe.scalar_units([column]) == {column: unit}


def test_parts_preserve_raw_values_units_and_one_complete_attributed_quote():
    text = "One operator's full note. " * 12
    rec = record(human=entries(("PHYSICS_OPERATOR", "a", "10:00", text),
                               ("CHIEF_OPERATOR", "b", "10:05", "Other note.")))
    parts = describe.describe_parts(rec)
    assert parts["header"] == "Shot 161172 (2015-01-13)."
    assert parts["blurb"] is None and parts["blurb_source"] is None
    assert parts["segment"] == {"name": "flat_top", "t0_s": .5295, "t1_s": 5.199}
    scalars = {s["name"]: s for s in parts["scalars"]}
    assert scalars["ip_mean"] == {"name": "ip_mean", "value": 985281.3, "units": "A"}
    assert scalars["pech_total_on_frac"]["value"] == 0
    assert parts["outcome"]["ip_target_err"] == -.009
    assert parts["operator_quote"] == {
        "text": text.strip(), "role": "PHYSICS_OPERATOR", "author": "a", "time": "10:00",
    }
    assert isinstance(describe.describe(rec), str)


def test_parts_never_fall_back_to_another_segment_or_invent_a_measurement():
    rec = record()
    rec.segments[0].raw.update(ip_mean=None, bt_mean=float("inf"))
    parts = describe.describe_parts(rec)
    scalars = {s["name"]: s["value"] for s in parts["scalars"]}
    assert scalars["ip_mean"] is None and scalars["bt_mean"] is None
    missing = describe.describe_parts(rec, "ramp_up")
    assert missing["segment"] is None and missing["scalars"] == []
    assert missing["operator_quote"] is None


def test_parts_separate_forecasts_and_bound_observed_intervals(shot_design_db):
    db = _db_with(shot_design_db, [
        *[_event(100, f"seen-{i}", t0_s=2 + i / 10, t1_s=2.05 + i / 10)
          for i in range(5)],
        _event(100, "risk", source="label_forecast", evidence_kind="forecast",
               phenomenon="tearing", t0_s=3, t1_s=4),
    ])
    parts = describe.describe_parts(db.get(100), db=db)
    row = next(p for p in parts["phenomena"] if p["id"] == "tearing")
    assert row["n_observed"] == 5 and row["n_forecast"] == 1
    assert len(row["first_intervals"]) == 3
    assert all(iv["evidence_kind"] == "detector" for iv in row["first_intervals"])
    assert [iv["event_id"] for iv in row["intervals"]] == [f"seen-{i}" for i in range(5)]
    assert row["first_intervals"] == row["intervals"][:3]
    assert all(iv["evidence_kind"] == "detector" for iv in row["intervals"])
    assert row["coverage_note"] and row["coverage_windows"]
    assert describe.describe_parts(db.get(100), db=db) == parts


def test_locate_and_events_share_the_same_full_segment_domain(
    client, shot_design_db,  # noqa: F811
):
    _db_with(shot_design_db, [_event(100, "observed")])
    table = pd.read_parquet(shot_design_db / "db/shots.parquet")
    rec = json.loads(table.loc[100, "record_json"])
    rec["segments"].append({"name": "full", "t0_ms": -3400, "t1_ms": 10501})
    table.loc[100, "record_json"] = json.dumps(rec)
    table.to_parquet(shot_design_db / "db/shots.parquet")
    tools.reset_cache()
    hit = next(h for h in client.get("/api/locate?phenomenon=tearing").json()
               if h["shot"] == 100)
    events = client.get("/api/shot/100/events").json()
    assert hit["domain"] == events["domain"] == {
        "t0_s": -4, "t1_s": 12, "source": "full segment",
    }


@pytest.mark.parametrize("segments,expected", [
    ([("full", 13, 6944), ("flat_top", -10000, 95000)],
     {"t0_s": -2, "t1_s": 8, "source": "full segment"}),
    ([("full", -3400, 10501)],
     {"t0_s": -4, "t1_s": 12, "source": "full segment"}),
    ([("ramp_up", -2100, 987), ("flat_top", 987, 5145), ("ramp_down", 5145, 8500)],
     {"t0_s": -3, "t1_s": 9, "source": "segments"}),
    ([], {"t0_s": -2, "t1_s": 8, "source": "default"}),
    ([("full", float("nan"), 6944), ("flat_top", 987, 5145)],
     {"t0_s": -2, "t1_s": 8, "source": "segments"}),
    ([("full", 8000, 1000)], {"t0_s": -2, "t1_s": 8, "source": "default"}),
])
def test_events_domain_uses_record_segments_not_coverage(
    client, shot_design_db, segments, expected,  # noqa: F811
):
    """Shot 199607 spans 0.013..6.944 s despite actuator coverage -10..95 s."""
    from shot_design.labels import event_sources

    table = pd.read_parquet(shot_design_db / "db/shots.parquet")
    rec = json.loads(table.loc[100, "record_json"])
    rec["segments"] = [
        {"name": name, "t0_ms": a, "t1_ms": b} for name, a, b in segments
    ]
    table.loc[100, "record_json"] = json.dumps(rec)
    table.to_parquet(shot_design_db / "db/shots.parquet")
    event_sources.write_sources(shot_design_db / "db/event_sources.parquet", [
        event_sources.source_row(100, "actuator", diag="ech_power_total",
                                 t_cov0_s=-10, t_cov1_s=95, n_events=0),
    ])
    tools.reset_cache()
    for params in ({}, {"t0_s": 1.0, "t1_s": 2.0, "phenomenon": "eho"}):
        response = client.get("/api/shot/100/events", params=params)
        assert response.status_code == 200
        payload = response.json()
        assert payload.pop("domain") == expected
        assert isinstance(payload.pop("phenomena"), list)
        assert payload == tools.get_events(100, **params)
        if not params:  # The EHO filter correctly excludes actuator coverage.
            coverage = payload["coverage"]["sources"][0]
            assert (coverage["t_cov0_s"], coverage["t_cov1_s"]) == (-10, 95)


@pytest.mark.parametrize("value,source", [
    (BLURB, "llm"), (BLURB, "template"), (BLURB, None), (None, None),
])
def test_shot_search_locate_apis_carry_stored_blurb_fields(
    client, shot_design_db, value, source,  # noqa: F811
):
    _db_with(shot_design_db, [_event(100, "observed")])
    write_blurb(shot_design_db / "db", value=value, source=source)
    tools.reset_cache()
    data = client.get("/api/shot/100").json()
    for row in (data, data["record"], data["describe_parts"]):
        assert_blurb_fields(row, value, source)
    assert data["units"]["ip_mean"] == "A"
    assert data["description"] == tools.describe_shot(100)["description"]
    assert data["describe_parts"]["scalars"]
    results = client.post("/api/search", json={"ref_shot": 101}).json()["results"]
    hit = next(row for row in results if row["shot"] == 100)
    assert_blurb_fields(hit, value, source)
    located = client.get("/api/locate", params={"phenomenon": "tearing"}).json()
    assert_blurb_fields(next(row for row in located if row["shot"] == 100), value, source)
