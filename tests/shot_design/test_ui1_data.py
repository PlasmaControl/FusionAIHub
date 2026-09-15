"""Read-only summary slots and structured descriptions; all tables live in tmp_path."""

import pandas as pd
import pytest

from shot_design import schema
from shot_design.mcp import tools
from shot_design.retrieval import describe, phenomena, rank
from shot_design.shotdb.store import ShotDB

from .test_describe import entries, record
from .test_phenomena import _db_with, _event
from .test_ui import client, local_auxiliary_paths  # noqa: F401


def write_summaries(db_dir, **over):
    values = {
        "shot": pd.Series([100], dtype="int32"),
        "summary": ["Reach H-mode. Target met. An EHO appeared."],
        "goal": ["Reach H-mode."], "outcome": ["Target met."],
        "findings": ["An EHO appeared."], "model": ["offline-model"],
        "prompt_version": [1], "written_at": ["2026-09-15T00:00:00Z"],
    }
    values.update(over)
    pd.DataFrame(values).to_parquet(db_dir / "summaries.parquet", index=False)


def test_absent_summary_is_none_on_records_search_and_locate(ideate_db):
    db = _db_with(ideate_db, [_event(100, "observed")])
    assert db.get(100).summary is None
    assert rank.search(schema.QueryState(ref_shot=101), db).items[0].summary is None
    assert phenomena.locate("tearing", db)[0].summary is None
    assert schema.SearchHit(shot=100, phenomenon="tearing", score=1).summary is None


def test_summary_table_is_joined_by_shot_without_mutating_stored_records(ideate_db):
    db_dir = ideate_db / "db"
    before = (db_dir / "shots.parquet").read_bytes()
    _db_with(ideate_db, [_event(100, "observed")])
    write_summaries(db_dir)
    db = ShotDB.load(db_dir)
    summary = "Reach H-mode. Target met. An EHO appeared."
    assert db.get(100).summary == summary
    assert db.get(101).summary is None
    assert db.summaries.iloc[0]["model"] == "offline-model"
    assert next(h for h in rank.search(schema.QueryState(ref_shot=101), db).items
                if h.shot == 100).summary == summary
    assert phenomena.locate("tearing", db)[0].summary == summary
    assert (db_dir / "shots.parquet").read_bytes() == before
    tools.reset_cache()
    assert tools.describe_shot(100)["summary"] == summary
    assert tools.describe_shot(100)["record"]["summary"] == summary


@pytest.mark.parametrize("value", [None, ""])
def test_null_or_empty_summary_is_none(ideate_db, value):
    write_summaries(ideate_db / "db", summary=[value])
    assert ShotDB.load(ideate_db / "db").get(100).summary is None


def test_corrupt_summary_table_does_not_hide_core_records(ideate_db):
    (ideate_db / "db/summaries.parquet").write_bytes(b"torn parquet")
    db = ShotDB.load(ideate_db / "db")
    assert db.get(100).summary is None
    assert "summaries" in db.load_errors


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
    assert parts["summary"] is None
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


def test_parts_separate_forecasts_and_bound_observed_intervals(ideate_db):
    db = _db_with(ideate_db, [
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
    assert row["coverage_note"] and row["coverage_windows"]
    assert describe.describe_parts(db.get(100), db=db) == parts


def test_shot_api_adds_parts_units_and_optional_summary(client, ideate_db):  # noqa: F811
    write_summaries(ideate_db / "db")
    tools.reset_cache()
    data = client.get("/api/shot/100").json()
    assert data["summary"] == data["record"]["summary"] == data["describe_parts"]["summary"]
    assert data["units"]["ip_mean"] == "A"
    assert data["description"] == tools.describe_shot(100)["description"]
    assert data["describe_parts"]["scalars"]


def test_bad_summary_table_is_reported_by_shot_api(client, ideate_db):  # noqa: F811
    (ideate_db / "db/summaries.parquet").write_bytes(b"broken")
    tools.reset_cache()
    data = client.get("/api/shot/100").json()
    assert data["summary"] is None
    assert any("summaries.parquet" in c for c in data["caveats"])
