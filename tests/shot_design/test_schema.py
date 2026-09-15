"""Ported from shot-recommender-system (shotrec) @565d548."""

# tests/test_schema.py
import json
from datetime import UTC, datetime

from shot_design import schema


def _record() -> schema.ShotRecord:
    seg = schema.Segment(
        name="flat_top",
        t0_ms=800.0,
        t1_ms=4800.0,
        raw={"ip_mean": 1.2e6, "pnbi_total_mean": None},
        derived={"betan_mean": 1.9},
    )
    return schema.ShotRecord(
        shot=161172,
        campaign="2014_2015",
        segments=[seg],
        derived_provenance={
            "betan": schema.Provenance(tool="EFIT", version="01", tree="efit01", assumed=True)
        },
        raw_sources={"ip": "staged"},
        human=schema.HumanTier(mpid="2014-21-20", mp_title="QH-mode access at low NBI torque"),
        labels=schema.Labels(regime="QH", regime_source="text", operational={"disruption_free"}),
        outcome=schema.Outcome(ip_target_err=-0.02, ip_target_hit=True),
        coverage={"ip": "present", "pech_LEIA": "unavailable"},
        built_at=datetime.now(UTC),
        builder_sha="abc123",
    )


def test_record_round_trips_through_json():
    rec = _record()
    back = schema.ShotRecord.model_validate_json(rec.model_dump_json())
    assert back == rec
    assert json.loads(rec.model_dump_json())["labels"]["operational"] == ["disruption_free"]


def test_labels_operational_serializes_sorted():
    # A single-element set can't distinguish "sorted" from "insertion order", so this uses
    # enough elements, in an unsorted insertion order, that only an actual sort produces this
    # exact sequence -- guarding the round-trip determinism the field_serializer exists for.
    labels = schema.Labels(operational={"z_flag", "a_flag", "m_flag", "disruption_free"})
    dumped = json.loads(labels.model_dump_json())["operational"]
    assert dumped == sorted(dumped) == ["a_flag", "disruption_free", "m_flag", "z_flag"]


def test_segment_lookup_and_summary():
    rec = _record()
    assert rec.segment("flat_top").raw["ip_mean"] == 1.2e6
    assert rec.segment("ramp_up") is None
    s = schema.to_summary(rec, description="Shot 161172 ...")
    assert s.flat_top["betan_mean"] == 1.9 and s.coverage_fraction == 0.5


def test_query_state_defaults():
    q = schema.QueryState(
        text="QH-mode at low torque", constraints={"ip_mean": schema.Range(lo=1.0e6, hi=1.4e6)}
    )
    assert q.segment == "flat_top" and q.n == 10 and q.actuators == {}
