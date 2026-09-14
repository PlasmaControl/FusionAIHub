"""The common CSV contract rejects corrupt labels before they become events."""
import pandas as pd
import pytest

from labelmaker.events import databases as db


def format_frame(**over):
    row = {"shot": 158015, "t0_s": 2.613, "t1_s": 2.613, "phenomenon": "rwm",
               "evidence_kind": "database", "source": "database:fixture",
               "confidence": "", "attrs": '{"NTOR": 2}'}
    row.update(over)
    return pd.DataFrame([row])


def test_common_schema_accepts_points_intervals_and_empty_confidence():
    frame = db.validate_format(format_frame(t1_s=3.0))
    assert list(frame.columns) == list(db.FORMAT_COLUMNS)
    assert frame.shot.dtype.kind == "i"
    assert frame.confidence.isna().all()
    assert frame.t1_s.item() == 3.0
    assert db.validate_format(format_frame()).t0_s.item() == 2.613


@pytest.mark.parametrize("over, field", [
    ({"phenomenon": "resistive_wall_mode"}, "phenomenon"),
    ({"attrs": "not json"}, "attrs"),
    ({"attrs": "[]"}, "attrs"),
    ({"attrs": '{"bad": NaN}'}, "attrs"),
    ({"attrs": '{"bad": 1e999}'}, "attrs"),
    ({"t1_s": 2.0}, "t1_s"),
    ({"t0_s": float("inf")}, "t0_s"),
    ({"t0_s": "soon"}, "t0_s"),
    ({"shot": 158015.5}, "shot"),
    ({"shot": float("inf")}, "shot"),
    ({"confidence": 1.1}, "confidence"),
    ({"confidence": "unknown"}, "confidence"),
    ({"confidence": float("inf")}, "confidence"),
    ({"source": ""}, "source"),
    ({"evidence_kind": "certain"}, "evidence_kind"),
])
def test_common_schema_rejects_bad_values_with_file_and_field(over, field):
    with pytest.raises(db.DatabaseError, match=field) as exc:
        db.validate_format(format_frame(**over), where="bad.csv")
    assert "bad.csv" in str(exc.value)


@pytest.mark.parametrize("change", ["missing", "extra", "reordered"])
def test_common_schema_rejects_bad_columns(change):
    frame = format_frame()
    if change == "missing":
        frame = frame.drop(columns="attrs")
    elif change == "extra":
        frame["extra"] = 1
    else:
        frame = frame[list(reversed(frame.columns))]
    with pytest.raises(db.DatabaseError, match="columns"):
        db.validate_format(frame)


def test_empty_table_is_a_valid_common_schema():
    frame = db.validate_format(format_frame().iloc[:0])
    assert frame.empty
    assert frame.shot.dtype.kind == "i"
