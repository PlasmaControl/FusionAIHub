"""The gated sawtooth review source: physics rows, the q = 1 gate, table rows."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/labeler"))
import sawtooth_source as source

ABSENT, PRESENT, UNCERTAIN, NOT_OBSERVABLE = 0, 1, 2, 3


def shard(path, rows):
    pd.DataFrame(
        [
            {
                "shot": shot,
                "category": category,
                "t_start": a,
                "t_end": b,
                "crowd": crowd,
                "confidence": confidence,
                "attrs": json.dumps(attrs),
            }
            for shot, category, a, b, crowd, confidence, attrs in rows
        ]
    ).to_csv(path, index=False)


def test_physics_rows_join_shards_close_rounding_gaps_and_fill_wider_ones(tmp_path):
    shard(
        tmp_path / "cohort-000.csv",
        [
            (1, ABSENT, 0.0, 100.0, True, 0.9, {"state": "absent"}),
            (1, PRESENT, 100.05, 200.0, True, 0.8, {"state": "present"}),
            (1, PRESENT, 150.0, 150.0, False, 1.0, {"state": "crash"}),  # a point
        ],
    )
    shard(
        tmp_path / "population-000.csv",
        [
            (1, ABSENT, 0.0, 100.0, True, 0.9, {"state": "absent"}),  # duplicate
            (1, UNCERTAIN, 205.0, 300.0, True, 1.0, {"state": "uncertain"}),
            (
                2,
                UNCERTAIN,
                0.0,
                50.0,
                True,
                1.0,
                {"state": "q_prior", "reason": "q_prior_untested"},
            ),
        ],
    )
    shard(
        tmp_path / "population-001.csv",
        [(2, NOT_OBSERVABLE, 50.0, 80.0, True, 1.0, {"state": "unassessed"})],
    )
    rows = source.physics_rows(tmp_path)
    assert rows[1] == [
        (ABSENT, 0.0, 100.05, 0.9, "absent"),  # the 0.05 ms gap is closed
        (PRESENT, 100.05, 200.0, 0.8, "present"),
        (NOT_OBSERVABLE, 200.0, 205.0, "", "gap"),  # a 5 ms gap is not observable
        (UNCERTAIN, 205.0, 300.0, 1.0, "uncertain"),
    ]
    assert rows[2] == [
        (UNCERTAIN, 0.0, 50.0, 1.0, "q_prior_untested"),  # the reason names it
        (NOT_OBSERVABLE, 50.0, 80.0, 1.0, "unassessed"),
    ]
    shard(
        tmp_path / "population-002.csv",
        [(2, ABSENT, 70.0, 90.0, True, 1.0, {"state": "absent"})],
    )
    with pytest.raises(ValueError, match="overlap"):
        source.physics_rows(tmp_path)


def test_gated_splits_a_row_across_several_no_q1_spans_and_keeps_present():
    rows = [
        (UNCERTAIN, 0.0, 1000.0, 0.5, "uncertain"),
        (PRESENT, 1000.0, 1100.0, 0.9, "present"),
        (NOT_OBSERVABLE, 1100.0, 1300.0, "", "unassessed"),
    ]
    absent = [(100.0, 200.0), (500.0, 700.0), (900.0, 1050.0), (1250.0, 1400.0)]
    out = source.gated(rows, absent, keep=(PRESENT, ABSENT))
    assert out == [
        (UNCERTAIN, 0.0, 100.0, 0.5, "uncertain"),
        (ABSENT, 100.0, 200.0, "", "no_q1:uncertain"),
        (UNCERTAIN, 200.0, 500.0, 0.5, "uncertain"),
        (ABSENT, 500.0, 700.0, "", "no_q1:uncertain"),
        (UNCERTAIN, 700.0, 900.0, 0.5, "uncertain"),
        (ABSENT, 900.0, 1000.0, "", "no_q1:uncertain"),
        (PRESENT, 1000.0, 1100.0, 0.9, "present"),  # present is never gated
        (NOT_OBSERVABLE, 1100.0, 1250.0, "", "unassessed"),
        (ABSENT, 1250.0, 1300.0, "", "no_q1:unassessed"),  # clipped to the row
    ]
    # The crash table keeps everything but present.
    crash = source.gated(
        [(PRESENT, 0.0, 300.0, "", ""), (ABSENT, 300.0, 400.0, "", "")],
        [(100.0, 350.0)],
        keep=(ABSENT, NOT_OBSERVABLE, UNCERTAIN),
    )
    assert [r[:3] for r in crash] == [
        (PRESENT, 0.0, 100.0),
        (ABSENT, 100.0, 300.0),
        (ABSENT, 300.0, 400.0),
    ]


def test_merged_joins_touching_rows_of_one_category_and_confidence():
    rows = [
        (UNCERTAIN, 0.0, 100.0, 0.5, "uncertain"),
        (ABSENT, 100.0, 200.0, "", "no_q1:uncertain"),
        (ABSENT, 200.0, 200.0, "", "no_q1:uncertain"),  # empty: dropped
        (ABSENT, 200.0, 300.123456, "", "no_q1:unassessed"),
        (ABSENT, 300.123456, 400.0, 0.7, "absent"),  # another confidence
        (PRESENT, 400.0, 450.5, 0.9, "present"),
    ]
    assert source.merged(7, rows) == [
        [7, UNCERTAIN, 0, 100, 0.5],
        [7, ABSENT, 100, 300.1235, ""],
        [7, ABSENT, 300.1235, 400, 0.7],
        [7, PRESENT, 400, 450.5, 0.9],
    ]


def test_build_crash_drops_blind_cohort_shots(tmp_path, monkeypatch):
    from types import SimpleNamespace

    table = tmp_path / "crash_v1.csv"
    pd.DataFrame(
        [
            {"shot": s, "category": c, "t_start": a, "t_end": b, "confidence": ""}
            for s in (1, 2)
            for c, a, b in ((ABSENT, 0, 100), (PRESENT, 100, 200), (ABSENT, 200, 300))
        ]
    ).to_csv(table, index=False)
    table.with_suffix(".meta.json").write_text("{}")
    monkeypatch.setattr(source.suggestions, "table_path", lambda *args: table)
    monkeypatch.setattr(source, "blind_shots", lambda tables: {2})
    monkeypatch.setattr(source, "set_shots", lambda paths, names: {})
    gated = []

    def gate_spans(shots, paths, workers):
        gated.extend(shots)
        return {s: [(150.0, 300.0, "no_q1", "efit02")] for s in shots}, {
            s: [] for s in shots
        }

    monkeypatch.setattr(source, "gate_spans", gate_spans)
    args = SimpleNamespace(from_version="v1", version="v9", workers=1, set=[])
    built = source.build_crash(args, SimpleNamespace(label_tables=tmp_path))
    assert gated == [1] and {row[0] for row in built["rows"]} == {1}
    assert built["meta"]["blind_shots_dropped"] == 1
    assert list(built["meta"]["per_shot"]) == ["1"]
    assert built["meta"]["per_shot"]["1"]["present_ms"] == 50.0
