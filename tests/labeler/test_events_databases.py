"""Common-format tables become events; curated lists never claim coverage."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import yaml

from labeler.config import Paths
from labeler.events import databases as db
from labeler.events import schema
from labeler.events.lexicon import load_lexicon


def _manifest(root, entries, *, version=1):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tables.yaml").write_text(
        yaml.safe_dump({"version": version, "tables": list(entries)}),
        encoding="utf-8",
    )
    return root


def _entry(**over):
    entry = {
        "stem": "rwm_fixture", "dir": "resistive_wall_mode", "phenomenon": "rwm",
        "kind": "point", "shot_col": "SHOT", "t_col": "ONSET_TIME",
        "t_units": "ms", "attr_cols": ["NTOR", "MODE_TYPE"],
        "attr_types": {"NTOR": "int", "MODE_TYPE": "str"}, "provenance": "a fixture",
        "raw_file": "original.csv", "format_stem": "normalized", "converter": "csv",
        "made_at": "2026-09-13T00:00:00Z",
    }
    entry.update(over)
    return entry


def _csv(root, entry, rows=None, **over):
    if rows is None:
        rows = [(158015, 2.613, 2, "n2rwm"), (158015, 2.613, 2, "n2rwm"),
                (156785, 0.856, 1, "rwm")]
    frame = pd.DataFrame([{
        "shot": shot, "t0_s": t, "t1_s": t, "phenomenon": entry["phenomenon"],
        "evidence_kind": "database", "source": "database:" + entry["stem"],
        "confidence": "", "attrs": json.dumps({"NTOR": ntor, "MODE_TYPE": mode,
                                                "table": entry["stem"]}), **over,
    } for shot, t, ntor, mode in rows], columns=db.FORMAT_COLUMNS)
    path = root / entry["dir"] / "format" / f"{entry['format_stem']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def table(tmp_path):
    root = tmp_path / "labels"
    entry = _entry()
    _manifest(root, [entry])
    _csv(root, entry)
    return root


def test_manifest_separates_raw_fields_format_path_and_source(table):
    (spec,) = db.load_manifest(table)
    assert spec.source == "database:rwm_fixture"
    assert spec.path(table) == table / "resistive_wall_mode/format/normalized.csv"
    assert spec.raw_path(table) == table / "resistive_wall_mode/raw/original.csv"
    assert spec.kind == "point" and spec.t_units == "ms"
    assert spec.attr_cols == ("NTOR", "MODE_TYPE")


def test_a_duplicate_stem_is_an_error_naming_the_file(tmp_path):
    _manifest(tmp_path, [_entry(), _entry(dir="elsewhere")])
    with pytest.raises(db.DatabaseError, match="rwm_fixture") as exc:
        db.load_manifest(tmp_path)
    assert "tables.yaml" in str(exc.value)


def test_two_tables_cannot_overwrite_the_same_format_file(tmp_path):
    _manifest(tmp_path, [_entry(), _entry(stem="other")])
    with pytest.raises(db.DatabaseError, match="format_stem"):
        db.load_manifest(tmp_path)


def test_a_phenomenon_the_lexicon_does_not_have_is_an_error(tmp_path):
    _manifest(tmp_path, [_entry(phenomenon="resistive_wall_mode")])
    with pytest.raises(db.DatabaseError, match="resistive_wall_mode"):
        db.load_manifest(tmp_path)


def test_every_manifest_phenomenon_is_a_lexicon_id():
    assert {s.phenomenon for s in db.load_manifest()} <= set(load_lexicon().ids)


@pytest.mark.parametrize("over, wanted", [
    ({"t_units": "minutes"}, "t_units"), ({"kind": "region"}, "kind"),
    ({"t_col": None}, "t_col"), ({"stem": ""}, "stem"),
    ({"attr_types": {"NTOR": "complex"}}, "attr_types"),
    ({"attr_types": {"NOT_A_COLUMN": "int"}}, "attr_cols"),
    ({"provenance": ""}, "provenance"), ({"raw_file": None}, "raw_file"),
    ({"format_stem": None}, "format_stem"), ({"converter": None}, "converter"),
    ({"made_at": None}, "made_at"),
])
def test_a_malformed_entry_is_an_error_that_names_the_field(tmp_path, over, wanted):
    entry = {k: v for k, v in _entry(**over).items() if v is not None}
    _manifest(tmp_path, [entry])
    with pytest.raises(db.DatabaseError, match=wanted):
        db.load_manifest(tmp_path)


def test_loader_uses_stored_intervals_evidence_and_confidence(tmp_path):
    entry = _entry(kind="interval", t0_col="START", t1_col="END")
    _manifest(tmp_path, [entry])
    _csv(tmp_path, entry, [(190000, 1.5, 1, "rwm")], t1_s=2.25,
         evidence_kind="human", confidence=0.6)
    (event,), (record,) = db.events_for_shot(190000, root=tmp_path)
    assert (event.t0_s, event.t1_s) == (1.5, 2.25)
    assert event.evidence_kind == "human" and event.confidence == 0.6
    assert record["n_events"] == 1


def test_loader_reads_format_only_and_never_requires_raw(table):
    (spec,) = db.load_manifest(table)
    before = spec.path(table).read_bytes()
    frame = db.read_table(spec, table)
    assert frame.t0_s.tolist() == [2.613, 2.613, 0.856]
    assert spec.path(table).read_bytes() == before
    raw = spec.raw_path(table)
    raw.parent.mkdir()
    raw.write_text("this is deliberately not a CSV")
    assert len(db.read_table(spec, table)) == 3
    spec.path(table).unlink()
    with pytest.raises(db.DatabaseError, match="format/normalized.csv"):
        db.read_table(spec, table)


def test_a_missing_manifest_is_an_error_and_not_an_empty_answer(tmp_path):
    with pytest.raises(db.DatabaseError):
        db.load_manifest(tmp_path / "nothing")


def test_a_duplicate_row_inside_one_table_is_kept(table):
    events, _ = db.events_for_shot(158015, root=table)
    assert len(events) == 2


@pytest.mark.parametrize("over, field", [
    ({"t0_s": "soon"}, "t0_s"), ({"t1_s": 0}, "t1_s"),
    ({"shot": 158015.5}, "shot"), ({"attrs": "oops"}, "attrs"),
    ({"source": "database:wrong"}, "source"), ({"phenomenon": "elm"}, "phenomenon"),
])
def test_loader_validates_the_format_file(table, over, field):
    path = _csv(table, _entry(), **over)
    (spec,) = db.load_manifest(table)
    with pytest.raises(db.DatabaseError, match=field) as exc:
        db.read_table(spec, table)
    assert str(path) in str(exc.value)


def test_shots_is_the_membership_set(table):
    (spec,) = db.load_manifest(table)
    assert db.shots(spec, table) == frozenset({158015, 156785})


# --------------------------------------------------------------- the events


def test_a_row_becomes_a_point_event_with_the_table_as_its_source(table):
    events, _ = db.events_for_shot(156785, root=table)
    (event,) = events
    assert event.shot == 156785
    assert event.source == "database:rwm_fixture"
    assert event.evidence_kind == "database"
    assert event.phenomenon == "rwm"          # the manifest's, not the dir's
    assert event.t0_s == event.t1_s == 0.856  # a point event
    assert event.diag == "" and event.channel == -1 and event.pass_name == ""


def test_the_coverage_is_nan_because_a_listing_is_not_a_coverage_claim(table):
    # The load-bearing rule. A database that names a shot says nothing about
    # which interval of that shot anybody examined, so a shot ABSENT from
    # the table is not a negative - and the NaN is what stops a downstream
    # consumer from reading it as one.
    events, _ = db.events_for_shot(156785, root=table)
    (event,) = events
    assert math.isnan(event.t_cov0_s) and math.isnan(event.t_cov1_s)
    # A curated list has no calibrated probability either; 1.0 would let a
    # ranker treat a human list as a perfectly confident detector.
    assert math.isnan(event.confidence)


def test_an_event_with_nan_coverage_is_valid_and_reaches_the_file(table,
                                                                  tmp_path):
    events, _ = db.events_for_shot(156785, root=table)
    out = schema.write_events(tmp_path / "e.parquet", 156785, events,
                              run_id="test")
    assert len(out) == 1
    assert out["t_cov0_s"].isna().all() and out["t_cov1_s"].isna().all()
    assert out["confidence"].isna().all()
    assert out["evidence_kind"].tolist() == ["database"]


def test_the_extra_columns_are_carried_verbatim_plus_the_table(table):
    events, _ = db.events_for_shot(156785, root=table)
    (event,) = events
    assert event.attrs == {"NTOR": 1, "MODE_TYPE": "rwm",
                           "table": "rwm_fixture"}
    assert isinstance(event.attrs["NTOR"], int)
    assert isinstance(event.attrs["MODE_TYPE"], str)
    # `table` is there so a row is traceable without parsing `source`.
    assert json.loads(schema._attrs_json(event.attrs))["NTOR"] == 1


def test_a_shot_no_table_names_gets_no_event_and_no_source_record(table):
    events, records = db.events_for_shot(199999, root=table)
    assert events == [] and records == []


def test_a_table_that_names_the_shot_records_one_ok_source_with_nan_coverage(
    table,
):
    _, records = db.events_for_shot(158015, root=table)
    (record,) = records
    assert record["source"] == "database:rwm_fixture"
    assert record["status"] == "ok"
    assert record["n_events"] == 2
    # The NaN coverage IS the signal, and it is the only one: an `ok` row
    # carries no reason (`schema._source_row` refuses one), so the sentence
    # lives in `db.COVERAGE_REASON` for the docs and never on the row.
    assert math.isnan(record["t_cov0_s"]) and math.isnan(record["t_cov1_s"])
    assert record["reason"] == ""
    assert "coverage unknown" in db.COVERAGE_REASON
    # The sources file's key is `(source, diag, channel, pass_name)`, so two
    # tables for one phenomenon coexist and a re-run replaces its own rows.
    assert (record["diag"], record["channel"], record["pass_name"]) == ("", -1, "")


def test_two_tables_naming_one_shot_stay_two_sources(tmp_path):
    root = tmp_path / "labels"
    first, second = _entry(), _entry(stem="rwm_other", format_stem="other")
    _manifest(root, [first, second])
    _csv(root, first, [(158015, 1.0, 1, "rwm")])
    _csv(root, second, [(158015, 2.0, 2, "n2rwm")])
    events, records = db.events_for_shot(158015, root=root)
    assert sorted(e.source for e in events) == [
        "database:rwm_fixture", "database:rwm_other",
    ]
    assert sorted(r["source"] for r in records) == [
        "database:rwm_fixture", "database:rwm_other",
    ]
    assert {e.attrs["table"] for e in events} == {"rwm_fixture", "rwm_other"}


def test_the_event_ids_keep_the_colon_and_number_per_source_in_time_order(
    table, tmp_path,
):
    # Decided once (proposal open question 8) and pinned here: the source
    # spelling is `database:<stem>`, colon and all, and the id is the
    # schema's own `{shot}-{source}-{n:05d}`. Changing either is a migration
    # of every events file ever written, so it fails here first.
    events, _ = db.events_for_shot(158015, root=table)
    out = schema.write_events(tmp_path / "e.parquet", 158015, events,
                              run_id="test")
    assert out["event_id"].tolist() == [
        "158015-database:rwm_fixture-00000",
        "158015-database:rwm_fixture-00001",
    ]


def test_the_index_gets_one_row_per_table_per_shot(tmp_path):
    root = tmp_path / "labels"
    first, second = _entry(), _entry(stem="rwm_other", format_stem="other")
    _manifest(root, [first, second])
    _csv(root, first, [(158015, 1.0, 1, "rwm")])
    _csv(root, second, [(158015, 2.0, 2, "n2rwm"),
                        (158015, 2.5, 2, "n2rwm")])
    events, _ = db.events_for_shot(158015, root=root)
    path = tmp_path / "e.parquet"
    schema.write_events(path, 158015, events, run_id="test")
    rows = schema.index_rows(path)
    assert [(r["source"], r["phenomenon"], r["n_events"]) for r in rows] == [
        ("database:rwm_fixture", "rwm", 1),
        ("database:rwm_other", "rwm", 2),
    ]
    assert {r["evidence_kind"] for r in rows} == {"database"}


def test_the_per_table_sources_are_offered_as_documentation(table):
    # `schema.KNOWN_SOURCES` has the bare "database"; the manifest is what
    # says which tables exist, and it can change without a code edit.
    assert db.known_sources(table) == ("database:rwm_fixture",)
    assert "database" in schema.KNOWN_SOURCES


# ------------------------------------------------- the committed RWM tables


def test_the_committed_tables_are_the_two_rwm_onset_databases():
    specs = db.load_manifest(Paths().label_tables)
    assert [s.stem for s in specs] == ["rwm_onsets_2017", "rwm_onsets_2024"]
    assert {s.phenomenon for s in specs} == {"rwm"}
    assert all(s.kind == "point" and s.t_units == "ms" for s in specs)
    assert all("Hansen" in s.provenance for s in specs)


def test_the_committed_tables_parse_and_every_time_is_a_plausible_shot_time():
    # A data-integrity test, so a bad hand edit of a CSV fails here. 20 s is
    # generous: a DIII-D shot is ~5 s, and an onset time that arrived in
    # seconds instead of milliseconds would land at 2.6e-3 s, which the
    # lower bound catches.
    root = Paths().label_tables
    total, named = 0, set()
    for spec in db.load_manifest(root):
        frame = db.read_table(spec, root)
        total += len(frame)
        named |= set(frame["shot"].tolist())
        assert frame["shot"].dtype.kind == "i"
        assert (frame["shot"] > 100_000).all()
        assert frame["t0_s"].between(0.1, 20.0).all()
        assert (frame["t1_s"] == frame["t0_s"]).all()
        attrs = frame["attrs"].map(json.loads)
        assert {a["MODE_TYPE"] for a in attrs} <= {"rwm", "n2rwm"}
        assert {a["NTOR"] for a in attrs} <= {1, 2}
        assert spec.path(root).parent.name == "format"
    assert (total, len(named)) == (56, 33)


def test_the_rwm_tables_name_no_shot_the_corpus_holds():
    # Recorded, not fixed: the tables span 156785-176092 and the corpus
    # starts at 185601, so a `--databases-only` run over any corpus shot
    # list writes nothing. That is the honest answer, and the reason the
    # loader had to land before a table that overlaps arrives.
    root = Paths().label_tables
    named: set[int] = set()
    for spec in db.load_manifest(root):
        named |= db.shots(spec, root)
    assert max(named) < 185601


def test_reading_the_same_table_twice_does_not_reread_the_file(table,
                                                               monkeypatch):
    # A `--databases-only` run over 16,909 shots asks each table for each
    # shot; re-parsing the CSV 16,909 times is the difference between
    # seconds and minutes.
    (spec,) = db.load_manifest(table)
    db.read_table(spec, table)
    calls = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    db.read_table(spec, table)
    db.events_for_shot(158015, root=table)
    assert calls == []


def test_a_changed_file_is_reread(table):
    (spec,) = db.load_manifest(table)
    assert len(db.read_table(spec, table)) == 3
    _csv(table, _entry(), [(158015, 2.613, 2, "n2rwm")])
    assert len(db.read_table(spec, table)) == 1
