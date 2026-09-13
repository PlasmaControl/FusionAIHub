"""Curated label tables: the manifest, the loader, and what a listing means.

Two kinds of test here, and they are guarding different things. Most of
them build a manifest and a CSV inside `tmp_path`, because what is under
test is the loader's contract - ms to seconds, NaN coverage, an absent shot
producing nothing at all - and a fixture is the only way to say "this
table, these rows, this answer". The last few read the REAL committed
tables under `data/labels/`, because those are edited by hand by whoever
sends us the next one, and a shot number that stopped being an integer or
an onset time that stopped being in milliseconds should fail here rather
than three stages downstream.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from labelmaker.config import Paths
from labelmaker.events import databases as db
from labelmaker.events import schema
from labelmaker.events.lexicon import load_lexicon

POINT_TABLE = """\
SHOT,ONSET_TIME,NTOR,MODE_TYPE
158015,2.61300E+03,2,n2rwm
158015,2.61300E+03,2,n2rwm
156785,8.56000E+02,1,rwm
"""


def _manifest(root, entries, *, version=1):
    """Write `tables.yaml` under `root` from a list of dicts."""
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    (root / "tables.yaml").write_text(
        yaml.safe_dump({"version": version, "tables": list(entries)}),
        encoding="utf-8",
    )
    return root


def _entry(**over):
    entry = {
        "stem": "rwm_fixture",
        "dir": "resistive_wall_mode",
        "phenomenon": "rwm",
        "kind": "point",
        "shot_col": "SHOT",
        "t_col": "ONSET_TIME",
        "t_units": "ms",
        "attr_cols": ["NTOR", "MODE_TYPE"],
        "attr_types": {"NTOR": "int", "MODE_TYPE": "str"},
        "provenance": "a fixture",
    }
    entry.update(over)
    return entry


def _csv(root, entry, text=POINT_TABLE):
    path = root / entry["dir"] / f"{entry['stem']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def table(tmp_path):
    """A one-table root: the manifest, the CSV, and the spec they make."""
    root = tmp_path / "labels"
    entry = _entry()
    _manifest(root, [entry])
    _csv(root, entry)
    return root


# ------------------------------------------------------------- the manifest


def test_the_manifest_is_read_and_a_table_names_its_own_source(table):
    (spec,) = db.load_manifest(table)
    assert spec.stem == "rwm_fixture"
    assert spec.phenomenon == "rwm"
    assert spec.kind == "point"
    assert spec.t_units == "ms"
    assert spec.attr_cols == ("NTOR", "MODE_TYPE")
    assert spec.provenance == "a fixture"
    # The source is the join key everything downstream filters on, and it
    # is derived from the stem so that two tables of one phenomenon can
    # never collapse into one source.
    assert spec.source == "database:rwm_fixture"
    assert spec.path(table) == table / "resistive_wall_mode" / "rwm_fixture.csv"


def test_a_duplicate_stem_is_an_error_naming_the_file(tmp_path):
    root = tmp_path / "labels"
    _manifest(root, [_entry(), _entry(dir="elsewhere")])
    with pytest.raises(db.DatabaseError) as exc:
        db.load_manifest(root)
    assert "rwm_fixture" in str(exc.value)
    assert str(root / "tables.yaml") in str(exc.value)


def test_a_phenomenon_the_lexicon_does_not_have_is_an_error(tmp_path):
    # The lexicon is the single vocabulary: a phenomenon id is a join key
    # against labels, events and ideate, so one that loaded silently would
    # be a phenomenon with no evidence anywhere.
    root = tmp_path / "labels"
    _manifest(root, [_entry(phenomenon="resistive_wall_mode")])
    with pytest.raises(db.DatabaseError) as exc:
        db.load_manifest(root)
    assert "resistive_wall_mode" in str(exc.value)
    assert "rwm" in str(exc.value)          # the ids it could have been


def test_every_manifest_phenomenon_is_a_lexicon_id():
    ids = set(load_lexicon().ids)
    for spec in db.load_manifest(Paths().label_tables):
        assert spec.phenomenon in ids


@pytest.mark.parametrize(
    "over, wanted",
    [
        ({"t_units": "minutes"}, "t_units"),
        ({"kind": "region"}, "kind"),
        ({"t_col": None}, "t_col"),
        ({"stem": ""}, "stem"),
        ({"attr_types": {"NTOR": "complex"}}, "attr_types"),
        ({"attr_types": {"NOT_A_COLUMN": "int"}}, "attr_cols"),
        ({"provenance": ""}, "provenance"),
    ],
)
def test_a_malformed_entry_is_an_error_that_names_the_field(tmp_path, over,
                                                            wanted):
    root = tmp_path / "labels"
    entry = _entry(**over)
    entry = {k: v for k, v in entry.items() if v is not None}
    _manifest(root, [entry])
    with pytest.raises(db.DatabaseError) as exc:
        db.load_manifest(root)
    assert wanted in str(exc.value)


def test_an_interval_table_names_two_time_columns(tmp_path):
    root = tmp_path / "labels"
    entry = _entry(stem="qh_windows", phenomenon="qh", kind="interval",
                   t0_col="START", t1_col="END", t_units="s",
                   attr_cols=[], attr_types={})
    del entry["t_col"]
    _manifest(root, [entry])
    _csv(root, entry, "SHOT,START,END\n190000,1.5,2.25\n")
    (spec,) = db.load_manifest(root)
    frame = db.read_table(spec, root)
    assert list(frame["t0_s"]) == [1.5] and list(frame["t1_s"]) == [2.25]
    events, _ = db.events_for_shot(190000, root=root)
    assert [(e.t0_s, e.t1_s) for e in events] == [(1.5, 2.25)]


def test_a_missing_manifest_is_an_error_and_not_an_empty_answer(tmp_path):
    # Silence here would look exactly like "no table names this shot",
    # which is the one thing a curated list must never be confused with.
    with pytest.raises(db.DatabaseError):
        db.load_manifest(tmp_path / "nothing")


# ---------------------------------------------------------------- the table


def test_the_times_are_converted_to_seconds_on_load_and_never_on_disk(table):
    (spec,) = db.load_manifest(table)
    before = (spec.path(table)).read_text(encoding="utf-8")
    frame = db.read_table(spec, table)
    assert list(frame["t0_s"]) == [2.613, 2.613, 0.856]
    assert list(frame["t1_s"]) == [2.613, 2.613, 0.856]
    assert list(frame["shot"]) == [158015, 158015, 156785]
    assert spec.path(table).read_text(encoding="utf-8") == before


def test_a_duplicate_row_inside_one_table_is_kept(table):
    (spec,) = db.load_manifest(table)
    frame = db.read_table(spec, table)
    # Two onsets at the same time is a claim the table makes. De-duplicating
    # it here would be us editing somebody else's list.
    assert len(frame) == 3
    events, _ = db.events_for_shot(158015, root=table)
    assert len(events) == 2


def test_a_missing_column_is_an_error_naming_the_file_and_the_column(tmp_path):
    root = tmp_path / "labels"
    entry = _entry()
    _manifest(root, [entry])
    path = _csv(root, entry, "SHOT,TIME,NTOR,MODE_TYPE\n158015,2613,1,rwm\n")
    (spec,) = db.load_manifest(root)
    with pytest.raises(db.DatabaseError) as exc:
        db.read_table(spec, root)
    assert "ONSET_TIME" in str(exc.value) and str(path) in str(exc.value)


def test_a_missing_csv_is_an_error_naming_the_path(tmp_path):
    root = tmp_path / "labels"
    _manifest(root, [_entry()])
    (spec,) = db.load_manifest(root)
    with pytest.raises(db.DatabaseError) as exc:
        db.read_table(spec, root)
    assert "rwm_fixture.csv" in str(exc.value)


@pytest.mark.parametrize("bad", ["nan", "", "soon"])
def test_a_non_numeric_or_non_finite_time_raises_at_load_not_at_write(tmp_path,
                                                                     bad):
    # At load, so the file is named in the error. An Event with a NaN t0_s
    # would raise in `Event.__post_init__` instead, five frames away from
    # the row that caused it and with no path in the message.
    root = tmp_path / "labels"
    entry = _entry()
    _manifest(root, [entry])
    _csv(root, entry, f"SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,{bad},1,rwm\n")
    (spec,) = db.load_manifest(root)
    with pytest.raises(db.DatabaseError) as exc:
        db.read_table(spec, root)
    assert "ONSET_TIME" in str(exc.value)


def test_a_non_integer_shot_raises_at_load(tmp_path):
    root = tmp_path / "labels"
    entry = _entry()
    _manifest(root, [entry])
    _csv(root, entry, "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015.5,2613,1,rwm\n")
    (spec,) = db.load_manifest(root)
    with pytest.raises(db.DatabaseError) as exc:
        db.read_table(spec, root)
    assert "SHOT" in str(exc.value)


def test_an_interval_that_runs_backwards_raises_at_load(tmp_path):
    root = tmp_path / "labels"
    entry = _entry(stem="backwards", phenomenon="qh", kind="interval",
                   t0_col="START", t1_col="END", t_units="s",
                   attr_cols=[], attr_types={})
    del entry["t_col"]
    _manifest(root, [entry])
    _csv(root, entry, "SHOT,START,END\n190000,2.5,1.5\n")
    (spec,) = db.load_manifest(root)
    with pytest.raises(db.DatabaseError):
        db.read_table(spec, root)


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
    assert math.isnan(record["t_cov0_s"]) and math.isnan(record["t_cov1_s"])
    assert record["reason"] == db.COVERAGE_REASON
    assert "coverage unknown" in record["reason"]
    # The sources file's key is `(source, diag, channel, pass_name)`, so two
    # tables for one phenomenon coexist and a re-run replaces its own rows.
    assert (record["diag"], record["channel"], record["pass_name"]) == ("", -1, "")


def test_two_tables_naming_one_shot_stay_two_sources(tmp_path):
    root = tmp_path / "labels"
    first, second = _entry(), _entry(stem="rwm_other")
    _manifest(root, [first, second])
    _csv(root, first, "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,1000,1,rwm\n")
    _csv(root, second, "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,2000,2,n2rwm\n")
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
    first, second = _entry(), _entry(stem="rwm_other")
    _manifest(root, [first, second])
    _csv(root, first, "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,1000,1,rwm\n")
    _csv(root, second,
         "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,2000,2,n2rwm\n"
         "158015,2500,2,n2rwm\n")
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
        assert set(frame["MODE_TYPE"]) <= {"rwm", "n2rwm"}
        assert set(frame["NTOR"]) <= {1, 2}
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
    _csv(table, {"dir": "resistive_wall_mode", "stem": "rwm_fixture"},
         "SHOT,ONSET_TIME,NTOR,MODE_TYPE\n158015,2613,2,n2rwm\n")
    assert len(db.read_table(spec, table)) == 1
