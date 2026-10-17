"""Silver shots share ordinary event saves and one provenance-bearing frame grid."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths, sha256_of
from labeler.events.review import labels, silver
from labeler.events.ui import app as ui

from .test_review_browser import (
    DRIVER,
    NODE,
    SHELLS,
    TOKEN,
    _store,
)
from .test_review_browser import (
    served as served_browser,  # noqa: F401
)
from .test_review_page import _node, needs_node

SHOT = 170815
OFF_ROSTER = 170816
GOLD_OFF_ROSTER = 171234


def table(path, spans, shot=SHOT):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "shot,category,t_start,t_end,confidence\n"
        + "".join(f"{shot},{c},{a},{b},\n" for a, b, c in spans)
    )
    return path


@pytest.fixture
def paths(tmp_path):
    tables = tmp_path / "events"
    tables.mkdir()
    (tables / "events.yaml").write_text(
        "taxonomy:\n"
        "  mhd_n1:\n"
        "    display: MHD n=1\n"
        "    subcategories:\n"
        "      m1: {display: '1/1'}\n"
    )
    for event in ("confinement", "mhd_n1", "disruption", "high_confinement_mode"):
        directory = tables / event
        directory.mkdir()
        (directory / "shots.csv").write_text(
            "shot,tier,holdout,reviewers,verified_on,notes\n"
            f"{SHOT},unverified,false,,,\n"
        )
    listing = tables / "silver" / "silver_shots.csv"
    listing.parent.mkdir()
    listing.write_text(
        "shot,batch,order,pool,classes,sources,notes\n"
        f"{SHOT},1,1,corpus,mhd_n1:present;sawtooth:present,format,\n"
        f"{OFF_ROSTER},2,1,ae_audited,confinement:h,extend,\n"
    )
    gold = tables / "gold" / "gold_shots.csv"
    gold.parent.mkdir()
    gold.write_text(
        "shot,run_day,year,classes,sources,notes\n"
        f"{SHOT},20260101,2026,mhd_n1:saved,saved,\n"
        f"{GOLD_OFF_ROSTER},20260101,2026,,,\n"
    )
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables, config_dir=tables, raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def client(paths, monkeypatch):
    monkeypatch.setattr(ui.Builds, "status", lambda *args: (Path("built"), None))
    monkeypatch.setattr(ui.rows, "meta", lambda *args: {
        "t_range": [0, 100], "rows": [], "grid": {},
    })
    transport = TestClient(ui.create_app(paths=paths, token="secret"))
    transport.cookies.set(ui.COOKIE, "secret")
    return transport


def test_silver_list_is_validated_and_cached_on_file_version(paths):
    listing = paths.label_tables / "silver" / "silver_shots.csv"
    found = silver.read_silver(paths.label_tables)
    assert silver.read_silver(paths.label_tables) is found
    assert silver.is_silver(paths.label_tables, OFF_ROSTER)
    assert not silver.is_silver(paths.label_tables, 1)
    original = listing.read_text()
    listing.write_text(original.replace(str(OFF_ROSTER), str(SHOT)))
    with pytest.raises(ValueError, match="duplicate"):
        silver.read_silver(paths.label_tables)
    listing.write_text(original.replace(",2,1,", ",6,1,"))
    with pytest.raises(ValueError, match="batch"):
        silver.read_silver(paths.label_tables)
    listing.write_text(original.replace(",2,1,", ",2.5,1,"))
    with pytest.raises(ValueError, match="integer"):
        silver.read_silver(paths.label_tables)


def test_layers_are_ordered_snapped_and_take_latest_suggestion(paths, caplog):
    directory = paths.label_tables / "disruption"
    table(directory / "review" / "labels.csv", [(0, 31, 0), (31, 34, 1)])
    source = table(directory / "extend_rule" / "source.csv", [(0, 70, 1)])
    labels.write_pointer(directory, source, method="rule", version="v1")
    table(directory / "format" / "disruption_format_v1.csv", [(0, 100, 2)])
    table(directory / "extend_rule" / "nested" / "other.csv", [(0, 100, 1)])
    table(directory / "format" / "bad.csv", [(0, 100, 1)]).write_text("bad\n1\n")
    for version in ("v2", "v10"):
        table(
            paths.root / "suggestions" / "rule" / version
            / f"disruption_suggest_rule_{version}.csv", [(0, 100, 1)],
        )
    found = silver.layers(paths, "disruption", SHOT)
    # Curated before machine; a pointer at a curated table keeps that table's rank.
    assert [layer.kind for layer in found] == [
        "saved", "format", "extend", "extend", "suggest",
    ]
    assert found[0].label.window == (0, 40)
    assert found[0].label.intervals == ((30, 40, 1),)
    assert "v10" in found[-1].name and "v2" not in found[-1].name
    assert "bad.csv" in caplog.text
    labels.write_pointer(
        directory,
        paths.root / "suggestions/rule/v10/disruption_suggest_rule_v10.csv",
        method="rule", version="v10",
    )
    assert [layer.kind for layer in silver.layers(paths, "disruption", SHOT)] == [
        "saved", "format", "extend", "extend", "source",
    ]


def test_sawtooth_physics_crowd_schema_reads_and_other_shots_skip(paths):
    # extend_saw_physics chunks carry a boolean `crowd` column, not attrs.
    chunk = paths.label_tables / "sawtooth_oscillation/extend_saw_physics/cohort-000.csv"
    chunk.parent.mkdir(parents=True)
    chunk.write_text(
        "shot,category,t_start,t_end,crowd,confidence,attrs\n"
        f"{SHOT},3,0.0,20.0,True,1.0,\"{{\"\"state\"\": \"\"unassessed\"\"}}\"\n"
        f"{SHOT},1,20.0,61.0,True,1.0,{{}}\n"
    )
    other = chunk.with_name("cohort-001.csv")
    other.write_text(chunk.read_text().replace(str(SHOT), "1"))
    found = [layer for layer in silver.layers(paths, "mhd_n1", SHOT)
             if layer.kind == "extend"]
    assert [layer.name for layer in found] == [
        "extend:sawtooth_oscillation/extend_saw_physics/cohort-000"
    ]
    assert found[0].label.intervals == ((20, 60, 1),)
    assert found[0].label.iscrowd == (1,)


def test_folding_maps_categories_and_keeps_not_observable_unassessed(paths):
    table(paths.label_tables / "high_confinement_mode/format/h.csv",
          [(0, 20, 1), (20, 40, 3), (40, 60, 2)])
    found = silver.layers(paths, "confinement", SHOT)
    assert found[0].name == "format:high_confinement_mode/h"
    assert found[0].label.intervals == ((0, 20, 1), (40, 60, 5))
    assert found[0].unknown == ((20, 40),)


def test_folded_saves_rank_after_saves_and_before_curated_tables(paths):
    table(paths.label_tables / "sawtooth_oscillation/review/labels.csv", [(0, 50, 1)])
    table(paths.label_tables / "mhd_n1/format/f.csv", [(0, 100, 2)])
    found = silver.layers(paths, "mhd_n1", SHOT)
    assert [(layer.kind, layer.name) for layer in found] == [
        ("folded", "folded:sawtooth_oscillation/review/labels"),
        ("format", "format:mhd_n1/f"),
    ]
    assert silver.layer_file(paths, found[0]) == (
        paths.label_tables / "sawtooth_oscillation/review/labels.csv"
    )


def test_silver_list_refuses_blind_cohort_shots(paths, monkeypatch):
    import pandas as pd

    cohort = paths.label_tables / "catalog" / "cohort.csv"
    cohort.parent.mkdir()
    cohort.write_text("frozen\n")
    monkeypatch.setattr(silver, "read_cohort", lambda path: pd.DataFrame(
        {"shot": [OFF_ROSTER, 5], "blind": [True, False]}
    ))
    with pytest.raises(ValueError, match="blind"):
        silver.read_silver(paths.label_tables)


def test_blind_shots_fail_closed_without_the_cohort(paths, monkeypatch):
    import pandas as pd

    assert not (paths.label_tables / "catalog" / "cohort.csv").exists()
    with pytest.raises(FileNotFoundError, match="cohort.csv is missing"):
        silver.blind_shots(paths.label_tables)
    # Only re-validating a built list may go on without it.
    assert silver.blind_shots(paths.label_tables, missing_ok=True) == set()
    assert SHOT in set(silver.read_silver(paths.label_tables).shot)
    # The writers' guards refuse too.
    monkeypatch.setattr(Paths, "from_env", lambda: paths)
    with pytest.raises(FileNotFoundError, match="cohort.csv"):
        silver.main(["export", "--out", str(paths.root / "grid")])

    cohort = paths.label_tables / "catalog" / "cohort.csv"
    cohort.parent.mkdir()
    cohort.write_text("frozen\n")
    monkeypatch.setattr(silver, "read_cohort", lambda path: pd.DataFrame(
        {"shot": [OFF_ROSTER, 5], "blind": [True, False]}
    ))
    assert silver.blind_shots(paths.label_tables) == {OFF_ROSTER}


def test_children_are_locked_then_clipped_and_folded(paths):
    table(paths.label_tables / "sawtooth_oscillation/format/s.csv", [(0, 100, 1)])
    child = paths.label_tables / "mhd_n1/subcategories/m1"
    table(child / "review/labels.csv", [(0, 100, 2)])
    assert silver.layers(paths, "mhd_n1.m1", SHOT) == []
    table(paths.label_tables / "mhd_n1/review/labels.csv",
          [(0, 20, 0), (20, 31, 1), (31, 100, 0)])
    found = silver.layers(paths, "mhd_n1.m1", SHOT)
    assert [layer.kind for layer in found] == ["saved", "format"]
    assert found[0].label.intervals == ((20, 30, 2),)
    assert found[1].label.intervals == ((20, 30, 1),)
    table(paths.label_tables / "mhd_n1/review/labels.csv", [(0, 100, 0)])
    assert silver.layers(paths, "mhd_n1.m1", SHOT) == []


def test_progress_counts_parents_and_only_eligible_children(paths):
    assert silver.review_events(paths) == [
        "confinement", "disruption", "mhd_n1", "mhd_n1.m1",
    ]
    manifest = paths.events_manifest
    taxonomy_text = manifest.read_text()
    manifest.write_text(taxonomy_text + "  disruption: {review: false}\n")
    assert silver.review_events(paths) == ["confinement", "mhd_n1", "mhd_n1.m1"]
    manifest.write_text(taxonomy_text)
    assert silver.progress(paths, SHOT) == {"saved": [], "total": 3}
    table(paths.label_tables / "mhd_n1/review/labels.csv", [(0, 100, 1)])
    assert silver.progress(paths, SHOT) == {"saved": ["mhd_n1"], "total": 4}
    table(paths.label_tables / "mhd_n1/subcategories/m1/review/labels.csv",
          [(0, 100, 0)])
    assert silver.progress(paths, SHOT) == {
        "saved": ["mhd_n1", "mhd_n1.m1"], "total": 4,
    }


def rows_table(path, saved, header="shot,category,t_start,t_end,confidence", tail=""):
    """A saved table of several shots: `saved` maps a shot to its (a, b, category)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "".join(
        f"{shot},{c},{a},{b},{tail}\n"
        for shot, spans in saved.items() for a, b, c in spans
    ))


def test_progress_many_equals_progress_shot_for_shot(paths):
    locked, saved, unsaved, open_, only, orphan = (
        170001, 170002, 170003, 170004, 170005, 170006)
    root = paths.label_tables
    # A parent saved all absent opens nothing; saved with a span, it opens its children.
    rows_table(root / "mhd_n1/review/labels.csv", {
        locked: [(0, 100, 0)], saved: [(0, 20, 0), (20, 31, 1), (31, 100, 0)],
        open_: [(0, 100, 1)],
    })
    rows_table(root / "mhd_n1/subcategories/m1/review/labels.csv", {
        locked: [(0, 100, 2)], saved: [(20, 30, 2)], orphan: [(0, 100, 2)],
    })
    rows_table(root / "confinement/review/labels.csv", {only: [(0, 100, 1)]})
    shots = [locked, saved, unsaved, open_, only, orphan, 999]  # 999 is on no table
    found = silver.progress_many(paths, shots)
    assert list(found) == shots
    assert found == {shot: silver.progress(paths, shot) for shot in shots}
    # Not vacuous: the three cases the lock rule tells apart.
    assert found[saved] == {"saved": ["mhd_n1", "mhd_n1.m1"], "total": 4}
    assert found[open_] == {"saved": ["mhd_n1"], "total": 4}
    assert found[locked] == {"saved": ["mhd_n1"], "total": 3}  # child locked, not counted
    assert found[orphan] == found[unsaved] == {"saved": [], "total": 3}
    assert found[only] == {"saved": ["confinement"], "total": 3}
    assert silver.progress_many(paths, []) == {}


def test_progress_many_reads_tables_as_progress_does(paths):
    root = paths.label_tables
    # A crowd-schema table, where a sub-millisecond row saves nothing, and a table
    # that cannot be read: each is what `progress` makes of it, shot by shot.
    rows_table(root / "disruption/review/labels.csv",
               {1: [(0, 100, 1)], 2: [(5, 5, 1)]},
               header="shot,category,t_start,t_end,crowd", tail="true")
    (root / "confinement/review").mkdir(parents=True)
    (root / "confinement/review/labels.csv").write_text("shot,category\nx,y\n")
    found = silver.progress_many(paths, [1, 2, 3])
    assert found == {shot: silver.progress(paths, shot) for shot in (1, 2, 3)}
    assert found[1]["saved"] == ["disruption"] and found[2]["saved"] == []


def test_export_distinguishes_unassessed_absent_and_falls_through(paths, tmp_path):
    directory = paths.label_tables / "disruption"
    table(directory / "review/labels.csv", [(0, 30, 0)])
    table(directory / "format/disruption_format_v1.csv",
          [(0, 20, 1), (20, 50, 2)])
    table(paths.label_tables / "confinement/format/c.csv", [(20, 60, 1)])
    out = tmp_path / "export"
    silver.export(paths, [SHOT], out)
    with np.load(out / f"{SHOT}.npz", allow_pickle=False) as arrays:
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)
        assert arrays["labels"].dtype == arrays["origin"].dtype == np.int8
        assert arrays["grid_ms"] == 10 and arrays["t0_ms"] == 0
        events = list(arrays["events"])
        row = arrays["labels"][events.index("disruption")]
        assert row.tolist() == [0, 0, 0, 2, 2, -1]
        origins = arrays["origins"]
        provenance = arrays["origin"][events.index("disruption")]
        assert list(origins[provenance[:5]]) == [
            "saved:disruption/review/labels", "saved:disruption/review/labels",
            "saved:disruption/review/labels", "format:disruption/disruption_format_v1",
            "format:disruption/disruption_format_v1",
        ]
        assert provenance[5] == -1
        assert arrays["labels"][events.index("confinement")].tolist() == [-1, -1, 1, 1, 1, 1]
    index = json.loads((out / "index.json").read_text())
    assert index["shots"] == [SHOT] and index["grid_ms"] == 10
    assert index["events"] == events
    assert index["categories"]["confinement"]["1"] == "high"
    assert index["origins"] and index["made_at"] and index["grid_rule"]


def test_export_leaves_not_observable_unassessed_so_lower_layers_speak(paths, tmp_path):
    table(paths.label_tables / "high_confinement_mode/format/h.csv",
          [(0, 20, 1), (20, 40, 3)])
    table(paths.label_tables / "confinement/extend_x/e.csv", [(0, 60, 2)])
    silver.export(paths, [SHOT], tmp_path / "export")
    with np.load(tmp_path / "export" / f"{SHOT}.npz", allow_pickle=False) as arrays:
        row = list(arrays["events"]).index("confinement")
        assert arrays["labels"][row].tolist() == [1, 1, 2, 2, 2, 2]
        names = arrays["origins"][arrays["origin"][row]]
        assert [name.split(":")[0] for name in names] == ["format"] * 2 + ["extend"] * 4
    index = json.loads((tmp_path / "export" / "index.json").read_text())
    assert "regime" in index["zero"]["confinement"]
    assert index["zero"]["mhd_n1"].startswith("absent")


def test_export_child_is_unassessed_outside_parent_region(paths, tmp_path):
    table(paths.label_tables / "mhd_n1/review/labels.csv",
          [(0, 20, 0), (20, 40, 1), (40, 60, 0)])
    table(paths.label_tables / "mhd_n1/subcategories/m1/format/c.csv", [(0, 60, 0)])
    silver.export(paths, [SHOT], tmp_path / "export")
    with np.load(tmp_path / "export" / f"{SHOT}.npz", allow_pickle=False) as arrays:
        row = list(arrays["events"]).index("mhd_n1.m1")
        assert arrays["labels"][row].tolist() == [-1, -1, 0, 0, -1, -1]


def _exported(paths, tmp_path, event):
    silver.export(paths, [SHOT], tmp_path / "export")
    with np.load(tmp_path / "export" / f"{SHOT}.npz", allow_pickle=False) as arrays:
        return arrays["labels"][list(arrays["events"]).index(event)].tolist()


def test_a_folded_tables_zero_is_not_the_events_absent(paths, tmp_path):
    table(paths.label_tables / "high_confinement_mode/format/h.csv",
          [(0, 20, 1), (20, 60, 0)])
    table(paths.label_tables / "low_confinement_mode/format/l.csv", [(0, 60, 1)])
    assert _exported(paths, tmp_path, "confinement") == [1, 1, 2, 2, 2, 2]
    table(paths.label_tables / "neoclassical_tearing_mode/format/t.csv",
          [(0, 20, 0), (20, 40, 1)])
    assert _exported(paths, tmp_path, "mhd_n1") == [-1, -1, 1, 1, -1, -1]


def test_a_regime_tables_zero_falls_through_but_the_owners_does_not(paths, tmp_path):
    table(paths.label_tables / "confinement/format/f.csv", [(0, 30, 1), (30, 60, 0)])
    table(paths.label_tables / "confinement/extend_x/e.csv", [(0, 60, 2)])
    assert _exported(paths, tmp_path, "confinement") == [1, 1, 1, 2, 2, 2]
    table(paths.label_tables / "confinement/review/labels.csv", [(0, 60, 0)])
    assert _exported(paths, tmp_path, "confinement") == [0] * 6


def _with_defaults(paths):
    """The shipped defaults: confinement reads unmarked time as low (2), detachment as
    attached (1). The fixture has no detachment roster yet."""
    directory = paths.label_tables / "detachment"
    directory.mkdir(exist_ok=True)
    (directory / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n"
        f"{SHOT},unverified,false,,,\n"
    )
    paths.events_manifest.write_text(
        paths.events_manifest.read_text()
        + "  confinement: {default: 2}\n  detachment: {default: 1}\n"
    )


def test_the_owners_unmarked_time_is_the_events_default_class(paths, tmp_path):
    save = paths.label_tables / "confinement/review/labels.csv"
    table(save, [(0, 20, 1), (20, 60, 0)])
    assert _exported(paths, tmp_path, "confinement") == [1, 1, 0, 0, 0, 0]  # no default
    _with_defaults(paths)
    assert _exported(paths, tmp_path, "confinement") == [1, 1, 2, 2, 2, 2]
    table(paths.label_tables / "detachment/review/labels.csv",
          [(0, 20, 2), (20, 60, 0)])
    assert _exported(paths, tmp_path, "detachment") == [2, 2, 1, 1, 1, 1]
    # An all-gap save, as Butt's WPQH-only saves were, now says L-mode throughout...
    table(save, [(0, 60, 0)])
    assert _exported(paths, tmp_path, "confinement") == [2] * 6
    # ...and 5 (uncertain) is its own class, never overwritten by the default.
    table(save, [(0, 20, 5), (20, 40, 4), (40, 60, 0)])
    assert _exported(paths, tmp_path, "confinement") == [5, 5, 4, 4, 2, 2]
    # The stored label is untouched; only the grid reads the 0 as the default.
    assert labels.read_saved(paths.label_tables / "confinement")[SHOT].intervals == (
        (0, 20, 5), (20, 40, 4),
    )


def test_other_tables_zero_and_other_events_keep_their_meaning_beside_a_default(
    paths, tmp_path
):
    _with_defaults(paths)
    table(paths.label_tables / "confinement/format/f.csv", [(0, 30, 1), (30, 60, 0)])
    table(paths.label_tables / "confinement/extend_x/e.csv", [(0, 60, 3)])
    # No save: a regime table's 0 still falls through to the next table (no default).
    assert _exported(paths, tmp_path, "confinement") == [1, 1, 1, 3, 3, 3]
    layers = silver.layers(paths, "confinement", SHOT)
    assert [(layer.kind, layer.fill) for layer in layers] == [
        ("format", 0), ("extend", 0),
    ]
    table(paths.label_tables / "confinement/review/labels.csv", [(0, 20, 4), (20, 60, 0)])
    assert _exported(paths, tmp_path, "confinement") == [4, 4, 2, 2, 2, 2]
    assert [(layer.kind, layer.fill) for layer in silver.layers(
        paths, "confinement", SHOT)] == [("saved", 2), ("format", 0), ("extend", 0)]
    # An event with no default reads its save's 0 as absent, as before.
    table(paths.label_tables / "disruption/review/labels.csv", [(0, 30, 0)])
    assert _exported(paths, tmp_path, "disruption") == [0, 0, 0, -1, -1, -1]
    assert [layer.fill for layer in silver.layers(paths, "disruption", SHOT)] == [0]


def test_the_export_index_names_the_default_and_the_zero_it_changes(paths, tmp_path):
    _with_defaults(paths)
    silver.export(paths, [SHOT], tmp_path / "export")
    index = json.loads((tmp_path / "export" / "index.json").read_text())
    assert index["default"]["confinement"] == 2 and index["default"]["detachment"] == 1
    assert index["default"]["mhd_n1"] == index["default"]["disruption"] == 0
    assert index["zero"]["confinement"].startswith("low (2), the event's default")
    assert index["zero"]["detachment"].startswith("attached (1), the event's default")
    assert "another table's 0 says nothing" in index["zero"]["confinement"]
    assert index["zero"]["disruption"].startswith("absent")
    assert "default" in index["grid_rule"]


def test_the_by_shot_view_sends_each_layers_fill(client, paths):
    _with_defaults(paths)
    table(paths.label_tables / "confinement/review/labels.csv", [(0, 20, 1), (20, 60, 0)])
    table(paths.label_tables / "confinement/format/f.csv", [(0, 60, 3)])
    found = client.get(f"/api/silver/shot?shot={SHOT}").json()
    event = next(row for row in found["events"] if row["event"] == "confinement")
    assert event["default"] == 2
    assert [(layer["kind"], layer["fill"]) for layer in event["layers"]] == [
        ("saved", 2), ("format", 0),
    ]
    assert event["layers"][0]["label"]["intervals"] == [[0, 20, 1]]  # as stored
    assert next(row for row in found["events"]
                if row["event"] == "disruption")["default"] == 0


def test_a_child_table_says_nothing_past_its_own_window(paths, tmp_path):
    table(paths.label_tables / "mhd_n1/review/labels.csv", [(0, 100, 1)])
    table(paths.label_tables / "mhd_n1/subcategories/m1/format/c.csv",
          [(0, 20, 0), (20, 40, 1)])
    assert _exported(paths, tmp_path, "mhd_n1.m1") == [0, 0, 1, 1] + [-1] * 6


def test_the_page_draft_is_filled_as_the_strip_fills_it(paths):
    table(paths.label_tables / "high_confinement_mode/format/h.csv",
          [(0, 20, 1), (20, 60, 0)])
    label, first = silver.draft(paths, "confinement", SHOT)
    assert label == first.label and first.name == "format:high_confinement_mode/h"
    table(paths.label_tables / "low_confinement_mode/format/l.csv", [(0, 60, 1)])
    label, first = silver.draft(paths, "confinement", SHOT)
    assert first.name == "format:high_confinement_mode/h"
    assert label == labels.Label((0, 60), ((0, 20, 1), (20, 60, 2)))


def test_the_owners_not_observable_is_final_but_a_folded_saves_is_not(paths, tmp_path):
    directory = paths.label_tables / "disruption"
    table(directory / "review/labels.csv", [(0, 20, 1), (20, 40, 3)])
    table(directory / "format/f.csv", [(0, 60, 1)])
    silver.export(paths, [SHOT], tmp_path / "export")
    with np.load(tmp_path / "export" / f"{SHOT}.npz", allow_pickle=False) as arrays:
        row = list(arrays["events"]).index("disruption")
        assert arrays["labels"][row].tolist() == [1, 1, -1, -1, 1, 1]
        names = arrays["origins"][arrays["origin"][row]]
        assert [name.split(":")[0] for name in names] == ["saved"] * 4 + ["format"] * 2
    table(paths.label_tables / "sawtooth_oscillation/review/labels.csv",
          [(0, 20, 1), (20, 40, 3)])
    table(paths.label_tables / "mhd_n1/format/m.csv", [(0, 60, 2)])
    assert _exported(paths, tmp_path, "mhd_n1") == [1, 1, 2, 2, 2, 2]


def test_silver_api_shapes_locked_children_and_token_gate(client, paths):
    found = client.get("/api/silver").json()
    assert found["grid_ms"] == 10
    assert [batch["batch"] for batch in found["batches"]] == [1, 2, 3, 4, 5]
    shot = found["batches"][0]["shots"][0]
    assert shot == {
        "shot": SHOT, "order": 1, "pool": "corpus",
        "classes": ["mhd_n1:present", "sawtooth:present"],
        "saved": 0, "total": 3, "done": False,
    }
    overview = client.get(f"/api/silver/shot?shot={SHOT}").json()
    assert overview["shot"] == SHOT and overview["batch"] == overview["order"] == 1
    assert overview["grid_ms"] == 10 and len(overview["t_range"]) == 2
    child = next(row for row in overview["events"] if row["parent"])
    assert child["event"] == "mhd_n1.m1" and child["locked"]
    assert child["layers"] == [] and not child["saved"]
    table(paths.label_tables / "mhd_n1/review/labels.csv", [(0, 100, 1)])
    overview = client.get(f"/api/silver/shot?shot={SHOT}").json()
    parent = next(row for row in overview["events"] if row["event"] == "mhd_n1")
    assert parent["saved"] and parent["layers"][0]["kind"] == "saved"
    assert set(parent["layers"][0]) == {
        "kind", "name", "label", "fill", "unknown", "held",
    }
    assert parent["default"] == parent["layers"][0]["fill"] == 0
    assert overview["t_range"] == [0, 100]
    assert client.get("/api/silver/shot?shot=99").status_code == 404
    assert client.get("/api/gold/shots").json() == {"shots": [SHOT, GOLD_OFF_ROSTER]}
    client.cookies.clear()
    assert client.get("/api/silver").status_code == 401
    assert client.get("/api/gold/shots").status_code == 401
    assert client.get(f"/api/silver/shot?shot={SHOT}").status_code == 401


def test_missing_silver_list_reports_it_without_breaking_events(client, paths):
    (paths.label_tables / "silver/silver_shots.csv").unlink()
    assert "missing" in client.get("/api/silver").json()["error"].lower()
    assert client.get("/api/events").status_code == 200
    assert client.get(f"/api/shot?event=disruption&shot={SHOT}").status_code == 200


def test_off_roster_silver_opens_saves_reloads_and_records_fallback(client, paths):
    source = table(paths.label_tables / "disruption/format/elsewhere.csv",
                   [(3, 31, 0), (31, 34, 1)], shot=OFF_ROSTER)
    query = f"event=disruption&shot={OFF_ROSTER}"
    opened = client.get(f"/api/shot?{query}")
    assert opened.status_code == 200, opened.text
    found = opened.json()
    assert found["tier"] == "unverified" and found["silver"] is True
    assert found["source"] == {"window": [0, 40], "intervals": [[30, 40, 1]]}
    assert found["source_origin"] == "format:disruption/elsewhere"
    saved = client.post("/api/label", json={
        "event": "disruption", "shot": OFF_ROSTER,
        "window": [0, 40], "intervals": [[31, 34, 1]],
    })
    assert saved.status_code == 200, saved.text
    assert saved.json()["saved"] == found["source"]
    assert saved.json()["last_save"]["source"] == found["source_origin"]
    assert saved.json()["last_save"]["source_sha256"] == sha256_of(source)
    assert client.get(f"/api/shot?{query}").json()["saved"] == found["source"]
    assert client.get(f"/api/history?{query}").status_code == 200
    for route in ("shot", "rows", "history", "frame", "masks", "tokeye"):
        extra = "&t0=0&t1=100&camera=none" if route in {"rows", "frame"} else ""
        response = client.get(f"/api/{route}?event=disruption&shot=999{extra}")
        assert response.status_code == 404 and "roster" in response.json()["error"]
    assert client.post("/api/label", json={
        "event": "disruption", "shot": 999, "window": [0, 100], "intervals": [],
    }).status_code == 404


def test_a_gold_shot_off_an_events_roster_is_an_extra_row_that_opens_and_saves(client, paths):
    queue = client.get("/api/queue?event=disruption").json()
    assert [row["shot"] for row in queue["shots"] if not row.get("extra")] == [SHOT]
    extras = [row for row in queue["shots"] if row.get("extra")]
    assert [(r["shot"], r["tier"], r["state"]) for r in extras] == [
        (GOLD_OFF_ROSTER, "unverified", "unreviewed")
    ]
    query = f"event=disruption&shot={GOLD_OFF_ROSTER}"
    assert client.get(f"/api/shot?{query}").json()["tier"] == "unverified"
    saved = client.post("/api/label", json={
        "event": "disruption", "shot": GOLD_OFF_ROSTER,
        "window": [0, 40], "intervals": [[31, 34, 1]],
    })
    assert saved.status_code == 200, saved.text
    assert saved.json()["row"]["tier"] == "unverified"
    again = client.get("/api/queue?event=disruption").json()["shots"]
    assert next(r for r in again if r["shot"] == GOLD_OFF_ROSTER)["state"] != "unreviewed"
    (paths.label_tables / "gold/gold_shots.csv").unlink()
    assert not any(r.get("extra") for r in client.get("/api/queue?event=disruption").json()["shots"])
    # Off gold it is no longer an extra row, but as an open six-digit shot it still opens.
    assert client.get(f"/api/shot?{query}").status_code == 200


def test_silver_fallback_uses_source_row_before_other_layers(client, paths):
    directory = paths.label_tables / "disruption"
    source = table(directory / "extend_rule/source.csv", [(0, 50, 2)], OFF_ROSTER)
    labels.write_pointer(directory, source, method="rule", version="v1")
    table(directory / "format/other.csv", [(0, 100, 1)], OFF_ROSTER)
    found = client.get(f"/api/shot?event=disruption&shot={OFF_ROSTER}").json()
    assert found["source"]["intervals"] == [[0, 50, 2]]
    assert "source_origin" not in found


def test_export_cli_defaults_to_silver_and_accepts_a_shots_file(
    paths, tmp_path, monkeypatch
):
    import pandas as pd

    monkeypatch.setattr(Paths, "from_env", lambda: paths)
    # The export keeps blind shots out, so it needs the cohort (5 is blind, not listed).
    cohort = paths.label_tables / "catalog" / "cohort.csv"
    cohort.parent.mkdir()
    cohort.write_text("frozen\n")
    monkeypatch.setattr(silver, "read_cohort", lambda path: pd.DataFrame(
        {"shot": [5], "blind": [True]}
    ))
    out = tmp_path / "default"
    silver.main(["export", "--out", str(out)])
    assert sorted(path.stem for path in out.glob("*.npz")) == [str(SHOT), str(OFF_ROSTER)]
    shots = tmp_path / "shots.csv"
    shots.write_text(f"shot\n{OFF_ROSTER}\n")
    out = tmp_path / "selected"
    silver.main(["export", "--out", str(out), "--shots-file", str(shots)])
    assert sorted(path.stem for path in out.glob("*.npz")) == [str(OFF_ROSTER)]


@needs_node
def test_silver_strip_winners_fall_through_and_respect_parent_regions():
    event = {"allowed": [[10, 50]], "layers": [
        {"label": {"window": [0, 30], "intervals": []}},
        {"label": {"window": [0, 50], "intervals": [[0, 50, 1]]}},
    ]}
    found = _node(
        "(() => {const r = m.silverFrames(input, [0,60]); "
        "return [Array.from(r.cells), Array.from(r.origin)];})()", event,
    )
    assert found == [[-1, 0, 0, 1, 1, -1], [-1, 0, 0, 1, 1, -1]]


@needs_node
def test_silver_strip_unknown_stretches_fall_through():
    event = {"layers": [
        {"label": {"window": [0, 30], "intervals": [[0, 30, 1]]}, "unknown": [[10, 20]]},
        {"label": {"window": [0, 30], "intervals": [[0, 30, 2]]}},
    ]}
    found = _node(
        "(() => {const r = m.silverFrames(input, [0,30]); "
        "return [Array.from(r.cells), Array.from(r.origin)];})()", event,
    )
    assert found == [[1, 2, 1], [0, 1, 0]]
    event["layers"][0]["held"] = [[10, 20]]
    found = _node(
        "(() => {const r = m.silverFrames(input, [0,30]); "
        "return [Array.from(r.cells), Array.from(r.origin)];})()", event,
    )
    assert found == [[1, -1, 1], [0, 0, 0]], "the owner's own stays unassessed"


@needs_node
def test_silver_strip_unmarked_time_reads_as_the_layers_fill():
    """A save's unmarked time inside its window is the layer's `fill` (the event's
    default class), as in the export, and drawn lighter; a fill of 0 is as before."""
    event = {"layers": [
        {"label": {"window": [0, 50], "intervals": [[20, 30, 1]]}, "fill": 2},
        {"label": {"window": [0, 60], "intervals": [[50, 60, 3]]}, "fill": 0},
    ]}
    found = _node(
        "(() => {const r = m.silverFrames(input, [0,60]); "
        "return [Array.from(r.cells), Array.from(r.implied), m.frameRuns(r)];})()", event,
    )
    assert found == [
        [2, 2, 1, 2, 2, 3], [1, 1, 0, 1, 1, 0],
        [[0, 2, 2, True], [2, 3, 1, False], [3, 5, 2, True], [5, 6, 3, False]],
    ]


@pytest.mark.skipif(NODE is None or not SHELLS, reason="needs node and headless Chromium")
def test_silver_page_navigation_reload_off_roster_save_and_next_event(
    served_browser, tmp_path, monkeypatch,  # noqa: F811
):
    """The silver set at api 11: a `#silver/` link opens its batch's shot list in By
    shot, an off-roster shot opens, saves and reloads there, Next event walks on to
    the list's next shot, and the gold list stands in for the old "Gold set" box."""
    monkeypatch.setenv("LABELER_NO_FETCH", "1")  # a shot with no store fails its build
    base, event = served_browser
    shot = 170899
    listing = event.parent / "silver/silver_shots.csv"
    listing.parent.mkdir()
    listing.write_text(
        "shot,batch,order,pool,classes,sources,notes\n"
        f"{SHOT},1,1,corpus,alfven_eigenmode:present,format,\n"
        f"{shot},2,1,ae_audited,alfven_eigenmode:present,format,\n"
    )
    table(event / "format/extra.csv", [(0, 1000, 0), (1000, 1500, 1)], shot)
    gold = event.parent / "gold/gold_shots.csv"
    gold.parent.mkdir()
    gold.write_text(
        "shot,run_day,year,classes,sources,notes\n"
        f"{SHOT},20260101,2026,alfven_eigenmode:saved,saved,\n"
        f"{GOLD_OFF_ROSTER},20260101,2026,,,\n"
    )
    paths = Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=event.parent, raw_cache=tmp_path / "raw",
        config_dir=tmp_path / "config",
    )
    _store(paths, shot)
    # The batches as named lists (api 11); batch 2 walks on to 170815 after 170899.
    lists = event.parent / "shotlists"
    lists.mkdir()
    (lists / "silver_batch_1.csv").write_text(
        f"shot,batch,order,pool,classes\n{SHOT},1,1,corpus,alfven_eigenmode:present\n"
    )
    (lists / "silver_batch_2.csv").write_text(
        "shot,batch,order,pool,classes\n"
        f"{shot},2,1,ae_audited,alfven_eigenmode:present\n"
        f"{SHOT},2,2,corpus,alfven_eigenmode:present\n"
    )
    paths.shotlists_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.shotlists_manifest.write_text(
        "lists:\n"
        "  silver_batch_1: {title: \"Silver batch 1\", parent: silver}\n"
        "  silver_batch_2: {title: \"Silver batch 2\", parent: silver}\n"
    )
    script = tmp_path / "silver-browser.mjs"
    # Reuse the existing driver's browser transport, without its scenarios.
    prefix = DRIVER.read_text().split("const opened =", 1)[0]
    script.write_text(prefix + r"""
await send("Runtime.enable");
await send("Page.enable");
await send("Page.addScriptToEvaluateOnNewDocument", {
  source: 'sessionStorage.setItem("labeler:who", "Silver reviewer")',
});
const strips = (shot) => `typeof Lists !== "undefined" && Lists.byShot && Lists.strips?.shot === ${shot}`;
await send("Page.navigate", { url: `${BASE}/?token=${TOKEN}#silver/2/170899` });
await until(strips(170899));
assert.equal(await js('location.hash'), '#list/silver_batch_2/170899');
assert.equal(await js('[...$("event").options].some((o) => o.value === "silver")'), false);
assert.equal(await js('$("event").value'), "__all");
assert.deepEqual(await js('[...$("silver-batches").children].map((b) => b.getAttribute("aria-pressed"))'),
  ["false", "false", "true"]);
assert.equal(await js('$("silver-shots").textContent.includes("AE audited")'), true);
assert.equal(await js('$("silver-shots").textContent.includes("0/1")'), true);
await until('document.querySelector("#silver-strips canvas").width > 0');
await js('$("silver-all-layers").checked = true; $("silver-all-layers").dispatchEvent(new Event("change"))');
assert.equal(await js('document.querySelectorAll(".silver-layer").length > 0'), true);
await send("Page.reload");
await until(strips(170899));
await js('document.querySelector(".silver-strip").click()');
await until('S.shot === 170899 && S.meta && S.opened?.ticket === S.ticket && Lists.context');
assert.equal(await js('S.meta.silver && S.meta.tier === "unverified"'), true);
assert.equal(await js('location.hash'), '#list/silver_batch_2/170899/alfven_eigenmode');
assert.equal(await js('S.meta.source_origin'), 'format:alfven_eigenmode/extra');
assert.deepEqual(await js('S.queue.map((row) => [row.shot, Boolean(row.extra)])'), [[170899, true], [170815, false]]);
await js('edit([3,1501], [[1001,1004,1]]); save(false)');
await until('!S.saving && S.meta.saved?.intervals[0]?.[0] === 1000');
assert.equal(await js('dirty()'), false);
assert.equal(await js('S.meta.saved.intervals[0][1]'), 1010);
assert.equal(await js('$("silver-next-event").hidden'), false);
await js('$("silver-breadcrumb").click()');
await until(strips(170899));
assert.equal(await js('$("silver-batches").children[2].textContent'), 'Silver batch 2 · 1/2 done');
await js('byShotKey("Enter")');
await until('Lists.context && S.shot === 170899 && S.opened?.ticket === S.ticket');
await js('$("silver-next-event").click()');
await until('Lists.context && S.shot === 170815 && S.opened?.ticket === S.ticket');
assert.equal(await js('location.hash'), '#list/silver_batch_2/170815/alfven_eigenmode');
await send("Page.reload");
await until('typeof Lists !== "undefined" && Lists.context && S.shot === 170815 && S.meta');
await js('byShotKey("g")');
await until(strips(170815));
await js('byShotKey("Escape")');
await until('Lists.byShot && Lists.shot === null && Lists.overview?.list === "silver_batch_2"');
await js('byShotKey("k")');
await until(strips(170899));
await js('location.hash = "#alfven_eigenmode/170899"');
await until('S.shot === 170899 && S.meta && !Lists.byShot && !Lists.context');
assert.equal(await js('S.meta.saved.intervals[0][1]'), 1010);
// The gold list, not a "Gold set" box: its off-roster shot is an extra chip.
assert.equal(await js('$("gold-only-label").hidden'), true);
const offered = 'S.queue.map(row => row.shot).join()';
const everything = await js(offered);
const choose = (name) => js(`$("shotlist").value = "${name}"; $("shotlist").dispatchEvent(new Event("change")); void 0`);
await choose("gold");
await until('Lists.name === "gold" && S.queue.length === 2 && S.opened?.ticket === S.ticket');
assert.equal(await js('S.queue.some(row => row.shot === 171234 && row.extra)'), true);
assert.equal(await js('$("queue").children.length === S.queue.length'), true);
assert.equal(await js('$("queue").querySelector(\'[data-shot="171234"]\').classList.contains("extra")'), true);
assert.equal(await js('stored("labeler:list:alfven_eigenmode")'), "gold");
await choose("roster");
await until(`Lists.name === "roster" && ${offered} === ${JSON.stringify(everything)}`);
assert.equal(await js('stored("labeler:list:alfven_eigenmode")'), null);
assert.deepEqual(errors, []);
ws.close();
browser.kill();
console.log("silver browser passed");
""")
    result = subprocess.run(
        [NODE, str(script), base, TOKEN, str(SHELLS[-1]), str(tmp_path / "profile")],
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
    assert "silver browser passed" in result.stdout
