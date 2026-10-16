"""The review server's shot-list routes: the dropdown, a list's queue, the overview."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.catalog.cohort import COHORT_COLUMNS
from labeler.events.review import labels
from labeler.events.ui import app as ui

SHOT, OTHER = 170815, 170816  # on every roster
GOLD_ONLY = 171234  # gold, on no roster
LISTED = 180000  # on ae_list (and its child ae_small) only
LOOSE = 190000  # on a list file the registry does not name
ELM = "edge_localized_mode"
KEY = "edge_localized_mode/format/edge_localized_mode_challenge_format_2026_v1"
ROSTER = ("shot,tier,holdout,reviewers,verified_on,notes\n"
          f"{SHOT},gold,false,,,\n{OTHER},unverified,false,,,\n")
REGISTRY = """\
lists:
  silver_batch_1: {title: "Silver batch 1", parent: silver, kind: derived}
  ae_small: {title: "AE small", parent: ae_list}
  ae_list: {title: "AE list", kind: dataset}
events:
  alfven_eigenmode: {recommended: [ae_list, silver, gold], default: ae_list}
"""


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def paths(tmp_path):
    tables = tmp_path / "events"
    _write(tables / "events.yaml", "taxonomy:\n  mhd_n1:\n    display: MHD n=1\n"
           "    subcategories:\n      m1: {display: '1/1'}\n")
    for event in ("alfven_eigenmode", "edge_localized_mode", "mhd_n1"):
        _write(tables / event / "shots.csv", ROSTER)
    _write(tables / "gold/gold_shots.csv", "shot,run_day,year,classes,sources,notes\n"
           f"{SHOT},20260101,2026,,,\n{GOLD_ONLY},20260101,2026,,,\n")
    _write(tables / "silver/silver_shots.csv",
           "shot,batch,order,pool,classes,sources,notes\n"
           f"{SHOT},1,1,corpus,mhd_n1:present;sawtooth:present,format,\n"
           f"{OTHER},2,1,corpus,,format,\n")
    _write(tables / "shotlists/ae_list.csv",
           f"shot,note\n{OTHER},on the roster\n{LISTED},off it\n")
    _write(tables / "shotlists/ae_small.txt", f"{LISTED}\n")
    _write(tables / "shotlists/silver_batch_1.csv", f"shot,batch,order\n{SHOT},1,1\n")
    _write(tables / "shotlists/loose.txt", f"# not registered\n{LOOSE}\n")
    _write(tables / "shotlists.yaml", REGISTRY)
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables, config_dir=tables, raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def client(paths, monkeypatch):
    monkeypatch.setattr(ui.Builds, "status", lambda *args: (Path("built"), None))
    transport = TestClient(ui.create_app(paths=paths, token="secret"))
    transport.cookies.set(ui.COOKIE, "secret")
    return transport


def _save(client, event, shot, intervals=((100, 300, 1),)):
    saved = client.post("/api/label", json={
        "event": event, "shot": shot, "window": [0, 1000],
        "intervals": [list(span) for span in intervals],
    })
    assert saved.status_code == 200, saved.text


def _rows(queue):
    return [(row["shot"], row.get("extra", False)) for row in queue["shots"]]


def test_the_dropdown_offers_recommended_lists_then_the_others_then_the_roster(client):
    offered = client.get("/api/shotlists?event=alfven_eigenmode").json()
    # A child of a recommended list opens the others, in its parent's order.
    assert [item["name"] for item in offered] == [
        "ae_list", "silver", "gold", "ae_small", "silver_batch_1", "loose", "roster"]
    found = {item["name"]: item for item in offered}
    assert found["ae_list"] == {
        "name": "ae_list", "title": "AE list", "n": 2, "kind": "dataset",
        "recommended": True, "default": True, "parent": None, "protect": None,
        "n_saved": 0,
    }
    assert found["gold"]["protect"] == "test" and found["gold"]["recommended"]
    assert found["loose"]["kind"] == "other"
    assert found["ae_small"]["parent"] == "ae_list"
    assert found["roster"] == {
        "name": "roster", "title": "Event roster", "n": 2, "kind": "roster",
        "recommended": False, "default": False, "parent": None, "protect": None,
        "n_saved": 0,
    }
    # An event the registry does not name: silver and gold, then each child after
    # its parent, and its own roster as the default.
    offered = client.get("/api/shotlists?event=edge_localized_mode").json()
    found = [(item["name"], item["recommended"], item["default"]) for item in offered]
    assert found == [
        ("silver", True, False), ("gold", True, False),
        ("silver_batch_1", False, False), ("ae_list", False, False),
        ("ae_small", False, False), ("loose", False, False), ("roster", False, True)]
    assert client.get("/api/shotlists?event=no_such_event").status_code == 404


def test_the_queue_without_a_list_is_todays_and_names_the_default(client):
    queue = client.get("/api/queue?event=alfven_eigenmode").json()
    assert _rows(queue) == [(SHOT, False), (OTHER, False), (GOLD_ONLY, True)]
    assert queue["resume"] == SHOT and queue["list"] == "ae_list"
    assert "note" not in queue
    assert queue["lists"] == client.get("/api/shotlists?event=alfven_eigenmode").json()
    assert client.get("/api/queue?event=alfven_eigenmode&list=roster").json()["shots"] \
        == queue["shots"]
    queue = client.get("/api/queue?event=edge_localized_mode").json()
    assert queue["list"] == "roster" and _rows(queue)[-1] == (GOLD_ONLY, True)


def test_a_lists_queue_flags_off_roster_shots_and_follows_saves_and_files(
    client, paths,
):
    queue = client.get("/api/queue?event=alfven_eigenmode&list=ae_list").json()
    assert queue["list"] == "ae_list" and "note" not in queue
    assert queue["shots"] == [
        {"shot": OTHER, "tier": "unverified", "state": "unreviewed", "saved_at": None},
        {"shot": LISTED, "tier": "unverified", "state": "unreviewed", "saved_at": None,
         "extra": True},
    ]
    assert queue["resume"] == OTHER
    # Any shot on any list opens on any event, unverified off its roster.
    assert ui.shot_tier("alfven_eigenmode", LISTED, paths) == "unverified"
    assert ui.shot_tier("edge_localized_mode", LOOSE, paths) == "unverified"
    assert ui.shot_tier("alfven_eigenmode", SHOT, paths) == "gold"
    with pytest.raises(HTTPException) as refused:
        ui.shot_tier("alfven_eigenmode", 999, paths)
    assert refused.value.status_code == 404
    _save(client, "alfven_eigenmode", LISTED)
    queue = client.get("/api/queue?event=alfven_eigenmode&list=ae_list").json()
    row = queue["shots"][1]
    assert row["shot"] == LISTED and row["state"] == "changed" and row["saved_at"]
    assert {item["name"]: item["n_saved"] for item in queue["lists"]}["ae_list"] == 1
    _write(paths.label_tables / "shotlists/ae_list.csv", f"shot\n{LISTED}\n{SHOT}\n")
    queue = client.get("/api/queue?event=alfven_eigenmode&list=ae_list").json()
    assert _rows(queue) == [(LISTED, True), (SHOT, False)]


def test_a_list_that_is_gone_or_unreadable_falls_back_with_a_note(client, paths):
    today = client.get("/api/queue?event=alfven_eigenmode").json()
    gone = client.get("/api/queue?event=alfven_eigenmode&list=ae_old").json()
    assert gone["note"] == "List ae_old no longer exists; showing ae_list."
    assert gone["list"] == "ae_list" and gone["shots"] == today["shots"]
    _write(paths.label_tables / "shotlists/ae_list.csv", f"shot\n{OTHER}\n{OTHER}\n")
    offered = client.get("/api/shotlists?event=alfven_eigenmode").json()
    assert "ae_list" not in [item["name"] for item in offered]
    assert offered[-1]["name"] == "roster" and offered[-1]["default"]
    for query in ("", "&list=ae_list"):
        queue = client.get(f"/api/queue?event=alfven_eigenmode{query}").json()
        assert queue["list"] == "roster" and queue["shots"] == today["shots"]
        assert queue["note"] == "List ae_list cannot be read; showing roster."


def test_a_subcategorys_list_queue_keeps_the_shots_its_parent_marks(client):
    _save(client, "mhd_n1", SHOT)
    _save(client, "mhd_n1", GOLD_ONLY)
    _save(client, "mhd_n1", OTHER, intervals=())  # all absent: nothing to label in
    today = client.get("/api/queue?event=mhd_n1.m1").json()
    assert _rows(today) == [(SHOT, False)] and today["list"] == "roster"
    queue = client.get("/api/queue?event=mhd_n1.m1&list=gold").json()
    assert _rows(queue) == [(SHOT, False), (GOLD_ONLY, True)]
    assert queue["list"] == "gold"
    roster = queue["lists"][-1]
    assert roster["name"] == "roster" and roster["n"] == 2, "the parent's roster"


def test_a_list_and_its_overview_by_shot(client, paths):
    found = client.get("/api/shotlist?name=ae_list").json()
    assert found == {
        "name": "ae_list", "title": "AE list", "kind": "dataset",
        "rows": [{"shot": OTHER, "note": "on the roster"},
                 {"shot": LISTED, "note": "off it"}],
        "meta": {},
    }
    assert client.get("/api/shotlist?name=loose").json()["kind"] == "other"
    for name in ("roster", "nope", "../x", "a" * 300):
        assert client.get(f"/api/shotlist?name={name}").status_code == 404, name
    overview = client.get("/api/overview?list=silver").json()
    assert set(overview) == {"list", "title", "grid_ms", "events", "shots", "children"}
    assert overview["children"] == [
        {"name": "silver_batch_1", "title": "Silver batch 1", "n": 1}]
    assert [row["event"] for row in overview["events"]] == [
        "alfven_eigenmode", "edge_localized_mode", "mhd_n1", "mhd_n1.m1"]
    assert overview["shots"][0] == {
        "shot": SHOT, "batch": 1, "order": 1, "pool": "corpus",
        "classes": ["mhd_n1:present", "sawtooth:present"], "sources": "format",
        "notes": "",
        "saved": 0, "total": 3, "done": False,
    }
    _save(client, "mhd_n1", SHOT)
    row = client.get("/api/overview?list=silver").json()["shots"][0]
    assert (row["saved"], row["total"]) == (1, 4), "the child opened behind its parent"
    by_shot = client.get(f"/api/overview?list=ae_list&shot={LISTED}").json()
    assert set(by_shot) == {"note", "list", "shot", "grid_ms", "t_range", "events"}
    assert by_shot["note"] == "off it" and len(by_shot["events"]) == 4
    silver_shot = client.get(f"/api/silver/shot?shot={SHOT}").json()
    by_shot = client.get(f"/api/overview?list=silver&shot={SHOT}").json()
    assert by_shot["events"] == silver_shot["events"]
    assert by_shot["t_range"] == silver_shot["t_range"]
    assert client.get("/api/overview?list=silver&shot=999").status_code == 404
    assert client.get("/api/overview?list=nope").status_code == 404
    assert client.get(f"/api/overview?list={'a' * 300}").status_code == 404
    client.cookies.clear()
    for route in ("/api/shotlists?event=alfven_eigenmode", "/api/shotlist?name=ae_list",
                  "/api/overview?list=silver"):
        assert client.get(route).status_code == 401, route


def _table(path, rows):
    """A label table: `rows` of (shot, t_start, t_end, category)."""
    return _write(path, "shot,category,t_start,t_end,confidence\n"
                  + "".join(f"{shot},{c},{a},{b},\n" for shot, a, b, c in rows))


def _layers(client, shot, name):
    by_shot = client.get(f"/api/overview?list={name}&shot={shot}").json()
    return next([layer["name"] for layer in event["layers"]]
                for event in by_shot["events"] if event["event"] == ELM)


def test_drafts_stay_off_roster_shots_holdout_shots_and_withheld_tables(client, paths):
    held, more = 180445, 180002  # on the ELM roster: a holdout shot, a listed one
    away = 180003  # a holdout shot off the roster, on an ordinary table only
    tables = paths.label_tables
    _write(tables / ELM / "shots.csv", ROSTER + f"{held},unverified,true,,,\n"
           f"{more},unverified,false,,,\n")
    _write(paths.shotlists_manifest,
           REGISTRY.replace("lists:\n", "lists:\n  elm_key: {protect: holdout}\n"))
    _write(paths.shotlists / "elm_key.txt", f"{held}\n{away}\n")
    _write(paths.shotlists / "elm_more.txt", f"{more}\n")
    source = _table(tables / ELM / "extend_jump/elm_jump.csv", [(OTHER, 0, 50, 1)])
    labels.write_pointer(tables / ELM, source, method="jump", version="v1")
    _table(tables / f"{KEY}.csv", [(held, 100, 300, 1), (LISTED, 100, 300, 1)])
    _table(tables / ELM / "format/elm_curated.csv",
           [(LISTED, 0, 50, 1), (LOOSE, 0, 50, 1), (more, 0, 50, 1),
            (away, 0, 50, 1)])

    def opened(shot):
        return ui.shot_labels(ELM, shot, paths)[0]

    # Off the roster, a listed shot opens on its draft...
    assert opened(LOOSE)["source_origin"] == "format:edge_localized_mode/elm_curated"
    # ...but never on one the withheld answer key holds, nor a holdout shot (on the
    # roster or off it, `away` on an ordinary table only), and a
    # roster shot keeps today's behaviour: no source row, no draft.
    for shot in (LISTED, held, more, away):
        body = opened(shot)
        assert body["source"] is None and "source_origin" not in body, shot
    # The by-shot strips leave the withheld table out.
    curated = "format:edge_localized_mode/elm_curated"
    assert _layers(client, LISTED, "ae_list") == [curated]
    assert _layers(client, held, "elm_key") == []
    # A pointer at a missing table still fails for a roster shot on a list; a shot the
    # event borrows (off its roster, or gold) skips it as before.
    source.unlink()
    for shot in (more, held):
        with pytest.raises(FileNotFoundError):
            opened(shot)
    assert opened(LOOSE)["source_origin"] == curated
    assert opened(SHOT)["source"] is None


def test_a_blind_shot_is_on_no_list_and_opens_nowhere(client, paths):
    blind = 175000
    rows = [{"shot": shot, "queue_rank": rank, "blind": shot == blind}
            for rank, shot in enumerate([SHOT, OTHER, blind], 1)]
    blank = dict.fromkeys(COHORT_COLUMNS, "")
    cohort = pd.DataFrame([blank | row for row in rows], columns=list(COHORT_COLUMNS))
    (paths.label_tables / "catalog").mkdir()
    cohort.to_csv(paths.label_tables / "catalog/cohort.csv", index=False)
    _write(paths.shotlists / "loose.txt", f"{LOOSE}\n{blind}\n")
    names = [item["name"] for item in client.get(f"/api/shotlists?event={ELM}").json()]
    assert "loose" not in names, "a list holding a blind shot is not read"
    with pytest.raises(HTTPException) as refused:
        ui.shot_tier(ELM, blind, paths)
    assert refused.value.status_code == 404
    assert client.get(f"/api/overview?list=gold&shot={blind}").status_code == 404
