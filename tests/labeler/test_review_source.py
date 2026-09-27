"""The table a review page starts from: the pointer's, else the newest format table."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.review import labels
from labeler.events.ui.app import COOKIE, create_app

EVENT = "edge_localized_mode"
ROSTER = "shot,tier,holdout,reviewers,verified_on,notes\n7,unverified,false,,,\n"
LEGACY = "shot,category,t_start,t_end,confidence\n7,0,0,500,\n7,1,500,900,\n"
SUGGESTED = "shot,category,t_start,t_end,confidence\n7,0,100,300,\n7,1,300,400,\n"


@pytest.fixture
def event_dir(tmp_path):
    directory = tmp_path / "events" / EVENT
    (directory / "format").mkdir(parents=True)
    (directory / "format" / f"{EVENT}_format_2026_09_12.csv").write_text(LEGACY)
    (directory / "shots.csv").write_text(ROSTER)
    return directory


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "root" / "suggestions" / "elm_clock" / "v1" / "t.csv"
    path.parent.mkdir(parents=True)
    path.write_text(SUGGESTED)
    return path


def test_without_a_pointer_the_newest_format_table_is_the_source(event_dir):
    assert labels.source_path(event_dir).name == f"{EVENT}_format_2026_09_12.csv"
    assert labels.read_source(event_dir)[7].intervals == ((500, 900, 1),)


def test_a_pointer_names_the_table_and_what_made_it(event_dir, table, monkeypatch):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: "nc1514")
    entry = labels.write_pointer(event_dir, table, method="elm_clock", version="v1")
    assert json.loads(labels.pointer_path(event_dir).read_text()) == entry
    assert (entry["table"], entry["set_by"]) == (str(table.resolve()), "nc1514")
    assert (entry["method"], entry["version"]) == ("elm_clock", "v1")
    assert labels.source_path(event_dir) == table.resolve()
    source = labels.read_source(event_dir)[7]
    assert source.window == (100, 400) and source.intervals == ((300, 400, 1),)


def test_a_pointer_to_a_missing_table_is_refused(event_dir, table):
    labels.write_pointer(event_dir, table, method="elm_clock", version="v1")
    table.unlink()
    with pytest.raises(FileNotFoundError, match="source.json names"):
        labels.source_path(event_dir)
    with pytest.raises(FileNotFoundError, match="no suggestion table"):
        labels.write_pointer(event_dir, table, method="elm_clock", version="v1")


def test_the_page_opens_on_the_suggestion_and_saves_record_it(
    event_dir, table, tmp_path
):
    labels.write_pointer(event_dir, table, method="elm_clock", version="v1")
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=event_dir.parent,
        raw_cache=tmp_path / "raw",
    )
    client = TestClient(create_app(paths=paths, token="secret"))
    client.cookies.set(COOKIE, "secret")
    queue = client.get("/api/queue", params={"event": EVENT}).json()
    assert queue["shots"][0]["state"] == "unreviewed"
    body = {"event": EVENT, "shot": 7, "window": [100, 400]}
    saved = client.post("/api/label", json={**body, "intervals": [[300, 400, 1]]})
    saved = saved.json()
    assert saved["row"]["state"] == "confirmed"
    assert saved["last_save"]["source"] == "t.csv"
