"""A save records the sha256 of the table it was opened on; agreement matches by it.

`agreement` matched a save to the table by the table's name alone, so a table
rebuilt under its own name still counted the saves opened on the old one. Each
save now records the table's sha256 (`source_sha256`), and `agreement` compares
only the saves whose sha256 is the table's now. The history lines written
before have no sha256; they still match by name.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths, sha256_of
from labeler.events.review import agreement, labels
from labeler.events.ui.app import COOKIE, create_app

EVENT = "high_confinement_mode"
TABLE = f"{EVENT}_format_2026_v1.csv"
NEWER = f"{EVENT}_format_2027_v1.csv"
HEADER = "shot,category,t_start,t_end,confidence\n"
ROSTER = "shot,tier,holdout,reviewers,verified_on,notes\n7,unverified,false,,,\n"
#: The table shot 7 opens on: present at 100-300 ms of 0-1000 ms.
OPENED = HEADER + "7,0,0,100,\n7,1,100,300,\n7,0,300,1000,\n"
#: The same name rebuilt: present at 500-900 ms, and a shot more.
REBUILT = HEADER + "7,0,0,500,\n7,1,500,900,\n7,0,900,1000,\n8,0,0,1000,\n"
#: What the page posts for shot 7: the label it opened on, confirmed.
BODY = {"event": EVENT, "shot": 7, "window": [0, 1000], "intervals": [[100, 300, 1]]}


@pytest.fixture
def event_dir(tmp_path):
    directory = tmp_path / "events" / EVENT
    (directory / "format").mkdir(parents=True)
    (directory / "format" / TABLE).write_text(OPENED)
    (directory / "shots.csv").write_text(ROSTER)
    return directory


@pytest.fixture
def table(event_dir):
    return event_dir / "format" / TABLE


@pytest.fixture
def client(event_dir, tmp_path):
    paths = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=event_dir.parent,
        raw_cache=tmp_path / "raw",
    )
    transport = TestClient(create_app(paths=paths, token="secret"))
    transport.cookies.set(COOKIE, "secret")
    return transport


def _post(client) -> dict:
    """The page's save of shot 7: the history entry the server returns."""
    response = client.post("/api/label", json=BODY)
    assert response.status_code == 200, response.text
    return response.json()["last_save"]


def _confirm(event_dir, **opened) -> dict:
    """Shot 7 saved as the table has it; `opened` is `labels.save`'s `source` and
    `source_sha256`, the table the page opened it on."""
    label = labels.normalise((0, 1000), [(100, 300, 1)])
    return labels.save(event_dir, 7, label, **opened)


def test_a_save_with_the_tables_sha256_writes_it_to_the_history(event_dir, table):
    digest = sha256_of(table)
    entry = _confirm(event_dir, source=TABLE, source_sha256=digest)
    assert (entry["source"], entry["source_sha256"]) == (TABLE, digest)
    assert labels.read_history(event_dir) == [entry]


def test_opened_on_gives_the_name_and_sha256_of_each_shots_last_save(event_dir, table):
    digest = sha256_of(table)
    _confirm(event_dir, source="old.csv", source_sha256="0" * 64)
    _confirm(event_dir, source=TABLE, source_sha256=digest)
    labels.save(event_dir, 8, labels.Label((0, 1000)), source=TABLE)
    assert agreement.opened_on(event_dir) == {7: (TABLE, digest), 8: (TABLE, None)}


def test_a_save_on_a_table_since_rebuilt_is_left_out_though_its_name_matches(
    event_dir, table
):
    _confirm(event_dir, source=TABLE, source_sha256=sha256_of(table))
    assert agreement.agreement(event_dir)["shots"] == 1, "the table as it was opened"
    table.write_text(REBUILT)  # the same name, other bytes
    got = agreement.agreement(event_dir)
    assert (got["shots"], got["tp"], got["fp"], got["fn"]) == (0, 0, 0, 0)
    assert got["precision"] is None
    assert got["excluded_saves"] == {TABLE: 1}


def test_a_save_whose_sha256_matches_the_table_is_kept(event_dir, table):
    _confirm(event_dir, source=TABLE, source_sha256=sha256_of(table))
    got = agreement.agreement(event_dir)
    assert (got["shots"], got["unchanged"], got["tp"]) == (1, 1, 20)
    assert got["excluded_saves"] == {}


def test_a_save_is_kept_on_the_same_bytes_under_another_name(event_dir, table):
    _confirm(event_dir, source=TABLE, source_sha256=sha256_of(table))
    copy = table.with_name(NEWER)  # the newest format table by name: the source now
    copy.write_bytes(table.read_bytes())
    got = agreement.agreement(event_dir)
    assert got["table"] == str(copy)
    assert (got["shots"], got["excluded_saves"]) == (1, {})


def test_the_last_save_decides_which_table_a_shot_was_opened_on(event_dir, table):
    _confirm(event_dir, source=TABLE, source_sha256="0" * 64)
    assert agreement.agreement(event_dir)["shots"] == 0, "opened on other bytes"
    _confirm(event_dir, source=TABLE, source_sha256=sha256_of(table))  # reopened
    assert agreement.agreement(event_dir)["shots"] == 1


def test_a_save_with_no_sha256_is_matched_by_name(event_dir, table):
    entry = _confirm(event_dir, source=TABLE)
    assert "source_sha256" not in entry
    # The owner's oldest lines have no `name` either.
    old = {key: value for key, value in entry.items() if key != "name"}
    labels.history_path(event_dir).write_text(json.dumps(old) + "\n")
    assert agreement.opened_on(event_dir) == {7: (TABLE, None)}
    assert agreement.agreement(event_dir)["shots"] == 1
    table.write_text(REBUILT)  # by name alone, a rebuilt table is not told apart
    assert agreement.agreement(event_dir)["shots"] == 1
    table.with_name(NEWER).write_text(OPENED)  # the source has another name now
    got = agreement.agreement(event_dir)
    assert (got["shots"], got["excluded_saves"]) == (0, {TABLE: 1})


def test_a_label_with_no_history_line_is_left_out(event_dir):
    _confirm(event_dir, source=TABLE)
    labels.history_path(event_dir).unlink()  # a crash between the label and its line
    got = agreement.agreement(event_dir)
    assert got["shots"] == 0 and sum(got["excluded_saves"].values()) == 1


def test_the_save_route_records_the_served_tables_sha256(
    client, event_dir, table, tmp_path
):
    saved = _post(client)
    assert (saved["source"], saved["source_sha256"]) == (TABLE, sha256_of(table))
    assert agreement.agreement(event_dir)["shots"] == 1
    # A pointer moves the page to a suggestion table: that is the table it serves.
    pointed = tmp_path / "root" / "suggestions" / "dalpha_lh" / "v1" / "t.csv"
    pointed.parent.mkdir(parents=True)
    pointed.write_text(REBUILT)
    labels.write_pointer(event_dir, pointed, method="dalpha_lh", version="v1")
    moved = _post(client)
    assert (moved["source"], moved["source_sha256"]) == ("t.csv", sha256_of(pointed))
    assert moved["source_sha256"] != saved["source_sha256"]
    assert labels.read_history(event_dir) == [saved, moved]
    assert agreement.opened_on(event_dir) == {7: ("t.csv", sha256_of(pointed))}
    got = agreement.agreement(event_dir)  # hashed from the pointer's table
    assert (got["shots"], got["excluded_saves"]) == (1, {})


def test_the_save_route_on_an_event_without_a_table_records_no_sha256(
    client, event_dir, table
):
    table.unlink()
    assert labels.source_path(event_dir) is None
    saved = _post(client)
    assert saved["source"] is None and "source_sha256" not in saved
    assert agreement.opened_on(event_dir) == {7: (None, None)}
