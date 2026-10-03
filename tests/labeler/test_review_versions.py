"""A shot's saved versions and the reviewer's name, in the history and the API,
and the list of names the page asks from."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.review import labels, reviewers, versions
from labeler.events.review.labels import normalise
from labeler.events.ui.app import API_VERSION, COOKIE, create_app

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
    "178642,unverified,true,,,\n"
)
SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
    "178642,0,0,2000,\n"
)
TABLE = "alfven_eigenmode_format_2026_v1.csv"


@pytest.fixture
def event_dir(tmp_path):
    directory = tmp_path / "events" / "alfven_eigenmode"
    (directory / "format").mkdir(parents=True)
    (directory / "format" / TABLE).write_text(SOURCE)
    (directory / "shots.csv").write_text(ROSTER)
    return directory


@pytest.fixture
def client(event_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: "nc1514")
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


def _label(**change):
    body = {
        "event": "alfven_eigenmode",
        "shot": 170815,
        "window": [0, 2000],
        "intervals": [[100, 300, 1]],
    }
    return {**body, **change}


@pytest.mark.parametrize(
    ("typed", "kept"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("  Ada Lovelace ", "Ada Lovelace"),
        ("Zoë Ó Briain", "Zoë Ó Briain"),
        ("x" * versions.NAME_MAX, "x" * versions.NAME_MAX),
    ],
)
def test_a_typed_name_is_kept_without_its_outer_whitespace(typed, kept):
    assert versions.clean_name(typed) == kept


@pytest.mark.parametrize(
    ("typed", "reason"),
    [
        ("x" * (versions.NAME_MAX + 1), "at most 64"),
        ("Ada\nLovelace", "control"),
        ("Ada\tLovelace", "control"),
        ("Ada\x00", "control"),
        ("Ada\u200bLovelace", "control"),
    ],
)
def test_a_name_that_cannot_be_one_line_of_text_is_refused(typed, reason):
    with pytest.raises(ValueError, match=reason):
        versions.clean_name(typed)


def test_a_save_records_the_login_and_the_typed_name(event_dir, monkeypatch):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: "nc1514")
    label = normalise((0, 2000), [(100, 300, 1)])
    named = labels.save(event_dir, 170815, label, source=TABLE, name="Ada Lovelace")
    unnamed = labels.save(event_dir, 170815, label, source=TABLE)
    assert (named["reviewer"], named["name"]) == ("nc1514", "Ada Lovelace")
    assert (unnamed["reviewer"], unnamed["name"]) == ("nc1514", None)
    assert labels.read_history(event_dir) == [named, unnamed]


def test_versions_are_one_shots_saves_numbered_from_one(event_dir, monkeypatch):
    monkeypatch.setattr(labels.getpass, "getuser", lambda: "nc1514")
    first = normalise((0, 2000), [(100, 300, 1)])
    second = normalise((0, 2000), [(150, 400, 1), (900, 950, 1)])
    labels.save(event_dir, 170815, first, source=TABLE, name="Ada")
    labels.save(event_dir, 178642, normalise((0, 2000), []), source=TABLE)
    labels.save(event_dir, 170815, second, source=TABLE, name="Grace")

    found = versions.shot_versions(event_dir, 170815)
    assert [v["version"] for v in found] == [1, 2]
    assert [v["name"] for v in found] == ["Ada", "Grace"]
    assert {v["reviewer"] for v in found} == {"nc1514"}
    assert found[0]["intervals"] == [[100, 300, 1]]
    assert found[1]["intervals"] == [[150, 400, 1], [900, 950, 1]]
    assert found[1]["window"] == [0, 2000] and found[1]["source"] == TABLE
    assert [v["version"] for v in versions.shot_versions(event_dir, 178642)] == [1]
    assert versions.shot_versions(event_dir, 999999) == []


def test_a_history_line_written_before_names_existed_reads_as_unnamed(event_dir):
    old = {
        "shot": 170815,
        "reviewer": "nc1514",
        "saved_at": "2026-09-20T10:00:00+00:00",
        "window": [0, 2000],
        "intervals": [[100, 300, 1]],
        "source": TABLE,
    }
    path = labels.history_path(event_dir)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(old) + "\n")
    [version] = versions.shot_versions(event_dir, 170815)
    assert version["name"] is None and version["reviewer"] == "nc1514"


def test_the_server_reports_its_api_version(client):
    assert client.get("/api/version").json() == {"api": API_VERSION} == {"api": 9}


def test_the_history_route_is_behind_the_gate(client):
    ungated = TestClient(client.app)
    assert (
        ungated.get("/api/history?event=alfven_eigenmode&shot=170815").status_code
        == 401
    )
    assert ungated.get("/api/version").status_code == 401


def test_a_named_save_is_listed_among_the_shots_versions(client):
    posted = client.post("/api/label", json=_label(name="  Ada Lovelace ")).json()
    assert posted["last_save"]["name"] == "Ada Lovelace"
    assert posted["last_save"]["reviewer"] == "nc1514"
    client.post("/api/label", json=_label(intervals=[[150, 400, 1]]))

    body = client.get("/api/history?event=alfven_eigenmode&shot=170815").json()
    assert body["shot"] == 170815
    assert [(v["version"], v["name"]) for v in body["versions"]] == [
        (1, "Ada Lovelace"),
        (2, None),
    ]
    assert body["versions"][1]["intervals"] == [[150, 400, 1]]


def test_a_shot_never_saved_has_no_versions(client):
    body = client.get("/api/history?event=alfven_eigenmode&shot=178642").json()
    assert body == {"shot": 178642, "versions": []}


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("event=alfven_eigenmode&shot=1", "not on this event's roster"),
        ("event=nothing&shot=170815", "unknown event"),
        ("event=..&shot=170815", "unknown event"),
    ],
)
def test_history_off_the_roster_is_not_found(client, query, reason):
    response = client.get(f"/api/history?{query}")
    assert response.status_code == 404
    assert reason in response.json()["error"]


@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("Ada\nLovelace", 400, "control"),
        ("x" * (versions.NAME_MAX + 1), 422, "name"),
        (7, 422, "name"),
    ],
)
def test_a_bad_name_is_refused_and_nothing_is_written(
    client, event_dir, name, status, reason
):
    response = client.post("/api/label", json=_label(name=name))
    assert response.status_code == status
    assert reason in response.json()["error"]
    assert not (event_dir / labels.REVIEW).exists(), "a refused save wrote a file"


def test_the_login_still_cannot_be_set_by_the_page(client, event_dir):
    response = client.post("/api/label", json=_label(reviewer="mallory"))
    assert response.status_code == 422
    assert not (event_dir / labels.REVIEW).exists()


def test_the_list_is_the_saved_names_until_the_first_name_is_added(client, event_dir):
    label = normalise((0, 2000), [(100, 300, 1)])
    for name in ("Grace", "ada", None):
        labels.save(event_dir, 170815, label, source=TABLE, name=name)
    masks = event_dir / labels.REVIEW / "masks.jsonl"
    masks.write_text('{"shot": 170815, "name": "Linus"}\nnot json\n')
    assert client.get("/api/names").json() == {"names": ["ada", "Grace", "Linus"]}

    listed = client.post("/api/names", json={"name": " ADA "})
    assert listed.json() == {"names": ["ada", "Grace", "Linus"], "name": "ada"}
    names_file = reviewers.names_path(event_dir.parent)
    assert not names_file.exists(), "a name already listed wrote the file"

    added = client.post("/api/names", json={"name": "Barbara"}).json()
    assert added == {"names": ["ada", "Barbara", "Grace", "Linus"], "name": "Barbara"}
    assert names_file.read_text() == "ada\nGrace\nLinus\nBarbara\n"
    labels.save(event_dir, 170815, label, source=TABLE, name="Zed")
    assert client.get("/api/names").json() == {"names": added["names"]}


def test_the_owner_s_edits_to_the_file_are_read_once_each_and_kept(client, event_dir):
    names_file = reviewers.names_path(event_dir.parent)
    owner = "\ufeffGrace\n\n  Ada \nada\nbad\u0007name"
    names_file.write_text(owner, encoding="utf-8")
    assert client.get("/api/names").json() == {"names": ["Ada", "Grace"]}
    added = client.post("/api/names", json={"name": "Barbara"}).json()
    assert added["names"] == ["Ada", "Barbara", "Grace"]
    assert names_file.read_text() == owner.removeprefix("\ufeff") + "\nBarbara\n"


@pytest.mark.parametrize(
    ("name", "status", "reason"),
    [
        ("", 400, "required"),
        ("   ", 400, "required"),
        ("Ada\nLovelace", 400, "control"),
        ("x" * (versions.NAME_MAX + 1), 422, "name"),
        (7, 422, "name"),
    ],
)
def test_a_bad_name_is_not_added(client, event_dir, name, status, reason):
    response = client.post("/api/names", json={"name": name})
    assert response.status_code == status
    assert reason in response.json()["error"]
    assert not reviewers.names_path(event_dir.parent).exists()


def test_the_names_are_behind_the_gate(client):
    ungated = TestClient(client.app)
    assert ungated.get("/api/names").status_code == 401
    assert ungated.post("/api/names", json={"name": "Ada"}).status_code == 401


def test_every_reviewer_of_a_shot_is_listed_once_in_the_order_they_came(
    client, event_dir
):
    for name in ["Ada Lovelace", "Grace Hopper", "ada lovelace", None]:
        response = client.post("/api/label", json=_label(name=name))
        assert response.status_code == 200
    masks = event_dir / labels.REVIEW / "masks.jsonl"
    other = {"shot": 178642, "name": "Alan Turing", "saved_at": "2026-09-01T00:00:00"}
    mine = {"shot": 170815, "name": "Emmy Noether", "saved_at": "2099-01-01T00:00:00"}
    masks.write_text(f"{json.dumps(other)}\nnot json\n{json.dumps(mine)}\n")
    listed = reviewers.shot_reviewers(event_dir, 170815)
    # The save made without a name credits no one, and the server's login is not shown.
    assert [(r["name"], r["saves"]) for r in listed] == [
        ("Ada Lovelace", 2),
        ("Grace Hopper", 1),
        ("Emmy Noether", 1),
    ]
    assert all(set(r) == {"name", "saves", "first", "last"} for r in listed)
    assert response.json()["reviewers"] == listed[:2]
    assert reviewers.shot_reviewers(event_dir, 178642)[0]["name"] == "Alan Turing"


def test_a_shot_never_saved_has_no_reviewers(event_dir):
    assert reviewers.shot_reviewers(event_dir, 170815) == []
