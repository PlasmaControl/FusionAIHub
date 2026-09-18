"""The browser review surface: its gate, its reads, and what it refuses to write."""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.ui.app import COOKIE, create_app

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
    "178642,unverified,true,,,\n"
)
OTHER_ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
)

NO_TOKEN = "no token: reopen the link printed by the verify server"
BAD_TOKEN = "bad token"


def _tmp_paths(tmp_path, tables):
    """A `Paths` rooted entirely under `tmp_path` - never the real corpus."""
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables,
        raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def tables(tmp_path):
    """A label_tables root holding two events with rosters."""
    root = tmp_path / "events"
    event = root / "alfven_eigenmode"
    (event / "format").mkdir(parents=True)
    (event / "shots.csv").write_text(ROSTER)
    other = root / "detachment"
    other.mkdir(parents=True)
    (other / "shots.csv").write_text(OTHER_ROSTER)
    # A directory with no roster is not an event the page can offer.
    (root / "scratch").mkdir()
    return root


@pytest.fixture
def app(tables, tmp_path):
    return create_app(paths=_tmp_paths(tmp_path, tables), token="secret")


@pytest.fixture
def client(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


def test_no_cookie_and_no_token_is_refused(app):
    response = TestClient(app).get("/api/events")
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}


def test_a_wrong_token_is_refused(app):
    response = TestClient(app).get("/api/events?token=wrong")
    assert response.status_code == 401
    assert response.json() == {"error": BAD_TOKEN}
    assert COOKIE not in response.cookies


def test_an_empty_token_is_refused(app):
    response = TestClient(app).get("/api/events?token=")
    assert response.status_code == 401
    assert response.json() == {"error": BAD_TOKEN}


def test_a_wrong_cookie_is_refused(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "wrong")
    response = transport.get("/api/events")
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}


def test_a_token_in_the_wrong_place_is_refused(app):
    """A header or a form field is not the gate; only query or cookie is."""
    transport = TestClient(app)
    header = transport.get("/api/events", headers={"Authorization": "secret"})
    assert header.status_code == 401
    assert header.json() == {"error": NO_TOKEN}
    posted = transport.post("/api/events", data={"token": "secret"})
    assert posted.status_code == 401
    assert posted.json() == {"error": NO_TOKEN}


def test_the_token_is_compared_in_constant_time(app, monkeypatch):
    seen = []
    real = secrets.compare_digest

    def spy(a, b):
        seen.append((a, b))
        return real(a, b)

    monkeypatch.setattr(secrets, "compare_digest", spy)
    TestClient(app).get("/api/events?token=wrong")
    assert seen == [(b"wrong", b"secret")]


def test_the_right_token_sets_the_cookie(app):
    response = TestClient(app, follow_redirects=False).get("/api/events?token=secret")
    assert response.status_code == 303
    assert response.headers["location"] == "/api/events"
    assert response.cookies[COOKIE] == "secret"
    cookie_header = response.headers["set-cookie"]
    assert "HttpOnly" in cookie_header
    assert "samesite=lax" in cookie_header.lower()


def test_an_authorized_response_is_not_cached(client):
    response = client.get("/api/events")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_events_lists_what_is_on_disk(client):
    payload = client.get("/api/events").json()
    names = [row["event"] for row in payload["events"]]
    assert names == ["alfven_eigenmode", "detachment"], (
        "a roster-less dir is not an event"
    )
    row = next(r for r in payload["events"] if r["event"] == "alfven_eigenmode")
    assert row["builder"] == "alfven_eigenmode", "not the generic fallback"
    assert row["n_shots"] == 2
    fallback = next(r for r in payload["events"] if r["event"] == "detachment")
    assert fallback["builder"] == "generic"
    assert fallback["n_shots"] == 1


def test_shots_returns_the_roster_with_correction_counts(client, tables):
    review = tables / "alfven_eigenmode" / "review"
    review.mkdir()
    (review / "178642__nc1514__20260918T120000Z.csv").write_text(
        "shot,category,t_start,t_end,notes\n178642,1,350,1300,\n"
    )
    payload = client.get("/api/shots?event=alfven_eigenmode").json()
    assert payload["event"] == "alfven_eigenmode"
    rows = payload["shots"]
    by_shot = {row["shot"]: row for row in rows}
    assert by_shot[170815] == {
        "shot": 170815,
        "tier": "gold",
        "holdout": False,
        "reviewers": ["alice"],
        "verified_on": "2026-01-01",
        "notes": "",
        "n_corrections": 0,
    }
    assert by_shot[178642] == {
        "shot": 178642,
        "tier": "unverified",
        "holdout": True,
        "reviewers": [],
        "verified_on": "",
        "notes": "",
        "n_corrections": 1,
    }


def test_guidance_comes_back_with_the_event(client):
    payload = client.get("/api/events").json()
    row = next(r for r in payload["events"] if r["event"] == "alfven_eigenmode")
    assert "80-250 kHz" in row["guidance"]


def test_reading_the_page_never_writes_the_roster(client, tables):
    roster = tables / "alfven_eigenmode" / "shots.csv"
    before = (roster.read_bytes(), roster.stat().st_mtime_ns)
    client.get("/api/events")
    client.get("/api/shots?event=alfven_eigenmode")
    assert (roster.read_bytes(), roster.stat().st_mtime_ns) == before


@pytest.fixture
def foreign(tmp_path):
    """A real, valid roster OUTSIDE label_tables, for the traversal tests."""
    outside = tmp_path / "secret_area"
    outside.mkdir()
    (outside / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n999999,gold,false,,,\n"
    )
    return outside


def test_a_traversing_event_is_not_found(client, foreign):
    response = client.get("/api/shots", params={"event": "../secret_area"})
    assert response.status_code == 404
    assert "999999" not in response.text, "the foreign roster was read"


def test_an_absolute_event_is_not_found(client, foreign):
    # pathlib's `/` DISCARDS the root when the right operand is absolute, so
    # an unchecked event name reaches any directory the server's uid can read.
    response = client.get("/api/shots", params={"event": str(foreign)})
    assert response.status_code == 404
    assert "999999" not in response.text, "the foreign roster was read"


def test_an_unknown_event_is_not_found(client):
    response = client.get("/api/shots", params={"event": "no_such_event"})
    assert response.status_code == 404


def test_one_bad_roster_does_not_hide_the_others(client, tables):
    bad = tables / "broken_event"
    bad.mkdir()
    (bad / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n170815,gold,True,,,\n"
    )
    payload = client.get("/api/events").json()
    rows = {row["event"]: row for row in payload["events"]}
    assert "alfven_eigenmode" in rows and "detachment" in rows
    assert "holdout" in rows["broken_event"]["error"], "the reason is not reported"


def test_a_missing_label_tables_root_is_not_a_crash(tmp_path):
    app = create_app(paths=_tmp_paths(tmp_path, tmp_path / "absent"), token="secret")
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    response = transport.get("/api/events")
    assert response.status_code == 200
    assert response.json() == {"events": []}


def test_a_duplicated_token_takes_the_last_value(app):
    """Starlette's QueryParams.get returns the LAST value; pin that."""
    transport = TestClient(app, follow_redirects=False)
    assert transport.get("/api/events?token=wrong&token=secret").status_code == 303
    assert transport.get("/api/events?token=secret&token=wrong").status_code == 401


def test_the_static_mount_is_behind_the_gate(app):
    response = TestClient(app).get("/")
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}
