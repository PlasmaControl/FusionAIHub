"""Build status in `/api/shot`'s 202 body (plan section C): where a store build stands
(`stage`), how long since it was asked for (`elapsed_s`), its place among the waiting
(`queue_position`, 0 while it runs), whether a registered shot list holds the shot
(`prebuilt`), the signals being fetched (`fetching`) and the activity bar's line
(`note`)."""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from labeler.events import raw, shotlists
from labeler.events.review import build as review_build
from labeler.events.ui import overlay_routes
from labeler.events.ui.app import COOKIE, create_app

from .test_events_ui import _one_trace, _tmp_paths

SHOTS = (170815, 170816, 170817)
ROSTER = "shot,tier,holdout,reviewers,verified_on,notes\n" + "".join(
    f"{shot},unverified,false,,,\n" for shot in SHOTS
)


@pytest.fixture
def paths(tmp_path):
    root = tmp_path / "events"
    (root / "detachment").mkdir(parents=True)
    (root / "detachment" / "shots.csv").write_text(ROSTER)
    return _tmp_paths(tmp_path, root)


@pytest.fixture
def app(paths):
    return create_app(paths=paths, token="secret")


@pytest.fixture
def client(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


@pytest.fixture
def held(monkeypatch):
    """A detachment builder that waits for the test to let it finish."""
    gate, started = threading.Event(), []

    def slow(event, shot, paths):
        started.append(shot)
        assert gate.wait(30), "the test never let the build finish"
        return _one_trace(event, shot, paths)

    monkeypatch.setitem(review_build.BUILDERS, "detachment", slow)
    yield gate, started
    gate.set()


def _until(check, timeout=10.0):
    end = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def _open(client, shot):
    return client.get(f"/api/shot?event=detachment&shot={shot}")


def test_waiting_and_running_builds_say_where_they_stand(client, app, held, monkeypatch):
    gate, started = held
    monkeypatch.setattr(shotlists, "holding",
                        lambda paths, shot: ["silver"] if shot == SHOTS[0] else [])
    for shot in SHOTS:
        first = _open(client, shot)
        assert first.status_code == 202
        body = first.json()
        assert body["building"] is True and body["progress"] is None
        assert set(body) == {"building", "progress", "stage", "fetching", "note",
                             "elapsed_s", "queue_position", "prebuilt"}
        assert body["fetching"] == [] and len(body["note"]) <= 40
    _until(lambda: len(started) == 2)  # a pool of two: the third waits
    running, waiting = _open(client, SHOTS[0]).json(), _open(client, SHOTS[2]).json()
    assert running["stage"] == "building" and running["queue_position"] == 0
    assert running["prebuilt"] is True and running["elapsed_s"] >= 0
    assert waiting["stage"] == "queued" and waiting["queue_position"] == 1
    assert waiting["prebuilt"] is False

    gate.set()
    for shot in SHOTS:
        future = app.state.builds.running.get(("detachment", shot))
        if future is not None:
            future.result(timeout=30)
    opened = _open(client, SHOTS[2])
    assert opened.status_code == 200 and len(opened.json()["rows"]) == 1
    assert ("detachment", SHOTS[2]) not in app.state.builds.asked
    assert ("detachment", SHOTS[2]) not in app.state.builds.started


def test_a_running_build_shows_fetching_then_building(client, app, paths, held, monkeypatch):
    gate, started = held
    shot = SHOTS[1]
    _open(client, shot)
    _until(lambda: shot in started)
    fetch = {"done": 2, "total": 7, "stage": "fetching co2"}
    monkeypatch.setattr(raw, "progress_for", lambda s: dict(fetch) if s == shot else None)
    body = _open(client, shot).json()
    assert body["stage"] == "fetching" and body["progress"] == fetch
    assert body["fetching"] == ["co2"] and body["note"] == "Fetching co2"
    monkeypatch.setattr(raw, "progress_for", lambda s: None)
    store = paths.spectrogram_file("detachment", shot)
    tmp = store.with_name(store.name + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(b"")
    try:
        assert _open(client, shot).json()["stage"] == "building"
    finally:
        tmp.unlink()
    gate.set()
    app.state.builds.running[("detachment", shot)].result(timeout=30)
    assert _open(client, shot).status_code == 200


def test_a_build_is_reading_while_the_shots_file_is_open(paths, tmp_path):
    shot = SHOTS[0]
    assert not overlay_routes.reading(shot)
    assert overlay_routes.build_stage(paths, "detachment", shot) == "building"
    with open(tmp_path / f"{shot}_processed.h5", "wb"):
        assert overlay_routes.reading(shot)
        assert not overlay_routes.reading(SHOTS[1])
        assert overlay_routes.build_stage(paths, "detachment", shot) == "reading"
    assert not overlay_routes.reading(shot)


def test_a_bad_shot_list_never_fails_the_status(paths, monkeypatch):
    def broken(paths, shot):
        raise ValueError("bad shotlists.yaml")

    monkeypatch.setattr(shotlists, "holding", broken)
    assert overlay_routes.prebuilt(paths, SHOTS[0]) is False
    monkeypatch.setattr(shotlists, "holding", lambda paths, shot: ["gold"])
    assert overlay_routes.prebuilt(paths, SHOTS[0]) is True
