"""The signal workspace's routes (`ui.workspace_routes`) in the app, with a fake fetch
worker: /api/version, /api/catalog, /api/signal, /api/profile, /api/fetch/status and
/api/views, their error bodies and codes, the blind shots and the token gate."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.raw import NO_FETCH_ENV
from labeler.events.review import catalog as cat
from labeler.events.review.fetch_worker import Fdp, Job
from labeler.events.ui import workspace_routes
from labeler.events.ui.app import API_VERSION, COOKIE, create_app

SHOT, BLIND = 180001, 180009
REPO_CONFIG = Path(__file__).parents[2] / "configs" / "labeler"
CATALOG = """
signals:
  - {id: fast, label: Fast, group: T, units: keV, kind: trace, events: [mhd_n1],
     legend: [a, b, c], sources: [{corpus: ece, rows: "0-2"}]}
  - {id: point, label: Point, group: T, units: V, kind: trace,
     sources: [{ptdata: PT1}]}
  - {id: qprof, label: q, group: P, units: "", kind: profile, axis: psi_n,
     sources: [{recipe: park_q}]}
"""
PRESETS = "presets: {mhd_n1: [{name: Preset, layout: {v: 1, rows: []}}]}\n"


class FakeFetcher:
    def __init__(self):
        self.jobs: dict = {}

    def submit(self, key, request, *, shot, ident, where, limit_s, on_result):
        return self.jobs.setdefault(key, Job(key, request, shot, ident, where,
                                             limit_s, on_result))

    def job(self, key):
        return self.jobs.get(key)

    def take_error(self, key):
        return None

    def position(self, job):
        return 1

    def summary(self):
        return {"queue": len(self.jobs), "running": None}


class Watch:
    def __init__(self, state="ok"):
        self.state = state

    def get(self):
        return Fdp(self.state != "unavailable", "2030-01-01T00:00:00Z", self.state)


@pytest.fixture
def paths(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(REPO_CONFIG / "events.yaml", config / "events.yaml")
    (config / cat.CATALOG_FILE).write_text(CATALOG)
    (config / "views.yaml").write_text(PRESETS)
    p = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus",
              text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
              label_tables=tmp_path / "events", config_dir=config,
              raw_cache=tmp_path / "raw")
    (p.label_tables / "mhd_n1").mkdir(parents=True)
    (p.label_tables / "mhd_n1" / "shots.csv").write_text("shot\n180001\n")
    p.corpus.mkdir()
    for shot in (SHOT, BLIND):
        with h5py.File(p.corpus_file(shot), "w") as f:
            g = f.create_group("ece")
            g.create_dataset("xdata", data=np.linspace(0, 2, 2000, dtype=np.float32))
            g.create_dataset("ydata", data=np.ones((3, 2000), np.float32))
    park = p.root / "mhd" / "efit02"
    park.mkdir(parents=True)
    times = np.array([100.0, 200.0, 300.0])
    grid = np.zeros((3, 9, 9))
    np.savez(park / f"{SHOT}.npz", times=times, qpsi=np.ones((3, 5)),
             fpol=np.ones((3, 5)), psirz=grid, r=np.linspace(1, 2, 9),
             z=np.linspace(-1, 1, 9), ssimag=np.zeros(3), ssibry=np.ones(3),
             rmaxis=np.full(3, 1.5))
    return p


@pytest.fixture
def app(paths, monkeypatch):
    monkeypatch.delenv(NO_FETCH_ENV, raising=False)
    monkeypatch.setattr(workspace_routes, "WATCH", Watch())
    made = create_app(paths=paths, token="secret")
    made.state.workspace.fetcher = FakeFetcher()
    made.state.workspace.watch = Watch()
    return made


@pytest.fixture
def client(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


def test_version_says_the_workspace_and_fdp(client):
    got = client.get("/api/version").json()
    assert got == {"api": API_VERSION, "workspace": True,
                   "fdp": {"env": True, "expires_at": "2030-01-01T00:00:00Z"}}


def test_the_catalog_for_an_event_and_a_shot(client):
    got = client.get("/api/catalog", params={"event": "mhd_n1", "shot": SHOT}).json()
    assert got["trees"] == list(cat.TREES) and got["node_max"] == cat.NODE_MAX
    assert got["ptdata_pattern"] == cat.PTDATA.pattern
    assert got["node_pattern"] == cat.NODE_PATTERN
    by_id = {e["id"]: e for e in got["entries"]}
    assert set(by_id["fast"]) == {"id", "label", "group", "units", "kind", "channels",
                                  "axis", "suggested", "source", "where", "local",
                                  "description", "points"}
    assert by_id["fast"]["points"] == [] and by_id["point"]["points"] == ["PT1"]
    assert (by_id["fast"]["local"], by_id["fast"]["suggested"]) == ("disk", True)
    assert (by_id["fast"]["channels"], by_id["fast"]["source"]) == (3, "corpus")
    assert (by_id["point"]["local"], by_id["point"]["suggested"]) == ("fetch", False)
    assert by_id["qprof"]["local"] == "disk" and by_id["qprof"]["axis"] == "psi_n"
    other = client.get("/api/catalog", params={"shot": 180002}).json()
    assert {e["id"]: e["local"] for e in other["entries"]} == {
        "fast": "missing", "point": "fetch", "qprof": "missing"}
    plain = client.get("/api/catalog").json()
    assert all(not e["suggested"] for e in plain["entries"])


def test_the_catalog_refuses_an_unknown_event_or_a_bad_shot(client):
    assert client.get("/api/catalog", params={"event": "nope"}).status_code == 404
    got = client.get("/api/catalog", params={"shot": "18x"})
    assert got.status_code == 422 and got.json()["kind"] == "invalid"


def test_a_signal_window(client):
    got = client.get("/api/signal", params={"shot": SHOT, "id": "fast", "t0": 0,
                                            "t1": 2000, "cols": 100})
    assert got.status_code == 200
    assert got.headers["content-type"] == "application/octet-stream"
    assert json.loads(got.headers["X-Grid"]) == {"t0": 0.0, "t1": 2000.0, "n": 100}
    signal = json.loads(got.headers["X-Signal"])
    assert (signal["label"], signal["units"], signal["legend"], signal["source"]) == (
        "Fast", "keV", ["a", "b", "c"], "corpus")
    values = np.frombuffer(got.content, "<f4").reshape(2, 3, 100)
    assert np.allclose(values, 1.0)


def test_a_fetched_signal_waits_with_a_202(client, app):
    got = client.get("/api/signal", params={"shot": SHOT, "id": "custom:DUSBRADIAL",
                                            "t0": 0, "t1": 2000})
    assert got.status_code == 202
    assert set(got.json()) == {"state", "stage", "elapsed_s", "queue_position"}
    assert got.json()["state"] == "queued"
    status = client.get("/api/fetch/status").json()
    assert status == {"fdp": "ok", "queue": 1, "running": None}
    app.state.workspace.watch = Watch("expired")
    got = client.get("/api/signal", params={"shot": SHOT, "id": "point", "t0": 0,
                                            "t1": 2000})
    assert got.status_code == 503 and got.json()["kind"] == "auth"


@pytest.mark.parametrize("params, status, kind", [
    ({"id": "custom:ip()"}, 422, "invalid"),
    ({"id": "mds:CER:\\CER::X"}, 422, "invalid"),
    ({"id": "nope"}, 422, "invalid"),
    ({"id": None}, 422, "invalid"),
    ({"t0": None}, 422, "invalid"),
    ({"t1": "soon"}, 422, "invalid"),
    ({"t0": 5, "t1": 5}, 422, "invalid"),
    ({"cols": "-1"}, 422, "invalid"),
    ({"cols": "8"}, 422, "invalid"),
    ({"shot": "../x"}, 422, "invalid"),
    ({"shot": 180002}, 404, "missing"),
    ({"shot": BLIND}, 404, "missing"),
])
def test_a_bad_signal_request(client, paths, monkeypatch, params, status, kind):
    (paths.label_tables / "catalog").mkdir()
    (paths.label_tables / "catalog" / "cohort.csv").write_text("blind\n")
    monkeypatch.setattr(workspace_routes.silver, "blind_shots",
                        lambda *a, **k: {BLIND})
    query = {"shot": SHOT, "id": "fast", "t0": 0, "t1": 2000, **params}
    got = client.get("/api/signal",
                     params={k: v for k, v in query.items() if v is not None})
    assert got.status_code == status
    assert got.json()["kind"] == kind and got.json()["error"]


def test_a_profile(client):
    got = client.get("/api/profile", params={"shot": SHOT, "id": "qprof"})
    assert got.status_code == 200
    header = json.loads(got.headers["X-Profile"])
    assert header["n_t"] == 3 and header["n_x"] == 5 and header["x_kind"] == "psi_n"
    assert header["label"] == "q (EFIT02)" and header["source"] == "park"
    assert len(got.content) == 4 * (3 + 5 + 15)
    trace = client.get("/api/profile", params={"shot": SHOT, "id": "fast"})
    assert trace.status_code == 422 and trace.json()["kind"] == "invalid"
    none = client.get("/api/profile", params={"shot": 180002, "id": "qprof"})
    assert none.status_code == 404 and none.json()["kind"] == "missing"


def test_views_through_the_api(client):
    got = client.get("/api/views", params={"event": "mhd_n1", "reviewer": "Ana"})
    assert got.json() == {"views": [{"name": "Preset", "layout": {"v": 1, "rows": []},
                                     "etag": got.json()["views"][0]["etag"],
                                     "preset": True}]}
    body = {"op": "save", "event": "mhd_n1", "reviewer": "Ana", "name": "Mine",
            "layout": {"v": 1, "rows": [{"id": "fast"}]}}
    saved = client.post("/api/views", json=body)
    assert saved.status_code == 200
    assert [v["name"] for v in saved.json()["views"]] == ["Mine", "Preset"]
    again = client.post("/api/views", json=body)  # another tab, no etag
    assert again.status_code == 409 and "another tab" in again.json()["error"]
    missing = client.post("/api/views", json={**body, "op": "delete", "name": "Gone"})
    assert missing.status_code == 404
    big = {**body, "name": "Big", "layout": {"pad": "x" * 70_000}}
    assert client.post("/api/views", json=big).status_code == 400
    huge = client.post("/api/views", content=b"{" + b" " * 300_000 + b"}")
    assert huge.status_code == 400
    assert client.post("/api/views", content=b"{nope").status_code == 400
    deep = client.post("/api/views", content=b"[" * 100_000 + b"]" * 100_000)
    assert deep.status_code == 400 and deep.json()["error"]
    assert "256 KB" in huge.json()["error"]
    assert client.post("/api/views", json={**body, "event": "nope"}).status_code == 404
    assert client.get("/api/views", params={"event": "nope"}).status_code == 404
    assert client.get("/api/views").status_code == 422


def test_status_says_off_while_fetching_is_off(client, monkeypatch):
    monkeypatch.setenv(NO_FETCH_ENV, "1")
    got = client.get("/api/fetch/status").json()
    assert got["fdp"] == "off" and NO_FETCH_ENV in got["reason"]


def test_every_route_needs_the_token(app):
    bare = TestClient(app)
    for url in ("/api/catalog", "/api/signal", "/api/profile", "/api/fetch/status",
                "/api/views?event=mhd_n1"):
        assert bare.get(url).status_code == 401
    assert bare.post("/api/views", json={}).status_code == 401
