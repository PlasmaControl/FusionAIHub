"""The overlay routes (`labeler.events.ui.overlay_routes`): per-row TokEye states in
`/api/shot` and `/api/overlays`, requests, the generalised `/api/tokeye`, and the
cache rule (versioned static files cached, the page and the API never)."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler.ae.seg import whole
from labeler.events.review import overlays
from labeler.events.review.rows import Grid
from labeler.events.ui import overlay_routes
from labeler.events.ui.app import COOKIE, create_app

from .test_events_ui import OTHER_ROSTER, ROSTER, _built, _tmp_paths
from .test_review_overlays import FakeSlurm, Pointwise, _ae_built

SHOT = 170815


@pytest.fixture(autouse=True)
def slurm(monkeypatch):
    fake = FakeSlurm()
    monkeypatch.setattr(overlays, "RUN", fake)
    overlays.forget_jobs()
    yield fake
    overlays.forget_jobs()


@pytest.fixture
def kicks(monkeypatch):
    """The drain thread's kicks, recorded instead of started."""
    found = []
    monkeypatch.setattr(overlay_routes.Drains, "kick", lambda self: found.append(1))
    return found


@pytest.fixture
def paths(tmp_path):
    root = tmp_path / "events"
    (root / "alfven_eigenmode" / "format").mkdir(parents=True)
    (root / "alfven_eigenmode" / "shots.csv").write_text(ROSTER)
    (root / "detachment").mkdir()
    (root / "detachment" / "shots.csv").write_text(OTHER_ROSTER)
    return _tmp_paths(tmp_path, root)


@pytest.fixture
def client(paths):
    transport = TestClient(create_app(paths=paths, token="secret"))
    transport.cookies.set(COOKIE, "secret")
    return transport


def test_shot_rows_carry_their_overlay_state(client, paths, kicks):
    _built(paths)
    body = client.get(f"/api/shot?event=alfven_eigenmode&shot={SHOT}").json()
    by_name = {row["name"]: row for row in body["rows"]}
    assert by_name["R0xV1"]["overlay"] == {"state": "not_computed", "reason": "No TokEye yet"}
    assert by_name["p1"]["overlay"] == {"state": "not_applicable",
                                        "reason": "not a spectrogram"}
    answer = client.post("/api/overlays/request", json={"event": "alfven_eigenmode",
                                                         "shot": SHOT})
    assert answer.status_code == 202
    assert answer.json() == {"state": "queued", "queue_position": 1}
    assert kicks == [1]
    body = client.get(f"/api/shot?event=alfven_eigenmode&shot={SHOT}").json()
    overlay = body["rows"][0]["overlay"]
    assert overlay["state"] == "queued" and overlay["queue_position"] == 1
    assert overlay["reason"] == "waiting for a GPU job" and "elapsed_s" in overlay


def test_the_overlays_route_answers_every_row(client, paths, kicks):
    _built(paths)
    found = client.get(f"/api/overlays?event=alfven_eigenmode&shot={SHOT}")
    assert found.status_code == 200
    assert found.json() == {"event": "alfven_eigenmode", "shot": SHOT, "rows": {
        "R0xV1": {"state": "not_computed", "reason": "No TokEye yet"},
        "p1": {"state": "not_applicable", "reason": "not a spectrogram"},
    }}
    missing = client.get("/api/overlays?event=alfven_eigenmode&shot=178642")
    assert missing.status_code == 404
    assert missing.json() == {
        "error": "shot 178642 has no alfven_eigenmode review store yet: open it first",
    }
    assert client.get(f"/api/overlays?event=nothing&shot={SHOT}").status_code == 404
    assert client.get("/api/overlays?event=alfven_eigenmode&shot=1").status_code == 404
    gated = TestClient(client.app).get(f"/api/overlays?event=alfven_eigenmode&shot={SHOT}")
    assert gated.status_code == 401


@pytest.mark.parametrize("body", [
    {"event": "alfven_eigenmode", "shot": "170815"},
    {"event": "alfven_eigenmode", "shot": 170815.0},
    {"event": "alfven_eigenmode", "shot": 0},
    {"event": "alfven_eigenmode; rm", "shot": SHOT},
    {"event": "alfven_eigenmode", "shot": SHOT, "list": "x"},
    {"event": "alfven_eigenmode"},
])
def test_a_request_takes_validated_ints_only(client, paths, kicks, body):
    _built(paths)
    answer = client.post("/api/overlays/request", json=body)
    assert answer.status_code == 422 and "error" in answer.json()
    assert overlays.pending(paths) == [] and kicks == []


def test_a_request_needs_a_store_and_a_spectrogram(client, paths, kicks):
    answer = client.post("/api/overlays/request",
                         json={"event": "alfven_eigenmode", "shot": SHOT})
    assert answer.status_code == 404
    _built(paths, event="detachment")
    detach = client.post("/api/overlays/request", json={"event": "detachment",
                                                        "shot": SHOT})
    assert detach.status_code == 202 and detach.json()["state"] == "queued"


def test_tokeye_serves_every_done_row_for_other_events(client, paths):
    _built(paths, event="detachment")
    url = f"/api/tokeye?event=detachment&shot={SHOT}"
    assert client.get(url).status_code == 404
    summary = overlays.build(paths, [("detachment", SHOT)], device="cpu", workers=0,
                             model=Pointwise(coherent=0.0))
    assert summary["done"] == 1
    body = client.get(url).json()
    assert list(body) == ["rows"] and list(body["rows"]) == ["R0xV1"]
    layer = body["rows"]["R0xV1"]
    assert layer["grid"] == {"t0_ms": 0.0, "dt_ms": 1.0, "n": 2000}
    assert (layer["n_y"], layer["y0"], layer["dy"]) == (4, 0.0, 1.0)
    assert (layer["y0_khz"], layer["dy_khz"], layer["layer"]) == (0.0, 1.0, "tokeye-4afc3948")
    assert len(layer["bits"]) > 0
    shot = client.get(f"/api/shot?event=detachment&shot={SHOT}").json()
    assert shot["rows"][0]["overlay"]["state"] == "done"


def test_tokeye_keeps_the_ae_layer_as_it_was(client, paths):
    lit = np.zeros((257, 98), dtype=bool)
    lit[100, 10:20] = True
    whole.save(whole.layer_file(paths, SHOT), Grid(0.0, 2.048, 98), 0.0, 0.977, lit)
    body = client.get(f"/api/tokeye?event=alfven_eigenmode&shot={SHOT}").json()
    assert body == whole.view(paths, SHOT)
    assert client.get("/api/tokeye?event=alfven_eigenmode&shot=178642").status_code == 404


def test_the_ae_page_keeps_tokeye_full_until_the_gate_passes(client, paths, monkeypatch):
    _ae_built(paths, monkeypatch)
    url = f"/api/tokeye?event=alfven_eigenmode&shot={SHOT}"
    assert client.get(url).json()["layer"] == overlays.LAYER  # no tokeye-full layer
    lit = np.zeros((257, 98), dtype=bool)
    lit[30:40, 50:60] = True
    whole.save(whole.layer_file(paths, SHOT), Grid(0.0, 2.048, 98), 0.0, 0.9765625, lit)
    assert client.get(url).json() == whole.view(paths, SHOT)  # no gate
    assert not overlays.compare(paths)["passed"]
    assert client.get(url).json() == whole.view(paths, SHOT)  # a failed gate
    shot = client.get(f"/api/shot?event=alfven_eigenmode&shot={SHOT}").json()
    assert {row["overlay"].get("layer") for row in shot["rows"]} == {"tokeye-full"}
    overlays.gate_file(paths).write_text(json.dumps(
        {"passed": True, "model_sha256": overlays.MODEL_SHA256}))
    assert client.get(url).json()["layer"] == overlays.LAYER  # a passed gate


def test_a_malformed_overlay_record_never_fails_the_shot(client, paths):
    _built(paths)
    record = overlays.state_file(paths, "alfven_eigenmode", SHOT)
    record.parent.mkdir(parents=True)
    record.write_text('["not", "a", "record"]')
    opened = client.get(f"/api/shot?event=alfven_eigenmode&shot={SHOT}")
    assert opened.status_code == 200
    assert opened.json()["rows"][0]["overlay"]["state"] == "not_computed"
    assert client.get(f"/api/overlays?event=alfven_eigenmode&shot={SHOT}").status_code == 200


def test_the_page_carries_versioned_urls_for_every_local_file(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["cache-control"] == "no-store"
    assert client.get("/index.html").text == page.text
    sources = re.findall(r"""<(?:script|link)\b[^>]*?\s(?:src|href)=["']([^"']+)""",
                         page.text)
    local = [url for url in sources
             if "://" not in url and not url.startswith(("//", "data:"))]
    assert local and all("?v=" in url or "&v=" in url for url in local)
    for name in ("app.js", "style.css"):
        digest = hashlib.sha256((overlay_routes.STATIC / name).read_bytes()).hexdigest()
        assert f"/{name}?v={digest[:8]}" in page.text


def test_only_a_current_versioned_static_file_is_cached(client, paths):
    digest = hashlib.sha256((overlay_routes.STATIC / "app.js").read_bytes()).hexdigest()
    cached = client.get(f"/app.js?v={digest[:8]}")
    assert cached.status_code == 200
    assert cached.headers["cache-control"] == "private, max-age=31536000, immutable"
    for url in ("/app.js", "/app.js?v=00000000", "/api/events?v=" + digest[:8],
                f"/?v={digest[:8]}"):
        assert client.get(url).headers["cache-control"] == "no-store", url
    gated = TestClient(client.app).get(f"/app.js?v={digest[:8]}")
    assert gated.status_code == 401 and gated.headers["cache-control"] == "no-store"


def test_events_are_recomputed_when_a_saves_table_changes(client, paths):
    first = client.get("/api/events").json()
    assert client.get("/api/events").json() == first
    roster = paths.label_tables / "detachment" / "shots.csv"
    roster.write_text(OTHER_ROSTER + "178642,unverified,false,,,\n")
    after = {e["event"]: e for e in client.get("/api/events").json()["events"]}
    assert after["detachment"]["n_shots"] == 2


def test_annotate_never_fails_a_shot(paths, monkeypatch):
    def broken(*args):
        raise OSError("disk gone")

    monkeypatch.setattr(overlays, "row_states", broken)
    rows = [{"name": "a", "kind": "image", "y_units": "kHz"},
            {"name": "b", "kind": "trace"}]
    overlay_routes.annotate(paths, "mhd_n1", SHOT, rows)
    assert rows[0]["overlay"]["state"] == "failed"
    assert rows[1]["overlay"]["state"] == "not_applicable"
