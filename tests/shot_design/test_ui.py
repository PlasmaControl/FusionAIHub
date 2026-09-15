"""Hermetic browser transport contracts: preserve the tools' answers and CLI JSON."""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import pytest

from shot_design import cli, config
from shot_design.mcp import tools
from shot_design.retrieval import phenomena

# Starlette 1.6's import-time type alias uses AnyIO 4.15's deprecated spelling.
# Ignore only this upstream import warning; all test execution remains under -W error.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="The anyio.abc.BlockingPortal alias is deprecated.*",
        category=DeprecationWarning,
    )
    from fastapi.testclient import TestClient


def wire(value):
    return json.loads(json.dumps(value, default=str))


@pytest.fixture(autouse=True)
def local_auxiliary_paths(ideate_db, tmp_path, monkeypatch):
    """The shared DB fixture relocates db, but not the curated CSV or model paths."""
    import yaml

    values = config.load_paths().model_dump(mode="json")
    for key in values:
        if key not in ("data_root", "sentence_transformers_model"):
            values[key] = "${data_root}/" + ("db" if key == "db_dir" else key)
    path = tmp_path / "paths.yaml"
    path.write_text(yaml.safe_dump(values))
    monkeypatch.setenv("SHOT_DESIGN_PATHS", str(path))
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "labelmaker"))
    monkeypatch.setenv("SHOT_DESIGN_CORPUS", str(tmp_path / "corpus"))
    tools.reset_cache()
    phenomena._database_shots.cache_clear()
    yield
    tools.reset_cache()
    phenomena._database_shots.cache_clear()


@pytest.fixture
def client(ideate_db):
    from shot_design.ui.app import create_app

    tools.reset_cache()
    with TestClient(create_app(token="test-token")) as client:
        client.get("/?token=test-token")
        yield client
    tools.reset_cache()


@pytest.mark.parametrize("body", [
    {"ref_shot": 100, "n": 3},
    {"ref_shot": 100, "segment": "flattop", "constraints": {"ip_mean": [1.3e6, None]},
     "require_labels": ["L"], "avoid_labels": ["dud"], "n": 5},
    {"ref_shot": 100, "actuators": {"nbi.total": 2e6}, "n": 2},
    {"constraints": {"not_a_column": [1, None]}},
    {},
])
def test_search_is_tool_json(client, body):
    response = client.post("/api/search", json=body)
    assert response.status_code == 200
    assert response.json() == wire(tools.search_shots(**body))


def test_text_search_uses_tools_without_loading_a_model(client, monkeypatch):
    import numpy as np

    from shot_design.shotdb import text

    # Only the external encoder is substituted; the ranking and route are real.
    monkeypatch.setattr(text, "embed_texts", lambda texts: np.tile(
        np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1)
    ))
    body = {"text": "QH-mode", "n": 3}
    reply = client.post("/api/search", json=body).json()
    assert reply == wire(tools.search_shots(**body))
    assert reply["results"]


def test_transport_does_not_drop_new_fields_or_coerce_values(client, monkeypatch):
    from datetime import date

    payload = {"caveats": ["visible caveat"], "status": "uncovered", "extra": date(2026, 9, 1),
               "events": [], "forecasts": [], "text_mentions": [], "unknown": None}
    monkeypatch.setattr(tools, "get_events", lambda *args, **kw: payload)
    assert client.get("/api/shot/100/events").json() == wire(payload)


@pytest.mark.parametrize("shot,segment", [(100, "flat_top"), (200, "full"),
                                          (100, "flattop"), (999, "flat_top")])
def test_shot_is_tool_json(client, shot, segment):
    response = client.get(f"/api/shot/{shot}", params={"segment": segment})
    assert response.status_code == 200
    payload = response.json()
    if "error" not in payload:
        assert payload.pop("describe_parts")["header"].startswith(f"Shot {shot}")
        assert isinstance(payload.pop("units"), dict)
    assert payload == wire(tools.describe_shot(shot, segment))


@pytest.fixture
def event_db(ideate_db):
    # Reuse the production-shaped event writer; it writes only in this fixture's tmp_path.
    from .test_mcp import _event, write_events

    config.load_paths().qh_database_csv.write_text("shot\n100\n")
    write_events(ideate_db / "db", [
        _event(100, "coherent_mode", 2.0, 3.0, event_id="observed", source="tokeye_track",
               f0_khz=4.0, f1_khz=6.0, attrs={"n_harmonics": 3}),
        _event(100, "eho", 2.5, 2.5, event_id="point", evidence_kind="heuristic"),
        _event(100, "eho", 2.0, 3.0, event_id="forecast", evidence_kind="forecast"),
        _event(100, "eho", 2.0, 3.0, event_id="text", evidence_kind="text",
               attrs={"quote": "Operator says EHO"}),
        _event(100, "eho", 2.0, 3.0, event_id="database", evidence_kind="database"),
    ])
    tools.reset_cache()
    return ideate_db


@pytest.mark.parametrize("params", [{}, {"phenomenon": "eho", "t0_s": 2.0, "t1_s": 2.5},
                                    {"t0_s": 20.0}, {"t0_s": 3.0, "t1_s": 1.0}])
def test_events_is_tool_json_and_preserves_evidence(client, event_db, params):
    response = client.get("/api/shot/100/events", params=params)
    assert response.status_code == 200
    assert response.json() == wire(tools.get_events(100, **params))
    if "error" not in response.json():
        assert {"status", "caveats", "events", "forecasts", "text_mentions",
                "database_intervals", "coverage"} <= response.json().keys()
    if not params:
        assert len(response.json()["events"]) == 2
        assert len(response.json()["forecasts"]) == 1
        assert len(response.json()["text_mentions"]) == 1
        assert len(response.json()["database_intervals"]) == 1


@pytest.mark.parametrize("name,avoid", [("eho", []), ("QH-mode", ["phenomenon:elm"])])
def test_locate_is_cli_json_with_reply_notes(client, event_db, capsys, name, avoid):
    args = ["phenomenon", name, "--n", "5", "--segment", "flat_top",
            "--min-confidence", "0.1", "--json"]
    for item in avoid:
        args += ["--avoid", item]
    assert cli.main(args) == 0
    output = capsys.readouterr()
    params = [("phenomenon", name), ("n", "5"), ("segment", "flat_top"),
              ("min_confidence", "0.1"), *[("avoid", item) for item in avoid]]
    response = client.get("/api/locate", params=params)
    assert response.status_code == 200
    assert response.json() == json.loads(output.out)
    assert response.json(), "CLI equality must exercise real hits, not two empty lists"
    if name == "eho":
        assert response.json()[0]["intervals"]
    assert json.loads(response.headers["X-Ideate-Caveats"]) == output.err.splitlines()


def test_registry_classification_and_literal_events_stay_distinct(client, event_db):
    hit = client.get("/api/locate?phenomenon=eho").json()[0]
    assert hit["intervals"]
    literal = client.get("/api/shot/100/events?phenomenon=eho").json()
    context = client.get("/api/shot/100/events").json()
    assert all(row["phenomenon"] == "eho" for row in literal["events"])
    assert any(row["phenomenon"] == "coherent_mode" for row in context["events"])


@pytest.mark.parametrize("shot,params,status", [
    (999, {}, "unindexed"), (200, {}, "unprocessed"),
    (100, {}, "observed"), (100, {"t0_s": 20.0}, "uncovered"),
])
def test_each_event_status_survives_transport(client, event_db, shot, params, status):
    from shot_design.labels import event_sources

    event_sources.write_sources(event_db / "db/event_sources.parquet", [
        event_sources.source_row(100, "tokeye_track", diag="mhr", channel=0,
                                 t_cov0_s=0.0, t_cov1_s=6.0, n_events=2),
    ])
    response = client.get(f"/api/shot/{shot}/events", params=params)
    assert response.json() == wire(tools.get_events(shot, **params))
    assert response.json()["status"] == status


def test_incomplete_database_uses_the_mcp_guard(ideate_db, client):
    (ideate_db / "db/segments.parquet").unlink()
    response = client.get("/api/shot/100")
    assert response.status_code == 200
    assert response.json() == wire(tools.never_raises(tools.describe_shot)(100))
    assert response.json()["caveats"]


@pytest.mark.parametrize("path", ["/", "/api/meta"])
def test_token_gate_and_cookie(ideate_db, path):
    from shot_design.ui.app import COOKIE, create_app

    with TestClient(create_app(token="a token & more")) as client:
        assert client.get(path).status_code == 401
        assert client.get(path, params={"token": "wrong"}).status_code == 401
        response = client.get(path, params={"token": "a token & more"},
                              follow_redirects=False)
        assert response.status_code == 303
        assert "token=" not in response.headers["location"]
        assert "HttpOnly" in response.headers["set-cookie"]
        assert COOKIE in client.cookies
        assert client.get(path).status_code == 200
        assert client.get(path, params={"token": "wrong"}).status_code == 401


class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href"):
                self.paths.append(value)
            if key == "id":
                self.ids.add(value)


def test_static_assets_and_three_views(client):
    from shot_design.ui.app import STATIC

    response = client.get("/")
    assert response.status_code == 200
    assert response.text == (STATIC / "index.html").read_text()
    parser = Assets()
    parser.feed(response.text)
    assert {"view-search", "view-shot", "view-locate"} <= parser.ids
    assert "Forecasts (model estimates)" in response.text
    assert "<h1>Shot Designer</h1>" in response.text
    assert parser.paths
    for path in parser.paths:
        assert ":" not in path and not path.startswith("//")
        asset = (STATIC / path.lstrip("/")).resolve()
        assert asset.is_relative_to(STATIC.resolve()) and asset.is_file()
        assert client.get("/" + path.lstrip("/")).status_code == 200
    assert client.get("/api/no-such-route").status_code == 404
    assert client.get("/api/no-such-route").headers["content-type"] == "application/json"


def test_meta_and_registry(client):
    reg = phenomena.registry()
    response = client.get("/api/phenomena")
    expected = [{"id": pid, "title": ph.title, "aliases": ph.aliases,
                 "sources": ph.sources, "covering_sources": ph.covering_sources}
                for pid, ph in sorted(reg.items())]
    assert response.json() == wire(expected)
    meta = client.get("/api/meta").json()
    assert meta["db"]["n_shots"] == 4
    assert meta["db"]["shot_range"] == [100, 201]
    assert [p["id"] for p in meta["phenomena"]] == sorted(reg)
    for ph in meta["phenomena"]:
        assert ph["has_detector"] == bool(reg[ph["id"]].covering_sources)


def test_unbuilt_db_preserves_tool_errors(ideate_db, tmp_path, monkeypatch):
    from shot_design.ui.app import create_app

    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path / "unbuilt"))
    with TestClient(create_app(token="test")) as client:
        client.get("/?token=test")
        replies = [
            (client.post("/api/search", json={"ref_shot": 100}),
             tools.search_shots(ref_shot=100)),
            (client.get("/api/shot/100"), tools.describe_shot(100)),
            (client.get("/api/shot/100/events"), tools.get_events(100)),
        ]
        for response, expected in replies:
            assert response.status_code == 200
            assert response.json() == wire(expected)
            assert "error" in response.json() and "caveats" in response.json()
            assert "Traceback" not in response.text
        assert "error" in client.get("/api/locate?phenomenon=eho").json()


def test_app_paths_are_isolated_between_concurrent_requests(ideate_db, tmp_path):
    from shot_design.ui.app import create_app

    original = config.load_paths()
    missing = tmp_path / "custom-db"
    custom = original.model_copy(update={"data_root": tmp_path / "custom-root"})
    with TestClient(create_app(token="a", paths=original)) as a, TestClient(
        create_app(token="b", paths=custom, db_dir=missing)
    ) as b:
        a.get("/?token=a")
        b.get("/?token=b")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(client.get, "/api/shot/100") for client in (a, b) * 4]
            results = [f.result().json() for f in futures]
        for good, bad in zip(results[::2], results[1::2], strict=True):
            assert good["shot"] == 100
            assert str(missing) in bad["error"]
    assert config.load_paths() == original
    assert not missing.exists()


def test_locate_does_not_reuse_another_apps_curated_list(ideate_db, tmp_path):
    from shot_design.ui.app import create_app

    paths = config.load_paths()
    a_csv, b_csv = tmp_path / "a.csv", tmp_path / "b.csv"
    a_csv.write_text("shot\n100\n")
    b_csv.write_text("shot\n200\n")
    with TestClient(create_app(token="a", paths=paths.model_copy(
        update={"qh_database_csv": a_csv}
    ))) as a, TestClient(create_app(token="b", paths=paths.model_copy(
        update={"qh_database_csv": b_csv}
    ))) as b:
        a.get("/?token=a")
        b.get("/?token=b")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(c.get, "/api/locate?phenomenon=qh") for c in (a, b) * 3]
            shots = [[hit["shot"] for hit in f.result().json()] for f in futures]
        assert shots == [[100], [200]] * 3


def test_serve_cli_prints_link_and_forwards_options(ideate_db, monkeypatch, capsys):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    assert cli.main(["serve", "--port", "8767", "--token", "a&b",
                     "--db-dir", str(ideate_db / "db")]) == 0
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8767/?token=a%26b" in output
    assert "ssh -L 8767:localhost:8767 stellar" in output
    app, kw = calls[0]
    assert kw["host"] == "127.0.0.1" and kw["port"] == 8767
    assert kw["access_log"] is False
    assert app.state.paths.db_dir == ideate_db / "db"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--host", "0.0.0.0"])
