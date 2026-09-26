"""The label review server: its gate, what it serves, and the one write it makes."""

from __future__ import annotations

import json
import secrets

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from labeler.config import Paths
from labeler.events.interval_tables import validate_intervals
from labeler.events.review import build as review_build
from labeler.events.review import labels
from labeler.events.review import rows as review_rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow
from labeler.events.ui.app import COOKIE, create_app
from labeler.events.verify import NoDataError

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
    "178642,unverified,true,,,\n"
)
OTHER_ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
)

SOURCE = (
    "shot,category,t_start,t_end,confidence\n"
    "170815,0,0,100,\n"
    "170815,1,100,300,\n"
    "170815,0,300,2000,\n"
    "178642,0,0,2000,\n"
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
def paths(tables, tmp_path):
    return _tmp_paths(tmp_path, tables)


@pytest.fixture
def app(paths):
    return create_app(paths=paths, token="secret")


@pytest.fixture
def client(app):
    transport = TestClient(app)
    transport.cookies.set(COOKIE, "secret")
    return transport


@pytest.fixture
def source(tables):
    """The event's format table: 170815 has one AE span, 178642 none."""
    path = tables / "alfven_eigenmode/format/alfven_eigenmode_format_2026_v1.csv"
    path.write_text(SOURCE)
    return path


def _built(paths, event="alfven_eigenmode", shot=170815):
    """A store file of two rows over 0-2000 ms at 1 ms: a 4-bin image and a trace."""
    image = ImageRow(
        "R0", "R0", (np.arange(8000) % 256).reshape(4, 2000).astype("uint8"),
        y0=0.0, dy=1.0, y_units="kHz", z_lo=-3.0, z_hi=27.0, z_units="dB",
        band=(80.0, 250.0),
    )
    trace = TraceRow(
        "p1", "Density", np.stack([np.zeros((1, 2000)), np.ones((1, 2000))])
    )
    path = paths.spectrogram_file(event, shot)
    review_rows.write(path, Grid(0.0, 1.0, 2000), [image, trace], event=event)
    return path


def _label(**change):
    """What the page posts for 170815 when the reviewer confirms its source label."""
    label = {
        "event": "alfven_eigenmode",
        "shot": 170815,
        "window": [0, 2000],
        "intervals": [[100, 300, 1]],
    }
    return {**label, **change}


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


def test_the_right_token_also_sets_a_cookie_a_framed_page_can_keep(app):
    """VS Code's Simple Browser frames the page; a Lax cookie is dropped
    there. The second cookie is SameSite=None and Secure, which 127.0.0.1
    over http is allowed to keep, and the gate accepts it alone."""
    from labeler.events.ui.app import FRAMED_COOKIE

    response = TestClient(app).get("/?token=secret", follow_redirects=False)
    assert response.status_code == 303
    framed = [
        h for h in response.headers.get_list("set-cookie") if FRAMED_COOKIE in h
    ]
    assert len(framed) == 1
    assert "samesite=none" in framed[0].lower()
    assert "secure" in framed[0].lower()
    assert "HttpOnly" in framed[0]

    transport = TestClient(app)
    transport.cookies.set(FRAMED_COOKIE, "secret")
    assert transport.get("/api/events").status_code == 200
    wrong = TestClient(app)
    wrong.cookies.set(FRAMED_COOKIE, "wrong")
    assert wrong.get("/api/events").status_code == 401


def test_an_authorized_response_is_not_cached(client):
    response = client.get("/api/events")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_events_lists_each_roster_and_how_much_of_it_is_reviewed(client, source):
    # "scratch" has no roster, so it is not an event. AE is a catalog phenomenon.
    states = {"1": "present", "2": "uncertain", "3": "not_observable"}
    assert client.get("/api/events").json() == {
        "events": [
            {"event": "alfven_eigenmode", "n_shots": 2, "n_reviewed": 0,
             "categories": states},
            {"event": "detachment", "n_shots": 1, "n_reviewed": 0,
             "categories": {"1": "present"}},
        ]
    }


def test_the_queue_is_the_roster_in_order_with_where_to_resume(client, source):
    assert client.get("/api/queue?event=alfven_eigenmode").json() == {
        "shots": [
            {"shot": 170815, "tier": "gold", "state": "unreviewed",
             "saved_at": None},
            {"shot": 178642, "tier": "unverified", "state": "unreviewed",
             "saved_at": None},
        ],
        "resume": 170815,
    }


def test_a_built_shot_opens_with_its_grid_its_rows_and_its_labels(
    client, paths, source
):
    _built(paths)
    body = client.get("/api/shot?event=alfven_eigenmode&shot=170815").json()
    assert body["tier"] == "gold"
    assert body["grid"] == {"t0": 0.0, "dt": 1.0, "n": 2000}
    assert body["t_range"] == [0.0, 2000.0]
    assert [row["name"] for row in body["rows"]] == ["R0", "p1"]
    assert body["rows"][0]["band"] == [80.0, 250.0]
    assert body["source"] == {"window": [0, 2000], "intervals": [[100, 300, 1]]}
    assert (body["saved"], body["last_save"], body["state"]) == (
        None, None, "unreviewed"
    )


def _one_trace(event, shot, paths):
    """A builder standing in for the panel builder: one 10 ms trace."""
    trace = TraceRow("p0", "Trace", np.zeros((2, 1, 10), dtype="float32"))
    return Grid(0.0, 1.0, 10), [trace], {"params": {}}


def test_a_shot_with_no_rows_is_built_on_first_open(client, app, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "detachment", _one_trace)
    first = client.get("/api/shot?event=detachment&shot=170815")
    assert first.status_code == 202
    assert first.json() == {"building": True, "progress": None}
    app.state.builds.running[("detachment", 170815)].result()
    body = client.get("/api/shot?event=detachment&shot=170815").json()
    assert body["grid"] == {"t0": 0.0, "dt": 1.0, "n": 10}
    assert body["source"] is None and body["state"] == "unreviewed"


def test_a_shot_that_cannot_be_built_says_why_and_is_retried(
    client, app, monkeypatch
):
    def broken(event, shot, paths):
        raise NoDataError(f"no features for {shot}")

    monkeypatch.setitem(review_build.BUILDERS, "detachment", broken)
    url = "/api/shot?event=detachment&shot=170815"
    for _ in range(2):  # the second open retries rather than repeating the error
        assert client.get(url).status_code == 202
        with pytest.raises(NoDataError):
            app.state.builds.running[("detachment", 170815)].result()
        response = client.get(url)
        assert response.status_code == 502
        assert response.json() == {"error": "NoDataError: no features for 170815"}


def test_a_shot_off_the_roster_is_not_found_and_not_built(client, app):
    response = client.get("/api/shot?event=alfven_eigenmode&shot=1")
    assert response.status_code == 404
    assert response.json() == {"error": "shot 1 is not on this event's roster"}
    assert app.state.builds.running == {}


def test_rows_come_back_as_bytes_with_the_grid_they_cover(client, paths):
    _built(paths)
    response = client.get(
        "/api/rows?event=alfven_eigenmode&shot=170815&t0=0&t1=2000&cols=500"
    )
    assert response.headers["content-type"] == "application/octet-stream"
    grid = json.loads(response.headers["x-grid"])
    assert grid == {"t0": 0.0, "t1": 2000.0, "n": 500}
    # The rows in store order, every 4 columns pooled into one: the image as
    # uint8 (4, 500), then the trace as float32 (min/max, 1 channel, 500).
    image = np.frombuffer(response.content[:2000], dtype=np.uint8).reshape(4, 500)
    trace = np.frombuffer(response.content[2000:], dtype="<f4").reshape(2, 1, 500)
    assert image[0, :3].tolist() == [3, 7, 11]
    assert (trace[0] == 0).all() and (trace[1] == 1).all()


@pytest.mark.parametrize(
    ("query", "status", "reason"),
    [
        ("shot=178642&t0=0&t1=2000", 404, "has no rows yet"),
        ("shot=170815&t0=5000&t1=6000", 400, "outside the record"),
        ("shot=170815&t0=0&t1=2000&cols=8", 422, "cols:"),
    ],
)
def test_a_bad_rows_request_is_refused_with_its_reason(
    client, paths, query, status, reason
):
    _built(paths)
    response = client.get(f"/api/rows?event=alfven_eigenmode&{query}")
    assert response.status_code == status
    assert reason in response.json()["error"]


def test_confirming_the_source_label_saves_it_as_a_format_table(
    client, tables, source
):
    body = client.post("/api/label", json=_label()).json()
    assert body["row"]["state"] == "confirmed" and body["row"]["tier"] == "gold"
    assert body["saved"] == {"window": [0, 2000], "intervals": [[100, 300, 1]]}
    assert body["last_save"]["saved_at"] == body["row"]["saved_at"]
    table = pd.read_csv(tables / "alfven_eigenmode/review/labels.csv")
    written = validate_intervals(table)[["shot", "category", "t_start", "t_end"]]
    assert written.values.tolist() == [
        [170815, 0, 0, 100],
        [170815, 1, 100, 300],
        [170815, 0, 300, 2000],
    ]
    queue = client.get("/api/queue?event=alfven_eigenmode").json()
    assert queue["shots"][0]["state"] == "confirmed"
    assert queue["resume"] == 178642
    assert client.get("/api/events").json()["events"][0]["n_reviewed"] == 1


def test_a_changed_label_is_saved_merged_and_shown_beside_its_source(
    client, paths, tables, source
):
    _built(paths)
    client.post("/api/label", json=_label())
    client.post("/api/label", json=_label(shot=178642, intervals=[]))
    changed = [[150, 250.4, 1], [240, 400, 1]]
    body = client.post("/api/label", json=_label(intervals=changed)).json()
    assert body["saved"] == {"window": [0, 2000], "intervals": [[150, 400, 1]]}
    assert body["row"]["state"] == "changed"
    view = client.get("/api/shot?event=alfven_eigenmode&shot=170815").json()
    assert view["source"]["intervals"] == [[100, 300, 1]]
    assert view["saved"]["intervals"] == [[150, 400, 1]]
    assert view["state"] == "changed"
    history = labels.read_history(tables / "alfven_eigenmode")
    assert [entry["shot"] for entry in history] == [170815, 178642, 170815]


def test_saving_a_shot_with_attrs_returns_conflict_and_keeps_both_files(client, tables):
    review = tables / "alfven_eigenmode" / "review"
    review.mkdir()
    (review / "labels.csv").write_text(
        'shot,category,t_start,t_end,confidence,attrs\n'
        '170815,1,0,2000,,"{""type"": ""TAE""}"\n'
    )
    (review / "history.jsonl").write_text('{"shot": 170815}\n')
    before = {name: (review / name).read_bytes() for name in [
        "labels.csv", "history.jsonl"
    ]}
    response = client.post("/api/label", json=_label())
    assert response.status_code == 409
    assert "170815" in response.json()["error"]
    assert "attrs" in response.json()["error"]
    assert {name: (review / name).read_bytes() for name in before} == before


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"shot": 1}, 404),
        ({"window": [0, 30000]}, 400),
        ({"window": [2000, 0], "intervals": []}, 400),
        ({"intervals": [[300, 100, 1]]}, 400),
        ({"intervals": [[100, 300, 7]]}, 400),
        ({"reviewer": "mallory"}, 422),
        ({"event": "../secret_area"}, 404),
    ],
)
def test_a_bad_label_is_refused_and_nothing_is_written(
    client, tables, foreign, source, change, status
):
    response = client.post("/api/label", json=_label(**change))
    assert response.status_code == status
    assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode/review").exists()
    assert not (foreign / "review").exists()


def test_a_non_finite_time_is_refused(client, tables, source):
    content = json.dumps(_label()).replace("[0, 2000]", "[0, Infinity]")
    response = client.post(
        "/api/label", content=content, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert not (tables / "alfven_eigenmode/review").exists()


def test_a_label_is_behind_the_token_gate(app, tables, source):
    response = TestClient(app).post("/api/label", json=_label())
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}
    assert not (tables / "alfven_eigenmode/review").exists()


def test_the_roster_is_read_and_never_written(client, tables, source):
    roster = tables / "alfven_eigenmode" / "shots.csv"
    before = (roster.read_bytes(), roster.stat().st_mtime_ns)
    client.get("/api/events")
    client.get("/api/queue?event=alfven_eigenmode")
    assert client.post("/api/label", json=_label()).status_code == 200
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
    response = client.get("/api/queue", params={"event": "../secret_area"})
    assert response.status_code == 404
    assert "999999" not in response.text, "the foreign roster was read"


def test_an_absolute_event_is_not_found(client, foreign):
    # pathlib's `/` DISCARDS the root when the right operand is absolute, so
    # an unchecked event name reaches any directory the server's uid can read.
    response = client.get("/api/queue", params={"event": str(foreign)})
    assert response.status_code == 404
    assert "999999" not in response.text, "the foreign roster was read"


def test_an_unknown_event_is_not_found(client):
    response = client.get("/api/queue", params={"event": "no_such_event"})
    assert response.status_code == 404


def test_an_overlong_event_is_not_found_not_a_500(client):
    """`Path.is_file` re-raises ENAMETOOLONG rather than reporting False -

    it is not in pathlib's `_IGNORED_ERRNOS` - so an unchecked `is_file` call
    turns a client-supplied `event` this long into an unhandled 500 instead
    of the same 404 every other rejected `event` gets.
    """
    response = client.get("/api/queue", params={"event": "a" * 5000})
    assert response.status_code == 404
    assert set(response.json()) == {"error"}


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


def test_a_bad_roster_is_a_json_error_from_the_queue_too(client, tables):
    """The roster `/api/events` flagged answers here in the page's one shape.

    Starlette's own 500 is bare text, and the page reads every refusal as
    `{"error": ...}`; an event offered precisely so its broken roster can be
    found must not be the one that breaks that contract.
    """
    bad = tables / "broken_event"
    bad.mkdir()
    (bad / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n170815,gold,True,,,\n"
    )
    response = client.get("/api/queue", params={"event": "broken_event"})
    assert response.status_code == 500
    assert set(response.json()) == {"error"}
    assert "holdout" in response.json()["error"], "the reason is not reported"


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


def test_every_request_is_logged_by_path_and_never_by_token(app, caplog):
    """The request log is how a reviewer tells an unreachable server from a
    slow one. It carries the path and the status, and never the query
    string, because the token rides there."""
    import logging

    caplog.set_level(logging.INFO, logger="labeler.events.ui.app")
    with TestClient(app) as client:
        client.get(f"/api/queue?event=alfven_eigenmode&token={app.state.token}")
        client.get("/api/queue?event=alfven_eigenmode")
        TestClient(app).get("/api/events")
    lines = [r.getMessage() for r in caplog.records if " -> " in r.getMessage()]
    assert any(line.startswith("GET /api/queue -> 303") for line in lines)
    assert any(line.startswith("GET /api/queue -> 200") for line in lines)
    assert any(line.startswith("GET /api/events -> 401") for line in lines)
    assert all(app.state.token not in line for line in lines)
    assert all("token=" not in line and "event=" not in line for line in lines)


def test_the_page_and_its_two_files_are_served(client):
    """The reviewer opens `/` and gets the page, the script and the styles."""
    page = client.get("/")
    assert page.status_code == 200
    assert "/app.js" in page.text and "/style.css" in page.text
    for path in ("/app.js", "/style.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.content, path


def test_label_data_is_never_cached(client, paths, source):
    """Every answer carries label data or the page that shows it; none is held."""
    _built(paths)
    answers = [
        client.get(path)
        for path in (
            "/",
            "/api/events",
            "/api/queue?event=alfven_eigenmode",
            "/api/shot?event=alfven_eigenmode&shot=170815",
            "/api/rows?event=alfven_eigenmode&shot=170815&t0=0&t1=100",
        )
    ]
    answers.append(client.post("/api/label", json=_label()))
    for response in answers:
        assert response.status_code == 200, response.url
        assert response.headers["cache-control"] == "no-store", response.url


def test_the_page_is_behind_the_token_gate(app):
    """The static mount is inside the gate, not beside it."""
    transport = TestClient(app)
    for path in ("/", "/app.js", "/style.css"):
        response = transport.get(path)
        assert response.status_code == 401, path
        assert response.json() == {"error": NO_TOKEN}, path


# --- the launcher -----------------------------------------------------------


def _off_the_real_tree(monkeypatch, tmp_path):
    """Point `Paths.from_env` at `tmp_path` so `main()` cannot open the corpus."""
    paths = _tmp_paths(tmp_path, tmp_path / "events")
    monkeypatch.setenv("LABELER_ROOT", str(paths.root))
    monkeypatch.setenv("LABELER_CORPUS", str(paths.corpus))
    monkeypatch.setenv("LABELER_TEXT_ROOT", str(paths.text_root))
    monkeypatch.setenv("LABELER_LOGS_JSONL", str(paths.logs_jsonl))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(paths.label_tables))
    monkeypatch.setenv("LABELER_RAW_CACHE", str(paths.raw_cache))


def _runner(monkeypatch, serve):
    """Replace the only layer that would bind a socket, recording its call."""
    calls = []

    def fake(app, host, port):
        calls.append({"app": app, "host": host, "port": port})
        return 0

    monkeypatch.setattr(serve, "_run", fake)
    return calls


def test_serve_refuses_any_host_but_the_loopback():
    from labeler.events.ui import serve

    with pytest.raises(ValueError, match="127.0.0.1"):
        serve.main(host="0.0.0.0")


def test_serve_prints_the_token_link_and_the_forward(monkeypatch, capsys, tmp_path):
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "_run", lambda app, host, port: 0)
    serve.main(token="secret", port=9999)
    printed = capsys.readouterr().out
    assert "http://127.0.0.1:9999/?token=secret" in printed
    assert "ssh -L 9999:localhost:9999" in printed


def test_serve_binds_the_loopback_and_hands_over_the_gated_app(
    monkeypatch, capsys, tmp_path
):
    """The real `main` path: what the runner is given, value by value."""
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    calls = _runner(monkeypatch, serve)
    assert serve.main(token="secret") == 0
    capsys.readouterr()
    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == serve.DEFAULT_PORT == 8811
    assert calls[0]["app"].state.token == "secret"


def test_serve_never_puts_the_token_bearing_url_in_uvicorn_s_access_log(monkeypatch):
    """The token rides in the query string, so an access log would print the
    credential into the terminal and into anything capturing it.
    """
    import uvicorn

    from labeler.events.ui import serve

    seen = {}

    def fake_run(app, **kwargs):
        seen["app"] = app
        seen.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    sentinel = object()
    assert serve._run(sentinel, "127.0.0.1", 8811) == 0
    assert seen["app"] is sentinel
    assert seen["access_log"] is False
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8811


def test_serve_mints_an_unguessable_token_that_differs_between_runs(
    monkeypatch, capsys, tmp_path
):
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    calls = _runner(monkeypatch, serve)
    serve.main(port=9999)
    first = capsys.readouterr().out
    serve.main(port=9999)
    second = capsys.readouterr().out

    minted = [call["app"].state.token for call in calls]
    assert minted[0] != minted[1]
    # 16 bytes of hex: too wide to guess from the loopback, and every
    # character accounted for so a truncated or non-hex source shows up.
    for token in minted:
        assert len(token) == 32
        assert set(token) <= set("0123456789abcdef")
    for token, printed in zip(minted, (first, second)):
        assert f"http://127.0.0.1:9999/?token={token}" in printed

    # The printed token is the only one that opens the page.
    app = calls[0]["app"]
    refused = TestClient(app).get(f"/api/events?token={minted[1]}")
    assert refused.status_code == 401
    assert refused.json() == {"error": BAD_TOKEN}


def test_serve_takes_its_token_from_secrets(monkeypatch, capsys, tmp_path):
    """A token from `random` would read the same but be predictable."""
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    _runner(monkeypatch, serve)
    seen = []
    real = secrets.token_hex

    def spy(n=None):
        seen.append(n)
        return real(n)

    monkeypatch.setattr(secrets, "token_hex", spy)
    serve.main(port=9999)
    capsys.readouterr()
    assert seen == [16]


def test_serve_warns_when_the_process_is_not_under_the_fdp_wrapper(
    monkeypatch, capsys, tmp_path
):
    """A shot outside the corpus needs a live fetch, and a live fetch needs
    the wrapper; saying so at startup costs a restart, saying so at the first
    fetch costs the review up to that point.
    """
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    _runner(monkeypatch, serve)
    for marker in serve.FDP_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    serve.main(token="secret")
    assert serve.FDP_COMMAND in capsys.readouterr().out


def test_serve_stays_quiet_when_it_is_under_the_fdp_wrapper(
    monkeypatch, capsys, tmp_path
):
    """The markers are the ones `fdp run` really sets; a note on every single
    run is a note nobody reads.
    """
    from labeler.events.ui import serve

    _off_the_real_tree(monkeypatch, tmp_path)
    _runner(monkeypatch, serve)
    for marker in serve.FDP_MARKERS:
        monkeypatch.setenv(marker, "set-by-fdp-run")
    serve.main(token="secret")
    assert "fdp" not in capsys.readouterr().out
