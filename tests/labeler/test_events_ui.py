"""The browser review surface: its gate, its reads, and what it refuses to write."""

from __future__ import annotations

import secrets

import numpy as np
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


def test_an_overlong_event_is_not_found_not_a_500(client):
    """`Path.is_file` re-raises ENAMETOOLONG rather than reporting False -

    it is not in pathlib's `_IGNORED_ERRNOS` - so an unchecked `is_file` call
    turns a client-supplied `event` this long into an unhandled 500 instead
    of the same 404 every other rejected `event` gets.
    """
    response = client.get("/api/shots", params={"event": "a" * 5000})
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


def test_a_bad_roster_is_a_json_error_from_shots_too(client, tables):
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
    response = client.get("/api/shots", params={"event": "broken_event"})
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


def test_the_static_mount_is_behind_the_gate(app):
    response = TestClient(app).get("/")
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}


def _co2_stub(seen, *, base_ms=2000.0, span_ms=200.0, rate_hz=1_000_000.0):
    """A stand-in for `raw_signal` that slices the way the real one does.

    It RECORDS its arguments, so a `t_range` the endpoint dropped or mangled
    on its way through is visible here rather than hidden behind a window
    that merely rendered something. The time base starts at a shot-like
    2000 ms on purpose: the corpus stores seconds and `raw_signal` returns
    milliseconds, and a base of 0 satisfies both, so it cannot fail.
    """
    from labeler.features.store import FeatureArray

    count = int(span_ms * rate_hz / 1000.0)
    x = base_ms + np.arange(count) / rate_hz * 1000.0
    y = np.random.default_rng(4).normal(size=(4, count)).astype("float32")

    def fake_raw_signal(shot, group, *, channels=None, t_range=None, paths=None):
        seen["shot"] = shot
        seen["group"] = group
        seen["t_range"] = t_range
        seen["paths"] = paths
        if t_range is None:
            return FeatureArray(x=x, y=y, attrs={})
        keep = (x >= t_range[0]) & (x <= t_range[1])
        return FeatureArray(x=x[keep], y=y[:, keep], attrs={})

    return fake_raw_signal


@pytest.fixture
def co2(monkeypatch):
    """The only layer stubbed: the one that reads bytes off disk."""
    from labeler.events.panels import alfven_eigenmode as ae

    seen = {}
    monkeypatch.setattr(ae, "raw_signal", _co2_stub(seen))
    return seen


def test_panels_honours_the_window_it_is_given(client, co2):
    response = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=2100&t1=2150"
    )
    assert response.status_code == 200
    payload = response.json()
    # Milliseconds all the way across the wire, floats on both sides.
    assert co2["t_range"] == (2100.0, 2150.0)
    assert co2["shot"] == 178642
    assert co2["group"] == "co2"
    assert payload["t_range"] == [2100.0, 2150.0]
    assert payload["event"] == "alfven_eigenmode"
    assert payload["shot"] == 178642
    assert len(payload["panels"]) == 3
    first = payload["panels"][0]
    assert first["kind"] == "heatmap"
    assert first["ylabel"] == "kHz"
    assert first["title"] == "CO2 crosspower DENR0UF x DENV1UF"
    assert first["bands"] == [[80.0, 250.0]]
    assert first["hlines"] == []
    assert len(first["z"]) == len(first["y"])
    assert len(first["z"][0]) == len(first["x"])
    assert 2100.0 <= min(first["x"]) and max(first["x"]) <= 2150.0


def test_panels_without_a_window_asks_for_the_whole_shot(client, co2):
    payload = client.get("/api/panels?event=alfven_eigenmode&shot=178642").json()
    assert co2["t_range"] is None
    assert payload["t_range"] is None
    assert min(payload["panels"][0]["x"]) >= 2000.0


def test_panels_is_given_the_apps_own_paths_not_the_environments(client, co2, tables):
    """A `Paths` that fell back to `from_env` would read the real corpus."""
    client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert co2["paths"] is not None
    assert co2["paths"].label_tables == tables


def test_panels_reports_a_missing_signal_as_a_message_not_a_500(client, monkeypatch):
    """A shot no tier has and no fetch route can reach is an ordinary outcome.

    404, not 502: nothing upstream failed, the shot is simply not here.
    """
    from labeler.events import raw
    from labeler.events.panels import alfven_eigenmode as ae

    def absent(*args, **kwargs):
        raise raw.NoDataError("shot 999999 has no 'co2' in the corpus or the cache")

    monkeypatch.setattr(ae, "raw_signal", absent)
    response = client.get("/api/panels?event=alfven_eigenmode&shot=999999")
    assert response.status_code == 404
    assert "999999" in response.json()["error"]


def test_an_upstream_fetch_failure_is_the_only_bad_gateway(client, monkeypatch):
    from labeler.events import raw
    from labeler.events.panels import alfven_eigenmode as ae

    def unreachable(*args, **kwargs):
        raise raw.UpstreamError("PTDATA: getservbyname failed for task PTSERVER")

    monkeypatch.setattr(ae, "raw_signal", unreachable)
    response = client.get("/api/panels?event=alfven_eigenmode&shot=999999")
    assert response.status_code == 502
    assert "PTSERVER" in response.json()["error"]


def test_unreadable_bytes_do_not_put_a_server_path_on_the_page(client, monkeypatch):
    """The reviewer gets a sentence; the errno and the absolute path get logged."""
    from labeler.events.panels import alfven_eigenmode as ae

    def broken(*args, **kwargs):
        raise OSError(
            2,
            "Unable to synchronously open file (unable to open file: "
            "name = '/scratch/gpfs/EKOLEMEN/foundation_model/178642_features.h5')",
        )

    monkeypatch.setattr(ae, "raw_signal", broken)
    response = client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert response.status_code == 502
    message = response.json()["error"]
    assert "178642" in message
    assert "/scratch" not in message and "Errno" not in message


def test_a_builder_bug_is_not_reported_as_a_bad_window(client, monkeypatch):
    """A genuine regression must not read to the reviewer as bad input.

    It is a 500 - the decision that a builder bug is a server error rather
    than a dressed-up 400 is correct and stays - but the body must still be
    this module's one {"error": ...} shape, and must not carry the raw
    exception text, which is logged instead.
    """
    from labeler.events.panels import alfven_eigenmode as ae

    def buggy(*args, **kwargs):
        raise ValueError("operands could not be broadcast together")

    monkeypatch.setattr(ae, "raw_signal", buggy)
    response = client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert response.status_code == 500
    body = response.json()
    assert set(body) == {"error"}
    assert "operands could not be broadcast together" not in body["error"]


def test_a_bad_window_says_which_window(client, co2):
    """With a window, a ValueError IS about the window - and names it."""
    response = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=2100&t1=2100.0001"
    )
    assert response.status_code == 400
    assert "2100.0" in response.json()["error"]


def test_a_corrupt_label_grid_is_not_the_same_as_a_missing_one(client, co2, tables):
    grid = tables / "alfven_eigenmode" / "format" / "shots" / "178642.npz"
    grid.parent.mkdir(parents=True, exist_ok=True)
    grid.write_bytes(b"PK\x03\x04 not really a zip")
    payload = client.get("/api/panels?event=alfven_eigenmode&shot=178642").json()
    assert payload["note"] == "LABEL ROW UNREADABLE"
    assert len(payload["panels"]) == 3, "the diagnostic panels survived"


def test_a_non_finite_band_edge_still_leaves_valid_json(client, monkeypatch):
    """`bands`/`hlines`/`zmin`/`zmax` are floats too, and NaN is not JSON."""
    from labeler.events import panels as registry
    from labeler.events.verify import Panel

    def one_panel(event, shot, *, t_range=None, paths=None):
        return [
            Panel(
                title="t",
                kind="line",
                x=np.array([0.0, 1.0]),
                y=np.array([[0.0, 1.0]]),
                ylabel="",
                bands=[(np.nan, 250.0)],
                hlines=[np.inf],
                zmin=np.nan,
            )
        ]

    monkeypatch.setattr(registry, "build", one_panel)
    response = client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert response.status_code == 200
    assert "NaN" not in response.text and "Infinity" not in response.text
    panel = response.json()["panels"][0]
    assert panel["bands"] == [[None, 250.0]]
    assert panel["hlines"] == [None]
    assert panel["zmin"] is None


def test_panels_refuses_a_window_the_shot_has_no_samples_in(client, co2):
    """The builder raises on a degenerate window; that is the user's input."""
    response = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=9000&t1=9100"
    )
    assert response.status_code == 400
    assert "0 sample(s)" in response.json()["error"]


def test_panels_refuses_a_malformed_window(client, co2):
    response = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=early&t1=2150"
    )
    assert response.status_code == 422
    assert "t0" in response.json()["error"]
    assert "t_range" not in co2, "the builder ran on an unparsed window"


def test_panels_says_when_there_is_no_label_row(client, co2):
    payload = client.get("/api/panels?event=alfven_eigenmode&shot=178642").json()
    assert payload["note"] == "NO LABEL ROW"
    assert len(payload["panels"]) == 3


def test_panels_appends_the_label_row_when_there_is_one(client, co2, tables):
    from labeler.events.interval_tables import write_label_grid

    time_ms = np.arange(2000.0, 2500.0, 50.0)
    labels = np.zeros((len(time_ms), 20))
    labels[2:4, :] = 1.0
    labels[0, :] = np.nan
    write_label_grid(
        tables / "alfven_eigenmode" / "format" / "shots" / "178642.npz",
        time_ms,
        labels,
        categories={"0": "none", "1": "ae"},
    )
    payload = client.get("/api/panels?event=alfven_eigenmode&shot=178642").json()
    assert payload["note"] == ""
    assert len(payload["panels"]) == 4
    row = payload["panels"][-1]
    assert row["title"] == "labels (format/shots)"
    assert row["kind"] == "heatmap"
    assert row["zmin"] == 0.0 and row["zmax"] == 1.0
    # Cell EDGES: one more coordinate on each axis than the grid has cells.
    assert len(row["z"]) == len(row["y"]) - 1
    assert len(row["z"][0]) == len(row["x"]) - 1
    assert row["x"][0] == 2000.0 and row["x"][-1] == 2500.0
    # NaN is not JSON. An unknown cell must arrive as null, which plotly
    # draws as a gap; a bare NaN token makes `JSON.parse` throw on the lot.
    assert row["z"][0][0] is None
    assert (
        "NaN" not in client.get("/api/panels?event=alfven_eigenmode&shot=178642").text
    )


def test_panels_for_an_unknown_event_is_not_found(client):
    response = client.get("/api/panels?event=no_such_event&shot=178642")
    assert response.status_code == 404
    assert response.json() == {"error": "unknown event 'no_such_event'"}


def test_a_traversing_event_has_no_panels(client, foreign):
    response = client.get("/api/panels", params={"event": "../secret_area", "shot": 1})
    assert response.status_code == 404
    assert "999999" not in response.text


def test_an_overlong_event_has_no_panels_not_a_500(client):
    """Same ENAMETOOLONG hazard as `/api/shots`, on the other endpoint."""
    response = client.get("/api/panels", params={"event": "a" * 5000, "shot": 1})
    assert response.status_code == 404
    assert set(response.json()) == {"error"}


def test_an_absolute_event_has_no_panels(client, foreign):
    response = client.get("/api/panels", params={"event": str(foreign), "shot": 1})
    assert response.status_code == 404
    assert "999999" not in response.text


def test_drawing_panels_never_writes_the_roster(client, co2, tables):
    roster = tables / "alfven_eigenmode" / "shots.csv"
    before = (roster.read_bytes(), roster.stat().st_mtime_ns)
    client.get("/api/panels?event=alfven_eigenmode&shot=178642&t0=2100&t1=2150")
    assert (roster.read_bytes(), roster.stat().st_mtime_ns) == before


def test_a_generic_events_panels_come_back_as_lines_in_milliseconds(
    client, monkeypatch
):
    """`_generic` reads seconds off disk; the wire is milliseconds either way."""
    from labeler.events.panels import _generic
    from labeler.features.store import FeatureArray

    seen = []

    def fake_read_feature(path, name):
        seen.append(name)
        seconds = np.arange(2.0, 2.2, 0.001)
        return FeatureArray(x=seconds, y=np.ones((1, len(seconds))) * 7.0, attrs={})

    monkeypatch.setattr(_generic, "read_feature", fake_read_feature)
    payload = client.get(
        "/api/panels?event=detachment&shot=178642&t0=2100&t1=2150"
    ).json()
    assert seen == ["ip", "betan", "pinj_total"]
    assert [panel["title"] for panel in payload["panels"]] == seen
    first = payload["panels"][0]
    assert first["kind"] == "line"
    assert first["legend"] == ["ch 0"]
    assert first["y"] == [[7.0] * len(first["x"])]
    assert "z" not in first
    # Seconds on disk would have put this window at 2.1-2.15, i.e. empty.
    # The sample spacing is 1 ms, so the window's edges land within one of it.
    assert first["x"][0] == pytest.approx(2101.0, abs=1.0)
    assert first["x"][-1] == pytest.approx(2150.0, abs=1.0)
    assert len(first["x"]) == pytest.approx(50, abs=1)


def _write_corpus_co2(paths, shot, *, base_ms=2000.0, span_ms=200.0, rate_hz=1e6):
    """A real corpus-layout file, so `raw_signal` runs for real in the test."""
    from labeler.events import raw

    count = int(span_ms * rate_hz / 1000.0)
    times_ms = base_ms + np.arange(count) / rate_hz * 1000.0
    values = np.random.default_rng(7).normal(size=(4, count)).astype("float32")
    raw.write_group(paths.corpus / f"{shot}_processed.h5", "co2", times_ms, values)


@pytest.fixture
def never_fetch(monkeypatch):
    """Records every live fetch. The reviewer's scroll wheel must never fire one.

    A live PTDATA fetch moves ~240 MB and takes minutes, so it is a thing the
    server may do on a deliberate first open of a shot and never as a side
    effect of panning past the end of one.
    """
    from labeler.events import raw

    calls = []

    def recording(shot, exprs, **kwargs):
        calls.append(shot)
        raise AssertionError("a live fetch ran")

    monkeypatch.setattr(raw, "fdp_signal", recording)
    return calls


def test_a_window_past_the_end_of_a_shot_does_not_fetch(client, paths, never_fetch):
    _write_corpus_co2(paths, 178642)
    response = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=9000&t1=9100"
    )
    assert never_fetch == [], "scrolling past the end of a shot fetched it live"
    assert response.status_code == 400
    assert "9000" in response.json()["error"]


def test_a_backwards_window_is_refused_before_anything_is_read(client, never_fetch):
    response = client.get(
        "/api/panels?event=alfven_eigenmode&shot=178642&t0=2150&t1=2100"
    )
    assert response.status_code == 400
    assert never_fetch == []


def test_the_whole_shot_render_stays_parseable(client, monkeypatch):
    """The default open of every shot is the widest render the server can make."""
    from labeler.events.panels import alfven_eigenmode as ae

    monkeypatch.setattr(ae, "raw_signal", _co2_stub({}, span_ms=1000.0))
    response = client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert response.status_code == 200
    size = len(response.content)
    cells = sum(
        len(panel["z"]) * len(panel["z"][0]) for panel in response.json()["panels"]
    )
    assert cells <= 500_000, f"{cells} heatmap cells in one whole-shot render"
    # The real render is ~2.4 MB; 4,000,000 is 1.7x headroom. 6,000,000 left
    # only a 0.4% margin against the DECIMALS=None mutation (6,025,699
    # bytes), thin enough that a numpy or json repr change could flip this
    # green while that regression was back.
    assert size <= 4_000_000, f"{size} bytes for one whole-shot render"


# --- /api/save: the one endpoint in this app that writes anything ---------


def _body(**overrides):
    """The payload the page posts, with a shot-like millisecond window."""
    body = {
        "event": "alfven_eigenmode",
        "shot": 178642,
        "reviewer": "nc1514",
        "marks": [{"t_start": 2100.0, "t_end": 2150.0, "category": 1}],
    }
    body.update(overrides)
    return body


@pytest.fixture
def no_roster_write(monkeypatch):
    """Records every `record_review` call - i.e. every write of shots.csv.

    `ReviewSession.save()` calls it, so a `/api/save` that went through the
    session instead of `write_corrections` shows up here by name rather than
    only as a byte difference in one other test.
    """
    from labeler.events import rosters

    calls = []
    real = rosters.record_review

    def recording(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(rosters, "record_review", recording)
    return calls


def test_save_writes_one_corrections_file(client, tables):
    response = client.post("/api/save", json=_body())
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"written", "n"}
    assert payload["n"] == 1
    written = tables / "alfven_eigenmode" / "review" / payload["written"]
    assert written.is_file()
    assert payload["written"].startswith("178642__nc1514__")
    assert payload["written"].endswith(".csv")
    assert "178642,1,2100" in written.read_text().replace(".0", "")


def test_a_saved_correction_round_trips_in_milliseconds(client, tables):
    """The wire is milliseconds, and so is the file. No 1000x anywhere."""
    from labeler.events.verify import corrections_for, read_corrections

    assert client.post("/api/save", json=_body()).status_code == 200
    saved = corrections_for("alfven_eigenmode", 178642, root=tables)
    assert len(saved) == 1
    frame = read_corrections(saved[0])
    assert list(frame.columns) == ["shot", "category", "t_start", "t_end", "confidence"]
    assert len(frame) == 1
    row = frame.iloc[0]
    assert int(row.shot) == 178642
    assert int(row.category) == 1
    assert float(row.t_start) == 2100.0
    assert float(row.t_end) == 2150.0


def test_every_mark_in_one_post_is_written(client, tables):
    from labeler.events.verify import corrections_for, read_corrections

    payload = client.post(
        "/api/save",
        json=_body(
            marks=[
                {"t_start": 2100.0, "t_end": 2150.0, "category": 1},
                {"t_start": 2300.5, "t_end": 2400.5, "category": 2},
            ]
        ),
    ).json()
    assert payload["n"] == 2
    frame = read_corrections(
        corrections_for("alfven_eigenmode", 178642, root=tables)[0]
    )
    assert list(frame.t_start) == [2100.0, 2300.5]
    assert list(frame.t_end) == [2150.0, 2400.5]
    assert list(frame.category) == [1, 2]


def test_a_second_save_writes_a_second_file_and_destroys_nothing(client, tables):
    first = client.post("/api/save", json=_body()).json()["written"]
    second = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": 2400.0, "t_end": 2900.0, "category": 0}]),
    ).json()["written"]
    assert first != second
    review = tables / "alfven_eigenmode" / "review"
    assert len(list(review.glob("178642__*.csv"))) == 2
    assert "2100" in (review / first).read_text(), "the first save was rewritten"


def test_save_does_not_touch_shots_csv(client, tables, no_roster_write):
    """The guarantee the reviewer is given, asserted on the bytes.

    An accidental `ReviewSession.save()` here would call `record_review` and
    break this without any other test noticing.
    """
    roster = tables / "alfven_eigenmode" / "shots.csv"
    before = (roster.read_bytes(), roster.stat().st_mtime_ns)
    assert client.post("/api/save", json=_body()).status_code == 200
    assert (roster.read_bytes(), roster.stat().st_mtime_ns) == before
    assert no_roster_write == [], "a save edited the roster"


def test_save_with_no_marks_is_refused(client, tables):
    response = client.post("/api/save", json=_body(marks=[]))
    assert response.status_code == 400
    assert "no marks" in response.json()["error"]
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_an_interval_that_runs_backwards(client, tables):
    response = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": 2150.0, "t_end": 2100.0, "category": 1}]),
    )
    assert response.status_code == 400
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_zero_length_interval(client, tables):
    """A drag that never moved marks nothing; `/api/panels` refuses it too."""
    response = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": 2100.0, "t_end": 2100.0, "category": 1}]),
    )
    assert response.status_code == 400
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_time_off_any_shot(client, tables):
    """A stray 1e30 is a bug upstream, not a range anybody dragged."""
    response = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": 2100.0, "t_end": 1e30, "category": 1}]),
    )
    assert response.status_code == 400
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_non_finite_time(client, tables):
    """`json.loads` takes the bare `NaN` and `Infinity` tokens as floats."""
    import json as jsonlib

    body = jsonlib.dumps(_body()).replace("2150.0", "NaN")
    assert "NaN" in body
    response = client.post(
        "/api/save", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    # Named, not just refused: without this the NaN falls through to
    # `validate_intervals`, which also refuses it, and the check here could
    # be deleted with nothing going red.
    assert "finite" in response.json()["error"]
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_non_numeric_time(client, tables):
    response = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": "early", "t_end": 2150.0, "category": 1}]),
    )
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_mark_missing_a_field(client, tables):
    response = client.post(
        "/api/save", json=_body(marks=[{"t_start": 2100.0, "category": 1}])
    )
    assert response.status_code == 422
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_fractional_category(client, tables):
    """`int()` would truncate 1.5 to 1 and write a label nobody chose."""
    response = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": 2100.0, "t_end": 2150.0, "category": 1.5}]),
    )
    assert response.status_code == 422
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_negative_category(client, tables):
    """The interval schema owns this rule; the endpoint must not swallow it."""
    response = client.post(
        "/api/save",
        json=_body(marks=[{"t_start": 2100.0, "t_end": 2150.0, "category": -1}]),
    )
    assert response.status_code == 400
    assert "nonnegative" in response.json()["error"]
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_body_that_is_missing_fields(client, tables):
    for body in ({}, {"event": "alfven_eigenmode"}, _body(shot=None)):
        response = client.post("/api/save", json=body)
        assert response.status_code == 422, body
        assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_field_of_the_wrong_type(client, tables):
    for body in (
        _body(shot="178642"),
        _body(marks={"t_start": 2100.0}),
        _body(marks=[[2100.0, 2150.0, 1]]),
        _body(event=178642),
    ):
        response = client.post("/api/save", json=body)
        assert response.status_code == 422, body
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_body_that_is_not_an_object(client, tables):
    """A bare list or string has no `.get`, so this is a 500 if unchecked."""
    for body in ([_body()], "alfven_eigenmode", 7):
        response = client.post("/api/save", json=body)
        assert response.status_code == 422, body
        assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_a_body_that_is_not_json(client, tables):
    response = client.post(
        "/api/save", content=b"not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_a_deeply_nested_body_reads_as_malformed_json_not_a_bare_500(client, tables):
    """`json.loads` raises `RecursionError`, not `ValueError`, past ~1000 levels.

    Unhandled, that falls through this endpoint's own `except ValueError`
    and Starlette answers with its plain-text 500, which breaks the one
    `{"error": ...}` shape every other refusal on this endpoint reads.
    """
    body = b"[" * 60_000
    response = client.post(
        "/api/save", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_save_refuses_an_absurd_number_of_marks(client, tables):
    mark = {"t_start": 2100.0, "t_end": 2150.0, "category": 1}
    response = client.post("/api/save", json=_body(marks=[mark] * 5000))
    assert response.status_code == 400
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_the_mark_cap_boundary_is_unmoved_by_checking_it_early(client, tables):
    """The cap now guards the parse loop instead of following it - same edge."""
    from labeler.events.ui.app import MAX_MARKS

    mark = {"t_start": 2100.0, "t_end": 2150.0, "category": 1}

    at_cap = client.post("/api/save", json=_body(marks=[mark] * MAX_MARKS))
    assert at_cap.status_code == 200
    assert at_cap.json()["n"] == MAX_MARKS

    over_cap = client.post("/api/save", json=_body(marks=[mark] * (MAX_MARKS + 1)))
    assert over_cap.status_code == 400
    assert f"{MAX_MARKS + 1} marks in one save" in over_cap.json()["error"]
    assert f"{MAX_MARKS} a review can hold" in over_cap.json()["error"]


def test_a_traversing_event_writes_nothing(client, foreign):
    """The read endpoints leaked a file; here the same hole is a WRITE."""
    response = client.post("/api/save", json=_body(event="../secret_area"))
    assert response.status_code == 404
    assert not (foreign / "review").exists(), "a correction was written outside"


def test_an_absolute_event_writes_nothing(client, foreign):
    response = client.post("/api/save", json=_body(event=str(foreign)))
    assert response.status_code == 404
    assert not (foreign / "review").exists(), "a correction was written outside"


def test_saving_to_an_unknown_event_is_not_found(client, tables):
    response = client.post("/api/save", json=_body(event="no_such_event"))
    assert response.status_code == 404
    assert not (tables / "no_such_event").exists(), "a directory was created"


def test_saving_to_an_overlong_event_is_not_found_not_a_500(client):
    response = client.post("/api/save", json=_body(event="a" * 5000))
    assert response.status_code == 404
    assert set(response.json()) == {"error"}


def test_a_reviewer_id_cannot_escape_the_review_directory(client, tables, tmp_path):
    """`reviewer` reaches a filename; `correction_path` sanitises it.

    An unsanitised `../../evil` is embedded inside one filename component
    (`<shot>__<reviewer>__<stamp>.csv`), so an escape it somehow achieved
    would land under the EVENT directory, not at `tmp_path` - `review/..`
    only reaches `alfven_eigenmode/`, two levels short of `tmp_path` itself.
    """
    payload = client.post("/api/save", json=_body(reviewer="../../evil")).json()
    written = tables / "alfven_eigenmode" / "review" / payload["written"]
    assert written.is_file()
    assert "/" not in payload["written"] and ".." not in payload["written"]
    assert not (tables / "alfven_eigenmode" / "evil").exists()


def test_a_reviewer_id_with_nothing_usable_is_refused(client, tables):
    response = client.post("/api/save", json=_body(reviewer="///"))
    assert response.status_code == 400
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_a_non_string_reviewer_is_refused_like_every_other_typed_field(client, tables):
    """`reviewer` was the one posted field with no type check.

    Unchecked, `178642` and `["a"]` both survive `str()` into a filename
    (`178642__178642__...csv`) - harmless after sanitisation, but out of
    step with the strict typing every other field on this endpoint gets.
    """
    for reviewer in (178642, ["a"], 1.5, False):
        response = client.post("/api/save", json=_body(reviewer=reviewer))
        assert response.status_code == 422, reviewer
        assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_an_overlong_reviewer_is_a_400_not_a_500(client, tables):
    """A 5000-character id survives sanitisation and then raises `ENAMETOOLONG`
    on `path.exists()`, which is a client-caused refusal, not a server fault -
    the same reasoning commit 70b561f applied to an over-long `event`.
    """
    response = client.post("/api/save", json=_body(reviewer="r" * 5000))
    assert response.status_code == 400
    assert set(response.json()) == {"error"}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_a_refused_save_leaves_an_existing_review_directory_exactly_as_it_was(
    client, tables
):
    review = tables / "alfven_eigenmode" / "review"
    review.mkdir()
    existing = review / "178642__alice__20260101T000000Z.csv"
    existing.write_text("shot,category,t_start,t_end,confidence\n178642,1,10,20,\n")
    before = (existing.read_bytes(), existing.stat().st_mtime_ns)
    assert client.post("/api/save", json=_body(marks=[])).status_code == 400
    assert client.post("/api/save", json=_body(shot="nope")).status_code == 422
    assert client.post("/api/save", json=_body(event="../secret")).status_code == 404
    assert [p.name for p in review.iterdir()] == [existing.name]
    assert (existing.read_bytes(), existing.stat().st_mtime_ns) == before


def test_an_unwritable_disk_does_not_put_a_server_path_on_the_page(
    client, tables, monkeypatch
):
    """Mirrors `test_unreadable_bytes_do_not_put_a_server_path_on_the_page`
    for `/api/panels`: the errno and the absolute path get logged, and the
    reviewer's page gets a sentence.
    """
    from labeler.events.ui import app as app_module

    def broken(frame, path):
        raise OSError(28, "No space left on device: /scratch/gpfs/EKOLEMEN/foo")

    monkeypatch.setattr(app_module, "write_corrections", broken)
    response = client.post("/api/save", json=_body())
    assert response.status_code == 500
    message = response.json()["error"]
    assert "/scratch" not in message and "Errno" not in message
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_a_save_shows_up_in_the_shot_list(client):
    """The count `/api/shots` reports is files on disk, so a save moves it."""
    before = client.get("/api/shots?event=alfven_eigenmode").json()["shots"]
    assert {row["shot"]: row["n_corrections"] for row in before}[178642] == 0
    client.post("/api/save", json=_body())
    after = client.get("/api/shots?event=alfven_eigenmode").json()["shots"]
    assert {row["shot"]: row["n_corrections"] for row in after}[178642] == 1


def test_save_is_behind_the_token_gate(app, tables):
    response = TestClient(app).post("/api/save", json=_body())
    assert response.status_code == 401
    assert response.json() == {"error": NO_TOKEN}
    assert not (tables / "alfven_eigenmode" / "review").exists()


def test_the_page_and_its_two_files_are_served(client):
    """The reviewer opens `/` and gets the page, the script and the styles."""
    page = client.get("/")
    assert page.status_code == 200
    assert "/app.js" in page.text and "/style.css" in page.text
    for path in ("/app.js", "/style.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.content, path


def test_plotly_is_served_off_disk(client):
    """No CDN link anywhere: this runs on a node with no route off the
    cluster, reached through an SSH forward, so the bundle comes from the
    installed plotly package instead.
    """
    assert "//cdn" not in client.get("/").text
    response = client.get("/vendor/plotly.min.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/javascript")
    # The real bundle, not a stub: megabytes of it.
    assert len(response.content) > 1_000_000
    # The one response allowed to be held; it carries nothing about any shot
    # and re-sending 4.8 MB on every reload is a second of a reviewer's time.
    assert response.headers["cache-control"] == "max-age=86400"


def test_label_data_is_still_never_cached(client):
    """`setdefault` on the gate's `Cache-Control` must not have let an answer
    built from `label_tables` become cacheable.
    """
    for path in ("/", "/api/events", "/api/shots?event=alfven_eigenmode"):
        assert client.get(path).headers["cache-control"] == "no-store", path


def test_the_panels_and_the_save_are_never_cached_either(client, co2, tables):
    """The two responses carrying the most label data of any on this app."""
    panels = client.get("/api/panels?event=alfven_eigenmode&shot=178642")
    assert panels.status_code == 200
    assert panels.headers["cache-control"] == "no-store"
    saved = client.post("/api/save", json=_body())
    assert saved.status_code == 200
    assert saved.headers["cache-control"] == "no-store"


def test_the_page_and_the_bundle_are_behind_the_token_gate(app):
    """The static mount is inside the gate, not beside it."""
    transport = TestClient(app)
    for path in ("/", "/app.js", "/style.css", "/vendor/plotly.min.js"):
        response = transport.get(path)
        assert response.status_code == 401, path
        assert response.json() == {"error": NO_TOKEN}, path
