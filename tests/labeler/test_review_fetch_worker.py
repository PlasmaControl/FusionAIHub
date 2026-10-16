"""The fetch worker (`review.fetch_worker`): the fdp token read and its state, error
classes, the record checks, the child's loop, and the parent's queue with a real
spawned child running a fake fetch (dedup, kinds, timeout and restart, a new token).
Nothing here reaches DIII-D, and every token is one the test builds."""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import time
import types
from multiprocessing import Pipe

import numpy as np
import pytest

from labeler.events.raw import NO_FETCH_ENV
from labeler.events.review import fetch_worker as fw
from labeler.events.review.fetch_worker import Fdp, FdpWatch, FetchError

NOW = 2_000_000_000
MARKERS = {marker: "x" for marker in fw.FDP_MARKERS}


def _jwt(exp) -> str:
    def part(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'exp': exp})}.c2ln"


def _home(tmp_path, token=None, cache=()):
    home = tmp_path / "home"
    (home / ".fdp" / "cache").mkdir(parents=True)
    if token is not None:
        (home / ".fdp" / "token").write_text(token + "\n")
    for name, text in cache:
        (home / ".fdp" / "cache" / name).write_text(text)
    return home


# ---- the token ----


def test_no_fdp_wrapper_is_unavailable_whatever_the_token(tmp_path):
    home = _home(tmp_path, _jwt(NOW + 3600))
    got = fw.read_fdp({}, home, NOW)
    assert (got.env, got.state) == (False, "unavailable")
    assert got.expires_at == "2033-05-18T04:33:20Z"


def test_a_valid_bearer_token_wins(tmp_path):
    given = _jwt(NOW + 600)
    home = _home(tmp_path, _jwt(NOW + 3600))
    got = fw.read_fdp({**MARKERS, fw.TOKEN_ENV: given}, home, NOW)
    assert (got.env, got.state, got.token) == (True, "ok", given)
    assert got.expires_at == "2033-05-18T03:43:20Z"


def test_an_expired_bearer_token_falls_to_the_device_cache(tmp_path):
    cached = _jwt(NOW + 900)
    home = _home(tmp_path, _jwt(NOW + 3600), cache=[("a.token", _jwt(NOW - 5)),
                                                     ("b.token", cached),
                                                     ("c.other", _jwt(NOW + 99))])
    got = fw.read_fdp({**MARKERS, fw.TOKEN_ENV: _jwt(NOW - 10)}, home, NOW)
    assert (got.state, got.token) == ("ok", cached)


def test_an_opaque_bearer_token_is_used_without_an_expiry(tmp_path):
    got = fw.read_fdp({**MARKERS, fw.TOKEN_ENV: "opaque"}, _home(tmp_path), NOW)
    assert (got.state, got.token, got.expires_at) == ("ok", "opaque", None)


def test_the_token_file_is_last(tmp_path):
    token = _jwt(NOW + 3600)
    got = fw.read_fdp(MARKERS, _home(tmp_path, token), NOW)
    assert (got.state, got.token) == ("ok", token)


@pytest.mark.parametrize("exp", [NOW - 1, NOW + fw.TOKEN_MARGIN_S - 1])
def test_a_token_at_or_near_its_expiry_is_expired(tmp_path, exp):
    got = fw.read_fdp(MARKERS, _home(tmp_path, _jwt(exp)), NOW)
    assert (got.state, got.token) == ("expired", None)
    assert got.expires_at is not None  # the latest expiry, for the page


def test_no_token_at_all_is_expired_under_fdp(tmp_path):
    got = fw.read_fdp(MARKERS, tmp_path / "nobody", NOW)
    assert (got.state, got.expires_at, got.token) == ("expired", None, None)


def test_the_token_never_shows(tmp_path):
    given = _jwt(NOW + 600)
    got = fw.read_fdp({**MARKERS, fw.TOKEN_ENV: given}, _home(tmp_path), NOW)
    assert given not in repr(got) and given not in json.dumps(got.public())
    assert set(got.public()) == {"env", "expires_at"}
    assert got.digest != Fdp(True, None, "ok", "other").digest
    assert len(got.digest) == 16 and given not in got.digest


@pytest.mark.parametrize("token", [None, "", "a.b", "a.!!!.c", "a.e30.c"])
def test_jwt_exp_of_a_malformed_token_is_none(token):
    assert fw.jwt_exp(token) is None


def test_the_watch_reuses_its_read():
    calls = []

    def reader():
        calls.append(1)
        return Fdp(True, None, "ok")

    watch = FdpWatch(reader, ttl_s=60)
    watch.get()
    watch.get()
    assert len(calls) == 1
    watch.clear()
    watch.get()
    assert len(calls) == 2
    fresh = FdpWatch(reader, ttl_s=0)
    fresh.get()
    fresh.get()
    assert len(calls) == 4


def test_refusal(monkeypatch):
    monkeypatch.delenv(NO_FETCH_ENV, raising=False)
    assert fw.refusal(Fdp(True, None, "ok")) is None
    for state, words in (("unavailable", "fdp wrapper"), ("expired", "fdp login")):
        error = fw.refusal(Fdp(state == "expired", None, state))
        assert error.kind == "auth" and words in error.message
    monkeypatch.setenv(NO_FETCH_ENV, "1")
    error = fw.refusal(Fdp(True, None, "ok"))
    assert error.kind == "auth" and NO_FETCH_ENV in error.message


# ---- errors and records ----


@pytest.mark.parametrize("text, kind", [
    ("%TREE-E-TreeNODATA, No data available for this node", "missing"),
    ("%TREE-W-TreeNNF, Node Not Found", "missing"),
    ("HTTP 401 from the gateway", "auth"),
    ("403 Forbidden", "auth"),
    ("Unauthorized", "auth"),
    ("the token has expired", "auth"),
    ("%TREE-E-FOPENR, Error opening file read-only", "upstream"),
    ("getservbyname failed for task 'PTSERVER'", "upstream"),
    ("shot 184013 is odd", "upstream"),
])
def test_classify(text, kind):
    assert fw.classify(RuntimeError(text))[0] == kind


def test_classify_scrubs_the_token_and_cuts_the_message():
    kind, message = fw.classify(ValueError("bad  \n token tok-123 " + "x" * 400),
                                token="tok-123")
    assert kind == "upstream" and "tok-123" not in message and "<token>" in message
    assert "\n" not in message and len(message) == 300


def test_an_unknown_kind_is_upstream():
    assert FetchError("weird", "m").kind == "upstream"
    assert [fw.STATUS[k] for k in fw.KINDS] == [422, 404, 503, 502, 504]


def _rec(data, **dims):
    units = {"data": "kA", **{name: unit for name, (_, unit) in dims.items()}}
    return {"data": data, **{name: values for name, (values, _) in dims.items()},
            "units": units}


def test_a_1d_trace_in_seconds_is_in_milliseconds():
    got = fw.as_trace(_rec(np.array([3.0, 1.0, 2.0, 4.0]),
                           times=(np.array([0.3, 0.1, np.nan, 0.4]), "s")), "X")
    assert got["t_ms"].tolist() == pytest.approx([100.0, 300.0, 400.0])
    assert got["y"].tolist() == [[1.0, 3.0, 4.0]] and got["y"].dtype == np.float32
    assert got["units"] == "kA"


def test_a_2d_record_puts_channels_first():
    t = np.arange(5.0)
    rows = np.arange(15.0).reshape(3, 5)
    by_row = fw.as_trace(_rec(rows, times=(t, "ms"), dim1=(np.arange(3.0), "")), "X")
    by_col = fw.as_trace(_rec(rows.T, times=(t, "ms"), dim1=(np.arange(3.0), "")), "X")
    assert by_row["y"].shape == by_col["y"].shape == (3, 5)
    assert np.array_equal(by_row["y"], by_col["y"])


@pytest.mark.parametrize("record, kind, words", [
    (_rec(np.ones((4, 4)), times=(np.arange(4.0), "ms")), "invalid", "ambiguous"),
    (_rec(np.ones(4), dim0=(np.arange(4.0), "m")), "invalid", "none of its"),
    (_rec(np.ones((2, 2, 2)), times=(np.arange(2.0), "s")), "invalid", "not a time"),
    (_rec(np.array([]), times=(np.array([]), "s")), "invalid", "not a time"),
    (_rec(np.ones((fw.MAX_CHANNELS + 1, 3)), times=(np.arange(3.0), "s")),
     "invalid", "channels"),
    (_rec(np.ones(2), times=(np.array([0.1, np.nan]), "s")), "missing", "two"),
])
def test_a_record_that_is_no_trace(record, kind, words):
    with pytest.raises(FetchError) as caught:
        fw.as_trace(record, "X")
    assert caught.value.kind == kind and words in caught.value.message


def test_a_free_2d_node_is_asked_again_for_both_dimensions(monkeypatch):
    calls = []
    t = np.linspace(0.0, 1.0, 6)

    def fake_mds(node, tree, shot, dims=None):
        calls.append(dims)
        if dims is None:  # its default dimension is the channel axis, not time
            return _rec(np.ones((3, 6)), dim0=(np.arange(3.0), ""))
        return _rec(np.ones((3, 6)), dim0=(np.arange(3.0), ""), dim1=(t, "s"))

    monkeypatch.setattr(fw, "_mds", fake_mds)
    got = fw._free_node("\\MHD::X", "MHD", 1)
    assert calls == [None, ("dim0", "dim1")] and got["y"].shape == (3, 6)
    calls.clear()
    monkeypatch.setattr(fw, "_mds", lambda *a, **k: (
        calls.append(1), _rec(np.ones(6), dim0=(t, "m")))[1])
    with pytest.raises(FetchError):
        fw._free_node("\\MHD::X", "MHD", 1)
    assert calls == [1]  # a 1-D record is not asked again


def test_ptdata_retries_once_on_the_server_lookup(monkeypatch):
    attempts, sleeps = [], []
    errors = ["getservbyname failed for task 'PTSERVER'"]

    class PtDataSignal:
        def __init__(self, name):
            self.name = name

        def fetch(self, shot):
            attempts.append(shot)
            if errors:
                raise RuntimeError(errors.pop(0))
            return {"data": [1.0]}

    monkeypatch.setitem(sys.modules, "toksearch_d3d",
                        types.SimpleNamespace(PtDataSignal=PtDataSignal))
    assert fw._ptdata("IP", 7, sleep=sleeps.append) == {"data": [1.0]}
    assert attempts == [7, 7] and sleeps == [fw.RETRY_DELAY_S]
    errors[:] = ["getservbyname", "getservbyname"]
    with pytest.raises(RuntimeError):
        fw._ptdata("IP", 7, sleep=sleeps.append)
    errors[:] = ["PTDATA error -1"]
    attempts.clear()
    with pytest.raises(RuntimeError):
        fw._ptdata("IP", 7, sleep=sleeps.append)
    assert attempts == [7]


def test_efit01_q_is_sorted_and_positive(monkeypatch):
    q = -np.vstack([np.full(4, 2.0), np.full(4, 1.0), np.full(4, 3.0)]).T  # (n_x, n_t)

    def fake_mds(node, tree, shot, dims=None):
        assert tree == "EFIT01"
        if node == fw.EFIT01_GTIME:
            return {"data": np.array([300.0, 100.0, 500.0])}
        return {"data": q}

    monkeypatch.setattr(fw, "_mds", fake_mds)
    got = fw._efit01_q(1)
    assert got["t_ms"].tolist() == [100.0, 300.0, 500.0]
    assert got["y"].shape == (3, 4) and got["y"][:, 0].tolist() == [1.0, 2.0, 3.0]
    assert got["x"][0] == 0.0 and got["x"][-1] == 1.0
    monkeypatch.setattr(fw, "_mds", lambda node, tree, shot, dims=None: {
        "data": np.ones(5) if node == fw.EFIT01_QPSI else np.arange(3.0)})
    with pytest.raises(FetchError) as caught:
        fw._efit01_q(1)
    assert caught.value.kind == "missing"


def test_run_refuses_an_unknown_fetch():
    with pytest.raises(FetchError) as caught:
        fw.run({"via": "shell"}, 1)
    assert caught.value.kind == "invalid"


# ---- the child's loop, in this process ----


def _work(request, shot):
    """The fake fetch the child runs: an echo, a sleep, a typed failure, a plain
    exception carrying the token, or a crash."""
    what = request["do"]
    if what == "echo":
        return {"shot": shot, "pid": os.getpid(),
                "token": os.environ.get(fw.TOKEN_ENV)}
    if what == "sleep":
        time.sleep(request["s"])
        return {"slept": request["s"]}
    if what == "fail":
        raise FetchError(request["kind"], "it failed")
    if what == "raise":
        raise RuntimeError(f"401 for token {os.environ.get(fw.TOKEN_ENV)}")
    if what == "crash":
        os._exit(3)
    raise AssertionError(what)


def _nothing():
    return None


def _no_toksearch():
    raise ImportError("No module named 'toksearch'")


def fake_serve(conn, token):
    fw.serve(conn, token, work=_work, prepare=_nothing)


def broken_serve(conn, token):
    fw.serve(conn, token, work=_work, prepare=_no_toksearch)


def test_serve_answers_each_request(monkeypatch):
    monkeypatch.setenv(fw.TOKEN_ENV, "before")  # serve sets it; the undo restores it
    parent, child = Pipe()
    for request in ({"do": "echo"}, {"do": "fail", "kind": "missing"},
                    {"do": "raise"}):
        parent.send((request, 9))
    parent.send(None)
    fw.serve(child, "tok-123", work=_work, prepare=_nothing)
    assert parent.recv() == (fw.READY, None)
    status, body = parent.recv()
    assert status == "ok" and body["shot"] == 9 and body["token"] == "tok-123"
    assert parent.recv() == ("error", ("missing", "it failed"))
    status, (kind, message) = parent.recv()
    assert kind == "auth" and "tok-123" not in message and "<token>" in message


def test_serve_says_when_toksearch_does_not_import():
    parent, child = Pipe()
    fw.serve(child, None, work=_work, prepare=_no_toksearch)
    status, problem = parent.recv()
    assert status == fw.READY and "does not import" in problem


# ---- the parent, with a real spawned child ----


def _wait(predicate, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting")


class Watcher:
    def __init__(self):
        self.fdp = Fdp(True, None, "ok", "tok-1")

    def get(self):
        return self.fdp


@pytest.fixture
def fetcher(monkeypatch):
    monkeypatch.delenv(NO_FETCH_ENV, raising=False)
    made = fw.Fetcher(target=fake_serve, watch=Watcher(), startup_s=60)
    yield made
    made.stop()


def _submit(fetcher, key, request, limit_s=30.0):
    got, done = [], threading.Event()

    def on_result(payload):
        got.append(payload)
        done.set()

    job = fetcher.submit(key, request, shot=key[0], ident=f"id{key[1]}",
                         where=f"node {key[1]}", limit_s=limit_s, on_result=on_result)
    return job, got, done


def _failed(fetcher, key):
    _wait(lambda: (fetcher.job(key) or fw.Job(*[None] * 7)).state == "failed")
    return fetcher.take_error(key)


def test_a_job_runs_once_however_often_it_is_asked(fetcher):
    job, got, done = _submit(fetcher, (5, "a"), {"do": "echo"})
    again, _, _ = _submit(fetcher, (5, "a"), {"do": "echo"})
    assert again is job
    assert done.wait(60) and got[0]["shot"] == 5 and got[0]["token"] == "tok-1"
    assert got[0]["pid"] != os.getpid()
    _wait(lambda: fetcher.job((5, "a")) is None)
    assert fetcher.take_error((5, "a")) is None and fetcher.started == 1
    _, got2, done2 = _submit(fetcher, (5, "b"), {"do": "echo"})
    assert done2.wait(60) and got2[0]["pid"] == got[0]["pid"]  # the same child


def test_the_queue_reports_what_runs_and_what_waits(fetcher):
    first, _, done = _submit(fetcher, (1, "slow"), {"do": "sleep", "s": 1.5})
    second, _, _ = _submit(fetcher, (2, "next"), {"do": "echo"})
    _wait(lambda: first.stage == "fetching node slow")  # after the child's start
    assert first.state == "fetching"
    summary = fetcher.summary()
    assert summary["queue"] == 1 and summary["running"]["id"] == "idslow"
    assert summary["running"]["shot"] == 1
    assert fetcher.position(first) == 0 and fetcher.position(second) == 1
    assert done.wait(60)


def test_a_failure_keeps_its_kind_and_is_told_once(fetcher):
    _submit(fetcher, (3, "m"), {"do": "fail", "kind": "missing"})
    assert _failed(fetcher, (3, "m")) == ("missing", "it failed")
    assert fetcher.take_error((3, "m")) is None and fetcher.job((3, "m")) is None
    _submit(fetcher, (3, "r"), {"do": "raise"})
    kind, message = _failed(fetcher, (3, "r"))
    assert kind == "auth" and "tok-1" not in message


def test_a_failed_job_is_forgotten_after_a_while(fetcher, monkeypatch):
    _submit(fetcher, (3, "m"), {"do": "fail", "kind": "upstream"})
    _wait(lambda: (fetcher.job((3, "m")) or fw.Job(*[None] * 7)).state == "failed")
    monkeypatch.setattr(fw, "ERROR_KEEP_S", -1.0)
    assert fetcher.job((3, "m")) is None


def test_a_job_past_its_limit_restarts_the_child(fetcher):
    _submit(fetcher, (4, "slow"), {"do": "sleep", "s": 30}, limit_s=1.0)
    kind, message = _failed(fetcher, (4, "slow"))
    assert kind == "timeout" and "restarted" in message
    _, _, done = _submit(fetcher, (4, "after"), {"do": "echo"})
    assert done.wait(60) and fetcher.started == 2


def test_a_crashed_child_is_upstream_and_replaced(fetcher):
    _submit(fetcher, (6, "crash"), {"do": "crash"})
    kind, _ = _failed(fetcher, (6, "crash"))
    assert kind == "upstream"
    _, _, done = _submit(fetcher, (6, "after"), {"do": "echo"})
    assert done.wait(60) and fetcher.started == 2


def test_a_new_token_starts_a_new_child(fetcher):
    _, got, done = _submit(fetcher, (7, "a"), {"do": "echo"})
    assert done.wait(60)
    fetcher._watch.fdp = Fdp(True, None, "ok", "tok-2")
    _, got2, done2 = _submit(fetcher, (7, "b"), {"do": "echo"})
    assert done2.wait(60) and got2[0]["token"] == "tok-2"
    assert got2[0]["pid"] != got[0]["pid"] and fetcher.started == 2


def test_no_fdp_fails_before_a_child_starts(fetcher):
    fetcher._watch.fdp = Fdp(True, None, "expired")
    _submit(fetcher, (8, "a"), {"do": "echo"})
    kind, message = _failed(fetcher, (8, "a"))
    assert kind == "auth" and "fdp login" in message and fetcher.started == 0


def test_a_child_without_toksearch_is_upstream(monkeypatch):
    monkeypatch.delenv(NO_FETCH_ENV, raising=False)
    fetcher = fw.Fetcher(target=broken_serve, watch=Watcher(), startup_s=60)
    try:
        _submit(fetcher, (9, "a"), {"do": "echo"})
        kind, message = _failed(fetcher, (9, "a"))
        assert kind == "upstream" and "does not import" in message
    finally:
        fetcher.stop()
