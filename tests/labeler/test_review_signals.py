"""On-demand signals (`review.signals`) with a fake fetch worker: the trace wire format,
local sources and their fall-through, the pyramid, the 202 flow and the cache, the
negative cache, every error kind, and the LRU."""

from __future__ import annotations

import os
import time

import h5py
import numpy as np
import pytest

from labeler.config import Paths
from labeler.events.review import fetch_worker, signals
from labeler.events.review.fetch_worker import Fdp, FdpWatch, FetchError, Job
from labeler.events.review.signals import Cache, Pending, Signals, Trace

SHOT = 180001
CATALOG = r"""
signals:
  - {id: fast, label: Fast, group: T, units: keV, kind: trace, legend: [a, b, c],
     sources: [{corpus: ece, rows: "0-2"}]}
  - {id: summed, label: Summed, group: T, units: MW, kind: trace,
     sources: [{corpus: pinj, rows: "0-1", total: true, scale: 1.0e-6}]}
  - {id: slow, label: Slow, group: T, units: "", kind: trace,
     sources: [{corpus: slow, rows: 0}]}
  - {id: nan_then_point, label: Point, group: T, units: V, kind: trace,
     sources: [{corpus: sxr, rows: 0}, {ptdata: PT1}]}
  - {id: two_nodes, label: Two, group: T, units: "", kind: trace,
     sources: [{mds: '\MHD::A', tree: MHD}, {ptdata: PT2}]}
  - {id: absent, label: Absent, group: T, units: "", kind: trace,
     sources: [{corpus: absent, rows: 0}]}
  - {id: feat, label: Ip, group: T, units: MA, kind: trace,
     sources: [{feature: ip, scale: 1.0e-6}]}
  - {id: q95, label: q95, group: T, units: "", kind: trace, sources: [{park: a_q95}]}
  - {id: feat_corpus, label: Slow feature, group: T, units: "", kind: trace,
     sources: [{feature: slow}]}
  - {id: feat_sentinel, label: Absent feature, group: T, units: "", kind: trace,
     sources: [{feature: absent}]}
  - {id: qprof, label: q, group: P, units: "", kind: profile, axis: psi_n,
     sources: [{recipe: park_q}]}
"""
FAST_N = 200_000  # 2 s at 100 kHz
SPIKE = 123_457


class FakeFetcher:
    """The fetcher's interface without a process: the test finishes or fails jobs."""

    def __init__(self):
        self.jobs: dict = {}
        self.errors: dict = {}
        self.submitted: list = []

    def submit(self, key, request, *, shot, ident, where, limit_s, on_result):
        if key in self.jobs:
            return self.jobs[key]
        job = Job(key, request, shot, ident, where, limit_s, on_result)
        self.jobs[key] = job
        self.submitted.append(job)
        return job

    def job(self, key):
        return self.jobs.get(key)

    def take_error(self, key):
        return self.errors.pop(key, None)

    def position(self, job):
        return list(self.jobs.values()).index(job) + 1

    def summary(self):
        return {"queue": len(self.jobs), "running": None}

    def finish(self, job, payload):
        del self.jobs[job.key]
        job.on_result(payload)

    def fail(self, job, kind, message="it failed"):
        del self.jobs[job.key]
        self.errors[job.key] = (kind, message)


def _watch(state="ok"):
    return FdpWatch(reader=lambda: Fdp(env=state != "unavailable", expires_at=None,
                                       state=state, token=None))


def _group(f, name, x, y):
    g = f.create_group(name)
    g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))
    g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))


@pytest.fixture
def paths(tmp_path):
    p = Paths(root=tmp_path / "root", corpus=tmp_path / "corpus",
              text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
              label_tables=tmp_path / "events", config_dir=tmp_path / "config",
              raw_cache=tmp_path / "raw")
    p.config_dir.mkdir()
    (p.config_dir / "signals.yaml").write_text(CATALOG)
    p.corpus.mkdir()
    x = np.arange(FAST_N) * 1e-5
    fast = np.stack([np.sin(x * 40), x, -x]).astype(np.float32)
    fast[0, SPIKE] = 50.0
    slow_x = np.arange(0, 2.0, 0.02)
    slow_x = slow_x[(slow_x < 1.0) | (slow_x > 1.5)]  # a 0.5 s gap
    with h5py.File(p.corpus_file(SHOT), "w") as f:
        _group(f, "ece", x, fast)
        _group(f, "sxr", np.arange(1000) * 1e-3, np.full((2, 1000), np.nan))
        _group(f, "pinj", np.arange(1000) * 2e-3,
               np.stack([np.full(1000, 1e6), np.full(1000, 2e6)]))
        _group(f, "slow", slow_x, (slow_x * 10)[None, :])
        _group(f, "absent", [0.0], [[0.0]])
    p.features.mkdir(parents=True)
    with h5py.File(p.features_file(SHOT), "w") as f:
        g = f.create_group("ip")
        g.create_dataset("xdata", data=np.arange(0, 2.0, 1e-3))
        g.create_dataset("ydata", data=np.full((1, 2000), 1.2e6, dtype=np.float32))
        g.attrs["resolver"] = "fdp"
    park = p.root / "mhd" / "efit02"
    park.mkdir(parents=True)
    times = np.arange(100.0, 2000.0, 20.0)
    np.savez(park / f"{SHOT}.npz", times=times, atime=times,
             a_q95=np.full(times.size, 4.5, dtype=np.float32),
             a_qmin=np.full(times.size, 1.2, dtype=np.float32))
    return p


@pytest.fixture
def service(paths):
    return Signals(paths, fetcher=FakeFetcher(), watch=_watch())


def _decode(trace: Trace):
    n = trace.grid["n"]
    values = np.frombuffer(trace.data, "<f4")
    channels = values.size // (2 * n)
    assert values.size == 2 * channels * n
    return values.reshape(2, channels, n)


def test_a_window_is_the_rows_wire_format(service):
    trace = service.trace(SHOT, "fast", 100.0, 300.0, 50)
    assert trace.grid == {"t0": 100.0, "t1": 300.0, "n": 50}
    assert trace.signal["legend"] == ["a", "b", "c"]
    assert (trace.signal["units"], trace.signal["source"], trace.signal["label"]) == (
        "keV", "corpus", "Fast")
    got = _decode(trace)
    x = (np.arange(FAST_N) * 1e-5).astype(np.float32).astype(np.float64)
    with h5py.File(service.paths.corpus_file(SHOT)) as f:
        y = f["ece"]["ydata"][:]
    step = 200.0 / 50
    for j in range(50):
        lo, hi = (100.0 + j * step) / 1e3, (100.0 + (j + 1) * step) / 1e3
        inside = (x >= lo) & (x < hi)
        assert np.allclose(got[0, :, j], y[:, inside].min(axis=1))
        assert np.allclose(got[1, :, j], y[:, inside].max(axis=1))


def test_rows_are_summed_and_scaled(service):
    got = _decode(service.trace(SHOT, "summed", 0.0, 1000.0, 100))
    assert got.shape == (2, 1, 100)
    assert np.allclose(got[:, 0, :], 3.0)


def test_a_slow_signal_is_joined_between_samples_but_not_across_a_gap(service):
    got = _decode(service.trace(SHOT, "slow", 0.0, 2000.0, 1000))
    centres = (np.arange(1000) + 0.5) * 2.0
    joined = (centres > 20) & (centres < 980)
    assert np.isfinite(got[0, 0, joined]).all()
    assert np.allclose(got[0, 0, joined], centres[joined] / 100, atol=0.2)
    gap = (centres > 1010) & (centres < 1490)
    assert np.isnan(got[0, 0, gap]).all()


def test_a_wide_window_uses_the_pyramid_and_keeps_a_spike(service, monkeypatch):
    exact = _decode(service.trace(SHOT, "fast", 0.0, 2000.0, 100))
    monkeypatch.setattr(signals, "RAW_BUDGET", 10_000)
    built = []
    real = signals.build_pyramid
    monkeypatch.setattr(signals, "build_pyramid",
                        lambda record: built.append(1) or real(record))
    coarse = _decode(service.trace(SHOT, "fast", 0.0, 2000.0, 100))
    again = _decode(service.trace(SHOT, "fast", 500.0, 1500.0, 100))
    assert built == [1]  # built once, then reused
    column = int(SPIKE * 1e-5 * 1e3 / 20)
    assert coarse[1, 0, column] == exact[1, 0, column] == 50.0
    assert np.nanmax(coarse[1]) == np.nanmax(exact[1])
    assert np.isfinite(again).all()


def test_nan_rows_fall_through_to_a_fetch_and_202(service):
    got = service.trace(SHOT, "nan_then_point", 0.0, 1000.0, 100)
    assert isinstance(got, Pending)
    assert got.body() == {"state": "queued", "stage": "waiting for the fetch worker",
                          "elapsed_s": got.elapsed_s, "queue_position": 1}
    [job] = service.fetcher.submitted
    assert job.request == {"via": "ptdata", "name": "PT1"}
    assert job.limit_s == fetch_worker.SHORT_S and job.ident == "nan_then_point"


def test_the_202_flow_ends_in_a_cache_hit(service, paths):
    first = service.trace(SHOT, "nan_then_point", 0.0, 1000.0, 100)
    second = service.trace(SHOT, "nan_then_point", 0.0, 500.0, 100)
    assert isinstance(first, Pending) and isinstance(second, Pending)
    [job] = service.fetcher.submitted  # deduplicated
    t_ms = np.arange(0.0, 1000.0, 1.0)
    service.fetcher.finish(job, {"t_ms": t_ms, "y": (t_ms / 1000)[None, :],
                                 "units": "volt"})
    cache_file = paths.root / "signals" / f"{SHOT}.h5"
    assert cache_file.is_file()
    trace = service.trace(SHOT, "nan_then_point", 0.0, 1000.0, 100)
    assert isinstance(trace, Trace) and trace.signal["source"] == "ptdata"
    assert trace.signal["units"] == "V"  # the catalog's units win
    got = _decode(trace)
    assert np.allclose(got[0, 0], np.arange(100) * 0.01, atol=1e-6)
    service.trace(SHOT, "nan_then_point", 200.0, 300.0, 100)
    assert len(service.fetcher.submitted) == 1
    assert not list(cache_file.parent.glob(".*.tmp"))


def test_a_free_id_takes_the_fetched_units(service):
    job_ident = "custom:DUSBRADIAL"
    assert isinstance(service.trace(SHOT, job_ident, 0.0, 10.0, 16), Pending)
    [job] = service.fetcher.submitted
    service.fetcher.finish(job, {"t_ms": np.arange(10.0), "y": np.ones((1, 10)),
                                 "units": "V"})
    trace = service.trace(SHOT, job_ident, 0.0, 10.0, 16)
    assert trace.signal["units"] == "V" and trace.signal["label"] == "DUSBRADIAL"


def test_a_missing_node_is_remembered_and_the_next_source_tried(service):
    service.trace(SHOT, "two_nodes", 0.0, 10.0, 16)
    [mds] = service.fetcher.submitted
    assert mds.request == {"via": "mds", "node": "\\MHD::A", "tree": "MHD",
                           "free": False}
    service.fetcher.fail(mds, "missing", "TreeNNF")
    assert isinstance(service.trace(SHOT, "two_nodes", 0.0, 10.0, 16), Pending)
    point = service.fetcher.submitted[-1]
    assert point.request == {"via": "ptdata", "name": "PT2"}
    service.fetcher.fail(point, "missing", "TreeNODATA")
    with pytest.raises(FetchError) as caught:
        service.trace(SHOT, "two_nodes", 0.0, 10.0, 16)
    assert caught.value.kind == "missing"
    assert signals.http_status(caught.value) == 404
    with pytest.raises(FetchError):
        service.trace(SHOT, "two_nodes", 0.0, 10.0, 16)
    assert len(service.fetcher.submitted) == 2  # both misses cached: no refetch
    assert service.cache.miss(SHOT, mds.key[1])["kind"] == "missing"


def test_a_stale_miss_is_fetched_again(service, monkeypatch):
    service.trace(SHOT, "two_nodes", 0.0, 10.0, 16)
    service.fetcher.fail(service.fetcher.submitted[0], "missing")
    service.trace(SHOT, "two_nodes", 0.0, 10.0, 16)
    later = time.time() + signals.MISS_TTL_S + 1
    monkeypatch.setattr(signals.time, "time", lambda: later)
    service.fetcher.jobs.clear()
    service.trace(SHOT, "two_nodes", 0.0, 10.0, 16)
    assert service.fetcher.submitted[-1].request["via"] == "mds"


@pytest.mark.parametrize("kind, status", [("invalid", 422), ("auth", 503),
                                          ("upstream", 502), ("timeout", 504)])
def test_each_error_kind_and_its_code_and_none_is_cached(service, kind, status):
    service.trace(SHOT, "custom:PT9", 0.0, 10.0, 16)
    job = service.fetcher.submitted[-1]
    service.fetcher.fail(job, kind, f"{kind} happened")
    with pytest.raises(FetchError) as caught:
        service.trace(SHOT, "custom:PT9", 0.0, 10.0, 16)
    assert (caught.value.kind, caught.value.message) == (kind, f"{kind} happened")
    assert signals.http_status(caught.value) == status
    assert service.cache.miss(SHOT, job.key[1]) is None
    assert isinstance(service.trace(SHOT, "custom:PT9", 0.0, 10.0, 16), Pending)
    assert len(service.fetcher.submitted) == 2


@pytest.mark.parametrize("state", ["unavailable", "expired"])
def test_no_fdp_is_refused_before_a_job(paths, state):
    service = Signals(paths, fetcher=FakeFetcher(), watch=_watch(state))
    with pytest.raises(FetchError) as caught:
        service.trace(SHOT, "custom:PT1", 0.0, 10.0, 16)
    assert caught.value.kind == "auth" and not service.fetcher.submitted
    # A local source still answers.
    assert isinstance(service.trace(SHOT, "fast", 0.0, 10.0, 16), Trace)


def test_fetching_off_is_refused(service, monkeypatch):
    monkeypatch.setenv("LABELER_NO_FETCH", "1")
    with pytest.raises(FetchError, match="LABELER_NO_FETCH") as caught:
        service.trace(SHOT, "custom:PT1", 0.0, 10.0, 16)
    assert caught.value.kind == "auth"


@pytest.mark.parametrize("ident, t0, t1, cols", [
    ("qprof", 0.0, 10.0, 16),  # a profile is a panel
    ("nope", 0.0, 10.0, 16),
    ("custom:bad-name", 0.0, 10.0, 16),
    ("mds:MHD:dim_of(\\MHD::A)", 0.0, 10.0, 16),
    ("fast", 10.0, 10.0, 16),
    ("fast", 0.0, float("inf"), 16),
    ("fast", 0.0, 10.0, 8),
    ("fast", 0.0, 10.0, 9000),
])
def test_an_invalid_request_is_422_before_any_fetch(service, ident, t0, t1, cols):
    with pytest.raises(FetchError) as caught:
        service.trace(SHOT, ident, t0, t1, cols)
    assert caught.value.kind == "invalid" and signals.http_status(caught.value) == 422
    assert not service.fetcher.submitted


def test_a_shot_without_the_data_is_missing(service):
    with pytest.raises(FetchError) as caught:
        service.trace(SHOT, "absent", 0.0, 10.0, 16)
    assert caught.value.kind == "missing"
    with pytest.raises(FetchError):
        service.trace(SHOT + 1, "fast", 0.0, 10.0, 16)


def test_a_feature_and_a_park_trace(service):
    ip = _decode(service.trace(SHOT, "feat", 500.0, 600.0, 20))
    assert np.allclose(ip, 1.2)
    q95 = service.trace(SHOT, "q95", 100.0, 1900.0, 100)
    assert q95.signal["label"] == "q95 (EFIT02)" and q95.signal["source"] == "park"
    assert np.allclose(_decode(q95)[:, 0, 2:-2], 4.5)


def test_availability(service):
    entries = service.catalog().entries
    found = {k: (state, source.kind) for k, (state, source)
             in service.availability(SHOT, entries).items()}
    assert found["fast"] == ("disk", "corpus")
    assert found["nan_then_point"] == ("fetch", "ptdata")
    assert found["absent"] == ("missing", "corpus")
    assert found["q95"] == ("disk", "park") and found["qprof"] == ("disk", "recipe")
    assert found["feat"] == ("disk", "feature")
    # A corpus group of a feature's name counts only when the trace reads samples.
    assert found["feat_corpus"] == ("disk", "feature")
    assert found["feat_sentinel"] == ("missing", "feature")
    assert service.status() == {"fdp": "ok", "queue": 0, "running": None}


def test_status_says_why_nothing_can_be_fetched(service, monkeypatch):
    monkeypatch.setenv(signals.raw.NO_FETCH_ENV, "1")
    got = service.status()
    assert got["fdp"] == "off" and signals.raw.NO_FETCH_ENV in got["reason"]
    monkeypatch.delenv(signals.raw.NO_FETCH_ENV)
    service.watch = _watch("expired")
    got = service.status()
    assert got["fdp"] == "expired" and "fdp login" in got["reason"]


def test_coarse_blocks_draw_a_step_envelope_not_a_ramp():
    xs = np.array([0.5, 1.5, 2.5, 3.5, 8.5])  # block centres; a gap after 3.5
    lo = np.array([[0.0, 10.0, 20.0, 30.0, 80.0]])
    got = signals._columns(xs, lo, lo + 1, 0.0, 4.0, 16, blocks=True)
    assert got[0, 0].tolist() == np.repeat([0, 10, 20, 30], 4).tolist()
    assert got[1, 0].tolist() == np.repeat([1, 11, 21, 31], 4).tolist()
    wide = signals._columns(xs, lo, lo + 1, 0.0, 9.0, 36, blocks=True)
    assert np.isnan(wide[0, 0, 24])  # 6.1 s: in the gap, no block covers it
    ramp = signals._columns(xs, lo, lo + 1, 0.0, 4.0, 16)
    assert not set(ramp[0, 0][np.isfinite(ramp[0, 0])]) <= {0, 10, 20, 30}


def test_the_lru_evicts_the_oldest_shot_but_never_the_new_one(tmp_path):
    cache = Cache(tmp_path / "signals", limit_bytes=10**9)
    big = {"xdata": np.arange(50_000.0), "ydata": np.ones((1, 50_000), np.float32)}
    for shot in (1, 2, 3):
        cache.store(shot, "k", big, {"units": ""})
    for age, shot in ((300, 1), (200, 2), (100, 3)):
        stamp = time.time() - age
        os.utime(cache.path(shot), (stamp, stamp))
    size = cache.path(1).stat().st_size
    cache.limit = int(size * 2.5)
    cache.store(4, "k", big, {"units": ""})
    assert not cache.path(1).exists() and not cache.path(2).exists()
    assert cache.path(3).exists() and cache.path(4).exists()
    cache.limit = 1
    cache.store(5, "k", big, {"units": ""})
    assert cache.path(5).exists()  # just written: kept even alone over the limit


def test_a_rewrite_keeps_the_other_records(tmp_path):
    cache = Cache(tmp_path / "signals")
    cache.store(7, "a", {"xdata": np.arange(3.0), "ydata": np.ones((1, 3))}, {})
    cache.store_miss(7, "b", "TreeNNF")
    cache.store(7, "c", {"xdata": np.arange(3.0), "ydata": np.ones((1, 3))}, {})
    with h5py.File(cache.path(7)) as f:
        assert set(f) == {"a", "c"}
    assert cache.miss(7, "b")["error"] == "TreeNNF"
    cache.store(7, "b", {"xdata": np.arange(3.0), "ydata": np.ones((1, 3))}, {})
    assert cache.miss(7, "b") is None
