"""The review server's fetch worker: one spawn-context process that fetches from
DIII-D, one job at a time (plan section B).

toksearch's MdsTreeRegistry is process-global and unlocked, and libfdpio is not
fork-safe, so fetches never run in the server's own threads: they run in one child
process, started with the `spawn` context the first time something is fetched. The
child only fetches and answers with arrays; the parent (`signals`) writes the cache.

- Jobs are deduplicated by key (`(shot, source key)`): a second request for a job in
  flight joins it.
- A job's limit is `SHORT_S` for a PTDATA point or a catalog node and `LONG_S` for a
  free node or a recipe of several nodes. A job past its limit kills the child; the
  next job starts a new one.
- The child gets the best fdp token when it starts (`Fdp.token`), and is restarted
  when that token changes, so a long-running server picks up an `fdp login`.

Errors are `FetchError(kind, message)`, kind one of `KINDS`: `invalid` (the node is
not a time trace), `missing` (the shot has no such data), `auth` (no fdp, or its
token expired), `upstream` (fdp failed) and `timeout`. Messages never carry the
token or the environment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import multiprocessing
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: What `fdp run` leaves in the environment (`ui.serve.FDP_MARKERS`, measured
#: 2026-09-18); without them a fetch fails with PTSERVER / TREE-E-FOPENR.
FDP_MARKERS = ("PTDATA_LIBRARY", "default_tree_path")
#: How to restart the review server so it can fetch (`raw.FDP_UI_COMMAND`).
FDP_COMMAND = "pixi run -e labeler labeler-verify"
LOGIN_COMMAND = "pixi run -e labeler fdp login"
TOKEN_ENV = "BEARER_TOKEN"
#: A token this close to its expiry counts as expired.
TOKEN_MARGIN_S = 60
#: How long the token read is reused (`/api/version` and every fetch read it).
WATCH_TTL_S = 60.0

SHORT_S = 30.0
LONG_S = 120.0
#: How long a new child may take to import toksearch, before its first job's clock.
STARTUP_S = 90.0
#: PTDATA's "getservbyname failed for task 'PTSERVER'" has been seen once on a shot
#: that fetched cleanly seconds later (`raw.RETRY_DELAY_S`): one retry.
RETRY_DELAY_S = 2.0
#: A failed job's error is kept this long for the page's next poll.
ERROR_KEEP_S = 60.0

KINDS = ("invalid", "missing", "auth", "upstream", "timeout")
STATUS = {"invalid": 422, "missing": 404, "auth": 503, "upstream": 502, "timeout": 504}

#: The most samples a fetched record may hold (channels x times).
MAX_SAMPLES = 64_000_000
MAX_CHANNELS = 512
READY = "ready"


class FetchError(Exception):
    """A fetch that did not give values; `kind` is one of `KINDS`."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind if kind in KINDS else "upstream"
        self.message = message


# ---- fdp and its token ----


def under_fdp(environ=None) -> bool:
    """Whether this process runs under `fdp run`."""
    environ = os.environ if environ is None else environ
    return all(environ.get(marker) for marker in FDP_MARKERS)


def jwt_exp(token) -> int | None:
    """A JWT's `exp` (unix seconds), or None for an opaque token. Never verified,
    never logged."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return int(exp) if exp is not None else None
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _read_token(path: Path) -> str | None:
    try:
        text = path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def _iso(exp: int | None) -> str | None:
    if exp is None:
        return None
    return datetime.fromtimestamp(exp, UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Fdp:
    """What the server knows about fdp: whether it runs under it (`env`), the best
    token's expiry, and `state` (ok, expired or unavailable)."""

    env: bool
    expires_at: str | None
    state: str
    token: str | None = field(default=None, repr=False)

    @property
    def digest(self) -> str:
        """Which token the worker runs with, without the token."""
        return hashlib.sha256((self.token or "").encode()).hexdigest()[:16]

    def public(self) -> dict:
        return {"env": self.env, "expires_at": self.expires_at}


def read_fdp(environ=None, home=None, now=None) -> Fdp:
    """The best usable token as `fdp.auth.get_valid_token` picks it: $BEARER_TOKEN
    unless it is an expired JWT, then the device caches, then ~/.fdp/token. No
    network call."""
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    now = time.time() if now is None else now
    env = under_fdp(environ)
    best, best_exp, latest = None, None, None
    given = environ.get(TOKEN_ENV)
    if given:
        exp = jwt_exp(given)
        latest = exp
        if exp is None or exp > now + TOKEN_MARGIN_S:
            best, best_exp = given, exp
    cache = home / ".fdp" / "cache"
    files = sorted(cache.glob("*.token")) if cache.is_dir() else []
    for path in [*files, home / ".fdp" / "token"]:
        token = _read_token(path)
        exp = jwt_exp(token) if token else None
        if exp is None:
            continue
        latest = exp if latest is None else max(latest, exp)
        if best is None and exp > now + TOKEN_MARGIN_S:
            best, best_exp = token, exp
    if not env:
        state = "unavailable"
    else:
        state = "ok" if best is not None else "expired"
    return Fdp(env=env, expires_at=_iso(best_exp if best is not None else latest),
               state=state, token=best)


class FdpWatch:
    """`read_fdp`, reused for `WATCH_TTL_S`."""

    def __init__(self, reader=read_fdp, ttl_s: float = WATCH_TTL_S):
        self._reader, self._ttl = reader, ttl_s
        self._lock = threading.Lock()
        self._value: Fdp | None = None
        self._at = 0.0

    def get(self) -> Fdp:
        with self._lock:
            now = time.monotonic()
            if self._value is None or now - self._at >= self._ttl:
                self._value, self._at = self._reader(), now
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None


WATCH = FdpWatch()


def off_reason() -> str | None:
    """Why this server never fetches (`LABELER_NO_FETCH` is set), or None."""
    from ..raw import NO_FETCH_ENV, fetching_disabled

    if fetching_disabled():
        return (f"fetching is off on this server ({NO_FETCH_ENV} is set): only "
                "signals on disk can be shown")
    return None


def refusal(fdp: Fdp) -> FetchError | None:
    """Why nothing can be fetched now, before a job is queued; None when it can."""
    off = off_reason()
    if off is not None:
        return FetchError("auth", off)
    if fdp.state == "unavailable":
        return FetchError("auth", "this server was not started under the fdp wrapper, "
                                  f"so it cannot fetch: restart it as '{FDP_COMMAND}'")
    if fdp.state == "expired":
        return FetchError("auth", "the fdp token has expired: run "
                                  f"'{LOGIN_COMMAND}', then try again")
    return None


# ---- the child ----


#: Dim units that mean "this axis is time", and their factor to milliseconds.
TIME_UNITS = {"ms": 1.0, "s": 1e3, "sec": 1e3, "secs": 1e3, "second": 1e3,
              "seconds": 1e3}
#: MDSplus's "no such node" and "node holds nothing" (`verify.MISSING_NODE_ERRORS`).
MISSING_MARKERS = ("TreeNODATA", "TreeNNF")
AUTH = re.compile(r"\b40[13]\b|unauthori[sz]ed|forbidden|bearer|token (has )?expired|"
                  r"invalid token", re.IGNORECASE)
TRANSIENT_MARKERS = ("getservbyname",)


def classify(error: BaseException, token: str | None = None) -> tuple[str, str]:
    """`(kind, message)` for an exception raised by toksearch. TREE-E-FOPENR (the
    tree would not open) and PTSERVER failures are upstream: never cached."""
    text = f"{type(error).__name__}: {error}"
    if token:
        text = text.replace(token, "<token>")
    text = re.sub(r"\s+", " ", text).strip()[:300]
    if any(marker in text for marker in MISSING_MARKERS):
        return "missing", text
    if AUTH.search(text):
        return "auth", f"fdp refused the request ({text}); run '{LOGIN_COMMAND}'"
    return "upstream", text


def _dims(rec: dict) -> list[tuple[str, np.ndarray, str]]:
    """A record's dimensions: `(name, values, lowercased units)`."""
    units = rec.get("units") if isinstance(rec.get("units"), dict) else {}
    out = []
    for key, value in rec.items():
        if key in ("data", "units"):
            continue
        try:
            with np.errstate(invalid="ignore"):
                arr = np.asarray(value, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size > 1:
            out.append((key, arr, str(units.get(key, "")).strip().lower()))
    return out


def _time_axis(rec: dict, sizes) -> tuple[np.ndarray, int] | None:
    """`(t_ms, axis)`: the record's time dimension and which data axis it is."""
    for _, arr, unit in _dims(rec):
        if unit in TIME_UNITS:
            for axis, size in enumerate(sizes):
                if arr.size == size:
                    return arr * TIME_UNITS[unit], axis
    return None


def as_trace(rec: dict, where: str) -> dict:
    """`{"t_ms", "y" (C, n), "units"}` from a 1-D time trace or a 2-D record with a
    time axis; FetchError `invalid` for anything else."""
    with np.errstate(invalid="ignore"):
        data = np.asarray(rec.get("data"), dtype=np.float64)
    if data.ndim not in (1, 2) or data.size == 0:
        raise FetchError("invalid", f"{where} is not a time trace (data {data.shape})")
    found = _time_axis(rec, data.shape)
    if found is None:
        raise FetchError("invalid", f"{where} is not a time trace: none of its "
                                    "dimensions is time")
    t_ms, axis = found
    if data.ndim == 2 and data.shape[0] == data.shape[1]:
        raise FetchError("invalid", f"{where} is not a time trace: its time axis is "
                                    "ambiguous")
    y = data[None, :] if data.ndim == 1 else (data if axis == 1 else data.T)
    if y.shape[0] > MAX_CHANNELS:
        raise FetchError("invalid", f"{where} has {y.shape[0]} channels; at most "
                                    f"{MAX_CHANNELS} can be shown")
    if y.size > MAX_SAMPLES:
        raise FetchError("invalid", f"{where} is too large to show ({y.size} samples)")
    keep = np.isfinite(t_ms)
    order = np.argsort(t_ms[keep], kind="stable")
    t_ms, y = t_ms[keep][order], y[:, keep][:, order]
    if t_ms.size < 2:
        raise FetchError("missing", f"{where} has fewer than two samples")
    units = rec.get("units") if isinstance(rec.get("units"), dict) else {}
    return {"t_ms": t_ms, "y": y.astype(np.float32),
            "units": str(units.get("data", "")).strip()}


def _ptdata(name: str, shot: int, sleep=time.sleep) -> dict:
    from toksearch_d3d import PtDataSignal

    for attempt in (1, 2):
        try:
            return PtDataSignal(name).fetch(int(shot))
        except Exception as error:  # toksearch raises its own and plain errors
            if attempt == 1 and any(m in str(error) for m in TRANSIENT_MARKERS):
                sleep(RETRY_DELAY_S)
                continue
            raise
    raise AssertionError("unreachable")


def _mds(node: str, tree: str, shot: int, dims=None) -> dict:
    from toksearch import MdsSignal

    signal = MdsSignal(node, tree, dims=list(dims)) if dims else MdsSignal(node, tree)
    return signal.fetch(int(shot))


def _free_node(node: str, tree: str, shot: int) -> dict:
    """A free node: its own dimensions first; a 2-D record whose default dimension is
    not time is asked again for both of its dimensions."""
    rec = _mds(node, tree, shot)
    try:
        return as_trace(rec, node)
    except FetchError as error:
        data = np.asarray(rec.get("data"))
        if error.kind != "invalid" or data.ndim != 2:
            raise
    return as_trace(_mds(node, tree, shot, dims=("dim0", "dim1")), node)


EFIT01_QPSI = "\\efit01::top.results.geqdsk:qpsi"
EFIT01_GTIME = "\\efit01::top.results.geqdsk:gtime"


def _efit01_q(shot: int) -> dict:
    """EFIT01's q(psi_N): `{"t_ms", "x", "y" (n_t, n_x), "units"}`."""
    rec = _mds(EFIT01_QPSI, "EFIT01", shot)
    gtime = _mds(EFIT01_GTIME, "EFIT01", shot)
    with np.errstate(invalid="ignore"):
        q = np.abs(np.asarray(rec["data"], dtype=np.float64))
        t_ms = np.asarray(gtime["data"], dtype=np.float64).ravel()
    if q.ndim != 2 or t_ms.size < 2:
        raise FetchError("missing", "EFIT01 has no q profile for this shot")
    if q.shape[0] != t_ms.size and q.shape[1] == t_ms.size:
        q = q.T
    if q.shape[0] != t_ms.size:
        raise FetchError("upstream", f"EFIT01 q {q.shape} does not match its "
                                     f"{t_ms.size} times")
    order = np.argsort(t_ms, kind="stable")
    return {"t_ms": t_ms[order], "x": np.linspace(0.0, 1.0, q.shape[1]),
            "y": q[order].astype(np.float32), "units": ""}


def run(request: dict, shot: int) -> dict:
    """One fetch, in the child."""
    via = request.get("via")
    if via == "ptdata":
        return as_trace(_ptdata(request["name"], shot), f"PTDATA {request['name']}")
    if via == "mds":
        node, tree = request["node"], request["tree"]
        if request.get("free"):
            return _free_node(node, tree, shot)
        return as_trace(_mds(node, tree, shot), node)
    if via == "efit01_q":
        return _efit01_q(shot)
    raise FetchError("invalid", f"unknown fetch {via!r}")


def _prepare() -> None:
    import toksearch  # noqa: F401
    import toksearch_d3d  # noqa: F401


def serve(conn, token: str | None, work=run, prepare=_prepare) -> None:
    """The child's loop: import toksearch, say `READY`, then answer each
    `(request, shot)` with `("ok", payload)` or `("error", (kind, message))`."""
    if token:
        os.environ[TOKEN_ENV] = token
    try:
        prepare()
    except Exception as error:  # noqa: BLE001 - any import failure is the answer
        conn.send((READY, f"toksearch does not import here: {classify(error)[1]}"))
        return
    conn.send((READY, None))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        request, shot = message
        try:
            reply = ("ok", work(request, shot))
        except FetchError as error:
            reply = ("error", (error.kind, error.message))
        except Exception as error:  # noqa: BLE001 - toksearch raises plain errors
            reply = ("error", classify(error, token))
        conn.send(reply)


# ---- the parent ----


@dataclass
class Job:
    key: tuple
    request: dict
    shot: int
    ident: str
    where: str
    limit_s: float
    on_result: object
    submitted: float = field(default_factory=time.monotonic)
    state: str = "queued"
    stage: str = "waiting for the fetch worker"
    error: tuple[str, str] | None = None
    ended: float | None = None

    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.submitted, 1)


class Fetcher:
    """The worker process and its queue. `on_result(payload)` runs in the dispatcher
    thread once the child answers; the job leaves the table when it returns."""

    def __init__(self, target=serve, watch: FdpWatch | None = None,
                 startup_s: float = STARTUP_S):
        self._target, self._watch = target, WATCH if watch is None else watch
        self._startup_s = startup_s
        self._cond = threading.Condition()
        self._queue: deque[Job] = deque()
        self._jobs: dict[tuple, Job] = {}
        self._running: Job | None = None
        self._thread: threading.Thread | None = None
        self._process = None
        self._conn = None
        self._digest: str | None = None
        self.started = 0  # child processes started, for tests and the log

    # -- the table --

    def submit(self, key: tuple, request: dict, *, shot: int, ident: str, where: str,
               limit_s: float, on_result) -> Job:
        """The job for `key`: the one in flight, else a new one at the queue's end.
        `ident` is the page's id, `where` the node or point (the stage names it)."""
        with self._cond:
            job = self._jobs.get(key)
            if job is not None and job.state != "failed":
                return job
            job = Job(key, request, int(shot), ident, where, limit_s, on_result)
            self._jobs[key] = job
            self._queue.append(job)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True,
                                                name="fetch-dispatch")
                self._thread.start()
            self._cond.notify_all()
            return job

    def job(self, key: tuple) -> Job | None:
        """The job for `key` while it is queued, fetching or recently failed."""
        with self._cond:
            self._prune()
            return self._jobs.get(key)

    def take_error(self, key: tuple) -> tuple[str, str] | None:
        """A failed job's `(kind, message)`, reported once."""
        with self._cond:
            job = self._jobs.get(key)
            if job is None or job.state != "failed":
                return None
            del self._jobs[key]
            return job.error

    def position(self, job: Job) -> int:
        """0 while it runs, else its place in the queue (1 is next)."""
        with self._cond:
            if job is self._running:
                return 0
            for index, queued in enumerate(self._queue):
                if queued is job:
                    return index + 1
            return 0

    def summary(self) -> dict:
        with self._cond:
            running = self._running
            return {
                "queue": len(self._queue),
                "running": None if running is None else {
                    "id": running.ident, "shot": running.shot,
                    "elapsed_s": running.elapsed_s(),
                },
            }

    def _prune(self) -> None:
        now = time.monotonic()
        for key, job in list(self._jobs.items()):
            if job.state == "failed" and now - (job.ended or now) > ERROR_KEEP_S:
                del self._jobs[key]

    # -- the dispatcher --

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    if not self._cond.wait(timeout=300) and not self._queue:
                        self._thread = None  # idle: the next submit starts another
                        return
                job = self._queue.popleft()
                alive = self._process is not None and self._process.is_alive()
                job.state, job.stage = "fetching", (
                    f"fetching {job.where}" if alive else "starting the fetch worker")
                self._running = job
            try:
                payload = self._fetch(job)
                job.stage = "writing the cache"
                job.on_result(payload)
            except FetchError as error:
                self._fail(job, error.kind, error.message)
            except Exception as error:  # the cache write, or a bug: never silent
                log.exception("fetch %s for %s failed", job.where, job.shot)
                self._fail(job, "upstream", f"{type(error).__name__}: {error}")
            else:
                with self._cond:
                    self._jobs.pop(job.key, None)
            finally:
                with self._cond:
                    self._running = None

    def _fail(self, job: Job, kind: str, message: str) -> None:
        with self._cond:
            job.state, job.error, job.ended = "failed", (kind, message), time.monotonic()

    def _fetch(self, job: Job):
        fdp = self._watch.get()
        problem = refusal(fdp)
        if problem is not None:
            raise problem
        if self._process is not None and fdp.digest != self._digest:
            self.stop()  # a new token: a new child
        if self._process is None or not self._process.is_alive():
            job.stage = "starting the fetch worker"
            self._start(fdp)
        job.stage = f"fetching {job.where}"
        try:
            self._conn.send((job.request, job.shot))
            if not self._conn.poll(job.limit_s):
                self.stop()
                raise FetchError("timeout", f"{job.where} did not answer within "
                                            f"{job.limit_s:.0f} s; the fetch worker "
                                            "was restarted")
            status, body = self._conn.recv()
        except (EOFError, OSError, BrokenPipeError):
            self.stop()
            raise FetchError("upstream", "the fetch worker stopped; try again") from None
        if status == "error":
            kind, message = body
            raise FetchError(kind, message)
        return body

    def _start(self, fdp: Fdp) -> None:
        self.stop()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=self._target, args=(child, fdp.token),
                                  daemon=True, name="labeler-fetch")
        process.start()
        child.close()
        self.started += 1
        self._process, self._conn, self._digest = process, parent, fdp.digest
        if not parent.poll(self._startup_s):
            self.stop()
            raise FetchError("timeout", "the fetch worker did not start within "
                                        f"{self._startup_s:.0f} s")
        try:
            status, problem = parent.recv()
        except (EOFError, OSError):
            self.stop()
            raise FetchError("upstream", "the fetch worker stopped on start") from None
        if status != READY or problem:
            self.stop()
            raise FetchError("upstream", problem or "the fetch worker did not start")

    def stop(self) -> None:
        """Kill the child, if any; the next job starts another."""
        process, conn = self._process, self._conn
        self._process = self._conn = self._digest = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if process is not None and process.is_alive():
            process.kill()
            process.join(5)
