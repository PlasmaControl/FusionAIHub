"""On-demand signals for the review page's added rows and dock panels (plan section B).

`Signals.trace` answers `/api/signal`: an entry of the catalog (`review.catalog`), or a
free `custom:`/`mds:` id, over `t0`-`t1` ms in `cols` columns, in the `/api/rows`
trace format byte for byte: little-endian float32 `(2, C, cols)`, the per-column
minima then maxima, column `j` centred at `t0 + (j + 0.5) * (t1 - t0) / cols`, NaN
where there is nothing (the page breaks the line there). Columns that fall between
two samples of a slow signal (EFIT every 20 ms in a 2 ms column) take the value
interpolated at their centre, unless the two samples are a gap apart (more than
`GAP_FACTOR` times the median spacing).

An entry's sources are tried in order, local first:

- `corpus`/`raw`: a corpus-layout group (`xdata` seconds, `ydata` (C, n)), its rows,
  summed when `total`. A group the shot does not have, a sentinel-width group, or
  rows that are NaN wherever probed fall through to the next source.
- `feature`: `events.equilibrium.signal` (features store, corpus, raw cache; the
  archive's 25 ms stamp lag removed).
- `park`: the MHD park's A-EQDSK q95 or q_min (EFIT02, else the fallback tree).
- `ptdata`/`mds`: the signals cache, else the fetch worker (`review.fetch_worker`):
  the answer is then 202 `{state, stage, elapsed_s, queue_position}` until the
  record is cached; the page asks again.

A window wider than `RAW_BUDGET` samples is drawn from a min/max pyramid (`LEVELS`
samples per block), built once per record from chunked reads and kept in memory
(`PYRAMID_BYTES`, least recently used out). When the blocks are coarser than the
columns, each column shows the minimum and maximum of the block that covers it (a step
envelope), never values interpolated between blocks.

The cache, `$LABELER_ROOT/signals/<shot>.h5`, holds what the worker fetched: one
group per source (`Source.key`), corpus layout, plus a profile's `t_ms`/`x`/`y`; a
missing node is remembered for `MISS_TTL_S` in the file's `misses` attribute (an auth
failure, an upstream failure or a timeout never is). The parent alone writes it, as a
new file renamed into place under a per-file lock; past `CACHE_LIMIT_BYTES` the
least recently used shots' files are deleted.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from ...config import Paths
from .. import raw
from . import catalog as cat
from . import profile_fetch
from .fetch_worker import (
    LONG_S,
    SHORT_S,
    STATUS,
    WATCH,
    FdpWatch,
    Fetcher,
    FetchError,
    off_reason,
    refusal,
)

log = logging.getLogger(__name__)

SIGNALS_DIR = "signals"
CACHE_LIMIT_BYTES = 20 * 1024**3
MISS_TTL_S = 24 * 3600.0
#: A cache file read is marked recently used at most this often.
TOUCH_EVERY_S = 600.0
#: The most samples (channels x times) read from disk for one window.
RAW_BUDGET = 4_000_000
LEVELS = (64, 4096)
#: Samples per chunk while building a pyramid: a multiple of every level.
CHUNK = 64 * 4096
PYRAMID_BYTES = 1 << 30
GAP_FACTOR = 3.0
#: A fraction of a column added before flooring a sample's column index.
EDGE_NUDGE = 1e-9
#: Points probed (64 samples each) to tell whether a group's rows hold anything.
PROBES = 16
COLS_MIN, COLS_MAX, COLS_DEFAULT = 16, 8192, 1500
MISSES_ATTR = "misses"


@dataclass(frozen=True)
class Pending:
    """A fetch in flight: the 202 body."""

    state: str
    stage: str
    elapsed_s: float
    queue_position: int

    def body(self) -> dict:
        return {"state": self.state, "stage": self.stage, "elapsed_s": self.elapsed_s,
                "queue_position": self.queue_position}


@dataclass(frozen=True)
class Trace:
    """A trace window: the bytes, `X-Grid` and `X-Signal`."""

    data: bytes
    grid: dict
    signal: dict


def http_status(error: FetchError) -> int:
    return STATUS.get(error.kind, 502)


def invalid(message: str) -> FetchError:
    return FetchError("invalid", message)


# ---- reading a record ----


def _stamp(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return str(path), st.st_mtime_ns, st.st_size


def _bisect(x, value: float, n: int, right: bool = False) -> int:
    """`np.searchsorted` on a sorted 1-D h5py dataset or array, by point reads."""
    if isinstance(x, np.ndarray):
        return int(np.searchsorted(x, value, side="right" if right else "left"))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        v = float(x[mid])
        if v < value or (right and v == value):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _sum_rows(block: np.ndarray) -> np.ndarray:
    """The rows' sum, NaN where every row is NaN."""
    total = np.nansum(block, axis=0, dtype=np.float64)
    total[np.isnan(block).all(axis=0)] = np.nan
    return total[None, :].astype(np.float32)


class Record:
    """Values against time: `x` seconds (a dataset or an array, sorted) and the
    `rows` of `y` (C_all, n), summed into one channel when `total`."""

    def __init__(self, x, y, rows=None, total=False, key=None, where=""):
        self.x, self.y, self.total, self.key, self.where = x, y, total, key, where
        self.n = int(y.shape[-1])
        self.rows = tuple(range(y.shape[0])) if rows is None else tuple(rows)
        self.channels = 1 if total else len(self.rows)

    def block(self, i0: int, i1: int) -> np.ndarray:
        """`(channels, i1 - i0)` float32."""
        rows = self.rows
        if rows == tuple(range(rows[0], rows[-1] + 1)):
            values = self.y[rows[0]:rows[-1] + 1, i0:i1]
        else:
            values = np.stack([self.y[r, i0:i1] for r in rows])
        with np.errstate(invalid="ignore"):
            values = np.asarray(values, dtype=np.float32)
        return _sum_rows(values) if self.total else values

    def times(self, i0: int, i1: int) -> np.ndarray:
        return np.asarray(self.x[i0:i1], dtype=np.float64)

    def has_values(self) -> bool:
        """Whether the rows hold a finite value at any of `PROBES` points."""
        width = min(64, self.n)
        for k in range(PROBES):
            start = int((self.n - width) * (k + 0.5) / PROBES)
            if np.isfinite(self.block(start, start + width)).any():
                return True
        return False


def _columns(xs, lo, hi, t0: float, t1: float, cols: int,
             blocks: bool = False) -> np.ndarray:
    """`(2, C, cols)` float32: minima and maxima of the samples (or blocks) at `xs`
    per column; an empty column between two samples less than a gap apart takes
    the value interpolated at its centre. With `blocks` (pyramid blocks centred at
    `xs`), an empty column takes the minimum and maximum of the block covering it."""
    out = np.full((2, lo.shape[0], cols), np.nan, dtype=np.float32)
    if xs.size == 0:
        return out
    step = (t1 - t0) / cols
    inside = np.flatnonzero((xs >= t0) & (xs < t1))
    hit = np.zeros(cols, dtype=bool)
    if inside.size:
        # The nudge keeps a sample on a column's edge in that column after the
        # cache's ms -> s -> ms round trip (940 ms reads back as 939.9999999999999).
        index = np.clip(((xs[inside] - t0) / step + EDGE_NUDGE).astype(np.int64), 0,
                        cols - 1)
        starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
        where = index[starts]
        with np.errstate(invalid="ignore"):
            out[0][:, where] = np.fmin.reduceat(lo[:, inside], starts, axis=1)
            out[1][:, where] = np.fmax.reduceat(hi[:, inside], starts, axis=1)
        hit[where] = True
    if xs.size < 2 or hit.all():
        return out
    empty = np.flatnonzero(~hit)
    centres = t0 + (empty + 0.5) * step
    if blocks:
        return _block_envelope(out, xs, lo, hi, empty, centres)
    right = np.searchsorted(xs, centres)
    ok = (right > 0) & (right < xs.size)
    empty, centres, right = empty[ok], centres[ok], right[ok]
    if not empty.size:
        return out
    left = right - 1
    gap = xs[right] - xs[left]
    spacing = float(np.median(np.diff(xs)))
    ok = (gap > 0) & (gap <= GAP_FACTOR * spacing)
    empty, centres, left, right, gap = (a[ok] for a in (empty, centres, left, right,
                                                        gap))
    if empty.size:
        w = ((centres - xs[left]) / gap).astype(np.float32)
        out[0][:, empty] = lo[:, left] * (1 - w) + lo[:, right] * w
        out[1][:, empty] = hi[:, left] * (1 - w) + hi[:, right] * w
    return out


def _block_envelope(out, xs, lo, hi, empty, centres) -> np.ndarray:
    """Fill each empty column with the block whose span (half the median block
    spacing either side of its centre) holds the column's centre."""
    half = 0.5 * float(np.median(np.diff(xs))) * (1 + 1e-6)
    right = np.clip(np.searchsorted(xs, centres), 1, xs.size - 1)
    left = right - 1
    pick = np.where(centres - xs[left] <= xs[right] - centres, left, right)
    near = np.abs(centres - xs[pick]) <= half
    out[0][:, empty[near]] = lo[:, pick[near]]
    out[1][:, empty[near]] = hi[:, pick[near]]
    return out


@dataclass
class Pyramid:
    """Per level: block centres (seconds) and the blocks' minima and maxima."""

    levels: dict
    nbytes: int


def build_pyramid(record: Record) -> Pyramid:
    """Min/max blocks of `LEVELS[0]` samples from chunked reads, and each coarser
    level from the one before."""
    first = LEVELS[0]
    starts_t, ends_t, los, his = [], [], [], []
    for a in range(0, record.n, CHUNK):
        b = min(record.n, a + CHUNK)
        xs, values = record.times(a, b), record.block(a, b)
        starts = np.arange(0, b - a, first)
        ends = np.minimum(starts + first, b - a) - 1
        with np.errstate(invalid="ignore"):
            los.append(np.fmin.reduceat(values, starts, axis=1))
            his.append(np.fmax.reduceat(values, starts, axis=1))
        starts_t.append(xs[starts])
        ends_t.append(xs[ends])
    t_start, t_end = np.concatenate(starts_t), np.concatenate(ends_t)
    lo, hi = np.concatenate(los, axis=1), np.concatenate(his, axis=1)
    levels = {first: ((t_start + t_end) / 2, lo, hi)}
    for previous, level in itertools.pairwise(LEVELS):
        factor = level // previous
        starts = np.arange(0, t_start.size, factor)
        ends = np.minimum(starts + factor, t_start.size) - 1
        with np.errstate(invalid="ignore"):
            lo = np.fmin.reduceat(lo, starts, axis=1)
            hi = np.fmax.reduceat(hi, starts, axis=1)
        t_start, t_end = t_start[starts], t_end[ends]
        levels[level] = ((t_start + t_end) / 2, lo, hi)
    nbytes = sum(c.nbytes + lo.nbytes + hi.nbytes for c, lo, hi in levels.values())
    return Pyramid(levels, nbytes)


class _Lru:
    """Pyramids by key, bounded in bytes."""

    def __init__(self, limit: int):
        self.limit, self.size = limit, 0
        self.items: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.building: dict = {}

    def get(self, key, build):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                return self.items[key]
            gate = self.building.setdefault(key, threading.Lock())
        with gate:  # one build per key; a second caller waits for it
            with self.lock:
                if key in self.items:
                    return self.items[key]
            value = build()
            with self.lock:
                self.building.pop(key, None)
                self.items[key] = value
                self.size += value.nbytes
                while self.size > self.limit and len(self.items) > 1:
                    _, old = self.items.popitem(last=False)
                    self.size -= old.nbytes
            return value


def window(record: Record, t0_ms: float, t1_ms: float, cols: int,
           pyramids: _Lru | None = None) -> np.ndarray:
    """`(2, C, cols)` over `t0_ms`-`t1_ms`."""
    t0, t1 = t0_ms / 1e3, t1_ms / 1e3
    n = record.n
    i0 = max(0, _bisect(record.x, t0, n) - 1)
    i1 = min(n, _bisect(record.x, t1, n, right=True) + 1)
    count = i1 - i0
    if count <= 0:
        return np.full((2, record.channels, cols), np.nan, dtype=np.float32)
    width = len(record.rows)
    if count * width <= RAW_BUDGET or pyramids is None or record.key is None:
        if count * width > 4 * RAW_BUDGET:
            raise invalid(f"{record.where}: {count * width} samples is too many to "
                          "read at once")
        values = record.block(i0, i1)
        return _columns(record.times(i0, i1), values, values, t0, t1, cols)
    pyramid = pyramids.get(record.key, lambda: build_pyramid(record))
    level = next((lv for lv in LEVELS if count // lv * width <= RAW_BUDGET), LEVELS[-1])
    centres, lo, hi = pyramid.levels[level]
    j0 = max(0, int(np.searchsorted(centres, t0)) - 1)
    j1 = min(centres.size, int(np.searchsorted(centres, t1, side="right")) + 1)
    return _columns(centres[j0:j1], lo[:, j0:j1], hi[:, j0:j1], t0, t1, cols,
                    blocks=True)


# ---- the cache ----


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Cache:
    """`$LABELER_ROOT/signals/<shot>.h5`: what the fetch worker brought back."""

    def __init__(self, root: Path, limit_bytes: int = CACHE_LIMIT_BYTES):
        self.root, self.limit = Path(root), limit_bytes
        self._locks: dict = {}
        self._guard = threading.Lock()
        self._touched: dict = {}

    def path(self, shot: int) -> Path:
        return self.root / f"{int(shot)}.h5"

    def _lock(self, path: Path) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(str(path), threading.Lock())

    def open(self, shot: int):
        """The shot's cache file, read-only, or None."""
        path = self.path(shot)
        if not path.is_file():
            return None
        try:
            handle = h5py.File(path, "r")
        except OSError:
            return None
        self._touch(path)
        return handle

    def _touch(self, path: Path) -> None:
        now = time.time()
        if now - self._touched.get(str(path), 0.0) < TOUCH_EVERY_S:
            return
        self._touched[str(path)] = now
        try:
            os.utime(path)
        except OSError:
            pass

    def miss(self, shot: int, key: str) -> dict | None:
        """The remembered miss of `key`, while it is fresh."""
        handle = self.open(shot)
        if handle is None:
            return None
        with handle:
            misses = _misses(handle)
        found = misses.get(key)
        if found is None or time.time() - found.get("at", 0) > MISS_TTL_S:
            return None
        return found

    def has(self, shot: int, key: str) -> bool:
        handle = self.open(shot)
        if handle is None:
            return False
        with handle:
            return key in handle

    def store(self, shot: int, key: str, arrays: dict, attrs: dict) -> None:
        """Add (or replace) one record."""

        def edit(out):
            if key in out:
                del out[key]
            group = out.create_group(key)
            for name, values in arrays.items():
                group.create_dataset(name, data=values)
            for name, value in {**attrs, "fetched_at": _now_iso()}.items():
                group.attrs[name] = value
            misses = _misses(out)
            if misses.pop(key, None) is not None:
                out.attrs[MISSES_ATTR] = json.dumps(misses)

        self._rewrite(shot, edit)

    def store_miss(self, shot: int, key: str, message: str) -> None:
        def edit(out):
            now = time.time()
            misses = {k: v for k, v in _misses(out).items()
                      if now - v.get("at", 0) <= MISS_TTL_S}
            misses[key] = {"kind": "missing", "error": message[:300], "at": now}
            out.attrs[MISSES_ATTR] = json.dumps(misses)

        self._rewrite(shot, edit)

    def _rewrite(self, shot: int, edit) -> None:
        path = self.path(shot)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock(path):
            scratch = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with h5py.File(scratch, "w") as out:
                    if path.is_file():
                        try:
                            with h5py.File(path, "r") as old:
                                for name in old:
                                    old.copy(old[name], out, name=name)
                                for name, value in old.attrs.items():
                                    out.attrs[name] = value
                        except OSError:
                            log.warning("%s was unreadable; starting it again", path)
                    edit(out)
                os.replace(scratch, path)
            finally:
                if scratch.exists():
                    scratch.unlink()
        self.evict(keep=path)

    def evict(self, keep: Path | None = None) -> list[Path]:
        """Delete the least recently used files while the cache is over its limit."""
        files = []
        for path in self.root.glob("*.h5"):
            try:
                st = path.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, path))
        total = sum(size for _, size, _ in files)
        removed = []
        for _, size, path in sorted(files, key=lambda item: item[0]):
            if total <= self.limit:
                break
            if keep is not None and path == keep:
                continue
            with self._lock(path):
                try:
                    path.unlink()
                except OSError:
                    continue
            total -= size
            removed.append(path)
        return removed


def _misses(handle) -> dict:
    try:
        found = json.loads(handle.attrs.get(MISSES_ATTR, "{}"))
    except (TypeError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


# ---- the service ----


class Signals:
    """The catalog, the local readers, the cache and the fetch worker, for one app."""

    def __init__(self, paths: Paths, fetcher: Fetcher | None = None,
                 watch: FdpWatch | None = None, cache: Cache | None = None):
        self.paths = paths
        self.watch = WATCH if watch is None else watch
        self.fetcher = fetcher if fetcher is not None else Fetcher(
            target=profile_fetch.serve, watch=self.watch)
        self.cache = Cache(paths.root / SIGNALS_DIR) if cache is None else cache
        self.pyramids = _Lru(PYRAMID_BYTES)
        self._probes: dict = {}
        self._features: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def catalog(self) -> cat.Catalog:
        return cat.load(self.paths.config_dir / cat.CATALOG_FILE)

    def resolve(self, ident) -> cat.Entry:
        try:
            return self.catalog().resolve(ident)
        except cat.Invalid as error:
            raise invalid(str(error)) from None

    def status(self) -> dict:
        """`{fdp, reason?, queue, running}`: fdp is "off" while `LABELER_NO_FETCH`
        refuses every fetch, else the token's state; `reason` says why nothing can be
        fetched, when it cannot."""
        fdp = self.watch.get()
        problem = refusal(fdp)
        state = "off" if off_reason() is not None else fdp.state
        return {"fdp": state, **({"reason": problem.message} if problem else {}),
                **self.fetcher.summary()}

    # -- a trace --

    def trace(self, shot: int, ident, t0: float, t1: float,
              cols: int = COLS_DEFAULT) -> Trace | Pending:
        """The window of a trace entry, or the fetch it waits on. Raises FetchError."""
        entry = self.resolve(ident)
        if entry.kind != "trace":
            raise invalid(f"{entry.id} is a profile: add it as a dock panel")
        check_window(t0, t1, cols)
        for source in entry.sources:
            if source.local:
                got = self._local(shot, entry, source, t0, t1, cols)
            else:
                got = self._fetched_trace(shot, entry, source, t0, t1, cols)
            if got is not None:
                return got
        wheres = "; ".join(s.where() for s in entry.sources)
        raise FetchError("missing", f"shot {shot} has no {entry.label} ({wheres})")

    def _answer(self, entry, source, values, t0, t1, cols, units=None, tag=None):
        if source.scale != 1.0:
            values = values * np.float32(source.scale)
        channels = values.shape[1]
        tag = source.tag if tag is None else tag
        label = entry.label + (f" ({tag})" if tag else "")
        signal = {"label": label, "units": entry.units or (units or ""),
                  "legend": entry.legend_for(channels), "source": source.wire,
                  "where": source.where(), "id": entry.id}
        grid = {"t0": t0, "t1": t1, "n": cols}
        return Trace(values.astype("<f4").tobytes(), grid, signal)

    def _local(self, shot, entry, source, t0, t1, cols):
        if source.kind in ("corpus", "raw"):
            path = self._group_file(shot, source.kind)
            with _opened(path) as handle:
                record = self._group_record(handle, path, source)
                if record is None:
                    return None
                values = window(record, t0, t1, cols, self.pyramids)
            return self._answer(entry, source, values, t0, t1, cols)
        record = self._array_record(shot, source)
        if record is None:
            return None
        values = window(record, t0, t1, cols, self.pyramids)
        tag = park_tag(self.paths, shot) if source.kind == "park" else None
        return self._answer(entry, source, values, t0, t1, cols, tag=tag)

    def _group_file(self, shot: int, kind: str) -> Path:
        if kind == "corpus":
            return self.paths.corpus_file(shot)
        return raw.cache_path(shot, paths=self.paths)

    def _group_record(self, handle, path: Path, source) -> Record | None:
        """The group's rows as a Record, or None when the file does not hold them."""
        if handle is None or source.name not in handle:
            return None
        group = handle[source.name]
        if "xdata" not in group or "ydata" not in group:
            return None
        y = group["ydata"]
        if y.ndim != 2 or y.shape[-1] <= raw.SENTINEL_WIDTH:
            return None
        if max(source.rows) >= y.shape[0]:
            return None
        stamp = _stamp(path)
        key = (stamp, source.name, source.rows, source.total)
        record = Record(group["xdata"], y, source.rows, source.total, key=key,
                        where=source.where())
        if not self._probe(key, record):
            return None
        return record

    def _probe(self, key, record: Record) -> bool:
        with self._lock:
            known = self._probes.get(key)
        if known is None:
            known = record.has_values()
            with self._lock:
                if len(self._probes) > 50_000:
                    self._probes.clear()
                self._probes[key] = known
        return known

    def _array_record(self, shot: int, source) -> Record | None:
        """A feature or a park trace, read whole (they are small) and kept."""
        if source.kind == "feature":
            stamps = tuple(_stamp(p) for p in (
                self.paths.features_file(shot), self.paths.corpus_file(shot),
                raw.cache_path(shot, paths=self.paths)))
        else:
            from ...mhd import equilibrium as mhd_eq

            found = mhd_eq.equilibrium_file(self.paths, shot)
            stamps = None if found is None else found
            if stamps is None:
                return None
        key = (source.kind, source.name, int(shot), stamps)
        with self._lock:
            if key in self._features:
                self._features.move_to_end(key)
                return self._features[key]
        record = (_feature_record(self.paths, shot, source) if source.kind == "feature"
                  else _park_record(stamps, source))
        with self._lock:
            self._features[key] = record
            while len(self._features) > 128:
                self._features.popitem(last=False)
        return record

    def _fetched_trace(self, shot, entry, source, t0, t1, cols):
        key = source.key()
        got = self._fetch_state(shot, entry, source, {"via": source.kind} | (
            {"name": source.name} if source.kind == "ptdata" else
            {"node": source.name, "tree": source.tree, "free": source.free}))
        if got is not None:
            return got if isinstance(got, Pending) else None
        handle = self.cache.open(shot)
        with _closing(handle):
            if handle is None or key not in handle:
                return None
            group = handle[key]
            stamp = (int(shot), key, str(group.attrs.get("fetched_at", "")))
            record = Record(group["xdata"], group["ydata"], key=stamp,
                            where=source.where())
            values = window(record, t0, t1, cols, self.pyramids)
            units = str(group.attrs.get("units", ""))
        return self._answer(entry, source, values, t0, t1, cols, units=units)

    def _fetch_state(self, shot, entry, source, request) -> Pending | bool | None:
        """None when the cache holds the record; False when the shot lacks it (the
        caller falls through); Pending while it is fetched. Raises FetchError for an
        auth, upstream or timeout failure."""
        key = source.key()
        job_key = (int(shot), key)
        error = self.fetcher.take_error(job_key)
        if error is not None:
            kind, message = error
            if kind != "missing":
                raise FetchError(kind, message)
            self.cache.store_miss(shot, key, message)
            return False
        job = self.fetcher.job(job_key)
        if job is not None:
            return self._pending(job)
        if self.cache.has(shot, key):
            return None
        if self.cache.miss(shot, key) is not None:
            return False
        problem = refusal(self.watch.get())
        if problem is not None:
            raise problem
        job = self.fetcher.submit(
            job_key, request, shot=shot, ident=entry.id, where=source.where(),
            limit_s=LONG_S if source.long_limit else SHORT_S,
            on_result=lambda payload: self._store(shot, entry, source, payload),
        )
        return self._pending(job)

    def _pending(self, job) -> Pending:
        state = "queued" if job.state == "queued" else "fetching"
        return Pending(state, job.stage, job.elapsed_s(), self.fetcher.position(job))

    def _store(self, shot, entry, source, payload: dict) -> None:
        attrs = {"spec": json.dumps(source.spec(), sort_keys=True), "id": entry.id,
                 "units": str(payload.get("units", ""))}
        if "x" in payload:  # a profile
            arrays = {"t_ms": np.asarray(payload["t_ms"], dtype=np.float64),
                      "x": np.asarray(payload["x"], dtype=np.float32),
                      "y": np.asarray(payload["y"], dtype=np.float32)}
        else:
            arrays = {"xdata": np.asarray(payload["t_ms"], dtype=np.float64) / 1e3,
                      "ydata": np.asarray(payload["y"], dtype=np.float32)}
        self.cache.store(shot, source.key(), arrays, attrs)

    def fetched_profile(self, shot, entry, source) -> Pending | dict | None:
        """A fetching recipe's `{t_ms, x, y, units}` from the cache, the fetch it
        waits on, or None when the shot lacks it."""
        got = self._fetch_state(shot, entry, source, {"via": source.name})
        if isinstance(got, Pending):
            return got
        if got is False:
            return None
        handle = self.cache.open(shot)
        with _closing(handle):
            if handle is None or source.key() not in handle:
                return None
            group = handle[source.key()]
            return {"t_ms": group["t_ms"][:], "x": group["x"][:], "y": group["y"][:],
                    "units": str(group.attrs.get("units", ""))}

    # -- the catalog listing --

    def availability(self, shot: int, entries) -> dict[str, tuple]:
        """Per entry id, `(state, source)`: "disk" and the first source on this
        machine for the shot, else "fetch" and the first that would be fetched, else
        "missing" and the first source. Metadata only, plus the memoized probes."""
        from . import profiles

        groups = {kind: self._group_shapes(self._group_file(shot, kind))
                  for kind in ("corpus", "raw")}
        features = _names(self.paths.features_file(shot))
        handle = self.cache.open(shot)
        with _closing(handle):
            cached = set(handle) if handle is not None else set()
            misses = _misses(handle) if handle is not None else {}
        from ...mhd import equilibrium as mhd_eq

        park = mhd_eq.equilibrium_file(self.paths, shot) is not None
        out = {}
        for entry in entries:
            state, chosen = "missing", entry.sources[0]
            for source in entry.sources:
                found = self._source_state(shot, source, groups, features, park,
                                           cached, misses, profiles)
                if found == "disk":
                    state, chosen = "disk", source
                    break
                if found == "fetch" and state == "missing":
                    state, chosen = "fetch", source
            out[entry.id] = (state, chosen)
        return out

    def _source_state(self, shot, source, groups, features, park, cached, misses,
                      profiles) -> str:
        if source.kind in ("corpus", "raw"):
            shape = groups[source.kind].get(source.name)
            if shape is None or shape[-1] <= raw.SENTINEL_WIDTH or (
                max(source.rows) >= shape[0]
            ):
                return "missing"
            path = self._group_file(shot, source.kind)
            with _opened(path) as handle:
                record = self._group_record(handle, path, source)
            return "disk" if record is not None else "missing"
        if source.kind == "feature":
            # A corpus or raw group of the same name counts only when the trace's
            # own reader (`equilibrium.signal`, memoized) gets samples from it.
            stored = source.name in features or any(
                (shape := groups[kind].get(source.name)) is not None
                and shape[-1] > raw.SENTINEL_WIDTH for kind in ("corpus", "raw"))
            return "disk" if stored and self._array_record(shot, source) is not None \
                else "missing"
        if source.kind == "park":
            return "disk" if park else "missing"
        if source.kind == "recipe" and not source.fetches:
            return "disk" if profiles.available(self.paths, shot, source.name) else (
                "missing")
        key = source.key()
        if key in cached:
            return "disk"
        miss = misses.get(key)
        if miss is not None and time.time() - miss.get("at", 0) <= MISS_TTL_S:
            return "missing"
        return "fetch"

    def _group_shapes(self, path: Path) -> dict:
        with _opened(path) as handle:
            if handle is None:
                return {}
            out = {}
            for name, group in handle.items():
                if isinstance(group, h5py.Group) and "ydata" in group:
                    out[name] = group["ydata"].shape
            return out


def check_window(t0, t1, cols) -> None:
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
        raise invalid("t0 and t1 are finite milliseconds, with t0 < t1")
    if not COLS_MIN <= cols <= COLS_MAX:
        raise invalid(f"cols is {COLS_MIN} to {COLS_MAX}")


def _names(path: Path) -> set[str]:
    with _opened(path) as handle:
        return set(handle) if handle is not None else set()


class _opened:
    """`h5py.File(path, "r")`, or None when it is absent or unreadable."""

    def __init__(self, path: Path):
        self.path, self.handle = Path(path), None

    def __enter__(self):
        if self.path.is_file():
            try:
                self.handle = h5py.File(self.path, "r")
            except OSError:
                self.handle = None
        return self.handle

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()


class _closing:
    def __init__(self, handle):
        self.handle = handle

    def __enter__(self):
        return self.handle

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()


def _feature_record(paths: Paths, shot: int, source) -> Record | None:
    from ..equilibrium import signal
    from ..verify import NoDataError

    try:
        array = signal(int(shot), source.name, paths)
    except (NoDataError, OSError, KeyError, ValueError):
        return None
    x = np.asarray(array.x, dtype=np.float64)
    y = np.atleast_2d(np.asarray(array.y, dtype=np.float32))
    if x.size < 2 or not np.isfinite(y).any():
        return None
    return Record(x, y, key=("feature", source.name, int(shot)),
                  where=source.where())


def _park_record(found, source) -> Record | None:
    key, _tree = found
    try:
        with np.load(key[0]) as d:
            if source.name not in d.files or "atime" not in d.files:
                return None
            t_ms = np.asarray(d["atime"], dtype=np.float64)
            y = np.asarray(d[source.name], dtype=np.float32).ravel()
    except (OSError, ValueError, KeyError):
        return None
    if t_ms.size != y.size or t_ms.size < 2:
        return None
    y = np.where(np.isfinite(y) & (y > 0), y, np.nan).astype(np.float32)
    keep = np.isfinite(t_ms)
    order = np.argsort(t_ms[keep], kind="stable")
    return Record(t_ms[keep][order] / 1e3, y[keep][order][None, :],
                  key=("park", source.name, key), where=source.where())


def park_tag(paths: Paths, shot: int) -> str:
    """The EFIT tree the park's file for this shot holds, upper case."""
    from ...mhd import equilibrium as mhd_eq

    found = mhd_eq.equilibrium_file(paths, shot)
    return "" if found is None else found[1].upper()
