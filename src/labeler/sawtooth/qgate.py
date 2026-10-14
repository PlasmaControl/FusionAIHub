"""The q = 1 surface gate: time with no q = 1 surface, where no 1/1 mode can live.

A sawtooth (and any 1/1 internal kink) needs q = 1 inside the plasma. Time where
q_min stays at or above a tree's cut for at least 100 ms has no q = 1 surface
(`NO_Q1`). Each whole-ms cell is decided by the best equilibrium that covers it
(owner 2026-10-14: "if efit02 doesnt exist, use efit01 or efit02er"):

- EFIT02 (MSE-constrained) `a_qmin >= 1.2`; the 0.2 margin is the MHD surface
  rule's (q_min - 0.2);
- else EFIT02ER (also MSE-constrained, trusted like EFIT02) `>= 1.2`;
- else the magnetics-only EFIT01 `>= 2.0`: EFIT01 reads about 0.1 above EFIT02
  at the median and 0.5 at the 90th percentile, and on 62 shots EFIT02 was
  below 1.2 in 0.6% of the time EFIT01 >= 2.0 gates.

A source decides only where every better one is silent: outside its valid time
base, or across a gap of more than `MAX_GAP_MS` between valid slices (a tree
with one slice covers nothing). Nothing is extrapolated: a cell no source
covers is `UNKNOWN`. Everything else is `Q1_POSSIBLE` (a q = 1 surface may
exist; nothing more). Each sample records the tree that decided it, and the
gate records every equilibrium file it read with its sha256, so a table or a
benchmark key built on it can be checked against the files later (`shot_spans`,
which keeps no record, hashes nothing).

Files: EFIT02 in `$LABELER_ROOT/mhd/efit02/<shot>.npz`; EFIT02ER or EFIT01 in
`$LABELER_ROOT/mhd/efit_fallback/<shot>.npz` (its `tree` key), written by
`dev/mhd/corpus/fetch_eq.py`; both hold `atime` (ms) and `a_qmin`. Without an
EFIT01 fallback file, EFIT01 q_min comes from the local stores (features,
corpus or raw cache; seconds there), never fetched.

The gate is evaluated on whole-ms cells `[g, g + 1)` at their midpoints, so the
spans do not depend on the caller's sample grid.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import Paths

NO_Q1, Q1_POSSIBLE, UNKNOWN = "no_q1", "q1_possible", "unknown"
EFIT02, EFIT02ER, EFIT01 = "efit02", "efit02er", "efit01"
#: Best first: a tree decides a cell only where every earlier one is silent.
TREES = (EFIT02, EFIT02ER, EFIT01)
#: q_min at or above which a tree says "no q = 1 surface".
THRESHOLDS = {EFIT02: 1.2, EFIT02ER: 1.2, EFIT01: 2.0}
SUSTAIN_MS = 100.0
#: Two valid slices further apart than this leave the time between them unknown.
MAX_GAP_MS = 100.0
RULE = {
    "no_q1": "q_min >= the deciding tree's cut, sustained >= 100 ms on whole-ms "
    "cells; each cell is decided by the best tree covering it: EFIT02 a_qmin >= "
    "1.2, else EFIT02ER >= 1.2 (MSE-constrained), else EFIT01 >= 2.0",
    "order": list(TREES),
    "thresholds": THRESHOLDS,
    "sustain_ms": SUSTAIN_MS,
    "max_gap_ms": MAX_GAP_MS,
    "efit02": "$LABELER_ROOT/mhd/efit02/<shot>.npz (atime ms, a_qmin)",
    "efit02er": "$LABELER_ROOT/mhd/efit_fallback/<shot>.npz with tree 'efit02er'",
    "efit01": "$LABELER_ROOT/mhd/efit_fallback/<shot>.npz with tree 'efit01', "
    "else labeler.events.equilibrium.signal(shot, 'qmin'): features, corpus or "
    "raw cache, never fetched",
    "extrapolation": "none: outside a tree's valid time base, or across a gap "
    "> max_gap_ms, the tree is silent; with no tree the time is unknown",
    "files": "every equilibrium file read is recorded with its size and sha256",
}

_STATES = np.array([Q1_POSSIBLE, NO_Q1, UNKNOWN])
_SOURCES = np.array(["", *TREES])


@dataclass(frozen=True)
class Gate:
    """Per-sample gate state, the tree that decided it ("" if none), and the
    record of every equilibrium file read (`file_record`; empty for a gate
    built from arrays)."""

    t_ms: np.ndarray
    state: np.ndarray
    source: np.ndarray
    files: tuple = ()

    @property
    def no_q1(self) -> np.ndarray:
        return self.state == NO_Q1


def efit02_dir(paths: Paths) -> Path:
    return paths.root / "mhd" / "efit02"


def fallback_dir(paths: Paths) -> Path:
    return paths.root / "mhd" / "efit_fallback"


def _parked(path: Path, default_tree: str | None):
    """`(tree, time ms, a_qmin)` of a parked equilibrium file, None if absent.

    A file that exists but cannot be read raises: a silently skipped tree
    would change the gate without a trace.
    """
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            tree = str(data["tree"]) if "tree" in data.files else default_tree
            t = np.asarray(data["atime"], float)
            q = np.asarray(data["a_qmin"], float)
    except Exception as error:  # zip, pickle and key errors alike
        raise ValueError(f"unreadable equilibrium file {path}: {error}") from error
    if tree not in THRESHOLDS:
        raise ValueError(f"{path}: unknown equilibrium tree {tree!r}")
    return tree, t, q


def read_efit02(shot: int, paths: Paths):
    """(time ms, a_qmin) from the parked EFIT02 file, or None."""
    found = _parked(efit02_dir(paths) / f"{int(shot)}.npz", EFIT02)
    return found[1:] if found is not None and found[0] == EFIT02 else None


def read_fallback(shot: int, paths: Paths):
    """(tree, time ms, a_qmin) from the fallback file (EFIT02ER or EFIT01), or None."""
    return _parked(fallback_dir(paths) / f"{int(shot)}.npz", None)


def _efit01_store(shot: int, paths: Paths):
    """(time ms, q_min, store path) of EFIT01 from the local stores, or None."""
    from ..events import equilibrium
    from ..events.verify import NoDataError

    try:
        array = equilibrium.signal(int(shot), "qmin", paths, fetch=False)
    except NoDataError:  # no store holds it; any other error is a broken store
        return None
    t_ms = np.asarray(array.x, float) * 1000.0  # the stores keep seconds
    return t_ms, np.asarray(array.y, float)[0], Path(array.attrs["store"])


def read_efit01(shot: int, paths: Paths):
    """(time ms, q_min) of EFIT01 from the local stores, or None. Never fetches."""
    found = _efit01_store(shot, paths)
    return None if found is None else found[:2]


_SHA256: dict[tuple, str] = {}


def series_sha256(series) -> str:
    """sha256 of a `(time ms, q_min)` pair as float64 bytes: what the gate read,
    unchanged when a store file is rewritten for other groups."""
    digest = hashlib.sha256()
    for values in series:
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def file_record(path, tree: str, paths: Paths, series=None) -> dict:
    """`{tree, path, bytes, sha256[, series_sha256]}` of one equilibrium file;
    the path is relative to `$LABELER_ROOT` when it lies inside it."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA256:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _SHA256[key] = digest.hexdigest()
    try:
        shown = str(path.resolve().relative_to(Path(paths.root).resolve()))
    except ValueError:
        shown = str(path)
    found = {"tree": tree, "path": shown, "bytes": stat.st_size, "sha256": _SHA256[key]}
    if series is not None:
        found["series_sha256"] = series_sha256(series)
    return found


def equilibria(shot: int, paths: Paths, *, records: bool = True) -> tuple[dict, tuple]:
    """`(series, files)`: each available tree's `(time ms, q_min)` for `shot`,
    and the record of every equilibrium file read (`file_record`; empty when
    `records` is False, which skips hashing the files: a store can be 300 MB).

    EFIT02 comes from its parked file, EFIT02ER or EFIT01 from the fallback
    file (its `tree`), and EFIT01 from the local stores only when no fallback
    file holds it (the two agree sample for sample where both exist).
    """
    shot = int(shot)
    series, files = {}, []
    for path, default in (
        (efit02_dir(paths) / f"{shot}.npz", EFIT02),
        (fallback_dir(paths) / f"{shot}.npz", None),
    ):
        found = _parked(path, default)
        if found is None:
            continue
        tree, t, q = found
        if records:
            files.append(file_record(path, tree, paths, (t, q)))
        series.setdefault(tree, (t, q))
    if EFIT01 not in series:
        found = _efit01_store(shot, paths)
        if found is not None:
            series[EFIT01] = found[:2]
            if records:
                files.append(file_record(found[2], EFIT01, paths, found[:2]))
    return series, tuple(files)


def _valid(series):
    """Finite, positive samples on strictly increasing times, or None."""
    if series is None:
        return None
    t, q = (np.asarray(a, float).ravel() for a in series)
    if t.shape != q.shape:
        return None
    keep = np.isfinite(t) & np.isfinite(q) & (q > 0)
    t, q = t[keep], q[keep]
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    first = np.r_[True, np.diff(t) > 0]
    t, q = t[first], q[first]
    return (t, q) if len(t) else None


def _source_cells(lo, n, series, threshold, max_gap_ms):
    """Cover and high flags of the whole-ms cells `lo .. lo + n` for one source."""
    cover = np.zeros(n, dtype=bool)
    high = np.zeros(n, dtype=bool)
    if series is None:
        return cover, high
    t, q = series
    mid = lo + np.arange(n) + 0.5
    inside = (lo + np.arange(n) >= t[0]) & (lo + np.arange(n) + 1 <= t[-1])
    if len(t) > 1:
        right = np.clip(np.searchsorted(t, mid), 1, len(t) - 1)
        inside &= (t[right] - t[right - 1]) <= max_gap_ms
    cover[:] = inside
    high[:] = inside & (np.interp(mid, t, q) >= threshold)
    return cover, high


def gate_cells(
    efit02=None,
    efit01=None,
    *,
    efit02er=None,
    sustain_ms=SUSTAIN_MS,
    max_gap_ms=MAX_GAP_MS,
):
    """`(lo, state, source)`: codes for the whole-ms cells from `lo` on.

    `efit02`, `efit02er`, `efit01` are `(time ms, q_min)` or None. State codes
    index `_STATES` (0 possible, 1 no q = 1, 2 unknown), source codes
    `_SOURCES` (0 none, then `TREES` in order).
    """
    given = {EFIT02: efit02, EFIT02ER: efit02er, EFIT01: efit01}
    series = {name: _valid(value) for name, value in given.items()}
    present = [s for s in series.values() if s is not None]
    if not present:
        return 0, np.zeros(0, np.int8), np.zeros(0, np.int8)
    lo = int(np.ceil(min(s[0][0] for s in present)))
    hi = int(np.floor(max(s[0][-1] for s in present)))
    n = max(0, hi - lo)
    source = np.zeros(n, np.int8)
    high = np.zeros(n, dtype=bool)
    # Worst first, so each better tree paints over the ones below it.
    for code, name in reversed(list(enumerate(TREES, start=1))):
        cover, above = _source_cells(lo, n, series[name], THRESHOLDS[name], max_gap_ms)
        source[cover] = code
        high[cover] = above[cover]
    state = np.where(source > 0, 0, 2).astype(np.int8)
    for a, b in _runs(high):
        if b - a >= sustain_ms:
            state[a:b] = 1
    return lo, state, source


def _runs(mask):
    """(start, stop) index pairs of the True runs of `mask`."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def gate_spans(efit02=None, efit01=None, **kwargs):
    """`(start ms, stop ms, state, source)` runs tiling the gate's cells."""
    lo, state, source = gate_cells(efit02, efit01, **kwargs)
    if not len(state):
        return []
    key = state.astype(np.int16) * len(_SOURCES) + source
    edges = np.flatnonzero(np.diff(key)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(key)]
    return [
        (float(lo + a), float(lo + b), str(_STATES[state[a]]), str(_SOURCES[source[a]]))
        for a, b in zip(starts, stops)
    ]


def sample(t_ms, efit02=None, efit01=None, **kwargs) -> Gate:
    """The gate at each of `t_ms`: the state of the whole-ms cell holding it."""
    t_ms = np.asarray(t_ms, dtype=float)
    lo, state, source = gate_cells(efit02, efit01, **kwargs)
    cell = np.floor(np.where(np.isfinite(t_ms), t_ms, 0)).astype(np.int64) - lo
    inside = np.isfinite(t_ms) & (cell >= 0) & (cell < len(state))
    index = np.where(inside, cell, 0)
    states = np.where(inside, state[index] if len(state) else 2, 2)
    sources = np.where(inside, source[index] if len(state) else 0, 0)
    return Gate(t_ms=t_ms, state=_STATES[states], source=_SOURCES[sources])


def _arguments(series: dict) -> dict:
    return {
        "efit02": series.get(EFIT02),
        "efit01": series.get(EFIT01),
        "efit02er": series.get(EFIT02ER),
    }


def q1_gate(shot: int, paths: Paths, t_ms) -> Gate:
    """The gate of `shot` at `t_ms` from its local equilibria, with their files."""
    series, files = equilibria(shot, paths)
    return replace(sample(t_ms, **_arguments(series)), files=files)


def shot_gate(shot: int, paths: Paths, *, records: bool = True):
    """`(spans, files)`: the gate spans of `shot` (see `gate_spans`) and the
    record of every equilibrium file read (empty when `records` is False)."""
    series, files = equilibria(shot, paths, records=records)
    return gate_spans(**_arguments(series)), files


def shot_spans(shot: int, paths: Paths):
    """The gate spans of `shot` from its local equilibria (see `gate_spans`),
    without hashing the files read: for callers that keep no file record."""
    return shot_gate(shot, paths, records=False)[0]


def files_digest(records) -> str:
    """One sha256 over `(shot, tree, sha256)` records, order-free."""
    rows = sorted({(int(s), str(t), str(h)) for s, t, h in records})
    text = "\n".join(f"{s} {t} {h}" for s, t, h in rows)
    return hashlib.sha256(text.encode()).hexdigest()
