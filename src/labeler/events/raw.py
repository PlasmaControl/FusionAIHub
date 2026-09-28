"""One way to get a raw signal, whatever tier it happens to live on.

Three places are tried in order: the corpus, the fetch cache (`$LABELER_ROOT/raw`),
and a live fetch that writes the cache. A caller cannot tell which one answered
except by `attrs["tier"]`, which exists for diagnostics, not for branching.
Nothing here writes the corpus.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Paths
from ..features.store import FeatureArray
from .verify import (
    CO2_CHORDS,
    ECE_POINT,
    ECE_TREE,
    NoDataError,
    corpus_signal,
    fdp_signal,
)

#: A group whose `ydata` is this narrow carries the corpus' absent-signal
#: sentinel rather than a record.
SENTINEL_WIDTH = 1

#: ECE is 48 radiometer channels, matching the corpus group's channel order
#: so a fetched shot and a corpus shot index the same.
ECE_CHANNELS = tuple(range(1, 49))


class WindowEmptyError(NoDataError):
    """The record is on disk; the window asked for falls outside it.

    A subclass so every existing `except NoDataError` keeps catching it -
    this narrows what a caller CAN say about the failure, it does not change
    what reaches one that does not care.
    """


class UpstreamError(NoDataError):
    """A live fetch reached fdp and fdp could not answer.

    The only one of these that is a server-side fault rather than "this shot
    is not here", which is why it is the only one a caller should report as
    a bad gateway.
    """


@dataclass(frozen=True)
class FetchSpec:
    """How to fetch one corpus group live, when neither tier has it."""

    exprs: tuple[str, ...]
    via: str
    tree: str = ECE_TREE


#: Only groups listed here can be fetched. An unlisted group missing from
#: both tiers raises rather than guessing at a point name: a wrong guess
#: puts the wrong diagnostic in front of a reviewer, which is worse than a
#: refusal.
FETCH_SPECS: dict[str, FetchSpec] = {
    "co2": FetchSpec(exprs=CO2_CHORDS, via="ptdata"),
    "ece": FetchSpec(
        exprs=tuple(ECE_POINT.format(channel=c) for c in ECE_CHANNELS),
        via="mds",
    ),
    # Plasma current in amperes; 0.5 ms steps on the 2021 shots, 0.05 ms after.
    "ip": FetchSpec(exprs=("ip",), via="ptdata"),
    # The divertor D-alpha photodiode the ELM review draws. Not a corpus group,
    # so every shot's comes from here; the point name is PTDATA's, like DENR0UF.
    "pcphd03": FetchSpec(exprs=("PCPHD03",), via="ptdata"),
}

#: One lock per resolved path, so two `write_group` calls for different
#: shots don't wait on each other, but two threads targeting the same shot
#: do. Only covers this process - see `write_group`'s docstring.
_write_locks: dict[Path, threading.Lock] = {}
#: Guards `_write_locks` itself. Without this, two threads racing to write
#: the SAME new path for the first time could each create their own Lock,
#: defeating the point - both would proceed thinking they hold "the" lock.
_write_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """The lock serializing writers to one resolved path, within this process."""
    with _write_locks_guard:
        lock = _write_locks.get(path)
        if lock is None:
            lock = _write_locks[path] = threading.Lock()
        return lock


def cache_path(shot: int, *, paths: Paths | None = None) -> Path:
    """Where a fetched shot parks, in the corpus' own naming."""
    paths = Paths.from_env() if paths is None else paths
    return paths.raw_cache / f"{int(shot)}_processed.h5"


def write_group(path, group: str, times_ms, values) -> None:
    """Add one group to a corpus-layout file, atomically and additively.

    Additive because the server fetches groups one panel at a time: asking
    for `ece` on a shot whose cache already holds `co2` must not throw the
    240 MB of CO2 away. Atomic because it is doing that concurrently with
    itself, and a reader must never meet a half-written group.

    The server calling this runs synchronous routes in Starlette's
    threadpool, so "concurrently with itself" means multiple THREADS in one
    process, not separate processes - `co2` and `ece` for the same shot can
    land on different threads at once. This function serializes those
    threads against each other (see `_lock_for`), so the additivity promise
    above actually holds. It does NOT serialize separate OS processes: the
    store build hands each shot to one worker process, so its workers never
    share a file, but the server fetching a shot the build is fetching at
    the same moment could race. That is rare enough to go without file
    locking.

    `times_ms` arrives in milliseconds, the convention `corpus_signal` and
    `fdp_signal` both return, and is stored in SECONDS, the convention the
    corpus file itself uses. That conversion is the whole reason this
    function exists rather than a bare h5py call.
    """
    import h5py

    path = Path(path)
    times_ms = np.asarray(times_ms, dtype="float64")
    values = np.asarray(values, dtype="float32")
    # Validated before any file is touched, so a bad call leaves nothing on
    # disk at all - not an empty file, not a temp file.
    if values.ndim != 2:
        raise ValueError(f"{group!r} ydata must be (C, T), got {values.shape}")
    if values.shape[-1] != times_ms.shape[-1]:
        raise ValueError(
            f"{group!r} xdata {times_ms.shape} and ydata {values.shape} disagree"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # uuid4, not id(values): id() is a memory address, not a unique token -
    # it isn't unique across processes and gets reused once an object is
    # garbage collected, so two unrelated writes could end up sharing a
    # scratch name. A plain fixed ".tmp" suffix has the same collision
    # problem one level out: it's only safe here because the lock below
    # keeps writers to one path from overlapping, and that lock is
    # process-local (see the docstring), so a second process writing the
    # same path would still be free to clash on a fixed name.
    scratch = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    # Serializes the whole copy-modify-rename sequence per resolved path so
    # two threads writing different groups to the same shot can't both copy
    # the pre-write file, each add only their own group, and have the
    # loser's `scratch.replace(path)` silently discard the winner's group.
    with _lock_for(path.resolve()):
        try:
            if path.is_file():
                # h5py cannot add to a file another process may be reading, and
                # cannot shrink one in place either. Copy, add, rename.
                import shutil

                shutil.copy2(path, scratch)
            with h5py.File(scratch, "a") as f:
                if group in f:
                    del f[group]
                g = f.create_group(group)
                g.create_dataset("xdata", data=(times_ms / 1000.0).astype("float32"))
                # Chunked per channel: a whole-channel read touches contiguous
                # chunks and a time slice touches one chunk per channel. No
                # compression - the record is broadband noise that gzip barely
                # shrinks while costing minutes per shot.
                g.create_dataset(
                    "ydata",
                    data=values,
                    chunks=(1, min(values.shape[-1], 1 << 20)),
                )
            scratch.replace(path)
        finally:
            scratch.unlink(missing_ok=True)


def raw_signal(
    shot: int,
    group: str,
    *,
    channels: Sequence[int] | None = None,
    t_range: tuple[float, float] | None = None,
    paths: Paths | None = None,
) -> FeatureArray:
    """One group of one shot: `x` milliseconds, `y` `(C, T)` float32.

    Corpus, then cache, then a live fetch that fills the cache.
    """
    paths = Paths.from_env() if paths is None else paths
    for tier, root in (("corpus", paths.corpus), ("cache", paths.raw_cache)):
        try:
            array = corpus_signal(
                shot, group, channels=channels, t_range=t_range, corpus=root
            )
        except NoDataError:
            # `corpus_signal` says "not here" and "here, but your window
            # selects nothing" with the same exception, and only the first
            # of those is a reason to try the next tier. Treating the second
            # as a miss walks all the way to `_fetch`, which pulls ~240 MB
            # over PTDATA for minutes and then slices the same empty window
            # out of it - so a reviewer scrolling past the end of a shot
            # hangs the server. Asking the tier whether it holds the record
            # is a metadata-only open and settles it without reading the
            # exception's prose.
            if t_range is not None and _holds_record(shot, group, root):
                raise WindowEmptyError(
                    f"shot {int(shot)} {group!r} has no samples in "
                    f"{t_range[0]}-{t_range[1]} ms; the record is on disk, "
                    f"so this window is outside the shot"
                ) from None
            continue
        return FeatureArray(x=array.x, y=array.y, attrs={**array.attrs, "tier": tier})
    return _fetch(shot, group, channels=channels, t_range=t_range, paths=paths)


def _holds_record(shot: int, group: str, root: Path) -> bool:
    """Whether this tier holds a real record for `group` - metadata only.

    A sentinel-width group is NOT a record: the diagnostic did not run, and
    the fetch tier is exactly the right next thing to try for it.

    `raw_signal`'s tier loop stops at the first tier this says holds ANY
    record - it does not compare how much of the shot each tier's record
    covers. If a narrower record ever sat in an earlier tier than a wider
    one (say, the corpus holding less of a shot than the cache), a window
    the corpus's record does not cover but the cache's does would raise
    WindowEmptyError here instead of falling through to the tier that could
    have answered it. Unreachable today - `write_group` always writes a
    whole record - but a future partial-record cache would need to guard
    against this.
    """
    import h5py

    path = Path(root) / f"{int(shot)}_processed.h5"
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as f:
            if group not in f:
                return False
            return f[group]["ydata"].shape[-1] > SENTINEL_WIDTH
    except (OSError, KeyError):
        return False


#: What a live fetch has got to, by shot. `progress_for` is the read side;
#: the browser page polls it while a whole-shot render is in flight, because
#: the fetch itself is one HTTP request that cannot say anything until it
#: is finished. An entry exists only WHILE a fetch runs: absent means no
#: fetch is in progress for that shot, never that one finished.
_progress: dict[int, dict] = {}
_progress_guard = threading.Lock()


def progress_for(shot: int) -> dict | None:
    """`{"done", "total", "stage"}` for a fetch in flight, else None."""
    with _progress_guard:
        record = _progress.get(int(shot))
        return None if record is None else dict(record)


def _report(shot: int, done: int, total: int, stage: str) -> None:
    with _progress_guard:
        _progress[int(shot)] = {"done": int(done), "total": int(total), "stage": stage}


def _forget(shot: int) -> None:
    with _progress_guard:
        _progress.pop(int(shot), None)


#: The archive the fdp wrapper points PTDATA at is reached over the network
#: (a pelican cache), and a lookup that fails there makes the client fall
#: back to a live PTSERVER connection, which no stellar node can make. The
#: message for that is "getservbyname failed for task 'PTSERVER'", and it
#: has been seen once on a shot that fetched cleanly seconds later. One
#: retry after a short pause covers that; a shot that really is absent
#: costs one extra round trip, about ten seconds.
RETRY_DELAY_S = 2.0

#: How to restart the browser server so a live fetch can work. `fdp_signal`
#: names the notebook kernel's command in its own hint, which is the wrong
#: thing to tell someone looking at the browser page.
FDP_UI_COMMAND = "pixi run -e labelmaker labeler-verify"


def _fetch_with_one_retry(shot, spec: FetchSpec, total: int) -> FeatureArray:
    last: NoDataError | None = None
    for attempt in (1, 2):
        try:
            return fdp_signal(
                int(shot),
                list(spec.exprs),
                tree=spec.tree,
                via=spec.via,
                on_progress=lambda done, _n, expr: _report(
                    shot, done, total, f"fetching {expr}"
                ),
            )
        except NoDataError as error:
            last = error
            if attempt == 1:
                _report(shot, 0, total, "fetch failed, retrying once")
                time.sleep(RETRY_DELAY_S)
    # fdp being unreachable, or answering with nothing, is the one failure
    # here that is about the upstream rather than about this shot not
    # existing on disk. The kernel hint is cut off and replaced with the
    # server's own.
    message = str(last).split(". If this kernel")[0]
    raise UpstreamError(
        f"{message} (tried twice). This is usually a passing failure to "
        f"reach the fdp archive: pick the shot again to retry. If the "
        f"server was not started under '{FDP_UI_COMMAND}', restart it that "
        f"way."
    ) from last


def _fetch(shot, group, *, channels, t_range, paths) -> FeatureArray:
    """Fetch one group live, cache the WHOLE record, return the slice.

    The cache is never the `t_range` window. Widening a window later would
    otherwise refetch a shot that is already on disk - minutes, and hundreds
    of megabytes over the wire, for a drag of the mouse.
    """
    spec = FETCH_SPECS.get(group)
    if spec is None:
        raise NoDataError(
            f"shot {int(shot)} has no {group!r} in the corpus or the cache, "
            f"and there is no fetch route for {group!r}; known routes are "
            f"{sorted(FETCH_SPECS)}"
        )
    # One step per point plus one for writing the cache. The render that
    # follows is not counted: it is seconds against a fetch of minutes.
    total = len(spec.exprs) + 1
    _report(shot, 0, total, f"fetching {group}")
    try:
        fetched = _fetch_with_one_retry(shot, spec, total)
        _report(shot, total - 1, total, "writing the cache")
        write_group(cache_path(shot, paths=paths), group, fetched.x, fetched.y)
    finally:
        _forget(shot)
    array = corpus_signal(
        shot, group, channels=channels, t_range=t_range, corpus=paths.raw_cache
    )
    return FeatureArray(x=array.x, y=array.y, attrs={**array.attrs, "tier": "fetch"})


#: The groups each review editor draws that the corpus lacks on some shots: CO2
#: before about 197545, and PCPHD03 on every shot. Fetching needs fdp, so the
#: login node fills the cache ahead of a review (`main`).
EDITOR_FETCHES = {
    "edge_localized_mode": ("co2", "pcphd03"),
    "high_confinement_mode": ("co2",),
}


def fill_cache(shot: int, groups: Sequence[str], paths: Paths) -> dict[str, str]:
    """Fetch into the cache each group neither tier holds; say where each is."""
    where = {}
    for group in groups:
        if _holds_record(shot, group, paths.corpus):
            where[group] = "corpus"
        elif _holds_record(shot, group, paths.raw_cache):
            where[group] = "cache"
        else:
            try:
                _fetch(shot, group, channels=[0], t_range=None, paths=paths)
            except NoDataError as error:
                where[group] = f"missing: {error}"
            else:
                where[group] = "fetched"
    return where


def main(argv=None) -> int:
    """`python -m labeler.events.raw --event E`: fill the cache for E's roster."""
    import argparse
    import json
    from collections import Counter

    from .rosters import read_roster, roster_path

    parser = argparse.ArgumentParser(
        prog="python -m labeler.events.raw",
        description="Fetch what a review editor draws into the raw cache (fdp).",
    )
    parser.add_argument("--event", required=True, choices=sorted(EDITOR_FETCHES))
    parser.add_argument("--shots", type=int, nargs="+", help="default: the roster")
    parser.add_argument("--limit", type=int, default=0, help="only the first N")
    parser.add_argument(
        "--pace", type=float, default=1.0, help="seconds to wait after a fetch"
    )
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    roster = roster_path(args.event, root=paths.label_tables)
    shots = args.shots or [int(shot) for shot in read_roster(roster).shot]
    if args.limit:
        shots = shots[: args.limit]
    counts: Counter[str] = Counter()
    for shot in shots:
        where = fill_cache(shot, EDITOR_FETCHES[args.event], paths)
        print(json.dumps({"shot": shot, **where}), flush=True)
        counts.update(value.split(":")[0] for value in where.values())
        if "fetched" in where.values():
            time.sleep(args.pace)
    print(json.dumps({"shots": len(shots), **dict(sorted(counts.items()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
