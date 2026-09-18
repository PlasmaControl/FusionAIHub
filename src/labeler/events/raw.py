"""One way to get a raw signal, whatever tier it happens to live on.

Three places are tried in order: the corpus, the project's fetch cache, and
a live fetch that writes the cache. A caller cannot tell which one answered
except by looking at `attrs["tier"]`, which exists for diagnostics and for
the promote command, not for branching.

The split between the two on-disk roots is about what the storage is FOR.
EKOLEMEN holds the long-term bulk raw record and has the capacity for it.
The project directory has room but is meant for temporary and smaller
things, so a fetch lands there as scratch. Nothing here ever writes the
corpus; `promote` does, deliberately and by hand.
"""

from __future__ import annotations

import threading
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


def groups_in(path) -> set[str]:
    """The group names one corpus-layout file holds; empty if it is absent."""
    import h5py

    path = Path(path)
    if not path.is_file():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


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
    above actually holds. It does NOT serialize across separate OS
    processes: this module's own CLI running alongside the server, or a
    future multi-worker deployment, could still race. No file locking is
    used to close that gap - this deployment is single-process, and adding
    it now would guard against a case that doesn't exist yet.

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
            continue
        return FeatureArray(
            x=array.x, y=array.y, attrs={**array.attrs, "tier": tier}
        )
    return _fetch(shot, group, channels=channels, t_range=t_range, paths=paths)


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
    fetched = fdp_signal(
        int(shot), list(spec.exprs), tree=spec.tree, via=spec.via
    )
    write_group(cache_path(shot, paths=paths), group, fetched.x, fetched.y)
    array = corpus_signal(
        shot, group, channels=channels, t_range=t_range, corpus=paths.raw_cache
    )
    return FeatureArray(x=array.x, y=array.y, attrs={**array.attrs, "tier": "fetch"})


#: The 32 groups a complete corpus shot holds, from
#: `src/tokamak_foundation_model/data/config/modalities/modalities.yaml`.
#: `promote` compares against this to decide whether a cache entry is a
#: whole shot or a verification fetch of one diagnostic.
CORPUS_GROUPS: tuple[str, ...] = (
    "mhr", "ece", "co2",
    "gas", "gas_raw", "ech", "ech_raw", "pin", "tin",
    "d_alpha", "mse", "ts_core_density", "ts_core_temp",
    "ts_tan_density", "ts_tan_temp", "cer_rot", "filterscopes",
    "ip", "betan", "pinj", "tinj", "li", "q95", "qmin", "qpsi",
    "kappa", "tritop", "tribot", "aminor", "rmaxis", "zmaxis", "wmhd",
)


def promote(shot: int, *, partial: bool = False, paths: Paths | None = None) -> Path:
    """Move a cached shot into the corpus, where it lives long-term.

    Manual and separate from anything the reviewer clicks: this moves
    hundreds of megabytes, and a Save that did it as a side effect would be
    a Save that can half-fail.

    The default refuses an incomplete shot. Training loaders glob
    `*_processed.h5` in the corpus root, and a verification fetch
    materialises the one group a panel asked for - so a partial file there
    is one those globs hand to training with the rest of the groups
    missing.
    """
    paths = Paths.from_env() if paths is None else paths
    source = cache_path(shot, paths=paths)
    if not source.is_file():
        raise FileNotFoundError(f"shot {int(shot)} is not in {paths.raw_cache}")
    target = paths.corpus / source.name
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; promote never overwrites a corpus "
            f"shot. Inspect both and remove one by hand."
        )
    present = groups_in(source)
    if not partial and len(present) < len(CORPUS_GROUPS):
        missing = sorted(set(CORPUS_GROUPS) - present)
        raise ValueError(
            f"shot {int(shot)} holds {len(present)} of {len(CORPUS_GROUPS)} "
            f"groups; missing {', '.join(missing)}. Training globs "
            f"*_processed.h5 in the corpus root and would read this as a "
            f"whole shot. Pass --partial if that is what you want."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # `replace` is atomic within a filesystem and falls back to copy+unlink
    # across one, which the cache and the corpus may well be.
    try:
        source.replace(target)
    except OSError:
        import shutil

        shutil.copy2(source, target)
        source.unlink()
    return target


def clean(*, paths: Paths | None = None) -> int:
    """Delete the whole fetch cache; return the bytes recovered."""
    import shutil

    paths = Paths.from_env() if paths is None else paths
    root = paths.raw_cache
    if not root.is_dir():
        return 0
    freed = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    shutil.rmtree(root)
    return freed


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m labeler.events.raw promote 178642 [--partial]` / `clean`."""
    import argparse

    parser = argparse.ArgumentParser(prog="labeler.events.raw")
    sub = parser.add_subparsers(dest="command", required=True)
    move = sub.add_parser("promote", help="move a cached shot into the corpus")
    move.add_argument("shot", type=int)
    move.add_argument(
        "--partial", action="store_true",
        help="promote a shot that does not hold all 32 groups",
    )
    sub.add_parser("clean", help="delete the whole fetch cache")
    args = parser.parse_args(argv)

    paths = Paths.from_env()
    if args.command == "clean":
        freed = clean(paths=paths)
        print(f"removed {freed / 1e9:.2f} GB from {paths.raw_cache}")
        return 0
    try:
        landed = promote(args.shot, partial=args.partial, paths=paths)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"error: {error}")
        return 1
    print(f"promoted {args.shot} -> {landed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
