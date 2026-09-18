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

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import Paths
from ..features.store import FeatureArray
from .verify import NoDataError, corpus_signal

#: A group whose `ydata` is this narrow carries the corpus' absent-signal
#: sentinel rather than a record.
SENTINEL_WIDTH = 1


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
    scratch = path.with_name(f".{path.name}.{id(values):x}.tmp")
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
    # Filled in by Task 3. Until then, say so rather than returning
    # something a panel would silently plot.
    raise NoDataError(
        f"shot {int(shot)} has no {group!r} in the corpus or the cache, and "
        f"there is no fetch route for {group!r}"
    )
