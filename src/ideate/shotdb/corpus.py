"""Read a shot from the FAITH corpus: `<corpus_dir>/<shot>_processed.h5`.

The corpus is 16,909 files (shots 185601-204999) written for the foundation model, not for us,
and four of its properties decide almost everything in this module. All four are measured, and
all four are silent failures if ignored:

* **Seconds.** `xdata` is a seconds axis. Everything else in ideate -- features, segments, the
  schema, every threshold in configs/ideate -- is milliseconds. The conversion happens here, at
  the boundary, exactly once.
* **`(C, 1)` placeholders.** A diagnostic the corpus did not record for a shot is still written,
  as a one-sample placeholder: `ydata` of shape `(C, 1)` and `xdata` of shape `(1,)`, all NaN.
  The time axis is the LAST one for a waveform group, so the presence test is `shape[-1] >= 2`.
  Testing `shape[0] > 1` instead -- the channel axis -- is what the old shotsearch manifest did,
  and it counted every placeholder as a populated group.
* **A trailing pad sample.** The 500 kHz groups (mhr, ece, co2, bes, mirnov) are 2^k+1 samples
  long and the last sample is NaN on every channel, while `xdata[-1]` is finite. Left in, it
  poisons any percentile or normalisation computed over the array. `read` strips it; `coverage`
  does not, because the diagnostic did cover that instant.
* **2.3 % of the files do not open.** Truncated writes. h5py raises `OSError`; that becomes
  `ShotFailed`, which is a different fact from "this diagnostic was not recorded" (`Unavailable`)
  and has to stay different all the way into the database's coverage table.

`build` is wired onto this reader through the `Reader` protocol, never by importing it directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from .reader import ShotFailed, Unavailable

SUFFIX = "_processed.h5"

# The smallest number of samples on a group's time axis that means "this diagnostic recorded
# something". One sample is the corpus's placeholder for "it did not"; zero does not occur.
MIN_SAMPLES = 2


def shot_of(path: Path) -> int:
    """The shot number a corpus filename carries: `204988_processed.h5` -> 204988."""
    return int(Path(path).name[: -len(SUFFIX)])


def describe(shape: Sequence[int]) -> tuple[str, int, int]:
    """`(kind, n_channels, n_samples)` for a `ydata` shape, without reading it.

    A waveform group is `(C, n)` -- or, for a single channel, `(n,)`. A video group is
    `(C, n, H, W)`: its time axis is the SECOND one, so `shape[-1]` (the frame width) is not the
    sample count and, applied to a video group, the placeholder test would be nonsense.
    """
    if len(shape) > 2:
        return "video", int(shape[0]), int(shape[1])
    if len(shape) == 2:
        return "signal", int(shape[0]), int(shape[1])
    return "signal", 1, int(shape[0]) if shape else 0


def open_h5(path: Path) -> h5py.File:
    """Open a corpus file read-only without taking an HDF5 lock.

    `locking=False` for the same reason `legacy_raw.open_h5` uses it: the corpus lives on a
    shared GPFS filesystem under another group's ownership, and taking a lock on a file someone
    else is writing turns a read into a BlockingIOError. We only ever read.

    Raises `ShotFailed`, never `OSError`: a caller of this module is deciding what to do about a
    shot, not about a file handle.
    """
    path = Path(path)
    try:
        return h5py.File(path, "r", locking=False)
    except OSError as e:
        raise ShotFailed(f"{path}: {e}") from e


class CorpusReader:
    """The FAITH corpus as a `Reader`: one file per shot, groups of unnamed channels, in ms.

    Each call opens the shot's file and closes it again, so it holds no handles between calls and
    is safe to fork with (a `multiprocessing` worker inheriting an open HDF5 handle is a known way
    to corrupt reads). The cost is one open per group read -- 0.3 ms warm against GPFS, against
    97 ms for one 500 kHz group's samples -- which is the right trade for a reader whose
    caller reads a handful of groups per shot, and the wrong one for a caller reading thirty; that
    caller should get a batched read here rather than a cached handle.
    """

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)

    def __repr__(self) -> str:
        return f"CorpusReader({str(self.corpus_dir)!r})"

    # -------------------------------------------------------------------------- the file

    def path(self, shot: int) -> Path:
        return self.corpus_dir / f"{int(shot)}{SUFFIX}"

    def available(self, shot: int) -> bool:
        try:
            with open_h5(self.path(shot)):
                return True
        except ShotFailed:
            return False

    def groups(self, shot: int) -> list[str]:
        with open_h5(self.path(shot)) as f:
            return sorted(name for name in f if self._present(f[name]))

    @staticmethod
    def _present(g) -> bool:
        if not isinstance(g, h5py.Group) or "ydata" not in g or "xdata" not in g:
            return False
        return describe(g["ydata"].shape)[2] >= MIN_SAMPLES

    # -------------------------------------------------------------------------- the samples

    def read(
        self, shot: int, group: str, channels: Sequence[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """`group` as `(t_ms float64 (n,), y float32 (C, n))`, in the channel order asked for.

        The trailing pad is stripped: samples at the end where EVERY returned channel is
        non-finite are dropped, and `t_ms` is truncated with them so the two stay aligned. The
        rule is written over the returned channels rather than over the fast-group names, so a
        group that grows a pad -- or loses one -- needs no list edited here.
        """
        path = self.path(shot)
        with open_h5(path) as f:
            x, y = self._datasets(f, path, group)
            kind, n_channels, n_samples = describe(y.shape)
            if n_samples < MIN_SAMPLES:
                raise Unavailable(f"{path}: {group} is a ({n_channels}, {n_samples}) placeholder")
            if kind == "video":
                raise ValueError(f"{path}: {group} is a video group; read() returns waveforms")
            idx = self._channel_index(path, group, n_channels, channels)
            t_ms = np.asarray(x[()], dtype=np.float64).ravel() * 1000.0
            if t_ms.size != n_samples:
                raise ShotFailed(
                    f"{path}: {group} has {t_ms.size} timestamps for {n_samples} samples"
                )
            vals = self._rows(y, idx, n_channels)
        keep = self._recorded_length(vals)
        return t_ms[:keep], vals[:, :keep]

    def coverage(self, shot: int, group: str) -> tuple[float, float]:
        """`(t0_ms, t1_ms)` from `xdata` alone -- two scalar reads, whatever the group's size.

        Deliberately NOT the span of what `read` returns: the pad sample's timestamp is finite
        and real, and this span is the answer to "was the diagnostic even looking", which is what
        keeps "no phenomenon here" apart from "no data here".
        """
        path = self.path(shot)
        with open_h5(path) as f:
            x, _ = self._datasets(f, path, group)
            if x.shape[0] < MIN_SAMPLES:
                raise Unavailable(f"{path}: {group} has a {x.shape} time axis (placeholder)")
            return float(x[0]) * 1000.0, float(x[-1]) * 1000.0

    # -------------------------------------------------------------------------- internals

    @staticmethod
    def _datasets(f: h5py.File, path: Path, group: str) -> tuple[h5py.Dataset, h5py.Dataset]:
        g = f.get(group)
        if not isinstance(g, h5py.Group) or "xdata" not in g or "ydata" not in g:
            raise Unavailable(f"{path}: no group {group!r}")
        return g["xdata"], g["ydata"]

    @staticmethod
    def _channel_index(
        path: Path, group: str, n_channels: int, channels: Sequence[int] | None
    ) -> list[int]:
        if channels is None:
            return list(range(n_channels))
        idx = [int(c) for c in channels]
        bad = [c for c in idx if not 0 <= c < n_channels]
        if bad:
            raise IndexError(f"{path}: {group} has {n_channels} channels, asked for {bad}")
        return idx

    @staticmethod
    def _rows(y: h5py.Dataset, idx: list[int], n_channels: int) -> np.ndarray:
        """The requested channel rows as one contiguous `(len(idx), n)` float32 array.

        h5py wants an increasing, duplicate-free selection, so the read is done sorted and the
        rows are put back into the caller's order afterwards -- one selection either way, and the
        corpus datasets are contiguous, so a row is a cheap strided read.
        """
        if len(idx) == n_channels and idx == list(range(n_channels)):
            vals = y[()]
        else:
            order = sorted(set(idx))
            got = np.atleast_2d(y[order, :])
            vals = got[[order.index(c) for c in idx], :]
        return np.ascontiguousarray(np.atleast_2d(vals), dtype=np.float32)

    @staticmethod
    def _recorded_length(vals: np.ndarray) -> int:
        """How many leading samples to keep: everything up to the last one where any returned
        channel is finite.

        The one exception is a selection with no finite sample anywhere -- a channel that
        recorded nothing at all. Stripping that to length zero would turn "recorded nothing" into
        "has no timebase", and the two are answered differently downstream (a NaN statistic with
        a coverage span, versus an empty array whose `t[0]` raises). So it is returned whole.
        """
        recorded = np.isfinite(vals).any(axis=0)
        if not recorded.any():
            return vals.shape[1]
        return int(np.flatnonzero(recorded)[-1]) + 1
