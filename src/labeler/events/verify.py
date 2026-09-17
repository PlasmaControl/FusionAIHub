"""The human review surface: show a shot's signals, take back corrections.

This module knows nothing about plasma physics. A notebook computes whatever
arrays settle its own phenomenon - a spectrogram, a set of raw channels, a
scalar trace - and hands them over as `Panel`s. Deciding which traces settle
which phenomenon is case-by-case work that does not generalise, so it lives
in each category's `verification.ipynb` rather than here.

What is shared is everything else: reading the corpus without loading it,
stacking the panels on one time axis, turning a dragged range into an
interval, and writing the two files a review produces - the corrections under
`review/` and the roster row in `shots.csv`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..config import Paths
from ..features.store import FeatureArray

#: A corpus group whose `ydata` is this narrow carries the absent-signal
#: sentinel - `(C, 1)` - rather than a record. `resolve_corpus` writes it for
#: a diagnostic that did not run, and it is not data.
SENTINEL_WIDTH = 1


class NoDataError(RuntimeError):
    """A signal a panel asked for is not on disk for this shot."""


def corpus_path(shot: int, *, corpus: Path | None = None) -> Path:
    """The corpus file for one shot."""
    corpus = Paths.from_env().corpus if corpus is None else Path(corpus)
    return corpus / f"{int(shot)}_processed.h5"


def corpus_signal(
    shot: int,
    group: str,
    *,
    channels: Sequence[int] | None = None,
    t_range: tuple[float, float] | None = None,
    corpus: Path | None = None,
) -> FeatureArray:
    """One corpus group, sliced: `x` in milliseconds, `y` as `(C, T)` float32.

    ECE is `(48, 3.1e6)` and CO2 `(4, 4.5e6)`; neither is ever read whole.
    `t_range` is milliseconds and is applied with an h5py slice, so a 100 ms
    window costs a 100 ms read. `channels` selects rows by index.
    """
    import h5py

    path = corpus_path(shot, corpus=corpus)
    if not path.is_file():
        raise NoDataError(
            f"shot {int(shot)} has no corpus file at {path}. The corpus covers "
            f"185601-204999; a shot outside it has to be fetched."
        )
    with h5py.File(path, "r") as f:
        if group not in f:
            raise NoDataError(f"shot {int(shot)} has no {group!r} group in {path}")
        x = f[group]["xdata"]
        y = f[group]["ydata"]
        if y.shape[-1] <= SENTINEL_WIDTH:
            raise NoDataError(
                f"shot {int(shot)} carries the absent-signal sentinel for "
                f"{group!r}: ydata is {y.shape}, so this diagnostic did not run"
            )
        start, stop = 0, x.shape[0]
        if t_range is not None:
            times = np.asarray(x, dtype="float64") * 1000.0
            start = int(np.searchsorted(times, t_range[0], side="left"))
            stop = int(np.searchsorted(times, t_range[1], side="right"))
            if stop <= start:
                raise NoDataError(
                    f"shot {int(shot)} {group!r} has no samples in "
                    f"{t_range[0]}-{t_range[1]} ms"
                )
        rows = list(range(y.shape[0])) if channels is None else list(channels)
        values = np.stack([y[row, start:stop] for row in rows]).astype("float32")
        times_ms = np.asarray(x[start:stop], dtype="float64") * 1000.0
    return FeatureArray(
        x=times_ms,
        y=values,
        attrs={
            "group": group,
            "shot": str(int(shot)),
            "channels": ",".join(str(row) for row in rows),
            "units": "ms",
        },
    )
