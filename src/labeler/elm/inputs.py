"""The ELM detector's inputs: filterscope D-alpha and line density on a 10 kHz grid.

`elm-ours` reads the three filterscope channels FS02-FS04 (50 kS/s) and the
two fast interferometer chords DENV2F and DENV3F (100 kS/s), the records
`scripts/labeler/elmo_fetch.py` stored under
`$LABELER_ROOT/benchmarks/elm/elmo/signals/<shot>.npz`. It does not read BES,
so it covers every shot that has the two fetched records, which the 46 review
shots without BES also have.

Every record is brought to one grid of 0.1 ms cells, `[GRID0_MS + 0.1 i,
GRID0_MS + 0.1 (i + 1))`, so a 1 ms cell is ten grid cells and a 50 ms bin is
five hundred, and both start on whole milliseconds of the shot's clock. The
filterscopes take the largest sample of a cell (an ELM is a burst; the maximum
keeps its height), the interferometers the mean.

**Channels** (`CHANNELS`, 11 rows):

* `fs02`, `fs03`, `fs04`: `(log10(max(x, 1e12)) - 15) / 1.5` of the cell maximum.
  D-alpha spans four decades between shots and is offset near zero at low
  signal, so a clipped logarithm is used rather than a per-shot scale: an ELM is
  an additive step in it whatever the shot's gain.
* `fs02_c`, `fs03_c`, `fs04_c`: that channel minus its running median over
  `BASELINE_S` (0.5 s), the contrast against the shot's own baseline.
* `ne2f`, `ne3f`: the line density in units of 1e14 m^-2, clipped to
  `[-3, 12]`. A chord whose median magnitude exceeds `BAD_DENSITY` is a failed
  digitiser (194445 reads 1e18) and is set to zero.
* `ne2f_hp`, `ne3f_hp`: ten times the density minus its running mean over
  `HP_S` (0.2 s): the density drop an ELM crash leaves.
* `valid`: 1 where both records have samples in the cell.

A cell without samples is zero in every row but `valid`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

GRID0_MS = -50.0
DT_MS = 0.1
CELLS_PER_MS = 10
FS_FLOOR = 1e12
FS_CENTRE = 15.0
FS_SCALE = 1.5
DENSITY_UNIT = 1e14
DENSITY_RANGE = (-3.0, 12.0)
BAD_DENSITY = 1e16
BASELINE_S = 0.5
HP_S = 0.2
HP_GAIN = 10.0
BLOCK_CELLS = 100  # 10 ms, the grid the running baselines are computed on
CHANNELS = (
    "fs02",
    "fs03",
    "fs04",
    "fs02_c",
    "fs03_c",
    "fs04_c",
    "ne2f",
    "ne3f",
    "ne2f_hp",
    "ne3f_hp",
    "valid",
)
N_CHANNELS = len(CHANNELS)
FS_LEVEL = (0, 1, 2)
FS_CONTRAST = (3, 4, 5)
DENSITY = (6, 7)
DENSITY_HP = (8, 9)
VALID = 10


def block_reduce(
    t_ms: np.ndarray, y: np.ndarray, n: int, how: str
) -> tuple[np.ndarray, np.ndarray]:
    """`y` `(C, T)` at times `t_ms`, reduced into the `n` cells of the grid.

    Returns the `(C, n)` reduction (`how` is `max` or `mean`) and the `(n,)`
    count of samples in each cell; a cell with none is 0.
    """
    cell = np.floor((np.asarray(t_ms, dtype=np.float64) - GRID0_MS) / DT_MS + 1e-6)
    cell = cell.astype(np.int64)
    keep = (cell >= 0) & (cell < n)
    cell, y = cell[keep], np.asarray(y)[:, keep]
    out = np.zeros((y.shape[0], n), dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    if cell.size == 0:
        return out, count
    first = np.flatnonzero(np.r_[True, np.diff(cell) != 0])
    where = cell[first]
    count[where] = np.diff(np.r_[first, cell.size])
    if how == "max":
        out[:, where] = np.maximum.reduceat(y, first, axis=1)
    else:
        out[:, where] = (
            np.add.reduceat(y.astype(np.float64), first, axis=1) / count[where]
        )
    return out, count


def grid_length(t_fs_ms: np.ndarray, t_int_ms: np.ndarray) -> int:
    """Whole milliseconds, in cells, from `GRID0_MS` to the shorter record's end."""
    end = min(float(t_fs_ms[-1]), float(t_int_ms[-1]))
    return int(np.floor(end - GRID0_MS)) * CELLS_PER_MS


def _blocks(x: np.ndarray, how: str) -> np.ndarray:
    """`x` `(C, N)` cut into `BLOCK_CELLS` blocks (the tail is dropped), reduced."""
    n = x.shape[1] // BLOCK_CELLS
    b = x[:, : n * BLOCK_CELLS].reshape(x.shape[0], n, BLOCK_CELLS)
    return np.median(b, axis=2) if how == "median" else b.mean(axis=2)


def _upsample(blocks: np.ndarray, n: int) -> np.ndarray:
    """A block series back to `n` cells, linear between block centres."""
    centres = (np.arange(blocks.shape[1]) + 0.5) * BLOCK_CELLS
    cells = np.arange(n) + 0.5
    return np.stack([np.interp(cells, centres, row) for row in blocks])


def running_median(x: np.ndarray, seconds: float) -> np.ndarray:
    """The running median of `x` `(C, N)` over `seconds`, on 10 ms blocks."""
    size = max(3, round(seconds * 1000 / (BLOCK_CELLS * DT_MS)) | 1)
    blocks = median_filter(_blocks(x, "median"), size=(1, size), mode="nearest")
    return _upsample(blocks, x.shape[1])


def running_mean(x: np.ndarray, seconds: float) -> np.ndarray:
    """The running mean of `x` `(C, N)` over `seconds`, on 10 ms blocks."""
    size = max(3, round(seconds * 1000 / (BLOCK_CELLS * DT_MS)) | 1)
    blocks = uniform_filter1d(_blocks(x, "mean"), size=size, axis=1, mode="nearest")
    return _upsample(blocks, x.shape[1])


def channels(
    t_fs_ms: np.ndarray,
    fs: np.ndarray,
    t_int_ms: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    """The `(N_CHANNELS, n)` float32 input of one shot from its two records.

    `fs` is `(3, T)` (FS02-FS04) at `t_fs_ms`, `density` `(2, T')` (DENV2F, DENV3F)
    at `t_int_ms`, both in ms on the shot's clock.
    """
    n = grid_length(t_fs_ms, t_int_ms)
    fmax, fcount = block_reduce(t_fs_ms, fs, n, "max")
    dmean, dcount = block_reduce(t_int_ms, density, n, "mean")
    out = np.zeros((N_CHANNELS, n), dtype=np.float32)
    covered = (fcount > 0) & (dcount > 0)
    level = np.log10(np.maximum(fmax, FS_FLOOR))
    # a cell with no sample holds the clip value; give the baselines the median instead
    level[:, fcount == 0] = np.median(level[:, fcount > 0], axis=1, keepdims=True)
    contrast = level - running_median(level, BASELINE_S)
    out[list(FS_LEVEL)] = (level - FS_CENTRE) / FS_SCALE
    out[list(FS_CONTRAST)] = contrast / FS_SCALE
    d = dmean / DENSITY_UNIT
    bad = np.median(np.abs(dmean[:, dcount > 0]), axis=1) > BAD_DENSITY
    d[:, dcount == 0] = np.median(d[:, dcount > 0], axis=1, keepdims=True)
    hp = d - running_mean(d, HP_S)
    d = np.clip(d, *DENSITY_RANGE)
    d[bad], hp[bad] = 0.0, 0.0
    out[list(DENSITY)] = d
    out[list(DENSITY_HP)] = np.clip(HP_GAIN * hp, -10.0, 10.0)
    out[:, ~covered] = 0.0
    out[VALID] = covered
    return out


def read_channels(path: Path) -> np.ndarray:
    """`channels` of the fetched record `<shot>.npz`."""
    with np.load(path) as z:
        return channels(
            z["t_fs_ms"], z["filterscopes"], z["t_int_ms"], z["interferometer"]
        )


def valid_intervals(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where a shot's input is valid, as `(starts, stops)` in ms of the shot's clock.

    Only stretches of at least one whole millisecond count; they are closed to
    1 ms cells so every bin built on them lies on the 1 ms grid.
    """
    ms = np.asarray(valid, dtype=bool)[: len(valid) // CELLS_PER_MS * CELLS_PER_MS]
    ms = ms.reshape(-1, CELLS_PER_MS).all(axis=1)
    edge = np.flatnonzero(np.diff(np.r_[0, ms.astype(np.int8), 0]))
    return GRID0_MS + edge[0::2].astype(float), GRID0_MS + edge[1::2].astype(float)
