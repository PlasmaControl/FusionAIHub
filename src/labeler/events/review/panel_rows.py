"""Any event's panels as review rows.

Heatmaps become image rows, scaled over their 1st-99.5th percentile unless
the panel pins `zmin`/`zmax`, or coded by `verify.mode_bytes` when the panel
has `modes`; lines become trace rows. Every row goes onto
one grid at the finest panel spacing, but no finer than 0.05 ms: finer data
is min/max-binned, coarser data is interpolated (lines) or nearest-sampled
(heatmaps). Explicit indicator bins remain steps over their full intervals,
including isolated valid bins between invalid bins.
"""

from __future__ import annotations

import math

import numpy as np

from ...config import Paths
from .. import panels
from ..verify import NoDataError, mode_bytes
from .rows import Grid, ImageRow, TraceRow

FINEST_DT_MS = 0.05
PERCENTILES = (1.0, 99.5)


def build(event: str, shot: int, paths: Paths) -> tuple[Grid, list, dict]:
    built = [p for p in panels.build(event, int(shot), paths=paths) if len(p.x)]
    if not built:
        raise NoDataError(f"no panels for shot {int(shot)}")
    xs = [
        _centred(p.x, p.z.shape[1]) if p.kind == "heatmap" else np.asarray(p.x, float)
        for p in built
    ]
    grid = _grid(xs)
    rows = [
        (_image if p.kind == "heatmap" else _trace)(f"p{i}", p, x, grid)
        for i, (p, x) in enumerate(zip(built, xs))
    ]
    params = {"finest_dt_ms": FINEST_DT_MS, "percentiles": list(PERCENTILES)}
    metadata = {f"p{i}": p.metadata for i, p in enumerate(built) if p.metadata}
    if metadata:
        params["panel_metadata"] = metadata
    return grid, rows, {"params": params}


def _centred(x, n: int) -> np.ndarray:
    """Cell centres from `n + 1` edges; centres pass through."""
    x = np.asarray(x, dtype=float)
    return (x[:-1] + x[1:]) / 2 if len(x) == n + 1 else x


def _spacing(x) -> float:
    return float(np.median(np.diff(x))) if len(x) > 1 else math.inf


def _grid(xs) -> Grid:
    lo = min(float(x[0]) for x in xs)
    hi = max(float(x[-1]) for x in xs)
    finest = min(_spacing(x) for x in xs)
    dt = max(FINEST_DT_MS, finest) if math.isfinite(finest) else 1.0
    n = max(1, math.ceil((hi - lo) / dt) + 1)
    return Grid(lo - dt / 2, dt, n)


def _columns(x, grid: Grid) -> np.ndarray:
    cols = np.floor((x - grid.t0_ms) / grid.dt_ms).astype(np.int64)
    return np.clip(cols, 0, grid.n - 1)


def _bin(values, cols, n: int, reduce, fill) -> np.ndarray:
    """Reduce the samples falling in each column; empty columns get `fill`."""
    out = np.full((*values.shape[:-1], n), fill, dtype=values.dtype)
    starts = np.flatnonzero(np.r_[True, np.diff(cols) > 0])
    out[..., cols[starts]] = reduce.reduceat(values, starts, axis=-1)
    return out


def _fine(x, grid: Grid) -> bool:
    return len(x) < 2 or _spacing(x) <= grid.dt_ms


def _centres(grid: Grid) -> np.ndarray:
    return grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms


def _trace(name: str, panel, x, grid: Grid) -> TraceRow:
    y = np.atleast_2d(np.asarray(panel.y, dtype=np.float32))
    if (panel.metadata or {}).get("trace_style") == "step":
        values = _steps(panel, y, grid)
    elif _fine(x, grid):
        cols = _columns(x, grid)
        low = _bin(y, cols, grid.n, np.fmin, np.nan)
        high = _bin(y, cols, grid.n, np.fmax, np.nan)
        values = np.stack([low, high])
    else:
        centres = _centres(grid)
        line = np.stack([np.interp(centres, x, channel) for channel in y])
        outside = (centres < x[0] - grid.dt_ms / 2) | (centres > x[-1] + grid.dt_ms / 2)
        line[:, outside] = np.nan
        values = np.stack([line, line]).astype(np.float32)
    legend = list(panel.legend) if panel.legend else [f"ch {i}" for i in range(len(y))]
    return TraceRow(name, panel.title, values, y_units=panel.ylabel, legend=legend,
                    hlines=[float(v) for v in panel.hlines])


def _steps(panel, y, grid: Grid) -> np.ndarray:
    """Min/max of bin values overlapping each column; invalid bins stay gaps."""
    starts = np.asarray(panel.metadata["bin_start_ms"], dtype=float)
    ends = np.asarray(panel.metadata["bin_end_ms"], dtype=float)
    if starts.shape != panel.x.shape or ends.shape != starts.shape:
        raise ValueError("step intervals disagree with panel clock")
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("step intervals must be finite")
    if np.any(ends <= starts):
        raise ValueError("step intervals must have positive width")
    values = np.full((2, len(y), grid.n), np.nan, dtype=np.float32)
    # A fine bin that does not contain a grid centre must still contribute.
    # Use overlap with column edges, retaining extrema when several bins pool.
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        lo = max(0, math.floor((start - grid.t0_ms) / grid.dt_ms))
        hi = min(grid.n, math.ceil((end - grid.t0_ms) / grid.dt_ms))
        if hi <= lo:
            continue
        value = y[:, i, None]
        values[0, :, lo:hi] = np.fmin(values[0, :, lo:hi], value)
        values[1, :, lo:hi] = np.fmax(values[1, :, lo:hi], value)
    return values


def _limits(z, zmin=None, zmax=None) -> tuple[float, float]:
    finite = z[np.isfinite(z)]
    lo, hi = np.percentile(finite, PERCENTILES) if finite.size else (0.0, 1.0)
    lo = float(lo if zmin is None else zmin)
    hi = float(hi if zmax is None else zmax)
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _image(name: str, panel, x, grid: Grid) -> ImageRow:
    z = np.asarray(panel.z, dtype=float)
    lo, hi = _limits(z, panel.zmin, panel.zmax)
    modes = None
    if panel.modes is not None:
        q = mode_bytes(z, panel.modes, panel.mode_colours, lo, hi)
        n = sorted(panel.mode_colours)
        modes = {
            "n": [int(v) for v in n],
            "levels": 256 // len(n),
            "colours": [panel.mode_colours[v] for v in n],
        }
    else:
        scaled = np.rint((np.nan_to_num(z, nan=lo) - lo) * 255 / (hi - lo))
        q = np.clip(scaled, 0, 255).astype(np.uint8)
    if _fine(x, grid):
        values = _bin(q, _columns(x, grid), grid.n, np.maximum, 0)
    else:
        centres = _centres(grid)
        nearest = np.clip(np.searchsorted(x, centres), 1, len(x) - 1)
        nearest -= (centres - x[nearest - 1]) < (x[nearest] - centres)
        values = q[:, nearest]
        values[:, np.abs(x[nearest] - centres) > _spacing(x) / 2] = 0
    y = _centred(panel.y, z.shape[0])
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    band = tuple(panel.bands[0]) if panel.bands else None
    return ImageRow(name, panel.title, values, y0=float(y[0]), dy=dy,
                    y_units=panel.ylabel, z_lo=lo, z_hi=hi, z_units="", band=band,
                    modes=modes)
