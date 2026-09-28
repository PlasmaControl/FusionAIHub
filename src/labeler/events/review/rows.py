"""Review rows on one time grid, stored as a pyramid a window can be read from fast.

A store file holds every row of one shot. Column `i` of the grid spans
`[t0_ms + i * dt_ms, t0_ms + (i + 1) * dt_ms)`. An image row is `uint8`
`(n_y, n)`, row 0 the lowest y; a trace row is `float32` `(2, n_channels, n)`,
each column's minimum and then its maximum, so pooling never hides a spike.
Each row is kept at three pooling factors, `rows/<name>/{1,8,64}`, and a read
takes the coarsest one that still gives the page a column per pixel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import h5py
import numpy as np

from ...config import atomic_path

LEVELS = (1, 8, 64)
CHUNK_COLUMNS = 512


@dataclass(frozen=True)
class Grid:
    t0_ms: float
    dt_ms: float
    n: int


@dataclass(frozen=True)
class ImageRow:
    name: str
    title: str
    values: np.ndarray
    y0: float
    dy: float
    y_units: str
    z_lo: float
    z_hi: float
    z_units: str
    band: tuple[float, float] | None = None
    #: A modes row's `{"n", "colours", "levels"}`: its bytes are
    #: `verify.mode_bytes`' codes, not a scale.
    modes: dict | None = None
    kind = "image"

    def meta(self) -> dict:
        meta = {
            "kind": self.kind,
            "title": self.title,
            "n_y": int(self.values.shape[0]),
            "y0": float(self.y0),
            "dy": float(self.dy),
            "y_units": self.y_units,
            "z_lo": float(self.z_lo),
            "z_hi": float(self.z_hi),
            "z_units": self.z_units,
        }
        if self.band is not None:
            meta["band"] = [float(v) for v in self.band]
        if self.modes is not None:
            meta["modes"] = self.modes
        return meta


@dataclass(frozen=True)
class TraceRow:
    name: str
    title: str
    values: np.ndarray
    y_units: str = ""
    legend: list[str] = field(default_factory=list)
    hlines: list[float] = field(default_factory=list)
    kind = "trace"

    def meta(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "n_channels": int(self.values.shape[1]),
            "y_units": self.y_units,
            "legend": list(self.legend),
            "hlines": [float(v) for v in self.hlines],
        }


def pool(values: np.ndarray, level: int, kind: str) -> np.ndarray:
    """Pool the last axis by `level`: max for images, (min, max) for traces.

    The last block pads with its own last column. NaN is skipped unless a
    whole block is NaN.
    """
    if level == 1:
        return values
    pad = -values.shape[-1] % level
    if pad:
        edge = np.repeat(values[..., -1:], pad, axis=-1)
        values = np.concatenate([values, edge], axis=-1)
    blocks = values.reshape(*values.shape[:-1], -1, level)
    if kind == "image":
        return blocks.max(axis=-1)
    low = np.fmin.reduce(blocks[0], axis=-1)
    high = np.fmax.reduce(blocks[1], axis=-1)
    return np.stack([low, high])


def write(path, grid: Grid, rows, **info) -> None:
    """Write one shot's rows; `info` becomes file attributes (JSON if not scalar)."""
    with atomic_path(path) as tmp, h5py.File(tmp, "w") as f:
        f.attrs.update(
            {
                "t0_ms": float(grid.t0_ms),
                "dt_ms": float(grid.dt_ms),
                "n": int(grid.n),
                "rows": json.dumps([row.name for row in rows]),
            }
        )
        for key, value in info.items():
            scalar = isinstance(value, (str, int, float))
            f.attrs[key] = value if scalar else json.dumps(value)
        for row in rows:
            dtype = "uint8" if row.kind == "image" else "float32"
            values = np.asarray(row.values, dtype=dtype)
            if values.shape[-1] != grid.n:
                raise ValueError(
                    f"row {row.name} has {values.shape[-1]} columns, the grid {grid.n}"
                )
            group = f.create_group(f"rows/{row.name}")
            group.attrs["meta"] = json.dumps(row.meta())
            for level in LEVELS:
                data = pool(values, level, row.kind)
                group.create_dataset(
                    str(level),
                    data=data,
                    chunks=(*data.shape[:-1], min(CHUNK_COLUMNS, data.shape[-1])),
                    compression="gzip",
                    compression_opts=1,
                )


def meta(path, hide=frozenset()) -> dict:
    """The grid, the time range and every row's description, less `hide`."""
    with h5py.File(path, "r") as f:
        t0, dt, n = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"]), int(f.attrs["n"])
        rows = [
            {"name": name, **json.loads(f["rows"][name].attrs["meta"])}
            for name in _names(f, hide)
        ]
    grid = {"t0": t0, "dt": dt, "n": n}
    return {"grid": grid, "t_range": [t0, t0 + n * dt], "rows": rows}


def read_window(
    path, t0: float, t1: float, cols: int, hide=frozenset()
) -> tuple[bytes, dict]:
    """Every row but `hide` over `t0`-`t1` ms at about `cols` columns, as bytes.

    Rows follow each other in store order: images as `uint8` `(n_y, k)`,
    traces as little-endian `float32` `(2, n_channels, k)`. The dict is the
    grid actually returned, `{"t0", "t1", "n": k}`.
    """
    with h5py.File(path, "r") as f:
        g0, dt, n = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"]), int(f.attrs["n"])
        i0 = max(0, math.floor((t0 - g0) / dt))
        i1 = min(n, math.ceil((t1 - g0) / dt))
        if i1 <= i0:
            end = g0 + n * dt
            raise ValueError(f"{t0}-{t1} ms is outside the record {g0}-{end} ms")
        level = max((lv for lv in LEVELS if (i1 - i0) // lv >= cols), default=1)
        j0, j1 = i0 // level, math.ceil(i1 / level)
        factor = math.ceil((j1 - j0) / cols)
        k = math.ceil((j1 - j0) / factor)
        parts = []
        for name in _names(f, hide):
            group = f["rows"][name]
            kind = json.loads(group.attrs["meta"])["kind"]
            block = pool(group[str(level)][..., j0:j1], factor, kind)
            parts.append(block.astype("uint8" if kind == "image" else "<f4").tobytes())
    start = g0 + j0 * level * dt
    return b"".join(parts), {"t0": start, "t1": start + k * factor * level * dt, "n": k}


def _names(f, hide) -> list[str]:
    return [name for name in json.loads(f.attrs["rows"]) if name not in hide]
