"""Sub-frame features from a shot's review-store rows (round three, Part B; spec §3.2).

`features` reads the rows a spec's roles name, each the first row whose title
starts with the role's (`find_role`), so a suffix such as `CLIPPED` does not
matter. It reads them at the coarsest level that still puts `MIN_COLUMNS` of its
columns in a sub-frame (`level_for`), over the whole 10 ms frames inside a window
(`window_frames`); a column belongs to the sub-frame its centre falls in, as in
`ae.xpower.data.frame_inputs`. The roles are walked in the spec's order, each
giving its own block, since roles may share a name (the sawtooth's four "ece"):
- a trace: each channel's minimum and maximum over each sub-frame, both scaled
  0-1 on one range per channel (D64): its `robust_limits` over the window's
  sub-frames, the page's percentiles and margin. A pooled role then averages
  its channels, leaving out any with no range (dead or flat);
- an image: its bins max-pooled to the role's `groups` frequency groups (a
  store's own pooling of an image), the mean over each sub-frame, / 255;
- a modes row, `verify.mode_bytes`' codes: each n's highest level over
  `MODES_KHZ`, over its top level, the mean over each sub-frame (D55);
- an optional role adds a presence channel, 1 where its row has a value; a
  missing one gives zeros and a presence of 0.
A sub-frame no column falls in is 0, and a frame is observed when half its
sub-frames hold every required role's row. A tearing-mode store from before
d9fb57d, whose n row has the old title (`STALE_TITLE`) or no modes meta, raises
`StaleStore` (D55, D65): it is rebuilt, never read.

`store` is a store's path, or a shot's rows in memory as `panel_rows.build`
returns them, `(grid, rows, ...)`: the population shots no store is written for
(D48).
"""

from __future__ import annotations

import json

import h5py
import numpy as np

from ..ae.xpower.data import band_slice, window_frames
from ..events.panels._shared import robust_limits
from ..events.panels.neoclassical_tearing_mode import MAX_KHZ
from ..events.review.rows import LEVELS, Grid, pool
from ..scoring.frames import FRAME_MS
from . import EventSpec, Role

#: The n row's title before d9fb57d, when it held n values and no modes meta.
STALE_TITLE = "toroidal mode number n"
#: The band a modes row's levels are read over, kHz: the tearing-mode panel's.
MODES_KHZ = (0.0, MAX_KHZ)
#: Each row kind's dtype, as `rows.write` keeps it.
DTYPE = {"trace": "float32", "image": "uint8"}
#: The fewest columns a sub-frame holds at the level read, so that its min/max
#: and timing do not hang on how many coarse columns it happens to hold.
MIN_COLUMNS = 4


class StaleStore(ValueError):
    """A tearing-mode store from before d9fb57d: its n row has no modes meta."""


def level_for(dt_ms_by_level: dict[int, float], sub_ms) -> int:
    """The largest level whose sub-frame of `sub_ms` holds at least `MIN_COLUMNS`
    of its columns (`MIN_COLUMNS * dt <= sub_ms`); the finest when none does."""
    limit = sub_ms * (1 + 1e-9)
    fits = [lv for lv, dt in dt_ms_by_level.items() if MIN_COLUMNS * dt <= limit]
    return max(fits) if fits else min(dt_ms_by_level)


def find_role(rows, role: Role) -> dict | None:
    """The first of `rows` (a store's row descriptions, as `rows.meta` gives them)
    whose title starts with the role's; None when none does."""
    return next((row for row in rows if row["title"].startswith(role.title)), None)


def features(store, spec: EventSpec, window) -> tuple[np.ndarray, np.ndarray]:
    """`(C, n_sub)` float32 in [0, 1] over the whole frames inside `window`, and
    `(n_frames,)` observed. The module's docstring gives the blocks."""
    first, n = window_frames(window)
    if isinstance(store, tuple):
        grid, built = store[0], store[1]
        described = [{"name": row.name, **row.meta()} for row in built]
        found = match_roles(described, spec, "the rows in memory")
        level, cols, sub = _columns(grid, spec, first, n)
        values = [
            None if row is None else _pooled(built, row, grid, level)[..., cols]
            for row in found
        ]
    else:
        with h5py.File(store, "r") as f:
            t0, dt = float(f.attrs["t0_ms"]), float(f.attrs["dt_ms"])
            grid = Grid(t0, dt, int(f.attrs["n"]))
            described = [
                {"name": name, **json.loads(f["rows"][name].attrs["meta"])}
                for name in json.loads(f.attrs["rows"])
            ]
            found = match_roles(described, spec, str(store))
            level, cols, sub = _columns(grid, spec, first, n)
            values = [
                None if row is None else f["rows"][row["name"]][str(level)][..., cols]
                for row in found
            ]
    return _assemble(spec, found, values, sub, n)


def match_roles(described: list[dict], spec: EventSpec, where: str) -> list:
    """Each role's row description in the spec's order, None for a missing
    optional one; raises for a missing required one or a stale n row
    (`StaleStore`). `described` is a store's rows, as `features` reads them."""
    found = []
    for role in spec.roles:
        row = find_role(described, role)
        if role.kind == "modes" and (row is None or "modes" not in row):
            old = row or next(
                (r for r in described if r["title"].startswith(STALE_TITLE)), None
            )
            if old is not None:
                raise StaleStore(
                    f"{where}: its n row {old['title']!r} has no modes meta, from "
                    "before d9fb57d; the store is rebuilt, never read (D65)"
                )
        if row is None:
            if not role.optional:
                raise ValueError(f"{where}: no row's title starts with {role.title!r}")
            found.append(None)
            continue
        kind = "modes" if "modes" in row else row["kind"]
        if kind != role.kind:
            raise ValueError(
                f"{where}: the row {row['title']!r} is {kind}, the role {role.kind}"
            )
        found.append(row)
    return found


def _pooled(built, row: dict, grid: Grid, level: int) -> np.ndarray:
    """An in-memory row at `level`, as `rows.write` stores it."""
    (values,) = [r.values for r in built if r.name == row["name"]]
    values = np.asarray(values, dtype=DTYPE[row["kind"]])
    if values.shape[-1] != grid.n:
        raise ValueError(
            f"row {row['name']} has {values.shape[-1]} columns, the grid {grid.n}"
        )
    return pool(values, level, row["kind"])


def _columns(grid: Grid, spec: EventSpec, first: int, n: int):
    """The level read, the slice of its columns centred in frames `first ..
    first + n - 1`, and each such column's sub-frame."""
    level = level_for({lv: lv * grid.dt_ms for lv in LEVELS}, spec.sub_ms)
    centres = grid.t0_ms + (np.arange(-(-grid.n // level)) + 0.5) * grid.dt_ms * level
    sub = np.floor((centres - first * FRAME_MS) / spec.sub_ms).astype(np.int64)
    inside = np.flatnonzero((sub >= 0) & (sub < _subs(spec) * n))
    if not inside.size:
        return level, slice(0, 0), sub[:0]
    cols = slice(int(inside[0]), int(inside[-1]) + 1)
    return level, cols, sub[cols]


def _assemble(spec: EventSpec, found, values, sub, n: int):
    subs = _subs(spec)
    width = subs * n
    counts = np.bincount(sub, minlength=width)
    covered = np.ones(width, dtype=bool)
    blocks = []
    for role, row, v in zip(spec.roles, found, values, strict=True):
        if row is None:
            blocks.append(np.zeros((_width(role), width)))
            continue
        if role.kind == "trace":
            block, seen = _trace(role, row, np.asarray(v, dtype=np.float64), sub, width)
        else:
            if role.kind == "image":
                per_column = _groups(v, role.groups) / 255.0
            else:
                per_column = _modes(v, row, role.groups)
            block = _reduce(per_column, sub, width, np.add, 0.0) / np.maximum(counts, 1)
            seen = counts > 0
        if role.optional:
            block = np.vstack([block, seen])
        else:
            covered &= seen
        blocks.append(block)
    observed = covered.reshape(n, subs).mean(axis=1) >= 0.5
    return np.concatenate(blocks).astype(np.float32), observed


def _reduce(values, sub, width: int, ufunc, fill: float) -> np.ndarray:
    """`ufunc` over each sub-frame's columns (`sub`, sorted); `fill` where none."""
    out = np.full((*values.shape[:-1], width), fill, dtype=np.float64)
    if sub.size:
        starts = np.flatnonzero(np.r_[True, np.diff(sub) > 0])
        out[..., sub[starts]] = ufunc.reduceat(values, starts, axis=-1)
    return out


def _trace(role: Role, row: dict, values, sub, width: int):
    """A trace's block, its minima then its maxima, and the sub-frames it has a
    value in. `values` is the store's `(2, C, n)`, each column's min and max."""
    if not role.pooled and row["n_channels"] != role.channels:
        raise ValueError(
            f"the row {row['title']!r} has {row['n_channels']} channels, "
            f"the role {role.channels}"
        )
    low = _reduce(values[0], sub, width, np.fmin, np.nan)
    high = _reduce(values[1], sub, width, np.fmax, np.nan)
    seen = np.isfinite(high).any(axis=0)
    both = np.concatenate([low, high], axis=-1)
    # One range for both extremes (D64); x only sizes robust_limits' mask.
    lo, hi = (v[:, None] for v in robust_limits(np.zeros(both.shape[-1]), both, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(hi > lo, np.clip((both - lo) / (hi - lo), 0.0, 1.0), np.nan)
    if role.pooled:
        scaled = _channel_mean(scaled)
    low, high = np.split(np.nan_to_num(scaled, nan=0.0), 2, axis=-1)
    return np.concatenate([low, high]), seen


def _channel_mean(values) -> np.ndarray:
    """`(1, n)`: the mean of the channels with a value; nan where none has one."""
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)[None]


def _groups(values, groups: int) -> np.ndarray:
    """`(groups, n)`: an image's bins, max-pooled to `groups` frequency groups."""
    values = np.asarray(values)
    if len(values) < groups:
        raise ValueError(f"{len(values)} bins cannot make {groups} groups")
    split = np.array_split(values, groups)
    return np.stack([block.max(axis=0) for block in split]).astype(np.float64)


def _modes(codes, row: dict, groups: int) -> np.ndarray:
    """`(K, n)`: each n's highest level over `MODES_KHZ`, 0-1 (D55)."""
    meta = row["modes"]
    k = len(meta["n"])
    if k != groups:
        raise ValueError(f"the row {row['title']!r} codes {k} n, the role {groups}")
    band = band_slice(row["y0"], row["dy"], row["n_y"], MODES_KHZ)
    level, index = np.divmod(np.asarray(codes)[band].astype(np.int64), k)
    top = int(meta["levels"]) - 1
    by_n = [np.where(index == i, level, 0).max(axis=0) for i in range(k)]
    return np.stack(by_n) / top


def _width(role: Role) -> int:
    """The channels a role gives, its presence included."""
    if role.kind == "trace":
        base = 2 * (1 if role.pooled else role.channels)
    else:
        base = role.groups
    return base + int(role.optional)


def _subs(spec: EventSpec) -> int:
    """The sub-frames in a frame."""
    subs = FRAME_MS / spec.sub_ms
    if subs < 1 or subs != round(subs):
        raise ValueError(
            f"{spec.method}: {spec.sub_ms} ms sub-frames do not divide "
            f"a {FRAME_MS} ms frame"
        )
    return round(subs)
