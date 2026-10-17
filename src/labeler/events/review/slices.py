"""2-D slices at a time, for the dock beside the rows (plan D.B).

Two kinds, per shot, read from local caches only (this module never fetches):

* `equilibrium`: `$LABELER_ROOT/equilibria/<shot>.npz`, written by
  scripts/labeler/detach_default_fetch.py from EFIT02, or EFIT01 when EFIT02 is
  missing or sparse (its `source` says which). The separatrix, X-points, strike
  points, wall and psi_N contours of the EFIT slice nearest the asked time.
* `inversion`: `$LABELER_ROOT/round4/detach/inversions/<shot>.npz`, the
  producer's TangTV emission inversions (`frames` (t, Z, R) on `radii` and
  `elevation`). The nearest frame as a grayscale PNG on the shot's robust range.

A slice further than `MAX_GAP_MS` from the asked time is missing, as in
`geometry.nearest`. Loaded files are cached by path and modification time.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure

from ...config import Paths
from .geometry import MAX_GAP_MS, real

#: The kinds a shot may have, with the label the dock shows.
KINDS = {"equilibrium": "EFIT equilibrium", "inversion": "TangTV inversion"}
#: The flux surfaces drawn: the core, the separatrix and two SOL surfaces.
LEVELS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.02, 1.05)
#: The inversion's grey scale: these percentiles of the shot's finite values.
PERCENTILES = (1.0, 99.5)
#: X-points then strike points, as AEQDSK (r, z) node pairs.
XPOINTS = (("rxpt1", "zxpt1"), ("rxpt2", "zxpt2"))
STRIKES = (("rvsod", "zvsod"), ("rvsid", "zvsid"), ("rvsou", "zvsou"),
           ("rvsiu", "zvsiu"))
#: Where a real DIII-D position lies (m); EFIT's sentinels fall outside or are
#: excluded by `geometry.real`.
R_BOUNDS, Z_BOUNDS = (0.8, 2.6), (-1.7, 1.7)


class SliceMissing(LookupError):
    """The shot has no slice of this kind on disk, or none near the time."""


def path(kind: str, shot: int, paths: Paths) -> Path:
    if kind == "equilibrium":
        return paths.root / "equilibria" / f"{int(shot)}.npz"
    if kind == "inversion":
        return paths.root / "round4/detach/inversions" / f"{int(shot)}.npz"
    raise ValueError(f"unknown slice kind {kind!r}")


def available(shot: int, paths: Paths) -> list[dict]:
    """What the shot has on disk: `[{kind, label, n_t, t_range_ms}]`.

    Reads each file's clock only (an inversion's frames are not loaded), and
    leaves out a kind whose file is unreadable, so one broken file never fails
    the listing; `/api/slice` names the error for that kind.
    """
    kinds = []
    for kind, label in KINDS.items():
        file = path(kind, shot, paths)
        try:
            stat = file.stat()
            t, tree = _clock(kind, str(file), stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError, KeyError):
            continue
        if not len(t):
            continue
        if kind == "equilibrium" and tree != "EFIT02":
            label = f"{label} ({tree})"
        kinds.append({
            "kind": kind, "label": label, "n_t": len(t),
            "t_range_ms": [float(t[0]), float(t[-1])],
        })
    return kinds


def equilibrium(shot: int, t_ms: float, paths: Paths) -> dict:
    """The EFIT slice nearest `t_ms`, in metres (the dock's JSON)."""
    data = _load("equilibrium", shot, paths)
    i = _nearest(data["t_ms"], t_ms, "equilibrium", shot)
    return _equilibrium_slice(data["key"], i)


def inversion(shot: int, t_ms: float, paths: Paths) -> tuple[bytes, dict]:
    """The inversion frame nearest `t_ms`: PNG bytes (row 0 = z_max, column 0 =
    r_min) and the X-Slice record."""
    data = _load("inversion", shot, paths)
    i = _nearest(data["t_ms"], t_ms, "inversion", shot)
    return _inversion_png(data["key"], i), {
        "t_ms": round(float(data["t_ms"][i]), 3),
        "r_range": data["r_range"],
        "z_range": data["z_range"],
        "vmin": data["vmin"],
        "vmax": data["vmax"],
        "units": "a.u.",
        "source": data["source"],
    }


def _nearest(times, t_ms, kind, shot) -> int:
    gaps = np.where(np.isfinite(times), np.abs(times - t_ms), np.inf)
    i = int(np.argmin(gaps))
    if not gaps[i] <= MAX_GAP_MS:
        raise SliceMissing(
            f"shot {shot}: no {kind} slice within {MAX_GAP_MS:g} ms of {t_ms:g} ms "
            f"(nearest {float(times[i]):g} ms)"
        )
    return i


def _load(kind: str, shot: int, paths: Paths) -> dict:
    file = path(kind, shot, paths)
    try:
        stat = file.stat()
    except FileNotFoundError:
        raise SliceMissing(f"shot {shot} has no {kind} on disk") from None
    try:
        data = _read(kind, str(file), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, KeyError) as error:
        raise SliceMissing(f"shot {shot}: its {kind} file is unreadable ({error})") from None
    if not len(data["t_ms"]):
        raise SliceMissing(f"shot {shot}: its {kind} file holds no slices")
    return data


@lru_cache(maxsize=64)
def _clock(kind: str, file: str, mtime_ns: int, size: int) -> tuple[np.ndarray, str]:
    """A file's slice times and (for an equilibrium) its tree, nothing else."""
    with np.load(file, allow_pickle=False) as source:
        if kind == "equilibrium":
            return np.asarray(source["t_ms"], dtype=float), str(source["tree"])
        return np.asarray(source["times_ms"], dtype=float), ""


@lru_cache(maxsize=8)
def _read(kind: str, file: str, mtime_ns: int, size: int) -> dict:
    """One file's arrays; the key (path, mtime, size) keys the slice caches."""
    key = (kind, file, mtime_ns, size)
    with np.load(file, allow_pickle=False) as source:
        data = {name: source[name] for name in source.files}
    if kind == "equilibrium":
        t = np.asarray(data["t_ms"], dtype=float)
        return {**data, "key": key, "t_ms": t, "tree": str(data["tree"])}
    frames = data["frames"]
    t = np.asarray(data["times_ms"], dtype=float)
    r = np.asarray(data["radii"], dtype=float)
    z = np.asarray(data["elevation"], dtype=float)
    if frames.shape != (len(t), len(z), len(r)):
        raise ValueError(f"{file}: frames {frames.shape} disagree with the axes")
    finite = frames[np.isfinite(frames)].astype(np.float32)
    vmin, vmax = 0.0, 1.0
    if finite.size:
        vmin, vmax = (float(v) for v in np.percentile(finite, PERCENTILES))
    return {
        "key": key, "t_ms": t, "frames": frames,
        "flip_rows": bool(z[-1] > z[0]), "flip_columns": bool(r[0] > r[-1]),
        "r_range": _span(r),
        "z_range": _span(z),
        "vmin": vmin, "vmax": vmax if vmax > vmin else vmin + 1.0,
        "source": Path(str(data.get("source", "not recorded"))).name,
    }


def _span(values) -> list[float] | None:
    """[min, max] to 4 decimals; None for no values."""
    if not np.size(values):
        return None
    return [round(float(np.min(values)), 4), round(float(np.max(values)), 4)]


def _points(pairs) -> list[list[float]]:
    return [[round(float(r), 4), round(float(z), 4)] for r, z in pairs]


@lru_cache(maxsize=512)
def _equilibrium_slice(key, i: int) -> dict:
    data = _read(*key)
    # The fetch writes `nbbbs` only when EFIT has it; NaN padding is dropped below.
    n = int(data["nbbbs"][i]) if "nbbbs" in data else data["rbbbs"].shape[1]
    boundary = np.column_stack([data["rbbbs"][i, :n], data["zbbbs"][i, :n]])
    boundary = boundary[np.isfinite(boundary).all(axis=1)]
    wall = np.asarray(data["lim"], dtype=float)
    wall = wall[np.isfinite(wall).all(axis=1)]

    def located(pairs):
        found = []
        for rn, zn in pairs:
            r, z = float(data[rn][i]), float(data[zn][i])
            if real(r, *R_BOUNDS) and real(z, *Z_BOUNDS):
                found.append((r, z))
        return _points(found)

    have_grid = all(k in data for k in ("r", "z", "psirz", "ssimag", "ssibry"))
    # The psi grid's extent, else the wall's, else the boundary's (None if none).
    outline = wall if len(wall) else boundary
    r_axis = np.asarray(data["r"], dtype=float) if have_grid else outline[:, 0]
    z_axis = np.asarray(data["z"], dtype=float) if have_grid else outline[:, 1]
    return {
        "t_ms": round(float(data["t_ms"][i]), 3),
        "source": data["tree"],
        "source_rule": str(data["source"]),
        "r_range": _span(r_axis),
        "z_range": _span(z_axis),
        "boundary": _points(boundary),
        "xpoints": located(XPOINTS),
        "strikes": located(STRIKES),
        "wall": _points(wall),
        "levels": _levels(data, i, wall) if have_grid else [],
    }


def _levels(data, i, wall) -> list[dict]:
    """psi_N contours, cut where they leave the wall; [] when the slice is bad."""
    axis, edge = float(data["ssimag"][i]), float(data["ssibry"][i])
    psi = np.asarray(data["psirz"][i], dtype=float)  # (z, r)
    if not (np.isfinite(axis) and np.isfinite(edge) and edge != axis):
        return []
    psin = (psi - axis) / (edge - axis)
    if not np.isfinite(psin).all():
        return []
    r, z = np.asarray(data["r"], dtype=float), np.asarray(data["z"], dtype=float)
    levels = []
    for level in LEVELS:
        paths = []
        for contour in measure.find_contours(psin, level):
            points = np.column_stack([
                np.interp(contour[:, 1], np.arange(len(r)), r),
                np.interp(contour[:, 0], np.arange(len(z)), z),
            ])
            inside = (
                measure.points_in_poly(points, wall) if len(wall) >= 3
                else np.ones(len(points), dtype=bool)
            )
            if not inside.all() and np.allclose(points[0], points[-1]):
                # A closed contour starts at its first outside point, so the
                # run through its seam stays one path.
                first = int(np.argmin(inside[:-1]))
                points = np.roll(points[:-1], -first, axis=0)
                inside = np.roll(inside[:-1], -first)
            # Split at the wall: each run of inside points is one path.
            edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
            for start, stop in zip(edges[::2], edges[1::2], strict=True):
                if stop - start >= 2:
                    paths.append(_points(points[start:stop]))
        levels.append({"psi_n": level, "paths": paths})
    return levels


@lru_cache(maxsize=256)
def _inversion_png(key, i: int) -> bytes:
    data = _read(*key)
    frame = np.asarray(data["frames"][i], dtype=np.float32)
    if data["flip_rows"]:
        frame = frame[::-1]
    if data["flip_columns"]:
        frame = frame[:, ::-1]
    scaled = (frame - data["vmin"]) / (data["vmax"] - data["vmin"])
    pixels = np.clip(np.nan_to_num(scaled, nan=0.0) * 255 + 0.5, 0, 255)
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()
