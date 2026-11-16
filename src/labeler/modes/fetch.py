"""The inputs omnimode's mode labeller reads, from a shot's staged FDP view (StellarAI all-shot run).

`scripts/labeler/mode_all_shots.py` stages each shot's native files on the login node and
extracts them here on a CPU node with no network. What is written is what omnimode reads, in
its formats, so its `batch` command runs on the shot unchanged:

* ``<corpus>/<shot>_processed.h5``: the corpus groups `mirnov` (29 rows, the 14 of `ROWS`
  filled, the rest NaN as a dead channel) and `ip` (1 row, amperes), written by
  `labeler.events.raw.write_group` (xdata float32 seconds, ydata float32, one-row chunks).
  Mirnov is placed on one 500 kHz grid (`assemble`): omnimode refuses a record at any other
  rate. The whole finite record is kept, as TokEye's masks need it.
* ``<inputs>/<tree>/<shot>.npz`` for `EQ_TREES` (the "er" ladder's EFIT rungs), in the format
  of ``dev/mhd/corpus/fetch_eq.py`` (`eq_record`); the ECE setup is NaN (the ermap-magrot
  variant reads no ECE).
* ``<inputs>/rot/<shot>.npz``: ZIPFIT01 and ZIPFIT02 TROTFIT, as ``dev/mhd/fusion/fetch_rot.py``.

The readers (`read_ptdata`, `read_mds`) need the fdp environment and `use_local_stage` called
first in a fresh process (the PTDATA engine reads its configuration once per process).
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

from labeler.events.raw import MIRNOV_POINTS, MIRNOV_WIDTH, write_group

#: The corpus `mirnov` rows the ermap-magrot labeller reads: the wall pair (0, 8), the
#: poloidal pairs M3 (1, 9), M5 (2, 10) and M89 (3, 11), and the six toroidal probes
#: (omnimode `modes.tracks.TOR`, `modes.labelmodel.PAIRS`).
ROWS = (0, 1, 2, 3, 8, 9, 10, 11, 15, 16, 18, 20, 21, 22)
POINTS = {row: MIRNOV_POINTS[row] for row in ROWS}
#: The toroidal probes, the reference first: the grid follows the first one present.
TOROIDAL = (15, 16, 18, 20, 21, 22)
FS_HZ = 500_000.0
#: A reference record whose rate is off 500 kHz by more than this is refused (`RateError`).
RATE_TOL = 0.01
#: A step longer than this many median steps splits a record into segments (no interpolation
#: across it: the grid is NaN there).
GAP_STEPS = 1.5
IP_POINT = "ip"
EQ_TREES = ("efit02er", "efit02", "efitrt2")
ROT_TREES = ("zipfit01", "zipfit02")
TREES = EQ_TREES + ROT_TREES
G_NODES = (
    "gtime",
    "psirz",
    "ssimag",
    "ssibry",
    "rmaxis",
    "zmaxis",
    "r",
    "z",
    "qpsi",
    "fpol",
)
A_NODES = ("atime", "qmin", "q95")


class RateError(ValueError):
    """The reference probe is not sampled at 500 kHz."""


def corpus_path(corpus, shot: int) -> Path:
    return Path(corpus) / f"{int(shot)}_processed.h5"


def segments(t_ms: np.ndarray) -> list[tuple[int, int]]:
    """`[(i0, i1)]`: the stretches of `t_ms` with no step over `GAP_STEPS` median steps."""
    t = np.asarray(t_ms, dtype=np.float64)
    if t.size < 2:
        return [(0, t.size)] if t.size else []
    step = np.diff(t)
    cuts = np.flatnonzero(step > GAP_STEPS * float(np.median(step))) + 1
    edges = [0, *cuts.tolist(), t.size]
    return [(a, b) for a, b in itertools.pairwise(edges) if b > a]


def place(t_ms, y, grid0_ms: float, n: int, fs_hz: float = FS_HZ) -> np.ndarray:
    """One record on the grid `grid0_ms + k / fs_hz` (k < n), float32: linear interpolation
    inside each of its segments (`segments`), NaN elsewhere. A record on the grid's own times
    comes back unchanged."""
    t = np.asarray(t_ms, dtype=np.float64)
    v = np.asarray(y, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float32)
    dt = 1000.0 / fs_hz
    for a, b in segments(t):
        if b - a < 2:
            continue
        k0 = max(0, int(np.ceil((t[a] - grid0_ms) / dt - 1e-6)))
        k1 = min(n - 1, int(np.floor((t[b - 1] - grid0_ms) / dt + 1e-6)))
        if k1 < k0:
            continue
        grid = grid0_ms + np.arange(k0, k1 + 1) * dt
        out[k0 : k1 + 1] = np.interp(grid, t[a:b], v[a:b])
    return out


def assemble(
    records: dict, fs_hz: float = FS_HZ
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """`(times_ms, Y, rows)`: the fetched Mirnov records `{row: (t_ms, y)}` on one grid.

    The grid runs at `fs_hz` from the first to the last sample of the reference probe (the
    first of `TOROIDAL` present); every row is `place`d on it. `Y` is float32
    (len(rows), T), `rows` sorted. ValueError without a toroidal probe; `RateError` when the
    reference's median step is off `fs_hz` by more than `RATE_TOL`."""
    ref = next((row for row in TOROIDAL if row in records), None)
    if ref is None:
        raise ValueError("no toroidal Mirnov probe")
    t_ref = np.asarray(records[ref][0], dtype=np.float64)
    if t_ref.size < 2:
        raise ValueError(f"reference probe {POINTS[ref]} holds {t_ref.size} samples")
    rate = 1000.0 / float(np.median(np.diff(t_ref)))
    if abs(rate - fs_hz) > RATE_TOL * fs_hz:
        raise RateError(
            f"reference probe {POINTS[ref]} at {rate:.0f} Hz, not {fs_hz:.0f}"
        )
    dt = 1000.0 / fs_hz
    n = round((t_ref[-1] - t_ref[0]) / dt) + 1
    times = t_ref[0] + np.arange(n) * dt
    rows = sorted(records)
    Y = np.stack([place(*records[row], t_ref[0], n, fs_hz) for row in rows])
    return times, Y, rows


def write_corpus(path, times_ms, Y, rows, ip_t_ms, ip_a) -> int:
    """The shot's corpus file, `mirnov` and `ip`, written into `<path>.part` and renamed, so
    a file at `path` is complete. Returns its size in bytes."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    part.unlink(missing_ok=True)
    write_group(part, "mirnov", times_ms, Y, rows=rows, n_rows=MIRNOV_WIDTH)
    write_group(part, "ip", ip_t_ms, np.asarray(ip_a, dtype=np.float32)[None, :])
    os.replace(part, path)
    return path.stat().st_size


def orient(psirz: np.ndarray, r, z, ssimag, rmaxis, zmaxis) -> np.ndarray:
    """`psirz` as (time, z, r): the order whose flux at the axis matches `ssimag`
    (``scripts/labeler/detach_fetch_efit.py``'s `orient`, which fetch_eq.py uses)."""
    nt, nz, nr = len(ssimag), len(z), len(r)
    candidates = {}
    if psirz.shape == (nt, nz, nr):
        candidates["tzr"] = psirz
    if psirz.shape == (nt, nr, nz):
        candidates["trz"] = np.swapaxes(psirz, 1, 2)
    if psirz.shape == (nr, nz, nt):
        candidates["rzt"] = np.transpose(psirz, (2, 1, 0))
    if psirz.shape == (nz, nr, nt):
        candidates["zrt"] = np.transpose(psirz, (2, 0, 1))
    best, best_err = None, np.inf
    picks = np.unique(np.linspace(0, nt - 1, 9).astype(int))
    for grid in candidates.values():
        err = 0.0
        for k in picks:
            iz = int(np.argmin(np.abs(z - zmaxis[k])))
            ir = int(np.argmin(np.abs(r - rmaxis[k])))
            err += abs(grid[k, iz, ir] - ssimag[k]) / (abs(ssimag[k]) + 1e-6)
        if err < best_err:
            best, best_err = grid, err
    if best is None or best_err / len(picks) > 0.2:
        raise ValueError(f"cannot orient psirz {psirz.shape} (error {best_err:.3g})")
    return best


def profile(a: np.ndarray, nt: int) -> np.ndarray:
    """(time, psi) from either axis order; |q| grows toward the edge on average (fetch_eq.py)."""
    if a.shape[0] == nt and a.shape[1] != nt:
        return a
    if a.shape[1] == nt and a.shape[0] != nt:
        return a.T
    return a if np.nanmean(np.abs(a[:, -1]) - np.abs(a[:, 0])) > 0 else a.T


def eq_record(g: dict, a: dict, tree: str) -> dict:
    """One EFIT tree's npz fields, as fetch_eq.py's `fetch_tree` writes them, from the G-EQDSK
    nodes `g` (`G_NODES`) and A-EQDSK nodes `a` (`A_NODES`). ValueError under two slices."""
    times = np.asarray(g["gtime"], dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"{tree}: {times.size} time slice(s)")
    nt = times.size
    r, z = np.asarray(g["r"], dtype=float), np.asarray(g["z"], dtype=float)
    if r.ndim == 2:
        r, z = r[0], z[0]
    psirz = orient(np.asarray(g["psirz"]), r, z, g["ssimag"], g["rmaxis"], g["zmaxis"])
    return {
        "times": times,
        "psirz": psirz.astype("float32"),
        "r": r,
        "z": z,
        "ssimag": np.asarray(g["ssimag"]),
        "ssibry": np.asarray(g["ssibry"]),
        "rmaxis": np.asarray(g["rmaxis"]),
        "qpsi": profile(np.asarray(g["qpsi"]), nt),
        "fpol": profile(np.asarray(g["fpol"]), nt),
        "atime": np.asarray(a["atime"], dtype=float),
        "a_qmin": np.asarray(a["qmin"]),
        "a_q95": np.asarray(a["q95"]),
        "tree": np.array(tree),
        "ece_freq": np.array([np.nan]),
        "ece_zh": np.array([np.nan]),
    }


def save_npz(path, fields: dict) -> None:
    """`np.savez_compressed`, written beside `path` and renamed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **fields)
    os.replace(tmp, path)


# --- readers: fdp environment, a staged shot, a fresh process --------------------------------


def use_local_stage(shot_dir) -> None:
    """Point MDSplus and the PTDATA engine at a staged shot (no network); call before toksearch
    or MDSplus is imported (``scripts/data_fetching_fdp/fetch_raw_fdp.py``'s `use_local_stage`)."""
    shot_dir = Path(shot_dir)
    for key in [k for k in os.environ if k.endswith("_path")]:
        os.environ.pop(key)
    for key in (
        "PTDATA_JSON_INDEX_DIR",
        "PTDATA_JSON_INDEX_PATTERN",
        "PTDATA_PLUGIN_LIB",
    ):
        os.environ.pop(key, None)
    prefix = Path(sys.prefix)
    os.environ.update(
        {
            "SYS_D3": str(shot_dir / "ptdata"),
            "SYS_D3_DELIM": ";",
            "PTDATA_LOC": "1",
            "PTDATA_LIBRARY": str(prefix / "lib/libd3.so"),
            "MDS_PATH": str(prefix / "tdi"),
            "D3DATA": "yes",
            "default_tree_path": str(shot_dir / "mdsplus/shots"),
        }
    )


def read_ptdata(name: str, shot: int) -> tuple[np.ndarray, np.ndarray]:
    """`(t_ms float64, y float32)` of one PTDATA point (`PtDataSignal`, ical 1, as the corpus
    build reads it)."""
    from toksearch_d3d import PtDataSignal

    rec = PtDataSignal(name).fetch(int(shot))
    unit = (rec.get("units") or {}).get("times", "ms")
    if unit != "ms":
        raise ValueError(f"{name} times in {unit!r}, not ms")
    return np.asarray(rec["times"], dtype=np.float64), np.asarray(
        rec["data"], dtype=np.float32
    )


def read_mds(expr: str, tree: str, shot: int, dims=None) -> dict:
    from toksearch import MdsSignal

    sig = MdsSignal(expr, tree, dims=list(dims)) if dims else MdsSignal(expr, tree)
    return sig.fetch(int(shot))


def read_eq(tree: str, shot: int) -> dict:
    """One EFIT tree's npz fields (`eq_record`) from the staged tree."""
    g = {k: np.asarray(read_mds(rf"\{tree}::{k}", tree, shot)["data"]) for k in G_NODES}
    a = {
        k: np.asarray(read_mds(rf"\{tree}::top.results.aeqdsk:{k}", tree, shot)["data"])
        for k in A_NODES
    }
    return eq_record(g, a, tree)


def read_rot(shot: int) -> tuple[dict, dict[str, str]]:
    """`(fields, errors)`: TROTFIT of each ZIPFIT tree (w, rho, t float32), as fetch_rot.py."""
    out, errors = {}, {}
    for tree in ROT_TREES:
        k = tree[-2:]
        try:
            rec = read_mds(
                rf"\{tree.upper()}::TOP.PROFILES.TROTFIT",
                tree.upper(),
                shot,
                dims=("rho", "times"),
            )
            out[f"w{k}"] = np.asarray(rec["data"], np.float32)
            out[f"rho{k}"] = np.asarray(rec["rho"], np.float32)
            out[f"t{k}"] = np.asarray(rec["times"], np.float32)
        except Exception as error:  # noqa: BLE001 - a missing tree must not stop the shot
            errors[tree] = f"{type(error).__name__}: {error}"[:200]
    return out, errors
