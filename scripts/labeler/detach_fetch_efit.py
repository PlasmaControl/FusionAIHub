#!/usr/bin/env python
"""Fetch the EFIT flux maps the detachment figure overlays on the camera views.

Run on the login node under the fdp wrapper, never in a SLURM job:

    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/detach_fetch_efit.py --shots 189057 189100 --pace 1

Per shot, one ``<shot>.npz`` under ``$LABELER_ROOT/round4/detach/efit`` with the
EFIT01 G-EQDSK record: ``gtime_ms``, ``r`` and ``z`` (grid, m), ``psirz`` as
``(time, z, r)``, ``ssimag``, ``ssibry``, ``rmaxis``, ``zmaxis`` and the boundary
``rbbbs`` / ``zbbbs`` (padded to the longest, NaN-filled).

The 2-D nodes come back with an unlabelled axis order, and the grid is square
(65 x 65) so the order cannot be read off the shape; it is decided here from the
flux itself: with the right order the extremum of ``psirz`` sits at the magnetic
axis and equals ``ssimag``. An authentication failure stops the run (exit 3).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

NODES = (
    "psirz",
    "ssimag",
    "ssibry",
    "rmaxis",
    "zmaxis",
    "gtime",
    "rgrid",
    "zgrid",
    "rbbbs",
    "zbbbs",
)
AUTH_WORDS = ("auth", "token", "login", "credential", "401", "403", "permission")


def out_dir() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach" / "efit"


def orient(psirz: np.ndarray, r, z, ssimag, rmaxis, zmaxis) -> np.ndarray:
    """`psirz` as (time, z, r): the order whose flux at the axis matches `ssimag`."""
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


def fetch(shot: int) -> dict:
    from labeler.features import resolve_fdp

    got = {}
    for name in NODES:
        expr = rf"\efit01::top.results.geqdsk:{name}"
        got[name] = np.asarray(resolve_fdp._fetch_mds(expr, "efit01", shot)["data"])
    gtime = got["gtime"].astype("float64")
    r, z = got["rgrid"].astype("float64"), got["zgrid"].astype("float64")
    psirz = orient(
        got["psirz"], r, z, got["ssimag"], got["rmaxis"], got["zmaxis"]
    ).astype("float32")
    rb, zb = got["rbbbs"], got["zbbbs"]
    if rb.ndim == 2 and rb.shape[0] != len(gtime):
        rb, zb = rb.T, zb.T
    return {
        "gtime_ms": gtime,
        "r": r,
        "z": z,
        "psirz": psirz,
        "ssimag": got["ssimag"].astype("float32"),
        "ssibry": got["ssibry"].astype("float32"),
        "rmaxis": got["rmaxis"].astype("float32"),
        "zmaxis": got["zmaxis"].astype("float32"),
        "rbbbs": rb.astype("float32"),
        "zbbbs": zb.astype("float32"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--pace", type=float, default=1.0)
    args = parser.parse_args()
    out_dir().mkdir(parents=True, exist_ok=True)
    for shot in args.shots:
        path = out_dir() / f"{shot}.npz"
        if path.is_file():
            print(shot, "already parked", flush=True)
            continue
        try:
            arrays = fetch(shot)
        except Exception as error:  # noqa: BLE001  report, keep going
            text = f"{type(error).__name__}: {str(error)[:200]}"
            print(shot, "ERROR", text, flush=True)
            if any(word in text.lower() for word in AUTH_WORDS):
                print("AUTH ERROR: stopping, the login has lapsed", file=sys.stderr)
                return 3
            continue
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        print(shot, "ok", arrays["psirz"].shape, flush=True)
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
