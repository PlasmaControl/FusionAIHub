#!/usr/bin/env python
"""Fetch where each BES channel looks, per shot, for the block choice of factor d.

For every shot of the fetched 1 MHz BES set (``$LABELER_ROOT/round4/conf/bes1mhz``) this
reads the 64 channels' positions ``\\BES::BES_R`` and ``\\BES::BES_Z`` (cm) and the EFIT
flux map (``\\efit01::top.results.geqdsk``: ``psirz``, ``ssimag``, ``ssibry``, ``r``,
``z``, ``rmaxis``, ``zmaxis``, ``gtime``; the ``efit02`` tree when ``efit01`` holds
none), and writes ``<shot>.npz`` with the positions, the EFIT times and each channel's
normalised flux ``psi_N`` at each time (``labeler.confinement.bes_geometry``), a few
hundred KB at most. A shot whose BES positions or flux map the archive lacks gets
``<shot>.missing.json`` and is not retried; a failed fetch is tried again. It stops at
the first authentication error, so run it on the login node under fdp while logged in,
at most 3 workers at pace 1::

    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/confinement_bes_geometry_fetch.py --part 0/3 --pace 1

The cohort's blind ``test`` shots are never fetched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_geometry as geo

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
WORK = LABELER / "round4/conf"
DEFAULT_OUT = WORK / "geometry"
SOURCE = WORK / "bes1mhz"
TREES = ("efit01", "efit02")
FIELDS = ("gtime", "psirz", "ssimag", "ssibry", "r", "z", "rmaxis", "zmaxis")
MISSING_ERRORS = ("TreeNODATA", "TreeNNF", "TreeFOPENR", "TreeNOT_OPEN")
AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")
CANARY = 149992
MAX_CONSECUTIVE_FAILURES = 8


def fetch_array(expr: str, tree: str, shot: int) -> np.ndarray | None:
    """One MDSplus node's data, or None when the shot holds nothing there."""
    from labeler.features.resolve_fdp import _fetch_mds

    try:
        rec = _fetch_mds(expr, tree, int(shot))
    except Exception as error:
        if type(error).__name__ in MISSING_ERRORS:
            return None
        raise
    return np.asarray(rec["data"], dtype=np.float64)


def process_shot(shot: int) -> dict:
    """One shot's channel positions and flux; the arrays to write."""
    r_cm = fetch_array(r"\BES::BES_R", "BES", shot)
    z_cm = fetch_array(r"\BES::BES_Z", "BES", shot)
    if r_cm is None or z_cm is None or r_cm.size != 64 or z_cm.size != 64:
        raise LookupError(f"{shot}: no 64-channel BES_R/BES_Z")
    out: dict[str, np.ndarray] = {"r_cm": r_cm, "z_cm": z_cm}
    for tree in TREES:
        got = {
            name: fetch_array(rf"\{tree}::top.results.geqdsk:{name}", tree, shot)
            for name in FIELDS
        }
        if any(v is None or v.size == 0 for v in got.values()):
            continue
        psin, residual = geo.channel_psin(
            got["psirz"],
            got["r"],
            got["z"],
            got["ssimag"],
            got["ssibry"],
            got["rmaxis"],
            got["zmaxis"],
            r_cm,
            z_cm,
        )
        out.update(
            gtime_ms=got["gtime"],
            psin=psin,
            axis_residual=np.array(residual),
            tree=np.array(tree),
        )
        return out
    out.update(
        gtime_ms=np.zeros(0),
        psin=np.zeros((0, 64), np.float32),
        axis_residual=np.array(np.nan),
        tree=np.array(""),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--shots", type=int, nargs="+", help="these shots only")
    ap.add_argument("--part", default="0/1", help="k/n: every n-th shot")
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between shots")
    ap.add_argument("--no-canary", action="store_true")
    args = ap.parse_args(argv)

    shots = sorted(
        int(f.stem)
        for f in args.source.glob("*.npz")
        if f.stem.isdigit() and not f.name.endswith(".tmp.npz")
    )
    if args.shots:
        shots = [s for s in args.shots if s in set(shots)]
    k, n = (int(v) for v in args.part.split("/"))
    shots = shots[k::n]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_canary:
        try:
            if fetch_array(r"\BES::BES_R", "BES", CANARY) is None:
                raise LookupError("canary shot returned nothing")
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"stopped": "canary", "error": str(error)[:300]}))
            return 2
    failures = 0
    for shot in shots:
        done = args.out_dir / f"{shot}.npz"
        gone = args.out_dir / f"{shot}.missing.json"
        if done.exists() or gone.exists():
            continue
        started = time.time()
        try:
            arrays = process_shot(shot)
        except LookupError as error:
            gone.write_text(json.dumps({"shot": shot, "error": str(error)[:500]}))
            print(
                json.dumps(
                    {"shot": shot, "status": "missing", "error": str(error)[:160]}
                ),
                flush=True,
            )
            continue
        except Exception as error:  # noqa: BLE001
            message = str(error)
            if any(w in message.lower() for w in AUTH_WORDS):
                print(
                    json.dumps(
                        {
                            "shot": shot,
                            "stopped": "authentication",
                            "error": message[:300],
                        }
                    )
                )
                return 2
            failures += 1
            print(
                json.dumps({"shot": shot, "status": "failed", "error": message[:160]}),
                flush=True,
            )
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(json.dumps({"stopped": f"{failures} failures in a row"}))
                return 3
            continue
        failures = 0
        tmp = done.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.rename(done)
        print(
            json.dumps(
                {
                    "shot": shot,
                    "status": "fetched",
                    "efit_times": int(arrays["gtime_ms"].size),
                    "tree": str(arrays["tree"]),
                    "axis_residual": float(arrays["axis_residual"]),
                    "seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
