#!/usr/bin/env python
"""Fetch the 0D traces the confinement segmenter reads, one small file per shot.

``confine-ours`` (``confinement_ours.py``) labels L / H / QH / WPQH from
zero-dimensional signals, so it can run on the whole roster, where BES is rare. The
corpus and the raw cache already hold D-alpha (``filterscopes``) and the beam powers
(``pinj``) for the roster; the rest comes from DIII-D through fdp:

* ``dens``   ``\\BCI::DENV2F``, the CO2 V2 interferometer's fast voltage (proportional
  to the line-integrated density; fringe jumps are not unwrapped), averaged to 1 ms
  bins;
* ``wmhd``   ``\\efit01::top.results.aeqdsk:wmhd`` (J) and ``betan`` (``:betan``), as
  the equilibrium reconstruction gives them (one point every 15-30 ms);
* ``pinj``   the eight beams' injected power (W), only for the shots whose corpus file
  and raw cache both lack it.

Output: ``$LABELER_ROOT/round4/conf/zerod/<shot>.npz``. A signal the shot does not hold
is an empty array and is named in ``missing``. Run on the login node under fdp while
logged in, with at most 3 workers in all::

    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/confinement_zerod_fetch.py --part 0/1 --pace 1

It stops at the first authentication error. The cohort's blind ``test`` shots are never
fetched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_windows as bw

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
DEFAULT_OUT = LABELER / "round4/conf/zerod"
ROSTER = Path(
    os.environ.get(
        "CONFINEMENT_ROSTER",
        "/scratch/gpfs/nc1514/FusionAIHub/data/events/confinement/shots.csv",
    )
)
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
RAW = LABELER / "raw"
BEAMS = ("15L", "15R", "21L", "21R", "30L", "30R", "33L", "33R")
DENSITY = (r"\BCI::DENV2F", "bci")
EFIT = {
    "wmhd": (r"\efit01::top.results.aeqdsk:wmhd", "efit01"),
    "betan": (r"\efit01::top.results.aeqdsk:betan", "efit01"),
}
BIN_MS = 1.0
MISSING_ERRORS = ("TreeNODATA", "TreeNNF", "TreeFOPENR", "TreeNOT_OPEN")
AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")
MAX_CONSECUTIVE_FAILURES = 8


def bin_mean(
    t_ms: np.ndarray, y: np.ndarray, step: float = BIN_MS
) -> tuple[float, np.ndarray]:
    """Mean of ``y`` in ``step``-ms bins: the first bin's left edge and the means (NaN
    if empty)."""
    ok = np.isfinite(t_ms) & np.isfinite(y)
    t_ms, y = t_ms[ok], y[ok].astype(np.float64)
    if t_ms.size == 0:
        return 0.0, np.zeros(0, dtype=np.float32)
    t0 = float(np.floor(t_ms.min() / step) * step)
    idx = np.floor((t_ms - t0) / step).astype(np.int64)
    n = int(idx.max()) + 1
    total = np.bincount(idx, weights=y, minlength=n)
    count = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    return t0, mean.astype(np.float32)


def fetch_node(expr: str, tree: str, shot: int, dims: tuple[str, ...] = ()):
    """One node's time (ms) and values, or None if the shot holds nothing there."""
    from labeler.features.resolve_fdp import _fetch_mds

    try:
        rec = _fetch_mds(expr, tree, int(shot), dims=list(dims))
    except Exception as error:
        if type(error).__name__ in MISSING_ERRORS:
            return None
        raise
    key = dims[0] if dims else "times"
    return np.asarray(rec[key], dtype=np.float64), np.asarray(
        rec["data"], dtype=np.float32
    )


def has_local_pinj(shot: int) -> bool:
    """True if the corpus file or the raw cache holds a beam-power record."""
    import h5py

    for base in (CORPUS, RAW):
        path = base / f"{shot}_processed.h5"
        if path.exists():
            with h5py.File(path, "r") as f:
                if "pinj" in f and f["pinj/ydata"].shape[-1] > 1:
                    return True
    return False


def process_shot(shot: int) -> dict:
    """Fetch one shot's traces; the arrays to write."""
    out: dict[str, np.ndarray] = {}
    missing: list[str] = []
    got = fetch_node(DENSITY[0], DENSITY[1], shot, dims=("dim0",))
    if got is None:
        missing.append("dens")
    else:
        t0, y = bin_mean(*got)
        out["dens"], out["dens_t0_ms"] = y, np.array(t0)
    for name, (expr, tree) in EFIT.items():
        got = fetch_node(expr, tree, shot)
        if got is None or got[0].size < 2:
            missing.append(name)
            continue
        out[f"{name}_t_ms"], out[name] = got
    if not has_local_pinj(shot):
        rows, t0 = [], None
        for beam in BEAMS:
            got = fetch_node(
                rf"\D3D::TOP.NB.NB{beam}:PINJ_{beam}", "D3D", shot, dims=("dim0",)
            )
            if got is None:
                rows.append(None)
                continue
            start, y = bin_mean(*got)
            t0 = start if t0 is None else min(t0, start)
            rows.append((start, y))
        if t0 is None:
            missing.append("pinj")
        else:
            length = max(len(r[1]) + round((r[0] - t0) / BIN_MS) for r in rows if r)
            grid = np.zeros((len(BEAMS), length), dtype=np.float32)
            for i, r in enumerate(rows):
                if r:
                    lo = round((r[0] - t0) / BIN_MS)
                    grid[i, lo : lo + len(r[1])] = np.nan_to_num(r[1])
            out["pinj"], out["pinj_t0_ms"] = grid, np.array(t0)
    out["missing"] = np.array(json.dumps(missing))
    return out


def wanted_shots() -> list[int]:
    """The roster and the curated shots outside the blind test split, ascending."""
    curated = set(bw.curated_intervals().shot)
    roster = set(pd.read_csv(ROSTER).shot)
    return sorted(int(s) for s in roster | curated)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--part", default="0/1", help="k/n: every n-th shot")
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between shots")
    ap.add_argument("--shots", type=int, nargs="+", help="these shots only")
    ap.add_argument("--no-canary", action="store_true")
    args = ap.parse_args(argv)

    shots = args.shots or wanted_shots()
    k, n = (int(v) for v in args.part.split("/"))
    shots = shots[k::n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_canary:
        try:
            fetch_node(*EFIT["betan"], 149992)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"stopped": "canary", "error": str(error)[:300]}))
            return 2
    failures = 0
    for shot in shots:
        done = args.out_dir / f"{shot}.npz"
        if done.exists():
            continue
        started = time.time()
        try:
            arrays = process_shot(shot)
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
                    "missing": json.loads(str(arrays["missing"])),
                    "seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
