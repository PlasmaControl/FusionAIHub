#!/usr/bin/env python
"""Window features of the corpus BES (500 kHz, 64 channels) for the curated shots.

The same windows, features and file layout as ``confinement_bes_fetch.py`` (one
``<shot>.npz`` per shot under ``$LABELER_ROOT/round4/conf/bes500k``), so a loader reads
either rate. Here a 1024-sample window is 2.05 ms, as in the first benchmark
(``confinement_bes_benchmark.py``), but all 64 channels are kept and the 15L and 15R
beam powers come from the corpus ``pinj`` group (rows 0 and 1, watts), so rows, beam
gating and transition exclusion can be chosen afterwards. Reads the corpus only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_features as bf
from labeler.confinement import bes_windows as bw

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
DEFAULT_OUT = LABELER / "round4/conf/bes500k"
FS = 500e3


def corpus_shot(shot: int, intervals: pd.DataFrame, out_dir: Path) -> dict:
    """Write one shot's file; return a one-line status."""
    path = CORPUS / f"{shot}_processed.h5"
    if not path.exists():
        return {"shot": shot, "status": "no corpus file"}
    with h5py.File(path, "r") as f:
        if "bes" not in f or f["bes/ydata"].shape[-1] <= 1:
            return {"shot": shot, "status": "no bes"}
        x = f["bes/xdata"]
        n, t0 = x.shape[0], float(x[0]) * 1000.0
        dt = (float(x[-1]) * 1000.0 - t0) / (n - 1)
        if abs(dt * FS / 1000.0 - 1.0) > 0.01:
            raise ValueError(f"{shot}: BES step {dt * 1000:.4f} us, not {1e6 / FS} us")
        bes = f["bes/ydata"][:].astype(np.float32)
        beams = {}
        if "pinj" in f and f["pinj/ydata"].shape[-1] > 1:
            tp = f["pinj/xdata"][:].astype(np.float64) * 1000.0
            for name, row in (("15L", 0), ("15R", 1)):
                beams[name] = (tp, f["pinj/ydata"][row].astype(np.float32))
    out = bw.shot_windows(intervals, bes, t0, dt, FS)
    if not out:
        return {"shot": shot, "status": "no window"}
    span = bf.WINDOW * dt
    for name in ("15L", "15R"):
        t_ms, y = beams.get(name, (None, None))
        for stat, values in bw.window_beam_power(
            t_ms, y, out["start_ms"], span
        ).items():
            out[f"p{name}_{stat}"] = values
    out["record"] = np.array([t0, dt, n, 0])
    out["channel_std"] = np.nanstd(bes[:, :: max(1, n // 200_000)], axis=1)
    tmp = out_dir / f"{shot}.tmp.npz"
    np.savez(tmp, **out)
    tmp.rename(out_dir / f"{shot}.npz")
    return {"shot": shot, "status": "written", "windows": int(out["feats"].shape[0])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    table = bw.curated_intervals()
    shots = [
        int(s)
        for s in sorted(table.shot.unique())
        if (CORPUS / f"{s}_processed.h5").exists()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(corpus_shot, s, table[table.shot == s], args.out_dir)
            for s in shots
            if not (args.out_dir / f"{s}.npz").exists()
        ]
        for fut in futures:
            print(json.dumps(fut.result()), flush=True)
            done += 1
    print(json.dumps({"shots_run": done, "seconds": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
