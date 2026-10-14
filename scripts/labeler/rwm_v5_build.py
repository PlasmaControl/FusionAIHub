#!/usr/bin/env python
"""Build the v5 RWM slices (round five, delta-Bp): rwm-ours's inputs.

    pixi run --frozen -e labeler python scripts/labeler/rwm_v5_build.py [--workers 8]

Round four's slices (`$LABELER_ROOT/round4/rwm/slices.parquet`) keep their rows,
labels and columns; each shot gains `features.BP_FEATURES` from the raw cache
(`rwm.data.load_signals`, never fetching; NaN where a shot lacks an input), into
`$LABELER_ROOT/round5/rwm/slices_v5.parquet`. Nothing about the round-four
columns changes, so `rwm_classify.py` scores the two input sets on the same rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.events.verify import NoDataError
from labeler.rwm import data, features


def shot_columns(job):
    """`(shot, frame)`: the shot's `BP_FEATURES` on its round-four slice times."""
    shot, times = job
    paths = Paths.from_env()
    try:
        signals = data.load_signals(shot, paths)
    except NoDataError:
        signals = {}
    columns = features.bp_columns(signals, times)
    return shot, pd.DataFrame({"shot": shot, "t_ms": times, **columns})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = Paths.from_env()
    slices = pd.read_parquet(paths.root / "round4" / "rwm" / "slices.parquet")
    jobs = [
        (int(shot), part.t_ms.to_numpy(float))
        for shot, part in slices.groupby("shot", sort=True)
    ]
    with Pool(args.workers) as pool:
        frames = [frame for _, frame in pool.imap(shot_columns, jobs, chunksize=2)]
    added = pd.concat(frames, ignore_index=True)
    merged = slices.merge(added, on=["shot", "t_ms"], how="left", validate="1:1")
    if len(merged) != len(slices):
        raise ValueError("the v5 columns changed the slice rows")
    out_dir = paths.root / "round5" / "rwm"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_dir / "slices_v5.parquet", index=False)

    coverage = {
        name: float(np.isfinite(merged[name]).mean()) for name in features.BP_FEATURES
    }
    print(
        json.dumps(
            {
                "rows": len(merged),
                "finite_fraction": coverage,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
