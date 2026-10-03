#!/usr/bin/env python
"""Magnetics features per 10 ms bin for the cohort: the input of `tm-ours`.

Reads each shot's six midplane Mirnov probes from the corpus (`mirnov` rows 15, 16, 18,
20, 21, 22) over the shot's catalog window and reduces them with
`labeler.tearing.magfeatures.shot_features`. One file per shot,
``$LABELER_ROOT/round4/tm/magfeatures/<shot>.npz`` (`centres_ms`, `features`, `names`);
a shot whose probes the corpus lacks gets ``<shot>.missing.json`` and is counted.

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_magfeatures.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.events.panels._shared import finite
from labeler.events.verify import NoDataError, corpus_signal
from labeler.tearing import magfeatures

OUT = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm/magfeatures"
)
CATALOG = REPO / "data/events/catalog"
#: Samples either side of the window, so the first and last bins have STFT columns.
MARGIN_MS = 300.0


def one(job):
    shot, w0, w1 = job
    path = OUT / f"{shot}.npz"
    if path.is_file():
        return shot, "have", 0.0
    started = time.monotonic()
    try:
        array = corpus_signal(
            shot,
            "mirnov",
            channels=list(magfeatures.PROBE_ROWS),
            t_range=(w0 - MARGIN_MS, w1 + MARGIN_MS),
        )
        values = np.stack([finite(y) for y in array.y])
        centres, feats = magfeatures.shot_features(array.x, values, (w0, w1))
    except (NoDataError, KeyError, OSError, ValueError) as exc:
        (OUT / f"{shot}.missing.json").write_text(
            json.dumps({"shot": shot, "reason": f"{type(exc).__name__}: {exc}"[:300]})
        )
        return shot, "missing", time.monotonic() - started
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        centres_ms=centres,
        features=feats.astype(np.float32),
        names=np.array(magfeatures.FEATURE_NAMES),
    )
    tmp.rename(path)
    return shot, "ok", time.monotonic() - started


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shots", type=int, nargs="+")
    args = ap.parse_args(argv)
    if args.workers > 8:
        raise SystemExit("at most 8 workers")
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(CATALOG / "cohort.csv")
    if args.shots:
        cohort = cohort[cohort.shot.isin(args.shots)]
    jobs = [
        (int(r.shot), float(r.window_start_ms), float(r.window_end_ms))
        for r in cohort.itertuples(index=False)
    ]
    tally: dict[str, int] = {}
    with Pool(args.workers) as pool:
        for k, (shot, status, secs) in enumerate(pool.imap_unordered(one, jobs), 1):
            tally[status] = tally.get(status, 0) + 1
            if status != "ok" or k % 50 == 0:
                print(f"{k}/{len(jobs)} shot {shot} {status} {secs:.1f}s", flush=True)
    print(json.dumps(tally))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
