#!/usr/bin/env python
"""Fetch the RWM baseline's inputs for the Hanson and comparison shots into the raw cache.

One ``<shot>_processed.h5`` per shot under ``$LABELER_ROOT/raw`` (the cache beside the
corpus layout), one group per canonical feature (`labeler.features.namespace`), merged so a
rerun only fetches what a shot lacks. The inputs are those the RWM labels and model read:
the n = 1 and n = 2 magnetic RMS, beta_N, l_i, Ip, Bt, q95, qmin, W_MHD, the ZIPFIT toroidal
rotation profile and the locked-mode detector. Fetching needs fdp, so run it on the login
node while logged in; it stops after several shots in a row fetch nothing, which is what a
lapsed login looks like::

    pixi run --frozen -e labelmaker fdp run python scripts/labeler/rwm_fetch.py \\
        --shots-csv $LABELER_ROOT/round4/rwm/shots_pool.csv --workers 3 --pace 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

#: What the RWM labels and model read, in fetch order.
FEATURES = (
    "ip",
    "bt",
    "betan",
    "li",
    "q95",
    "qmin",
    "wmhd",
    "n1rms",
    "n2rms",
    "rot_zipfit",
    "dusbradial",
)
#: Consecutive shots that fetched nothing before the run stops.
MAX_EMPTY_IN_A_ROW = 6

_PACE = 1.0
_RETRIES = 2


def _init(pace: float, retries: int) -> None:
    global _PACE, _RETRIES
    _PACE, _RETRIES = pace, retries


def fetch_shot(shot: int) -> dict:
    """Fetch what the shot's cache file lacks, one `resolve` call, one merge."""
    # Imported here: the pool is forked first, and toksearch's reader is not fork-safe.
    from labeler.config import Paths
    from labeler.events import raw
    from labeler.features import resolve_fdp, store

    path = raw.cache_path(shot, paths=Paths.from_env())
    have = store.present(path)
    permanent = store.permanent_names(path)
    want = [n for n in FEATURES if n not in have and n not in permanent]
    row = {
        "shot": shot,
        "have": sorted(have & set(FEATURES)),
        "fetched": [],
        "missing": {},
    }
    if not want:
        return row
    started = time.monotonic()
    arrays, missing = resolve_fdp.resolve(shot, want, retries=_RETRIES)
    store.write_features(path, shot, arrays, missing, merge=True)
    row["fetched"] = sorted(arrays)
    row["missing"] = {n: c for n, c in missing.items() if n not in arrays}
    row["seconds"] = round(time.monotonic() - started, 1)
    time.sleep(_PACE)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots-csv", type=Path, required=True)
    ap.add_argument("--shots", type=int, nargs="+", help="these shots instead")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", type=Path, help="append one JSON line per shot")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 3:
        ap.error("at most 3 workers")

    import pandas as pd

    shots = args.shots or sorted(int(s) for s in pd.read_csv(args.shots_csv).shot)
    if args.limit:
        shots = shots[: args.limit]
    log = args.log.open("a") if args.log else None
    empty = done = 0
    with Pool(
        args.workers, initializer=_init, initargs=(args.pace, args.retries)
    ) as pool:
        for row in pool.imap_unordered(fetch_shot, shots):
            done += 1
            text = json.dumps(row)
            print(f"[{done}/{len(shots)}] {text}", flush=True)
            if log:
                log.write(text + "\n")
                log.flush()
            if row["missing"] and not row["fetched"]:
                empty += 1
            elif row["fetched"]:
                empty = 0
            if empty >= MAX_EMPTY_IN_A_ROW:
                print(
                    json.dumps(
                        {
                            "stopped": f"{empty} shots in a row fetched nothing "
                            "(login lapsed?)"
                        }
                    ),
                    flush=True,
                )
                pool.terminate()
                return 3
    return 0


if __name__ == "__main__":
    os._exit(main())
