#!/usr/bin/env python
"""Fetch the n = 1 and n = 2 magnetic records the tearing-mode interval label reads.

``\\MHD::N1RMS`` and ``\\MHD::N2RMS`` (tree ``mhd``, gauss, 1 kHz) are the traces the
lab's tearing-mode labels are written on (Fu 2020: n = 1 RMS above 10 G for 50 ms;
Farre-Kaga 2025: above 12 G for 50 ms); ``\\MHD::N1FREQ`` and ``\\MHD::N2FREQ`` (kHz)
give each mode's frequency, which falls to zero as it locks (``--which freq``). The corpus
holds none of them, so they come from MDSplus. Each shot's file, ``<shot>.npz`` under the
output directory, holds the whole records (``t_ms`` and the two traces, float64 /
float32). A shot fdp says has no such record gets ``<shot>.missing.json`` and is not
retried; a fetch that failed otherwise is tried again on the next run. It stops at the
first authentication error.

Needs fdp, so run it on the login node while logged in::

    pixi run --frozen -e labelmaker fdp run python scripts/labeler/tm_fetch.py \\
        --from cohort --part 0/3

``--from cohort`` is the 500 cohort shots, ``--from population`` every population shot
(the cohort is a subset of it); ``--shots`` names shots outright.
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

from labeler.events.verify import NoDataError, fdp_signal

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
OUT_ROOT = LABELER / "round4/tm"
CATALOG = REPO / "data/events/catalog"
#: Which records: the MDSplus nodes, the names they are saved under, the default folder.
KINDS = {
    "rms": ((r"\MHD::N1RMS", r"\MHD::N2RMS"), ("n1rms", "n2rms"), "signals"),
    "freq": ((r"\MHD::N1FREQ", r"\MHD::N2FREQ"), ("n1freq", "n2freq"), "signals_freq"),
}
#: What fdp says of a record the shot does not have, as against a fetch that failed.
ABSENT_WORDS = ("NODATA", "NNF", "No data", "TreeNNF", "TreeNODATA")
AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")
MAX_CONSECUTIVE_FAILURES = 8


def catalog_shots(which: str) -> list[int]:
    """The cohort's, or the population's (a superset), shots."""
    name = {"cohort": "cohort.csv", "population": "population.csv"}[which]
    return sorted(int(s) for s in pd.read_csv(CATALOG / name, usecols=["shot"]).shot)


def fetch_shot(shot: int, kind: str = "rms") -> dict[str, np.ndarray]:
    """Both records of one shot; a failed fetch is retried once after a pause."""
    exprs, names, _ = KINDS[kind]
    for attempt in (1, 2):
        try:
            record = fdp_signal(shot, list(exprs), tree="mhd", via="mds")
            return {
                "t_ms": np.asarray(record.x, dtype=np.float64),
                names[0]: np.asarray(record.y[0], dtype=np.float32),
                names[1]: np.asarray(record.y[1], dtype=np.float32),
            }
        except NoDataError:
            if attempt == 2:
                raise
            time.sleep(5.0)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--from", dest="source", choices=("cohort", "population"), default="cohort"
    )
    ap.add_argument(
        "--shots", type=int, nargs="+", help="these shots instead of --from"
    )
    ap.add_argument(
        "--part",
        default="0/1",
        help="k/n: this process takes every n-th shot, offset k",
    )
    ap.add_argument("--which", choices=tuple(KINDS), default="rms")
    ap.add_argument(
        "--only-with",
        type=Path,
        help="a CSV with a shot column: only those of the shots",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between shots")
    args = ap.parse_args(argv)

    shots = sorted(set(args.shots)) if args.shots else catalog_shots(args.source)
    if args.only_with:
        keep = set(pd.read_csv(args.only_with, usecols=["shot"]).shot.astype(int))
        shots = [s for s in shots if s in keep]
    args.out_dir = args.out_dir or OUT_ROOT / KINDS[args.which][2]
    k, n = (int(v) for v in args.part.split("/"))
    shots = shots[k::n]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for shot in shots:
        done = args.out_dir / f"{shot}.npz"
        gone = args.out_dir / f"{shot}.missing.json"
        if done.exists() or gone.exists():
            continue
        started = time.time()
        try:
            record = fetch_shot(shot, args.which)
        except NoDataError as error:
            message = str(error)
            if any(word in message.lower() for word in AUTH_WORDS):
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
            permanent = any(word in message for word in ABSENT_WORDS)
            if permanent:
                gone.write_text(json.dumps({"shot": shot, "error": message[:500]}))
            failures += 1
            status = "missing" if permanent else "failed"
            print(
                json.dumps({"shot": shot, "status": status, "error": message[:160]}),
                flush=True,
            )
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(json.dumps({"stopped": f"{failures} failures in a row"}))
                return 3
            continue
        failures = 0
        scratch = done.with_suffix(".tmp.npz")
        np.savez(scratch, **record)
        scratch.rename(done)
        print(
            json.dumps(
                {
                    "shot": shot,
                    "status": "fetched",
                    "samples": int(record["t_ms"].size),
                    "seconds": round(time.time() - started, 1),
                }
            ),
            flush=True,
        )
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
