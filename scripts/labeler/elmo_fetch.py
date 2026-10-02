#!/usr/bin/env python
"""Fetch the ELM-O detector's native-rate inputs for the ELM benchmark, one file per shot.

ELM-O (O'Shea et al. 2023) reads two interferometer chords, ``DENV2F`` and ``DENV3F``
(100 kS/s), and three filterscopes, ``FS02``-``FS04`` (50 kS/s). The corpus holds the
filterscopes only as a 10 kHz resampling and not the fast chords at all, so both come from
MDSplus: ``\\BCI::DENV2F`` and ``\\BCI::DENV3F`` (tree ``bci``), ``\\SPECTROSCOPY::FS02``-``04``.
Each shot's file, ``<shot>.npz`` under the output directory, holds the whole records
(``t_int_ms``, ``interferometer`` (2, T), ``t_fs_ms``, ``filterscopes`` (3, T), float32).
A shot fdp says has no such record gets ``<shot>.missing.json`` and is not retried;
a fetch that failed otherwise is tried again on the next run.

The shots are David Smith's labelled ELM windows (``--from smith``: 211 shots) and the
shots of the ELM review labels (``--from review``). Fetching needs fdp, so run it on the
login node while logged in; it stops at the first authentication error::

    pixi run --frozen -e labelmaker fdp run python scripts/labeler/elmo_fetch.py \\
        --from smith --part 0/2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.events.verify import NoDataError, fdp_signal

LABELER = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
DEFAULT_OUT = LABELER / "benchmarks/elm/elmo/signals"
SMITH_DIR = Path("/projects/EKOLEMEN/dsmith/data")
SMITH_FILES = ("labeled-elm-events.hdf5", "labeled_elm_events_long_windows_20220921.hdf5")
REVIEW_LABELS = REPO / "data/events/edge_localized_mode/review/labels.csv"
INTERFEROMETER = (r"\BCI::DENV2F", r"\BCI::DENV3F")
FILTERSCOPES = tuple(rf"\SPECTROSCOPY::FS{i:02d}" for i in (2, 3, 4))
#: What fdp says of a record the shot does not have, as against a fetch that failed.
ABSENT_WORDS = ("NODATA", "NNF", "No data")
AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")
MAX_CONSECUTIVE_FAILURES = 8


def smith_shots() -> list[int]:
    """Shots of David Smith's labelled ELM windows (read in place from /projects)."""
    shots: set[int] = set()
    for name in SMITH_FILES:
        with h5py.File(SMITH_DIR / name, "r") as f:
            shots.update(int(group.attrs["shot"]) for group in f.values())
    return sorted(shots)


def review_shots() -> list[int]:
    """Shots of the ELM review labels."""
    return sorted(int(s) for s in pd.read_csv(REVIEW_LABELS, usecols=["shot"]).shot.unique())


def fetch_shot(shot: int) -> dict[str, np.ndarray]:
    """Both records of one shot; a failed fetch is retried once after a pause."""
    for attempt in (1, 2):
        try:
            inter = fdp_signal(shot, list(INTERFEROMETER), tree="bci", via="mds")
            fil = fdp_signal(shot, list(FILTERSCOPES), tree="SPECTROSCOPY", via="mds")
            return {
                "t_int_ms": np.asarray(inter.x, dtype=np.float64),
                "interferometer": np.asarray(inter.y, dtype=np.float32),
                "t_fs_ms": np.asarray(fil.x, dtype=np.float64),
                "filterscopes": np.asarray(fil.y, dtype=np.float32),
            }
        except NoDataError:
            if attempt == 2:
                raise
            time.sleep(5.0)
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="source", choices=("smith", "review", "both"), default="smith")
    ap.add_argument("--shots", type=int, nargs="+", help="these shots instead of --from")
    ap.add_argument("--part", default="0/1", help="k/n: this process takes every n-th shot, offset k")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pace", type=float, default=1.0, help="seconds between shots")
    args = ap.parse_args(argv)

    if args.shots:
        shots = sorted(set(args.shots))
    else:
        shots = set()
        if args.source in ("smith", "both"):
            shots.update(smith_shots())
        if args.source in ("review", "both"):
            shots.update(review_shots())
        shots = sorted(shots)
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
            record = fetch_shot(shot)
        except NoDataError as error:
            message = str(error)
            if any(word in message.lower() for word in AUTH_WORDS):
                print(json.dumps({"shot": shot, "stopped": "authentication", "error": message[:300]}))
                return 2
            permanent = any(word in message for word in ABSENT_WORDS)
            if permanent:
                gone.write_text(json.dumps({"shot": shot, "error": message[:500]}))
            failures += 1
            status = "missing" if permanent else "failed"
            print(json.dumps({"shot": shot, "status": status, "error": message[:160]}), flush=True)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(json.dumps({"stopped": f"{failures} failures in a row"}))
                return 3
            continue
        failures = 0
        np.savez(done.with_suffix(".tmp.npz"), **record)
        done.with_suffix(".tmp.npz").rename(done)
        print(json.dumps({
            "shot": shot, "status": "fetched",
            "interferometer": list(record["interferometer"].shape),
            "filterscopes": list(record["filterscopes"].shape),
            "seconds": round(time.time() - started, 1),
        }), flush=True)
        time.sleep(args.pace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
