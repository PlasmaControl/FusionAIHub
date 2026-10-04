#!/usr/bin/env python
"""Fetch the date of each curated confinement shot, for the coverage figure's marginal.

No table on disk dates the shots before 2021, so this reads the time MDSplus recorded
when the shot's EFIT reconstruction was inserted
(``\\EFIT01::TOP.RESULTS.GEQDSK:GTIME``, a 64-bit count of 100 ns ticks since
1858-11-17; the BES tree's own nodes are the fallback). EFIT01 runs within hours of a
shot, so this is the shot's day; a shot whose date is more than 30 days from the
median of its 11 neighbouring shots (an EFIT run again later) is flagged and takes
the neighbours' median date for its year. Output:
``$LABELER_ROOT/round4/conf/dates.csv`` (shot, inserted_utc, year, source_node,
consistent). A single worker; run on the login node under fdp while logged in; it stops
at the first authentication error::

    pixi run --frozen -e labelmaker fdp run python \\
        scripts/labeler/confinement_shot_dates_fetch.py

The cohort's blind ``test`` shots are never fetched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_windows as bw

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
DEFAULT_OUT = LABELER / "round4/conf/dates.csv"
VMS_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)
NODES = (
    ("EFIT01", r"\EFIT01::TOP.RESULTS.GEQDSK:GTIME"),
    ("BES", r"\BES::BES_R"),
)
AUTH_WORDS = ("auth", "credential", "kerberos", "permission denied", "expired", "kinit")
FIRST, LAST = 2005, 2030


def vms_to_utc(ticks: int) -> datetime:
    """A 64-bit MDSplus/VMS time (100 ns ticks since 1858-11-17) as a UTC datetime."""
    return VMS_EPOCH + timedelta(microseconds=int(ticks) / 10.0)


def inserted(shot: int) -> tuple[datetime, str] | None:
    """When the shot's first usable node was written, and the node's tree."""
    import MDSplus

    for tree, node in NODES:
        try:
            node_ = MDSplus.Tree(tree, int(shot), "readonly").getNode(node)
            ticks = int(node_.time_inserted)
        except Exception as error:
            if any(w in str(error).lower() for w in AUTH_WORDS):
                raise
            continue
        if ticks > 0:
            when = vms_to_utc(ticks)
            if FIRST <= when.year <= LAST:
                return when, tree
    return None


def assign_years(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``consistent`` and ``year`` to a shot-sorted frame of ``inserted_utc``.

    A shot is consistent when it has a stamp within 30 days of the median stamp of its
    11 neighbouring shots; any other shot (no stamp, or an EFIT run again later) takes
    the year of that neighbours' median.
    """
    when = pd.to_datetime(frame.inserted_utc, utc=True, errors="coerce")
    seconds = pd.Series(
        [t.timestamp() if pd.notna(t) else float("nan") for t in when],
        index=frame.index,
    )
    near = seconds.rolling(11, center=True, min_periods=3).median()
    consistent = when.notna() & ((seconds - near).abs() / 86400 <= 30)
    year = [
        w.year if ok else (datetime.fromtimestamp(a, UTC).year if pd.notna(a) else None)
        for w, a, ok in zip(when, near, consistent, strict=True)
    ]
    return frame.assign(consistent=consistent, year=year)


def summary(frame: pd.DataFrame) -> dict:
    """Shot count, dated count, inconsistent shots and the shots per year."""
    stamped = pd.to_datetime(frame.inserted_utc, utc=True, errors="coerce")
    return {
        "shots": len(frame),
        "dated": int(stamped.notna().sum()),
        "inconsistent": frame.shot[~frame.consistent].tolist(),
        "years": {
            str(int(y)): int(n)
            for y, n in frame.year.value_counts().sort_index().items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pace", type=float, default=0.2)
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="re-derive consistent and year from the stamps already in --out",
    )
    args = ap.parse_args(argv)
    if args.from_csv:
        frame = assign_years(pd.read_csv(args.out, keep_default_na=False))
        frame.to_csv(args.out, index=False)
        print(json.dumps(summary(frame)))
        return 0
    shots = sorted(int(s) for s in bw.curated_intervals().shot.unique())
    rows = []
    for shot in shots:
        try:
            got = inserted(shot)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"shot": shot, "stopped": str(error)[:200]}))
            return 2
        when, tree = got if got else (None, "")
        rows.append(
            {
                "shot": shot,
                "inserted_utc": when.isoformat(timespec="seconds") if when else "",
                "year": None,
                "source_node": tree,
            }
        )
        time.sleep(args.pace)
    frame = assign_years(pd.DataFrame(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(json.dumps(summary(frame)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
