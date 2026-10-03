#!/usr/bin/env python
"""Write the whole-interval tearing-mode labels for the cohort (and the population).

Reads the fetched n = 1 / n = 2 RMS records (`tm_fetch.py`), applies the frozen rule
(`labeler.tearing.rule`) inside each shot's catalog window, and writes

* ``data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv`` (+ meta):
  the cohort's rows in the catalog's interval schema, small enough to commit;
* ``$LABELER_ROOT/round4/tm/labels/`` the same for the population, and one row per
  interval with the amplitudes the catalog schema has no place for.

Run from the worktree::

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_label.py --from cohort
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths, git_sha
from labeler.events import spans
from labeler.events.interval_tables import write_interval_table
from labeler.tearing import rule

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
OUT_ROOT = LABELER / "round4/tm"
SIGNALS = OUT_ROOT / "signals"
FREQUENCIES = OUT_ROOT / "signals_freq"
CATALOG = REPO / "data/events/catalog"
COHORT_OUT = REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval"


def load_signals(shot: int, directory: Path):
    """`(t_ms, n1rms, n2rms)` of a fetched shot, or None."""
    path = directory / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        return npz["t_ms"], npz["n1rms"], npz["n2rms"]


def lock_times(shot: int, directory: Path):
    """`{n: times}` the toroidal modes' frequency fell to zero, or None: no record."""
    path = directory / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        return {
            1: rule.frequency_locks(npz["t_ms"], npz["n1freq"]),
            2: rule.frequency_locks(npz["t_ms"], npz["n2freq"]),
        }


def shot_label(shot: int, window, paths: Paths, directory: Path, freq_dir=None):
    """`(label, how, locks_known)` of one shot, or None: its record was not fetched."""
    record = load_signals(shot, directory)
    if record is None:
        return None
    t_ms, n1, n2 = record
    start, how = spans.plasma_start(shot, paths, window)
    locks = None if freq_dir is None else lock_times(shot, freq_dir)
    label = rule.label_shot(shot, t_ms, n1, n2, window, start, lock_ms=locks)
    return label, how, locks is not None


def table_for(shots: pd.DataFrame, paths: Paths, directory: Path, freq_dir=None):
    """`(rows, intervals, labels, missing, starts, unlocked)` over the shots.

    `unlocked` lists the shots with an interval and no frequency record, whose
    `locked` is therefore unknown (left unset).
    """
    labels, tables, missing, starts, unlocked = [], [], [], {}, []
    for row in shots.itertuples(index=False):
        window = (float(row.window_start_ms), float(row.window_end_ms))
        made = shot_label(int(row.shot), window, paths, directory, freq_dir)
        if made is None:
            missing.append(int(row.shot))
            continue
        label, how, known = made
        labels.append(label)
        starts[int(row.shot)] = {"start_ms": label.start_ms, "from": how}
        if label.intervals and not known and freq_dir is not None:
            unlocked.append(int(row.shot))
        tables.append(rule.shot_table(label))
    rows = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return rows, rule.intervals_frame(labels), labels, missing, starts, unlocked


def meta_for(which, frame, labels, missing, unlocked, shots, rules, extra=None):
    present = frame[(frame.category == 1) & (frame.t_end > frame.t_start)]
    return {
        "category": rule.CATEGORY,
        "table_kind": "intervals",
        "made_by": "scripts/labeler/tm_label.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "shots": which,
        "n_requested_shots": len(shots),
        "n_labelled_shots": len(labels),
        "n_missing_signal_shots": len(missing),
        "n_shots_with_a_mode": int(present.shot.nunique()),
        "signals": "\\MHD::N1RMS, \\MHD::N2RMS (tree mhd, gauss, 1 kHz), fetched",
        "rule": {f"n{r.n}": asdict(r) for r in rules},
        "rule_doc": "docs/labeler/tearing_detection.md; labeler.tearing.rule",
        "crowd": "span rows iscrowd 1 (the whole interval), onset point rows "
        "iscrowd 0; t_start == t_end marks a point",
        "absent": "the rest of the shot's catalog window; ramp-up uncertain where the "
        "rule fires in it; stretches the record did not cover not observable",
        "plasma_start": "labeler.events.spans.plasma_start: the rule runs from the "
        "time Ip reaches its flat-top fraction; each shot's start is in "
        "$LABELER_ROOT/round4/tm/labels/plasma_start_<set>.json",
        "missing_signal_shots": missing[:50],
        "locked": {
            "signal": "\\MHD::N1FREQ, \\MHD::N2FREQ (kHz)",
            "rule": "labeler.tearing.rule.frequency_locks / apply_locking: the mode's "
            "frequency falls to <= 1 kHz for 20 ms after >= 1.5 kHz, within the "
            "interval's last 150 ms or 100 ms after it; set only when true",
            "intervals_without_a_frequency_record_shots": unlocked[:50],
        },
        **(extra or {}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--from", dest="source", choices=("cohort", "population"), default="cohort"
    )
    ap.add_argument("--signals-dir", type=Path, default=SIGNALS)
    ap.add_argument("--freq-dir", type=Path, default=FREQUENCIES)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    paths = Paths.from_env()
    table = pd.read_csv(CATALOG / f"{args.source}.csv")
    shots = table[["shot", "window_start_ms", "window_end_ms"]]
    rows, intervals, labels, missing, starts, unlocked = table_for(
        shots, paths, args.signals_dir, args.freq_dir
    )
    out_dir = args.out_dir or (
        COHORT_OUT if args.source == "cohort" else OUT_ROOT / "labels"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_for(args.source, rows, labels, missing, unlocked, shots, rule.RULES)
    name = (
        "tm_interval.csv" if args.source == "cohort" else "tm_interval_population.csv"
    )
    write_interval_table(rows, out_dir / name, meta)
    big = OUT_ROOT / "labels"
    big.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(big / f"tm_intervals_full_{args.source}.csv", index=False)
    (big / f"plasma_start_{args.source}.json").write_text(
        json.dumps(starts, indent=0) + "\n", encoding="utf-8"
    )
    print(
        f"{args.source}: {len(labels)} shots labelled, {len(missing)} without a "
        f"record, {len(intervals)} intervals on {intervals.shot.nunique()} shots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
