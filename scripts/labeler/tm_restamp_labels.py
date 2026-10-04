#!/usr/bin/env python
"""Carry trained score records over an attrs-only change of the tearing-mode labels.

The GPU records (`tm-ours`, the retrained CNN and DSM) hold the sha256 of the label
table they were fitted against. A later change that touches only the `attrs` column
of the table, here the `onset_window_degenerate` flag of one onset row, leaves every
per-bin target, mask and threshold as it was, so refitting would only reshuffle
random numbers. This script proves the change is attrs-only and then writes the new
sha256 into the named records, keeping the old one beside it:

* the committed table of `--previous-rev` and the current table have the same rows,
  columns and values everywhere except `attrs`;
* the `attrs` of every row differ at most in `onset_window_degenerate`;
* the per-bin targets and masks (`scoring.label_bins`, both modes) of every
  development shot are identical.

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_restamp_labels.py tm_ours_magnetics_cv ...
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.events.interval_tables import parse_attrs
from labeler.tearing import scoring

REPO = Path(__file__).resolve().parents[2]
TM = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
#: The only attribute the change may touch.
FLAG = "onset_window_degenerate"


def previous_table(rev: str) -> bytes:
    path = str(LABELS.relative_to(REPO))
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"],
        capture_output=True,
        check=True,
    ).stdout


def assert_attrs_only(old: pd.DataFrame, new: pd.DataFrame) -> int:
    """Raise unless `new` differs from `old` only in `FLAG`; the rows that differ."""
    assert list(old.columns) == list(new.columns)
    assert len(old) == len(new)
    others = [c for c in old.columns if c != "attrs"]
    pd.testing.assert_frame_equal(old[others], new[others], check_exact=True)
    changed = 0
    for before, after in zip(old["attrs"], new["attrs"], strict=True):
        a, b = parse_attrs(before), parse_attrs(after)
        if before != after:
            changed += 1
        a.pop(FLAG, None)
        b.pop(FLAG, None)
        assert a == b, (before, after)
    return changed


def assert_same_bins(old: pd.DataFrame, new: pd.DataFrame, shots) -> None:
    """Per-bin targets and masks of every shot, both scoring modes, are identical."""
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv").set_index("shot")
    for shot in shots:
        row = cohort.loc[shot]
        centres = scoring.bin_centres((row.window_start_ms, row.window_end_ms))
        for negative in (False, True):
            got = [
                scoring.label_bins(
                    table[table.shot == shot], centres, uncertain_negative=negative
                )
                for table in (old, new)
            ]
            for a, b in zip(got[0], got[1], strict=True):
                assert np.array_equal(a, b), (shot, negative)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+", help="records under results/, without .json")
    ap.add_argument("--previous-rev", default="HEAD")
    args = ap.parse_args(argv)
    old_bytes, new_bytes = previous_table(args.previous_rev), LABELS.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    new_sha = hashlib.sha256(new_bytes).hexdigest()
    old = pd.read_csv(io.BytesIO(old_bytes))
    new = pd.read_csv(io.BytesIO(new_bytes))
    changed = assert_attrs_only(old, new)
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    assert_same_bins(old, new, dev)
    print(f"attrs-only change: {changed} rows; targets and masks identical")
    for stem in args.stems:
        path = TM / "results" / f"{stem}.json"
        record = json.loads(path.read_text())
        assert record["labels_sha256"] == old_sha, (stem, record["labels_sha256"])
        record["labels_sha256_trained_on"] = old_sha
        record["labels_sha256"] = new_sha
        record["labels_restamp"] = (
            "the labels changed only in the attrs column (the onset_window_degenerate "
            "flag); per-bin targets and masks are identical, so the fit was not "
            "repeated (scripts/labeler/tm_restamp_labels.py)"
        )
        path.write_text(json.dumps(record, indent=1, default=float) + "\n")
        print("restamped", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
