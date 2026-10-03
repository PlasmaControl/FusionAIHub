#!/usr/bin/env python
"""Regenerate full non-test sensitivity grids with the current extraction code.

Unpositioned raw sweep diagnostics are omitted: they do not contribute votes.
Positioned probe, Prad, camera and availability extraction are identical to the
primary run. Regenerated 50 ms arrays must agree with the primary grid.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import detach_bins
import detach_label as dl
import numpy as np

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"


def process_shot(shot):
    records = []
    for width in (20, 50, 100):
        directory = ROOT / f"bins_w{width}"
        records.append(
            detach_bins.process(shot, width, directory, raw_probe_diagnostics=False)
        )
    main_path = ROOT / "bins" / f"{shot}.npz"
    if main_path.exists():
        with (
            np.load(main_path) as main,
            np.load(ROOT / "bins_w50" / f"{shot}.npz") as alt,
        ):
            for key in (
                "start_ms",
                "afrac_valid",
                "afrac_vote",
                "prad_value",
                "prad_valid",
                "prad_vote",
                "tangtv_valid",
                "tangtv_vote",
                "aux_elm_share",
                "aux_p_in_w",
            ):
                if not np.array_equal(main[key], alt[key], equal_nan=True):
                    raise ValueError(f"50 ms regeneration mismatch {shot}/{key}")
    return {"shot": shot, "widths": records, "primary_50ms_check": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    shots = [
        int(s) for s in (ROOT / "shots_sensitivity_non_test.txt").read_text().split()
    ]
    split = dl.cohort_split(shots)
    shots = [s for s in shots if split[s] != "test"]
    with mp.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_shot, shots), 1):
            print(i, len(shots), result, flush=True)


if __name__ == "__main__":
    main()
