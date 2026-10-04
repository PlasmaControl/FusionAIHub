#!/usr/bin/env python
"""Measure the range of the fast-density input the occupancy U-Net reads.

The input is the DENV2F/DENV3F native ordinate divided by `inputs.DENSITY_UNIT`
(1e14) and clipped to `inputs.DENSITY_RANGE`. This reads each reviewed shot's prepared
input array, takes the two density channels over the valid cells between 1 and 4 s
(the flat-top window of the reviewed shots), drops chords the failed-digitiser screen
zeroed, and records the pooled median, 5-95% range and the share of cells at the
clip bounds. The physical ordinate unit stays unverified; this reports the numerical
range of the input only. Offline: no network access.

    python scripts/labeler/elm_density_range.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import inputs, labels, prepare

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/density_range.json"
WINDOW_MS = (1000, 4000)


def main() -> int:
    paths = Paths.from_env()
    shots = sorted(
        map(int, labels.review_table(prepare.review_csv(paths)).shot.unique())
    )
    lo, hi = inputs.DENSITY_RANGE
    cells0, cells1 = (int(t * inputs.CELLS_PER_MS) for t in WINDOW_MS)
    pooled, rejected, chords, per_shot = [], 0, 0, {}
    for shot in shots:
        x = np.load(prepare.inputs_dir(paths) / f"{shot}.npy", mmap_mode="r")
        valid = np.asarray(x[inputs.VALID][cells0:cells1]) > 0
        row = {"valid_cells": int(valid.sum()), "chords_used": 0}
        for channel in inputs.DENSITY:
            values = np.asarray(x[channel][cells0:cells1])[valid]
            chords += 1
            if values.size == 0 or not np.any(values):
                rejected += 1  # screened to zero, or no valid cell in the window
                continue
            pooled.append(values.astype(np.float64))
            row["chords_used"] += 1
        per_shot[str(shot)] = row
    values = np.concatenate(pooled)
    q05, q50, q95 = np.percentile(values, [5, 50, 95])
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "purpose": "numerical range of the fast-density input (native ordinate / "
        "1e14, clipped), flat-top window; units of the ordinate stay unverified",
        "input_divisor_native_units": inputs.DENSITY_UNIT,
        "clip_range": [lo, hi],
        "window_ms": list(WINDOW_MS),
        "shots": len(shots),
        "chords": chords,
        "chords_zeroed_or_empty": rejected,
        "cells": int(values.size),
        "median": float(q50),
        "p05": float(q05),
        "p95": float(q95),
        "share_at_upper_clip": float(np.mean(values >= hi)),
        "share_at_lower_clip": float(np.mean(values <= lo)),
        "per_shot": per_shot,
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "per_shot"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
