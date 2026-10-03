#!/usr/bin/env python
"""Audit the roster's flat-top starts and the saved OPERATIONS n=1 probe traces.

Reads the roster `$LABELER_ROOT/round4/rwm/shots.csv` (written by `rwm_build.py`), the
cached Ip trace of the one negative flat-top start (comparison 157975) and the three
probe shots' saved `sensor_probe/*.npz` arrays (`rwm_sensor_probe.py`); nothing is
fetched or refitted. Writes `outputs/labeler/rwm/data_audit.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.rwm import data, features

OUT = REPO / "outputs" / "labeler" / "rwm" / "data_audit.json"
#: Times (ms) at which the outlier's Ip is quoted, nearest sample.
SAMPLE_TIMES_MS = (-300, -250, -200, -150, -100, -50, 0, 100, 200, 350)


def outlier_record(shot: int, paths: Paths) -> dict:
    t, ip = data.load_signals(shot, paths)["ip"]
    level = np.abs(ip)
    peak = float(np.nanmax(level))
    window = features.flattop_window(t, ip, 0.5)

    def first_above(threshold):
        return float(t[np.flatnonzero(level >= threshold)[0]])

    return {
        "shot": shot,
        "peak_ip_ma": peak / 1e6,
        "half_peak_ip_ma": 0.5 * peak / 1e6,
        "first_time_ms": float(t[0]),
        "flattop_start_ms": window[0],
        "flattop_end_ms": window[1],
        "first_ip_at_least_0p05_ma_ms": first_above(0.05e6),
        "first_ip_at_least_0p5_ma_ms": first_above(0.5e6),
        "ip_ma_at_ms": {
            str(ms): float(ip[int(np.argmin(np.abs(t - ms)))] / 1e6)
            for ms in SAMPLE_TIMES_MS
        },
        "reading": (
            "the saved trace is a smooth rise from about zero current to half of its "
            "peak before t = 0; the negative start is the 50%-of-peak crossing of that "
            "rise on a low-peak shot, not a corrupted time base (interpretation of "
            "the numbers above; the discharge was not otherwise inspected)"
        ),
    }


def probe_statistics(paths: Paths) -> dict:
    """How each saved OPERATIONS n=1 amplitude behaves over its shot's flat-top."""
    probes = json.loads((OUT.parent / "sensor_probe.json").read_text())["probes"]
    windows, out = {}, {}
    for row in probes:
        shot, name = int(row["shot"]), row["name"]
        if shot not in windows:
            windows[shot] = features.flattop_window(
                *data.load_signals(shot, paths)["ip"], 0.5
            )
        with np.load(Path(row["array"])) as saved:
            x, y = saved["x"], saved["y"]
        start, end = windows[shot]
        inside = y[(x >= start) & (x <= end)]
        median = float(np.median(inside))
        out[f"{shot}_{name}"] = {
            "shot": shot,
            "campaign": 2014 if shot < 170000 else 2018,
            "units": row["units"]["data"],
            "flattop_ms": [start, end],
            "median": median,
            "p5": float(np.percentile(inside, 5)),
            "p95": float(np.percentile(inside, 95)),
            "std": float(inside.std()),
            "share_within_5_percent_of_median": float(
                np.mean(np.abs(inside - median) <= 0.05 * abs(median))
            ),
        }
    return out


def main() -> None:
    paths = Paths.from_env()
    roster = pd.read_csv(paths.root / "round4" / "rwm" / "shots.csv")
    used = roster[roster.selected & roster.flattop_start_ms.notna()]
    start = used.flattop_start_ms
    negative = used[start < 0]
    others = start[start >= 0].sort_values()
    record = {
        "script": "scripts/labeler/rwm_data_audit.py",
        "scope": (
            "flat-top window = longest run with |Ip| >= 50% of the shot's peak "
            "(`features.flattop_window`); roster is the Hanson and selected "
            "comparison shots of `shots.csv`"
        ),
        "operations_n1_probe": {
            "scope": (
                "saved OPERATIONS CN1BAMP/ILN1BAMP/IUN1BAMP traces of the three "
                "probe shots over each shot's flat-top; semantics unverified"
            ),
            "traces": probe_statistics(paths),
        },
        "roster": {
            "shots": len(used),
            "by_role": {str(k): int(v) for k, v in used.role.value_counts().items()},
            "negative_flattop_start_shots": [int(s) for s in negative.shot],
            "smallest_nonnegative_start_ms": float(others.iloc[0]),
            "median_start_ms": float(start.median()),
            "largest_start_ms": float(start.max()),
        },
        "outliers": {
            str(int(row.shot)): {
                "role": row.role,
                "campaign": int(row.campaign),
                **outlier_record(int(row.shot), paths),
            }
            for row in negative.itertuples()
        },
        "consequence": (
            "comparison shots enter the baseline only through alarm incidence and the "
            "interval export (and the comparison-negative sensitivity's training); a "
            "negative start only moves where that shot's unlabelled span begins"
        ),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
