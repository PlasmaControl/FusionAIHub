"""Verify cached cohort clocks using the exact native reader, without reading Te."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from sawtooth_physics import OUTPUT, REPO, REVIEW, WORK, save_json

from labeler.config import Paths
from labeler.sawtooth.preprocessing import sample_native


class VirtualZeroChannel:
    """Supply one finite dummy channel solely to exercise native-clock checks."""

    ndim = 2

    def __init__(self, length):
        self.shape = (1, length)

    def __getitem__(self, key):
        _, columns = key
        start, stop, step = columns.indices(self.shape[1])
        return np.zeros((1, len(range(start, stop, step))), dtype=np.float32)


def audit_shot(shot):
    try:
        with h5py.File(Paths.from_env().corpus_file(shot), "r", locking=False) as file:
            clock = file["ece/xdata"]
            native_t, _ = sample_native(
                {"xdata": clock, "ydata": VirtualZeroChannel(len(clock))}
            )
        row = json.loads((WORK / "shots" / f"{shot}.json").read_text())
        lo, hi = row["window_s"]
        selected = native_t[(native_t >= lo) & (native_t <= hi)]
        with np.load(WORK / "signals" / f"{shot}.npz") as cache:
            matches = np.array_equal(selected, cache["t"])
        return {"shot": shot, "native_clock_valid": True, "cached_time_match": matches}
    except (OSError, KeyError, ValueError) as error:
        return {"shot": shot, "error": f"{type(error).__name__}: {error}"}


def main():
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    shots = sorted(set(cohort.shot) | set(pd.read_csv(REVIEW).shot))
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(audit_shot, shots, chunksize=1))
    summary = {
        "requested_shots": len(shots),
        "native_clock_valid_shots": sum(
            row.get("native_clock_valid", False) for row in rows
        ),
        "errors": {row["shot"]: row["error"] for row in rows if "error" in row},
        "cached_time_mismatches": [
            row["shot"] for row in rows if row.get("cached_time_match") is False
        ],
        "method": (
            "Exact sample_native checks on real xdata; virtual zero channel "
            "avoids ydata IO. No physical input or labels modified."
        ),
        "by_shot": rows,
    }
    save_json(WORK / "clock_audit.json", summary)
    small = {key: value for key, value in summary.items() if key != "by_shot"}
    small["source_details"] = str(WORK / "clock_audit.json")
    save_json(OUTPUT / "clock_audit.json", small)
    print(json.dumps(small), flush=True)
    assert not summary["errors"] and not summary["cached_time_mismatches"]


if __name__ == "__main__":
    main()
