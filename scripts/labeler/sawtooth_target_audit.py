"""Audit Gaussian supervision eligibility on all 500 cached native shot grids.

Compare the prior raw-present-center targets with corrected assessed-center
targets, before and after the float16 conversion used by training windows.
Unknown-bin differences are reported separately from actual supervised changes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_benchmark import (
    assessed_points,
    masks_at,
    present_rows,
    record,
    targets,
)
from sawtooth_physics import OUTPUT, REPO, WORK, save_json

from labeler.sawtooth.metrics import spans_at
from labeler.sawtooth.models import soft_crash_target


def audit(work):
    shots = sorted(pd.read_csv(REPO / "data/events/catalog/cohort.csv").shot)
    counts, rows, changed_bins = Counter(), [], []
    for shot in shots:
        shot = int(shot)
        with np.load(work / "signals" / f"{shot}.npz") as data:
            signal = {key: data[key] for key in ("t", "observable", "assessed")}
        t = signal["t"]
        rec = record(work, shot)
        assessed = masks_at(signal, t)[1]
        observable_from_record = spans_at(t, rec["observable_spans"])
        assessed_from_record = spans_at(t, rec["assessed_spans"])
        observable_mismatch = int(
            np.count_nonzero(observable_from_record != signal["observable"])
        )
        assessed_mismatch = int(np.count_nonzero(assessed_from_record != assessed))
        assert not observable_mismatch, f"observable reconstruction differs: {shot}"
        assert not assessed_mismatch, f"assessed reconstruction differs: {shot}"
        centers = np.array(
            [r["time_s"] for r in present_rows(rec, "crashes")], dtype=float
        )
        eligible_centers = assessed_points(signal, centers)
        before = soft_crash_target(t, centers)
        after, _, _ = targets(t, rec, boundary=0)
        expected = soft_crash_target(t, eligible_centers)
        assert np.array_equal(after, expected), f"center filtering differs: {shot}"
        changed = before != after
        supervised_changed = changed & assessed
        before_training = before.astype(np.float16)
        after_training = after.astype(np.float16)
        training_changed = (before_training != after_training) & assessed
        row = {
            "shot": shot,
            "native_bins": len(t),
            "assessed_bins": int(assessed.sum()),
            "raw_present_centers": len(centers),
            "eligible_centers": len(eligible_centers),
            "demoted_centers": len(centers) - len(eligible_centers),
            "unmasked_target_changed_bins": int(changed.sum()),
            "assessed_target_changed_bins": int(supervised_changed.sum()),
            "assessed_float16_target_changed_bins": int(training_changed.sum()),
            "observable_reconstruction_mismatch": observable_mismatch,
            "assessed_reconstruction_mismatch": assessed_mismatch,
        }
        for key, value in row.items():
            if key != "shot":
                counts[key] += value
        counts["audited_shots"] += 1
        counts["shots_with_demoted_centers"] += bool(row["demoted_centers"])
        counts["shots_with_assessed_target_changes"] += bool(
            row["assessed_target_changed_bins"]
        )
        for index in np.flatnonzero(supervised_changed):
            changed_bins.append(
                {
                    "shot": shot,
                    "time_s": float(t[index]),
                    "old_target": float(before[index]),
                    "new_target": float(after[index]),
                    "old_float16_training_target": float(before_training[index]),
                    "new_float16_training_target": float(after_training[index]),
                }
            )
        rows.append(row)
    full = {
        "protocol": {
            "cohort": "all fixed 500 cohort shots, no threshold selection",
            "prior": "Gaussian from crash attrs.state=present, irrespective of mask",
            "corrected": (
                "Crash attrs.state=present and center assessed on the nearest "
                "native grid bin; reconstruct masks from explicit record spans"
            ),
            "gaussian_sigma_ms": 0.5,
            "gaussian_truncation_sigma": 3,
            "window_sampling": "unchanged; raw-present-center sampling retained",
            "hl3_supervision": "classes and window selection unchanged; picks unused",
        },
        "counts": dict(counts),
        "shot_ids": list(map(int, shots)),
        "changed_assessed_bins": changed_bins,
        "assessed_training_targets_bit_identical": (
            counts["assessed_float16_target_changed_bins"] == 0
        ),
        "retraining_required": counts["assessed_float16_target_changed_bins"] > 0,
        "by_shot": rows,
    }
    save_json(work / "target_mask_audit.json", full)
    summary = {key: value for key, value in full.items() if key != "by_shot"}
    summary["by_shot_count"] = len(rows)
    summary["by_shot_source"] = str(work / "target_mask_audit.json")
    summary["shots_with_demoted_centers"] = [
        row for row in rows if row["demoted_centers"]
    ]
    save_json(OUTPUT / "target_mask_audit.json", summary)
    print(counts, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    audit(args.work)


if __name__ == "__main__":
    main()
