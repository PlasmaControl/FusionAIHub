#!/usr/bin/env python
"""Backfill ELM run provenance and disclose saved preliminary training runs.

This audits existing files; it never retrains or regenerates held-out predictions.
The revision and explanation must be supplied because no training-time source
snapshot exists for the historical cv2 run.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from labeler.config import Paths
from labeler.elm import inputs, provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="cv2")
    ap.add_argument(
        "--revision", required=True, help="committed training-code reference"
    )
    ap.add_argument(
        "--evidence", required=True, help="basis for reconstructing revision"
    )
    ap.add_argument("--history-json", type=Path, required=True)
    ap.add_argument("--prior-log", action="append", default=[], metavar="RUN=PATH")
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    cv = paths.root / "round4" / "elm" / "cv"
    logs = dict(value.split("=", 1) for value in args.prior_log)
    record = provenance.backfill(cv / args.run, paths, args.revision, args.evidence)
    history = {
        "observed_at": provenance.observed_at(),
        "source": str(cv),
        "reported_run": args.run,
        "runs": {},
        "caveats": {
            "inputs": "FS02-FS04 and two density chords; FS01 was not used.",
            "normalization": (
                "GroupNorm statistics are computed over 4096 ms training crops and "
                "over padded whole-shot inputs during validation and inference."
            ),
            "seed": "One base seed; per-fold seed offsets are not seed replicates.",
            "history": (
                "Saved artifacts establish preliminary held-out predictions existed; "
                "they do not establish whether those predictions influenced choices. "
                "No complete configuration survives for cv1."
            ),
        },
        "channels": list(inputs.CHANNELS),
    }
    for run_dir in sorted(p for p in cv.iterdir() if p.is_dir()):
        run_file = run_dir / "run.json"
        run_record = json.loads(run_file.read_text()) if run_file.exists() else None
        folds = []
        for fold_file in sorted(run_dir.glob("fold*/fold.json")):
            row = json.loads(fold_file.read_text())
            folds.append(
                {
                    "fold": row["fold"],
                    "record": provenance.file_record(fold_file),
                    "epochs_recorded": len(row["history"]),
                    "train": row["train"],
                    "inner_val": row["inner_val"],
                    "test": row["test"],
                    "best": row["best"],
                    "threshold": row["threshold"],
                    "onset_threshold_recorded": "onset_threshold" in row,
                }
            )
        summary = {
            "run_record": provenance.file_record(run_file) if run_record else None,
            "config_recorded": run_record["config"] if run_record else None,
            "completed_folds": folds,
            "checkpoint_count": len(list(run_dir.glob("fold*/model.pt"))),
            "held_out_prediction_count": len(list((run_dir / "pred").glob("*.npz"))),
        }
        if run_dir.name in logs:
            log_file = Path(logs[run_dir.name])
            summary["log"] = provenance.file_record(log_file)
            summary["fold_start_messages"] = [
                line
                for line in log_file.read_text().splitlines()
                if line.startswith("fold ")
            ]
        history["runs"][run_dir.name] = summary

    cohort = pd.read_csv(paths.catalog / "cohort.csv").set_index("shot")
    day_folds = defaultdict(lambda: defaultdict(list))
    for k, shots in enumerate(record["folds"]):
        for shot in shots:
            day = str(int(cohort.loc[shot, "run_id"]))
            day_folds[day][k].append(shot)
    history["same_day_outer_fold_audit"] = {
        "cohort": provenance.file_record(paths.catalog / "cohort.csv"),
        "run_days": len(day_folds),
        "run_days_spanning_multiple_folds": sum(len(v) > 1 for v in day_folds.values()),
        "days": [
            {"run_id": day, "folds": dict(sorted(folds.items()))}
            for day, folds in sorted(day_folds.items())
            if len(folds) > 1
        ],
        "reviewer_examples": [
            {
                "shot": shot,
                "run_id": int(cohort.loc[shot, "run_id"]),
                "fold": next(k for k, f in enumerate(record["folds"]) if shot in f),
            }
            for shot in (192721, 192732, 192751, 190637, 190643)
        ],
    }
    args.history_json.parent.mkdir(parents=True, exist_ok=True)
    args.history_json.write_text(json.dumps(history, indent=1) + "\n")
    print(
        json.dumps(
            {
                "run_json": str(cv / args.run / "run.json"),
                "history_json": str(args.history_json),
                "producing_revision_reconstructed": record["git"],
                "fold_records": len(record["fold_records"]),
                "inputs_hashed": len(record["provenance"]["data"]["inputs"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
