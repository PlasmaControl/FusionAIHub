#!/usr/bin/env python
r"""Summarise the run-day-grouped `elm-ours` cross-validation beside the headline.

The headline folds are grouped by shot; 16 of 94 run days then cross folds, so
neighbouring shots of one day can sit on both sides of a split. `labeler.elm.train
--group-by run_day` repeats the five-fold CV with every run day (the cohort `run_id`)
whole inside one fold and the same recipe. First evaluate it:

    elm_ours_evaluate.py --run cv2_runday \
        --out-dir $LABELER_ROOT/round4/elm/run_day_cv/eval

This script keeps the figures the paper quotes and the fold composition, which the
greedy day-balanced dealing does not stratify by annotation kind. It is a sensitivity
beside the headline, not a replacement: its folds were not tuned or selected on
performance.

Output: `outputs/labeler/elm/ours/run_day_cv.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import methods, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/ours/run_day_cv.json"
HEADLINE = REPO / "outputs/labeler/elm/ours/evaluation.json"
METRICS = ("auroc", "auprc", "f1", "precision", "recall", "false_alarm_bin_rate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="cv2_runday")
    ap.add_argument("--evaluation", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    root = paths.root / "round4/elm"
    evaluation = args.evaluation or root / "run_day_cv/eval/evaluation.json"
    ev = json.loads(evaluation.read_text())
    headline = json.loads(HEADLINE.read_text())
    oof = methods.Oof(root / "cv" / args.run)
    data = train.load(paths)
    days = train.run_days(paths, sorted(data))
    where: dict[str, set[int]] = {}
    for shot, fold in oof.fold_of.items():
        where.setdefault(days[shot], set()).add(fold)
    folds = []
    for k in range(len(oof.record["folds"])):
        shots = sorted(s for s, f in oof.fold_of.items() if f == k)
        kinds = [set(data[s].spans.kind) for s in shots]
        folds.append(
            {
                "fold": k,
                "test_shots": len(shots),
                "run_days": len({days[s] for s in shots}),
                "threshold": oof.record["fold_records"][k]["threshold"],
                "shots_with_non_crowd_spans": sum("non_crowd" in x for x in kinds),
                "shots_with_crowd_spans": sum("crowd" in x for x in kinds),
            }
        )
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "run": args.run,
        "source_evaluation": str(evaluation),
        "source_evaluation_sha256": sha256_of(evaluation),
        "fold_grouping": oof.record.get("fold_grouping"),
        "protocol": "Same recipe, seed and scoring as the headline cv2 run; only "
        "the fold dealing differs: every cohort run day lies wholly in one test "
        "fold (greedy balance by shot count, no stratification by annotation "
        "kind). Sensitivity beside the headline, not a replacement.",
        "cohort_test_shots_used": 0,
        "run_days": len(where),
        "run_days_spanning_multiple_folds": sum(len(v) > 1 for v in where.values()),
        "folds": folds,
        "sets": {},
    }
    for tag, body in ev["sets"].items():
        record["sets"][tag] = {
            "n_shots": body["n_shots"],
            "bins": body["bins"],
            "methods": {
                name: {
                    "point": {m: res["point"].get(m) for m in METRICS},
                    "ci95": {m: res.get("ci95", {}).get(m) for m in METRICS},
                }
                for name, res in body["methods"].items()
            },
            "paired": {
                key: {"value": row["value"], "ci95": row["ci95"]}
                for key, row in body["paired"].items()
                if key.rsplit(": ", 1)[-1] in ("auroc", "auprc", "f1")
            },
            "headline_elm_ours": {
                m: headline["sets"][tag]["methods"]["elm-ours"]["point"][m]
                for m in ("auroc", "auprc", "f1")
            },
        }
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    for tag, body in record["sets"].items():
        print(tag, body["methods"]["elm-ours"]["point"], body["headline_elm_ours"])
    print(
        "run days",
        record["run_days"],
        "spanning",
        record["run_days_spanning_multiple_folds"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
