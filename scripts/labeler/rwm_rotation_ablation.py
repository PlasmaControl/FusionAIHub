#!/usr/bin/env python
"""Bound ZIPFIT dependence with one no-rotation reference-split nested CV.

Run under a 30-minute timeout. Save separate predictions, leaving every original
fit and score untouched. --rescore-saved recomputes summaries without another fit.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rwm_evaluate as baseline

from labeler.config import Paths
from labeler.rwm import evaluate as ev
from labeler.rwm import labels, metrics
from labeler.rwm.records import load_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rescore-saved", action="store_true")
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    start = time.monotonic()
    out = baseline.REPO / "outputs/labeler/rwm/rotation_ablation.json"
    if out.exists() and not args.rescore_saved:
        parser.error(
            "ablation already exists; use --rescore-saved to avoid another fit"
        )
    root = Paths.from_env().root / "round4/rwm"
    baseline._init(root / "slices.parquet", args.replicates)
    target, other = baseline._STATE["onsets"], baseline._STATE["other"]
    every = {
        s: sorted(target.get(s, []) + other.get(s, []))
        for s in set(target) | set(other)
    }
    reference = load_evaluation(
        baseline.REPO / "outputs/labeler/rwm/evaluation.json"
    )["configs"]["rwm-brf"]
    original, original_alarms, _ = baseline.replay(reference, target, every)
    original_groups = ev.shot_records(original, original_alarms, target)
    removed = [c for c in baseline.ALL if c.startswith("rot_")]
    config = {
        "kind": "brf",
        "columns": [c for c in baseline.ALL if c not in removed],
        "comparison": False,
    }
    saved = json.loads(out.read_text()) if args.rescore_saved else None
    if saved is not None:
        oof, alarms, rules = baseline.replay(saved, target, every)
        path = Path(saved["predictions"])
    else:
        table = ev.relabel(baseline._STATE["slices"], target, other, labels.HORIZON_MS)
        folds = [
            part.shot.unique().tolist()
            for _, part in original.groupby("fold", sort=True)
        ]
        oof, alarms, rules = ev.cross_validate(
            table,
            baseline.factory(config),
            target,
            explanation_onsets=every,
            seed=baseline.SEED,
            outer_shot_folds=folds,
        )
        path = root / "predictions_rwm-brf-no-rotation_seed0.parquet"
        oof.to_parquet(path, index=False)
        print("completed the single no-rotation CV refit", flush=True)
    ordered = original.sort_values(["shot", "t_ms"])
    ablated = oof.sort_values(["shot", "t_ms"])
    for column in ("shot", "t_ms", "fold", "label", "label_broad"):
        if not np.array_equal(ordered[column], ablated[column]):
            raise ValueError(f"ablation differs from reference in {column}")
    summary, groups = baseline.summarise(oof, alarms, target, rules)
    paired = metrics.paired_bootstrap(
        groups,
        original_groups,
        ev.statistic,
        replicates=args.replicates,
        seed=baseline.SEED,
        method="basic",
    )
    paired["phase_controlled_auroc"] = ev.phase_controlled_bootstrap(
        groups, original_groups, replicates=args.replicates, seed=baseline.SEED
    )
    record = {
        "script": "scripts/labeler/rwm_rotation_ablation.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "completed",
        "model": "rwm-brf-no-rotation",
        "options": config,
        "removed_columns": removed,
        "predictions": str(path),
        "reference_predictions": reference["predictions"],
        "fit_elapsed_seconds": (
            saved["fit_elapsed_seconds"]
            if saved is not None
            else time.monotonic() - start
        ),
        "protocol": {
            "outer_folds": ev.OUTER_FOLDS,
            "inner_folds": ev.INNER_FOLDS,
            "fold_seed": baseline.SEED,
            "forest": baseline.FOREST,
            "bootstrap_replicates": args.replicates,
            "paired_direction": "no rotation minus original forest",
            "intervals": (
                "95% basic paired shot-bootstrap, fixed predictions; "
                "unadjusted for multiplicity"
            ),
            "scope": (
                "one reference-split CV; identical outer shot sets, inner splits "
                "and random seeds; training-only imputation; no comparison "
                "fitting/tuning"
            ),
            "limitation": (
                "input-dependence sensitivity, not a measurement of upstream "
                "ZIPFIT timing bias or all acausal leakage"
            ),
        },
        "reference_metrics": reference["metrics"],
        "paired_change": paired,
        **summary,
    }
    out.write_text(json.dumps(baseline.clean(record), indent=2, allow_nan=False) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
