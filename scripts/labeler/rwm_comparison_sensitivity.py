#!/usr/bin/env python
"""One reference-split CV using comparisons as label-noisy training negatives.

Run under ``timeout 1800``; headline scores use the original Hanson primary mask.
Comparisons never tune the primary cutoffs or alarm rules. Original predictions
and scores are preserved. --rescore-saved reuses this sensitivity's predictions.
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rwm_evaluate as baseline

from labeler.config import Paths
from labeler.rwm import evaluate as ev
from labeler.rwm import labels, metrics
from labeler.rwm.records import load_evaluation, write_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rescore-saved", action="store_true")
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    out = baseline.REPO / "outputs/labeler/rwm/comparison_sensitivity.json"
    if out.exists() and not args.rescore_saved:
        parser.error("sensitivity exists; use --rescore-saved to avoid another fit")
    root = Paths.from_env().root / "round4/rwm"
    baseline._init(root / "slices.parquet", args.replicates)
    target, other = baseline._STATE["onsets"], baseline._STATE["other"]
    every = {
        shot: sorted(target.get(shot, []) + other.get(shot, []))
        for shot in set(target) | set(other)
    }
    reference = load_evaluation(baseline.REPO / "outputs/labeler/rwm/evaluation.json")[
        "configs"
    ]["rwm-brf"]
    original, original_alarms, _ = baseline.replay(reference, target, every)
    original_groups = ev.shot_records(original, original_alarms, target)
    config = {"kind": "brf", "columns": baseline.ALL, "comparison": True}
    saved = load_evaluation(out) if args.rescore_saved else None
    if saved is not None:
        oof, alarms, rules = baseline.replay(saved, target, every)
        path = Path(saved["predictions"])
        fit_seconds = saved["fit_elapsed_seconds"]
    else:
        table = ev.relabel(baseline._STATE["slices"], target, other, labels.HORIZON_MS)
        folds = [
            part.shot.unique().tolist()
            for _, part in original.groupby("fold", sort=True)
        ]
        start = time.monotonic()
        oof, alarms, rules = ev.cross_validate(
            table,
            baseline.factory(config),
            target,
            explanation_onsets=every,
            seed=baseline.SEED,
            outer_shot_folds=folds,
        )
        fit_seconds = time.monotonic() - start
        path = root / "predictions_rwm-brf-comparison-negative_seed0.parquet"
        oof.to_parquet(path, index=False)
        print(f"single comparison-negative CV: {fit_seconds:.1f} s", flush=True)
    ordered = original.sort_values(["shot", "t_ms"])
    sensitivity = oof.sort_values(["shot", "t_ms"])
    for column in ("shot", "t_ms", "fold", "label", "label_broad"):
        if not np.array_equal(ordered[column], sensitivity[column]):
            raise ValueError(f"sensitivity differs from reference in {column}")
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
        "script": "scripts/labeler/rwm_comparison_sensitivity.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "completed",
        "model": "rwm-brf-comparison-negative",
        "options": config,
        "predictions": str(path),
        "reference_predictions": reference["predictions"],
        "fit_elapsed_seconds": fit_seconds,
        "protocol": {
            "outer_folds": ev.OUTER_FOLDS,
            "inner_folds": ev.INNER_FOLDS,
            "fold_seed": baseline.SEED,
            "forest": baseline.FOREST,
            "bootstrap_replicates": args.replicates,
            "training": (
                "primary Hanson slices plus unlabelled comparisons as negatives"
            ),
            "scoring": "unchanged primary Hanson mask and phase-controlled AUROC",
            "cutoff_tuning": "inner-OOF primary Hanson slices; no comparison tuning",
            "alarm_tuning": (
                "unchanged inner-OOF Hanson traces through last explanation "
                "onset +100 ms; no comparison tuning"
            ),
            "scope": "one CV; identical outer shots, inner splits and random seeds",
            "paired_direction": "comparison-negative sensitivity minus original forest",
            "intervals": "95% shot-bootstrap; percentile scores, basic paired changes",
            "limitation": "label-noisy negatives; comparisons are not verified stable",
        },
        "reference_metrics": reference["metrics"],
        "reference_phase_controlled_auroc": reference["phase_controlled_auroc"],
        "paired_change": paired,
        **summary,
    }
    write_evaluation(
        baseline.clean(record), out, root / "comparison_sensitivity_details.json"
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
