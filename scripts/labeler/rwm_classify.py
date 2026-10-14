#!/usr/bin/env python
"""Score RWM presence per 10 ms bin: rwm-ours against rwm-brf (owner, 2026-10-14).

    pixi run --frozen -e labeler python scripts/labeler/rwm_classify.py \
        [--table CSV] [--replicates 1000] [--workers 2]

Both models are the balanced forest of Piccione et al. 2022 (`rwm.forest`, 300
trees, depth 8, leaves of 5) fitted as a presence classifier at t on Hanson's 33
shots (`labeler.rwm.classify`: the slice at t is the bin [t - 10, t); present
positive, the rest of the measured time negative, uncertain and not observable
time out of both classes), one calendar run day held out at a time with the F1
cut chosen on the other days:

- `rwm-ours`: the v5 inputs at t, `features.V5_FEATURES` less lock_v: the round-
  four inputs and the n = 1 delta-Bp (mean, max, log slope), the q = 2 rotation
  and the I-coil activity, from `$LABELER_ROOT/round5/rwm/slices_v5.parquet`;
- `rwm-brf`: the forest recipe on the round-four inputs (`features.FEATURES` less
  lock_v, N1RMS-based), on the same rows, bins and folds.

The table defaults to the one the RWM review page opens on (`review/source.json`,
`rwm_onsets` v5). Scores are bin-level F1 (each fold's cut), AUROC and AUPRC
with 95 % whole-shot bootstrap intervals, their paired difference
(rwm-ours minus rwm-brf, the same shots drawn for both), the constant predictor
on the same bins as the trivial baseline, and the view with uncertain time
scored negative. The NSTX forest's reported numbers ride along as `legacy`: a
100 ms forecast on another machine. The record goes to
`docs/labeler/results/rwm_classifier.json` (or `--out`), held-out scores beside the
slices (named after `--out` when it is not the default).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths, git_sha
from labeler.events.review import labels as review_labels
from labeler.events.review.silver import event_dir
from labeler.rwm import classify, features, metrics

EVENT = "resistive_wall_mode"
SEED = 0
#: DUSBRADIAL is zero on most 2014 traces and flagged corrupted on 2018 Hanson
#: shots (round four left it out the same way).
MODELS = {
    "rwm-ours": tuple(f for f in features.V5_FEATURES if f != "lock_v"),
    "rwm-brf": tuple(f for f in features.FEATURES if f != "lock_v"),
}
OUT = REPO / "docs" / "labeler" / "results" / "rwm_classifier.json"
LEGACY_RECORD = REPO / "outputs" / "labeler" / "rwm" / "evaluation.json"


def rows(paths: Paths, table_path: Path) -> tuple[pd.DataFrame, dict]:
    """Hanson's slices with their bin's category, targets and calendar day."""
    slices = pd.read_parquet(paths.root / "round5" / "rwm" / "slices_v5.parquet")
    slices = slices[slices.role == "hanson"].reset_index(drop=True)
    roster = pd.read_csv(paths.root / "round4" / "rwm" / "shots.csv")
    run = roster.set_index("shot").run.astype(str)
    slices["day"] = slices.shot.map(run).map(classify.calendar_day)
    if slices.day.isna().any():
        raise ValueError("a Hanson shot has no run record in the round-four roster")
    table = pd.read_csv(table_path)
    by_shot = dict(tuple(table.groupby("shot")))
    category = np.full(len(slices), -1, dtype=np.int8)
    for shot, index in slices.groupby("shot").indices.items():
        part = by_shot.get(int(shot), table.iloc[:0])
        category[index] = classify.bin_categories(part, slices.t_ms.to_numpy()[index])
    slices["category"] = category
    slices["y"] = classify.targets(category)
    slices["y_unc"] = classify.targets(category, classify.UNCERTAIN_NEGATIVE)
    hanson = table[table.shot.isin(slices.shot.unique())]
    present = hanson[hanson.category == classify.PRESENT]
    in_slices = int((category == classify.PRESENT).sum())
    audit = {
        "table_shots_hanson": int(hanson.shot.nunique()),
        "bins_by_category": {
            str(int(c)): int(n)
            for c, n in zip(*np.unique(category, return_counts=True), strict=True)
        },
        "present_ms_in_table": float((present.t_end - present.t_start).sum()),
        "present_bins_scored": in_slices,
        "note": "the slices span the high-current window (|Ip| at least half its "
        "flat-top value, `features.slice_table`); category -1 is a bin outside "
        "the table's window",
    }
    return slices, audit


def clean(value):
    """NaN and infinity as JSON null (a day with no call has no precision)."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_model(job):
    name, frame, columns = job
    oof = classify.cross_validate(frame, columns, seed=SEED)
    return name, oof


def summarise(oof: pd.DataFrame, replicates: int) -> dict:
    out = {}
    for view, column in (("primary", "y"), (classify.UNCERTAIN_NEGATIVE, "y_unc")):
        records = classify.shot_records(oof, column)
        groups = {"hanson": records}
        entry = {
            "counts": classify.counts(records),
            "metrics": metrics.shot_bootstrap(
                groups, classify.statistic, replicates=replicates, seed=SEED
            ),
        }
        if view == "primary":
            entry["by_day"] = {
                day: {
                    "counts": classify.counts(part),
                    "point": classify.statistic({"hanson": part}),
                }
                for day in sorted({r["day"] for r in records})
                for part in [[r for r in records if r["day"] == day]]
            }
        out[view] = entry
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, help="default: the review source")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    table_path = args.table or review_labels.source_path(event_dir(paths, EVENT))
    meta_path = Path(str(table_path).replace(".csv", ".meta.json"))
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    frame, audit = rows(paths, table_path)
    with Pool(min(args.workers, len(MODELS))) as pool:
        oofs = dict(pool.map(run_model, [(n, frame, c) for n, c in MODELS.items()]))
    prediction_dir = paths.root / "round5" / "rwm"
    # another --out keeps its own predictions beside the default record's
    stem = "classifier" if args.out.resolve() == OUT else args.out.stem
    result = {}
    for name, oof in oofs.items():
        oof["y_unc"] = frame.y_unc.to_numpy()
        path = prediction_dir / f"{stem}_{name}.parquet"
        oof.to_parquet(path, index=False)
        scored = summarise(oof, args.replicates)
        result[name] = {
            "inputs": list(MODELS[name]),
            **scored["primary"],
            "cuts": {
                day: float(oof.cut[oof.day == day].iloc[0])
                for day in sorted(oof.day.unique())
            },
            classify.UNCERTAIN_NEGATIVE: scored[classify.UNCERTAIN_NEGATIVE],
            "predictions": str(path),
        }
    paired = {}
    for view, column in (("primary", "y"), (classify.UNCERTAIN_NEGATIVE, "y_unc")):
        a, b = (
            {"hanson": classify.shot_records(oofs[n], column)}
            for n in ("rwm-ours", "rwm-brf")
        )
        if [r["shot"] for r in a["hanson"]] != [r["shot"] for r in b["hanson"]]:
            raise ValueError("the two models are scored on different shots")
        paired[view] = metrics.paired_bootstrap(
            a, b, classify.statistic, replicates=args.replicates, seed=SEED
        )
    counts = result["rwm-ours"]["counts"]
    if counts != result["rwm-brf"]["counts"]:
        raise ValueError("the two models are scored on different bins")
    legacy = json.loads(LEGACY_RECORD.read_text())["legacy"]
    record = {
        "script": "scripts/labeler/rwm_classify.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "task": "detector: RWM presence per 10 ms bin",
        "labels": {
            "table": str(table_path),
            "method": meta.get("method"),
            "version": meta.get("version"),
            "table_git_sha": meta.get("git_sha"),
            "positive": "present (1)",
            "negative": "unassessed (4) or absent (0) on Hanson's shots: Hanson's "
            "lists taken as complete there, as round four assumed",
            "excluded": "uncertain (2) and not observable (3), and bins outside "
            "the table's window",
            "bin": "the slice at t is the frame [t - 10, t) its trailing inputs read",
            "audit": audit,
        },
        "protocol": {
            "shots": "Hanson's 33 shots (round four's roster, role hanson)",
            "folds": "leave one calendar run day out: "
            + ", ".join(
                f"{day} ({frame[frame.day == day].shot.nunique()} shots)"
                for day in sorted(frame.day.unique())
            )
            + "; the run records 20180314 and 20180314A share a day",
            "cut": "per fold, the F1-best cut on an inner leave-one-day-out over "
            "the training days",
            "model": {
                "name": "balanced random forest",
                **classify.FOREST,
                "seed": SEED,
                "imputer": "training-fold medians",
            },
            "bootstrap": f"{args.replicates} whole-shot resamples, seed {SEED}, "
            "percentile 95 % intervals; paired: the same shots drawn for both",
        },
        "counts": counts,
        "baseline": classify.baseline(counts["prevalence"]),
        "baseline_uncertain_negative": classify.baseline(
            result["rwm-ours"][classify.UNCERTAIN_NEGATIVE]["counts"]["prevalence"]
        ),
        "models": result,
        "paired": {"rwm-ours_minus_rwm-brf": paired},
        "legacy": {
            **legacy,
            "task": "forecast: a 100 ms forecast on NSTX, another machine; slice "
            "AUROC only (no F1, no AUPRC), not comparable with a DIII-D presence "
            "score",
            "record": str(LEGACY_RECORD.relative_to(REPO)),
        },
        "caveats": [
            (
                "The v5 present spans end where the delta-Bp falls back below the "
                "screen threshold and the uncertain stretches are delta-Bp runs, so "
                "rwm-ours reads the labels' own sensor: its score is partly "
                "agreement with the delta-Bp rule, not an independent truth."
            ),
            (
                "Negatives are assumed: Hanson's lists are taken as complete on his "
                "shots; no rule tested absence there."
            ),
            (
                "Three calendar days, 33 shots: the intervals resample shots at "
                "fixed held-out predictions, not run days."
            ),
        ],
        "dropped": "the forecaster (100 ms horizon), the 20 ms onset gap, the "
        "gap-aware alarms and the phase-controlled AUROC (owner, 2026-10-14 12:58)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(clean(record), indent=1) + "\n")
    for name in MODELS:
        m = result[name]["metrics"]
        print(name, {k: round(m[k]["estimate"], 4) for k in ("f1", "auroc", "auprc")})
    print("counts", counts, "baseline", record["baseline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
