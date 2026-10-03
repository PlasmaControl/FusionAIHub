#!/usr/bin/env python
"""Score one log D-alpha burst feature with the frozen review CV partitions."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, dsm, feature, methods, score, train

REPO = Path(__file__).resolve().parents[2]


def common_bin_audit(paths, data, sets, rows, oof):
    """Assert that the feature's common bins are the benchmark's common bins.

    `elm-feature-only` restricts the primary bins with `dsm.bins_with_rows`; the
    common-bin control table restricts them with `compare.bin_support` over every
    method's support. Both must keep the same bins, shot by shot, or the feature
    row would be scored on bins the other rows were not.
    """
    work = paths.root / "round4/elm/dsm"
    fits = json.loads((work / "fits.json").read_text())
    detection = {
        s: dsm.load_rows(s, work / "repaired_raw_rows" / f"{s}.npz") for s in data
    }
    dscores = compare.DsmScores.load(
        work, rows, variants=tuple(fits["detectors"]), detection_rows=detection
    )
    audit = {}
    for tag, sdef in sets.items():
        total = 0
        for s in sdef.shots:
            bins = sdef.bins[s]
            masks = compare.bin_support(
                s, bins, sdef.cover[s], oof.trace(s)[0], dscores
            )
            if sdef.has_elmo:
                masks[compare.NAME["elmo"]] = compare.covered_bin_mask(
                    bins, sdef.elmo_cover[s]
                )
            keep = np.logical_and.reduce(list(masks.values()))
            rowed = dsm.bins_with_rows(
                bins, rows[s].usable, lags=(compare.FORECAST_LAG_ROWS, 0)
            )
            if set(bins.t0[keep].tolist()) != set(rowed.t0.tolist()):
                raise AssertionError(
                    f"{tag} shot {s}: feature common bins differ from the "
                    "benchmark's common bins"
                )
            total += int(keep.sum())
        audit[tag] = {"shots": len(sdef.shots), "common_bins": total}
    return {"bin_sets_equal": True, "sets": audit}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="cv2")
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    data = train.load(paths)
    run = paths.root / "round4/elm/cv" / args.run
    folds = [json.loads((run / f"fold{k}/fold.json").read_text()) for k in range(5)]
    # Fit only outer-training bins, choose the operating point on inner validation.
    features = {s: feature.bin_feature(d.x, d.bins) for s, d in data.items()}
    models, threshold, fold_records = {}, {}, []
    for fold in folds:
        fit_shots, val_shots = fold["train"], fold["inner_val"]
        x = np.concatenate([features[s] for s in fit_shots])
        y = np.concatenate([data[s].bins.truth for s in fit_shots])
        model = feature.fit_logistic(x, y)
        val_x = np.concatenate([features[s] for s in val_shots])
        val_y = np.concatenate([data[s].bins.truth for s in val_shots])
        thr, val_f1 = score.best_threshold(val_y, feature.logistic_scores(model, val_x))
        for s in fold["test"]:
            models[s], threshold[s] = model, thr
        fold_records.append(
            {k: fold[k] for k in ("fold", "train", "inner_val", "test")}
            | {
                "threshold": thr,
                "val_f1": val_f1,
                "coefficient": float(model[0]),
                "intercept": float(model[1]),
            }
        )
    sets = compare.load_sets(paths, data)
    rows = {
        s: dsm.load_rows(s, paths.root / "round4/elm/dsm" / f"{s}.npz") for s in data
    }
    oof = methods.Oof(run)
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "name": "elm-feature-only",
        "feature": "max across FS02-FS04 of within-bin max minus median log10 "
        "D-alpha on the existing 0.1ms cell-maximum input grid",
        "recipe": "one feature, logistic regression C=1; no class weighting, "
        "normalization or hyperparameter selection; inner-val F1 threshold",
        "folds": fold_records,
        "cohort_test_shots_used": 0,
        "primary_run": str(run / "run.json"),
        "primary_run_sha256": sha256_of(run / "run.json"),
        "common_bin_audit": common_bin_audit(paths, data, sets, rows, oof),
        "sets": {},
    }
    for scope in ("primary", "common"):
        record["sets"][scope] = {}
        for name, subset in sets.items():
            parts = []
            for s in subset.shots:
                bins = subset.bins[s]
                if scope == "common":
                    bins = dsm.bins_with_rows(
                        bins, rows[s].usable, lags=(compare.FORECAST_LAG_ROWS, 0)
                    )
                f = feature.bin_feature(data[s].x, bins)
                p = feature.logistic_scores(models[s], f)
                parts.append(
                    score.ShotScore(s, bins.truth, bins.kind, p >= threshold[s], p)
                )
            record["sets"][scope][name] = {
                "n_shots": len(subset.shots),
                "bins": sum(len(p.truth) for p in parts),
                "methods": {
                    "elm-feature-only": score.summarise(parts, score.draws(len(parts)))
                },
            }
    out = REPO / "outputs/labeler/elm/ours/feature_only.json"
    out.write_text(json.dumps(record, indent=1) + "\n")
    for scope, subsets in record["sets"].items():
        for tag, res in subsets.items():
            print(scope, tag, res["bins"], res["methods"]["elm-feature-only"]["point"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
