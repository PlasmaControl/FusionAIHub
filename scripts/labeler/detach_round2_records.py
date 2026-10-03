#!/usr/bin/env python
"""Snapshot and compare round-two populations, including unknown ELM coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import detach_label as dl
import numpy as np
import pandas as pd
from detach_json import dumps

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULT = dl.REPO / "docs/labeler/results/detachment_round2.json"
BEFORE = ROOT / "fix_round2_before.json"


def summarize():
    bins = dl.load_all(ROOT / "bins")
    labels = pd.read_csv(ROOT / "labels_bins.csv.gz")
    intervals = pd.read_csv(dl.OUT / "detach_shots.csv")
    certain = labels.state_lm.isin((1, 2, 3))
    spans = intervals.loc[intervals.category.isin((1, 2, 3))]
    cohort = dl.cohort_split(labels.shot.unique())
    split = labels.shot.map(cohort)
    survey = pd.read_csv(ROOT / "survey/corpus_survey.csv")
    result = {
        "sources": {
            "bins": str(ROOT / "bins"),
            "labels": str(ROOT / "labels_bins.csv.gz"),
            "intervals": str(dl.OUT / "detach_shots.csv"),
            "survey": str(ROOT / "survey/corpus_survey.csv"),
        },
        "population": {"bins": len(bins), "shots": int(bins.shot.nunique())},
        "assessed": {"bins": len(labels), "shots": int(labels.shot.nunique())},
        "certain": {
            "bins": int(certain.sum()),
            "shots": int(labels.loc[certain, "shot"].nunique()),
            "seconds": float(certain.sum() * 0.05),
            "intervals": len(spans),
            "single_50ms_intervals": int(((spans.t_end - spans.t_start) == 50).sum()),
            "by_split": {
                s: {
                    "bins": int((certain & split.eq(s)).sum()),
                    "shots": int(labels.loc[certain & split.eq(s), "shot"].nunique()),
                }
                for s in ("train", "val", "test", "outside")
            },
        },
        "states": {str(k): int(v) for k, v in labels.state_lm.value_counts().items()},
        "tiers": {str(k): int(v) for k, v in labels.tier.value_counts().items()},
        "corpus_survey": {
            "shots": len(survey),
            **{
                name: int((survey[column] > 1).sum())
                for name, column in (
                    ("tangtv", "n_tangtv"),
                    ("bolo", "n_bolo"),
                    ("langmuir", "n_langmuir"),
                    ("irtv", "n_irtv"),
                )
            },
        },
        "inversion_shots": len(list((ROOT / "inversions").glob("*.npz"))),
        "surrogate_valid_bins": int(
            (bins.tangtv_valid & bins.tangtv_source.eq("surrogate")).sum()
        ),
    }
    result["elm_coverage"] = {}
    for name, rows in (
        ("discharge", bins),
        ("assessed", labels),
        ("certain", labels.loc[certain]),
    ):
        share = rows.get("aux_elm_share", pd.Series(np.nan, index=rows.index))
        known = np.isfinite(share)
        result["elm_coverage"][name] = {
            "bins": len(rows),
            "finite": int(known.sum()),
            "nan": int((~known).sum()),
            "majority": int((share > 0.5).sum()),
        }
    # These are the actual inversion shots in Chen's specified 189057--189101 range.
    chen = bins[bins.shot.between(189057, 189101)]
    result["chen_hmode"] = {
        "selection": (
            "All discharge bins on cached inversion shots 189057--189101; no new "
            "H-mode inference or threshold selection."
        ),
        "bins": len(chen),
        "shots": int(chen.shot.nunique()),
        "indicators": {
            name: {
                "valid_bins": int(chen[f"{name}_valid"].sum()),
                "voting_bins": int((chen[f"{name}_vote"] > 0).sum()),
                "shots": int(chen.loc[chen[f"{name}_valid"], "shot"].nunique()),
            }
            for name in ("tangtv", "prad", "afrac")
        },
        "per_shot": {
            str(int(shot)): {
                "bins": len(group),
                **{
                    name: int(group[f"{name}_valid"].sum())
                    for name in ("tangtv", "prad", "afrac")
                },
                "elm_nan_bins": int((~np.isfinite(group.aux_elm_share)).sum()),
            }
            for shot, group in chen.groupby("shot")
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    current = summarize()
    if args.snapshot:
        if BEFORE.exists():
            raise SystemExit(f"Refusing to replace baseline snapshot {BEFORE}")
        BEFORE.write_text(dumps(current, indent=1))
        print(dumps(current, indent=1))
        return
    result = {"before": json.loads(BEFORE.read_text()), "after": current}
    RESULT.write_text(dumps(result, indent=1))
    print(dumps({k: v["certain"] for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
