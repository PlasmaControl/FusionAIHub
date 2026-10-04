#!/usr/bin/env python
"""f_div as a within-shot corroborator: per-shot AUROC against TangTV and against Te.

f_div (Prad,div,L over the heating power) is not a vote in the exported label: pooled
over shots its level moves with the shot (input power, seeding, strike-point geometry)
and as a vote it only created conflicts (owner decision 2026-10-04, after Opus review
5, I1). Within one shot the same quantity ranks the bins, so this script reports it
as a corroborator, shot by shot, with the shot counts beside every summary:

* the score is f_div, either over the shot's own baseline (`prad_rel_value`, the
  relative family, valid where `prad_valid`) or as the absolute ratio
  (`prad_value`, valid where `prad_abs_valid`, the 201081-anchored family);
* against TangTV: positive class the TangTV detached votes, negative the attached
  votes, upper-shelf bins where TangTV is valid;
* against Te: positive class the divertor-Thomson-cold bins (<= 5 eV), negative the
  warm ones (>= 10 eV), the same Te as `detach_te_check.py`;
* per shot, with at least `MIN_CLASS_BINS` bins in each class, then the mean over
  those shots with a shot-bootstrap interval, the number of shots, and the number of
  shots where the AUROC exceeds 0.5; and the pooled AUROC (shots mixed) with its
  shot-bootstrap interval, which is the number a vote would have to live with.

Nothing here is fitted and no threshold is chosen.

    pixi run --frozen -e labelmaker python scripts/labeler/detach_fdiv_check.py
"""

from __future__ import annotations

import os
from pathlib import Path

import detach_benchmark as bench
import detach_te_check as te
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
OUT = REPO / "docs/labeler/results/detachment_fdiv_check.json"
MIN_CLASS_BINS = 5
FAMILIES = {
    "relative": ("prad_rel_value", "prad_valid"),
    "absolute": ("prad_value", "prad_abs_valid"),
}


def classes(frame: pd.DataFrame) -> dict:
    """Positive / negative masks of the two references on the bins of `frame`."""
    tangtv = (
        frame.tangtv_tier.eq("upper_shelf")
        & frame.tangtv_valid.astype(bool)
        & frame.tangtv_vote.isin((core.ATTACHED, core.DETACHED))
    )
    warm = np.isfinite(frame.te_ev) & (frame.te_ev >= te.TE_ATTACHED_MIN_EV)
    cold = np.isfinite(frame.te_ev) & (frame.te_ev <= te.TE_DETACHED_MAX_EV)
    return {
        "tangtv": (
            tangtv & frame.tangtv_vote.eq(core.DETACHED),
            tangtv & frame.tangtv_vote.eq(core.ATTACHED),
        ),
        "te": (cold, warm),
    }


def shot_row(rows: pd.DataFrame, family: str, reference: str) -> dict:
    """One shot's AUROC of f_div for one reference (NaN under `MIN_CLASS_BINS`)."""
    value, valid = FAMILIES[family]
    ok = rows[valid].astype(bool) & np.isfinite(rows[value])
    positive, negative = classes(rows)[reference]
    positive, negative = positive & ok, negative & ok
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    area = float("nan")
    if min(n_pos, n_neg) >= MIN_CLASS_BINS:
        use = (positive | negative).to_numpy()
        area = bench.auroc(rows[value].to_numpy(float)[use], positive.to_numpy()[use])
    return {"n_detached": n_pos, "n_attached": n_neg, "auroc": area}


def summary(per_shot: dict, frame: pd.DataFrame, family: str, reference: str, rng):
    """Within-shot mean over the shots with both classes, and the pooled AUROC."""
    areas = np.asarray(
        [v[family][reference]["auroc"] for v in per_shot.values()], dtype=float
    )
    areas = areas[np.isfinite(areas)]
    draws = [
        float(np.mean(areas[rng.integers(0, len(areas), len(areas))]))
        for _ in range(bench.REPLICATES if len(areas) else 0)
    ]
    value, valid = FAMILIES[family]
    ok = frame[valid].astype(bool) & np.isfinite(frame[value])
    positive, negative = classes(frame)[reference]
    use = ((positive | negative) & ok).to_numpy()
    pooled = bench.auroc_boot(
        frame[value].to_numpy(float)[use],
        positive.to_numpy()[use],
        frame.shot.to_numpy()[use],
        rng,
    )
    return {
        "within_shot": {
            "min_class_bins": MIN_CLASS_BINS,
            "n_shots": len(areas),
            "mean": float(areas.mean()) if len(areas) else None,
            "mean_ci95": bench.interval(draws),
            "shots_above_chance": int((areas > 0.5).sum()),
            "median": float(np.median(areas)) if len(areas) else None,
        },
        "pooled": pooled,
    }


def main() -> int:
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    frame = te.attach_te(frame)
    rng = np.random.default_rng(0)
    split = frame.drop_duplicates("shot").set_index("shot").split.to_dict()
    per_shot = {}
    for shot, rows in frame.groupby("shot"):
        entry = {"split": split[shot]}
        for family in FAMILIES:
            entry[family] = {
                reference: shot_row(rows, family, reference)
                for reference in ("tangtv", "te")
            }
        has = any(
            entry[f][r]["n_detached"] + entry[f][r]["n_attached"] > 0
            for f in FAMILIES
            for r in ("tangtv", "te")
        )
        if has:
            per_shot[int(shot)] = entry
    record = {
        "role": "f_div is a within-shot corroborator, not a vote",
        "bin_ms": core.BIN_MS,
        "references": {
            "tangtv": "positive: TangTV detached vote; negative: attached vote "
            "(upper shelf, TangTV valid)",
            "te": "positive: divertor Thomson Te <= "
            f"{te.TE_DETACHED_MAX_EV:g} eV; negative: Te >= "
            f"{te.TE_ATTACHED_MIN_EV:g} eV",
        },
        "families": {
            "relative": "f_div over the shot's own baseline (prad_rel_value)",
            "absolute": "f_div over the heating power (prad_value), the "
            "201081-anchored family",
        },
        "summary": {
            family: {
                reference: summary(per_shot, frame, family, reference, rng)
                for reference in ("tangtv", "te")
            }
            for family in FAMILIES
        },
        "per_shot": {str(k): v for k, v in per_shot.items()},
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    for family in FAMILIES:
        for reference in ("tangtv", "te"):
            s = record["summary"][family][reference]
            print(
                family,
                reference,
                "within",
                s["within_shot"]["n_shots"],
                s["within_shot"]["mean"],
                "pooled",
                round(s["pooled"]["value"], 3),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
