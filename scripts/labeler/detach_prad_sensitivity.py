#!/usr/bin/env python
"""Label composition under the Prad,div cutoff family: primary, relative, band sweep.

The primary f_div cutoffs are absolute and anchored on shot 201081
(`thresholds.prad_cutoffs`: the midpoint of its measured attached and detached
Prad,div,L plus or minus `PRAD_BAND_MW`, over its measured P_in). The sensitivity
alternative votes on f_div over the shot's own baseline (`prad_rel_value`,
`thresholds.prad_relative_cutoffs`). This script recomputes the primary compatibility
rule from the exported bins under every variant (the band swept from 0.025 to
0.3 MW for both families, plus the published 1.6/2.2 MW values for the anchor) and
records, per variant, the certain bins by state, the shots and cohort shots they
sit on and whether the three-state criterion below holds. Two variants drop the
Afrac vote (the Jsat-ratio proxy is uncalibrated and its conflicts block
certainty). Nothing is selected from it: the variants only show how much the
composition moves.

Three-state criterion (the paper decision): certain attached AND certain detached
bins on at least 3 shots (the same shots), at least one of them in the cohort
(`train`, `val` or `test` of `data/events/catalog/cohort.csv`).

    pixi run --frozen -e labelmaker python scripts/labeler/detach_prad_sensitivity.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, label_model, prad
from labeler.events.detachment import thresholds as th

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
OUT = REPO / "docs/labeler/results/detachment_prad_sensitivity.json"
BANDS_MW = (0.025, 0.05, 0.1, 0.15, 0.2, 0.3)
MIN_CRITERION_SHOTS = 3
PUBLISHED_ATTACHED_MW, PUBLISHED_DETACHED_MW = 1.6, 2.2
COHORT = ("train", "val", "test")


def matrices(frame, prad_vote):
    votes = np.stack(
        [
            frame.afrac_vote.to_numpy(),
            prad_vote,
            frame.tangtv_vote.to_numpy(),
        ],
        axis=1,
    ).astype(int)
    valid = np.stack(
        [frame[f"{n}_valid"].to_numpy(bool) for n in label_model.LF_NAMES], axis=1
    )
    return votes, valid


def composition(frame, votes, valid) -> dict:
    state, tier = label_model.compatibility_decide(
        votes,
        valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share.to_numpy(float)),
    )
    certain = tier == "certain"
    split = frame.split.to_numpy()
    out = {"certain_bins": {}, "certain_shots": {}, "certain_cohort_shots": {}}
    shots = {}
    for code in core.VOTE_STATES:
        name = core.STATE_NAMES[code]
        hit = certain & (state == code)
        shots[code] = set(frame.shot[hit].astype(int))
        out["certain_bins"][name] = int(hit.sum())
        out["certain_bins"][f"{name}_by_split"] = {
            s: int((hit & (split == s)).sum()) for s in (*COHORT, "outside")
        }
        out["certain_shots"][name] = len(shots[code])
        out["certain_cohort_shots"][name] = len(
            {
                s
                for s in shots[code]
                if frame.loc[frame.shot == s, "split"].iloc[0] in COHORT
            }
        )
    both = shots[core.ATTACHED] & shots[core.DETACHED]
    in_cohort = {
        s for s in both if frame.loc[frame.shot == s, "split"].iloc[0] in COHORT
    }
    out["shots_with_certain_attached_and_detached"] = len(both)
    out["cohort_shots_with_certain_attached_and_detached"] = len(in_cohort)
    out["three_state_criterion_met"] = bool(
        len(both) >= MIN_CRITERION_SHOTS and len(in_cohort) >= 1
    )
    out["uncertain_bins"] = int((state == core.UNCERTAIN).sum())
    out["candidate_marfe_bins"] = int(
        ((state == core.UNCERTAIN) & frame.tangtv_marfe_candidate.to_numpy(bool)).sum()
    )
    return out


def main() -> int:
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    for name in label_model.LF_NAMES:
        frame[f"{name}_valid"] = frame[f"{name}_valid"].astype(bool)
    pvalid = frame.prad_valid.to_numpy(bool)
    absolute = frame.prad_value.to_numpy(float)
    relative = frame.prad_rel_value.to_numpy(float)

    def absolute_vote(lo, hi):
        return np.where(pvalid, prad.fdiv_vote(absolute, lo, hi), core.ABSTAIN)

    def relative_vote(lo, hi):
        return np.where(pvalid, prad.fdiv_vote(relative, lo, hi), core.ABSTAIN)

    variants = {}

    def add(name, vote, cutoffs, family, drop_afrac=False):
        votes, valid = matrices(frame, vote)
        if drop_afrac:
            # The Jsat-ratio proxy casts no vote (it stays a valid measurement).
            votes[:, label_model.LF_NAMES.index("afrac")] = core.ABSTAIN
        prad_votes = vote[pvalid]
        variants[name] = {
            "family": family,
            "afrac_votes_removed": drop_afrac,
            "cutoffs": [float(cutoffs[0]), float(cutoffs[1])],
            "prad_votes": {
                "attached": int((prad_votes == core.ATTACHED).sum()),
                "detached": int((prad_votes == core.DETACHED).sum()),
                "abstain": int((prad_votes == core.ABSTAIN).sum()),
            },
            **composition(frame, votes, valid),
        }

    primary = th.prad_cutoffs()
    add("primary_absolute", absolute_vote(*primary), primary, "absolute")
    relative_primary = th.prad_relative_cutoffs()
    add("relative", relative_vote(*relative_primary), relative_primary, "relative")
    add(
        "primary_absolute_afrac_abstains",
        absolute_vote(*primary),
        primary,
        "absolute",
        drop_afrac=True,
    )
    add(
        "relative_afrac_abstains",
        relative_vote(*relative_primary),
        relative_primary,
        "relative",
        drop_afrac=True,
    )
    mid = 0.5 * (th.PRAD_ANCHOR_ATTACHED_MW + th.PRAD_ANCHOR_DETACHED_MW)
    published = (
        (0.5 * (PUBLISHED_ATTACHED_MW + PUBLISHED_DETACHED_MW) - th.PRAD_BAND_MW)
        / th.PRAD_ANCHOR_P_IN_MW,
        (0.5 * (PUBLISHED_ATTACHED_MW + PUBLISHED_DETACHED_MW) + th.PRAD_BAND_MW)
        / th.PRAD_ANCHOR_P_IN_MW,
    )
    add(
        "absolute_published_anchor_values",
        absolute_vote(*published),
        published,
        "absolute",
    )
    for band in BANDS_MW:
        cut = th.prad_cutoffs(band_mw=band)
        add(f"absolute_band_{band:g}_mw", absolute_vote(*cut), cut, "absolute")
        cut = th.prad_relative_cutoffs(band_mw=band)
        add(f"relative_band_{band:g}_mw", relative_vote(*cut), cut, "relative")
    record = {
        "anchor": {
            "shot": th.PRAD_ANCHOR_SHOT,
            "attached_mw": th.PRAD_ANCHOR_ATTACHED_MW,
            "detached_mw": th.PRAD_ANCHOR_DETACHED_MW,
            "midpoint_mw": mid,
            "p_in_mw": th.PRAD_ANCHOR_P_IN_MW,
            "band_mw": th.PRAD_BAND_MW,
            "record": "docs/labeler/results/detachment_prad_anchor.json",
        },
        "criterion": (
            "certain attached and certain detached bins on at least "
            f"{MIN_CRITERION_SHOTS} of the same shots, at least one in the cohort"
        ),
        "criterion_min_shots": MIN_CRITERION_SHOTS,
        "n_assessed_bins": len(frame),
        "variants": variants,
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    for name in (
        "primary_absolute",
        "relative",
        "primary_absolute_afrac_abstains",
        "relative_afrac_abstains",
    ):
        v = variants[name]
        print(name, v["certain_bins"], v["three_state_criterion_met"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
