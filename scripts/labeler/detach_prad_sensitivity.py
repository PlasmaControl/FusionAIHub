#!/usr/bin/env python
"""Label composition when f_div is added to the rule: relative and absolute cutoffs.

The exported rule has two voting indicators, TangTV and Afrac
(`label_model.SECOND_VOTERS`); f_div (Prad,div,L over the heating power) is a
within-shot corroborator and creates no conflict (owner decision 2026-10-04, after
Opus review 5, I1: its relative vote is at or below chance pooled over shots). This
script recomputes the compatibility rule from the exported bins with f_div ADDED as a
second voter, under the per-shot RELATIVE cutoffs (`prad_vote`,
`thresholds.prad_relative_cutoffs`) and under the ABSOLUTE cutoffs anchored on shot
201081 (`prad_abs_vote`, `thresholds.prad_cutoffs`: the midpoint of its measured
attached and detached Prad,div,L plus or minus `PRAD_BAND_MW`, over its measured
P_in), the band swept from 0.025 to 0.3 MW for both families, plus the published
1.6/2.2 MW values for the anchor. It also records the exported rule itself
(`primary`, which must reproduce the exported labels) and TangTV alone (`tangtv_alone`).
Per variant: the certain and tangtv_only bins by state, the shots and cohort shots they
sit on and whether the three-state criterion below holds. Nothing is selected from
it: the variants only show how much the composition moves.

It also records, per shot, the range of f_div (absolute and relative) in TangTV
attached and detached bins, so the per-shot dependence is visible, and the published
point 180257 at 4800 ms, where the absolute f_div voted attached.

Three-state criterion (the paper decision): attached AND detached bins of the tier
on at least 3 shots (the same shots), at least one of them in the cohort
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
PUBLISHED_POINT = (180257, 4800.0)


def matrices(frame, prad_vote, prad_valid):
    votes = np.stack(
        [
            frame.afrac_vote.to_numpy(),
            prad_vote,
            frame.tangtv_vote.to_numpy(),
        ],
        axis=1,
    ).astype(int)
    valid = np.stack(
        [
            frame.afrac_valid.to_numpy(bool),
            prad_valid,
            frame.tangtv_valid.to_numpy(bool),
        ],
        axis=1,
    )
    return votes, valid


def criterion(frame, hit_by_state, split) -> dict:
    """Shots with attached and detached bins of the tier, and the cohort shots."""
    shots = {
        code: set(frame.shot[hit_by_state[code]].astype(int))
        for code in (core.ATTACHED, core.DETACHED)
    }
    both = shots[core.ATTACHED] & shots[core.DETACHED]
    in_cohort = {s for s in both if split.get(s) in COHORT}
    return {
        "shots_with_attached_and_detached": len(both),
        "cohort_shots_with_attached_and_detached": len(in_cohort),
        "criterion_met": bool(len(both) >= MIN_CRITERION_SHOTS and len(in_cohort) >= 1),
    }


def composition(frame, votes, valid, second) -> dict:
    state, tier = label_model.compatibility_decide(
        votes,
        valid,
        tangtv_tier=frame.tangtv_tier.to_numpy(),
        elm_known=np.isfinite(frame.aux_elm_share.to_numpy(float)),
        second=second,
    )
    # the same relabel as `detach_label.py`: an uncertain bin with a persistent
    # high-front candidate is the tier `candidate_marfe` (the state stays uncertain)
    candidate = frame.tangtv_marfe_candidate.to_numpy(bool)
    tier = np.where(
        candidate
        & (state == core.UNCERTAIN)
        & ~np.isin(tier, ("lower_shelf_window", "elm_unknown", "geometry_unknown")),
        "candidate_marfe",
        tier,
    )
    split_by_shot = frame.drop_duplicates("shot").set_index("shot").split.to_dict()
    split = frame.split.to_numpy()
    out = {}
    for label, tiers in (
        ("certain", ("certain",)),
        ("tangtv_only", ("tangtv_only",)),
        ("certain_or_tangtv_only", ("certain", "tangtv_only")),
    ):
        selected = np.isin(tier, tiers)
        hits = {code: selected & (state == code) for code in core.VOTE_STATES}
        block = {
            "bins": {core.STATE_NAMES[c]: int(h.sum()) for c, h in hits.items()},
            "bins_by_split": {
                core.STATE_NAMES[c]: {
                    s: int((h & (split == s)).sum()) for s in (*COHORT, "outside")
                }
                for c, h in hits.items()
            },
            "shots": {
                core.STATE_NAMES[c]: int(frame.shot[h].nunique())
                for c, h in hits.items()
            },
            **criterion(frame, hits, split_by_shot),
        }
        out[label] = block
    out["uncertain_bins"] = int((state == core.UNCERTAIN).sum())
    out["candidate_marfe_bins"] = int((tier == "candidate_marfe").sum())
    out["tier_counts"] = {
        str(k): int(v) for k, v in pd.Series(tier).value_counts().items()
    }
    return out, state, tier


def per_shot_fdiv(frame: pd.DataFrame) -> dict:
    """f_div per shot in TangTV attached and detached upper-shelf bins (D2)."""
    tangtv = (
        frame.tangtv_tier.eq("upper_shelf")
        & frame.tangtv_valid.astype(bool)
        & frame.tangtv_vote.isin((core.ATTACHED, core.DETACHED))
    )

    def quantiles(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return [float(v) for v in np.percentile(x, [10, 50, 90])] if len(x) else None

    def shares(rows, vote, valid):
        n = int(valid.sum())
        return {
            "n_valid_bins": n,
            "attached": float((valid & vote.eq(core.ATTACHED)).sum() / n)
            if n
            else None,
            "detached": float((valid & vote.eq(core.DETACHED)).sum() / n)
            if n
            else None,
        }

    shots = {}
    for shot, rows in frame.groupby("shot"):
        has = rows.prad_abs_valid.astype(bool) | rows.prad_valid.astype(bool)
        if not has.any():
            continue
        relative = rows.prad_rel_value.to_numpy(float)
        absolute = rows.prad_value.to_numpy(float)
        ok = np.isfinite(relative) & np.isfinite(absolute) & (relative > 0)
        entry = {
            "bins_valid_absolute": int(rows.prad_abs_valid.astype(bool).sum()),
            "bins_valid_relative": int(rows.prad_valid.astype(bool).sum()),
            "p_in_mw_median": float(np.nanmedian(rows.aux_p_in_w) / 1e6)
            if np.isfinite(rows.aux_p_in_w).any()
            else None,
            "baseline_f_div": float(np.median(absolute[ok] / relative[ok]))
            if ok.any()
            else None,
            "f_div_absolute_10_50_90": quantiles(absolute[rows.prad_abs_valid]),
            "f_div_relative_10_50_90": quantiles(relative),
        }
        for name, code in (("attached", core.ATTACHED), ("detached", core.DETACHED)):
            sel = tangtv.loc[rows.index] & rows.tangtv_vote.eq(code)
            entry[f"tangtv_{name}_bins"] = int(sel.sum())
            entry[f"f_div_absolute_median_tangtv_{name}"] = (
                float(np.nanmedian(rows.prad_value[sel & rows.prad_abs_valid]))
                if (sel & rows.prad_abs_valid).any()
                else None
            )
            entry[f"f_div_relative_median_tangtv_{name}"] = (
                float(np.nanmedian(rows.prad_rel_value[sel & rows.prad_valid]))
                if (sel & rows.prad_valid.astype(bool)).any()
                else None
            )
            entry[f"absolute_votes_in_tangtv_{name}"] = shares(
                rows,
                pd.Series(
                    np.where(
                        rows.prad_abs_valid.astype(bool),
                        rows.prad_abs_vote,
                        core.ABSTAIN,
                    ),
                    index=rows.index,
                ),
                (sel & rows.prad_abs_valid.astype(bool)),
            )
            entry[f"relative_votes_in_tangtv_{name}"] = shares(
                rows, rows.prad_vote, (sel & rows.prad_valid.astype(bool))
            )
        shots[str(int(shot))] = entry
    pooled = {}
    for name, code in (("attached", core.ATTACHED), ("detached", core.DETACHED)):
        sel = tangtv & frame.tangtv_vote.eq(code)
        for family, valid, vote in (
            ("absolute", frame.prad_abs_valid.astype(bool), frame.prad_abs_vote),
            ("relative", frame.prad_valid.astype(bool), frame.prad_vote),
        ):
            use = sel & valid
            pooled[f"{family}_in_tangtv_{name}"] = {
                "bins": int(use.sum()),
                "shots": int(frame.shot[use].nunique()),
                "voted_attached": int((use & vote.eq(core.ATTACHED)).sum()),
                "voted_detached": int((use & vote.eq(core.DETACHED)).sum()),
                "abstained": int((use & (vote <= 0)).sum()),
                "shots_with_an_attached_vote": int(
                    frame.shot[use & vote.eq(core.ATTACHED)].nunique()
                ),
            }
    return {
        "population": "upper-shelf bins where TangTV is valid and votes attached "
        "or detached, per shot; f_div in MW/MW (absolute) and over the shot "
        "baseline (relative)",
        "pooled": pooled,
        "shots": shots,
    }


def published_point(frame: pd.DataFrame) -> dict:
    """The published detached point 180257 at 4800 ms, indicator by indicator."""
    shot, time = PUBLISHED_POINT
    rows = frame[
        (frame.shot == shot) & (frame.start_ms <= time) & (time < frame.start_ms + 50)
    ]
    path = ROOT / "bins" / f"{shot}.npz"
    with np.load(path) as f:
        bins = {k: f[k] for k in f.files}
    k = int(
        np.flatnonzero((bins["start_ms"] <= time) & (time < bins["start_ms"] + 50))[0]
    )

    def at(key):
        value = bins[key][k]
        return value.item() if hasattr(value, "item") else value

    out = {
        "shot": shot,
        "published_time_ms": time,
        "published_state": "detached",
        "bin_start_ms": float(bins["start_ms"][k]),
        "f_div_absolute": float(at("prad_value")),
        "f_div_absolute_vote": int(at("prad_abs_vote"))
        if at("prad_abs_valid")
        else None,
        "f_div_relative": float(at("prad_rel_value")),
        "f_div_relative_vote": int(at("prad_vote")) if at("prad_valid") else None,
        "f_div_relative_valid": bool(at("prad_valid")),
        "f_div_relative_reason": str(at("prad_reason")),
        "p_in_mw": float(at("aux_p_in_w")) / 1e6,
        "afrac_valid": bool(at("afrac_valid")),
        "afrac_value": float(at("afrac_value")),
        "afrac_vote": int(at("afrac_vote")),
        "afrac_reason": str(at("afrac_reason")),
        "tangtv_valid": bool(at("tangtv_valid")),
        "tangtv_vote": int(at("tangtv_vote")),
        "tangtv_tier": str(at("tangtv_tier")),
        "regime": str(at("regime")),
        "exported_state": core.STATE_NAMES.get(int(rows.state_rule.iloc[0]), "unknown")
        if len(rows)
        else "not_assessed",
        "exported_tier": str(rows.tier.iloc[0]) if len(rows) else "not_assessed",
    }
    return out


def main() -> int:
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    for name in label_model.LF_NAMES:
        frame[f"{name}_valid"] = frame[f"{name}_valid"].astype(bool)
    frame["prad_abs_valid"] = frame.prad_abs_valid.astype(bool)
    relative_valid = frame.prad_valid.to_numpy(bool)
    absolute_valid = frame.prad_abs_valid.to_numpy(bool)
    absolute = frame.prad_value.to_numpy(float)
    relative = frame.prad_rel_value.to_numpy(float)

    def absolute_vote(lo, hi):
        return np.where(absolute_valid, prad.fdiv_vote(absolute, lo, hi), core.ABSTAIN)

    def relative_vote(lo, hi):
        return np.where(relative_valid, prad.fdiv_vote(relative, lo, hi), core.ABSTAIN)

    variants = {}

    def add(name, vote, valid, cutoffs, family, second):
        """One variant: `second` names the indicators that vote beside TangTV."""
        votes, valid_matrix = matrices(frame, vote, valid)
        prad_votes = vote[valid]
        block, state, tier = composition(frame, votes, valid_matrix, second)
        variants[name] = {
            "family": family,
            "second_voters": list(second),
            "cutoffs": None
            if cutoffs is None
            else [float(cutoffs[0]), float(cutoffs[1])],
            "prad_votes": {
                "attached": int((prad_votes == core.ATTACHED).sum()),
                "detached": int((prad_votes == core.DETACHED).sum()),
                "abstain": int((prad_votes == core.ABSTAIN).sum()),
            },
            **block,
        }
        return state, tier

    relative_cut = th.prad_relative_cutoffs()
    absolute_cut = th.prad_cutoffs()
    plain = relative_vote(*relative_cut)  # a bystander in the exported rule
    state, tier = add(
        "primary", plain, relative_valid, None, "none", label_model.SECOND_VOTERS
    )
    variants["primary"]["reproduces_exported_labels"] = bool(
        np.array_equal(state, frame.state_rule.to_numpy())
        and np.array_equal(tier, frame.tier.to_numpy())
    )
    add("tangtv_alone", plain, relative_valid, None, "none", ())
    both = ("afrac", "prad")
    add(
        "with_relative_fdiv",
        relative_vote(*relative_cut),
        relative_valid,
        relative_cut,
        "relative",
        both,
    )
    add(
        "with_absolute_fdiv",
        absolute_vote(*absolute_cut),
        absolute_valid,
        absolute_cut,
        "absolute",
        both,
    )
    add(
        "with_relative_fdiv_afrac_abstains",
        relative_vote(*relative_cut),
        relative_valid,
        relative_cut,
        "relative",
        ("prad",),
    )
    add(
        "with_absolute_fdiv_afrac_abstains",
        absolute_vote(*absolute_cut),
        absolute_valid,
        absolute_cut,
        "absolute",
        ("prad",),
    )
    mid = 0.5 * (th.PRAD_ANCHOR_ATTACHED_MW + th.PRAD_ANCHOR_DETACHED_MW)
    published = (
        (0.5 * (PUBLISHED_ATTACHED_MW + PUBLISHED_DETACHED_MW) - th.PRAD_BAND_MW)
        / th.PRAD_ANCHOR_P_IN_MW,
        (0.5 * (PUBLISHED_ATTACHED_MW + PUBLISHED_DETACHED_MW) + th.PRAD_BAND_MW)
        / th.PRAD_ANCHOR_P_IN_MW,
    )
    add(
        "with_absolute_fdiv_published_anchor_values",
        absolute_vote(*published),
        absolute_valid,
        published,
        "absolute",
        both,
    )
    for band in BANDS_MW:
        cut = th.prad_relative_cutoffs(band_mw=band)
        add(
            f"with_relative_fdiv_band_{band:g}_mw",
            relative_vote(*cut),
            relative_valid,
            cut,
            "relative",
            both,
        )
        cut = th.prad_cutoffs(band_mw=band)
        add(
            f"with_absolute_fdiv_band_{band:g}_mw",
            absolute_vote(*cut),
            absolute_valid,
            cut,
            "absolute",
            both,
        )
    record = {
        "primary": "TangTV and Afrac vote; f_div is not a vote",
        "second_voters": list(label_model.SECOND_VOTERS),
        "anchor": {
            "shot": th.PRAD_ANCHOR_SHOT,
            "attached_mw": th.PRAD_ANCHOR_ATTACHED_MW,
            "detached_mw": th.PRAD_ANCHOR_DETACHED_MW,
            "midpoint_mw": mid,
            "p_in_mw": th.PRAD_ANCHOR_P_IN_MW,
            "p_in_definition": "median of the 250 ms-averaged P_in over the 32 "
            "50 ms bins of the anchor shot's attached window",
            "band_mw": th.PRAD_BAND_MW,
            "relative_cutoffs": list(relative_cut),
            "absolute_cutoffs_global": list(absolute_cut),
            "record": "docs/labeler/results/detachment_prad_anchor.json",
        },
        "criterion": (
            "attached and detached bins of the tier on at least "
            f"{MIN_CRITERION_SHOTS} of the same shots, at least one in the cohort"
        ),
        "criterion_min_shots": MIN_CRITERION_SHOTS,
        "n_assessed_bins": len(frame),
        "variants": variants,
        "per_shot_f_div": per_shot_fdiv(frame),
        "published_point_180257_4800_ms": published_point(frame),
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    for name in (
        "primary",
        "tangtv_alone",
        "with_relative_fdiv",
        "with_absolute_fdiv",
    ):
        v = variants[name]
        print(name, v["certain"]["bins"], v["tangtv_only"]["bins"])
    print(dumps(record["published_point_180257_4800_ms"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
