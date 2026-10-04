#!/usr/bin/env python
"""Record the current detachment label set: coverage, indicator provenance, D9.

    python scripts/labeler/detach_current_state.py

Reads `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` (every assessed bin, written
by `detach_label.py`) and the extracted per-shot grids in `bins/`, and writes
`docs/labeler/results/detachment_current.json`:

* the population by state, split and tier;
* Afrac: coverage on the upper shelf, the reasons a bin had no valid probe, and the
  provenance of the probe that voted (distance to the outer strike point, psiN,
  margin, how many probes were eligible);
* Prad,div: vote counts by tier with the invalid reasons, and the upper-shelf
  vote table against TangTV;
* MARFE: the evidence chain counted bin by bin (TangTV MARFE vote, the spatial cue,
  the density cue, persistence) and what became certain or `candidate_marfe`;
* the paper criterion (D9): certain attached and certain detached bins on the same
  shots, and whether one of them is a cohort shot.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from detach_json import dumps
from detach_label import load_all

from labeler.events.detachment import core, thresholds

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
OUT = REPO / "docs/labeler/results/detachment_current.json"
COHORT_SPLITS = ("train", "val", "test")
#: The criterion for presenting a three-state label set rather than an agreement
#: appendix: certain attached and certain detached bins on this many of the same
#: shots, at least one of them in the fixed cohort.
D9_MIN_SHOTS = 3


def population(frame: pd.DataFrame, keep=None) -> dict:
    sel = frame if keep is None else frame.loc[np.asarray(keep, bool)]
    return {
        "bins": len(sel),
        "shots": int(sel.shot.nunique()),
        "seconds": float(len(sel) * core.BIN_MS / 1000),
        "shot_ids": sorted(int(s) for s in sel.shot.unique()),
    }


def counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def population_block(labels: pd.DataFrame) -> dict:
    assessed = labels.assessed.to_numpy(bool)
    certain = assessed & labels.state_rule.isin(core.VOTE_STATES).to_numpy()
    out = {
        "assessed": population(labels, assessed),
        "certain": population(labels, certain),
        "by_state": {},
        "by_split": {},
        "by_tier": {},
    }
    for state in (1, 2, 3, 4):
        name = core.STATE_NAMES[state]
        out["by_state"][name] = population(
            labels, assessed & labels.state_rule.eq(state)
        )
    for split, rows in labels.groupby("split"):
        out["by_split"][str(split)] = {
            "assessed_bins": len(rows),
            "assessed_shots": int(rows.shot.nunique()),
            "certain_bins_by_state": {
                core.STATE_NAMES[s]: int((rows.state_rule == s).sum())
                for s in (1, 2, 3)
            },
        }
    out["by_tier"] = {
        str(tier): population(rows)
        | {
            "state_bins": {
                core.STATE_NAMES[s]: int((rows.state_rule == s).sum())
                for s in (1, 2, 3, 4)
            }
        }
        for tier, rows in labels.groupby("tier")
    }
    out["tangtv_geometry_tier_of_assessed_bins"] = counts(labels.tangtv_tier)
    return out


def earlier_gate_check(bins: pd.DataFrame) -> dict:
    """How often the selected probes meet the earlier gate (within 2 cm of the outer
    strike AND psiN >= 1.01). Few do, so the gate, not sparse EFIT flux maps, is
    why the earlier export had no upper-shelf Afrac vote."""
    have = bins[np.isfinite(bins.aux_jsat_selected_psin)]
    near = have.aux_jsat_selected_distance_m <= 0.02
    sol = have.aux_jsat_selected_psin >= 1.01
    return {
        "rule": "within 2 cm of the outer strike and psiN >= 1.01",
        "bins_with_selected_probe": len(have),
        "within_2cm": int(near.sum()),
        "psin_at_least_1.01": int(sol.sum()),
        "both": int((near & sol).sum()),
    }


def afrac_block(bins: pd.DataFrame, labels: pd.DataFrame) -> dict:
    upper = bins[bins.tangtv_tier.eq("upper_shelf")]
    valid = upper[upper.afrac_valid.astype(bool)]
    vote = valid[valid.afrac_vote > 0]
    return {
        "population": "every extracted 50 ms bin on a fetched shot, TangTV upper shelf",
        "all_bins": population(bins),
        "upper_shelf_bins": population(upper),
        "upper_shelf_valid": population(upper, upper.afrac_valid.astype(bool)),
        "upper_shelf_votes": {
            core.STATE_NAMES[s]: int((vote.afrac_vote == s).sum()) for s in (1, 2, 3)
        },
        "valid_bins_all_tiers": population(bins, bins.afrac_valid.astype(bool)),
        "reasons_upper_shelf": counts(upper.afrac_reason),
        "reasons_all_bins": counts(bins.afrac_reason),
        "selected_probe_provenance_valid_bins": {
            "distinct_probes": int(valid.aux_jsat_selected_probe.nunique()),
            "distance_to_outer_strike_m_p05_p50_p95": [
                float(v)
                for v in np.nanpercentile(
                    valid.aux_jsat_selected_distance_m, [5, 50, 95]
                )
            ]
            if len(valid)
            else None,
            "psin_min_median_max": [
                float(valid.aux_jsat_selected_psin.min()),
                float(valid.aux_jsat_selected_psin.median()),
                float(valid.aux_jsat_selected_psin.max()),
            ]
            if len(valid)
            else None,
            "radial_margin_m_min": float(valid.aux_jsat_radial_margin_m.min())
            if len(valid)
            else None,
            "eligible_probes_per_bin_median": float(
                valid.afrac_probe_n_eligible.median()
            )
            if len(valid)
            else None,
        },
        "provenance_exported_for_invalid_bins": {
            "columns": [
                "aux_jsat_selected_probe",
                "aux_jsat_selected_r_m",
                "aux_jsat_selected_z_m",
                "aux_jsat_selected_psin",
                "aux_jsat_selected_distance_m",
                "aux_jsat_radial_margin_m",
                "afrac_probe_n_eligible",
                "afrac_reason",
            ],
            "upper_shelf_invalid_bins": int((~upper.afrac_valid.astype(bool)).sum()),
            "of_which_with_reported_probe_psin": int(
                (
                    ~upper.afrac_valid.astype(bool)
                    & np.isfinite(upper.aux_jsat_selected_psin)
                ).sum()
            ),
        },
        "eligibility": {
            "sol_side": (
                f"psiN > {thresholds.PROBE_SOL_PSI_N_MIN:.3f} and at least "
                f"{thresholds.PROBE_STRIKE_MARGIN_M * 1e3:.0f} mm outboard of the "
                "outer strike point"
            ),
            "window": f"psiN <= {thresholds.PROBE_SOL_PSI_N_MAX:.2f}",
            "choice": "peak Jsat among the eligible probes",
        },
        "earlier_gate_check": earlier_gate_check(bins),
        "afrac_votes_in_certain_labels": {
            "certain_bins": int(labels.state_rule.isin(core.VOTE_STATES).sum()),
            "certain_bins_with_afrac_cast": int(
                (
                    labels.state_rule.isin(core.VOTE_STATES)
                    & labels.afrac_valid.astype(bool)
                    & (labels.afrac_vote > 0)
                ).sum()
            ),
        },
    }


def prad_block(bins: pd.DataFrame, labels: pd.DataFrame) -> dict:
    out = {"by_tier": {}}
    for tier, rows in bins.groupby("tangtv_tier"):
        valid = rows.prad_valid.astype(bool)
        out["by_tier"][str(tier)] = {
            "bins": len(rows),
            "valid": int(valid.sum()),
            "votes": {
                "attached": int((valid & rows.prad_vote.eq(1)).sum()),
                "detached": int((valid & rows.prad_vote.eq(2)).sum()),
                "abstain_between_cutoffs": int((valid & rows.prad_vote.lt(1)).sum()),
            },
            "invalid_reasons": counts(rows.prad_reason[~valid]),
        }
    upper = labels[labels.tangtv_tier.eq("upper_shelf")]
    pair = upper[upper.prad_valid.astype(bool) & upper.tangtv_valid.astype(bool)]
    table = pd.crosstab(
        pair.tangtv_vote.map({-1: "abstain", 1: "attached", 2: "detached", 3: "marfe"}),
        pair.prad_vote.map({-1: "abstain", 1: "attached", 2: "detached", 3: "marfe"}),
    )
    out["upper_shelf_vote_table_tangtv_rows_prad_columns"] = {
        str(r): {str(c): int(table.loc[r, c]) for c in table.columns}
        for r in table.index
    }
    out["upper_shelf_both_valid"] = population(pair)
    return out


def marfe_block(labels: pd.DataFrame) -> dict:
    marfe_vote = labels.tangtv_vote.eq(core.MARFE)
    return {
        "evidence_rule": (
            f"TangTV MARFE vote (DZ >= {thresholds.DZ_MARFE_MIN:.1f} held for "
            f"{thresholds.MARFE_MIN_BINS} adjacent valid bins), an emission peak "
            "inside the separatrix near the X-point, and the density cue "
            f"fG >= {thresholds.GREENWALD_CUE_MIN:.1f} (or a recorded H-L "
            "transition cue); Prad,div and Afrac do not corroborate a MARFE"
        ),
        "tangtv_marfe_vote": population(labels, marfe_vote),
        "persistent_height": population(
            labels, labels.tangtv_marfe_candidate.astype(bool)
        ),
        "spatial_cue": population(labels, labels.tangtv_marfe_spatial.astype(bool)),
        "density_cue": population(labels, labels.tangtv_marfe_second_cue.astype(bool)),
        "certain_marfe": population(labels, labels.state_rule.eq(core.MARFE)),
        "candidate_marfe_tier": population(labels, labels.tier.eq("candidate_marfe")),
        "marfe_vote_but_other_indicator_attached": population(
            labels,
            marfe_vote
            & (labels.prad_vote.eq(1) | labels.afrac_vote.eq(1))
            & ~labels.state_rule.eq(core.MARFE),
        ),
    }


def d9_block(labels: pd.DataFrame) -> dict:
    certain = labels[labels.state_rule.isin(core.VOTE_STATES)]
    att = set(certain.loc[certain.state_rule.eq(core.ATTACHED), "shot"].astype(int))
    det = set(certain.loc[certain.state_rule.eq(core.DETACHED), "shot"].astype(int))
    both = sorted(att & det)
    cohort = {
        int(s)
        for s, split in labels.drop_duplicates("shot")[["shot", "split"]].itertuples(
            index=False
        )
        if split in COHORT_SPLITS
    }
    both_cohort = [s for s in both if s in cohort]
    attached_cohort = sorted(att & cohort)
    detached_cohort = sorted(det & cohort)
    ok = len(both) >= D9_MIN_SHOTS and len(both_cohort) >= 1
    return {
        "criterion": "certain attached AND certain detached bins on at least "
        f"{D9_MIN_SHOTS} of the same shots, at least one of them a cohort shot "
        "(split train, val or test)",
        "shots_with_certain_attached": sorted(att),
        "shots_with_certain_detached": sorted(det),
        "shots_with_both": both,
        "cohort_shots_with_both": both_cohort,
        "cohort_shots_with_certain_attached": attached_cohort,
        "cohort_shots_with_certain_detached": detached_cohort,
        "met": bool(ok),
        "presentation": "three_state_label_set_exploratory_with_te_check"
        if ok
        else "indicator_agreement_appendix",
    }


def main() -> None:
    source = ROOT / "labels_bins.csv.gz"
    labels = pd.read_csv(source)
    bins = load_all(ROOT / "bins")
    bins = bins[bins.shot.isin(labels.shot.unique())].reset_index(drop=True)
    record = {
        "bin_ms": core.BIN_MS,
        "labels_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script": "scripts/labeler/detach_current_state.py",
        "scope": "exploratory labels; no independent benchmark",
        "population": population_block(labels),
        "afrac": afrac_block(bins, labels),
        "prad": prad_block(bins, labels),
        "marfe": marfe_block(labels),
        "paper_criterion_d9": d9_block(labels),
        "p_in": {
            "definition": "NBI + EFIT POH + ECH, centered 250 ms mean",
            "nbi_recovery": "docs/labeler/results/detachment_nbi_calibration.json",
        },
        "lower_shelf_window": {
            "note": "the lower-shelf TangTV extraction is not a label column and "
            "is absent from the detach-ui handoff; its bins carry tier "
            "lower_shelf_window and state uncertain",
            "bins": population(labels, labels.tier.eq("lower_shelf_window")),
        },
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    print(dumps(record["paper_criterion_d9"], indent=1))
    print(dumps(record["population"]["by_state"], indent=1)[:1200])


if __name__ == "__main__":
    main()
