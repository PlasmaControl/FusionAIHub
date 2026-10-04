#!/usr/bin/env python
"""Record the current detachment label set: coverage, indicator provenance, D9.

    python scripts/labeler/detach_current_state.py

Reads `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` (every assessed bin, written
by `detach_label.py`) and the extracted per-shot grids in `bins/`, and writes
`docs/labeler/results/detachment_current.json`:

* the population by state, split and tier (`certain`: TangTV vote plus an agreeing
  Afrac vote; `tangtv_only`: TangTV vote with Afrac abstaining or invalid, silver;
  `conflict`: TangTV and Afrac disagree);
* Afrac: coverage on the upper shelf, the reasons a bin had no valid probe, and the
  provenance of the probe that voted (psiN, its attached reference, how many probes
  were inside the window), and the L/H gate's abstentions;
* Prad,div (f_div, a within-shot corroborator, not a vote): vote counts by tier with
  the invalid reasons (the relative vote and the absolute one, both sensitivities),
  and the upper-shelf vote tables against TangTV;
* MARFE: the evidence chain counted bin by bin (TangTV MARFE vote, the spatial cue,
  the density cue, persistence), `candidate_marfe` (no MARFE state is exported) and
  the recall on the one published MARFE (199166 at 3705 ms);
* the paper criterion (D9): attached and detached bins on the same shots, for the
  certain tier alone and for certain plus `tangtv_only`, and whether one of them is
  a cohort shot.
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
    certain = assessed & labels.tier.eq("certain").to_numpy()
    silver = assessed & labels.tier.eq("tangtv_only").to_numpy()
    labelled = assessed & labels.state_rule.isin((core.ATTACHED, core.DETACHED))
    out = {
        "assessed": population(labels, assessed),
        "certain": population(labels, certain),
        "tangtv_only": population(labels, silver),
        "labelled_certain_or_tangtv_only": population(labels, labelled),
        "by_state": {},
        "by_state_certain": {},
        "by_state_tangtv_only": {},
        "by_split": {},
        "by_tier": {},
    }
    for state in (1, 2, 4):
        name = core.STATE_NAMES[state]
        out["by_state"][name] = population(
            labels, assessed & labels.state_rule.eq(state)
        )
    for state in (1, 2):
        name = core.STATE_NAMES[state]
        out["by_state_certain"][name] = population(
            labels, certain & labels.state_rule.eq(state)
        )
        out["by_state_tangtv_only"][name] = population(
            labels, silver & labels.state_rule.eq(state)
        )
    for split, rows in labels.groupby("split"):
        out["by_split"][str(split)] = {
            "assessed_bins": len(rows),
            "assessed_shots": int(rows.shot.nunique()),
            "certain_bins_by_state": {
                core.STATE_NAMES[s]: int(
                    ((rows.state_rule == s) & rows.tier.eq("certain")).sum()
                )
                for s in (1, 2)
            },
            "tangtv_only_bins_by_state": {
                core.STATE_NAMES[s]: int(
                    ((rows.state_rule == s) & rows.tier.eq("tangtv_only")).sum()
                )
                for s in (1, 2)
            },
        }
    out["by_tier"] = {
        str(tier): population(rows)
        | {
            "state_bins": {
                core.STATE_NAMES[s]: int((rows.state_rule == s).sum())
                for s in (1, 2, 4)
            }
        }
        for tier, rows in labels.groupby("tier")
    }
    out["tangtv_geometry_tier_of_assessed_bins"] = counts(labels.tangtv_tier)
    return out


def afrac_block(bins: pd.DataFrame, labels: pd.DataFrame) -> dict:
    upper = bins[bins.tangtv_tier.eq("upper_shelf")]
    valid = upper[upper.afrac_valid.astype(bool)]
    vote = valid[valid.afrac_vote > 0]
    labelled = labels.state_rule.isin((core.ATTACHED, core.DETACHED))
    certain = labels.tier.eq("certain")
    silver = labels.tier.eq("tangtv_only")
    valid_afrac = labels.afrac_valid.astype(bool)
    cast = valid_afrac & (labels.afrac_vote > 0)
    return {
        "population": "every extracted 50 ms bin on a fetched shot, TangTV upper shelf",
        "method": "per_probe_reference",
        "all_bins": population(bins),
        "upper_shelf_bins": population(upper),
        "upper_shelf_valid": population(upper, upper.afrac_valid.astype(bool)),
        "upper_shelf_votes": {
            core.STATE_NAMES[s]: int((vote.afrac_vote == s).sum()) for s in (1, 2)
        },
        "valid_bins_all_tiers": population(bins, bins.afrac_valid.astype(bool)),
        "reasons_upper_shelf": counts(upper.afrac_reason),
        "reasons_all_bins": counts(bins.afrac_reason),
        "l_mode_abstentions": population(bins, bins.afrac_reason.eq("l_mode")),
        "regime_of_all_bins": counts(bins.regime),
        "regime_source_of_all_bins": counts(bins.regime_source),
        "selected_probe_provenance_valid_bins": {
            "distinct_probes": int(valid.aux_jsat_selected_probe.nunique()),
            "psin_min_median_max": [
                float(valid.aux_jsat_selected_psin.min()),
                float(valid.aux_jsat_selected_psin.median()),
                float(valid.aux_jsat_selected_psin.max()),
            ]
            if len(valid)
            else None,
            "probes_inside_window_per_bin_median": float(
                valid.afrac_probe_n_eligible.median()
            )
            if len(valid)
            else None,
            "attached_reference_median": float(valid.aux_jsat_reference.median())
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
                "aux_jsat_reference",
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
            "window": f"|psiN - 1| <= {thresholds.AFRAC_PSI_WINDOW}",
            "reference": f"per probe: the {thresholds.AFRAC_REFERENCE_QUANTILE} "
            "quantile of its own model-normalised current over its bins in the "
            f"window, at least {thresholds.AFRAC_REFERENCE_MIN_BINS} bins",
            "choice": "the probe nearest the separatrix in flux among those with a "
            "reference",
            "gate": "ELM, ramp, low power, and known L-mode bins abstain",
        },
        "afrac_in_labelled_bins": {
            "labelled_bins_certain_or_tangtv_only": int(labelled.sum()),
            "certain_bins": int(certain.sum()),
            "certain_bins_with_afrac_cast": int((certain & cast).sum()),
            "tangtv_only_bins": int(silver.sum()),
            "tangtv_only_afrac_valid_between_cutoffs": int(
                (silver & valid_afrac & ~cast).sum()
            ),
            "tangtv_only_afrac_invalid": int((silver & ~valid_afrac).sum()),
            "tangtv_only_afrac_invalid_reasons": counts(
                labels.afrac_reason[silver & ~valid_afrac]
            ),
        },
    }


def prad_block(bins: pd.DataFrame, labels: pd.DataFrame) -> dict:
    out = {
        "role": "within-shot corroborator, not a vote: the relative and absolute "
        "votes below are sensitivities",
        "by_tier": {},
        "absolute_sensitivity_by_tier": {},
    }
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
        absolute = rows.prad_abs_valid.astype(bool)
        out["absolute_sensitivity_by_tier"][str(tier)] = {
            "valid": int(absolute.sum()),
            "votes": {
                "attached": int((absolute & rows.prad_abs_vote.eq(1)).sum()),
                "detached": int((absolute & rows.prad_abs_vote.eq(2)).sum()),
                "abstain_between_cutoffs": int(
                    (absolute & rows.prad_abs_vote.lt(1)).sum()
                ),
            },
        }
    upper = labels[labels.tangtv_tier.eq("upper_shelf")]
    names = {-1: "abstain", 1: "attached", 2: "detached", 3: "marfe"}
    for key, valid_column, vote_column in (
        ("upper_shelf_vote_table_tangtv_rows_prad_columns", "prad_valid", "prad_vote"),
        (
            "upper_shelf_vote_table_tangtv_rows_absolute_prad_columns",
            "prad_abs_valid",
            "prad_abs_vote",
        ),
    ):
        pair = upper[upper[valid_column].astype(bool) & upper.tangtv_valid.astype(bool)]
        table = pd.crosstab(pair.tangtv_vote.map(names), pair[vote_column].map(names))
        out[key] = {
            str(r): {str(c): int(table.loc[r, c]) for c in table.columns}
            for r in table.index
        }
        out[key.replace("vote_table", "both_valid")] = population(pair)
    return out


def published_marfe(labels: pd.DataFrame) -> dict:
    """The one published MARFE: 199166 at 3705 ms (Chen 2026), bin by bin."""
    start = 3700.0
    rows = labels[
        (labels.shot == 199166) & labels.start_ms.between(start - 50, start + 100)
    ]
    cols = [
        "start_ms",
        "tangtv_vote",
        "tangtv_value",
        "tangtv_marfe_candidate",
        "tangtv_marfe_spatial",
        "tangtv_marfe_second_cue",
        "aux_greenwald_fraction",
        "tier",
        "state_rule",
    ]
    bin_row = rows[rows.start_ms.eq(start)]
    is_candidate = bool(len(bin_row) and bin_row.tier.eq("candidate_marfe").all())
    has_marfe = bool(len(bin_row) and bin_row.state_rule.eq(core.MARFE).any())
    return {
        "shot": 199166,
        "published_time_ms": 3705.0,
        "bins": [
            {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in rows[cols].to_dict("records")
        ],
        "bin_containing_the_published_time_is_candidate_marfe": is_candidate,
        "bin_containing_the_published_time_has_marfe_state": has_marfe,
        "recall_marfe_state": f"{int(has_marfe)}/1",
        "recall_candidate_marfe": f"{int(is_candidate)}/1",
    }


def marfe_block(labels: pd.DataFrame) -> dict:
    marfe_vote = labels.tangtv_vote.eq(core.MARFE)
    return {
        "evidence_rule": (
            f"TangTV MARFE vote (DZ >= {thresholds.DZ_MARFE_MIN:.1f} held for "
            f"{thresholds.MARFE_MIN_BINS} adjacent valid bins), an emission peak "
            "inside the separatrix near the X-point, and the density cue "
            f"fG >= {thresholds.GREENWALD_CUE_MIN:.1f}; Prad,div and Afrac do not "
            "corroborate a MARFE"
        ),
        "export_policy": "no MARFE state is exported: the fG cue has no literature "
        "source (Dong 2025 gives fG ~> 0.5 on HL-3 with a core-point density and a "
        "core-Te condition), so a TangTV MARFE vote is the tier `candidate_marfe` "
        "with state uncertain",
        "tangtv_marfe_vote": population(labels, marfe_vote),
        "persistent_height": population(
            labels, labels.tangtv_marfe_candidate.astype(bool)
        ),
        "spatial_cue": population(labels, labels.tangtv_marfe_spatial.astype(bool)),
        "density_cue": population(labels, labels.tangtv_marfe_second_cue.astype(bool)),
        "marfe_state_bins": population(labels, labels.state_rule.eq(core.MARFE)),
        "candidate_marfe_tier": population(labels, labels.tier.eq("candidate_marfe")),
        "published_marfe_199166": published_marfe(labels),
    }


def d9_block(labels: pd.DataFrame) -> dict:
    cohort = {
        int(s)
        for s, split in labels.drop_duplicates("shot")[["shot", "split"]].itertuples(
            index=False
        )
        if split in COHORT_SPLITS
    }

    def criterion(selected: pd.DataFrame) -> dict:
        att = set(
            selected.loc[selected.state_rule.eq(core.ATTACHED), "shot"].astype(int)
        )
        det = set(
            selected.loc[selected.state_rule.eq(core.DETACHED), "shot"].astype(int)
        )
        both = sorted(att & det)
        both_cohort = [s for s in both if s in cohort]
        ok = len(both) >= D9_MIN_SHOTS and len(both_cohort) >= 1
        return {
            "shots_with_attached": sorted(att),
            "shots_with_detached": sorted(det),
            "shots_with_both": both,
            "cohort_shots_with_both": both_cohort,
            "cohort_shots_with_attached": sorted(att & cohort),
            "cohort_shots_with_detached": sorted(det & cohort),
            "met": bool(ok),
        }

    certain = criterion(labels[labels.tier.eq("certain")])
    both_tiers = criterion(labels[labels.tier.isin(("certain", "tangtv_only"))])
    return {
        "criterion": "attached AND detached bins on at least "
        f"{D9_MIN_SHOTS} of the same shots, at least one of them a cohort shot "
        "(split train, val or test)",
        "certain": certain,
        "certain_or_tangtv_only": both_tiers,
        "met": certain["met"],
        "presentation": "three_state_label_set_exploratory_with_te_check"
        if certain["met"]
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
            "anchor_p_in_mw": thresholds.PRAD_ANCHOR_P_IN_MW,
            "anchor_p_in_definition": "median over the 32 attached-window bins of "
            "shot 201081 of the 250 ms-averaged P_in",
            "nbi_recovery": "docs/labeler/results/detachment_nbi_calibration.json",
        },
        "lower_shelf_window": {
            "note": "the lower-shelf TangTV extraction is invalid as a vote "
            "(reason lower_shelf_window, value kept) and absent from the detach-ui "
            "handoff; its bins carry tier lower_shelf_window and state uncertain",
            "bins": population(labels, labels.tier.eq("lower_shelf_window")),
        },
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    print(dumps(record["paper_criterion_d9"]["certain"], indent=1)[:600])
    print(dumps(record["population"]["by_tier"], indent=1)[:2400])


if __name__ == "__main__":
    main()
