#!/usr/bin/env python
"""Write the detachment Figure 2 panel: coverage by tier, agreement with Te and f_div.

There is no independent detachment benchmark. The label set is the geometry-gated
TangTV state, checked against the divertor Thomson Te (a temperature none of the
three indicators uses; Te has since informed rule decisions, so this is a consistency
check, not a validation). The panel shows

* the coverage of the exported label by tier, labelled bins first (TangTV + Afrac
  agreement, tier `certain`, which is agreement and not a confidence level; TangTV
  only, silver; `tangtv_only_lmode`, uncertain because TangTV votes detached on a
  known L-mode or probable-L phase; the `candidate_marfe` tier; every other
  uncertain bin), bins and shots;
* the agreement of the TangTV state with Te: the AUROC of -Te for a detached against
  an attached vote, pooled over shots with a 95% shot-bootstrap interval. Filled
  markers are the TangTV vote before the L-mode gate (every shot, without the anchor
  shot 201081, and by regime: known L-mode, probable L-mode, probable H-mode,
  no regime; the known-H bins are one shot, so not estimable and not drawn); open
  markers are the label tiers (labelled bins, the agreement tier, the TangTV-only
  tier) and the two indicators' votes (f_div is drawn for reference, it is not a vote
  of the label). The share of detached votes at or below 5 eV is written under the
  regime rows. Whether the second vote improves the Te agreement is the
  `second_vote_effect` of the Te record, copied into the panel source. A row on
  fewer than `detach_benchmark.MIN_INTERVAL_SHOTS` shots has a value and no interval
  (the record says "n/a (N shots)") and is not drawn;
* f_div as a within-shot corroborator: its AUROC against the TangTV vote and against
  Te, pooled over shots (filled blue diamond) and the mean over shots (open blue
  triangle; the shapes and colour differ from the Te panel's black circles), for the
  per-shot relative value and the absolute ratio, with the shot counts.

Every number is read from `docs/labeler/results/detachment_benchmark.json`,
`detachment_te_check.json`, `detachment_fdiv_check.json` and the exported bins. The
panel source is `docs/labeler/figure2_detach.json`; there is no second copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import detach_benchmark as bench
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detach_json import dumps
from detach_label import load_all
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from labeler.events.detachment import core, thresholds

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "labeler" / "results"
PANEL_JSON = REPO / "docs" / "labeler" / "figure2_detach.json"
AGREEMENT_JSON = RESULTS / "detachment_benchmark.json"
TE_JSON = RESULTS / "detachment_te_check.json"
FDIV_JSON = RESULTS / "detachment_fdiv_check.json"
CURRENT_JSON = RESULTS / "detachment_current.json"
FIG_HEIGHT_IN = 8.8
#: Colour and shapes of the f_div panel: not the Te panel's black circles.
FDIV_COLOUR = "#0072B2"
LF_NAMES = ("afrac", "prad", "tangtv")
#: Okabe-Ito colours of the exported classes; silver (TangTV only) is the tint.
CLASS_COLOUR = {
    "attached_certain": "#0072B2",
    "attached_silver": "#8EC1E3",
    "detached_certain": "#E69F00",
    "detached_silver": "#F4D58D",
    "lmode_gate": "#009E73",
    "candidate_marfe": "#CC79A7",
    "uncertain": "#999999",
}
CLASS_LABEL = {
    "attached_certain": "attached, agreement tier",
    "attached_silver": "attached, TangTV only",
    "detached_certain": "detached, agreement tier",
    "detached_silver": "detached, TangTV only",
    "lmode_gate": "uncertain, L-mode gate",
    "candidate_marfe": "candidate MARFE",
    "uncertain": "uncertain, other",
}
SETTINGS = {
    "afrac": "Jsat ratio against the probe's own attached reference, probe nearest "
    "the separatrix within 0.01 in psiN, L-mode bins abstain, uncalibrated",
    "prad": "Prad,div,L over P_in, 250 ms inter-ELM means; a within-shot "
    "corroborator, not a vote of the label (its relative and 201081-anchored "
    "absolute votes are sensitivities)",
    "tangtv": "C-III front height with shelf geometry, quality and MARFE gates",
}
PAPER_TIER = "upper_shelf"


def clean_ci(ci) -> list[float] | None:
    """The interval as floats, or None where it is missing (a null in the record is
    "n/a": fewer than `bench.MIN_INTERVAL_SHOTS` shots, or no valid draw)."""
    if not ci or any(x is None or not np.isfinite(x) for x in ci):
        return None
    return [float(x) for x in ci]


def n_a(n_shots) -> str:
    """The text for an interval that does not exist: 'n/a (1 shot)'."""
    n = int(n_shots or 0)
    if n >= bench.MIN_INTERVAL_SHOTS:
        return "n/a (no valid bootstrap draw)"
    return f"n/a ({n} shot{'' if n == 1 else 's'})"


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def population(frame: pd.DataFrame, keep) -> dict:
    """Count disjoint 50 ms bins; shot counts across states need not add up."""
    selected = frame.loc[np.asarray(keep, bool)]
    return {
        "bins": len(selected),
        "shots": int(selected.shot.nunique()),
        "seconds": float(len(selected) * core.BIN_MS / 1000),
        "shot_ids": sorted(int(s) for s in selected.shot.unique()),
    }


def label_class(frame: pd.DataFrame) -> pd.Series:
    """The class of each bin's exported label (tier-aware)."""
    state, tier = frame.state_rule, frame.tier
    out = pd.Series("uncertain", index=frame.index)
    for name, code in (("attached", core.ATTACHED), ("detached", core.DETACHED)):
        out[(tier == "certain") & (state == code)] = f"{name}_certain"
        out[(tier == "tangtv_only") & (state == code)] = f"{name}_silver"
    out[tier == "tangtv_only_lmode"] = "lmode_gate"
    out[tier == "candidate_marfe"] = "candidate_marfe"
    return out


def coverage_group(frame: pd.DataFrame) -> dict:
    """Disjoint classes of the assessed bins, and the groups the figure draws."""
    assessed = frame.assessed.to_numpy(bool)
    klass = label_class(frame)
    by_class = {
        name: population(frame, assessed & klass.eq(name).to_numpy())
        for name in CLASS_COLOUR
    }
    certain = assessed & frame.tier.eq("certain").to_numpy()
    labelled = assessed & frame.tier.isin(("certain", "tangtv_only")).to_numpy()
    return {
        "assessed": population(frame, assessed),
        "labelled_certain_or_tangtv_only": population(frame, labelled),
        "certain": population(frame, certain),
        "by_class": by_class,
        "by_tier": {
            str(tier): population(frame, assessed & frame.tier.eq(tier).to_numpy())
            for tier in sorted(frame.tier.dropna().unique())
        },
    }


def indicator_coverage(labels: pd.DataFrame) -> list[dict]:
    """Measurement coverage on all extracted grids, before eligibility selection."""
    bins = load_all(root() / "bins")
    index = pd.MultiIndex.from_frame(bins[["shot", "start_ms"]])
    tiers = labels.set_index(["shot", "start_ms"]).tier.reindex(index).to_numpy()
    table = []
    for name in LF_NAMES:
        valid = bins[f"{name}_valid"].to_numpy(bool)
        cast = valid & bins[f"{name}_vote"].isin(core.VOTE_STATES).to_numpy()
        table.append(
            {
                "indicator": name,
                "setting": SETTINGS[name],
                "population": "all extracted shot/bin grids, every split",
                "population_counts": population(bins, np.ones(len(bins), bool)),
                "measurement": population(
                    bins, np.isfinite(bins[f"{name}_value"].to_numpy(float))
                ),
                "valid_measurement": population(bins, valid),
                "vote": population(bins, cast),
                "certain_label": population(bins, cast & (tiers == "certain")),
                "labelled_certain_or_tangtv_only": population(
                    bins, cast & np.isin(tiers, ("certain", "tangtv_only"))
                ),
            }
        )
    return table


def agreement_rows(benchmark: dict) -> list[dict]:
    """The drawn agreement statistics, each with its interval and sample."""
    free = benchmark["threshold_free_agreement"].get(PAPER_TIER, {}).get("prad__tangtv")
    pair = benchmark["pairwise_agreement"]["by_tangtv_tier"].get(PAPER_TIER, {})
    pair = pair.get("prad__tangtv", {})
    rows = []

    def add(key, label, chance, value, ci, n_bins, n_shots, kind):
        finite = value is not None and np.isfinite(value)
        rows.append(
            {
                "key": key,
                "label": label,
                "kind": kind,
                "chance": chance,
                "value": value if finite else None,
                "ci95": clean_ci(ci) if finite else None,
                "ci_note": None
                if finite and clean_ci(ci)
                else (n_a(n_shots) if finite else None),
                "n_bins": n_bins,
                "n_shots": n_shots,
                "drawn": bool(finite),
                "undefined_reason": None
                if finite
                else "undefined: one indicator cast a single class",
            }
        )

    if free:
        a = free["auroc_pooled"]
        add(
            "auroc_pooled",
            "AUROC pooled",
            0.5,
            a["value"],
            a["ci95"],
            a["n_bins"],
            a["n_shots"],
            "threshold_free",
        )
        w = free["within_shot"]["auroc"]
        add(
            "auroc_within_shot",
            "AUROC per shot",
            0.5,
            w["mean"],
            w["mean_ci95"],
            None,
            w["n_shots"],
            "threshold_free",
        )
        r = free["spearman_pooled"]
        add(
            "spearman_pooled",
            r"$\rho$ pooled",
            0.0,
            r["value"],
            r["ci95"],
            r["n_bins"],
            r["n_shots"],
            "threshold_free",
        )
        w = free["within_shot"]["spearman"]
        add(
            "spearman_within_shot",
            r"$\rho$ per shot",
            0.0,
            w["mean"],
            w["mean_ci95"],
            None,
            w["n_shots"],
            "threshold_free",
        )
    if pair:
        for key, label, field in (
            ("kappa_binary", r"$\kappa$ binary", "binary_kappa"),
            ("kappa_3class", r"$\kappa$ 3-class", "kappa"),
        ):
            k = pair[field]
            add(
                key,
                label,
                0.0,
                k["value"],
                k["ci95"],
                pair["both_vote_bins"],
                pair["both_vote_shots"],
                "kappa",
            )
    return rows


#: The Te bands behind `share_in_band` (the record's `bands_ev`).
BAND_TEXT = {"attached": ">= 10 eV", "detached": "<= 5 eV"}


def share_in_band(entry: dict) -> dict:
    """Share of the attached bins at or above 10 eV and of the detached bins at or
    below 5 eV, with the bin counts and the shots behind them."""
    out = {}
    for state in ("attached", "detached"):
        block = entry.get(state, {})
        n = int(block.get("n_bins", 0))
        out[state] = {
            "band_ev": BAND_TEXT[state],
            "n_bins": n,
            "n_in_band": int(block.get("n_in_band", 0)) if n else 0,
            "share": block.get("share_in_band") if n else None,
            "n_shots": int(block.get("n_shots", 0)),
        }
    return out


def te_rows(te: dict) -> list[dict]:
    """AUROC of -Te for detached against attached, by what casts the vote.

    `population` says which vote the row scores: `vote_before_gate` is the TangTV
    vote with no L-mode gate (the check against Te, filled markers); `label_tier`
    and `indicator_vote` are the exported tiers and the single-indicator votes (open
    markers). The shares in band of a label tier are post-selection: the gate was
    added after the Te check flagged the L-mode bins.
    """
    auroc = "auroc_neg_te_detached_vs_attached"
    regime = te["by_regime"]["tangtv_vote_before_gate"]
    before = te["by_tier"]
    gate_note = (
        "TangTV vote before the L-mode gate (every valid upper-shelf vote), the "
        "check against Te; the share in band is not post-selection"
    )
    post_note = (
        "exported tier, after the L-mode gate; the share in band is post-selection "
        "(the gate was added after the Te check flagged the L-mode bins)"
    )
    vote_note = "single-indicator vote on the assessed bins of the label set"
    sources = (
        (
            "labelled",
            "labelled tiers",
            "label_tier",
            before["certain_or_tangtv_only"],
            post_note,
        ),
        ("certain", "agreement tier", "label_tier", before["certain"], post_note),
        (
            "tangtv_only",
            "TangTV-only tier",
            "label_tier",
            before["tangtv_only"],
            post_note,
        ),
        (
            "tangtv_vote_before_gate",
            "TangTV, no gate",
            "vote_before_gate",
            before["tangtv_vote_before_gate"],
            gate_note,
        ),
        (
            "tangtv_vote_before_gate_without_201081",
            "  w/o shot 201081",
            "vote_before_gate",
            before["tangtv_vote_before_gate_without_201081"],
            gate_note + "; the anchor shot 201081 (it sets the DZ cutoffs) left out",
        ),
        (
            "tangtv_vote_before_gate_regime_H",
            "  known H-mode",
            "vote_before_gate",
            regime["H"],
            gate_note + "; known H-mode bins",
        ),
        (
            "tangtv_vote_before_gate_regime_L",
            "  known L-mode",
            "vote_before_gate",
            regime["L"],
            gate_note + "; bins of a known L-mode phase, where the gate acts",
        ),
        (
            "tangtv_vote_before_gate_regime_probable_L",
            "  probable L-mode",
            "vote_before_gate",
            regime["probable_L"],
            gate_note + "; regime unknown, ELM-free, median P_in under 2 MW (an "
            "a-priori proxy), where the gate also acts",
        ),
        (
            "tangtv_vote_before_gate_regime_probable_H",
            "  probable H-mode",
            "vote_before_gate",
            regime["probable_H"],
            gate_note + "; regime unknown with an ELM share (report only, the gate "
            "does not act on it)",
        ),
        (
            "tangtv_vote_before_gate_regime_unknown",
            "  no regime",
            "vote_before_gate",
            regime["unknown"],
            gate_note + "; regime unknown and neither proxy applies",
        ),
        (
            "afrac_vote",
            "Afrac vote",
            "indicator_vote",
            te["indicator_votes"]["afrac"],
            vote_note,
        ),
        (
            "f_div_vote",
            r"$f_{\mathrm{div}}$ votes (unused)",
            "indicator_vote",
            te["indicator_votes"]["prad"],
            vote_note + "; f_div is not a vote of the label",
        ),
    )
    rows = []
    for key, label, population, entry, note in sources:
        pooled = entry[auroc]["pooled"]
        n_shots = pooled["n_shots"]
        finite = pooled["value"] is not None and np.isfinite(pooled["value"])
        ci = clean_ci(pooled["ci95"]) if finite else None
        shares = share_in_band(entry)
        reason = None
        if not finite:
            reason = (
                "no attached bins"
                if not shares["attached"]["n_bins"]
                else "no detached bins"
                if not shares["detached"]["n_bins"]
                else "one class in every resample"
            )
        rows.append(
            {
                "key": f"te_{key}",
                "label": label.strip(),
                "population": population,
                "chance": 0.5,
                "value": pooled["value"] if finite else None,
                "ci95": ci,
                "ci_note": None if ci or not finite else n_a(n_shots),
                "not_estimable_reason": reason,
                "n_bins": pooled["n_bins"],
                "n_not_attached": pooled["n_not_attached"],
                "n_shots": n_shots,
                "share_in_band": shares,
                "note": note,
                "within_shot": entry[auroc]["within_shot"],
                "drawn": bool(finite and ci),
            }
        )
    return rows


def fdiv_rows(record: dict) -> list[dict]:
    """f_div AUROC against TangTV and Te: pooled over shots and mean per shot."""
    rows = []
    for family in ("relative", "absolute"):
        for reference, name in (("tangtv", "TangTV"), ("te", "Te")):
            entry = record["summary"][family][reference]
            pooled, within = entry["pooled"], entry["within_shot"]

            def finite(x):
                return x is not None and np.isfinite(x)

            rows.append(
                {
                    "key": f"fdiv_{family}_vs_{reference}",
                    "label": f"{family} vs {name}",
                    "chance": 0.5,
                    "pooled": {
                        "value": pooled["value"] if finite(pooled["value"]) else None,
                        "ci95": clean_ci(pooled["ci95"])
                        if finite(pooled["value"])
                        else None,
                        "ci_note": None
                        if clean_ci(pooled["ci95"]) or not finite(pooled["value"])
                        else n_a(pooled["n_shots"]),
                        "n_bins": pooled["n_bins"],
                        "n_shots": pooled["n_shots"],
                    },
                    "within_shot": {
                        "value": within["mean"] if finite(within["mean"]) else None,
                        "ci95": clean_ci(within["mean_ci95"])
                        if finite(within["mean"])
                        else None,
                        "ci_note": None
                        if clean_ci(within["mean_ci95"]) or not finite(within["mean"])
                        else n_a(within["n_shots"]),
                        "n_shots": within["n_shots"],
                        "shots_above_chance": within["shots_above_chance"],
                    },
                }
            )
    return rows


def build() -> dict:
    source = root() / "labels_bins.csv.gz"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    labels = pd.read_csv(source)
    benchmark = json.loads(AGREEMENT_JSON.read_text())
    if benchmark.get("sources", {}).get("labels_sha256") != digest:
        raise ValueError("Pairwise agreement is stale; rerun detach_benchmark.py")
    coverage = coverage_group(labels)
    current = json.loads(CURRENT_JSON.read_text())
    if current["labels_sha256"] != digest:
        raise ValueError("Current-state record is stale; rerun detach_current_state.py")
    never = current["population"]["assessed_bins_that_can_never_carry_a_state"]
    coverage["assessed_that_can_never_carry_a_state"] = {
        "reason": never["reason"],
        "bins": never["bins"],
        "shots": never["shots"],
        "source": str(CURRENT_JSON),
    }
    coverage["by_tangtv_tier"] = {
        str(tier): coverage_group(group)
        for tier, group in labels.groupby("tangtv_tier", dropna=False)
    }
    coverage["by_split"] = {
        str(split): coverage_group(group)
        for split, group in labels.groupby("split", dropna=False)
    }
    pair = benchmark["pairwise_agreement"]["by_tangtv_tier"].get(PAPER_TIER, {})
    pair = pair.get("prad__tangtv", {})
    rows = agreement_rows(benchmark)
    te = json.loads(TE_JSON.read_text())
    fdiv = json.loads(FDIV_JSON.read_text())
    result = {
        "schema": "detachment_coverage_agreement",
        "task": "detachment",
        "scope": "exploratory label-set coverage and indicator agreement; "
        "no independent benchmark",
        "independent_benchmark": {
            "status": "unavailable",
            "reason": "No independent physical state reference with evaluable "
            "coverage; the divertor Thomson check is in "
            "docs/labeler/results/detachment_te_check.json.",
        },
        "label_set": "geometry-gated TangTV state, checked against divertor Thomson Te",
        "interpretation": "TangTV + Afrac agreement (tier certain; agreement of two "
        "indicators, not a confidence level) is the TangTV vote with an agreeing "
        "Afrac vote; TangTV only labels (silver) are the TangTV vote with Afrac "
        "abstaining or invalid; bins where TangTV votes detached on a known L-mode "
        "phase or in a probable-L window (regime unknown, ELM-free, median P_in under "
        "2 MW) are uncertain (tier tangtv_only_lmode). None is independent truth. "
        "Agreement does not establish accuracy; the Te rows of the vote before the "
        "gate are the check, and the shares in band of the label tiers are "
        "post-selection (the gate was added after the Te check flagged the L-mode "
        "bins).",
        "bin_ms": core.BIN_MS,
        "figure": {"width_in": 3.25, "height_in": FIG_HEIGHT_IN, "min_font_pt": 7},
        "population": "all exported assessed eligible-shot bins, every split",
        "paper_tier": PAPER_TIER,
        "agreement_population": {
            "both_valid_bins": pair.get("both_valid_bins"),
            "both_vote_bins": pair.get("both_vote_bins"),
            "both_vote_shots": pair.get("both_vote_shots"),
            "both_vote_shot_ids": pair.get("both_vote_shot_ids"),
            "vote_table_prad_rows_tangtv_columns": pair.get("counts"),
        },
        "bootstrap": {
            "unit": "shot",
            "replicates": benchmark["replicates"],
            "ci": 0.95,
            "seed": benchmark["bootstrap"]["seed"],
            "method": "percentile",
            "undefined_draws": "excluded; valid_replicates in the benchmark record",
            "source": "docs/labeler/results/detachment_benchmark.json",
        },
        "coverage": coverage,
        "te_agreement": te_rows(te),
        "second_vote_effect": te["by_tier"]["second_vote_effect"],
        "fdiv_corroborator": fdiv_rows(fdiv),
        "fdiv_source": str(FDIV_JSON),
        "te_source": str(TE_JSON),
        "agreement": rows,
        "coverage_table": indicator_coverage(labels),
        "definitions": {
            "assessed": "at least two valid measurements on an eligible shot, or a "
            "TangTV vote",
            "certain": "TangTV + Afrac agreement: assessed bin where the TangTV vote "
            "and an agreeing valid Afrac vote state attached or detached (not a "
            "confidence level)",
            "tangtv_only_lmode": "uncertain: TangTV votes detached (0.5 <= DZ < "
            "1.2) on a known L-mode phase or in a probable-L window; the DZ cutoffs "
            "come from an H-mode shot. Only the detached vote is gated: L-mode "
            "inner-SOL leakage biases DZ upward, so a low DZ stays trustworthy",
            "probable_L": "regime unknown from every source, ELM coverage known, no "
            "ELM share in the window and median P_in under 2 MW (the Martin 2008 "
            "L-H threshold scaling gives about 1.7 to 2.4 MW for typical DIII-D "
            "parameters); an a-priori proxy, not a measured regime",
            "vote_before_gate": "the TangTV vote with no L-mode gate (filled "
            "markers of the Te panel); not the same population as the "
            "`no_second_voter` label composition, which has the gate applied",
            "tangtv_only": "assessed bin where TangTV votes and Afrac abstains or "
            "is invalid (silver)",
            "conflict": "TangTV and Afrac vote against each other; uncertain",
            "candidate_marfe": "uncertain bin with a sustained TangTV high front; "
            "no MARFE state is exported",
            "uncertain": "every other assessed bin; the tier says why",
            "te_auroc": "P(Te of a detached-vote bin < Te of an attached-vote bin); "
            "0.5 is chance",
            "seconds": "bin count times bin_ms / 1000; disjoint bins, not spans",
            "shots": "unique contributing shots; state shot counts may overlap",
            "measurement": "finite scalar before indicator validity gates",
            "valid_measurement": "passes the indicator validity gates",
            "vote": "valid attached/detached/marfe vote, excluding abstentions",
            "certain_label": "vote bin that also has a certain exported state",
            "auroc": "P(f_div of a TangTV not-attached bin > f_div of a TangTV "
            "attached bin); 0.5 is chance",
            "spearman": "rank correlation of f_div with DZ, both larger when more "
            "detached; 0 is chance",
            "within_shot": "mean over shots of the per-shot statistic, shots with "
            "enough bins of both classes",
            "kappa_binary": "Cohen's kappa of attached versus not attached "
            "(MARFE merged with detached); undefined, not drawn, when either "
            "indicator cast one class",
        },
        "thresholds": {
            "prad_relative_cutoffs": list(thresholds.prad_relative_cutoffs()),
            "prad_absolute_cutoffs_global": list(thresholds.prad_cutoffs()),
            "prad_cutoffs_note": "f_div is not a vote of the label. Its relative "
            "votes (f_div over the shot baseline) and its absolute votes (global "
            "cutoffs, one pair for every shot, anchored on 201081) are "
            "sensitivities: docs/labeler/results/detachment_prad_anchor.json",
            "afrac_window_psin": thresholds.AFRAC_PSI_WINDOW,
            "afrac_votes": [
                thresholds.AFRAC_DETACHED_MAX,
                thresholds.AFRAC_ATTACHED_MIN,
            ],
            "prad_averaging_ms": thresholds.PRAD_AVERAGING_MS,
        },
        "sources": {
            "labels": str(source),
            "bins": str(root() / "bins"),
            "labels_sha256": digest,
            "benchmark": str(AGREEMENT_JSON),
            "te_check": str(TE_JSON),
            "fdiv_check": str(FDIV_JSON),
        },
    }
    PANEL_JSON.write_text(dumps(result, indent=1))
    return result


def interval_plot(ax, rows, colour_for, *, tick_for=None) -> None:
    """Dot and 95% interval per row, the chance level as a grey tick. `tick_for`
    writes the row's tick label (default: label and shot count)."""
    for y, row in enumerate(rows):
        ax.plot([row["chance"]], [y], marker="|", ms=7, color=".6", mew=0.8, zorder=1)
        colour, face = colour_for(row)
        if row["value"] is None:
            # no AUROC (a class is empty): say why instead of leaving the row blank
            ax.text(
                0.22,
                y,
                f"n/a: {row['not_estimable_reason']}",
                fontsize=7,
                va="center",
                zorder=4,
                bbox={"fc": "white", "ec": "none", "pad": 0.6},
            )
            continue
        ax.plot(row["value"], y, "o", ms=4, color=colour, mfc=face, zorder=3)
        if row["ci95"] and all(np.isfinite(row["ci95"])):
            ax.hlines(y, *row["ci95"], color=colour, lw=1, zorder=2)
    default = lambda row: (
        f"{row['label']} ({row['n_shots']})" if row.get("n_shots") else row["label"]
    )
    ax.set_yticks(range(len(rows)), [(tick_for or default)(row) for row in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.spines[["top", "right"]].set_visible(False)


def te_tick(row: dict) -> str:
    """Row label, shot count and, for the vote before the gate, the share of its
    detached votes at or below 5 eV (the L-mode row is the one that matters)."""
    text = f"{row['label']} ({row['n_shots']})"
    detached = row["share_in_band"]["detached"]
    if "_regime_" in row["key"] and detached["n_bins"]:
        text += f"\ndetached \u22645 eV: {detached['n_in_band']}/{detached['n_bins']}"
    return text


def draw(data: dict, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (coverage_ax, te_ax, agreement_ax) = plt.subplots(
        3,
        1,
        figsize=(3.25, FIG_HEIGHT_IN),
        gridspec_kw={"height_ratios": [1.2, 4.0, 1.5]},
    )
    coverage = data["coverage"]
    groups = (
        (
            "Labelled",
            "labelled_certain_or_tangtv_only",
            (
                "attached_certain",
                "attached_silver",
                "detached_certain",
                "detached_silver",
            ),
        ),
        ("Agreement", "certain", ("attached_certain", "detached_certain")),
        ("Assessed", "assessed", tuple(CLASS_COLOUR)),
    )
    drawn_classes = []
    ticks = []
    for y, (_, key, classes) in enumerate(groups):
        left = 0
        for name in classes:
            count = coverage["by_class"][name]["bins"]
            if count:
                coverage_ax.barh(
                    y, count, left=left, color=CLASS_COLOUR[name], height=0.62
                )
                if name not in drawn_classes:
                    drawn_classes.append(name)
            left += count
        entry = coverage[key]
        ticks.append(f"{groups[y][0]}\n{entry['bins']:,} bins, {entry['shots']} shots")
    coverage_ax.set_yticks(range(len(groups)), ticks)
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, coverage["assessed"]["bins"] * 1.02)
    coverage_ax.set_xticks([0, 500, 1000, 1500])
    coverage_ax.spines[["top", "right"]].set_visible(False)
    never = coverage["assessed_that_can_never_carry_a_state"]
    coverage_ax.set_xlabel(
        f"50 ms bins; {never['bins']:,} assessed bins\ncan never carry a state"
    )
    fig.legend(
        [Patch(facecolor=CLASS_COLOUR[n]) for n in drawn_classes],
        [CLASS_LABEL[n] for n in drawn_classes],
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7,
        handlelength=1,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    te_rows_drawn = [
        row
        for row in data["te_agreement"]
        if row["drawn"]
        or (
            row["population"] == "vote_before_gate"
            and row["value"] is None
            and row["share_in_band"]["attached"]["n_bins"]
            + row["share_in_band"]["detached"]["n_bins"]
        )
    ]

    def te_colour(row):
        filled = row["population"] == "vote_before_gate"
        return ("#222222", "#222222" if filled else "white")

    interval_plot(te_ax, te_rows_drawn, te_colour, tick_for=te_tick)
    te_ax.legend(
        [
            Line2D([], [], marker="o", ms=4, color="#222222", mfc="#222222", ls="-"),
            Line2D([], [], marker="o", ms=4, color="#222222", mfc="white", ls="-"),
        ],
        ["TangTV vote before the gate", "label tiers, indicator votes"],
        loc="lower right",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        labelspacing=0.25,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    te_ax.set_xlim(0.2, 1.02)
    te_ax.set_xlabel(r"AUROC of $-T_e$" "\n(detached vs attached;\nshots in brackets)")
    rows = data["fdiv_corroborator"]
    for y, row in enumerate(rows):
        agreement_ax.plot(
            [row["chance"]], [y], marker="|", ms=7, color=".6", mew=0.8, zorder=1
        )
        for part, dy, marker, face in (
            ("pooled", -0.16, "D", FDIV_COLOUR),
            ("within_shot", 0.16, "^", "white"),
        ):
            entry = row[part]
            if entry["value"] is None:
                continue
            agreement_ax.plot(
                entry["value"],
                y + dy,
                marker,
                ms=3.6,
                color=FDIV_COLOUR,
                mfc=face,
                zorder=3,
            )
            if entry["ci95"] and all(np.isfinite(entry["ci95"])):
                agreement_ax.hlines(y + dy, *entry["ci95"], color=FDIV_COLOUR, lw=1)
    agreement_ax.set_yticks(
        range(len(rows)),
        [
            f"{row['label']}\n({row['pooled']['n_shots']} / "
            f"{row['within_shot']['n_shots']} shots)"
            for row in rows
        ],
    )
    agreement_ax.set_ylim(len(rows) - 0.4, -0.6)
    agreement_ax.spines[["top", "right"]].set_visible(False)
    lows = [
        r[p]["ci95"][0] for r in rows for p in ("pooled", "within_shot") if r[p]["ci95"]
    ]
    highs = [
        r[p]["ci95"][1] for r in rows for p in ("pooled", "within_shot") if r[p]["ci95"]
    ]
    agreement_ax.set_xlim(
        min([0.5, *lows]) - 0.03 if lows else 0.0,
        min(max([*highs, 0.6]) + 0.02, 1.02),
    )
    agreement_ax.set_xlabel(
        r"AUROC of $f_{\mathrm{div}}$" "\n(shots: pooled / per shot)"
    )
    agreement_ax.legend(
        [
            Line2D([], [], marker="D", ms=3.6, color=FDIV_COLOUR, mfc=FDIV_COLOUR),
            Line2D([], [], marker="^", ms=3.6, color=FDIV_COLOUR, mfc="white"),
        ],
        ["pooled", "per shot"],
        loc="lower right",
        bbox_to_anchor=(1.02, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.5, right=0.97, top=0.89, bottom=0.11, hspace=0.85)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_detachment_figure2.pdf", metadata={"CreationDate": None})
    fig.savefig(out / "fig_detachment_figure2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=root() / "figure")
    args = parser.parse_args()
    draw(build(), args.out_dir)
    print("wrote", PANEL_JSON, "and", args.out_dir / "fig_detachment_figure2.pdf")


if __name__ == "__main__":
    main()
