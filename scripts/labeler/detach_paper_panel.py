#!/usr/bin/env python
"""Write the detachment Figure 2 panel: coverage by tier, agreement with Te and f_div.

There is no independent detachment benchmark. The label set is the geometry-gated
TangTV state, validated by the divertor Thomson Te (a temperature none of the three
indicators uses). The panel shows

* the coverage of the exported label by tier (certain: TangTV plus an agreeing Afrac
  vote; TangTV only, silver; the `candidate_marfe` tier; every other uncertain bin),
  bins and shots;
* the agreement of the TangTV state with Te: the AUROC of -Te for a detached against
  an attached vote, pooled over shots with a 95% shot-bootstrap interval, for the
  TangTV vote alone, the certain tier, the TangTV-only tier and the two indicators'
  votes (f_div is drawn for reference, it is not a vote of the label). Whether the
  second vote improves the Te agreement is the `second_vote_effect` of the Te
  record, copied into the panel source;
* f_div as a within-shot corroborator: its AUROC against the TangTV vote and against
  Te, pooled over shots (filled) and the mean over shots (open), for the per-shot
  relative value and the absolute ratio, with the shot counts.

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detach_json import dumps
from detach_label import load_all
from matplotlib.patches import Patch

from labeler.events.detachment import core, thresholds

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "labeler" / "results"
PANEL_JSON = REPO / "docs" / "labeler" / "figure2_detach.json"
AGREEMENT_JSON = RESULTS / "detachment_benchmark.json"
TE_JSON = RESULTS / "detachment_te_check.json"
FDIV_JSON = RESULTS / "detachment_fdiv_check.json"
FIG_HEIGHT_IN = 5.4
LF_NAMES = ("afrac", "prad", "tangtv")
#: Okabe-Ito colours of the exported classes; silver (TangTV only) is the tint.
CLASS_COLOUR = {
    "attached_certain": "#0072B2",
    "attached_silver": "#8EC1E3",
    "detached_certain": "#E69F00",
    "detached_silver": "#F4D58D",
    "candidate_marfe": "#CC79A7",
    "uncertain": "#999999",
}
CLASS_LABEL = {
    "attached_certain": "attached, certain",
    "attached_silver": "attached, TangTV only",
    "detached_certain": "detached, certain",
    "detached_silver": "detached, TangTV only",
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
                "ci95": [float(x) for x in ci] if finite and ci else None,
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


def te_rows(te: dict) -> list[dict]:
    """AUROC of -Te for detached against attached, by what casts the vote."""
    auroc = "auroc_neg_te_detached_vs_attached"
    sources = (
        ("tangtv_alone", "TangTV alone", te["by_tier"]["tangtv_vote_alone"]),
        ("certain", "certain tier", te["by_tier"]["certain"]),
        ("tangtv_only", "TangTV only", te["by_tier"]["tangtv_only"]),
        ("afrac_vote", "Afrac vote", te["indicator_votes"]["afrac"]),
        (
            "f_div_vote",
            r"$f_{\mathrm{div}}$ votes (unused)",
            te["indicator_votes"]["prad"],
        ),
    )
    rows = []
    for key, label, entry in sources:
        pooled = entry[auroc]["pooled"]
        finite = pooled["value"] is not None and np.isfinite(pooled["value"])
        rows.append(
            {
                "key": f"te_{key}",
                "label": label,
                "chance": 0.5,
                "value": pooled["value"] if finite else None,
                "ci95": [float(x) for x in pooled["ci95"]] if finite else None,
                "n_bins": pooled["n_bins"],
                "n_not_attached": pooled["n_not_attached"],
                "n_shots": pooled["n_shots"],
                "within_shot": entry[auroc]["within_shot"],
                "drawn": bool(finite),
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
                        "ci95": [float(x) for x in pooled["ci95"]]
                        if finite(pooled["value"])
                        else None,
                        "n_bins": pooled["n_bins"],
                        "n_shots": pooled["n_shots"],
                    },
                    "within_shot": {
                        "value": within["mean"] if finite(within["mean"]) else None,
                        "ci95": [float(x) for x in within["mean_ci95"]]
                        if finite(within["mean"])
                        else None,
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
        "label_set": "geometry-gated TangTV state, validated by divertor Thomson Te",
        "interpretation": "Certain labels are the TangTV vote with an agreeing "
        "Afrac vote; TangTV only labels (silver) are the TangTV vote with Afrac "
        "abstaining or invalid. Neither is independent truth. Agreement does not "
        "establish accuracy.",
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
            "certain": "assessed bin where the TangTV vote and an agreeing valid "
            "Afrac vote state attached or detached",
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


def interval_plot(ax, rows, colour_for, *, chance_label=None) -> None:
    """Dot and 95% interval per row, the chance level as a grey tick."""
    for y, row in enumerate(rows):
        ax.plot([row["chance"]], [y], marker="|", ms=7, color=".6", mew=0.8, zorder=1)
        colour, face = colour_for(row)
        ax.plot(row["value"], y, "o", ms=4, color=colour, mfc=face, zorder=3)
        if row["ci95"] and all(np.isfinite(row["ci95"])):
            ax.hlines(y, *row["ci95"], color=colour, lw=1, zorder=2)
    ax.set_yticks(
        range(len(rows)),
        [
            f"{row['label']} ({row['n_shots']})" if row.get("n_shots") else row["label"]
            for row in rows
        ],
    )
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.spines[["top", "right"]].set_visible(False)


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
        gridspec_kw={"height_ratios": [0.8, 1.0, 1.7]},
    )
    coverage = data["coverage"]
    groups = (
        ("Assessed", "assessed", tuple(CLASS_COLOUR)),
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
        ("Certain", "certain", ("attached_certain", "detached_certain")),
    )
    drawn_classes = []
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
        coverage_ax.text(
            max(left, 1) + coverage["assessed"]["bins"] * 0.02,
            y,
            f"{entry['bins']:,} bins, {entry['shots']} shots",
            ha="left",
            va="center",
            fontsize=7,
        )
    coverage_ax.set_yticks(range(len(groups)), [g[0] for g in groups])
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, coverage["assessed"]["bins"] * 2.2)
    coverage_ax.set_xticks([0, 1000])
    coverage_ax.spines[["top", "right"]].set_visible(False)
    coverage_ax.set_xlabel("50 ms bins")
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
    )
    te_rows_drawn = [row for row in data["te_agreement"] if row["drawn"]]

    def te_colour(row):
        return ("#222222", "#222222") if "alone" in row["key"] else ("#222222", "white")

    interval_plot(te_ax, te_rows_drawn, te_colour)
    te_ax.set_xlim(0.2, 1.02)
    te_ax.set_xlabel(r"AUROC of $-T_e$, detached vs attached" "\n(shots in brackets)")
    rows = data["fdiv_corroborator"]
    for y, row in enumerate(rows):
        agreement_ax.plot(
            [row["chance"]], [y], marker="|", ms=7, color=".6", mew=0.8, zorder=1
        )
        for part, dy, marker, face in (
            ("pooled", -0.16, "o", "#222222"),
            ("within_shot", 0.16, "s", "white"),
        ):
            entry = row[part]
            if entry["value"] is None:
                continue
            agreement_ax.plot(
                entry["value"],
                y + dy,
                marker,
                ms=3.6,
                color="#222222",
                mfc=face,
                zorder=3,
            )
            if entry["ci95"] and all(np.isfinite(entry["ci95"])):
                agreement_ax.hlines(y + dy, *entry["ci95"], color="#222222", lw=1)
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
        r"AUROC of $f_{\mathrm{div}}$"
        "\nfilled: pooled, open: per shot"
        "\n(shots: pooled / per shot)"
    )
    fig.subplots_adjust(left=0.4, right=0.97, top=0.85, bottom=0.15, hspace=1.0)
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
