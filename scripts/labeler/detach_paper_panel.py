#!/usr/bin/env python
"""Write the detachment Figure 2 panel: per-state coverage and indicator agreement.

There is no independent detachment benchmark. The agreement shown is between two
indicators, Prad,div/P_in and the TangTV C-III front, on the upper-shelf bins where
both cast a vote: threshold-free (AUROC of the f_div value against the TangTV
attached versus not-attached vote; Spearman rho of f_div against DZ), pooled over
shots and within shots, each with a 95% shot-bootstrap interval, and the binary
kappa of the cast votes where it is defined. A kappa on a table in which either
indicator used a single class is undefined and is not drawn.

Every number is read from `docs/labeler/results/detachment_benchmark.json` (one
bootstrap, written by `detach_benchmark.py`) and the exported bins. The panel
source is `docs/labeler/figure2_detach.json`; there is no second copy.
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
LF_NAMES = ("afrac", "prad", "tangtv")
STATE_COLOUR = {1: "#0072B2", 2: "#E69F00", 3: "#CC79A7", 4: "#999999"}
STATE_LABEL = {1: "attached", 2: "detached", 3: "MARFE", 4: "uncertain"}
SETTINGS = {
    "afrac": "Jsat ratio at the peak SOL-side target probe, uncalibrated",
    "prad": "Prad,div,L over P_in, 250 ms inter-ELM means, absolute cutoffs "
    "anchored on 201081",
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


def coverage_group(frame: pd.DataFrame) -> dict:
    assessed = frame.assessed.to_numpy(bool)
    certain = assessed & frame.state_lm.isin(core.VOTE_STATES).to_numpy()
    return {
        "assessed": population(frame, assessed),
        "certain": population(frame, certain),
        "by_state": {
            core.STATE_NAMES[state]: {
                "assessed": population(frame, assessed & frame.state_lm.eq(state)),
                "certain": population(frame, certain & frame.state_lm.eq(state)),
            }
            for state in (1, 2, 3, 4)
        },
    }


def indicator_coverage(labels: pd.DataFrame) -> list[dict]:
    """Measurement coverage on all extracted grids, before eligibility selection."""
    bins = load_all(root() / "bins")
    exported = (
        labels.set_index(["shot", "start_ms"])
        .state_lm.reindex(pd.MultiIndex.from_frame(bins[["shot", "start_ms"]]))
        .to_numpy()
    )
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
                "certain_label": population(
                    bins, cast & np.isin(exported, core.VOTE_STATES)
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
        "interpretation": "Certain labels require compatible indicator votes; they "
        "are not independent truth. Agreement does not establish accuracy.",
        "bin_ms": core.BIN_MS,
        "figure": {"width_in": 3.25, "height_in": 3.5, "min_font_pt": 7},
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
        "agreement": rows,
        "coverage_table": indicator_coverage(labels),
        "definitions": {
            "assessed": "at least two valid measurements on an eligible shot",
            "certain": "assessed bin with a certain primary state "
            "(attached, detached or MARFE)",
            "uncertain": "every other assessed bin; the tier says why",
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
            "prad_cutoffs_mw_over_p_in": "per shot from the 201081 anchor: see "
            "docs/labeler/results/detachment_prad_anchor.json",
            "prad_averaging_ms": thresholds.PRAD_AVERAGING_MS,
        },
        "sources": {
            "labels": str(source),
            "bins": str(root() / "bins"),
            "labels_sha256": digest,
            "benchmark": str(AGREEMENT_JSON),
        },
    }
    PANEL_JSON.write_text(dumps(result, indent=1))
    return result


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
    fig, (coverage_ax, agreement_ax) = plt.subplots(
        2, 1, figsize=(3.25, 3.5), gridspec_kw={"height_ratios": [0.8, 1.5]}
    )
    coverage = data["coverage"]
    drawn_states = set()
    for y, kind in enumerate(("assessed", "certain")):
        left = 0
        for state in (1, 2, 3, 4):
            count = coverage["by_state"][core.STATE_NAMES[state]][kind]["bins"]
            if count:
                coverage_ax.barh(
                    y, count, left=left, color=STATE_COLOUR[state], height=0.6
                )
                drawn_states.add(state)
            left += count
    coverage_ax.set_yticks([0, 1], ["Assessed", "Certain"])
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, max(coverage["assessed"]["bins"], 1))
    for y, kind in enumerate(("assessed", "certain")):
        entry = coverage[kind]
        text = f"{entry['bins']:,} bins, {entry['shots']} shots"
        if kind == "assessed":
            coverage_ax.text(
                coverage["assessed"]["bins"] * 0.985,
                y,
                text,
                ha="right",
                va="center",
                fontsize=7,
                color="black",
            )
        else:
            coverage_ax.text(
                entry["bins"] + coverage["assessed"]["bins"] * 0.03,
                y,
                text,
                ha="left",
                va="center",
                fontsize=7,
            )
    coverage_ax.set_xlabel("50 ms bins")
    fig.legend(
        [Patch(facecolor=STATE_COLOUR[s]) for s in sorted(drawn_states)],
        [STATE_LABEL[s] for s in sorted(drawn_states)],
        ncol=len(drawn_states),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7,
        handlelength=1,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    rows = [row for row in data["agreement"] if row["drawn"]]
    lows = [row["ci95"][0] for row in rows if row["ci95"]]
    highs = [row["ci95"][1] for row in rows if row["ci95"]]
    for y, row in enumerate(rows):
        agreement_ax.plot(
            [row["chance"]], [y], marker="|", ms=7, color=".6", mew=0.8, zorder=1
        )
        colour = "#222222" if row["kind"] == "threshold_free" else "#777777"
        face = colour if row["kind"] == "threshold_free" else "white"
        agreement_ax.plot(row["value"], y, "o", ms=4, color=colour, mfc=face, zorder=3)
        if row["ci95"] and all(np.isfinite(row["ci95"])):
            agreement_ax.hlines(y, *row["ci95"], color=colour, lw=1, zorder=2)
    agreement_ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    agreement_ax.set_ylim(len(rows) - 0.4, -0.6)
    low = min([0.0, *lows]) - 0.05 if lows else -0.05
    high = max([*highs, 1.0]) if highs else 1.0
    agreement_ax.set_xlim(low, min(high + 0.02, 1.02))
    agreement_ax.set_xlabel("Prad,div vs TangTV, 95% shot CI")
    for ax in (coverage_ax, agreement_ax):
        ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.9, bottom=0.12, hspace=0.75)
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
