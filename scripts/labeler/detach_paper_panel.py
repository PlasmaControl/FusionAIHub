#!/usr/bin/env python
"""Write exploratory detachment coverage and indicator-agreement sources.

There is no independent detachment benchmark. No indicator or learned model is
scored against the compatible label that its measurements help define. The
agreement population instead includes every exported bin where Prad and TangTV
both cast valid votes, stratified by TangTV geometry tier, without selecting
certain labels. Only upper-shelf agreement is drawn in the paper panel.
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
BOOTSTRAPS = 1000
SEED = 7
STATE_COLOUR = {1: "#0072B2", 2: "#E69F00", 3: "#CC79A7", 4: "#999999"}
SETTINGS = {
    "afrac": "Uncalibrated local Jsat ratio; attached-current fit unavailable",
    "prad": "Local Prad,div/P_in thresholds with heating and radiation smoothing",
    "tangtv": "C-III front with upper-shelf geometry, quality and MARFE gates",
}


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


def kappa(table: np.ndarray) -> float:
    total = table.sum()
    if not total:
        return float("nan")
    expected = float(table.sum(axis=0) @ table.sum(axis=1)) / total**2
    if 1 - expected <= np.finfo(float).eps:
        return float("nan")
    return float((np.trace(table) / total - expected) / (1 - expected))


def agreement_metrics(table: np.ndarray) -> dict:
    """Prad rows, TangTV columns; binary merges MARFE with detached on both axes."""
    binary = np.array(
        [[table[0, 0], table[0, 1:].sum()], [table[1:, 0].sum(), table[1:, 1:].sum()]]
    )
    return {
        "kappa_3class": kappa(table),
        "kappa_binary": kappa(binary),
        "agreement_3class": float(np.trace(table) / table.sum())
        if table.sum()
        else float("nan"),
        "agreement_binary": float(np.trace(binary) / binary.sum())
        if binary.sum()
        else float("nan"),
    }


def paired_agreement(frame: pd.DataFrame) -> dict:
    """Bootstrap whole shots, including all paired bins on each sampled shot."""
    both_valid = frame.prad_valid.to_numpy(bool) & frame.tangtv_valid.to_numpy(bool)
    keep = (
        both_valid
        & frame.prad_vote.isin(core.VOTE_STATES).to_numpy()
        & frame.tangtv_vote.isin(core.VOTE_STATES).to_numpy()
    )
    paired = frame.loc[keep]
    shots = sorted(int(s) for s in paired.shot.unique())
    tables = np.zeros((len(shots), 3, 3), dtype=int)
    for i, shot in enumerate(shots):
        rows = paired[paired.shot == shot]
        np.add.at(
            tables[i],
            (rows.prad_vote.to_numpy(int) - 1, rows.tangtv_vote.to_numpy(int) - 1),
            1,
        )
    table = tables.sum(axis=0)
    point = agreement_metrics(table)
    rng = np.random.default_rng(SEED)
    draws = [
        agreement_metrics(tables[rng.integers(0, len(shots), len(shots))].sum(axis=0))
        for _ in range(BOOTSTRAPS if shots else 0)
    ]
    result = {}
    for key, value in point.items():
        samples = np.array([draw[key] for draw in draws], float)
        samples = samples[np.isfinite(samples)]
        limits = np.percentile(samples, [2.5, 97.5]) if len(samples) else [np.nan] * 2
        result[key] = {
            "value": value,
            "ci95": [float(x) for x in limits],
            "valid_replicates": len(samples),
            "replicates": BOOTSTRAPS,
        }
    return {
        "both_valid": population(frame, both_valid),
        "both_vote": population(frame, keep),
        "state_order": ["attached", "detached", "marfe"],
        "table_axes": {"rows": "prad_vote", "columns": "tangtv_vote"},
        "confusion_counts": table.tolist(),
        "metrics": result,
        "conflicts": {
            "tangtv_attached_prad_detached": population(
                frame, keep & frame.tangtv_vote.eq(1) & frame.prad_vote.eq(2)
            ),
            "tangtv_attached_all_prad_votes": population(
                frame, keep & frame.tangtv_vote.eq(1)
            ),
            "tangtv_attached_prad_valid": population(
                frame, both_valid & frame.tangtv_vote.eq(1)
            ),
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
    agreement = {
        "population": "every exported eligible-shot bin where Prad and TangTV "
        "both cast valid votes, without conditioning on the exported state",
        "binary_mapping": {
            "attached": "attached",
            "detached": "detached",
            "marfe": "detached",
        },
        "by_tangtv_tier": {
            str(tier): paired_agreement(group)
            for tier, group in labels.groupby("tangtv_tier", dropna=False)
        },
        "paper_tier": "upper_shelf",
        "bootstrap_source": str(AGREEMENT_JSON),
    }
    # One published bootstrap record: independently reconstruct its population
    # and count table before using its intervals, so stale inputs cannot mix.
    metric_keys = {
        "kappa_3class": "kappa",
        "kappa_binary": "binary_kappa",
        "agreement_3class": "agreement",
        "agreement_binary": "binary_agreement",
    }
    for tier, entry in agreement["by_tangtv_tier"].items():
        published = benchmark["pairwise_agreement"]["by_tangtv_tier"][tier][
            "prad__tangtv"
        ]
        if (
            published["counts"] != entry["confusion_counts"]
            or published["both_vote_shot_ids"] != entry["both_vote"]["shot_ids"]
            or published["both_valid_bins"] != entry["both_valid"]["bins"]
        ):
            raise ValueError(f"Pairwise population differs for TangTV tier {tier}")
        for key, published_key in metric_keys.items():
            value = published[published_key]["value"]
            own = entry["metrics"][key]["value"]
            if value is not None and not np.isclose(value, own):
                raise ValueError(f"Pairwise metric differs: {tier}, {key}")
            entry["metrics"][key] = published[published_key]
    upper = agreement["by_tangtv_tier"].get("upper_shelf", paired_agreement(labels[:0]))
    rows = []
    for name in ("assessed", "certain"):
        entry = coverage[name]
        rows.append(
            {
                "panel": "detachment",
                "kind": "coverage",
                "series": name,
                "group": "Exploratory coverage",
                "metric": "bins",
                "value": entry["bins"],
                "n_shots": entry["shots"],
                "seconds": entry["seconds"],
                "ci_lo": None,
                "ci_hi": None,
                "source": "docs/labeler/figure2_detach.json",
                "key": f"coverage.{name}.bins",
            }
        )
    for metric in ("kappa_3class", "kappa_binary"):
        entry = upper["metrics"][metric]
        rows.append(
            {
                "panel": "detachment",
                "kind": "agreement",
                "series": metric,
                "group": "Upper-shelf indicator agreement",
                "metric": "kappa",
                "value": entry["value"],
                "ci_lo": entry["ci95"][0],
                "ci_hi": entry["ci95"][1],
                "n_bins": upper["both_vote"]["bins"],
                "n_shots": upper["both_vote"]["shots"],
                "source": "docs/labeler/figure2_detach.json",
                "key": f"indicator_agreement.by_tangtv_tier.upper_shelf."
                f"metrics.{metric}.value",
            }
        )
    result = {
        "schema": "detachment_coverage_agreement",
        "task": "detachment",
        "scope": "exploratory label-set coverage and indicator agreement; "
        "no independent benchmark",
        "independent_benchmark": {
            "status": "unavailable",
            "reason": "No independent physical state reference with evaluable coverage.",
        },
        "interpretation": "Certain labels require compatible indicator votes; "
        "they are not independent truth. Agreement does not establish accuracy. "
        "MARFE candidates come from one shot and remain within cue uncertainty.",
        "bin_ms": core.BIN_MS,
        "figure": {"width_in": 3.25, "height_in": 3.4, "min_font_pt": 7},
        "population": "all exported assessed eligible-shot bins, every split",
        "bootstrap": {
            "unit": "shot",
            "replicates": BOOTSTRAPS,
            "ci": 0.95,
            "seed": benchmark["bootstrap"]["seed"],
            "method": "percentile",
            "undefined_draws": "excluded; valid_replicates reported",
            "source": str(AGREEMENT_JSON),
        },
        "coverage": coverage,
        "indicator_agreement": agreement,
        "coverage_table": indicator_coverage(labels),
        "definitions": {
            "assessed": "at least two valid measurements on an eligible shot",
            "certain": "assessed compatible primary state attached/detached/marfe",
            "seconds": "bin count times bin_ms / 1000; disjoint bins, not spans",
            "shots": "unique contributing shots; state shot counts may overlap",
            "measurement": "finite scalar before indicator validity gates",
            "valid_measurement": "passes the indicator validity gates",
            "vote": "valid attached/detached/marfe vote, excluding abstentions",
            "certain_label": "vote bin also has a certain exported primary state",
        },
        "thresholds": {
            "prad_attached_max": thresholds.PRAD_ATTACHED_MAX,
            "prad_detached_min": thresholds.PRAD_DETACHED_MIN,
            "prad_averaging_ms": thresholds.PRAD_AVERAGING_MS,
        },
        "sources": {
            "labels": str(source),
            "bins": str(root() / "bins"),
            "labels_sha256": digest,
            "pairwise_agreement": str(AGREEMENT_JSON),
        },
        "rows": rows,
    }
    PANEL_JSON.write_text(dumps(result, indent=1))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "detachment_figure2.json").write_text(dumps(result, indent=1))
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
        2, 1, figsize=(3.25, 3.4), gridspec_kw={"height_ratios": [1.0, 1.05]}
    )
    coverage = data["coverage"]
    for y, kind in enumerate(("assessed", "certain")):
        left = 0
        for state in (1, 2, 3, 4):
            count = coverage["by_state"][core.STATE_NAMES[state]][kind]["bins"]
            coverage_ax.barh(
                y, count, left=left, color=STATE_COLOUR[state], height=0.52
            )
            left += count
        entry = coverage[kind]
        coverage_ax.text(
            left + max(coverage["assessed"]["bins"] * 0.02, 1),
            y,
            f"{left:,} bins\n{entry['shots']} shots; {entry['seconds']:.2f} s",
            va="center",
            fontsize=7,
        )
    coverage_ax.set_yticks([0, 1], ["Assessed", "Certain"])
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, max(coverage["assessed"]["bins"], 1) * 1.75)
    coverage_ax.set_xlabel("50 ms bins")
    coverage_ax.legend(
        [Patch(facecolor=STATE_COLOUR[s]) for s in (1, 2, 3, 4)],
        ["attached", "detached", "MARFE cand.", "uncertain"],
        ncol=2,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.63),
        fontsize=7,
    )
    upper = data["indicator_agreement"]["by_tangtv_tier"].get("upper_shelf")
    interval_limits = [0.0, 1.0]
    for y, key in enumerate(("kappa_3class", "kappa_binary")):
        entry = upper["metrics"][key] if upper else {}
        value, ci = entry.get("value"), entry.get("ci95", [None, None])
        if value is not None and np.isfinite(value):
            agreement_ax.plot(value, y, "o", ms=4, color="#0072B2")
            if all(v is not None and np.isfinite(v) for v in ci):
                interval_limits.extend(ci)
                agreement_ax.hlines(y, ci[0], ci[1], color="#0072B2", lw=1)
                agreement_ax.text(
                    0.99,
                    y + 0.25,
                    f"{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                    ha="right",
                    fontsize=7,
                )
        else:
            agreement_ax.text(0.5, y, "undefined", ha="center", fontsize=7)
    agreement_ax.set_yticks([0, 1], ["3 class", "Binary"])
    agreement_ax.set_ylim(1.55, -0.45)
    xmin = min(-0.1, min(interval_limits) - 0.05)
    agreement_ax.set_xlim(xmin, max(1.02, max(interval_limits) + 0.02))
    ticks = [-0.5, 0, 0.5, 1] if xmin < -0.3 else [-0.2, 0, 0.5, 1]
    agreement_ax.set_xticks([value for value in ticks if value >= xmin])
    agreement_ax.axvline(0, color=".65", lw=0.6, ls="--")
    agreement_ax.set_xlabel(r"Prad–TangTV $\kappa$ (95% shot CI)")
    if upper:
        count = upper["both_vote"]
        prad_counts = np.asarray(upper["confusion_counts"]).sum(axis=1)
        constant_vote = np.flatnonzero(prad_counts)
        note = (
            "\nPrad votes: " + core.STATE_NAMES[int(constant_vote[0]) + 1] + " only"
            if len(constant_vote) == 1
            else ""
        )
        agreement_ax.set_title(
            f"Upper shelf: {count['bins']} bins / {count['shots']} shots" + note,
            fontsize=7.5,
        )
    for ax in (coverage_ax, agreement_ax):
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, 0.985, "Exploratory detachment labels", va="top", ha="center", fontsize=8
    )
    fig.text(
        0.5,
        0.035,
        "No independent benchmark\nBinary: MARFE merged with detached",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.77, bottom=0.22, hspace=1.1)
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
