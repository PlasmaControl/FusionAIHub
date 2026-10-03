#!/usr/bin/env python
"""Figure 2 detachment coverage panel from the small legacy/Tokamak-SI JSON."""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    repo = Path(__file__).resolve().parents[2]
    data = json.loads(
        (repo / "docs/labeler/results/detachment_figure2.json").read_text()
    )
    out = Path(os.environ["LABELER_ROOT"]) / "round4/detach/figure"
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    labels = ["Afrac (Eldon)", "Prad,div (Eldon)", "TangTV (Chen)", "Redundant label"]
    coverage = [r["coverage"] for r in data["legacy"]] + [data["Tokamak-SI"]["certain"]]
    fig, ax = plt.subplots(figsize=(6.75, 2.4))
    for i, (label, c) in enumerate(zip(labels, coverage, strict=True)):
        ax.barh(i, c["bins"], color="#E69F00" if i < 3 else "#0072B2", height=0.62)
        ax.text(
            c["bins"] + max(v["bins"] for v in coverage) * 0.02,
            i,
            f"{c['bins']:,} bins / {c['shots']} shots"
            if c["bins"]
            else "published setting not reproduced",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Covered bins (50 ms)")
    ax.set_xlim(0, max(v["bins"] for v in coverage) * 1.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.97, bottom=0.24)
    fig.savefig(out / "fig_detachment_figure2.pdf")
    fig.savefig(out / "fig_detachment_figure2.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
