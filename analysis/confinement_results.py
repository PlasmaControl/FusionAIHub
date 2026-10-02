"""Generate paper figures from frozen confinement evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labeler.confinement import run_directory


def make_figures(run_dir: Path):
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    folder = run_dir / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    full = evaluation["models"]["full"]
    bootstrap = evaluation["paired_bootstrap"]["full_cohort"]
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.4), constrained_layout=True)
    matrix = np.array(full["metrics"]["confusion_matrix_LH"])
    image = axes[0].imshow(
        matrix / matrix.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap="Blues"
    )
    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                f"{matrix[i, j]:,}\n{matrix[i, j] / matrix[i].sum():.1%}",
                ha="center",
                va="center",
                color="white" if i == j else "black",
            )
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["L", "H-family"],
        yticks=[0, 1],
        yticklabels=["L", "H-family"],
        xlabel="Predicted",
        ylabel="Annotated",
        title="Primary detector: 28 shots, 1,403 bins",
    )
    fig.colorbar(image, ax=axes[0], fraction=0.045, label="Recall within each row")
    models = ["full", "all_H"]
    labels = ["Primary", "Always H"]
    colors = ["#1565a5", "#7d858a"]
    metrics = ["macro_f1", "balanced_accuracy"]
    x = np.arange(2)
    for n, (name, label, color) in enumerate(zip(models, labels, colors, strict=True)):
        measured = full["metrics"] if name == "full" else full["all_H_baseline"]
        values = np.array([measured[m] for m in metrics])
        intervals = np.array([bootstrap["metric_95ci"][name][m] for m in metrics])
        axes[1].errorbar(
            x + (n - 0.5) * 0.16,
            values,
            yerr=np.vstack((values - intervals[:, 0], intervals[:, 1] - values)),
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[1].set(
        xticks=x,
        xticklabels=["Macro-F1", "Balanced accuracy"],
        ylim=(0.35, 1.03),
        ylabel="Score",
        title="95% intervals from whole-shot resampling",
    )
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(loc="center", bbox_to_anchor=(0.65, 0.48))
    fig.suptitle(
        "Exploratory test of imported labels; L occurs on five shots", fontsize=11
    )
    for suffix in ("pdf", "png"):
        fig.savefig(folder / f"detector_results.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=run_directory())
    make_figures(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
