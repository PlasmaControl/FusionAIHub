"""Stage-1 evaluation — Phase 2 plots.

Consumes the CSV.gz tables produced by ``eval_e2e_stage1_phase1.py`` and
produces the plots specified in ``docs/eval_stage1_plan.md`` §5.

This first cut delivers the **aggregate-quality scatter** only (§2-Q1):
one scatter per (split, modality), one dot per shot, y = model MAE vs
x = copy-baseline MAE. Below-diagonal = model beats persistence. The
title carries the **percent of shots below diagonal** — the single
most-quotable summary number for "did the model learn anything for
this modality?".

Per-shot 2×2 summary plots (which require re-inference for GT-vs-pred
panels) and stitched-window plots are deferred to Phase 2.1 / Phase 3
respectively.

Run::

    pixi run python scripts/training/eval_e2e_stage1_phase2_plots.py \\
        --output_dir eval_runs/stage1_phase1_e2e_stage1_best_4609988

Plots are written to ``<output_dir>/plots/<split>/<modality>/_aggregate_scatter.png``.

Follows the §5 quality bar:
- Honest axes (no rainbow colormaps; physical units in labels).
- Self-documenting titles (modality, split, n_shots, %-below-diagonal).
- Equal aspect ratio so the y=x diagonal reads 45° to the eye.
- Dot color encodes a *second* shot-level statistic
  (frac_windows_below_diag) so dense clusters resolve into "shots
  where the model wins consistently" vs "wins on average, loses on
  key windows".
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("eval_stage1_phase2_plots")


# ─────────────────────────────────────────────────────────────────────
# Style conventions (§5 quality bar — applied globally)
# ─────────────────────────────────────────────────────────────────────

# Perceptually uniform colormap for the dot-color statistic. Never
# rainbow (it distorts ordering perception).
_DOT_CMAP = "viridis"
# Color for the y=x diagonal reference line.
_DIAGONAL_COLOR = "0.4"
# Color for the linear-fit reference (set to off-axis so it doesn't
# confuse with the diagonal).
_FIT_COLOR = "tab:red"


# ─────────────────────────────────────────────────────────────────────
# Aggregate-quality scatter
# ─────────────────────────────────────────────────────────────────────


def plot_aggregate_scatter(
    per_shot_df: pd.DataFrame,
    split: str,
    modality: str,
    kind: str,
    out_path: Path,
) -> None:
    """Per-shot scatter of model MAE vs copy-baseline MAE.

    See §5 of ``docs/eval_stage1_plan.md``.

    Parameters
    ----------
    per_shot_df : pd.DataFrame
        Slice of ``per_shot_metrics.csv.gz`` filtered to one
        ``(split, modality)``.
    split, modality, kind : str
        Identifiers for the title and output path.
    out_path : pathlib.Path
        Where to write the PNG (parent dir will be created).
    """
    # Drop shots whose mae_ratio_mean is non-finite — they have no
    # copy denominator (e.g., modality absent everywhere). Reporting
    # them as plotted points is misleading; reporting them as a count
    # in the title is honest.
    n_shots_total = len(per_shot_df)
    df = per_shot_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mae_mean", "copy_mae_mean", "mae_ratio_mean"]
    )
    n_shots_kept = len(df)
    n_dropped = n_shots_total - n_shots_kept

    if n_shots_kept == 0:
        # Defensive: empty modality (e.g., absent everywhere in this
        # split). Write a placeholder so the missing plot is visible
        # rather than silently absent in the output dir.
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(
            0.5, 0.5,
            f"{modality} ({split}): no plottable shots\n"
            f"(all {n_shots_total} have undefined mae_ratio)",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return

    x = df["copy_mae_mean"].to_numpy(dtype=np.float64)
    y = df["mae_mean"].to_numpy(dtype=np.float64)
    c = df["frac_windows_below_diag"].to_numpy(dtype=np.float64)

    # Percent below the diagonal = the headline number.
    pct_below = float((y < x).mean()) * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Diagonal first (back-most), so dots draw on top.
    lim_lo = float(min(x.min(), y.min())) * 0.95
    lim_hi = float(max(x.max(), y.max())) * 1.05
    if lim_lo == lim_hi:
        # Degenerate scale (all identical) — pad arbitrarily.
        lim_lo -= 0.05
        lim_hi += 0.05
    ax.plot(
        [lim_lo, lim_hi], [lim_lo, lim_hi],
        color=_DIAGONAL_COLOR, linewidth=1.2, linestyle="--",
        zorder=1, label="y = x (copy baseline)",
    )

    # Scatter with frac-windows-below-diag as the color.
    sc = ax.scatter(
        x, y,
        c=c,
        cmap=_DOT_CMAP,
        vmin=0.0, vmax=1.0,
        s=40, alpha=0.85,
        edgecolor="white", linewidth=0.3,
        zorder=3,
    )

    # Colorbar with explicit unit (a fraction, clearly named).
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("frac_windows_below_diag (per shot)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Title encodes everything a future-you needs to interpret the plot.
    title_lines = [
        f"{modality} ({kind}) — split: {split}",
        f"{n_shots_kept} shots plotted "
        + (f"(+{n_dropped} dropped: undefined ratio)" if n_dropped else "")
        + f" · {pct_below:.1f}% below diagonal",
    ]
    ax.set_title("\n".join(title_lines), fontsize=10)

    ax.set_xlabel(
        "copy-baseline MAE per shot (= MAE between input(t) and target(t+50ms))",
        fontsize=9,
    )
    ax.set_ylabel("model MAE per shot", fontsize=9)

    # Equal aspect so the diagonal is visually 45°.
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)

    # Bottom-left corner legend so it doesn't overlap the colorbar.
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    # Light grid for reading off values.
    ax.grid(True, alpha=0.3, linewidth=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output_dir", type=Path, required=True,
        help="Existing eval output directory (produced by "
             "eval_e2e_stage1_phase1.py). Must contain "
             "per_shot_metrics.csv.gz.",
    )
    p.add_argument(
        "--plots_subdir", type=str, default="plots",
        help="Subdirectory of --output_dir to write plots into.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    ps_path = args.output_dir / "per_shot_metrics.csv.gz"
    if not ps_path.exists():
        raise SystemExit(
            f"per_shot_metrics.csv.gz not found at {ps_path}. "
            f"Run Phase 1 first (eval_e2e_stage1_phase1.py)."
        )

    per_shot = pd.read_csv(ps_path, compression="gzip")
    logger.info(f"Loaded {len(per_shot):,} per-shot rows from {ps_path.name}")

    # Phase 1 now emits one row per (shot, modality, k). For aggregate
    # scatter plots we show the final-step rollout (k = max present);
    # for Stage 1 (K=1) this is a no-op.
    if "k" in per_shot.columns and per_shot["k"].nunique() > 1:
        k_render = int(per_shot["k"].max())
        per_shot = per_shot[per_shot["k"] == k_render].copy()
        logger.info(f"Rendering scatter at k={k_render} (final rollout step)")

    plots_root = args.output_dir / args.plots_subdir

    # Aggregate scatter: one per (split, modality).
    n_plots = 0
    for (split, modality, kind), group in per_shot.groupby(
        ["split", "modality", "kind"], sort=False
    ):
        out_path = plots_root / split / modality / "_aggregate_scatter.png"
        plot_aggregate_scatter(
            per_shot_df=group,
            split=split, modality=modality, kind=kind,
            out_path=out_path,
        )
        logger.info(
            f"Wrote {out_path.relative_to(args.output_dir)} "
            f"({len(group)} shots)"
        )
        n_plots += 1

    logger.info(f"Phase 2 (aggregate scatter): {n_plots} plots written to "
                f"{plots_root.relative_to(args.output_dir)}/")


if __name__ == "__main__":
    main()
