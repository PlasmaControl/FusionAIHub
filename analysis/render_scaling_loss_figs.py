"""Paper-quality loss-curve figures for the IGNITE backbone scaling study.

Re-renders from the DURABLE loss records — no training needed:
  * per-cell metrics json ``e2e_overfit_200729.json`` -> ``loss_curve`` (per-step; written by
    tests/ignite/test_e2e_overfit_realshot.py for every run from 2026-08-07 on)
  * fallback: the run logs' ``[e2e] backbone step N masked_ce=X`` lines (50-step samples;
    the only record for the pre-json cells — logs are append-only, so this always works)
  * (production runs: ``loss_history.jsonl`` has the same role; point --cell at it if needed)

Outputs (PDF vector + PNG preview, overwritten in place in the scaling dir):
  fig_scaling_curves.{pdf,png}   1x3 small multiples (one panel per width; color = depth)
  fig_scaling_summary.{pdf,png}  final masked CE vs relative compute (log-log)

Design: categorical hues from the validated palette (CVD-checked), color follows DEPTH
consistently across panels, one y-scale everywhere, recessive grid, ink-colored labels.

Usage:  python analysis/render_scaling_loss_figs.py [--dir eval_runs/ignite_e2e_scaling_200729]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

WIDTHS = (256, 512, 1024)
DEPTHS = (4, 8, 16)
# validated categorical palette (dataviz reference, light surface) — color follows DEPTH
DEPTH_COLOR = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a"}
INK, INK2 = "#1a1a19", "#5f5e56"

# legacy cells whose curves live only in logs (runs before the json loss_curve existed)
LOG_FALLBACK = {
    "d256x4": "e2e_scale_d256x4_5189391.out",
    "d256x8": "e2e_scale_d256x8_5189393.out",
    "d512x4": "e2e_scale_d512x4_5189392.out",
    "d512x8": "e2e_scale_d512x8_5189644.out",
    "d256x16": "e2e_scale_d256x16_5190810.out",
}
_PAT = re.compile(r"backbone step\s+(\d+)\s+masked_ce=([\d.]+)")


def _bin50(steps: np.ndarray, vals: np.ndarray):
    """Mean-bin a per-step curve to 50-step resolution (comparable with log-sampled cells)."""
    if len(steps) < 2 or (steps[1] - steps[0]) >= 50:
        return steps, vals
    edges = np.arange(0, steps.max() + 50, 50)
    idx = np.digitize(steps, edges)
    out_s, out_v = [], []
    for b in np.unique(idx):
        m = idx == b
        out_s.append(steps[m].mean())
        out_v.append(vals[m].mean())
    return np.asarray(out_s), np.asarray(out_v)


def load_cell(scaling_dir: Path, width: int, depth: int):
    """-> (steps, ce, final_ce) or None. Prefers the json per-step curve; falls back to logs."""
    cell = f"d{width}x{depth}"
    j = scaling_dir / cell / "e2e_overfit_200729.json"
    if j.exists():
        m = json.loads(j.read_text())
        if m.get("loss_curve"):
            v = np.asarray(m["loss_curve"], dtype=float)
            s = np.arange(1, len(v) + 1, dtype=float)
            s, v = _bin50(s, v)
            return s, v, float(m["maskgit_ce_end"])
    lf = LOG_FALLBACK.get(cell)
    if lf and (REPO / "logs" / lf).exists():
        txt = (REPO / "logs" / lf).read_text(errors="ignore")
        pairs = [(int(a), float(b)) for a, b in _PAT.findall(txt)]
        if len(pairs) >= 3:
            s = np.asarray([p[0] for p in pairs], dtype=float)
            v = np.asarray([p[1] for p in pairs], dtype=float)
            final = float(v[-3:].mean())
            jm = scaling_dir / cell / "e2e_overfit_200729.json"
            if jm.exists():
                final = float(json.loads(jm.read_text()).get("maskgit_ce_end", final))
            return s, v, final
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_runs/ignite_e2e_scaling_200729")
    args = ap.parse_args()
    sdir = (REPO / args.dir) if not Path(args.dir).is_absolute() else Path(args.dir)

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7, "axes.edgecolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed TrueType (journal requirement)
    })

    cells = {}
    for w in WIDTHS:
        for d in DEPTHS:
            got = load_cell(sdir, w, d)
            if got:
                cells[(w, d)] = got
    if not cells:
        raise SystemExit(f"no loss records found under {sdir}")

    # ---- Fig 1: training curves, small multiples by width, color = depth ------------------- #
    ylo = min(v.min() for _s, v, _f in cells.values()) * 0.8
    yhi = max(v.max() for _s, v, _f in cells.values()) * 1.15
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(7.0, 2.4), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        for d in DEPTHS:
            if (w, d) not in cells:
                continue
            s, v, f = cells[(w, d)]
            ax.plot(s, v, color=DEPTH_COLOR[d], lw=1.4, solid_capstyle="round",
                    label=f"depth {d}")
            ax.annotate(f"{f:.2f}", (s[-1], v[-1]), xytext=(3, 0),
                        textcoords="offset points", fontsize=6.5, color=INK, va="center")
        ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax.set_title(f"$d_\\mathrm{{model}}$ = {w}")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel("masked-token CE")
    handles = [plt.Line2D([], [], color=DEPTH_COLOR[d], lw=1.4, label=f"depth {d}")
               for d in DEPTHS]
    axes[-1].legend(handles=handles, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(sdir / f"fig_scaling_curves.{ext}", dpi=300)
    plt.close(fig)

    # ---- Fig 2: final CE vs relative compute (log-log) ------------------------------------- #
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for d in DEPTHS:
        xs, ys, labs = [], [], []
        for w in WIDTHS:
            if (w, d) not in cells:
                continue
            xs.append((w / 256) ** 2 * (d / 4))
            ys.append(cells[(w, d)][2])
            labs.append(f"d{w}")
        if not xs:
            continue
        ax.plot(xs, ys, "o-", color=DEPTH_COLOR[d], lw=1.2, ms=4,
                mec="white", mew=0.6, label=f"depth {d}")
        for x, y, t in zip(xs, ys, labs):
            ax.annotate(t, (x, y), xytext=(4, 3), textcoords="offset points",
                        fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative compute  $(d/256)^2 \\times (\\mathrm{depth}/4)$")
    ax.set_ylabel("final masked-token CE")
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(sdir / f"fig_scaling_summary.{ext}", dpi=300)
    plt.close(fig)
    print(f"rendered {len(cells)} cells -> {sdir}/fig_scaling_curves.[pdf|png], "
          f"fig_scaling_summary.[pdf|png]")


if __name__ == "__main__":
    main()
