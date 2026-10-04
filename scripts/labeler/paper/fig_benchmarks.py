r"""fig_benchmarks: F1 per label set, each model named "<task>-<model>", in two
label settings, "legacy" and "Tokamak-SI".

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig_benchmarks.py \
        [--out dev/label_paper/figures/fig_benchmarks.pdf] [--png PATH] [--table PATH]

Every score drawn is read from an evaluation JSON (`SOURCES`), with the 95 %
shot-bootstrap interval stored beside it where there is one; none is typed in
here. F1 only. Per set, "legacy" is the setting the model's own paper (or the
older annotation) used and "Tokamak-SI" is our labels:

- AE: `ae-ours` (the SELDnet model), `ae-rcn` and `ae-lstm` (chord
  spectrogram, each chord scored) on the 19 held-out shots, 10 ms frames;
  legacy = against the Heidbrink annotation (`results.annotated`), Tokamak-SI =
  against the dense labels (`results.reviewed`).
- Confinement: `confine-cnn`; legacy = the earlier paper's macro F1,
  the mean of its four per-class F1 (`paper` in the first retrain's
  evaluation), one bar, no interval; Tokamak-SI = our reimplementation under
  the paper's selection and split, the ablation's protocol row
  `full_cum_abcdrgef` (142 distinct test shots in 200 shot-tests from five
  random by-shot splits whose test sets overlap, windows pooled; macro F1,
  AUROC and AUPRC with shot-bootstrap intervals), the one confine-cnn score of
  the paper. The first retrain's own score is not drawn.
- ELMs: `elm-elmo`; legacy = D. Smith's windows at the published setting
  (`smith.published_setting`, its F1 from the stored counts); Tokamak-SI = the
  reviewed spans in 50 ms bins (`review.paper`).
- Tearing modes, sawtooth and RWM onsets have no score yet: two empty
  hatched slots each (legacy, Tokamak-SI).

`--metric auc` draws the same three scored sets for AUROC (top row) and AUPRC
(bottom row) into `fig_benchmarks_auc.pdf`: a legacy score only where its paper
reported one (neither confine-cnn's nor elm-elmo's did), else a hatched "not
reported" slot; elm-elmo's AUROC and AUPRC come from a sweep of its threshold.

A hatched empty slot is a model not yet scored. `--table` writes every number
drawn, with its JSON key, as CSV; the same rows are printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from labeler.paper import FONT_PT, PAGE_IN, style

REPO = Path(__file__).resolve().parents[3]
OUTPUTS = REPO / "outputs" / "labeler"
SOURCES = {
    "ae": OUTPUTS / "ae" / "baselines" / "evaluation.json",
    # the earlier paper's per-class scores (`paper`), pasted from its table
    "confinement": OUTPUTS / "confinement" / "bes" / "evaluation.json",
    # the protocol row's record (confinement_bes_ablation.py summarize)
    "confinement_si": OUTPUTS
    / "confinement"
    / "bes"
    / "ablation_rows"
    / "full_cum_abcdrgef.json",
    "elm": OUTPUTS / "elm" / "elmo" / "evaluation.json",
}
DEFAULT_OUT = REPO / "dev" / "label_paper" / "figures" / "fig_benchmarks.pdf"
DEFAULT_AUC_OUT = DEFAULT_OUT.with_name("fig_benchmarks_auc.pdf")

# Okabe-Ito, one colour per label setting (checked with the dataviz palette
# validator, light surface).
LEGACY = "#E69F00"
SI = "#0072B2"
INK = "#333333"
PENDING_EDGE = "#999999"
VALUE_PT = FONT_PT - 1.5
BAR = 0.62
YMAX = 1.13
YTICKS = np.arange(0.0, 1.01, 0.25)

AE_MODELS = (  # (key in the evaluation, "<task>-<model>")
    ("seldnet", "ae-ours"),
    ("rcn_spec_each", "ae-rcn"),
    ("lstm_spec_each", "ae-lstm"),
)
AE_SETTINGS = (  # (setting, truth block, what it is scored against, colour)
    ("legacy", "annotated", "Heidbrink annotation", LEGACY),
    ("Tokamak-SI", "reviewed", "dense labels", SI),
)
CLASSES = ("L", "H", "QH", "WP")
METRICS = {"f1": "F1", "auroc": "AUROC", "auprc": "AUPRC"}
PENDING = ("Tearing Modes", "Sawtooth", "Resistive Wall Modes")


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class Rows:
    """Every number drawn, with where it came from."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, panel, series, group, metric, value, ci, source, key) -> None:
        lo, hi = (None, None) if ci is None else ci
        self.rows.append(
            {
                "panel": panel,
                "series": series,
                "group": group,
                "metric": metric,
                "value": value,
                "ci_lo": lo,
                "ci_hi": hi,
                "source": str(Path(source).relative_to(REPO)),
                "key": key,
            }
        )

    def write(self, path: Path | None) -> None:
        fields = list(self.rows[0])
        out = csv.DictWriter(sys.stdout, fields, lineterminator="\n")
        out.writeheader()
        out.writerows(self.rows)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                writer.writerows(self.rows)


def axes_style(ax, ylabel: str | None) -> None:
    ax.set_ylim(0, YMAX)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([f"{t:g}" for t in YTICKS])
    ax.grid(axis="y", color="#e4e4e4", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", length=2, pad=1.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
        ax.spines[side].set_linewidth(0.6)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)


def scored_bar(ax, x, value, ci, colour, digits=3) -> None:
    """A bar with its interval, if it has one, and its value written above."""
    ax.bar(x, value, BAR, color=colour, edgecolor="none", zorder=2)
    top = value
    if ci is not None:
        lo, hi = ci
        ax.errorbar(
            x,
            value,
            yerr=[[value - lo], [hi - value]],
            fmt="none",
            ecolor=INK,
            elinewidth=0.6,
            capsize=1.5,
            capthick=0.6,
            zorder=3,
        )
        top = hi
    ax.text(
        x,
        top + 0.015,
        f"{value:.{digits}f}",
        ha="center",
        va="bottom",
        fontsize=VALUE_PT,
        color=INK,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
    )


def pending_slot(ax, x, width, edge, height=1.0, text=None) -> None:
    """An empty hatched slot: no model of this role scored yet."""
    ax.add_patch(
        Rectangle(
            (x - width / 2, 0),
            width,
            height,
            facecolor="none",
            edgecolor=edge,
            hatch="////",
            lw=0.6,
            ls=(0, (2, 1.5)),
            zorder=1,
        )
    )
    if text:
        ax.text(
            x,
            height / 2,
            text,
            rotation=90,
            ha="center",
            va="center",
            fontsize=VALUE_PT,
            color=INK,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )


def group_labels(ax, centres, labels, y) -> None:
    """The setting of each group of bars, below the bar names."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for c, text in zip(centres, labels):
        ax.text(
            c,
            y,
            text,
            transform=trans,
            ha="center",
            va="top",
            fontsize=FONT_PT - 1,
            color=INK,
        )


def finish_groups(ax, ticks, names, centres, labels, y=-0.12) -> None:
    ax.set_xticks(ticks, names)
    ax.tick_params(axis="x", labelsize=FONT_PT - 1.5)
    ax.set_xlim(min(ticks) - 0.7, max(ticks) + 0.7)
    group_labels(ax, centres, labels, y)
    axes_style(ax, None)


def draw_ae(ax, ae: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """ae-ours, ae-rcn and ae-lstm against each truth: legacy, then Tokamak-SI."""
    ticks, names, centres, labels = [], [], [], []
    step = len(AE_MODELS) + 0.7
    for g, (setting, truth, truth_name, colour) in enumerate(AE_SETTINGS):
        block = ae["results"][truth]
        xs = [g * step + i for i in range(len(AE_MODELS))]
        for x, (key, name) in zip(xs, AE_MODELS):
            entry = block[key]
            ci = entry["ci95"][metric]
            scored_bar(ax, x, entry[metric], ci, colour)
            ticks.append(x)
            names.append(name)
            rows.add(
                panel,
                name,
                setting,
                METRICS[metric],
                entry[metric],
                ci,
                SOURCES["ae"],
                f"results.{truth}.{key}.{metric}",
            )
        centres.append(float(np.mean(xs)))
        labels.append(setting)
    finish_groups(ax, ticks, names, centres, labels)
    ax.set_ylabel(METRICS[metric], labelpad=2)


def confinement_si(record: dict) -> dict:
    """The protocol row's record (`ablation_rows/full_cum_abcdrgef.json`) as the
    macro scores and intervals `draw_confinement` reads: macro F1 from the row's
    own windows, macro AUROC and AUPRC from its `ranking`, each with the 95 % shot
    bootstrap interval stored beside it."""
    own, rank = record["own_population"], record["ranking"]
    return {
        "macro": {
            "f1": own["macro_f1"],
            "auroc": rank["auroc"]["macro"],
            "auprc": rank["auprc"]["macro"],
        },
        "ci95": {
            "macro_f1": own["ci95"]["macro_f1"],
            "macro_auroc": rank["ci95"]["auroc"],
            "macro_auprc": rank["ci95"]["auprc"],
        },
        "keys": {
            "f1": "own_population.macro_f1",
            "auroc": "ranking.auroc.macro",
            "auprc": "ranking.auprc.macro",
        },
    }


def draw_confinement(
    ax, conf: dict, si: dict, rows: Rows, panel: str, metric: str = "f1"
) -> None:
    """confine-cnn's macro score: the earlier paper's (legacy, from `conf`; F1
    only, it published no AUROC or AUPRC) and our reimplementation under its
    selection and split (Tokamak-SI, from `si`, see `confinement_si`)."""
    name = "macro " + METRICS[metric]
    if metric == "f1":
        published = conf["paper"]
        paper_macro = float(np.mean([published[k]["f1"] for k in CLASSES]))
        scored_bar(ax, 0, paper_macro, None, LEGACY, digits=2)
        rows.add(
            panel,
            "confine-cnn",
            "legacy",
            name,
            paper_macro,
            None,
            SOURCES["confinement"],
            "mean of paper.{L,H,QH,WP}.f1",
        )
    else:
        pending_slot(ax, 0, 0.8, PENDING_EDGE, text="not reported")
    value = si["macro"][metric]
    ci = si["ci95"][f"macro_{metric}"]
    scored_bar(ax, 1.7, value, ci, SI)
    rows.add(
        panel,
        "confine-cnn",
        "Tokamak-SI",
        name,
        value,
        ci,
        SOURCES["confinement_si"],
        si["keys"][metric],
    )
    finish_groups(
        ax,
        [0, 1.7],
        ["confine-cnn"] * 2,
        [0, 1.7],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_ylabel(name, labelpad=2)


def draw_elm(ax, elm: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """elm-elmo on D. Smith's windows (legacy; F1 only, there is no threshold
    sweep there) and on the reviewed spans (Tokamak-SI; AUROC and AUPRC come
    from a sweep of its detection threshold)."""
    name = METRICS[metric]
    if metric == "f1":
        smith = elm["smith"]["published_setting"]
        scored_bar(ax, 0, smith["f1"], None, LEGACY)
        rows.add(
            panel,
            "elm-elmo",
            "legacy",
            name,
            smith["f1"],
            None,
            SOURCES["elm"],
            "smith.published_setting.f1",
        )
        review = elm["review"]["paper"]
        key = "review.paper.f1"
    else:
        pending_slot(ax, 0, 0.8, PENDING_EDGE, text="not reported")
        review = elm["review"]["eta_sweep"]
        key = f"review.eta_sweep.{metric}"
    ci = review["ci95"][metric]
    scored_bar(ax, 1.7, review[metric], ci, SI)
    rows.add(
        panel, "elm-elmo", "Tokamak-SI", name, review[metric], ci, SOURCES["elm"], key
    )
    finish_groups(
        ax,
        [0, 1.7],
        ["elm-elmo"] * 2,
        [0, 1.7],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_ylabel(name, labelpad=2)


def draw_pending(ax) -> None:
    """A set with no score yet: an empty slot for each setting."""
    pending_slot(ax, 0, 0.8, LEGACY)
    pending_slot(ax, 1.7, 0.8, SI)
    finish_groups(ax, [0, 1.7], ["", ""], [0, 1.7], ["legacy", "Tokamak-SI"], y=-0.04)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def draw(out: Path, png: Path | None, table: Path | None) -> None:
    """The F1 figure: every set, with the sets not yet scored."""
    ae, conf, elm = (load(SOURCES[k]) for k in ("ae", "confinement", "elm"))
    si = confinement_si(load(SOURCES["confinement_si"]))
    rows = Rows()
    with style():
        fig = Figure(figsize=(PAGE_IN, 4.0))
        outer = fig.add_gridspec(
            2,
            1,
            left=0.06,
            right=0.995,
            top=0.88,
            bottom=0.04,
            hspace=0.5,
            height_ratios=[1.0, 0.55],
        )
        top = outer[0].subgridspec(1, 3, width_ratios=[2.2, 1, 1.15], wspace=0.22)
        low = outer[1].subgridspec(1, 3, wspace=0.22)
        a, b, c = (fig.add_subplot(top[0, i]) for i in range(3))
        pend = [fig.add_subplot(low[0, i]) for i in range(3)]
        draw_ae(a, ae, rows, "a")
        draw_confinement(b, conf, si, rows, "b")
        draw_elm(c, elm, rows, "c")
        for ax in pend:
            draw_pending(ax)
        shots = len(ae["shots"]["fair"])
        titles = {
            a: f"Alfv\u00e9n Eigenmodes ({shots} held-out shots)",
            b: "Confinement",
            c: "Edge Localized Modes",
        }
        titles.update(zip(pend, PENDING))
        for ax, text in titles.items():
            ax.set_title(text, loc="left", fontsize=FONT_PT, pad=4)
        handles = [
            Patch(color=LEGACY, label="legacy"),
            Patch(color=SI, label="Tokamak-SI"),
            Patch(
                facecolor="none",
                edgecolor=PENDING_EDGE,
                hatch="////",
                label="not yet scored",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=1.6,
            columnspacing=1.6,
            fontsize=FONT_PT - 1,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="pdf", metadata={"CreationDate": None})
        if png is not None:
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=200)
    rows.write(table)
    gits = {k: load(p).get("git") for k, p in SOURCES.items()}
    print(f"wrote {out}; sources at git {gits}", file=sys.stderr)


def draw_auc(out: Path, png: Path | None, table: Path | None) -> None:
    """AUROC (top row) and AUPRC (bottom row) for the scored sets."""
    ae, conf, elm = (load(SOURCES[k]) for k in ("ae", "confinement", "elm"))
    si = confinement_si(load(SOURCES["confinement_si"]))
    rows = Rows()
    with style():
        fig = Figure(figsize=(PAGE_IN, 4.6))
        grid = fig.add_gridspec(
            2,
            3,
            left=0.06,
            right=0.995,
            top=0.9,
            bottom=0.09,
            hspace=0.78,
            wspace=0.22,
            width_ratios=[2.2, 1, 1.15],
        )
        shots = len(ae["shots"]["fair"])
        titles = (
            f"Alfv\u00e9n Eigenmodes ({shots} held-out shots)",
            "Confinement",
            "Edge Localized Modes",
        )
        for r, metric in enumerate(("auroc", "auprc")):
            axes = [fig.add_subplot(grid[r, i]) for i in range(3)]
            draw_ae(axes[0], ae, rows, "a" + str(r), metric)
            draw_confinement(axes[1], conf, si, rows, "b" + str(r), metric)
            draw_elm(axes[2], elm, rows, "c" + str(r), metric)
            for ax, text in zip(axes, titles):
                ax.set_title(text, loc="left", fontsize=FONT_PT, pad=4)
        handles = [
            Patch(color=LEGACY, label="legacy"),
            Patch(color=SI, label="Tokamak-SI"),
            Patch(
                facecolor="none",
                edgecolor=PENDING_EDGE,
                hatch="////",
                label="not reported",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=1.6,
            columnspacing=1.6,
            fontsize=FONT_PT - 1,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="pdf", metadata={"CreationDate": None})
        if png is not None:
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=200)
    rows.write(table)
    print(f"wrote {out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--png", type=Path, help="also save a PNG preview here")
    parser.add_argument("--table", type=Path, help="write the numbers drawn as CSV")
    parser.add_argument(
        "--metric",
        choices=("f1", "auc"),
        default="f1",
        help="f1: the main figure; auc: AUROC and AUPRC, for the appendix "
        f"(default out: {DEFAULT_AUC_OUT.name})",
    )
    args = parser.parse_args(argv)
    if args.metric == "auc":
        out = DEFAULT_AUC_OUT if args.out == DEFAULT_OUT else args.out
        draw_auc(out, args.png, args.table)
    else:
        draw(args.out, args.png, args.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
