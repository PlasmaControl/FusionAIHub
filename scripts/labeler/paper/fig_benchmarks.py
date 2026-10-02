r"""fig_benchmarks: per label set, the prior papers' benchmark models beside the
best model, one panel per set.

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig_benchmarks.py \
        [--out dev/label_paper/figures/fig_benchmarks.pdf] [--png PATH] [--table PATH]

Every score drawn is read from an evaluation JSON (`SOURCES`), with the 95 %
shot-bootstrap interval stored beside it; none is typed in here. The models are
the ones the paper keeps: per set, the prior-paper benchmark and the best model,
nothing else.

- (a) AE, F1 and (b) AE, AUROC: RCN and LSTM (chord spectrogram, each chord
  scored) and SELDnet on the 19 held-out shots, 10 ms frames, against the dense
  labels (`results.reviewed`) and, lighter, against the Heidbrink annotation
  (`results.annotated`); the always-present constant (`always`) dashed across
  each group. AUROC is drawn because every bar in the AE panels has one.
- (c) Confinement, F1: the retrained BES-CNN per class and macro, out of fold
  on 119 shots (`results.all_shots`), beside the prior paper's own per-class F1
  (`paper`) as a marker; its macro marker is the mean of those four, as the
  paper's "Average" row (0.94). F1 only: the published scores have no AUROC.
- (d) ELMs: ELM-O's bin F1 on the reviewed spans (`review.paper`), then its
  precision and recall on D. Smith's windows (`smith.published_setting`)
  beside the published ones at the same setting (`paper.table`).
- (e) Tearing modes, sawteeth and RWM onsets have no score yet: two empty
  hatched slots each (benchmark, best model), the benchmark named where one is
  chosen (`PENDING`). Disruptions have no labels yet.

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
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from labeler.paper import FONT_PT, PAGE_IN, style

REPO = Path(__file__).resolve().parents[3]
OUTPUTS = REPO / "outputs" / "labeler"
SOURCES = {
    "ae": OUTPUTS / "ae" / "baselines" / "evaluation.json",
    "confinement": OUTPUTS / "confinement" / "bes" / "evaluation.json",
    "elm": OUTPUTS / "elm" / "elmo" / "evaluation.json",
}
DEFAULT_OUT = REPO / "dev" / "label_paper" / "figures" / "fig_benchmarks.pdf"

# Okabe-Ito, one colour per role (checked with the dataviz palette validator,
# all pairs, light surface): prior-paper benchmark, best model, published score.
BENCHMARK = "#E69F00"
OURS = "#0072B2"
PUBLISHED = "#CC79A7"
INK = "#333333"
REFERENCE = "#555555"
PENDING_EDGE = "#999999"
LIGHTER = 0.45  # the annotation group's tint: this share of the colour, the rest white
VALUE_PT = FONT_PT - 1.5
BAR = 0.62
YMAX = 1.13
YTICKS = np.arange(0.0, 1.01, 0.25)

AE_MODELS = (  # benchmarks first, then the best model
    ("rcn_spec_each", "RCN", BENCHMARK),
    ("lstm_spec_each", "LSTM", BENCHMARK),
    ("seldnet", "SELDnet", OURS),
)
AE_TRUTHS = (
    ("reviewed", "dense labels", False),
    ("annotated", "Heidbrink annotation", True),
)
CLASSES = (("L", "L"), ("H", "H"), ("QH", "QH"), ("WP", "WPQH"))
PENDING = (  # (set, its prior-paper benchmark, or None where none is chosen)
    ("Tearing modes", "tearing-onset CNN,\ntime-to-event model"),
    ("Sawteeth", None),
    ("RWMs", None),
)


def tint(colour: str, share: float = LIGHTER) -> tuple[float, float, float]:
    return tuple(share * c + (1 - share) for c in to_rgb(colour))


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


def scored_bar(ax, x, value, ci, colour, name=None) -> None:
    """A bar with its interval and its value written above; `name`, if given,
    written up the bar from its foot."""
    ax.bar(x, value, BAR, color=colour, edgecolor="none", zorder=2)
    if name:
        ax.text(
            x,
            0.025,
            name,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=VALUE_PT,
            color="white" if colour == OURS else INK,
            zorder=4,
        )
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
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=VALUE_PT,
        color=INK,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
    )


def published_mark(ax, x, value) -> None:
    ax.plot(
        x,
        value,
        marker="D",
        ms=3.6,
        mfc=PUBLISHED,
        mec=INK,
        mew=0.5,
        ls="none",
        zorder=5,
    )


def pending_slot(ax, x, width, edge, text=None, height=1.0) -> None:
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


def group_labels(ax, centres, labels, y=-0.13) -> None:
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


def draw_ae(ax, ae: dict, metric: str, rows: Rows, panel: str) -> None:
    """RCN, LSTM and SELDnet against both truths, the constant dashed across."""
    name = "F1" if metric == "f1" else "AUROC"
    ticks, centres, labels = [], [], []
    step = len(AE_MODELS) + 0.6
    for g, (truth, truth_name, lighter) in enumerate(AE_TRUTHS):
        block = ae["results"][truth]
        xs = [g * step + i for i in range(len(AE_MODELS))]
        for x, (key, label, colour) in zip(xs, AE_MODELS):
            entry = block[key]
            ci = entry["ci95"][metric]
            fill = tint(colour) if lighter else colour
            scored_bar(ax, x, entry[metric], ci, fill, name=label)
            ticks.append(x)
            rows.add(
                panel,
                label,
                truth_name,
                name,
                entry[metric],
                ci,
                SOURCES["ae"],
                f"results.{truth}.{key}.{metric}",
            )
        always = block["always"][metric]
        ax.hlines(
            always,
            xs[0] - 0.5,
            xs[-1] + 0.5,
            colors=REFERENCE,
            linestyles=(0, (3, 2)),
            lw=0.8,
            zorder=1.5,  # behind the bars, so it never runs through their text
        )
        rows.add(
            panel,
            "always present",
            truth_name,
            name,
            always,
            None,
            SOURCES["ae"],
            f"results.{truth}.always.{metric}",
        )
        share = round(100 * block["seldnet"]["truth_fraction"])
        centres.append(float(np.mean(xs)))
        labels.append(f"{truth_name}\n({share}% present)")
    ax.set_xticks([])
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    group_labels(ax, centres, labels, y=-0.04)
    axes_style(ax, name)


def draw_confinement(ax, conf: dict, rows: Rows, panel: str) -> None:
    """BES-CNN per class and macro, the published F1 beside, the best model pending."""
    result = conf["results"]["all_shots"]
    published = conf["paper"]
    step = 1.35
    offset = 0.6  # the published marker, right of its value
    ticks, ticklabels = [], []
    for i, (key, label) in enumerate(CLASSES):
        x = i * step
        value = result["classes"][key]["f1"]
        ci = result["ci95"][f"f1_{key}"]
        scored_bar(ax, x, value, ci, BENCHMARK)
        published_mark(ax, x + offset, published[key]["f1"])
        ticks.append(x + offset / 2)
        ticklabels.append(label)
        rows.add(
            panel,
            "BES-CNN",
            label,
            "F1",
            value,
            ci,
            SOURCES["confinement"],
            f"results.all_shots.classes.{key}.f1",
        )
        rows.add(
            panel,
            "BES-CNN as published",
            label,
            "F1",
            published[key]["f1"],
            None,
            SOURCES["confinement"],
            f"paper.{key}.f1",
        )
    x = len(CLASSES) * step
    value = result["macro"]["f1"]
    ci = result["ci95"]["macro_f1"]
    scored_bar(ax, x, value, ci, BENCHMARK)
    paper_macro = float(np.mean([published[k]["f1"] for k, _ in CLASSES]))
    published_mark(ax, x + offset, paper_macro)
    ticks.append(x + offset / 2)
    ticklabels.append("macro")
    rows.add(
        panel,
        "BES-CNN",
        "macro",
        "F1",
        value,
        ci,
        SOURCES["confinement"],
        "results.all_shots.macro.f1",
    )
    rows.add(
        panel,
        "BES-CNN as published",
        "macro",
        "F1",
        paper_macro,
        None,
        SOURCES["confinement"],
        "mean of paper.{L,H,QH,WP}.f1",
    )
    slot = x + step + 0.15
    pending_slot(ax, slot, 0.8, OURS, "best model: pending")
    ticks.append(slot)
    ticklabels.append("best")
    ax.set_xticks(ticks, ticklabels)
    ax.set_xlim(-0.55, slot + 0.55)
    axes_style(ax, "F1")
    windows = result["windows"]
    group_labels(
        ax,
        [np.mean(ticks[:-1])],
        [f"BES-CNN, {result['shots']} shots out of fold ({windows:,} windows)"],
    )


def draw_elm(ax, elm: dict, rows: Rows, panel: str) -> None:
    """ELM-O's F1 on the reviewed spans, the best model pending; then its precision and
    recall on D. Smith's windows beside the published ones."""
    review = elm["review"]["paper"]
    scored_bar(ax, 0, review["f1"], review["ci95"]["f1"], BENCHMARK)
    rows.add(
        panel,
        "ELM-O",
        "reviewed spans",
        "F1",
        review["f1"],
        review["ci95"]["f1"],
        SOURCES["elm"],
        "review.paper.f1",
    )
    pending_slot(ax, 1.0, 0.8, OURS, "best model: pending")
    smith = elm["smith"]["published_setting"]
    paper = elm["paper"]["table"][f"{smith['threshold']:.1f}"]
    offset = 0.42
    ticks, ticklabels = [0, 1.0], ["ELM-O\nF1", "best\nF1"]
    for x, metric, label in ((2.45, "precision", "P"), (3.65, "recall", "R")):
        scored_bar(ax, x, smith[metric], smith["ci95"][metric], BENCHMARK)
        published_mark(ax, x + offset, paper[metric])
        ticks.append(x + offset / 2)
        ticklabels.append(f"ELM-O\n{label}")
        rows.add(
            panel,
            "ELM-O",
            "Smith's windows",
            label,
            smith[metric],
            smith["ci95"][metric],
            SOURCES["elm"],
            f"smith.published_setting.{metric}",
        )
        rows.add(
            panel,
            "ELM-O as published",
            "Smith's windows",
            label,
            paper[metric],
            None,
            SOURCES["elm"],
            f"paper.table.{smith['threshold']:.1f}.{metric}",
        )
    ax.axvline(1.72, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
    ax.set_xticks(ticks, ticklabels)
    ax.set_xlim(-0.55, 4.6)
    axes_style(ax, "score")
    group_labels(
        ax,
        [0.5, 3.05 + offset / 2],
        [
            f"reviewed spans\n({elm['review']['shots']} shots)",
            f"D. Smith's windows\n({elm['smith']['shots']} shots)",
        ],
        y=-0.2,
    )


def draw_pending(ax) -> None:
    """The sets with no score yet: two empty slots each, benchmark and best model."""
    step = 2.6
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ticks, ticklabels = [], []
    for i, (name, benchmark) in enumerate(PENDING):
        x = i * step
        pending_slot(ax, x, 0.8, BENCHMARK, "benchmark: pending")
        pending_slot(ax, x + 1, 0.8, OURS, "best model: pending")
        ticks += [x, x + 1]
        ticklabels += ["prior", "best"]
        ax.text(
            x + 0.5,
            1.0,
            name,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=FONT_PT,
            color=INK,
        )
        ax.text(
            x + 0.5,
            -0.13,
            f"benchmark:\n{benchmark}" if benchmark else "benchmark:\nnot yet chosen",
            transform=trans,
            ha="center",
            va="top",
            fontsize=FONT_PT - 1,
            color=INK,
        )
    x = len(PENDING) * step + 0.3
    ax.text(
        x,
        0.5,
        "Disruptions:\nno labels yet",
        transform=trans,
        ha="center",
        va="center",
        fontsize=FONT_PT - 1,
        color=INK,
        style="italic",
    )
    ax.set_xticks(ticks, ticklabels)
    ax.set_xlim(-0.6, x + 0.9)
    axes_style(ax, None)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def draw(out: Path, png: Path | None, table: Path | None) -> None:
    ae, conf, elm = (load(SOURCES[k]) for k in ("ae", "confinement", "elm"))
    rows = Rows()
    with style():
        fig = Figure(figsize=(PAGE_IN, 4.1))
        outer = fig.add_gridspec(
            2, 1, left=0.055, right=0.995, top=0.85, bottom=0.12, hspace=0.62
        )
        top = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.24)
        low = outer[1].subgridspec(1, 2, width_ratios=[1.05, 2.25], wspace=0.12)
        a, b, c = (fig.add_subplot(top[0, i]) for i in range(3))
        d, e = (fig.add_subplot(low[0, i]) for i in range(2))
        draw_ae(a, ae, "f1", rows, "a")
        draw_ae(b, ae, "auroc", rows, "b")
        draw_confinement(c, conf, rows, "c")
        draw_elm(d, elm, rows, "d")
        draw_pending(e)
        shots = len(ae["shots"]["fair"])
        titles = {
            a: f"(a) AE, F1 ({shots} held-out shots)",
            b: f"(b) AE, AUROC ({shots} held-out shots)",
            c: "(c) Confinement regime, F1 by class",
            d: "(d) ELMs",
            e: "(e) No score yet",
        }
        for ax, text in titles.items():
            ax.set_title(text, loc="left", fontsize=FONT_PT, pad=13 if ax is e else 4)
        handles = [
            Patch(color=BENCHMARK, label="prior-paper benchmark"),
            Patch(color=OURS, label="best model"),
            Patch(
                facecolor=tint(BENCHMARK),
                edgecolor="none",
                label="lighter: vs. Heidbrink annotation",
            ),
            Line2D(
                [],
                [],
                marker="D",
                ms=3.6,
                mfc=PUBLISHED,
                mec=INK,
                mew=0.5,
                ls="none",
                label="as published (its paper's own test)",
            ),
            Line2D(
                [], [], color=REFERENCE, ls=(0, (3, 2)), lw=0.8, label="always present"
            ),
            Patch(
                facecolor="none",
                edgecolor=PENDING_EDGE,
                hatch="////",
                label="no model yet",
            ),
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
            handlelength=1.6,
            columnspacing=1.2,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--png", type=Path, help="also save a PNG preview here")
    parser.add_argument("--table", type=Path, help="write the numbers drawn as CSV")
    args = parser.parse_args(argv)
    draw(args.out, args.png, args.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
