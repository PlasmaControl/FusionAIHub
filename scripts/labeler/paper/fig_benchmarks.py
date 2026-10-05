r"""fig_benchmarks: Figure 2 of the paper. Every label set, legacy against Tokamak-SI:
the F1 of each model "<task>-<model>" in two label settings (two rows of panels),
and below them how much labelled data each setting holds (the coverage row).

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig_benchmarks.py \
        [--out dev/label_paper/figures/fig_benchmarks.pdf] [--png PATH] [--table PATH]

Every score drawn is read from a committed record (`SOURCES`), with the 95 %
shot-bootstrap interval stored beside it where there is one; none is typed in
here. Per set, "legacy" is the setting the model's own paper (or the older
annotation) used and "Tokamak-SI" is our labels:

- AE: `ae-ours` (the SELDnet model), `ae-rcn` and `ae-lstm` (chord
  spectrogram, each chord scored) on the 19 held-out shots, 10 ms frames;
  legacy = against the Heidbrink annotation (`results.annotated`), Tokamak-SI =
  against the dense labels (`results.reviewed`).
- Confinement: `confine-cnn`; legacy = the earlier paper's macro F1, the mean
  of its four per-class F1 (`paper` in `gill_2024_published.json`, its results
  table, committed beside the other records), one bar, no interval;
  Tokamak-SI = our reimplementation under the paper's selection and split, the
  ablation's protocol row `full_cum_abcdrgef` (142 distinct test shots in 200
  shot-tests from five random by-shot splits whose test sets overlap, windows
  pooled; macro F1, AUROC and AUPRC with shot-bootstrap intervals), the one
  confine-cnn score of the paper. The first retrain's own score is not drawn.
- ELMs: `elm-elmo` and `elm-ours` from the ELM stream's final records. Legacy =
  D. Smith's windows at the published setting (`reimplemented_elmo_overlap` of
  `smith/evaluation.json`: overlap-region F1, conditional on the selected
  windows; the F1 is checked against the one the stored counts give). Tokamak-SI
  = the reviewed spans in 50 ms bins on the 73 shots with BES, the set both
  models are scored on (`sets.bes73` of `ours/evaluation.json`); `elm-ours` has
  no legacy bar, and its score on all 119 reviewed shots is in the CSV only.
- Tearing modes: `tm-onsetcnn` and `tm-dsm` (published model at its published
  threshold on the Seo cohort, 25 ms bins, against the retrained twin on the
  Tokamak-SI labels, 10 ms bins) and `tm-ours` (Tokamak-SI only), from
  `docs/labeler/figure2_tm.json`. The Tokamak-SI F1 excludes uncertain time;
  an open diamond marks the same model with uncertain time scored negative.
- RWM: `rwm-brf` on the Tokamak-SI labels (`outputs/labeler/rwm/evaluation.json`),
  pooled slice F1, with a tick at the elapsed-time baseline (the legend's
  "trivial baseline", as the always-present tick of the sawtooth panel); the legacy
  forest was scored on another machine and has no comparable F1 (a hatched slot).
- Sawtooth: `saw-derivative` (the single-channel derivative picker, which reads the same
  ECE edges the labels are built from), `saw-hl3` (the OuYang 2025 HL-3 network,
  replicated) and `saw-ours`
  on the Tokamak-SI labels, from the sawtooth stream's benchmark record
  (`outputs/labeler/sawtooth/fix5/benchmark.json`): presence F1 on 2 ms bins,
  out of fold over the shot-grouped three-fold split of the TRAIN shots (the
  record's `crash_tolerance_2ms.presence`; the presence threshold is selected
  per fold on inner-selection shots), with the record's 95 % shot-bootstrap
  interval and a tick at the always-present baseline. The score is agreement
  with the physics rule on assessed bins, not an independent truth. HL-3's own
  paper was scored on another machine: a hatched slot, its stated three-class
  window accuracy in the CSV only.

The coverage row reads `docs/labeler/figure2_coverage.json` (written by
`fig2_coverage.py`): labelled shots and labelled time per set, legacy against
Tokamak-SI, with a darker inner bar for the human-reviewed subset.

`--metric auc` draws the same sets for AUROC and AUPRC (no coverage row) into
`fig_benchmarks_auc.pdf`: a legacy score only where its paper reported one, else
a hatched grey slot (confine-cnn's and elm-elmo's AUPRC, elm-elmo's AUROC; the
NSTX forest's slice AUROC is another machine and is in the CSV only). The
confine-cnn paper reported a one-vs-rest AUC of at least 0.99 for every class and
no macro value, so its legacy AUROC slot is a hatched bar at 0.99 labelled as
that bound, not a macro score (`auc_one_vs_rest` in `gill_2024_published.json`);
the RWM AUROC is the phase-controlled one. elm-elmo's AUROC and AUPRC come from a
sweep of its threshold.

A hatched empty slot is a value not available; its reason is written in it.
`--table` writes every number drawn, and the numbers kept out of the figure, with
their JSON key as CSV; the same rows are printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from labeler.paper import FONT_PT, PAGE_IN, style

REPO = Path(__file__).resolve().parents[3]
OUTPUTS = REPO / "outputs" / "labeler"
SOURCES = {
    "ae": OUTPUTS / "ae" / "baselines" / "evaluation.json",
    # the earlier paper's per-class scores (`paper`) and AUC bound
    # (`auc_one_vs_rest`), from its results table
    "confinement": OUTPUTS / "confinement" / "bes" / "gill_2024_published.json",
    # the protocol row's record (confinement_bes_ablation.py summarize)
    "confinement_si": OUTPUTS
    / "confinement"
    / "bes"
    / "ablation_rows"
    / "full_cum_abcdrgef.json",
    # the ELM stream's final records, the ones its paper tables read
    "elm": OUTPUTS / "elm" / "ours" / "evaluation.json",
    "elm_smith": OUTPUTS / "elm" / "smith" / "evaluation.json",
    "tm": REPO / "docs" / "labeler" / "figure2_tm.json",
    "rwm": OUTPUTS / "rwm" / "evaluation.json",
    # the sawtooth stream's benchmark: out-of-fold presence scores per model
    "saw": OUTPUTS / "sawtooth" / "fix5" / "benchmark.json",
    "coverage": REPO / "docs" / "labeler" / "figure2_coverage.json",
}
DEFAULT_OUT = REPO / "dev" / "label_paper" / "figures" / "fig_benchmarks.pdf"
DEFAULT_AUC_OUT = DEFAULT_OUT.with_name("fig_benchmarks_auc.pdf")

# Okabe-Ito, one colour per label setting (checked with the dataviz palette
# validator, light surface); the reviewed subset is a darker shade of Tokamak-SI.
LEGACY = "#E69F00"
SI = "#0072B2"
REVIEWED = "#0A2A47"
INK = "#333333"
PENDING_EDGE = "#999999"
# every text in the figure is at least FONT_PT (7 pt)
TEXT = {
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
}
VALUE_PT = FONT_PT
BAR = 0.62
YMAX = 1.2
YTICKS = np.arange(0.0, 1.01, 0.25)
S_PER_H = 3600.0
# layout in inches
LEFT_IN, RIGHT_IN, GAP_IN = 0.45, 0.06, 0.5
TITLE_IN, NAMES_IN = 0.2, 0.46
AXES_IN = 0.82  # height of a score panel
COVER_IN, COVER_NAMES_IN = 0.8, 0.55  # the coverage row, and its rotated names
SLOT = 0.8  # width of an empty slot, in bar units
GROUP_GAP = 1.7  # distance from the legacy group to the Tokamak-SI group
BELOW_PT = 26  # the group label's distance under the axis, in points

AE_MODELS = (  # (key in the evaluation, "<task>-<model>")
    ("seldnet", "ae-ours"),
    ("rcn_spec_each", "ae-rcn"),
    ("lstm_spec_each", "ae-lstm"),
)
AE_SETTINGS = (  # (setting, truth block, what it is scored against, colour)
    ("legacy", "annotated", "Heidbrink annotation", LEGACY),
    ("Tokamak-SI", "reviewed", "dense labels", SI),
)
TM_MODELS = (  # (architecture in the record, "<task>-<model>")
    ("onsetcnn", "tm-onsetcnn"),
    ("dsm", "tm-dsm"),
    ("magnetic-detector", "tm-ours"),
)
# the record's key for each metric: (legacy, Tokamak-SI)
TM_KEYS = {
    "f1": ("f1_published_threshold", "f1"),
    "auroc": ("auroc", "auroc"),
    "auprc": ("auprc", "auprc"),
}
SAW_MODELS = (  # the models of the record, named "<task>-<model>" there
    "saw-derivative",
    "saw-hl3",
    "saw-ours",
)
SAW_TRIVIAL = "saw-always-present"
SAW_BLOCK = "crash_tolerance_2ms"  # holds the presence scores (the same at 1 ms)
CLASSES = ("L", "H", "QH", "WP")
METRICS = {"f1": "F1", "auroc": "AUROC", "auprc": "AUPRC"}
STATUS_TEXT = {
    "none": "no legacy set",
    "pending": "pending",
    "point_events": "point events",
}
TITLES = {
    "tm": "Tearing Modes",
    "sawtooth": "Sawtooth",
    "rwm": "Resistive Wall Modes",
    "conf": "Confinement",
    "elm": "Edge Localized Modes",
}


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


class Rows:
    """Every number drawn (and the numbers kept out of the figure), with where it
    came from."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(
        self,
        panel,
        series,
        group,
        metric,
        value,
        ci,
        source,
        key,
        note="",
        drawn=True,
    ) -> None:
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
                "drawn": drawn,
                "note": note,
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


def two_line(name: str) -> str:
    """`ae-ours` as `ae-` over `ours`: the names sit under bars a few tenths of an
    inch apart."""
    head, _, tail = name.partition("-")
    return f"{head}-\n{tail}" if tail else name


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


def uncertain_marker(ax, x, value) -> None:
    """The same model with uncertain time scored negative: an open diamond."""
    ax.plot(
        [x],
        [value],
        marker="D",
        ms=3.6,
        mfc="white",
        mec=INK,
        mew=0.8,
        ls="none",
        zorder=5,
    )


def baseline_tick(ax, x, value) -> None:
    """A baseline's score on the model's bar: a short tick across it."""
    half = BAR / 2 * 1.4
    for colour, width in (("white", 3.0), (INK, 1.4)):
        ax.plot(
            [x - half, x + half],
            [value, value],
            color=colour,
            lw=width,
            solid_capstyle="butt",
            zorder=5,
        )


def pending_slot(
    ax,
    x,
    width,
    edge,
    height=1.0,
    text=None,
    bottom=0.0,
    ytext=None,
) -> None:
    """An empty hatched slot: a value not available, with its reason written in."""
    ax.add_patch(
        Rectangle(
            (x - width / 2, bottom),
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
            bottom + height / 2 if ytext is None else ytext,
            text,
            rotation=90,
            ha="center",
            va="center",
            fontsize=VALUE_PT,
            linespacing=1.1,
            color=INK,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )


def bound_slot(ax, x, width, value, colour, text) -> None:
    """A hatched bar up to a published lower bound that holds for every class: not a
    macro score, so it carries no value label and says what it is."""
    ax.add_patch(
        Rectangle(
            (x - width / 2, 0),
            width,
            value,
            facecolor="none",
            edgecolor=colour,
            hatch="////",
            lw=0.6,
            zorder=1,
        )
    )
    ax.plot(
        [x - width / 2, x + width / 2], [value, value], color=colour, lw=1.2, zorder=3
    )
    ax.text(
        x,
        value / 2,
        text,
        rotation=90,
        ha="center",
        va="center",
        fontsize=VALUE_PT,
        linespacing=1.1,
        color=INK,
        zorder=4,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.95},
    )


def group_labels(ax, centres, labels, below_pt=BELOW_PT) -> None:
    """The setting of each group of bars, under the bar names."""
    for c, text in zip(centres, labels):
        ax.annotate(
            text,
            xy=(c, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -below_pt),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=FONT_PT,
            color=INK,
            annotation_clip=False,
        )


def finish_groups(ax, ticks, names, centres, labels, below_pt=BELOW_PT) -> None:
    ax.set_xticks(ticks, [two_line(n) for n in names])
    ax.tick_params(axis="x", labelsize=FONT_PT)
    ax.set_xlim(min(ticks) - 0.7, max(ticks) + 0.7)
    group_labels(ax, centres, labels, below_pt)
    axes_style(ax, None)


def cell(c: dict) -> tuple[float, tuple | None]:
    """A record's `{value, lo, hi}` cell: its value and its 95 % interval (none when
    the interval is a point)."""
    lo, hi = c["lo"], c["hi"]
    return c["value"], ((lo, hi) if hi > lo else None)


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
                note=f"scored against the {truth_name}",
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
    """confine-cnn's macro score: the earlier paper's (legacy, from `conf`: macro
    F1; for AUROC only its bound of 0.99 per class, one-vs-rest, no macro value;
    no AUPRC) and our reimplementation under its selection and split (Tokamak-SI,
    from `si`, see `confinement_si`)."""
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
    elif metric == "auroc":
        bound = conf["auc_one_vs_rest"]["per_class_at_least"]
        bound_slot(
            ax,
            0,
            SLOT,
            bound,
            LEGACY,
            f"≥ {bound:.2f}\nper class",
        )
        rows.add(
            panel,
            "confine-cnn",
            "legacy",
            "per-class AUROC, lower bound (one-vs-rest)",
            bound,
            None,
            SOURCES["confinement"],
            "auc_one_vs_rest.per_class_at_least",
        )
    else:
        pending_slot(ax, 0, SLOT, PENDING_EDGE, text="not reported")
    value = si["macro"][metric]
    ci = si["ci95"][f"macro_{metric}"]
    scored_bar(ax, GROUP_GAP, value, ci, SI)
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
        [0, GROUP_GAP],
        ["confine-cnn"] * 2,
        [0, GROUP_GAP],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_ylabel(name, labelpad=2)


def smith_legacy(smith: dict) -> tuple[float, tuple]:
    """elm-elmo's legacy score: D. Smith's windows at the published setting, the
    ELM stream's reimplementation (`reimplemented_elmo_overlap`). Its F1 must be the
    one its stored counts and its stored precision and recall give. Returns the F1
    and its interval."""
    block = smith["reimplemented_elmo_overlap"]
    c, point = block["counts"], block["point"]
    from_counts = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"])
    p, r = point["precision"], point["recall"]
    from_pr = 2 * p * r / (p + r)
    for name, other in (("counts", from_counts), ("precision and recall", from_pr)):
        if abs(point["f1"] - other) > 1e-9:
            raise ValueError(f"Smith F1 {point['f1']} is not its {name}' {other}")
    return point["f1"], tuple(block["ci95"]["f1"])


def draw_elm(
    ax, elm: dict, smith: dict, rows: Rows, panel: str, metric: str = "f1"
) -> None:
    """elm-elmo on D. Smith's windows (legacy; F1 only, there is no threshold sweep
    there) and on the reviewed spans, and elm-ours on the reviewed spans (Tokamak-SI,
    the shots with BES that both models are scored on; AUROC and AUPRC come from a
    sweep of the detection threshold)."""
    name = METRICS[metric]
    src = SOURCES["elm"]
    models = (("elm-elmo", "elm-elmo"), ("elm-ours", "elm-ours"))
    if metric == "f1":
        value, ci = smith_legacy(smith)
        scored_bar(ax, 0, value, ci, LEGACY)
        rows.add(
            panel,
            "elm-elmo",
            "legacy",
            name,
            value,
            ci,
            SOURCES["elm_smith"],
            "reimplemented_elmo_overlap.point.f1",
            note="D. Smith's windows at the published setting; overlap-region F1, "
            "conditional on the selected windows",
        )
        old = smith["original_cached_elmo_overlap"]["point"]["f1"]
        rows.add(
            panel,
            "elm-elmo",
            "legacy, original cached sweep",
            name,
            old,
            tuple(smith["original_cached_elmo_overlap"]["ci95"]["f1"]),
            SOURCES["elm_smith"],
            "original_cached_elmo_overlap.point.f1",
            note="the earlier figure's bar, kept out",
            drawn=False,
        )
    else:
        pending_slot(ax, 0, SLOT, PENDING_EDGE, text="not reported")
        rows.add(
            panel,
            "elm-elmo",
            "legacy, window occupancy",
            name,
            smith["methods"]["elm-elmo"]["occupancy_1ms"]["point"][metric],
            None,
            SOURCES["elm_smith"],
            f"methods.elm-elmo.occupancy_1ms.point.{metric}",
            note="reimplementation scored on the Smith windows' 1 ms cells; "
            "the earlier paper reported none",
            drawn=False,
        )
    bes = elm["sets"]["bes73"]
    xs = [GROUP_GAP, GROUP_GAP + 1]
    for x, (key, label) in zip(xs, models):
        res = bes["methods"][key]
        if metric == "f1":
            value, ci = res["point"]["f1"], res["ci95"]["f1"]
            where = f"sets.bes73.methods.{key}.point.f1"
        else:
            value, ci = res["point"][metric], res["ci95"][metric]
            where = f"sets.bes73.methods.{key}.point.{metric}"
        scored_bar(ax, x, value, ci, SI)
        rows.add(
            panel,
            label,
            "Tokamak-SI",
            name,
            value,
            ci,
            src,
            where,
            note="reviewed spans, 50 ms bins, the shots with BES",
        )
    res = elm["sets"]["all119"]["methods"]["elm-ours"]
    rows.add(
        panel,
        "elm-ours",
        "Tokamak-SI, all reviewed shots",
        name,
        res["point"][metric],
        res["ci95"][metric],
        src,
        f"sets.all119.methods.elm-ours.point.{metric}",
        note="kept out of the figure: elm-elmo needs BES, so the bars share bes73",
        drawn=False,
    )
    finish_groups(
        ax,
        [0, *xs],
        ["elm-elmo"] * 2 + ["elm-ours"],
        [0, float(np.mean(xs))],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_ylabel(name, labelpad=2)


def draw_tm(ax, tm: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """tm-onsetcnn and tm-dsm as the published model on its own labels (legacy, at
    its published threshold) and as the retrained twin on the Tokamak-SI labels, and
    tm-ours on Tokamak-SI only; each Tokamak-SI bar has an open diamond at the same
    model with uncertain time scored negative."""
    legacy_key, si_key = TM_KEYS[metric]
    by = {r["architecture"]: r for r in tm["rows"]}
    src = SOURCES["tm"]
    name = METRICS[metric]
    legacy_models = [(a, n) for a, n in TM_MODELS if by[a].get("legacy")]
    ticks, names = [], []
    xs = list(range(len(legacy_models)))
    for x, (arch, label) in zip(xs, legacy_models):
        block = by[arch]["legacy"]
        value, ci = cell(block[legacy_key])
        scored_bar(ax, x, value, ci, LEGACY)
        ticks.append(x)
        names.append(label)
        rows.add(
            panel,
            label,
            "legacy",
            name,
            value,
            ci,
            src,
            f"rows[{arch}].legacy.{legacy_key}",
            note=f"{block['model']}, published threshold, "
            f"{block['bin_ms']:g} ms bins, {block['shots']} shots",
        )
        if metric == "f1":
            tuned, tuned_ci = cell(block["f1_tuned"])
            rows.add(
                panel,
                label,
                "legacy, tuned threshold",
                name,
                tuned,
                tuned_ci,
                src,
                f"rows[{arch}].legacy.f1_tuned",
                note="secondary; kept out of the figure",
                drawn=False,
            )
    legacy_centre = float(np.mean(xs))
    x0 = (xs[-1] if xs else 0) + GROUP_GAP
    si_xs = [x0 + i for i in range(len(TM_MODELS))]
    for x, (arch, label) in zip(si_xs, TM_MODELS):
        row = by[arch]
        value, ci = cell(row["tokamak_si"][si_key])
        scored_bar(ax, x, value, ci, SI)
        unc, _ = cell(row["tokamak_si_uncertain_negative"][si_key])
        uncertain_marker(ax, x, unc)
        ticks.append(x)
        names.append(label)
        si = row["tokamak_si"]
        rows.add(
            panel,
            label,
            "Tokamak-SI",
            name,
            value,
            ci,
            src,
            f"rows[{arch}].tokamak_si.{si_key}",
            note=f"{row['tokamak_si_model']}, {si['bin_ms']:g} ms bins, "
            f"{si['shots']} shots, uncertain time excluded",
        )
        rows.add(
            panel,
            label,
            "Tokamak-SI, uncertain time scored negative",
            name,
            unc,
            None,
            src,
            f"rows[{arch}].tokamak_si_uncertain_negative.{si_key}",
            note="open diamond",
        )
    finish_groups(
        ax,
        ticks,
        names,
        [legacy_centre, float(np.mean(si_xs))],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_ylabel(name, labelpad=2)


def rwm_cell(config: dict, metric: str) -> tuple[float, tuple, str]:
    """rwm-brf's (or a baseline's) value, interval and key for `metric`: pooled
    slice F1 and AUPRC, and the phase-controlled AUROC the paper's table reports."""
    if metric == "auroc":
        c, key = config["phase_controlled_auroc"], "phase_controlled_auroc"
    else:
        key = {"f1": "metrics.slice_f1", "auprc": "metrics.slice_auprc"}[metric]
        c = config["metrics"][key.split(".")[1]]
    return c["estimate"], (c["low"], c["high"]), key + ".estimate"


def draw_rwm(ax, rwm: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """The legacy forest (another machine: a hatched slot, its slice AUROC in the
    CSV only) and rwm-brf on the Tokamak-SI labels, with a tick at the elapsed-time
    baseline."""
    name = METRICS[metric]
    src = SOURCES["rwm"]
    legacy = rwm["legacy"]
    reason = {
        "f1": "other machine,\nno F1",
        "auroc": "other machine,\nnot comparable",
        "auprc": "other machine,\nno AUPRC",
    }[metric]
    pending_slot(ax, 0, SLOT, PENDING_EDGE, text=reason)
    legacy_key = {"f1": "slice_f1", "auroc": "slice_auroc", "auprc": None}[metric]
    rows.add(
        panel,
        "rwm-brf",
        "legacy",
        name,
        legacy[legacy_key] if legacy_key else None,
        None,
        src,
        f"legacy.{legacy_key}" if legacy_key else "legacy",
        note=f"{legacy['model']}: {legacy['context']}; not drawn",
        drawn=False,
    )
    brf, clock = (rwm["configs"][k] for k in ("rwm-brf", "rwm-rule-elapsed-time"))
    value, ci, key = rwm_cell(brf, metric)
    base, _, base_key = rwm_cell(clock, metric)
    scored_bar(ax, GROUP_GAP, value, ci, SI)
    baseline_tick(ax, GROUP_GAP, base)
    rows.add(
        panel,
        "rwm-brf",
        "Tokamak-SI",
        name,
        value,
        ci,
        src,
        f"configs.rwm-brf.{key}",
        note="100 ms horizon; "
        + ("phase-controlled" if metric == "auroc" else "pooled slices"),
    )
    rows.add(
        panel,
        "rwm-rule-elapsed-time",
        "Tokamak-SI, elapsed-time baseline",
        name,
        base,
        None,
        src,
        f"configs.rwm-rule-elapsed-time.{base_key}",
        note="tick",
    )
    finish_groups(
        ax,
        [0, GROUP_GAP],
        ["", "rwm-brf"],
        [0, GROUP_GAP],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_ylabel(name, labelpad=2)


def saw_presence(saw: dict, model: str) -> tuple[dict, dict]:
    """A model's out-of-fold presence scores and their 95 % shot-bootstrap
    intervals (`crash_tolerance_2ms` of the benchmark record). The presence block
    does not depend on the crash tolerance; the record must say the same at 1 ms."""
    entry = saw["Tokamak-SI"][model]
    if entry["crash_tolerance_1ms"]["presence"] != entry[SAW_BLOCK]["presence"]:
        raise ValueError(f"{model}: presence differs between the crash tolerances")
    return entry[SAW_BLOCK]["presence"], entry[SAW_BLOCK]["ci95"]


def saw_title(saw: dict) -> str:
    """The panel title, with the bin width the record states."""
    return f"{TITLES['sawtooth']} (presence, {saw['protocol']['bin_ms']:g} ms bins)"


def draw_sawtooth(ax, saw: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """The derivative baseline, `saw-hl3` and `saw-ours` on the Tokamak-SI labels
    (presence, out of fold; a tick at the always-present baseline) and, as the
    legacy setting, a hatched slot: the HL-3 paper was scored on another machine."""
    name = METRICS[metric]
    src = SOURCES["saw"]
    reason = {
        "f1": "other machine,\nno F1",
        "auroc": "other machine,\nnot comparable",
        "auprc": "other machine,\nno AUPRC",
    }[metric]
    pending_slot(ax, 0, SLOT, PENDING_EDGE, text=reason)
    legacy = saw["legacy"]
    rows.add(
        panel,
        "saw-hl3",
        "legacy",
        name,
        None,
        None,
        src,
        "legacy",
        note=f"{legacy['source']}: no presence {name}; not drawn",
        drawn=False,
    )
    block = f"Tokamak-SI.{{}}.{SAW_BLOCK}.presence"
    base_presence, _ = saw_presence(saw, SAW_TRIVIAL)
    base = base_presence[metric]
    xs = [GROUP_GAP + i for i in range(len(SAW_MODELS))]
    for x, model in zip(xs, SAW_MODELS):
        presence, ci95 = saw_presence(saw, model)
        value = presence[metric]
        ci = tuple(ci95[f"presence_{metric}"])
        scored_bar(ax, x, value, ci, SI)
        baseline_tick(ax, x, base)
        entry = saw["Tokamak-SI"][model]
        rows.add(
            panel,
            model,
            "Tokamak-SI",
            name,
            value,
            ci,
            src,
            f"{block.format(model)}.{metric}",
            note=f"presence on {saw['protocol']['bin_ms']:g} ms bins, out of fold "
            f"over {entry['coverage']['supported_shots']} supported TRAIN shots "
            f"({entry['coverage']['scored_shots']} with assessed bins; "
            f"{presence['positive_bins']} present and {presence['negative_bins']} "
            "tested-absent bins), scored against the physics rule on assessed bins"
            + (
                "; presence threshold selected per fold on inner-selection shots"
                if metric == "f1"
                else ""
            ),
        )
        if metric == "f1":
            fixed = entry["presence_fixed_threshold"]
            rows.add(
                panel,
                model,
                "Tokamak-SI, fixed threshold",
                name,
                fixed["pooled"]["f1"],
                tuple(fixed["ci95"]["presence_f1"]),
                src,
                f"Tokamak-SI.{model}.presence_fixed_threshold.pooled.f1",
                note=f"the same bins at the single threshold {fixed['threshold']:g}"
                "; kept out of the figure",
                drawn=False,
            )
    rows.add(
        panel,
        SAW_TRIVIAL,
        "Tokamak-SI, trivial baseline",
        name,
        base,
        None,
        src,
        f"{block.format(SAW_TRIVIAL)}.{metric}",
        note="presence = 1 everywhere; tick",
    )
    if metric == "f1":
        hl3 = saw["Tokamak-SI"]["saw-hl3"]
        for mode, key in (("real-time", "real_time"), ("offline", "offline")):
            rows.add(
                panel,
                "saw-hl3",
                "legacy",
                f"three-class window accuracy, {mode} (stated)",
                legacy[key]["accuracy_stated"],
                None,
                src,
                f"legacy.{key}.accuracy_stated",
                note="the HL-3 paper's own metric on its own machine and windows; "
                "not an F1, not comparable; kept out of the figure",
                drawn=False,
            )
        rows.add(
            panel,
            "saw-hl3",
            "Tokamak-SI, three-class",
            "three-class window accuracy",
            hl3["three_class_window_accuracy"],
            tuple(hl3["three_class_accuracy_ci95"]),
            src,
            "Tokamak-SI.saw-hl3.three_class_window_accuracy",
            note="the same three-class windows scored on DIII-D, out of fold; "
            "kept out of the figure",
            drawn=False,
        )
    finish_groups(
        ax,
        [0, *xs],
        ["", *SAW_MODELS],
        [0, float(np.mean(xs))],
        ["legacy", "Tokamak-SI"],
    )
    ax.set_xlim(-0.6, xs[-1] + 0.55)  # a little tighter, so the tearing title fits
    ax.set_ylabel(name, labelpad=2)


def draw_coverage(ax, cov: dict, rows: Rows, panel: str, field: str) -> None:
    """Labelled shots (`field` "shots") or labelled time in hours ("seconds") per
    set, log y: legacy against Tokamak-SI, each Tokamak-SI bar with a darker inner
    bar for its human-reviewed subset; a set with no value in a setting gets a
    hatched slot that says why."""
    unit = "labelled shots" if field == "shots" else "labelled time (h)"
    scale = 1.0 if field == "shots" else 1.0 / S_PER_H
    order = cov["order"]
    value_of = {}
    for key in order:
        for side in ("legacy", "tokamak_si"):
            v = cov["sets"][key][side][field]
            value_of[key, side] = None if v is None else v * scale
    drawn = [v for v in value_of.values() if v is not None]
    for key in order:  # the reviewed subset can be the smallest bar
        reviewed = cov["sets"][key]["tokamak_si"].get("reviewed")
        if reviewed and reviewed["status"] == "measured":
            drawn.append(reviewed[field] * scale)
    lo = 10.0 ** np.floor(np.log10(min(drawn)))
    hi = 10.0 ** np.ceil(np.log10(max(drawn)))
    width = 0.36
    for i, key in enumerate(order):
        entry = cov["sets"][key]
        for sign, side, colour, group in (
            (-1, "legacy", LEGACY, "legacy"),
            (1, "tokamak_si", SI, "Tokamak-SI"),
        ):
            x = i + sign * 0.2
            block = entry[side]
            value = value_of[key, side]
            record_key = f"sets.{key}.{side}.{field}"
            if value is None:
                pending_slot(
                    ax,
                    x,
                    width,
                    PENDING_EDGE,
                    height=hi - lo,
                    bottom=lo,
                    text=STATUS_TEXT[block["status"]],
                    ytext=float(np.sqrt(lo * hi)),
                )
                rows.add(
                    panel,
                    entry["name"],
                    group,
                    unit,
                    None,
                    None,
                    SOURCES["coverage"],
                    record_key,
                    note=STATUS_TEXT[block["status"]],
                )
                continue
            ax.bar(x, value - lo, width, bottom=lo, color=colour, lw=0, zorder=2)
            rows.add(
                panel,
                entry["name"],
                group,
                unit,
                value,
                None,
                SOURCES["coverage"],
                record_key,
                note=block["definition"]
                if block["definition"] != "shared definition"
                else "",
            )
            reviewed = block.get("reviewed")
            if reviewed and reviewed["status"] == "measured":
                inner = reviewed[field] * scale
                ax.bar(
                    x,
                    inner - lo,
                    width / 2,
                    bottom=lo,
                    color=REVIEWED,
                    lw=0,
                    zorder=3,
                )
                rows.add(
                    panel,
                    entry["name"],
                    "Tokamak-SI, reviewed subset",
                    unit,
                    inner,
                    None,
                    SOURCES["coverage"],
                    f"sets.{key}.{side}.reviewed.{field}",
                )
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [cov["sets"][k]["name"] for k in order],
        rotation=40,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=0, pad=2, labelsize=FONT_PT)
    ax.tick_params(axis="y", labelsize=FONT_PT, length=2, pad=1.5)
    ax.grid(axis="y", color="#e4e4e4", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
        ax.spines[side].set_linewidth(0.6)
    ax.set_title(unit, loc="left", fontsize=FONT_PT, pad=4)


def row_axes(fig, fig_h, top_in, height_in, units):
    """One row of panels `height_in` high, its top `top_in` under the figure's top,
    each as wide as its bars need (`units`, in bar spacings)."""
    n = len(units)
    width = PAGE_IN - LEFT_IN - RIGHT_IN - GAP_IN * (n - 1)
    grid = fig.add_gridspec(
        1,
        n,
        left=LEFT_IN / PAGE_IN,
        right=1 - RIGHT_IN / PAGE_IN,
        top=1 - top_in / fig_h,
        bottom=1 - (top_in + height_in) / fig_h,
        wspace=GAP_IN / (width / n),
        width_ratios=units,
    )
    return [fig.add_subplot(grid[0, i]) for i in range(n)]


def legend_handles(reviewed: bool, bound: bool) -> list:
    handles = [
        Patch(color=LEGACY, label="legacy"),
        Patch(color=SI, label="Tokamak-SI"),
    ]
    if reviewed:
        handles.append(Patch(color=REVIEWED, label="human-reviewed subset"))
    if bound:
        handles.append(
            Patch(
                facecolor="none",
                edgecolor=LEGACY,
                hatch="////",
                label="bound for every class, no macro value",
            )
        )
    handles.append(
        Patch(
            facecolor="none",
            edgecolor=PENDING_EDGE,
            hatch="////",
            label="no value (reason in slot)",
        )
    )
    handles += [
        Line2D(
            [],
            [],
            marker="D",
            ms=3.6,
            mfc="white",
            mec=INK,
            mew=0.8,
            ls="none",
            label="uncertain time scored negative",
        ),
        Line2D([], [], color=INK, lw=1.4, label="trivial baseline"),
    ]
    return handles


def save(fig, out: Path, png: Path | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None})
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=150)


def tm_title(tm: dict) -> str:
    """The panel title, with the two bin widths the record states."""
    legacy = {r["legacy"]["bin_ms"] for r in tm["rows"] if r.get("legacy")}
    si = {r["tokamak_si"]["bin_ms"] for r in tm["rows"]}
    if len(legacy) != 1 or len(si) != 1:
        raise ValueError(f"the TM rows differ in bin width: {legacy}, {si}")
    return (
        f"{TITLES['tm']} ({legacy.pop():g} ms legacy, {si.pop():g} ms Tokamak-SI bins)"
    )


def panels(axes, records, rows, metric, tag):
    """The six panels of one metric: AE, confinement, ELM over TM, sawtooth, RWM."""
    ae, conf, si, elm, smith, tm, rwm, saw = records
    top, bottom = axes
    shots = len(ae["shots"]["fair"])
    draw_ae(top[0], ae, rows, f"a{tag}", metric)
    draw_confinement(top[1], conf, si, rows, f"b{tag}", metric)
    draw_elm(top[2], elm, smith, rows, f"c{tag}", metric)
    draw_tm(bottom[0], tm, rows, f"d{tag}", metric)
    draw_sawtooth(bottom[1], saw, rows, f"e{tag}", metric)
    draw_rwm(bottom[2], rwm, rows, f"f{tag}", metric)
    titles = (
        f"Alfvén Eigenmodes ({shots} held-out shots)",
        TITLES["conf"],
        TITLES["elm"],
        tm_title(tm),
        saw_title(saw),
        TITLES["rwm"],
    )
    for ax, text in zip([*top, *bottom], titles):
        ax.set_title(text, loc="left", fontsize=FONT_PT, pad=4)


# bar spacings each panel needs: AE (two groups of three), confinement, ELM (one
# legacy bar, two Tokamak-SI) over tearing modes (two, three), sawtooth (a legacy
# slot, three Tokamak-SI), RWM
TOP_UNITS = (7.1, 3.1, 4.1)
BOTTOM_UNITS = (6.4, 4.8, 3.1)
LEGEND_IN = 0.34


def read_records():
    ae, conf, elm, smith, tm, rwm, saw = (
        load(SOURCES[k])
        for k in ("ae", "confinement", "elm", "elm_smith", "tm", "rwm", "saw")
    )
    si = confinement_si(load(SOURCES["confinement_si"]))
    return ae, conf, si, elm, smith, tm, rwm, saw


@contextmanager
def plot_style() -> Iterator[None]:
    """The paper's text sizes, raised so that nothing is under FONT_PT; make and
    save a figure inside it."""
    with style(), matplotlib.rc_context(TEXT):
        yield


def add_legend(fig, reviewed: bool, bound: bool) -> None:
    fig.legend(
        handles=legend_handles(reviewed, bound),
        loc="upper center",
        ncol=3 if reviewed else 4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.6,
        columnspacing=1.4,
        labelspacing=0.3,
        borderaxespad=0.1,
    )


def figure_f1(records, cov: dict, rows: Rows) -> Figure:
    """The F1 figure: two rows of panels, then the coverage row. Call inside
    `plot_style`."""
    block = TITLE_IN + AXES_IN + NAMES_IN
    fig_h = LEGEND_IN + 2 * block + TITLE_IN + COVER_IN + COVER_NAMES_IN + 0.04
    fig = Figure(figsize=(PAGE_IN, fig_h))
    top = LEGEND_IN
    axes = []
    for units in (TOP_UNITS, BOTTOM_UNITS):
        axes.append(row_axes(fig, fig_h, top + TITLE_IN, AXES_IN, units))
        top += block
    panels(axes, records, rows, "f1", "")
    cover = row_axes(fig, fig_h, top + TITLE_IN, COVER_IN, (1, 1))
    draw_coverage(cover[0], cov, rows, "g", "shots")
    draw_coverage(cover[1], cov, rows, "h", "seconds")
    add_legend(fig, reviewed=True, bound=False)
    return fig


def figure_auc(records, rows: Rows) -> Figure:
    """AUROC then AUPRC, each as two rows of panels (no coverage row). Call
    inside `plot_style`."""
    block = TITLE_IN + AXES_IN + NAMES_IN
    legend_in = LEGEND_IN + 0.1
    fig_h = legend_in + 4 * block + 0.04
    fig = Figure(figsize=(PAGE_IN, fig_h))
    top = legend_in
    for metric, tag in (("auroc", "0"), ("auprc", "1")):
        axes = []
        for units in (TOP_UNITS, BOTTOM_UNITS):
            axes.append(row_axes(fig, fig_h, top + TITLE_IN, AXES_IN, units))
            top += block
        panels(axes, records, rows, metric, tag)
    add_legend(fig, reviewed=False, bound=True)
    return fig


def draw(out: Path, png: Path | None, table: Path | None) -> None:
    """The F1 figure, with the coverage row, and the numbers behind it."""
    rows = Rows()
    with plot_style():
        fig = figure_f1(read_records(), load(SOURCES["coverage"]), rows)
        save(fig, out, png)
    rows.write(table)
    gits = {k: load(p).get("git") for k, p in SOURCES.items()}
    print(
        f"wrote {out}; {PAGE_IN} x {fig.get_figheight():.2f} in; sources at git {gits}",
        file=sys.stderr,
    )


def draw_auc(out: Path, png: Path | None, table: Path | None) -> None:
    """The AUROC and AUPRC figure, and the numbers behind it."""
    rows = Rows()
    with plot_style():
        fig = figure_auc(read_records(), rows)
        save(fig, out, png)
    rows.write(table)
    print(f"wrote {out}; {PAGE_IN} x {fig.get_figheight():.2f} in", file=sys.stderr)


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
