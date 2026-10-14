r"""fig_benchmarks: Figure 2 of the paper. Every label set, original against
Tokamak-SI: the F1 of each model "<task>-<model>" (two rows of panels, one x tick
per model, the original and the Tokamak-SI score as a pair of bars on it), and
below them how much labelled data each setting holds (the coverage panel).

    PYTHONPATH=src pixi run --frozen -e labeler \
        python scripts/labeler/paper/fig_benchmarks.py \
        [--out dev/label_paper/figures/fig_benchmarks.pdf] [--png PATH] [--table PATH]

Every score drawn is read from a committed record (`SOURCES`), with the 95 %
shot-bootstrap interval stored beside it where there is one; none is typed in
here. Per set, "legacy" (drawn as "original") is the setting the model's own
paper (or the older annotation) used and "Tokamak-SI" is our labels. An original
bar whose value comes from another dataset (the earlier paper's own data, or our
replication on it) has REPORTED written along it in white, no interval and no
baseline; every other bar has its own trivial baseline, a short dashed black
segment across just that bar, computed on the bins or slices the bar is scored on
(the F1 of calling everything positive, 2p / (1 + p) at prevalence p; AUROC 0.5;
AUPRC = p), and `--table` has one row per bar, series "trivial baseline":

- AE: `ae-rcn` and `ae-lstm` (chord spectrogram, each chord scored) and
  `ae-ours` (the SELDnet model), in that order, on the 19 held-out shots, 10 ms
  frames; legacy = against the Heidbrink annotation (`results.annotated`), Tokamak-SI =
  against the dense labels (`results.reviewed`).
- Confinement: `confine-cnn`; legacy = the earlier paper's macro F1, the mean
  of its four per-class F1 (`paper` in `gill_2024_published.json`, its results
  table, committed beside the other records), one bar, REPORTED, no interval;
  Tokamak-SI = our reimplementation under the paper's selection and split, the
  ablation's protocol row `full_cum_abcdrgef` (142 distinct test shots in 200
  shot-tests from five random by-shot splits whose test sets overlap, windows
  pooled; macro F1, AUROC and AUPRC with shot-bootstrap intervals), the one
  confine-cnn score of the paper. The first retrain's own score is not drawn.
  Its baseline: the always-majority predictor's macro F1 (the majority class's
  F1 over the four classes), from the classes' supports on the same windows.
  `confine-ours` (Tokamak-SI only) is scored on exactly those windows, labels and
  bootstrap (`confinement_ours.py protocol`; each shot predicted by networks that
  never trained on its run day); its record must hold confine-cnn's own score on
  the same windows, and the paired difference of the two is in the CSV only.
- ELMs: `elm-elmo` and `elm-ours` from the ELM stream's final records. Legacy =
  D. Smith's windows at the published setting (`reimplemented_elmo_overlap` of
  `smith/evaluation.json`: overlap-region F1, conditional on the selected
  windows; the F1 is checked against the one the stored counts give; REPORTED,
  its interval is in the CSV only). Tokamak-SI
  = the reviewed spans in 50 ms bins on the 73 shots with BES, the set both
  models are scored on (`sets.bes73` of `ours/evaluation.json`, baseline its
  `always present` method); `elm-ours` has no legacy bar, and its score on all
  119 reviewed shots is in the CSV only.
- Tearing modes: `tm-cnn` (the records' `onsetcnn`) and `tm-dsm` (published model at its published
  threshold on the Seo cohort, 25 ms bins, against the retrained twin on the
  Tokamak-SI labels, 10 ms bins) and `tm-ours` (Tokamak-SI only), from
  `docs/labeler/figure2_tm.json`. Both legacy and Tokamak-SI bars are scored by us
  on DIII-D shots, so none is REPORTED; each baseline is its own bins'
  `prevalence`. The Tokamak-SI F1 excludes uncertain time; the same model with
  uncertain time scored negative is in the CSV only.
- RWM: presence classifiers at t (owner, 2026-10-14), from
  `docs/labeler/results/rwm_classifier.json` (`rwm_classify.py`): `rwm-brf`, the
  NSTX balanced-forest recipe on the round-four (N1RMS-based) inputs retrained on
  the Tokamak-SI labels, and `rwm-ours`, the same forest on the v5 inputs with the
  n = 1 delta-Bp, alone on its tick. Bin-level F1, AUROC and AUPRC on 10 ms bins
  of Hanson's shots, one calendar run day held out at a time, with 95 %
  whole-shot bootstrap intervals; one baseline, the prevalence of the bins both
  are scored on (`counts.prevalence`, checked against its bin counts and each
  model's). The paired difference and the uncertain-scored-negative view are in
  the CSV only. The NSTX forest was a forecast on another machine and reported a
  slice AUROC only: a REPORTED bar beside `rwm-brf` in the AUROC figure, nothing
  in the F1 and AUPRC figures.
- Sawtooth: `saw-hl3` (record `saw-hl3-full`: the OuYang 2025 HL-3 network,
  replicated, on the paper's nine offline inputs; the earlier four-input run,
  `saw-hl3-ece`, is in the CSV only)
  and `saw-ours` (v2: the ECE network with the derivative picker's edge score as
  input and gate, which reads the same ECE edges the labels are built from)
  on the Tokamak-SI labels, from the sawtooth stream's benchmark record
  (`outputs/labeler/sawtooth/fix6/benchmark.json`): presence F1 on 2 ms bins,
  out of fold over the shot-grouped three-fold split of the TRAIN shots (the
  record's `crash_tolerance_2ms.presence`; the presence threshold is selected
  per fold on inner-selection shots), with the record's 95 % shot-bootstrap
  interval and the always-present baseline of the record on the same bins (checked
  to be the bins of every model). Once the record
  holds the no-q=1-negatives view (`no_q1_negatives`, time with no q=1 surface
  scored as tested-absent) for every model, the panel draws that view instead
  (`saw_view`). The score is agreement with the physics rule on assessed bins,
  not an independent truth. HL-3's own paper was scored on another machine: its
  stated offline three-class window accuracy (`legacy.offline.accuracy_stated`,
  not an F1) is drawn as a REPORTED bar beside `saw-hl3-full` in the F1 figure.
- Detachment: `detach-ours` (windows of 0-D signals, `detach_ours.py`), one
  Tokamak-SI bar on its own tick, as no earlier detector or label set exists:
  macro F1 from 5-fold shot-grouped cross-validation on the detachment labels
  (`cv_shots.macro_f1` of `docs/labeler/results/detachment_ours.json`), with its
  shot-bootstrap interval and the always-majority baseline on the same bins
  (owner, 2026-10-14: the panel shows a model named and scored like the others).
  No AUROC or AUPRC is stored for it. The consistency check of the TangTV vote
  against divertor Thomson Te (`detachment_check`: agreement, kappa, and the
  AUROC of -Te before the L-mode gate) is in the CSV only.

The coverage panel reads `docs/labeler/figure2_coverage.json` (written by
`fig2_coverage.py`): labelled hours per set (bars, log y), original against
Tokamak-SI, with a darker inner bar for the human-reviewed subset and the
labelled shot count written on each bar.

`--metric auc` draws the same sets for AUROC and AUPRC (no coverage panel) into
`fig_benchmarks_auc.pdf`: an original score only where its paper reported one
(confine-cnn's and elm-elmo's AUPRC, elm-elmo's AUROC are notes; the NSTX
forest's slice AUROC, a forecast on another machine: a REPORTED bar). The
confine-cnn paper reported a one-vs-rest AUC of at least 0.99 for every class and
no macro value, so its original AUROC is a REPORTED bar at 0.99 written as that
bound (≥ .99), not a macro score (`auc_one_vs_rest` in `gill_2024_published.json`).
elm-elmo's AUROC and AUPRC come from a sweep of its threshold.

A value not available is a short grey note where its bar would be. A score is
written above its bar without the leading zero (.836): a pair's two values at
7 pt are only that narrow enough to leave 2 pt between neighbouring pairs.
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
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator
from matplotlib.transforms import offset_copy

from labeler.paper import FONT_PT, PAGE_IN, style, theme

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
    # confine-ours on that row's windows (confinement_ours.py protocol)
    "confinement_ours": OUTPUTS / "confinement" / "ours" / "protocol_base3.json",
    # the ELM stream's final records, the ones its paper tables read
    "elm": OUTPUTS / "elm" / "ours" / "evaluation.json",
    "elm_smith": OUTPUTS / "elm" / "smith" / "evaluation.json",
    "tm": REPO / "docs" / "labeler" / "figure2_tm.json",
    # rwm-brf and rwm-ours, presence per 10 ms bin (rwm_classify.py)
    "rwm": REPO / "docs" / "labeler" / "results" / "rwm_classifier.json",
    # the sawtooth stream's benchmark: out-of-fold presence scores per model
    "saw": OUTPUTS / "sawtooth" / "fix6" / "benchmark.json",
    "coverage": REPO / "docs" / "labeler" / "figure2_coverage.json",
    # detach-ours, the panel's bar: 5-fold shot-grouped CV on the detachment labels
    "detach_ours": REPO / "docs" / "labeler" / "results" / "detachment_ours.json",
    # the detachment consistency check, CSV only: AUROC of -Te against the TangTV
    # vote (with its shot-bootstrap interval), and the vote's agreement with Te's state
    "detach_te": REPO / "docs" / "labeler" / "results" / "detachment_te_check.json",
    "detach_source": REPO
    / "docs"
    / "labeler"
    / "results"
    / "detachment_source_v3.json",
}
DEFAULT_OUT = REPO / "dev" / "label_paper" / "figures" / "fig_benchmarks.pdf"
DEFAULT_AUC_OUT = DEFAULT_OUT.with_name("fig_benchmarks_auc.pdf")

# the paper theme (`labeler.paper.theme`): Tokamak-SI sea green, the original label
# sets coral; the reviewed subset is a darker shade of Tokamak-SI.
LEGACY = theme.CORAL
SI = theme.SEA_GREEN
REVIEWED = theme.SEA_GREEN_DARK
INK = "#333333"
NOTE = "#6b6b6b"  # the grey of a note that stands in for a missing value
REPORTED = "REPORTED"  # written along an original bar reported on another dataset
BASELINE = "trivial baseline"  # the legend's entry, and the series of its table rows
BASELINE_DASH = (0, (4, 2))
CHANCE_AUROC = 0.5  # the AUROC of a constant score, by definition
REPORTED_LIFT = 0.03  # a REPORTED bar's value stands this much higher (data units)
# every text in the figure is at least FONT_PT (7 pt)
TEXT = {
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
}
VALUE_PT = FONT_PT
# a pair's two values start this far outside their bars' inner edges, so that
# they keep 2 pt apart with their white boxes
VALUE_GAP_PT = 1.0
# x layout, in tick spacings (one tick per model)
HALF = 0.22  # a pair's two bars sit this far either side of the model's tick
BAR = 0.4  # the width of every bar
MARGIN = 0.6  # a panel's room beyond its outer ticks
YMAX = 1.2
YTICKS = np.arange(0.0, 1.01, 0.25)
S_PER_H = 3600.0
# layout in inches
LEFT_IN, RIGHT_IN, GAP_IN = 0.45, 0.06, 0.42
TITLE_IN, NAMES_IN = 0.2, 0.34
AXES_IN = 0.82  # height of a score panel
COVER_IN, COVER_NAMES_IN = 0.85, 0.2  # the coverage panel, and its set names
COVER_TICKS_IN = 0.08  # its y tick labels (0.001) are wider than the scores'
LEGEND_IN = 0.34
HEADING_IN = 0.22  # the AUC figure's heading above each metric's block of panels
#: The AUC figure's two blocks repeat the panel titles: a heading names each.
AUC_HEADINGS = {
    "auroc": "AUROC: area under the ROC curve",
    "auprc": "AUPRC: area under the precision–recall curve",
}

AE_MODELS = (  # (key in the evaluation, "<task>-<model>"); ours last, as every panel's
    ("rcn_spec_each", "ae-rcn"),
    ("lstm_spec_each", "ae-lstm"),
    ("seldnet", "ae-ours"),
)
AE_SETTINGS = (  # (setting, truth block, what it is scored against, colour)
    ("legacy", "annotated", "Heidbrink annotation", LEGACY),
    ("Tokamak-SI", "reviewed", "dense labels", SI),
)
TM_MODELS = (  # (architecture in the record, "<task>-<model>")
    ("onsetcnn", "tm-cnn"),  # Seo et al.'s onset CNN; the record keeps "onsetcnn"
    ("dsm", "tm-dsm"),
    ("magnetic-detector", "tm-ours"),
)
# the record's key for each metric: (legacy, Tokamak-SI)
TM_KEYS = {
    "f1": ("f1_published_threshold", "f1"),
    "auroc": ("auroc", "auroc"),
    "auprc": ("auprc", "auprc"),
}
# legacy TM checkpoints with no published operating point: their F1 bar is drawn at
# the inner-validation threshold (the DSM's 500 ms risk peaks at 0.279 on these bins
# and is capped at 0.323, so the upstream helper's default 0.3 can never fire)
TM_TUNED_LEGACY = ("dsm",)
SAW_HL3 = "saw-hl3-full"  # the HL-3 bar: the paper's full offline inputs
SAW_HL3_ECE = "saw-hl3-ece"  # the earlier four-input run: in the CSV only
SAW_TICKS = {SAW_HL3: "saw-hl3"}  # the tick's name where it is not the record's
SAW_MODELS = (  # the models of the record, named "<task>-<model>" there
    SAW_HL3,
    "saw-ours",
)
SAW_TRIVIAL = "saw-always-present"
SAW_BLOCK = "crash_tolerance_2ms"  # holds the presence scores (the same at 1 ms)
# the record's no-q=1-negatives view (time with no q = 1 surface, q_min above the
# gate's cut, added as negatives): drawn instead of the primary scores once every
# model of the record has it
SAW_VIEW = "no_q1_negatives"
CLASSES = ("L", "H", "QH", "WP")
METRICS = {"f1": "F1", "auroc": "AUROC", "auprc": "AUPRC"}
SHOWN = {"legacy": "original"}  # the paper calls the earlier label sets "original"
STATUS_TEXT = {
    "none": "no original",
    "pending": "pending",
    "point_events": "point events",
}
TITLES = {
    "ae": "Alfvén Eigenmodes",
    "detach": "Detachment",
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
    inch apart. A name with two hyphens (`saw-hl3-full`) breaks at the last one, so
    its second line stays as narrow as its neighbours'."""
    split = name.rpartition if name.count("-") > 1 else name.partition
    head, sep, tail = split("-")
    return f"{head}-\n{tail}" if sep else name


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


def score_text(value: float, digits: int = 3) -> str:
    """A score as written above its bar: `digits` decimals, with no leading zero
    below 1 (".836"), the width at which neighbouring pairs keep apart."""
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("0.") else text


def value_text(ax, x, y, text, side=0, lift=0.0) -> None:
    """A bar's value, written above `y` (and `lift` higher, in data units). `side` 0
    centres it on the bar; -1 (the left bar of a pair) ends it `VALUE_GAP_PT` left
    of the bar's right edge and 1 (the right bar) starts it as far right of the
    bar's left edge, so a pair's two values spread outward from the tick and never
    meet."""
    ax.text(
        x - side * BAR / 2,
        y + 0.015 + lift,
        text,
        transform=offset_copy(
            ax.transData, fig=ax.figure, x=side * VALUE_GAP_PT, units="points"
        ),
        ha={0: "center", -1: "right", 1: "left"}[side],
        va="bottom",
        fontsize=VALUE_PT,
        color=INK,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
    )


def baseline_segment(ax, x, value) -> None:
    """A bar's trivial baseline: a short black dashed segment across just that
    bar's width, over the bar."""
    ax.plot(
        [x - BAR / 2, x + BAR / 2],
        [value, value],
        color="black",
        lw=0.7,
        ls=BASELINE_DASH,
        zorder=4,
    )


def scored_bar(ax, x, value, ci, colour, digits=3, side=0, base=None) -> None:
    """A bar with its interval, if it has one, its trivial baseline `base`, if it
    has one, and its value written above them both."""
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
    if base is not None:
        baseline_segment(ax, x, base)
        top = max(top, base)  # the value clears the dash
    value_text(ax, x, top, score_text(value, digits), side)


def reported_bar(ax, x, value, digits=3, side=0, bound=False) -> None:
    """An original bar whose value was reported on another dataset (the earlier
    paper's own data, or our replication on it): REPORTED written along it in
    white, no interval and no baseline. A `bound` is a published lower bound, and
    is written as one."""
    ax.bar(x, value, BAR, color=LEGACY, edgecolor="none", zorder=2)
    ax.text(
        x,
        value / 2,
        REPORTED,
        rotation=90,
        ha="center",
        va="center",
        fontsize=VALUE_PT,
        fontfamily=matplotlib.rcParams["font.family"],  # the labels' own face
        fontweight="bold",
        color="white",
        zorder=5,
    )
    text = f"≥ {score_text(value, 2)}" if bound else score_text(value, digits)
    # REPORTED nearly fills a bar of 0.9: lift the value off the end of the word
    value_text(ax, x, value, text, side, lift=REPORTED_LIFT)


def note(ax, x, text, y=0.03) -> None:
    """A short grey note where a bar would be: why there is no value."""
    ax.text(
        x,
        y,
        text,
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=VALUE_PT,
        linespacing=1.1,
        color=NOTE,
        zorder=4,
    )


def finish(ax, names, ylabel, margin=MARGIN) -> None:
    """One tick per model, `names` in order from 0."""
    ax.set_xticks(range(len(names)), [two_line(n) for n in names])
    ax.tick_params(axis="x", labelsize=FONT_PT)
    ax.set_xlim(-margin, len(names) - 1 + margin)
    axes_style(ax, ylabel)


def always_f1(prevalence: float) -> float:
    """The F1 of calling every bin positive: 2p / (1 + p) at prevalence p."""
    return 2 * prevalence / (1 + prevalence)


def baseline_row(
    rows, panel, series, group, metric, value, source, key, detail=""
) -> None:
    """A bar's trivial baseline in the table: one row per bar, series "trivial
    baseline", the bar's own model and setting as its group."""
    rows.add(
        panel,
        BASELINE,
        f"{series}, {group}",
        metric,
        value,
        None,
        source,
        key,
        note=detail,
    )


def cell(c: dict) -> tuple[float, tuple | None]:
    """A record's `{value, lo, hi}` cell: its value and its 95 % interval (none when
    the interval is a point)."""
    lo, hi = c["lo"], c["hi"]
    return c["value"], ((lo, hi) if hi > lo else None)


def draw_ae(ax, ae: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """ae-rcn, ae-lstm and ae-ours, each against the legacy truth and the
    Tokamak-SI truth: a pair of bars per model, each with the baseline of the
    always-present predictor on its own truth (`results.<truth>.always`)."""
    for i, (key, name) in enumerate(AE_MODELS):
        for side, (setting, truth, truth_name, colour) in zip((-1, 1), AE_SETTINGS):
            entry = ae["results"][truth][key]
            ci = entry["ci95"][metric]
            base = ae["results"][truth]["always"][metric]
            x = i + side * HALF
            scored_bar(ax, x, entry[metric], ci, colour, side=side, base=base)
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
            baseline_row(
                rows,
                panel,
                name,
                setting,
                METRICS[metric],
                base,
                SOURCES["ae"],
                f"results.{truth}.always.{metric}",
                f"always present, on the {truth_name} frames",
            )
    finish(ax, [name for _, name in AE_MODELS], METRICS[metric])


def confinement_si(record: dict) -> dict:
    """The protocol row's record (`ablation_rows/full_cum_abcdrgef.json`) as the
    macro scores and intervals `draw_confinement` reads: macro F1 from the row's
    own windows, macro AUROC and AUPRC from its `ranking`, each with the 95 % shot
    bootstrap interval stored beside it. Its trivial baselines come from the
    classes' supports on the same windows: the always-majority predictor's macro F1
    (its class's F1 over the number of classes, the others scoring 0) and the macro
    AUPRC of a constant score (the classes' mean prevalence)."""
    own, rank = record["own_population"], record["ranking"]
    support = [c["support"] for c in own["classes"].values()]
    total = sum(support)
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
        "base": {
            "f1": always_f1(max(support) / total) / len(support),
            "auroc": CHANCE_AUROC,
            "auprc": sum(n / total for n in support) / len(support),
        },
        "base_keys": {
            "f1": "own_population.classes.*.support: majority class F1 / classes",
            "auroc": "chance level by definition",
            "auprc": "own_population.classes.*.support: mean class prevalence",
        },
    }


def confinement_ours(record: dict, row: dict) -> dict:
    """confine-ours on confine-cnn's protocol (`confinement_ours.py protocol`) as
    `confinement_si` reads the row: its macro scores, intervals and baselines from
    the record's `confine-ours` entry, and the paired difference ours minus the CNN
    (`ours_minus_cnn`) with its interval. The record's own `confine-cnn` entry must
    be the protocol row's score on the same windows (windows, shots and macro F1),
    or the two bars would not share them: it is refused otherwise."""
    cnn, own = record["confine-cnn"], row["own_population"]
    same = (cnn["windows"], cnn["shots"]) == (own["windows"], own["shots"])
    if not same or abs(cnn["macro_f1"] - own["macro_f1"]) > 1e-9:
        raise ValueError(
            f"confine-ours's record scores confine-cnn {cnn['macro_f1']} on "
            f"{cnn['windows']} windows of {cnn['shots']} shots, the protocol row "
            f"{own['macro_f1']} on {own['windows']} of {own['shots']}"
        )
    entry = record["confine-ours"]
    out = confinement_si({"own_population": entry, "ranking": entry["ranking"]})
    out["keys"] = {
        "f1": "confine-ours.macro_f1",
        "auroc": "confine-ours.ranking.auroc.macro",
        "auprc": "confine-ours.ranking.auprc.macro",
    }
    out["base_keys"] = {
        k: v.replace("own_population.", "confine-ours.")
        for k, v in out["base_keys"].items()
    }
    diff = record["ours_minus_cnn"]
    out["difference"] = {
        metric: (diff[key]["difference"], tuple(diff[key]["ci95"]))
        for metric, key in (("f1", "macro_f1"), ("auroc", "auroc"), ("auprc", "auprc"))
    }
    return out


def draw_confinement(
    ax, conf: dict, si: dict, rows: Rows, panel: str, metric: str = "f1"
) -> None:
    """confine-cnn's macro score: the earlier paper's (legacy, from `conf`: macro
    F1, REPORTED; for AUROC only its bound of 0.99 per class, one-vs-rest, no macro
    value, REPORTED as a bound; no AUPRC) and our reimplementation under its
    selection and split (Tokamak-SI, from `si`, see `confinement_si`; with its
    baseline); then, when `si` holds it (`si["ours"]`, see `confinement_ours`),
    confine-ours on the same windows, one Tokamak-SI bar on its own tick, and the
    paired difference in the CSV only."""
    name = "macro " + METRICS[metric]
    if metric == "f1":
        published = conf["paper"]
        paper_macro = float(np.mean([published[k]["f1"] for k in CLASSES]))
        reported_bar(ax, -HALF, paper_macro, digits=2, side=-1)
        rows.add(
            panel,
            "confine-cnn",
            "legacy",
            name,
            paper_macro,
            None,
            SOURCES["confinement"],
            "mean of paper.{L,H,QH,WP}.f1",
            note="REPORTED: the earlier paper's own score on its data",
        )
    elif metric == "auroc":
        bound = conf["auc_one_vs_rest"]["per_class_at_least"]
        reported_bar(ax, -HALF, bound, digits=2, side=-1, bound=True)
        rows.add(
            panel,
            "confine-cnn",
            "legacy",
            "per-class AUROC, lower bound (one-vs-rest)",
            bound,
            None,
            SOURCES["confinement"],
            "auc_one_vs_rest.per_class_at_least",
            note="REPORTED: the earlier paper's bound on its data",
        )
    else:
        note(ax, -HALF, "not reported")
    value = si["macro"][metric]
    ci = si["ci95"][f"macro_{metric}"]
    scored_bar(ax, HALF, value, ci, SI, side=1, base=si["base"][metric])
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
    baseline_row(
        rows,
        panel,
        "confine-cnn",
        "Tokamak-SI",
        name,
        si["base"][metric],
        SOURCES["confinement_si"],
        si["base_keys"][metric],
        "a constant predictor on the same windows",
    )
    ours = si.get("ours")
    if ours is None:
        finish(ax, ["confine-cnn"], METRICS[metric], margin=2 * HALF + BAR)  # "≥ 0.99"
        return
    value = ours["macro"][metric]
    ci = ours["ci95"][f"macro_{metric}"]
    scored_bar(ax, 1, value, ci, SI, base=ours["base"][metric])
    rows.add(
        panel,
        "confine-ours",
        "Tokamak-SI",
        name,
        value,
        ci,
        SOURCES["confinement_ours"],
        ours["keys"][metric],
        note="confine-cnn's windows, labels and shot bootstrap; each shot "
        "predicted by networks that never trained on its run day",
    )
    baseline_row(
        rows,
        panel,
        "confine-ours",
        "Tokamak-SI",
        name,
        ours["base"][metric],
        SOURCES["confinement_ours"],
        ours["base_keys"][metric],
        "a constant predictor on the same windows",
    )
    diff, diff_ci = ours["difference"][metric]
    rows.add(
        panel,
        "confine-ours minus confine-cnn",
        "Tokamak-SI",
        name,
        diff,
        diff_ci,
        SOURCES["confinement_ours"],
        f"ours_minus_cnn.{'macro_f1' if metric == 'f1' else metric}.difference",
        note="paired shot bootstrap: the same shots drawn for both models",
        drawn=False,
    )
    finish(ax, ["confine-cnn", "confine-ours"], METRICS[metric])  # axis: no "macro"
    ax.set_xlim(-(2 * HALF + BAR), 1 + MARGIN)  # room for "≥ 0.99" on the left


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
    """elm-elmo on D. Smith's windows (legacy, REPORTED: another dataset; F1 only,
    there is no threshold sweep there) and on the reviewed spans, and elm-ours on the
    reviewed spans (Tokamak-SI, the shots with BES that both models are scored on;
    AUROC and AUPRC come from a sweep of the detection threshold), each Tokamak-SI
    bar with the record's always-present predictor as its baseline. elm-elmo is a
    pair of bars, elm-ours one."""
    name = METRICS[metric]
    src = SOURCES["elm"]
    models = (("elm-elmo", "elm-elmo"), ("elm-ours", "elm-ours"))
    if metric == "f1":
        value, ci = smith_legacy(smith)
        reported_bar(ax, -HALF, value, side=-1)
        rows.add(
            panel,
            "elm-elmo",
            "legacy",
            name,
            value,
            ci,
            SOURCES["elm_smith"],
            "reimplemented_elmo_overlap.point.f1",
            note="REPORTED: D. Smith's windows at the published setting; "
            "overlap-region F1, conditional on the selected windows; "
            "interval not drawn",
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
        note(ax, -HALF, "not reported")
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
    # elm-elmo's Tokamak-SI bar beside its legacy one; elm-ours alone on its tick
    places = ((HALF, 1), (1, 0))
    for (x, side), (key, label) in zip(places, models):
        res = bes["methods"][key]
        if metric == "f1":
            value, ci = res["point"]["f1"], res["ci95"]["f1"]
            where = f"sets.bes73.methods.{key}.point.f1"
        else:
            value, ci = res["point"][metric], res["ci95"][metric]
            where = f"sets.bes73.methods.{key}.point.{metric}"
        base = bes["methods"]["always present"]["point"][metric]
        scored_bar(ax, x, value, ci, SI, side=side, base=base)
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
        baseline_row(
            rows,
            panel,
            label,
            "Tokamak-SI",
            name,
            base,
            src,
            f"sets.bes73.methods.always present.point.{metric}",
            "the always-present predictor on the same bins",
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
    finish(ax, [label for _, label in models], name)


def prevalence_base(metric: str, prevalence: float) -> float:
    """The score of a predictor that carries no information, from the prevalence
    of the bins a bar is scored on: always-present F1, the chance AUROC, and the
    prevalence itself as the AUPRC."""
    return {
        "f1": always_f1(prevalence),
        "auroc": CHANCE_AUROC,
        "auprc": prevalence,
    }[metric]


def prevalence_key(metric: str, key: str) -> str:
    """Where in the table a baseline made by `prevalence_base` comes from."""
    return {
        "f1": f"2p/(1+p) at {key}",
        "auroc": "chance level by definition",
        "auprc": key,
    }[metric]


def draw_tm(ax, tm: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """tm-cnn and tm-dsm as the published model on its own labels (legacy, at
    its published threshold, scored by us on the Seo cohort) beside the retrained
    twin on the Tokamak-SI labels, a pair of bars per model, and tm-ours on
    Tokamak-SI only; each bar has the baseline of its own bins' prevalence. The
    published F1 is at the model's published threshold, but the DSM checkpoint has
    none (`TM_TUNED_LEGACY`): its bar is the inner-validation threshold's. The
    same models with uncertain time scored negative are in the table only."""
    legacy_key, si_key = TM_KEYS[metric]
    by = {r["architecture"]: r for r in tm["rows"]}
    src = SOURCES["tm"]
    name = METRICS[metric]
    for i, (arch, label) in enumerate(TM_MODELS):
        row = by[arch]
        block = row.get("legacy")
        if block:
            tuned = metric == "f1" and arch in TM_TUNED_LEGACY
            key = "f1_tuned" if tuned else legacy_key
            value, ci = cell(block[key])
            prevalence = block["prevalence"]["value"]
            base = prevalence_base(metric, prevalence)
            scored_bar(ax, i - HALF, value, ci, LEGACY, side=-1, base=base)
            rows.add(
                panel,
                label,
                "legacy",
                name,
                value,
                ci,
                src,
                f"rows[{arch}].legacy.{key}",
                note=f"{block['model']}, "
                + (
                    "no published operating point for this checkpoint; "
                    "inner-validation threshold, "
                    if tuned
                    else "published threshold, "
                )
                + f"{block['bin_ms']:g} ms bins, {block['shots']} shots",
            )
            baseline_row(
                rows,
                panel,
                label,
                "legacy",
                name,
                base,
                src,
                prevalence_key(metric, f"rows[{arch}].legacy.prevalence"),
                f"prevalence {prevalence:.4f} of the bins it is scored on",
            )
            if metric == "f1":
                other, other_key = (
                    ("legacy, published threshold", "f1_published_threshold")
                    if tuned
                    else ("legacy, tuned threshold", "f1_tuned")
                )
                other_value, other_ci = cell(block[other_key])
                rows.add(
                    panel,
                    label,
                    other,
                    name,
                    other_value,
                    other_ci,
                    src,
                    f"rows[{arch}].legacy.{other_key}",
                    note=(
                        "risk >= 0.3 is an upstream helper default; 500 ms risk "
                        "peaks at 0.279 on these bins (model cap 0.323)"
                        if tuned
                        else "secondary; kept out of the figure"
                    ),
                    drawn=False,
                )
        x, side = (i + HALF, 1) if block else (i, 0)
        value, ci = cell(row["tokamak_si"][si_key])
        prevalence = row["tokamak_si"]["prevalence"]["value"]
        base = prevalence_base(metric, prevalence)
        scored_bar(ax, x, value, ci, SI, side=side, base=base)
        unc, _ = cell(row["tokamak_si_uncertain_negative"][si_key])
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
        baseline_row(
            rows,
            panel,
            label,
            "Tokamak-SI",
            name,
            base,
            src,
            prevalence_key(metric, f"rows[{arch}].tokamak_si.prevalence"),
            f"prevalence {prevalence:.4f} of the bins it is scored on",
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
            note="kept out of the figure",
            drawn=False,
        )
    finish(ax, [label for _, label in TM_MODELS], name)


def rwm_cell(model: dict, metric: str) -> tuple[float, tuple, str]:
    """A presence classifier's bin-level `metric` (F1 at its folds' cuts, AUROC,
    AUPRC): its value, 95 % whole-shot interval and key under the model."""
    c = model["metrics"][metric]
    return c["estimate"], (c["low"], c["high"]), f"metrics.{metric}.estimate"


def rwm_prevalence(record: dict) -> float:
    """The prevalence of the bins both classifiers are scored on (`counts.
    prevalence`, which must be its own present and negative bin counts, and every
    model's counts must be the record's)."""
    counts = record["counts"]
    own = counts["positive_bins"] / (counts["positive_bins"] + counts["negative_bins"])
    if abs(counts["prevalence"] - own) > 1e-9:
        raise ValueError(
            f"RWM prevalence {counts['prevalence']} is not its counts' {own}"
        )
    for name, model in record["models"].items():
        if model["counts"] != counts:
            raise ValueError(f"{name} is scored on other bins than the record's")
    return counts["prevalence"]


def draw_rwm(ax, rwm: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """The presence classifiers on the Tokamak-SI labels: `rwm-brf` (the NSTX
    forest's recipe on the round-four inputs) and `rwm-ours` (delta-Bp), each with
    the baseline of the bins' prevalence; for AUROC the NSTX forest's reported
    slice AUROC beside rwm-brf (REPORTED: a forecast on another machine; it has
    no F1 and no AUPRC). The paired difference and the uncertain-scored-negative
    view are in the table only."""
    name = METRICS[metric]
    src = SOURCES["rwm"]
    legacy = rwm["legacy"]
    prevalence = rwm_prevalence(rwm)
    base = prevalence_base(metric, prevalence)
    counts = rwm["counts"]
    reported = metric == "auroc"  # the NSTX forest reported a slice AUROC only
    if reported:
        reported_bar(ax, -HALF, legacy["slice_auroc"], side=-1)
    rows.add(
        panel,
        "rwm-brf",
        "legacy",
        name,
        legacy["slice_auroc"] if reported else None,
        None,
        src,
        "legacy.slice_auroc" if reported else "legacy",
        note=(
            f"REPORTED: {legacy['model']}, {legacy['task']}"
            if reported
            else f"{legacy['model']}: {legacy['task']}; no {name}; not drawn"
        ),
        drawn=reported,
    )
    for tick, model in enumerate(("rwm-brf", "rwm-ours")):
        entry = rwm["models"][model]
        value, ci, key = rwm_cell(entry, metric)
        x, side = (HALF, 1) if reported and tick == 0 else (tick, 0)
        scored_bar(ax, x, value, ci, SI, side=side, base=base)
        rows.add(
            panel,
            model,
            "Tokamak-SI",
            name,
            value,
            ci,
            src,
            f"models.{model}.{key}",
            note=f"presence per 10 ms bin, {counts['shots']} Hanson shots, one "
            "calendar run day held out at a time"
            + ("; cut chosen on the training days" if metric == "f1" else ""),
        )
        baseline_row(
            rows,
            panel,
            model,
            "Tokamak-SI",
            name,
            base,
            src,
            prevalence_key(metric, "counts.prevalence"),
            f"prevalence {prevalence:.4f} of the bins both models are scored on "
            f"({counts['positive_bins']} present, {counts['negative_bins']} "
            "negative)",
        )
        unc = entry["uncertain_negative"]["metrics"][metric]
        rows.add(
            panel,
            model,
            "Tokamak-SI, uncertain time scored negative",
            name,
            unc["estimate"],
            (unc["low"], unc["high"]),
            src,
            f"models.{model}.uncertain_negative.metrics.{metric}.estimate",
            note="the same predictions; kept out of the figure",
            drawn=False,
        )
    diff = rwm["paired"]["rwm-ours_minus_rwm-brf"]["primary"][metric]
    rows.add(
        panel,
        "rwm-ours minus rwm-brf",
        "Tokamak-SI",
        name,
        diff["estimate"],
        (diff["low"], diff["high"]),
        src,
        f"paired.rwm-ours_minus_rwm-brf.primary.{metric}.estimate",
        note="paired shot bootstrap: the same shots drawn for both models",
        drawn=False,
    )
    finish(ax, ["rwm-brf", "rwm-ours"], name)


def saw_view(saw: dict) -> str | None:
    """The view of the record the sawtooth panel draws: `SAW_VIEW` (time with no
    q=1 surface scored as tested-absent) once every model has it, the drawn ones,
    the trivial baseline and the CSV-only run; else None, the primary scores. A
    view that only some models have is refused."""
    models = (*SAW_MODELS, SAW_TRIVIAL, SAW_HL3_ECE)
    have = {m: SAW_VIEW in saw["Tokamak-SI"][m] for m in models}
    if all(have.values()):
        return SAW_VIEW
    if any(have.values()):
        raise ValueError(f"only some sawtooth models have {SAW_VIEW}: {have}")
    return None


def saw_presence(saw: dict, model: str, view: str | None = None) -> tuple[dict, dict]:
    """A model's out-of-fold presence scores and their 95 % shot-bootstrap
    intervals (`crash_tolerance_2ms` of the benchmark record, inside `view` when
    one is given). The presence block does not depend on the crash tolerance; the
    record must say the same at 1 ms."""
    entry = saw["Tokamak-SI"][model]
    if view is not None:
        entry = entry[view]
    if entry["crash_tolerance_1ms"]["presence"] != entry[SAW_BLOCK]["presence"]:
        raise ValueError(f"{model}: presence differs between the crash tolerances")
    return entry[SAW_BLOCK]["presence"], entry[SAW_BLOCK]["ci95"]


def draw_sawtooth(ax, saw: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """`saw-hl3-full` and `saw-ours` on the Tokamak-SI labels
    (presence, out of fold, in the record's `saw_view`), each with the
    always-present baseline on its own bins; in the F1 figure `saw-hl3-full` has
    beside it the HL-3 paper's stated offline three-class window accuracy (REPORTED:
    its own machine and windows, not an F1)."""
    name = METRICS[metric]
    src = SOURCES["saw"]
    view = saw_view(saw)
    legacy = saw["legacy"]
    if metric != "f1":
        rows.add(
            panel,
            SAW_HL3,
            "legacy",
            name,
            None,
            None,
            src,
            "legacy",
            note=f"{legacy['source']}: no presence {name}; not drawn",
            drawn=False,
        )
    block = f"Tokamak-SI.{{}}.{view + '.' if view else ''}{SAW_BLOCK}.presence"
    seen = (
        f"; the {view} view: {saw['Tokamak-SI'][SAW_MODELS[0]][view]['definition']}"
        if view
        else ""
    )
    base_presence, _ = saw_presence(saw, SAW_TRIVIAL, view)
    base = base_presence[metric]
    for tick, model in enumerate(SAW_MODELS):
        presence, ci95 = saw_presence(saw, model, view)
        bins = (presence["positive_bins"], presence["negative_bins"])
        if bins != (base_presence["positive_bins"], base_presence["negative_bins"]):
            raise ValueError(f"{model} is scored on other bins than {SAW_TRIVIAL}")
        value = presence[metric]
        ci = tuple(ci95[f"presence_{metric}"])
        x, side = tick, 0
        if metric == "f1" and model == SAW_HL3:  # the stated value beside it
            stated = legacy["offline"]["accuracy_stated"]
            reported_bar(ax, tick - HALF, stated, side=-1)
            x, side = tick + HALF, 1
        scored_bar(ax, x, value, ci, SI, side=side, base=base)
        baseline_row(
            rows,
            panel,
            model,
            "Tokamak-SI",
            name,
            base,
            src,
            f"{block.format(SAW_TRIVIAL)}.{metric}",
            "presence = 1 everywhere, on the same bins" + seen,
        )
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
            )
            + seen,
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
                + ("; the primary scores, not the drawn view" if view else "")
                + "; kept out of the figure",
                drawn=False,
            )
    ece_presence, ece_ci95 = saw_presence(saw, SAW_HL3_ECE, view)
    ece_entry = saw["Tokamak-SI"][SAW_HL3_ECE]
    rows.add(
        panel,
        SAW_HL3_ECE,
        "Tokamak-SI, four-input run",
        name,
        ece_presence[metric],
        tuple(ece_ci95[f"presence_{metric}"]),
        src,
        f"{block.format(SAW_HL3_ECE)}.{metric}",
        note="the earlier four-input HL-3 run (EFIT-axis and outer ECE, Mirnov, Ip), "
        f"same {ece_entry['coverage']['supported_shots']} supported TRAIN shots "
        "and bins; kept out of the figure",
        drawn=False,
    )
    if metric == "f1":
        for model in (SAW_HL3, SAW_HL3_ECE):
            entry = saw["Tokamak-SI"][model]
            group = (
                "Tokamak-SI, three-class"
                if model == SAW_HL3
                else "Tokamak-SI, four-input run, three-class"
            )
            rows.add(
                panel,
                model,
                group,
                "three-class window accuracy",
                entry["three_class_window_accuracy"],
                tuple(entry["three_class_accuracy_ci95"]),
                src,
                f"Tokamak-SI.{model}.three_class_window_accuracy",
                note="the same three-class windows scored on DIII-D, out of fold; "
                "kept out of the figure",
                drawn=False,
            )
            majority = entry["three_class_majority_baseline"]
            rows.add(
                panel,
                model,
                group,
                "three-class window accuracy, majority-class floor",
                majority["accuracy"],
                tuple(majority["accuracy_ci95"]),
                src,
                f"Tokamak-SI.{model}.three_class_majority_baseline.accuracy",
                note="the fitting-chosen majority class on the same windows; "
                "kept out of the figure",
                drawn=False,
            )
        for mode, key in (("real-time", "real_time"), ("offline", "offline")):
            offline = key == "offline"
            rows.add(
                panel,
                SAW_HL3,
                "legacy",
                f"three-class window accuracy, {mode} (stated)",
                legacy[key]["accuracy_stated"],
                None,
                src,
                f"legacy.{key}.accuracy_stated",
                note=(
                    "REPORTED: a three-class window accuracy stated in the HL-3 "
                    "paper (offline, the nine-input model), on its own machine and "
                    "windows, not an F1; drawn in the F1 panel"
                    if offline
                    else "the HL-3 paper's own metric on its own machine and "
                    "windows; not an F1, not comparable; kept out of the figure"
                ),
                drawn=offline,
            )
    finish(ax, [SAW_TICKS.get(m, m) for m in SAW_MODELS], name)


DETACH_TIER = "tangtv_vote_before_gate"  # the check: every valid TangTV vote
DETACH_NAME = "detach-ours"  # the one tick of the panel: the 0-D window model
DETACH_VOTE = "detach-vote"  # the TangTV vote's consistency check, CSV only


def detachment_check(te: dict, source: dict) -> dict:
    """The detachment consistency check as `draw_detachment` reads it: the AUROC of
    -Te (divertor Thomson) against the TangTV vote, with its shot-bootstrap
    interval (`te`, `detachment_te_check.json`), and the vote's agreement with Te's
    state where both name one, a rate with no interval (`source`,
    `detachment_source_v3.json`). The stored agreement must be its own counts. The
    baseline of the agreement is the share of the majority state among the compared
    bins (Te's state is the reference: a vote that calls that state everywhere), read
    from the vote-by-Te table, whose bins must be the compared bins."""
    auroc = te["by_tier"][DETACH_TIER]["auroc_neg_te_detached_vs_attached"]["pooled"]
    states = source["agreement_vote_vs_te"]["states"]
    from_counts = states["agree"] / states["bins"]
    if abs(states["agreement"] - from_counts) > 1e-9:
        raise ValueError(
            f"detachment agreement {states['agreement']} is not its counts' "
            f"{from_counts}"
        )
    table = source["agreement_vote_vs_te"]["vote_by_te"]
    named = ("attached", "detached")
    te_state = {t: sum(table[v][t] for v in named) for t in named}
    if sum(te_state.values()) != states["bins"]:
        raise ValueError(
            f"the vote-by-Te table holds {te_state}, not the compared bins"
        )
    return {
        "auroc": {
            "value": auroc["value"],
            "ci": tuple(auroc["ci95"]),
            "bins": auroc["n_bins"],
            "shots": auroc["n_shots"],
            "key": f"by_tier.{DETACH_TIER}.auroc_neg_te_detached_vs_attached.pooled",
        },
        "agreement": {
            "value": states["agreement"],
            "bins": states["bins"],
            "agree": states["agree"],
            "shots": states["shots"],
            "kappa": states["kappa"],
            "key": "agreement_vote_vs_te.states",
            "base": max(te_state.values()) / states["bins"],
            "base_key": "agreement_vote_vs_te.vote_by_te: majority Te state share",
        },
    }


def detachment_ours(record: dict) -> dict:
    """detach-ours (`detach_ours.py`: windows of 0-D signals, 5-fold shot-grouped
    cross-validation on the detachment labels) as `draw_detachment` reads it: its
    macro F1 with the shot-bootstrap interval, and the always-majority predictor's
    macro F1 on the same bins (the confinement panel's baseline), from the record's
    class supports, which must be its bins."""
    cv = record["cv_shots"]
    support = [cv["class_support"][c]["n_bins"] for c in cv["reference_classes"]]
    if sum(support) != cv["n_bins"]:
        raise ValueError(
            f"detach-ours's class supports {support} are not its {cv['n_bins']} bins"
        )
    return {
        "value": cv["macro_f1"]["value"],
        "ci": tuple(cv["macro_f1"]["ci95"]),
        "bins": cv["n_bins"],
        "shots": cv["n_shots"],
        "key": "cv_shots.macro_f1",
        "base": always_f1(max(support) / sum(support)) / len(support),
        "base_key": "cv_shots.class_support: always-majority macro F1",
    }


def draw_detachment(ax, det: dict, rows: Rows, panel: str, metric: str = "f1") -> None:
    """detach-ours on the detachment labels, one Tokamak-SI bar on its own tick
    (there is no earlier detector or label set to set beside it). F1 figure: its
    macro F1 with its interval and the always-majority baseline. AUROC, AUPRC: not
    reported (the record holds none). The consistency check of the TangTV vote
    against divertor Thomson Te (`detachment_check`: the agreement and kappa, and
    the AUROC of -Te) stays in the CSV only."""
    name = METRICS[metric]
    if metric == "f1":
        ours = det["ours"]
        src = SOURCES["detach_ours"]
        scored_bar(ax, 0, ours["value"], ours["ci"], SI, base=ours["base"])
        rows.add(
            panel,
            DETACH_NAME,
            "Tokamak-SI",
            "macro F1",
            ours["value"],
            ours["ci"],
            src,
            f"{ours['key']}.value",
            note=f"5-fold shot-grouped CV on the detachment labels, {ours['bins']} "
            f"bins, {ours['shots']} shots; agreement with constructed labels",
        )
        baseline_row(
            rows,
            panel,
            DETACH_NAME,
            "Tokamak-SI",
            "macro F1",
            ours["base"],
            src,
            ours["base_key"],
            "a constant predictor on the same bins",
        )
    else:
        note(ax, 0, "not reported")
        rows.add(
            panel,
            DETACH_NAME,
            "Tokamak-SI",
            name,
            None,
            None,
            SOURCES["detach_ours"],
            "-",
            note=f"no {name} is stored for detach-ours; not drawn",
            drawn=False,
        )
    check = (
        "consistency check of the TangTV vote, not a detector; kept out of the figure"
    )
    agree, auroc = det["agreement"], det["auroc"]
    if metric == "f1":
        for group, value, key in (
            ("agreement", agree["value"], f"{agree['key']}.agreement"),
            ("Cohen's kappa", agree["kappa"], f"{agree['key']}.kappa"),
        ):
            rows.add(
                panel,
                DETACH_VOTE,
                f"Tokamak-SI, {group}",
                group,
                value,
                None,
                SOURCES["detach_source"],
                key,
                note=f"TangTV vote against divertor Thomson Te state where both name "
                f"one: {agree['agree']} of {agree['bins']} bins; {check}",
                drawn=False,
            )
    elif metric == "auroc":
        rows.add(
            panel,
            DETACH_VOTE,
            "Tokamak-SI, AUROC",
            "AUROC",
            auroc["value"],
            auroc["ci"],
            SOURCES["detach_te"],
            f"{auroc['key']}.value",
            note=f"-Te against the TangTV vote before the L-mode gate, "
            f"{auroc['bins']} bins, {auroc['shots']} shots; {check}",
            drawn=False,
        )
    finish(ax, [DETACH_NAME], name)


def draw_coverage(ax, cov: dict, rows: Rows, panel: str) -> None:
    """Labelled time in hours per set (bars, log y) with the labelled shot count
    written on each bar: original against Tokamak-SI, each Tokamak-SI bar with a
    darker inner bar for its human-reviewed subset; a set with no time in a
    setting gets a short grey note where its bar would be."""
    unit, count = "labeled time (h)", "labeled shots"
    order = cov["order"]
    hours = {}
    for key in order:
        for side in ("legacy", "tokamak_si"):
            v = cov["sets"][key][side]["seconds"]
            hours[key, side] = None if v is None else v / S_PER_H
    drawn = [v for v in hours.values() if v is not None]
    for key in order:  # the reviewed subset can be the smallest bar
        reviewed = cov["sets"][key]["tokamak_si"].get("reviewed")
        if reviewed and reviewed["status"] == "measured":
            drawn.append(reviewed["seconds"] / S_PER_H)
    lo = 10.0 ** np.floor(np.log10(min(drawn)))
    # room above the tallest bar for its count
    hi = 10.0 ** (np.log10(max(drawn)) + 0.75)
    width = 0.38
    for i, key in enumerate(order):
        entry = cov["sets"][key]
        for sign, side, colour, group in (
            (-1, "legacy", LEGACY, "legacy"),
            (1, "tokamak_si", SI, "Tokamak-SI"),
        ):
            x = i + sign * 0.2
            block = entry[side]
            value, shots = hours[key, side], block["shots"]
            record_key = f"sets.{key}.{side}"
            definition = (
                block["definition"]
                if block.get("definition", "shared definition") != "shared definition"
                else ""
            )
            if value is None:
                status = STATUS_TEXT[block["status"]]
                text = status if shots is None else f"{status}\n{shots} shots"
                note(ax, x, text, y=lo * 1.25)
            else:
                ax.bar(x, value - lo, width, bottom=lo, color=colour, lw=0, zorder=2)
                ax.text(
                    x,
                    value * 1.12,
                    f"{shots}",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_PT,
                    color=INK,
                    zorder=6,
                )
            for metric, v, field in ((unit, value, "seconds"), (count, shots, "shots")):
                rows.add(
                    panel,
                    entry["name"],
                    group,
                    metric,
                    v,
                    None,
                    SOURCES["coverage"],
                    f"{record_key}.{field}",
                    note=STATUS_TEXT[block["status"]] if value is None else definition,
                    drawn=v is not None,
                )
            reviewed = block.get("reviewed")
            if reviewed and reviewed["status"] == "measured":
                inner = reviewed["seconds"] / S_PER_H
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
                    f"{record_key}.reviewed.seconds",
                )
                rows.add(
                    panel,
                    entry["name"],
                    "Tokamak-SI, reviewed subset",
                    count,
                    reviewed["shots"],
                    None,
                    SOURCES["coverage"],
                    f"{record_key}.reviewed.shots",
                    note="not written on the figure",
                    drawn=False,
                )
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(range(len(order)), [cov["sets"][k]["name"] for k in order])
    ax.tick_params(axis="x", length=0, pad=2, labelsize=FONT_PT)
    ax.tick_params(axis="y", labelsize=FONT_PT, length=2, pad=1.5)
    ax.grid(axis="y", color="#e4e4e4", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
        ax.spines[side].set_linewidth(0.6)
    ax.set_ylabel("hours", labelpad=2)
    ax.set_title(
        "Labeled data per set",
        loc="left",
        fontsize=FONT_PT,
        pad=4,
    )


def place(ax, fig_h, left_in, top_in, width_in, height_in) -> None:
    """Put `ax` at a box given in inches from the figure's left and top edges."""
    ax.set_position(
        (
            left_in / PAGE_IN,
            1 - (top_in + height_in) / fig_h,
            width_in / PAGE_IN,
            height_in / fig_h,
        )
    )


def place_rows(rows_of_axes, fig_h, tops_in, height_in) -> None:
    """Size every drawn panel to its x range so that a tick spacing is as wide in
    every row: the width the row that needs most room allows (its panels
    `GAP_IN` apart); the other rows spread their panels apart. Each row is
    `height_in` high, its top `tops_in` under the figure's top."""
    spans = [[float(np.diff(ax.get_xlim())[0]) for ax in row] for row in rows_of_axes]
    room = PAGE_IN - LEFT_IN - RIGHT_IN
    unit = min(
        (room - GAP_IN * (len(row) - 1)) / sum(span)
        for row, span in zip(rows_of_axes, spans)
    )
    for row, span, top in zip(rows_of_axes, spans, tops_in):
        gap = (room - unit * sum(span)) / (len(row) - 1)
        left = LEFT_IN
        for ax, s in zip(row, span):
            place(ax, fig_h, left, top, unit * s, height_in)
            left += unit * s + gap


def legend_handles(reviewed: bool) -> list:
    handles = [
        Patch(color=LEGACY, label=SHOWN["legacy"]),
        Patch(color=SI, label="Tokamak-SI"),
    ]
    if reviewed:
        handles.append(Patch(color=REVIEWED, label="human-reviewed subset"))
    handles.append(
        Line2D([], [], color="black", lw=0.7, ls=BASELINE_DASH, label=BASELINE)
    )
    return handles


def save(fig, out: Path, png: Path | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", metadata={"CreationDate": None})
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=150)


def panels(axes, records, rows, metric, tag):
    """The seven panels of one metric: AE, confinement, ELM over TM, sawtooth, RWM,
    detachment."""
    ae, conf, si, elm, smith, tm, rwm, saw, det = records
    top, bottom = axes
    draw_ae(top[0], ae, rows, f"a{tag}", metric)
    draw_confinement(top[1], conf, si, rows, f"b{tag}", metric)
    draw_elm(top[2], elm, smith, rows, f"c{tag}", metric)
    draw_tm(bottom[0], tm, rows, f"d{tag}", metric)
    draw_sawtooth(bottom[1], saw, rows, f"e{tag}", metric)
    draw_rwm(bottom[2], rwm, rows, f"f{tag}", metric)
    draw_detachment(bottom[3], det, rows, f"h{tag}", metric)
    titles = (
        TITLES["ae"],
        TITLES["conf"],
        TITLES["elm"],
        TITLES["tm"],
        TITLES["sawtooth"],
        TITLES["rwm"],
        TITLES["detach"],
    )
    for ax, text in zip([*top, *bottom], titles):
        # the one-tick panels are narrower than their names: the RWM name ends
        # at its panel's right edge rather than run into the detachment panel
        loc = "right" if ax is bottom[2] else "left"
        ax.set_title(text, loc=loc, fontsize=FONT_PT, pad=4)


def read_records():
    ae, conf, elm, smith, tm, rwm, saw = (
        load(SOURCES[k])
        for k in ("ae", "confinement", "elm", "elm_smith", "tm", "rwm", "saw")
    )
    row = load(SOURCES["confinement_si"])
    si = confinement_si(row)
    si["ours"] = confinement_ours(load(SOURCES["confinement_ours"]), row)
    det = detachment_check(load(SOURCES["detach_te"]), load(SOURCES["detach_source"]))
    det["ours"] = detachment_ours(load(SOURCES["detach_ours"]))
    return ae, conf, si, elm, smith, tm, rwm, saw, det


@contextmanager
def plot_style() -> Iterator[None]:
    """The paper's text sizes, raised so that nothing is under FONT_PT; make and
    save a figure inside it."""
    with style(), matplotlib.rc_context(TEXT):
        yield


def add_legend(fig, reviewed: bool) -> None:
    # with the reviewed entry, two columns filled down: the colours, then the
    # reviewed subset over the baseline dash
    fig.legend(
        handles=legend_handles(reviewed),
        loc="upper center",
        ncol=2 if reviewed else 3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.6,
        columnspacing=1.4,
        labelspacing=0.3,
        borderaxespad=0.1,
    )


def figure_f1(records, cov: dict, rows: Rows) -> Figure:
    """The F1 figure: two rows of panels, then the coverage panel. Call inside
    `plot_style`."""
    block = TITLE_IN + AXES_IN + NAMES_IN
    fig_h = LEGEND_IN + 2 * block + TITLE_IN + COVER_IN + COVER_NAMES_IN + 0.04
    fig = Figure(figsize=(PAGE_IN, fig_h))
    axes = [[fig.add_axes((0, 0, 1, 1)) for _ in range(n)] for n in (3, 4)]
    panels(axes, records, rows, "f1", "")
    tops = [LEGEND_IN + TITLE_IN, LEGEND_IN + block + TITLE_IN]
    place_rows(axes, fig_h, tops, AXES_IN)
    cover = fig.add_axes((0, 0, 1, 1))
    draw_coverage(cover, cov, rows, "g")
    top = LEGEND_IN + 2 * block + TITLE_IN
    left = LEFT_IN + COVER_TICKS_IN
    place(cover, fig_h, left, top, PAGE_IN - left - RIGHT_IN, COVER_IN)
    add_legend(fig, reviewed=True)
    return fig


def figure_auc(records, rows: Rows) -> Figure:
    """AUROC then AUPRC, each as two rows of panels under a bold heading that
    names the metric (`AUC_HEADINGS`; the panel titles repeat), no coverage panel.
    Call inside `plot_style`."""
    block = TITLE_IN + AXES_IN + NAMES_IN
    legend_in = LEGEND_IN + 0.1
    fig_h = legend_in + 4 * block + 2 * HEADING_IN + 0.04
    fig = Figure(figsize=(PAGE_IN, fig_h))
    axes = []
    for metric, tag in (("auroc", "0"), ("auprc", "1")):
        pair = [[fig.add_axes((0, 0, 1, 1)) for _ in range(n)] for n in (3, 4)]
        panels(pair, records, rows, metric, tag)
        axes += pair
    # each block's first row has its heading above it
    tops = [
        legend_in + (i // 2 + 1) * HEADING_IN + TITLE_IN + i * block for i in range(4)
    ]
    place_rows(axes, fig_h, tops, AXES_IN)
    for top, text in zip(tops[::2], AUC_HEADINGS.values(), strict=True):
        fig.text(
            0.01,
            1 - (top - TITLE_IN - HEADING_IN) / fig_h,
            text,
            fontsize=FONT_PT + 1,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
        )
    add_legend(fig, reviewed=False)
    return fig


def draw(out: Path, png: Path | None, table: Path | None) -> None:
    """The F1 figure, with the coverage row, and the numbers behind it."""
    rows = Rows()
    with plot_style():
        fig = figure_f1(read_records(), load(SOURCES["coverage"]), rows)
        save(fig, out, png)
    rows.write(table)
    records = {k: load(p) for k, p in SOURCES.items()}
    gits = {k: r.get("git") or r.get("git_sha") for k, r in records.items()}
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
