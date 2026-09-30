r"""fig_scores, the selected models' F1, ROC and precision-recall curve, and
the score tables.

`draw_scores` is one model per paper phenomenon (`ORDER`), the one selected for
main inference (`labeler.paper.roc.SELECTED`): ae_xpower for AE, the frame
models for the other four. Three panels, side by side: each model's test F1
with its 95 % shot-bootstrap interval, from its `evaluation.json`, its value
written beside it; each model's ROC on the same test shots, from its
`roc.json` (`labeler.paper.roc`), its threshold's point marked, beside the
dashed chance diagonal; and its precision-recall curve, its threshold's point
marked, with the phenomenon's positive share (the chance level of a PR curve)
as a short dotted tick in its colour at the right edge. One key below the
three panels gives each phenomenon's AUROC and AUPRC ("AE: AUROC 0.99, AUPRC
0.95"), then the chance entries and the threshold's point. A phenomenon with
no evaluation has no dot and its tick says `NOT_SCORED`; one with no ROC has
no curve and its key says `NO_ROC`; a `roc.json` with no `auprc` draws its ROC
but no PR curve, and its key says `NO_PR`. No other method and no baseline is
drawn: those are the tables'.

`table_ae`, `table_segmentation` and `table_differences` give the AE methods'
and the segmentation's scores and their paired differences as LaTeX
`tabular`s, at three decimals, for the owner to caption.

**The tables fit ICML's 487.8 pt text width** at `\small` or larger, in Times.
A score is written `$v^{+a}_{-b}$`, its 95 % interval's ends `v + a` and `v - b`
(`INTERVAL_NOTE`, said once in each table's `%` comment), exact where
`v ± half-width` would not be: the bootstrap intervals are not symmetric. Its
numbers, and those of table_differences' conditions, are set in amsmath's
`\text{}`, so their digits print in the text font, Times, and not in Computer
Modern, the math font `times` leaves; the signs stay math symbols. The
manuscript loads amsmath, but `icml2025.sty` does not, so each table that uses
`\text{}` says once in its `%` comment that it needs amsmath (`AMSMATH`). A
negative number outside math is written `$-$0.188`, and one that rounds to zero
is `0.000`, unsigned, in math too (`number`, `text_number`).

The AE methods are the same six in table_ae_scores (`AE_METHODS`): the model,
the two detectors it is compared with, the start table, the UCI windows and
the always-present baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from . import FONT_PT, PAGE_IN, save, style, title

AE_NAMES = {
    "ae_xpower": "ae_xpower",
    "seldnet": "SELDnet",
    "tokeye": "TokEye",
    "source": "start table",
    "uci": "UCI windows",
    "always": "always",
}
AE_METHODS = tuple(AE_NAMES)  # in table_ae_scores
SEG_NAMES = {"ae_seg": "ae_seg", "recipe": "recipe", "tokeye": "TokEye"}
SEG_METRICS = {
    "dice": "Dice",
    "frame_precision": "frame P",
    "frame_recall": "frame R",
    "frame_f1": "frame F1",
    "fp_rate_mhd": "MHD FP",
}
#: fig_scores' colour per phenomenon, the same in both panels.
PHENOMENON_COLOURS = {
    "alfven_eigenmode": "#d62728",
    "neoclassical_tearing_mode": "#1f77b4",
    "high_confinement_mode": "#2ca02c",
    "edge_localized_mode": "#9467bd",
    "sawtooth_oscillation": "#ff7f0e",
}
NOT_SCORED = "not scored"  # a phenomenon's tick, without its evaluation
NO_ROC = "no ROC"  # its key, without its roc.json
NO_PR = "no PR"  # its key, with a roc.json that has no auprc
CHANCE = "chance: diagonal"
CHANCE_PR = "chance: positive share"
AT_THRESHOLD = "at the model's threshold"
F1_LABEL = "test F1 (95 % shot-bootstrap interval)"
SHARE_TICK = 0.12  # the positive-share tick's length, in axes units
KEY_COLUMNS = 3
SCORES_HEIGHT_IN = 2.8  # 0.4 in over the two-panel figure, for the key
DECIMALS = 3
AMSMATH = r"needs amsmath for its \text{}"  # in each table that uses it, once
INTERVAL_NOTE = (
    "$v^{+a}_{-b}$ is the value with its 95% shot-bootstrap interval, "
    "[v - b, v + a] (a = high - value, b = value - low, each end rounded to "
    f"{DECIMALS} decimals)"
)
# (record, key, comparison, bar, condition from the thresholds, its test)
DIFFERENCES = (
    (
        "ae",
        "f1_minus_seldnet",
        "AE F1: ae_xpower $-$ SELDnet",
        "A1",
        lambda t: rf"low $\geq {text_number(t['f1_vs_seldnet_low'], 'g')}$",
        lambda lo, hi, t: lo >= t["f1_vs_seldnet_low"],
    ),
    (
        "ae",
        "mhd_fp_minus_seldnet",
        "AE MHD FP: ae_xpower $-$ SELDnet",
        "A2",
        lambda t: r"high $< \text{0}$",
        lambda lo, hi, t: hi < 0,
    ),
    (
        "ae",
        "f1_minus_always",
        "AE F1: ae_xpower $-$ always",
        "A3",
        lambda t: r"low $> \text{0}$",
        lambda lo, hi, t: lo > 0,
    ),
    ("seg", "dice_minus_recipe", "Seg Dice: ae_seg $-$ recipe", None, None, None),
    (
        "seg",
        "frame_f1_minus_recipe",
        "Seg frame F1: ae_seg $-$ recipe",
        None,
        None,
        None,
    ),
)


class Selected(NamedTuple):
    """One phenomenon's model selected for main inference, as fig_scores draws
    it: its evaluation's F1 estimate (`stats.Estimate.as_json`; None when it is
    not scored) and its `roc.json` record (None when there is none)."""

    category: str
    method: str
    f1: dict | None
    roc: dict | None


def read(path) -> dict | None:
    """An evaluation record, or None if the run has not written it."""
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def interval(estimate: dict) -> tuple[float, float, float]:
    """`(value, low, high)` of a `stats.Estimate.as_json`, NaN where undefined."""
    return tuple(
        math.nan if estimate.get(k) is None else float(estimate[k])
        for k in ("value", "low", "high")
    )


def _scored(s: Selected) -> bool:
    return s.f1 is not None and not math.isnan(interval(s.f1)[0])


def _f1_panel(ax, selected: Sequence[Selected]) -> None:
    """A dot per scored phenomenon, its interval a whisker, its value beside it."""
    for i, s in enumerate(selected):
        if not _scored(s):
            continue
        v, lo, hi = interval(s.f1)
        colour = PHENOMENON_COLOURS[s.category]
        whisker = np.nan_to_num(np.array([[v - lo], [hi - v]]))
        ax.errorbar(
            [i],
            [v],
            yerr=whisker,
            fmt="o",
            ms=4,
            color=colour,
            ecolor=colour,
            elinewidth=0.8,
            capsize=2,
            clip_on=False,
            label=title(s.category),
        )
        ax.annotate(
            f"{v:.2f}",
            (i, v),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=FONT_PT - 1,
        )
    ticks = [
        title(s.category) if _scored(s) else f"{title(s.category)}\n{NOT_SCORED}"
        for s in selected
    ]
    ax.set_xticks(range(len(selected)), ticks)
    ax.set_xlim(-0.5, len(selected) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel(F1_LABEL)


def roc_key(s: Selected) -> str:
    """The phenomenon's entry in the shared key: its AUROC (or `NO_ROC`) and
    its AUPRC (or `NO_PR`), as "AE: AUROC 0.99, AUPRC 0.95"."""
    if s.roc is None:
        return f"{title(s.category)}: {NO_ROC}"
    auprc = s.roc.get("auprc")
    pr = NO_PR if auprc is None else f"AUPRC {auprc:.2f}"
    return f"{title(s.category)}: AUROC {s.roc['auroc']:.2f}, {pr}"


def _mark(ax, x, y, colour) -> None:
    """A model's threshold point."""
    ax.plot(
        [x],
        [y],
        "o",
        ms=3.5,
        color=colour,
        mec="white",
        mew=0.5,
        zorder=3,
        clip_on=False,
    )


def _roc_panel(ax, selected: Sequence[Selected]) -> None:
    """A curve per phenomenon with a ROC, its threshold's point marked, and
    the chance diagonal."""
    for s in selected:
        if s.roc is None:
            continue
        colour = PHENOMENON_COLOURS[s.category]
        points = s.roc["curve"]
        ax.plot(points["fpr"], points["tpr"], color=colour, lw=1.0)
        at = s.roc["threshold"]
        _mark(ax, at["fpr"], at["tpr"], colour)
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")


def _pr_panel(ax, selected: Sequence[Selected]) -> None:
    """A curve per phenomenon whose `roc.json` has a PR curve, its threshold's
    point marked, and its positive share as a short dotted tick at the right
    edge."""
    for s in selected:
        if s.roc is None or "auprc" not in s.roc:
            continue
        colour = PHENOMENON_COLOURS[s.category]
        points = s.roc["pr"]
        # a step at each point, as the average precision sums it: a straight
        # line between two would claim precision no threshold has
        ax.plot(
            points["recall"],
            points["precision"],
            color=colour,
            lw=1.0,
            drawstyle="steps-pre",
        )
        at = s.roc["threshold"]
        _mark(ax, at["recall"], at["precision"], colour)
        share = s.roc["positive_share"]
        ax.plot([1 - SHARE_TICK, 1], [share, share], ls=":", color=colour, lw=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")


def _key(selected: Sequence[Selected]) -> list:
    """The shared key's handles: a coloured line per phenomenon, the ROC's
    chance diagonal, the PR's chance tick and the threshold's point."""
    handles = []
    for s in selected:
        colour = PHENOMENON_COLOURS[s.category]
        dead = s.roc is None
        handles.append(Line2D([], [], color=colour, lw=1.0, ls=":" if dead else "-"))
        handles[-1].set_label(roc_key(s))
    chance = Line2D([], [], ls="--", color="#999999", lw=0.7, label=CHANCE)
    share = Line2D([], [], ls=":", color="#999999", lw=1.2, label=CHANCE_PR)
    point = Line2D([], [], ls="none", marker="o", ms=3.5, color="#444444")
    point.set_label(AT_THRESHOLD)
    return [*handles, chance, share, point]


def draw_scores(selected: Sequence[Selected], stem: Path) -> Figure:
    """The selected models' test F1, ROC and precision-recall curve, side by
    side, with one key below (module docstring), one per phenomenon in the
    order given."""
    with style():
        fig = Figure(figsize=(PAGE_IN, SCORES_HEIGHT_IN), layout="constrained")
        f1, roc, pr = fig.subplots(1, 3, width_ratios=[1.3, 1, 1])
        _f1_panel(f1, selected)
        _roc_panel(roc, selected)
        _pr_panel(pr, selected)
        fig.legend(
            handles=_key(selected),
            loc="outside lower center",
            ncol=KEY_COLUMNS,
            frameon=False,
        )
        save(fig, stem)
    return fig


def number(x: float, signed: bool = False) -> str:
    """`x` at `DECIMALS` for LaTeX text: `$-$` for a minus, `+` when `signed`,
    and a value that rounds to zero as `0.000`, unsigned."""
    text = f"{abs(x):.{DECIMALS}f}"
    if float(text) == 0:
        return text
    if x < 0:
        return f"$-${text}"
    return f"+{text}" if signed else text


def text_number(x: float, spec: str) -> str:
    r"""`x` for math mode, formatted by `spec`: its digits in amsmath's
    `\text{}`, so in the text font, and its minus a math symbol; a value that
    rounds to zero is unsigned, as `number` writes it."""
    text = format(abs(x), spec)
    minus = "-" if x < 0 and float(text) != 0 else ""
    return rf"{minus}\text{{{text}}}"


def _fmt(estimate: dict) -> str:
    """A score and its interval, `$v^{+a}_{-b}$` (`INTERVAL_NOTE`), each number
    in the text font (`text_number`)."""
    v, lo, hi = interval(estimate)
    if math.isnan(v):
        return "--"
    if math.isnan(lo) or math.isnan(hi):
        return number(v)
    value, low, high = (round(x, DECIMALS) for x in (v, lo, hi))
    up, down = max(high - value, 0.0), max(value - low, 0.0)
    value, up, down = (text_number(x, f".{DECIMALS}f") for x in (value, up, down))
    return f"${value}^{{+{up}}}_{{-{down}}}$"


def _fmt_difference(estimate: dict) -> str:
    """A paired difference, signed, with its interval in brackets."""
    v, lo, hi = interval(estimate)
    if math.isnan(v):
        return "--"
    if math.isnan(lo) or math.isnan(hi):
        return number(v, signed=True)
    ends = ", ".join(number(x, signed=True) for x in (lo, hi))
    return f"{number(v, signed=True)} [{ends}]"


def _tex(text: str) -> str:
    return text.replace("_", r"\_")


def tabular(header: Sequence[str], rows: Sequence[Sequence[str]], comment: str) -> str:
    r"""A booktabs `tabular`, a `%` comment line above it. A table that sets
    anything in `\text{}` ends its comment with `AMSMATH`, so whoever pastes it
    into a document without amsmath learns why it stops at its first cell."""
    cells = [*header, *(cell for row in rows for cell in row)]
    if any(r"\text{" in cell for cell in cells):
        comment = f"{comment}; {AMSMATH}"
    lines = [
        f"% {comment}",
        rf"\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _verdict(bar: dict) -> str:
    return ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in bar.items())


def table_ae(ae: dict, second_look: str | None = None) -> str:
    """The AE frame scores; the comment ends with the second look, when there
    is one (`labeler.paper.build.second_look`)."""
    keys = ("precision", "recall", "f1", "fp_rate_mhd", "fp_rate_other")
    rows = [
        [_tex(AE_NAMES[m]), *(_fmt(ae["methods"][m][k]) for k in keys)]
        for m in AE_METHODS
        if m in ae["methods"]
    ]
    n, meta = ae["frames"], ae.get("meta", {})
    comment = (
        f"AE frame scores: {n['shots']} test shots, {n['scored']} frames "
        f"({n['present']} present, {n['mhd_absent']} MHD); {INTERVAL_NOTE}; "
        f"{meta.get('candidate', '?')} at {meta.get('threshold', '?')}; "
        f"bar {_verdict(ae.get('bar', {}))}"
    )
    if second_look:
        comment += f"; {second_look}"
    header = ("Method", "Precision", "Recall", "F1", "FP (MHD)", "FP (other)")
    return tabular(header, rows, comment)


def table_segmentation(seg: dict) -> str:
    rows = [
        [_tex(SEG_NAMES[m]), *(_fmt(seg["methods"][m][k]) for k in SEG_METRICS)]
        for m in SEG_NAMES
        if m in seg["methods"]
    ]
    c = seg["counts"]
    comment = (
        f"AE segmentation: {c['shots']} test shots, {c['scored_pixels']} scored "
        f"pixels ({c['ae_pixels']} AE), {c['frames']} frames; {INTERVAL_NOTE}; "
        f"bar {_verdict(seg.get('bar', {}))}"
    )
    if beside := seg.get("meta", {}).get("ae_model"):
        comment += f"; evaluated beside {'/'.join(Path(beside).parts[-4:-1])}"
    header = ("Method", "Dice", "Frame P", "Frame R", "Frame F1", "FP (MHD)")
    return tabular(header, rows, comment)


def table_differences(
    ae: dict | None, seg: dict | None, second_look: str | None = None
) -> str:
    """The paired differences the records hold, with the part of the bar each
    decides: its condition, whether it holds, and the record's verdict on that
    bar. Only a record that exists gives rows; the AE rows' second look, when
    there is one, ends the comment."""
    records = {"ae": ae, "seg": seg}
    rows = []
    for which, key, name, bar, condition, holds in DIFFERENCES:
        record = records[which]
        if record is None or key not in record.get("differences", {}):
            continue
        estimate = record["differences"][key]
        cells = [_tex(name), _fmt_difference(estimate)]
        if bar is None:
            cells += ["--", "--", "--"]
        else:
            _, lo, hi = interval(estimate)
            met = holds(lo, hi, record["bar_thresholds"])
            verdict = record.get("bar", {}).get(bar)
            cells += [
                condition(record["bar_thresholds"]),
                "yes" if met else "no",
                "--" if verdict is None else f"{bar} {'pass' if verdict else 'fail'}",
            ]
        rows.append(cells)
    comment = (
        "Paired differences with 95% shot-bootstrap intervals [low, high]; "
        "Condition is the part of the bar the difference decides, Bar the "
        "record's verdict on it"
    )
    if ae is not None and second_look:
        comment += f"; AE: {second_look}"
    header = ("Comparison", "Difference", "Condition", "Holds", "Bar")
    return tabular(header, rows, comment)
