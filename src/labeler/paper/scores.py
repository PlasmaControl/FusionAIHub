r"""Frame scores: the AE methods, their MHD check, and the segmentation.

`draw_scores` is one panel per paper phenomenon (`ORDER`): frame precision,
recall and F1 with 95 % shot-bootstrap intervals. AE's panel, across the top,
is `models/ae_xpower/<version>/evaluation.json` (`labeler.ae.xpower.evaluate`);
the other four, below it, are coming. Its dots and whiskers sit on an axis from
`SCORE_FLOOR`, so the differences A1 tests show; a value below the floor is
drawn on it, with its number. `draw_mhd` is AE's alone: each method's
false-positive rate on MHD frames (the owner says absent and TokEye sees a
0-60 kHz line) beside its rate on the other absent frames, against the bar, on
an axis capped at `MHD_CAP`; a bar past the cap runs to the edge and carries
its value.

**Each AE figure states its bar's verdict as the record gives it** (`bar`),
under its axis, with the numbers it turns on (`a1_said`, `a2_said`): A1's
dashed marks are only its absolute floors (`FLOORS_LABEL`), so fig_scores says
A1's paired clause, F1 − SELDnet's lower bound against its bar, and any floor
the model misses; fig_mhd says both of A2's clauses, the MHD rate against its
bar and MHD FP − SELDnet's upper bound against 0. They are stated, not drawn: a
difference sits on its own scale around 0, not on a score axis from
`SCORE_FLOOR` or a rate axis to `MHD_CAP`, and a bound 0.002 from its bar is a
gap no whisker shows at this size, where the text gives both numbers. A record
that passes says pass. `draw_segmentation` is
`models/ae_seg/v1/evaluation.json` (`labeler.ae.seg.evaluate`), with every bar
G1-G3 set (G1's on the Dice and on its interval's lower end); its MHD
false-positive rate, where lower is better, has its own axis from 0 to
`SEG_FP_TOP` so G3 reads. Recipe and TokEye are hatched, as the sources of the
pseudo-masks the segmentation is scored against. `table_ae`,
`table_segmentation` and `table_differences` give the same numbers as LaTeX
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

The AE methods are the same six everywhere (`AE_METHODS`): the model, the two
detectors it is compared with, the start table, the UCI windows and the
always-present baseline.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from . import AE, COLUMN_IN, FONT_PT, ORDER, PAGE_IN, placeholder, save, style, title

AE_NAMES = {
    "ae_xpower": "ae_xpower",
    "seldnet": "SELDnet",
    "tokeye": "TokEye",
    "source": "start table",
    "uci": "UCI windows",
    "always": "always",
}
AE_METHODS = tuple(AE_NAMES)  # in fig_scores, fig_mhd and table_ae_scores
METRICS = {"precision": "precision", "recall": "recall", "f1": "F1"}
SCORE_FLOOR = 0.6
MHD_CAP = 0.4
MARKERS = {
    "ae_xpower": "o",
    "seldnet": "s",
    "tokeye": "D",
    "source": "^",
    "uci": "v",
    "always": "X",
}
SEG_NAMES = {"ae_seg": "ae_seg", "recipe": "recipe", "tokeye": "TokEye"}
PSEUDO_SOURCES = ("recipe", "tokeye")  # the pseudo-masks are built from these
PSEUDO_LABEL = "pseudo-mask source"
HATCH = "////"
SEG_METRICS = {
    "dice": "Dice",
    "frame_precision": "frame P",
    "frame_recall": "frame R",
    "frame_f1": "frame F1",
    "fp_rate_mhd": "MHD FP",
}
SEG_BARS = {
    "dice": ("dice", "dice_low"),
    "frame_precision": ("frame_precision",),
    "fp_rate_mhd": ("mhd_fp_rate",),
}
SEG_FP = "fp_rate_mhd"  # lower is better: its own axis
SEG_SCORES = tuple(m for m in SEG_METRICS if m != SEG_FP)
SEG_FP_TOP = 0.2  # the axis reaches further only for an interval past it
SEG_FP_STEP = 0.05
LOWER_BETTER = "lower is better"
AE_BARS = {"precision": "precision", "recall": "recall", "f1": "f1"}  # A1
BAR_LABEL = "bar"
LOW_BAR_LABEL = "bar on the lower bound"
FLOORS_LABEL = "A1 floors"  # fig_scores' dashes: A1's absolute floors alone
MARKS = (BAR_LABEL, LOW_BAR_LABEL, FLOORS_LABEL)  # after the methods in a legend
COLOURS = {
    "ae_xpower": "#d62728",
    "ae_seg": "#d62728",
    "seldnet": "#1f77b4",
    "tokeye": "#2ca02c",
    "recipe": "#ff7f0e",
    "source": "#9467bd",
    "uci": "#8c564b",
    "always": "#7f7f7f",
}
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


def _bars(
    ax,
    groups: Sequence[str],
    methods: Sequence[str],
    get: Callable[[str, str], tuple[float, float, float]],
    names: dict[str, str],
) -> None:
    """Grouped bars with interval whiskers; `get(method, group)` is an interval.
    The pseudo-masks' sources are hatched."""
    width = 0.8 / len(methods)
    x = np.arange(len(groups))
    for i, m in enumerate(methods):
        v, lo, hi = np.array([get(m, g) for g in groups]).T
        at = x - 0.4 + (i + 0.5) * width
        ax.bar(
            at,
            v,
            width,
            color=COLOURS[m],
            label=names[m],
            hatch=HATCH if m in PSEUDO_SOURCES else None,
            edgecolor="white" if m in PSEUDO_SOURCES else None,
            linewidth=0,
        )
        whisker = np.nan_to_num(np.array([v - lo, hi - v]))
        ax.errorbar(
            at, v, yerr=whisker, fmt="none", ecolor="black", elinewidth=0.6, capsize=1
        )


def _dots(ax, ae: dict, methods: Sequence[str]) -> None:
    """Dot and whisker per method and metric, on an axis from `SCORE_FLOOR`."""
    width = 0.8 / len(methods)
    x = np.arange(len(METRICS))
    for i, m in enumerate(methods):
        v, lo, hi = np.array([interval(ae["methods"][m][g]) for g in METRICS]).T
        at = x - 0.4 + (i + 0.5) * width
        shown = np.maximum(v, SCORE_FLOOR)
        whisker = np.nan_to_num(
            np.array(
                [shown - np.maximum(lo, SCORE_FLOOR), np.maximum(hi, shown) - shown]
            )
        )
        whisker[:, v < SCORE_FLOOR] = 0
        ax.errorbar(
            at,
            shown,
            yerr=whisker,
            fmt=MARKERS[m],
            ms=3,
            color=COLOURS[m],
            ecolor=COLOURS[m],
            elinewidth=0.7,
            capsize=0,
            label=AE_NAMES[m],
            clip_on=False,
        )
        for a, value in zip(at, v, strict=True):
            if value < SCORE_FLOOR:
                ax.annotate(
                    f"{value:.2f}",
                    (a, SCORE_FLOOR),
                    xytext=(0, 4),  # clear of the marker drawn at the floor
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=FONT_PT - 1,
                    color=COLOURS[m],
                )
    bars = ae.get("bar_thresholds", {})
    first = True
    for k, metric in enumerate(METRICS):
        if AE_BARS.get(metric) in bars:
            ax.hlines(
                bars[AE_BARS[metric]],
                k - 0.45,
                k + 0.45,
                colors="black",
                linestyles="--",
                lw=0.6,
                label=FLOORS_LABEL if first else "_" + FLOORS_LABEL,
            )
            first = False


def _bars_last(handles: list, names: list) -> tuple[list, list]:
    """The methods first, then the bars' marks."""
    order = sorted(range(len(names)), key=lambda i: names[i] in MARKS)
    return [handles[i] for i in order], [names[i] for i in order]


def _said(x: float, spec: str = f".{DECIMALS}f") -> str:
    """`x` for a figure's text, its minus U+2212 as the ticks write it; a value
    that rounds to zero is unsigned."""
    text = format(abs(x), spec)
    return f"\u2212{text}" if x < 0 and float(text) != 0 else text


def _holds(key: str, lo: float, hi: float, thresholds: dict) -> bool:
    """Whether the paired difference `key` meets its part of the bar, by the
    test `table_differences` applies (`DIFFERENCES`)."""
    [test] = [d[5] for d in DIFFERENCES if d[1] == key]
    return test(lo, hi, thresholds)


def _decided(ae: dict, bar: str) -> bool | None:
    """The record's verdict on `bar`, None when it gives none."""
    return ae.get("bar", {}).get(bar)


def _head(ae: dict, bar: str) -> str:
    """`bar`'s verdict from the record, as in "A1 fail:"."""
    return f"{bar} {'pass' if _decided(ae, bar) else 'fail'}:"


def a1_said(ae: dict) -> str | None:
    """A1's verdict and what its floors cannot show: F1 − SELDnet's lower bound
    against its bar, and each floor the model's score misses; None when the
    record gives no verdict on A1."""
    if _decided(ae, "A1") is None:
        return None
    t = ae["bar_thresholds"]
    _, lo, hi = interval(ae["differences"]["f1_minus_seldnet"])
    sign = "\u2265" if _holds("f1_minus_seldnet", lo, hi, t) else "<"
    clauses = [
        (
            f"F1 \u2212 {AE_NAMES['seldnet']} lower bound {_said(lo)} {sign} "
            f"{_said(t['f1_vs_seldnet_low'], 'g')}"
        )
    ]
    model = ae["methods"]["ae_xpower"]
    for metric, key in AE_BARS.items():
        value = interval(model[metric])[0]
        if key in t and not value >= t[key]:
            clauses.append(f"{METRICS[metric]} {_said(value)} < {_said(t[key], 'g')}")
    return "\n".join([_head(ae, "A1"), *clauses])  # a panel a third of the page


def a2_said(ae: dict) -> str | None:
    """A2's verdict and both its clauses: the model's MHD rate against its bar,
    and MHD FP − SELDnet's upper bound against 0; None when the record gives
    no verdict on A2."""
    if _decided(ae, "A2") is None:
        return None
    t = ae["bar_thresholds"]
    rate = interval(ae["methods"]["ae_xpower"]["fp_rate_mhd"])[0]
    _, lo, hi = interval(ae["differences"]["mhd_fp_minus_seldnet"])
    under = "\u2264" if rate <= t["mhd_fp_rate"] else ">"
    below = "<" if _holds("mhd_fp_minus_seldnet", lo, hi, t) else "\u2265"
    return "\n".join(
        [
            (
                f"{_head(ae, 'A2')} MHD FP {_said(rate)} {under} "
                f"{_said(t['mhd_fp_rate'], 'g')}"
            ),
            f"MHD FP \u2212 {AE_NAMES['seldnet']} upper bound {_said(hi)} {below} 0",
        ]
    )


def draw_scores(ae: dict, stem: Path) -> Figure:
    """The phenomena's frame scores: AE across the top, the rest coming below."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 3.2), layout="constrained")
        grid = fig.add_gridspec(2, len(ORDER) - 1)
        axes = [fig.add_subplot(grid[0, :])]
        axes += [fig.add_subplot(grid[1, i]) for i in range(len(ORDER) - 1)]
        for ax, category in zip(axes, ORDER, strict=True):
            if category != AE:
                placeholder(ax, title(category))
                continue
            methods = [m for m in AE_METHODS if m in ae["methods"]]
            _dots(ax, ae, methods)
            for k in range(1, len(METRICS)):
                ax.axvline(k - 0.5, color="#dddddd", lw=0.5)
            ax.set_xticks(range(len(METRICS)), list(METRICS.values()))
            ax.set_xlim(-0.5, len(METRICS) - 0.5)
            ax.set_ylim(SCORE_FLOOR, 1)
            ax.set_ylabel(f"frame score (axis from {SCORE_FLOOR:g})")
            if said := a1_said(ae):
                ax.set_xlabel(said)
            n = ae["frames"]
            ax.set_title(f"{title(AE)}: {n['shots']} test shots, {n['scored']} frames")
            handles, names = _bars_last(*ax.get_legend_handles_labels())
        fig.legend(handles, names, loc="outside lower center", ncols=4)
        save(fig, stem)
    return fig


def draw_mhd(ae: dict, stem: Path) -> Figure:
    """Each AE method's false-positive rate on MHD frames and on the rest."""
    methods = [m for m in AE_METHODS if m in ae["methods"]]
    y = np.arange(len(methods))[::-1].astype(float)
    with style():
        fig = Figure(figsize=(COLUMN_IN, 2.4), layout="constrained")  # A2's lines
        ax = fig.subplots()
        for offset, key, colour, ink, name in (
            (0.18, "fp_rate_mhd", "#1f77b4", "white", "MHD frames"),
            (-0.18, "fp_rate_other", "#bbbbbb", "black", "other absent frames"),
        ):
            v, lo, hi = np.array([interval(ae["methods"][m][key]) for m in methods]).T
            past = v > MHD_CAP
            ax.barh(y + offset, np.minimum(v, MHD_CAP), 0.34, color=colour, label=name)
            whisker = np.nan_to_num(np.array([v - lo, hi - v]))
            whisker[:, past] = 0
            ax.errorbar(
                np.minimum(v, MHD_CAP),
                y + offset,
                xerr=whisker,
                fmt="none",
                ecolor="black",
                elinewidth=0.6,
            )
            for at, value in zip(y[past] + offset, v[past], strict=True):
                ax.text(
                    MHD_CAP * 0.99,
                    at,
                    f"{value:.2f}",
                    ha="right",
                    va="center",
                    color=ink,
                    fontsize=FONT_PT - 1,
                )
        bar = ae["bar_thresholds"]["mhd_fp_rate"]
        ax.axvline(bar, color="black", ls="--", lw=0.6)
        ax.set_yticks(y, [AE_NAMES[m] for m in methods])
        ax.set_xlim(0, MHD_CAP)
        label = f"false-positive rate (bar: MHD ≤ {bar:g}; axis to {MHD_CAP:g})"
        ax.set_xlabel("\n".join(x for x in (label, a2_said(ae)) if x))
        n = ae["frames"]
        ax.set_title(
            f"{n['mhd_absent']} MHD frames in {n['shots_with_mhd_absent']} test shots"
        )
        fig.legend(loc="outside lower center", ncols=2, frameon=False)
        save(fig, stem)
    return fig


def _seg_bars(ax, seg: dict, metrics: Sequence[str], drawn: dict) -> None:
    """The G bars on `metrics`' groups, each key once in the legend."""
    for i, metric in enumerate(metrics):
        for k, key in enumerate(SEG_BARS.get(metric, ())):
            if key not in seg["bar_thresholds"]:
                continue
            name = BAR_LABEL if k == 0 else LOW_BAR_LABEL
            ax.hlines(
                seg["bar_thresholds"][key],
                i - 0.45,
                i + 0.45,
                colors="black",
                linestyles="--" if k == 0 else ":",
                lw=0.6,
                label=name if not drawn[name] else "_" + name,
            )
            drawn[name] = True


def fp_top(seg: dict, methods: Sequence[str]) -> float:
    """`SEG_FP_TOP`, or the next `SEG_FP_STEP` past the highest value or upper
    interval end; one undefined (NaN) is left out, so it hides none after it."""
    ends = (v for m in methods for v in interval(seg["methods"][m][SEG_FP])[::2])
    highest = max((v for v in ends if not math.isnan(v)), default=0.0)
    steps = math.ceil(round(highest / SEG_FP_STEP, 6))
    return max(SEG_FP_TOP, steps * SEG_FP_STEP)


def draw_segmentation(seg: dict, stem: Path) -> Figure:
    """The segmentation's pixel and frame scores, and beside them its MHD
    false-positive rate on its own zoomed axis, each bar G1-G3 marked."""
    methods = [m for m in SEG_NAMES if m in seg["methods"]]

    def get(m: str, g: str) -> tuple[float, float, float]:
        return interval(seg["methods"][m][g])

    with style():
        fig = Figure(figsize=(COLUMN_IN, 2.3), layout="constrained")
        ax, fp = fig.subplots(1, 2, width_ratios=[len(SEG_SCORES), 1.25])
        drawn = {BAR_LABEL: False, LOW_BAR_LABEL: False}
        _bars(ax, list(SEG_SCORES), methods, get, SEG_NAMES)
        _seg_bars(ax, seg, SEG_SCORES, drawn)
        ax.set_xticks(range(len(SEG_SCORES)), [SEG_METRICS[m] for m in SEG_SCORES])
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        _bars(fp, [SEG_FP], methods, get, SEG_NAMES)
        _seg_bars(fp, seg, (SEG_FP,), drawn)
        fp.set_xticks([0], [SEG_METRICS[SEG_FP]])
        fp.set_xlim(-0.5, 0.5)
        fp.set_ylim(0, fp_top(seg, methods))
        fp.set_title(LOWER_BETTER, fontsize=FONT_PT - 1, style="italic")
        c = seg["counts"]
        fig.suptitle(
            f"{c['shots']} test shots: {c['ae_pixels']:,} AE pixels of "
            f"{c['scored_pixels']:,}",
            fontsize=FONT_PT,
        )
        found: dict[str, object] = {}
        for axes in (ax, fp):
            for handle, name in zip(*axes.get_legend_handles_labels(), strict=True):
                found.setdefault(name, handle)
        handles, names = _bars_last(list(found.values()), list(found))
        if any(m in PSEUDO_SOURCES for m in methods):
            handles.append(Patch(facecolor="white", edgecolor="black", hatch=HATCH))
            names.append(PSEUDO_LABEL)
        fig.legend(handles, names, loc="outside lower center", ncols=3)
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


def table_segmentation(seg: dict, reuse: str | None = None) -> str:
    """The segmentation scores; the comment ends with the reuse note, when its
    test is a second use of an earlier version's test shots
    (`labeler.paper.build.seg_reuse`)."""
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
    if reuse:
        comment += f"; {reuse}"
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
