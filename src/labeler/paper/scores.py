"""Frame scores: the AE methods, their MHD check, and the segmentation.

`draw_scores` is the six-phenomenon grid: frame precision, recall and F1 with
95 % shot-bootstrap intervals, one panel per catalog phenomenon. AE's panel is
`models/ae_xpower/v1/evaluation.json` (`labeler.ae.xpower.evaluate`); the other
five are coming. `draw_mhd` is AE's alone: each method's false-positive rate on
MHD frames (the owner says absent and TokEye sees a 0-60 kHz line) beside its
rate on the other absent frames, against the bar. `draw_segmentation` is
`models/ae_seg/v1/evaluation.json` (`labeler.ae.seg.evaluate`), with each
metric's bar marked. `table_ae` and `table_segmentation` give the same numbers
as LaTeX `tabular`s, for the owner to caption.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from . import AE, COLUMN_IN, ORDER, PAGE_IN, placeholder, save, style, title

AE_NAMES = {
    "ae_xpower": "ae_xpower",
    "seldnet": "SELDnet",
    "tokeye": "TokEye",
    "source": "start table",
    "uci": "UCI windows",
    "always": "always",
}
FIG_METHODS = ("ae_xpower", "seldnet", "tokeye", "always")
METRICS = {"precision": "precision", "recall": "recall", "f1": "F1"}
SEG_NAMES = {"ae_seg": "ae_seg", "recipe": "recipe", "tokeye": "TokEye"}
SEG_METRICS = {
    "dice": "Dice",
    "frame_precision": "frame P",
    "frame_recall": "frame R",
    "frame_f1": "frame F1",
    "fp_rate_mhd": "MHD FP",
}
SEG_BARS = {
    "dice": "dice",
    "frame_precision": "frame_precision",
    "fp_rate_mhd": "mhd_fp_rate",
}
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
    """Grouped bars with interval whiskers; `get(method, group)` is an interval."""
    width = 0.8 / len(methods)
    x = np.arange(len(groups))
    for i, m in enumerate(methods):
        v, lo, hi = np.array([get(m, g) for g in groups]).T
        at = x - 0.4 + (i + 0.5) * width
        ax.bar(at, v, width, color=COLOURS[m], label=names[m])
        whisker = np.nan_to_num(np.array([v - lo, hi - v]))
        ax.errorbar(
            at, v, yerr=whisker, fmt="none", ecolor="black", elinewidth=0.6, capsize=1
        )


def draw_scores(ae: dict, stem: Path) -> Figure:
    """The six-phenomenon grid of frame scores; AE filled, the rest coming."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 3.2), layout="constrained")
        axes = fig.subplots(2, 3).ravel()
        for ax, category in zip(axes, ORDER, strict=True):
            if category != AE:
                placeholder(ax, title(category))
                continue
            methods = [m for m in FIG_METHODS if m in ae["methods"]]
            _bars(
                ax,
                list(METRICS),
                methods,
                lambda m, g: interval(ae["methods"][m][g]),
                AE_NAMES,
            )
            ax.set_xticks(range(len(METRICS)), list(METRICS.values()))
            ax.set_ylim(0, 1)
            ax.set_ylabel("frame score")
            n = ae["frames"]
            ax.set_title(f"{title(AE)}: {n['shots']} test shots, {n['scored']} frames")
            handles, names = ax.get_legend_handles_labels()
        fig.legend(handles, names, loc="outside lower center", ncols=len(names))
        save(fig, stem)
    return fig


def draw_mhd(ae: dict, stem: Path) -> Figure:
    """Each AE method's false-positive rate on MHD frames and on the rest."""
    methods = [m for m in AE_NAMES if m in ae["methods"]]
    y = np.arange(len(methods))[::-1].astype(float)
    with style():
        fig = Figure(figsize=(COLUMN_IN, 2.1), layout="constrained")
        ax = fig.subplots()
        for offset, key, colour, name in (
            (0.18, "fp_rate_mhd", "#1f77b4", "MHD frames"),
            (-0.18, "fp_rate_other", "#bbbbbb", "other absent frames"),
        ):
            v, lo, hi = np.array([interval(ae["methods"][m][key]) for m in methods]).T
            ax.barh(y + offset, v, 0.34, color=colour, label=name)
            whisker = np.nan_to_num(np.array([v - lo, hi - v]))
            ax.errorbar(
                v, y + offset, xerr=whisker, fmt="none", ecolor="black", elinewidth=0.6
            )
        bar = ae["bar_thresholds"]["mhd_fp_rate"]
        ax.axvline(bar, color="black", ls="--", lw=0.6)
        ax.set_yticks(y, [AE_NAMES[m] for m in methods])
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"false-positive rate (bar: MHD <= {bar:g})")
        n = ae["frames"]
        ax.set_title(
            f"{n['mhd_absent']} MHD frames in {n['shots_with_mhd_absent']} test shots"
        )
        ax.legend(loc="upper right", frameon=False)
        save(fig, stem)
    return fig


def draw_segmentation(seg: dict, stem: Path) -> Figure:
    """The segmentation's pixel and frame scores, each metric's bar dashed."""
    methods = [m for m in SEG_NAMES if m in seg["methods"]]
    with style():
        fig = Figure(figsize=(COLUMN_IN, 2.1), layout="constrained")
        ax = fig.subplots()
        _bars(
            ax,
            list(SEG_METRICS),
            methods,
            lambda m, g: interval(seg["methods"][m][g]),
            SEG_NAMES,
        )
        for i, metric in enumerate(SEG_METRICS):
            if metric in SEG_BARS:
                bar = seg["bar_thresholds"][SEG_BARS[metric]]
                ax.hlines(bar, i - 0.45, i + 0.45, colors="black", ls="--", lw=0.6)
        ax.set_xticks(range(len(SEG_METRICS)), list(SEG_METRICS.values()))
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        c = seg["counts"]
        ax.set_title(
            f"{c['shots']} test shots: {c['ae_pixels']:,} AE pixels of "
            f"{c['scored_pixels']:,}"
        )
        fig.legend(loc="outside lower center", ncols=len(methods))
        save(fig, stem)
    return fig


def _fmt(estimate: dict) -> str:
    v, lo, hi = interval(estimate)
    if math.isnan(v):
        return "--"
    if math.isnan(lo) or math.isnan(hi):
        return f"{v:.2f}"
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


def _tex(text: str) -> str:
    return text.replace("_", r"\_")


def tabular(header: Sequence[str], rows: Sequence[Sequence[str]], comment: str) -> str:
    """A booktabs `tabular`, a `%` comment line above it."""
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


def table_ae(ae: dict) -> str:
    keys = ("precision", "recall", "f1", "fp_rate_mhd", "fp_rate_other")
    rows = [
        [_tex(AE_NAMES[m]), *(_fmt(ae["methods"][m][k]) for k in keys)]
        for m in AE_NAMES
        if m in ae["methods"]
    ]
    n, meta = ae["frames"], ae.get("meta", {})
    comment = (
        f"AE frame scores: {n['shots']} test shots, {n['scored']} frames "
        f"({n['present']} present, {n['mhd_absent']} MHD); 95% shot-bootstrap "
        f"intervals; {meta.get('candidate', '?')} at {meta.get('threshold', '?')}; "
        f"bar {_verdict(ae.get('bar', {}))}"
    )
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
        f"pixels ({c['ae_pixels']} AE), {c['frames']} frames; 95% shot-bootstrap "
        f"intervals; bar {_verdict(seg.get('bar', {}))}"
    )
    header = ("Method", "Dice", "Frame P", "Frame R", "Frame F1", "FP (MHD)")
    return tabular(header, rows, comment)
