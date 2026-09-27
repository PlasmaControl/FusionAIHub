"""How much the catalog holds per phenomenon: reviewed, positive, suggested.

For each catalog phenomenon: the shots a person reviewed, those with any
present span and their present time; how the model split the reviewed shots
(train, val, test); and the shots the extension suggested labels for, with those
it calls positive, by campaign year. AE's come from the owner's review
(`data/events/alfven_eigenmode/review/labels.csv`), the chosen `ae_xpower`
model's `split.csv` and the extension's `summary.csv`; the other five are
coming. A suggested shot is a suggestion, not a label (v1 spec §3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..ae.xpower.train import read_split
from ..events.catalog.states import PRESENT
from ..events.review import labels
from . import AE, COMING, ORDER, PAGE_IN, placeholder, save, style, title
from .scores import tabular

SPLITS = ("train", "val", "test")
BARS = {
    "reviewed": "#bbbbbb",
    "positive": "#d62728",
    "suggested": "#9ecae1",
    "suggested positive": "#1f77b4",
}
UNKNOWN_YEAR = 0


@dataclass(frozen=True)
class Counts:
    """One phenomenon's shots. `by_year` maps a campaign year to (suggested,
    suggested positive); `UNKNOWN_YEAR` holds shots without one."""

    reviewed: int
    positive: int
    present_s: float
    split: Mapping[str, int] = field(default_factory=dict)
    by_year: Mapping[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def suggested(self) -> int:
        return sum(n for n, _ in self.by_year.values())

    @property
    def suggested_positive(self) -> int:
        return sum(k for _, k in self.by_year.values())

    def bar(self, name: str) -> int:
        return {
            "reviewed": self.reviewed,
            "positive": self.positive,
            "suggested": self.suggested,
            "suggested positive": self.suggested_positive,
        }[name]


def ae_counts(
    event_dir: Path, split_csv: Path | None, summary_csv: Path | None
) -> Counts:
    """AE's counts; a missing `split_csv` or `summary_csv` leaves its part empty."""
    saved = labels.read_saved(event_dir)
    positive = sum(
        any(c == PRESENT for _, _, c in label.intervals) for label in saved.values()
    )
    present_ms = sum(
        b - a for label in saved.values() for a, b, c in label.intervals if c == PRESENT
    )
    split = dict.fromkeys(SPLITS, 0)
    if split_csv is not None and Path(split_csv).is_file():
        for shot, which in read_split(split_csv).items():
            if shot in saved:
                split[which] = split.get(which, 0) + 1
    by_year = {}
    if summary_csv is not None and Path(summary_csv).is_file():
        summary = pd.read_csv(summary_csv)
        years = summary["year"].fillna(UNKNOWN_YEAR).astype(int)
        for year, group in summary.groupby(years):
            by_year[int(year)] = (len(group), int((group.present_frames > 0).sum()))
    return Counts(len(saved), positive, round(present_ms / 1000, 3), split, by_year)


def draw_coverage(counts: Mapping[str, Counts], stem: Path) -> Figure:
    """Shots per phenomenon (log scale), and AE's suggestions by campaign year."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 2.3), layout="constrained")
        left, right = fig.subplots(1, 2, width_ratios=[3, 2])
        x = np.arange(len(ORDER))
        width = 0.8 / len(BARS)
        for i, (name, colour) in enumerate(BARS.items()):
            values = [counts[c].bar(name) if c in counts else 0 for c in ORDER]
            shown = [v if v > 0 else np.nan for v in values]  # nothing on a log axis
            bars = left.bar(
                x - 0.4 + (i + 0.5) * width, shown, width, color=colour, label=name
            )
            left.bar_label(
                bars, labels=[f"{v}" if v > 0 else "" for v in values], fontsize=4
            )
        for k, category in enumerate(ORDER):
            if category not in counts:
                left.text(
                    k,
                    0.5,
                    COMING,
                    transform=left.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    color="#888888",
                    style="italic",
                )
        left.set_yscale("log")
        left.set_ylim(bottom=1)
        left.set_xticks(x, [title(c) for c in ORDER])
        left.set_ylabel("shots")
        left.legend(frameon=False, ncols=2, loc="upper right")
        ae = counts.get(AE)
        if ae is None or not ae.by_year:
            placeholder(right, "AE suggestions by campaign year")
        else:
            years = sorted(ae.by_year, key=lambda y: (y == UNKNOWN_YEAR, y))
            names = ["?" if y == UNKNOWN_YEAR else str(y) for y in years]
            at = np.arange(len(years))
            right.bar(
                at,
                [ae.by_year[y][0] for y in years],
                color=BARS["suggested"],
                label="suggested",
            )
            right.bar(
                at,
                [ae.by_year[y][1] for y in years],
                color=BARS["suggested positive"],
                label="with AE",
            )
            right.set_xticks(at, names)
            right.set_title("AE suggestions by campaign year")
            right.set_ylabel("shots")
            right.legend(frameon=False)
        save(fig, stem)
    return fig


def table_datasets(counts: Mapping[str, Counts]) -> str:
    header = (
        "Phenomenon",
        "Reviewed",
        "Positive",
        "Present (s)",
        "Train / val / test",
        "Suggested",
        "Suggested positive",
    )
    rows = []
    for category in ORDER:
        c = counts.get(category)
        if c is None:
            rows.append(
                [
                    title(category),
                    rf"\multicolumn{{{len(header) - 1}}}{{c}}{{{COMING}}}",
                ]
            )
            continue
        rows.append(
            [
                title(category),
                str(c.reviewed),
                str(c.positive),
                f"{c.present_s:.1f}",
                " / ".join(str(c.split.get(s, 0)) for s in SPLITS),
                str(c.suggested),
                str(c.suggested_positive),
            ]
        )
    comment = (
        "Shots per phenomenon: reviewed by a person, with any present span, the "
        "model's split of the reviewed shots, and the extension's suggestions "
        "(not labels)"
    )
    return tabular(header, rows, comment)
