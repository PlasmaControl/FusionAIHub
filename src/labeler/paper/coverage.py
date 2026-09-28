"""How much the catalog holds per phenomenon: reviewed, positive, suggested.

For each catalog phenomenon: the shots a person reviewed, those with any
present span and their present time; how the model split the reviewed shots
(train, val, test) and, apart, the reviewed shots in no split (saved after the
model was trained); and the shots the extension suggested labels for, with
those it calls positive, by campaign year. AE's come from the owner's live
review (`data/events/alfven_eigenmode/review/labels.csv`), the chosen
`ae_xpower` model's `split.csv` and the extension's `summary.csv`; the other
five are coming. So the reviewed count is the split's three plus "no split". A
suggested shot is a suggestion, not a label (v1 spec §3).

A cross-validated version (v2: its `chosen.json` names `folds_sha256`, or its
models directory holds `cv/folds.csv`) holds no shot out for validation: its
train shots are dealt into folds that choose the model. The panel draws its
train and test shots alone, with no validation bar, whatever its folds, and the
table's validation cell is `--`, its comment saying why. The train shots are
marked with the number of folds (the distinct `fold` values, `fold_count`) only
when the build has checked `cv/folds.csv` against `chosen.json`
(`build.folds_check`). A validation split's three bars are drawn as they always
were.

A part whose input is missing is not a zero: without `split.csv` (no model
chosen) or `summary.csv` (the extension did not run) its panel says "not run"
and the table prints `--`. Why it did not run goes in the build's manifest
(`partial`), not in the figure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..events.catalog.states import PRESENT
from ..events.review.labels import Label
from . import AE, COMING, FONT_PT, ORDER, PAGE_IN, placeholder, save, style, title
from .scores import tabular

SPLITS = ("train", "val", "test")
UNSPLIT = "no\nsplit"  # the split panel's tick, on two lines: clear of "test"
UNSPLIT_COLOUR = "#cccccc"
NOT_RUN = "not run"
MISSING = "--"
SHOT_BARS = {"reviewed": "#bbbbbb", "with a present span": "#d62728"}
PRESENT_COLOUR = "#d62728"
SPLIT_COLOUR = "#7f7f7f"
SUGGESTED = {"suggested": "#9ecae1", "suggested with AE": "#1f77b4"}
UNKNOWN_YEAR = 0
VALUE_PT = FONT_PT - 1


@dataclass(frozen=True)
class Counts:
    """One phenomenon's shots. `split` maps train/val/test to reviewed shots,
    `unsplit` counts the reviewed shots in none of them, and `by_year` maps a
    campaign year to (suggested, suggested positive), `UNKNOWN_YEAR` holding
    shots without one; None where the input is missing (not run).
    `cross_validated` says the train shots are dealt into folds, with no shot
    held out for validation, and `folds` is their number: None for a
    validation split, or for folds that could not be counted."""

    reviewed: int
    positive: int
    present_s: float
    split: Mapping[str, int] | None = None
    by_year: Mapping[int, tuple[int, int]] | None = None
    unsplit: int | None = None
    folds: int | None = None
    cross_validated: bool = False

    @property
    def suggested(self) -> int | None:
        if self.by_year is None:
            return None
        return sum(n for n, _ in self.by_year.values())

    @property
    def suggested_positive(self) -> int | None:
        if self.by_year is None:
            return None
        return sum(k for _, k in self.by_year.values())


def ae_counts(
    saved: Mapping[int, Label],
    model_split: Mapping[int, str] | None,
    summary: pd.DataFrame | None,
    *,
    folds: int | None = None,
    cross_validated: bool = False,
) -> Counts:
    """AE's counts from the owner's saved labels (`labels.read_saved`), the
    chosen model's split (`read_split`) and the extension's `summary.csv`; a
    missing split or summary (None) leaves its part None. `folds` is a
    cross-validated split's number of folds (`fold_count`), which makes it
    `cross_validated`; so does `cross_validated` alone, its folds not counted."""
    positive = sum(
        any(c == PRESENT for _, _, c in label.intervals) for label in saved.values()
    )
    present_ms = sum(
        b - a for label in saved.values() for a, b, c in label.intervals if c == PRESENT
    )
    split = unsplit = None
    if model_split is not None:
        split = dict.fromkeys(SPLITS, 0)
        for shot, which in model_split.items():
            if shot in saved:
                split[which] = split.get(which, 0) + 1
        unsplit = sum(shot not in model_split for shot in saved)
    by_year = None
    if summary is not None:
        years = summary["year"].fillna(UNKNOWN_YEAR).astype(int)
        by_year = {
            int(year): (len(group), int((group.present_frames > 0).sum()))
            for year, group in summary.groupby(years)
        }
    return Counts(
        len(saved),
        positive,
        round(present_ms / 1000, 3),
        split=split,
        by_year=by_year,
        unsplit=unsplit,
        folds=folds,
        cross_validated=cross_validated or folds is not None,
    )


def fold_count(folds: pd.DataFrame) -> int | None:
    """The number of folds in a `cv/folds.csv` (shot, split, fold): its
    distinct `fold` values, the test shots having none; None for a table with
    no `fold` column."""
    if "fold" not in folds.columns:
        return None
    return int(folds["fold"].dropna().nunique())


def _n_folds(n: int) -> str:
    return f"{n} fold{'' if n == 1 else 's'}"


def _splits(c: Counts) -> tuple[str, ...]:
    """The splits drawn and counted: train and test alone when the train shots
    are cross-validated, which holds no shot out for validation (whether or not
    their folds were counted); all three otherwise."""
    return ("train", "test") if c.cross_validated else SPLITS


def _split_tick(which: str, c: Counts) -> str:
    """A split's tick: cross-validated train shots name their folds on a
    second line, as "no split" is on two."""
    if which == "train" and c.folds is not None:
        return f"train\n({_n_folds(c.folds)})"
    return which


def _coming_rows(ax, rows: np.ndarray, counts: Mapping[str, Counts]) -> None:
    for y, category in zip(rows, ORDER, strict=True):
        if category not in counts:
            ax.text(
                0.5,
                y,
                COMING,
                transform=ax.get_yaxis_transform(),
                ha="center",
                va="center",
                color="#888888",
                style="italic",
            )


def _room(ax, top: float) -> None:
    """Room past the longest bar for its value."""
    ax.set_xlim(0, 1.3 * top if top > 0 else 1)


def _shots_panel(ax, counts: Mapping[str, Counts], rows: np.ndarray) -> None:
    height = 0.8 / len(SHOT_BARS)
    top = 0.0
    for i, (name, colour) in enumerate(SHOT_BARS.items()):
        at, values = [], []
        for y, category in zip(rows, ORDER, strict=True):
            c = counts.get(category)
            v = None if c is None else (c.reviewed, c.positive)[i]
            if v:  # nothing drawn, and no key, for a series without data
                at.append(y + 0.4 - (i + 0.5) * height)  # reviewed on top
                values.append(v)
        if values:
            bars = ax.barh(at, values, height, color=colour, label=name)
            ax.bar_label(bars, padding=1, fontsize=VALUE_PT)
            top = max(top, *values)
    _coming_rows(ax, rows, counts)
    _room(ax, top)
    ax.set_yticks(rows, [title(c) for c in ORDER])
    ax.set_ylim(rows.min() - 0.6, rows.max() + 0.6)
    ax.set_title("shots")


def _present_panel(ax, counts: Mapping[str, Counts], rows: np.ndarray) -> None:
    at, values = [], []
    for y, category in zip(rows, ORDER, strict=True):
        c = counts.get(category)
        if c is not None and c.present_s > 0:
            at.append(y)
            values.append(c.present_s)
    if values:
        bars = ax.barh(at, values, 0.5, color=PRESENT_COLOUR)
        ax.bar_label(
            bars, labels=[f"{v:.1f}" for v in values], padding=1, fontsize=VALUE_PT
        )
    _coming_rows(ax, rows, counts)
    _room(ax, max(values, default=0))
    ax.set_yticks(rows, [])
    ax.set_title("present time (s)")


def _split_panel(ax, ae: Counts | None) -> None:
    heading = f"{title(AE)}: model split"
    if ae is None or ae.split is None:
        placeholder(ax, heading, NOT_RUN)
        return
    splits = _splits(ae)
    values = [ae.split.get(s, 0) for s in splits] + [ae.unsplit or 0]
    colours = [SPLIT_COLOUR] * len(splits) + [UNSPLIT_COLOUR]
    bars = ax.bar(range(len(values)), values, color=colours)
    ax.bar_label(bars, padding=1, fontsize=VALUE_PT)
    ticks = [_split_tick(s, ae) for s in splits]
    ax.set_xticks(range(len(values)), [*ticks, UNSPLIT])
    ax.set_ylim(0, 1.2 * max(max(values), 1))
    ax.set_ylabel("reviewed shots")
    ax.set_title(heading)


def _years_panel(ax, ae: Counts | None) -> None:
    heading = f"{title(AE)} suggestions by year"
    if ae is None or ae.by_year is None:
        placeholder(ax, heading, NOT_RUN)
        return
    years = sorted(ae.by_year, key=lambda y: (y == UNKNOWN_YEAR, y))
    at = np.arange(len(years))
    for k, (name, colour) in enumerate(SUGGESTED.items()):
        ax.bar(at, [ae.by_year[y][k] for y in years], color=colour, label=name)
    ax.set_xticks(at, ["?" if y == UNKNOWN_YEAR else str(y) for y in years])
    ax.set_xlabel("campaign year")
    ax.set_ylabel("shots")
    ax.set_title(heading)
    if years:
        ax.legend(frameon=False)


def draw_coverage(counts: Mapping[str, Counts], stem: Path) -> Figure:
    """Per phenomenon the shots reviewed and positive, and their present time;
    AE's model split; and AE's suggestions by campaign year."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 2.3), layout="constrained")
        axes = fig.subplots(1, 4, width_ratios=[1.5, 1.1, 1, 1.3])
        rows = np.arange(len(ORDER))[::-1].astype(float)
        _shots_panel(axes[0], counts, rows)
        _present_panel(axes[1], counts, rows)
        axes[1].set_ylim(axes[0].get_ylim())
        _split_panel(axes[2], counts.get(AE))
        _years_panel(axes[3], counts.get(AE))
        handles, names = axes[0].get_legend_handles_labels()
        if handles:  # a key only for a series with data
            fig.legend(handles, names, loc="outside lower left", ncols=len(names))
        save(fig, stem)
    return fig


def _cell(value) -> str:
    return MISSING if value is None else str(value)


def table_datasets(counts: Mapping[str, Counts]) -> str:
    header = (
        "Phenomenon",
        "Reviewed",
        "Positive",
        "Present (s)",
        "Train / val / test",
        "No split",
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
        split = (
            MISSING
            if c.split is None
            else " / ".join(
                str(c.split.get(s, 0)) if s in _splits(c) else MISSING for s in SPLITS
            )
        )
        rows.append(
            [
                title(category),
                str(c.reviewed),
                str(c.positive),
                f"{c.present_s:.1f}",
                split,
                _cell(c.unsplit),
                _cell(c.suggested),
                _cell(c.suggested_positive),
            ]
        )
    comment = (
        "Shots per phenomenon: reviewed by a person, with any present span, the "
        "model's split of the reviewed shots and those in no split (reviewed = "
        "train + val + test + no split), and the extension's suggestions (not "
        f"labels); {MISSING} where that run has not happened"
    )
    for category in ORDER:
        c = counts.get(category)
        if c is not None and c.split is not None and c.cross_validated:
            over = "" if c.folds is None else f" over {_n_folds(c.folds)}"
            comment += (
                f"; {title(category)}'s train shots are cross-validated{over}, so "
                f"no shot is held out for validation ({MISSING})"
            )
    return tabular(header, rows, comment)
