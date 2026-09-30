"""How much the catalog holds per phenomenon: labelled, positive, suggested.

For each catalog phenomenon: the shots with a label ("labelled"), those with
any present span and their present time; how the model split them (train,
val, test); and, in the table alone, the shots the extension suggested labels
for, with those it calls positive. A suggested shot is a suggestion, not a
label (v1 spec §3). The figure's shots and present-time panels are on log
axes, their bars from 1 (the owner, 2026-09-29 23:51 and 2026-09-30 01:05).
The figure has no suggestions panel (the owner, 2026-09-29 23:51: "remove the
suggestions by year graphs").

AE's labels are the owner's live review
(`data/events/alfven_eigenmode/review/labels.csv`), so its labelled shots are
its reviewed ones; its split is the chosen `ae_xpower` model's `split.csv`,
with "no split" the reviewed shots in none of it (saved after the model was
trained), so AE's labelled = train + val + test + no split; its suggestions are
the extension's `summary.csv`.

Each frame-model phenomenon (`FRAME_SOURCES`: NTM, H-mode with L-mode, ELMing
and sawteeth) is counted from whichever of its inputs exist (D59). Its labels
are its original table's with the owner's reviews over them (F2): the labelled
and positive shots and the present time are its split's meta's
(`labelled_shots`, `positive_shots`, `present_s`, `frames.shots_meta_file`),
counted over every labelled shot, blind and left-out ones too (D60), never
re-read from the grids here. The owner's live review is counted apart
("Reviewed by the owner" in the table). Its split is the frame model's (train,
val and test only, F2: no owner split), of the labelled shots whose inputs are
on disk; the labelled shots in none of it are those `frames.shots` left out,
so the table's No split is `--` for it. Its suggestions come from the
application's `summary.csv`, and whether its model failed its primary bar (the
first criterion of its spec's bar: E1, H1, N1 or S1) or is effectively the
`always` baseline (`frames.evaluate`, F6) from the suggestion table's meta: its
split panel's heading says so ("bar not met", "≈ always") and the table puts a
dagger on its Suggested cell (F8). Sawteeth's labels are those of the
ece_sawtooth v3 detector (ECE and SXR crashes; D56 as amended), not a person's,
so its tick says "(detector)". The frame
phenomena's splits take a row of their own below AE's. A phenomenon with none of
these inputs is still "coming". Disruption is not a paper phenomenon
(`paper.LEFT_OUT`): it is neither counted nor drawn.

A cross-validated version (v2: its `chosen.json` names `folds_sha256`, or its
models directory holds `cv/folds.csv`) holds no shot out for validation: its
train shots are dealt into folds that choose the model. The panel draws its
train and test shots alone, with no validation bar, whatever its folds, and the
table's validation cell is `--`, its comment saying why. The train shots are
marked with the number of folds (the distinct `fold` values, `fold_count`) only
when the build has checked `cv/folds.csv` against `chosen.json`
(`build.folds_check`). A validation split's three bars are drawn as they always
were.

A part whose input is missing is not a zero: without its split (no model
chosen, or a frame model's shots not split yet) or its split's meta, its panel
says "not run", and without that or `summary.csv` (the extension did not run)
the table prints `--`. Why it did not run goes in the build's manifest (`partial`),
not in the figure. A counted phenomenon with no labelled shot is a true zero:
its row of the shots panel says "none reviewed" (AE) or "none labelled".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .. import frames
from ..events.catalog.states import PRESENT
from ..events.review.labels import Label
from . import AE, COMING, FONT_PT, ORDER, PAGE_IN, placeholder, save, style, title
from .scores import tabular

SPLITS = ("train", "val", "test")
UNSPLIT = "no\nsplit"  # the split panel's tick, on two lines: clear of "test"
UNSPLIT_COLOUR = "#cccccc"
NOT_RUN = "not run"
MISSING = "--"
SHOT_BARS = {"labelled": "#bbbbbb", "with a present span": "#d62728"}
PRESENT_COLOUR = "#d62728"
SPLIT_COLOUR = "#7f7f7f"
UNKNOWN_YEAR = 0
VALUE_PT = FONT_PT - 1
ROW_IN = 2.3  # AE's row's height, inches: the whole figure's when it is alone
ROW_WIDTHS = (1.5, 1.1, 1)  # its panels: shots, present time, split
NONE_REVIEWED = "none reviewed"  # AE's, whose labels are its reviews
NONE_LABELLED = "none labelled"  # a frame phenomenon's
#: A frame split panel's marks (F8): its model failed its primary bar, or is
#: effectively the `always` baseline.
BAR_NOT_MET = "bar not met"
ALWAYS = "≈ always"
DAGGER = r"$^\dagger$"
FRAME_ROW_IN = 1.5  # the frame row's height, inches
#: The room past the longest bar for its number (`_room`): `ROOM` times it on a
#: linear axis, and `ROOM` times its decades on a log axis.
ROOM = 1.3
#: Where the log axis's bars start: 0 has no place on it.
BAR_FROM = 1


@dataclass(frozen=True)
class FrameSource:
    """Where a frame-model phenomenon's coverage comes from (D59)."""

    method: str  # the frame model (a `labeler.frames.SPECS` key)
    origin: str  # its original labels, as the table's comment names them
    also: str | None = None  # a state counted with it (H-mode's "L-mode", D38)
    detector: bool = False  # its labels are a detector's, not a person's

    @property
    def primary(self) -> str:
        """The model's primary bar: the first criterion of its spec's bar."""
        return next(iter(frames.SPECS[self.method].bar))


#: The frame-model phenomena, in `ORDER`'s order; never disruption (D61).
FRAME_SOURCES = {
    "neoclassical_tearing_mode": FrameSource("ntm_frames", "the tearing archive"),
    "high_confinement_mode": FrameSource(
        "hmode_frames", "Jalal Butt's table", "L-mode"
    ),
    "edge_localized_mode": FrameSource("elm_frames", "Hiro's table"),
    "sawtooth_oscillation": FrameSource(
        "sawtooth_frames",
        "the ece_sawtooth v3 detector (ECE and SXR crashes; D56 as amended)",
        detector=True,
    ),
}


def tick(category: str) -> str:
    """A phenomenon's tick in the shots panel: its title, and "(detector)" for
    one whose labels are a detector's (sawteeth's, D56 as amended)."""
    source = FRAME_SOURCES.get(category)
    if source is not None and source.detector:
        return f"{title(category)} (detector)"
    return title(category)


@dataclass(frozen=True)
class Counts:
    """One phenomenon's shots. `reviewed` counts the owner's saved shots, and
    `positive` and `present_s` the labelled shots with a present span and
    their present time. `split` maps train/val/test to shots, `unsplit` counts
    the reviewed shots in none of them, and `by_year` maps a campaign year to
    (suggested, suggested positive), `UNKNOWN_YEAR` holding shots without one;
    None where the input is missing (not run). `cross_validated` says the train
    shots are dealt into folds, with no shot held out for validation, and
    `folds` is their number: None for a validation split, or for folds that
    could not be counted.

    `frame` says these are a frame model's counts: `labelled`, `positive` and
    `present_s` are its split's meta's (None: not read), over the original
    labels with the owner's over them; `split` is of its own shots, and
    `unsplit` None. `bar_met` is whether its model met its primary bar and
    `always` whether it is effectively always, from the suggestion table's
    meta (None: not read)."""

    reviewed: int
    positive: int | None
    present_s: float | None
    split: Mapping[str, int] | None = None
    by_year: Mapping[int, tuple[int, int]] | None = None
    unsplit: int | None = None
    folds: int | None = None
    cross_validated: bool = False
    labelled: int | None = None  # a frame model's labelled shots; None: not read
    frame: bool = False  # a frame model's counts (split of its own shots)
    bar_met: bool | None = None  # its primary bar (E1, H1, N1, S1); None: unread
    always: bool | None = None  # effectively the always baseline; None: unread

    @property
    def labelled_shots(self) -> int | None:
        """The labelled shots: AE's reviewed ones, a frame model's `labelled`."""
        return self.labelled if self.frame else self.reviewed

    @property
    def marks(self) -> list[str]:
        """A frame model's marks (F8): `BAR_NOT_MET`, `ALWAYS`, in that order."""
        found = []
        if self.frame and self.bar_met is False:
            found.append(BAR_NOT_MET)
        if self.frame and self.always:
            found.append(ALWAYS)
        return found

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


def frame_split(table: pd.DataFrame, where: str = "a frame split") -> dict[int, str]:
    """A frame model's `frames.shots_file` (shot, split, positive, roster, owner)
    as {shot: split}; a ValueError naming `where` (the method and its file) and
    each split that is not one of `SPLITS` (v1's "owner" among them)."""
    odd = sorted({str(which) for which in table["split"]} - set(SPLITS))
    if odd:
        raise ValueError(
            f"{where}: a split is one of {', '.join(SPLITS)}, not {', '.join(odd)}"
        )
    return {
        int(shot): str(which)
        for shot, which in zip(table["shot"], table["split"], strict=True)
    }


#: The split meta's keys the coverage reads (`frames.shots`, F4).
META_KEYS = ("labelled_shots", "positive_shots", "present_s")
#: The suggestion table meta's keys the coverage reads (`frames.apply`, F6).
TABLE_META_KEYS = ("bar", "effectively_always")


def frame_counts(
    saved: Mapping[int, Label],
    model_split: Mapping[int, str] | None,
    summary: pd.DataFrame | None,
    *,
    meta: Mapping | None = None,
    table_meta: Mapping | None = None,
    primary: str | None = None,
) -> Counts:
    """A frame-model phenomenon's counts: `reviewed`, the owner's saved shots;
    the labelled and positive shots and the present time from its split's
    `meta` (`META_KEYS`; None without it); the count of each of `SPLITS` over
    ALL the model's shots (`frame_split`), every split a key, 0 included, None
    without the split; the suggestions from the application's `summary.csv`,
    as `ae_counts` counts them; and, from the suggestion table's `table_meta`,
    whether the `primary` criterion of its bar was met and whether it is
    effectively always. `unsplit` is None: its labelled shots in no split are
    those `frames.shots` left out. Never cross-validated."""
    split = None
    if model_split is not None:
        split = dict.fromkeys(SPLITS, 0)
        for which in model_split.values():
            split[which] += 1
    labelled = positive = present_s = None
    if meta is not None:
        missing = [k for k in META_KEYS if k not in meta]
        if missing:
            raise KeyError(f"the frame split's meta has no {', '.join(missing)}")
        labelled, positive = int(meta["labelled_shots"]), int(meta["positive_shots"])
        present_s = float(meta["present_s"])
    bar_met = always = None
    if table_meta is not None:
        missing = [k for k in TABLE_META_KEYS if k not in table_meta]
        if missing:
            raise KeyError(f"the suggestion table's meta has no {', '.join(missing)}")
        bar_met = bool(table_meta["bar"][primary])
        always = bool(table_meta["effectively_always"])
    return replace(
        ae_counts(saved, None, summary),
        positive=positive,
        present_s=present_s,
        split=split,
        unsplit=None,
        labelled=labelled,
        frame=True,
        bar_met=bar_met,
        always=always,
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


def _row_text(ax, y: float, text: str) -> None:
    """`text` across the row at `y`, grey italic: COMING, NOT_RUN,
    NONE_REVIEWED or NONE_LABELLED."""
    ax.text(
        0.5,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        ha="center",
        va="center",
        color="#888888",
        style="italic",
    )


def _coming_rows(ax, rows: np.ndarray, counts: Mapping[str, Counts]) -> None:
    for y, category in zip(rows, ORDER, strict=True):
        if category not in counts:
            _row_text(ax, y, COMING)


def _room(ax, top: float, log: bool = False) -> None:
    """Room past the longest bar for its value: `ROOM` times it on a linear
    axis from 0. On a log axis from `BAR_FROM`, multiplicative: `ROOM` times
    its decades, the right end `top ** ROOM` (top times `top ** (ROOM - 1)`),
    so the numbers get the same share of the panel; at least a decade's."""
    if log:
        ax.set_xlim(BAR_FROM, max(top, 10) ** ROOM)
    else:
        ax.set_xlim(0, ROOM * top if top > 0 else 1)


def _at_start(ax, y: float, text: str) -> None:
    """A value with no bar on a log axis: its number where `bar_label` would
    put it, at the axis's start."""
    ax.annotate(
        text,
        (BAR_FROM, y),
        xytext=(1, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=VALUE_PT,
    )


def _shots_panel(ax, counts: Mapping[str, Counts], rows: np.ndarray) -> None:
    """The labelled shots and those with a present span, on a log axis: each
    bar from `BAR_FROM` to its count, its number beside it. A count of 0 draws
    no bar but keeps its number; a count not read (None) draws neither."""
    ax.set_xscale("log")
    height = 0.8 / len(SHOT_BARS)
    top = 0.0
    for i, (name, colour) in enumerate(SHOT_BARS.items()):
        at, values, zeros = [], [], []
        for y, category in zip(rows, ORDER, strict=True):
            c = counts.get(category)
            v = None if c is None else (c.labelled_shots, c.positive)[i]
            if v is None:
                continue
            y_bar = y + 0.4 - (i + 0.5) * height  # labelled on top
            if v:
                at.append(y_bar)
                values.append(v)
            else:
                zeros.append(y_bar)
        if values:  # nothing drawn, and no key, for a series without a bar
            widths = [v - BAR_FROM for v in values]
            bars = ax.barh(at, widths, height, left=BAR_FROM, color=colour, label=name)
            numbers = [f"{v:,}" for v in values]
            ax.bar_label(bars, labels=numbers, padding=1, fontsize=VALUE_PT)
            top = max(top, *values)
        for y_bar in zeros:
            _at_start(ax, y_bar, "0")
    _coming_rows(ax, rows, counts)
    for y, category in zip(rows, ORDER, strict=True):
        c = counts.get(category)
        if c is None:
            continue
        if c.labelled_shots is None:
            _row_text(ax, y, NOT_RUN)  # a frame split with no meta
        elif c.labelled_shots == 0:  # counted: a true zero
            _row_text(ax, y, NONE_LABELLED if c.frame else NONE_REVIEWED)
    _room(ax, top, log=True)
    ax.set_yticks(rows, [tick(c) for c in ORDER])
    ax.set_ylim(rows.min() - 0.6, rows.max() + 0.6)
    ax.set_title("shots")


def _present_panel(ax, counts: Mapping[str, Counts], rows: np.ndarray) -> None:
    """The present time, on a log axis as the shots panel's: each bar from
    `BAR_FROM` s to its value, its number beside it. A time of at most
    `BAR_FROM` s draws no bar but keeps its number; none (0, or not read)
    draws neither."""
    ax.set_xscale("log")
    at, values = [], []
    for y, category in zip(rows, ORDER, strict=True):
        c = counts.get(category)
        if c is None or not c.present_s:
            continue
        if c.present_s > BAR_FROM:
            at.append(y)
            values.append(c.present_s)
        else:
            _at_start(ax, y, f"{c.present_s:,.1f}")
    if values:
        widths = [v - BAR_FROM for v in values]
        bars = ax.barh(at, widths, 0.5, left=BAR_FROM, color=PRESENT_COLOUR)
        ax.bar_label(
            bars, labels=[f"{v:,.1f}" for v in values], padding=1, fontsize=VALUE_PT
        )
    _coming_rows(ax, rows, counts)
    _room(ax, max(values, default=0), log=True)
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


def _frame_split_panel(ax, category: str, c: Counts | None) -> None:
    """A frame model's split of its own shots, train, val and test (F2: no
    owner split), the first bar on top. A model that failed its primary bar,
    or is effectively always, says so on the heading's second line (F8)."""
    heading = f"{title(category)}: model split"
    if c is not None and c.marks:
        heading += f"\n({', '.join(c.marks)})"
    if c is None:
        placeholder(ax, heading)
        return
    if c.split is None:
        placeholder(ax, heading, NOT_RUN)
        return
    values = [c.split.get(s, 0) for s in SPLITS]
    at = np.arange(len(values))
    bars = ax.barh(at, values, color=SPLIT_COLOUR)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=1, fontsize=VALUE_PT)
    ax.set_yticks(at, list(SPLITS))
    ax.invert_yaxis()
    _room(ax, max(values))
    ax.set_xlabel("shots")
    ax.set_title(heading)


def draw_coverage(counts: Mapping[str, Counts], stem: Path) -> Figure:
    """Per phenomenon the shots labelled and positive, on a log axis, and
    their present time, and AE's model split. While a frame-model phenomenon
    is counted, a row below: each one's model split, its heading marked when
    its model is (F8). One key, the shots panel's; no suggestions panel."""
    framed = any(category in counts for category in FRAME_SOURCES)
    with style():
        if framed:
            fig = Figure(figsize=(PAGE_IN, ROW_IN + FRAME_ROW_IN), layout="constrained")
            outer = fig.add_gridspec(2, 1, height_ratios=[ROW_IN, FRAME_ROW_IN])
            axes = outer[0].subgridspec(1, 3, width_ratios=ROW_WIDTHS).subplots()
        else:
            fig = Figure(figsize=(PAGE_IN, ROW_IN), layout="constrained")
            axes = fig.subplots(1, 3, width_ratios=ROW_WIDTHS)
        rows = np.arange(len(ORDER))[::-1].astype(float)
        _shots_panel(axes[0], counts, rows)
        _present_panel(axes[1], counts, rows)
        axes[1].set_ylim(axes[0].get_ylim())
        _split_panel(axes[2], counts.get(AE))
        if framed:
            splits = outer[1].subgridspec(1, len(FRAME_SOURCES)).subplots()
            for ax, category in zip(splits, FRAME_SOURCES, strict=True):
                _frame_split_panel(ax, category, counts.get(category))
        handles, names = axes[0].get_legend_handles_labels()
        if handles:  # a key only for a series with data
            fig.legend(handles, names, loc="outside lower left", ncols=len(names))
        save(fig, stem)
    return fig


def _cell(value) -> str:
    return MISSING if value is None else str(value)


def _listed(names: list[str]) -> str:
    """The names as a sentence lists them: "A", "A and B", "A, B and C"."""
    if len(names) < 2:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _frame_clause(counts: Mapping[str, Counts]) -> str:
    """The comment's clause for the counted frame-model phenomena, in `ORDER`:
    what their labels are, what their split and No split hold, and where their
    labels come from; empty while none is."""
    counted = [c for c in ORDER if c in FRAME_SOURCES and c in counts]
    if not counted:
        return ""
    tables = []
    for category in counted:
        source = FRAME_SOURCES[category]
        also = f" (and {source.also})" if source.also else ""
        tables.append(f"{title(category)}{also}: {source.origin}")
    return (
        f"; for {_listed([title(c) for c in counted])}, Labelled counts every shot "
        "of the original labels with the owner's reviews over them (blind and "
        "left-out shots too), Train, Val and Test the frame model's split of "
        "those with its inputs on disk, and No split is "
        f"{MISSING}: their other labelled shots are those the split left out "
        "(blind, or without the model's inputs on disk, or with no labelled bin "
        f"in their window); original labels: {', '.join(tables)}"
    )


def _dagger_clause(counts: Mapping[str, Counts]) -> str:
    """The comment's clause for the daggered Suggested cells (F8), naming each
    model's marks; empty while none is marked."""
    marked = [
        f"{title(category)} ({', '.join(counts[category].marks)})"
        for category in ORDER
        if category in counts and counts[category].marks
    ]
    if not marked:
        return ""
    return (
        f"; {DAGGER}: the frame model failed its primary bar (E1, H1, N1 or S1: "
        f"{BAR_NOT_MET}) or is effectively the always-present baseline ({ALWAYS}), "
        f"so its suggestions say little: {_listed(marked)}"
    )


def _suggested(c: Counts) -> str:
    """The Suggested cell, a dagger on a marked frame model's (F8)."""
    cell = _cell(c.suggested)
    return f"{cell}{DAGGER}" if c.marks and c.suggested is not None else cell


def _present(c: Counts) -> str:
    return MISSING if c.present_s is None else f"{c.present_s:.1f}"


def table_datasets(counts: Mapping[str, Counts]) -> str:
    header = (
        "Phenomenon",
        "Labelled",
        "Reviewed by the owner",
        "Positive",
        "Present (s)",
        "Train",
        "Val",
        "Test",
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
        if c.split is None:
            split = [MISSING] * len(SPLITS)
        else:
            split = [
                str(c.split.get(s, 0)) if s in _splits(c) else MISSING for s in SPLITS
            ]
        rows.append(
            [
                title(category),
                _cell(c.labelled_shots),
                str(c.reviewed),
                _cell(c.positive),
                _present(c),
                *split,
                MISSING if c.frame else _cell(c.unsplit),
                _suggested(c),
                _cell(c.suggested_positive),
            ]
        )
    comment = (
        (
            "Shots per phenomenon: labelled, reviewed by the owner, with any present "
            "span and their present time, the model's split (Train, Val, Test) and the "
            "extension's suggestions (not labels); "
            f"{MISSING} where that run has not happened; AE's labels are the owner's "
            "reviews, so its Labelled = Reviewed = Train + Val + Test + No split (the "
            "reviewed shots saved after the model was trained)"
        )
        + _frame_clause(counts)
        + _dagger_clause(counts)
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
