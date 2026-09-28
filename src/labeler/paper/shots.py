"""One discharge as the interpreter shows it, and AE examples from the test shots.

`picture` is what a shot viewer shows for AE: the review store's first
cross-power row (R0 x V1), 0-250 kHz; the owner's frames; the chosen
`ae_xpower` model's P(AE) per 10 ms frame, run here on the CPU as the gallery
runs it; and the segmentation's points of interest (`poi.csv`). The owner's
frames are the model's own copy of the labels, `<candidate>/review/labels.csv`,
the ones it was trained and scored on (D18), never the live table the owner
keeps saving; `ae_shot` reads them from disk. `draw_interpreter` is the paper's
one-discharge figure: that spectrogram and its points of interest over one
track per catalog phenomenon, AE's holding the owner's frames above the
model's, the other five coming. `draw_examples` stacks a few test shots (the
model's `split.csv`), which `pick_examples` takes: the best, the median and the
worst F1 against the owner.

**The F1 of a shot** is over the 10 ms frames of the scored 0-2 s that the owner
called present or absent (`scored_f1`), the window the paper's headline F1 is
over; the gallery's `f1_vs_owner` is over the owner's whole window. The build
ranks by the 0-2 s F1 (`rank_keys`), so its picks can differ from a ranking by
the gallery's. The evaluation's own frames also need TokEye's record and the
source table's window (`labeler.ae.xpower.evaluate`); a shot's F1 here does not.

**The points of interest** are thin outlines, clipped to the axes, dashed and
fainter when the peak lies after the scored 2 s (`in_scored_window` false); only
the `LABEL_CAP` largest by `pixels` carry their `region` number (`poi.csv`). A
dashed line at 2 s marks the scored window whenever a shot runs past it.

**The interpreter's shot** (`interpreter_pick`, `INTERPRETER_RULE`) shows AE
turning off and on. Its pool is the test shots with a point of interest (all
test shots when none has one) where the owner calls frames absent after the
first frame they call present, inside 0-2 s (`owner_mixed`): the lead-in before
breakdown, absent on every shot, does not count. The pick is the best F1 at
three decimals in the pool, ties to fewer points of interest, then to the lower
shot number. Only when no shot has such an absent stretch does the same rule
run over all those test shots (`POOL_FALLBACK`).

Every legend lists only what some panel draws: it is built from the drawn
artists' own labels, so its keys have their style.
"""

from __future__ import annotations

import io
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..ae.xpower import EVENT
from ..ae.xpower.data import BAND_KHZ, CROSS_ROWS, store_rows, targets, window_frames
from ..ae.xpower.gallery import STATE_COLOURS
from ..ae.xpower.train import f1_of, frame_cells, load, probabilities, read_split
from ..config import Paths
from ..events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.review import labels
from ..events.review.labels import Label
from ..events.review.rows import Grid
from ..scoring.frames import FRAME_MS
from . import AE, COMING, FONT_PT, ORDER, PAGE_IN, save, style, title

MARGIN_MS = 50.0
PICTURE_LEVEL = 8
MODEL_COLOUR = "#222222"
POI_COLOUR = "#00e5ff"
POI_LW = 0.4
POI_ALPHA = 0.75
POI_AFTER_ALPHA = 0.4
LABEL_CAP = 10  # number at most the 10 largest points of interest (by pixels)
SCORED_MS = 2000.0  # the scored window is 0-2 s
SCORED_LABEL = "scored: 0-2 s"
F1_DECIMALS = 3
STATE_NAMES = {
    PRESENT: "present",
    UNCERTAIN: "uncertain",
    NOT_OBSERVABLE: "not observable",
}
POI_LABEL = "point of interest"
POI_AFTER_LABEL = "point of interest, after 2 s"
MODEL_LABEL = "model: present"
THRESHOLD_LABEL = "model threshold"
INTERPRETER_RULE = (
    "among the test shots with a point of interest (all test shots if none has "
    "one), those whose owner's AE track has absent frames after its first "
    f"present frame in 0-2 s; the best F1 over 0-2 s at {F1_DECIMALS} decimals, "
    "ties to fewer points of interest, then to the lower shot number; if no "
    "shot has such absent frames, the same over all those test shots"
)
POOL_GAP = "AE turns off after its onset in 0-2 s"
POOL_FALLBACK = "no shot has AE turning off after its onset in 0-2 s: all of them"
EXAMPLES_RULE = (
    "the reviewed test shots ranked by F1 over 0-2 s (ties by the lower shot "
    "number), taken evenly from the best to the worst"
)


def in_scored(first: int, n: int) -> np.ndarray:
    """Whether each frame `first .. first + n - 1` lies inside the scored 0-2 s."""
    start = (first + np.arange(n)) * FRAME_MS
    return (start >= 0) & (start + FRAME_MS <= SCORED_MS)


def scored_f1(prob, owner, threshold: float, *, first: int) -> float:
    """F1 over the frames of 0-2 s the owner called present or absent."""
    inside = in_scored(first, len(prob))
    return f1_of(frame_cells(np.asarray(prob)[inside], owner[inside], threshold))


def owner_mixed(owner, *, first: int) -> int:
    """The owner's absent frames in 0-2 s after their first present frame there:
    AE turning off, not the lead-in before breakdown. 0 without a present frame."""
    inside = np.asarray(owner)[in_scored(first, len(owner))]
    present = np.flatnonzero(inside == PRESENT)
    return int((inside[present[0] :] == ABSENT).sum()) if len(present) else 0


@dataclass(frozen=True)
class AEShot:
    """One shot's AE picture. `owner` and `prob` are per frame from `first`;
    `f1` is over 0-2 s, `f1_window` over the owner's whole window."""

    shot: int
    split: str
    grid: Grid
    image: np.ndarray  # (n_y, n) bytes: R0 x V1 at `PICTURE_LEVEL`
    y0: float
    dy: float
    first: int
    owner: np.ndarray
    prob: np.ndarray
    threshold: float
    f1: float
    f1_window: float = float("nan")
    boxes: tuple[dict, ...] = ()

    @property
    def edges(self) -> np.ndarray:
        return (self.first + np.arange(len(self.prob) + 1)) * FRAME_MS


@dataclass(frozen=True)
class ShotScore:
    """A test shot's rank keys: F1 over 0-2 s, and the owner's absent frames
    after their AE's onset in 0-2 s (`owner_mixed`)."""

    shot: int
    f1: float
    f1_window: float
    gap: int


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """`(start, stop)` of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flags, np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


@dataclass(frozen=True)
class Model:
    """The chosen model, loaded once: the network, its saved record and split."""

    net: object
    blob: Mapping
    split: Mapping[int, str]

    @property
    def threshold(self) -> float:
        return float(self.blob["threshold"])

    @classmethod
    def load(cls, model_file, split: Mapping[int, str] | None = None) -> Model:
        """`model_file` a path or file object; the split beside it by default."""
        net, blob = load(model_file)
        if split is None:
            split = read_split(Path(model_file).parent / "split.csv")
        return cls(net, blob, split)


def test_shots(split: Mapping[int, str]) -> list[int]:
    """The model's test shots, as its evaluation scores them."""
    return sorted(int(s) for s, which in split.items() if which == "test")


def picture(
    shot: int, *, label: Label, model: Model, store, boxes: Sequence[dict] = ()
) -> AEShot:
    """One labelled shot with `model` run over its review `store`: a path, or
    the file's bytes as the build read them."""

    def rows(level: int = 1):
        source = io.BytesIO(store) if isinstance(store, bytes) else store
        return store_rows(source, level=level)

    first, n = window_frames(label.window)
    prob, _ = probabilities(model.net, rows(), first, n, band=model.blob["band_khz"])
    grid, values, y0, dy = rows(PICTURE_LEVEL)
    owner = targets(label, first, n)
    return AEShot(
        shot=shot,
        split=model.split.get(shot, ""),
        grid=grid,
        image=values[0],
        y0=y0,
        dy=dy,
        first=first,
        owner=owner,
        prob=prob,
        threshold=model.threshold,
        f1=scored_f1(prob, owner, model.threshold, first=first),
        f1_window=f1_of(frame_cells(prob, owner, model.threshold)),
        boxes=tuple(boxes),
    )


def boxes_of(poi: pd.DataFrame | None, shot: int) -> tuple[dict, ...]:
    """The shot's points of interest as records."""
    return () if poi is None else tuple(poi[poi.shot == shot].to_dict("records"))


def rank_keys(s: AEShot) -> ShotScore:
    """A drawn shot's rank keys."""
    return ShotScore(
        shot=s.shot,
        f1=s.f1,
        f1_window=s.f1_window,
        gap=owner_mixed(s.owner, first=s.first),
    )


def ae_shot(
    paths: Paths, shot: int, *, model_file: Path, poi: pd.DataFrame | None = None
) -> AEShot:
    """A shot in the copy of the labels saved beside `model_file` (D18), with that
    model run over its store."""
    label = labels.read_saved(Path(model_file).parent).get(shot)
    if label is None:
        raise KeyError(f"{shot}: the owner has not saved an AE label")
    return picture(
        shot,
        label=label,
        model=Model.load(model_file),
        store=paths.spectrogram_file(EVENT, shot),
        boxes=boxes_of(poi, shot),
    )


def _ranked(f1: Mapping[int, float]) -> list[int]:
    """The shots with an F1, best first, ties by the lower shot."""
    scored = [(float(v), int(s)) for s, v in f1.items() if not np.isnan(v)]
    return [s for _, s in sorted(scored, key=lambda x: (-x[0], x[1]))]


def pick_examples(f1: Mapping[int, float], n: int = 3) -> list[int]:
    """`n` test shots spread from the best F1 to the worst (`EXAMPLES_RULE`);
    `f1` maps each test shot to its F1 over 0-2 s."""
    ranked = _ranked(f1)
    if len(ranked) <= n:
        return ranked
    if n <= 1:
        return ranked[: max(n, 0)]
    step = (len(ranked) - 1) / (n - 1)
    return list(dict.fromkeys(ranked[round(i * step)] for i in range(n)))


def interpreter_pick(
    f1: Mapping[int, float],
    poi: pd.DataFrame | None,
    gaps: Mapping[int, int] | None = None,
) -> dict:
    """The shot `INTERPRETER_RULE` picks, the branch that fired and its pool:
    each pool shot's F1, points of interest and `owner_mixed` count. `f1` maps
    each test shot to its F1 over 0-2 s, `gaps` to its `owner_mixed`."""
    ranked = _ranked(f1)
    if not ranked:
        raise ValueError("no test shot has an F1")
    gaps = gaps or {}
    points = (
        pd.Series(dtype=int) if poi is None else poi.shot.astype(int).value_counts()
    )
    marked = [s for s in ranked if s in points.index] or ranked
    pool = [s for s in marked if gaps.get(s, 0) > 0]
    branch = POOL_GAP if pool else POOL_FALLBACK
    pool = pool or marked

    def key(s: int) -> tuple:
        return (-round(float(f1[s]), F1_DECIMALS), int(points.get(s, 0)), s)

    return {
        "shot": min(pool, key=key),
        "branch": branch,
        "pool": {
            str(s): {
                "f1": round(float(f1[s]), 4),
                "poi": int(points.get(s, 0)),
                "absent_after_onset": int(gaps.get(s, 0)),
            }
            for s in sorted(pool)
        },
    }


def _spectrogram(ax, s: AEShot) -> None:
    extent = (
        s.grid.t0_ms,
        s.grid.t0_ms + s.grid.n * s.grid.dt_ms,
        s.y0 - s.dy / 2,
        s.y0 + (s.image.shape[0] - 0.5) * s.dy,
    )
    ax.imshow(
        s.image,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax.axhline(BAND_KHZ[0], color="white", lw=0.6, ls="--")
    ax.set_ylim(0, 250)
    ax.set_ylabel(f"{CROSS_ROWS[0].replace('x', ' × ')}\nkHz")


def _scored(box: dict) -> bool:
    return bool(box.get("in_scored_window", True))


def numbered(boxes: Sequence[dict], xlim, ylim) -> list[dict]:
    """The `LABEL_CAP` largest points (by `pixels`) whose number would sit inside
    the axes, at the top left of the box."""
    (x0, x1), (y0, y1) = sorted(xlim), sorted(ylim)
    inside = [
        b
        for b in boxes
        if x0 <= float(b["t_start_ms"]) <= x1 and y0 <= float(b["f_hi_khz"]) <= y1
    ]
    return sorted(inside, key=lambda b: -float(b.get("pixels", 0)))[:LABEL_CAP]


def _boxes(ax, s: AEShot) -> None:
    """Thin clipped outlines; the `LABEL_CAP` largest numbered. Call it after the
    axes' limits are set."""
    labelled = {POI_LABEL: False, POI_AFTER_LABEL: False}
    for box in s.boxes:
        inside = _scored(box)
        name = POI_LABEL if inside else POI_AFTER_LABEL
        ax.add_patch(
            Rectangle(
                (box["t_start_ms"], box["f_lo_khz"]),
                box["t_end_ms"] - box["t_start_ms"],
                box["f_hi_khz"] - box["f_lo_khz"],
                fill=False,
                ec=POI_COLOUR,
                lw=POI_LW,
                ls="-" if inside else "--",
                alpha=POI_ALPHA if inside else POI_AFTER_ALPHA,
                clip_on=True,
                label=name if not labelled[name] else "_" + name,
            )
        )
        labelled[name] = True
    for box in numbered(s.boxes, ax.get_xlim(), ax.get_ylim()):
        x, y = float(box["t_start_ms"]), float(box["f_hi_khz"])
        ax.text(
            x,
            y,
            f"{int(box['region'])}",
            color=POI_COLOUR,
            fontsize=FONT_PT - 1,
            ha="left",
            va="top",
            alpha=POI_ALPHA if _scored(box) else POI_AFTER_ALPHA,
            clip_on=True,
        )


def _scored_line(axes, s: AEShot, *, dark: Collection) -> None:
    """A dashed line at 2 s on each axis, labelled on the first, when the shot
    runs past it."""
    if s.edges[-1] <= SCORED_MS:
        return
    for k, ax in enumerate(axes):
        colour = "white" if ax in dark else "black"
        ax.axvline(SCORED_MS, color=colour, lw=0.6, ls="--", label="_scored")
        if k == 0:
            ax.text(
                SCORED_MS,
                0.97,
                SCORED_LABEL,
                transform=ax.get_xaxis_transform(),
                ha="right",
                va="top",
                color=colour,
                fontsize=FONT_PT - 1,
                clip_on=True,
            )


def _owner_bars(ax, s: AEShot, y: tuple[float, float]) -> None:
    edges = s.edges
    for state, colour in STATE_COLOURS.items():
        spans = [(edges[a], edges[b] - edges[a]) for a, b in runs(s.owner == state)]
        if spans:
            ax.broken_barh(
                spans, y, color=colour, lw=0, label=f"owner: {STATE_NAMES[state]}"
            )


def _model_bars(ax, s: AEShot, y: tuple[float, float]) -> None:
    edges = s.edges
    spans = [(edges[a], edges[b] - edges[a]) for a, b in runs(s.prob >= s.threshold)]
    if spans:
        ax.broken_barh(spans, y, color=MODEL_COLOUR, lw=0, label=MODEL_LABEL)


LEGEND_ORDER = (
    *(f"owner: {name}" for name in STATE_NAMES.values()),
    MODEL_LABEL,
    THRESHOLD_LABEL,
    POI_LABEL,
    POI_AFTER_LABEL,
)


def _legend(fig: Figure) -> None:
    """One key per label some panel draws, with that artist's own style."""
    found: dict[str, object] = {}
    for ax in fig.axes:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            found.setdefault(name, handle)
    names = sorted(
        found, key=lambda n: LEGEND_ORDER.index(n) if n in LEGEND_ORDER else 99
    )
    if names:
        fig.legend(
            [found[n] for n in names],
            names,
            loc="outside lower center",
            ncols=min(len(names), 5),
        )


def draw_interpreter(s: AEShot, stem: Path) -> Figure:
    """The spectrogram with AE's points of interest over a track per phenomenon."""
    t0, t1 = s.edges[0] - MARGIN_MS, s.edges[-1] + MARGIN_MS
    with style():
        fig = Figure(figsize=(PAGE_IN, 3.4), layout="constrained")
        spec, *tracks = fig.subplots(
            1 + len(ORDER),
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [4] + [0.45] * len(ORDER)},
        )
        _spectrogram(spec, s)
        spec.set_xlim(t0, t1)
        _boxes(spec, s)
        spec.set_title(f"shot {s.shot}: what the interpreter marks")
        for ax, category in zip(tracks, ORDER, strict=True):
            ax.set_ylim(0, 2)
            ax.set_yticks([])
            ax.set_ylabel(title(category), rotation=0, ha="right", va="center")
            if category != AE:
                ax.text(
                    0.5,
                    0.5,
                    COMING,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#888888",
                    style="italic",
                )
                continue
            _owner_bars(ax, s, (1.05, 0.9))
            _model_bars(ax, s, (0.05, 0.9))
        _scored_line([spec, *tracks], s, dark={spec})
        for ax in tracks[:-1]:
            ax.tick_params(bottom=False)
        tracks[-1].set_xlabel("time (ms)")
        _legend(fig)
        save(fig, stem)
    return fig


def draw_examples(shots: Sequence[AEShot], stem: Path) -> Figure:
    """Each shot: the spectrogram, the owner's strip, and P(AE) with the model's
    present frames shaded; each title gives the shot's F1 over 0-2 s."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 2.0 * len(shots) + 0.3), layout="constrained")
        axes = fig.subplots(
            3 * len(shots), 1, gridspec_kw={"height_ratios": [3, 0.35, 1] * len(shots)}
        )
        for k, s in enumerate(shots):
            spec, strip, model = axes[3 * k : 3 * k + 3]
            t0, t1 = s.edges[0] - MARGIN_MS, s.edges[-1] + MARGIN_MS
            for ax in (spec, strip, model):
                ax.set_xlim(t0, t1)
            _spectrogram(spec, s)
            _boxes(spec, s)
            spec.set_title(f"shot {s.shot} ({s.split}): F1 (0-2 s) {s.f1:.2f}")
            _owner_bars(strip, s, (0, 1))
            strip.set_yticks([])
            strip.set_ylabel("owner", rotation=0, ha="right", va="center")
            centres = s.edges[:-1] + FRAME_MS / 2
            for i, (a, b) in enumerate(runs(s.prob >= s.threshold)):
                model.axvspan(
                    s.edges[a],
                    s.edges[b],
                    color=MODEL_COLOUR,
                    alpha=0.15,
                    lw=0,
                    label=MODEL_LABEL if i == 0 else "_" + MODEL_LABEL,
                )
            model.plot(centres, s.prob, color=MODEL_COLOUR, lw=0.6, label="_P(AE)")
            model.axhline(
                s.threshold, color=MODEL_COLOUR, lw=0.5, ls=":", label=THRESHOLD_LABEL
            )
            model.set_ylim(0, 1)
            model.set_ylabel("P(AE)")
            _scored_line([spec, strip, model], s, dark={spec})
            for ax in (spec, strip):
                ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel("time (ms)")
        _legend(fig)
        save(fig, stem)
    return fig
