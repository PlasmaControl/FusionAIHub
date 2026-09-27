"""One discharge as the interpreter shows it, and AE examples from the test shots.

`ae_shot` reads what a shot viewer shows for AE: the review store's first
cross-power row (R0 x V1), 0-250 kHz; the owner's frames; the chosen
`ae_xpower` model's P(AE) per 10 ms frame, run here on the CPU as the gallery
runs it; and the segmentation's points of interest (`poi.csv`), each a box
labelled AE. `draw_interpreter` is the paper's one-discharge figure: that
spectrogram and its points of interest over one track per catalog phenomenon,
AE's holding the owner's frames above the model's, the other five coming.
`draw_examples` stacks a few test shots, which `pick_examples` takes from the
gallery's index: the best, the median and the worst F1 against the owner.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from ..ae.xpower import EVENT, event_dir
from ..ae.xpower.data import BAND_KHZ, CROSS_ROWS, store_rows, targets, window_frames
from ..ae.xpower.gallery import STATE_COLOURS
from ..ae.xpower.train import f1_of, frame_cells, load, probabilities, read_split
from ..config import Paths
from ..events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.review import labels
from ..events.review.rows import Grid
from ..scoring.frames import FRAME_MS
from . import AE, COMING, ORDER, PAGE_IN, save, style, title

MARGIN_MS = 50.0
PICTURE_LEVEL = 8
MODEL_COLOUR = "#222222"
POI_COLOUR = "#00e5ff"


@dataclass(frozen=True)
class AEShot:
    """One shot's AE picture. `owner` and `prob` are per frame from `first`."""

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
    boxes: tuple[dict, ...] = ()

    @property
    def edges(self) -> np.ndarray:
        return (self.first + np.arange(len(self.prob) + 1)) * FRAME_MS


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """`(start, stop)` of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flags, np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def ae_shot(
    paths: Paths, shot: int, *, model_file: Path, poi: pd.DataFrame | None = None
) -> AEShot:
    """A reviewed AE shot, with the model at `model_file` run over its store."""
    label = labels.read_saved(event_dir(paths)).get(shot)
    if label is None:
        raise KeyError(f"{shot}: the owner has not saved an AE label")
    model, blob = load(model_file)
    first, n = window_frames(label.window)
    store = paths.spectrogram_file(EVENT, shot)
    prob, _ = probabilities(model, store_rows(store), first, n, band=blob["band_khz"])
    grid, values, y0, dy = store_rows(store, level=PICTURE_LEVEL)
    owner = targets(label, first, n)
    threshold = float(blob["threshold"])
    boxes = () if poi is None else tuple(poi[poi.shot == shot].to_dict("records"))
    return AEShot(
        shot=shot,
        split=read_split(Path(model_file).parent / "split.csv").get(shot, ""),
        grid=grid,
        image=values[0],
        y0=y0,
        dy=dy,
        first=first,
        owner=owner,
        prob=prob,
        threshold=threshold,
        f1=f1_of(frame_cells(prob, owner, threshold)),
        boxes=boxes,
    )


def _tested(index: pd.DataFrame) -> pd.DataFrame:
    """The gallery's reviewed test shots with an F1, best first, ties by shot."""
    test = index[(index.group == "reviewed") & (index.split == "test")].copy()
    test["f1"] = pd.to_numeric(test.f1_vs_owner, errors="coerce")
    test = test[test.f1.notna()]
    return test.sort_values(["f1", "shot"], ascending=[False, True])


def pick_examples(index: pd.DataFrame, n: int = 3) -> list[int]:
    """`n` test shots spread from the best F1 to the worst."""
    ranked = [int(s) for s in _tested(index).shot]
    if len(ranked) <= n:
        return ranked
    if n <= 1:
        return ranked[: max(n, 0)]
    step = (len(ranked) - 1) / (n - 1)
    return list(dict.fromkeys(ranked[round(i * step)] for i in range(n)))


def interpreter_shot(index: pd.DataFrame, poi: pd.DataFrame | None) -> int:
    """The best-F1 test shot with a point of interest (else the best test shot)."""
    ranked = [int(s) for s in _tested(index).shot]
    if not ranked:
        raise ValueError("the gallery index has no reviewed test shot with an F1")
    marked = set() if poi is None else {int(s) for s in poi.shot}
    return next((s for s in ranked if s in marked), ranked[0])


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


def _boxes(ax, s: AEShot) -> None:
    for i, box in enumerate(s.boxes, 1):
        ax.add_patch(
            Rectangle(
                (box["t_start_ms"], box["f_lo_khz"]),
                box["t_end_ms"] - box["t_start_ms"],
                box["f_hi_khz"] - box["f_lo_khz"],
                fill=False,
                ec=POI_COLOUR,
                lw=0.8,
            )
        )
        ax.text(
            box["t_start_ms"],
            box["f_hi_khz"],
            f"AE {i}",
            color=POI_COLOUR,
            fontsize=5,
            va="bottom",
        )


def _owner_bars(ax, s: AEShot, y: tuple[float, float]) -> None:
    edges = s.edges
    for state, colour in STATE_COLOURS.items():
        spans = [(edges[a], edges[b] - edges[a]) for a, b in runs(s.owner == state)]
        if spans:
            ax.broken_barh(spans, y, color=colour, lw=0)


def _model_bars(ax, s: AEShot, y: tuple[float, float]) -> None:
    edges = s.edges
    spans = [(edges[a], edges[b] - edges[a]) for a, b in runs(s.prob >= s.threshold)]
    if spans:
        ax.broken_barh(spans, y, color=MODEL_COLOUR, lw=0)


def _keys(poi: bool) -> list[Patch]:
    keys = [
        Patch(color=STATE_COLOURS[PRESENT], label="owner: present"),
        Patch(color=STATE_COLOURS[UNCERTAIN], label="owner: uncertain"),
        Patch(color=STATE_COLOURS[NOT_OBSERVABLE], label="owner: not observable"),
        Patch(color=MODEL_COLOUR, label="model: present"),
    ]
    if poi:
        keys.append(Patch(fill=False, ec=POI_COLOUR, label="point of interest"))
    return keys


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
        for ax in tracks[:-1]:
            ax.tick_params(bottom=False)
        tracks[-1].set_xlabel("time (ms)")
        spec.set_xlim(t0, t1)
        fig.legend(handles=_keys(bool(s.boxes)), loc="outside lower center", ncols=5)
        save(fig, stem)
    return fig


def draw_examples(shots: Sequence[AEShot], stem: Path) -> Figure:
    """Each shot: the spectrogram, the owner's strip, and P(AE) with the model's
    present frames shaded."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 2.0 * len(shots) + 0.3), layout="constrained")
        axes = fig.subplots(
            3 * len(shots), 1, gridspec_kw={"height_ratios": [3, 0.35, 1] * len(shots)}
        )
        for k, s in enumerate(shots):
            spec, strip, model = axes[3 * k : 3 * k + 3]
            t0, t1 = s.edges[0] - MARGIN_MS, s.edges[-1] + MARGIN_MS
            _spectrogram(spec, s)
            _boxes(spec, s)
            spec.set_title(
                f"shot {s.shot} ({s.split}): F1 against the owner {s.f1:.2f}"
            )
            _owner_bars(strip, s, (0, 1))
            strip.set_yticks([])
            strip.set_ylabel("owner", rotation=0, ha="right", va="center")
            centres = s.edges[:-1] + FRAME_MS / 2
            for a, b in runs(s.prob >= s.threshold):
                model.axvspan(
                    s.edges[a], s.edges[b], color=MODEL_COLOUR, alpha=0.15, lw=0
                )
            model.plot(centres, s.prob, color=MODEL_COLOUR, lw=0.6)
            model.axhline(s.threshold, color=MODEL_COLOUR, lw=0.5, ls=":")
            model.set_ylim(0, 1)
            model.set_ylabel("P(AE)")
            for ax in (spec, strip, model):
                ax.set_xlim(t0, t1)
            for ax in (spec, strip):
                ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel("time (ms)")
        poi = any(s.boxes for s in shots)
        fig.legend(handles=_keys(poi), loc="outside lower center", ncols=5)
        save(fig, stem)
    return fig
