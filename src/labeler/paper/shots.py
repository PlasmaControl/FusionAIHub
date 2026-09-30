"""One discharge as the interpreter shows it, and AE examples from the test shots.

`picture` is what a shot viewer shows for AE: the review store's first
cross-power row (R0 x V1), 0-250 kHz; the owner's frames; the chosen
`ae_xpower` model's P(AE) per 10 ms frame, run here on the CPU as the gallery
runs it; and the segmentation's mask, SegNet run here too. The owner's
frames are the model's own copy of the labels, `<candidate>/review/labels.csv`,
the ones it was trained and scored on (D18), never the live table the owner
keeps saving; `ae_shot` reads them from disk. `draw_interpreter` is the paper's
one-discharge figure: that spectrogram and its mask over one
track per catalog phenomenon, AE's holding the owner's frames above the
model's. `draw_examples` stacks a few test shots (the
model's `split.csv`), which `pick_examples` takes: the best, the median and the
worst F1 against the owner; a whole-window version's picks hold at least one
shot whose owner window runs past 2 s (`LONG_MS`) when a test shot's does.

**The scored window** is the version's (`scored_ms`, from
`labeler.ae.xpower.scored_until`): 0-2 s for v1 and v2, whose TokEye masks end
at 2 s, and the owner's whole window for v3 (None). A shot's F1, the
interpreter's pick, its off-period, the rules' texts (`pick_texts`) and the
dashed line all use it; the module's constants (`INTERPRETER_RULE` and the rest)
are the 0-2 s texts, as v1's and v2's manifests have them.

**The F1 of a shot** is over the 10 ms frames of the scored window that the
owner called present or absent (`scored_f1`), the window the paper's headline
F1 is over. The gallery's `f1_vs_owner` is over the owner's whole window, as
v3's F1 is; the build ranks by the scored F1 (`rank_keys`), so a 0-2 s
version's picks can differ from a ranking by the gallery's. Every shot also
keeps its F1 over 0-2 s (`AEShot.f1_0_2s`) and over the owner's whole window
(`f1_window`), whatever its version. The evaluation's own frames also need
TokEye's record and, over 0-2 s, the source table's window, or, over the whole
window (v3), TokEye's whole-shot masks and an observed store
(`labeler.ae.xpower.evaluate`); a shot's F1 here needs none of them.

**The mask** is what the segmentation says, semantically: every pixel SegNet
calls AE, P(AE) at its threshold inside the band its blob records
(`labeler.ae.seg.train.blob_band`: 0-250 kHz for SegNet v2, 80-250 kHz for v1's
blob, which records none; `labeler.ae.seg.poi.ae_pixels`, the call its
evaluation scores, there only over the pseudo-mask's scored pixels), a
translucent fill with a thin outline. It is
run on the picture's own rows, at the store level it reads (`PICTURE_LEVEL`), so
it lies on the picture's pixels. Its regions, the points of interest of
`poi.csv`, are not drawn: they are a table for tools, which the interpreter's
pick still counts. A dashed line at 2 s marks a 0-2 s version's scored window
whenever a shot runs past it; a whole-window version's shots have none. No
picture has a line at 80 kHz: that floor was only the labelling view.

**The interpreter's shot** (`interpreter_pick`, `INTERPRETER_RULE`) shows AE
turning off and back on. Its pool is the test shots with a point of interest
(all test shots when none has one) where the owner calls at least
`MIN_GAP_FRAMES` whole consecutive 10 ms frames absent between the first and
the last frame they call present, inside the scored window (`longest_off`):
neither the lead-in before breakdown, absent on every shot, nor AE that turns
off and does not come back before the window ends (2 s for v1 and v2) counts,
nor fewer frames, such as one absent frame or
several short runs. A frame a present span touches is present, so 5 absent
frames are an off-period of at least 50 ms, but one of 50-59 ms off the 10 ms
grid can hold only 4 (one of 60 ms or more always holds 5). So the texts give
the rule in frames (`OFF_FRAMES`), and 50 ms only as what the frames imply
(`OFF_MS`), never of a shot the rule leaves out. The pick is the best F1 at
three decimals in the pool, ties to fewer points of interest, then to the lower
shot number. When that pool is empty the same rule runs over all those test
shots, and the branch says why: no test shot has that many absent frames
(`POOL_FALLBACK`), or the ones that do have no point of interest
(`POOL_UNMARKED`, which names them).

**The other four tracks** (F9) are what `paper.build` found for the shot,
passed in (`draw_interpreter`'s `tracks`), so the figure reads nothing: the
suggested states per frame where the shot is in that phenomenon's frame-model
suggestion table, drawn as `paper.roster` draws its tracks (`state_bars`,
keyed `SUGGESTED`'s, never the owner's); `NO_DATA` ("no filterscopes data",
say) where the shot lacks one of the model's required groups on disk;
`NOT_APPLIED` where the table exists without the shot; and `COMING` where
there is no table, or nothing was passed.

Every legend lists only what some panel draws: it is built from the drawn
artists' own labels, so its keys have their style (the mask's, an image's, as a
patch of its fill and outline). A mask with no pixel in its axes' view (one
only after 2 s on a shot drawn to 2 s, say) is drawn, clipped away, with no
label, so it adds no key that nothing in view shows.
"""

from __future__ import annotations

import io
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ..ae.seg import train as seg_train
from ..ae.seg.poi import ae_pixels
from ..ae.seg.pseudo import LEVEL as SEG_LEVEL
from ..ae.xpower import EVENT, scored_until
from ..ae.xpower.data import CROSS_ROWS, store_rows, targets, window_frames
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
PICTURE_LEVEL = SEG_LEVEL  # SegNet reads level 8, so its mask is on these pixels
MODEL_COLOUR = "#222222"
MASK_COLOUR = "#00e5ff"
MASK_ALPHA = 0.3  # the fill's; the chirps under it stay visible
MASK_LW = 0.4  # the outline's
SCORED_MS = 2000.0  # v1's and v2's scored window is 0-2 s (`scored_ms`)
#: A whole-window version's examples show a shot whose owner window runs past it.
LONG_MS = SCORED_MS


def window_name(until_ms: float | None) -> str:
    """The name of the scored window ending at `until_ms`: "0-2 s" for 2000 ms,
    "whole window" for None, the owner's whole window."""
    return "whole window" if until_ms is None else f"0-{until_ms / 1000:g} s"


SCORED_LABEL = f"scored: {window_name(SCORED_MS)}"
F1_DECIMALS = 3
STATE_NAMES = {
    PRESENT: "present",
    UNCERTAIN: "uncertain",
    NOT_OBSERVABLE: "not observable",
}
MASK_LABEL = "segmentation: AE"
#: The roster figure's fill, SegNet over its whole picture (`paper.roster`).
ROSTER_MASK_LABEL = "segmentation"
MASK_LABELS = (MASK_LABEL, ROSTER_MASK_LABEL)
MODEL_LABEL = "model: present"
THRESHOLD_LABEL = "model threshold"
#: A frame-model track's key (F9): what it holds is a suggestion.
SUGGESTED = "suggested: {}"
#: A track's text where the shot lacks one of the model's groups on disk.
NO_DATA = "no {} data"
NOT_APPLIED = "not applied to this shot"
TEXT_COLOUR = "#888888"
MIN_GAP_FRAMES = 5  # whole absent frames: 50 ms or more, but 50-59 ms can be 4
GAP_MS = MIN_GAP_FRAMES * FRAME_MS
OFF_MS = f"so an off-period of at least {GAP_MS} ms"  # what those frames imply


@dataclass(frozen=True)
class PickTexts:
    """The interpreter's and the examples' rules, said over one scored window."""

    off_frames: str  # what `longest_off` and `interpreter_pick` check
    interpreter: str
    gap: str
    fallback: str
    unmarked: str  # holds "{shots}", which `interpreter_pick` fills
    examples: str


def pick_texts(until_ms: float | None = SCORED_MS) -> PickTexts:
    """The rules' texts over the scored window ending at `until_ms` (0-2 s by
    default; None, the owner's whole window)."""
    where = "the whole window" if until_ms is None else window_name(until_ms)
    off = (
        f"at least {MIN_GAP_FRAMES} whole consecutive absent {FRAME_MS} ms frames "
        f"between the owner's first and last present frames in {where}"
    )
    return PickTexts(
        off_frames=off,
        interpreter=(
            "among the test shots with a point of interest (all test shots if none "
            f"has one), those with {off} (AE turning off and back on, {OFF_MS}); "
            f"the best F1 over {where} at {F1_DECIMALS} decimals, ties to fewer "
            "points of interest, then to the lower shot number; if there is none, "
            "the same over all those test shots"
        ),
        gap=f"AE turns off and back on: {off} ({OFF_MS})",
        fallback=(
            f"no test shot has {off}, so the pool is every test shot with a point "
            "of interest (every test shot if none has one)"
        ),
        unmarked=(
            "only test shots with no point of interest ({shots}) have "
            f"{off}, so the pool is every test shot with one"
        ),
        examples=(
            f"the reviewed test shots ranked by F1 over {where} (ties by the lower "
            "shot number), taken evenly from the best to the worst"
            + (
                ""
                if until_ms is not None
                else "; if none of those taken has an owner window running past "
                f"{LONG_MS:g} ms, the middle one is replaced by the shot nearest "
                "it in rank that has one"
            )
        ),
    )


# The 0-2 s texts, v1's and v2's, each the string it has always been.
_TEXTS = pick_texts(SCORED_MS)
OFF_FRAMES = _TEXTS.off_frames
INTERPRETER_RULE = _TEXTS.interpreter
POOL_GAP = _TEXTS.gap
POOL_FALLBACK = _TEXTS.fallback
POOL_UNMARKED = _TEXTS.unmarked
EXAMPLES_RULE = _TEXTS.examples


def scored_ms(version: str | None = None) -> float | None:
    """Where `version`'s scored window ends, ms (`xpower.scored_until`): 2 s for
    v1 and v2 (and no version), None for v3, the owner's whole window."""
    return scored_until(version)


def in_scored(first: int, n: int, until_ms: float | None = SCORED_MS) -> np.ndarray:
    """Whether each frame `first .. first + n - 1` lies inside the scored window,
    0 to `until_ms` (0-2 s by default); every frame for None, the owner's whole
    window."""
    if until_ms is None:
        return np.ones(n, dtype=bool)
    start = (first + np.arange(n)) * FRAME_MS
    return (start >= 0) & (start + FRAME_MS <= until_ms)


def scored_f1(
    prob, owner, threshold: float, *, first: int, until_ms: float | None = SCORED_MS
) -> float:
    """F1 over the frames of the scored window (`in_scored`) the owner called
    present or absent."""
    inside = in_scored(first, len(prob), until_ms)
    return f1_of(frame_cells(np.asarray(prob)[inside], owner[inside], threshold))


def longest_off(owner, *, first: int, until_ms: float | None = SCORED_MS) -> int:
    """The owner's longest off-period in the scored window (`in_scored`: 0-2 s
    by default, the owner's whole window for None), in frames: the longest run
    of consecutive absent frames between their first and last present frames
    there, AE turning off and back on. Neither the lead-in before breakdown nor
    AE that turns off and does not come back before the window ends counts, and
    an uncertain frame ends a run; 0 without a present frame."""
    inside = np.asarray(owner)[in_scored(first, len(owner), until_ms)]
    present = np.flatnonzero(inside == PRESENT)
    if not len(present):
        return 0
    span = inside[present[0] : present[-1]] == ABSENT
    return max((int(b - a) for a, b in runs(span)), default=0)


@dataclass(frozen=True)
class AEShot:
    """One shot's AE picture. `owner` and `prob` are per frame from `first`;
    `f1` is over the scored window ending at `scored_until_ms` (0-2 s by
    default; None, the owner's whole window), `f1_window` over the owner's whole
    window and `f1_0_2s` over 0-2 s."""

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
    mask: np.ndarray | None = None  # (n_y, n) bool: SegNet's AE pixels, or none run
    scored_until_ms: float | None = SCORED_MS

    @property
    def edges(self) -> np.ndarray:
        return (self.first + np.arange(len(self.prob) + 1)) * FRAME_MS

    @property
    def f1_0_2s(self) -> float:
        """The F1 over 0-2 s, whatever the scored window."""
        return scored_f1(self.prob, self.owner, self.threshold, first=self.first)


@dataclass(frozen=True)
class ShotScore:
    """A test shot's rank keys: F1 over its scored window, and the owner's
    longest off-period there, in frames (`longest_off`)."""

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


@dataclass(frozen=True)
class Segmentation:
    """SegNet, loaded once, and its saved record."""

    net: object
    blob: Mapping

    @property
    def threshold(self) -> float:
        return float(self.blob["threshold"])

    @classmethod
    def load(cls, model_file) -> Segmentation:
        """`model_file` a path or file object."""
        return cls(*seg_train.load(model_file))


def test_shots(split: Mapping[int, str]) -> list[int]:
    """The model's test shots, as its evaluation scores them."""
    return sorted(int(s) for s, which in split.items() if which == "test")


def picture(
    shot: int,
    *,
    label: Label,
    model: Model,
    store,
    segmentation: Segmentation | None = None,
    scored_until_ms: float | None = SCORED_MS,
) -> AEShot:
    """One labelled shot with `model`, and `segmentation` if given, run over its
    review `store`: a path, or the file's bytes as the build read them. Its F1
    is over the scored window ending at `scored_until_ms` (None: the owner's
    whole window), and the mask over the band SegNet's blob records."""

    def rows(level: int = 1):
        source = io.BytesIO(store) if isinstance(store, bytes) else store
        return store_rows(source, level=level)

    first, n = window_frames(label.window)
    prob, _ = probabilities(model.net, rows(), first, n, band=model.blob["band_khz"])
    grid, values, y0, dy = rows(PICTURE_LEVEL)
    owner = targets(label, first, n)
    mask = None
    if segmentation is not None:
        seg_prob = seg_train.predict(segmentation.net, values)
        band = seg_train.blob_band(segmentation.blob)
        mask = ae_pixels(seg_prob, segmentation.threshold, y0, dy, band=band)
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
        f1=scored_f1(
            prob, owner, model.threshold, first=first, until_ms=scored_until_ms
        ),
        f1_window=f1_of(frame_cells(prob, owner, model.threshold)),
        mask=mask,
        scored_until_ms=scored_until_ms,
    )


def rank_keys(s: AEShot) -> ShotScore:
    """A drawn shot's rank keys, over its scored window."""
    return ShotScore(
        shot=s.shot,
        f1=s.f1,
        f1_window=s.f1_window,
        gap=longest_off(s.owner, first=s.first, until_ms=s.scored_until_ms),
    )


def ae_shot(
    paths: Paths,
    shot: int,
    *,
    model_file: Path,
    seg_file: Path | None = None,
    scored_until_ms: float | None = SCORED_MS,
) -> AEShot:
    """A shot in the copy of the labels saved beside `model_file` (D18), with that
    model, and the segmentation in `seg_file` if given, run over its store and
    scored over the window ending at `scored_until_ms` (`picture`)."""
    label = labels.read_saved(Path(model_file).parent).get(shot)
    if label is None:
        raise KeyError(f"{shot}: the owner has not saved an AE label")
    return picture(
        shot,
        label=label,
        model=Model.load(model_file),
        store=paths.spectrogram_file(EVENT, shot),
        segmentation=None if seg_file is None else Segmentation.load(seg_file),
        scored_until_ms=scored_until_ms,
    )


def _ranked(f1: Mapping[int, float]) -> list[int]:
    """The shots with an F1, best first, ties by the lower shot."""
    scored = [(float(v), int(s)) for s, v in f1.items() if not np.isnan(v)]
    return [s for _, s in sorted(scored, key=lambda x: (-x[0], x[1]))]


def pick_examples(
    f1: Mapping[int, float], n: int = 3, long: Collection[int] | None = None
) -> list[int]:
    """`n` test shots spread from the best F1 to the worst (`EXAMPLES_RULE` over
    0-2 s, `pick_texts(until_ms).examples` for another window); `f1` maps each
    test shot to its F1 over the scored window. `long`, for a whole-window
    version, holds the shots whose owner window runs past LONG_MS: when no pick
    is one of them, the middle pick (index len // 2) is replaced by the `long`
    shot nearest it in rank (ties to the better rank), if a ranked shot is."""
    ranked = _ranked(f1)
    if len(ranked) <= n:
        picks = ranked
    elif n <= 1:
        picks = ranked[: max(n, 0)]
    else:
        step = (len(ranked) - 1) / (n - 1)
        picks = list(dict.fromkeys(ranked[round(i * step)] for i in range(n)))
    if long is None or not picks or any(s in long for s in picks):
        return picks
    others = [i for i, s in enumerate(ranked) if s in long and s not in picks]
    if not others:
        return picks
    middle = len(picks) // 2
    at = ranked.index(picks[middle])
    picks[middle] = ranked[min(others, key=lambda i: (abs(i - at), i))]
    return picks


def interpreter_pick(
    f1: Mapping[int, float],
    poi: pd.DataFrame | None,
    gaps: Mapping[int, int] | None = None,
    *,
    until_ms: float | None = SCORED_MS,
) -> dict:
    """The shot the interpreter rule picks (`INTERPRETER_RULE` over 0-2 s,
    `pick_texts(until_ms).interpreter` otherwise), the branch that fired and its pool:
    each pool shot's F1, points of interest and longest off-period in frames
    (`longest_off_frames`). `f1` maps each test shot to its F1 over the scored
    window ending at `until_ms` (a shot without one is left out), `gaps` to its
    `longest_off` there; a shot is back on after at least `MIN_GAP_FRAMES` whole
    absent frames (`OFF_FRAMES`). The branch is said over that window
    (`pick_texts`)."""
    ranked = _ranked(f1)
    if not ranked:
        raise ValueError("no test shot has an F1")
    texts = pick_texts(until_ms)
    gaps = gaps or {}
    points = (
        pd.Series(dtype=int) if poi is None else poi.shot.astype(int).value_counts()
    )
    marked = [s for s in ranked if s in points.index] or ranked
    back = sorted(s for s in ranked if gaps.get(s, 0) >= MIN_GAP_FRAMES)
    pool = [s for s in marked if s in back]
    if pool:
        branch = texts.gap
    elif back:
        branch = texts.unmarked.format(shots=", ".join(map(str, back)))
    else:
        branch = texts.fallback
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
                "longest_off_frames": int(gaps.get(s, 0)),
            }
            for s in sorted(pool)
        },
    }


def _extent(s: AEShot) -> tuple[float, float, float, float]:
    """The picture's pixel edges: ms left and right, kHz below and above."""
    return (
        s.grid.t0_ms,
        s.grid.t0_ms + s.grid.n * s.grid.dt_ms,
        s.y0 - s.dy / 2,
        s.y0 + (s.image.shape[0] - 0.5) * s.dy,
    )


def show_image(ax, image, extent, top_khz: float, ylabel: str) -> None:
    """A store's image row, bytes 0-255, on the paper's frequency axis: 0 to
    `top_khz`, each pixel as it is (the spectrograms' one style)."""
    ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        vmin=0,
        vmax=255,
        interpolation="nearest",
    )
    ax.set_ylim(0, top_khz)
    ax.set_ylabel(ylabel)


def _spectrogram(ax, s: AEShot) -> None:
    ylabel = f"{CROSS_ROWS[0].replace('x', ' × ')}\nkHz"
    show_image(ax, s.image, _extent(s), 250, ylabel)


def mask_rgba(mask: np.ndarray) -> np.ndarray:
    """The fill as RGBA pixels, its alpha in the data: `imshow(alpha=...)` over
    another image is applied twice when a PDF composites the two into one."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = to_rgba(MASK_COLOUR, MASK_ALPHA)
    return rgba


def _mask(ax, s: AEShot, label: str = MASK_LABEL) -> None:
    """SegNet's pixels, a translucent fill with a thin outline, labelled
    `label` for the legend when some pixel lies in view. Call it after the
    axes' limits are set."""
    if s.mask is None or not s.mask.any():
        return
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    times = s.grid.t0_ms + (np.arange(s.grid.n) + 0.5) * s.grid.dt_ms
    freqs = s.y0 + np.arange(s.mask.shape[0]) * s.dy
    seen = s.mask[np.ix_((freqs >= y0) & (freqs <= y1), (times >= x0) & (times <= x1))]
    ax.imshow(
        mask_rgba(s.mask),
        origin="lower",
        aspect="auto",
        extent=_extent(s),
        interpolation="nearest",
        label=label if seen.any() else "_" + label,
    )
    ax.contour(
        times, freqs, s.mask, levels=[0.5], colors=MASK_COLOUR, linewidths=MASK_LW
    )


def _scored_line(axes, s: AEShot, *, dark: Collection) -> None:
    """A dashed line where the shot's scored window ends (2 s for v1 and v2) on
    each axis, labelled on the first, when the shot runs past it; none when the
    window is the owner's whole window."""
    until = s.scored_until_ms
    if until is None or s.edges[-1] <= until:
        return
    for k, ax in enumerate(axes):
        colour = "white" if ax in dark else "black"
        ax.axvline(until, color=colour, lw=0.6, ls="--", label="_scored")
        if k == 0:
            ax.text(
                until,
                0.97,
                f"scored: {window_name(until)}",
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


def text_track(ax, text: str) -> None:
    """`text` across a track, grey italic: `COMING`, `NO_DATA`'s and the rest."""
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT_COLOUR,
        style="italic",
    )


def state_bars(ax, s, states, y: tuple[float, float] = (0.1, 0.8)) -> None:
    """One bar per suggested state `states` (per frame of `s.edges`) holds,
    keyed as a suggestion (`SUGGESTED`)."""
    edges = s.edges
    states = np.asarray(states)
    for state, colour in STATE_COLOURS.items():
        spans = [(edges[a], edges[b] - edges[a]) for a, b in runs(states == state)]
        if spans:
            label = SUGGESTED.format(STATE_NAMES[state])
            ax.broken_barh(spans, y, color=colour, lw=0, label=label)


LEGEND_ORDER = (
    *(f"owner: {name}" for name in STATE_NAMES.values()),
    MODEL_LABEL,
    THRESHOLD_LABEL,
    *(SUGGESTED.format(name) for name in STATE_NAMES.values()),
    *MASK_LABELS,
)


def _legend(fig: Figure) -> None:
    """One key per label some panel draws, with that artist's own style; the
    mask, an image, which a legend cannot key, as a patch of its fill and
    outline (`MASK_LABELS`: the paper's "segmentation: AE" and the roster
    figure's "segmentation")."""
    found: dict[str, object] = {}
    for ax in fig.axes:
        for handle, name in zip(*ax.get_legend_handles_labels(), strict=True):
            found.setdefault(name, handle)
        for label in MASK_LABELS:
            if any(image.get_label() == label for image in ax.images):
                found.setdefault(
                    label,
                    Patch(
                        facecolor=to_rgba(MASK_COLOUR, MASK_ALPHA),
                        edgecolor=MASK_COLOUR,
                        linewidth=MASK_LW,
                    ),
                )
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


def draw_interpreter(
    s: AEShot, stem: Path, tracks: Mapping[str, object] | None = None
) -> Figure:
    """The spectrogram with AE's mask over a track per phenomenon. `tracks`
    gives each other phenomenon's track (F9): its suggested state per frame of
    `s`, or a text (`NO_DATA`'s, `NOT_APPLIED`, `COMING`); one not given is
    `COMING`."""
    tracks = tracks or {}
    t0, t1 = s.edges[0] - MARGIN_MS, s.edges[-1] + MARGIN_MS
    with style():
        fig = Figure(figsize=(PAGE_IN, 3.4), layout="constrained")
        spec, *axes = fig.subplots(
            1 + len(ORDER),
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [4] + [0.45] * len(ORDER)},
        )
        _spectrogram(spec, s)
        spec.set_xlim(t0, t1)
        _mask(spec, s)
        spec.set_title(f"shot {s.shot}: what the interpreter marks")
        for ax, category in zip(axes, ORDER, strict=True):
            ax.set_ylim(0, 2)
            ax.set_yticks([])
            ax.set_ylabel(title(category), rotation=0, ha="right", va="center")
            if category == AE:
                _owner_bars(ax, s, (1.05, 0.9))
                _model_bars(ax, s, (0.05, 0.9))
                continue
            track = tracks.get(category, COMING)
            if isinstance(track, str):
                text_track(ax, track)
            else:
                state_bars(ax, s, track, (0.2, 1.6))
        _scored_line([spec, *axes], s, dark={spec})
        for ax in axes[:-1]:
            ax.tick_params(bottom=False)
        axes[-1].set_xlabel("time (ms)")
        _legend(fig)
        save(fig, stem)
    return fig


def draw_examples(shots: Sequence[AEShot], stem: Path) -> Figure:
    """Each shot: the spectrogram with the mask, the owner's strip, and P(AE)
    with the model's present frames shaded; each title gives the shot's F1 over
    its scored window, and names the window."""
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
            _mask(spec, s)
            window = window_name(s.scored_until_ms)
            spec.set_title(f"shot {s.shot} ({s.split}): F1 ({window}) {s.f1:.2f}")
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
