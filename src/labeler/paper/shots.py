"""AE examples from the test shots, and the pieces the interpreter figure shares.

`picture` is what a shot viewer shows for AE: the review store's first
cross-power row (R0 x V1), 0-250 kHz; the owner's frames; the chosen
`ae_xpower` model's P(AE) per 10 ms frame, run here on the CPU as the gallery
runs it; and the segmentation's mask, SegNet run here too. The owner's
frames are the model's own copy of the labels, `<candidate>/review/labels.csv`,
the ones it was trained and scored on (D18), never the live table the owner
keeps saving; `ae_shot` reads them from disk. `draw_examples` stacks a few
test shots (the model's `split.csv`), which `pick_examples` takes: the best,
the median and the worst F1 against the owner; a whole-window version's picks
hold at least one shot whose owner window runs past 2 s (`LONG_MS`) when a
test shot's does.

**The interpreter figure** is `paper.roster`'s, the paper's `fig_interpreter`:
the models' suggestions on one roster shot, never reviewed. It is drawn with
the pieces here: the spectrogram (`show_image`), the mask (`_mask`, keyed
`ROSTER_MASK_LABEL`), the suggested states (`state_bars`, every key
`SUGGESTED`'s, never the owner's), a track's or a panel's text (`text_track`:
`NO_DATA`'s, `NOT_APPLIED`) and the legend (`_legend`).

**The scored window** is the version's (`scored_ms`, from
`labeler.ae.xpower.scored_until`): 0-2 s for v1 and v2, whose TokEye masks end
at 2 s, and the owner's whole window for v3 (None). A shot's F1, the examples'
rule (`pick_texts`) and the dashed line all use it; `EXAMPLES_RULE` is the
0-2 s text, as v1's and v2's manifests have it.

**The F1 of a shot** is over the 10 ms frames of the scored window that the
owner called present or absent (`scored_f1`), the window the paper's headline
F1 is over. The gallery's `f1_vs_owner` is over the owner's whole window, as
v3's F1 is; the build ranks by the scored F1 (`AEShot.f1`), so a 0-2 s
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
`poi.csv`, are not drawn: they are a table for tools. A dashed line at 2 s
marks a 0-2 s version's scored window whenever a shot runs past it; a
whole-window version's shots have none. No picture has a line at 80 kHz: that
floor was only the labelling view.

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
from ..events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.review import labels
from ..events.review.labels import Label
from ..events.review.rows import Grid
from ..scoring.frames import FRAME_MS
from . import FONT_PT, PAGE_IN, save, style

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
STATE_NAMES = {
    PRESENT: "present",
    UNCERTAIN: "uncertain",
    NOT_OBSERVABLE: "not observable",
}
MASK_LABEL = "segmentation: AE"
#: The interpreter figure's fill, SegNet over its whole picture (`paper.roster`).
ROSTER_MASK_LABEL = "segmentation"
MASK_LABELS = (MASK_LABEL, ROSTER_MASK_LABEL)
MODEL_LABEL = "model: present"
THRESHOLD_LABEL = "model threshold"
#: A frame-model track's key (F9): what it holds is a suggestion.
SUGGESTED = "suggested: {}"
#: A signal panel's text where the shot's store lacks its rows (`paper.roster`).
NO_DATA = "no {} data"
#: A track's text where the frame model's table exists without the shot.
NOT_APPLIED = "not applied to this shot"
TEXT_COLOUR = "#888888"


@dataclass(frozen=True)
class PickTexts:
    """The examples' rule, said over one scored window."""

    examples: str


def pick_texts(until_ms: float | None = SCORED_MS) -> PickTexts:
    """The rule's text over the scored window ending at `until_ms` (0-2 s by
    default; None, the owner's whole window)."""
    where = "the whole window" if until_ms is None else window_name(until_ms)
    return PickTexts(
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


# The 0-2 s text, v1's and v2's, the string it has always been.
EXAMPLES_RULE = pick_texts(SCORED_MS).examples


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


def text_track(ax, text: str) -> None:
    """`text` across a track or panel, grey italic: `NO_DATA`'s, `NOT_APPLIED`
    and the rest."""
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
    outline (`MASK_LABELS`: fig_examples_ae's "segmentation: AE" and the
    interpreter figure's "segmentation")."""
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
