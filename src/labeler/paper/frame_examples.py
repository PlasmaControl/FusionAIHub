"""Examples of the four frame models' test shots: `fig_examples_<phenomenon>`.

Each is the frames gallery's test picture (`labeler.frames.gallery`, drawn with
its `draw_rows`, `draw_target` and `draw_prob`) in the paper's style, for three
of the selected model's test shots (`roc.SELECTED`, `frames.VERSION`), taken by
`shots.pick_examples` over each shot's F1: the best, the median and the worst.

**The F1 of a shot** is `frames.evaluate.score`'s, the evaluation's own: over
the bins the model scored (ABSENT or PRESENT_T, observed, with a finite P) at
the model's threshold, `blob["threshold"]` of the loader. Only the shots with
a present bin among those scored bins are ranked (`scored_present`), so each
has an F1.

**A shot's panels** are the model's input rows (one block a role, scaled 0-1),
the target per bin (present, absent, uncertain, unknown: the original's labels
with the owner's over them) and P per frame, its present frames shaded and its
threshold dotted. A shot is drawn from its prepared features, as the gallery
draws a test shot, so the sawtooth stores are not needed.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ..ae.xpower.gallery import STATE_COLOURS
from ..config import Paths
from ..events.catalog.states import NOT_OBSERVABLE
from ..frames import apply as frames_apply
from ..frames import evaluate as frames_evaluate
from ..frames import gallery
from ..frames.targets import ABSENT, UNKNOWN
from ..scoring.frames import FRAME_MS
from . import PAGE_IN, roc, save, style
from . import shots as ae_shots

RULE = (
    "the test shots with a present bin among the bins the model scored (absent "
    "or present, observed, with a finite P), ranked by F1 over those bins at its "
    "threshold, ties by the lower shot number, taken evenly: the best, the "
    "median and the worst; a shot with no present bin scored is left out"
)
#: Blank is absent, as in fig_examples_ae's owner strip; unknown is its
#: not-observable grey.
TARGET_COLOURS = {ABSENT: None, UNKNOWN: STATE_COLOURS[NOT_OBSERVABLE]}
TARGET_LABEL = "target: {}"
MODEL_LABEL = ae_shots.MODEL_LABEL
THRESHOLD_LABEL = ae_shots.THRESHOLD_LABEL
LEGEND_ORDER = (
    *(TARGET_LABEL.format(n) for n in gallery.TARGET_NAMES.values()),
    MODEL_LABEL,
    THRESHOLD_LABEL,
)


@dataclass(frozen=True)
class FrameShot:
    """One test shot as drawn: its features and target, P per frame, the
    threshold and the F1 over the scored bins."""

    shot: int
    x: np.ndarray
    first: int
    bins: np.ndarray
    states: np.ndarray
    prob: np.ndarray
    threshold: float
    f1: float
    split: str = "test"


def figure_name(method: str) -> str:
    """The product's name: `fig_examples_ntm` for `ntm_frames`."""
    return f"fig_examples_{method.removesuffix('_frames')}"


def shot_f1(prob, scored: frames_evaluate.Shot, threshold: float) -> float:
    """The shot's F1 from its P per bin over its scored bins
    (`frames.evaluate.score`); NaN where undefined."""
    f1 = frames_evaluate.score(prob, scored.states, threshold, scored.observed)["f1"]
    return float("nan") if f1 is None else float(f1)


def scored_present(prob, scored: frames_evaluate.Shot) -> bool:
    """Whether a present bin is among the bins the model scored: observed, with
    a finite P (`roc.scored_bins`, the bins `frames.evaluate.score` counts)."""
    return bool(roc.scored_bins(prob, scored.states, scored.observed)[1].any())


def read_features(
    paths: Paths, method: str, shots: Sequence[int]
) -> tuple[list[frames_evaluate.Shot], dict[int, bytes], dict[str, str]]:
    """The test shots with features as scoring reads them, their features'
    bytes by shot (the ones parsed), and the shots without them with why
    (`frames.evaluate.read_shots`)."""
    data: dict[int, bytes] = {}
    found, left = frames_evaluate.read_shots(paths, method, shots, data=data)
    return found, data, left


def scores_of(
    model, spec, threshold: float, shots: Sequence[frames_evaluate.Shot]
) -> tuple[dict[int, float], set[int]]:
    """Each shot's F1 (`shot_f1`), and the shots with a present bin among the
    bins the model scored (`scored_present`), from one P per shot."""
    f1, present = {}, set()
    for scored in shots:
        prob = frames_evaluate.model_probs(model, spec, scored)
        f1[scored.shot] = shot_f1(prob, scored, threshold)
        if scored_present(prob, scored):
            present.add(scored.shot)
    return f1, present


def pick(f1: dict[int, float], n: int = 3) -> list[int]:
    """The examples: `shots.pick_examples` over the F1s."""
    return ae_shots.pick_examples(f1, n)


def picture(
    model, spec, threshold: float, shot: int, blob: bytes, f1: float
) -> FrameShot:
    """One shot ready to draw: P per frame as the gallery's test picture has it."""
    with np.load(io.BytesIO(blob)) as z:
        x, observed, first = z["x"], z["observed"], int(z["first"])
        bins, states = z["bins"], z["states"]
    prob = frames_apply.frame_probs(model, spec, x, observed, first)
    return FrameShot(shot, x, first, bins, states, prob, threshold, f1)


def _legend(fig: Figure) -> None:
    """One key per label the panels draw, built from the artists."""
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
            ncols=len(names),
        )


def draw_examples(spec, shots: Sequence[FrameShot], stem: Path) -> Figure:
    """Each shot: the input rows, the target strip and P with its present frames
    shaded and its threshold dotted; the title gives the shot's F1 and its
    unit."""
    with style():
        fig = Figure(figsize=(PAGE_IN, 2.2 * len(shots) + 0.3), layout="constrained")
        axes = fig.subplots(
            3 * len(shots),
            1,
            gridspec_kw={"height_ratios": [3, 0.35, 1] * len(shots)},
        )
        for k, s in enumerate(shots):
            rows, strip, model = axes[3 * k : 3 * k + 3]
            t0, t1 = s.first * FRAME_MS, (s.first + len(s.prob)) * FRAME_MS
            for ax in (rows, strip, model):
                ax.set_xlim(t0, t1)
            gallery.draw_rows(rows, spec, s.x, t0, t1)
            rows.set_title(
                f"shot {s.shot} ({s.split}): F1 ({spec.bin_ms:g} ms bins) {s.f1:.2f}"
            )
            gallery.draw_target(
                strip,
                spec,
                (s.bins, s.states),
                label=TARGET_LABEL,
                fontsize=6,
                colours=TARGET_COLOURS,
            )
            gallery.draw_prob(
                model,
                s.prob,
                s.first,
                s.threshold,
                colour=ae_shots.MODEL_COLOUR,
                alpha=0.15,
                shade=ae_shots.MODEL_COLOUR,
                lw=(0.6, 0.5),
                labels=(MODEL_LABEL, THRESHOLD_LABEL),
            )
            for ax in (rows, strip):
                ax.tick_params(labelbottom=False)
        axes[-1].set_xlabel("time (ms)")
        _legend(fig)
        save(fig, stem)
    return fig


def draw_one(spec) -> Callable:
    """`draw_examples` for `spec`, as the build's `figure(name, draw, *args)`
    calls it."""
    return lambda shots, stem: draw_examples(spec, shots, stem)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
