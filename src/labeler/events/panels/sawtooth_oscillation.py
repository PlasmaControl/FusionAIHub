"""Sawteeth, as the inversion of adjacent ECE channels across the q = 1 surface,
and the soft X-ray chords that drop at each crash.

The SXR row draws the `CHOSEN` chords of the first lit fan with the most
crash-like drops over the Ip flat-top (`crash_drops`), not the brightest: on
about 25 shots the brightest sit near 4.6 V and barely move (189061's chords 10
and 12, against 9 and 11 that crash). A shot without ECE, or without SXR, gets
the other's rows alone.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ...config import Paths
from .. import spans
from ..raw import raw_signal
from ..verify import NoDataError, Panel
from ._shared import optional, plasma_window

#: Four rows of four ADJACENT channels covering 20-35, sixteen in all. The
#: flip a sawtooth crash makes is a RELATIVE thing - inner channels drop as
#: outer ones rise - so channels are overplotted in adjacent groups rather
#: than drawn one per panel.
CHANNEL_ROWS = (
    (20, 21, 22, 23),
    (24, 25, 26, 27),
    (28, 29, 30, 31),
    (32, 33, 34, 35),
)
#: The SXR fans tried in order, by their first row in the corpus's 320 (32
#: chords each). The first with `MIN_CHORDS` chords finite over at least half
#: the record is drawn: its `CHOSEN` chords with the most crash-like drops.
SXR_ARRAYS = (
    ("SX90RM1F", 192),
    ("SX90RP1F", 256),
    ("SX90RM1S", 224),
    ("SX90RP1S", 288),
)
CHORDS = 32
MIN_CHORDS = 8
CHOSEN = 4
#: A crash-like drop: a sample where the `SMOOTH_SAMPLES`-sample mean falls by
#: more than `DROP_SIGMA` standard deviations of its own sample-to-sample
#: change, that deviation taken per `BLOCK_MS` of the flat-top so a stretch
#: where the fan misbehaves (189061's first second) sets only its own scale.
#: The judge's measure: on 189061 it ranks chords 8, 9, 11 and 14 first.
SMOOTH_SAMPLES = 5
DROP_SIGMA = 6.0
BLOCK_MS = 1000.0
MIN_BLOCK_SAMPLES = 20


def crash_drops(x_ms, y, span=None) -> np.ndarray:
    """Each row's crash-like drops inside `span` (ms; None: the whole record)."""
    x = np.asarray(x_ms, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    lo, hi = (x[0], x[-1]) if span is None else span
    counts = np.zeros(len(y), dtype=np.int64)
    for start in np.arange(lo, hi, BLOCK_MS):
        keep = (x >= start) & (x < min(start + BLOCK_MS, hi))
        if keep.sum() < MIN_BLOCK_SAMPLES:
            continue
        for row, values in enumerate(y[:, keep]):
            finite = np.isfinite(values)
            if finite.sum() < MIN_BLOCK_SAMPLES:
                continue
            values = np.where(finite, values, np.median(values[finite]))
            step = np.diff(uniform_filter1d(values, SMOOTH_SAMPLES, mode="nearest"))
            counts[row] += int(np.sum(step < -DROP_SIGMA * step.std()))
    return counts


def chord_span(shot, paths) -> tuple[float, float] | None:
    """Where the chords are compared: the Ip flat-top inside the plasma window,
    else the window, else (None) the whole record."""
    window = plasma_window(shot, paths)
    if window is None:
        return None
    paths = Paths.from_env() if paths is None else paths
    return spans.plasma_flattop(int(shot), paths, window) or window


def ece_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    built = []
    for row in CHANNEL_ROWS:
        array = raw_signal(
            int(shot), "ece", channels=list(row), t_range=t_range, paths=paths
        )
        built.append(
            Panel(
                title=f"ECE ch {row[0]}-{row[-1]}",
                x=array.x,
                y=array.y,
                ylabel="keV",
                legend=[f"ch {c}" for c in row],
            )
        )
    return built


def sxr_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The first lit fan's `CHOSEN` chords with the most crash-like drops, the
    brighter first among equals, chosen over the whole record whatever the view."""
    span = chord_span(shot, paths)
    for name, first in SXR_ARRAYS:
        rows = list(range(first, first + CHORDS))
        array = raw_signal(int(shot), "sxr", channels=rows, paths=paths)
        y = np.asarray(array.y)
        lit = np.isfinite(y).mean(axis=1) >= 0.5
        if lit.sum() < MIN_CHORDS:
            continue
        drops = np.full(CHORDS, -1)
        drops[lit] = crash_drops(array.x, y[lit], span)
        level = np.full(CHORDS, -np.inf)
        level[lit] = np.nanmedian(y[lit], axis=1)
        top = np.sort(np.lexsort((-level, -drops))[:CHOSEN])
        x = np.asarray(array.x)
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
            x, y = x[keep], y[:, keep]
        return [
            Panel(
                title=f"SXR {name}, the {CHOSEN} chords with the most crash-like drops",
                x=x,
                y=y[top],
                legend=[f"{name}{c + 1:02d}" for c in top],
            )
        ]
    raise NoDataError(f"shot {int(shot)}: no SXR fan has {MIN_CHORDS} finite chords")


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return optional("ECE", shot, lambda: ece_panels(shot, **kwargs)) + optional(
        "SXR", shot, lambda: sxr_panels(shot, **kwargs)
    )
