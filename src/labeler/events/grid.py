"""The one time grid every DIII-D label sits on: 10 ms frames from t = 0.

Frame k covers [10k, 10k + 10) ms. A boundary goes to the nearest frame edge,
halves up (the page's `Math.floor(t / 10 + 0.5) * 10`). A span shorter than a
frame that would vanish claims the frame holding its midpoint, so nothing
labelled is lost; a point (zero length) stays a point. A window's ends snap the
same way, so a label ending at 4573 ms does not gain an absent frame it never
assessed. Every event, subcategory and source then fits one (event x frame)
array per shot.
"""

from __future__ import annotations

import math

import numpy as np

GRID_MS = 10
#: Frames outside a label's window: nobody said anything there.
UNASSESSED = -1


def snap(t, grid_ms: int = GRID_MS) -> int:
    """The frame edge nearest `t` ms, halves up."""
    value = float(t)
    if not math.isfinite(value):
        raise ValueError("time must be finite")
    return grid_ms * math.floor(value / grid_ms + 0.5)


def snap_span(a, b, grid_ms: int = GRID_MS) -> tuple[int, int]:
    """Both ends on frame edges; a short span keeps the frame of its midpoint."""
    a, b = float(a), float(b)
    if b < a:
        raise ValueError(f"span {a}-{b} ms runs backwards")
    lo, hi = snap(a, grid_ms), snap(b, grid_ms)
    if hi > lo or b == a:
        return lo, hi
    start = grid_ms * math.floor((a + b) / 2 / grid_ms)
    return start, start + grid_ms


def snap_window(lo, hi, grid_ms: int = GRID_MS) -> tuple[int, int]:
    """A window's ends, snapped as a span's are."""
    lo, hi = float(lo), float(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
        raise ValueError(f"window {lo}-{hi} ms is not a window")
    return snap_span(lo, hi, grid_ms)


def frames(window, intervals, t0_ms: int, n: int, grid_ms: int = GRID_MS,
           unknown=(), fill: int = 0) -> np.ndarray:
    """`n` frames from `t0_ms` (a frame edge): UNASSESSED outside the window,
    `fill` inside it (0, absent, unless the event's default says another class,
    `taxonomy.Taxonomy.default`), and each span's category, later spans painting
    over earlier; `unknown` spans (a table's not-observable stretches) are
    UNASSESSED again."""
    if t0_ms % grid_ms:
        raise ValueError(f"t0 {t0_ms} ms is not a frame edge")
    out = np.full(n, UNASSESSED, dtype=np.int8)

    def index(t):
        return min(max((t - t0_ms) // grid_ms, 0), n)

    lo, hi = snap_window(*window, grid_ms)
    out[index(lo):index(hi)] = fill
    for a, b, c in intervals:
        a, b = snap_span(a, b, grid_ms)
        out[index(a):index(b)] = c
    for a, b in unknown:
        a, b = snap_span(a, b, grid_ms)
        out[index(a):index(b)] = UNASSESSED
    return out
