"""saw-ours v2: the learned ECE picker combined with the single-channel derivative.

The derivative picker (`saw-derivative` in `scripts/labeler/sawtooth_benchmark.py`)
calls presence wherever an axis-channel edge of at least z robust sigma lies
within +/-125 ms. Its per-sample *edge score* is the largest such edge amplitude,
so the derivative presence at any z is ``edge_score >= z``. saw-ours v2 reads the
edge score twice: as two extra input channels of the network (the axis-channel
z-trace and the edge score, log-compressed), and as an AND on the output, where
presence needs both the network's probability and an edge of at least z. The
threshold and z are chosen on inner-selection shots only; z = 0 drops the AND.

The derivative reads the same ECE edges the physics-rule labels are built from,
so the combination inherits that partial circularity.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

#: Edge reach of the derivative presence rule (s), as in `derivative_prediction`.
EDGE_REACH_S = 0.125


def derivative_trace(signal):
    """The axis-channel derivative in robust sigma, on the native grid.

    The same steps as `derivative_candidates`: observable finite samples of the
    axis-selected channel, interpolated across gaps, a 2.5-sample Gaussian first
    derivative (negative, so a crash is positive) and the MAD scale over samples
    with complete filter support. Returns ``(z, complete)``; with fewer than two
    finite samples z is zero and nothing is complete.
    """
    center = int(np.asarray(signal["central_channel"]).item())
    y = np.asarray(signal["y"])
    t = np.asarray(signal["t"], float)
    finite = np.asarray(signal["observable"], bool) & np.isfinite(y[center])
    if finite.sum() < 2:
        return np.zeros(len(t)), np.zeros(len(t), bool)
    core = np.interp(t, t[finite], y[center, finite].astype(float))
    derivative = -gaussian_filter1d(core, 2.5, order=1)
    complete = (
        maximum_filter1d((~finite).astype(np.uint8), 21, mode="constant", cval=1) == 0
    )
    values = derivative[complete]
    if not len(values):
        return np.zeros(len(t)), complete
    scale = 1.4826 * np.median(np.abs(values - np.median(values)))
    return derivative / max(scale, 1e-8), complete


def edge_score(t, edge_times, amplitudes, reach_s=EDGE_REACH_S):
    """Largest edge amplitude within +/- reach_s of each time (0 where none).

    ``edge_times`` are the observable derivative peaks and ``amplitudes`` their
    z values, so ``edge_score(...) >= z`` is the derivative presence at z.
    """
    t = np.asarray(t, float)
    order = np.argsort(edge_times)
    times = np.asarray(edge_times, float)[order]
    values = np.asarray(amplitudes, float)[order]
    score = np.zeros(len(t))
    if not len(times):
        return score
    # A superset by searchsorted, then the exact closed test of
    # `derivative_prediction` (|t - edge| <= 0.125) on each candidate.
    lo = np.searchsorted(times, t - reach_s - 1e-9, side="left")
    hi = np.searchsorted(times, t + reach_s + 1e-9, side="right")
    count = hi - lo
    for k in range(int(count.max()) if len(count) else 0):
        rows = np.flatnonzero(count > k)
        index = lo[rows] + k
        inside = np.abs(t[rows] - times[index]) <= reach_s
        rows, index = rows[inside], index[inside]
        score[rows] = np.maximum(score[rows], values[index])
    return score


def input_channels(z, edge):
    """The two derivative input channels: sign-preserving log1p of the z-trace
    and log1p of the edge score. Both are order one for the global normaliser."""
    z = np.nan_to_num(np.asarray(z, float))
    return np.stack(
        [np.sign(z) * np.log1p(np.abs(z)), np.log1p(np.maximum(edge, 0))]
    ).astype(np.float32)


def combined_presence(presence, edge, z):
    """Network presence where an edge of at least z lies within reach, else 0.

    z = 0 (or below) leaves the network's presence unchanged.
    """
    presence = np.asarray(presence, np.float32)
    if z <= 0:
        return presence.copy()
    return np.where(np.asarray(edge) >= z, presence, 0).astype(np.float32)


def gated_presence(presence, no_q1):
    """Presence forced to 0 in time with no q = 1 surface (output q gate)."""
    return np.where(np.asarray(no_q1, bool), 0, presence).astype(np.float32)
