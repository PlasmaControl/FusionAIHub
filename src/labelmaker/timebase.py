"""Sampling a per-shot record at chosen times.

These are IGNITE's conventions, copied rather than imported: labelmaker
depends on no model code (see ignite/gate.py's reuse note). The original
and the reason both halves matter are in
src/tokamak_foundation_model/ignite/train_dynamics.py:177-192.
"""
from __future__ import annotations

import numpy as np

MS_PER_S = 1000.0


def sample_rate(x: np.ndarray) -> float:
    """Samples per second, from the record span.

    Not `1/median(diff(x))`: `xdata` is float32 in the corpus, which loses
    the sample step to cancellation and drifts by up to 0.8% over a shot.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        raise ValueError(f"need at least two samples, got {x.size}")
    span = float(x[-1] - x[0])
    if span <= 0.0:
        raise ValueError(f"non-increasing time axis: {x[0]} .. {x[-1]}")
    return (x.size - 1) / span


def index_at(x: np.ndarray, t) -> np.ndarray:
    """Nearest sample index for each requested time, clamped into the record.

    Counted from the record start, not from zero: actuator groups begin at
    negative times, where an unclamped index would wrap to the array tail.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    idx = np.round((t - x[0]) * sample_rate(x)).astype(np.int64)
    return np.clip(idx, 0, x.size - 1)


def sample_at(x: np.ndarray, y: np.ndarray, t, *, max_gap: float | None = None):
    """`y` at each time in `t` (nearest sample), last axis sampled.

    `max_gap` guards the clamp: a time further than `max_gap` seconds from
    the nearest sample yields NaN instead of the edge value.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    idx = index_at(x, t)
    out = y[..., idx].astype(np.float64, copy=True)
    if max_gap is not None:
        far = np.abs(x[idx] - t) > max_gap
        out[..., far] = np.nan
    return out


def window_mean(x: np.ndarray, y: np.ndarray, t, width: float) -> np.ndarray:
    """Mean of `y` over each half-open window `[t, t + width)`.

    NaN where the window holds no finite sample, so a missing stretch never
    silently becomes an edge value.

    The mean is accumulated by hand rather than with `np.nanmean`, which
    raises "Mean of empty slice" through the `warnings` module on an
    all-NaN window - `np.errstate` does not catch that, and real channels
    have dropout stretches. Same approach as `decimate_to_step` below.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.full((y.shape[0], t.size), np.nan, dtype=np.float64)
    lo = np.searchsorted(x, t, side="left")
    hi = np.searchsorted(x, t + width, side="left")
    for j, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            continue
        seg = y[:, a:b]
        good = np.isfinite(seg)
        counts = good.sum(axis=1)
        totals = np.where(good, seg, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = np.where(counts > 0, totals / counts, np.nan)
    return out[0] if out.shape[0] == 1 else out


def decimate_to_step(x: np.ndarray, y: np.ndarray, step: float):
    """Bin-average `y` onto a uniform grid of the given step.

    Used to bound the size of high-rate PTDATA before it is stored: a 10 kHz
    channel over 6 s is 60k samples per shot, and nothing downstream asks
    for more than 1 ms resolution. Empty bins are NaN; NaNs inside a bin are
    ignored.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    if x.size < 2:
        raise ValueError(f"need at least two samples, got {x.size}")
    n = int(np.floor((x[-1] - x[0]) / step)) + 1
    bins = np.clip(((x - x[0]) / step).astype(np.int64), 0, n - 1)
    out = np.empty((y.shape[0], n), dtype=np.float64)
    for c in range(y.shape[0]):
        good = np.isfinite(y[c])
        totals = np.bincount(bins, weights=np.where(good, y[c], 0.0), minlength=n)
        counts = np.bincount(bins, weights=good.astype(np.float64), minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[c] = np.where(counts > 0, totals / counts, np.nan)
    return x[0] + step * np.arange(n, dtype=np.float64), out
