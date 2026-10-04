"""Native-rate anti-aliasing and explicit diagnostic assessment support."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly


def sample_native(group, *, rows=None, fs=10000, chunk_samples=250000, rms=False):
    """FIR-filter every native sample before integer decimation, with halos.

    The default polyphase FIR has half-length 10 times the decimation factor.
    Chunks and halos align to the output grid; reflected endpoint padding and
    complete finite FIR support preserve DC and avoid zero-filled missing data.
    Memory scales with channel count times chunk size, not shot duration.
    Low-rate diagnostics retain their native cadence for caller interpolation.
    """
    x, y = group["xdata"], group["ydata"]
    if len(x) < 32 or y.ndim != 2 or y.shape[1] != len(x):
        raise ValueError("absent waveform or incompatible clock")
    dt = (float(x[-1]) - float(x[0])) / (len(x) - 1)
    if dt <= 0:
        raise ValueError("nonincreasing clock")
    factor = max(1, round(1 / (dt * fs)))
    block = max(factor, chunk_samples // factor * factor)
    halo = 10 * factor if factor > 1 else 0
    rows = slice(None) if rows is None else rows
    tx, values = [], []
    for start in range(0, len(x), block):
        end = min(len(x), start + block)
        lo, hi = max(0, start - halo), min(len(x), end + halo)
        clock = np.asarray(x[lo:hi], dtype=float)
        if not np.isfinite(clock).all() or (np.diff(clock) <= 0).any():
            raise ValueError("native clock must be finite and strictly increasing")
        expected = float(x[0]) + np.arange(lo, hi) * dt
        tolerance = max(1e-7, 4 * np.finfo(x.dtype).eps * np.max(np.abs(clock)))
        if not np.all(np.abs(clock - expected) <= tolerance):
            raise ValueError("nonuniform waveform")
        v = np.asarray(y[rows, lo:hi], dtype=np.float32)
        if rms:
            v = v**2
        if factor > 1:
            finite = np.isfinite(v)
            bad = maximum_filter1d(
                (~finite).astype(np.uint8), 2 * halo + 1, axis=1, mode="reflect"
            )[:, ::factor]
            v = resample_poly(
                np.where(finite, v, 0), 1, factor, axis=1, padtype="reflect"
            )
            v[bad.astype(bool)] = np.nan
        if rms:
            v = np.sqrt(np.maximum(v, 0))
        first = (start - lo) // factor
        count = (end - start + factor - 1) // factor
        values.append(v[:, first : first + count])
        tx.append(expected[first * factor : first * factor + count * factor : factor])
    return np.concatenate(tx), np.concatenate(values, axis=1)


def mask_spans(t, mask):
    """Half-open sample-cell support, preserving every missing-data gap."""
    t, mask = np.asarray(t), np.asarray(mask, dtype=bool)
    dt = float(np.median(np.diff(t)))
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [
        (float(t[a]), float(t[b]) if b < len(t) else float(t[-1] + dt))
        for a, b in zip(ends[::2], ends[1::2], strict=True)
    ]


STATES = ("present", "absent", "absent_q_prior", "uncertain", "unassessed")


def state_spans(t, observable, present, uncertain, *, absent=None, q_prior=None):
    """Five states; only an explicit ECE absence test supplies negatives.

    Observable bins without an accepted train or a tested absence remain
    uncertain. ``absent`` is a sample mask, typically ``Detection.absent_mask``.
    ``q_prior`` marks time only a sustained EFIT01 q prior calls quiet: state
    ``absent_q_prior``, not assessed and never a negative. A tested absence or
    an uncertain interval overrides it. Missing diagnostic support overrides
    every assessment.
    """
    from .metrics import spans_at

    observable = np.asarray(observable, dtype=bool)
    if observable.shape != np.shape(t):
        raise ValueError("observability must have one boolean per sample")
    states = np.full(len(t), "uncertain", dtype="U14")
    for mask, label in ((q_prior, "absent_q_prior"), (absent, "absent")):
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != np.shape(t):
            raise ValueError("absence evidence must have one boolean per sample")
        states[mask] = label
    states[spans_at(t, present)] = "present"
    doubt = spans_at(t, uncertain)
    states[doubt] = "uncertain"
    states[~observable] = "unassessed"
    spans = sorted(
        [
            {"start_s": a, "end_s": b, "state": state}
            for state in STATES
            for a, b in mask_spans(t, states == state)
        ],
        key=lambda r: r["start_s"],
    )
    return spans, (states == "present") | (states == "absent")
