"""Sawtooth crash times from ECE Te steps, ported from `dev/label_dev/sawtooth.ipynb`.

A sawtooth crash is a step: the core's Te falls and, at the same instant, a
channel outside the inversion radius rises. A small crash's step (~0.1-0.2 keV)
is smaller than the slow swings of Te, and the noise differs between channels
and over a shot, so no fixed threshold works. What always holds is the pattern,
so:

1. `step_z`: each channel's step (the mean of the next `W_MS` minus the mean of
   the previous `W_MS`, every `GRID_MS`) in units of that channel's own typical
   step, the median of |step| over the surrounding `NOISE_MS` (a Haar wavelet in
   noise units; nothing is fitted).
2. `crash_score`: `min(-z_core, z_off)`, large only when both happen; the
   best of the off-axis channels counts, since the one outside the inversion
   radius is the one that rises. An AND, not a difference: a difference is large
   when either happens, which scores an ELM almost like a crash.
3. `find_elms`: an ELM is a sharp D-alpha rise, the same step in noise units on
   the divertor photodiode, above `ELM_Z`. It does not cool the core, but its
   0.3 ms spike reaches the ECE and the crash score can catch it, so a score
   peak within `TOGETHER_MS` of an ELM counts as a crash only if the core is
   still down 1-3 ms later by at least `LASTING` noise units (`find_events`).

The constants are the notebook's: a score threshold of `THRESHOLD`, whose noise
is about 1 (its 99th percentile 1.5 on 206279).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

#: The step's half-width (a crash lasts ~0.1-0.5 ms), the window its typical
#: size is taken over, and the grid the steps are computed on, all in ms.
W_MS = 0.5
NOISE_MS = 200.0
GRID_MS = 0.1
#: A crash-score peak above this (noise units) is a crash candidate.
THRESHOLD = 3.0
#: A D-alpha rise (0.3 ms steps) above this (noise units) is an ELM.
ELM_Z = 5.0
ELM_W_MS = 0.3
#: Peaks no closer than this are separate: ELMs, and crash-score peaks.
MIN_SEPARATION_MS = 2.0
#: A score peak this close to an ELM is the ELM's unless the core stays down.
TOGETHER_MS = 1.0
#: ...by this many noise units, 1-3 ms later (`step_z` with the spike left out).
LASTING = 20.0
LASTING_W_MS = 2.0
LASTING_GAP_MS = 1.0
#: The floor of a channel's typical step, so a dead (constant) channel divides
#: to a huge number only in its own units, not to infinity.
MIN_FLOOR = 1e-6


def _filled(y) -> np.ndarray:
    """`y` as float64 with its gaps set to its median (all-NaN: zeros)."""
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.all():
        return y
    return np.where(ok, y, np.median(y[ok]) if ok.any() else 0.0)


def _dt(t) -> float:
    """The sample step of `t` from its span, not a median step: a float32 clock
    quantises."""
    t = np.asarray(t, dtype=np.float64)
    return float((t[-1] - t[0]) / (len(t) - 1))


def step_z(t, y, w_ms=W_MS, noise_ms=NOISE_MS, grid_ms=GRID_MS, gap_ms=0.0):
    """`(t, z)`: how big a step is at each moment, in units of its channel's own.

    `step` is the mean of `y` over the next `w_ms` minus its mean over the
    previous `w_ms`, taken every `grid_ms`; its typical size is the running
    median of |step - its running median| over `noise_ms` (x 1.4826, MAD to
    sigma). Both returned arrays are on the `grid_ms` grid. `gap_ms` leaves that
    much out on each side of the moment, so a spike shorter than the gap does
    not count as a step. Empty arrays when the record is shorter than the window.
    """
    t = np.asarray(t, dtype=np.float64)
    y = _filled(y)
    if len(t) < 2:
        return np.empty(0), np.empty(0)
    dt = _dt(t)
    n, gap = max(1, round(w_ms / dt)), round(gap_ms / dt)
    stride = max(1, round(grid_ms / dt))
    c = np.concatenate([[0.0], np.cumsum(y)])  # any window mean is two lookups
    i = np.arange(n + gap, len(y) - n - gap, stride)
    if not len(i):
        return np.empty(0), np.empty(0)
    step = (c[i + gap + n] - c[i + gap]) / n - (c[i - gap] - c[i - gap - n]) / n
    k = int(noise_ms / (dt * stride)) | 1  # window length in samples, odd
    centre = median_filter(step, size=k, mode="nearest")
    floor = 1.4826 * median_filter(np.abs(step - centre), size=k, mode="nearest")
    return t[i], (step - centre) / np.maximum(floor, MIN_FLOOR)


def crash_score(t, core, outer, w_ms=W_MS, noise_ms=NOISE_MS):
    """`(t, score)`: `min(-z_core, z_off)`, the best off-axis channel's.

    `core` is the core channel's Te, `outer` the off-axis channels' `(C, T)`,
    all on the clock `t`. It is large only when the core drops AND an off-axis
    channel rises at the same instant.
    """
    outer = np.atleast_2d(outer)
    if not len(outer):
        raise ValueError("a crash score needs at least one off-axis channel")
    grid, z_core = step_z(t, core, w_ms, noise_ms)
    scores = [np.minimum(-z_core, step_z(t, row, w_ms, noise_ms)[1]) for row in outer]
    return grid, np.max(scores, axis=0)


def find_elms(t, dalpha, z_min=ELM_Z):
    """`(t, z)` of the ELM onsets: sharp D-alpha rises, at least 2 ms apart."""
    grid, z = step_z(t, dalpha, w_ms=ELM_W_MS)
    if len(grid) < 2:
        return np.empty(0), np.empty(0)
    distance = max(1, round(MIN_SEPARATION_MS / _dt(grid)))
    peaks, _ = find_peaks(z, height=z_min, distance=distance)
    return grid[peaks], z[peaks]


def find_events(
    score,
    t,
    core,
    dalpha=None,
    threshold=THRESHOLD,
    elm_z=ELM_Z,
    together_ms=TOGETHER_MS,
    lasting=LASTING,
):
    """Crashes and ELMs, labelled together: a time-sorted list of dicts
    `{"t": ms, "kind": "sawtooth" | "elm", "size": noise units}`.

    `score` is `crash_score`'s `(t, score)`, `core` the core Te on the clock `t`,
    `dalpha` the photodiode's `(t, y)` or None (then no ELM is excluded). An ELM
    is a D-alpha rise above `elm_z`. A sawtooth is a score peak above
    `threshold`, unless it comes within `together_ms` of an ELM and the core is
    back up 1-3 ms later (less than `lasting` noise units down): that was the
    ELM's spike on the ECE, and the ELM gets it as `"crash": score`.
    """
    grid, s = score
    if len(grid) < 2:
        return []
    distance = max(1, round(MIN_SEPARATION_MS / _dt(grid)))
    peaks, _ = find_peaks(s, height=threshold, distance=distance)
    elms = []
    if dalpha is not None:
        elms = [
            {"t": float(te), "kind": "elm", "size": float(z)}
            for te, z in zip(*find_elms(dalpha[0], dalpha[1], elm_z), strict=True)
        ]
    down_t, down = step_z(t, core, w_ms=LASTING_W_MS, gap_ms=LASTING_GAP_MS)
    saws = []
    for p in peaks:
        near = [e for e in elms if abs(e["t"] - grid[p]) <= together_ms]
        k = int(np.searchsorted(down_t, grid[p]))
        window = down[max(k - 3, 0) : k + 4]
        if near and len(window) and window.min() > -lasting:
            near[0]["crash"] = float(s[p])
        else:
            saws.append({"t": float(grid[p]), "kind": "sawtooth", "size": float(s[p])})
    return sorted(elms + saws, key=lambda e: e["t"])


def find_crashes(t, core, outer, dalpha=None, **kwargs) -> np.ndarray:
    """The crash times (ms), sorted: the sawtooth events of `find_events` over
    `crash_score(t, core, outer)`. `kwargs` are `find_events`' thresholds."""
    score = crash_score(t, core, outer)
    events = find_events(score, t, core, dalpha, **kwargs)
    return np.array([e["t"] for e in events if e["kind"] == "sawtooth"], dtype=float)
