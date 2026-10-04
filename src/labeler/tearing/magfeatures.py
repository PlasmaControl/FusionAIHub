"""Magnetics features for the tearing-mode detector: the Mirnov array below 30 kHz.

A tearing mode is a coherent line below ~30 kHz on the toroidal Mirnov array with a
toroidal number n from the probes' phases. `shot_features` cuts a shot's six midplane
probes (the review editor's: `labeler.events.panels.neoclassical_tearing_mode.PROBES`)
into a short-time Fourier transform and reduces each column to `FEATURE_NAMES`:

* the array-mean power in eight bands (absolute, and above the shot's own quiet level);
* the amplitude of the coherent lines whose phases fit n = 1, 2 and 3 (best fit among
  n = -4 to 5, as the review editor's map, and at least 0.9; either sense of rotation)
  as the largest value in each of four bands;
* at the strongest line of the column, its frequency, its prominence and how well its
  phases fit n = 1, 2 and 3.

The columns (2.56 ms apart) are averaged into 10 ms bins. None of this reads the lab's
`\\MHD::N1RMS` / `N2RMS`, which are what the interval labels are drawn from;
`rms_bin_features` adds them for the ablation that asks how much of the detector is
those two traces.
"""

from __future__ import annotations

import numpy as np

from ..events.panels._shared import stft
from ..events.panels.neoclassical_tearing_mode import PROBES
from . import rule, scoring

#: The probes (corpus `mirnov` rows) and their toroidal angles (degrees).
PROBE_ROWS = tuple(PROBES)
PROBE_PHI = tuple(PROBES.values())
RATE_HZ = 100_000
NPERSEG = 1024
HOP = 256
MAX_KHZ = 30.0
POWER_BANDS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 12), (12, 16), (16, 22), (22, 30))
N_BANDS = ((0.5, 5), (5, 10), (10, 20), (20, 30))
N_VALUES = (1, 2, 3)
#: The strongest line is looked for between these (kHz): below 1 kHz sits locked-mode
#: and equilibrium drift, which has no phase to fit.
LINE_KHZ = (1.0, 30.0)
FLOOR_QUANTILE = 0.2
EPS = 1e-30
#: The toroidal numbers the phases are fitted for, the review editor's range.
FIT_N = tuple(range(-4, 6))
#: A cell counts as a coherent mode of its best-fit n at or above this phase fit.
COHERENT_FIT = 0.9
#: Added before the log of a band's n-resolved amplitude: a band with no such mode.
AMP_FLOOR = 1e-9

FEATURE_NAMES = (
    tuple(f"logp_{lo}_{hi}" for lo, hi in POWER_BANDS)
    + tuple(f"rel_{lo}_{hi}" for lo, hi in POWER_BANDS)
    + tuple(f"a{n}_{lo}_{hi}" for n in N_VALUES for lo, hi in N_BANDS)
    + tuple(f"fit{n}" for n in N_VALUES)
    + ("line_khz", "line_prominence_db")
)


def harmonic_amplitude(spec, n: int) -> np.ndarray:
    """`(F, T)`: the n-th toroidal harmonic of the probes' complex amplitudes.

    `spec` is `(P, F, T)`. The larger of the two senses of rotation is kept.
    """
    phi = np.deg2rad(np.asarray(PROBE_PHI, dtype=float))
    out = None
    for sign in (1, -1):
        turns = np.exp(-1j * sign * n * phi).astype(np.complex64)
        amp = np.abs(np.tensordot(turns, spec, axes=(0, 0))) / spec.shape[0]
        out = amp if out is None else np.maximum(out, amp)
    return out


def phase_fit(spec, n: int) -> np.ndarray:
    """`(F, T)` in 0..1: how nearly the probes' phases turn n times round the torus."""
    unit = spec / np.maximum(np.abs(spec), EPS)
    return harmonic_amplitude(unit, n)


def best_n(spec) -> tuple[np.ndarray, np.ndarray]:
    """`(n, fit)` per cell: the toroidal number whose phases fit the probes best.

    `n` runs over `FIT_N` (signed, as the review editor's mode-number map takes it) and
    `fit` is the phase fit of that n, 0..1.
    """
    unit = spec / np.maximum(np.abs(spec), EPS)
    phi = np.deg2rad(np.asarray(PROBE_PHI, dtype=float))
    fit = np.full(unit.shape[1:], -1.0, dtype=np.float32)
    which = np.zeros(unit.shape[1:], dtype=np.int8)
    for n in FIT_N:
        turns = np.exp(-1j * n * phi).astype(np.complex64)
        value = np.abs(np.tensordot(turns, unit, axes=(0, 0))) / unit.shape[0]
        which[value > fit] = n
        np.maximum(fit, value, out=fit)
    return which, fit


def column_features(spec, f_khz, floor_columns=None) -> np.ndarray:
    """`(T, len(FEATURE_NAMES))` float32 from `(P, F, T)` spectra up to `MAX_KHZ`."""
    f = np.asarray(f_khz, dtype=float)
    amp = np.sqrt((np.abs(spec) ** 2).mean(axis=0))  # (F, T), array-mean amplitude
    power = amp**2
    n_cols = power.shape[1]
    logp = []
    for lo, hi in POWER_BANDS:
        sel = (f >= lo) & (f < hi if hi < MAX_KHZ else f <= hi)
        logp.append(np.log10(power[sel].sum(axis=0) + EPS))
    logp = np.stack(logp, axis=0)  # (bands, T)
    feats = [logp]
    use = np.ones(n_cols, bool) if floor_columns is None else np.asarray(floor_columns)
    floor = np.quantile(logp[:, use] if use.any() else logp, FLOOR_QUANTILE, axis=1)
    feats.append(logp - floor[:, None])
    which, fit = best_n(spec)
    coherent = fit >= COHERENT_FIT
    for n in N_VALUES:
        mine = amp * (coherent & (np.abs(which) == n))
        rows = []
        for lo, hi in N_BANDS:
            sel = (f >= lo) & (f <= hi)
            rows.append(np.log10(mine[sel].max(axis=0) + AMP_FLOOR))
        feats.append(np.stack(rows, axis=0))
    line = (f >= LINE_KHZ[0]) & (f <= LINE_KHZ[1])
    idx = np.flatnonzero(line)
    peak = idx[np.argmax(power[line], axis=0)]  # (T,)
    cols = np.arange(n_cols)
    feats.append(np.stack([phase_fit(spec, n)[peak, cols] for n in N_VALUES], axis=0))
    inside = power[line]
    prominence = 10 * np.log10(
        (inside.max(axis=0) + EPS) / (np.median(inside, axis=0) + EPS)
    )
    feats.append(np.stack([f[peak], prominence], axis=0))
    return np.concatenate(feats, axis=0).T.astype(np.float32)


def bin_columns(t_ms, values, centres, bin_ms: float = scoring.BIN_MS) -> np.ndarray:
    """`(len(centres), C)`: the mean of the columns in each bin, NaN where none are."""
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(values, dtype=float)
    edges = np.concatenate((centres - bin_ms / 2.0, [centres[-1] + bin_ms / 2.0]))
    which = np.searchsorted(edges, t, side="right") - 1
    ok = (which >= 0) & (which < len(centres))
    out = np.full((len(centres), v.shape[1]), np.nan)
    count = np.bincount(which[ok], minlength=len(centres)).astype(float)
    for c in range(v.shape[1]):
        total = np.bincount(which[ok], weights=v[ok, c], minlength=len(centres))
        out[count > 0, c] = total[count > 0] / count[count > 0]
    return out


def shot_features(t_ms, y, window, bin_ms: float = scoring.BIN_MS):
    """`(centres_ms, features)` of one shot's probes `y` `(P, N)` over `window`.

    `t_ms` is the probes' time base. Bins are `scoring.bin_centres(window)`; a bin no
    STFT column falls in is NaN.
    """
    t_cols, f_hz, spec = stft(t_ms, y, rate_hz=RATE_HZ, nperseg=NPERSEG, hop=HOP)
    keep = f_hz <= MAX_KHZ * 1000
    spec = spec[:, keep].astype(np.complex64)
    inside = (t_cols >= window[0]) & (t_cols <= window[1])
    feats = column_features(spec, f_hz[keep] / 1000.0, inside)
    centres = scoring.bin_centres(window, bin_ms)
    return centres, bin_columns(t_cols, feats, centres, bin_ms)


RMS_NAMES = ("log_n1rms", "log_n2rms")


def rms_bin_features(t_ms, n1, n2, centres, bin_ms: float = scoring.BIN_MS):
    """`(len(centres), 2)`: log10 of the smoothed n = 1 and n = 2 RMS, bin maxima."""
    t, a, dt = rule.uniform(t_ms, n1)
    b = rule.uniform(t_ms, n2)[1]
    cols = []
    for trace in (a, b):
        sm = rule.smoothed(trace, dt, 5.0)
        edges = np.concatenate((centres - bin_ms / 2.0, [centres[-1] + bin_ms / 2.0]))
        which = np.searchsorted(edges, t, side="right") - 1
        out = np.full(len(centres), np.nan)
        ok = (which >= 0) & (which < len(centres)) & np.isfinite(sm)
        np.fmax.at(out, which[ok], np.log10(np.maximum(sm[ok], 1e-3)))
        cols.append(out)
    return np.stack(cols, axis=1)
