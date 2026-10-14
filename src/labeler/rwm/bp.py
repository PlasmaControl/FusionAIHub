"""The resistive wall mode's own sensor: the low-frequency n = 1 delta-Bp.

Okabayashi et al. 2005 (Nucl. Fusion 45 1715, sec. 3-4) detect the RWM with the
midplane internal poloidal-field probes, "better suited ... as these probes detect
the RWM signals enhanced by the eddy currents on the resistive wall", from DC to a
few hundred Hz: the mode grows in 1-4 ms and rotates below 30 Hz. `\\MHD::N1RMS` is
an FFT-band amplitude from 0.5 kHz up and cannot see it. The four PTDATA points
MPID067U/097U/127U/157U are 180-degree probe differences D(phi) = B(phi) -
B(phi + 180), in which every even n cancels and an n = 1 perturbation of amplitude
A and phase phi0 reads D = 2 A cos(phi - phi0) (phase4/rwm_inputs.md, `lib.py::dn1`).
The points are in fact half that difference (gain 0.46-0.49 against the single
probes on 2018 shots, `SINGLES`), so the fitted A is half the n = 1 field
amplitude; every threshold on it (`rwm.BP_FLOOR_G` ...) was set on this scale.
From 2021 the differences are formed from the single probes on the same scale.

`amplitude` takes the cached 0.5 ms bin means (`labeler.features.namespace`
`mpid067u` ...) and, per channel:

1. puts them on one 0.5 ms grid, tesla to gauss;
2. subtracts a `BASELINE_MS` running median (the raw channels carry 5-20 G static
   offsets): centred for the labels, trailing (`causal=True`) for rwm-ours;
3. removes applied-field pickup (`pickup_fit`): the MPID "U" channels see the
   I-coils directly (Spearman 0.83 between delta-Bp and I-coil activity on
   158021). The channel's deviation is regressed on the coil currents'
   deviations (the six independent I-coils and the three odd C-coil differences,
   each less the same running median) by a two-pass least squares that drops
   samples more than `TRIM_MAD` robust deviations off the first fit, so a burst
   of plasma response does not set the coupling; the fitted pickup is
   subtracted. The fit uses the whole shot and no onset time: nine coupling
   numbers per channel, a calibration, the one non-causal step of the causal
   amplitude;
4. averages over `SMOOTH_MS` (centred, or trailing when causal), which cancels a
   mode rotating at a multiple of 1 / SMOOTH_MS (500 Hz) and attenuates any
   faster one, before the least-squares n = 1 fit.

The amplitude is in gauss and the phase in degrees. A rotating tearing mode below
about 1 kHz, or a locked one, still reaches it: delta-Bp alone does not tell an RWM
from a locked or slowly rotating tearing mode, which is why the screen keeps the
Mirnov track gate and N1RMS rejects rotating modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The probe differences, by canonical feature name, and their toroidal angles.
ANGLES_DEG = {
    "mpid067u": 67.5,
    "mpid097u": 97.4,
    "mpid127u": 127.0,
    "mpid157u": 157.6,
}
#: From 2021 (185805 on) PTDATA has no MPID*U points; each difference is formed
#: from the fast integrated single probes behind it (`differences`). MEASURED on
#: 176088 and 178872 (2018), which have both, after the running median: MPID{a}U
#: = 0.46-0.49 x (MPI66M{a}F - its partner), correlation 0.98-0.99. The 337 deg
#: probe is gone from 2018 on; its partner is the 340 deg probe, 2.4 deg off the
#: antipode (at most 2 % of the n = 1 difference).
SINGLES = {
    "mpid067u": ("mpi66m067f", "mpi66m247f"),
    "mpid097u": ("mpi66m097f", "mpi66m277f"),
    "mpid127u": ("mpi66m127f", "mpi66m307f"),
    "mpid157u": ("mpi66m157f", "mpi66m340f"),
}
SINGLE_NAMES = tuple(name for pair in SINGLES.values() for name in pair)
#: An MPID point is half the probe difference (the measured gain above).
DIFFERENCE_GAIN = 0.5
#: The independent I-coil currents: IU/IL 210, 270 and 330 run antiparallel to
#: 30, 90 and 150 (correlation -1.00 on 156787 and 176088).
ICOILS = ("iu30", "iu90", "iu150", "il30", "il90", "il150")
#: C-coil pairs 180 deg apart; their difference is the odd-n current a
#: probe difference can see (C139 and C319 correlate at only -0.71 on 176088).
CCOIL_PAIRS = (("c79", "c259"), ("c139", "c319"), ("c199", "c19"))
COILS = ICOILS + tuple(c for pair in CCOIL_PAIRS for c in pair)
#: Grid step of the stored bin means and of every series here.
STEP_MS = 0.5
#: Running-median width that removes static offsets and slow drifts.
BASELINE_MS = 100.0
#: Boxcar applied before the fit: zeros at multiples of 500 Hz.
SMOOTH_MS = 2.0
#: Samples further than this many robust deviations off the first pickup fit
#: are left out of the second.
TRIM_MAD = 5.0
#: A coil regressor whose deviation varies less than this (A) is left out: it
#: carries no current to pick up and would only condition the fit badly.
MIN_COIL_STD_A = 5.0
T_TO_G = 1.0e4


def _on_grid(t_src, y_src, grid):
    """The nearest source sample within half a step of each grid time, else NaN."""
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    if len(t_src) < 2:
        return np.full(len(grid), np.nan)
    index = np.clip(np.searchsorted(t_src, grid), 1, len(t_src) - 1)
    left = index - 1
    nearer = np.where(
        np.abs(t_src[index] - grid) < np.abs(t_src[left] - grid), index, left
    )
    out = y_src[nearer].copy()
    out[np.abs(t_src[nearer] - grid) > STEP_MS / 2 + 1e-6] = np.nan
    return out


def _held(t_src, y_src, grid):
    """A slower record (a 5 ms C-coil) linearly interpolated onto the grid, NaN
    outside it and across gaps longer than four of its own steps."""
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    good = np.isfinite(y_src)
    t_src, y_src = t_src[good], y_src[good]
    if len(t_src) < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, t_src, y_src, left=np.nan, right=np.nan)
    step = float(np.median(np.diff(t_src)))
    index = np.clip(np.searchsorted(t_src, grid), 1, len(t_src) - 1)
    out[(t_src[index] - t_src[index - 1]) > 4 * step] = np.nan
    return out


def running_median(y, width_ms=BASELINE_MS, *, causal=False):
    """Median over `width_ms` of samples, centred or trailing (`(t - width, t]`),
    from at least a quarter of a full window; NaNs are skipped."""
    width = max(round(width_ms / STEP_MS), 1)
    series = pd.Series(np.asarray(y, dtype=float))
    rolled = series.rolling(width, center=not causal, min_periods=max(width // 4, 1))
    return rolled.median().to_numpy()


def _mean(y, width_ms, *, causal):
    """Boxcar mean of the rows of `y` over `width_ms`, centred or trailing."""
    width = max(round(width_ms / STEP_MS), 1)
    frame = pd.DataFrame(np.atleast_2d(y).T)
    return frame.rolling(width, center=not causal, min_periods=1).mean().to_numpy().T


def differences(signals):
    """`signals` with each missing probe difference formed from its two single
    probes (`SINGLES`): `DIFFERENCE_GAIN` x (probe - partner) on the probe's
    samples, NaN where the partner has none within half a step. A cached MPID
    point is kept as it is."""
    out = dict(signals)
    for name, (probe, partner) in SINGLES.items():
        if name in out or probe not in out or partner not in out:
            continue
        t, y = (np.asarray(v, dtype=float) for v in out[probe])
        out[name] = (t, DIFFERENCE_GAIN * (y - _on_grid(*out[partner], t)))
    return out


def probe_source(signals) -> str:
    """Which records the four differences come from: "MPID", "single probes",
    "mixed", or "" without all four."""
    have = [name in signals for name in ANGLES_DEG]
    formed = [name in differences(signals) for name in ANGLES_DEG]
    if not all(formed):
        return ""
    if all(have):
        return "MPID"
    return "single probes" if not any(have) else "mixed"


def probe_grid(signals):
    """`(t_ms, Y_gauss)`: the four probe differences (an MPID point, else formed
    from its single probes, `differences`) on one 0.5 ms grid (their common
    span); None without all four."""
    signals = differences(signals)
    if any(name not in signals for name in ANGLES_DEG):
        return None
    starts = [signals[n][0][0] for n in ANGLES_DEG]
    ends = [signals[n][0][-1] for n in ANGLES_DEG]
    grid = np.arange(max(starts), min(ends) + 1e-9, STEP_MS)
    if len(grid) < 2:
        return None
    rows = [_on_grid(*signals[name], grid) * T_TO_G for name in ANGLES_DEG]
    return grid, np.array(rows)


def coil_regressors(signals, grid):
    """`(names, X)`: coil currents in kA on `grid`, the six I-coils as they are and
    the C-coil pairs as differences (a missing partner counts as zero current
    only when the other is present); rows with no record are left out."""
    names, rows = [], []
    for coil in ICOILS:
        if coil in signals:
            names.append(coil)
            rows.append(_held(*signals[coil], grid) / 1000.0)
    for a, b in CCOIL_PAIRS:
        have = [c for c in (a, b) if c in signals]
        if not have:
            continue
        left = _held(*signals[a], grid) if a in signals else 0.0
        right = _held(*signals[b], grid) if b in signals else 0.0
        names.append(f"{a}-{b}")
        rows.append((left - right) / 1000.0 + np.zeros(len(grid)))
    return names, (np.array(rows) if rows else np.zeros((0, len(grid))))


def pickup_fit(dev, X, names):
    """Per channel, the two-pass least-squares coupling of `dev` (C, T) to the coil
    deviations `X` (K, T), with an intercept. Returns `(pickup, info)`: the fitted
    pickup (C, T, NaN where a used regressor is) and per channel the coefficients
    (G per kA), the samples used and the variance share removed."""
    use = [k for k in range(len(X)) if np.nanstd(X[k]) * 1000.0 >= MIN_COIL_STD_A]
    info = {"regressors": [names[k] for k in use], "channels": []}
    pickup = np.zeros_like(dev)
    if not use:
        return pickup, info
    A = np.vstack([np.ones(X.shape[1]), X[use]]).T
    rows_ok = np.isfinite(A).all(axis=1)
    pickup[:, ~rows_ok] = np.nan
    for c, y in enumerate(dev):
        ok = rows_ok & np.isfinite(y)
        if ok.sum() < 10 * A.shape[1]:
            info["channels"].append(None)
            continue
        coef = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
        resid = y[ok] - A[ok] @ coef
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        keep = np.abs(resid - np.median(resid)) <= TRIM_MAD * max(mad, 1e-6)
        index = np.flatnonzero(ok)[keep]
        coef = np.linalg.lstsq(A[index], y[index], rcond=None)[0]
        fitted = A[rows_ok, 1:] @ coef[1:]
        pickup[c, rows_ok] = fitted
        before = np.var(y[index])
        after = np.var(y[index] - A[index] @ coef)
        info["channels"].append(
            {
                "coef_g_per_ka": [round(float(v), 3) for v in coef[1:]],
                "samples": len(index),
                "variance_removed": float(1 - after / before) if before > 0 else 0.0,
            }
        )
    return pickup, info


def n1_fit(Y):
    """`(amplitude_g, phase_deg)` of D(phi) = 2 A cos(phi - phi0) fitted to the
    rows of `Y` (in `ANGLES_DEG` order); NaN where a channel is."""
    phi = np.deg2rad(list(ANGLES_DEG.values()))
    M = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    c = np.linalg.pinv(M) @ np.nan_to_num(Y)
    amplitude = 0.5 * np.hypot(c[0], c[1])
    phase = np.rad2deg(np.arctan2(c[1], c[0]))
    bad = ~np.isfinite(Y).all(axis=0)
    amplitude[bad] = phase[bad] = np.nan
    return amplitude, phase


def amplitude(signals, *, causal=False, pickup=True, fit=None):
    """The n = 1 delta-Bp of one shot from `{name: (t_ms, y)}`, or None without
    all four probes. Returns `t_ms`, `amp_g`, `phase_deg`, `pickup` (the fit's
    info; `removed` False with no coil record or `pickup=False`) and `probes`
    (`probe_source`).

    `causal` takes the trailing running median and boxcar, so a value at t reads
    no sample after t, except through the whole-shot pickup coupling; `fit` reuses
    a coupling found before (`pickup_fit`'s info with its pickup coefficients),
    as the causal amplitude does with the centred one's."""
    got = probe_grid(signals)
    if got is None:
        return None
    grid, Y = got
    dev = Y - np.array([running_median(y, causal=causal) for y in Y])
    info = {"removed": False}
    if pickup:
        names, X = coil_regressors(signals, grid)
        if len(names):
            Xd = X - np.array([running_median(x, causal=causal) for x in X])
            if fit is None:
                found, details = pickup_fit(dev, Xd, names)
            else:
                found, details = _apply(fit, Xd, names), fit
            dev = dev - found
            info = {"removed": bool(details.get("regressors")), **details}
    smooth = _mean(dev, SMOOTH_MS, causal=causal)
    amp, phase = n1_fit(smooth)
    return {
        "t_ms": grid,
        "amp_g": amp,
        "phase_deg": phase,
        "pickup": info,
        "probes": probe_source(signals),
    }


def _apply(fit, X, names):
    """The pickup a previous `pickup_fit` (its info) predicts from `X`."""
    out = np.zeros((len(ANGLES_DEG), X.shape[1]))
    if not fit.get("regressors"):
        return out
    index = [names.index(n) for n in fit["regressors"]]
    rows = X[index]
    for c, channel in enumerate(fit["channels"]):
        if channel is not None:
            out[c] = np.asarray(channel["coef_g_per_ka"]) @ rows
    return out


def both(signals):
    """`(centred, causal)` amplitudes sharing one pickup coupling (the centred
    fit's), or `(None, None)` without the probes."""
    centred = amplitude(signals)
    if centred is None:
        return None, None
    fit = centred["pickup"] if centred["pickup"].get("removed") else None
    causal = amplitude(signals, causal=True, pickup=fit is not None, fit=fit)
    return centred, causal


def coil_activity(signals, grid, *, causal=True):
    """`(icoil_dev_ka, icoil_level_ka)` on `grid`: the largest I-coil deviation from
    its running median (feedback or programmed changes) and the root mean square
    of the I-coil currents (the applied level); NaN without I-coil records."""
    rows = [_held(*signals[c], grid) / 1000.0 for c in ICOILS if c in signals]
    if not rows:
        nan = np.full(len(grid), np.nan)
        return nan, nan
    X = np.array(rows)
    dev = np.abs(X - np.array([running_median(x, causal=causal) for x in X]))
    finite = np.isfinite(X)
    count = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.sqrt(np.where(finite, X**2, 0.0).sum(axis=0) / count)
    return np.fmax.reduce(dev, axis=0), np.where(count > 0, level, np.nan)
