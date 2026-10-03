"""Trailing time-slice calculations on offline inputs for the RWM baseline.

Piccione et al. 2022 resample every input causally to 5 ms and use betaN, its
no-wall and with-wall limits, the rotation and collisionality inside the
pedestal, and the RMS and peak frequency of the low-frequency odd-n magnetics.
DIII-D stores betaN, l_i, q95, W_MHD and the ZIPFIT rotation on the EFIT and
ZIPFIT cadence (20 ms or slower) and the n = 1 and n = 2 magnetic RMS at 1 kHz,
so this module holds a value from its last sample (dropping stale values) and
calculates rates over trailing windows. This does not establish real-time input
availability: N1RMS/N2RMS are postprocessed magnetic amplitudes with uncertain
upstream timing, and ZIPFIT's time smoothing is mildly acausal. Holding samples
cannot remove that processing. The high-current analysis window also uses the
whole-shot peak and is retrospective.

Every function takes plain arrays (time in milliseconds) and is vectorised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Grid step of the slice table, in ms (the paper uses 5 ms; DIII-D's EFIT is 20 ms).
STEP_MS = 10.0
#: A held EFIT or ZIPFIT value older than this is missing, not stale.
EFIT_MAX_AGE_MS = 50.0
ZIPFIT_MAX_AGE_MS = 100.0
#: The RMS window of the paper's MHD feature.
RMS_WINDOW_MS = 5.0
#: Window of the growth-rate feature, about four wall times (tau_w ~ 5 ms).
GROWTH_WINDOW_MS = 20.0
#: Radii of the two rotation features: the core and near the q = 2 surface.
ROTATION_RHO = (0.25, 0.625)
#: The empirical no-wall limit betaN ~ 4 l_i (DIII-D rule of thumb).
NO_WALL_FACTOR = 4.0
#: A growth rate needs this floor so an RMS of exactly zero stays finite.
LOG_FLOOR_G = 0.05
TIME_COLUMN = "time_since_flattop_ms"

FEATURES = (
    "betan",
    "li",
    "q95",
    "qmin",
    "wmhd_mj",
    "betan_over_li",
    "betan_minus_4li",
    "ip_ma",
    "n1rms_g",
    "n1rms_max_g",
    "n1rms_growth_per_s",
    "n2rms_g",
    "rot_core_khz",
    "rot_mid_khz",
    "lock_v",
)


def hold(t_src_ms, y_src, t_grid_ms, max_age_ms):
    """Each grid time's last source value at or before it, NaN when older than the cap."""
    t_src = np.asarray(t_src_ms, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    grid = np.asarray(t_grid_ms, dtype=float)
    index = np.searchsorted(t_src, grid, side="right") - 1
    ok = index >= 0
    index = np.clip(index, 0, max(len(t_src) - 1, 0))
    out = np.where(ok, y_src[index], np.nan)
    age = np.where(ok, grid - t_src[index], np.inf)
    out[age > max_age_ms] = np.nan
    return out


def _window_bounds(t_src, grid, width_ms):
    """Source index ranges `[lo, hi)` of the samples in `(t - width, t]`."""
    hi = np.searchsorted(t_src, grid, side="right")
    lo = np.searchsorted(t_src, grid - width_ms, side="right")
    return lo, hi


def trailing_mean(t_src_ms, y_src, t_grid_ms, width_ms, min_fraction=0.5):
    """Mean of the finite samples in `(t - width, t]`, NaN when too few arrived.

    `min_fraction` is of the samples a full window would hold at the source's median
    step, so a gap in the record gives NaN and not a mean of one stray sample.
    """
    t_src = np.asarray(t_src_ms, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    grid = np.asarray(t_grid_ms, dtype=float)
    finite = np.isfinite(y_src)
    csum = np.concatenate([[0.0], np.cumsum(np.where(finite, y_src, 0.0))])
    ccount = np.concatenate([[0], np.cumsum(finite)])
    lo, hi = _window_bounds(t_src, grid, width_ms)
    count = ccount[hi] - ccount[lo]
    step = float(np.median(np.diff(t_src))) if len(t_src) > 1 else np.inf
    need = max(1.0, min_fraction * width_ms / step)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (csum[hi] - csum[lo]) / count
    return np.where(count >= need, mean, np.nan)


def trailing_max(t_src_ms, y_src, t_grid_ms, width_ms):
    """Maximum of the finite samples in `(t - width, t]`, NaN when there are none."""
    t_src = np.asarray(t_src_ms, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    grid = np.asarray(t_grid_ms, dtype=float)
    lo, hi = _window_bounds(t_src, grid, width_ms)
    out = np.full(len(grid), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            window = y_src[a:b]
            if np.isfinite(window).any():
                out[i] = np.nanmax(window)
    return out


def trailing_log_slope(t_src_ms, y_src, t_grid_ms, width_ms, floor):
    """Growth rate in 1/s: least-squares slope of `log(max(y, floor))` in the window.

    NaN with fewer than half the window's samples or a flat time axis. The slope of a
    log amplitude is the exponential growth rate, which is what a resistive wall mode
    shows on the wall time while a rotating mode's RMS does not grow steadily.
    """
    t_src = np.asarray(t_src_ms, dtype=float) / 1000.0
    y_src = np.asarray(y_src, dtype=float)
    grid = np.asarray(t_grid_ms, dtype=float)
    finite = np.isfinite(y_src)
    z = np.where(finite, np.log(np.maximum(y_src, floor)), 0.0)
    t = np.where(finite, t_src, 0.0)
    cumulative = {
        "n": np.cumsum(finite),
        "t": np.cumsum(t),
        "z": np.cumsum(z),
        "tt": np.cumsum(t * t),
        "tz": np.cumsum(t * z),
    }
    cumulative = {k: np.concatenate([[0], v]) for k, v in cumulative.items()}
    lo, hi = _window_bounds(t_src * 1000.0, grid, width_ms)
    s = {k: v[hi] - v[lo] for k, v in cumulative.items()}
    step = float(np.median(np.diff(t_src))) * 1000.0 if len(t_src) > 1 else np.inf
    need = max(3.0, 0.5 * width_ms / step)
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = s["n"] * s["tt"] - s["t"] ** 2
        slope = (s["n"] * s["tz"] - s["t"] * s["z"]) / denominator
    return np.where((s["n"] >= need) & (denominator > 1e-12), slope, np.nan)


def profile_at(t_src_ms, profile, rho_grid, rho, t_grid_ms, max_age_ms):
    """A held `(rho, T)` profile sampled at one radius by linear interpolation."""
    profile = np.asarray(profile, dtype=float)
    rho_grid = np.asarray(rho_grid, dtype=float)
    low = int(
        np.clip(np.searchsorted(rho_grid, rho, side="right") - 1, 0, len(rho_grid) - 2)
    )
    weight = (rho - rho_grid[low]) / (rho_grid[low + 1] - rho_grid[low])
    row = (1 - weight) * profile[low] + weight * profile[low + 1]
    return hold(t_src_ms, row, t_grid_ms, max_age_ms)


def flattop_window(t_ms, ip, fraction):
    """`(start, end)` ms of the longest run of samples with |Ip| above `fraction` of peak."""
    t = np.asarray(t_ms, dtype=float)
    level = np.abs(np.asarray(ip, dtype=float))
    finite = np.isfinite(level)
    if not finite.any():
        return None
    above = finite & (level >= fraction * np.max(level[finite]))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    best = None
    for a, b in zip(edges[::2], edges[1::2]):
        if best is None or t[b - 1] - t[a] > best[1] - best[0]:
            best = (float(t[a]), float(t[b - 1]))
    return best


def time_since_flattop(t_ms, ip, grid_ms, threshold_a=0.5e6):
    """Causal elapsed time from the first |Ip| >= 0.5 MA sample, in ms.

    This fixed crossing is an operational proxy for flat-top start, independent of
    the future peak used to select the analysis window. Before crossing it is NaN.
    """
    t, level, grid = (np.asarray(v, dtype=float) for v in (t_ms, ip, grid_ms))
    crossed = np.flatnonzero(np.isfinite(level) & (np.abs(level) >= threshold_a))
    if not len(crossed):
        return np.full(len(grid), np.nan)
    elapsed = grid - t[crossed[0]]
    return np.where(elapsed >= 0, elapsed, np.nan)


def slice_table(signals, *, step_ms=STEP_MS, ip_fraction=0.5):
    """One offline-input feature row per grid time in the high-current window.

    `signals` maps a canonical feature name to `(t_ms, y)` with `y` one scalar trace,
    or `(33, T)` for `rot_zipfit` (rows on the ZIPFIT radial grid `rho_grid`), plus the
    key `rho_grid` for that. A missing signal gives an all-NaN column, so a shot with
    no rotation profile still yields rows and the caller decides how to treat them.
    """
    ip_t, ip_y = signals["ip"]
    window = flattop_window(ip_t, ip_y, ip_fraction)
    if window is None:
        return pd.DataFrame(columns=["t_ms", *FEATURES, TIME_COLUMN])
    grid = np.arange(np.ceil(window[0] / step_ms) * step_ms, window[1] + 1e-9, step_ms)
    nan = np.full(len(grid), np.nan)

    def scalar(name):
        if name not in signals:
            return nan
        t, y = signals[name]
        return hold(t, y, grid, EFIT_MAX_AGE_MS)

    betan, li = scalar("betan"), scalar("li")
    table = {
        "t_ms": grid,
        "betan": betan,
        "li": li,
        "q95": scalar("q95"),
        "qmin": scalar("qmin"),
    }
    table["wmhd_mj"] = scalar("wmhd") / 1e6
    with np.errstate(invalid="ignore", divide="ignore"):
        table["betan_over_li"] = np.where(li > 0, betan / li, np.nan)
    table["betan_minus_4li"] = betan - NO_WALL_FACTOR * li
    table["ip_ma"] = trailing_mean(ip_t, np.abs(ip_y), grid, step_ms) / 1e6
    for name, key in (("n1rms", "n1rms_g"), ("n2rms", "n2rms_g")):
        if name in signals:
            t, y = signals[name]
            table[key] = trailing_mean(t, y, grid, RMS_WINDOW_MS)
        else:
            table[key] = nan
    if "n1rms" in signals:
        t, y = signals["n1rms"]
        table["n1rms_max_g"] = trailing_max(t, y, grid, GROWTH_WINDOW_MS)
        table["n1rms_growth_per_s"] = trailing_log_slope(
            t, y, grid, GROWTH_WINDOW_MS, LOG_FLOOR_G
        )
    else:
        table["n1rms_max_g"] = table["n1rms_growth_per_s"] = nan
    if "rot_zipfit" in signals and "rho_grid" in signals:
        t, profile = signals["rot_zipfit"]
        for key, rho in zip(("rot_core_khz", "rot_mid_khz"), ROTATION_RHO):
            table[key] = profile_at(
                t, profile, signals["rho_grid"], rho, grid, ZIPFIT_MAX_AGE_MS
            )
    else:
        table["rot_core_khz"] = table["rot_mid_khz"] = nan
    if "dusbradial" in signals:
        t, y = signals["dusbradial"]
        table["lock_v"] = trailing_mean(t, np.abs(y), grid, RMS_WINDOW_MS)
    else:
        table["lock_v"] = nan
    table[TIME_COLUMN] = time_since_flattop(ip_t, ip_y, grid)
    return pd.DataFrame(table, columns=["t_ms", *FEATURES, TIME_COLUMN])
