"""Read the per-shot inputs the indicators need, from the corpus and the fetch cache.

Two homes, one rule. The corpus (`/scratch/gpfs/EKOLEMEN/foundation_model`) holds
what it holds; the fetch cache (`$LABELER_ROOT/round4/detach/cache/<shot>.npz`,
written by `scripts/labeler/detach_fetch.py`) holds the rest. A signal neither holds
is `None`, and the indicator that needs it is invalid for that reason; this module
never fetches.

All times returned are milliseconds, all powers watts.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def labeler_root() -> Path:
    return Path(os.environ["LABELER_ROOT"])


def cache_file(shot: int) -> Path:
    return labeler_root() / "round4" / "detach" / "cache" / f"{int(shot)}.npz"


def load_cache(shot: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every fetched node of a shot as `{name: (t_ms, y)}`; `{}` if never fetched."""
    path = cache_file(shot)
    if not path.is_file():
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(path) as npz:
        for key in npz.files:
            if key.endswith("__y"):
                name = key[:-3]
                out[name] = (npz[name + "__t"], npz[key])
    return out


def corpus_group(
    shot: int, group: str, channels=None
) -> tuple[np.ndarray, np.ndarray] | None:
    """`(t_ms, y[C, T])` of a corpus group, or None when absent or a stub."""
    import h5py

    path = CORPUS / f"{int(shot)}_processed.h5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        if group not in f:
            return None
        ydata = f[group]["ydata"]
        if ydata.shape[-1] <= 1:
            return None
        t = np.asarray(f[group]["xdata"][:], dtype=float) * 1000.0
        if channels is None:
            y = np.asarray(ydata[:], dtype=np.float32)
        else:
            y = np.asarray(ydata[list(channels), :], dtype=np.float32)
    return t, y


def decimate_median(t_ms: np.ndarray, y: np.ndarray, dt_ms: float, *, mean=False):
    """Block decimation to `dt_ms`, for records far finer than any use.

    The median by default (it ignores a spike); `mean=True` for a power, where
    a modulated beam's duty cycle is the signal and a median would read it as 0.
    """
    reduce = np.nanmean if mean else np.nanmedian
    t_ms = np.asarray(t_ms, dtype=float)
    step = max(1, round(dt_ms / max(np.median(np.diff(t_ms)), 1e-9)))
    n = (len(t_ms) // step) * step
    if n == 0:
        return t_ms, np.asarray(y)
    t = t_ms[:n].reshape(-1, step).mean(axis=1)
    if y.ndim == 1:
        return t, reduce(y[:n].reshape(-1, step), axis=1)
    return t, reduce(y[:, :n].reshape(y.shape[0], -1, step), axis=2)


def window_mean(t_src, y_src, t_dst, width_ms, *, keep=None):
    """Centered mean with complete source coverage; missing windows remain NaN.

    A keep mask removes measured ELM samples from radiation means. It does not
    turn missing source samples into measurements. Gaps/boundaries are checked
    before reduction, independently of the mask. Both endpoints are included
    in the reduction and in its availability check.
    """
    from .core import sample_windows_known

    t_src = np.asarray(t_src, dtype=float)
    y = np.asarray(y_src, dtype=float)
    good = np.isfinite(y)
    selected = good if keep is None else good & np.asarray(keep, dtype=bool)
    csum = np.r_[0.0, np.cumsum(np.where(selected, y, 0.0))]
    counts = np.r_[0, np.cumsum(selected)]
    starts, stops = np.asarray(t_dst) - width_ms / 2, np.asarray(t_dst) + width_ms / 2
    lo = np.searchsorted(t_src, starts, side="left")
    hi = np.searchsorted(t_src, stops, side="right")
    n = counts[hi] - counts[lo]
    known = sample_windows_known(t_src, good, starts, stops, closed_right=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(known & (n > 0), (csum[hi] - csum[lo]) / n, np.nan)


def beam_power(shot: int, cache, t_dst, width_ms: float):
    """Neutral-beam power (W) averaged over `width_ms` windows at `t_dst`, or None.

    The corpus `pinj` (rows are beams, in watts) when it is a real record; else the
    fetched `\\NB::PINJ` total (kW); else the fetched PTDATA `BMSPINJ` total
    (`pinj_bms`, megawatts: the beam management system's total, checked against the
    corpus beams in `detachment_nbi_calibration.json`). Averaged, not sampled: a
    beam may blip. None when no source holds a record: Prad,div cannot be
    normalised without the beam power, and a missing record is never zero.
    """
    corpus = corpus_group(shot, "pinj")
    if corpus is not None:
        t, y = corpus
        return window_mean(t, np.sum(y, axis=0), t_dst, width_ms)
    if "pinj_total" in cache and np.isfinite(cache["pinj_total"][1]).any():
        t, y = cache["pinj_total"]
        return window_mean(t, y, t_dst, width_ms) * 1e3
    if "pinj_bms" in cache and np.isfinite(cache["pinj_bms"][1]).any():
        t, y = cache["pinj_bms"]
        return window_mean(t, y, t_dst, width_ms) * 1e6
    return None


#: ECHPWR is the total gyrotron power in kW; readings under this are the offset.
ECH_NOISE_KW = 100.0


def heating_power(shot: int, cache=None):
    """`(t_ms, p_heat_W, p_sol_W)` on the EFIT time base, or None without EFIT.

    `p_heat` = neutral beams + ohmic (EFIT `poh`) + ECH; `p_sol` subtracts the
    stored-energy rate dW/dt from EFIT's `wmhd` (Eldon 2022: P_SOL = P_NBI + P_OHM
    - dW/dt). Core radiation is NOT subtracted, so `p_sol` is an upper bound of the
    power crossing the separatrix, off by a slowly varying factor that the Afrac
    reference level absorbs. A shot with
    no beam record at all (stub corpus, no fetched beams) returns None: Prad,div
    cannot be normalised without knowing its input power.
    """
    cache = load_cache(shot) if cache is None else cache
    if "poh" not in cache or "wmhd" not in cache:
        return None
    t, poh = (np.asarray(v, dtype=float) for v in cache["poh"])
    beams = beam_power(shot, cache, t, 20.0)
    if beams is None:
        return None
    pheat = beams + poh
    if "echpwr" in cache:
        te, ye = cache["echpwr"]
        ech = window_mean(te, ye, t, 20.0) * 1e3
        ech = np.where(np.isfinite(ech), np.maximum(ech, 0.0), np.nan)
        pheat = pheat + np.where(ech > ECH_NOISE_KW * 1e3, ech, ech * 0.0)
    tw, w = (np.asarray(v, dtype=float) for v in cache["wmhd"])
    smooth = np.convolve(w, np.ones(5) / 5.0, mode="same")
    dwdt = np.gradient(smooth, tw / 1000.0)
    dwdt[:3] = dwdt[-3:] = 0.0
    return t, pheat, pheat - np.interp(t, tw, dwdt)


def line_density(cache):
    """`(t_ms, ne)` line-integrated density, arbitrary units, or None.

    The vertical CO2 chord V2, with R0 and V3 as fallbacks. Only ratios of this
    enter the Afrac model (the ne^2 scaling, referenced to the shot's own level),
    so the unit does not matter.
    """
    for name in ("denv2uf", "denr0uf", "denv3uf"):
        if name in cache and np.isfinite(cache[name][1]).any():
            t, y = cache[name]
            return np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    return None


def density_line_si(y, units):
    """Line density in m^-2 only from explicit units, never from magnitude.

    DIII-D ELECTRONS::TOP.BCI.MAIN:DENV2 commonly uses `m/cm3`: path in
    metres times density in cm^-3 (also documented in shot_design.flags.rules).
    Multiplication by 1e6 converts that mixed convention to m^-2.
    """
    unit = str(units).lower().replace(" ", "").replace("**", "^")
    factors = {
        "m/cm3": 1e6,
        "m/cm^3": 1e6,
        "cm^-2": 1e4,
        "cm-2": 1e4,
        "1/cm2": 1e4,
        "m^-2": 1.0,
        "m-2": 1.0,
        "1/m2": 1.0,
    }
    if unit not in factors:
        return None
    return np.asarray(y, dtype=float) * factors[unit]


def greenwald_fraction(edges, cache):
    """Unit-confirmed V2 line-average / nG, using nG=|Ip[MA]|/(pi*a^2).

    The elliptical chord at R=1.94 m ignores triangularity and is approximate
    (about 10-20 percent per the existing Shot Designer method). It is a
    corroborating density-limit cue, not independent MARFE truth.
    """
    from .core import bin_median

    # AEQDSK R0 is the geometric centre; ROUT is another quantity and is not
    # an axis major radius (it can be near 0.1 on these shots).
    required = ("density_v2_si", "aminor", "kappa", "r0", "ipmeas")
    if any(k not in cache for k in required):
        return np.full(len(edges) - 1, np.nan)
    density, a, kappa, axis, ip = [bin_median(*cache[k], edges)[0] for k in required]
    with np.errstate(invalid="ignore", divide="ignore"):
        length = 2 * kappa * np.sqrt(a * a - (1.94 - axis) ** 2)
        n_g = np.abs(ip) / 1e6 / (np.pi * a * a)
        fraction = density / length / 1e20 / n_g
    good = (density > 0) & (a > 0) & (kappa > 0) & (n_g > 0)
    return np.where(good & np.isfinite(fraction), fraction, np.nan)


def load_flux_map(shot):
    """Usable multi-slice EFIT01 or EFIT02 flux map, with its source retained."""
    path = labeler_root() / "round4/detach/efit" / f"{shot}.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        if len(f["gtime_ms"]) < 2 or str(f.get("source", "EFIT01")) not in (
            "EFIT01",
            "EFIT02",
        ):
            return None
        return {k: f[k] for k in f.files}


def flux_at_positions(maps, t_ms, positions, max_gap_ms=40.0):
    """psiN at fixed (R,Z) probe positions from close multi-slice EFIT maps.

    Both EFIT01 and EFIT02 are accepted; a single equilibrium cannot describe
    an evolving discharge. Return (probe,time), with NaN outside coverage.
    """
    from scipy.interpolate import RegularGridInterpolator

    t_ms, positions = np.asarray(t_ms), np.asarray(positions)
    out = np.full((len(positions), len(t_ms)), np.nan)
    if maps is None or len(maps["gtime_ms"]) < 2:
        return out
    mt = maps["gtime_ms"]
    near = np.abs(mt[:, None] - t_ms[None, :]).argmin(axis=0)
    close = np.abs(mt[near] - t_ms) <= max_gap_ms
    for k in np.unique(near[close]):
        scale = maps["ssibry"][k] - maps["ssimag"][k]
        if not np.isfinite(scale) or scale == 0:
            continue
        psin = (maps["psirz"][k] - maps["ssimag"][k]) / scale
        interp = RegularGridInterpolator(
            (maps["z"], maps["r"]), psin, bounds_error=False, fill_value=np.nan
        )
        out[:, close & (near == k)] = interp(positions[:, ::-1])[:, None]
    return out


def elm_mask(shot: int, cache=None):
    """`(t_ms, flag)` at 10 kHz: True where the divertor D-alpha is inside an ELM.

    The median over the live corpus filterscope rows FS01-FS08 (each divided by its
    own median), against a 50 ms running median; a sample is in an ELM when it
    exceeds that baseline by `ELM_SIGMA` robust standard deviations of the residual
    (MAD x 1.4826) and by `ELM_MIN_REL_RISE` of the baseline, widened by +/-2 ms.
    Prad and Jsat use this inter-ELM mask. Camera frames integrate ELMs as in
    Chen 2026, so TangTV requires coverage but applies no overlap veto.
    Cached FS01-FS04 supplement a missing corpus record; None if neither home
    holds a live D-alpha channel.
    """
    from scipy.ndimage import maximum_filter1d, median_filter

    from labeler.events.detachment import thresholds

    got = corpus_group(shot, "filterscopes", channels=range(8))
    if got is not None and not any(
        np.isfinite(row).mean() > 0.9 and np.nanmedian(row) > 0 for row in got[1]
    ):
        got = None
    if got is None:
        cache = load_cache(shot) if cache is None else cache
        records = []
        for name in ("fs01", "fs02", "fs03", "fs04"):
            if name not in cache:
                continue
            tc, yc = cache[name]
            if len(tc) > 1 and np.isfinite(yc).mean() > 0.9 and np.nanmedian(yc) > 0:
                records.append((tc, yc))
        if not records:
            return None
        # Intersect the live records; do not extrapolate unknown D-alpha time.
        start, stop = max(t[0] for t, _ in records), min(t[-1] for t, _ in records)
        if stop <= start:
            return None
        t = np.arange(start, stop + 0.05, 0.1)
        rows = []
        for tc, yc in records:
            row = np.interp(t, tc, yc)
            step_c = float(np.median(np.diff(tc)))
            for j in np.flatnonzero(np.diff(tc) > max(2 * step_c, 2.0)):
                row[(t > tc[j]) & (t < tc[j + 1])] = np.nan
            rows.append(row)
        got = t, np.vstack(rows)
    t, y = got
    live = [r for r in y if np.isfinite(r).mean() > 0.9 and np.nanmedian(r) > 0]
    if not live:
        return None
    normalized = np.asarray([r / np.nanmedian(r) for r in live])
    x = np.ma.median(np.ma.masked_invalid(normalized), axis=0).filled(np.nan)
    available = np.isfinite(x)
    # Fill only for the running-filter calculation; restore unknown samples below.
    filtered_input = np.interp(t, t[available], x[available])
    step = float(np.median(np.diff(t)))
    base = median_filter(
        filtered_input, size=max(3, round(50.0 / step)), mode="nearest"
    )
    resid = x - base
    sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    flag = (resid > thresholds.ELM_SIGMA * max(sigma, 1e-9)) & (
        resid > thresholds.ELM_MIN_REL_RISE * base
    )
    half = round(thresholds.ELM_MASK_HALF_WIDTH_MS / step)
    widened = maximum_filter1d(
        flag.astype(np.uint8),
        size=2 * half + 1,
        mode="constant",
    )
    return t, np.where(available, widened.astype(float), np.nan)


#: The suggestion table the confinement review opens on: curated spans where a shot
#: has them, else the D-alpha H-mode method's. Categories 1 high, 2 low, 3 qh,
#: 4 wpqh, 5 uncertain; 0 is time in which nothing is observable.
REGIME_TABLE = "suggestions/regimes/v1/confinement_suggest_regimes_v1.csv"
_REGIME_L = (2,)
_REGIME_H = (1, 3, 4)


def regime(shot: int, centres_ms) -> tuple[np.ndarray, str]:
    """`(regime, source)`: `L`, `H` or `unknown` on each bin centre, and where from.

    Eldon 2022's Afrac model holds in H-mode, so the exporter gates L-mode out. The
    regime comes from, in order, the confinement suggestion table when it marks
    the shot's bins low or high (`regime_table`), else the D-alpha H-mode method
    (`spans.detect_hmode`: L is the stretch it measured outside its H-mode spans,
    `dalpha_detector`), else it is `unknown` (`none`) and nothing is gated. A bin
    in the H-mode method's uncertain spans, or outside what it measured, is
    `unknown`. Coverage is partial (the method needs the corpus D-alpha, CO2 and
    beams), which is why the caller exports the regime beside the gate.
    """
    import pandas as pd

    from labeler.config import Paths
    from labeler.events import spans

    centres = np.asarray(centres_ms, dtype=float)
    out = np.full(len(centres), "unknown", dtype="U7")
    paths = Paths.from_env()
    table = paths.root / REGIME_TABLE
    if table.is_file():
        rows = pd.read_csv(table).query("shot == @shot")
        for row in rows.itertuples():
            hit = (centres >= row.t_start) & (centres < row.t_end)
            out[hit & np.isin(row.category, _REGIME_L)] = "L"
            out[hit & np.isin(row.category, _REGIME_H)] = "H"
        if (out != "unknown").any():
            return out, "regime_table"
    try:
        found = spans.detect_hmode(int(shot), paths)
    except spans.INPUT_MISSING:
        return out, "none"
    measured = [(a, b) for a, b in found.measured]
    held = [(a, b) for a, b, _ in found.spans]
    for a, b in spans.minus(measured, held):
        out[(centres >= a) & (centres < b)] = "L"
    for a, b, state in found.spans:
        if state == spans.PRESENT:
            out[(centres >= a) & (centres < b)] = "H"
    return out, "dalpha_detector"


#: Regime values the TangTV L-mode gate acts on: known L-mode and the probable-L proxy.
GATED_REGIMES = ("L", "probable_L")


def probable_regimes(
    regime, source, elm_known, elm_share, p_in_w, width_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(regime, source)` with the unknown-regime bins typed by ELMs and power.

    The confinement table and the D-alpha H-mode detector leave most shots without a
    regime (the detector needs CO2, absent on every 189xxx and 190xxx shot). The bins
    of a shot whose regime is `unknown`, whose ELM coverage is known and whose input
    power is measured form its window: the stretch where both inputs of the proxy
    exist, so the ELM test and the power test read the same bins (an ELM flag on a
    bin with no input power, in the current ramp or after the end of the heating, is
    outside the window). The window is

    * `probable_L` when the D-alpha ELM detector flags no ELM in any bin of it
      (zero ELM share) and its median input power (`p_in_w`, W) is below
      `thresholds.PROBABLE_L_MAX_P_IN_W` (the Martin 2008 L-H threshold range). The
      TangTV detached vote there is gated like a known L-mode one;
    * `probable_H` when it has any ELM share and is not `probable_L`: ELMs are the
      textbook H-mode signature. Report only; nothing is gated on it.

    A window shorter than `PROBABLE_REGIME_MIN_MS` (in bins at `width_ms`) and every
    bin of known regime keep what they had. The new value is also written to the source, so a table can split known
    from probable. Known coverage is the only basis: an unknown ELM coverage never
    types a bin.
    """
    from labeler.events.detachment import thresholds

    regime = np.asarray(regime)
    out = np.array(regime, dtype="U12")
    src = np.array(np.broadcast_to(np.asarray(source), regime.shape), dtype="U16")
    window = (
        (regime == "unknown")
        & np.asarray(elm_known, bool)
        & np.isfinite(np.asarray(elm_share, dtype=float))
        & np.isfinite(np.asarray(p_in_w, dtype=float))
    )
    need = thresholds.min_bins(thresholds.PROBABLE_REGIME_MIN_MS, width_ms)
    if window.sum() < need:
        return out, src
    power = np.asarray(p_in_w, dtype=float)[window]
    share = np.asarray(elm_share, dtype=float)[window]
    quiet = share.max() == 0.0
    if quiet and float(np.median(power)) < thresholds.PROBABLE_L_MAX_P_IN_W:
        label = "probable_L"
    elif share.max() > 0.0:
        label = "probable_H"
    else:
        return out, src
    out[window] = label
    src[window] = label
    return out, src


def tangtv_geometry(shot, cache=None):
    """EFIT02 for camera geometry; explicit EFIT01 fallback for missing records.

    Heating/current remain EFIT01. The surrogate requires EFIT02 and does not
    deploy across sources. Inversions can use a flagged EFIT01 fallback.
    """
    cache = load_cache(shot) if cache is None else dict(cache)
    path = labeler_root() / "round4/detach/geometry02" / f"{shot}.npz"
    if path.exists():
        with np.load(path) as f:
            if len(f["t_ms"]) < 2:
                return cache, "EFIT01_fallback_EFIT02_sparse"
            for name in ("rvsod", "zvsod", "rxpt1", "zxpt1"):
                cache[name] = (f["t_ms"], f[name])
        return cache, "EFIT02"
    return cache, "EFIT01_fallback"
