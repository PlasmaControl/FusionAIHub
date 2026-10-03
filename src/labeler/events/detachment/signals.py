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


def window_mean(t_src, y_src, t_dst, width_ms):
    """Mean of `y_src` over `width_ms` centred on each `t_dst` (NaN where empty)."""
    t_src = np.asarray(t_src, dtype=float)
    y = np.nan_to_num(np.asarray(y_src, dtype=float))
    csum = np.r_[0.0, np.cumsum(y)]
    lo = np.searchsorted(t_src, np.asarray(t_dst) - width_ms / 2, side="left")
    hi = np.searchsorted(t_src, np.asarray(t_dst) + width_ms / 2, side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (csum[hi] - csum[lo]) / n, np.nan)


def beam_power(shot: int, cache, t_dst, width_ms: float):
    """Neutral-beam power (W) averaged over `width_ms` windows at `t_dst`, or None.

    The corpus `pinj` (rows are beams, in watts) when it is a real record, else the
    fetched `\\NB::PINJ` total (kW). Averaged, not sampled: a beam may blip.
    """
    corpus = corpus_group(shot, "pinj")
    if corpus is not None:
        t, y = corpus
        return window_mean(t, np.nansum(y, axis=0), t_dst, width_ms)
    if "pinj_total" in cache and np.isfinite(cache["pinj_total"][1]).any():
        t, y = cache["pinj_total"]
        return window_mean(t, y, t_dst, width_ms) * 1e3
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
    pheat = np.nan_to_num(beams) + np.nan_to_num(poh)
    if "echpwr" in cache:
        te, ye = cache["echpwr"]
        ech = window_mean(te, ye, t, 20.0) * 1e3
        pheat = pheat + np.where(np.nan_to_num(ech) > ECH_NOISE_KW * 1e3, ech, 0.0)
    tw, w = (np.asarray(v, dtype=float) for v in cache["wmhd"])
    smooth = np.convolve(np.nan_to_num(w), np.ones(5) / 5.0, mode="same")
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


def elm_mask(shot: int):
    """`(t_ms, flag)` at 10 kHz: True where the divertor D-alpha is inside an ELM.

    The median over the live corpus filterscope rows FS01-FS08 (each divided by its
    own median), against a 50 ms running median; a sample is in an ELM when it
    exceeds that baseline by `ELM_SIGMA` robust standard deviations of the residual
    (MAD x 1.4826), widened by 1 ms each side. None if the corpus has no record.
    """
    from scipy.ndimage import maximum_filter1d, median_filter

    from labeler.events.detachment import thresholds

    got = corpus_group(shot, "filterscopes", channels=range(8))
    if got is None:
        return None
    t, y = got
    live = [r for r in y if np.isfinite(r).mean() > 0.9 and np.nanmedian(r) > 0]
    if not live:
        return None
    x = np.nanmedian([np.nan_to_num(r) / np.nanmedian(r) for r in live], axis=0)
    step = float(np.median(np.diff(t)))
    base = median_filter(x, size=max(3, round(50.0 / step)), mode="nearest")
    resid = x - base
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    flag = resid > thresholds.ELM_SIGMA * max(sigma, 1e-9)
    widen = max(1, round(1.0 / step)) * 2 + 1
    return t, maximum_filter1d(flag.astype(np.uint8), size=widen).astype(bool)
