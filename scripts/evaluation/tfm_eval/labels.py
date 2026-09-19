"""Downstream-task labels derived from RAW HDF5 shot files.

Produces per-window supervision targets for the eval suite directly from
``<shot>_processed.h5`` (flat groups of ``xdata`` (T,) / ``ydata`` (C, T)
float32), *not* via :class:`TokamakH5Dataset`.  Windows follow the eval grid
convention: ``t_s = warmup_s + idx * stride`` with ``window_s`` (=
``chunk_duration_s``) long windows; all times in this API are SECONDS.

Labels
------
* :func:`elm_labels` — ELM detection on the fs00–fs07 D-alpha filterscopes
  (``filterscopes`` channels 0:8): per-window ``elm_now`` / ``elm_next``
  (prediction target) booleans, peak counts, composite maxima, plus shot-level
  peak list and ELM rate.
* :func:`band_power_labels` — per-window log10 band power (and broadband RMS)
  of high-rate magnetics (``mirnov``, ``mhr``) without ever loading the full
  ``ydata`` (mirnov is ~950 MB per shot).

Data quirks (measured on /lustre/orion/fus187/proj-shared/foundation_model)
---------------------------------------------------------------------------
* **xdata is stored in SECONDS in these files**, although older docs describe
  it as milliseconds.  Evidence: ``filterscopes`` spans -0.05..6.95 with 70001
  samples = 10 kHz only if seconds (10 MHz if ms); ``mirnov``/``mhr``/``ece``
  come out at 500 kHz and Thomson at 100 Hz under the seconds reading, and
  ``data_loader.py`` itself treats ``xdata`` as seconds.  Unit is
  auto-detected per group from the time span (span > ``_MS_SPAN_THRESHOLD``
  file units => milliseconds) so the module keeps working either way.
* Filterscope ``ydata`` is in photon-flux-like units (~1e14–1e16); all ELM
  thresholds are therefore scale-free (per-shot MAD based).
* The first and last sample of each filterscope channel are NaN; channels are
  sanitised before use.
* Missing modalities are length-1 stubs (``xdata`` shape ``(1,)``).

ELM detector
------------
Composite = max over the channel slice of per-channel median-subtracted
D-alpha.  ``scipy.signal.find_peaks`` with per-shot adaptive thresholds:
prominence >= ``k_mad * MAD(composite)`` (global MAD), height >= rolling
median baseline + ``k_mad * MAD``, distance >= ``min_distance_s``.  A
*threshold-stability gate* keeps no-ELM shots from producing garbage: the
detection is repeated at ``stability_factor``x the threshold, and if fewer
than ``stability_min_ratio`` of the candidates survive, the candidate
population is threshold-riding noise; only peaks exceeding the expected
extreme of a pure-noise record (``noise_extreme_mult * sigma *
sqrt(2 ln N)``, ``sigma = 1.4826 MAD``) are kept — typically none.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

__all__ = [
    "modality_present",
    "elm_labels",
    "band_power_labels",
    "detect_elm_peaks",
]

# Time spans larger than this (in raw file units) are taken to be
# milliseconds; DIII-D shots last <~ 15 s, i.e. >~ 1000 in ms.
_MS_SPAN_THRESHOLD = 60.0

# Relative tolerance on the local sample spacing when verifying that a
# group's time base is uniform (measured deviation on these files: ~3e-5).
_UNIFORMITY_RTOL = 5e-3


# ---------------------------------------------------------------------------
# Time-base helpers
# ---------------------------------------------------------------------------
def modality_present(h5: "h5py.File", name: str) -> bool:
    """True if group ``name`` exists and is not a length-1 missing-data stub."""
    if name not in h5:
        return False
    group = h5[name]
    if "xdata" not in group:
        return False
    return group["xdata"].shape[0] > 1


def _unit_scale_to_seconds(span_file_units: float) -> float:
    """Multiplier converting the group's raw time units to seconds."""
    return 1e-3 if span_file_units > _MS_SPAN_THRESHOLD else 1.0


class _TimeBase:
    """Uniform time base of one HDF5 group, read WITHOUT loading full xdata.

    Attributes are in seconds: ``t0_s`` (first sample time), ``dt_s``
    (sample spacing), ``n`` (number of samples), ``t_end_s`` (last sample).
    """

    __slots__ = ("t0_s", "dt_s", "n", "t_end_s")

    def __init__(self, t0_s: float, dt_s: float, n: int):
        self.t0_s = float(t0_s)
        self.dt_s = float(dt_s)
        self.n = int(n)
        self.t_end_s = self.t0_s + (self.n - 1) * self.dt_s

    @classmethod
    def from_group(cls, group: "h5py.Group", n_check: int = 64) -> "_TimeBase":
        x = group["xdata"]
        n = int(x.shape[0])
        if n < 2:
            raise ValueError("length-1 stub group has no time base")
        x0 = float(x[0])
        x_last = float(x[-1])
        scale = _unit_scale_to_seconds(x_last - x0)
        dt = (x_last - x0) / (n - 1)
        # Verify uniformity on a small subsample (never the full array).
        idx = np.unique(np.linspace(0, n - 1, min(n_check, n)).astype(np.int64))
        xs = x[idx].astype(np.float64)
        local = np.diff(xs) / np.diff(idx)
        max_dev = float(np.max(np.abs(local - dt))) if len(local) else 0.0
        if dt <= 0 or max_dev > _UNIFORMITY_RTOL * abs(dt):
            warnings.warn(
                f"non-uniform time base in group '{group.name}': "
                f"dt={dt:.3e}, max local deviation={max_dev:.3e}",
                stacklevel=2,
            )
        return cls(x0 * scale, dt * scale, n)

    def window_slice(self, t_start_s: float, n_win: int) -> tuple[int, int]:
        """[i0, i1) sample slice for a window starting at ``t_start_s``.

        Returned without clipping; callers decide how to treat windows that
        fall (partially) outside ``[0, n)``.
        """
        i0 = int(np.ceil((t_start_s - self.t0_s) / self.dt_s - 1e-9))
        return i0, i0 + n_win


# ---------------------------------------------------------------------------
# ELM labels (filterscopes fs00–fs07 D-alpha)
# ---------------------------------------------------------------------------
def _load_dalpha_composite(
    h5: "h5py.File", channels: slice
) -> tuple[np.ndarray, np.ndarray] | None:
    """(times_s, composite) from filterscope ``channels``, or None if unusable.

    Composite = max over channels of per-channel median-subtracted signal.
    Non-finite samples (first/last sample of every channel in these files)
    are excluded from the max; channels that are entirely non-finite are
    dropped.
    """
    if not modality_present(h5, "filterscopes"):
        return None
    group = h5["filterscopes"]
    x = group["xdata"][:].astype(np.float64)
    scale = _unit_scale_to_seconds(x[-1] - x[0])
    x_s = x * scale
    y = group["ydata"][channels, :].astype(np.float64)
    if y.ndim == 1:
        y = y[None, :]
    finite = np.isfinite(y)
    usable = finite.any(axis=1)
    if not usable.any():
        warnings.warn("filterscopes: no usable (non-NaN) channel in slice")
        return None
    y, finite = y[usable], finite[usable]
    med = np.nanmedian(np.where(finite, y, np.nan), axis=1, keepdims=True)
    resid = np.where(finite, y - med, -np.inf)
    comp = resid.max(axis=0)
    comp[~np.isfinite(comp)] = 0.0  # samples where every channel was NaN
    return x_s, comp


def detect_elm_peaks(
    comp: np.ndarray,
    times_s: np.ndarray,
    k_mad: float = 6.0,
    baseline_window_s: float = 0.051,
    min_distance_s: float = 1e-3,
    stability_factor: float = 1.5,
    stability_min_ratio: float = 0.10,
    noise_extreme_mult: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Detect ELM peaks on a D-alpha composite trace.

    Returns ``(peak_indices, info)``; ``info`` carries the rolling baseline,
    MAD, effective height threshold and gate diagnostics for plotting/QA.
    """
    n = comp.size
    dt = float(times_s[-1] - times_s[0]) / max(n - 1, 1)
    win = max(3, int(round(baseline_window_s / dt)) | 1)  # odd
    baseline = median_filter(comp, size=win, mode="nearest")
    mad = float(np.median(np.abs(comp - np.median(comp))))
    distance = max(1, int(round(min_distance_s / dt)))
    info: dict = {
        "baseline": baseline,
        "mad": mad,
        "height": baseline + k_mad * mad,
        "gated": False,
        "stability_ratio": np.nan,
    }
    if mad <= 0 or not np.isfinite(mad):
        # Dead/constant channel bank: nothing to detect.
        return np.empty(0, dtype=np.int64), info

    peaks, _ = find_peaks(
        comp, prominence=k_mad * mad, height=info["height"], distance=distance
    )
    strict, _ = find_peaks(
        comp,
        prominence=stability_factor * k_mad * mad,
        height=baseline + stability_factor * k_mad * mad,
        distance=distance,
    )
    ratio = len(strict) / max(len(peaks), 1)
    info["stability_ratio"] = ratio
    if len(peaks) and ratio < stability_min_ratio:
        # Candidates evaporate under a modestly raised threshold => they are
        # the extreme order statistics of a continuous noise band, not an
        # ELM population.  Keep only excursions beyond the expected extreme
        # of a pure-noise record of this length.
        sigma = 1.4826 * mad
        extreme = noise_extreme_mult * sigma * np.sqrt(2.0 * np.log(n))
        peaks = peaks[(comp[peaks] - baseline[peaks]) > extreme]
        info["gated"] = True
        info["noise_extreme"] = extreme
    return peaks, info


def elm_labels(
    h5_path: "str | Path",
    window_starts_s: np.ndarray,
    window_s: float = 0.05,
    channels: slice = slice(0, 8),
    *,
    k_mad: float = 6.0,
    baseline_window_s: float = 0.051,
    min_distance_s: float = 1e-3,
    stability_factor: float = 1.5,
    stability_min_ratio: float = 0.10,
    noise_extreme_mult: float = 1.5,
) -> "dict[str, np.ndarray] | None":
    """Per-window ELM labels from the D-alpha filterscopes.

    Parameters
    ----------
    window_starts_s
        Window start times in seconds (the eval grid ``warmup_s + idx*stride``).
    window_s
        Window length in seconds (eval ``chunk_duration_s``).
    channels
        Filterscope channel slice; 0:8 are the fs00–fs07 D-alpha chords.

    Returns
    -------
    None if the ``filterscopes`` group is missing (length-1 stub) or has no
    usable channel; otherwise a dict with per-window arrays aligned to
    ``window_starts_s``:

    * ``elm_now``  (bool)  >=1 detected peak in ``[t, t+window_s)``
    * ``elm_next`` (bool)  >=1 peak in ``[t+window_s, t+2*window_s)`` —
      the prediction target
    * ``n_peaks_now`` (int16) peak count in the *now* window
    * ``dalpha_max_now`` (float32) composite max in the *now* window
      (NaN where the window has no samples)
    * ``in_range`` (bool) whole ``[t, t+2*window_s)`` span lies inside the
      filterscope time base — labels outside are False/0/NaN by construction

    plus shot-level meta: ``n_peaks_total`` (int), ``elm_rate_hz`` (float,
    peaks over the full digitised span), ``peak_times_s`` (float64 array).
    """
    window_starts_s = np.asarray(window_starts_s, dtype=np.float64)
    with h5py.File(str(h5_path), "r") as h5:
        loaded = _load_dalpha_composite(h5, channels)
    if loaded is None:
        return None
    times_s, comp = loaded

    peaks, _info = detect_elm_peaks(
        comp,
        times_s,
        k_mad=k_mad,
        baseline_window_s=baseline_window_s,
        min_distance_s=min_distance_s,
        stability_factor=stability_factor,
        stability_min_ratio=stability_min_ratio,
        noise_extreme_mult=noise_extreme_mult,
    )
    peak_times = times_s[peaks]

    n_w = window_starts_s.size
    t0, t1 = window_starts_s, window_starts_s + window_s
    t2 = window_starts_s + 2.0 * window_s
    # Peak counts via searchsorted on the (sorted) peak times.
    c0 = np.searchsorted(peak_times, t0, side="left")
    c1 = np.searchsorted(peak_times, t1, side="left")
    c2 = np.searchsorted(peak_times, t2, side="left")
    n_now = (c1 - c0).astype(np.int16)
    n_next = (c2 - c1).astype(np.int16)

    # Composite max per window (raw composite, not baseline-subtracted).
    i0 = np.searchsorted(times_s, t0, side="left")
    i1 = np.searchsorted(times_s, t1, side="left")
    dmax = np.full(n_w, np.nan, dtype=np.float32)
    for j in range(n_w):
        if i1[j] > i0[j]:
            dmax[j] = comp[i0[j] : i1[j]].max()

    span = float(times_s[-1] - times_s[0])
    return {
        "elm_now": n_now > 0,
        "elm_next": n_next > 0,
        "n_peaks_now": n_now,
        "dalpha_max_now": dmax,
        "in_range": (t0 >= times_s[0]) & (t2 <= times_s[-1]),
        "n_peaks_total": int(peaks.size),
        "elm_rate_hz": float(peaks.size / span) if span > 0 else 0.0,
        "peak_times_s": peak_times.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Band-power labels (mirnov / mhr — high-rate magnetics)
# ---------------------------------------------------------------------------
def _merge_slices(
    slices: Sequence[tuple[int, int]], max_len: int
) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent [i0, i1) reads into contiguous blocks."""
    order = sorted(s for s in slices)
    blocks: list[tuple[int, int]] = []
    for i0, i1 in order:
        if blocks and i0 <= blocks[-1][1] and (i1 - blocks[-1][0]) <= max_len:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], i1))
        else:
            blocks.append((i0, i1))
    return blocks


def _band_power_group(
    group: "h5py.Group",
    window_starts_s: np.ndarray,
    window_s: float,
    f_lo: float,
    f_hi: float,
    eps: float,
    taper: "str | None",
) -> tuple[np.ndarray, np.ndarray]:
    """(log10 band power, log10 broadband RMS) per window for one group.

    Never loads full ``ydata``: reads only ``ydata[:, i0:i1]`` per window,
    merging overlapping/adjacent windows into contiguous block reads.
    """
    n_w = window_starts_s.size
    bp = np.full(n_w, np.nan, dtype=np.float32)
    rms = np.full(n_w, np.nan, dtype=np.float32)

    tb = _TimeBase.from_group(group)
    n_win = int(round(window_s / tb.dt_s))
    if n_win < 8:
        warnings.warn(
            f"{group.name}: window of {window_s}s is only {n_win} samples"
        )
        return bp, rms

    freqs = np.fft.rfftfreq(n_win, d=tb.dt_s)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        warnings.warn(
            f"{group.name}: band [{f_lo}, {f_hi}] Hz empty at "
            f"fs={1.0 / tb.dt_s:.0f} Hz, n={n_win}"
        )
        return bp, rms
    if taper == "hann":
        w = np.hanning(n_win)
        u = float(np.mean(w**2))  # taper power compensation
    elif taper is None:
        w, u = None, 1.0
    else:
        raise ValueError(f"unknown taper {taper!r}")

    # Windows fully inside the time base; others stay NaN.
    win_slices: dict[int, tuple[int, int]] = {}
    for j, t in enumerate(window_starts_s):
        i0, i1 = tb.window_slice(float(t), n_win)
        if i0 >= 0 and i1 <= tb.n:
            win_slices[j] = (i0, i1)
    if not win_slices:
        return bp, rms

    ydata = group["ydata"]
    blocks = _merge_slices(list(win_slices.values()), max_len=max(4 * n_win, 1 << 20))
    for b0, b1 in blocks:
        block = ydata[:, b0:b1].astype(np.float64)  # (C, L) — bounded read
        for j, (i0, i1) in win_slices.items():
            if i0 < b0 or i1 > b1:
                continue
            seg = block[:, i0 - b0 : i1 - b0]
            seg = seg - seg.mean(axis=1, keepdims=True)
            # log10 broadband RMS: per-channel RMS, mean over channels.
            rms[j] = np.log10(
                max(float(np.sqrt(np.mean(seg**2, axis=1)).mean()), eps)
            )
            spec = np.fft.rfft(seg * w if w is not None else seg, axis=1)
            # One-sided band power (variance in band, Parseval-consistent):
            # sum_band 2|X_k|^2 / (n^2 * U).
            p = 2.0 * np.abs(spec[:, band]) ** 2 / (n_win**2 * u)
            bp[j] = np.log10(max(float(p.sum(axis=1).mean()), eps))
    return bp, rms


def band_power_labels(
    h5_path: "str | Path",
    window_starts_s: np.ndarray,
    window_s: float = 0.05,
    spec: Iterable[tuple[str, str, float, float]] = (
        ("mirnov", "mirnov_bp", 1e3, 1e4),
        ("mhr", "mhr_bp", 1e3, 1e4),
    ),
    *,
    eps: float = 1e-30,
    taper: "str | None" = "hann",
) -> "dict[str, np.ndarray]":
    """Per-window log10 band power of high-rate magnetics groups.

    For each ``(group, label, f_lo, f_hi)`` in ``spec`` emits:

    * ``<label>``      log10 band power in ``[f_lo, f_hi]`` Hz — one-sided
      rFFT power (Hann-tapered, power-compensated, per-window mean removed),
      summed over band bins, mean over channels, floored at ``eps``.
    * ``<label>_rms``  log10 broadband RMS (per-channel RMS of the
      mean-removed window, mean over channels), floored at ``eps``.

    Missing groups (length-1 stubs) and windows outside a group's time span
    yield NaN.  Full ``ydata`` is never loaded: only the per-window sample
    slices are read (merged into contiguous blocks where windows overlap).
    """
    window_starts_s = np.asarray(window_starts_s, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    with h5py.File(str(h5_path), "r") as h5:
        for group_name, label, f_lo, f_hi in spec:
            if not modality_present(h5, group_name):
                out[label] = np.full(window_starts_s.size, np.nan, np.float32)
                out[f"{label}_rms"] = np.full(
                    window_starts_s.size, np.nan, np.float32
                )
                continue
            bp, rms = _band_power_group(
                h5[group_name], window_starts_s, window_s, f_lo, f_hi, eps, taper
            )
            out[label] = bp
            out[f"{label}_rms"] = rms
    return out


# ---------------------------------------------------------------------------
# Self-test / validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DATA_DIR = Path("/lustre/orion/fus187/proj-shared/foundation_model")
    REPO_ROOT = Path(__file__).resolve().parents[3]
    OUT_DIR = REPO_ROOT / "data" / "outputs" / "eval_suite" / "label_check"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    SHOTS = [190000, 190014, 190030]
    WARMUP_S, STRIDE_S, WINDOW_S = 1.0, 0.25, 0.05

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    timing_reported = False

    for row, shot in enumerate(SHOTS):
        path = DATA_DIR / f"{shot}_processed.h5"
        with h5py.File(path, "r") as h5:
            present = {
                m: modality_present(h5, m)
                for m in ("filterscopes", "mirnov", "mhr", "co2")
            }
            fs_end = float(h5["filterscopes"]["xdata"][-1]) * (
                _unit_scale_to_seconds(
                    float(h5["filterscopes"]["xdata"][-1])
                    - float(h5["filterscopes"]["xdata"][0])
                )
            )
        print(f"=== shot {shot}  modality_present: {present}")

        # Eval window grid; extend slightly past the filterscope span so the
        # in_range machinery is exercised.
        n_idx = int(np.floor((fs_end + 0.4 - WARMUP_S) / STRIDE_S)) + 1
        grid = WARMUP_S + STRIDE_S * np.arange(n_idx)

        lab = elm_labels(path, grid, window_s=WINDOW_S)
        assert lab is not None, f"{shot}: filterscopes unexpectedly missing"
        inr = lab["in_range"]
        prev_next = float(lab["elm_next"][inr].mean()) if inr.any() else np.nan
        prev_now = float(lab["elm_now"][inr].mean()) if inr.any() else np.nan
        print(
            f"    n_peaks_total={lab['n_peaks_total']}  "
            f"elm_rate_hz={lab['elm_rate_hz']:.1f}  "
            f"elm_now prevalence={prev_now:.3f}  "
            f"elm_next prevalence={prev_next:.3f}  "
            f"(windows: {inr.sum()}/{grid.size} in range, stride {STRIDE_S}s)"
        )
        print(
            f"    dalpha_max_now range [{np.nanmin(lab['dalpha_max_now']):.2e}, "
            f"{np.nanmax(lab['dalpha_max_now']):.2e}]  "
            f"n_peaks_now max={lab['n_peaks_now'].max()}"
        )
        assert (~inr).sum() > 0, "grid should exercise out-of-range windows"
        assert not lab["elm_now"][~inr].any() and not lab["elm_next"][~inr].any()

        t_bp = time.perf_counter()
        bp = band_power_labels(path, grid, window_s=WINDOW_S)
        t_bp = time.perf_counter() - t_bp
        for key, arr in bp.items():
            ok = np.isfinite(arr)
            rng = (
                f"[{np.nanmin(arr):7.3f}, {np.nanmax(arr):7.3f}]"
                if ok.any()
                else "all-NaN"
            )
            print(f"    {key:14s} log10 range {rng}   ({ok.sum()}/{arr.size} finite)")
        if not timing_reported:
            print(
                f"    TIMING: band_power_labels({shot}, full {grid.size}-window "
                f"grid, mirnov+mhr) took {t_bp:.2f} s  (< 60 s required)"
            )
            timing_reported = True

        # ---- validation figure row -------------------------------------
        with h5py.File(path, "r") as h5:
            times_s, comp = _load_dalpha_composite(h5, slice(0, 8))
        peaks, info = detect_elm_peaks(comp, times_s)
        pk_t = times_s[peaks]
        if pk_t.size:
            dense = pk_t[
                np.argmax([(np.abs(pk_t - t) < 0.75).sum() for t in pk_t])
            ]
        else:
            dense = 0.5 * (times_s[0] + times_s[-1])
        lo, hi = dense - 0.75, dense + 0.75
        m = (times_s >= lo) & (times_s <= hi)

        ax = axes[row]
        for t, is_elm, ok in zip(grid, lab["elm_now"], inr):
            if not ok or t + WINDOW_S < lo or t > hi:
                continue
            ax.axvspan(
                t,
                t + WINDOW_S,
                color="#d62728" if is_elm else "#1f77b4",
                alpha=0.30 if is_elm else 0.10,
                lw=0,
            )
        ax.plot(times_s[m], comp[m], "k-", lw=0.6, label="D-alpha composite")
        pm = (pk_t >= lo) & (pk_t <= hi)
        ax.plot(pk_t[pm], comp[peaks][pm], "rv", ms=6, label="detected ELM peak")
        ax.plot(
            times_s[m],
            info["height"][m],
            color="#ff7f0e",
            ls="--",
            lw=0.9,
            label="height threshold (baseline + k*MAD)",
        )
        gated = " [stability-GATED: no ELM population]" if info["gated"] else ""
        ax.set_title(
            f"shot {shot}: {lab['n_peaks_total']} peaks, "
            f"{lab['elm_rate_hz']:.1f} Hz, elm_next prevalence "
            f"{prev_next:.2f} @ stride {STRIDE_S}s{gated}",
            fontsize=11,
        )
        ax.set_xlim(lo, hi)
        ax.set_ylabel("D-alpha (ph flux, a.u.)")
        if row == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        if row == len(SHOTS) - 1:
            ax.set_xlabel("time (s)")

    fig.suptitle(
        "ELM label validation — composite of filterscopes ch 0:8 (fs00-fs07)\n"
        "find_peaks(prominence >= 6*MAD, height >= rolling median + 6*MAD, "
        "distance >= 1 ms); 50 ms window shading: elm_now (red=True, blue=False)",
        fontsize=10,
    )
    fig_path = OUT_DIR / "elm_validation.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Validation figure: {fig_path}")
