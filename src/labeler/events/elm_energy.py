"""Provisional ELM losses from an EFIT-scaled diamagnetic-loop trace.

Remove a line fitted only to supplied quiet baseline windows, then calibrate
the loop's multiplicative scale against overlapping positive EFIT WMHD samples.
Find fast losses on the calibrated loop independently of D-alpha; retain a size
only for a one-to-one D-alpha match with complete, isolated pre/post coverage.
Noise estimation and loss measurements stay inside the calibration interval;
quiet off-plasma samples must not determine the plasma loss threshold.
Times are milliseconds. Input loop units and polarity are preserved in the fit.
This estimates the observed diagnostic drop, without deconvolving wall response.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks

from ..config import Paths
from ..features.store import read_feature
from . import equilibrium, spans, transients
from .raw import raw_signal
from .verify import NoDataError


@dataclass(frozen=True)
class DriftFit:
    slope_per_ms: float
    intercept: float
    baseline_windows_ms: tuple[tuple[float, float], ...]
    samples: int
    rms_residual: float


@dataclass(frozen=True)
class ScaleFit:
    gain_j_per_unit: float
    calibration_window_ms: tuple[float, float]
    matched_samples: int
    median_relative_error: float


@dataclass(frozen=True)
class Settings:
    pre_window_ms: tuple[float, float] = (-2.0, -0.25)
    post_window_ms: tuple[float, float] = (0.25, 2.0)
    match_tolerance_ms: float = 3.0
    min_separation_ms: float = 3.0
    noise_sigma: float = 5.0
    min_fraction: float = 0.002
    min_samples: int = 2

    def __post_init__(self):
        pre, post = _window(self.pre_window_ms), _window(self.post_window_ms)
        if pre[1] > 0 or post[0] < 0:
            raise ValueError("pre/post measurement windows must bracket the crash")
        scalars = (self.match_tolerance_ms, self.min_separation_ms,
                   self.noise_sigma, self.min_fraction)
        if not np.isfinite(scalars).all() or min(scalars) <= 0:
            raise ValueError("loss detection settings must be finite and positive")
        if type(self.min_samples) is not int or self.min_samples < 2:
            raise ValueError("loss windows need at least two samples")


@dataclass(frozen=True)
class Loss:
    loop_time_ms: float
    dalpha_time_ms: float
    energy_before_j: float
    energy_after_j: float
    loss_j: float
    fraction: float


@dataclass(frozen=True)
class Analysis:
    time_ms: np.ndarray
    corrected_loop: np.ndarray
    energy_j: np.ndarray
    drift: DriftFit
    calibration: ScaleFit
    settings: Settings
    losses: tuple[Loss, ...]
    unmatched_dalpha_ms: tuple[float, ...]
    provenance: dict | None = None

    def metadata(self):
        drift, calibration, settings = (
            asdict(self.drift), asdict(self.calibration), asdict(self.settings)
        )
        drift["baseline_windows_ms"] = [list(w) for w in self.drift.baseline_windows_ms]
        return {
            "method": "diamagnetic_loop_v2", "status": "provisional",
            "drift": drift, "calibration": calibration, "settings": settings,
            "measurement_window_ms": list(self.calibration.calibration_window_ms),
            "losses": [asdict(loss) for loss in self.losses],
            "unmatched_dalpha_ms": list(self.unmatched_dalpha_ms),
            "provenance": {} if self.provenance is None else self.provenance,
        }


def _trace(t_ms, values):
    t, y = np.asarray(t_ms, dtype=float), np.asarray(values, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or (
        np.diff(t) <= 0
    ).any():
        raise ValueError("trace times must be finite and strictly increasing")
    if y.ndim != 1 or y.shape != t.shape:
        raise ValueError("one trace and its matching time axis are required")
    return t, y


def _window(bounds):
    values = np.asarray(bounds, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or values[1] <= values[0]:
        raise ValueError("a window needs two finite increasing bounds")
    return tuple(float(v) for v in values)


def _runs(t, y):
    index = np.flatnonzero(np.isfinite(y))
    if not len(index):
        return []
    cuts = np.flatnonzero(
        (np.diff(index) > 1)
        | (np.diff(t[index]) > 1.5 * np.median(np.diff(t)))
    ) + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(index, cuts)]


def _sample(target, t, y):
    result = np.full(len(target), np.nan)
    for lo, hi in _runs(t, y):
        keep = (target >= t[lo]) & (target <= t[hi - 1])
        result[keep] = np.interp(target[keep], t[lo:hi], y[lo:hi])
    return result


def correct_drift(t_ms, loop, baseline_windows_ms):
    """Subtract a line fitted to at least two distinct, supplied quiet windows."""
    t, y = _trace(t_ms, loop)
    windows = tuple(sorted(_window(w) for w in baseline_windows_ms))
    if len(windows) < 2 or any(a[1] >= b[0] for a, b in pairwise(windows)):
        raise ValueError("baseline requires at least two non-overlapping quiet windows")
    keep = np.zeros(len(t), dtype=bool)
    for a, b in windows:
        inside = (t >= a) & (t <= b) & np.isfinite(y)
        if inside.sum() < 3:
            raise ValueError("baseline windows need at least three finite samples each")
        keep |= inside
    origin = float(t[keep].mean())
    design = np.column_stack([t[keep] - origin, np.ones(keep.sum())])
    slope, centred_intercept = np.linalg.lstsq(design, y[keep], rcond=None)[0]
    intercept = float(centred_intercept - slope * origin)
    corrected = y - (slope * t + intercept)
    fit = DriftFit(float(slope), intercept, windows, int(keep.sum()),
                   float(np.sqrt(np.mean(corrected[keep] ** 2))))
    return corrected, fit


def scale_to_efit(t_ms, corrected_loop, efit_t_ms, wmhd_j, *, calibration_window_ms):
    """Median signed EFIT/loop gain at native EFIT samples; no gap extrapolation."""
    t, loop = _trace(t_ms, corrected_loop)
    ef_t, reference = _trace(efit_t_ms, wmhd_j)
    window = _window(calibration_window_ms)
    sampled = _sample(ef_t, t, loop)
    finite = np.abs(sampled[np.isfinite(sampled)])
    floor = 64 * np.finfo(float).eps * max(1.0, finite.max(initial=0))
    keep = (
        (ef_t >= window[0]) & (ef_t <= window[1])
        & np.isfinite(reference) & (reference > 0)
        & np.isfinite(sampled) & (np.abs(sampled) > floor)
    )
    if keep.sum() < 3:
        raise ValueError("calibration needs at least three overlapping EFIT/loop samples")
    ratios = reference[keep] / sampled[keep]
    gain = float(np.median(ratios))
    same_sign = np.sign(ratios) == np.sign(gain)
    if not np.isfinite(gain) or gain == 0 or same_sign.mean() < 0.9:
        raise ValueError("loop polarity is inconsistent across the calibration window")
    matched = np.flatnonzero(keep)[same_sign]
    gain = float(np.median(reference[matched] / sampled[matched]))
    error = float(np.median(
        np.abs(gain * sampled[matched] - reference[matched]) / reference[matched]
    ))
    if error > 0.2:
        raise ValueError("calibration median relative error exceeds 20 percent")
    return loop * gain, ScaleFit(gain, window, len(matched), error)


def _window_mean(t, y, bounds, min_samples):
    a, b = bounds
    left = np.searchsorted(t, t + a, side="left")
    right = np.searchsorted(t, t + b, side="right")
    count = right - left
    sums = np.r_[0.0, np.cumsum(y)]
    covered = (t + a >= t[0]) & (t + b <= t[-1]) & (count >= min_samples)
    return np.divide(
        sums[right] - sums[left], count,
        out=np.full(len(t), np.nan), where=covered,
    )


def detect_losses(t_ms, energy_j, dalpha_peaks_ms, *, settings=None):
    """Size only independent loop drops with an unambiguous isolated match."""
    settings = Settings() if settings is None else settings
    t, energy = _trace(t_ms, energy_j)
    peaks = np.asarray(dalpha_peaks_ms, dtype=float)
    if peaks.ndim != 1 or not np.isfinite(peaks).all() or (np.diff(peaks) <= 0).any():
        raise ValueError("D-alpha peaks must be finite and strictly increasing")
    before, after = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for lo, hi in _runs(t, energy):
        before[lo:hi] = _window_mean(
            t[lo:hi], energy[lo:hi], settings.pre_window_ms, settings.min_samples,
        )
        after[lo:hi] = _window_mean(
            t[lo:hi], energy[lo:hi], settings.post_window_ms, settings.min_samples,
        )
    drop = before - after
    finite = drop[np.isfinite(drop)]
    if not len(finite):
        return (), tuple(float(p) for p in peaks)
    noise = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    positive = before[np.isfinite(before) & (before > 0)]
    threshold = max(settings.noise_sigma * noise,
                    settings.min_fraction * np.median(positive) if len(positive) else 0)
    if threshold <= 0:
        return (), tuple(float(p) for p in peaks)
    candidates = []
    for lo, hi in _runs(t, drop):
        distance = max(1, int(np.ceil(settings.min_separation_ms / np.median(
            np.diff(t[lo:hi])
        )))) if hi - lo > 1 else 1
        indices, _ = find_peaks(drop[lo:hi], height=threshold,
                                prominence=threshold, distance=distance)
        candidates.extend(int(lo + i) for i in indices)
    links = {}
    contenders = np.zeros(len(peaks), dtype=int)
    for i in candidates:
        # Another D-alpha peak in the measurement support makes a single-event
        # attribution unresolved, even when the loop trace merges both drops.
        neighbors = np.flatnonzero(
            (peaks >= t[i] + settings.pre_window_ms[0])
            & (peaks <= t[i] + settings.post_window_ms[1])
        )
        if len(neighbors) > 1 or before[i] <= 0 or after[i] < 0:
            continue
        possible = np.flatnonzero(np.abs(t[i] - peaks) <= settings.match_tolerance_ms)
        if len(neighbors) == 1:
            # A farther match must not attribute the observed loss to a
            # different D-alpha event than the one inside its support.
            possible = possible[possible == neighbors[0]]
        links[i] = possible
        contenders[possible] += 1
    used_dalpha, losses = set(), []
    for i, possible in links.items():
        # Crowded timing stays unresolved; nearest-first matching can assign
        # one arbitrary size when several loops share a D-alpha candidate.
        if len(possible) != 1 or contenders[possible[0]] != 1:
            continue
        j = int(possible[0])
        losses.append(Loss(float(t[i]), float(peaks[j]), float(before[i]),
                           float(after[i]), float(drop[i]), float(drop[i] / before[i])))
        used_dalpha.add(j)
    losses.sort(key=lambda loss: loss.dalpha_time_ms)
    unmatched = tuple(float(p) for j, p in enumerate(peaks) if j not in used_dalpha)
    return tuple(losses), unmatched


def analyze(
    t_ms, loop, efit_t_ms, wmhd_j, dalpha_peaks_ms, *, baseline_windows_ms,
    calibration_window_ms, settings=None,
):
    """Calibrate the full loop; size losses only inside the calibrated regime."""
    corrected, drift = correct_drift(t_ms, loop, baseline_windows_ms)
    window = _window(calibration_window_ms)
    if any(max(a, window[0]) < min(b, window[1]) for a, b in drift.baseline_windows_ms):
        raise ValueError("baseline and calibration windows must not overlap")
    energy, calibration = scale_to_efit(
        t_ms, corrected, efit_t_ms, wmhd_j, calibration_window_ms=window,
    )
    settings = Settings() if settings is None else settings
    t = np.asarray(t_ms, dtype=float)
    inside = (t >= window[0]) & (t <= window[1])
    # Keep all D-alpha peaks in the matcher: unsupported times remain unsized,
    # and neighboring peaks still prevent ambiguous individual attribution.
    losses, unmatched = detect_losses(
        t[inside], energy[inside], dalpha_peaks_ms, settings=settings,
    )
    return Analysis(t, corrected, energy, drift,
                    calibration, settings, losses, unmatched)


def load(shot, paths=None, *, loop_name="diamagnetic_loop"):
    """Analyze a local loop with explicit per-shot baseline/calibration metadata.

    The loop group uses seconds on xdata, a single native integrated-loop trace
    on ydata, and JSON attrs baseline_windows_ms and calibration_window_ms.
    Optional elm_energy_settings overrides the documented provisional defaults.
    There is no guessed live loop node or inferred drift interval.
    """
    paths = Paths.from_env() if paths is None else paths
    for path in (paths.features_file(shot), paths.corpus_file(shot),
                 paths.raw_cache / f"{int(shot)}_processed.h5"):
        try:
            loop = read_feature(path, loop_name)
        except (OSError, KeyError):
            continue
        except (ValueError, TypeError) as error:
            raise NoDataError(
                f"shot {shot}: malformed {loop_name}: {error}"
            ) from error
        break
    else:
        raise NoDataError(f"shot {shot}: no local {loop_name} signal")
    try:
        if loop.y.shape[0] != 1:
            raise ValueError("diamagnetic loop must contain one scalar trace")
        for key in ("baseline_windows_ms", "calibration_window_ms"):
            if key not in loop.attrs:
                raise ValueError(f"diamagnetic loop needs explicit {key} metadata")
        baseline = json.loads(loop.attrs["baseline_windows_ms"])
        calibration = json.loads(loop.attrs["calibration_window_ms"])
        settings = Settings(**json.loads(loop.attrs.get("elm_energy_settings", "{}")))
        reference = equilibrium.signal(int(shot), "wmhd", paths, fetch=True)
        if reference.y.shape[0] != 1:
            raise ValueError("WMHD calibration must contain one scalar trace")
        dalpha = raw_signal(int(shot), "filterscopes", channels=list(range(8)),
                            paths=paths)
        channel = spans.dalpha_channel(dalpha.y, int(shot))
        events = transients.elm_clock_events(
            dalpha.y[channel], dalpha.x / 1000, shot=int(shot), channel=channel,
        )
        peaks = [e.t0_s * 1000 for e in events if e.phenomenon == "elm"]
        analysis = analyze(
            loop.x * 1000, loop.y[0], reference.x * 1000, reference.y[0], peaks,
            baseline_windows_ms=baseline, calibration_window_ms=calibration,
            settings=settings,
        )
        return replace(analysis, provenance={
            "loop_feature": loop_name, "loop_store": str(path),
            "loop_source": loop.attrs.get("source", loop.attrs.get("locator", "")),
            "loop_units": loop.attrs.get("units", "unspecified"),
            "efit_store": reference.attrs.get("store", ""),
            "efit_reference": "EFIT01 WMHD (J)",
            "dalpha_channel": f"FS{channel + 1:02d}",
            "response_corrected": False,
        })
    except (ValueError, TypeError) as error:
        raise NoDataError(f"shot {shot}: unusable diamagnetic energy: {error}") from error
