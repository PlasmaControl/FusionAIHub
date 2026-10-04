"""Gude-inspired multichannel edge detection with explicit geometry abstentions.

The profile test is a DIII-D adaptation, not a byte-for-byte implementation of
Gude's two-sided SXR test. Without calibrated radius, channel order establishes
an inversion boundary but does not establish its physical radius or q=1 proximity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from ..events.panels.ece_geometry import align_q
from ..events.schema import Event


@dataclass(frozen=True)
class Rule:
    sigma_ms: float = 0.25
    frame_ms: float = 10.3
    posr_threshold: float = 6.0
    coincidence_ms: float = 0.5
    minimum_channels: int = 2
    significance: float = 0.02
    maximum_net: float = 0.9
    minimum_block: int = 2
    pulse_reach: int = 6
    minimum_period_ms: float = 20.0
    maximum_period_ms: float = 250.0
    period_ratio: float = 2.5
    minimum_train: int = 3
    qmin_margin: float = 0.4
    qmin_absence: float = 1.5
    qmin_sustain_ms: float = 50.0
    relaxation_minimum_period_ms: float = 10.0
    relaxation_minimum_edges: int = 6
    periodicity_null_replicates: int = 199
    periodicity_null_alpha: float = 0.05
    radius_tolerance: float = 0.15
    central_relative_drop: float = 0.05
    te_floor_kev: float = 0.5
    inversion_spread_channels: float = 2.0
    minimum_ip_ma: float = 0.3
    neutron_relative_drop: float = 0.02
    neutron_noise_k: float = 3.0
    minimum_nbi_power_w: float = 1e5
    maximum_channel_to_core: float = 1.5
    mirnov_burst_z: float = 6.0
    dalpha_burst_z: float = 6.0
    # ECE-side validity (second-harmonic density cutoff and mapping failures).
    ece_validity_smooth_ms: float = 5.0
    ece_validity_sustain_ms: float = 20.0
    ece_step_ratio: float = 2.0
    ece_step_max_rho: float = 0.7
    ece_axis_to_max: float = 0.6
    ece_axis_max_distance_m: float = 0.1
    # Tested absence: no periodic POSR edge on any valid channel inside this
    # nominal geometric rho, and no isolated one within this context.
    quiet_core_max_rho: float = 0.5
    isolated_edge_context_ms: float = 5.15


DEFAULT_RULE = Rule(
    **json.loads(Path(__file__).with_name("freeze.json").read_text())["rule"]
)


@dataclass
class Detection:
    crashes: list[Event]
    intervals: list[Event]
    rejected: dict[str, int]
    candidates: int
    observable: np.ndarray
    uncertain_intervals: list[Event]
    absent_mask: np.ndarray
    absence_diagnostics: dict
    # Sustained high EFIT01 q_min without an ECE quiet-core test. This is a
    # prior, never a tested negative: it is exported as uncertain.
    q_prior_mask: np.ndarray | None = None


def _runs(mask):
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(ends[::2], ends[1::2], strict=True))


def _smooth_finite(values, width):
    """Moving mean over finite samples only; NaN where under half is finite."""
    finite = np.isfinite(values)
    total = uniform_filter1d(
        np.where(finite, values, 0.0).astype(np.float32), width, axis=1, mode="nearest"
    )
    count = uniform_filter1d(finite.astype(np.float32), width, axis=1, mode="nearest")
    return np.divide(
        total, count, out=np.full(values.shape, np.nan, np.float32), where=count > 0.5
    )


def ece_validity(values, t, rule=DEFAULT_RULE, *, radius=None):
    """Mark time whose ECE profile cannot be a Te profile (cutoff, bad mapping).

    Two sustained tests, both on a short moving mean of the valid channels:
    a step between adjacent channels inside nominal rho < ``ece_step_max_rho``
    larger than ``ece_step_ratio``, or the channel nearest the EFIT axis
    reading below ``ece_axis_to_max`` of the profile maximum. A condition must
    hold for at least ``ece_validity_sustain_ms`` to remove time. Edge channels
    are excluded from the step test because a pedestal can legitimately exceed
    a factor of two per channel. Without nominal geometry nothing is tested.
    """
    values = np.asarray(values, dtype=np.float32)
    flag = np.zeros(len(t), dtype=bool)
    info = {
        "status": "geometry_unavailable_not_tested",
        "step_ratio": rule.ece_step_ratio,
        "step_max_rho": rule.ece_step_max_rho,
        "axis_to_max": rule.ece_axis_to_max,
        "sustain_ms": rule.ece_validity_sustain_ms,
        "smooth_ms": rule.ece_validity_smooth_ms,
        "step_samples": 0,
        "axis_samples": 0,
        "removed_samples": 0,
    }
    if radius is None or radius.nominal_rho is None:
        return flag, info
    dt = float(np.median(np.diff(t)))
    width = max(1, round(rule.ece_validity_smooth_ms / 1000 / dt)) | 1
    stop = min(40, len(values))
    smooth = _smooth_finite(values[:stop], width)
    valid = np.isfinite(smooth) & (smooth > 0)
    rho = np.asarray(radius.nominal_rho)[:stop]
    pair = valid[:-1] & valid[1:]
    pair &= (rho[:-1] < rule.ece_step_max_rho) & (rho[1:] < rule.ece_step_max_rho)
    low = np.minimum(smooth[:-1], smooth[1:])
    high = np.maximum(smooth[:-1], smooth[1:])
    step = (pair & (high > rule.ece_step_ratio * low)).any(axis=0)
    peak = np.max(np.where(valid, smooth, -np.inf), axis=0)
    distance = np.abs(np.asarray(radius.R_m)[:stop] - radius.axis_R_m[None])
    distance = np.where(valid & np.isfinite(distance), distance, np.inf)
    nearest = np.argmin(distance, axis=0)
    columns = np.arange(len(t))
    axis_te = smooth[nearest, columns]
    evaluable = (
        np.isfinite(peak)
        & (peak > 0)
        & (distance[nearest, columns] <= rule.ece_axis_max_distance_m)
    )
    axis = evaluable & (axis_te < rule.ece_axis_to_max * peak)
    sustain = round(rule.ece_validity_sustain_ms / 1000 / dt)
    for lo, hi in _runs(step | axis):
        if hi - lo >= sustain:
            flag[lo:hi] = True
    info.update(
        status="tested_nominal_geometry",
        step_samples=int(step.sum()),
        axis_samples=int(axis.sum()),
        removed_samples=int(flag.sum()),
        axis_max_distance_m=rule.ece_axis_max_distance_m,
    )
    return flag, info


def posr(frame, peak, remove):
    """Gude's POSR: remove the kernel-length largest absolute filtered values."""
    frame = np.asarray(frame, dtype=float)
    if len(frame) <= remove + 4 or not np.isfinite(frame).all():
        return 0.0
    order = np.argsort(np.abs(frame))
    noise = frame[order[: len(frame) - remove]]
    scale = noise.std()
    return float(abs(peak - noise.mean()) / scale) if scale > 1e-12 else 0.0


def noise_calibration(rule=DEFAULT_RULE, *, fs=10000, frames=5000, seed=20261003):
    """Simulated Gaussian-noise frame maxima; calibration does not use shots."""
    rng = np.random.default_rng(seed)
    sigma = rule.sigma_ms * fs / 1000
    width = round(rule.frame_ms * fs / 1000) | 1
    remove = int(np.ceil(7 * sigma))
    pad = int(np.ceil(3.5 * sigma))
    weights = []
    for _ in range(frames):
        x = rng.normal(size=width + 2 * pad)
        c = gaussian_filter1d(x, sigma, order=1, truncate=3.5)[pad:-pad]
        weights.append(posr(c, c[np.argmax(np.abs(c))], remove))
    return {
        "seed": seed,
        "frames": frames,
        "fs_hz": fs,
        "rule": asdict(rule),
        "posr_quantiles": dict(
            zip(("95", "99", "99.85"), np.quantile(weights, [0.95, 0.99, 0.9985]))
        ),
        "false_alarm_fraction": float(
            np.mean(np.array(weights) >= rule.posr_threshold)
        ),
        "note": "Per noise frame, before channel/profile/period tests; not a shot FPR.",
    }


def inversion_profile(step, level, rule=DEFAULT_RULE, *, core_level=None):
    """Contiguous core loss bordered by an outer gain, with Gude's A_norm/A_net.

    The hottest part of an uncalibrated array is only a temperature proxy for the
    core. Store the channel boundary separately from a calibrated physical radius.
    """
    step, level = np.asarray(step), np.asarray(level)
    valid = np.isfinite(step) & np.isfinite(level) & (level > 0)
    valid[40:] = False
    if core_level is not None and np.isfinite(core_level) and core_level > 0:
        valid &= level <= rule.maximum_channel_to_core * core_level
    if valid.sum() < 4:
        return "coverage", {}
    profile = np.where(valid, step, 0.0)
    total = np.abs(profile).sum()
    norm = total / np.where(valid, np.abs(level), 0).sum()
    net = abs(profile.sum()) / max(total, 1e-12)
    attrs = {
        "a_norm": float(norm),
        "a_net": float(net),
        "masked_channels": np.flatnonzero(~valid).tolist(),
        "maximum_channel_to_core": rule.maximum_channel_to_core,
        "core_reference_kev": float(core_level) if core_level is not None else None,
    }
    if norm < rule.significance:
        return "significance", attrs
    if net >= rule.maximum_net:
        return "redistribution", attrs
    # Ignore sign flicker smaller than 0.5% of the local temperature.
    drops = valid & (profile < -0.005 * level)
    rises = valid & (profile > 0.005 * level)
    blocks = [(a, b) for a, b in _runs(drops) if b - a >= rule.minimum_block]
    if not blocks:
        return "drop_block", attrs
    a, b = max(blocks, key=lambda ab: -profile[ab[0] : ab[1]].sum())
    gain_blocks = [(c, d) for c, d in _runs(rises) if d - c >= rule.minimum_block]
    nearby = [
        (c, d)
        for c, d in gain_blocks
        if 0 <= c - b <= rule.pulse_reach or 0 <= a - d <= rule.pulse_reach
    ]
    if not nearby:
        return "rise_block", attrs
    # The nearest contiguous gain borders this loss. A stronger remote block
    # may be harmonic overlap or a different event and cannot move the boundary.
    c, d = min(
        nearby,
        key=lambda cd: (
            cd[0] - b if cd[0] >= b else a - cd[1],
            -profile[cd[0] : cd[1]].sum(),
        ),
    )
    boundary = (b + c - 1) / 2 if c >= b else (a + d - 1) / 2
    hot = int(np.argmax(np.where(valid, level, -np.inf)))
    core_moves = a - 2 <= hot < b + 2 or level[a:b].mean() >= 0.7 * level[hot]
    attrs.update(
        inversion_channel=float(boundary),
        drop_start=int(a),
        drop_stop=int(b),
        rise_start=int(c),
        rise_stop=int(d),
        core_moves=bool(core_moves),
        inversion_rho=None,
        q1_rho=None,
        inversion_R_m=None,
        q1_R_m=None,
        geometry_status="channel_order_only",
    )
    return "accept", attrs


def trains(times_s, rule=DEFAULT_RULE):
    """Maximal quasi-periodic groups of at least three crashes; no gap bridging."""
    times = np.asarray(times_s, dtype=float)
    if len(times) < rule.minimum_train:
        return []
    groups, start, previous = [], 0, None
    for i, gap in enumerate(np.diff(times) * 1000):
        acceptable = rule.minimum_period_ms <= gap <= rule.maximum_period_ms
        if previous is not None:
            acceptable &= max(gap, previous) <= rule.period_ratio * min(gap, previous)
        if not acceptable:
            if i + 1 - start >= rule.minimum_train:
                groups.append((start, i + 1))
            start, previous = i + 1, None
        else:
            previous = gap
    if len(times) - start >= rule.minimum_train:
        groups.append((start, len(times)))
    return groups


def periodicity_null(edge_times, rule=DEFAULT_RULE):
    """Conditional shuffled-time null, preserving count, span and peak holdoff.

    Interior event times are shuffled uniformly between fixed first/last edges.
    A 5.15 ms refractory offset matches find_peaks, so its imposed minimum
    spacing cannot itself establish physical periodicity. The statistic is
    gap coefficient of variation; smaller values indicate greater regularity.
    This is a phase-level diagnostic, not an independent label validation.
    """
    times = np.asarray(edge_times, dtype=float)
    gaps = np.diff(times)
    if len(times) < rule.relaxation_minimum_edges or (gaps <= 0).any():
        return {"p_value": 1.0, "gap_cv": None, "replicates": 0}
    cv = float(gaps.std() / gaps.mean())
    holdoff = min(rule.frame_ms / 2000, float(gaps.min()))
    free_span = max(0.0, times[-1] - times[0] - len(gaps) * holdoff)
    rng = np.random.default_rng(20261003)
    interior = np.sort(
        rng.uniform(0, free_span, (rule.periodicity_null_replicates, len(times) - 2)),
        axis=1,
    )
    shuffled = (
        np.diff(
            np.column_stack(
                (np.zeros(len(interior)), interior, np.full(len(interior), free_span))
            ),
            axis=1,
        )
        + holdoff
    )
    null_cv = shuffled.std(axis=1) / shuffled.mean(axis=1)
    return {
        "p_value": float((1 + np.sum(null_cv <= cv)) / (len(null_cv) + 1)),
        "gap_cv": cv,
        "replicates": len(null_cv),
        "holdoff_ms": holdoff * 1000,
        "method": "fixed_count_span_refractory_uniform_time_shuffle",
    }


def _relaxation_groups(edges, rule):
    """Physical-gap grouping, shared exactly by data and shuffled replicates."""
    groups, start, previous = [], 0, None
    for index, gap in enumerate(np.diff(edges)):
        physical = gap * 1000 >= rule.relaxation_minimum_period_ms
        agrees = physical and (
            previous is None
            or (max(gap, previous) <= rule.period_ratio * min(gap, previous))
        )
        if not agrees:
            if index + 1 - start >= rule.relaxation_minimum_edges:
                groups.append((start, index + 1))
            start = index + 1 if not physical else index
        previous = gap if physical else None
    if len(edges) - start >= rule.relaxation_minimum_edges:
        groups.append((start, len(edges)))
    return groups


def _selected_phase_null(edges, groups, rule):
    """Minimum CV over all selected groups: selection and multiplicity included."""
    gaps = np.diff(edges)
    holdoff = min(rule.frame_ms / 2000, float(gaps.min()))
    free_span = max(0.0, edges[-1] - edges[0] - len(gaps) * holdoff)
    rng = np.random.default_rng(20261003)
    interior = np.sort(
        rng.uniform(0, free_span, (rule.periodicity_null_replicates, len(edges) - 2)),
        axis=1,
    )
    null_times = np.column_stack(
        (np.zeros(len(interior)), interior, np.full(len(interior), free_span))
    )
    null_times += np.arange(len(edges))[None] * holdoff
    null_min_cv = np.full(len(interior), np.inf)
    for index, shuffled in enumerate(null_times):
        selected = _relaxation_groups(shuffled, rule)
        for a, b in selected:
            periods = np.diff(shuffled[a:b])
            null_min_cv[index] = min(null_min_cv[index], periods.std() / periods.mean())
    results = []
    for a, b in groups:
        periods = np.diff(edges[a:b])
        cv = float(periods.std() / periods.mean())
        results.append(
            {
                "p_value": float((1 + np.sum(null_min_cv <= cv)) / (len(interior) + 1)),
                "gap_cv": cv,
                "replicates": len(interior),
                "method": (
                    "fixed_count_span_refractory_shuffle_same_grouping_minimum_CV"
                ),
            }
        )
    return results


def core_relaxation_phases(edge_times, observable_spans, rule=DEFAULT_RULE):
    """Protect only POSR-qualified, physically spaced, non-noise-like phases.

    Callers supply POSR-qualified negative edges. At least six edges and a
    shuffled-time p<=0.05 are needed to expand first-to-last support. Isolated
    qualified edges still have their individual uncertainty context. Phases
    never cross an unobserved interval.
    """
    times = np.unique(np.asarray(edge_times, dtype=float))
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("core edges must be finite times in seconds")
    horizon = 1.5 * rule.maximum_period_ms / 1000
    phases = []
    for lo, hi in observable_spans:
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("observable spans must have finite increasing bounds")
        left, right = np.searchsorted(times, [lo, hi])
        edges = times[left:right]
        if len(edges) < rule.relaxation_minimum_edges:
            continue
        groups = _relaxation_groups(edges, rule)
        if not groups:
            continue
        nulls = _selected_phase_null(edges, groups, rule)
        for (a, b), null in zip(groups, nulls, strict=True):
            periods = np.diff(edges[a:b]) * 1000
            if null["p_value"] > rule.periodicity_null_alpha:
                continue
            phases.append(
                {
                    "start_s": float(max(lo, edges[a] - horizon)),
                    "end_s": float(min(hi, edges[b - 1] + horizon)),
                    "first_edge_s": float(edges[a]),
                    "last_edge_s": float(edges[b - 1]),
                    "edges": int(b - a),
                    "period_ms": float(np.median(periods)),
                    "minimum_gap_ms": float(periods.min()),
                    "maximum_gap_ms": float(periods.max()),
                    "null_p_value": null["p_value"],
                    "gap_cv": null["gap_cv"],
                    "null_method": null["method"],
                    "null_replicates": null["replicates"],
                }
            )
    return phases


def _window(trace, crash, lower, upper):
    """Finite native auxiliary samples in a time window, without interpolation."""
    tx, values = (np.asarray(x) for x in trace)
    a, b = np.searchsorted(tx, [crash + lower, crash + upper])
    segment = values[a:b]
    return segment[np.isfinite(segment)]


def _neutron_evidence(trace, nbi, crash, rule):
    """Positive evidence only, with native-window noise and known NBI support.

    Quadrature MAD scatter is conservative: native samples may be correlated,
    so their count does not artificially shrink the reported noise threshold.
    """
    attrs = {
        "neutron_relative_drop": None,
        "neutron_drop_corroboration": None,
        "neutron_window_noise": None,
        "neutron_noise_k": rule.neutron_noise_k,
        "neutron_noise_method": "quadrature_native_window_MAD",
        "neutron_drop_absolute": None,
        "neutron_evidence_status": "unavailable",
        "nbi_power_w": None,
        "nbi_on": None,
    }
    if nbi is not None:
        power = align_q([crash], nbi[0], np.atleast_2d(nbi[1]))[0, 0]
        if np.isfinite(power):
            attrs["nbi_power_w"] = float(power)
            attrs["nbi_on"] = bool(power >= rule.minimum_nbi_power_w)
    if trace is None:
        return attrs
    before = _window(trace, crash, -0.0015, -0.0003)
    after = _window(trace, crash, 0.0003, 0.0015)
    attrs["neutron_window_samples"] = [len(before), len(after)]
    if min(len(before), len(after)) < 4:
        attrs["neutron_evidence_status"] = "insufficient_window_samples"
        return attrs
    pre, post = np.median(before), np.median(after)
    scales = [
        1.4826 * np.median(abs(before - pre)),
        1.4826 * np.median(abs(after - post)),
    ]
    noise = float(np.hypot(*scales))
    attrs["neutron_window_noise"] = noise
    attrs["neutron_drop_absolute"] = float(pre - post)
    if pre <= 0 or pre <= rule.neutron_noise_k * max(scales):
        attrs["neutron_evidence_status"] = "noise_dominated"
        return attrs
    attrs["neutron_relative_drop"] = float((pre - post) / pre)
    if attrs["nbi_on"] is not True:
        attrs["neutron_evidence_status"] = (
            "nbi_unknown" if attrs["nbi_on"] is None else "nbi_off"
        )
        return attrs
    threshold = max(rule.neutron_relative_drop * pre, rule.neutron_noise_k * noise)
    attrs["neutron_drop_threshold_absolute"] = float(threshold)
    if pre - post >= threshold:
        attrs["neutron_drop_corroboration"] = True
        attrs["neutron_evidence_status"] = "positive"
    else:
        attrs["neutron_evidence_status"] = "below_noise_threshold"
    return attrs


def _core_reference(values, membership):
    """Hottest coherent three-channel core median, robust to isolated spikes.

    A broad proxy can include cooler flanks. Its full median would incorrectly
    screen a steep physical central peak, so compare with the hottest local
    neighborhood having at least two measured core channels.
    """
    valid = membership & np.isfinite(values) & (values > 0)
    reference = np.full(values.shape[1], np.nan)
    for lo in range(len(values) - 2):
        supported = valid[lo : lo + 3]
        missing = supported.sum(axis=0) < 2
        if missing.all():
            continue
        selected = np.where(supported, values[lo : lo + 3], np.nan)
        selected[0, missing] = 0
        median = np.nanmedian(selected, axis=0)
        median[missing] = np.nan
        reference = np.fmax(reference, median)
    return reference


def _channel_edge_times(edge, observable, t, rule, sigma):
    """POSR-qualified negative edges of one channel's fractional edge trace."""
    dt = float(np.median(np.diff(t)))
    width = round(rule.frame_ms / 1000 / dt) | 1
    half, remove = width // 2, int(np.ceil(7 * sigma))
    times = []
    peaks, _ = find_peaks(-edge, distance=max(1, half))
    for k in peaks:
        if k < half or k + half >= len(t) or not observable[k]:
            continue
        relative = -edge[k] * np.sqrt(2 * np.pi) * sigma
        if not np.isfinite(relative) or relative < rule.significance:
            continue
        if posr(edge[k - half : k + half + 1], edge[k], remove) >= rule.posr_threshold:
            times.append(float(t[k]))
    return times


def _absence_evidence(t, observable, edges, support, candidates, rule, sigma):
    """Test quiet core support independently of profile/train acceptance.

    ``edges`` holds each supported channel's fractional edge trace (channels,
    time) and ``support`` its validity. Tested absence is a negative result of
    the frozen edge test on every supported channel: no POSR-qualified periodic
    train (per channel, 20-250 ms, no bridging of unobserved time), no isolated
    qualified edge within ``isolated_edge_context_ms``, no profile-passing
    candidate or slow relaxation phase nearby, a full +/-1.5 maximum-period
    observable context, and at least ``minimum_channels`` channels whose local
    noise could resolve an edge at the profile significance threshold.
    Quiet time is never inferred from EFIT q. This remains a research policy.
    """
    dt = float(np.median(np.diff(t)))
    width = round(rule.frame_ms / 1000 / dt) | 1
    block = max(width, round(rule.maximum_period_ms / 1000 / dt))
    scale = np.sqrt(2 * np.pi) * sigma
    runs = _runs(observable)
    channel_times, periodic_by_channel = [], []
    for row in range(len(edges)):
        edge = np.where(support[row], edges[row], 0.0)
        times = np.asarray(_channel_edge_times(edge, observable, t, rule, sigma))
        channel_times.append(times)
        periodic = []
        # A periodicity claim must not cross an unobserved diagnostic gap.
        for lo, hi in runs:
            left = np.searchsorted(times, t[lo])
            right = np.searchsorted(times, t[hi - 1], side="right")
            for a, b in trains(times[left:right], rule):
                periodic.extend(times[left + a : left + b])
        periodic_by_channel.append(np.asarray(periodic))
    all_times = np.sort(np.concatenate(channel_times)) if channel_times else np.empty(0)
    # Merge simultaneous edges of different channels into one reported event.
    merged = []
    for time in all_times:
        if not merged or time - merged[-1] > 2 * dt:
            merged.append(float(time))
    ambiguous_core_times = merged
    periodic_times = (
        np.unique(np.concatenate(periodic_by_channel))
        if periodic_by_channel
        else np.empty(0)
    )
    periodic = list(map(float, periodic_times))
    horizon = 1.5 * rule.maximum_period_ms / 1000
    radius = int(np.ceil(horizon / dt))
    complete = (
        maximum_filter1d(
            (~observable).astype(np.uint8), 2 * radius + 1, mode="constant", cval=1
        )
        == 0
    )
    # A failed edge test is informative only if its noise floor could resolve
    # an edge at the profile significance threshold. Estimate each channel's
    # scatter in maximum-period blocks and require enough resolving channels.
    resolved_channels = np.zeros(len(t), dtype=int)
    noise_rows = []
    for lo in range(0, len(t), block):
        hi = min(len(t), lo + block)
        noises = []
        for row in range(len(edges)):
            supported = observable[lo:hi] & support[row, lo:hi]
            if supported.sum() < width:
                continue
            relative_edge = edges[row, lo:hi][supported] * scale
            noises.append(
                float(1.4826 * np.median(abs(relative_edge - np.median(relative_edge))))
            )
        resolving = sum(rule.posr_threshold * n < rule.significance for n in noises)
        resolved_channels[lo:hi] = resolving
        if noises:
            noise_rows.append(
                {
                    "start_s": float(t[lo]),
                    "end_s": float(t[hi - 1] + dt),
                    "median_relative_edge_noise": float(np.median(noises)),
                    "channels": len(noises),
                    "resolving_channels": int(resolving),
                }
            )
    detectable = resolved_channels >= rule.minimum_channels
    noise_resolved = (
        maximum_filter1d(
            (~detectable).astype(np.uint8), 2 * radius + 1, mode="constant", cval=1
        )
        == 0
    )

    def proximity(times, reach=horizon):
        near = np.zeros(len(t), dtype=bool)
        for time in times:
            lo, hi = np.searchsorted(t, [time - reach, time + reach])
            # Include both boundary samples in the exclusion window.
            near[lo : min(len(t), hi + 1)] = True
        return near

    profile_near = proximity(candidates)
    periodic_near = proximity(periodic)
    core_edge_near = proximity(
        ambiguous_core_times, rule.isolated_edge_context_ms / 1000
    )
    observable_spans = [
        (float(t[lo]), float(t[hi]) if hi < len(t) else float(t[-1] + dt))
        for lo, hi in runs
    ]
    phase_spans, seen = [], set()
    for times in channel_times:
        for phase in core_relaxation_phases(times, observable_spans, rule):
            # Neighbouring channels see the same relaxation: keep one record.
            key = (phase["first_edge_s"], phase["last_edge_s"], phase["edges"])
            if key not in seen:
                seen.add(key)
                phase_spans.append(phase)
    phase_spans.sort(key=lambda phase: phase["start_s"])
    phase_support = np.zeros(len(t), dtype=bool)
    for phase in phase_spans:
        lo, hi = np.searchsorted(t, [phase["start_s"], phase["end_s"]])
        phase_support[lo:hi] = True
    absent = (
        observable
        & complete
        & noise_resolved
        & ~profile_near
        & ~periodic_near
        & ~core_edge_near
        & ~phase_support
    )
    return absent, {
        "policy": "complete_quiet_core_context",
        "validation_status": "unvalidated_research_policy",
        "context_radius_ms": horizon * 1000,
        "profile_passing_candidates": len(candidates),
        "core_relaxation_test": {
            "method": "per_channel_POSR_qualified_negative_fractional_edges",
            "aggregation": "union_over_supported_channels",
            "supported_channels": len(edges),
            "minimum_relative_edge": rule.significance,
            "posr_threshold": rule.posr_threshold,
            "minimum_period_ms": rule.minimum_period_ms,
            "maximum_period_ms": rule.maximum_period_ms,
            "minimum_train": rule.minimum_train,
            "candidate_edges": int(sum(len(x) for x in channel_times)),
            "ambiguous_edges": len(ambiguous_core_times),
            "ambiguous_edge_times_s": ambiguous_core_times,
            "periodic_edges": len(periodic),
            "periodic_edge_times_s": periodic,
            "phase_spans": phase_spans,
            "phase_grouping": {
                "edge_source": "per_channel_POSR_qualified_edge_times_s",
                "posr_threshold": rule.posr_threshold,
                "minimum_train": rule.relaxation_minimum_edges,
                "period_ratio": rule.period_ratio,
                "minimum_period_ms": rule.relaxation_minimum_period_ms,
                "maximum_period_ms": None,
                "context_radius_ms": horizon * 1000,
                "null_alpha": rule.periodicity_null_alpha,
                "null_replicates": rule.periodicity_null_replicates,
            },
            "positive_train_period_bounds": {
                "minimum_period_ms": rule.minimum_period_ms,
                "maximum_period_ms": rule.maximum_period_ms,
            },
            "noise_method": (
                "per_channel_maximum_period_block_relative_edge_MAD; "
                f"at least {rule.minimum_channels} resolving channels"
            ),
            "noise_windows": noise_rows,
        },
        "reason_samples": {
            "unobservable": int((~observable).sum()),
            "incomplete_context": int((observable & ~complete).sum()),
            "unresolved_core_noise": int((observable & ~noise_resolved).sum()),
            "near_profile_candidate": int((observable & profile_near).sum()),
            "periodic_core_relaxation": int((observable & periodic_near).sum()),
            "core_edge_ambiguous": int((observable & core_edge_near).sum()),
            "core_relaxation_phase": int((observable & phase_support).sum()),
            "tested_absence": int(absent.sum()),
        },
    }


def _burst(trace, crash, threshold, *, absolute=False):
    if trace is None:
        return None
    background = _window(trace, crash, -0.025, 0.025)
    near = _window(trace, crash, -0.002, 0.002)
    if len(background) <= 10 or not len(near):
        return None
    if absolute:
        background, near = abs(background), abs(near)
    baseline = np.median(background)
    scale = 1.4826 * np.median(abs(background - baseline))
    return bool(np.max(near) > baseline + threshold * max(scale, 1e-9))


def _stable_groups(accepted, observable, t, rule):
    """Period and inversion-stable groups, split at every unobserved sample."""
    times = np.asarray([time for time, _ in accepted])
    groups = []
    for start, stop in _runs(observable):
        lo, hi = np.searchsorted(times, [t[start], t[stop - 1]], side="left")
        hi = int(np.searchsorted(times, t[stop - 1], side="right"))
        for a, b in trains(times[lo:hi], rule):
            a, b = int(lo + a), int(lo + b)
            anchor = a
            channels = []
            for index in range(a, b):
                channel = accepted[index][1]["inversion_channel"]
                if channels and (
                    max(*channels, channel) - min(*channels, channel)
                    > rule.inversion_spread_channels
                ):
                    if index - anchor >= rule.minimum_train:
                        groups.append((anchor, index))
                    anchor, channels = index, []
                channels.append(channel)
            if b - anchor >= rule.minimum_train:
                groups.append((anchor, b))
    return groups


def detect(
    t_s,
    y,
    *,
    shot,
    rule=DEFAULT_RULE,
    qmin=None,
    geometry=None,
    dalpha=None,
    sxr=None,
    core_channels=None,
    ip=None,
    observability=None,
    observability_reference=None,
    neutron=None,
    mirnov=None,
    nbi=None,
    q_source="EFIT01",
    radius_geometry=None,
    spatially_verified=None,
):
    """ECE (channels,time) -> point crashes and train spans; all times seconds.

    Optional scalar diagnostics are (seconds, values). EFIT01 conflicts above
    1.4 flag uncertainty. Tested absence is the ECE quiet-core test (no POSR
    periodic edge on any valid channel inside nominal rho < 0.5); sustained
    q>=1.5 is a prior, not evidence: time it alone supports is returned in
    ``Detection.q_prior_mask`` and is never absent. Calibrated psi must show loss
    inside and gain outside q=1. Uncalibrated SXR has no corroboration claim.
    Observability can supply a density/cutoff mask; ``ece_validity`` removes
    further time whose ECE profile shows cutoff or mapping failure.
    Neutron evidence needs known NBI power in watts and a measured-noise drop;
    missing auxiliary evidence never downgrades an ECE crash.
    """
    t = np.asarray(t_s, dtype=float)
    values = np.array(y, dtype=np.float32, copy=True)
    if t.ndim != 1 or len(t) < 32 or values.ndim != 2 or values.shape[1] != len(t):
        raise ValueError("ECE needs (channels,time) and at least 32 times")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("times must be finite and increasing")
    if np.max(np.abs(np.diff(t) - dt)) > 0.05 * dt:
        raise ValueError("edge filter requires uniform sampling")
    sigma = rule.sigma_ms / 1000 / dt
    if sigma < 2:
        raise ValueError("at least two samples per Gaussian sigma are required")
    # Terminal channels have no verified radial ordering, even when their
    # apparent temperature looks plausible. They cannot supply an inversion.
    values[40:] = np.nan
    if radius_geometry is not None and radius_geometry.lcfs_outer_R_m is not None:
        if radius_geometry.R_m.shape != values.shape:
            raise ValueError("radius geometry and ECE sample axes disagree")
        overlap = radius_geometry.R_m < 2 / 3 * radius_geometry.lcfs_outer_R_m[None]
        values[overlap] = np.nan
    finite = np.isfinite(values)
    proxy = np.asarray(
        np.arange(len(values)) if core_channels is None else core_channels, dtype=int
    )
    if (
        proxy.ndim != 1
        or len(proxy) < rule.minimum_channels
        or ((proxy < 0).any() or (proxy >= len(values)).any())
    ):
        raise ValueError("core_channels needs valid ECE channel indices")
    calibrated_core = None
    if geometry is not None and geometry.psi is not None:
        positions = align_q(t * 1000, geometry.time_ms, geometry.psi)
        surfaces = align_q(t * 1000, geometry.time_ms, geometry.q1_psi[None])[0]
        valid_position = np.isfinite(positions) & (positions >= 0) & (positions <= 1)
        calibrated_core = valid_position & (positions < surfaces[None])
        # Without a q=1 surface, the closest measured channels to the axis
        # provide support only; the candidate remains equilibrium-uncertain.
        unknown = ~np.isfinite(surfaces)
        nearest = np.argsort(np.where(valid_position, positions, np.inf), axis=0)
        closest = np.zeros_like(valid_position)
        np.put_along_axis(closest, nearest[: rule.minimum_channels], True, axis=0)
        calibrated_core[:, unknown] = (closest & valid_position)[:, unknown]
        membership = calibrated_core
    else:
        membership = np.zeros_like(finite)
        membership[proxy] = True
    core_level = _core_reference(values, membership)
    # The local pre-crash core level is the comparison scale. A central crash
    # itself must not make a genuine outer rise look hotter than the core.
    reference_width = round(rule.frame_ms / 1000 / dt) | 1
    core_level = maximum_filter1d(
        np.where(np.isfinite(core_level), core_level, 0), reference_width
    )
    finite &= (values > 0) & (
        values <= rule.maximum_channel_to_core * core_level[None, :]
    )
    core_valid = finite & membership & (values >= rule.te_floor_kev)
    observable = (core_valid.sum(axis=0) >= rule.minimum_channels) & (
        finite.sum(axis=0) >= 4
    )
    accounting = {"core_observable_samples": int(observable.sum())}
    if observability_reference is not None:
        # Accounting only: what the superseded reference-field guard would remove.
        accounting["removed_by_reference_density_guard"] = int(
            (observable & ~np.asarray(observability_reference, dtype=bool)).sum()
        )
    if observability is not None:
        supplied = np.asarray(observability, dtype=bool)
        if supplied.shape != t.shape:
            raise ValueError("observability must have one boolean per ECE sample")
        accounting["removed_by_density_guard"] = int((observable & ~supplied).sum())
        observable &= supplied
    validity_flag, validity_info = ece_validity(values, t, rule, radius=radius_geometry)
    accounting["removed_by_ece_validity"] = int((observable & validity_flag).sum())
    validity_info["observable_removed_samples"] = accounting["removed_by_ece_validity"]
    observable &= ~validity_flag
    accounting["observable_samples_after_guards"] = int(observable.sum())
    clean = np.where(finite, values, 0)
    # No interpolation across missing data: exclude the whole filter support.
    bad = maximum_filter1d(
        (~finite).astype(np.uint8), 2 * int(np.ceil(3.5 * sigma)) + 1
    )
    c = gaussian_filter1d(clean, sigma, axis=1, order=1, truncate=3.5)
    step = c * (np.sqrt(2 * np.pi) * sigma)
    width = round(rule.frame_ms / 1000 / dt) | 1
    half, remove = width // 2, int(np.ceil(7 * sigma))
    observable &= (
        maximum_filter1d(
            (~observable).astype(np.uint8), 2 * int(np.ceil(3.5 * sigma)) + 1
        )
        == 0
    )
    observable[:half] = False
    observable[-half:] = False
    votes = []
    for channel, trace in enumerate(c):
        peaks, _ = find_peaks(np.abs(trace), distance=max(1, half))
        for k in peaks:
            if k < half or k + half >= len(t) or bad[channel, k]:
                continue
            if abs(step[channel, k]) < 0.005 * max(values[channel, k], 0):
                continue
            weight = posr(trace[k - half : k + half + 1], trace[k], remove)
            if weight >= rule.posr_threshold:
                votes.append((k, channel, abs(float(step[channel, k]))))
    votes.sort()
    clusters = []
    for vote in votes:
        if (
            not clusters
            or (vote[0] - clusters[-1][0][0]) * dt * 1000 > rule.coincidence_ms
        ):
            clusters.append([vote])
        else:
            clusters[-1].append(vote)
    rejected, accepted, profile_candidates = {}, [], []

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for cluster in clusters:
        if len({v[1] for v in cluster}) < rule.minimum_channels:
            reject("coincidence")
            continue
        k = round(np.average([v[0] for v in cluster], weights=[v[2] for v in cluster]))
        if not observable[k]:
            reject("unobservable_core")
            continue
        if accepted and t[k] - accepted[-1][0] < rule.coincidence_ms / 1000:
            reject("holdoff")
            continue
        profile = np.where(bad[:, k], np.nan, step[:, k])
        level = clean[:, max(0, k - half) : k + half + 1].mean(axis=1)
        verdict, attrs = inversion_profile(
            profile, level, rule, core_level=float(core_level[k])
        )
        if verdict != "accept":
            reject(verdict)
            continue
        profile_candidates.append(float(t[k]))
        attrs["uncertainty_reasons"] = []
        if spatially_verified is False:
            attrs["uncertainty_reasons"].append("unverified_spatial_adjacency")
        if core_channels is not None and (geometry is None or geometry.psi is None):
            loss = np.isfinite(profile[proxy]) & (
                profile[proxy] < -0.005 * level[proxy]
            )
            attrs["core_proxy_channels"] = proxy.tolist()
            attrs["core_moves"] = bool(loss.sum() >= rule.minimum_channels)
        elm = _burst(dalpha, t[k], rule.dalpha_burst_z)
        attrs["dalpha_coincident"] = elm
        if elm and not attrs["core_moves"]:
            reject("elm_edge_only")
            continue
        if not attrs["core_moves"] and (geometry is None or geometry.psi is None):
            reject("edge_only_proxy")
            continue
        if ip is not None:
            current = align_q([t[k]], ip[0], np.atleast_2d(ip[1]))[0, 0]
            if np.isfinite(current) and abs(current) < rule.minimum_ip_ma * 1e6:
                reject("low_plasma_current")
                continue
        attrs["qmin"] = None
        attrs["q_source"] = q_source if qmin is not None else None
        if qmin is not None:
            q = align_q([t[k]], qmin[0], np.atleast_2d(qmin[1]))[0, 0]
            if np.isfinite(q) and q > 0:
                attrs["qmin"] = float(q)
                conflict = (
                    1.05
                    if str(q_source).startswith("MSE-constrained")
                    else 1 + rule.qmin_margin
                )
                attrs["qmin_conflict_threshold"] = conflict
                if q > conflict:
                    attrs["uncertainty_reasons"].append("qmin_conflict")
        if radius_geometry is not None:
            from .geometry import radius_evidence

            nominal = radius_evidence(radius_geometry, t[k], attrs["inversion_channel"])
            attrs.update(nominal)
            attrs["geometry_status"] = "nominal_second_harmonic_R"
            if nominal["inversion_R_m"] is None:
                attrs["uncertainty_reasons"].append("unverified_spatial_adjacency")
            delta = nominal["q1_radius_difference_m"]
            if delta is not None:
                # This is a major-radius tolerance in metres, retained separately
                # from the calibrated sqrt(psi) tolerance below.
                attrs["q1_R_tolerance_m"] = 0.15
                if abs(delta) > 0.15:
                    attrs["uncertainty_reasons"].append("nominal_q1_radius_mismatch")
            elif attrs.get("qmin") is not None and attrs["qmin"] <= 1:
                attrs["q1_comparison_status"] = "EFIT_q1_mapping_unavailable"
            else:
                attrs["q1_comparison_status"] = "no_EFIT01_q1_or_missing_slice"
        if geometry is not None and geometry.psi is not None:
            psi = align_q([t[k] * 1000], geometry.time_ms, geometry.psi)[:, 0]
            surface = align_q(
                [t[k] * 1000], geometry.time_ms, geometry.q1_psi[None, :]
            )[0, 0]
            q = align_q([t[k] * 1000], geometry.time_ms, geometry.q)[:, 0]
            if np.isfinite(q).sum() >= 4 and not np.isfinite(surface):
                attrs["uncertainty_reasons"].append("no_q1_surface")
            if np.isfinite(surface) and surface > 0:
                losses = np.arange(attrs["drop_start"], attrs["drop_stop"])
                gains = np.arange(attrs["rise_start"], attrs["rise_stop"])
                inner_loss = np.isfinite(psi[losses]) & (psi[losses] < surface)
                outer_gain = np.isfinite(psi[gains]) & (psi[gains] > surface)
                wrong_loss = np.isfinite(psi[losses]) & (psi[losses] > surface)
                wrong_gain = np.isfinite(psi[gains]) & (psi[gains] < surface)
                if wrong_loss.any() or wrong_gain.any():
                    reject("calibrated_direction")
                    continue
                if (
                    inner_loss.sum() < rule.minimum_channels
                    or outer_gain.sum() < rule.minimum_channels
                ):
                    attrs["uncertainty_reasons"].append("incomplete_spatial_membership")
                attrs["core_moves"] = bool(inner_loss.sum() >= rule.minimum_channels)
            channel = attrs["inversion_channel"]
            lo, hi = int(np.floor(channel)), int(np.ceil(channel))
            if hi < len(psi) and np.isfinite(psi[[lo, hi]]).all() and surface > 0:
                radius = np.sqrt(float(np.interp(channel, np.arange(len(psi)), psi)))
                q1 = float(np.sqrt(surface))
                attrs.update(
                    inversion_rho=float(radius),
                    q1_rho=q1,
                    geometry_status="calibrated_psi",
                )
                if abs(radius - q1) > rule.radius_tolerance:
                    attrs["uncertainty_reasons"].append("q1_radius_mismatch")
        before = slice(max(0, k - round(0.0015 / dt)), k - round(0.0003 / dt))
        after = slice(k + round(0.0003 / dt), min(len(t), k + round(0.0015 / dt)))
        before_values, after_values = values[:, before], values[:, after]
        pre_count = np.isfinite(before_values).sum(axis=1)
        post_count = np.isfinite(after_values).sum(axis=1)
        pre = np.divide(
            np.where(np.isfinite(before_values), before_values, 0).sum(axis=1),
            pre_count,
            out=np.full(len(values), np.nan),
            where=pre_count > 0,
        )
        post = np.divide(
            np.where(np.isfinite(after_values), after_values, 0).sum(axis=1),
            post_count,
            out=np.full(len(values), np.nan),
            where=post_count > 0,
        )
        central = (
            np.flatnonzero(calibrated_core[:, k])
            if calibrated_core is not None
            else proxy
            if core_channels is not None
            else np.arange(attrs["drop_start"], attrs["drop_stop"])
        )
        usable = central[np.isfinite(pre[central]) & np.isfinite(post[central])]
        if not len(usable):
            reject("unobservable_central_channel")
            continue
        central_selection = "hottest_available_core_proxy"
        if radius_geometry is not None:
            distances = np.abs(
                radius_geometry.R_m[usable, k] - radius_geometry.axis_R_m[k]
            )
            mapped = np.isfinite(distances)
            if mapped.any():
                central = int(usable[mapped][np.argmin(distances[mapped])])
                central_selection = "nearest_nominal_EFIT_magnetic_axis"
            else:
                central = int(usable[np.argmax(pre[usable])])
        elif geometry is not None and geometry.psi is not None:
            mapped = np.isfinite(psi[usable])
            if mapped.any():
                central = int(usable[mapped][np.argmin(psi[usable[mapped]])])
                central_selection = "minimum_calibrated_poloidal_flux"
            else:
                central = int(usable[np.argmax(pre[usable])])
        else:
            central = int(usable[np.argmax(pre[usable])])
        amplitude = float((pre[central] - post[central]) / pre[central])
        attrs.update(
            central_channel=central,
            central_selection=central_selection,
            central_te_pre_kev=float(pre[central]),
            central_te_post_kev=float(post[central]),
            central_relative_drop=amplitude,
        )
        if pre[central] < rule.te_floor_kev or amplitude < rule.central_relative_drop:
            reject("central_relative_drop")
            continue
        attrs["sxr_corroboration"] = None
        attrs["sxr_geometry_status"] = (
            "unverified_spatial_pairing" if sxr is not None else "unavailable"
        )
        attrs.update(_neutron_evidence(neutron, nbi, t[k], rule))
        burst = _burst(mirnov, t[k], rule.mirnov_burst_z, absolute=True)
        attrs["mirnov_burst_corroboration"] = True if burst else None
        attrs.update(
            crowd=False,
            period_ms=None,
            coincident_channels=len({v[1] for v in cluster}),
        )
        accepted.append((float(t[k]), attrs))
    times = [a for a, _ in accepted]
    groups = _stable_groups(accepted, observable, t, rule)
    train_members = set()
    for train_index, (a, b) in enumerate(groups):
        train_members.update(range(a, b))
        period = float(np.median(np.diff(times[a:b])) * 1000)
        for _, attrs in accepted[a:b]:
            attrs["period_ms"] = period
            attrs["train_id"] = f"{shot}:{train_index}"
            attrs["state"] = "uncertain" if attrs["uncertainty_reasons"] else "present"
    kwargs = {
        "shot": int(shot),
        "source": "saw_physics",
        "phenomenon": "sawtooth_oscillation",
        "diag": "ece",
        "evidence_kind": "heuristic",
        "t_cov0_s": float(t[0]),
        "t_cov1_s": float(t[-1]),
    }
    crashes = [
        Event(
            t0_s=a,
            t1_s=a,
            confidence=min(1.0, attrs["coincident_channels"] / 8),
            attrs=attrs,
            **kwargs,
        )
        for index, (a, attrs) in enumerate(accepted)
        if index in train_members
    ]
    intervals, uncertain_intervals = [], []
    state_groups = []
    for train_start, train_stop in groups:
        boundaries = [train_start]
        for index in range(train_start + 1, train_stop):
            previous, current = accepted[index - 1][1], accepted[index][1]
            if (previous["state"], previous["uncertainty_reasons"]) != (
                current["state"],
                current["uncertainty_reasons"],
            ):
                boundaries.append(index)
        boundaries.append(train_stop)
        state_groups.extend(
            (a, b, train_start, train_stop) for a, b in pairwise(boundaries)
        )
    for a, b, train_start, train_stop in state_groups:
        radii = [
            attrs["inversion_rho"]
            for _, attrs in accepted[a:b]
            if attrs["inversion_rho"] is not None
        ]
        attrs = {
            "crowd": True,
            "crashes": b - a,
            "train_id": accepted[a][1]["train_id"],
            "period_ms": accepted[a][1]["period_ms"],
            "inversion_rho": float(np.median(radii)) if radii else None,
            "q1_rho": accepted[a][1]["q1_rho"],
            "inversion_R_m": None,
            "q1_R_m": None,
            "geometry_status": accepted[a][1]["geometry_status"],
            "inversion_channel": float(
                np.median([r[1]["inversion_channel"] for r in accepted[a:b]])
            ),
            "inversion_channel_spread": float(
                np.ptp([r[1]["inversion_channel"] for r in accepted[a:b]])
            ),
            "central_relative_drop": float(
                np.median([r[1]["central_relative_drop"] for r in accepted[a:b]])
            ),
            "state": accepted[a][1]["state"],
            "uncertainty_reasons": accepted[a][1]["uncertainty_reasons"],
        }
        output = uncertain_intervals if attrs["state"] == "uncertain" else intervals
        start = times[a] if a == train_start else (times[a - 1] + times[a]) / 2
        end = times[b - 1] + dt if b == train_stop else (times[b - 1] + times[b]) / 2
        if attrs["state"] == "uncertain":
            # Cover the final uncertain point and its finite filter support.
            left = int(np.searchsorted(t, times[a]))
            right = int(np.searchsorted(t, times[b - 1]))
            run = next(
                (lo, hi) for lo, hi in _runs(observable) if lo <= left <= right < hi
            )
            padding = rule.frame_ms / 2000
            if a == train_start:
                start = max(float(t[run[0]]), start - padding)
            run_stop = t[run[1]] if run[1] < len(t) else t[-1] + dt
            if b == train_stop:
                end = min(float(run_stop), end + padding)
        output.append(
            Event(t0_s=start, t1_s=end, confidence=1.0, attrs=attrs, **kwargs)
        )
    for index, (time, attrs) in enumerate(accepted):
        if index in train_members:
            continue
        attrs = dict(
            attrs,
            state="uncertain",
            uncertainty_reasons=sorted(
                set(attrs["uncertainty_reasons"] + ["isolated_candidate"])
            ),
        )
        left = max(0, np.searchsorted(t, time) - round(rule.frame_ms / 2000 / dt))
        right = min(
            len(t) - 1, np.searchsorted(t, time) + round(rule.frame_ms / 2000 / dt)
        )
        # Restrict the complete uncertain interval to this observable run.
        index = int(np.searchsorted(t, time))
        missing_left = np.flatnonzero(~observable[left:index])
        if len(missing_left):
            left += int(missing_left[-1]) + 1
        missing_right = np.flatnonzero(~observable[index : right + 1])
        if len(missing_right):
            right = index + int(missing_right[0]) - 1
        uncertain_intervals.append(
            Event(
                t0_s=float(t[left]),
                t1_s=float(t[right]),
                confidence=0.5,
                attrs=attrs,
                **kwargs,
            )
        )
    if radius_geometry is not None and radius_geometry.nominal_rho is not None:
        # Every valid channel inside nominal rho < quiet_core_max_rho, not only
        # the axis-selected proxy: a quiet core means quiet at all of them.
        core_support = finite & (radius_geometry.nominal_rho < rule.quiet_core_max_rho)
        quiet_support = "nominal_rho_all_valid_channels"
    else:
        core_support = finite & membership
        quiet_support = "core_proxy_channels"
    # Each supported channel keeps its own fractional edge: a signed proxy mean
    # can dilute a drop in a subset of channels or cancel it against rises.
    rows = np.flatnonzero(core_support.any(axis=1))
    edges = np.zeros((len(rows), len(t)), dtype=np.float32)
    for index, channel in enumerate(rows):
        channel_level = maximum_filter1d(clean[channel], reference_width)
        np.divide(
            c[channel],
            channel_level,
            out=edges[index],
            where=core_support[channel] & (channel_level > 0),
        )
    absent_mask, absence_diagnostics = _absence_evidence(
        t, observable, edges, core_support[rows], profile_candidates, rule, sigma
    )
    absence_diagnostics["profile_passing_candidate_times_s"] = profile_candidates
    absence_diagnostics["core_relaxation_test"]["support"] = {
        "channels": quiet_support,
        "maximum_nominal_rho": rule.quiet_core_max_rho,
        "isolated_edge_context_ms": rule.isolated_edge_context_ms,
    }
    absence_diagnostics["ece_validity"] = validity_info
    absence_diagnostics["guard_accounting"] = accounting
    high_q = np.zeros(len(t), dtype=bool)
    if qmin is not None:
        qvalues = align_q(t, qmin[0], np.atleast_2d(qmin[1]))[0]
        evidence = observable & np.isfinite(qvalues) & (qvalues >= rule.qmin_absence)
        for lo, hi in _runs(evidence):
            if (hi - lo) * dt >= rule.qmin_sustain_ms / 1000:
                high_q[lo:hi] = True
        # Contradictory core/profile drops remain uncertain. Use accepted
        # physical candidates rather than profile-only noisy coincidences.
        conflicts = [times[index] for index in train_members]
        horizon = 1.5 * rule.maximum_period_ms / 1000
        for time in conflicts:
            lo, hi = np.searchsorted(t, [time - horizon, time + horizon])
            high_q[lo : hi + 1] = False
        # Significant isolated core edges contradict a crash-free bin even
        # without a complete inversion. They protect their finite edge support,
        # rather than invalidating an entire high-q phase.
        for time in absence_diagnostics["core_relaxation_test"][
            "ambiguous_edge_times_s"
        ]:
            lo, hi = np.searchsorted(
                t, [time - rule.frame_ms / 2000, time + rule.frame_ms / 2000]
            )
            high_q[lo : hi + 1] = False
        for event in uncertain_intervals + intervals:
            lo, hi = np.searchsorted(t, [event.t0_s, event.t1_s])
            high_q[lo:hi] = False
    # Magnetics-only EFIT01 q_min is a prior, not a measurement of the core:
    # high q neither makes nor is required for tested absence. Time it alone
    # supports is a separate state, exported as uncertain and never a negative.
    q_prior = high_q & ~absent_mask
    absence_diagnostics["qmin_absence_test"] = {
        "threshold": rule.qmin_absence,
        "sustain_ms": rule.qmin_sustain_ms,
        "q_source": q_source if qmin is not None else None,
        "role": "supporting_prior_only; neither necessary nor sufficient",
        "conflicts_preserved": (
            "inversion_qualified_trains_full_context; "
            "isolated_POSR_core_edges_finite_support"
        ),
        "validation_status": "physics_rule_evidence_not_expert_validated",
    }
    absence_diagnostics["reason_samples"].update(
        sustained_high_q=int(high_q.sum()),
        tested_absence=int(absent_mask.sum()),
        tested_absence_with_high_q=int((absent_mask & high_q).sum()),
        tested_absence_without_high_q=int((absent_mask & ~high_q).sum()),
        q_prior_only=int(q_prior.sum()),
    )
    return Detection(
        crashes,
        intervals,
        rejected,
        len(clusters),
        observable,
        uncertain_intervals,
        absent_mask,
        absence_diagnostics,
        q_prior,
    )
