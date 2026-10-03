"""Gude-inspired multichannel edge detection with explicit geometry abstentions.

The profile test is a DIII-D adaptation, not a byte-for-byte implementation of
Gude's two-sided SXR test. Without calibrated radius, channel order establishes
an inversion boundary but does not establish its physical radius or q=1 proximity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
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
    minimum_period_ms: float = 5.0
    maximum_period_ms: float = 250.0
    period_ratio: float = 2.5
    minimum_train: int = 3
    qmin_margin: float = 0.05
    radius_tolerance: float = 0.15


DEFAULT_RULE = Rule()


@dataclass
class Detection:
    crashes: list[Event]
    intervals: list[Event]
    rejected: dict[str, int]
    candidates: int


def _runs(mask):
    ends = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(ends[::2], ends[1::2], strict=True))


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


def inversion_profile(step, level, rule=DEFAULT_RULE):
    """Contiguous core loss bordered by an outer gain, with Gude's A_norm/A_net.

    The hottest part of an uncalibrated array is only a temperature proxy for the
    core. Store the channel boundary separately from a calibrated physical radius.
    """
    step, level = np.asarray(step), np.asarray(level)
    valid = np.isfinite(step) & np.isfinite(level) & (level > 0)
    if valid.sum() < 4:
        return "coverage", {}
    profile = np.where(valid, step, 0.0)
    total = np.abs(profile).sum()
    norm = total / np.where(valid, np.abs(level), 0).sum()
    net = abs(profile.sum()) / max(total, 1e-12)
    attrs = {"a_norm": float(norm), "a_net": float(net)}
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
    c, d = max(nearby, key=lambda cd: profile[cd[0] : cd[1]].sum())
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
):
    """ECE (channels,time) -> point crashes and train spans; all times seconds.

    Optional qmin/dalpha are (seconds, scalar values); geometry is a calibrated
    Geometry object; SXR is a same-time array used only as corroborating evidence.
    Missing q or calibration never becomes an invented surface. Finite qmin>1.05
    and known finite q profiles without a q=1 surface veto a candidate.
    """
    t = np.asarray(t_s, dtype=float)
    values = np.asarray(y, dtype=np.float32)
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
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0)
    # No interpolation across missing data: exclude the whole filter support.
    bad = maximum_filter1d(
        (~finite).astype(np.uint8), 2 * int(np.ceil(3.5 * sigma)) + 1
    )
    c = gaussian_filter1d(clean, sigma, axis=1, order=1, truncate=3.5)
    step = c * (np.sqrt(2 * np.pi) * sigma)
    width = round(rule.frame_ms / 1000 / dt) | 1
    half, remove = width // 2, int(np.ceil(7 * sigma))
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
    rejected, accepted = {}, []

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    for cluster in clusters:
        if len({v[1] for v in cluster}) < rule.minimum_channels:
            reject("coincidence")
            continue
        k = round(np.average([v[0] for v in cluster], weights=[v[2] for v in cluster]))
        if accepted and t[k] - accepted[-1][0] < rule.minimum_period_ms / 1000:
            reject("holdoff")
            continue
        profile = np.where(bad[:, k], np.nan, step[:, k])
        level = gaussian_filter1d(
            clean[:, max(0, k - half) : k + half + 1].mean(axis=1), 1
        )
        verdict, attrs = inversion_profile(profile, level, rule)
        if verdict != "accept":
            reject(verdict)
            continue
        if core_channels is not None and (geometry is None or geometry.psi is None):
            proxy = np.asarray(core_channels, dtype=int)
            loss = np.isfinite(profile[proxy]) & (
                profile[proxy] < -0.005 * level[proxy]
            )
            attrs["core_proxy_channels"] = proxy.tolist()
            if loss.sum() < rule.minimum_channels:
                reject("edge_only_proxy")
                continue
            attrs["core_moves"] = True
        if ip is not None:
            current = align_q([t[k]], ip[0], np.atleast_2d(ip[1]))[0, 0]
            if np.isfinite(current) and abs(current) < 0.3e6:
                reject("low_plasma_current")
                continue
        attrs["qmin"] = None
        if qmin is not None:
            q = align_q([t[k]], qmin[0], np.atleast_2d(qmin[1]))[0, 0]
            if np.isfinite(q) and q > 0:
                attrs["qmin"] = float(q)
                if q > 1 + rule.qmin_margin:
                    reject("qmin_above_one")
                    continue
        if geometry is not None and geometry.psi is not None:
            psi = align_q([t[k] * 1000], geometry.time_ms, geometry.psi)[:, 0]
            surface = align_q(
                [t[k] * 1000], geometry.time_ms, geometry.q1_psi[None, :]
            )[0, 0]
            q = align_q([t[k] * 1000], geometry.time_ms, geometry.q)[:, 0]
            if np.isfinite(q).sum() >= 4 and not np.isfinite(surface):
                reject("no_q1_surface")
                continue
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
                    reject("q1_radius_mismatch")
                    continue
        elm = False
        if dalpha is not None:
            tx, trace = dalpha
            near = np.abs(np.asarray(tx) - t[k]) <= 0.002
            background = np.abs(np.asarray(tx) - t[k]) <= 0.025
            v = np.asarray(trace)[background]
            v = v[np.isfinite(v)]
            if len(v) > 10 and near.any():
                baseline = np.median(v)
                scale = 1.4826 * np.median(abs(v - baseline))
                elm = bool(
                    np.nanmax(np.asarray(trace)[near]) > baseline + 6 * max(scale, 1e-9)
                )
        if elm and not attrs["core_moves"]:
            reject("elm_edge_only")
            continue
        attrs["dalpha_coincident"] = elm
        attrs["sxr_corroboration"] = None
        if sxr is not None:
            near = sxr[:, max(0, k - 5) : k + 6]
            diff = np.diff(near, axis=1)
            attrs["sxr_corroboration"] = bool(
                np.sum(
                    np.max(np.abs(diff), axis=1)
                    > 0.02 * np.maximum(np.abs(near).mean(axis=1), 1e-9)
                )
                >= 2
            )
        attrs.update(
            crowd=False,
            period_ms=None,
            coincident_channels=len({v[1] for v in cluster}),
        )
        accepted.append((float(t[k]), attrs))
    times = [a for a, _ in accepted]
    groups = trains(times, rule)
    for a, b in groups:
        period = float(np.median(np.diff(times[a:b])) * 1000)
        for _, attrs in accepted[a:b]:
            attrs["period_ms"] = period
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
        for a, attrs in accepted
    ]
    intervals = []
    for a, b in groups:
        radii = [
            attrs["inversion_rho"]
            for _, attrs in accepted[a:b]
            if attrs["inversion_rho"] is not None
        ]
        attrs = {
            "crowd": True,
            "crashes": b - a,
            "period_ms": accepted[a][1]["period_ms"],
            "inversion_rho": float(np.median(radii)) if radii else None,
            "inversion_channel": float(
                np.median([r[1]["inversion_channel"] for r in accepted[a:b]])
            ),
        }
        intervals.append(
            Event(
                t0_s=times[a], t1_s=times[b - 1], confidence=1.0, attrs=attrs, **kwargs
            )
        )
    return Detection(crashes, intervals, rejected, len(clusters))
