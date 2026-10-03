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
    minimum_period_ms: float = 20.0
    maximum_period_ms: float = 250.0
    period_ratio: float = 2.5
    minimum_train: int = 3
    qmin_margin: float = 0.05
    radius_tolerance: float = 0.15
    central_relative_drop: float = 0.1
    te_floor_kev: float = 0.5
    inversion_spread_channels: float = 2.0
    minimum_ip_ma: float = 0.3
    neutron_relative_drop: float = 0.02
    mirnov_burst_z: float = 6.0
    dalpha_burst_z: float = 6.0


DEFAULT_RULE = Rule()


@dataclass
class Detection:
    crashes: list[Event]
    intervals: list[Event]
    rejected: dict[str, int]
    candidates: int
    observable: np.ndarray
    uncertain_intervals: list[Event]


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


def _window(trace, crash, lower, upper):
    """Finite native auxiliary samples in a time window, without interpolation."""
    tx, values = (np.asarray(x) for x in trace)
    a, b = np.searchsorted(tx, [crash + lower, crash + upper])
    segment = values[a:b]
    return segment[np.isfinite(segment)]


def _relative_drop(trace, crash):
    if trace is None:
        return None
    before = _window(trace, crash, -0.0015, -0.0003)
    after = _window(trace, crash, 0.0003, 0.0015)
    if not len(before) or not len(after) or np.median(before) <= 0:
        return None
    return float((np.median(before) - np.median(after)) / np.median(before))


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
    neutron=None,
    mirnov=None,
    q_source="EFIT01",
):
    """ECE (channels,time) -> point crashes and train spans; all times seconds.

    Optional scalar diagnostics are (seconds, values). Equilibrium conflicts flag
    uncertainty; they never assert absence. Calibrated psi must show loss inside
    and gain outside q=1. Uncalibrated SXR has no corroboration claim. Observability
    can supply a density/cutoff mask in addition to finite core ECE and its Te floor.
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
        core_valid = finite & calibrated_core & (values >= rule.te_floor_kev)
    else:
        core_valid = finite[proxy] & (values[proxy] >= rule.te_floor_kev)
    observable = (core_valid.sum(axis=0) >= rule.minimum_channels) & (
        finite.sum(axis=0) >= 4
    )
    if observability is not None:
        supplied = np.asarray(observability, dtype=bool)
        if supplied.shape != t.shape:
            raise ValueError("observability must have one boolean per ECE sample")
        observable &= supplied
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
    rejected, accepted = {}, []

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
        level = gaussian_filter1d(
            clean[:, max(0, k - half) : k + half + 1].mean(axis=1), 1
        )
        verdict, attrs = inversion_profile(profile, level, rule)
        if verdict != "accept":
            reject(verdict)
            continue
        attrs["uncertainty_reasons"] = []
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
                if q > 1 + rule.qmin_margin:
                    attrs["uncertainty_reasons"].append("qmin_conflict")
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
        central = int(usable[np.argmax(pre[usable])])
        amplitude = float((pre[central] - post[central]) / pre[central])
        attrs.update(
            central_channel=central,
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
        neutron_drop = _relative_drop(neutron, t[k])
        attrs["neutron_relative_drop"] = neutron_drop
        attrs["neutron_drop_corroboration"] = (
            None if neutron_drop is None else neutron_drop >= rule.neutron_relative_drop
        )
        attrs["mirnov_burst_corroboration"] = _burst(
            mirnov, t[k], rule.mirnov_burst_z, absolute=True
        )
        evidence = [
            attrs["neutron_drop_corroboration"],
            attrs["mirnov_burst_corroboration"],
        ]
        if any(item is not None for item in evidence) and not any(evidence):
            attrs["uncertainty_reasons"].append("auxiliary_not_corroborated")
        attrs.update(
            crowd=False,
            period_ms=None,
            coincident_channels=len({v[1] for v in cluster}),
        )
        accepted.append((float(t[k]), attrs))
    times = [a for a, _ in accepted]
    groups = _stable_groups(accepted, observable, t, rule)
    train_members = set()
    for a, b in groups:
        train_members.update(range(a, b))
        period = float(np.median(np.diff(times[a:b])) * 1000)
        reasons = sorted(
            {
                reason
                for _, attrs in accepted[a:b]
                for reason in attrs["uncertainty_reasons"]
            }
        )
        for _, attrs in accepted[a:b]:
            attrs["period_ms"] = period
            attrs["state"] = "uncertain" if reasons else "present"
            attrs["uncertainty_reasons"] = reasons
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
        start, end = times[a], times[b - 1]
        if attrs["state"] == "uncertain":
            # Cover the final uncertain point and its finite filter support.
            left = int(np.searchsorted(t, start))
            right = int(np.searchsorted(t, end))
            run = next(
                (lo, hi) for lo, hi in _runs(observable) if lo <= left <= right < hi
            )
            padding = rule.frame_ms / 2000
            start = max(float(t[run[0]]), start - padding)
            end = min(float(t[run[1] - 1] + dt), end + padding)
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
    return Detection(
        crashes, intervals, rejected, len(clusters), observable, uncertain_intervals
    )
