r"""A strong rotating n=1/n=2 mode (tearing-mode proxy), from magnetic RMS.

Farre-Kaga et al. (2025) require n1 RMS >12 G continuously for 50 ms, with
onset at 10% of the peak. Here `mode_intervals` requires that uninterrupted
seed crossing BEFORE joining any runs. Both measured and median-smoothed RMS
must exceed the seed; sub-release dips and acquisition gaps never count toward
the hold. The n2 seed of 6 G is a local extension.

Each qualifying seed grows backwards/forwards to max(1 G, 10% of its seed peak).
Release dips <=50 ms may be joined only across measured samples. Merged components
retain their individual release levels in `release_components`; the summary level
is their minimum and the summary peak is their largest qualified seed peak.

`label_shot` additionally requires a coherent n-resolved line below 30 kHz, including
a continuous 50 ms supported seed and >=80% support through the span. Unsupported
or short seeds and sustained coherent sub-seed lines are uncertain, not absent.
The caller supplies N1FREQ/N2FREQ or Mirnov phase-coherence evidence. These magnetics
alone do not establish an island's poloidal number or distinguish every MHD family.

Frequency drops alone are `locked_candidate`, with `lock_time_ms`. Only independent
locked-mode confirmation truncates the rotating interval and sets `locked`; without
frequency its end/locking status is unknown. Observed onsets are points (iscrowd 0),
present intervals spans (iscrowd 1). Acquisition gaps remain NaN and are unobservable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, percentile_filter

from ..events.interval_tables import WITH_ATTRS, attrs_text

ABSENT, PRESENT, UNCERTAIN, NOT_OBSERVABLE = 0, 1, 2, 3
CATEGORY = "neoclassical_tearing_mode"
#: Why an interval ended.
DECAY, PLASMA_END, LOCKED, UNKNOWN = "decay", "plasma_end", "locked", "unknown"


@dataclass(frozen=True)
class ModeRule:
    """The five numbers that make one toroidal mode number's intervals (see module)."""

    n: int
    onset_g: float
    hold_ms: float = 50.0
    release_fraction: float = 0.10
    release_floor_g: float = 1.0
    merge_gap_ms: float = 50.0
    smooth_ms: float = 5.0
    #: A mode that starts less than this after the window opens was already there:
    #: its onset was not seen.
    onset_margin_ms: float = 20.0
    #: Retained for compatibility with earlier callers; continuous seed duration
    #: supersedes joined-run duty, so this field no longer changes the rule.
    min_duty: float = 0.5
    #: n = 2 only: the RMS counts where it exceeds this multiple of the n = 1 RMS, so
    #: the n = 2 harmonic of a large n = 1 mode is not a second mode.
    harmonic_ratio: float | None = None
    #: Development-only absent-time 95th percentile (calibration_dev_fix1.json).
    weak_g: float = 2.028201377


#: Farre-Kaga et al. 2025: above 12 G continuously for 50 ms; onset at 10% of peak.
N1_RULE = ModeRule(n=1, onset_g=12.0)
#: The 6 G n2 seed is a local extension, without an island-number assignment.
#: The ratio is the rounded-up development-only p99 during strong n1-only modes
#: (scripts/labeler/tm_calibrate_rule.py; calibration_dev_fix1.json). No test reference.
N2_RULE = ModeRule(n=2, onset_g=6.0, harmonic_ratio=0.57, weak_g=1.827998042)
RULES = (N1_RULE, N2_RULE)


@dataclass(frozen=True)
class Interval:
    """One mode's interval, in ms; `onset_seen` is False if it began with the window."""

    n: int
    start_ms: float
    end_ms: float
    peak_g: float
    peak_ms: float
    ended: str = DECAY
    locked: bool = False
    onset_seen: bool = True
    release_g: float = float("nan")
    m: int | None = None
    seed_duration_ms: float = 0.0
    seed_start_ms: float | None = None
    locked_candidate: bool = False
    locked_known: bool = False
    lock_time_ms: float | None = None
    lock_candidates_ms: tuple[float, ...] = ()
    coherent_fraction: float = 0.0
    #: Each merged component retains its own peak-relative release, not 10% of
    #: the largest merged peak. The displayed release is their minimum.
    release_components: tuple[tuple[float, float, float], ...] = ()


def uniform(t_ms, y) -> tuple[np.ndarray, np.ndarray, float]:
    """`(t, y, dt)` on a uniform grid at the record's median step, gaps as NaN."""
    t = np.asarray(t_ms, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(t)
    t, y = t[keep], y[keep]
    if t.size < 2:
        raise ValueError("a record needs two samples")
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    dt = float(np.median(np.diff(t)))
    if not dt > 0:
        raise ValueError("the time base does not advance")
    if np.allclose(np.diff(t), dt, rtol=0, atol=dt * 1e-3):
        return t, y, dt
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    values = np.interp(grid, t, y, left=np.nan, right=np.nan)
    right = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    left = np.maximum(right - 1, 0)
    across_gap = (t[right] - t[left] > 1.5 * dt) & (grid != t[right])
    values[across_gap] = np.nan
    return grid, values, dt


def _runs(mask) -> tuple[np.ndarray, np.ndarray]:
    """Half-open index runs `[a, b)` over which `mask` is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _bridge(starts, stops, gap: int, available=None) -> tuple[np.ndarray, np.ndarray]:
    """The runs, those at most `gap` samples apart joined into one."""
    out_a, out_b = [], []
    for a, b in zip(starts, stops, strict=True):
        if (
            out_b
            and a - out_b[-1] <= gap
            and (available is None or np.all(available[out_b[-1] : a]))
        ):
            out_b[-1] = b
        else:
            out_a.append(int(a))
            out_b.append(int(b))
    return np.asarray(out_a, dtype=int), np.asarray(out_b, dtype=int)


def smoothed(y, dt_ms: float, smooth_ms: float) -> np.ndarray:
    """`y` through a running median of `smooth_ms`; a gap (NaN) counts as no signal."""
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(y)
    y = np.where(valid, y, 0.0)
    width = max(1, round(smooth_ms / dt_ms))
    if width % 2 == 0:
        width += 1
    out = median_filter(y, size=width, mode="nearest") if width > 1 else y
    out[~valid] = np.nan
    return out


def window_slice(t, window) -> tuple[int, int]:
    """Indices `[i0, i1)` of the samples inside `window` (the whole record: none)."""
    if window is None:
        return 0, len(t)
    return (
        int(np.searchsorted(t, window[0], side="left")),
        int(np.searchsorted(t, window[1], side="right")),
    )


def mode_intervals(
    t_ms, rms, rule: ModeRule, window=None, *, reference=None, include_short=False
):
    """One mode's intervals from its RMS trace, inside `window` (ms), in time order.

    `reference` is the n = 1 trace on the same samples, which `rule.harmonic_ratio`
    compares with. Samples outside `window` are not read.
    """
    t, y, dt = uniform(t_ms, rms)
    xs = smoothed(y, dt, rule.smooth_ms)
    if rule.harmonic_ratio is not None and reference is not None:
        _, ref, _ = uniform(t_ms, reference)
        ref = smoothed(ref, dt, rule.smooth_ms)
        if len(ref) == len(xs):
            xs = np.where(xs > rule.harmonic_ratio * ref, xs, 0.0)
    i0, i1 = window_slice(t, window)
    xs[:i0] = 0.0
    xs[i1:] = 0.0
    hold = int(np.ceil(rule.hold_ms / dt - 1e-9))
    gap = int(np.floor(rule.merge_gap_ms / dt + 1e-9))
    # Duration is measured BEFORE any merging, on an uninterrupted crossing of
    # both the measured and smoothed seed threshold. Smoothing cannot erase a dip.
    available = np.isfinite(y)
    seed_mask = (xs > rule.onset_g) & (y > rule.onset_g)
    seeds = []
    for a, b in zip(*_runs(seed_mask), strict=True):
        if not include_short and b - a < hold:
            continue
        peak = a + int(np.argmax(xs[a:b]))
        release = max(rule.release_floor_g, rule.release_fraction * xs[peak])
        starts, stops = _bridge(*_runs(xs > release), gap, available)
        k = int(np.searchsorted(starts, peak, side="right")) - 1
        seeds.append(
            (int(starts[k]), int(stops[k]), peak, float(release), int(b - a), int(a))
        )
    seeds.sort()
    merged: list[list] = []
    for a, b, peak, release, duration, seed_start in seeds:
        component = (float(t[a]), float(t[b - 1]), release)
        if merged and a - merged[-1][1] <= gap and np.all(available[merged[-1][1] : a]):
            last = merged[-1]
            last[1] = max(last[1], b)
            if xs[peak] > xs[last[2]]:
                last[2] = peak
            last[3] = min(last[3], release)
            if duration > last[4]:
                last[4:6] = [duration, seed_start]
            if component not in last[6]:
                last[6].append(component)
        else:
            merged.append([a, b, peak, release, duration, seed_start, [component]])
    found = []
    for a, b, peak, release, duration, seed_start, components in merged:
        touches_end = window is not None and b >= i1
        found.append(
            Interval(
                n=rule.n,
                start_ms=float(t[a]),
                end_ms=float(min(t[b - 1], window[1])) if window else float(t[b - 1]),
                peak_g=float(xs[peak]),
                peak_ms=float(t[peak]),
                ended=PLASMA_END if touches_end else DECAY,
                onset_seen=window is None or (a - i0) * dt >= rule.onset_margin_ms,
                release_g=release,
                seed_duration_ms=float(duration * dt),
                seed_start_ms=float(t[seed_start]),
                release_components=tuple(components),
            )
        )
    return found


def tearing_intervals(t_ms, n1, n2=None, window=None, rules=RULES):
    """Every mode's intervals, n = 1 first then n = 2, each in time order."""
    found = []
    for rule in rules:
        if rule.n == 1:
            found += mode_intervals(t_ms, n1, rule, window)
        elif rule.n == 2 and n2 is not None:
            found += mode_intervals(t_ms, n2, rule, window, reference=n1)
    return found


def frequency_locks(
    t_ms,
    freq_khz,
    *,
    lock_khz: float = 1.0,
    hold_ms: float = 20.0,
    spin_khz: float = 1.5,
    lookback_ms: float = 300.0,
) -> np.ndarray:
    """Times (ms) a mode's frequency fell to `lock_khz` and stayed, once rotating.

    `freq_khz` is `\\MHD::N<n>FREQ`, the toroidal mode's frequency (0.5 kHz steps). A
    locking is the first sample of a stretch of at least `hold_ms` at or below
    `lock_khz` that follows a frequency of at least `spin_khz` within `lookback_ms`: a
    mode born slow is not one that locked.
    """
    t, f, dt = uniform(t_ms, freq_khz)
    low = np.nan_to_num(f, nan=np.inf) <= lock_khz
    hold = max(1, round(hold_ms / dt))
    back = max(1, round(lookback_ms / dt))
    spinning = np.nan_to_num(f, nan=0.0)
    found = []
    for a, b in zip(*_runs(low), strict=True):
        if (
            b - a >= hold
            and spinning[max(0, a - back) : a].max(initial=0.0) >= spin_khz
        ):
            found.append(float(t[a]))
    return np.asarray(found, dtype=float)


def coherent_frequency(freq_khz, dt_ms, *, window_ms=50.0):
    """A sustained rotating line, excluding fast sweeps and stationary activity.

    The local 50 ms 10th--90th-percentile width must be <=max(2 kHz, 25% of
    median frequency), with 1.5<=f<=30 kHz. This local stability convention is
    an explicit extension to reject broadband/chirping bursts, not a published
    tearing/island discriminator. Missing samples remain unsupported.
    """
    freq = np.asarray(freq_khz, dtype=float)
    finite = np.isfinite(freq)
    safe = np.where(finite, freq, 0.0)
    width = max(1, round(window_ms / dt_ms))
    if width % 2 == 0:
        width += 1
    lo = percentile_filter(safe, 10, size=width, mode="nearest")
    hi = percentile_filter(safe, 90, size=width, mode="nearest")
    median = median_filter(safe, size=width, mode="nearest")
    return (
        finite
        & (freq >= 1.5)
        & (freq <= 30.0)
        & (hi - lo <= np.maximum(2.0, 0.25 * median))
    )


def apply_locking(
    intervals,
    lock_ms,
    *,
    confirmed_ms=None,
    tail_ms: float = 150.0,
    after_ms: float = 100.0,
):
    """Flag frequency drops as candidates; truncate only independently confirmed locks.

    `confirmed_ms` supplies times confirmed by a locked-mode diagnostic. A drop
    alone never sets `locked`. Missing frequency leaves end/locking unknown.
    A confirmation must coincide with a frequency drop (within 20 ms) inside the
    span; a candidate after a span's decay does not extend that rotating span.
    """
    out = []
    for item in intervals:
        times = lock_ms.get(item.n) if isinstance(lock_ms, dict) else lock_ms
        known = times is not None
        times = np.sort(np.asarray([] if times is None else times, dtype=float))
        confirmation = (
            confirmed_ms.get(item.n) if isinstance(confirmed_ms, dict) else confirmed_ms
        )
        item = replace(
            item,
            ended=item.ended if known else UNKNOWN,
            locked_known=known and confirmation is not None,
        )
        # Preserve every in-span drop; continued rotation makes it unconfirmed,
        # not grounds to silently erase the candidate. `tail_ms` is retained for
        # callers of the earlier end-only flagging API.
        hit = (times >= item.start_ms) & (times <= item.end_ms + after_ms)
        if hit.any():
            item = replace(
                item,
                locked_candidate=True,
                lock_time_ms=float(times[hit][0]),
                lock_candidates_ms=tuple(float(time) for time in times[hit]),
            )
        for time in times:
            if (
                not item.start_ms < time <= item.end_ms
                or confirmation is None
                or time < (item.seed_start_ms or item.start_ms) + 50.0
            ):
                continue
            if np.any(np.abs(np.asarray(confirmation) - time) <= 20.0):
                item = replace(
                    item,
                    ended=LOCKED,
                    locked=True,
                    locked_candidate=False,
                    lock_time_ms=float(time),
                    end_ms=float(time),
                )
                break
        out.append(item)
    return out


def present_mask(intervals, t_ms, n: int | None = None) -> np.ndarray:
    """True at each of `t_ms` inside any interval (of toroidal number `n`, if given)."""
    t = np.asarray(t_ms, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    for item in intervals:
        if n is None or item.n == n:
            mask |= (t >= item.start_ms) & (t <= item.end_ms)
    return mask


def coverage_gaps(t_ms, y, window, *, min_ms: float) -> list[tuple[float, float]]:
    """The stretches of `window` the record did not cover for at least `min_ms`.

    A stretch is uncovered where the record has no sample, or only gaps (NaN).
    """
    t = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t_ok = t[ok]
    w0, w1 = float(window[0]), float(window[1])
    if not t_ok.size:
        return [(w0, w1)] if w1 > w0 else []
    step = float(np.median(np.diff(t_ok))) if t_ok.size > 1 else 1.0
    edges = np.concatenate(([w0], t_ok[(t_ok > w0) & (t_ok < w1)], [w1]))
    gaps = []
    for a, b in pairwise(edges):
        if b - a > max(min_ms, 1.5 * step):
            gaps.append((float(a), float(b)))
    return gaps


def _minus(spans, holes):
    """`spans` (`(a, b)` pairs) less the `holes`."""
    out = list(spans)
    for h0, h1 in holes:
        nxt = []
        for a, b in out:
            if h1 <= a or h0 >= b:
                nxt.append((a, b))
                continue
            if h0 > a:
                nxt.append((a, h0))
            if h1 < b:
                nxt.append((h1, b))
        out = nxt
    return [(a, b) for a, b in out if b > a]


def _union(spans):
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


@dataclass(frozen=True)
class ShotLabel:
    """One shot's label: the plasma's intervals and what is not a clean absence."""

    shot: int
    window: tuple[float, float]
    start_ms: float
    intervals: tuple[Interval, ...]
    ramp_up: tuple[tuple[float, float], ...] = ()
    not_observable: tuple[tuple[float, float], ...] = ()
    uncertain: tuple[tuple[float, float, str, int], ...] = ()


def label_shot(
    shot: int,
    t_ms,
    n1,
    n2,
    window,
    start_ms: float | None = None,
    *,
    lock_ms=None,
    confirmed_lock_ms=None,
    coherent=None,
    seed_coherent=None,
    weak_coherent=None,
    weak_release_coherent=None,
    rules=RULES,
    gap_ms: float = 0.0,
    m_of=None,
) -> ShotLabel:
    """The shot's label over `window`, the plasma starting at `start_ms`.

    The rule runs from the plasma's start to the window's end. The ramp-up before it
    is uncertain where the rule fires on it, so a mode of the ramp-up neither makes an
    interval nor passes for an absence. Stretches of the window the n = 1 record did
    not cover for `gap_ms` are not observable. `m_of(n, start_ms, end_ms)`, if given,
    returns an interval's poloidal number or None (`surface.supported_m`).
    """
    w0, w1 = float(window[0]), float(window[1])
    start = w0 if start_ms is None else min(max(float(start_ms), w0), w1)
    plasma, uncertain = [], []
    t, first, dt = uniform(t_ms, n1)
    traces = {1: first}
    if n2 is not None:
        traces[2] = uniform(t_ms, n2)[1]
    for mode_rule in rules:
        n = mode_rule.n
        if n not in traces:
            continue
        y = traces[n]
        reference = first if n == 2 else None
        valid = np.isfinite(y)
        support = np.zeros(t.shape, bool)
        if coherent is not None and n in coherent:
            support = np.asarray(coherent[n], dtype=bool) & valid
        weak_support = support & (y > mode_rule.weak_g)
        if weak_coherent is not None and n in weak_coherent:
            weak_support |= np.asarray(weak_coherent[n], dtype=bool) & valid
        candidates = mode_intervals(
            t, y, mode_rule, (start, w1), reference=reference, include_short=True
        )
        strict = mode_intervals(t, y, mode_rule, (start, w1), reference=reference)
        for item in strict:
            inside = (t >= item.start_ms) & (t <= item.end_ms)
            fraction = float(support[inside].mean())
            seed_support = support
            if seed_coherent is not None and n in seed_coherent:
                seed_support = np.asarray(seed_coherent[n], dtype=bool) & valid
            seed = (y > mode_rule.onset_g) & seed_support & inside
            if n == 2:
                seed &= y > mode_rule.harmonic_ratio * first
            duration = max(
                (b - a for a, b in zip(*_runs(seed), strict=True)), default=0
            )
            if fraction >= 0.8 and duration * dt >= mode_rule.hold_ms:
                plasma.append(replace(item, coherent_fraction=fraction))
        accepted = [(i.start_ms, i.end_ms) for i in plasma if i.n == n]
        for item in candidates:
            reason = (
                "short_seed"
                if item.seed_duration_ms < mode_rule.hold_ms
                else "coherent_line_unconfirmed_or_above_30khz"
            )
            uncertain.extend(
                (a, b, reason, n)
                for a, b in _minus([(item.start_ms, item.end_ms)], accepted)
            )
        # Weak lines visible in Mirnov remain uncertain even below the quiet-RMS
        # p95. RMS above that p95 requires rotating-line evidence too.
        weak = weak_support & (t >= start) & (t <= w1)
        weak_starts, weak_stops = _runs(weak)
        cores = [
            (a, b)
            for a, b in zip(weak_starts, weak_stops, strict=True)
            if (b - a) * dt >= 100.0
        ]
        release = weak.copy()
        if weak_release_coherent is not None and n in weak_release_coherent:
            release |= np.asarray(weak_release_coherent[n], dtype=bool) & valid
            release &= (t >= start) & (t <= w1)
        release_starts, release_stops = _runs(release)
        for a, b in zip(
            *_bridge(release_starts, release_stops, int(50.0 / dt), valid),
            strict=True,
        ):
            # One uninterrupted 100 ms core establishes the weak track before
            # filling short evidence interruptions. Missing acquisition is never
            # bridged; no collection of short fragments can establish a core.
            if any(a <= lo and hi <= b for lo, hi in cores):
                uncertain.extend(
                    (lo, hi, "coherent_sub_seed", n)
                    for lo, hi in _minus([(float(t[a]), float(t[b - 1]))], accepted)
                )
    plasma = apply_locking(plasma, lock_ms, confirmed_ms=confirmed_lock_ms)
    if m_of is not None:
        plasma = [replace(i, m=m_of(i.n, i.start_ms, i.end_ms)) for i in plasma]
    ramp = ()
    if start > w0:
        early = tearing_intervals(t_ms, n1, n2, (w0, start), rules)
        ramp = tuple(_union([(i.start_ms, i.end_ms) for i in early]))
    gaps = coverage_gaps(t_ms, n1, (w0, w1), min_ms=gap_ms)
    return ShotLabel(
        int(shot), (w0, w1), start, tuple(plasma), ramp, tuple(gaps), tuple(uncertain)
    )


def _row(shot, category, a, b, attrs=None):
    return {
        "shot": int(shot),
        "category": int(category),
        "t_start": round(float(a), 3),
        "t_end": round(float(b), 3),
        "confidence": np.nan,
        "attrs": attrs_text(attrs or {}),
    }


def interval_attrs(item: Interval, *, crowd: int) -> dict:
    """The catalog attributes of one interval's span (`crowd` 1) or onset (0)."""
    attrs = {"iscrowd": int(crowd), "n": int(item.n)}
    attrs.update(ended=item.ended, locked_known=item.locked_known)
    if item.locked:
        attrs["locked"] = True
    if item.locked_candidate:
        attrs["locked_candidate"] = True
    if item.lock_time_ms is not None:
        attrs["lock_time_ms"] = float(item.lock_time_ms)
    if item.lock_candidates_ms:
        attrs["lock_candidates_ms"] = list(item.lock_candidates_ms)
    if item.m is not None:
        # m = n q needs the safety factor, which is the offline EFIT01 here
        attrs["m"] = int(item.m)
        attrs["efit_tree"] = "efit01"
    return attrs


def shot_table(label: ShotLabel) -> pd.DataFrame:
    """The shot's rows in the catalog's interval schema, sorted by time.

    Each interval is a present span (`iscrowd` 1) and, if its onset was seen, a present
    point at its start (`iscrowd` 0), both with the mode's `n` (and `locked`, and `m`
    where EFIT's q supports one). Where nothing else holds, the window is absent; the
    ramp-up is uncertain where the rule fired in it; and what the record did not cover
    is not observable.
    """
    rows = []
    for item in label.intervals:
        rows.append(
            _row(
                label.shot,
                PRESENT,
                item.start_ms,
                item.end_ms,
                interval_attrs(item, crowd=1),
            )
        )
        if item.onset_seen:
            rows.append(
                _row(
                    label.shot,
                    PRESENT,
                    item.start_ms,
                    item.start_ms,
                    interval_attrs(item, crowd=0),
                )
            )
    w0, w1 = label.window
    gaps = list(label.not_observable)
    present = [(i.start_ms, i.end_ms) for i in label.intervals]
    ramp = _minus(list(label.ramp_up), gaps + present)
    uncertain = []
    for a, b, reason, n in label.uncertain:
        uncertain.extend(
            (lo, hi, reason, n) for lo, hi in _minus([(a, b)], gaps + present)
        )
    busy = _union(present + ramp + gaps + [(a, b) for a, b, _, _ in uncertain])
    rows += [_row(label.shot, NOT_OBSERVABLE, a, b) for a, b in gaps]
    rows += [_row(label.shot, UNCERTAIN, a, b) for a, b in ramp]
    rows += [
        _row(label.shot, UNCERTAIN, a, b, {"reason": reason, "n": n})
        for a, b, reason, n in uncertain
    ]
    rows += [_row(label.shot, ABSENT, a, b) for a, b in _minus([(w0, w1)], busy)]
    frame = pd.DataFrame(rows, columns=list(WITH_ATTRS))
    return frame.sort_values(
        ["t_start", "t_end", "category"], kind="stable"
    ).reset_index(drop=True)


def intervals_frame(labels) -> pd.DataFrame:
    """One row per interval across shots, with what the catalog schema cannot hold."""
    rows = [
        {
            "shot": label.shot,
            "n": item.n,
            "t_start": item.start_ms,
            "t_end": item.end_ms,
            "duration_ms": item.end_ms - item.start_ms,
            "peak_g": item.peak_g,
            "peak_ms": item.peak_ms,
            "release_g": item.release_g,
            "ended": item.ended,
            "locked": item.locked,
            "locked_candidate": item.locked_candidate,
            "locked_known": item.locked_known,
            "lock_time_ms": item.lock_time_ms,
            "lock_candidates_ms": item.lock_candidates_ms,
            "seed_duration_ms": item.seed_duration_ms,
            "seed_start_ms": item.seed_start_ms,
            "coherent_fraction": item.coherent_fraction,
            "release_components": item.release_components,
            "onset_seen": item.onset_seen,
            "m": item.m,
        }
        for label in labels
        for item in label.intervals
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "shot",
            "n",
            "t_start",
            "t_end",
            "duration_ms",
            "peak_g",
            "peak_ms",
            "release_g",
            "ended",
            "locked",
            "locked_candidate",
            "locked_known",
            "lock_time_ms",
            "lock_candidates_ms",
            "seed_duration_ms",
            "seed_start_ms",
            "coherent_fraction",
            "release_components",
            "onset_seen",
            "m",
        ],
    )
