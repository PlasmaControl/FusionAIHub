r"""The whole-interval tearing-mode label: where a mode is present, by magnetic RMS.

The lab's tearing-mode labels so far were onsets: Seo's archive marks the growth phase
of an n = 1 mode, and the survival labels (Farre-Kaga et al. 2025) the time the n = 1
RMS first reaches a tenth of a peak above 12 G that holds for 50 ms (Fu et al. 2020 used
10 G for 50 ms). This module keeps that rule and makes it a whole interval, the way the
owner asked: a mode is present from its onset until it decays, locks or the plasma ends.

One mode of toroidal number `n` is read off `\MHD::N<n>RMS` (gauss, 1 kHz) in four
steps, each a number a `ModeRule` names:

1. The trace is smoothed by a running median of `smooth_ms`, so a spike of a few samples
   (an ELM, a sawtooth crash) is not a mode.
2. A seed is a stretch where the smoothed trace is above `onset_g` (Farre-Kaga's "peaks
   above 12 G"); runs of it at most `merge_gap_ms` apart are one stretch, a locking
   mode's RMS being ragged, if the runs fill `min_duty` of it (a train of ELM spikes
   does not). Its peak sets the release level of step 3.
3. The mode around the peak is the stretch that stays above `release_fraction` of the
   peak (Farre-Kaga's onset at 10 % of the peak), never below `release_floor_g`, the
   magnetics' noise; dips shorter than `merge_gap_ms` do not end it. It is a mode, and
   an interval, only if it lasts `hold_ms` (Farre-Kaga's 50 ms): a seed whose mode is
   shorter, such as a burst the trace falls from at once, is not. This is the
   hysteresis: a high threshold to start an interval, a lower one to end it.
4. Intervals closer than `merge_gap_ms` are one.

An interval ends at decay (the trace fell below its release level), at the end of the
plasma (the window's end), or at locking (`apply_locking`: a locked-mode detection
inside the interval or just after its end). The onset is a point event (`iscrowd` 0) at
the interval's start; the interval is a span (`iscrowd` 1). `shot_table` writes both in
the catalog's interval schema with the rest of the window absent, the ramp-up uncertain
where the rule fires in it, and what the record did not cover not observable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from ..events.interval_tables import WITH_ATTRS, attrs_text

ABSENT, PRESENT, UNCERTAIN, NOT_OBSERVABLE = 0, 1, 2, 3
CATEGORY = "neoclassical_tearing_mode"
#: Why an interval ended.
DECAY, PLASMA_END, LOCKED = "decay", "plasma_end", "locked"


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
    #: A seed's runs above `onset_g` must fill this fraction of the stretch they span,
    #: so a train of short spikes (ELMs) bridged by `merge_gap_ms` is not one mode.
    min_duty: float = 0.5
    #: n = 2 only: the RMS counts where it exceeds this multiple of the n = 1 RMS, so
    #: the n = 2 harmonic of a large n = 1 mode is not a second mode.
    harmonic_ratio: float | None = None


#: Farre-Kaga et al. 2025: a peak above 12 G, the mode lasting 50 ms, onset at 10 % of
#: the peak.
N1_RULE = ModeRule(n=1, onset_g=12.0)
#: No published n = 2 rule. Half the n = 1 onset: a mode's vacuum field falls off as
#: r^-(m+1), and the 3/2 sits further in than the 2/1. Harmonics of an n = 1 mode reach
#: 0.3 of it (shots 185805, 187043), hence the ratio.
N2_RULE = ModeRule(n=2, onset_g=6.0, harmonic_ratio=0.4)
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
    return grid, np.interp(grid, t, y, left=np.nan, right=np.nan), dt


def _runs(mask) -> tuple[np.ndarray, np.ndarray]:
    """Half-open index runs `[a, b)` over which `mask` is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _bridge(starts, stops, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """The runs, those at most `gap` samples apart joined into one."""
    out_a, out_b = [], []
    for a, b in zip(starts, stops, strict=True):
        if out_b and a - out_b[-1] <= gap:
            out_b[-1] = b
        else:
            out_a.append(int(a))
            out_b.append(int(b))
    return np.asarray(out_a, dtype=int), np.asarray(out_b, dtype=int)


def smoothed(y, dt_ms: float, smooth_ms: float) -> np.ndarray:
    """`y` through a running median of `smooth_ms`; a gap (NaN) counts as no signal."""
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0)
    width = max(1, round(smooth_ms / dt_ms))
    if width % 2 == 0:
        width += 1
    return median_filter(y, size=width, mode="nearest") if width > 1 else y


def window_slice(t, window) -> tuple[int, int]:
    """Indices `[i0, i1)` of the samples inside `window` (the whole record: none)."""
    if window is None:
        return 0, len(t)
    return (
        int(np.searchsorted(t, window[0], side="left")),
        int(np.searchsorted(t, window[1], side="right")),
    )


def mode_intervals(t_ms, rms, rule: ModeRule, window=None, *, reference=None):
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
    groups: list[list[int]] = []
    for a, b in zip(*_runs(xs > rule.onset_g), strict=True):
        if groups and a - groups[-1][1] <= gap:
            groups[-1][1] = int(b)
            groups[-1][2] += int(b - a)
        else:
            groups.append([int(a), int(b), int(b - a)])
    seeds = []
    for a, b, above in groups:
        if above < rule.min_duty * (b - a):
            continue
        peak = a + int(np.argmax(xs[a:b]))
        release = max(rule.release_floor_g, rule.release_fraction * xs[peak])
        starts, stops = _bridge(*_runs(xs > release), gap)
        k = int(np.searchsorted(starts, peak, side="right")) - 1
        if stops[k] - starts[k] >= hold:
            seeds.append((int(starts[k]), int(stops[k]), peak, float(release)))
    seeds.sort()
    merged: list[list] = []
    for a, b, peak, release in seeds:
        if merged and a - merged[-1][1] <= gap:
            last = merged[-1]
            last[1] = max(last[1], b)
            if xs[peak] > xs[last[2]]:
                last[2] = peak
            last[3] = min(last[3], release)
        else:
            merged.append([a, b, peak, release])
    found = []
    for a, b, peak, release in merged:
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


def apply_locking(
    intervals, lock_ms, *, tail_ms: float = 150.0, after_ms: float = 100.0
):
    """The intervals, each that ended by locking marked `locked`.

    A locked mode no longer rotates and leaves the RMS band, so the RMS collapses at
    locking. An interval is locked when a locking (`frequency_locks`) falls within its
    last `tail_ms` or up to `after_ms` after its end. `lock_ms` is the locking times, or
    a dict of them by toroidal number (none: nothing changes). An interval that the
    plasma ended keeps saying so; its `locked` is still set.
    """
    out = []
    for item in intervals:
        times = lock_ms.get(item.n) if isinstance(lock_ms, dict) else lock_ms
        times = np.sort(np.asarray([] if times is None else times, dtype=float))
        hit = (times >= item.end_ms - tail_ms) & (times <= item.end_ms + after_ms)
        hit &= times >= item.start_ms
        if hit.any():
            ended = LOCKED if item.ended == DECAY else item.ended
            item = replace(item, ended=ended, locked=True)
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


def label_shot(
    shot: int,
    t_ms,
    n1,
    n2,
    window,
    start_ms: float | None = None,
    *,
    lock_ms=None,
    rules=RULES,
    gap_ms: float = 50.0,
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
    plasma = tearing_intervals(t_ms, n1, n2, (start, w1), rules)
    plasma = apply_locking(plasma, lock_ms)
    if m_of is not None:
        plasma = [replace(i, m=m_of(i.n, i.start_ms, i.end_ms)) for i in plasma]
    ramp = ()
    if start > w0:
        early = tearing_intervals(t_ms, n1, n2, (w0, start), rules)
        ramp = tuple(_union([(i.start_ms, i.end_ms) for i in early]))
    gaps = coverage_gaps(t_ms, n1, (w0, w1), min_ms=gap_ms)
    return ShotLabel(int(shot), (w0, w1), start, tuple(plasma), ramp, tuple(gaps))


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
    if item.locked:
        attrs["locked"] = True
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
    ramp = _minus(list(label.ramp_up), gaps)
    busy = _union([(i.start_ms, i.end_ms) for i in label.intervals] + ramp + gaps)
    rows += [_row(label.shot, NOT_OBSERVABLE, a, b) for a, b in gaps]
    rows += [_row(label.shot, UNCERTAIN, a, b) for a, b in ramp]
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
            "onset_seen",
            "m",
        ],
    )
