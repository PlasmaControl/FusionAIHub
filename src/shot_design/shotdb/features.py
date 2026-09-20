"""Per-segment features from raw signals: discharge segmentation from Ip, summary statistics,
actuator-system totals, coarse waveform shapes, and the Ip/NBI target-vs-actual outcome.

Conventions (legacy notebooks, kept on purpose): "peak" is the 95th percentile of in-window
samples (nonzero subset for actuators, so an idle beam does not drag the value to zero) and
"mean" is the mean of that same subset. EFIT quantities need >= 3 slices in a window.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from ..config import SignalSpec, SystemSpec
from ..schema import Segment
from .legacy_raw import Signal


def window(sig: Signal, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    m = (sig.t_ms >= t0) & (sig.t_ms <= t1)
    return sig.t_ms[m], sig.y[m]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(int(a), int(b - 1)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _nanmedian_filter(y: np.ndarray, k: int) -> np.ndarray:
    """Median filter that skips NaN ("not recorded", never "zero" -- see module docstring)
    instead of letting it corrupt the window the way plain scipy.ndimage.median_filter does:
    confirmed empirically that NaN's undefined ordering there makes even a 3-sample gap poison a
    window's median, well short of any deliberate tolerance.

    Edge handling: deliberately NOT scipy's mode="nearest" (pad by repeating the boundary
    sample), because that repeats whatever the boundary sample is -- including NaN, whenever the
    boundary itself is unrecorded. That silently halves the healable gap right where it matters
    most: measured, an 11-wide kernel that fully heals a 10-sample gap in the interior only
    healed a 5-sample gap at the very start or end before this fix. Instead, each of the first/
    last k//2 positions reuses the nearest k-wide window that fits entirely inside the real,
    unpadded array (shifted inward just far enough, never shrunk and never filled with a
    repeated or fabricated sample) -- so every value feeding the median was actually recorded
    somewhere in y. A NaN-free signal's segment-relevant (deep-interior) values are unaffected:
    the shift and the old edge-repeat scheme only disagree inside the first/last k//2 samples,
    which this project's reference window and segment boundaries are always many samples clear of
    (see retrieval.yaml's t_min_ms/window_ms margins) -- confirmed numerically in task-7-report.md.

    A window with at least one recorded sample still yields a real median; only a window that is
    entirely NaN stays NaN. That makes the tolerance a direct property of the kernel width k: a
    gap of fewer than k samples is always fully healed -- even one that touches the very first or
    last sample of the array -- because every position's k-wide window (shifted, not shrunk, at
    the boundary) still reaches at least one recorded sample past the open end of the gap. A gap
    of k samples or more leaves a residual unresolved wherever no such window exists, exactly as
    before.
    """
    n = y.size
    if n == 0:
        return y.copy()
    r = k // 2
    k_eff = min(k, n)  # degenerate guard: fewer samples in the whole array than the kernel width
    windows = np.lib.stride_tricks.sliding_window_view(
        y, k_eff
    )  # (n - k_eff + 1, k_eff); no padding
    lo = np.clip(
        np.arange(n) - r, 0, n - k_eff
    )  # shift the window inward at the edges, never truncate
    with warnings.catch_warnings():
        # An entirely-unrecorded window is meant to come back NaN (see above); this is numpy
        # telling us it did exactly that, not a computation to second-guess.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        return np.nanmedian(windows[lo], axis=-1)


def _smooth_ip(ip: Signal, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    t, y = window(ip, cfg["t_min_ms"], cfg["t_max_ms"])
    if t.size < 10:
        # Too few samples to say anything about the reference level -- NaN, not 0.0, so
        # ip_outcome's finite-vs-not check (see its own comment) reports the current signal as
        # unusable rather than "no plasma": a handful of unrecorded samples is not evidence the
        # shot never had one.
        return t, y, float("nan")
    dt = float(np.median(np.diff(t))) or 1.0
    k = max(1, round(cfg["median_ms"] / dt) | 1)  # odd kernel
    y = _nanmedian_filter(np.abs(y).astype(np.float64), k)
    w0, w1 = cfg["window_ms"]
    inwin = (t >= w0) & (t <= w1)
    # Unchanged on purpose: np.percentile still returns NaN if even one sample in the window is
    # NaN. Combined with the healing above, that is exactly the right split -- a gap shorter than
    # the kernel is invisible by the time it gets here (zero NaN left to propagate), while a gap
    # at least as wide as the kernel still has a genuinely unresolved patch, and *that* should
    # still make the reference level -- and everything gated on it -- unusable rather than guessed.
    ip_ref = float(np.percentile(y[inwin], 95)) if inwin.any() else 0.0
    return t, y, ip_ref


def find_segments(ip: Signal, cfg: dict) -> list[Segment]:
    """ramp_up / flat_top / ramp_down / full from |Ip|; [] when there was no plasma."""
    t, y, ip_ref = _smooth_ip(ip, cfg)
    if ip_ref < cfg["min_ip_a"]:
        return []
    runs = _runs(y >= cfg["flat_top_frac"] * ip_ref)
    if not runs:
        return []
    s, e = max(runs, key=lambda r: r[1] - r[0])
    lo = y >= cfg["plasma_frac"] * ip_ref
    first, last = int(np.argmax(lo)), int(lo.size - 1 - np.argmax(lo[::-1]))
    segs = [
        Segment(name="full", t0_ms=float(t[first]), t1_ms=float(t[last])),
        Segment(name="ramp_up", t0_ms=float(t[first]), t1_ms=float(t[s])),
        Segment(name="flat_top", t0_ms=float(t[s]), t1_ms=float(t[e])),
        Segment(name="ramp_down", t0_ms=float(t[e]), t1_ms=float(t[last])),
    ]
    return [sg for sg in segs if sg.t1_ms > sg.t0_ms]


def proxy_segments(pulse_length_s: float | None, cfg: dict) -> list[Segment]:
    """full/ramp_up/flat_top/ramp_down from PULSE-LENGTH alone, for a cluster with no Ip
    trace.

    Frontier (task F2b) has no plasma-current signal at all -- no corpus `ip` group, no
    labeler feature file -- so `find_segments` is never reachable there. The DIII-D shot
    table's PULSE-LENGTH (seconds, counted from breakdown at t=0, same as the corpus
    `xdata` origin) is the only flat-top evidence left. `cfg["proxy_ramp_up_s"]` /
    `cfg["proxy_ramp_down_s"]` (retrieval.yaml) split it the same way
    `select.PROXY_RAMP_S` does for rule (d): ramp-up, then flat top, then ramp-down, in
    ms.

    `[]` for None, a non-finite pulse length, or a pulse too short to leave any flat
    top -- exactly `find_segments`'s "no plasma" answer, so a caller cannot tell a proxy
    shot with no flat top apart from a real one by the shape of the result alone.
    """
    if pulse_length_s is None or not np.isfinite(pulse_length_s):
        return []
    ramp_up_s, ramp_down_s = cfg["proxy_ramp_up_s"], cfg["proxy_ramp_down_s"]
    if pulse_length_s - ramp_up_s - ramp_down_s <= 0:
        return []
    pulse_ms = pulse_length_s * 1000.0
    ramp_up_ms = ramp_up_s * 1000.0
    ramp_down_ms = ramp_down_s * 1000.0
    return [
        Segment(name="full", t0_ms=0.0, t1_ms=pulse_ms),
        Segment(name="ramp_up", t0_ms=0.0, t1_ms=ramp_up_ms),
        Segment(name="flat_top", t0_ms=ramp_up_ms, t1_ms=pulse_ms - ramp_down_ms),
        Segment(name="ramp_down", t0_ms=pulse_ms - ramp_down_ms, t1_ms=pulse_ms),
    ]


def _subset(y: np.ndarray, actuator: bool) -> np.ndarray:
    return y[y > 0] if actuator else y


def _finite(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A NaN sample is the raw reader's mark for "not recorded" (see module docstring), never a
    real reading of zero -- drop it before any statistic sees it, the same way the peak
    convention already drops an actuator's non-positive samples. t and y are dropped together so
    slope keeps its (time, value) pairing."""
    # float64 before any arithmetic: the raw store is float32, and D-alpha runs at ~1e15-1e18
    # ph/cm^2/sr/s, so a float32 sum of squares overflows -- the first 200-shot build wrote
    # dalpha_std = inf for 5 segments and numpy warned "overflow encountered in reduce".
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    return t[ok], y[ok]


def _value(x) -> float | None:
    """A statistic is a finite float or None -- never inf, which pandas/Parquet would store and
    the PCA would then have to guard against."""
    v = float(x)
    return v if np.isfinite(v) else None


def stat_mean(t: np.ndarray, y: np.ndarray, actuator: bool) -> float | None:
    _, y = _finite(t, y)
    v = _subset(y, actuator)
    return _value(np.mean(v)) if v.size else None


def stat_peak(t: np.ndarray, y: np.ndarray, actuator: bool) -> float | None:
    _, y = _finite(t, y)
    v = _subset(y, actuator)
    return _value(np.percentile(v, 95)) if v.size else None


def stat_min(t: np.ndarray, y: np.ndarray, actuator: bool) -> float | None:
    _, y = _finite(t, y)
    return _value(np.min(y)) if y.size else None


def stat_std(t: np.ndarray, y: np.ndarray, actuator: bool) -> float | None:
    _, y = _finite(t, y)
    # Overflow means an unavailable statistic; keep every other warning visible.
    with np.errstate(over="ignore", invalid="ignore"):
        return _value(np.std(y)) if y.size > 1 else None


def stat_on_frac(t: np.ndarray, y: np.ndarray, actuator: bool) -> float | None:
    # stat_peak alone can't tell these apart -- it returns None both when nothing in the window
    # was recorded at all and when an actuator's recorded samples never rose above zero -- but
    # on_frac must: an idle gyrotron/beam/valve (recorded, and reading zero) is a confident "never
    # on" (0.0), while a channel with no finite sample at all is genuinely unknown (None). Check
    # recordedness directly first, then let peak (still legitimately undefined for an idle
    # actuator) decide only between those two remaining cases.
    _, y = _finite(t, y)
    if y.size == 0:
        return None
    v = _subset(y, actuator)
    if v.size == 0:
        return 0.0
    peak = float(np.percentile(v, 95))
    return float(np.mean(y > 0.05 * peak))


def stat_slope(t: np.ndarray, y: np.ndarray, actuator: bool) -> float | None:
    t, y = _finite(t, y)
    if t.size < 3 or t[-1] == t[0]:
        return None
    return _value(np.polyfit(t / 1000.0, y, 1)[0])  # per second


STATS: dict[str, Callable[[np.ndarray, np.ndarray, bool], float | None]] = {
    "mean": stat_mean,
    "peak": stat_peak,
    "min": stat_min,
    "std": stat_std,
    "on_frac": stat_on_frac,
    "slope": stat_slope,
}


def segment_scalars(
    signals: dict[str, Signal | None],
    seg: Segment,
    specs: list[SignalSpec],
    min_samples: int = 3,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    raw: dict[str, float | None] = {}
    derived: dict[str, float | None] = {}
    for spec in specs:
        target = derived if spec.tier == "derived" else raw
        sig = signals.get(spec.name)
        for stat in spec.stats:
            key = f"{spec.name}_{stat}"
            if sig is None:
                target[key] = None
                continue
            t, y = window(sig, seg.t0_ms, seg.t1_ms)
            n_recorded = int(np.isfinite(y).sum())  # unrecorded (NaN) samples are not data
            target[key] = (
                STATS[stat](t, y, spec.system is not None) if n_recorded >= min_samples else None
            )
    return raw, derived


def system_totals(
    signals: dict[str, Signal | None], systems: dict[str, SystemSpec]
) -> dict[str, Signal]:
    """Sum the members of each actuator system onto the first member's time grid.

    A member's NaN sample means it was not recorded there, not that it was off (confirmed against
    a real gyrotron column that is 99%+ NaN while genuinely installed -- see module docstring). So
    at each sample the total is the sum of only the members that actually recorded a value there:
    an unrecorded member contributes nothing and must not turn the whole system's total NaN. Only
    when *no* member recorded anything at a sample does the total itself become NaN, rather than a
    zero that would misreport "nobody was watching" as "the system was off".
    """
    out: dict[str, Signal] = {}
    for sysdef in systems.values():
        members = [signals.get(f"{sysdef.prefix}_{m}") for m in sysdef.members]
        members = [m for m in members if m is not None]
        if not members:
            continue
        ref = members[0]
        n = ref.t_ms.size
        # Accumulate one member at a time (a running sum plus a running recorded-count) instead
        # of stacking every member into an (n_members, n_samples) array first: measured on the
        # real 8-beam NBI grid (1,310,001 samples), stacking peaked around 189 MB against ~22 MB
        # here, and a full database build runs this once per shot per system.
        total = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.int32)
        for m in members:
            if m.t_ms.size == n and np.array_equal(m.t_ms, ref.t_ms):
                y = m.y  # already the reference grid; read-only below, so no copy needed
            else:
                # Outside its own recorded time span a member has no opinion at all -- extrapolate
                # with NaN, not the previous 0.0, which read as a real "this member is off"
                # reading there and could single-handedly turn a sample that truly nobody
                # recorded into a confident (and wrong) zero.
                y = np.interp(ref.t_ms, m.t_ms, m.y, left=np.nan, right=np.nan)
            recorded = np.isfinite(y)
            total += np.where(recorded, y, 0.0)  # one array, not the ~3 a boolean-indexed += makes
            count += recorded
        total[count == 0] = np.nan
        out[f"{sysdef.prefix}_total"] = Signal(
            ref.t_ms, total.astype(np.float32), ref.units, ref.source, ref.group, "total"
        )
    return out


def waveform_shape(sig: Signal | None, seg: Segment, n: int = 20) -> np.ndarray:
    """Coarse shape of a signal over a segment (n points), normalized by the maximum absolute
    value onto [-1, 1] -- not [0, 1], since a bipolar signal keeps its sign. NaN means a grid
    point with nothing recorded nearby (including every point, when the signal is absent or the
    segment recorded nothing at all); a fully recorded but identically zero trace (an idle
    gyrotron/beam/valve) instead comes back all zeros, its own real normalized shape -- the two
    must not collapse into each other."""
    if sig is None:
        return np.full(n, np.nan, dtype=np.float32)
    t, y = window(sig, seg.t0_ms, seg.t1_ms)
    if t.size < 2:
        return np.full(n, np.nan, dtype=np.float32)
    grid = np.linspace(seg.t0_ms, seg.t1_ms, n)
    v = np.interp(grid, t, y.astype(np.float64))
    # An unrecorded (NaN) source sample can leave some interpolated grid points NaN too; scale by
    # what was actually recorded instead of letting one gap turn the whole scale NaN and fall
    # through to raw physical-unit magnitudes (~1e6) sitting next to every other, normalized,
    # entry in the feature vector.
    finite = np.isfinite(v)
    if not finite.any():
        return np.full(n, np.nan, dtype=np.float32)
    scale = float(np.max(np.abs(v[finite])))
    if scale == 0.0:
        # Every recorded point in this segment reads exactly zero -- a real, idle waveform, not
        # an absent one. v's finite entries are already 0.0 (and any still-NaN entries are
        # genuinely unrecorded grid points), so return it as is rather than compute 0/0.
        return v.astype(np.float32)
    return (v / scale).astype(np.float32)


def ip_outcome(ip: Signal, ip_target: Signal | None, segs: list[Segment], cfg: dict) -> dict:
    """Did the plasma current follow the PCS program, and how did the discharge end?"""
    out: dict = {
        "ip_target_err": None,
        "ip_target_hit": None,
        "flat_top_ms": None,
        "ended_early": None,
        "fast_quench": None,
        "end_reason": None,
    }
    flat = next((s for s in segs if s.name == "flat_top"), None)
    if flat is None:
        # find_segments also returns [] when the smoothed reference level is NaN -- a NaN run in
        # Ip too wide for _nanmedian_filter to heal (at or beyond its kernel width) still leaves a
        # residual patch that poisons the reference level's percentile -- because every
        # `y >= frac * ip_ref` comparison is then False, the same empty result as a shot that
        # truly never had plasma. Tell the two apart: a NaN reference means the current signal
        # itself was unusable, not that there was no plasma to measure.
        _, _, ip_ref = _smooth_ip(ip, cfg)
        out["end_reason"] = "no_plasma" if np.isfinite(ip_ref) else "ip_signal_unusable"
        return out
    # "full" is built from the same first/last plasma-detection indices as "flat_top" in
    # find_segments and survives the same t1 > t0 filter, so it is always present alongside
    # flat_top; the fallback to `flat` only guards a future change to find_segments.
    full = next((s for s in segs if s.name == "full"), flat)
    t, y, ip_ref = _smooth_ip(ip, cfg)
    out["flat_top_ms"] = flat.t1_ms - flat.t0_ms
    if ip_target is not None:
        _, meas = window(ip, flat.t0_ms, flat.t1_ms)
        _, tgt = window(ip_target, flat.t0_ms, flat.t1_ms)
        # Recorded samples only (see module docstring): np.mean is NaN-propagating, so a single
        # unrecorded sample in either trace would otherwise turn err into NaN, and
        # `abs(NaN) <= tolerance` is always False -- a confident "missed" for a shot we actually
        # know nothing about. meas/tgt.size below then naturally means "recorded samples", so a
        # window with none leaves ip_target_err/hit at their None ("unknown") defaults instead of
        # guessing.
        meas, tgt = meas[np.isfinite(meas)], tgt[np.isfinite(tgt)]
        if meas.size and tgt.size and float(np.mean(tgt)) > 0:
            err = float(np.mean(meas)) / float(np.mean(tgt)) - 1.0
            out["ip_target_err"], out["ip_target_hit"] = err, abs(err) <= cfg["target_tolerance"]
    # fast quench: from above 50% to below 10% of ip_ref within quench_ms
    hi_idx = np.flatnonzero(y > 0.5 * ip_ref)
    lo_after = np.flatnonzero(y < 0.1 * ip_ref)
    fast: bool | None = False
    if hi_idx.size:
        last_hi = hi_idx[-1]
        # Both comparisons above are False for every NaN sample, the same shape as the
        # target-comparison issue below: a residual gap (wider than _nanmedian_filter's kernel,
        # left unhealed) spanning the *entire* region after the last above-50% sample would
        # otherwise read as a confident "never dropped below 10%" -- "no quench" -- even though
        # that region was never actually observed. Only when at least one sample past last_hi was
        # actually recorded does "no quench" mean anything.
        if np.isfinite(y[last_hi + 1 :]).any():
            nxt = lo_after[lo_after > last_hi]
            fast = bool(nxt.size and (t[nxt[0]] - t[last_hi]) <= cfg["quench_ms"])
        else:
            fast = None
    out["fast_quench"] = fast
    # None ("unknown"), not a confident False: an absent target signal (ip_target is None --
    # this channel was simply never fetched for this shot, a normal and expected data-coverage
    # gap, see legacy_raw.read_signal) is no evidence the discharge ended normally. Overwritten below
    # whenever the target signal is actually present -- including the honest None already reached
    # when it's present but the post-shot window itself has zero recorded samples.
    ended_early: bool | None = None
    if ip_target is not None:
        # Anchored on "full"'s end (Ip itself already below plasma_frac -- genuinely collapsed),
        # not "flat_top"'s end (only the 85% crossing, i.e. the *start* of a decline that may be
        # a perfectly normal multi-hundred-ms programmed ramp-down). Using flat_top here would
        # check the target only an instant after Ip begins its programmed fall, when the target
        # is -- by construction of a well-tracked shot -- still near its own high value, so a
        # deliberate, on-schedule rampdown would be misread as "ended early" every time.
        _, ty = window(ip_target, full.t1_ms, full.t1_ms + 500.0)
        # Recorded samples only, same reasoning as the flat-top error above: np.max is also
        # NaN-propagating, so one unrecorded sample here used to make "> 0.5*ip_ref" silently
        # False -- a confident "did not end early" rather than the honest "can't tell" a window
        # with zero recorded samples actually is.
        ty = ty[np.isfinite(ty)]
        ended_early = bool(np.max(ty) > 0.5 * ip_ref) if ty.size else None
    # fast is decisive on its own (it needs only Ip, never the target signal): it can confirm an
    # early end outright, but its being False cannot rule one out when the target-based check
    # above came back unknown -- that stays unknown too, rather than defaulting to a confident
    # False the way `ended_early or fast` (ordinary two-valued "or") would have.
    out["ended_early"] = True if fast else ended_early
    out["end_reason"] = (
        "fast_current_quench"
        if fast
        else "early_termination"
        if ended_early
        else "target_signal_unusable"
        if ended_early is None
        else "programmed_rampdown"
    )
    return out


def infer_nbi_target_mw(
    target_peak_raw: float | None, actual_peak_w: float | None
) -> tuple[float | None, str | None]:
    """PTDATA BMSPINJ is MW on most shots and kW on a minority. Infer the unit from the ratio to the
    measured total NBI power (PABS/PINJF are reliably W); the decade between the two bands is left
    unscored rather than guessed."""
    if not target_peak_raw or not actual_peak_w or target_peak_raw <= 0 or actual_peak_w <= 0:
        return None, None
    log_ratio = float(np.log10(actual_peak_w / target_peak_raw))
    if 5.0 < log_ratio <= 7.0:
        return float(target_peak_raw), "MW"
    if 2.0 <= log_ratio <= 4.0:
        return float(target_peak_raw) * 1e-3, "kW"
    return None, None
