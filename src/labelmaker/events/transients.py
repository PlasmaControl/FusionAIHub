"""A transient mask's column activity -> ELM times, an ELM clock, quiet time.

`masks.py` says WHERE the network saw transient activity. On a tokamak that
is very largely ELMs, and an ELM is not a shape in the time-frequency plane
the way a mode is - it is a MOMENT: a narrow vertical stripe that lights
most of the 512 frequency rows at once, because a filament crossing the
separatrix is broadband. So everything here works on ONE number per column,
the fraction of rows that are lit, and never looks at the mask again. That
number is `masks.block_arrays`' `_col_act` key, computed at the same
threshold and pinned equal to `column_activity` by a test, which is why
`transients_for_block` can read a masks file without unpacking anything.

**Two readings of that trace, because a burst and an ELM are different
questions.** `extract_bursts` is TokEye's
`elmspec/events.py::extract_elm_events`, ported: threshold the activity at
`ACTIVITY_MIN`, close quiet gaps of up to `MIN_GAP_COLS` columns, and every
remaining run is a burst. That answers "how long was the plasma disturbed".
`elm_events` is elmcycle's `elms.py::detect_elms`, parametrised the way it
is there: smooth the trace over `SMOOTH_MS` and take `scipy.signal
.find_peaks` with a `PROMINENCE` and a `MIN_DISTANCE_MS`. That answers
"when did each one happen". One burst with two peaks in it is one
disturbance and two ELMs, and both facts are wanted - the event rows carry
the peaks, and the burst a peak sits in is where its confidence comes from.
The thresholds are ours; only the two shapes of logic are the references'.

**The clock, because what matters about an ELM is usually the one before.**
Almost every quantity a consumer asks of an ELMy phase is relative to the
cycle rather than to the shot clock: a pedestal recovers over a fraction of
the inter-ELM interval, and a window feature computed two thirds of the way
through one is not comparable with one computed just after a crash. So
`elm_clock` returns, per column, the rate, the time since the last ELM, the
time to the next and the PHASE - the fraction of the current interval
elapsed - and the L9 script stores those four arrays beside the mask rather
than as events, because an array per column is not a discrete happening and
the events table is for discrete happenings.

**ELM-free is a claim about a RATE, not about a silence.** A QH-mode or an
EHO phase is defined by the ELMs having stopped, and "stopped" has to be
measurable on a detector that misses one now and then; `elm_free_intervals`
therefore asks where the rate counted in a `RATE_WINDOW_S` window stays at
or under `ELM_FREE_MAX_RATE_HZ`, not where the trace is flat. With the
defaults - 5 Hz in 100 ms - that is "no ELM within 50 ms either way", so
each ELM costs the record the 100 ms window around it and a gap of `g`
seconds yields an interval of `g - 0.1`. These intervals are what later
gates the QH/EHO candidates: an EHO inside an ELMy phase is a mode with a
different name.

Nothing here decides that a burst IS an ELM. The mask is a transient-
activity detector, a sawtooth crash and a disruption precursor are transient
too, and the `phenomenon="elm"` on an event row is a claim by
`source="tokeye_transient"` that a consumer can weigh against any other.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from ..ae.labels import N_BINS, PROB_THRESHOLD
from .coverage import clip_point_to_coverage, clipped_attrs
from .masks import read_mask
from .schema import Event

#: Probability a row counts as transient at. NOT ours: it is the threshold
#: `masks.block_arrays` binarised the stored mask at, so `column_activity`
#: of a probability map is the `_col_act` the file already holds.
ACTIVITY_THR = PROB_THRESHOLD
#: Fraction of the 512 rows a column needs before it is part of a burst.
#: TokEye's number: an ELM lights most of the spectrum, and 10% of the rows
#: is already far more than the salt a threshold leaves behind.
ACTIVITY_MIN = 0.1
#: Quiet columns two halves of one burst may be apart. An ELM crash is not
#: one column wide and the network's probability dips inside it.
MIN_GAP_COLS = 3
#: Boxcar the activity is smoothed over before peaks are taken, in ms.
#: elmcycle's, and small on purpose - 0.64 ms is under a twentieth of the
#: 15.36 ms ELM period the spec pins, so it removes single-column noise
#: without moving a crash.
SMOOTH_MS = 0.64
#: Prominence a smoothed peak needs to be an ELM. On a trace bounded by 0
#: and 1 this is 3% of full scale.
PROMINENCE = 0.03
#: How close two ELMs may be. 3 ms is a fifth of the pinned period; two
#: peaks nearer than that are one crash seen twice.
MIN_DISTANCE_MS = 3.0
#: Window the ELM rate is counted in, centred on the sample. 100 ms holds
#: six and a half of the pinned periods - long enough that missing one ELM
#: moves the rate by a sixth rather than by half, short enough to see a
#: phase change.
RATE_WINDOW_S = 0.1
#: A rate at or under this is not an ELMy phase. With `RATE_WINDOW_S` it
#: means "no ELM within 50 ms either way": half a count in the window.
ELM_FREE_MAX_RATE_HZ = 5.0
#: An ELM-free interval shorter than this is a missed ELM, not a phase.
ELM_FREE_MIN_S = 0.05

#: Who claims these events, and what they are claimed to be. The ELMs come
#: from the mask; the quiet comes from the clock built on top of it, and a
#: consumer that trusts one and not the other can tell them apart.
SOURCE = "tokeye_transient"
PHENOMENON = "elm"
FREE_SOURCE = "elm_clock"
FREE_PHENOMENON = "elm_free"


# -------------------------------------------------------- column activity

def column_activity(tra_prob, *, thr: float = ACTIVITY_THR) -> np.ndarray:
    """`(512, T)` transient probabilities -> `(T,)` lit fraction per column.

    The whole of an ELM detector's input. Written the way
    `masks.block_arrays` writes `_col_act` - `(prob >= thr).mean(axis=0)`
    in float32 - because a masks file stores that array and this module
    must not disagree with it by a rounding.
    """
    prob = np.asarray(tra_prob)
    if prob.ndim != 2 or prob.shape[0] != N_BINS:
        raise ValueError(
            f"expected a ({N_BINS}, T) transient map, got {prob.shape}"
        )
    return (prob >= float(thr)).mean(axis=0).astype(np.float32)


# ------------------------------------------------------------------ bursts

@dataclass(frozen=True)
class Burst:
    """One run of columns the transient channel is lit across.

    Half-open `[col0, col1)`, like every extent in this package; TokEye's
    `ElmEvent` is inclusive, so the port is here and the convention is ours.
    `peak_activity` is the largest lit fraction inside the run, which is
    what an ELM event's confidence is.
    """

    col0: int
    col1: int
    peak_activity: float

    @property
    def n_cols(self) -> int:
        return self.col1 - self.col0


def _runs(active) -> list[tuple[int, int]]:
    """Half-open `[start, stop)` of every True run.

    `tokeye.elmspec.events._contiguous_runs`, whose pairs are inclusive,
    with the `- 1` taken off the ends.
    """
    padded = np.concatenate(([False], np.asarray(active, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _fill_gaps(active, max_gap: int) -> np.ndarray:
    """Close quiet gaps of at most `max_gap` columns between True runs.

    `tokeye.elmspec.events._fill_gaps` on half-open runs: its
    `next_start - prev_end - 1` is this one's `next_start - prev_stop`.
    """
    active = np.asarray(active, dtype=bool)
    if max_gap <= 0:
        return active
    filled = active.copy()
    runs = _runs(active)
    for (_, prev_stop), (next_start, _) in itertools.pairwise(runs):
        if next_start - prev_stop <= max_gap:
            filled[prev_stop:next_start] = True
    return filled


def extract_bursts(
    activity,
    *,
    activity_min: float = ACTIVITY_MIN,
    min_gap_cols: int = MIN_GAP_COLS,
    min_duration_cols: int = 1,
) -> list[Burst]:
    """Column activity -> the runs of it that are lit. TokEye's, ported.

    Threshold at `activity_min`, close gaps of at most `min_gap_cols`
    quiet columns, and keep every run of at least `min_duration_cols`.
    This is the EXTENT of a disturbance; `elm_events` is its timing, and
    the two are deliberately separate - a burst that holds two peaks is one
    burst and two ELMs.
    """
    activity = np.asarray(activity, dtype=np.float64)
    if activity.ndim != 1:
        raise ValueError(f"expected one activity trace, got {activity.shape}")
    active = _fill_gaps(activity >= float(activity_min), int(min_gap_cols))
    return [
        Burst(col0=col0, col1=col1, peak_activity=float(activity[col0:col1].max()))
        for col0, col1 in _runs(active)
        if col1 - col0 >= int(min_duration_cols)
    ]


# --------------------------------------------------------------- ELM times

def _frame_s(t_s) -> float:
    """Seconds one column lasts, from the column grid itself."""
    t = np.asarray(t_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"a column grid needs at least two columns, got {t.shape}")
    frame = float(t[1] - t[0])
    if not frame > 0.0:
        raise ValueError(f"column times must increase; got {t[0]} then {t[1]}")
    return frame


def smooth_activity(activity, t_s, *, smooth_ms: float = SMOOTH_MS) -> np.ndarray:
    """The activity trace, boxcar-smoothed over `smooth_ms` of it.

    elmcycle's smoothing, width and all: `max(1, round(smooth_ms /
    frame_ms))` columns of `uniform_filter1d`. Public because the
    confidence of an ELM that fell outside every burst is this trace under
    the peak, and a caller should not have to reproduce the width.
    """
    activity = np.asarray(activity, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    if activity.shape != t.shape:
        raise ValueError(
            f"activity is {activity.shape}, its column times are {t.shape}"
        )
    # `round`, and banker's rounding at that: the wide pass' 0.64 / 0.256 is
    # exactly 2.5 and rounds to 2, an EVEN width, which `uniform_filter1d`
    # centres asymmetrically (one column back, two forward). That is
    # elmcycle's own arithmetic and the width every threshold here was set
    # against - do not "fix" it to 3 without re-measuring `PROMINENCE`.
    width = max(1, round(float(smooth_ms) / (_frame_s(t) * 1e3)))
    return uniform_filter1d(activity, size=width)


def elm_events(
    activity,
    t_s,
    *,
    smooth_ms: float = SMOOTH_MS,
    prominence: float = PROMINENCE,
    min_distance_ms: float = MIN_DISTANCE_MS,
) -> np.ndarray:
    """Column activity -> the times, in seconds, that ELMs happened.

    elmcycle's `detect_elms`: smooth, then `find_peaks` with a prominence
    and a minimum separation. A peak's time is its COLUMN's time, so every
    ELM lands on the mask's own grid and is exactly as precise as the
    column is (0.256 ms on the wide pass) - no sub-column interpolation,
    because the activity in a column is a count of rows and interpolating a
    count would invent a precision the detector does not have.

    elmcycle's `height` is not used: it defaults there to the trace's
    median plus 0.05, which is a decision about the shot's baseline that
    `prominence` already makes locally and better.
    """
    t = np.asarray(t_s, dtype=np.float64)
    smoothed = smooth_activity(activity, t, smooth_ms=smooth_ms)
    distance = max(1, math.ceil(float(min_distance_ms) / (_frame_s(t) * 1e3)))
    idx, _ = find_peaks(smoothed, prominence=float(prominence), distance=distance)
    return t[idx]


# --------------------------------------------------------------- the clock

def _sorted_times(elm_times_s) -> np.ndarray:
    """The ELM times as a sorted 1-D float64 array."""
    elm = np.asarray(elm_times_s, dtype=np.float64).ravel()
    if elm.size and not np.isfinite(elm).all():
        raise ValueError("ELM times must be finite")
    return np.sort(elm)


def _count_in_window(elm: np.ndarray, at, window_s: float) -> np.ndarray:
    """How many ELMs lie in `[t - w/2, t + w/2)` around each of `at`.

    Half-open on the same side as every other interval here, so an ELM
    exactly `w/2` before a sample counts and one exactly `w/2` after does
    not. The one definition of "the rate"; `elm_clock` samples it on the
    column grid and `elm_free_intervals` solves it in closed form.
    """
    at = np.asarray(at, dtype=np.float64)
    half = float(window_s) / 2.0
    lo = np.searchsorted(elm, at - half, side="left")
    hi = np.searchsorted(elm, at + half, side="left")
    return (hi - lo).astype(np.int64)


def elm_clock(
    elm_times_s, t_s, *, window_s: float = RATE_WINDOW_S
) -> dict[str, np.ndarray]:
    """ELM times -> the four per-column arrays that place a sample in a cycle.

    * `rate_hz` - ELMs counted in a `window_s` window centred on the sample,
      divided by the window. It is therefore QUANTISED to multiples of
      `1 / window_s`: a 65 Hz train gives 60 or 70 Hz, alternating, and 65
      on average. That is what a count is, and a smoother estimator would be
      a model of the rate rather than a measurement of it.
    * `time_since_last_s` - to the last ELM at or before the sample; NaN
      before the first.
    * `time_to_next_s` - to the first ELM strictly after; NaN from the last
      ELM on.
    * `phase` - `time_since_last / (time_since_last + time_to_next)`, the
      fraction of the CURRENT inter-ELM interval that has elapsed. Defined
      on `[first, last)` and NaN outside it: the last ELM opens an interval
      that never closes, and a fraction of an unfinished interval is not a
      number anybody should be given.

    Every array is float64 and as long as `t_s`. They are returned rather
    than stored: an array per column is not a discrete happening, and the
    events table is for discrete happenings.
    """
    elm = _sorted_times(elm_times_s)
    t = np.asarray(t_s, dtype=np.float64)
    nan = np.full(t.shape, np.nan, dtype=np.float64)
    if elm.size == 0:
        return {
            "rate_hz": np.zeros(t.shape, dtype=np.float64),
            "time_since_last_s": nan.copy(),
            "time_to_next_s": nan.copy(),
            "phase": nan.copy(),
        }
    rate = _count_in_window(elm, t, window_s).astype(np.float64) / float(window_s)
    nxt = np.searchsorted(elm, t, side="right")
    prev = nxt - 1
    since = np.where(prev >= 0, t - elm[np.clip(prev, 0, None)], np.nan)
    to_next = np.where(
        nxt < elm.size, elm[np.clip(nxt, None, elm.size - 1)] - t, np.nan
    )
    period = since + to_next
    phase = nan.copy()
    known = np.isfinite(period) & (period > 0.0)
    phase[known] = since[known] / period[known]
    return {
        "rate_hz": rate,
        "time_since_last_s": since,
        "time_to_next_s": to_next,
        "phase": phase,
    }


# ------------------------------------------------------ ELM-free intervals

def _rate_breakpoints(elm: np.ndarray, t0: float, t1: float,
                      window_s: float) -> np.ndarray:
    """Every time in `[t0, t1]` at which the windowed count can change.

    An ELM at `T` is counted for `t` in `(T - w/2, T + w/2]`, so the count
    is a step function whose steps are exactly these edges. Between two of
    them it is constant, which is what lets `elm_free_intervals` be exact
    rather than sampled on some grid the caller happens to have.
    """
    half = float(window_s) / 2.0
    edges = np.concatenate(([t0, t1], elm - half, elm + half))
    return np.unique(np.clip(edges, t0, t1))


def elm_free_intervals(
    elm_times_s,
    t_cov,
    *,
    max_rate_hz: float = ELM_FREE_MAX_RATE_HZ,
    min_duration_s: float = ELM_FREE_MIN_S,
    window_s: float = RATE_WINDOW_S,
) -> np.ndarray:
    """The maximal `[t0, t1)` of `t_cov` in which the ELM rate stays low.

    "Low" is `elm_clock`'s rate at or under `max_rate_hz`, which is a count
    of at most `max_rate_hz * window_s` ELMs in the window - with the
    defaults, at most half an ELM, i.e. none: an ELM at `T` disqualifies
    `(T - 50 ms, T + 50 ms]`, and a gap of `g` seconds in the train yields
    an interval of `g - window_s`. Intervals shorter than
    `min_duration_s` are dropped, because one missed ELM in an ELMy phase
    would otherwise be reported as a quiet phase 20 ms long.

    Solved on the count's own breakpoints rather than on any column grid,
    so the answer does not depend on a resolution nobody passed in, and
    clipped to the coverage window - "nobody looked here" is not "it was
    quiet here". `(0, 2)` when there is no such interval.
    """
    t0, t1 = float(t_cov[0]), float(t_cov[1])
    empty = np.zeros((0, 2), dtype=np.float64)
    if not t1 > t0:
        return empty
    elm = _sorted_times(elm_times_s)
    edges = _rate_breakpoints(elm, t0, t1, window_s)
    if edges.size < 2:
        return empty
    mid = 0.5 * (edges[:-1] + edges[1:])
    quiet = _count_in_window(elm, mid, window_s) <= float(max_rate_hz) * float(
        window_s
    )
    out: list[list[float]] = []
    for is_quiet, lo, hi in zip(quiet, edges[:-1], edges[1:], strict=True):
        if not is_quiet:
            continue
        if out and out[-1][1] == lo:
            out[-1][1] = float(hi)
        else:
            out.append([float(lo), float(hi)])
    kept = [pair for pair in out if pair[1] - pair[0] >= float(min_duration_s)]
    return np.array(kept, dtype=np.float64).reshape(-1, 2) if kept else empty


# ------------------------------------------------------------------ events

def _nearest_col(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The index of the column nearest each of `values`.

    ELM times come off `t_s` in the first place, so this is usually a
    lookup; it is written as a nearest-neighbour search because a caller
    may pass times a hand edit or another detector produced.
    """
    idx = np.clip(np.searchsorted(t, values), 1, t.size - 1)
    left = idx - 1
    return np.where(values - t[left] <= t[idx] - values, left, idx)


def transients_to_events(
    elm_times_s,
    bursts: Sequence[Burst],
    *,
    shot: int,
    diag: str,
    channel: int,
    pass_name: str,
    t_s,
    t_cov: tuple[float, float],
    unet_sha256: str,
    activity,
    smooth_ms: float = SMOOTH_MS,
    max_rate_hz: float = ELM_FREE_MAX_RATE_HZ,
    min_duration_s: float = ELM_FREE_MIN_S,
    window_s: float = RATE_WINDOW_S,
) -> list[Event]:
    """ELMs and quiet -> event rows: a point per ELM, an interval per lull.

    Two claims of two different kinds, so two sources. An ELM is a
    `detector` claim by `tokeye_transient` and a POINT event
    (`t1_s == t0_s`), with no frequency band at all - an ELM is broadband
    by definition and a band on it would be an artefact of where the
    spectrogram stops. Its confidence is the peak activity of the burst it
    falls inside, which is the fraction of the spectrum that lit up; an ELM
    that fell outside every burst - a peak too faint to reach
    `ACTIVITY_MIN` - is worth the smoothed trace under it instead, and NaN
    if the trace was not passed. `activity` has no default because that
    fallback is the exception: a caller reaches it by passing `None` and
    saying so, not by forgetting an argument.

    An ELM-free interval is a `heuristic` claim by `elm_clock`, derived
    from the same times by `elm_free_intervals`, and carries no confidence:
    it is arithmetic on the ELM list, and its uncertainty is the ELM list's.
    Its `attrs` record the threshold and window it was cut at, the highest
    rate actually reached inside it, and how many ELMs it contains - which
    is zero unless the caller relaxed `max_rate_hz`.

    An ELM whose column lies outside `t_cov` - the stitched transform's
    first and last columns do (`masks.COL_ORIGIN`) - is written AT the
    bound with `attrs["clipped"] = true`; see
    `coverage.clip_point_to_coverage` for why that and not a refusal.
    """
    elm = _sorted_times(elm_times_s)
    t = np.asarray(t_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError(f"a column grid needs at least two columns, got {t.shape}")
    smoothed = (
        None if activity is None
        else smooth_activity(activity, t, smooth_ms=smooth_ms)
    )
    cov = (float(t_cov[0]), float(t_cov[1]))
    common: dict[str, Any] = {
        "shot": int(shot),
        "diag": str(diag),
        "channel": int(channel),
        "pass_name": str(pass_name),
        "t_cov0_s": cov[0],
        "t_cov1_s": cov[1],
    }

    cols = _nearest_col(t, elm) if elm.size else np.zeros(0, dtype=np.int64)
    out: list[Event] = []
    for time, col in zip(elm.tolist(), cols.tolist(), strict=True):
        holder = next(
            (b for b in bursts if b.col0 <= col < b.col1), None
        )
        if holder is not None:
            confidence = float(holder.peak_activity)
        elif smoothed is not None:
            confidence = float(smoothed[col])
        else:
            confidence = math.nan
        # A column centre outside the record it was computed from is the
        # transform's edge padding (`masks.COL_ORIGIN`), not an ELM that
        # outlived the digitiser: the point is moved onto the bound and the
        # row says so. `col` still names the column the peak was found in,
        # and the clock's own arithmetic below runs on the measured times.
        at, clipped = clip_point_to_coverage(time, cov)
        out.append(
            Event(
                source=SOURCE,
                evidence_kind="detector",
                phenomenon=PHENOMENON,
                t0_s=at,
                t1_s=at,
                confidence=confidence,
                attrs=clipped_attrs({
                    "col": int(col),
                    "burst_col0": None if holder is None else int(holder.col0),
                    "burst_cols": None if holder is None else int(holder.n_cols),
                    "unet_sha256": str(unet_sha256),
                }, clipped),
                **common,
            )
        )

    free = elm_free_intervals(
        elm, cov, max_rate_hz=max_rate_hz,
        min_duration_s=min_duration_s, window_s=window_s,
    )
    for lo, hi in free.tolist():
        inside = elm[(elm >= lo) & (elm < hi)]
        edges = _rate_breakpoints(elm, lo, hi, window_s)
        mid = 0.5 * (edges[:-1] + edges[1:])
        peak = _count_in_window(elm, mid, window_s).max() if mid.size else 0
        out.append(
            Event(
                source=FREE_SOURCE,
                evidence_kind="heuristic",
                phenomenon=FREE_PHENOMENON,
                t0_s=float(lo),
                t1_s=float(hi),
                attrs={
                    "duration_s": float(hi - lo),
                    "max_rate_hz": float(max_rate_hz),
                    "max_rate_inside_hz": float(peak) / float(window_s),
                    "n_elms_inside": int(inside.size),
                    "window_s": float(window_s),
                    "unet_sha256": str(unet_sha256),
                },
                **common,
            )
        )
    return out


# ---------------------------------------------------- one stored block

@dataclass(eq=False)
class Transients:
    """Everything the transient channel of one block says, in one object.

    `eq=False` for the reason `tracks.Component` has it: this holds arrays,
    and a generated `__eq__` would compare them elementwise and raise on
    the truth value of the result.
    """

    activity: np.ndarray
    t_s: np.ndarray
    bursts: list[Burst]
    elm_times_s: np.ndarray
    clock: dict[str, np.ndarray]
    elm_free_s: np.ndarray
    t_cov: tuple[float, float]


def transients_for_block(
    block_prefix: str,
    masks_path,
    *,
    activity_min: float = ACTIVITY_MIN,
    min_gap_cols: int = MIN_GAP_COLS,
    smooth_ms: float = SMOOTH_MS,
    prominence: float = PROMINENCE,
    min_distance_ms: float = MIN_DISTANCE_MS,
    max_rate_hz: float = ELM_FREE_MAX_RATE_HZ,
    min_duration_s: float = ELM_FREE_MIN_S,
    window_s: float = RATE_WINDOW_S,
) -> Transients:
    """One stored `(diag, channel, pass)` block -> its transient reading.

    Two keys, and no unpacking: `_col_act` IS `column_activity` of the
    transient map, computed at the file's own threshold before the mask was
    packed, so the ELM path costs a masks file two small arrays where
    `tracks.tracks_for_block` costs it a whole `(512, T)` unpack. `t_cov`
    is the block's own column span - the coverage of the diagnostic that
    was looked at, which is what tells "no ELMs here" from "nobody looked" -
    and it runs half a column either side of the first and last column
    CENTRE, because a column is a span of record and not an instant.

    That equality is checked rather than assumed: `_meta["thr"]` must be
    `ACTIVITY_THR`, or the stored trace is `column_activity` of something
    else and every threshold in this module is being applied to the wrong
    numbers.
    """
    meta = read_mask(masks_path, f"{block_prefix}_meta")
    thr = float(meta["thr"])
    if thr != float(ACTIVITY_THR):
        raise ValueError(
            f"{block_prefix}: _col_act was cut at thr={thr}, but this module "
            f"reads it as column_activity at ACTIVITY_THR={float(ACTIVITY_THR)}"
        )
    activity = np.asarray(
        read_mask(masks_path, f"{block_prefix}_col_act"), dtype=np.float64
    )
    t_s = np.asarray(read_mask(masks_path, f"{block_prefix}_t_s"), dtype=np.float64)
    half = 0.5 * _frame_s(t_s)
    t_cov = (float(t_s[0]) - half, float(t_s[-1]) + half)
    elm = elm_events(
        activity, t_s, smooth_ms=smooth_ms,
        prominence=prominence, min_distance_ms=min_distance_ms,
    )
    return Transients(
        activity=activity,
        t_s=t_s,
        bursts=extract_bursts(
            activity, activity_min=activity_min, min_gap_cols=min_gap_cols
        ),
        elm_times_s=elm,
        clock=elm_clock(elm, t_s, window_s=window_s),
        elm_free_s=elm_free_intervals(
            elm, t_cov, max_rate_hz=max_rate_hz,
            min_duration_s=min_duration_s, window_s=window_s,
        ),
        t_cov=t_cov,
    )
