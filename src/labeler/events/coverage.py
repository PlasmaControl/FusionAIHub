"""What was actually looked at, per source, per quantity.

A source's `Coverage` holds disjoint finite intervals and its detector's
`min_gap_s`. It records what was measured even when no event was found.
`finite_intervals` excludes leading, trailing and interior missing samples;
only gaps shorter than the calling detector's resolution may merge.
Required inputs intersect their sets; any-channel inputs use their union.
The `t_cov0_s`/`t_cov1_s` pair remains a display hull, never a coverage test.

Each source keeps its own inputs: a gas recorder's long axis cannot extend
NBI coverage. Likewise, the first and last finite sample cannot establish
observation of the gap between them. `finite_span`, `feature_spans` and
`intersect` remain available for event extents and display-only callers.

**Clipping is the other half of the same honesty.** A transform pads: a
track stitched across tile boundaries ended up to 2.052 ms past its own
`t_cov1_s` on the three pilot shots (9, 11 and 15 rows of them). That
overrun is transform-edge support, not plasma, so `clip_to_coverage`
trims the extent back into the coverage and says `attrs["clipped"] = true`
where it had to - and `schema.Event.__post_init__` now REFUSES a row that
ends after its own coverage, so the invariant is enforced at the point the
row is built rather than repaired later by whoever notices. A POINT that
the detector's own grid or gate window put past the bound - an ELM on the
transform's edge column, an L-H transition within 5 ms of the beam record's
end - is moved onto the bound by `clip_point_to_coverage` and says so, since
refusing it would cost the shot the whole step and not the one row.

Nothing here opens a file or knows a locator; it takes arrays and spans.
"""
from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

#: An unknown span. Not `(0, 0)` and not `(-inf, inf)`: "nobody looked" is
#: a third answer, and a consumer that treats it as either is wrong in a
#: direction that cannot be detected downstream.
UNKNOWN: tuple[float, float] = (float("nan"), float("nan"))


def _known(span) -> bool:
    return bool(math.isfinite(float(span[0])) and math.isfinite(float(span[1])))


def finite_span(t_s, y=None) -> tuple[float, float]:
    """`(first, last)` time at which there is a finite sample.

    `y` is the trace measured on `t_s`: `(samples,)` or
    `(channels, samples)`, in which case a sample counts wherever ANY
    channel is finite. Without `y` the finiteness of `t_s` itself is the
    test, which is what a bare time axis can be asked.

    `UNKNOWN` when nothing is finite - an all-NaN channel, an empty axis -
    because a span computed from no samples is not a span.
    """
    t, ok = _finite_mask(t_s, y)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return UNKNOWN
    return (float(t[idx[0]]), float(t[idx[-1]]))


def _finite_mask(t_s, y) -> tuple[np.ndarray, np.ndarray]:
    """`(t, ok)`: the axis and, per sample, whether there was something to see."""
    t = np.asarray(t_s, dtype=np.float64).ravel()
    ok = np.isfinite(t)
    if y is not None:
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            sample = np.isfinite(arr)
        elif arr.ndim == 2:
            sample = np.isfinite(arr).any(axis=0)
        else:
            raise ValueError(f"y must be 1-D or 2-D; got {arr.shape}")
        if sample.size != t.size:
            raise ValueError(
                f"{sample.size} samples against {t.size} times"
            )
        ok = ok & sample
    return t, ok


# ----------------------------------------------------- interval sets
#
# The third iteration-0 critic's remaining cap. `finite_span` is a HULL:
# the real 198658 `filterscopes` group with every channel NaN over 1-2 s
# still gave the ELM clock one continuous -0.05..6.95 s span, so a window
# inside the gap came back from the MCP as `observed, n=0` - "an
# observation of nothing happening" over a second nobody measured. What a
# detector saw is the SET of finite runs; the hull is for display only.

#: A disjoint, sorted set of `(t0, t1)` intervals. `()` is the interval
#: analogue of `UNKNOWN`: nothing finite, nothing observed.
IntervalSet = tuple[tuple[float, float], ...]


def _check_min_gap(min_gap_s) -> float:
    try:
        gap = float(min_gap_s)
    except (TypeError, ValueError):
        gap = math.nan
    if not (math.isfinite(gap) and gap >= 0.0):
        raise ValueError(
            f"min_gap_s must be a finite non-negative number of seconds; "
            f"got {min_gap_s!r}"
        )
    return gap


def finite_intervals(t_s, y=None, *, min_gap_s) -> IntervalSet:
    """The maximal runs of finite samples on `t_s`, as `(first, last)` times.

    `y` is as for `finite_span`: `(samples,)`, or `(channels, samples)` in
    which case a sample counts wherever ANY channel is finite; without it
    the finiteness of `t_s` itself is the test.

    Two runs separated by a gap SHORTER than `min_gap_s` - measured from the
    last finite sample before it to the first after - are one interval. The
    constant is the guarded detector's own resolution: a dropout the ELM
    clock could not have resolved a peak inside anyway is not a hole in
    what it saw, while one it could have is. `0.0` bridges nothing.

    Leading and trailing NaN - the corpus' fast groups are 2^k + 1 long with
    a NaN last sample; a filterscope's head is NaN - are simply outside the
    first and last run and never make an interval of their own. All NaN is
    `()`.
    """
    gap = _check_min_gap(min_gap_s)
    t, ok = _finite_mask(t_s, y)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ()
    seen = t[idx]
    if np.any(np.diff(seen) < 0):
        raise ValueError("t_s must be non-decreasing")
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    merged: list[tuple[float, float]] = []
    for s, e in zip(starts, ends):
        lo, hi = float(t[s]), float(t[e])
        if merged and lo - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return tuple(merged)


def interval_hull(intervals: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """`(first start, last end)` of a sorted set - the DISPLAY hull - or `UNKNOWN`.

    This is what `t_cov0_s`/`t_cov1_s` hold once the intervals are the
    coverage: a number for a plot axis or a sentence, never the thing that
    decides whether a window was observed.
    """
    ivs = tuple(intervals)
    if not ivs:
        return UNKNOWN
    return (float(ivs[0][0]), float(ivs[-1][1]))


def union_intervals(*sets: Iterable[tuple[float, float]]) -> IntervalSet:
    """One sorted, disjoint set covering every interval of every input.

    Overlapping and touching intervals merge. For a step whose coverage is
    "any channel" - the QH proxy runs on whichever tokeye block produced a
    track. No input at all is `()`.
    """
    ivs = []
    for group in sets:
        for lo, hi in group:
            lo, hi = float(lo), float(hi)
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"intervals must be finite; got ({lo}, {hi})")
            if hi < lo:
                raise ValueError(
                    f"an interval's start must precede its end; got ({lo}, {hi})"
                )
            ivs.append((lo, hi))
    ivs.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in ivs:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _intersect_two(a: IntervalSet, b: IntervalSet) -> IntervalSet:
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi >= lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def intersect_intervals(*sets: Iterable[tuple[float, float]]) -> IntervalSet:
    """The set of times EVERY input covers.

    For a detector that needs several inputs at once - the L-H detector
    needs D-alpha AND the line density AND the injected power, the QH proxy
    its tracks AND the ELM clock AND the beam power AND the flat-top. One
    empty input makes the whole intersection `()`, as one `UNKNOWN` makes
    `intersect` unknown: a heuristic whose density trace was never
    measured observed nothing, not the D-alpha's stretch. No input at all
    is `()` for the same reason. Two intervals that only touch meet in a
    point.
    """
    if not sets:
        return ()
    result = union_intervals(sets[0])
    for other in sets[1:]:
        result = _intersect_two(result, union_intervals(other))
        if not result:
            return ()
    return result


@dataclasses.dataclass(frozen=True)
class Coverage:
    """What one source saw: its interval set and how small a gap could hide.

    `intervals` is sorted and disjoint; `min_gap_s` is the resolution the
    set was built at (`finite_intervals`' argument, or the coarsest of the
    inputs when sets were intersected), persisted beside the row so a
    reader knows that a dropout shorter than it would not show. `hull` is
    the display span the row's `t_cov0_s`/`t_cov1_s` carry; `known` is
    whether anything at all was seen.
    """

    intervals: IntervalSet
    min_gap_s: float

    def __post_init__(self) -> None:
        ivs = tuple((float(lo), float(hi)) for lo, hi in self.intervals)
        for lo, hi in ivs:
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"intervals must be finite; got ({lo}, {hi})")
            if hi < lo:
                raise ValueError(
                    f"an interval's start must precede its end; got ({lo}, {hi})"
                )
        for (plo, phi), (lo, _hi) in pairwise(ivs):
            if lo < plo:
                raise ValueError(f"intervals must be sorted; got {ivs}")
            if lo <= phi:
                raise ValueError(f"intervals must be disjoint; got {ivs}")
        object.__setattr__(self, "intervals", ivs)
        object.__setattr__(self, "min_gap_s", _check_min_gap(self.min_gap_s))

    @property
    def hull(self) -> tuple[float, float]:
        return interval_hull(self.intervals)

    @property
    def known(self) -> bool:
        return bool(self.intervals)

    @classmethod
    def measured(cls, t_s, y=None, *, min_gap_s: float) -> Coverage:
        """Finite sample runs at the calling detector's documented resolution."""
        return cls(finite_intervals(t_s, y, min_gap_s=min_gap_s), min_gap_s)

    def intersect(self, *others: Coverage) -> Coverage:
        """Required inputs intersect; retain the coarsest input resolution."""
        return Coverage(
            intersect_intervals(self.intervals, *(c.intervals for c in others)),
            max([self.min_gap_s, *(c.min_gap_s for c in others)]),
        )


def attach_intervals(events, ran):
    """Carry source interval sets on event rows for databases missing a source table.

    The source table remains authoritative, including for zero detections.
    Event hulls alone must never revive a known gap if that table is absent.
    """
    out = []
    for event in events:
        key = (event.source, event.diag, event.channel, event.pass_name)
        cov = ran.get(key, Coverage((), 0.0))
        out.append(dataclasses.replace(event, attrs={
            **event.attrs, "coverage_intervals": cov.intervals,
            "coverage_min_gap_s": cov.min_gap_s,
        }))
    return out


def feature_spans(
    features: Mapping[str, tuple[Any, Any]],
) -> dict[str, tuple[float, float]]:
    """`{name: (t0, t1)}`: each canonical feature's display hull, not its coverage set.

    `features` is what `pipeline._actuator_features` builds and
    `heuristics.actuator_intervals` reads - `{name: (t_s, y)}` in
    `features/namespace.py`'s terms - and this is the per-quantity coverage
    that replaces the union of all their axes.
    """
    return {
        name: finite_span(t_s, y) for name, (t_s, y) in features.items()
    }


def intersect(spans: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """The interval every one of `spans` covers, or `UNKNOWN`.

    For a detector that needs several inputs AT ONCE. One unknown span
    makes the intersection unknown rather than being skipped: a heuristic
    whose density trace was never measured did not observe the shot's
    D-alpha stretch, it observed nothing, and answering with the D-alpha's
    span would claim otherwise. An empty iterable is `UNKNOWN` for the
    same reason, and so is an intersection that comes out backwards - two
    inputs that never ran at the same time cover no time together.
    """
    spans = list(spans)
    if not spans or not all(_known(s) for s in spans):
        return UNKNOWN
    lo = max(float(s[0]) for s in spans)
    hi = min(float(s[1]) for s in spans)
    return (lo, hi) if hi >= lo else UNKNOWN


def clip_to_coverage(
    t0_s: float, t1_s: float, cov: tuple[float, float],
) -> tuple[float, float, bool]:
    """`(t0, t1, clipped)`: the extent trimmed into `[cov0, cov1]`.

    An unknown or half-unknown coverage clips nothing on the side it does
    not know, because there is no bound there to trim to.

    `cov` is the display hull. An event extent spanning an interior gap
    stays intact: the detector saw both sides; coverage of the gap itself
    is decided separately from the source's interval set.

    An extent lying WHOLLY outside its coverage comes back untouched, with
    `clipped` false. That is not a transform edge - it is a detector
    claiming an event in a stretch it did not measure - and moving it onto
    the boundary would turn a bug into a plausible-looking row. Left
    unclipped, `schema.Event.__post_init__` refuses it, which is where a
    caller finds out.
    """
    t0, t1 = float(t0_s), float(t1_s)
    cov0, cov1 = float(cov[0]), float(cov[1])
    lo = max(t0, cov0) if math.isfinite(cov0) else t0
    hi = min(t1, cov1) if math.isfinite(cov1) else t1
    if hi < lo:
        return (t0, t1, False)
    return (lo, hi, lo != t0 or hi != t1)


def clip_point_to_coverage(
    t_s: float, cov: tuple[float, float],
) -> tuple[float, bool]:
    """`(t, clipped)`: a POINT moved onto the coverage bound it overran.

    `clip_to_coverage` leaves an extent lying wholly outside its coverage
    alone, to be refused: an interval nobody measured is a detector claiming
    a stretch it did not look at. A point event is a different case when it
    is the detector's OWN grid or gate window that put it past the bound:

    * an ELM's time is its column's centre (`transients.elm_events`), and a
      stitched transform's first and last columns lie outside the record
      they were computed from - `masks.COL_ORIGIN` padding, measured at up
      to 2.052 ms on the pilot shots. Every column of the grid came from
      THIS record, so a centre outside the sample span cannot be a
      measurement made anywhere else;
    * an L-H transition clears its beam gate on `[when - 5 ms, when]`, so a
      `when` up to 5 ms after the NBI record's last finite sample is a
      claim whose every input WAS measured - and it lies past the
      intersection coverage all the same.

    In both the overrun is bounded by the detector's own time resolution,
    and `pipeline.finish_shot` isolates per STEP, not per row: refusing the
    one point would cost the shot the whole ELM clock, or its every L-H
    claim, over a millisecond. So the point is moved onto the bound and the
    row says so; the caller keeps the measured instant in `attrs` (`col`
    for an ELM, `t_measured_s` for a transition). An unknown bound moves
    nothing on its side. This clips only to the outer hull, never across
    an interior gap; source intervals separately decide observation.
    """
    t = float(t_s)
    cov0, cov1 = float(cov[0]), float(cov[1])
    if math.isfinite(cov0) and t < cov0:
        return (cov0, True)
    if math.isfinite(cov1) and t > cov1:
        return (cov1, True)
    return (t, False)


def clipped_attrs(attrs: Mapping[str, Any], clipped: bool) -> dict[str, Any]:
    """`attrs` with `clipped` recorded when, and only when, it happened.

    Absent rather than `false` on the ordinary row: `attrs` is stored as a
    JSON string per row and a key that is false on 99.9% of a campaign's
    events costs more than it says. A consumer asks `attrs.get("clipped")`.
    """
    out = dict(attrs)
    if clipped:
        out["clipped"] = True
    return out


# ------------------------------------------------- per-source completion

#: A skip whose step name starts with one of these is about a `(diag,
#: channel[, pass])` block, and the rest of the key names it. `norm` is
#: its own source and not `tokeye_track`'s, because a norm fallback is
#: recorded WHILE the block runs: they are two facts about one block, and
#: collapsing them onto one key would lose whichever came second.
_BLOCK_STEPS = {
    "channel": "tokeye_track",
    "read": "tokeye_track",
    "mask": "tokeye_track",
    "track": "tokeye_track",
    "norm": "norm",
}

#: Step name -> the sources it would have written. The D-alpha clock and
#: TokEye transients run independently. A step with no entry here is
#: its own source name - `qh_flattop` and `nbi_counter` are recorded skips
#: that are not event sources at all, and inventing a source for them would
#: be worse than letting them name themselves.
_STEP_SOURCES = {
    "elm_clock": ("elm_clock",),
    "sawtooth": ("ece_sawtooth",),
    "lh": ("dalpha_lh",),
    "actuator": ("actuator",),
    "qh": ("qh_proxy",),
    "text": ("text",),
    # The features-store read (task L-D2). Three step names, one source:
    # `features` is the whole file missing, `ip` and `qmin` are one
    # quantity each, and they land on the SAME keys the successful read
    # declares - `("features", "<quantity>", -1, "")` - so a shot's row for
    # `qmin` says either what it covered or why there was none, never both
    # and never neither.
    "features": ("features",),
    "ip": ("features",),
    "qmin": ("features",),
    # `nbi_counter` is a PHENOMENON of the actuator source, not a source,
    # and `heuristics._actuator_event` stamps `diag="tinj_total"` on its
    # rows and on no others. So a shot where it could not be evaluated and
    # a shot where it was land on the same key - `("actuator",
    # "tinj_total", ...)` - and a consumer reads one row rather than
    # having to know that a missing row means one thing under one name and
    # another under another.
    "nbi_counter": ("actuator",),
}

#: The diagnostic a non-block step reads, where it reads one. Used only to
#: fill the `diag` column of a SKIPPED step's row, so that a skipped
#: sawtooth row and a sawtooth row that ran carry the same key.
_STEP_DIAGS = {
    "elm_clock": "filterscopes",
    "sawtooth": "ece",
    "lh": "filterscopes",
    "ip": "ip",
    "qmin": "qmin",
    # `qmin_rule` is its own source and reads one canonical feature, so a
    # skipped rule and a rule that ran carry the same key.
    "qmin_rule": "qmin",
    "nbi_counter": "tinj_total",
}


def _split_block_key(rest: str) -> tuple[str, int, str]:
    """`"mhr:4:wide"` -> `("mhr", 4, "wide")`; a missing part is the default."""
    parts = rest.split(":")
    diag = parts[0] if parts and parts[0] else ""
    try:
        channel = int(parts[1])
    except (IndexError, ValueError):
        channel = -1
    pass_name = parts[2] if len(parts) > 2 else ""
    return diag, channel, pass_name


def skip_key(step: str) -> list[tuple[str, str, int, str]]:
    """One `ShotResult.skipped` step name -> the `(source, diag, channel,
    pass)` keys it is a fact about.

    The step names are `pipeline.process_shot`'s: `"channel bes:26"`,
    `"mask mhr:4:wide"`, `"actuator gas"`, `"elm_clock"`, `"sawtooth"`.
    Anything unrecognised becomes a row under its own name rather than
    being dropped - a skip nobody can read is still better than a skip
    nobody can see.
    """
    head, _, rest = step.partition(" ")
    if head in _BLOCK_STEPS and rest:
        diag, channel, pass_name = _split_block_key(rest)
        return [(_BLOCK_STEPS[head], diag, channel, pass_name)]
    if head == "actuator" and rest:
        # `"actuator gas"`: the group is what was not read, and the group
        # name is what the ran rows carry in `diag`.
        return [("actuator", rest, -1, "")]
    sources = _STEP_SOURCES.get(step, (step,))
    diag = _STEP_DIAGS.get(step, "")
    return [(source, diag, -1, "") for source in sources]


def source_records(
    shot: int,
    *,
    ran: Mapping[tuple[str, str, int, str], Coverage | tuple[float, float]],
    skipped: Mapping[str, str],
    events: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """One shot's per-source completion rows, for `schema.write_sources`.

    `ran` maps a `(source, diag, channel, pass_name)` key to the coverage
    that key was computed over - EVERY key that ran, whether or not it
    produced an event, which is the whole point: "the tracker ran on
    mhr_04_wide and saw no mode" and "mhr_04_wide was never processed" are
    different facts and an events file alone states neither.

    `skipped` is `ShotResult.skipped`; every entry becomes a row (two,
    where one step owns two sources), with its reason. A key that both ran
    and was skipped keeps the RAN row, because the step got far enough to
    look and `status == "ok"` carries an empty `reason` by contract; the
    step's own message is still in `ShotResult.skipped` and in the run's
    JSON. `norm` and `qh_flattop` have sources of their own precisely so
    that the caveats which fire while a step SUCCEEDS get rows rather than
    colliding with it.
    """
    counts: dict[tuple[str, str, int, str], int] = {}
    for event in events:
        key = (
            str(event.source), str(event.diag), int(event.channel),
            str(event.pass_name),
        )
        counts[key] = counts.get(key, 0) + 1

    rows: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for key, span in ran.items():
        rows[key] = _row(shot, key, "ok", "", span, counts.get(key, 0))
    for step, why in skipped.items():
        for key in skip_key(step):
            if key in ran:
                continue
            if key in rows:
                rows[key]["reason"] = f"{rows[key]['reason']}; {why}"
                continue
            rows[key] = _row(shot, key, "skipped", why, UNKNOWN,
                             counts.get(key, 0))
    # Any source that wrote events without declaring a key: recorded rather
    # than lost, with the coverage its own rows carry.
    for key, n in counts.items():
        if key not in rows:
            rows[key] = _row(shot, key, "ok", "", UNKNOWN, n)
    return [rows[key] for key in sorted(rows)]


def _row(shot: int, key, status: str, reason: str, span, n_events: int) -> dict:
    source, diag, channel, pass_name = key
    cov = span if isinstance(span, Coverage) else None
    span = cov.hull if cov is not None else span
    return {
        "shot": int(shot),
        "source": str(source),
        "status": str(status),
        "reason": str(reason),
        "t_cov0_s": float(span[0]),
        "t_cov1_s": float(span[1]),
        "n_events": int(n_events),
        "diag": str(diag),
        "channel": int(channel),
        "pass_name": str(pass_name),
        # None marks an older caller's hull. [] explicitly means no coverage.
        "intervals": json.dumps(cov.intervals) if cov is not None else None,
        "min_gap_s": cov.min_gap_s if cov is not None else math.nan,
    }


__all__ = [
    "UNKNOWN",
    "Coverage",
    "IntervalSet",
    "clip_point_to_coverage",
    "clip_to_coverage",
    "clipped_attrs",
    "feature_spans",
    "finite_intervals",
    "finite_span",
    "intersect",
    "intersect_intervals",
    "interval_hull",
    "skip_key",
    "source_records",
    "union_intervals",
]
