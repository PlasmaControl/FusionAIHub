"""Events: span boundaries and points, matched one to one within a tolerance.

A boundary is an event only where the state changes between absent and present.
A span that begins at the window's edge, or beside an uncertain or unobservable
stretch, has no observed onset, so it contributes none.

D21 scores boundaries strictly inside the common window. After nearest-first
matching, unmatched boundaries within the tolerance of the other reader's
uncertain or not-observable spans are excluded and recorded. Matched pairs
always count. A method's abstentions read as absent and never exclude; two
readers' abstentions exclude in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from .frames import ABSENT, PRESENT, Assessment

#: Float slack on the tolerance, so a pair exactly at the tolerance matches.
SLACK_MS = 1e-9


def _timing_option(value, name):
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _times(values):
    times = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(times).all():
        raise ValueError("event times must be finite")
    return times


def boundaries(
    assessment: Assessment, *, method: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Onsets and ends in ms; a method's abstentions read as absent."""
    runs = assessment.runs()
    if method:
        runs = [(a, b, PRESENT if s == PRESENT else ABSENT) for a, b, s in runs]
    onsets, ends = [], []
    for (_, _, before), (a, _, after) in pairwise(runs):
        if before == ABSENT and after == PRESENT:
            onsets.append(a)
        elif before == PRESENT and after == ABSENT:
            ends.append(a)
    return np.array(onsets, dtype=float), np.array(ends, dtype=float)


@dataclass(frozen=True)
class Matching:
    """Pairs and unmatched/excluded indices into the recorded times, in ms."""

    pairs: np.ndarray  # (k, 2): reference index, estimate index
    unmatched_reference: np.ndarray
    unmatched_estimate: np.ndarray
    offsets: np.ndarray  # |reference - estimate| of each pair, ms
    reference: np.ndarray
    estimate: np.ndarray
    excluded_reference: np.ndarray
    excluded_estimate: np.ndarray


def match(reference, estimate, tolerance_ms: float) -> Matching:
    """Pair times one to one, nearest first, within `tolerance_ms`.

    Candidate pairs are taken in order of distance, ties by the earlier reference
    time and then the earlier estimate, and a pair is kept when neither time is
    already taken.
    This nearest-first rule (spec section 7.1) is not maximum-cardinality:
    references [0, 10] and estimates [6, 14] at tolerance 6 give one pair,
    although two pairs are possible.
    """
    _timing_option(tolerance_ms, "tolerance_ms")
    ref = _times(reference)
    est = _times(estimate)
    distance = np.abs(ref[:, None] - est[None, :])
    i, j = np.nonzero(distance <= tolerance_ms + SLACK_MS)
    d = distance[i, j]
    order = np.lexsort((est[j], ref[i], d))
    taken_ref = np.zeros(len(ref), dtype=bool)
    taken_est = np.zeros(len(est), dtype=bool)
    pairs = []
    for k in order:
        a, b = i[k], j[k]
        if not (taken_ref[a] or taken_est[b]):
            taken_ref[a] = taken_est[b] = True
            pairs.append((a, b))
    pairs = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    return Matching(
        pairs=pairs,
        unmatched_reference=np.flatnonzero(~taken_ref),
        unmatched_estimate=np.flatnonzero(~taken_est),
        offsets=np.abs(ref[pairs[:, 0]] - est[pairs[:, 1]]),
        reference=ref,
        estimate=est,
        excluded_reference=np.array([], dtype=np.int64),
        excluded_estimate=np.array([], dtype=np.int64),
    )


def _near_abstention(times, assessment: Assessment, tolerance_ms: float):
    """Distance to closed abstention spans, including the tolerance slack."""
    near = np.zeros(len(times), dtype=bool)
    for a, b, state in assessment.spans:
        if state > PRESENT:
            distance = np.maximum(np.maximum(a - times, times - b), 0)
            near |= distance <= tolerance_ms + SLACK_MS
    return near


def event_matchings(
    reference: Assessment,
    estimate: Assessment,
    tolerance_ms: float,
    *,
    method: bool,
) -> dict[str, Matching]:
    """D21 onset/end scoring: common interior, match, then exclude abstentions.

    The reference is a reader; `method` selects how the estimate is read.
    Only unmatched boundaries can be excluded. Add the two matchings' cells
    for combined event F1, and use their offsets for the median timing error.
    """
    lo = max(reference.window[0], estimate.window[0])
    hi = min(reference.window[1], estimate.window[1])
    result = {}
    for key, ref, est in zip(
        ("onset", "end"), boundaries(reference), boundaries(estimate, method=method)
    ):
        ref = ref[(ref > lo) & (ref < hi)]
        est = est[(est > lo) & (est < hi)]
        matching = match(ref, est, tolerance_ms)
        ur, ue = matching.unmatched_reference, matching.unmatched_estimate
        exclude_ref = (
            np.zeros(len(ur), dtype=bool)
            if method
            else _near_abstention(ref[ur], estimate, tolerance_ms)
        )
        exclude_est = _near_abstention(est[ue], reference, tolerance_ms)
        result[key] = replace(
            matching,
            unmatched_reference=ur[~exclude_ref],
            unmatched_estimate=ue[~exclude_est],
            excluded_reference=ur[exclude_ref],
            excluded_estimate=ue[exclude_est],
        )
    return result


def event_cells(matching: Matching) -> np.ndarray:
    """`[tp, fp, fn]` of one matching."""
    return np.array(
        [
            len(matching.pairs),
            len(matching.unmatched_estimate),
            len(matching.unmatched_reference),
        ],
        dtype=float,
    )


def within(times, window, *, end_slack_ms: float = 0.0) -> np.ndarray:
    """Times in `[start, stop)`, plus `[stop, stop + end_slack_ms]` if positive.

    A disruption's t_D, t80 and t20 take D19's 2 ms end slack:
    `labeler.events.catalog.check.DISRUPTION_TIMING_TOLERANCE_MS`.
    """
    _timing_option(end_slack_ms, "end_slack_ms")
    times = _times(times)
    before_end = times < window[1]
    if end_slack_ms > 0:
        before_end |= times <= window[1] + end_slack_ms
    return times[(times >= window[0]) & before_end]
