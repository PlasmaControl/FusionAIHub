"""Events: span boundaries and points, matched one to one within a tolerance.

A boundary is an event only where the state changes between absent and present.
A span that begins at the window's edge, or beside an uncertain or unobservable
stretch, has no observed onset, so it contributes none.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .frames import ABSENT, PRESENT, Assessment

#: Float slack on the tolerance, so a pair exactly at the tolerance matches.
SLACK_MS = 1e-9


def boundaries(assessment: Assessment) -> tuple[np.ndarray, np.ndarray]:
    """Onsets (absent to present) and ends (present to absent), in ms."""
    runs = assessment.runs()
    onsets, ends = [], []
    for (_, _, before), (a, _, after) in pairwise(runs):
        if before == ABSENT and after == PRESENT:
            onsets.append(a)
        elif before == PRESENT and after == ABSENT:
            ends.append(a)
    return np.array(onsets, dtype=float), np.array(ends, dtype=float)


@dataclass(frozen=True)
class Matching:
    """Matched index pairs, and what each side left unmatched."""

    pairs: np.ndarray  # (k, 2): reference index, estimate index
    unmatched_reference: np.ndarray
    unmatched_estimate: np.ndarray
    offsets: np.ndarray  # |reference - estimate| of each pair, ms


def match(reference, estimate, tolerance_ms: float) -> Matching:
    """Pair times one to one, nearest first, within `tolerance_ms`.

    Candidate pairs are taken in order of distance, ties by the earlier reference
    time and then the earlier estimate, and a pair is kept when neither time is
    already taken.
    """
    ref = np.asarray(reference, dtype=float).ravel()
    est = np.asarray(estimate, dtype=float).ravel()
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
    )


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


def within(times, window) -> np.ndarray:
    """The times inside `[start, stop)`, for scoring points in a checked window."""
    times = np.asarray(times, dtype=float).ravel()
    return times[(times >= window[0]) & (times < window[1])]
