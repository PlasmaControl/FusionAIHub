"""Assessments on 10 ms frames.

An assessment is what one reader or one method said about one phenomenon on
one shot: a window `[start, stop)` in whole ms and the spans inside it, each
with a state. Time inside the window that no span covers is absent; time
outside it was not assessed.

Frames are the 10 ms cells `[10k, 10k + 10)` on the shot's own clock, so every
comparison of a shot uses the same cells. A frame belongs to an assessment only
if its window covers the whole frame, and it takes the highest state that
touches it: not observable over uncertain over present over absent. A frame half
present and half absent is therefore present, and one grazed by an uncertain
span is uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..events.catalog.states import (
    ABSENT,
    NOT_OBSERVABLE,
    PRESENT,
    STATE_NAMES,
    UNCERTAIN,
)

FRAME_MS = 10
STATES = tuple(STATE_NAMES)
OUTSIDE = -1


def whole_number(value) -> int:
    """Validate a whole numeric value before converting it to a plain int."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if (
        isinstance(value, (float, np.floating))
        and np.isfinite(value)
        and value == np.floor(value)
    ):
        return int(value)
    raise ValueError(f"value {value!r} is not a whole number")


@dataclass(frozen=True)
class Assessment:
    """A window and its non-overlapping spans `(start, stop, state)`, in ms."""

    window: tuple[int, int]
    spans: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        lo, hi = (whole_number(v) for v in self.window)
        if not lo < hi:
            raise ValueError(f"window {lo}-{hi} ms is empty")
        spans = tuple(
            sorted(tuple(whole_number(v) for v in span) for span in self.spans)
        )
        last = lo
        for a, b, state in spans:
            if state not in STATES:
                raise ValueError(f"state {state} is not one of {STATES}")
            if not (lo <= a < b <= hi):
                raise ValueError(f"span {a}-{b} ms is empty or outside {lo}-{hi} ms")
            if a < last:
                raise ValueError(f"span {a}-{b} ms overlaps the one before it")
            last = b
        object.__setattr__(self, "window", (int(lo), int(hi)))
        object.__setattr__(self, "spans", spans)

    @classmethod
    def from_label(cls, label) -> Assessment:
        """From a review `Label`: its intervals are the non-absent spans."""
        return cls(tuple(label.window), tuple(tuple(s) for s in label.intervals))

    @classmethod
    def from_rows(cls, rows) -> Assessment:
        """From nonempty `(t_start, t_end, state)` rows tiling one window."""
        rows = sorted(tuple(whole_number(v) for v in row) for row in rows)
        if not rows:
            raise ValueError("no rows")
        previous = None
        for row in rows:
            a, b, state = row
            if not a < b:
                raise ValueError(f"row {row} is empty or reversed")
            if state not in STATES:
                raise ValueError(f"state {state} is not one of {STATES}")
            if previous is not None and previous[1] != a:
                raise ValueError(f"rows {previous} and {row} have a gap or overlap")
            previous = row
        window = (rows[0][0], rows[-1][1])
        return cls(window, tuple(row for row in rows if row[2] != ABSENT))

    def runs(self) -> list[tuple[int, int, int]]:
        """The window as consecutive `(start, stop, state)` runs, gaps absent."""
        out = []
        cursor = self.window[0]
        for a, b, state in self.spans:
            if a > cursor:
                out.append((cursor, a, ABSENT))
            out.append((a, b, state))
            cursor = b
        if cursor < self.window[1]:
            out.append((cursor, self.window[1], ABSENT))
        merged = []
        for a, b, state in out:
            if merged and merged[-1][2] == state and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b, state)
            else:
                merged.append((a, b, state))
        return merged

    @classmethod
    def from_checked(cls, rows, allowed, *, category: str) -> Assessment:
        """Tiled whole-ms rows inside `allowed`, clipping only D19's final span.

        A reader may shrink the assessed window. Only a disruption's last present
        row may overrun the allowed end by the checker's timing tolerance.
        """
        from ..events.catalog.check import DISRUPTION_TIMING_TOLERANCE_MS

        rows = sorted(tuple(whole_number(v) for v in row) for row in rows)
        assessed = cls.from_rows(rows)
        lo, hi = cls(tuple(allowed)).window
        for start, stop, state in assessed.spans:
            if state == PRESENT and start >= hi:
                raise ValueError(
                    f"present span {start}-{stop} ms starts at or after allowed end {hi} ms"
                )
        a, b = assessed.window
        if a < lo or a >= hi:
            raise ValueError("assessed start is outside the allowed window")
        if b > hi:
            if category != "disruption" or b > hi + DISRUPTION_TIMING_TOLERANCE_MS:
                raise ValueError("assessed end is outside the allowed window")
            overrun = [row for row in rows if row[1] > hi]
            if len(overrun) != 1 or overrun[0][2] != PRESENT:
                raise ValueError("only the last present span may overrun allowed end")
        return cls(
            (a, min(b, hi)),
            tuple((a, min(b, hi), s) for a, b, s in assessed.spans),
        )


def frame_grid(*assessments: Assessment) -> tuple[int, int]:
    """The first frame index and the number of frames every window covers."""
    lo = max(a.window[0] for a in assessments)
    hi = min(a.window[1] for a in assessments)
    first = -(-lo // FRAME_MS)
    return first, max(0, hi // FRAME_MS - first)


def frame_states(assessment: Assessment, first: int, n: int) -> np.ndarray:
    """The state of frames `first .. first + n - 1`; `OUTSIDE` off the window."""
    starts = (first + np.arange(n)) * FRAME_MS
    lo, hi = assessment.window
    inside = (starts >= lo) & (starts + FRAME_MS <= hi)
    states = np.where(inside, ABSENT, OUTSIDE).astype(np.int8)
    for a, b, state in assessment.spans:
        j0 = max(0, a // FRAME_MS - first)
        j1 = min(n, -(-b // FRAME_MS) - first)
        if j1 > j0:
            touched = states[j0:j1]
            touched[inside[j0:j1]] = np.maximum(touched[inside[j0:j1]], state)
    return states


@dataclass(frozen=True)
class FrameCounts:
    """Frames both sides assessed, split by what each said.

    Counted frames are those the reference called present or absent. A method's
    uncertain or not-observable frame counts as not present; how many of the
    counted frames it called each is kept beside the four cells.
    Reference-only frames were present or absent on the reference's own grid,
    but the estimate's window did not cover them; they are outside the cells.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    excluded: int
    method_uncertain: int
    method_unobserved: int
    reference_only: int

    def cells(self) -> np.ndarray:
        """`[tp, fp, fn, tn]`, the order `stats` expects."""
        return np.array([self.tp, self.fp, self.fn, self.tn], dtype=float)


def frame_counts(reference: Assessment, estimate: Assessment) -> FrameCounts:
    """Count the estimate's frames against the reference's."""
    first, n = frame_grid(reference, estimate)
    ref = frame_states(reference, first, n)
    est = frame_states(estimate, first, n)
    both = (ref != OUTSIDE) & (est != OUTSIDE)
    counted = both & (ref <= PRESENT)
    truth = ref == PRESENT
    said = est == PRESENT
    ref_first, ref_n = frame_grid(reference)
    own_ref = frame_states(reference, ref_first, ref_n)
    own_est = frame_states(estimate, ref_first, ref_n)
    return FrameCounts(
        tp=int(np.sum(counted & truth & said)),
        fp=int(np.sum(counted & ~truth & said)),
        fn=int(np.sum(counted & truth & ~said)),
        tn=int(np.sum(counted & ~truth & ~said)),
        excluded=int(np.sum(both & (ref > PRESENT))),
        method_uncertain=int(np.sum(counted & (est == UNCERTAIN))),
        method_unobserved=int(np.sum(counted & (est == NOT_OBSERVABLE))),
        reference_only=int(np.sum((own_ref <= PRESENT) & (own_est == OUTSIDE))),
    )


def agreement_frames(*reads: Assessment) -> np.ndarray:
    """`(n_frames, n_reads)` 0/1: frames every read called present or absent."""
    first, n = frame_grid(*reads)
    states = np.stack([frame_states(read, first, n) for read in reads], axis=1)
    keep = np.isin(states, (ABSENT, PRESENT)).all(axis=1)
    return states[keep].astype(np.int8)


def shot_presence(assessment: Assessment, *, method: bool = False) -> int | None:
    """1 if present anywhere; 0 if absent over all observable time; else None.

    None means the shot-level answer is not known: something was uncertain and
    nothing present, or the whole window was not observable.
    With `method=True`, uncertain and not observable count as absent.
    """
    states = {state for _, _, state in assessment.spans}
    if PRESENT in states:
        return 1
    if method:
        return 0
    if UNCERTAIN in states:
        return None
    unobserved = sum(b - a for a, b, s in assessment.spans if s == NOT_OBSERVABLE)
    lo, hi = assessment.window
    return 0 if unobserved < hi - lo else None
