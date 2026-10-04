"""RWM labels from Jeremy Hanson's onset times.

An onset is a point in time. Two labels are made from it, and they are different
things:

* the **physical export**: an uncertain pre-onset window
  `[onset - ONSET_WINDOW_MS, onset)` (category 2), and minimal present time
  `[onset, onset + PRESENT_MS)` (category 1), lasting one 10 ms slice;
* the **forecast target** of the baseline, as Piccione et al. 2022 define their
  stability label: a time slice is positive when an onset follows within
  `HORIZON_MS` (100 ms there). Primary negatives are earlier than the last target
  onset. Slices after it are excluded; a broader negative set is a sensitivity.

Only the listed onset points are verified evidence. Physical absence before the
first precursor is assumed, conditional on onset-list completeness. Beyond the
minimal present slice the state is unassessed without termination evidence;
later forecast negatives do not establish physical absence. Comparison shots are
unlabelled; their slices stay `UNLABELLED`.
"""

from __future__ import annotations

import numpy as np

#: The paper's horizon: slices within 100 ms before the RWM time are positive.
HORIZON_MS = 100.0
#: Slices from an onset to this long after it are neither stable nor about to go
#: unstable; the mode is already acting, so they leave the training and scoring sets.
POST_MS = 100.0
#: Uncertain pre-onset interval motivated by the millisecond wall time tau_w.
#: ASSUMPTION: 20 ms is about four wall times if tau_w ~ 5 ms, which is an
#: order-of-magnitude choice (Piccione 2022 gives only "milliseconds"; a DIII-D
#: source is still needed, \citeph{DIII-D wall time}), not a measured growth time.
#: Hanson's ONSET_TIME has no documented detection/threshold meaning in the
#: supplied sources, so pre-onset presence and extent remain uncertain.
#: N1RMS is not RWM-specific; a random-time slope search gives similar maxima.
ONSET_WINDOW_MS = 20.0
#: Minimal post-onset present interval: one slice, not a measured mode duration.
PRESENT_MS = 10.0

#: Listed onsets this close are one event.
MERGE_MS = 10.0

POSITIVE, NEGATIVE, EXCLUDED = 1, 0, -1
#: A slice of a shot whose onsets nobody listed: neither positive nor negative.
UNLABELLED = -2

#: Interval-table categories written by `window_rows`.
PRESENT, ABSENT, UNCERTAIN = 1, 0, 2
UNASSESSED = 4


def slice_labels(
    t_ms,
    onsets_ms,
    *,
    horizon_ms=HORIZON_MS,
    post_ms=POST_MS,
    other_onsets_ms=(),
    negative_scope="piccione",
):
    """Per-slice label: 1 an onset within the next `horizon_ms`, 0 none, -1 excluded.

    A slice at `t` is positive when some onset `o` has `o - horizon <= t < o`. It is
    excluded when some onset has `o <= t < o + post` and it is not positive (an onset
    that follows soon after another keeps its own positive slices). Every other slice
    before the last target onset is an assumed negative. With no target onset every
    slice is excluded: an n=2-only Hanson shot is not a verified n=1-stable shot.
    This extends Piccione's single-onset label to multiple onsets using the last
    target onset as the end of negative coverage, with earlier aftermath excluded.
    `negative_scope="broad"` retains the former post-last-onset negatives solely
    for sensitivity scoring. A comparison shot must be assigned UNLABELLED by its
    caller regardless of this function's result.

    `other_onsets_ms` are onsets of a different kind (an n = 2 RWM when the target is
    n = 1): their slices from `o - horizon` to `o + post` are excluded, never
    positive and never negative, unless a target onset makes them positive.
    """
    t = np.asarray(t_ms, dtype=float)
    if negative_scope not in ("piccione", "broad"):
        raise ValueError("negative_scope must be 'piccione' or 'broad'")
    label = np.zeros(t.shape, dtype=np.int8)
    for other in np.asarray(other_onsets_ms, dtype=float):
        label[(t >= other - horizon_ms) & (t < other + post_ms)] = EXCLUDED
    onsets = np.sort(np.asarray(onsets_ms, dtype=float))
    if not len(onsets):
        return (
            np.full(t.shape, EXCLUDED, dtype=np.int8)
            if negative_scope == "piccione"
            else label
        )
    # First onset strictly after each slice time, and last at or before it.
    after = np.searchsorted(onsets, t, side="right")
    before = after - 1
    next_gap = np.where(
        after < len(onsets), onsets[np.minimum(after, len(onsets) - 1)] - t, np.inf
    )
    last_gap = np.where(before >= 0, t - onsets[np.maximum(before, 0)], np.inf)
    label[last_gap < post_ms] = EXCLUDED
    if negative_scope == "piccione":
        label[t >= onsets[-1]] = EXCLUDED
    label[next_gap <= horizon_ms] = POSITIVE
    return label


def merge_close(onsets_ms, within_ms=MERGE_MS):
    """Onsets with any that follows within `within_ms` of the one kept dropped.

    Hanson's tables list some onsets twice, a few tenths of a ms apart (156787 at
    1616.9 and 1617.5 ms, 156795 at 1656.0 and 1656.4 ms); they are one event for
    scoring, not two.
    """
    kept: list[float] = []
    for onset in sorted(float(o) for o in onsets_ms):
        if not kept or onset - kept[-1] > within_ms:
            kept.append(onset)
    return kept


def uncertain_onset_windows(onsets_ms, *, window_ms=ONSET_WINDOW_MS):
    """Merged uncertain pre-onset `(start, end)` windows with unverified extent."""
    merged: list[list[float]] = []
    for onset in sorted(float(o) for o in onsets_ms):
        start = onset - window_ms
        if merged and start <= merged[-1][1]:
            merged[-1][1] = onset
        else:
            merged.append([start, onset])
    return [(a, b) for a, b in merged]


def interval_onset_sources(category, start_ms, end_ms, source_onsets):
    """Original mode/time records contributing to an onset-derived interval.

    Merge within each NTOR exactly as the export does, retaining every original
    row of a merged event. Match the retained event's window by overlap, including
    uncertain pieces split by present-time precedence. ONSET_TIME is in ms.
    """
    if category not in (PRESENT, UNCERTAIN):
        return []
    anchors, found = {}, []
    for row in sorted(source_onsets, key=lambda r: (r["ntor"], r["t_ms"])):
        ntor, onset = int(row["ntor"]), float(row["t_ms"])
        if ntor not in anchors or onset - anchors[ntor] > MERGE_MS:
            anchors[ntor] = onset
        anchor = anchors[ntor]
        lo, hi = (
            (anchor, anchor + PRESENT_MS)
            if category == PRESENT
            else (anchor - ONSET_WINDOW_MS, anchor)
        )
        if max(start_ms, lo) < min(end_ms, hi):
            found.append(
                {
                    "NTOR": ntor,
                    "MODE_TYPE": str(row["mode_type"]),
                    "ONSET_TIME": onset,
                    "source": row["raw_source"],
                }
            )
    return sorted(found, key=lambda r: (r["ONSET_TIME"], r["NTOR"]))


def _subtract(span, holes):
    """`span` minus the union of `holes`, as a list of non-empty `(a, b)` pieces."""
    pieces = [list(span)]
    for lo, hi in sorted(holes):
        out = []
        for a, b in pieces:
            if hi <= a or lo >= b:
                out.append([a, b])
                continue
            if lo > a:
                out.append([a, lo])
            if hi < b:
                out.append([hi, b])
        pieces = out
    return [(a, b) for a, b in pieces if b > a]


def window_rows(
    shot,
    onsets_ms,
    flattop,
    *,
    assumed_absent,
    window_ms=ONSET_WINDOW_MS,
    horizon_ms=HORIZON_MS,
    post_ms=POST_MS,
):
    """Interval-table rows `(shot, category, t_start, t_end)` for one shot.

    Category 1 covers one minimal 10 ms slice after each onset; it does not measure
    mode duration. Category 2 is each uncertain pre-onset window. ONSET_TIME is not
    established as a detection/threshold time, so pre-onset presence is not inferred.
    With `assumed_absent=True`,
    category 0 covers flat-top time before the first onset's precursor, conditional
    on onset-list completeness; it never denotes verified coverage. Every remaining
    stretch is explicitly category 4 (unassessed), except later uncertain and
    minimal present windows. An onset list supplies no termination
    evidence.

    The rows tile the flat-top and any weak windows extending beyond it. With no
    flat-top they tile only the weak windows' bounding span. `post_ms` is retained
    for caller compatibility, but a forecast exclusion duration cannot establish
    physical recovery and does not change these state intervals.
    """
    onsets = sorted(float(o) for o in onsets_ms)
    windows = uncertain_onset_windows(onsets, window_ms=window_ms)
    present = uncertain_onset_windows(
        [o + PRESENT_MS for o in onsets], window_ms=PRESENT_MS
    )
    # Present evidence takes precedence when a later pre-onset window overlaps it.
    windows = [piece for span in windows for piece in _subtract(span, present)]
    occupied = windows + present
    rows = [(shot, UNCERTAIN, a, b) for a, b in windows]
    rows += [(shot, PRESENT, a, b) for a, b in present]
    bounds = occupied + ([flattop] if flattop is not None else [])
    if not bounds:
        return rows
    coverage = (min(a for a, _ in bounds), max(b for _, b in bounds))
    absent = []
    if assumed_absent and flattop is not None and onsets:
        end = min(flattop[1], onsets[0] - horizon_ms)
        if end > flattop[0]:
            absent = _subtract((flattop[0], end), occupied)
            rows += [(shot, ABSENT, a, b) for a, b in absent]
    rows += [
        (shot, UNASSESSED, a, b) for a, b in _subtract(coverage, occupied + absent)
    ]
    return sorted(rows, key=lambda r: (r[2], r[3]))
