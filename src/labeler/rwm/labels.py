"""RWM labels from Jeremy Hanson's onset times.

An onset is a point in time. Two labels are made from it, and they are different
things:

* the **growth window** `[onset - GROWTH_MS, onset]`, the catalog's "RWM present"
  interval (category 1), short because the mode grows on the wall time;
* the **forecast target** of the baseline, as Piccione et al. 2022 define their
  stability label: a time slice is positive when an onset follows within
  `HORIZON_MS` (100 ms there), negative otherwise. Slices just after an onset,
  while the mode is acting, are excluded from both classes.

Only a shot that was examined can have a negative. The onset tables list onsets and
say nothing about shots they omit, so a comparison shot is unlabelled; the labels
of its slices stay `UNLABELLED` and the baselines treat them as such.
"""

from __future__ import annotations

import numpy as np

#: The paper's horizon: slices within 100 ms before the RWM time are positive.
HORIZON_MS = 100.0
#: Slices from an onset to this long after it are neither stable nor about to go
#: unstable; the mode is already acting, so they leave the training and scoring sets.
POST_MS = 100.0
#: Length of the catalog's "RWM present" window before an onset. DIII-D's wall time
#: tau_w is a few ms, so a mode growing on it gains a factor e per few ms: 20 ms is
#: two to four e-foldings. The measured largest trailing 20 ms growth rate of the
#: n = 1 RMS around the 48 distinct n = 1 onsets has a median e-folding time of 9 ms
#: (quartiles 5.8 and 11.0 ms; `outputs/labeler/rwm/growth.json`), which is 2.2
#: e-foldings (1.8 to 3.5) in 20 ms. The 1 kHz RMS itself cannot place the start of the
#: growth more sharply, so this length rests on the wall time and is a convention.
GROWTH_MS = 20.0

#: Listed onsets this close are one event.
MERGE_MS = 10.0

POSITIVE, NEGATIVE, EXCLUDED = 1, 0, -1
#: A slice of a shot whose onsets nobody listed: neither positive nor negative.
UNLABELLED = -2

#: Interval-table categories written by `window_rows`.
PRESENT, ABSENT, UNCERTAIN = 1, 0, 2


def slice_labels(
    t_ms, onsets_ms, *, horizon_ms=HORIZON_MS, post_ms=POST_MS, other_onsets_ms=()
):
    """Per-slice label: 1 an onset within the next `horizon_ms`, 0 none, -1 excluded.

    A slice at `t` is positive when some onset `o` has `o - horizon <= t < o`. It is
    excluded when some onset has `o <= t < o + post` and it is not positive (an onset
    that follows soon after another keeps its own positive slices). Every other slice
    is negative. With no onset every slice is negative; the caller decides whether
    that means "examined" or "unlabelled".

    `other_onsets_ms` are onsets of a different kind (an n = 2 RWM when the target is
    n = 1): their slices from `o - horizon` to `o + post` are excluded, never
    positive and never negative, unless a target onset makes them positive.
    """
    t = np.asarray(t_ms, dtype=float)
    label = np.zeros(t.shape, dtype=np.int8)
    for other in np.asarray(other_onsets_ms, dtype=float):
        label[(t >= other - horizon_ms) & (t < other + post_ms)] = EXCLUDED
    onsets = np.sort(np.asarray(onsets_ms, dtype=float))
    if not len(onsets):
        return label
    # First onset strictly after each slice time, and last at or before it.
    after = np.searchsorted(onsets, t, side="right")
    before = after - 1
    next_gap = np.where(
        after < len(onsets), onsets[np.minimum(after, len(onsets) - 1)] - t, np.inf
    )
    last_gap = np.where(before >= 0, t - onsets[np.maximum(before, 0)], np.inf)
    label[last_gap < post_ms] = EXCLUDED
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


def growth_windows(onsets_ms, *, growth_ms=GROWTH_MS):
    """Merged `(start, end)` growth windows, one per onset (overlaps joined)."""
    merged: list[list[float]] = []
    for onset in sorted(float(o) for o in onsets_ms):
        start = onset - growth_ms
        if merged and start <= merged[-1][1]:
            merged[-1][1] = onset
        else:
            merged.append([start, onset])
    return [(a, b) for a, b in merged]


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
    examined,
    growth_ms=GROWTH_MS,
    horizon_ms=HORIZON_MS,
    post_ms=POST_MS,
):
    """Interval-table rows `(shot, category, t_start, t_end)` for one shot.

    Category 1 is each growth window. For an `examined` shot (one with listed onsets)
    category 0 is the shot's flat-top `flattop = (start, end)` ms outside every
    `[onset - horizon, onset + post]`; the precursor stretch before a growth window
    and the aftermath of an onset carry no row, which the interval-table convention
    reads as not assessed. An unexamined shot has no category 0 row at all.
    """
    rows = [
        (shot, PRESENT, a, b) for a, b in growth_windows(onsets_ms, growth_ms=growth_ms)
    ]
    if examined and flattop is not None:
        holes = [(o - horizon_ms, o + post_ms) for o in onsets_ms]
        rows += [(shot, ABSENT, a, b) for a, b in _subtract(flattop, holes)]
    return sorted(rows, key=lambda r: (r[2], r[3]))
