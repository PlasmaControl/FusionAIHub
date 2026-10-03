"""Non-crowd-span onsets from the onset head: peaks, and their match to the review.

The onset head's trace is a probability per 1 ms of the shot's clock that an
non-crowd span starts there (the target is a Gaussian of `labels.ONSET_SIGMA_MS`
at each reviewed non-crowd span's start). A detected onset is a local maximum of
the trace above a threshold, at least `MIN_SEP_MS` from a higher one. The match to
the review follows the reviewed onsets only where they are defined
(`labels.Dense.onset_mask`: absent and non-crowd spans, not crowds, whose ELMs are
unmarked):

* a reviewed onset (the start of an non-crowd span) is found when a detected one
  lies within `tol` ms of it, each detected onset used once;
* a detected onset is false when it is not matched and lies where onsets are
  defined, so in an absent span or in an non-crowd span away from its start.

Detections in crowds, uncertain or unlabelled time are neither counted nor
penalised.
"""

from __future__ import annotations

import numpy as np

MIN_SEP_MS = 10
TOLERANCES_MS = (5.0, 10.0)


def peaks(trace: np.ndarray, threshold: float, min_sep: int = MIN_SEP_MS) -> np.ndarray:
    """Indices of the local maxima of `trace` at or above `threshold`, thinned so
    that none lies within `min_sep` cells of a higher one."""
    x = np.asarray(trace, dtype=np.float64)
    if x.size < 3:
        return np.zeros(0, dtype=np.int64)
    inner = (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= threshold)
    cand = np.flatnonzero(inner) + 1
    keep: list[int] = []
    for i in cand[np.argsort(-x[cand], kind="stable")]:
        if all(abs(int(i) - j) >= min_sep for j in keep):
            keep.append(int(i))
    return np.array(sorted(keep), dtype=np.int64)


def match(
    found_ms: np.ndarray,
    truth_ms: np.ndarray,
    defined: np.ndarray,
    t0_ms: float,
    tol: float,
) -> tuple[int, int, int]:
    """`(tp, fp, fn)` of detected onsets `found_ms` against the reviewed `truth_ms`.

    `defined` is `labels.Dense.onset_mask`, on the 1 ms grid starting at `t0_ms`.
    """
    found = np.sort(np.asarray(found_ms, dtype=float))
    truth = np.sort(np.asarray(truth_ms, dtype=float))
    used = np.zeros(found.size, dtype=bool)
    tp = 0
    for t in truth:
        d = np.abs(found - t)
        d[used] = np.inf
        if d.size and d.min() <= tol:
            used[int(np.argmin(d))] = True
            tp += 1
    cell = np.floor(found - t0_ms).astype(int)
    inside = (cell >= 0) & (cell < len(defined))
    in_defined = np.zeros(found.size, dtype=bool)
    in_defined[inside] = defined[cell[inside]]
    fp = int((~used & in_defined).sum())
    return tp, fp, int(truth.size - tp)


def best_threshold(
    traces: list[np.ndarray],
    truths: list[np.ndarray],
    defined: list[np.ndarray],
    t0_ms: float,
    tol: float,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """The threshold maximising pooled onset F1 over shots, and that F1."""
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    best = (float(grid[0]), -1.0)
    for thr in grid:
        tp = fp = fn = 0
        for x, truth, mask in zip(traces, truths, defined):
            found = peaks(x, thr) + t0_ms + 0.5
            a, b, c = match(found, truth, mask, t0_ms, tol)
            tp, fp, fn = tp + a, fp + b, fn + c
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[1]:
            best = (float(thr), float(f1))
    return best
