"""Each method's output on the reviewed shots, as `score.ShotScore`s.

A `ShotScore` holds a method's calls on one shot's scored bins. Three kinds of
method meet here:

* a **trace** method (`elm-ours`): a 1 ms event probability whose mean over a bin is
  the bin's score; a bin is called present at the fold's threshold, and its detected
  spans are the stretches where the 50 ms moving mean of the trace reaches it;
* a **row** method (`elm-dsm`): a score per grid row, each row summarising the 50 ms
  before its time stamp; a bin takes the score of one row (the caller says which),
  and a row at or above the threshold marks the 50 ms it summarises as detected;
* a **span** method (ELM-O, the ELM clock): detected spans, a bin is called when one
  touches it.

The span counts of the benchmark (`score.SPAN_KEYS`) are those of
`scripts/labeler/elmo_benchmark.bin_table`; `span_counts` re-states its span rule and
the benchmark script's output is the test of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import inputs, labels, onset, score

ROW_MS = 25.0
WINDOW_MS = 50.0


def cover_frame(cov0, cov1) -> pd.DataFrame:
    return pd.DataFrame(
        {"t_start_ms": np.asarray(cov0, float), "t_end_ms": np.asarray(cov1, float)}
    )


def span_frame(starts, stops) -> pd.DataFrame:
    return pd.DataFrame(
        {"t_start_ms": np.asarray(starts, float), "t_end_ms": np.asarray(stops, float)}
    )


def _touches(starts, stops, lo, hi) -> np.ndarray:
    """Which of the intervals `[lo, hi)` any of the sorted disjoint spans touches."""
    if not len(starts):
        return np.zeros(len(lo), dtype=bool)
    i = np.minimum(np.searchsorted(stops, lo, side="right"), len(starts) - 1)
    return (stops[i] > lo) & (starts[i] < hi)


def span_counts(
    spans: pd.DataFrame, cover: pd.DataFrame, shot_spans: pd.DataFrame
) -> dict:
    """The benchmark's span counts for detected `spans` on one shot.

    `spans` (columns `t_start_ms`, `t_end_ms`) are the method's detected spans, sorted
    and not overlapping; `shot_spans` the shot's review rows. A labelled span with less
    than half its length analysed (`cover`) is skipped; an individual or absent span
    counts as hit when any detected span touches it.
    """
    starts = spans.t_start_ms.to_numpy(float)
    stops = spans.t_end_ms.to_numpy(float)
    cov0, cov1 = labels.merge_intervals(
        cover.t_start_ms.to_numpy(float), cover.t_end_ms.to_numpy(float)
    )
    out = dict.fromkeys(score.SPAN_KEYS, 0)
    for row in shot_spans.itertuples():
        if row.kind not in labels.SCORED_KINDS or row.t_end <= row.t_start:
            continue
        analysed = np.clip(
            np.minimum(cov1, row.t_end) - np.maximum(cov0, row.t_start), 0, None
        ).sum()
        if analysed < 0.5 * (row.t_end - row.t_start):
            continue
        hit = bool(
            _touches(
                starts,
                stops,
                np.array([float(row.t_start)]),
                np.array([float(row.t_end)]),
            )[0]
        )
        if row.kind == "absent":
            out["absent_spans"] += 1
            out["absent_span_alarm"] += int(hit)
        elif row.kind == "crowd":
            out["crowd_spans"] += 1
        else:
            out["individual_spans"] += 1
            out["individual_span_hit"] += int(hit)
    return out


def inside(x_ms, cov0, cov1) -> np.ndarray:
    """Which times of `x_ms` lie in analysed time."""
    x = np.asarray(x_ms, dtype=float)
    k = np.searchsorted(cov0, x, side="right") - 1
    return (k >= 0) & (x < np.asarray(cov1)[np.maximum(k, 0)])


def onset_counts(found_ms, shot_spans, onset_mask, n_ms, cov0, cov1, tol):
    """`(tp, fp, fn)` of detected onsets against the shot's reviewed individual starts,
    counted inside analysed time only."""
    ind = shot_spans[shot_spans.kind == "individual"].t_start.to_numpy(float)
    truth = ind[inside(ind, cov0, cov1)]
    found = np.asarray(found_ms, dtype=float)
    found = found[inside(found, cov0, cov1)]
    cells = inputs.GRID0_MS + np.arange(n_ms) + 0.5
    defined = onset_mask & inside(cells, cov0, cov1)
    return onset.match(found, truth, defined, inputs.GRID0_MS, tol)


def onset_summary(per_shot: np.ndarray, boot: np.ndarray) -> dict:
    """Pooled onset precision, recall and F1 of `(shots, 3)` tp/fp/fn rows, with intervals."""

    def rates(total):
        tp, fp, fn = total
        return {
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        }

    point = rates(per_shot.sum(axis=0))
    reps = [rates(per_shot[d].sum(axis=0)) for d in boot]
    ci = {k: score._ci(np.array([r[k] for r in reps])) for k in point}
    tp, fp, fn = (int(v) for v in per_shot.sum(axis=0))
    return {"counts": {"tp": tp, "fp": fp, "fn": fn}, "point": point, "ci95": ci}


class Oof:
    """The out-of-fold predictions of one `train` run and each shot's thresholds."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.record = json.loads((self.dir / "run.json").read_text())
        self.fold_of: dict[int, int] = {}
        self.threshold: dict[int, float] = {}
        self.onset_threshold: dict[int, float] = {}
        for k in range(len(self.record["folds"])):
            info = json.loads((self.dir / f"fold{k}" / "fold.json").read_text())
            for s in info["test"]:
                self.fold_of[s] = k
                self.threshold[s] = info["threshold"]
                self.onset_threshold[s] = info["onset_threshold"]

    def trace(self, shot: int) -> np.ndarray:
        with np.load(self.dir / "pred" / f"{shot}.npz") as z:
            return z["p"].astype(np.float32)


def trace_part(shot_spans, shot, bins, cover, trace, thr) -> score.ShotScore:
    """A trace method's part: bin means, calls at `thr`, spans from the moving mean."""
    s = labels.bin_scores(trace, bins)
    starts, stops = labels.runs_of(trace, thr)
    spans = span_counts(span_frame(starts, stops), cover, shot_spans)
    return score.ShotScore(shot, bins.truth, bins.kind, s >= thr, s, spans)


def row_part(shot_spans, shot, bins, cover, row_t_ms, row_score, thr, *, lag_rows=0):
    """A row method's part.

    `row_t_ms` are the rows' time stamps (each row summarises the `WINDOW_MS` before
    it); a bin's score is the row `lag_rows` rows after its end (0: the row summarising
    the bin itself) and `-WINDOW_MS / ROW_MS` rows is the row ending at its start.
    Bins with no such row are dropped from the part: the caller must give every
    method the same bins, so it passes bins already restricted to the rows.
    """
    end = bins.t0 + WINDOW_MS + lag_rows * ROW_MS
    k = np.rint((end - row_t_ms[0]) / ROW_MS).astype(int)
    if k.size and (k.min() < 0 or k.max() >= len(row_t_ms)):
        raise ValueError("a scored bin has no row; restrict the bins first")
    s = np.asarray(row_score, dtype=float)[k]
    on = np.asarray(row_score) >= thr
    run0, run1 = _row_runs(row_t_ms, on)
    spans = span_counts(span_frame(run0, run1), cover, shot_spans)
    return score.ShotScore(shot, bins.truth, bins.kind, s >= thr, s, spans)


def _row_runs(row_t_ms, on):
    """Merged `[t - WINDOW_MS, t)` of the rows that are on."""
    t = np.asarray(row_t_ms, dtype=float)[np.asarray(on, dtype=bool)]
    out0: list[float] = []
    out1: list[float] = []
    for e in t:
        a = e - WINDOW_MS
        if out1 and a <= out1[-1]:
            out1[-1] = max(out1[-1], e)
        else:
            out0.append(a)
            out1.append(e)
    return np.array(out0), np.array(out1)


def span_part(shot_spans, shot, bins, cover, spans: pd.DataFrame) -> score.ShotScore:
    """A span method's part: a bin is called when a detected span touches it."""
    spans = spans.sort_values("t_start_ms")
    call = labels.hard_hits(
        spans.t_start_ms.to_numpy(float), spans.t_end_ms.to_numpy(float), bins
    )
    counts = span_counts(spans, cover, shot_spans)
    return score.ShotScore(shot, bins.truth, bins.kind, call, None, counts)


def restrict_bins(bins: labels.Bins, keep: np.ndarray) -> labels.Bins:
    return labels.Bins(
        bins.t0[keep], bins.truth[keep], bins.kind[keep], bins.span[keep]
    )
