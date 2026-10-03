"""Segment-level scores for dense labels: the MS-TCN segmental F1 and edit score.

Farha and Gall (2019) score an action segmentation by segments rather than frames: a
predicted segment is a true positive when it overlaps a not yet matched ground-truth
segment of the same class with intersection over union at or above 10, 25 or 50 %; the
edit score is the normalised Levenshtein distance between the two segment label
sequences. A frame-wise score hides flicker (a prediction that alternates every few
milliseconds can still be right on most bins), and these two do not.

Here the sequences are the shot's labelled bins only: a bin with no label (-1) is
skipped, and a segment does not run across a gap of unlabelled bins.
"""

from __future__ import annotations

import numpy as np

OVERLAPS = (0.10, 0.25, 0.50)


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of one class over consecutive labelled bins: ``(class, first bin, last bin +
    1)``."""
    labels = np.asarray(labels)
    out: list[tuple[int, int, int]] = []
    start = None
    for i in range(len(labels) + 1):
        cur = int(labels[i]) if i < len(labels) else -1
        if start is not None and (cur != int(labels[start]) or cur < 0):
            out.append((int(labels[start]), start, i))
            start = None
        if start is None and cur >= 0:
            start = i
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance between two label sequences."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def edit_score(
    truth: list[tuple[int, int, int]], pred: list[tuple[int, int, int]]
) -> float:
    """100 * (1 - distance / longer length) of the two segment label sequences."""
    a, b = [s[0] for s in truth], [s[0] for s in pred]
    longest = max(len(a), len(b))
    return 100.0 if longest == 0 else 100.0 * (1.0 - edit_distance(a, b) / longest)


def match_counts(
    truth: list[tuple[int, int, int]], pred: list[tuple[int, int, int]], overlap: float
) -> tuple[int, int, int]:
    """True positives, false positives and false negatives at one IoU threshold."""
    used = [False] * len(truth)
    tp = 0
    for cls, lo, hi in pred:
        best, pick = -1.0, -1
        for j, (c, a, b) in enumerate(truth):
            if c != cls:
                continue
            inter = min(hi, b) - max(lo, a)
            if inter <= 0:
                continue
            iou = inter / (max(hi, b) - min(lo, a))
            if iou > best and not used[j]:
                best, pick = iou, j
        if pick >= 0 and best >= overlap:
            used[pick] = True
            tp += 1
    return tp, len(pred) - tp, len(truth) - tp


def shot_counts(truth_bins: np.ndarray, pred_bins: np.ndarray) -> dict:
    """The counts and edit score of one shot.

    ``truth_bins`` has -1 where unlabelled; ``pred_bins`` is a dense class per bin and
    is cut to the labelled bins and split at gaps the same way.
    """
    keep = truth_bins >= 0
    pred = np.where(keep, pred_bins, -1)
    t_seg, p_seg = segments(truth_bins), segments(pred)
    out = {
        "edit": edit_score(t_seg, p_seg),
        "segments_true": len(t_seg),
        "segments_pred": len(p_seg),
    }
    for ov in OVERLAPS:
        tp, fp, fn = match_counts(t_seg, p_seg, ov)
        (
            out[f"tp{int(ov * 100)}"],
            out[f"fp{int(ov * 100)}"],
            out[f"fn{int(ov * 100)}"],
        ) = tp, fp, fn
    return out


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    """Segmental F1 from summed counts; NaN with no segments at all."""
    denom = tp + 0.5 * (fp + fn)
    return float("nan") if denom == 0 else float(tp / denom)


def summarise(
    per_shot: list[dict], *, replicates: int = 1000, seed: int = 20261001
) -> dict:
    """F1@10/25/50 (counts summed over shots) and mean edit, with shot-bootstrap 95 %
    intervals."""
    keys = [f"{k}{int(o * 100)}" for o in OVERLAPS for k in ("tp", "fp", "fn")]
    counts = np.array([[s[k] for k in keys] for s in per_shot], dtype=np.float64)
    edit = np.array([s["edit"] for s in per_shot], dtype=np.float64)

    def score(rows: np.ndarray, e: np.ndarray) -> list[float]:
        total = rows.sum(axis=0).reshape(len(OVERLAPS), 3)
        return [f1_from_counts(*t) for t in total] + [float(e.mean())]

    point = score(counts, edit)
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            score(counts[i], edit[i])
            for i in (
                rng.integers(len(per_shot), size=len(per_shot))
                for _ in range(replicates)
            )
        ]
    )
    names = [f"f1_{int(o * 100)}" for o in OVERLAPS] + ["edit"]
    return {
        "shots": len(per_shot),
        **{
            n: {
                "value": point[i],
                "ci95": [
                    float(np.nanpercentile(draws[:, i], 2.5)),
                    float(np.nanpercentile(draws[:, i], 97.5)),
                ],
            }
            for i, n in enumerate(names)
        },
    }
