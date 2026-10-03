"""Rank metrics for the 4-class confinement scores, in plain numpy.

``auroc`` and ``average_precision`` are the Mann-Whitney AUROC and the step-wise average
precision (the same definitions scikit-learn uses, which the labelmaker environment
lacks); ``one_vs_rest`` gives them per class and averaged over the classes that occur.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

CLASSES = ("L", "H", "QH", "WP")


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Chance that a random positive outranks a random negative (ties count half)."""
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    # average ranks over ties
    boundaries = np.flatnonzero(np.r_[True, np.diff(sorted_scores) != 0, True])
    for lo, hi in pairwise(boundaries):
        ranks[order[lo:hi]] = 0.5 * (lo + hi + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(score: np.ndarray, positive: np.ndarray) -> float:
    """Sum over thresholds of (recall step) x precision, descending score."""
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hit = positive[order]
    # one operating point per distinct score
    last = np.flatnonzero(np.r_[np.diff(score[order]) != 0, True])
    tp = np.cumsum(hit)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def one_vs_rest(probs: np.ndarray, truth: np.ndarray) -> dict:
    """Per-class and macro AUROC and AP of ``probs`` (n, 4) against ``truth``."""
    out: dict = {"auroc": {}, "auprc": {}}
    for i, c in enumerate(CLASSES):
        positive = truth == i
        out["auroc"][c] = auroc(probs[:, i], positive)
        out["auprc"][c] = average_precision(probs[:, i], positive)
    for key in ("auroc", "auprc"):
        values = [v for v in out[key].values() if np.isfinite(v)]
        out[key]["macro"] = float(np.mean(values)) if values else float("nan")
    return out
