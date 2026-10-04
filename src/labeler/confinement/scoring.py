"""Rank metrics for the 4-class confinement scores, in plain numpy.

``auroc`` and ``average_precision`` are the Mann-Whitney AUROC and the step-wise average
precision (the same definitions scikit-learn uses, which the labelmaker environment
lacks); ``one_vs_rest`` gives them per class and averaged over the classes that occur.
"""

from __future__ import annotations

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
    fresh = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    first = np.flatnonzero(fresh)
    last = np.r_[first[1:], len(score)]
    ranks[order] = (0.5 * (first + last + 1))[np.cumsum(fresh) - 1]
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


def rank_with_ci(
    probs: np.ndarray,
    truth: np.ndarray,
    shots: np.ndarray,
    *,
    replicates: int = 1000,
    seed: int = 20261001,
) -> dict:
    """``one_vs_rest`` of the windows plus 95 % shot-bootstrap intervals of the macro
    AUROC and AUPRC (a replicate draws the shots with replacement and keeps all of each
    drawn shot's windows)."""
    out = one_vs_rest(probs, truth)
    ids = np.unique(shots)
    order = np.argsort(shots, kind="stable")
    bounds = np.searchsorted(shots[order], ids)
    ends = np.r_[bounds[1:], len(order)]
    members = [order[a:b] for a, b in zip(bounds, ends, strict=True)]
    rng = np.random.default_rng(seed)
    macro = {"auroc": [], "auprc": []}
    for _ in range(replicates):
        pick = np.concatenate(
            [members[i] for i in rng.integers(len(ids), size=len(ids))]
        )
        draw = one_vs_rest(probs[pick], truth[pick])
        for key, values in macro.items():
            values.append(draw[key]["macro"])
    out["ci95"] = {
        key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for key, v in macro.items()
    }
    out["replicates"] = replicates
    return out
