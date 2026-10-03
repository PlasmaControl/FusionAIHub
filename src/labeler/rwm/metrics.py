"""Slice metrics and the shot bootstrap, without scikit-learn."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _level_ends(sorted_scores):
    """Indices of the last row of each distinct score in a sorted array (inf ties too)."""
    s = np.asarray(sorted_scores)
    return np.flatnonzero(np.concatenate([s[1:] != s[:-1], [True]]))


def auroc(score, label):
    """Area under the ROC curve (ties get half credit); NaN with one class only."""
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(score, label):
    """Average precision: the sum of precision at each score level times its recall step."""
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    n_pos = int(label.sum())
    if not n_pos or n_pos == len(label):
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, y = score[order], label[order]
    last = _level_ends(s)
    true_pos = np.cumsum(y)[last]
    seen = last + 1
    recall = true_pos / n_pos
    step = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(step * true_pos / seen))


def roc_cutoff(score, label):
    """The score cutoff whose ROC point is nearest (0 FPR, 1 TPR), as in the paper.

    A slice is called positive when `score >= cutoff`. NaN with one class only.
    """
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    s, y = score[order], label[order]
    last = _level_ends(s)
    tpr = np.cumsum(y)[last] / n_pos
    fpr = np.cumsum(~y)[last] / n_neg
    return float(s[last][np.argmin(fpr**2 + (1 - tpr) ** 2)])


def confusion(score, label, cutoff):
    """`(tpr, fpr, precision, f1)` of the rule `score >= cutoff`."""
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    called = score >= cutoff
    tp, fp = int((called & label).sum()), int((called & ~label).sum())
    fn, tn = int((~called & label).sum()), int((~called & ~label).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return tpr, fpr, precision, f1


def _interval(values, point, level, method="percentile"):
    """`{"estimate", "low", "high"}` with a percentile or reflected basic interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": point, "low": float("nan"), "high": float("nan")}
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(values, [tail, 100 - tail])
    if method == "basic":
        low, high = 2 * point - high, 2 * point - low
    return {"estimate": point, "low": float(low), "high": float(high)}


def _summarise(estimate, draws, level, method="percentile"):
    if isinstance(estimate, dict):
        return {
            k: _interval([d[k] for d in draws], estimate[k], level, method)
            for k in estimate
        }
    return _interval(draws, estimate, level, method)


def shot_bootstrap(groups, statistic, *, replicates=1000, seed=0, level=0.95):
    """Percentile interval of `statistic` over shots resampled with replacement.

    `groups` maps a stratum name to a list of per-shot records; each stratum is
    resampled on its own, so the number of shots of each kind is the observed one.
    `statistic` takes `{stratum: [records]}` and returns a float or a dict of floats.
    Returns `{"estimate", "low", "high"}`, or one such entry per key of a dict result.
    """
    rng = np.random.default_rng(seed)
    estimate = statistic(groups)
    draws = []
    for _ in range(replicates):
        sample = {
            name: [records[i] for i in rng.integers(0, len(records), len(records))]
            for name, records in groups.items()
            if records
        }
        draws.append(statistic(sample))
    return _summarise(estimate, draws, level)


def _difference(first, second):
    if isinstance(first, dict):
        return {k: first[k] - second[k] for k in first}
    return first - second


def paired_bootstrap(
    groups_a,
    groups_b,
    statistic,
    *,
    replicates=1000,
    seed=0,
    level=0.95,
    method="percentile",
):
    """Interval of `statistic(a) - statistic(b)` over shared shot resamples.

    `groups_a` and `groups_b` score the same shots in the same order (two models on one
    cross-validation); every replicate draws one set of shots and applies it to both, so
    the interval is for the difference between the models, not for each on its own.
    The result has the shape of `shot_bootstrap`'s, for the difference `a - b`.
    `method="percentile"` preserves the default; `method="basic"` reflects the
    bootstrap percentile endpoints about the observed difference. Basic intervals
    can extend outside a metric's natural range and are not bias-corrected BCa.
    """
    if method not in {"percentile", "basic"}:
        raise ValueError("method must be 'percentile' or 'basic'")
    for name, records in groups_a.items():
        if len(records) != len(groups_b.get(name, [])):
            raise ValueError(f"stratum {name!r} holds different shots in the two sets")
    rng = np.random.default_rng(seed)
    estimate = _difference(statistic(groups_a), statistic(groups_b))
    draws = []
    for _ in range(replicates):
        picks = {
            name: rng.integers(0, len(records), len(records))
            for name, records in groups_a.items()
            if records
        }
        sample_a = {n: [groups_a[n][i] for i in idx] for n, idx in picks.items()}
        sample_b = {n: [groups_b[n][i] for i in idx] for n, idx in picks.items()}
        draws.append(_difference(statistic(sample_a), statistic(sample_b)))
    return _summarise(estimate, draws, level, method)
