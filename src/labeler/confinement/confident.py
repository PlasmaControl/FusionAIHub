"""Confident learning (Northcutt, Jiang and Chuang, 2021) for the confinement labels.

A classifier's held-out probabilities mark where the label disagrees with what the
signal says. Per class ``j`` the threshold ``t_j`` is the mean predicted probability of
class ``j`` over the windows labelled ``j``; a window labelled ``i`` is *confidently* of
class ``j`` when ``p_j >= t_j`` (the largest such ``p_j`` when several are). Counting
those gives the confident joint ``C[i, j]``; its off-diagonal is the estimate of label
noise. This is the algorithm cleanlab implements; cleanlab is not installed, so it is
written out here.

The labels are intervals, not windows, so ``interval_issues`` aggregates the window
verdicts to the interval: the share of its (non-margin) windows that are confidently of
another class, and which class. Windows near an interval end are left out, since a
transition makes the next regime show early or the last one linger, and the analysis is
done per label source because a label an expert gave on its own is not the same kind of
claim as one two sources agree on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("L", "H", "QH", "WP")


def class_thresholds(
    probs: np.ndarray, labels: np.ndarray, n_classes: int = 4
) -> np.ndarray:
    """Mean self-confidence per class (NaN for a class with no window)."""
    t = np.full(n_classes, np.nan)
    for j in range(n_classes):
        sel = labels == j
        if sel.any():
            t[j] = probs[sel, j].mean()
    return t


def confident_argmax(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """For each window the class it is confidently of, -1 if no class clears its
    threshold."""
    above = probs >= np.where(np.isfinite(thresholds), thresholds, np.inf)[None, :]
    masked = np.where(above, probs, -np.inf)
    best = masked.argmax(axis=1)
    return np.where(above.any(axis=1), best, -1)


def confident_joint(
    probs: np.ndarray, labels: np.ndarray, n_classes: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Confident joint ``C[given, confident]`` and each window's confident class."""
    t = class_thresholds(probs, labels, n_classes)
    guess = confident_argmax(probs, t)
    joint = np.zeros((n_classes, n_classes), dtype=np.int64)
    sel = guess >= 0
    np.add.at(joint, (labels[sel], guess[sel]), 1)
    return joint, guess


def calibrate_joint(joint: np.ndarray, label_counts: np.ndarray) -> np.ndarray:
    """Northcutt's calibrated joint: rows scaled to the given-label counts, whole
    matrix scaled to their total, so that it estimates the joint of given and true
    labels over all windows rather than over the confidently placed ones."""
    joint = joint.astype(np.float64)
    rows = joint.sum(axis=1, keepdims=True)
    scaled = np.divide(
        joint * label_counts[:, None], rows, out=np.zeros_like(joint), where=rows > 0
    )
    total = scaled.sum()
    return scaled * (label_counts.sum() / total) if total > 0 else scaled


def interval_issues(
    windows: pd.DataFrame,
    probs: np.ndarray,
    *,
    margin_ms: float = 20.0,
    min_windows: int = 5,
    min_share: float = 0.5,
) -> pd.DataFrame:
    """Per interval: how many of its windows are confidently another class, and which.

    ``windows`` has ``shot``, ``interval``, ``label`` and ``keep`` (False for windows to
    leave out, e.g. inside ``margin_ms`` of an interval end). An interval is flagged
    when at least ``min_windows`` windows are kept and at least ``min_share`` of them
    are confidently of one other class.
    """
    labels = windows.label.to_numpy()
    thresholds = class_thresholds(
        probs[windows.keep.to_numpy()], labels[windows.keep.to_numpy()]
    )
    guess = confident_argmax(probs, thresholds)
    frame = windows.assign(confident=guess)
    rows = []
    for (shot, interval), g in frame[frame.keep].groupby(["shot", "interval"]):
        given = int(g.label.iloc[0])
        counts = np.bincount(g.confident[g.confident >= 0], minlength=len(CLASSES))
        other = counts.copy()
        other[given] = 0
        top = int(other.argmax())
        n = len(g)
        rows.append(
            {
                "shot": int(shot),
                "interval": int(interval),
                "given": CLASSES[given],
                "windows": n,
                "share_given": counts[given] / n,
                "share_other": other[top] / n,
                "other": CLASSES[top] if other[top] else "",
                "flagged": bool(n >= min_windows and other[top] / n >= min_share),
            }
        )
    return pd.DataFrame(rows)
