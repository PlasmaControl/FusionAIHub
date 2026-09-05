"""Numpy-only prior adjustment and isotonic probability calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def prior_shift(p: np.ndarray, *, from_prevalence: float,
                to_prevalence: float) -> np.ndarray:
    """Multiply prediction odds by target/source prevalence odds."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    q1, p1 = np.clip([from_prevalence, to_prevalence], 1e-6, 1 - 1e-6)
    ratio = (p1 / (1 - p1)) * ((1 - q1) / q1)
    # This odds form avoids an unnecessary log/exp round trip at the tails.
    return p * ratio / (1 - p + p * ratio)


@dataclass(frozen=True)
class IsotonicMap:
    x: np.ndarray
    y: np.ndarray
    n_fit: int
    prevalence_fit: float

    @classmethod
    def fit(cls, scores: np.ndarray, truth: np.ndarray) -> IsotonicMap:
        """Fit non-decreasing probabilities with squared-error PAVA."""
        return fit_isotonic(scores, truth)

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Interpolate finite scores with endpoint clamping; non-finite scores stay NaN."""
        scores = np.asarray(scores, dtype=float)
        return np.where(np.isfinite(scores), np.interp(scores, self.x, self.y), np.nan)

    def to_dict(self) -> dict:
        """JSON-compatible knots and fitting population statistics."""
        return {"x": self.x.tolist(), "y": self.y.tolist(), "n_fit": self.n_fit,
                "prevalence_fit": self.prevalence_fit}

    @classmethod
    def from_dict(cls, data: dict) -> IsotonicMap:
        """Restore a serialized map."""
        return cls(np.asarray(data["x"], float), np.asarray(data["y"], float),
                   int(data["n_fit"]), float(data["prevalence_fit"]))


def fit_isotonic(scores: np.ndarray, truth: np.ndarray) -> IsotonicMap:
    """Pool score ties first, then merge adjacent decreasing block means."""
    scores, truth = np.asarray(scores, float), np.asarray(truth, float)
    if (scores.ndim != 1 or scores.shape != truth.shape or not scores.size
            or not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all()):
        raise ValueError("fit requires nonempty finite scores and matching binary truth")
    x, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=truth)
    blocks = []
    for i, (total, count) in enumerate(zip(sums, counts, strict=True)):
        blocks.append((i, i + 1, total, count))
        while len(blocks) > 1:
            a, b = blocks[-2:]
            if a[2] / a[3] <= b[2] / b[3]:
                break
            blocks[-2:] = [(a[0], b[1], a[2] + b[2], a[3] + b[3])]
    y = np.empty(x.size)
    for start, end, total, count in blocks:
        y[start:end] = total / count
    return IsotonicMap(x, y, int(scores.size), float(truth.mean()))
