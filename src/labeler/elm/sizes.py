"""Large or small: one detected ELM's size from the detector's input, for its type.

The challenge merit function prices a detection by the type it is reported as
(`spikes.prices`). Reported as a small ELM (type 2), a found large ELM costs 5.0 and a
false detection 3.0; reported as large (type 1), a found small ELM costs 3.0 and a
false detection 10.0. So the type is a decision on three probabilities, no event,
large and small (`spikes.report_type`), and this module gives two of them from three
measures of the detection's size (`features`):

* `rise`: the filterscope level's maximum over the 1 ms after the onset against its
  median 1.5 to 0.5 ms before, in decades;
* `contrast`: that maximum against the level's 0.5 s running median (the input's
  contrast rows), in decades;
* `density_drop`: the high-passed line density's mean over the 1 ms before against
  its minimum over the 5 ms after, in the input's density unit (1e14 of the chord's
  native ordinate), averaged over the chords the input screen kept.

The rise and the contrast are the median over FS02-FS04. None of the three uses the
distance to another ELM, but in a dense train the rise's baseline, 1.5 to 0.5 ms
before the onset, can sit on the last ELM's decay and lower the rise. The probabilities come from
two logistic regressions (`fit`) with a light ridge penalty on standardised features;
a measure the input cannot give (an invalid stretch, no usable chord) takes the
training median.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from . import inputs, inst

FEATURES = ("rise", "contrast", "density_drop")
BEFORE_MS = (-1.5, -0.5)
AFTER_MS = 1.0
DROP_BEFORE_MS = 1.0
DROP_AFTER_MS = 5.0
RIDGE = 1e-2


def _cells(ms: float) -> int:
    return round(ms / inputs.DT_MS)


def features(x: np.ndarray, onsets_ms) -> np.ndarray:
    """`(n, len(FEATURES))` size measures of the onsets `onsets_ms` in one shot's input
    `x` (`inputs.channels`); NaN where the input cannot give one."""
    t = np.asarray(onsets_ms, dtype=float)
    out = np.full((len(t), len(FEATURES)), np.nan)
    valid = x[inputs.VALID] > 0
    level = np.where(valid, x[list(inputs.FS_LEVEL)], np.nan) * inputs.FS_SCALE
    contrast = np.where(valid, x[list(inputs.FS_CONTRAST)], np.nan) * inputs.FS_SCALE
    chords = [r for r in inputs.DENSITY_HP if np.any(x[r] != 0)]
    hp = np.where(valid, x[chords], np.nan) / inputs.HP_GAIN if chords else None
    n = x.shape[1]
    b0, b1 = _cells(BEFORE_MS[0]), _cells(BEFORE_MS[1])
    a1 = _cells(AFTER_MS) + 1
    d0, d1 = _cells(DROP_BEFORE_MS), _cells(DROP_AFTER_MS) + 1
    with warnings.catch_warnings():
        # an all-NaN slice (an invalid stretch) gives NaN, as wanted
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, c in enumerate(inst.cell_of(t)):
            if c < 0 or c >= n:
                continue
            after = np.nanmax(level[:, c : c + a1], axis=1)
            before = np.nanmedian(level[:, max(0, c + b0) : max(0, c + b1)], axis=1)
            out[i, 0] = np.nanmedian(after - before)
            out[i, 1] = np.nanmedian(np.nanmax(contrast[:, c : c + a1], axis=1))
            if hp is not None:
                pre = np.nanmean(hp[:, max(0, c - d0) : c], axis=1)
                post = np.nanmin(hp[:, c : c + d1], axis=1)
                out[i, 2] = np.nanmean(pre - post)
    return out


def balanced(y) -> np.ndarray:
    """Sample weights giving each class of the 0/1 labels `y` half the total."""
    y = np.asarray(y, dtype=bool)
    w = np.empty(len(y))
    w[y] = 0.5 / max(int(y.sum()), 1)
    w[~y] = 0.5 / max(int((~y).sum()), 1)
    return w * len(y)


def fit(X: np.ndarray, y, weight=None, ridge: float = RIDGE) -> dict:
    """A logistic regression of the 0/1 labels `y` on `X`, as a JSON-able dict: NaN
    filled with the column's median, columns standardised, `ridge` on the slopes."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    w = w / w.mean()
    fill = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), fill, X)
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mean) / scale

    def loss(theta):
        z = theta[0] + Z @ theta[1:]
        p = 1 / (1 + np.exp(-z))
        value = np.mean(w * (np.logaddexp(0, z) - y * z))
        value += 0.5 * ridge * theta[1:] @ theta[1:]
        g = w * (p - y) / len(y)
        grad = np.concatenate([[g.sum()], Z.T @ g + ridge * theta[1:]])
        return value, grad

    res = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"logistic fit did not converge: {res.message}")
    return {
        "fill": fill.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(res.x[0]),
        "coef": res.x[1:].tolist(),
    }


def prob(model: dict, X: np.ndarray) -> np.ndarray:
    """The probability of label 1 under a `fit` model."""
    X = np.asarray(X, dtype=float)
    X = np.where(np.isnan(X), np.asarray(model["fill"]), X)
    Z = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    z = model["intercept"] + Z @ np.asarray(model["coef"])
    return 1 / (1 + np.exp(-z))


def auroc(score, y) -> float:
    """Area under the ROC curve of `score` for the 0/1 labels `y` (ties count half)."""
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
