"""A single D-alpha burst feature for the occupancy-development benchmark."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import inputs, labels


def bin_feature(x: np.ndarray, bins: labels.Bins) -> np.ndarray:
    """Maximum across FS02--04 of max minus median log D-alpha within each bin.

    Uses the same 0.1 ms maxima as the U-Net inputs. No density, running baseline,
    neighboring bins, shot labels, or fitted normalization enters this feature.
    """
    values = []
    width = int(labels.BIN_MS / inputs.DT_MS)
    for start in bins.t0:
        a = round((start - inputs.GRID0_MS) / inputs.DT_MS)
        b = a + width
        if a < 0 or b > x.shape[1] or not (x[inputs.VALID, a:b] > 0).all():
            raise ValueError("feature requires fully covered 50 ms bins")
        log_fs = x[list(inputs.FS_LEVEL), a:b] * inputs.FS_SCALE
        values.append(float((log_fs.max(axis=1) - np.median(log_fs, axis=1)).max()))
    return np.asarray(values, dtype=np.float64)


def fit_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L2 logistic regression with C=1, an unpenalized intercept, no class weight."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def objective(theta):
        logits = theta[0] * x + theta[1]
        loss = np.logaddexp(0, logits).sum() - np.dot(y, logits)
        residual = expit(logits) - y
        return loss + 0.5 * theta[0] ** 2, np.array(
            [np.dot(residual, x) + theta[0], residual.sum()]
        )

    result = minimize(objective, np.zeros(2), method="L-BFGS-B", jac=True)
    if not result.success:
        raise ValueError(f"logistic fit failed: {result.message}")
    return result.x


def logistic_scores(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Probability for the one-feature fit; no shot-level transform."""
    return expit(theta[0] * np.asarray(x) + theta[1])
