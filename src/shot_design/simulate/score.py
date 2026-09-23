"""Scores for an ensemble of rollouts against the measurement, numpy only.

``members`` is (M, F, C): M rollouts of one decoded feature over F frames and C channels.
``obs`` (F, C) is the same feature decoded from the measured codes, so the model is scored
in its codecs' space and the codecs' own reconstruction error does not count against it.
"""

from __future__ import annotations

import numpy as np

# An edit is resolved when it moves the ensemble mean at least this many noise floors.
RESOLVED = 2.0


def crps(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-element CRPS, E|X - y| - E|X - X'| / 2, the second mean over distinct pairs.

    This "fair" estimator is unbiased for the CRPS of the distribution the members are
    drawn from, whatever M (Ferro 2014), so ensembles of any size and a point forecast
    (M = 1, where it is the absolute error) compare directly.
    """
    m = members.shape[0]
    err = np.abs(members - obs).mean(axis=0)
    if m == 1:
        return err
    x = np.sort(members, axis=0)
    rank = (2 * np.arange(m) - m + 1).reshape((m,) + (1,) * (x.ndim - 1))
    return err - (rank * x).sum(axis=0) / (m * (m - 1))


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def spread_error(members: np.ndarray, obs: np.ndarray) -> float | None:
    """Ensemble spread over the error of the ensemble mean; 1 for a reliable ensemble.

    The sqrt((M + 1) / M) factor corrects for a finite ensemble (Fortin et al. 2014).
    Below 1 the ensemble is overconfident, above 1 it is wider than its errors.
    """
    m = members.shape[0]
    spread = float(np.sqrt((m + 1) / m * members.var(axis=0, ddof=1).mean()))
    return _ratio(spread, rmse(members.mean(axis=0), obs))


def effect(a: np.ndarray, b: np.ndarray) -> float:
    """RMS over frames and channels of the difference between two ensembles' means."""
    return rmse(a.mean(axis=0), b.mean(axis=0))


def modality_scores(gt: np.ndarray, arms: dict[str, np.ndarray], k0: int) -> dict:
    """One modality's entry in ``metrics.json``.

    ``gt`` (F, C) covers the whole window and ``arms`` {name: (M, F, C)} the same frames;
    frames [k0, F) are scored. Persistence holds the last seed frame, the seed mean holds
    the seed's average, and nRMSE divides by the measured standard deviation over the
    predicted frames (1 is no better than knowing the window's own mean). With a
    ``proposed`` and a ``null`` arm, the edit's effect is set against the noise floor.
    """
    obs = gt[k0:]
    persistence = np.broadcast_to(gt[k0 - 1], obs.shape)
    seed_mean = np.broadcast_to(gt[:k0].mean(axis=0), obs.shape)
    real = arms["real"][:, k0:]
    scale = float(obs.std())
    crps_real = float(crps(real, obs).mean())
    crps_persistence = float(np.abs(persistence - obs).mean())
    out = {
        "nrmse": {
            "real": _ratio(rmse(real.mean(axis=0), obs), scale),
            "persistence": _ratio(rmse(persistence, obs), scale),
            "seed_mean": _ratio(rmse(seed_mean, obs), scale),
        },
        "crps": {"real": crps_real, "persistence": crps_persistence},
        "skill": _skill(crps_real, crps_persistence),
        "spread_error": spread_error(real, obs) if real.shape[0] > 1 else None,
    }
    if "proposed" in arms and "null" in arms:
        size = effect(arms["proposed"][:, k0:], real)
        noise = effect(arms["null"][:, k0:], real)
        ratio = _ratio(size, noise)
        out.update(
            effect=size,
            noise=noise,
            effect_to_noise=ratio,
            resolved=ratio is not None and ratio >= RESOLVED,
        )
    return out


def _ratio(a: float, b: float) -> float | None:
    return a / b if b > 0 else None


def _skill(crps_model: float, crps_reference: float) -> float | None:
    """1 - CRPS / CRPS(reference): 1 is perfect, 0 no better than the reference."""
    return 1.0 - crps_model / crps_reference if crps_reference > 0 else None
