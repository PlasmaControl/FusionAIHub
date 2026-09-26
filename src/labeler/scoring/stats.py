"""Rates, kappas, shot weights and the stratified shot bootstrap.

Every metric here is a function of summed cells: per-shot counts, weighted and
added up. The cells are

- precision, recall, F1: `[tp, fp, fn]` (a fourth, `tn`, is ignored);
- Cohen's kappa: `[n00, n01, n10, n11]`, reader A's state first;
- Fleiss' kappa: `[n_frames, agreeing_pairs, present_votes]` (see `fleiss_cells`).

A metric takes the totals, shape `(..., k)`, and returns one value per leading
index, so the bootstrap evaluates all its replicates in one call. An undefined
value (nothing to divide by) is NaN.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

N_REPLICATES = 2000
LEVEL = 0.95


def _ratio(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def precision(totals):
    t = np.asarray(totals, dtype=float)
    return _ratio(t[..., 0], t[..., 0] + t[..., 1])


def recall(totals):
    t = np.asarray(totals, dtype=float)
    return _ratio(t[..., 0], t[..., 0] + t[..., 2])


def f1(totals):
    t = np.asarray(totals, dtype=float)
    return _ratio(2 * t[..., 0], 2 * t[..., 0] + t[..., 1] + t[..., 2])


def cohen_cells(frames) -> np.ndarray:
    """`[n00, n01, n10, n11]` from `(n_frames, 2)` 0/1 frames."""
    frames = np.asarray(frames, dtype=np.int64).reshape(-1, 2)
    return np.bincount(frames[:, 0] * 2 + frames[:, 1], minlength=4).astype(float)


def cohen_kappa(totals):
    t = np.asarray(totals, dtype=float)
    n = t.sum(axis=-1)
    agree = _ratio(t[..., 0] + t[..., 3], n)
    a = _ratio(t[..., 2] + t[..., 3], n)
    b = _ratio(t[..., 1] + t[..., 3], n)
    chance = a * b + (1 - a) * (1 - b)
    return _ratio(agree - chance, 1 - chance)


def fleiss_cells(frames) -> np.ndarray:
    """`[n_frames, agreeing_pairs, present_votes]` from `(n_frames, k)` 0/1 frames.

    `agreeing_pairs` sums, over frames, the ordered pairs of readers who agree:
    `p(p - 1) + a(a - 1)` for `p` present and `a` absent votes.
    """
    frames = np.asarray(frames, dtype=np.int64)
    present = frames.sum(axis=1)
    absent = frames.shape[1] - present
    pairs = present * (present - 1) + absent * (absent - 1)
    return np.array([len(frames), pairs.sum(), present.sum()], dtype=float)


def fleiss_kappa(totals, k: int):
    t = np.asarray(totals, dtype=float)
    n, pairs, present = t[..., 0], t[..., 1], t[..., 2]
    observed = _ratio(pairs, n * k * (k - 1))
    p = _ratio(present, n * k)
    chance = p**2 + (1 - p) ** 2
    return _ratio(observed - chance, 1 - chance)


def stratum_weights(strata: Sequence, population: Mapping) -> np.ndarray:
    """`N_h / n_h` per shot: `N_h` from `population`, `n_h` counted in `strata`."""
    strata = list(strata)
    counts = Counter(strata)
    missing = sorted(set(counts) - set(population), key=str)
    if missing:
        raise ValueError(f"no population count for strata {missing}")
    return np.array([population[h] / counts[h] for h in strata], dtype=float)


def replicate_weights(
    strata: Sequence, weights, n: int = N_REPLICATES, seed: int = 0
) -> np.ndarray:
    """`(n, n_shots)`: each replicate's weight per shot.

    Shots are drawn with replacement within their stratum, as many as the
    stratum holds; a shot's replicate weight is its weight times its draws.
    """
    strata = list(strata)
    codes = {h: i for i, h in enumerate(sorted(set(strata), key=str))}
    ids = np.array([codes[h] for h in strata], dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.zeros((n, len(strata)))
    for code in range(len(codes)):
        members = np.flatnonzero(ids == code)
        m = len(members)
        picks = rng.integers(0, m, size=(n, m)) + m * np.arange(n)[:, None]
        draws[:, members] = np.bincount(picks.ravel(), minlength=n * m).reshape(n, m)
    return draws * weights[None, :]


@dataclass(frozen=True)
class Estimate:
    """A weighted point estimate and its percentile interval."""

    value: float
    low: float
    high: float
    undefined: int  # replicates where the metric had nothing to divide by

    def as_json(self) -> dict:
        return {
            "value": _plain(self.value),
            "low": _plain(self.low),
            "high": _plain(self.high),
            "undefined_replicates": self.undefined,
        }


def _plain(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def _interval(replicates, level: float) -> tuple[float, float, int]:
    replicates = np.asarray(replicates, dtype=float)
    ok = np.isfinite(replicates)
    if not ok.any():
        return np.nan, np.nan, len(replicates)
    tail = 50 * (1 - level)
    low, high = np.percentile(replicates[ok], [tail, 100 - tail])
    return float(low), float(high), int((~ok).sum())


def estimate(
    cells,
    strata: Sequence,
    weights,
    metric: Callable,
    *,
    n: int = N_REPLICATES,
    seed: int = 0,
    level: float = LEVEL,
) -> Estimate:
    """`metric` of the weighted totals of `cells` `(n_shots, k)`, with its interval."""
    cells = np.asarray(cells, dtype=float)
    weights = np.asarray(weights, dtype=float)
    value = float(metric(weights @ cells))
    replicates = metric(replicate_weights(strata, weights, n, seed) @ cells)
    low, high, undefined = _interval(replicates, level)
    return Estimate(value, low, high, undefined)


def difference(
    cells_a,
    cells_b,
    strata: Sequence,
    weights,
    metric: Callable,
    **options,
) -> Estimate:
    """`metric(a) - metric(b)` on the same resampled shots (paired)."""
    a = np.asarray(cells_a, dtype=float)
    b = np.asarray(cells_b, dtype=float)
    k = a.shape[1]

    def paired(totals):
        return metric(totals[..., :k]) - metric(totals[..., k:])

    return estimate(np.hstack([a, b]), strata, weights, paired, **options)


def weighted_median(values, weights):
    """Weighted median of `values`; `weights` may carry leading replicate axes."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(values):
        return np.full(weights.shape[:-1], np.nan) if weights.ndim > 1 else np.nan
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[..., order], axis=-1)
    total = cumulative[..., -1:]
    index = np.argmax(cumulative >= total / 2, axis=-1)
    return np.where(total[..., 0] > 0, values[order][index], np.nan)


def median_estimate(
    values,
    owners,
    strata: Sequence,
    weights,
    *,
    n: int = N_REPLICATES,
    seed: int = 0,
    level: float = LEVEL,
) -> Estimate:
    """Weighted median of per-event `values`, each owned by shot `owners[i]`."""
    values = np.asarray(values, dtype=float)
    owners = np.asarray(owners, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    value = float(weighted_median(values, weights[owners]))
    shots = replicate_weights(strata, weights, n, seed)
    low, high, undefined = _interval(weighted_median(values, shots[:, owners]), level)
    return Estimate(value, low, high, undefined)
