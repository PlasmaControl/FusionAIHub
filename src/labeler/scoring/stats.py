"""Rates, kappas, shot weights and the stratified shot bootstrap.

Every metric here is a function of summed cells: per-shot counts, weighted and
added up. The cells are

- precision, recall, F1: `[tp, fp, fn]` (a fourth, `tn`, is ignored);
- Cohen's kappa: `[n00, n01, n10, n11]`, reader A's state first;
- Fleiss' kappa: `[n_frames, agreeing_pairs, present_votes]` (see `fleiss_cells`).

A metric takes the totals, shape `(..., k)`, and returns one value per leading
index, so the bootstrap evaluates all its replicates in one call. An undefined
value (nothing to divide by) is NaN.

The cohort weight is N_h / n_h for a group-year cell. D25's blind weight is
that first-stage weight times n_g / b_g, the group's cohort-to-blind ratio;
the blind subset is resampled within its groups.

Intervals use the specified percentile bootstrap (95% by default). Simulation
coverage depends on the design: the reviewed cohort design covered 0.948-0.953,
and the smaller blind design 0.912-0.932. No coverage correction is applied.
The blind figure was simulated with group-only weights and a 20/10/20
allocation, before D25.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .frames import whole_number

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
    frames = _binary_frames(frames, cohen=True)
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
    frames = _binary_frames(frames)
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


def _binary_frames(frames, *, cohen=False):
    frames = np.asarray(frames)
    if frames.ndim != 2 or (frames.shape[1] != 2 if cohen else frames.shape[1] < 2):
        raise ValueError("frames must have shape (n_frames, 2) or (n_frames, k >= 2)")
    if not np.isin(frames, (0, 1)).all():
        raise ValueError("every frame entry must be 0 or 1")
    return frames.astype(np.int64)


def stratum_weights(strata: Sequence, population: Mapping) -> np.ndarray:
    """`N_h / n_h` per shot: `N_h` from `population`, `n_h` counted in `strata`."""
    strata = list(strata)
    counts = Counter(strata)
    missing = sorted(set(counts) - set(population), key=str)
    if missing:
        raise ValueError(f"no population count for strata {missing}")
    for h, count in population.items():
        try:
            count = whole_number(count)
        except ValueError as exc:
            raise ValueError(f"population stratum {h!r}: {exc}") from exc
        if count < counts[h]:
            raise ValueError(f"population stratum {h!r} has invalid count {count!r}")
    unscored = sorted(
        (h for h, count in population.items() if count > 0 and h not in counts),
        key=str,
    )
    if unscored:
        raise ValueError(f"no scored shot for population strata {unscored}")
    return np.array([population[h] / counts[h] for h in strata], dtype=float)


def two_stage_weights(
    first_stage, groups: Sequence, cohort_counts: Mapping
) -> np.ndarray:
    """D25 blind weights: cohort weight times the group's `n_g / b_g`."""
    groups = list(groups)
    weights = np.asarray(first_stage, dtype=float)
    if weights.ndim != 1 or len(weights) != len(groups):
        raise ValueError("first_stage and groups must have one entry per blind shot")
    counts = Counter(groups)
    for g, b in counts.items():
        if g not in cohort_counts:
            raise ValueError(f"no cohort count for group {g!r}")
        try:
            n = whole_number(cohort_counts[g])
        except ValueError as exc:
            raise ValueError(f"group {g!r}: {exc}") from exc
        if b > n:
            raise ValueError(f"group {g!r}: blind count {b} exceeds cohort count {n}")
    for g, weight in zip(groups, weights):
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"group {g!r}: invalid first-stage weight {weight!r}")
    return weights * np.array([cohort_counts[g] / counts[g] for g in groups])


def _shot_inputs(strata, weights):
    strata = list(strata)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(strata):
        raise ValueError("strata and weights must have one entry per shot")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    return strata, weights


def _replicate_count(n):
    n = whole_number(n)
    if n < 1:
        raise ValueError(f"replicates {n} must be at least 1")
    return n


def _check_level(level):
    if not 0 < level < 1:
        raise ValueError(f"level {level!r} must be strictly between 0 and 1")


def _seed(seed):
    try:
        value = whole_number(seed)
    except ValueError as exc:
        raise ValueError(f"seed: {exc}") from exc
    if value < 0:
        raise ValueError("seed must be nonnegative")
    return value


def replicate_weights(
    strata: Sequence, weights, n: int = N_REPLICATES, seed: int = 0
) -> np.ndarray:
    """`(n, n_shots)`: each replicate's weight per shot.

    Shots are drawn with replacement within their stratum, as many as the
    stratum holds; a shot's replicate weight is its weight times its draws.
    """
    strata, weights = _shot_inputs(strata, weights)
    n = _replicate_count(n)
    ordered = sorted(set(strata), key=str)
    for a, b in pairwise(ordered):
        if str(a) == str(b):
            raise ValueError(f"distinct strata {a!r} and {b!r} have the same str")
    codes = {h: i for i, h in enumerate(ordered)}
    ids = np.array([codes[h] for h in strata], dtype=np.int64)
    rng = np.random.default_rng(_seed(seed))
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
    replicates: int
    seed: int
    level: float

    def as_json(self) -> dict:
        return {
            "value": _plain(self.value),
            "low": _plain(self.low),
            "high": _plain(self.high),
            "undefined_replicates": self.undefined,
            "replicates": int(self.replicates),
            "seed": int(self.seed),
            "level": float(self.level),
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
    strata, weights = _shot_inputs(strata, weights)
    if cells.ndim != 2 or len(cells) != len(strata):
        raise ValueError("cells must be two-dimensional with one row per shot")
    bad = np.flatnonzero((~np.isfinite(cells) | (cells < 0)).any(axis=1))
    if len(bad):
        raise ValueError(f"cells row {bad[0]} must be finite and nonnegative")
    seed = _seed(seed)
    _check_level(level)
    value = float(metric(weights @ cells))
    replicates = metric(replicate_weights(strata, weights, n, seed) @ cells)
    low, high, undefined = _interval(replicates, level)
    return Estimate(value, low, high, undefined, len(replicates), seed, level)


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
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            "cells_a and cells_b must be two-dimensional of the same shape"
        )
    k = a.shape[1]

    def paired(totals):
        return metric(totals[..., :k]) - metric(totals[..., k:])

    return estimate(np.hstack([a, b]), strata, weights, paired, **options)


def weighted_median(values, weights):
    """Weighted median of `values`; `weights` may carry leading replicate axes."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")
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
    owners = np.asarray(owners)
    strata, weights = _shot_inputs(strata, weights)
    if values.ndim != 1 or owners.ndim != 1 or len(values) != len(owners):
        raise ValueError("values and owners must be one-dimensional of the same length")
    seed = _seed(seed)
    indices = [whole_number(owner) for owner in owners]
    if any(owner < 0 or owner >= len(strata) for owner in indices):
        raise ValueError("each owner must be a shot index in [0, number of shots)")
    owners = np.array(indices, dtype=np.int64)
    _check_level(level)
    value = float(weighted_median(values, weights[owners]))
    shots = replicate_weights(strata, weights, n, seed)
    low, high, undefined = _interval(weighted_median(values, shots[:, owners]), level)
    return Estimate(value, low, high, undefined, len(shots), seed, level)
