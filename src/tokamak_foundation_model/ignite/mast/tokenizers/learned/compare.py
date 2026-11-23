"""Paired gaps, verdicts, collapse and the cost of a missing channel (plan 1c, rules 1, 2 and 7).

Everything here is a pure function of numbers or of a codec and one shot's cached values; the commands
that read the score trees and the cache (`run_tokenizer.py compare` and `seeds`, `train_tokenizer.py joint`)
call it. All errors are in TokaMark units (``(x - SignalSpec.mean) / SignalSpec.std``): the per-shot values of
`per_shot_harness` are the evaluator's `NRMSE_mean`, those of `per_shot_cache` are 1b's round-trip RMSE over a
shot's valid samples (`ladder.shot_error`). A shot without the signal, or without a valid sample, is left out
of a per-shot mapping and never counted as an error of 0; `paired_gap` takes a gap over the shots that both
sides have. A count of zero shots is "no data", not a verdict.

No torch at import time (the linear arms and the comparison run without it): `ignite.gate`, which `usage`
needs, is imported inside `usage`.
"""

from __future__ import annotations

import math
import types
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ...tasks import SignalSpec
from .. import ladder
from ..codecs import Codec
from . import recode

BOOT = 2000                # paired resamples of the shots (rule 1)
CLEAR = 0.10               # the gap that makes "beats" "clearly beats" (rule 2: the author's reading of the design)
_BOOT_BLOCK = 4_000_000    # resampled values held at once: a block of resamples is `_BOOT_BLOCK // n_shots` rows


# --------------------------------------------------------------------------------------- the gap

def paired_gap(a: Mapping[int, float], b: Mapping[int, float], *, n_boot: int = BOOT, seed: int = 0) -> dict:
    """The gap of tokenizer A against tokenizer B over the shots both have (rule 1).

    `a` and `b` map a shot to its error in TokaMark units. Only the shots with a finite value on both sides
    count. Returns ``n_shots`` (int), ``mean_a`` and ``mean_b`` (over those shots), ``gap`` =
    ``(mean_b - mean_a) / mean_b`` (positive when A is the lower error) and ``lo`` and ``hi``, the 2.5 and 97.5
    percentiles of that gap over `n_boot` resamples of those shots with replacement (the same resampled shots for
    A and B, from ``numpy.random.default_rng(seed)``). `gap`, `lo` and `hi` are NaN with fewer than two shots, or
    when ``mean_b`` is not above 0; the means are NaN with no shot.
    """
    shots = sorted(s for s in a if s in b and math.isfinite(a[s]) and math.isfinite(b[s]))
    n = len(shots)
    x = np.array([a[s] for s in shots], dtype=np.float64)
    y = np.array([b[s] for s in shots], dtype=np.float64)
    mean_a = float(x.mean()) if n else math.nan
    mean_b = float(y.mean()) if n else math.nan
    out = {"n_shots": n, "mean_a": mean_a, "mean_b": mean_b, "gap": math.nan, "lo": math.nan, "hi": math.nan}
    if n < 2 or not mean_b > 0:
        return out
    out["gap"] = (mean_b - mean_a) / mean_b
    rng = np.random.default_rng(seed)
    block = max(1, _BOOT_BLOCK // n)
    gaps = []
    for start in range(0, n_boot, block):
        idx = rng.integers(0, n, (min(block, n_boot - start), n))
        ma, mb = x[idx].mean(axis=1), y[idx].mean(axis=1)
        keep = mb > 0
        gaps.append((mb[keep] - ma[keep]) / mb[keep])
    gaps = np.concatenate(gaps)
    if len(gaps):
        out["lo"], out["hi"] = (float(v) for v in np.percentile(gaps, [2.5, 97.5]))
    return out


def seed_spread(gaps: Sequence[float]) -> float:
    """The spread of one tokenizer's gap over its seeds, largest minus smallest (rule 2); NaN for fewer than
    two gaps or when one of them is not a number (a gap not measured makes the spread unknown)."""
    g = np.asarray(list(gaps), dtype=np.float64)
    if len(g) < 2 or not np.isfinite(g).all():
        return math.nan
    return float(g.max() - g.min())


def beats(g: Mapping[str, float], *, seed_spread: float | None = None) -> bool:
    """Rule 2: A beats B if the interval of `g` (a `paired_gap` of A against B) lies above 0 and, when
    `seed_spread` is given (A or B is a network), the gap is larger than it. A NaN `seed_spread` (fewer than two
    seeds, or a seed not measured) is a spread that is not known, so the win does not stand."""
    if not (math.isfinite(g["lo"]) and g["lo"] > 0):
        return False
    return True if seed_spread is None else bool(g["gap"] > seed_spread)


def clearly_beats(g: Mapping[str, float], *, seed_spread: float | None = None, margin: float = CLEAR) -> bool:
    """Rule 2: A clearly beats B if A beats B and the gap is at least `margin` (0.10)."""
    return beats(g, seed_spread=seed_spread) and bool(g["gap"] >= margin)


def word(ab: Mapping[str, float], ba: Mapping[str, float], seed_spread: float | None = None) -> str:
    """The verdict between A and B in words: ``a clearly beats b``, ``a beats b``, ``b clearly beats a``,
    ``b beats a``, ``tie`` or ``no data``.

    `ab` is the `paired_gap` of A against B and `ba` that of B against A (the two rows of one comparison).
    ``no data`` is for a gap that is not a number on both sides (fewer than two shots), which is not a tie.
    """
    if not (math.isfinite(ab["gap"]) or math.isfinite(ba["gap"])):
        return "no data"
    for who, g in (("a", ab), ("b", ba)):
        if clearly_beats(g, seed_spread=seed_spread):
            return f"{who} clearly beats {'b' if who == 'a' else 'a'}"
        if beats(g, seed_spread=seed_spread):
            return f"{who} beats {'b' if who == 'a' else 'a'}"
    return "tie"


# --------------------------------------------------------------------------------- per-shot errors

def per_shot_harness(score_dir: Path, task: str, signal: str) -> dict[int, float]:
    """The evaluator's ``NRMSE_mean`` of each shot for `signal`, from ``<score_dir>/<task>/shots_metrics.csv``
    (TokaMark units; one row per shot and feature). ``{}`` if the file is missing."""
    path = Path(score_dir) / task / "shots_metrics.csv"
    if not path.is_file():
        return {}
    table = pd.read_csv(path)
    rows = table[table["feature_name"] == signal]
    return {int(s): float(v) for s, v in zip(rows["shot_id"], rows["NRMSE_mean"], strict=True)}


def per_shot_cache(codec: Codec, spec: SignalSpec, per_shot: Iterable[tuple[int, np.ndarray | None]],
                   mode: str = "unquantized") -> dict[int, float]:
    """`ladder.shot_error` of each shot: ``per_shot`` yields ``(shot, z_model)`` with `z_model` the cache's
    ``(*values_shape, T)`` in model units, or None for a shot without the signal. The error is the RMSE of the
    round-trip over the shot's valid samples in TokaMark units. A shot without the signal or without a valid
    sample is left out."""
    out: dict[int, float] = {}
    for shot, z in per_shot:
        if z is None:
            continue
        err = ladder.shot_error(codec, spec, z, mode)
        if err is not None:
            out[int(shot)] = err
    return out


def values_to_reach(curve: Mapping[int, float], target: float) -> int | None:
    """The smallest size (a count of values) of `curve` ``{size: error}`` whose error is at or under `target`
    (same unit as the errors); None if none is. A size whose error is not a number is not under anything."""
    for size in sorted(curve):
        if curve[size] <= target:
            return int(size)
    return None


def scaled_k(k: int, factor: float, step: int, limit: int) -> int | None:
    """``k * factor`` rounded down to a multiple of `step`; None if that is under `step` or over `limit`
    (a size that does not exist)."""
    scaled = int(math.floor(k * factor / step + 1e-9)) * step
    return scaled if step <= scaled <= limit else None


# ---------------------------------------------------------------------------------- collapse (rule 7)

def level_index(values: np.ndarray, levels: int) -> np.ndarray:
    """Values in [-1, 1] to the index of the nearest of `levels` even levels, ``rint((v + 1) / 2 * (levels - 1))``,
    int64 and of the shape of `values`."""
    return np.rint((np.asarray(values) + 1.0) / 2.0 * (levels - 1)).astype(np.int64)


def usage(values: np.ndarray, levels: int = 8, min_entropy: float = 0.3) -> dict:
    """How a network's tokens use the `levels` levels of each value (rule 7).

    `values` is ``(M, n_tok, n_val)``, the bounded values of a token network on its validation examples
    (unitless, in [-1, 1]; `ValueError` for a finite value outside that). Examples with a NaN anywhere are
    dropped, and `ValueError` if none is left (no examples is no verdict). Each token is read on its own through
    `ignite.gate.utilization` (levels ``(levels,) * n_val``), and each value's entropy is divided by
    ``log(levels)``. Returns ``first_value_entropy`` (the smallest over the tokens of the normalised entropy of
    value 0), ``active_values`` (how many of the `n_val` value positions have a smallest normalised entropy over
    the tokens of at least `min_entropy`; reported, it decides nothing), ``collapsed`` (``first_value_entropy <
    min_entropy``, the 0.3 of baseline IGNITE's ``gate_min_code_entropy``) and ``n_examples``.
    """
    from tokamak_foundation_model.ignite import gate          # imports torch: only here

    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError(f"values must be (M, n_tok, n_val), got {v.shape}")
    v = v[np.isfinite(v).all(axis=(1, 2))]
    if len(v) == 0:
        raise ValueError("no example without a NaN: there is nothing to judge")
    if v.min() < -1.0 - 1e-6 or v.max() > 1.0 + 1e-6:
        raise ValueError(f"values must lie in [-1, 1], got [{v.min():.4g}, {v.max():.4g}]")
    _, n_tok, n_val = v.shape
    cfg = types.SimpleNamespace(codebook_size=levels ** min(n_val, 20), fsq_levels=(levels,) * n_val)
    codes = level_index(np.clip(v, -1.0, 1.0), levels)
    entropy = np.array([gate.utilization(codes[:, t:t + 1, :], cfg, min_utilization=0.0)["dim_entropy_nats"]
                        for t in range(n_tok)]) / math.log(levels)                      # (n_tok, n_val)
    first = float(entropy[:, 0].min())
    return {"first_value_entropy": first, "active_values": int((entropy.min(axis=0) >= min_entropy).sum()),
            "collapsed": bool(first < min_entropy), "n_examples": int(len(v))}


# ---------------------------------------------------------------------------- a missing channel

def missing_stream_cost(codec: Codec, spec: SignalSpec, z_model: np.ndarray, channel: int) -> tuple[float, float] | None:
    """What one missing channel does to the others, for one shot of a series signal.

    `z_model` is the cache's ``(C, T)`` in model units. Returns ``(before, after)``: the round-trip RMSE in
    TokaMark units over the valid samples of the channels other than `channel`, with the shot as it is and with
    `channel` set to NaN. A per-channel codec gives two equal numbers; a concatenated one, which fills a
    missing stream, codes the other channels worse. None if the other channels have no valid sample (no data is
    not a cost of 0). `ValueError` for a shot that is not ``(C, T)`` or a `channel` outside ``0..C-1``.
    """
    z = np.asarray(z_model)
    if z.ndim != 2 or not 0 <= channel < z.shape[0]:
        raise ValueError(f"z_model must be (C, T) with channel in 0..C-1, got {z.shape} and channel {channel}")
    others = np.arange(z.shape[0]) != channel
    gone = z.copy()
    gone[channel] = np.nan
    pair = []
    for shot in (z, gone):
        sq, n = ladder.channel_errors(codec, spec, shot)
        if n[others].sum() == 0:
            return None
        pair.append(float(math.sqrt(sq[others].sum() / n[others].sum())))
    return pair[0], pair[1]


# ---------------------------------------------------------------------------------- the joint basis

def left_over_sums(train: np.ndarray, val: np.ndarray, sizes: Sequence[int], *,
                   masked: bool = False) -> tuple[list[float], float]:
    """``(residuals, total)`` behind `left_over`: the sums of squares of what the first `size` principal components
    of `train` leave (one number per size) and of `val` about the training mean. Both are in example units squared.

    `masked` False (the default): over the complete rows of `val`, the rows without a NaN. `masked` True: over every
    row of `val` and the entries of it that are finite. A NaN entry of `val` is a missing stream, which a
    concatenated recoder encodes as 0.0 (`RecodedCodec`); it is encoded so, and counts in neither sum. This is the
    measure of `train_tokenizer.py linear`, and the one that has data when a probe is dead in every frame.
    """
    train = np.asarray(train)
    val = np.asarray(val, dtype=np.float64)
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1]:
        raise ValueError(f"train and val must be (rows, d) with one d, got {train.shape} and {val.shape}")
    sizes = [int(s) for s in sizes]
    if not sizes:
        raise ValueError("sizes is empty")
    r = recode.fit_linear(train, max(sizes), concat=True)           # NaN set to 0.0 first: ruling C23 (and C21)
    if not masked:
        val = val[np.isfinite(val).all(axis=1)]
        centered = val - r.mean
        total = float((centered ** 2).sum())
        captured = np.cumsum((centered @ r.basis) ** 2, axis=1).sum(axis=0)       # (max size,)
        return [max(total - float(captured[s - 1]), 0.0) for s in sizes], total
    ok = np.isfinite(val)
    filled = np.where(ok, val, 0.0)
    coeff = (filled - r.mean) @ r.basis                                           # (M, max size)
    total = float((np.where(ok, filled - r.mean, 0.0) ** 2).sum())
    residuals = []
    for size in sizes:
        decoded = coeff[:, :size] @ r.basis[:, :size].T + r.mean
        residuals.append(float((np.where(ok, filled - decoded, 0.0) ** 2).sum()))
    return residuals, total


def left_over(train: np.ndarray, val: np.ndarray, sizes: Sequence[int]) -> list[float]:
    """The share of the mean square of `val` that the first `size` principal components of `train` leave, for
    each of `sizes` (0 leaves nothing, 1 leaves all).

    `train` is ``(M, d)`` and `val` is ``(M', d)``, both in example units (NaN allowed in `train`: a missing
    stream). The components are those of `recode.fit_linear(train, max(sizes), concat=True)`, so a NaN of `train` is
    0.0 in the fit, as it is for a concatenated recoder (ruling C23: the pairwise-complete covariance of rows with
    missing streams is not positive semi-definite and reorders the basis). The share is taken about the training
    mean, over the rows of `val` without a NaN. NaN for every size if no row of `val` is complete or `val` is
    all at the mean (no data is not a share of 0).
    """
    residuals, total = left_over_sums(train, val, sizes)
    return [r / total if total > 0 else math.nan for r in residuals]
