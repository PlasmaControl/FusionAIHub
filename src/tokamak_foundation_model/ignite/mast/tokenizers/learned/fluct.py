"""What the trend path leaves out of a 50 kHz channel, as a statistic (plan 1c, Task 7).

No tokenizer is built here (open decision 4). A 50 kHz channel (250 samples of 0.02 ms in a 5 ms frame) is
described by its first `k_trend` cosine values; this module measures the power in the band above them, by
frequency, so that the stage-2 plan can decide on fluctuation tokens with a number. Nothing in stage 1 is
decided by it.

Units. Every array is in the codec's domain (what a front codec's ``domain_from_model`` returns), and a power
is a mean square in that domain squared. The cosine values are 1b's: the orthonormal type-2 DCT of a frame divided
by ``sqrt(n)``, so the sum of the squared values of a frame is the frame's mean square (`codecs.dct_values`).
They are computed here with scipy directly: importing `codecs` loads torch, and this module and its spawned
readers must not; a test holds `cosine_values` equal to `codecs.dct_values(x, n)` to 1e-12.

Frequency. With 250 samples of 0.02 ms, cosine value ``j`` is at ``j * 100 Hz`` (the frame is 5 ms, so the
spacing is ``1 / (2 * 5 ms)``), and a group of 10 neighbouring values is 1 kHz wide.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.fft

from ... import cache

N_FLUCT_BINS = 25                                  # groups of neighbouring cosine values (1 kHz each at 250 samples)
BELOW_5KHZ = 5                                     # the first five groups


def cosine_values(x: np.ndarray) -> np.ndarray:
    """All ``n`` cosine values of each frame: `x` ``(..., n)`` samples to ``(..., n)``, float64.

    ``scipy.fft.dct(x, type=2, norm="ortho") / sqrt(n)``, the first ``n`` values of `codecs.dct_values`; a frame
    with a NaN is NaN in every value (a DCT needs every sample).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n < 1:
        raise ValueError("a frame needs at least one sample")
    out = scipy.fft.dct(x, type=2, norm="ortho", axis=-1) / math.sqrt(n)
    return np.where(np.isnan(x).any(axis=-1, keepdims=True), np.nan, out)


def _split(u: np.ndarray, k_trend: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """``(band, trend)`` powers of each frame: band ``(..., n_bins)``, trend ``(...)`` (values 1 to ``k_trend - 1``)."""
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[-1]
    if k_trend < 0:
        raise ValueError(f"k_trend must be at least 0, got {k_trend}")
    if n_bins < 1 or n % n_bins:
        raise ValueError(f"a frame of {n} samples does not split into {n_bins} groups of equal width")
    p = cosine_values(u) ** 2
    trend = p[..., 1:k_trend].sum(axis=-1)
    p[..., :k_trend] = 0.0
    return p.reshape(p.shape[:-1] + (n_bins, n // n_bins)).sum(axis=-1), trend


def fluct_power(u: np.ndarray, k_trend: int, n_bins: int = N_FLUCT_BINS) -> np.ndarray:
    """The power of each frame above the trend, by frequency: `u` ``(..., n)`` samples of one frame in the codec's
    domain to ``(..., n_bins)``.

    The squares of all ``n`` cosine values, those under index `k_trend` set to 0, summed in `n_bins` groups of
    ``n // n_bins`` neighbours. With ``k_trend`` 0 the sum over the groups is the frame's mean square. A frame with
    a NaN gives NaN; ``n % n_bins != 0`` (or a negative `k_trend`) raises `ValueError`.
    """
    return _split(u, k_trend, n_bins)[0]


def frame_persistence(p: np.ndarray) -> float:
    """How much a level carries from one frame to the next: `p` ``(F,)``, one power per frame, to one number.

    Pearson's correlation of ``log10(p[1:])`` with ``log10(p[:-1])`` over the pairs in which both are finite and
    positive. NaN with fewer than three such pairs, or when either side of the pairs is constant.
    """
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p) & (p > 0)
    pair = ok[1:] & ok[:-1]
    if pair.sum() < 3:
        return float("nan")
    a, b = np.log10(p[:-1][pair]), np.log10(p[1:][pair])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    da, db = a - a.mean(), b - b.mean()
    return float((da * db).sum() / math.sqrt((da ** 2).sum() * (db ** 2).sum()))


def summarize(per_shot: Iterable[np.ndarray], frame_len: int, k_trend: int) -> dict:
    """The band above the trend over many shots: each item of `per_shot` is ``(C, T)`` samples of one shot in the
    codec's domain. Whole frames only, counted from the shot's first sample (a trailing partial frame is dropped,
    as `store.stream_values` does); a frame with a NaN is left out.

    Keys: ``n_frames`` (channel-frames used), ``share_by_bin`` ``(N_FLUCT_BINS,)`` (the band's power per group as a
    share of its total, over all used frames), ``share_below_5khz`` (the first five groups' share),
    ``band_to_trend`` (the band's power over the power of the values 1 to ``k_trend - 1``; NaN for ``k_trend < 2``),
    ``persistence`` (the median over every (shot, channel) whose `frame_persistence` is finite, of that function
    of the band's total power per frame; NaN if there is none). Without a frame: ``n_frames`` 0 and NaN elsewhere.
    """
    if frame_len < 1:
        raise ValueError(f"frame_len must be at least 1, got {frame_len}")
    if k_trend < 0:
        raise ValueError(f"k_trend must be at least 0, got {k_trend}")
    n_frames, band_sum, trend_sum, persistence = 0, np.zeros(N_FLUCT_BINS), 0.0, []
    for z in per_shot:
        z = np.asarray(z, dtype=np.float64)
        f = z.shape[-1] // frame_len
        if f == 0:
            continue
        u = z[:, : f * frame_len].reshape(z.shape[0], f, frame_len)
        band, trend = _split(u, k_trend, N_FLUCT_BINS)
        total = band.sum(axis=-1)                                      # (C, F), NaN for a frame with a NaN
        used = np.isfinite(total)
        n_frames += int(used.sum())
        band_sum += band[used].sum(axis=0)
        trend_sum += float(trend[used].sum())
        persistence += [frame_persistence(row) for row in total]
    persistence = [v for v in persistence if np.isfinite(v)]
    nan = float("nan")
    total_band = band_sum.sum()
    share = band_sum / total_band if n_frames and total_band > 0 else np.full(N_FLUCT_BINS, nan)
    return {"n_frames": n_frames, "share_by_bin": share,
            "share_below_5khz": float(share[:BELOW_5KHZ].sum()),
            "band_to_trend": float(total_band / trend_sum) if n_frames and k_trend >= 2 and trend_sum > 0 else nan,
            "persistence": float(np.median(persistence)) if persistence else nan}


def shot_domain(job: tuple) -> np.ndarray | None:
    """``(shot, cache_root, stats_sha256, front)`` to ``(C, T)`` float64 samples of one shot in the front codec's
    domain, or None if the shot has no such signal. Module-level so `transform.map_shots` can pickle it.

    The cache file is read with the manifest's ``stats_sha256`` (a file made from other statistics raises
    `ValueError`); `front` is any codec with ``name``, ``n_tok`` and ``domain_from_model``.
    """
    shot, root, sha, front = job
    got = cache.read_shot(Path(root) / "shots" / f"{shot}.h5", [front.name], stats_sha256=sha)[front.name]
    if got is None:
        return None
    z = np.asarray(got.values)
    u = np.asarray(front.domain_from_model(z[None]), dtype=np.float64)[0]
    return u.reshape(front.n_tok, z.shape[-1])
