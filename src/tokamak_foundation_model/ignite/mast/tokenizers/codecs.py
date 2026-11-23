"""Codecs without learned weights: a signal's samples to values of 5 ms frames and back.

A codec turns a signal ``(N, *values_shape, T)`` (physical units, time last, NaN for missing) into
*values* ``(N, F, n_tok, n_val)``: one row of `n_val` numbers for each of `n_tok` tokens of each of the
`F` frames (`frames.frame_index`). The values are in the codec's *domain* (see `Codec`), ordered so
that the first ``k`` of them are the best ``k``-value description of the token; `grids.GridSet` turns
them into bins and back. A frame with no valid sample has NaN values and decodes to NaN: no data is
not a value.

This file grows in Tasks 4 (the linear codec) and 5 (the band cosine transform).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import scipy.fft

from .. import transform
from ..transform import ChannelStats
from . import frames
from .grid import READ_MODES, GridSet

DOMAINS = ("linear", "compressed", "signal")


# ------------------------------------------------------------------------------ cosine transform

def dct_values(x: np.ndarray, k: int) -> np.ndarray:
    """First `k` cosine values of each frame: `x` ``(..., n)`` samples (NaN missing) -> ``(..., k)``.

    The unit is the unit of `x`. For a frame without NaN the values are the first `k` coefficients of
    the orthonormal type-2 DCT divided by ``sqrt(n)`` (zero padded when ``k > n``): value 0 is the
    frame mean and the sum of the squared values is the frame's mean square. A frame with some NaN
    keeps the mean of its valid samples in value 0 and 0 in the others (a DCT needs every sample); a
    frame with no valid sample is NaN in every value. Float64.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n < 1:
        raise ValueError("a frame needs at least one sample")
    valid = ~np.isnan(x)
    count = valid.sum(-1)
    filled = np.where(valid, x, 0.0)
    full = scipy.fft.dct(filled, type=2, norm="ortho", axis=-1) / math.sqrt(n)
    kk = min(k, n)
    out = np.zeros(x.shape[:-1] + (k,))
    out[..., :kk] = full[..., :kk]
    # value 0 is the mean of the valid samples, computed as a mean: it equals the DCT's first
    # coefficient over sqrt(n) up to rounding, and is exact for a one-sample frame
    mean = np.divide(filled.sum(-1), count, out=np.zeros(count.shape), where=count > 0)
    part = (count > 0) & (count < n)
    out[part] = 0.0
    out[..., 0] = np.where(count > 0, mean, out[..., 0])
    out[count == 0] = np.nan
    return out


def dct_decode(values: np.ndarray, n: int) -> np.ndarray:
    """Samples of a frame from its values: ``(..., k)`` -> ``(..., n)``, in the unit of the values.

    The inverse of `dct_values` for any `k`: coefficients that are missing count as zero (``k < n``),
    coefficients beyond `n` are ignored. A row with a NaN value decodes to NaN. Float64.
    """
    v = np.asarray(values, dtype=np.float64)
    if n < 1:
        raise ValueError("a frame needs at least one sample")
    bad = np.isnan(v).any(-1)
    kk = min(v.shape[-1], n)
    coef = np.where(np.isnan(v[..., :kk]), 0.0, v[..., :kk])
    dc = coef[..., 0].copy()
    coef[..., 0] = 0.0
    # the mean is added as itself (the transform of the other coefficients has zero mean), which is
    # exact for a one-sample frame, where the transform alone is off in the last bit
    out = scipy.fft.idct(coef * math.sqrt(n), type=2, norm="ortho", n=n, axis=-1) + dc[..., None]
    out[bad] = np.nan
    return out


# ---------------------------------------------------------------------------------------- the base

class Codec:
    """Base of the codecs: the domain, the frames and the read-out; subclasses supply the values.

    Attributes: `name`; `dt` the sampling interval in seconds; `values_shape` the channel shape of the
    signal; `stats` its `ChannelStats`; `domain` the unit the values are in: ``"linear"``
    (`frames.to_linear`: the data's own shape, so a frame mean is a mean of the data), ``"compressed"``
    (the cache's model units, `transform.to_model`: a mean on the asinh scale) or ``"signal"``
    (TokaMark's units, ``(x - signal_mean) / signal_std``); `n_tok` tokens per frame and `n_val`
    values per token; `grids` the bins of the values (None before they are fitted).
    """

    def __init__(self, name: str, dt: float, values_shape: tuple[int, ...], stats: ChannelStats,
                 n_tok: int, n_val: int, domain: str = "linear", grids: GridSet | None = None,
                 signal_mean: float = 0.0, signal_std: float = 1.0) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}; one of {DOMAINS}")
        self.name = name
        self.dt = dt
        self.values_shape = tuple(values_shape)
        self.stats = stats
        self.domain = domain
        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.n_tok = n_tok
        self.n_val = n_val
        self.grids = grids

    # -- domain
    def _per_signal(self, fn: Callable[[np.ndarray], np.ndarray], a: np.ndarray) -> np.ndarray:
        """Apply a `transform` function that wants ``(*values_shape, T)`` to ``(..., *values_shape, T)``.

        The leading axes are folded into the time axis before the call and unfolded after it.
        """
        a = np.asarray(a)
        nv = len(self.values_shape)
        lead = a.shape[: a.ndim - nv - 1]
        if not lead:
            return fn(a)
        t = a.shape[-1]
        b = np.moveaxis(a.reshape((-1,) + a.shape[len(lead):]), 0, -2)      # (*values_shape, L, T)
        out = fn(b.reshape(self.values_shape + (-1,)))
        out = np.moveaxis(out.reshape(self.values_shape + (-1, t)), -2, 0)
        return out.reshape(lead + self.values_shape + (t,))

    def to_domain(self, x: np.ndarray) -> np.ndarray:
        """Physical ``(..., *values_shape, T)`` to the domain (float64, NaN kept)."""
        if self.domain == "linear":
            return frames.to_linear(x, self.stats)
        if self.domain == "compressed":
            return self._per_signal(lambda a: transform.to_model(a, self.stats), x)
        return (np.asarray(x, dtype=np.float64) - self.signal_mean) / self.signal_std

    def from_domain(self, u: np.ndarray) -> np.ndarray:
        """The domain to physical ``(..., *values_shape, T)`` (the inverse of `to_domain`)."""
        if self.domain == "linear":
            return frames.from_linear(u, self.stats)
        if self.domain == "compressed":
            return self._per_signal(lambda a: transform.to_physical(a, self.stats), u)
        return np.asarray(u, dtype=np.float64) * self.signal_std + self.signal_mean

    def domain_from_model(self, z: np.ndarray) -> np.ndarray:
        """Model units (the cache's) ``(..., *values_shape, T)`` to the domain."""
        if self.domain == "linear":
            return frames.linear_from_model(z, self.stats)
        if self.domain == "compressed":
            return np.asarray(z, dtype=np.float64)
        x = self._per_signal(lambda a: transform.to_physical(a, self.stats), z)
        return (x - self.signal_mean) / self.signal_std

    # -- values
    def encode_domain(self, u: np.ndarray) -> np.ndarray:
        """The domain ``(N, *values_shape, T)`` to values ``(N, F, n_tok, n_val)``."""
        raise NotImplementedError

    def decode_domain(self, values: np.ndarray, n: int) -> np.ndarray:
        """Values ``(N, F, n_tok, k)`` to the domain ``(N, *values_shape, n)``."""
        raise NotImplementedError

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Physical ``(N, *values_shape, T)`` to values ``(N, F, n_tok, n_val)``, ``F = frame_index(T, dt).max() + 1``."""
        x = np.asarray(x)
        if x.ndim != len(self.values_shape) + 2 or x.shape[1:-1] != self.values_shape:
            raise ValueError(f"x must be (N, *{self.values_shape}, T), got {x.shape}")
        if x.shape[-1] < 1:
            raise ValueError("a signal needs at least one sample")
        return self.encode_domain(self.to_domain(x))

    def decode(self, values: np.ndarray, n: int) -> np.ndarray:
        """Values ``(N, F, n_tok, k)`` to physical ``(N, *values_shape, n)``; NaN where a frame has no value."""
        return self.from_domain(self.decode_domain(values, n))

    def roundtrip(self, x: np.ndarray, mode: str = "unquantized") -> np.ndarray:
        """`x` physical ``(N, *values_shape, T)`` through the codec: ``decode(read(encode(x)), T)``.

        `read` is the identity for ``unquantized`` and ``grids.read(values, mode)`` otherwise;
        `ValueError` if the mode needs `grids` and there are none. NaN where the frame has no token.
        """
        if mode not in READ_MODES:
            raise ValueError(f"unknown read mode {mode!r}; one of READ_MODES {READ_MODES}")
        if mode != "unquantized" and self.grids is None:
            raise ValueError(f"read mode {mode!r} needs grids, and codec {self.name!r} has none")
        values = self.encode(x)
        if mode != "unquantized":
            values = self.grids.read(values, mode)
        return self.decode(values, np.asarray(x).shape[-1])


def _frame_groups(n: int, dt: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Frames of `n` samples at `dt`, grouped by length: ``[(length, frame ids, first samples)]``.

    Frames are contiguous, so a frame is its first sample and its length. Almost all have one length;
    the last may be shorter. Frame ids are int64 ``(m,)``; first samples likewise.
    """
    f = frames.frame_index(n, dt)
    n_frames = int(f.max()) + 1
    lens = np.bincount(f, minlength=n_frames)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = []
    for length in np.unique(lens):
        if length == 0:                                    # a frame no sample falls in (dt > 5 ms)
            continue
        ids = np.flatnonzero(lens == length)
        out.append((int(length), ids, starts[ids]))
    return out


# --------------------------------------------------------------------------------- the frame DCT

class FrameDCT(Codec):
    """One token per channel; its `k` values are the first cosine values of the channel's frame.

    ``n_tok = prod(values_shape)``, ``n_val = k``. Value 0 is the frame mean in the codec's domain
    and the values are nested: the first ``j`` are the best ``j``-value description (`dct_values`).
    """

    def __init__(self, name: str, dt: float, values_shape: tuple[int, ...], stats: ChannelStats, k: int,
                 domain: str = "linear", grids: GridSet | None = None, signal_mean: float = 0.0,
                 signal_std: float = 1.0) -> None:
        super().__init__(name, dt, values_shape, stats, n_tok=int(np.prod(values_shape, dtype=np.int64)),
                         n_val=k, domain=domain, grids=grids, signal_mean=signal_mean,
                         signal_std=signal_std)
        self.k = k

    def encode_domain(self, u: np.ndarray) -> np.ndarray:
        """Domain ``(N, *values_shape, T)`` to values ``(N, F, n_tok, k)`` float64."""
        u = np.asarray(u, dtype=np.float64)
        n, t = u.shape[0], u.shape[-1]
        flat = u.reshape(n, self.n_tok, t)
        n_frames = int(frames.frame_index(t, self.dt).max()) + 1
        out = np.full((n, n_frames, self.n_tok, self.k), np.nan)     # a frame with no sample stays NaN
        for length, ids, starts in _frame_groups(t, self.dt):
            cols = starts[:, None] + np.arange(length)               # (m, length)
            v = dct_values(flat[:, :, cols], self.k)                 # (N, n_tok, m, k)
            out[:, ids] = np.moveaxis(v, 2, 1)
        return out

    def decode_domain(self, values: np.ndarray, n: int) -> np.ndarray:
        """Values ``(N, F, n_tok, k)`` to the domain ``(N, *values_shape, n)``; NaN for a NaN frame."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 4 or v.shape[2] != self.n_tok:
            raise ValueError(f"values must be (N, F, {self.n_tok}, k), got {v.shape}")
        groups = _frame_groups(n, self.dt)
        n_frames = int(frames.frame_index(n, self.dt).max()) + 1
        if v.shape[1] < n_frames:
            raise ValueError(f"{n} samples need {n_frames} frames, values have {v.shape[1]}")
        out = np.full((v.shape[0], self.n_tok, n), np.nan)
        for length, ids, starts in groups:
            cols = starts[:, None] + np.arange(length)               # (m, length)
            d = dct_decode(np.moveaxis(v[:, ids], 1, 2), length)     # (N, n_tok, m, length)
            out[:, :, cols] = d
        return out.reshape((v.shape[0],) + self.values_shape + (n,))
