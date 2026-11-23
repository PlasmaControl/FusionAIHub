"""Codecs without learned weights: a signal's samples to values of 5 ms frames and back.

A codec turns a signal ``(N, *values_shape, T)`` (physical units, time last, NaN for missing) into
*values* ``(N, F, n_tok, n_val)``: one row of `n_val` numbers for each of `n_tok` tokens of each of the
`F` frames (`frames.frame_index`). The values are in the codec's *domain* (see `Codec`), ordered so
that the first ``k`` of them are the best ``k``-value description of the token; `grids.GridSet` turns
them into bins and back. A frame with no valid sample has NaN values and decodes to NaN: no data is
not a value.

This file grows in Task 5 (the band cosine transform).
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


# ------------------------------------------------------------------------------- principal components

def fit_pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Principal components of rows with missing points, from every pair of valid points.

    `x` is ``(N, D)``, NaN for missing, in whatever unit the caller wants the components in. Returns
    ``(mean (D,), basis (n, D), eigenvalues (n,), total_variance)`` with ``n = min(n_components, D)``.
    The mean of a point is over the rows where it is valid (0 if never). The covariance of two points
    is over the rows where both are valid, about the two points' means over those same rows
    (pairwise-complete), so a point that is never valid, or a pair never valid together, contributes 0.
    The basis rows are orthonormal eigenvectors by decreasing eigenvalue; a negative eigenvalue (which
    pairwise covariance can give) is set to 0. `total_variance` is the trace of the covariance.

    The rows are visited in chunks: three D by D float64 sums (of ``x x``, of ``x`` over the rows where
    the other point is valid, and of the pair counts) are kept, and `x` is never copied whole. The sums
    are taken about each point's own mean (a first pass) so that a large mean does not cancel the
    variance; the covariance does not depend on that shift.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"x must be (N, D), got {x.shape}")
    n_rows, d = x.shape
    n = min(int(n_components), d)
    if n < 0:
        raise ValueError(f"n_components must not be negative, got {n_components}")
    step = max(1, 4_000_000 // max(d, 1))
    total = np.zeros(d)
    count = np.zeros(d)
    for i in range(0, n_rows, step):
        c = np.asarray(x[i:i + step], dtype=np.float64)
        ok = np.isfinite(c)
        total += np.where(ok, c, 0.0).sum(0)
        count += ok.sum(0)
    mean = np.divide(total, count, out=np.zeros(d), where=count > 0)
    xx = np.zeros((d, d))                          # sum of x_i x_j over the rows where both are valid
    xv = np.zeros((d, d))                          # sum of x_i over the rows where j is valid, at [i, j]
    vv = np.zeros((d, d))                          # the number of rows where both are valid
    for i in range(0, n_rows, step):
        c = np.asarray(x[i:i + step], dtype=np.float64)
        ok = np.isfinite(c)
        z = np.where(ok, c - mean, 0.0)
        v = ok.astype(np.float64)
        xx += z.T @ z
        xv += z.T @ v
        vv += v.T @ v
    nn = np.where(vv > 0, vv, 1.0)
    cov = xx / nn - (xv / nn) * (xv.T / nn)
    cov[vv == 0] = 0.0
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:n]
    eig = np.maximum(evals[order], 0.0)
    return mean, evecs[:, order].T.copy(), eig, float(np.trace(cov))


class LinearCodec(Codec):
    """A profile or map as coefficients on principal components; one example is one sample in time.

    The signal ``(*values_shape)`` at one sample is flattened row-major to ``D = prod(values_shape)``
    points. ``n_values`` coefficients are split into ``n_tok = n_values // values_per_token`` tokens of
    ``n_val = values_per_token`` values; token ``i`` holds components ``i * n_val`` to
    ``(i + 1) * n_val - 1`` so a prefix of the tokens is a coarser profile. Every frame must hold one
    sample (`ValueError` otherwise); a frame no sample falls in (``dt`` above 5 ms) has NaN values.

    With `B` the first `n_values` basis rows restricted to the valid points and `e` their eigenvalues,
    the values `a` of a row `r` solve ``(B B^T + lam * diag(1 / e)) a = B (r - profile_mean)[valid]``
    with ``lam = max(ridge, 1e-10 * eigenvalues[0])``; components with ``e == 0`` are fixed to 0. For a
    row with no missing point the first tokens' values do not depend on `n_values` (for an orthonormal
    basis), with missing points they do, slightly. A row with no valid point has NaN values. `decode`
    gives ``profile_mean + a @ basis[:n_values]`` at every point, also where the input was missing;
    with ``n_values = 0`` there is no token and it is the mean profile (a NaN frame cannot be told
    apart then). `profile_mean` ``(D,)`` and `basis` ``(m, D)`` are in the codec's domain.
    """

    def __init__(self, name: str, dt: float, values_shape: tuple[int, ...], stats: ChannelStats,
                 profile_mean: np.ndarray, basis: np.ndarray, eigenvalues: np.ndarray, n_values: int,
                 values_per_token: int, domain: str = "linear", ridge: float = 0.0,
                 grids: GridSet | None = None, signal_mean: float = 0.0, signal_std: float = 1.0) -> None:
        if values_per_token < 1 or n_values % values_per_token:
            raise ValueError(f"n_values={n_values} must be a multiple of values_per_token={values_per_token}")
        basis = np.asarray(basis, dtype=np.float64)
        d = int(np.prod(values_shape, dtype=np.int64))
        if basis.ndim != 2 or basis.shape[1] != d:
            raise ValueError(f"basis must be (m, {d}), got {basis.shape}")
        if n_values > len(basis):
            raise ValueError(f"n_values={n_values} is more than the {len(basis)} basis rows")
        profile_mean = np.asarray(profile_mean, dtype=np.float64)
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        if profile_mean.shape != (d,) or eigenvalues.shape != (len(basis),):
            raise ValueError(f"profile_mean must be ({d},) and eigenvalues ({len(basis)},), "
                             f"got {profile_mean.shape} and {eigenvalues.shape}")
        super().__init__(name, dt, values_shape, stats, n_tok=n_values // values_per_token,
                         n_val=values_per_token, domain=domain, grids=grids, signal_mean=signal_mean,
                         signal_std=signal_std)
        self.profile_mean = profile_mean
        self.basis = basis
        self.eigenvalues = eigenvalues
        self.n_values = n_values
        self.ridge = ridge
        self.lam = max(ridge, 1e-10 * float(eigenvalues[0])) if len(eigenvalues) else ridge
        self.n_points = d

    def _frame_of_sample(self, n: int) -> np.ndarray:
        """Frame of each of `n` samples, ``(n,)`` int64; `ValueError` if two samples share a frame."""
        f = frames.frame_index(n, self.dt)
        if n > 1 and (np.diff(f) < 1).any():
            raise ValueError(f"the linear codec wants one sample per frame; dt={self.dt} gives more")
        return f

    def encode_domain(self, u: np.ndarray) -> np.ndarray:
        """Domain ``(N, *values_shape, T)`` to values ``(N, F, n_tok, n_val)`` float64."""
        u = np.asarray(u, dtype=np.float64)
        n, t = u.shape[0], u.shape[-1]
        f = self._frame_of_sample(t)
        out = np.full((n, int(f[-1]) + 1, self.n_tok, self.n_val), np.nan)
        if self.n_values == 0:
            return out
        r = np.moveaxis(u.reshape(n, self.n_points, t), 2, 1).reshape(n * t, self.n_points) - self.profile_mean
        a = np.full((n * t, self.n_values), np.nan)
        e = self.eigenvalues[: self.n_values]
        active = np.flatnonzero(e > 0)
        ok = np.isfinite(r)
        has = ok.any(1)
        a[has] = 0.0                                          # components with e == 0 stay 0
        if len(active):
            b = self.basis[: self.n_values][active]            # (m, D)
            pen = self.lam / e[active]
            packed = np.packbits(ok, axis=1)
            _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
            inverse = inverse.reshape(-1)
            for g in range(len(first)):
                rows = np.flatnonzero((inverse == g) & has)
                if not len(rows):
                    continue
                v = ok[first[g]]
                bv = b[:, v]
                sol = np.linalg.solve(bv @ bv.T + np.diag(pen), bv @ r[np.ix_(rows, np.flatnonzero(v))].T)
                a[np.ix_(rows, active)] = sol.T
        out[:, f] = a.reshape(n, t, self.n_tok, self.n_val)
        return out

    def decode_domain(self, values: np.ndarray, n: int) -> np.ndarray:
        """Values ``(N, F, n_tok, n_val)`` to the domain ``(N, *values_shape, n)``; NaN for a NaN frame."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 4 or v.shape[2:] != (self.n_tok, self.n_val):
            raise ValueError(f"values must be (N, F, {self.n_tok}, {self.n_val}), got {v.shape}")
        f = self._frame_of_sample(n)
        if v.shape[1] <= int(f[-1]):
            raise ValueError(f"{n} samples need {int(f[-1]) + 1} frames, values have {v.shape[1]}")
        a = v[:, f].reshape(v.shape[0] * n, self.n_values)
        out = self.profile_mean + a @ self.basis[: self.n_values]      # NaN in a row stays in that row
        out = out.reshape(v.shape[0], n, self.n_points)
        return np.moveaxis(out, 1, 2).reshape((v.shape[0],) + self.values_shape + (n,))
