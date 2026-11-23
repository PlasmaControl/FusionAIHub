"""Ordered bins: every (token, value) position has its own grid of `n_bins` centres.

A value is not one class among `n_bins` but a point between two neighbouring centres: its target is
*two-hot* (weight linear in the value on the two centres around it), so a model that predicts a
distribution over the bins is trained on an ordered target, and the mean of the distribution is a
value in the data's unit. The centres of a position are asinh-spaced: even near zero (within
`width`), logarithmic beyond it, so a signal with a heavy tail keeps its resolution near zero.

Units. Every array of values is in one unit, the caller's (the codecs pass linear units); the grid
has no opinion. Shapes: `lo`, `hi`, `width` are ``(n_tok, n_val)``; values are ``(..., n_tok, k)``
with ``k <= n_val`` (a prefix of the positions, read with the first grids); weights are
``(..., n_tok, k, n_bins)``. A position whose grid is NaN (never seen in the fit) reads NaN. NaN
stays NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

READ_MODES = ("unquantized", "ranged", "nearest")
MAD_TO_SIGMA = 1.4826            # a Gaussian's standard deviation over its median absolute deviation
MIN_WIDTH_OVER_SPAN = 1e-3       # the width never falls below this fraction of the fitted span


@dataclass(frozen=True)
class GridSet:
    """The ordered bins of an ``(n_tok, n_val)`` array of positions.

    Centres of position ``(i, j)``: ``width * sinh(linspace(asinh(lo / width), asinh(hi / width),
    n_bins))``, strictly increasing, the first exactly `lo` and the last exactly `hi`.
    """

    lo: np.ndarray
    hi: np.ndarray
    width: np.ndarray
    n_bins: int = 64

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")
        if self.lo.ndim != 2 or not (self.lo.shape == self.hi.shape == self.width.shape):
            raise ValueError(
                f"lo, hi, width must share one (n_tok, n_val) shape, got "
                f"{self.lo.shape}, {self.hi.shape}, {self.width.shape}")
        seen = np.isfinite(self.lo) & np.isfinite(self.hi) & np.isfinite(self.width)
        if not (self.lo[seen] < self.hi[seen]).all() or not (self.width[seen] > 0).all():
            raise ValueError("every fitted position needs lo < hi and width > 0")

    @property
    def n_tok(self) -> int:
        return self.lo.shape[0]

    @property
    def n_val(self) -> int:
        return self.lo.shape[1]

    def centres(self) -> np.ndarray:
        """The centres: ``(n_tok, n_val, n_bins)`` float64, increasing along the last axis."""
        lo, hi, w = (np.asarray(a, dtype=np.float64) for a in (self.lo, self.hi, self.width))
        a, b = np.arcsinh(lo / w), np.arcsinh(hi / w)
        t = np.linspace(0.0, 1.0, self.n_bins)
        c = w[..., None] * np.sinh(a[..., None] + (b - a)[..., None] * t)
        c[..., 0], c[..., -1] = lo, hi                     # the ends exactly, not within rounding
        return c

    def _locate(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lower neighbour, fraction toward the upper one, and the centres of each value.

        `values` ``(..., n_tok, k)`` -> ``(lower, frac, c)``: `lower` ``(..., n_tok, k)`` int64 in
        ``[0, n_bins - 2]``; `frac` the same shape, float64 in ``[0, 1]`` (NaN for a NaN value or an
        unfitted position), linear in the value, 0 below `lo` and 1 above `hi`; `c` the used centres
        ``(n_tok, k, n_bins)``.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.ndim < 2 or v.shape[-2] != self.n_tok or v.shape[-1] > self.n_val:
            raise ValueError(
                f"values must be (..., {self.n_tok}, k <= {self.n_val}), got {v.shape}")
        c = self.centres()[:, : v.shape[-1], :]
        lower = np.clip((c <= v[..., None]).sum(-1) - 1, 0, self.n_bins - 2)
        c_lo = np.take_along_axis(np.broadcast_to(c, v.shape + (self.n_bins,)), lower[..., None], -1)[..., 0]
        c_hi = np.take_along_axis(np.broadcast_to(c, v.shape + (self.n_bins,)), lower[..., None] + 1, -1)[..., 0]
        frac = np.clip((v - c_lo) / (c_hi - c_lo), 0.0, 1.0)
        return lower.astype(np.int64), frac, c

    def two_hot(self, values: np.ndarray) -> np.ndarray:
        """Targets: `values` ``(..., n_tok, k)`` -> weights ``(..., n_tok, k, n_bins)`` float64.

        The two centres around a value get weights linear in it (they sum to 1 and their weighted
        mean is the value); a value beyond ``[lo, hi]`` puts weight 1 on the outermost centre; a NaN
        value gets a row of NaN.
        """
        lower, frac, _ = self._locate(values)
        w = np.zeros(lower.shape + (self.n_bins,))
        np.put_along_axis(w, lower[..., None], (1.0 - frac)[..., None], -1)
        np.put_along_axis(w, lower[..., None] + 1, frac[..., None], -1)
        w[np.isnan(frac)] = np.nan
        return w

    def expected(self, weights: np.ndarray) -> np.ndarray:
        """Mean value of a distribution over the bins: ``(..., n_tok, k, n_bins)`` -> ``(..., n_tok, k)``."""
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim < 3 or w.shape[-3] != self.n_tok or w.shape[-2] > self.n_val or w.shape[-1] != self.n_bins:
            raise ValueError(
                f"weights must be (..., {self.n_tok}, k <= {self.n_val}, {self.n_bins}), got {w.shape}")
        return (w * self.centres()[:, : w.shape[-2], :]).sum(-1)

    def index(self, values: np.ndarray) -> np.ndarray:
        """Index of the nearest centre: ``(..., n_tok, k)`` -> same shape int64, -1 for NaN."""
        lower, frac, _ = self._locate(values)
        nearest = lower + (frac > 0.5)                     # linear in the value: a tie goes to the lower
        return np.where(np.isnan(frac), -1, nearest).astype(np.int64)

    def read(self, values: np.ndarray, mode: str) -> np.ndarray:
        """Values after quantisation, same shape and unit as `values`.

        ``unquantized`` returns them; ``ranged`` is ``expected(two_hot(values))`` (the value limited
        to ``[lo, hi]``, exact inside); ``nearest`` is the nearest centre. NaN stays NaN.
        """
        if mode not in READ_MODES:
            raise ValueError(f"unknown read mode {mode!r}; one of READ_MODES {READ_MODES}")
        if mode == "unquantized":
            return np.asarray(values, dtype=np.float64)
        if mode == "ranged":
            return self.expected(self.two_hot(values))
        idx = self.index(values)
        c = np.broadcast_to(self.centres()[:, : idx.shape[-1], :], idx.shape + (self.n_bins,))
        out = np.take_along_axis(c, np.maximum(idx, 0)[..., None], -1)[..., 0]
        return np.where(idx < 0, np.nan, out)


def fit_ranges(values: np.ndarray, margin: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Range and width of each position: `values` ``(N, n_tok, n_val)`` (NaN missing) -> `(lo, hi, width)`.

    Each output is ``(n_tok, n_val)`` float64 in the unit of `values`. `lo` and `hi` are the minimum
    and maximum of the finite values, each moved outward by `margin` times their span (so an outlier
    is covered). `width` is 1.4826 times the median absolute deviation about the median, at least
    ``1e-3`` times the span: a robust scale, which an outlier does not grow. A constant position
    (span 0) gets ``v - 1``, ``v + 1``, 1; a position with no finite value gets NaN in all three.
    Holds ``(N, n_tok * n_val)`` float64 in memory (a median needs the column).
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError(f"values must be (N, n_tok, n_val), got {v.shape}")
    shape = v.shape[1:]
    flat = v.reshape(v.shape[0], -1)
    flat = np.where(np.isfinite(flat), flat, np.nan)
    seen = np.isfinite(flat).any(0)
    lo = np.full(flat.shape[1], np.nan)
    hi, width = lo.copy(), lo.copy()
    if seen.any():
        x = flat[:, seen]                                  # no all-NaN column: nothing warns below
        vmin, vmax = np.nanmin(x, 0), np.nanmax(x, 0)
        med = np.nanmedian(x, 0)
        mad = np.nanmedian(np.abs(x - med), 0)
        span = vmax - vmin
        const = span == 0
        lo[seen] = np.where(const, vmin - 1.0, vmin - margin * span)
        hi[seen] = np.where(const, vmax + 1.0, vmax + margin * span)
        width[seen] = np.where(const, 1.0, np.maximum(MAD_TO_SIGMA * mad, MIN_WIDTH_OVER_SPAN * span))
    return lo.reshape(shape), hi.reshape(shape), width.reshape(shape)
