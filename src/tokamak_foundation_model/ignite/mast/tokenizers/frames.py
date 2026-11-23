"""Frames of 5 ms and the linear units the tokenizers work in.

Units. "Physical" is the unit of the data. "Model units" are the cache's (`transform.to_model`:
``(c(x) - mean) / std`` with ``c`` the identity or ``asinh(x / scale)``). "Linear units" undo the
compression but keep the data's own shape: ``(x - mean) / std`` for ``compress == "none"`` and
``x / scale`` for ``"asinh"``. Every array has the time axis last; the signal's constants broadcast
against the axes before it, so a leading batch axis is allowed. NaN stays NaN.
"""

from __future__ import annotations

import numpy as np

from ..headroom import cheap
from ..transform import ChannelStats, _check_compress

FRAME_S = cheap.FRAME_S          # 0.005, stage 0's frame


def frame_len(dt: float) -> int:
    """Samples of one 5 ms frame at sampling interval `dt` seconds (at least 1)."""
    return cheap.frame_len(dt)


def frame_index(n: int, dt: float) -> np.ndarray:
    """Frame of each of `n` samples at interval `dt` seconds: ``(n,)`` int64.

    Sample ``j`` belongs to frame ``floor((j + 0.5) * dt / FRAME_S)``, counted from the first sample;
    the half-sample offset puts a sample's centre, not its edge, in a frame (it keeps a spectrogram
    column on the frame it mostly lies in).
    """
    return np.floor((np.arange(n) + 0.5) * dt / FRAME_S).astype(np.int64)


def _const(a: np.ndarray) -> np.ndarray:
    """A constant of shape ``values_shape`` as float64 with a trailing time axis to broadcast."""
    return np.asarray(a, dtype=np.float64)[..., None]


def to_linear(x: np.ndarray, st: ChannelStats) -> np.ndarray:
    """Physical ``(..., *values_shape, T)`` to linear units (float64, NaN kept)."""
    _check_compress(st.compress)
    x = np.asarray(x, dtype=np.float64)
    if st.compress == "asinh":
        return x / _const(st.scale)
    return (x - _const(st.mean)) / _const(st.std)


def from_linear(u: np.ndarray, st: ChannelStats) -> np.ndarray:
    """Linear units to physical (the inverse of `to_linear`)."""
    _check_compress(st.compress)
    u = np.asarray(u, dtype=np.float64)
    if st.compress == "asinh":
        return u * _const(st.scale)
    return u * _const(st.std) + _const(st.mean)


def linear_from_model(z: np.ndarray, st: ChannelStats) -> np.ndarray:
    """Model units (the cache's) to linear units: `z` itself for ``"none"``, ``sinh(z * std + mean)`` for ``"asinh"``."""
    _check_compress(st.compress)
    z = np.asarray(z, dtype=np.float64)
    if st.compress == "asinh":
        return np.sinh(z * _const(st.std) + _const(st.mean))
    return z


def model_from_linear(u: np.ndarray, st: ChannelStats) -> np.ndarray:
    """Linear units to model units (the inverse of `linear_from_model`)."""
    _check_compress(st.compress)
    u = np.asarray(u, dtype=np.float64)
    if st.compress == "asinh":
        return (np.arcsinh(u) - _const(st.mean)) / _const(st.std)
    return u
