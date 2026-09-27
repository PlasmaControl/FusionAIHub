"""What the review editors' panels share.

A spectrogram on one colour scale for every shot, a trace averaged into
bins, and the rule for an input a shot does not have: that panel is left
out (and logged), the rest of the shot is still drawn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from fractions import Fraction

import numpy as np
from scipy import signal

from ...config import Paths
from ...features.store import read_feature
from ..verify import NoDataError, Panel

log = logging.getLogger(__name__)
#: What a shot may lack: a raw group (NoDataError), a feature (KeyError), a file.
MISSING = (NoDataError, KeyError, OSError)
#: dB above each frequency bin's own median, the scale every spectrogram shares.
Z_DB = (-3.0, 27.0)


def optional(what: str, shot: int, build: Callable[[], Iterable[Panel]]) -> list:
    """`build()`'s panels, or none when the shot lacks one of its inputs."""
    try:
        return list(build())
    except MISSING as error:
        log.info("shot %s: no %s panel: %s", shot, what, error)
        return []


def finite(y) -> np.ndarray:
    """One trace as float32, its gaps filled with its median; no data raises."""
    y = np.asarray(y, dtype=np.float32)
    ok = np.isfinite(y)
    if not ok.any():
        raise NoDataError("the trace has no finite sample")
    return np.where(ok, y, np.median(y[ok])).astype(np.float32)


def stft(x_ms, y, *, rate_hz: float, nperseg: int, hop: int):
    """`(t_ms, f_hz, spec)`: one trace resampled to `rate_hz`, a Hann STFT.

    The rate comes from the span, never a median step: float32 time vectors
    quantise. `y` may be `(T,)` or `(C, T)`.
    """
    x_ms = np.asarray(x_ms, dtype=np.float64)
    if len(x_ms) < 2 or x_ms[-1] <= x_ms[0]:
        raise NoDataError(f"{len(x_ms)} samples do not span a spectrogram")
    rate = (len(x_ms) - 1) / ((x_ms[-1] - x_ms[0]) / 1000)
    ratio = Fraction(rate_hz / rate).limit_denominator(100)
    values = signal.resample_poly(y, ratio.numerator, ratio.denominator, axis=-1)
    if values.shape[-1] < nperseg:
        raise NoDataError(f"{values.shape[-1]} samples, fewer than {nperseg}")
    fs = rate * ratio.numerator / ratio.denominator
    f_hz, t_s, spec = signal.stft(
        values,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary=None,
        padded=False,
        axis=-1,
    )
    return x_ms[0] + t_s * 1000, f_hz, spec


def above_floor_db(power, quantile: float = 0.5) -> np.ndarray:
    """Power in dB above each frequency bin's `quantile` over the shot.

    The median suits bursts. A mode that holds one frequency for most of the
    shot would sit at its own median and vanish, so its panel takes a lower
    quantile.
    """
    db = 10 * np.log10(np.asarray(power, dtype=np.float64) + 1e-30)
    return db - np.quantile(db, quantile, axis=-1, keepdims=True)


def power_panel(title: str, x_ms, y, *, rate_hz, nperseg, hop, max_khz, bands=()):
    """One trace's power spectrogram, 0 to `max_khz`, on the shared scale."""
    t_ms, f_hz, spec = stft(x_ms, finite(y), rate_hz=rate_hz, nperseg=nperseg, hop=hop)
    keep = f_hz <= max_khz * 1000
    return Panel(
        title=title,
        kind="heatmap",
        x=t_ms,
        y=f_hz[keep] / 1000,
        z=above_floor_db(np.abs(spec[keep]) ** 2),
        ylabel="kHz",
        bands=list(bands),
        zmin=Z_DB[0],
        zmax=Z_DB[1],
    )


def bin_mean(x_ms, y, width_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """`(times, means)` of `(C, T)` samples over bins `width_ms` wide.

    A bin's time is the mean of its samples' times, so none lies outside the
    record; a bin with no sample is left out, one with only gaps is nan.
    """
    x_ms = np.asarray(x_ms, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if not len(x_ms):
        raise NoDataError("no samples to average")
    index = np.floor((x_ms - x_ms[0]) / width_ms).astype(np.int64)
    n = int(index[-1]) + 1
    taken = np.bincount(index, minlength=n)
    used = taken > 0
    times = np.bincount(index, x_ms, minlength=n)[used] / taken[used]
    ok = np.isfinite(y)
    sums = np.stack(
        [np.bincount(index, np.where(k, v, 0.0), minlength=n) for v, k in zip(y, ok)]
    )
    counts = np.stack([np.bincount(index, k.astype(float), minlength=n) for k in ok])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums[:, used] / counts[:, used]
    return times, means.astype(np.float32)


def betan_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    """beta_N from the features store; seconds on disk, milliseconds here."""
    paths = Paths.from_env() if paths is None else paths
    betan = read_feature(paths.features_file(int(shot)), "betan")
    x, y = betan.x * 1000.0, np.atleast_2d(betan.y)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    return [Panel(title="beta_N", x=x, y=y, ylabel="β_N")]
