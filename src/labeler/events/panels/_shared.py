"""What the review editors' panels share.

A spectrogram on one colour scale for every shot, a trace averaged into
bins, and the rule for an input a shot does not have: that panel is left
out (and logged), the rest of the shot is still drawn.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from fractions import Fraction

import numpy as np
from scipy import signal

from ...config import Paths
from ...features.store import read_feature
from .. import spans
from ..raw import FetchDisabledError
from ..verify import NoDataError, Panel

log = logging.getLogger(__name__)
#: What a shot may lack: a raw group (NoDataError), a feature (KeyError), a file.
MISSING = (NoDataError, KeyError, OSError)
#: dB above each frequency bin's own median, the scale every spectrogram shares.
Z_DB = (-3.0, 27.0)
#: A trace's robust range: these percentiles over the plasma window, each side
#: widened by `ROBUST_MARGIN` of the span between them. The page scales a trace
#: to what it holds, so one spike at the end of a discharge (PCPHD03 on 192003:
#: 3.6 V against 0.02 V ELMs) would flatten everything else in the row.
ROBUST_PERCENTILES = (0.5, 99.5)
ROBUST_MARGIN = 1.0
#: What a row's title says when `robust_clip` moved a sample of it.
CLIPPED = ", clipped to its plasma range"


def optional(what: str, shot: int, build: Callable[[], Iterable[Panel]]) -> list:
    """`build()`'s panels, or none when the shot lacks one of its inputs."""
    try:
        return list(build())
    except MISSING as error:
        # Fetching off (a job) is a gap the job log should show, not a quiet one.
        disabled = isinstance(error, FetchDisabledError)
        level = logging.WARNING if disabled else logging.INFO
        log.log(level, "shot %s: no %s panel: %s", shot, what, error)
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


def above_floor_db(power, quantile: float = 0.5, columns=None) -> np.ndarray:
    """Power in dB above each frequency bin's `quantile` over the shot.

    The median suits bursts. A mode that holds one frequency for most of the
    shot would sit at its own median and vanish, so its panel takes a lower
    quantile. `columns` (a mask over the last axis) takes the floor over those
    columns only; every column is still returned.
    """
    db = 10 * np.log10(np.asarray(power, dtype=np.float64) + 1e-30)
    over = db if columns is None else db[..., np.asarray(columns, dtype=bool)]
    return db - np.quantile(over, quantile, axis=-1, keepdims=True)


def plasma_window(shot: int, paths: Paths | None = None):
    """`(start_ms, end_ms)`, the shot's v1 rule-4 Ip window, or None.

    The cohort's (`spans.queue`, every roster shot) first, then the population's
    (`spans.population`); a table that cannot be read is passed over.
    """
    paths = Paths.from_env() if paths is None else paths
    for table in (spans.queue, spans.population):
        try:
            frame = table(paths)
        except (OSError, ValueError, KeyError) as error:
            log.warning("shot %s: no %s window: %s", shot, table.__name__, error)
            continue
        for row in frame[frame.shot == int(shot)].itertuples(index=False):
            start, end = float(row.window_start_ms), float(row.window_end_ms)
            if np.isfinite(start) and np.isfinite(end) and end > start:
                return start, end
    return None


def plasma_columns(t_ms, band_power, window) -> np.ndarray:
    """Which spectrogram columns are the plasma's, for its floor to be taken over.

    A probe's record runs seconds past the plasma, and a floor over the whole
    record falls there and pins the plasma at the top of the scale. The columns
    inside `window` when it holds any; otherwise those whose `band_power` is
    above the record's median, which are the plasma's while it fills at least
    40 % of the record (a floor at the 20th percentile of that half stays on it).
    """
    t_ms = np.asarray(t_ms, dtype=float)
    if window is not None:
        inside = (t_ms >= window[0]) & (t_ms <= window[1])
        if inside.any():
            return inside
    band_power = np.asarray(band_power, dtype=float)
    return band_power > np.median(band_power)


def power_panel(
    title: str, x_ms, y, *, rate_hz, nperseg, hop, max_khz, bands=(), plasma=False
):
    """One trace's power spectrogram, 0 to `max_khz`, on the shared scale.

    `plasma` is the window (`plasma_window`) whose columns the floor is taken
    over (`plasma_columns`), or False for the whole record's.
    """
    t_ms, f_hz, spec = stft(x_ms, finite(y), rate_hz=rate_hz, nperseg=nperseg, hop=hop)
    keep = f_hz <= max_khz * 1000
    power = np.abs(spec[keep]) ** 2
    columns = (
        None if plasma is False else plasma_columns(t_ms, power.sum(axis=0), plasma)
    )
    return Panel(
        title=title,
        kind="heatmap",
        x=t_ms,
        y=f_hz[keep] / 1000,
        z=above_floor_db(power, columns=columns),
        ylabel="kHz",
        bands=list(bands),
        zmin=Z_DB[0],
        zmax=Z_DB[1],
    )


def robust_limits(x_ms, y, window, margin: float = ROBUST_MARGIN) -> np.ndarray:
    """`(2, C)`: each channel's robust range over `window` (the record without one).

    The `ROBUST_PERCENTILES` of its finite samples, pushed apart by `margin` of
    their span; nan for a channel with none.
    """
    x_ms = np.asarray(x_ms, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    inside = np.ones(len(x_ms), dtype=bool)
    if window is not None:
        inside = (x_ms >= window[0]) & (x_ms <= window[1])
        if not inside.any():
            inside[:] = True
    limits = np.full((2, len(y)), np.nan)
    for i, row in enumerate(y):
        v = row[inside][np.isfinite(row[inside])]
        if v.size:
            lo, hi = np.percentile(v, ROBUST_PERCENTILES)
            limits[:, i] = (lo - margin * (hi - lo), hi + margin * (hi - lo))
    return limits


def robust_clip(x_ms, y, window) -> tuple[np.ndarray, bool]:
    """`(y, clipped)`: `y` inside its `robust_limits`, and whether a sample was not.

    The page takes a trace row's y-range from its samples, so a clipped trace
    is how a store sets the range: the samples beyond it sit on its edge.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))
    lo, hi = (v[:, None] for v in robust_limits(x_ms, y, window))
    with np.errstate(invalid="ignore"):
        out = np.where(y < lo, lo, np.where(y > hi, hi, y)).astype(np.float32)
        clipped = bool(((y < lo) | (y > hi)).any())
    return out, clipped


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


def despike(x_ms, y, width_ms: float) -> np.ndarray:
    """`(C, T)`: each sample as the median of its bin, the bins `width_ms` wide
    from the first sample (`x_ms` ascending); a bin of only gaps stays nan.

    A spike narrower than half a bin is gone, where a mean would only spread it
    over the bin; a step stays a step, blurred into one bin at most. The samples
    keep their times, so a review grid built from them is the raw record's.
    """
    x_ms = np.asarray(x_ms, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))
    if not len(x_ms):
        return y
    index = np.floor((x_ms - x_ms[0]) / width_ms).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, np.diff(index) > 0])
    taken = np.diff(np.r_[starts, len(index)])
    bins = np.repeat(np.arange(len(starts)), taken)
    slot = np.arange(len(index)) - starts[bins]
    out = np.empty_like(y)
    for i, row in enumerate(y):  # one channel's (bins, samples) table at a time
        table = np.full((len(starts), taken.max()), np.nan, dtype=np.float32)
        table[bins, slot] = row
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            out[i] = np.nanmedian(table, axis=-1)[bins]
    return out


def betan_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    """beta_N from the features store; seconds on disk, milliseconds here."""
    paths = Paths.from_env() if paths is None else paths
    betan = read_feature(paths.features_file(int(shot)), "betan")
    x, y = betan.x * 1000.0, np.atleast_2d(betan.y)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    return [Panel(title="beta_N", x=x, y=y, ylabel="β_N")]
