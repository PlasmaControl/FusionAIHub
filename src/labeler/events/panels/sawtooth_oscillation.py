"""Sawteeth, as the inversion of adjacent ECE channels across the q = 1 surface,
the core electron temperature, and the soft X-ray chords that drop at each crash.

The ECE rows draw each 500 kHz sample as the median of its `ECE_BIN_MS`, so
the radiometer's spikes do not set the rows' range (`ece_panels`). The Te row
is Thomson scattering's hottest core chords, every 10 ms (`te_panels`).

The SXR row draws the `CHOSEN` chords of the first lit fan with the most
crash-like drops over the Ip flat-top (`crash_drops`), not the brightest: on
about 25 shots the brightest sit near 4.6 V and barely move (189061's chords 10
and 12, against 9 and 11 that crash). A shot without ECE, Thomson or SXR gets
the others' rows alone.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from ...config import Paths
from .. import spans
from ..raw import raw_signal
from ..verify import NoDataError, Panel
from ._shared import despike, optional, plasma_window, robust_limits

#: Four rows of four ADJACENT channels covering 20-35, sixteen in all. The
#: flip a sawtooth crash makes is a RELATIVE thing - inner channels drop as
#: outer ones rise - so channels are overplotted in adjacent groups rather
#: than drawn one per panel.
CHANNEL_ROWS = (
    (20, 21, 22, 23),
    (24, 25, 26, 27),
    (28, 29, 30, 31),
    (32, 33, 34, 35),
)
#: Each ECE sample as the median of its 0.05 ms (25 samples), the review grid's
#: finest column (`panel_rows.FINEST_DT_MS`), so the page loses no time it
#: could show. The radiometer's spikes are 1-4 samples wide and reach 22 keV
#: over a 3 keV core (185838 at 1.4 s; 44 keV at most): they set a trace row's
#: range and flattened the crashes under them. A median drops them where a mean
#: would spread them, and keeps a crash's drop within one bin (two columns on
#: the page, whose columns start half a column earlier). The samples keep their
#: times: one median a bin at its samples' mean time put the grid at 0.050000655
#: ms on the corpus's float32 seconds, and left 698 of its columns empty.
ECE_BIN_MS = 0.05
#: Thomson scattering's core Te, the corpus's `ts_core_temp`: 44 chords in eV
#: every 10 ms. Of the chords with a Te on at least half as many of the plasma
#: window's samples as the best-lit chord, the `TE_CHORDS` hottest by their
#: median over it are drawn in keV. A share of the best's, not of the window:
#: without a window it is the record's 14 s, which a plasma fills under half of
#: (192238: 45 %). A fit that failed reads 0 or below (1-9 % of samples) and is
#: left out. A bad one reads high (17 keV over 195786's 2 keV core, just after
#: its window) and would set the row's range, so the row is clipped above its
#: robust range (`_shared.robust_limits`) with a `TE_MARGIN` of its span, and
#: only above: a ramp's cooler Te stays as it is. The traces' margin of a whole
#: span left the opening view over 1.5 times the window's hottest Te on 58
#: roster shots; a quarter leaves at most 1.24.
TE_GROUP = "ts_core_temp"
TE_CHORDS = 4
TE_MARGIN = 0.25
TE_CLIPPED = ", clipped above its plasma range"
#: The SXR fans tried in order, by their first row in the corpus's 320 (32
#: chords each). The first with `MIN_CHORDS` chords finite over at least half
#: the record is drawn: its `CHOSEN` chords with the most crash-like drops.
SXR_ARRAYS = (
    ("SX90RM1F", 192),
    ("SX90RP1F", 256),
    ("SX90RM1S", 224),
    ("SX90RP1S", 288),
)
CHORDS = 32
MIN_CHORDS = 8
CHOSEN = 4
#: A crash-like drop: a sample where the `SMOOTH_SAMPLES`-sample mean falls by
#: more than `DROP_SIGMA` standard deviations of its own sample-to-sample
#: change, that deviation taken per `BLOCK_MS` of the flat-top so a stretch
#: where the fan misbehaves (189061's first second) sets only its own scale.
#: The judge's measure: on 189061 it ranks chords 8, 9, 11 and 14 first.
SMOOTH_SAMPLES = 5
DROP_SIGMA = 6.0
BLOCK_MS = 1000.0
MIN_BLOCK_SAMPLES = 20


def crash_drops(x_ms, y, span=None) -> np.ndarray:
    """Each row's crash-like drops inside `span` (ms; None: the whole record)."""
    x = np.asarray(x_ms, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    lo, hi = (x[0], x[-1]) if span is None else span
    counts = np.zeros(len(y), dtype=np.int64)
    for start in np.arange(lo, hi, BLOCK_MS):
        keep = (x >= start) & (x < min(start + BLOCK_MS, hi))
        if keep.sum() < MIN_BLOCK_SAMPLES:
            continue
        for row, values in enumerate(y[:, keep]):
            finite = np.isfinite(values)
            if finite.sum() < MIN_BLOCK_SAMPLES:
                continue
            values = np.where(finite, values, np.median(values[finite]))
            step = np.diff(uniform_filter1d(values, SMOOTH_SAMPLES, mode="nearest"))
            counts[row] += int(np.sum(step < -DROP_SIGMA * step.std()))
    return counts


def chord_span(shot, paths) -> tuple[float, float] | None:
    """Where the chords are compared: the Ip flat-top inside the plasma window,
    else the window, else (None) the whole record."""
    window = plasma_window(shot, paths)
    if window is None:
        return None
    paths = Paths.from_env() if paths is None else paths
    return spans.plasma_flattop(int(shot), paths, window) or window


def ece_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    built = []
    for row in CHANNEL_ROWS:
        array = raw_signal(
            int(shot), "ece", channels=list(row), t_range=t_range, paths=paths
        )
        built.append(
            Panel(
                title=f"ECE Te, ch {row[0]}-{row[-1]} ({ECE_BIN_MS:g} ms median)",
                x=array.x,
                y=despike(array.x, array.y, ECE_BIN_MS),
                ylabel="keV",
                legend=[f"ch {c}" for c in row],
            )
        )
    return built


def te_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The `TE_CHORDS` hottest Thomson core chords, chosen and clipped over the
    whole record whatever the view."""
    array = raw_signal(int(shot), TE_GROUP, paths=paths)
    x = np.asarray(array.x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        y = np.where(np.asarray(array.y) > 0, array.y / 1000, np.nan).astype(np.float32)
    window = plasma_window(shot, paths)
    inside = np.ones(len(x), dtype=bool)
    if window is not None and ((x >= window[0]) & (x <= window[1])).any():
        inside = (x >= window[0]) & (x <= window[1])
    share = np.isfinite(y[:, inside]).mean(axis=1)
    if not share.max() > 0:
        raise NoDataError(f"shot {int(shot)}: no Thomson core chord has a Te")
    lit = share >= 0.5 * share.max()
    level = np.full(len(y), -np.inf)
    level[lit] = np.nanmedian(y[lit][:, inside], axis=1)
    top = np.sort(np.argsort(-level, kind="stable")[: min(TE_CHORDS, lit.sum())])
    y = y[top]
    high = robust_limits(x, y, window, TE_MARGIN)[1][:, None]
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    with np.errstate(invalid="ignore"):
        clipped = bool((y > high).any())
        y = np.where(y > high, high, y).astype(np.float32)
    return [
        Panel(
            title=f"Te, Thomson core: the {len(top)} hottest chords"
            + (TE_CLIPPED if clipped else ""),
            x=x,
            y=y,
            ylabel="keV",
            legend=[f"chord {c}" for c in top],
        )
    ]


def sxr_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The first lit fan's `CHOSEN` chords with the most crash-like drops, the
    brighter first among equals, chosen over the whole record whatever the view."""
    span = chord_span(shot, paths)
    for name, first in SXR_ARRAYS:
        rows = list(range(first, first + CHORDS))
        array = raw_signal(int(shot), "sxr", channels=rows, paths=paths)
        y = np.asarray(array.y)
        lit = np.isfinite(y).mean(axis=1) >= 0.5
        if lit.sum() < MIN_CHORDS:
            continue
        drops = np.full(CHORDS, -1)
        drops[lit] = crash_drops(array.x, y[lit], span)
        level = np.full(CHORDS, -np.inf)
        level[lit] = np.nanmedian(y[lit], axis=1)
        top = np.sort(np.lexsort((-level, -drops))[:CHOSEN])
        x = np.asarray(array.x)
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
            x, y = x[keep], y[:, keep]
        return [
            Panel(
                title=f"SXR {name}, the {CHOSEN} chords with the most crash-like drops",
                x=x,
                y=y[top],
                legend=[f"{name}{c + 1:02d}" for c in top],
            )
        ]
    raise NoDataError(f"shot {int(shot)}: no SXR fan has {MIN_CHORDS} finite chords")


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return (
        optional("ECE", shot, lambda: ece_panels(shot, **kwargs))
        + optional("Te", shot, lambda: te_panels(shot, **kwargs))
        + optional("SXR", shot, lambda: sxr_panels(shot, **kwargs))
    )
