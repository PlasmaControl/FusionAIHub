"""Sawtooth in ECE inversion groups, core Te, and crash-sensitive SXR chords.

Calibrated channel geometry separates the core from outside q = 1; without it,
adjacent channel groups and their baseline-subtracted comparison show inversion
without assigning a physical radius.

The ECE rows draw each 500 kHz sample as the median of its `ECE_BIN_MS`, so
the radiometer's spikes do not set the rows' range (`ece_panels`). The Te row
is Thomson scattering's hottest core chords, every 10 ms (`te_panels`).

The SXR row draws the `CHOSEN` chords of the first lit fan with the most
crash-like drops over the Ip flat-top (`crash_drops`), not the brightest: on
about 25 shots the brightest sit near 4.6 V and barely move (189061's chords 10
and 12, against 9 and 11 that crash). Its chords are clipped to their robust
range over the plasma window (`_shared.robust_clip`), as the ELM D-alpha rows
are, so a spike after the plasma does not set the row's scale. A shot without
ECE, Thomson or SXR gets the others' rows alone. An optional n1rms row shows
the n=1 magnetic RMS amplitude in gauss below the SXR row.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import uniform_filter1d

from ...config import Paths
from .. import equilibrium, spans
from ..heuristics import SXR_CHORDS, SXR_FANS, SXR_LIT_FRAC, SXR_MIN_CHORDS
from ..raw import raw_signal
from ..verify import NoDataError, Panel
from . import ece_geometry
from ._shared import (
    CLIPPED,
    despike,
    optional,
    plasma_window,
    robust_clip,
    robust_limits,
)

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
# The corpus stores channel numbers only. No calibrated ECE channel radius,
# equilibrium time base, or q=1 crossing is stored alongside it, so these are
# inversion sides rather than claims about core or outside-q=1 locations.
ECE_INVERSION_SIDES = ("A", "A", "B", "B")
ECE_GEOMETRY_NOTE = "q=1 mapping unavailable"
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
#: without a window it is the record's 10-14 s, which a plasma fills under half of
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
#: the record is drawn: its `CHOSEN` chords with the most crash-like drops. The
#: rule is `heuristics.SXR_FANS`, which the sawtooth detector's SXR part reads.
SXR_ARRAYS = SXR_FANS
CHORDS = SXR_CHORDS
MIN_CHORDS = SXR_MIN_CHORDS
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
                title=(
                    f"ECE Te, inversion side {ECE_INVERSION_SIDES[len(built)]}, "
                    f"ch {row[0]}-{row[-1]} ({ECE_BIN_MS:g} ms median; "
                    f"{ECE_GEOMETRY_NOTE})"
                ),
                x=array.x,
                y=despike(array.x, array.y, ECE_BIN_MS),
                ylabel="keV",
                legend=[f"ch {c}" for c in row],
            )
        )
    return built


def _mapped_ece_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    geometry = ece_geometry.load_geometry(int(shot), paths)
    q_x, q_values = geometry.time_ms, geometry.q
    array = raw_signal(
        int(shot), "ece", channels=list(range(q_values.shape[0])),
        t_range=t_range, paths=paths
    )
    if geometry.psi is None:
        core_geometry, outside_geometry = ece_geometry.classify_q1(q_values)
    else:
        core_geometry, outside_geometry = ece_geometry.surface_membership(
            geometry.psi, geometry.q1_psi
        )
    core = ece_geometry.align_mask(array.x, q_x, core_geometry)
    outside = ece_geometry.align_mask(array.x, q_x, outside_geometry)
    values = despike(array.x, array.y, ECE_BIN_MS)
    panels = []
    names = (
        ("q<1 inversion group", core),
        ("q>1 inversion group", outside),
    )
    if geometry.psi is not None:
        names = (("core q<1", core), ("outside q=1", outside))
    for name, mask in names:
        selected = np.arange(min(values.shape[0], mask.shape[0]))
        visible = mask[selected].any(axis=1)
        if not visible.any():
            continue
        y = np.where(mask[selected[visible]], values[selected[visible]], np.nan)
        panels.append(
            Panel(
                title=f"ECE Te, {name} (measured q geometry)",
                x=array.x,
                y=y,
                ylabel="keV",
                legend=[f"ch {c}" for c in selected[visible]],
            )
        )
    if not panels:
        raise NoDataError("ECE q geometry has no valid q<1 or q>1 samples")
    return panels


def ece_inversion_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    array = raw_signal(
        int(shot), "ece", channels=list(range(36)), paths=paths
    )
    x, y = ece_inversion_from_arrays(
        array.x, despike(array.x, array.y, ECE_BIN_MS), t_range=t_range
    )
    return _inversion_panel(x, y)


def ece_inversion_from_arrays(x, values, *, t_range=None):
    channels = tuple(range(16)) if np.asarray(values).shape[0] < 36 else None
    if channels is None:
        x, y = ece_geometry.inversion_difference(x, values)
    else:
        x, y = ece_geometry.inversion_difference(x, values, channels=channels)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    return x, y


def _inversion_panel(x, y):
    return [
        Panel(
            title="ECE inversion side comparison (q=1 mapping unavailable)",
            x=x,
            y=y,
            ylabel="ΔTe, baseline-subtracted (keV)",
            legend=["side A: ch 20-27", "side B: ch 28-35"],
        )
    ]


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
    brighter first among equals, chosen and clipped over the whole record
    whatever the view."""
    span = chord_span(shot, paths)
    for name, first in SXR_ARRAYS:
        rows = list(range(first, first + CHORDS))
        array = raw_signal(int(shot), "sxr", channels=rows, paths=paths)
        y = np.asarray(array.y)
        lit = np.isfinite(y).mean(axis=1) >= SXR_LIT_FRAC
        if lit.sum() < MIN_CHORDS:
            continue
        drops = np.full(CHORDS, -1)
        drops[lit] = crash_drops(array.x, y[lit], span)
        level = np.full(CHORDS, -np.inf)
        level[lit] = np.nanmedian(y[lit], axis=1)
        top = np.sort(np.lexsort((-level, -drops))[:CHOSEN])
        x = np.asarray(array.x)
        chosen, clipped = robust_clip(x, y[top], plasma_window(shot, paths))
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
            x, chosen = x[keep], chosen[:, keep]
        return [
            Panel(
                title=f"SXR {name}, the {CHOSEN} chords with the most crash-like drops"
                + (CLIPPED if clipped else ""),
                x=x,
                y=chosen,
                legend=[f"{name}{c + 1:02d}" for c in top],
            )
        ]
    raise NoDataError(f"shot {int(shot)}: no SXR fan has {MIN_CHORDS} finite chords")


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    ece = optional("ECE", shot, lambda: _mapped_ece_panels(shot, **kwargs))
    inversion = []
    if not ece:
        ece = optional("ECE", shot, lambda: ece_panels(shot, paths=paths))
        if len(ece) >= 4:
            inversion = _inversion_panel(
                *ece_inversion_from_arrays(
                    ece[0].x,
                    np.concatenate([panel.y for panel in ece[:4]], axis=0),
                    t_range=t_range,
                )
            )
        if t_range is not None:
            viewed = []
            for panel in ece:
                keep = (panel.x >= t_range[0]) & (panel.x <= t_range[1])
                viewed.append(replace(panel, x=panel.x[keep], y=panel.y[:, keep]))
            ece = viewed
    return (
        ece
        + optional("Te", shot, lambda: te_panels(shot, **kwargs))
        + optional("SXR", shot, lambda: sxr_panels(shot, **kwargs))
        + optional(
            "n1rms",
            shot,
            lambda: equilibrium.line_panel(
                shot,
                "n1rms",
                "n1rms (n=1 magnetic RMS)",
                ylabel="G",
                t_range=t_range,
                paths=Paths.from_env() if paths is None else paths,
            ),
        )
        + inversion
    )
