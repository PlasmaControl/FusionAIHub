"""The 1/1 mode and the sawtooth: the ECE Te map, q_min, n1rms, D-alpha, mhr B1.

`BUILDERS["mhd_n1.m1"]` and `MARKERS["mhd_n1.m1"]`. A sawtooth crash is the 1/1
mode's collapse: the core's Te falls and the Te outside the inversion radius
rises at the same instant, then the mode that rose over the ramp is gone. The
rows (the sawtooth editor's, `sawtooth_oscillation`, drew the channels one group
at a time; this one draws all of them):

- `ECE Te map`: every radiometer channel in radius against time, the channels
  sorted by radius (the radiometer is second-harmonic X-mode, so a channel's
  frequency fixes its radius, and the radius falls as the channel number rises);
- `q=1 radius`: where the EFIT q = 1 surface falls on that axis, when the cached
  ECE geometry (`ece_geometry`: a calibrated psi per channel, EFIT's qpsi) holds
  it; no shot of the roster has one yet, so the row is usually left out;
- `q_min (EFIT01)` with a line at q = 1, `n1rms` (the n=1 magnetic RMS), D-alpha
  PCPHD03 (an ELM is a spike here and a drop in the off-axis Te), and B1's
  2-30 kHz spectrogram (`fishbone`), where the mode's precursor shows.

There is no crash-score row and no crash-step map: the crashes are only the
markers (`markers`), found by `crash.find_crashes` (the notebook's step in noise
units, ported) on the core channel, one channel 0.20 m outboard of it and one
0.30 m, inside the Ip flat-top.

The radius is nominal, not calibrated. `ECE_GHZ` is the radiometer's channel
frequencies (the notebook's `\\ELECTRONS::TOP.ECE.SETUP:FREQ`, the same on all
five shots it cached, 191871 to 206279). The core channel, the one the crash
search reads, is the one nearest the magnetic axis, as `dev/label_dev/sawtooth.ipynb`
chooses it: R = 2 x 27.992 GHz/T x Bt0 x 1.6955 m / f (`sawtooth.geometry`'s
`second_harmonic_R`) with Bt0 and the axis (`r0`) their medians over the plasma
window; the map's radius axis is that same R. Where a calibrated ECE psi is cached
(`ece_geometry`) the core is its least-psi channel, and where Bt or the axis is
missing it is the hottest channel, with the radius taking that channel to sit at
`R_AXIS_M`; the caveat says which. Channels 38-40 (the radiometer's band edge and
harmonic overlap) are never the core nor an off-axis channel.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ...config import Paths
from ...sawtooth import geometry
from .. import crash, equilibrium, spans
from ..raw import raw_signal
from ..verify import NoDataError, Panel
from . import ece_geometry, fishbone
from ._shared import (
    MISSING,
    despike,
    optional,
    plasma_window,
    with_caveat,
)
from .edge_localized_mode import flat, pcphd03_panel

#: The radiometer's first 40 channels, GHz (the corpus's `ece` rows 0-39; rows
#: 40-47 are not radiometer channels). 83.5 GHz up in 1 GHz steps to channel 16
#: (98.5 twice), then 1 GHz to 113.5, then 2 GHz to 129.5.
ECE_GHZ = np.r_[
    83.5 + np.arange(15), 98.499, 82.5 + np.arange(16, 32), 115.5 + 2.0 * np.arange(8)
]
N_CHANNELS = len(ECE_GHZ)
#: Where the hottest channel is taken to sit when there is no Bt or axis to place
#: the channels with, m (`sawtooth.ipynb`'s fallback axis).
R_AXIS_M = 1.75
#: Where DIII-D defines Bt0, m (the second-harmonic radius is R = 2 x 27.992 GHz/T
#: x Bt0 x this / f), and the axis R this side of which the EFIT `r0` is believed.
FIELD_REFERENCE_M = 1.6955
AXIS_RANGE_M = (1.0, 2.5)
#: Channels 38-40 (1-based; rows 37-39): the radiometer's band edge and where its
#: second harmonic overlaps the next. Never the core nor an off-axis channel.
EXCLUDED_CHANNELS = (37, 38, 39)
#: The off-axis channels the crash score reads, m outboard of the core's, and
#: how far a channel may be from that and still stand for it.
OFF_AXIS_M = (0.20, 0.30)
OFF_AXIS_TOLERANCE_M = 0.04
#: The map's time bin (the review grid's finest column is 0.05 ms) and its radius
#: step, m; a channel is 1.6-3.5 cm from the next.
MAP_BIN_MS = 0.1
MAP_STEP_M = 0.015
#: The crash search reads this much either side of the plasma window, ms: the
#: noise of a channel is its median step over 200 ms, which wants the context.
CONTEXT_MS = 250.0
#: Channels per read, so a 6 s record at 500 kHz is never held 40 deep.
CHUNK = 8
#: Samples per channel the hottest channel is chosen on: every 50th, 0.1 ms.
CHOOSE_STRIDE = 50
MAP_TITLE = "ECE Te map"
Q1_TITLE = "q=1 radius"
TITLES = (
    MAP_TITLE,
    Q1_TITLE,
    "q_min (EFIT01)",
    "n1rms (n=1 magnetic RMS)",
    "D-alpha PCPHD03",
    fishbone.TITLES[0],
)


def nominal_radius(f_ghz, f_core_ghz, r_core_m=R_AXIS_M) -> np.ndarray:
    """Each channel's nominal radius, m: `r_core_m * f_core / f`."""
    return r_core_m * f_core_ghz / np.asarray(f_ghz, dtype=float)


def bin_median(x_ms, y, width_ms) -> tuple[np.ndarray, np.ndarray]:
    """`(times, medians)` of `(C, T)` samples over bins `width_ms` wide.

    A bin's time is the mean of its samples' times; the median drops the
    radiometer's 1-4 sample spikes (`sawtooth_oscillation.ECE_BIN_MS`) where a
    mean would spread them.
    """
    x = np.asarray(x_ms, dtype=np.float64)
    index = np.floor((x - x[0]) / width_ms).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, np.diff(index) > 0])
    taken = np.diff(np.r_[starts, len(index)])
    return np.add.reduceat(x, starts) / taken, despike(x, y, width_ms)[:, starts]


def _read(shot, channels, *, t_range, paths):
    return raw_signal(
        int(shot), "ece", channels=list(channels), t_range=t_range, paths=paths
    )


def _channel_te(shot, *, t_range, paths) -> tuple[np.ndarray, np.ndarray]:
    """`(t_ms, Te (40, T))` binned to `MAP_BIN_MS`, read `CHUNK` channels at a time."""
    times, rows = None, []
    for first in range(0, N_CHANNELS, CHUNK):
        chunk = range(first, min(first + CHUNK, N_CHANNELS))
        array = _read(shot, chunk, t_range=t_range, paths=paths)
        if array.y.shape[0] < len(chunk):
            raise NoDataError(f"shot {int(shot)}: ECE has {array.y.shape[0]} channels")
        times, binned = bin_median(array.x, array.y, MAP_BIN_MS)
        rows.append(binned)
    return times, np.concatenate(rows)


def _median(y) -> np.ndarray:
    """`(C,)` medians of `(C, T)`, a gap counting as -inf: a channel that is dark
    for half the record is the coldest, not an error."""
    return np.median(np.where(np.isfinite(y), y, -np.inf), axis=1)


class Radii(NamedTuple):
    """Where the channels sit and which one is the core."""

    radius_m: np.ndarray  # (N_CHANNELS,) nominal radius of each channel, m
    core: int  # channel index (0-based)
    how: str  # "psi", "axis" or "hottest": what chose the core
    bt0: float = np.nan  # T, the median over the plasma window (how == "axis")
    axis_m: float = np.nan  # m, likewise

    def caveat(self) -> str:
        """What the map's radius axis and core are, for its caveat."""
        core = {
            "axis": f"the one nearest the EFIT magnetic axis ({self.axis_m:.2f} m)",
            "psi": "the one with the least calibrated ECE psi",
            "hottest": "the hottest",
        }[self.how]
        if np.isfinite(self.bt0):
            return (
                f"channels at R = 2 x 27.992 GHz/T x Bt0 x {FIELD_REFERENCE_M:g} m / f "
                f"with Bt0 = {self.bt0:.2f} T, nominal, no relativistic correction; "
                f"the core (channel {self.core + 1}) is {core}"
            )
        return (
            f"channels sorted by nominal radius, taking the core (channel "
            f"{self.core + 1}, {core}) to sit at {R_AXIS_M:g} m, good to a few cm; "
            "no Bt0 or EFIT axis was available to place them"
        )


def usable(n=N_CHANNELS) -> np.ndarray:
    """`(n,)` bool: False for the channels `EXCLUDED_CHANNELS`."""
    keep = np.ones(n, dtype=bool)
    keep[[c for c in EXCLUDED_CHANNELS if c < n]] = False
    return keep


def _plasma_median(x_s, y, window) -> float:
    """The median of the finite samples of `y` over `window` (ms; the record
    without, or when it holds none of the samples). `NoDataError` when none."""
    values = np.asarray(y, dtype=float)
    inside = np.ones(len(values), dtype=bool)
    if window is not None:
        t = np.asarray(x_s, dtype=float) * 1000.0
        if ((t >= window[0]) & (t <= window[1])).any():
            inside = (t >= window[0]) & (t <= window[1])
    values = values[inside]
    values = values[np.isfinite(values)]
    if not values.size:
        raise NoDataError("no finite samples")
    return float(np.median(values))


def axis_radii(shot, window, *, paths) -> tuple[np.ndarray, float, float]:
    """`(radius_m, Bt0, axis_m)`: each channel's second-harmonic radius at the
    plasma window's median |Bt| (`sawtooth.geometry.second_harmonic_R`) and the
    window's median EFIT axis R. `MISSING` when either is not cached (or fetched)
    or the axis is not a tokamak's."""
    bt = equilibrium.signal(int(shot), "bt", paths, fetch=True)
    axis = equilibrium.signal(int(shot), "r0", paths, fetch=True)
    bt0 = abs(_plasma_median(bt.x, bt.y[0], window))
    axis_m = _plasma_median(axis.x, axis.y[0], window)
    if not bt0 > 0 or not AXIS_RANGE_M[0] < axis_m < AXIS_RANGE_M[1]:
        raise NoDataError(f"shot {int(shot)}: Bt0 {bt0:g} T, axis {axis_m:g} m")
    # One time, the middle of a flat Bt0 clock: the radius is the same at every one.
    radius = geometry.second_harmonic_R(
        np.array([0.5]),
        ECE_GHZ * 1e9,
        (np.array([0.0, 1.0]), np.array([bt0, bt0])),
        FIELD_REFERENCE_M,
    )[:, 0]
    return radius, bt0, axis_m


def psi_core(shot, window, *, paths) -> int:
    """The channel with the least calibrated psi (the one nearest the axis),
    median over `window`; `MISSING` without a cached `ece_psi`."""
    got = ece_geometry.load_geometry(int(shot), paths)
    if got.psi is None:
        raise NoDataError(f"shot {int(shot)}: no calibrated ECE psi")
    psi = np.asarray(got.psi, dtype=float)[:N_CHANNELS]
    t = np.asarray(got.time_ms, dtype=float)
    inside = np.ones(len(t), dtype=bool)
    if window is not None and ((t >= window[0]) & (t <= window[1])).any():
        inside = (t >= window[0]) & (t <= window[1])
    median = np.array(
        [
            np.median(row[inside][np.isfinite(row[inside])])
            if np.isfinite(row[inside]).any()
            else np.inf
            for row in psi
        ]
    )
    median[~usable(len(median))] = np.inf
    if not np.isfinite(median).any():
        raise NoDataError(f"shot {int(shot)}: no finite ECE psi")
    return int(np.argmin(median))


def choose(shot, window, *, paths, hottest_channel) -> Radii:
    """The core and the radii: the calibrated-psi channel, or the one nearest the
    EFIT axis, or `hottest_channel()` (called only then, it reads the record)."""
    try:
        radius, bt0, axis_m = axis_radii(shot, window, paths=paths)
    except MISSING:
        radius, bt0, axis_m = None, np.nan, np.nan
    try:
        core, how = psi_core(shot, window, paths=paths), "psi"
    except (*MISSING, ValueError):
        core, how = None, None
    if core is None and radius is not None:
        gap = np.abs(radius - axis_m)
        gap[~usable()] = np.inf
        core, how = int(np.argmin(gap)), "axis"
    if core is None:
        core, how = int(hottest_channel()), "hottest"
    if radius is not None:
        return Radii(radius, core, how, bt0, axis_m)
    return Radii(nominal_radius(ECE_GHZ, ECE_GHZ[core]), core, how)


def hottest(te, t_ms, window) -> int:
    """The channel with the highest median Te over `window` (the record without),
    never one of `EXCLUDED_CHANNELS`."""
    inside = np.ones(len(t_ms), dtype=bool)
    if window is not None and ((t_ms >= window[0]) & (t_ms <= window[1])).any():
        inside = (t_ms >= window[0]) & (t_ms <= window[1])
    median = _median(te[:, inside])
    median[~usable(len(median))] = -np.inf
    return int(np.argmax(median))


def map_rows(te, radius_m) -> tuple[np.ndarray, np.ndarray]:
    """`(R, z)`: `te` `(C, T)` on a uniform radius axis, ascending, each row the
    nearest channel's (`radius_m` falls with the channel number)."""
    r = np.arange(radius_m.min(), radius_m.max() + MAP_STEP_M, MAP_STEP_M)
    nearest = np.abs(radius_m[None, :] - r[:, None]).argmin(axis=1)
    return r, te[nearest]


def q1_radius(geometry, radius_m) -> tuple[np.ndarray, np.ndarray]:
    """`(t_ms, R)`: where the q = 1 surface falls on the radiometer's radii.

    `geometry` is `ece_geometry.Geometry` with a calibrated `psi` `(C, T)` and
    the EFIT `q1_psi` `(T,)`. At each geometry time the channels from the
    outermost to the one nearest the axis (the least psi) are the outboard
    side, where psi rises with radius; the radius at `q1_psi` is interpolated
    along them, NaN where there is no q = 1 surface or it lies outside them.
    """
    if geometry.psi is None or geometry.q1_psi is None:
        raise NoDataError("the cached ECE geometry holds no psi or q = 1 surface")
    n = min(N_CHANNELS, geometry.psi.shape[0], len(radius_m))
    psi = np.asarray(geometry.psi, dtype=float)[:n]
    out = np.full(psi.shape[1], np.nan)
    for t in range(psi.shape[1]):
        target = geometry.q1_psi[t]
        column = psi[:, t]
        if not np.isfinite(target) or not np.isfinite(column).any():
            continue
        axis = int(np.nanargmin(column))
        side = column[axis::-1]  # axis -> outermost: psi up, radius up
        ok = np.isfinite(side)
        if ok.sum() < 2 or not np.all(np.diff(side[ok]) > 0):
            continue
        out[t] = np.interp(target, side[ok], radius_m[axis::-1][ok], np.nan, np.nan)
    return np.asarray(geometry.time_ms, dtype=float), out


def ece_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The ECE Te map, and the q = 1 radius when the cached geometry holds it."""
    paths_ = Paths.from_env() if paths is None else paths
    window = plasma_window(shot, paths_)
    t_ms, te = _channel_te(shot, t_range=t_range, paths=paths)
    placed = choose(
        shot, window, paths=paths_, hottest_channel=lambda: hottest(te, t_ms, window)
    )
    radius = placed.radius_m
    r, z = map_rows(te, radius)
    built = [
        with_caveat(
            Panel(
                title=MAP_TITLE,
                kind="heatmap",
                x=t_ms,
                y=r,
                z=z,
                ylabel="R (m)",
            ),
            f"Te in keV, each channel's median over {MAP_BIN_MS:g} ms; "
            f"{placed.caveat()}; channels 38-40 are never the core",
        )
    ]
    try:
        calibrated = ece_geometry.load_geometry(int(shot), paths_)
        x, line = q1_radius(calibrated, radius)
    except MISSING:
        return built
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, line = x[keep], line[keep]
    if len(x) and np.isfinite(line).any():
        built.append(
            with_caveat(
                Panel(title=Q1_TITLE, x=x, y=line[None], ylabel="R (m)"),
                "EFIT q = 1 surface mapped through the cached calibrated ECE psi, "
                "onto the map's nominal radii",
            )
        )
    return built


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    paths_ = Paths.from_env() if paths is None else paths
    window = plasma_window(shot, paths)
    return (
        optional("ECE", shot, lambda: ece_panels(shot, **kwargs))
        + optional(
            "q_min",
            shot,
            lambda: equilibrium.line_panel(
                shot, "qmin", TITLES[2], ylabel="q", hlines=[1.0],
                t_range=t_range, paths=paths_,
            ),
        )
        + optional(
            "n1rms",
            shot,
            lambda: equilibrium.line_panel(
                shot, "n1rms", TITLES[3], ylabel="G", t_range=t_range, paths=paths_
            ),
        )
        + optional(
            "PCPHD03",
            shot,
            lambda: pcphd03_panel(shot, window=window, **kwargs),
        )
        + optional("B1", shot, lambda: fishbone.b1_panel(shot, **kwargs))
    )


def _off_axis(radius_m, core) -> list[int]:
    """The channels `OFF_AXIS_M` outboard of `core` (within the tolerance), each
    once, never the core nor one of `EXCLUDED_CHANNELS`."""
    chosen = []
    for offset in OFF_AXIS_M:
        gap = np.abs(radius_m - (radius_m[core] + offset))
        gap[~usable(len(gap))] = np.inf
        channel = int(np.argmin(gap))
        if gap[channel] <= OFF_AXIS_TOLERANCE_M and channel not in chosen + [core]:
            chosen.append(channel)
    return chosen


def crash_times(shot, *, paths=None) -> np.ndarray:
    """The shot's crash times (ms) inside its Ip flat-top (after t = 0 without a
    plasma window), sorted.

    The core is the channel `choose` gives (the least calibrated psi, else the one
    nearest the EFIT axis at the window's Bt0, else, reading the record, the one
    with the highest median Te over the plasma window), the off-axis channels the
    ones nearest `OFF_AXIS_M` outboard of it on the same radii; `crash.find_crashes`
    runs on them at the radiometer's own rate, with PCPHD03 (unless flat) to tell
    ELMs from crashes. `NoDataError` and its kin when the ECE record is missing.
    """
    paths_ = Paths.from_env() if paths is None else paths
    window = plasma_window(shot, paths_)
    span = None
    if window is not None:
        span = (window[0] - CONTEXT_MS, window[1] + CONTEXT_MS)
    # Without a window nothing before the discharge (t < 0) counts.
    lo, hi = window if window is not None else (0.0, np.inf)

    def hottest_channel() -> int:
        medians = []
        for first in range(0, N_CHANNELS, CHUNK):
            chunk = range(first, min(first + CHUNK, N_CHANNELS))
            array = _read(shot, chunk, t_range=span, paths=paths)
            inside = (array.x >= lo) & (array.x <= hi)
            medians.append(_median(array.y[:, inside][:, ::CHOOSE_STRIDE]))
        median = np.concatenate(medians)
        median[~usable(len(median))] = -np.inf
        return int(np.argmax(median))

    placed = choose(shot, window, paths=paths_, hottest_channel=hottest_channel)
    core, radius = placed.core, placed.radius_m
    outer = _off_axis(radius, core)
    if not outer:
        return np.empty(0)
    array = _read(shot, [core, *outer], t_range=span, paths=paths)
    dalpha = None
    try:
        trace = raw_signal(int(shot), "pcphd03", paths=paths)
        if not flat(trace.x, trace.y, window):
            dalpha = (np.asarray(trace.x), np.asarray(trace.y)[0])
    except MISSING:
        pass
    times = crash.find_crashes(array.x, array.y[0], array.y[1:], dalpha)
    flattop = (lo, hi)
    if window is not None:
        flattop = spans.plasma_flattop(int(shot), paths_, window) or window
    return times[(times >= flattop[0]) & (times <= flattop[1])]


def markers(shot, *, t_range=None, paths=None) -> list[dict]:
    """`[{"label": "crash", "t_ms": [...]}]` inside `t_range`, or `[]` when the
    shot has no ECE (or no crash)."""
    try:
        times = crash_times(shot, paths=paths)
    except (*MISSING, ValueError):
        return []
    if t_range is not None:
        times = times[(times >= t_range[0]) & (times <= t_range[1])]
    if not len(times):
        return []
    return [{"label": "crash", "t_ms": [float(t) for t in times]}]
