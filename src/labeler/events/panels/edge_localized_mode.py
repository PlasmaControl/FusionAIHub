"""ELMs: D-alpha on the divertor chords that see the strike points, ELM onsets,
and the interferometer's spectrogram.

An ELM is a burst on the divertor D-alpha 2-5 ms wide; on the interferometer it
is a broadband stripe. The rows, in order:

1. D-alpha summed over the filterscope chords that look at the outer divertor
   leg (`strike_dalpha`): which divertor (upper or lower) and which chords come
   from EFIT's strike points while Ip is at least 80 % of its 95th percentile.
   Straight after it comes the one filterscope the ELM spans read
   (`spans_dalpha`, `spans.dalpha_channel`: FS01, or FS02 where FS01 is dark),
   titled exactly `D-alpha FSnn`: it is the row the frames model and the paper
   read (`frames.features.find_role`), never the sum. Where the strike points
   cannot be read it is the only D-alpha row, and its caveat says why.
2. The jump-and-hold row (`jump_and_hold`): how far D-alpha rose in the next
   0.25 ms, over its 99.5th-percentile rise in the Ip flat-top (`flat_top`);
   its onsets (`onsets`, rising crossings of `JUMP_THRESHOLD` inside the
   flat-top) are the page's "ELM onset" markers (`markers`).
3. PCPHD03, left out where it cannot be read or is flat over the plasma
   (`flat`); the first row's caveat says which.
4. The CO2 R0 density at 0.1 ms, EFIT's stored energy W_MHD and, where a
   diamagnetic loop exists, the drift-corrected loop energy and the ELM losses
   matched to D-alpha (the last three are provisional).
5. The CO2 R0 chord's own power from 0 to 125 kHz, its floor over the plasma
   window.

Both D-alpha traces and PCPHD03 are clipped to their robust range over the
plasma window (`_shared.robust_clip`), so a spike at the end of the discharge
does not set a row's scale.

The strike-point chord choice and the jump-and-hold detector are reimplemented
from David Eldon's OMFIT ELM module (`select_filterscope` and
`jump_and_hold_detect`), as `adaptation_of_omfit_elm.py` in DIII-D's
`elm_detection_challenge` at commit b71f991 has them; nothing is copied.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from scipy import signal as sp_signal

from ...config import Paths
from ...features.store import FeatureArray
from .. import elm_energy, equilibrium, spans
from ..raw import FetchDisabledError, raw_signal
from ..verify import NoDataError, Panel
from ._shared import (
    CLIPPED,
    MISSING,
    ROBUST_PERCENTILES,
    bin_mean,
    optional,
    plasma_window,
    power_panel,
    robust_clip,
    with_caveat,
)

log = logging.getLogger(__name__)

#: R0 resampled to 250 kHz, 256-sample windows every 64: 0.98 kHz bins and
#: 0.256 ms columns, the AE rows' resolution, up to 125 kHz.
RATE_HZ = 250_000
NPERSEG = 256
HOP = 64
MAX_KHZ = 125.0
#: PCPHD03 is flat when its 0.5-99.5 percentile range over the plasma window
#: is under this, in V. On the 450 queue shots that range clusters at
#: 0.0075-0.0101 V (26 shots), the digitiser's noise, and none of the 28 under
#: 0.011 V shows more than 10 spike samples inside a suggested ELM span; from
#: 0.0125 V up the record can carry real ELMs (196093, 190602, 192766 show 180
#: to 469), so the 0.02 V a first survey suggested would drop some.
FLAT_V = 0.011
#: The CO2 density row's bin, ms.
DENSITY_BIN_MS = 0.1

DALPHA_STRIKE_TITLE = "D-alpha FS strike sum"
JUMP_TITLE = "ELM jump-and-hold"
PCPHD03_TITLE = "D-alpha PCPHD03"
DENSITY_TITLE = "CO2 R0 density"
POWER_TITLE = "CO2 R0 power"
WMHD_TITLE = "W_MHD"
LOOP_TITLES = ("Loop stored energy", "ELM energy loss", "ELM loss fraction")

#: The DIII-D filterscope chords' (R, Z) in metres, in the order Eldon's module
#: lists them. The corpus' `filterscopes` rows 0-7 are FS01-FS08
#: (`LOWER_CORPUS`); `filterscopes_up` (`raw.FETCH_SPECS`) holds `UPPER_FETCHED`.
CHORDS: dict[str, tuple[float, float]] = {
    "FS00": (1.015, -1.113),
    "FS01": (1.049, -1.256),
    "FS07": (1.114, -1.321),
    "FS08": (1.143, -1.352),
    "FS02": (1.225, -1.362),
    "FS03": (1.307, -1.362),
    "FS04": (1.347, -1.362),
    "FS05": (1.491, -1.252),
    "FS06": (1.602, -1.252),
    "FS13UP": (1.015, 0.616),
    "FS12UP": (1.015, 0.861),
    "FS11UP": (1.015, 1.144),
    "FS00UP": (1.152, 1.176),
    "FS01UP": (1.262, 1.293),
    "FS02UP": (1.413, 1.261),
    "FS03UP": (1.545, 1.151),
    "FS04UP": (1.696, 1.079),
    "FS12": (1.016, -0.140),
}
LOWER_CORPUS = tuple(f"FS{i:02d}" for i in range(1, 9))
UPPER_FETCHED = ("FS13UP", "FS12UP", "FS11UP", "FS00UP", "FS01UP", "FS02UP",
                 "FS03UP", "FS04UP")

#: A strike point counts while Ip is at least this fraction of its 95th percentile.
IP_FRACTION = 0.8
IP_PERCENTILE = 95
#: Below this |mean Z| (m) the discharge may be a double null: the strike point
#: can change side, so only the times on the side it is mostly on are averaged.
DOUBLE_NULL_Z = 0.8
#: Chords nearer the midplane than this (m) see neither divertor.
MIDPLANE_Z = 0.5
#: The centre post (m): an inner strike point on it is moved 1 mm off it.
CENTRE_POST_R = 1.015
CENTRE_POST_CLEARANCE = 1.016

#: The jump window (ms), the normalised jump above which an ELM starts, and the
#: presmoothing's timescale (ms).
JUMP_WINDOW_MS = 0.25
JUMP_THRESHOLD = 0.12
PRESMOOTH_MS = 0.1
#: The jump leads its ELM by the window, so it is shifted back by this part of
#: the window left over once the threshold's share of the rise is taken.
ROLLBACK = 0.85
#: The rise is divided by this percentile of itself in the flat-top. Eldon divides
#: by the largest (100); one giant spike, 10 times the usual ELM, then leaves the
#: usual ELMs under the threshold (199563, 193273), so the scale is a high
#: percentile, which a handful of spikes cannot move.
SCALE_PERCENTILE = 99.5
#: ... and never under this many times the flat-top's median rise, in threshold
#: terms: a record whose ELMs fill under 0.5% of its samples has the percentile
#: in the noise, where every wiggle would be an onset.
NOISE_RISES = 10.0
#: Onsets closer than this (ms) are one ELM.
MERGE_MS = 1.0
#: The disruption-free mask: Ip over its 95th percentile, smoothed over
#: `IP_SMOOTH_MS`, may change by less than `IP_RATE_PER_MS` a ms; the mask is
#: then smoothed over `MASK_SMOOTH_MS`, which widens each rejected stretch.
IP_SMOOTH_MS = 3.0
IP_RATE_PER_MS = 1 / 20
MASK_SMOOTH_MS = 2.0


# --- the strike-point chord choice -------------------------------------------


class StrikeChoice(NamedTuple):
    """The chords chosen, with the averages they were chosen from."""

    names: tuple[str, ...]
    r_out: float
    r_in: float
    z_avg: float


def _mean(values) -> float:
    """The mean of the finite values, nan (without a warning) when there are none."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def select_chords(ip, r_out, r_in, z_out, available=tuple(CHORDS)) -> StrikeChoice:
    """The chords that see the outer divertor leg, from the strike points.

    `ip`, `r_out`, `r_in` and `z_out` (the outer strike point's major radius,
    the inner's, and the outer's height, m) are on one clock. EFIT writes 0.0
    where it found no strike point, so a sample with a zero radius is missing: the
    strike points are averaged where `|ip|` is at least `IP_FRACTION` of its 95th
    percentile and both radii were found. Under
    `DOUBLE_NULL_Z` the averages are repeated over the side the outer strike
    point is mostly on. An upper strike point (mean Z above 0) takes the chords
    above `MIDPLANE_Z`, a lower one those below `-MIDPLANE_Z`; of those a chord
    stays when it is outside the outer strike point or inside the inner one,
    which keeps it off the private-flux region. Only chords in `available`
    count. Raises `ValueError` when no strike point can be averaged; the names
    are empty when no chord stays.
    """
    ip, r_out, r_in, z_out = (
        np.asarray(a, dtype=np.float64) for a in (ip, r_out, r_in, z_out)
    )
    with np.errstate(invalid="ignore"):
        peak = np.nanpercentile(np.abs(ip), IP_PERCENTILE) if np.isfinite(ip).any() else np.nan
        z_out = np.where(r_out > 0, z_out, np.nan)
        use = (np.abs(ip) >= IP_FRACTION * peak) & (r_out > 0) & (r_in > 0)
        if not use.any():
            raise ValueError("no strike point found while Ip is high")
        r_avg_out, r_avg_in, z_avg = _mean(r_out[use]), _mean(r_in[use]), _mean(z_out[use])
        if abs(z_avg) < DOUBLE_NULL_Z:
            side = z_out >= 0 if z_avg >= 0 else z_out < 0
            use &= side
            if not use.any():
                raise ValueError("no strike point on the side of a double null")
            r_avg_out, r_avg_in, z_avg = _mean(r_out[use]), _mean(r_in[use]), _mean(z_out[use])
    if not np.isfinite([r_avg_out, r_avg_in, z_avg]).all():
        raise ValueError("the strike points hold no finite value")
    if r_avg_in <= CENTRE_POST_R:
        r_avg_in = CENTRE_POST_CLEARANCE
    names = []
    for name in available:
        r, z = CHORDS[name]
        seen = z > MIDPLANE_Z if z_avg > 0 else z < -MIDPLANE_Z
        if seen and (r_avg_out < r or r_avg_in > r):
            names.append(name)
    return StrikeChoice(tuple(names), r_avg_out, r_avg_in, z_avg)


class Dalpha(NamedTuple):
    """The D-alpha trace the ELM rows read, and how it was made."""

    x: np.ndarray
    y: np.ndarray  # (T,)
    title: str
    legend: str
    caveat: str


def _live(row) -> bool:
    row = np.asarray(row)
    return bool(np.isfinite(row).any() and (np.nan_to_num(row) != 0).any())


def _sum_chords(fs, rows) -> np.ndarray:
    """`(T,)`: the `rows` of `fs`, summed, a sample with no finite chord nan."""
    stack = np.asarray(fs.y, dtype=np.float64)[list(rows)]
    count = np.isfinite(stack).sum(axis=0)
    total = np.where(count > 0, np.nansum(stack, axis=0), np.nan)
    return total.astype(np.float32)


def strike_dalpha(shot, *, paths=None) -> Dalpha:
    """D-alpha summed over the chords `select_chords` picks, over the whole record.

    Reads the strike points (`strike_points`: RVSOUT, RVSIN, ZVSOUT) and Ip, the
    corpus' FS01-FS08 and, only when the strike point is in the upper divertor,
    the upper chords (`filterscopes_up`). Chords with no signal (all gaps or
    all zero) are dropped. Raises `NoDataError` or `ValueError` where one of
    those cannot be read or no chord stays.
    """
    paths = Paths.from_env() if paths is None else paths
    strike = raw_signal(int(shot), "strike_points", paths=paths)
    ip = raw_signal(int(shot), "ip", paths=paths)
    ip_at_strikes = np.interp(strike.x, ip.x, ip.y[0], np.nan, np.nan)
    choice = select_chords(ip_at_strikes, strike.y[0], strike.y[1], strike.y[2])
    if not choice.names:
        raise ValueError(
            f"no chord outside the strike points (R {choice.r_in:.3f}-"
            f"{choice.r_out:.3f} m, Z {choice.z_avg:+.2f} m)"
        )
    if choice.z_avg > 0:
        fs, order = raw_signal(int(shot), "filterscopes_up", paths=paths), UPPER_FETCHED
    else:
        fs = raw_signal(int(shot), "filterscopes", channels=list(range(8)), paths=paths)
        order = LOWER_CORPUS
    rows = [i for i, name in enumerate(order) if name in choice.names and _live(fs.y[i])]
    if not rows:
        raise NoDataError(f"shot {int(shot)}: no live chord among {list(choice.names)}")
    names = [order[i] for i in rows]
    caveat = (
        f"{'+'.join(names)} summed: the chords outside the outer strike point "
        f"(R {choice.r_out:.3f} m) or inside the inner (R {choice.r_in:.3f} m) of the "
        f"{'upper' if choice.z_avg > 0 else 'lower'} divertor (Z {choice.z_avg:+.2f} m), "
        f"from EFIT strike points at Ip above {IP_FRACTION:.0%} of its p95. "
        "After David Eldon's OMFIT ELM filterscope picker"
    )
    return Dalpha(fs.x, _sum_chords(fs, rows), DALPHA_STRIKE_TITLE, "strike sum", caveat)


def spans_dalpha(shot, *, paths=None, why="") -> Dalpha:
    """The filterscope the ELM spans read, chosen over the whole record."""
    fs = raw_signal(int(shot), "filterscopes", channels=list(range(8)), paths=paths)
    channel = spans.dalpha_channel(fs.y, int(shot))
    name = f"FS{channel + 1:02d}"
    caveat = (
        f"The strike points could not be used ({why}), so this is the one "
        "filterscope the ELM spans read" if why
        else "The one filterscope the ELM spans read"
    )
    return Dalpha(fs.x, np.asarray(fs.y[channel], dtype=np.float32),
                  f"D-alpha {name}", name, caveat)


def dalpha(shot, *, paths=None) -> Dalpha:
    """`strike_dalpha`, or the ELM spans' filterscope where the strike points,
    Ip or the chords cannot be read."""
    try:
        return strike_dalpha(shot, paths=paths)
    except (*MISSING, ValueError) as error:
        if isinstance(error, FetchDisabledError):
            log.warning("shot %s: no strike-point D-alpha: %s", shot, error)
        return spans_dalpha(shot, paths=paths, why=str(error)[:160])


# --- the jump-and-hold detector ----------------------------------------------


def _step_ms(x_ms) -> float:
    """The sample step from the span, not a median step: a float32 clock quantises."""
    x_ms = np.asarray(x_ms, dtype=np.float64)
    return float((x_ms[-1] - x_ms[0]) / (len(x_ms) - 1))


def lowpass(x_ms, y, timescale_ms: float) -> np.ndarray:
    """`y` through a first-order Butterworth low-pass, forward and back (no lag).

    The cutoff is `1 / timescale_ms` kHz, against the Nyquist frequency
    (clamped below it). Gaps stay gaps; only the finite samples are filtered.
    """
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(y)
    out = np.full(y.shape, np.nan)
    if ok.sum() < 8:
        return np.where(ok, y, np.nan)
    wn = min(0.9, (1.0 / timescale_ms) / (0.5 / _step_ms(x_ms)))
    b, a = sp_signal.butter(1, wn)
    out[ok] = sp_signal.filtfilt(b, a, y[ok])
    return out


def disruption_free(x_ms, ip_at_x) -> np.ndarray:
    """Where Ip is not changing fast: a bool mask over `x_ms`.

    `ip_at_x` is Ip on the clock `x_ms`, signed like the discharge. It is
    normalised by its 95th percentile and smoothed, and its rate must stay
    under `IP_RATE_PER_MS`; the rate's mask is then smoothed and floored, so a
    rejected stretch grows. A disruption's D-alpha flash is what this keeps
    from setting the jump's scale.
    """
    ip = np.asarray(ip_at_x, dtype=np.float64)
    if not np.isfinite(ip).any():
        return np.ones(len(ip), dtype=bool)
    ip = ip * np.sign(_mean(ip) or 1.0)
    peak = np.nanpercentile(ip, IP_PERCENTILE)
    if not peak > 0:
        return np.ones(len(ip), dtype=bool)
    smooth = lowpass(x_ms, ip / peak, IP_SMOOTH_MS)
    rate = np.gradient(np.nan_to_num(smooth, nan=0.0), np.asarray(x_ms, dtype=np.float64))
    steady = (np.abs(rate) < IP_RATE_PER_MS).astype(np.float64)
    mask = np.floor(lowpass(x_ms, steady, MASK_SMOOTH_MS) + 1e-12) >= 1
    return mask if mask.any() else np.ones(len(ip), dtype=bool)


def flat_top(x_ms, ip_at_x) -> np.ndarray:
    """Where the jump's scale is read and an onset may stand: a bool mask over `x_ms`.

    `disruption_free` (Ip is not changing fast) and Ip at least `IP_FRACTION` of
    its 95th percentile, the rule the chord choice uses for the strike points. So
    the ramps, the quench and a disruption's D-alpha flash are out; a sample with
    no Ip is out. All True where Ip has no finite sample, or where nothing is left.
    """
    ip = np.asarray(ip_at_x, dtype=np.float64)
    if not np.isfinite(ip).any():
        return np.ones(len(ip), dtype=bool)
    keep = disruption_free(x_ms, ip)
    level = np.abs(ip)
    peak = np.nanpercentile(level, IP_PERCENTILE)
    if peak > 0:
        with np.errstate(invalid="ignore"):
            keep &= level >= IP_FRACTION * peak
    return keep if keep.any() else np.ones(len(ip), dtype=bool)


def jump_and_hold(x_ms, y, ip_at_x=None, keep=None) -> np.ndarray:
    """`(T,)`: how far `y` rises over the next `JUMP_WINDOW_MS`, normalised.

    `y` is smoothed over `PRESMOOTH_MS`; at each sample the rise is the largest
    value in the next window minus its own, shifted later by `ROLLBACK` of the
    window's share that the threshold leaves, so the rise sits on the ELM.
    It is divided by its `SCALE_PERCENTILE`th percentile where the discharge is
    at its flat-top (`flat_top` of `ip_at_x`, or `keep`, a mask already made;
    everywhere without either), floored at `NOISE_RISES` thresholds' worth of its
    median there, so `JUMP_THRESHOLD` is a fraction of a big ELM's rise that a
    single spike cannot move and noise cannot reach. A gap is nan.
    """
    x_ms = np.asarray(x_ms, dtype=np.float64)
    smooth = lowpass(x_ms, y, PRESMOOTH_MS)
    dt = _step_ms(x_ms)
    n_win = max(1, int(np.ceil(JUMP_WINDOW_MS / dt)))
    shift = round(JUMP_WINDOW_MS * (1 - JUMP_THRESHOLD) * ROLLBACK / dt)
    padded = np.concatenate([np.where(np.isfinite(smooth), smooth, -np.inf),
                             np.full(n_win - 1, -np.inf)])
    ahead = np.lib.stride_tricks.sliding_window_view(padded, n_win).max(axis=1)
    rise = np.where(np.isfinite(smooth) & np.isfinite(ahead), ahead - smooth, np.nan)
    if shift:
        rise = np.concatenate([np.zeros(shift), rise[:-shift]])
    if keep is None:
        keep = np.ones(len(rise), dtype=bool) if ip_at_x is None else flat_top(x_ms, ip_at_x)
    inside = rise[keep]
    scale = np.nan
    if np.isfinite(inside).any():
        scale = max(
            np.nanpercentile(inside, SCALE_PERCENTILE),
            NOISE_RISES * np.nanmedian(inside) / JUMP_THRESHOLD,
        )
    if not scale > 0:
        raise NoDataError("D-alpha has no rise to normalise the jump by")
    return (rise / scale).astype(np.float32)


def onsets(x_ms, jump, threshold=JUMP_THRESHOLD, merge_ms=MERGE_MS, keep=None) -> np.ndarray:
    """The times (ms) the normalised `jump` rises through `threshold`.

    A start inside the first sample's flagged stretch is not one (the record
    is taken to begin between ELMs); onsets within `merge_ms` of the one before
    are one ELM. With `keep` (a mask over `x_ms`, `flat_top`), a rise that starts
    outside it is not an ELM onset either.
    """
    x_ms = np.asarray(x_ms, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        flag = np.asarray(jump) > threshold
    if flag.size and flag[0]:
        flag[: flag.size if flag.all() else int(np.argmin(flag))] = False
    rising = np.flatnonzero(flag[1:] & ~flag[:-1]) + 1
    if keep is not None:
        rising = rising[np.asarray(keep, dtype=bool)[rising]]
    kept: list[float] = []
    for t in x_ms[rising]:
        if not kept or t - kept[-1] >= merge_ms:
            kept.append(float(t))
    return np.array(kept)


def _ip_on(x_ms, shot, paths):
    """Ip on the clock `x_ms`, or None where the shot has none."""
    try:
        ip = raw_signal(int(shot), "ip", paths=paths)
    except MISSING:
        return None
    return np.interp(x_ms, ip.x, ip.y[0], np.nan, np.nan)


def markers(shot, *, t_range=None, paths=None) -> list[dict]:
    """The page's "ELM onset" markers: `[{"label": "ELM onset", "t_ms": [...]}]`.

    The jump-and-hold onsets over the whole record (the scale is the flat-top's),
    those inside the flat-top (`flat_top`) and inside `t_range` when it is given;
    `[]` where D-alpha cannot be read or holds no ELM.
    """
    paths = Paths.from_env() if paths is None else paths
    try:
        d = dalpha(shot, paths=paths)
        ip = _ip_on(d.x, shot, paths)
        keep = None if ip is None else flat_top(d.x, ip)
        jump = jump_and_hold(d.x, d.y, keep=keep)
    except (*MISSING, ValueError):
        return []
    times = onsets(d.x, jump, keep=keep)
    if t_range is not None:
        times = times[(times >= t_range[0]) & (times <= t_range[1])]
    return [{"label": "ELM onset", "t_ms": [float(t) for t in times]}] if len(times) else []


# --- the rows ------------------------------------------------------------------


def co2_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    co2 = raw_signal(int(shot), "co2", channels=[0], t_range=t_range, paths=paths)
    return [
        power_panel(
            POWER_TITLE,
            co2.x,
            co2.y[0],
            rate_hz=RATE_HZ,
            nperseg=NPERSEG,
            hop=HOP,
            max_khz=MAX_KHZ,
            plasma=plasma_window(shot, paths),
        )
    ]


def density_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The CO2 R0 chord averaged over `DENSITY_BIN_MS`."""
    co2 = raw_signal(int(shot), "co2", channels=[0], t_range=t_range, paths=paths)
    x, y = bin_mean(co2.x, co2.y, DENSITY_BIN_MS)
    return [with_caveat(Panel(title=DENSITY_TITLE, x=x, y=y, ylabel="n_e"),
                        f"{DENSITY_BIN_MS:g} ms mean")]


def wmhd_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    paths = Paths.from_env() if paths is None else paths
    panel = equilibrium.line_panel(
        shot, "wmhd", WMHD_TITLE, paths=paths, t_range=t_range, ylabel="J"
    )
    return [with_caveat(panel[0], "EFIT01 stored energy")]


def flat(x_ms, y, window) -> bool:
    """Whether a trace's 0.5-99.5 percentile range over `window` is under `FLAT_V`."""
    x_ms = np.asarray(x_ms, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))[0]
    inside = np.isfinite(y)
    if window is not None and (x_ms >= window[0]).any():
        inside &= (x_ms >= window[0]) & (x_ms <= window[1])
    if not inside.any():
        return True
    lo, hi = np.percentile(y[inside], ROBUST_PERCENTILES)
    return bool(hi - lo < FLAT_V)


def _trace(title, x, y, window, *, legend=None, notes=()) -> Panel:
    y, clipped = robust_clip(x, y, window)
    return with_caveat(
        Panel(title=title, x=x, y=y, ylabel="D-α", legend=legend),
        *notes,
        CLIPPED if clipped else "",
    )


def pcphd03_panel(shot, *, t_range=None, paths=None, window=None) -> list[Panel]:
    """PCPHD03, or none when it is flat; `NoDataError` when it cannot be read."""
    trace = raw_signal(int(shot), "pcphd03", paths=paths)
    if flat(trace.x, trace.y, window):
        return []
    x, y = np.asarray(trace.x), np.atleast_2d(trace.y)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    return [_trace(PCPHD03_TITLE, x, y, window)]


def jump_panel(d: Dalpha, ip, *, t_range=None) -> Panel:
    """The jump-and-hold row of `d`, over the whole record and cropped to the view."""
    jump = jump_and_hold(d.x, d.y, ip)
    x = np.asarray(d.x)
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, jump = x[keep], jump[keep]
    return with_caveat(
        Panel(title=JUMP_TITLE, x=x, y=jump[None], ylabel=f"rise / p{SCALE_PERCENTILE:g}",
              hlines=[JUMP_THRESHOLD]),
        f"D-alpha's rise over the next {JUMP_WINDOW_MS:g} ms after a {PRESMOOTH_MS:g} ms "
        f"smooth, over its {SCALE_PERCENTILE:g}th-percentile rise in the Ip flat-top (Ip "
        f"at least {IP_FRACTION:.0%} of its p95 and not changing fast, so a disruption's "
        f"flash does not set the scale), floored at {NOISE_RISES:g} thresholds' worth of "
        f"its median there so noise cannot reach the line. The line at {JUMP_THRESHOLD:g} is the "
        "ELM-onset threshold; its rising crossings inside the flat-top are the markers. "
        "After David Eldon's jump-and-hold ELM detector",
        "" if ip is not None else "Ip not found: the scale is the whole record's",
    )


def dalpha_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The D-alpha row (the strike-point sum and, after it, the spans' single
    chord), the jump-and-hold row and PCPHD03 (where it is not flat)."""
    window = plasma_window(shot, paths)
    kwargs = {"t_range": t_range, "paths": paths, "window": window}
    try:
        pcphd03 = pcphd03_panel(shot, **kwargs)
        note = "" if pcphd03 else "PCPHD03 is flat over the plasma, so it is left out"
    except MISSING as error:
        if isinstance(error, FetchDisabledError):
            log.warning("shot %s: no PCPHD03 panel: %s", shot, error)
        pcphd03, note = [], "PCPHD03 not found, so it is left out"

    def row(d, x, y, *notes):
        if t_range is not None:
            keep = (x >= t_range[0]) & (x <= t_range[1])
            x, y = x[keep], y[:, keep]
        return _trace(d.title, x, y, window, legend=[d.legend], notes=notes)

    def build():
        d = dalpha(shot, paths=paths)
        x, y = np.asarray(d.x), np.atleast_2d(d.y)
        # The row stands without the jump: a record with no rise to scale by
        # (a dead or constant channel) still draws the trace.
        jump = optional(
            "jump-and-hold",
            shot,
            lambda: [jump_panel(d, _ip_on(x, shot, paths), t_range=t_range)],
        )
        rows = [row(d, x, y, d.caveat, note)]
        if d.title == DALPHA_STRIKE_TITLE:
            # The frames model and the paper read the single chord (`D-alpha FSnn`),
            # never the sum: it stays, right after the sum.
            def spans_row():
                c = spans_dalpha(shot, paths=paths)
                return [row(c, np.asarray(c.x), np.atleast_2d(c.y), c.caveat)]

            rows += optional("D-alpha spans filterscope", shot, spans_row)
        return [*rows, *jump]

    return optional("D-alpha filterscope", shot, build) + pcphd03


def energy_panels(shot, *, t_range=None, paths=None):
    """Calibrated native loop energy and provisional D-alpha-matched loss sizes."""
    paths = Paths.from_env() if paths is None else paths

    def build():
        analysis = elm_energy.load(int(shot), paths)
        x = analysis.time_ms
        start, end = analysis.calibration.calibration_window_ms
        measured = (x >= start) & (x <= end) & np.isfinite(analysis.energy_j)
        energy = np.where(measured, analysis.energy_j, np.nan)
        loss = np.where(measured, 0.0, np.nan)
        fraction = loss.copy()
        for event in analysis.losses:
            index = np.searchsorted(x, event.loop_time_ms)
            loss[index], fraction[index] = event.loss_j / 1000, event.fraction * 100
        specs = (
            (LOOP_TITLES[0], energy, "J",
             "Diamagnetic loop energy, drift-corrected and scaled to EFIT's"),
            (LOOP_TITLES[1], loss, "kJ", "Loop drops matched to D-alpha; provisional"),
            (LOOP_TITLES[2], fraction, "%",
             "ELM energy loss over the pre-ELM energy; provisional"),
        )
        built = []
        for title, values, units, note in specs:
            time, y = equilibrium.plot_arrays(FeatureArray(x / 1000, values[None]))
            if t_range is not None:
                keep = (time >= t_range[0]) & (time <= t_range[1])
                time, y = time[keep], y[:, keep]
            built.append(with_caveat(
                Panel(title=title, x=time, y=y, ylabel=units,
                      metadata=analysis.metadata() if not built else None),
                note,
            ))
        return built

    return optional("diamagnetic energy", shot, build)


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return (
        dalpha_panels(shot, **kwargs)
        + optional("CO2 density", shot, lambda: density_panel(shot, **kwargs))
        + optional("W_MHD", shot, lambda: wmhd_panel(shot, **kwargs))
        + energy_panels(shot, **kwargs)
        + optional("CO2", shot, lambda: co2_panel(shot, **kwargs))
    )
