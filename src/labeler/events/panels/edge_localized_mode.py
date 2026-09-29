"""ELMs: the CO2 interferometer's spectrogram to 125 kHz, and D-alpha.

An ELM is a burst on the divertor D-alpha 2-5 ms wide; on the interferometer
it is a broadband stripe. The owner chose these rows: the R0 chord's own
power from 0 to 125 kHz, its floor over the plasma window, and the PCPHD03
photodiode as a trace, with no spectrogram. Beside it is the filterscope the
ELM spans were found on (`spans.dalpha_channel`: FS01, or FS02 where FS01 is
dark), so the page always shows the signal a suggestion came from.

PCPHD03 is left out where it cannot be read, or where it is flat over the
plasma (`flat`); the filterscope row's title says which. Both traces are
clipped to their robust range over the plasma window (`_shared.robust_clip`),
so a spike at the end of the discharge does not set the row's scale.
"""

from __future__ import annotations

import logging

import numpy as np

from .. import spans
from ..raw import FetchDisabledError, raw_signal
from ..verify import Panel
from ._shared import (
    CLIPPED,
    MISSING,
    ROBUST_PERCENTILES,
    optional,
    plasma_window,
    power_panel,
    robust_clip,
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


def co2_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    co2 = raw_signal(int(shot), "co2", channels=[0], t_range=t_range, paths=paths)
    return [
        power_panel(
            "CO2 R0 power",
            co2.x,
            co2.y[0],
            rate_hz=RATE_HZ,
            nperseg=NPERSEG,
            hop=HOP,
            max_khz=MAX_KHZ,
            plasma=plasma_window(shot, paths),
        )
    ]


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


def _trace(title, x, y, window, *, legend=None, note="") -> Panel:
    y, clipped = robust_clip(x, y, window)
    return Panel(
        title=title + (CLIPPED if clipped else "") + note,
        x=x,
        y=y,
        ylabel="D-α",
        legend=legend,
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
    return [_trace("D-alpha PCPHD03", x, y, window)]


def filterscope_panel(
    shot, *, t_range=None, paths=None, window=None, note=""
) -> list[Panel]:
    """The filterscope the ELM spans read, chosen over the whole record."""
    fs = raw_signal(int(shot), "filterscopes", channels=list(range(8)), paths=paths)
    channel = spans.dalpha_channel(fs.y, int(shot))
    x, y = np.asarray(fs.x), np.asarray(fs.y)[channel : channel + 1]
    if t_range is not None:
        keep = (x >= t_range[0]) & (x <= t_range[1])
        x, y = x[keep], y[:, keep]
    name = f"FS{channel + 1:02d}"
    title = f"D-alpha {name}, the ELM spans' channel"
    return [_trace(title, x, y, window, legend=[name], note=note)]


def dalpha_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    window = plasma_window(shot, paths)
    kwargs = {"t_range": t_range, "paths": paths, "window": window}
    try:
        pcphd03 = pcphd03_panel(shot, **kwargs)
        note = "" if pcphd03 else " (PCPHD03 flat, left out)"
    except MISSING as error:
        if isinstance(error, FetchDisabledError):
            log.warning("shot %s: no PCPHD03 panel: %s", shot, error)
        pcphd03, note = [], " (PCPHD03 not found)"
    fs = optional(
        "D-alpha filterscope",
        shot,
        lambda: filterscope_panel(shot, note=note, **kwargs),
    )
    return pcphd03 + fs


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return optional("CO2", shot, lambda: co2_panel(shot, **kwargs)) + dalpha_panels(
        shot, **kwargs
    )
