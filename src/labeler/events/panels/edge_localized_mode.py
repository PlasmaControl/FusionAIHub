"""ELMs: the CO2 interferometer's spectrogram to 125 kHz, and D-alpha.

An ELM is a burst on the divertor D-alpha 2-5 ms wide; on the interferometer
it is a broadband stripe. The owner chose these two rows: the R0 chord's own
power from 0 to 125 kHz, and the PCPHD03 photodiode as a trace, with no
spectrogram. Where PCPHD03 cannot be read the corpus's first D-alpha
filterscope stands in, and its title says so.
"""

from __future__ import annotations

from ..raw import raw_signal
from ..verify import NoDataError, Panel
from ._shared import optional, power_panel

#: R0 resampled to 250 kHz, 256-sample windows every 64: 0.98 kHz bins and
#: 0.256 ms columns, the AE rows' resolution, up to 125 kHz.
RATE_HZ = 250_000
NPERSEG = 256
HOP = 64
MAX_KHZ = 125.0


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
        )
    ]


def dalpha_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    try:
        trace = raw_signal(int(shot), "pcphd03", t_range=t_range, paths=paths)
        title = "D-alpha PCPHD03"
    except NoDataError:
        trace = raw_signal(
            int(shot), "filterscopes", channels=[0], t_range=t_range, paths=paths
        )
        title = "D-alpha FS01 (PCPHD03 not found)"
    return [Panel(title=title, x=trace.x, y=trace.y, ylabel="D-α")]


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return optional("CO2", shot, lambda: co2_panel(shot, **kwargs)) + optional(
        "D-alpha", shot, lambda: dalpha_panel(shot, **kwargs)
    )
