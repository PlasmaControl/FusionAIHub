"""H-mode: D-alpha, the line-averaged density, the beams and beta_N.

At the L-H transition the divertor D-alpha falls within a few ms while the
density rises (v1 spec §6.2); the beams and beta_N say whether the shot was
heated enough to get there and what the confinement did. D-alpha is the
corpus's filterscopes FS01-FS08, what `dalpha_lh` reads; the density is the
CO2 R0 chord averaged over 1 ms; the beams are the corpus's eight NBI powers
summed; beta_N is the features store's.
"""

from __future__ import annotations

import numpy as np

from ..raw import raw_signal
from ..verify import NoDataError, Panel
from ._shared import betan_panel, bin_mean, optional

DALPHA_CHANNELS = tuple(range(8))
DENSITY_BIN_MS = 1.0


def dalpha_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    fs = raw_signal(
        int(shot),
        "filterscopes",
        channels=list(DALPHA_CHANNELS),
        t_range=t_range,
        paths=paths,
    )
    lit = [i for i, row in enumerate(fs.y) if np.isfinite(row).any()]
    if not lit:
        raise NoDataError(f"shot {int(shot)}: no finite D-alpha filterscope")
    return [
        Panel(
            title="D-alpha filterscopes",
            x=fs.x,
            y=fs.y[lit],
            ylabel="D-α",
            legend=[f"FS{c + 1:02d}" for c in lit],
        )
    ]


def density_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    co2 = raw_signal(int(shot), "co2", channels=[0], t_range=t_range, paths=paths)
    x, y = bin_mean(co2.x, co2.y, DENSITY_BIN_MS)
    return [Panel(title="density, CO2 R0 (1 ms mean)", x=x, y=y, ylabel="n_e")]


def beams_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    pinj = raw_signal(int(shot), "pinj", t_range=t_range, paths=paths)
    total = np.nansum(np.asarray(pinj.y, dtype=np.float64), axis=0) * 1e-6
    return [Panel(title="NBI power", x=pinj.x, y=total[None, :], ylabel="MW")]


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    built = []
    for what, build in (
        ("D-alpha", dalpha_panel),
        ("density", density_panel),
        ("NBI", beams_panel),
        ("beta_N", betan_panel),
    ):
        built += optional(what, shot, lambda build=build: build(shot, **kwargs))
    return built
