"""H-mode: D-alpha, the line-averaged density, the beams, the edge Te and beta_N.

At the L-H transition the divertor D-alpha falls within a few ms while the
density rises (v1 spec §6.2); the beams and beta_N say whether the shot was
heated enough to get there and what the confinement did. D-alpha is the
corpus's filterscopes FS01-FS08, what `dalpha_lh` reads, each clipped to its
robust range over the plasma window (`_shared.robust_clip`) so a spike at the
end of the discharge does not set the row's scale; the density is the CO2 R0
chord averaged over 1 ms; the beams are the corpus's eight NBI powers summed;
beta_N is the features store's.

The edge Te row is the ECE's outermost channels inside the plasma, averaged
over 1 ms (`edge_te_panel`): an I-mode shows an edge Te pedestal with no
density pedestal, which the density row beside it settles. Titles are short;
what qualifies a row (a clip, a mean, which channels) is its caveat.
"""

from __future__ import annotations

import numpy as np

from ...config import Paths
from ..raw import ECE_EDGE_ROWS, raw_signal, record_tier
from ..verify import NoDataError, Panel
from . import ece_geometry
from ._shared import (
    CLIPPED,
    MISSING,
    betan_panel,
    bin_mean,
    optional,
    plasma_window,
    robust_clip,
    with_caveat,
)

DALPHA_CHANNELS = tuple(range(8))
DENSITY_BIN_MS = 1.0
EDGE_TE_BIN_MS = 1.0
#: The ECE radiometer is the corpus group's first 40 rows; R falls with the row.
RADIOMETER_CHANNELS = 40
#: How many channels the edge Te row draws, and the rho a channel must reach
#: (and stay inside) to count as the edge.
EDGE_CHANNELS = 4
EDGE_RHO = (0.8, 1.0)

TITLES = ("D-alpha filterscopes", "CO2 R0 density", "NBI power", "edge Te (ECE)", "beta_N")


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
    y, clipped = robust_clip(fs.x, fs.y[lit], plasma_window(shot, paths))
    return [
        with_caveat(
            Panel(
                title=TITLES[0],
                x=fs.x,
                y=y,
                ylabel="D-α",
                legend=[f"FS{c + 1:02d}" for c in lit],
            ),
            CLIPPED if clipped else "",
        )
    ]


def density_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    co2 = raw_signal(int(shot), "co2", channels=[0], t_range=t_range, paths=paths)
    x, y = bin_mean(co2.x, co2.y, DENSITY_BIN_MS)
    return [
        with_caveat(
            Panel(title=TITLES[1], x=x, y=y, ylabel="n_e"),
            f"CO2 R0 chord, {DENSITY_BIN_MS:g} ms mean",
        )
    ]


def beams_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    pinj = raw_signal(int(shot), "pinj", t_range=t_range, paths=paths)
    total = np.nansum(np.asarray(pinj.y, dtype=np.float64), axis=0) * 1e-6
    return [Panel(title="NBI power", x=pinj.x, y=total[None, :], ylabel="MW")]


def edge_channels(shot, rows, paths) -> tuple[list[int], str]:
    """`(rows, how)`: which of the ECE `rows` (0..`rows`-1) are the edge, and why.

    From the cached ECE geometry (`ece_geometry`: each channel's normalised
    flux), the `EDGE_CHANNELS` channels with the largest median rho inside
    `EDGE_RHO`; without it, the channels of largest nominal R, which are the
    first rows, and the caveat says they may lie outside the plasma.
    """
    try:
        psi = ece_geometry.load_geometry(int(shot), paths).psi
    except MISSING:
        psi = None
    if psi is not None:
        n = min(rows, len(psi))
        rho = np.full(n, np.nan)
        for i in range(n):
            finite = psi[i][np.isfinite(psi[i]) & (psi[i] >= 0)]
            if finite.size:
                rho[i] = np.sqrt(np.median(finite))
        inside = np.flatnonzero((rho >= EDGE_RHO[0]) & (rho < EDGE_RHO[1]))
        if inside.size:
            chosen = inside[np.argsort(rho[inside])[::-1][:EDGE_CHANNELS]]
            return sorted(int(c) for c in chosen), (
                f"The {len(chosen)} outermost ECE channels inside the plasma "
                f"(median rho {EDGE_RHO[0]:g}-{EDGE_RHO[1]:g} from the ECE geometry)"
            )
    return list(range(min(EDGE_CHANNELS, rows))), (
        f"No ECE geometry, so the {EDGE_CHANNELS} channels of largest nominal R "
        "(the lowest frequencies), which may lie outside the plasma"
    )


def edge_te_panel(shot, *, t_range=None, paths=None) -> list[Panel]:
    """The edge ECE channels' Te, averaged over `EDGE_TE_BIN_MS`.

    The corpus' `ece` (or the raw cache's) when it holds the shot, else the
    fetched `ece_edge` group (the eight largest-R channels, `raw.FETCH_SPECS`).
    """
    paths = Paths.from_env() if paths is None else paths
    if record_tier(int(shot), "ece", paths=paths):
        group, rows = "ece", RADIOMETER_CHANNELS
    else:
        group, rows = "ece_edge", len(ECE_EDGE_ROWS)
    chosen, how = edge_channels(shot, rows, paths)
    te = raw_signal(int(shot), group, channels=chosen, t_range=t_range, paths=paths)
    if not np.isfinite(te.y).any():
        raise NoDataError(f"shot {int(shot)}: the edge ECE channels hold no finite Te")
    x, y = bin_mean(te.x, te.y, EDGE_TE_BIN_MS)
    return [
        with_caveat(
            Panel(
                title=TITLES[3],
                x=x,
                y=y,
                ylabel="keV",
                legend=[f"ch {c}" for c in chosen],
            ),
            how,
            f"{EDGE_TE_BIN_MS:g} ms mean",
        )
    ]


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    built = []
    for what, build in (
        ("D-alpha", dalpha_panel),
        ("density", density_panel),
        ("NBI", beams_panel),
        ("edge Te", edge_te_panel),
        ("beta_N", betan_panel),
    ):
        built += optional(what, shot, lambda build=build: build(shot, **kwargs))
    return built
