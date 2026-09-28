"""Tearing modes: the magnetics below 30 kHz, the toroidal mode number, beta_N.

A tearing mode is a coherent line below ~30 kHz on the Mirnov probes, with its
toroidal mode number n from the probe array (v1 spec §6.4). The first
spectrogram is MPI66M322D's, in dB above each frequency's floor over the plasma:
the shot's v1 rule-4 Ip window, not the whole record, which runs seconds past it.
The second is pyspecview's picture of the six MPI66M midplane probes: every
time-frequency cell takes the n from -4 to 5 whose phases fit the probes' best,
and is drawn in that n's colour, as bright as the probes' mean power over the
same kind of floor. A line is then one colour, and each line gets its own n.

n > 0 is a mode travelling counter-clockwise seen from above, the co-current
direction of DIII-D's normal plasma current, so a rotating 2/1 shows n = 1 as
the catalog counts it. DIII-D's toroidal angle runs clockwise, and pyspecview,
which fits that angle as it stands, shows the same mode as n = -1.
beta_N is the features store's. EFIT q and the ECE cross-phase the sheet also
names come with v1 Part 7.
"""

from __future__ import annotations

import numpy as np

from ..raw import raw_signal
from ..verify import Panel
from ._shared import (
    above_floor_db,
    betan_panel,
    finite,
    optional,
    plasma_columns,
    plasma_window,
    stft,
)

#: Corpus `mirnov` row -> the probe's toroidal angle in degrees, from
#: pyspecview's DIII-D probe table (loaders_DIIID/Magnetics.py,
#: `BpDot_probes_R0`): MPI66M322D sits at 317.4 degrees whatever its name.
PROBES = {15: 317.4, 16: 132.5, 18: 312.4, 20: 19.5, 21: 97.4, 22: 307.0}
#: 100 kHz, 1024-sample windows every 256: 0.1 kHz bins, 2.56 ms columns.
RATE_HZ = 100_000
NPERSEG = 1024
HOP = 256
MAX_KHZ = 30.0
#: pyspecview's DIII-D range, -5 to 4 on its clockwise angle, turned round.
N_VALUES = np.arange(-4, 6)
#: Each n's colour: 1, 2 and 3 far apart, and each n far from -n.
N_COLOURS = {
    -4: "#a2845e",
    -3: "#b28dff",
    -2: "#ff9500",
    -1: "#00e5ff",
    0: "#c8c8c8",
    1: "#ff3b30",
    2: "#34c759",
    3: "#3d8bff",
    4: "#ffcc00",
    5: "#ff2dd4",
}
#: The spectrograms' floor: a tearing mode can hold one frequency for seconds.
#: It is each bin's quantile over the plasma's columns (`plasma_columns`).
FLOOR_QUANTILE = 0.2
#: The spectrograms' scale, higher than the shared (-3, 27): a flat top's broadband
#: power runs 27-40 dB over that floor on loud shots and would hide the lines.
Z_DB = (-3.0, 42.0)


def mode_numbers(spec, phi_deg) -> np.ndarray:
    """`(F, T)` int8: each time-frequency cell's n, pyspecview's way.

    `spec` is `(P, F, T)`, the probes' complex STFTs. Each n in `N_VALUES`
    scores a cell `|mean_k exp(i (arg A_k - n phi_k))|`, 1 when the probes'
    phases turn exactly n times round the torus, and the best score wins; the
    cell's power plays no part. pyspecview (`update_image`) scores
    `exp(i (arg A_k + n phi_k))`, the same fit on the opposite sign.
    """
    unit = spec / np.maximum(np.abs(spec), 1e-30)
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    best = np.full(unit.shape[1:], -1.0, dtype=np.float32)
    index = np.zeros(unit.shape[1:], dtype=np.int8)
    for i, n in enumerate(N_VALUES):  # one n at a time: a whole shot's cells are ~2M
        turns = np.exp(-1j * n * phi).astype(np.complex64)
        fit = np.abs(np.tensordot(turns, unit, axes=(0, 0)))
        index[fit > best] = i
        np.maximum(best, fit, out=best)
    return N_VALUES[index].astype(np.int8)


def magnetics_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    rows = list(PROBES)
    mirnov = raw_signal(
        int(shot), "mirnov", channels=rows, t_range=t_range, paths=paths
    )
    values = np.stack([finite(y) for y in mirnov.y])
    t_ms, f_hz, spec = stft(mirnov.x, values, rate_hz=RATE_HZ, nperseg=NPERSEG, hop=HOP)
    keep = f_hz <= MAX_KHZ * 1000
    spec = spec[:, keep].astype(np.complex64)
    power = np.abs(spec) ** 2
    plasma = plasma_columns(t_ms, power[0].sum(axis=0), plasma_window(shot, paths))
    return [
        Panel(
            title="MPI66M322D power",
            kind="heatmap",
            x=t_ms,
            y=f_hz[keep] / 1000,
            z=above_floor_db(power[0], FLOOR_QUANTILE, columns=plasma),
            ylabel="kHz",
            zmin=Z_DB[0],
            zmax=Z_DB[1],
        ),
        Panel(
            title="toroidal n, MPI66M probes",
            kind="heatmap",
            x=t_ms,
            y=f_hz[keep] / 1000,
            z=above_floor_db(power.mean(axis=0), FLOOR_QUANTILE, columns=plasma),
            ylabel="kHz",
            zmin=Z_DB[0],
            zmax=Z_DB[1],
            modes=mode_numbers(spec, list(PROBES.values())),
            mode_colours=N_COLOURS,
        ),
    ]


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return optional("magnetics", shot, lambda: magnetics_panels(shot, **kwargs)) + (
        optional("beta_N", shot, lambda: betan_panel(shot, **kwargs))
    )
