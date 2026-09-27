"""Tearing modes: the magnetics below 30 kHz, the toroidal mode number, beta_N.

A tearing mode is a coherent line below ~30 kHz on the Mirnov probes, with its
toroidal mode number n from the probe array (v1 spec §6.4). The spectrogram is
MPI66M322D's. The n strip takes, in each column, the strongest line between 1
and 30 kHz on the six MPI66M midplane probes and scores each n from -4 to 4
by how well the probes' phases line up with it: 1 is a perfect fit. A mode
cos(wt - n phi) has n > 0, phi being each probe's toroidal angle in its name.
beta_N is the features store's. EFIT q and the ECE cross-phase the sheet also
names come with v1 Part 7.
"""

from __future__ import annotations

import numpy as np

from ..raw import raw_signal
from ..verify import Panel
from ._shared import Z_DB, above_floor_db, betan_panel, finite, optional, stft

#: Corpus `mirnov` row -> the probe's toroidal angle in degrees (MPI66M<phi>D).
PROBES = {15: 322.0, 16: 132.0, 18: 312.0, 20: 20.0, 21: 97.0, 22: 307.0}
#: 100 kHz, 1024-sample windows every 256: 0.1 kHz bins, 2.56 ms columns.
RATE_HZ = 100_000
NPERSEG = 1024
HOP = 256
BAND_KHZ = (1.0, 30.0)
N_VALUES = np.arange(-4, 5)
#: The spectrogram's floor: a tearing mode can hold one frequency for seconds.
FLOOR_QUANTILE = 0.2
#: A column's strongest line must stand this far above the column's median over
#: the band, or its n is not scored. On the six probes' summed power a peak of
#: white noise stays within ~5 dB of it.
MIN_CONTRAST_DB = 15.0


def mode_numbers(spec, phi_deg, f_hz) -> np.ndarray:
    """`(len(N_VALUES), T)`: how well each n fits the phases at each column's peak.

    `spec` is `(P, F, T)`, the probes' complex STFTs. A column whose peak is
    under `MIN_CONTRAST_DB` above the column's median scores 0 for every n.
    """
    band = (f_hz >= BAND_KHZ[0] * 1000) & (f_hz <= BAND_KHZ[1] * 1000)
    power = (np.abs(spec[:, band]) ** 2).sum(axis=0)
    peak = np.argmax(power, axis=0)
    cols = np.arange(power.shape[1])
    contrast = power[peak, cols] / np.maximum(np.median(power, axis=0), 1e-30)
    strong = 10 * np.log10(contrast + 1e-30) >= MIN_CONTRAST_DB
    at_peak = spec[:, band][:, peak, cols]
    unit = at_peak / np.maximum(np.abs(at_peak), 1e-30)
    phi = np.deg2rad(np.asarray(phi_deg))
    turns = np.exp(1j * np.outer(N_VALUES, phi))
    fit = np.abs(turns @ unit) / len(phi)
    return np.where(strong, fit, 0.0)


def magnetics_panels(shot, *, t_range=None, paths=None) -> list[Panel]:
    rows = list(PROBES)
    mirnov = raw_signal(
        int(shot), "mirnov", channels=rows, t_range=t_range, paths=paths
    )
    values = np.stack([finite(y) for y in mirnov.y])
    t_ms, f_hz, spec = stft(mirnov.x, values, rate_hz=RATE_HZ, nperseg=NPERSEG, hop=HOP)
    keep = f_hz <= BAND_KHZ[1] * 1000
    return [
        Panel(
            title="MPI66M322D power",
            kind="heatmap",
            x=t_ms,
            y=f_hz[keep] / 1000,
            z=above_floor_db(np.abs(spec[0, keep]) ** 2, FLOOR_QUANTILE),
            ylabel="kHz",
            zmin=Z_DB[0],
            zmax=Z_DB[1],
        ),
        Panel(
            title="toroidal mode number n (MPI66M probes)",
            kind="heatmap",
            x=t_ms,
            y=N_VALUES.astype(float),
            z=mode_numbers(spec, list(PROBES.values()), f_hz),
            ylabel="n",
            zmin=0.0,
            zmax=1.0,
        ),
    ]


def panels(shot, *, t_range=None, paths=None):
    kwargs = {"t_range": t_range, "paths": paths}
    return optional("magnetics", shot, lambda: magnetics_panels(shot, **kwargs)) + (
        optional("beta_N", shot, lambda: betan_panel(shot, **kwargs))
    )
