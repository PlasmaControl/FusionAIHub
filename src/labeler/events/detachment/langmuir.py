"""Ion saturation current from the swept divertor Langmuir probes in the corpus.

The corpus `langmuir` group holds the raw PTDATA records `TPLANG01..TPLANG72` at
524 kHz (real on about half of the shots of 2018-2019, a one-sample stub on the
rest). They are not processed Jsat: the processed `\\WALL::JSAT*` nodes hold no data
on any shot tested, and no probe position or calibration is on disk.

**Channel layout** (found from the data, not from documentation; the checks are in
`tests/labeler/test_detachment_indicators.py`): the channels come in adjacent pairs
`(I, V)` = `(c, c + 1)` for odd `c`: the odd channel is the probe current, the even
one the bias voltage, swept at about 1.05 kHz and common to many probes. A
characteristic plotted from such a pair shows the ion plateau at the negative end,
the floating point where the current crosses its offset, and the electron branch at
the positive end; a pair the other way round (V, I) gives nonsense. A probe whose
current does not respond to its sweep (an open or dead channel) is dropped.

**Jsat per sweep** is the median current over the most negative `ION_FRACTION` of
the sweep's voltage range, minus the offset the current channel reads with no
plasma (`OFFSET_WINDOW_S`, the quiet 13 ms before t = 0; the 25 ms before it is the
digitiser's start-up transient). The median over the plateau samples rejects the
turn-around spikes. The unit is the digitiser volt: the ion current is NOT
calibrated, so the indicator built on it (`afrac.py`) is a ratio to the same probe's
own attached level and never uses the absolute value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")

#: The bias sweep of the DIII-D divertor probe array (measured: 1050.08 Hz).
SWEEP_HZ = 1050.0
#: The lowest fraction of a sweep's voltage range that counts as the ion branch.
ION_FRACTION = 0.2
#: Offset window (seconds): after the digitiser start-up and before the plasma.
OFFSET_WINDOW_S = (-0.015, -0.002)
#: A sweep is a sweep when its voltage swing exceeds this (volts at the digitiser).
MIN_SWEEP_VPP = 1.0
#: A probe is alive when, on its 90th-percentile sweep, the current differs
#: between the ion and electron ends by at least this (volts); a dead channel
#: differs by about 0.003. (Not the median: the record includes the dark seconds
#: before and after the discharge.)
MIN_RESPONSE = 0.03


def sweep_jsat(
    current: np.ndarray, voltage: np.ndarray, samples_per_sweep: int, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """`(jsat, response)` per sweep for one probe.

    `current` and `voltage` start at the same sample and are cut into consecutive
    blocks of one sweep period; a block is complete or dropped. `jsat` is the
    absolute ion-branch current minus the offset; `response` the absolute difference
    between the median current at the two ends of the sweep, the liveness test.
    """
    n = samples_per_sweep
    count = min(len(current), len(voltage)) // n
    i_blocks = np.asarray(current[: count * n], dtype=np.float64).reshape(count, n)
    v_blocks = np.asarray(voltage[: count * n], dtype=np.float64).reshape(count, n)
    vmin = v_blocks.min(axis=1, keepdims=True)
    vmax = v_blocks.max(axis=1, keepdims=True)
    swing = vmax - vmin
    low = v_blocks <= vmin + ION_FRACTION * swing
    high = v_blocks >= vmax - ION_FRACTION * swing
    ion = np.nanmedian(np.where(low, i_blocks, np.nan), axis=1)
    electron = np.nanmedian(np.where(high, i_blocks, np.nan), axis=1)
    jsat = np.abs(ion - offset)
    response = np.abs(electron - ion)
    flat = swing[:, 0] < MIN_SWEEP_VPP  # no sweep running: nothing to say
    jsat[flat] = np.nan
    response[flat] = np.nan
    return jsat, response


def probe_pairs(n_channels: int) -> list[tuple[int, int]]:
    """0-based `(current, voltage)` row pairs: channels `(c, c + 1)`, `c` odd."""
    return [(c - 1, c) for c in range(1, n_channels, 2)]


def read_shot(shot: int) -> dict | None:
    """Per-probe Jsat on the sweep clock, or None where the corpus has no probes.

    Returns `{"t_ms": (S,), "jsat": (P, S), "probe": (P,) 1-based current channel,
    "response": (P,) 90th-percentile response}` for the live probes only.
    """
    import h5py

    path = CORPUS / f"{int(shot)}_processed.h5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as handle:
        if "langmuir" not in handle:
            return None
        group = handle["langmuir"]
        if group["ydata"].shape[-1] <= 1:
            return None
        x = np.asarray(group["xdata"][:], dtype=np.float64)
        fs = 1.0 / float(np.median(np.diff(x)))
        n = round(fs / SWEEP_HZ)
        start = int(np.searchsorted(x, 0.0))
        quiet = slice(
            int(np.searchsorted(x, OFFSET_WINDOW_S[0])),
            int(np.searchsorted(x, OFFSET_WINDOW_S[1])),
        )
        ydata = group["ydata"]
        rows, probes, medians = [], [], []
        for i_row, v_row in probe_pairs(ydata.shape[0]):
            voltage = np.asarray(ydata[v_row, start:], dtype=np.float64)
            if not np.isfinite(voltage[:2000]).all():
                continue  # a channel the digitiser never filled
            if np.ptp(voltage[:20000]) < MIN_SWEEP_VPP:
                continue  # no bias sweep on this pair
            current = np.asarray(ydata[i_row, start:], dtype=np.float64)
            offset = float(np.nanmedian(ydata[i_row, quiet]))
            jsat, response = sweep_jsat(current, voltage, n, offset)
            finite = response[np.isfinite(response)]
            alive = float(np.percentile(finite, 90)) if finite.size else 0.0
            if not alive >= MIN_RESPONSE:
                continue
            rows.append(jsat.astype(np.float32))
            probes.append(i_row + 1)
            medians.append(alive)
    if not rows:
        return None
    count = rows[0].size
    t_ms = (x[start] + (np.arange(count) + 0.5) * n / fs) * 1000.0
    return {
        "t_ms": t_ms,
        "jsat": np.vstack(rows),
        "probe": np.asarray(probes),
        "response": np.asarray(medians),
    }
