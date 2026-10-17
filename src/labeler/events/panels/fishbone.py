"""Fishbones, as a coherent n = 1 burst locked across two Mirnov probes."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from ..raw import raw_signal
from ..verify import Panel
from ._shared import with_caveat

#: B1 and B5, the pair the cross-phase is taken over.
PROBES = (0, 4)
NPERSEG = 4096
MAX_HZ = 40_000.0
FISHBONE_BAND = (2.0, 30.0)


#: The rows' titles; the band they mark and the probes are in the caveat.
TITLES = ("mhr B1 spectrogram", "cross-phase B1 x B5")


def _spectrograms(shot, probes, *, t_range, paths):
    """`(t_ms, freq_hz, [complex spectrogram per probe])` of the `mhr` probes."""
    mhr = raw_signal(
        int(shot), "mhr", channels=list(probes), t_range=t_range, paths=paths
    )
    # A window this thin (or a zero-span one) divides to nan below rather
    # than raising: `scipy.signal.spectrogram` then runs with `fs=nan`,
    # returns `freq=[nan]`, and `freq <= MAX_HZ` is all-False since nan
    # comparisons are always False - a 0-row heatmap that renders blank
    # instead of erroring.
    if mhr.x.shape[0] < 2 or mhr.x[-1] == mhr.x[0]:
        raise ValueError(
            "fishbone needs a time vector spanning more than one instant; "
            f"got {mhr.x.shape[0]} sample(s). An out-of-range t_range slices "
            "the signal to an empty or single-sample window, which divides "
            "to nan and draws a blank panel."
        )
    # The rate comes from the SPAN, never from a median diff: xdata is
    # float32 and its spacing quantises at t ~ 3 s.
    rate = (mhr.x.shape[0] - 1) / ((mhr.x[-1] - mhr.x[0]) / 1000.0)
    nperseg = min(NPERSEG, mhr.x.shape[0])
    kwargs = {
        "fs": rate,
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "mode": "complex",
    }
    # Each probe is transformed on its own: the cross-phase below is the phase
    # of spec_a * conj(spec_b), the quantity that locks to a coherent n = 1
    # mode. The phase of a spectrogram of a complex signal built from the two
    # real probes (a + i*b) is a different, meaningless quantity.
    specs = []
    for row in mhr.y:
        freq, times, spec = scipy_signal.spectrogram(row, **kwargs)
        specs.append(spec)
    return times * 1000.0 + mhr.x[0], freq, specs


def b1_panel(shot, *, t_range=None, paths=None, band=FISHBONE_BAND) -> list[Panel]:
    """B1's power alone, over `band` (kHz): what the 1/1 view draws."""
    t_ms, freq, (spec,) = _spectrograms(shot, PROBES[:1], t_range=t_range, paths=paths)
    keep = (freq >= band[0] * 1000) & (freq <= band[1] * 1000)
    return [
        Panel(
            title=TITLES[0],
            kind="heatmap",
            x=t_ms,
            y=freq[keep] / 1000.0,
            z=np.log10(np.abs(spec[keep]) ** 2 + 1e-30),
            ylabel="kHz",
            metadata={"caveat": f"B1 only, {band[0]:g}-{band[1]:g} kHz, log10 power."},
        )
    ]


def panels(shot, *, t_range=None, paths=None):
    t_ms, freq, (spec_a, spec_b) = _spectrograms(
        shot, PROBES, t_range=t_range, paths=paths
    )
    cross_phase = np.angle(spec_a * np.conj(spec_b))
    # B1's own power, out of the complex spectrogram already computed rather
    # than a third `mode="psd"` pass over the same two million samples.
    # scipy's one-sided psd doubles every bin but DC and Nyquist, so this is
    # the same panel shifted by a constant log10(2) - and it is log-scaled.
    power = np.abs(spec_a) ** 2
    keep = freq <= MAX_HZ
    khz = freq[keep] / 1000.0
    caveat = f"The band {FISHBONE_BAND[0]:g}-{FISHBONE_BAND[1]:g} kHz is marked."
    return [
        with_caveat(
            Panel(
                title=TITLES[0],
                kind="heatmap",
                x=t_ms,
                y=khz,
                z=np.log10(power[keep] + 1e-30),
                ylabel="kHz",
                bands=[FISHBONE_BAND],
            ),
            caveat,
        ),
        with_caveat(
            Panel(
                title=TITLES[1],
                kind="heatmap",
                x=t_ms,
                y=khz,
                z=cross_phase[keep],
                ylabel="kHz",
                bands=[FISHBONE_BAND],
                metadata={"z_units": "rad"},
            ),
            caveat,
        ),
    ]
