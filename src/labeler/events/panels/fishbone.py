"""Fishbones, as a coherent n = 1 burst locked across two Mirnov probes."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from ..raw import raw_signal
from ..verify import Panel

#: B1 and B5, the pair the cross-phase is taken over.
PROBES = (0, 4)
NPERSEG = 4096
MAX_HZ = 40_000.0
FISHBONE_BAND = (2.0, 30.0)

GUIDANCE = (
    "<b>What you are looking for:</b> bursts in 2-30 kHz (the shaded band) "
    "that chirp DOWNWARD over a few milliseconds, with a cross-phase that "
    "stays flat across the burst - that flatness is what says the two probes "
    "are seeing one coherent mode rather than two patches of turbulence."
)


def panels(shot, *, t_range=None, paths=None):
    mhr = raw_signal(
        int(shot), "mhr", channels=list(PROBES), t_range=t_range, paths=paths
    )
    # A window this thin (or a zero-span one) divides to nan below rather
    # than raising: `scipy.signal.spectrogram` then runs with `fs=nan`,
    # returns `freq=[nan]`, and `freq <= MAX_HZ` is all-False since nan
    # comparisons are always False - a 0-row heatmap that renders blank
    # instead of erroring. Mirror alfven_eigenmode.crosspower's guard.
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
    # Cross-phase needs each probe's OWN complex spectrum: phase(spec_a *
    # conj(spec_b)) is the cross-spectrum's phase, the quantity that locks
    # to a coherent n = 1 mode. The phase of a spectrogram taken of a complex
    # signal built from the two real probes (a + i*b) is a different,
    # meaningless quantity, so each probe is transformed on its own.
    freq, times, spec_a = scipy_signal.spectrogram(mhr.y[0], **kwargs)
    _, _, spec_b = scipy_signal.spectrogram(mhr.y[1], **kwargs)
    cross_phase = np.angle(spec_a * np.conj(spec_b))
    # B1's own power, out of the complex spectrogram already computed rather
    # than a third `mode="psd"` pass over the same two million samples.
    # scipy's one-sided psd doubles every bin but DC and Nyquist, so this is
    # the same panel shifted by a constant log10(2) - and it is log-scaled.
    power = np.abs(spec_a) ** 2
    keep = freq <= MAX_HZ
    t_ms = times * 1000.0 + mhr.x[0]
    khz = freq[keep] / 1000.0
    return [
        Panel(
            title=f"mhr B{PROBES[0] + 1} spectrogram",
            kind="heatmap",
            x=t_ms,
            y=khz,
            z=np.log10(power[keep] + 1e-30),
            ylabel="kHz",
            bands=[FISHBONE_BAND],
        ),
        Panel(
            title=f"cross-phase B{PROBES[0] + 1} x B{PROBES[1] + 1}",
            kind="heatmap",
            x=t_ms,
            y=khz,
            z=cross_phase[keep],
            ylabel="kHz",
            bands=[FISHBONE_BAND],
        ),
    ]
