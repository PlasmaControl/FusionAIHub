"""Alfven eigenmodes, as coherent narrowband activity on more than one chord.

Lifted from `data/events/alfven_eigenmode/verification.ipynb`, which this
replaces. The two comments inside `crosspower` are load-bearing and were
each written after the corresponding bug.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from ..raw import raw_signal
from ..verify import CO2_CHORDS, Panel

#: Where TAEs and RSAEs live. Marked on the figure, not filtered for.
AE_BAND = (80.0, 250.0)

#: R0xV1, R0xV2, R0xV3 - the reference chord against each vertical one.
#: A feature on ONE pair only is more likely chord-specific noise than a
#: mode, which is the whole reason three pairs are drawn rather than one.
CHORD_PAIRS = ((0, 1), (0, 2), (0, 3))

#: A full-rate spectrogram of a 2 s window is ~3300 columns, and three of
#: those as plotly heatmaps is tens of megabytes of JSON and a wedged
#: browser tab. The server re-renders on zoom instead, so a window is always
#: this sharp however wide it is.
MAX_TIME_BINS = 1000

#: The whole-shot open is the widest render the server can be asked for, and
#: it is the FIRST one every shot pays: nothing narrows it, so it lands on
#: the cap. At 615 frequency rows x 976 columns x three pairs that measured
#: 36 MB of JSON and two seconds, before plotly has drawn anything. The
#: overview only has to show WHERE to look - the reviewer zooms for detail
#: and the server re-renders that window sharp - so both axes are coarser
#: here. A windowed render keeps the full frequency resolution, because
#: zooming in time is not a reason to lose it.
FULL_SHOT_MAX_TIME_BINS = 500
FULL_SHOT_MAX_FREQ_BINS = 256

GUIDANCE = (
    "<b>What you are looking for:</b> coherent narrowband activity inside "
    "80-250 kHz (between the dotted lines) that shows up on <b>more than one "
    "chord pair</b>. A feature on a single pair only is more likely "
    "chord-specific noise than a real mode. TAEs sit near the low end of the "
    "band; RSAEs sweep upward as the safety factor evolves through the shot."
    "<br><br>No AE-annotated shot is in the corpus - the 180 annotated shots "
    "span 170659-178879 and the corpus starts at 185601 - so every shot here "
    "is fetched live over PTDATA. The first open of a shot moves ~240 MB and "
    "takes several minutes; after that it is cached and reopening is instant."
)


def _block_mean(values, axis, max_bins):
    """Average `values` down to at most `max_bins` along `axis`.

    Ceiling division: floor division rounds the width DOWN, so it leaves the
    cap broken silently whenever max_bins does not divide the length - 389
    columns at max_bins=250 floor to width 1, which reduces nothing, and 3300
    at max_bins=1000 floor to width 3, which still ships 1100 bins. The
    reshape below "succeeds" either way, just on more bins than max_bins.

    The final partial block is dropped, so up to `width - 1` bins fall off
    the end - about 8 ms of a 2 s window at the default column cap. That is
    deliberate: a partial block averages fewer segments than its neighbours,
    so it would be brighter or dimmer for a reason that has nothing to do
    with the plasma.
    """
    values = np.moveaxis(np.asarray(values), axis, 0)
    if max_bins is None or len(values) <= max_bins:
        return np.moveaxis(values, 0, axis)
    width = -(-len(values) // max_bins)
    usable = (len(values) // width) * width
    shape = (-1, width, *values.shape[1:])
    return np.moveaxis(values[:usable].reshape(shape).mean(axis=1), 0, axis)


def crosspower(
    time_ms,
    a,
    b,
    *,
    nperseg=2048,
    max_khz=300.0,
    max_bins=MAX_TIME_BINS,
    max_freq_bins=None,
):
    """Log |S_a . conj(S_b)| for two chords, as `(freq_khz, t_ms, z)`.

    The rate comes from the SPAN, never from a median of successive
    differences: a float32 time vector quantises its spacing at t ~ 3 s, and
    the resulting rate is wrong by percents, in a biased direction, with
    nothing on screen to give it away.

    Downsampling to `max_bins` columns and `max_freq_bins` rows is done by
    `_block_mean`, whose docstring says what it drops and why.
    """
    time_ms = np.asarray(time_ms, dtype="float64")
    if len(time_ms) < 2 or time_ms[-1] == time_ms[0]:
        raise ValueError(
            "crosspower needs a time vector spanning more than one instant; "
            f"got {len(time_ms)} sample(s). An out-of-range t_range slices the "
            "signal to an empty or single-sample window, which divides to nan "
            "and draws a blank panel."
        )
    rate = (len(time_ms) - 1) / ((time_ms[-1] - time_ms[0]) / 1000.0)
    kwargs = {
        "fs": rate,
        "nperseg": min(nperseg, len(time_ms)),
        "noverlap": min(nperseg, len(time_ms)) // 2,
        "mode": "complex",
    }
    freq, times_s, spec_a = scipy_signal.spectrogram(np.asarray(a), **kwargs)
    _, _, spec_b = scipy_signal.spectrogram(np.asarray(b), **kwargs)
    cross = spec_a * np.conj(spec_b)

    keep = freq <= max_khz * 1000.0
    freq_khz = freq[keep] / 1000.0
    magnitude = np.abs(cross[keep])
    t_ms = times_s * 1000.0 + time_ms[0]

    # The average is taken over the POWER and the log comes AFTER it.
    # Averaging the log instead is a geometric mean, which is pulled down by
    # the quiet bins in a block and so suppresses exactly the short bursts
    # this panel exists to show.
    # Each axis' coordinates go through the same call as the data, so the
    # two cannot disagree about which blocks survived.
    magnitude = _block_mean(magnitude, 1, max_bins)
    t_ms = _block_mean(t_ms, 0, max_bins)
    magnitude = _block_mean(magnitude, 0, max_freq_bins)
    freq_khz = _block_mean(freq_khz, 0, max_freq_bins)
    return freq_khz, t_ms, np.log10(magnitude + 1e-30)


def panels(shot, *, t_range=None, paths=None):
    """One crosspower heatmap per chord pair, over the window asked for."""
    co2 = raw_signal(int(shot), "co2", t_range=t_range, paths=paths)
    coarse = t_range is None
    built = []
    for reference, vertical in CHORD_PAIRS:
        freq_khz, t_ms, power = crosspower(
            co2.x,
            co2.y[reference],
            co2.y[vertical],
            max_bins=FULL_SHOT_MAX_TIME_BINS if coarse else MAX_TIME_BINS,
            max_freq_bins=FULL_SHOT_MAX_FREQ_BINS if coarse else None,
        )
        built.append(
            Panel(
                title=f"CO2 crosspower {CO2_CHORDS[reference]} x {CO2_CHORDS[vertical]}",
                kind="heatmap",
                x=t_ms,
                y=freq_khz,
                z=power,
                ylabel="kHz",
                bands=[AE_BAND],
            )
        )
    return built
