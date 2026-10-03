"""Log-spectral BES window features, the recipe of Gill et al. (2024).

A window is 1024 samples of the BES array. The signal is band-passed 2.5-150 kHz
(4th-order Butterworth, causal); the window is split in 2 sub-windows of 512, each in
2 segments of 256; the FFT of each segment gives log10 of the squared magnitude of
bins 0-127; the 2 segments are averaged. One window is 2 x 128 features per channel.
Standardising a signal subtracts ``2 log10(sigma)`` from its log power, so the features
are stored for the band-passed signal and the per-channel offset is applied later, from
the training windows of a fold (``standardising_offset``).

The same functions serve any sampling rate: at 1 MHz a window is 1.02 ms, at the
corpus' 500 kHz it is 2.05 ms.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

WINDOW = 1024
SUB = 512
SEGMENT = 256
FREQS = 128
BAND = (2.5e3, 150e3)
#: Samples read ahead of a filtered stretch so the causal filter has settled.
PAD = 4096


def band_sos(fs: float) -> np.ndarray:
    """The 2.5-150 kHz 4th-order Butterworth band-pass, as second-order sections."""
    return butter(4, BAND, btype="bandpass", fs=fs, output="sos")


def window_starts(lo: int, hi: int, stride: int, window: int = WINDOW) -> np.ndarray:
    """Start samples of the windows wholly in ``[lo, hi)``, one every ``stride``."""
    if hi - lo < window:
        return np.zeros(0, dtype=np.int64)
    return lo + stride * np.arange((hi - lo - window) // stride + 1, dtype=np.int64)


def spectral_features(windows: np.ndarray) -> np.ndarray:
    """``(n, C, 1024)`` band-passed windows -> ``(n, 2, C, 128)`` log-spectral features.

    Axis 1 is the sub-window (first 512 samples, last 512). Float16, as stored.
    """
    n, channels, length = windows.shape
    if length != WINDOW:
        raise ValueError(f"windows are {length} samples, not {WINDOW}")
    seg = windows.reshape(n, channels, 2, 2, SEGMENT)
    spec = np.abs(np.fft.rfft(seg, axis=-1)[..., :FREQS]) ** 2
    logs = np.log10(spec + 1e-30).mean(axis=3)
    return logs.transpose(0, 2, 1, 3).astype(np.float16)


def filtered_windows(
    signal: np.ndarray, starts: np.ndarray, sos: np.ndarray, *, pad: int = PAD
) -> np.ndarray:
    """Band-pass ``signal`` (C, T) over the stretch ``starts`` cover; cut windows.

    The filter runs from ``pad`` samples before the first window so a start-up
    transient is gone by the first window. Returns ``(n, C, 1024)`` float64.
    """
    if starts.size == 0:
        return np.zeros((0, signal.shape[0], WINDOW))
    first = int(starts[0])
    read = max(0, first - pad)
    last = int(starts[-1]) + WINDOW
    filt = sosfilt(sos, signal[:, read:last].astype(np.float64), axis=1)
    cut = (starts - read)[:, None] + np.arange(WINDOW)
    return filt[:, cut].transpose(1, 0, 2)


def standardising_offset(power: np.ndarray) -> np.ndarray:
    """Per-channel additive offset of log10 power for unit variance.

    ``power`` is ``(n, C)``, each window's mean squared band-passed signal per channel;
    the training windows' mean is the variance the paper standardises by.
    """
    sigma2 = power.astype(np.float64).mean(axis=0)
    return -np.log10(np.maximum(sigma2, 1e-12))
