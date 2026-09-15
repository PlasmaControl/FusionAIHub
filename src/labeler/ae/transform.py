"""The AE spectrogram transform, ported off tokeye.

Task 7a built the training dataset by calling ``tokeye.transforms.compute_stft``
once per CO2 channel and then standardising the stack the way
``tokeye.inference.model_infer`` does. tokeye is importable only from
``/scratch/gpfs/nc1514/tokeye/.venv`` (read-only, torch 2.9), so labeler
cannot depend on it at inference time - the pixi env has to run this on its
own. ``compute_stft`` is fifteen lines of scipy, so it is ported here verbatim
rather than reimplemented, and
``outputs/labelmaker/ae/scripts/pin_transform.py`` runs both side by side on a
real corpus record and records the max absolute difference.

The pipeline, in the order the dataset script applies it:

1. per channel, ``scipy.signal.ShortTimeFFT(hann(1024), hop=128).stft(x)``,
   magnitude, ``log1p``, drop the DC bin (512 bins left), clip to the array's
   own 1st/99th percentiles - i.e. ``compute_stft`` with its defaults;
2. standardise **per channel** over the whole ``(512, frames)`` array,
   ``(x - mean) / (std + 1e-6)`` - ``model_infer``'s normalisation;
3. restrict to bins ``164:512`` - 348 bins, 80.57-250.00 kHz on the
   500 kHz grid - which is what the network was trained on.

Steps 2 and 3 are in that order and must stay in it: the mean and std are
taken over the FULL 512-bin array, not over the band. Reversing them changes
every value the model sees.

The frequency axis is fixed by the sample rate, and 250 kHz is the Nyquist of
the 500 kHz CO2 record - the band's upper edge is the instrument's ceiling,
not a modelling choice.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .labels import BAND_HI_BIN, BAND_LO_BIN, HOP, N_BINS, N_FFT

#: `tokeye.transforms.DEFAULT_CLIP_LOW` / `DEFAULT_CLIP_HIGH`.
CLIP_LOW = 1.0
CLIP_HIGH = 99.0
#: `tokeye.inference.model_infer`'s guard against a constant channel.
STD_EPS = 1e-6
#: Name written into every label group so a stored series says what made it.
TRANSFORM_NAME = (
    "tokeye compute_stft (hann 1024, hop 128, |.|, log1p, DC dropped, "
    "1/99 percentile clip) + per-channel standardisation, bins 164:512"
)
#: Sample rate the model was trained at and the corpus CO2 record carries.
NOMINAL_FS_HZ = 5.0e5


def _window() -> np.ndarray:
    return signal.get_window("hann", N_FFT)


def short_time_fft(fs_hz: float = 1.0) -> signal.ShortTimeFFT:
    """The one STFT configuration this model is defined by."""
    return signal.ShortTimeFFT(_window(), hop=HOP, fs=float(fs_hz))


def frame_times_s(n_samples: int, fs_hz: float) -> np.ndarray:
    """Frame centre times, in seconds, relative to sample 0 of the record.

    `ShortTimeFFT.stft` pads the start, so the first few frames sit at
    negative times; `t()` is the authority on the grid and is what the task
    7a dataset script used, so it is what is reproduced here.
    """
    return np.asarray(short_time_fft(float(fs_hz)).t(int(n_samples)), dtype=np.float64)


def n_frames(n_samples: int) -> int:
    """Number of STFT frames a record of `n_samples` produces."""
    return int(frame_times_s(n_samples, 1.0).size)


def compute_stft(x: np.ndarray) -> np.ndarray:
    """One channel -> `(512, frames)` float32; `tokeye.transforms.compute_stft`.

    Ported line for line from that function with its default arguments, for
    the single-row case (a two-row input would make tokeye form a
    cross-spectrum, which task 7a deliberately never does - each CO2 chord is
    transformed on its own).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected one channel, got shape {x.shape}")
    sxx = short_time_fft().stft(x[None, :])[0]
    sxx = np.log1p(np.abs(sxx))
    sxx = sxx[1:, :]                                    # drop DC
    vmin, vmax = np.percentile(sxx, [CLIP_LOW, CLIP_HIGH])
    return np.clip(sxx, vmin, vmax).astype(np.float32)


def spectrogram(signals: np.ndarray) -> np.ndarray:
    """`(C, N)` waveforms -> `(C, 512, frames)` float32, one channel at a time."""
    signals = np.asarray(signals)
    if signals.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {signals.shape}")
    return np.stack([compute_stft(row) for row in signals], axis=0)


def standardise(raw: np.ndarray) -> np.ndarray:
    """Per-channel z-score over the whole `(bins, frames)` array.

    `tokeye.inference.model_infer`'s normalisation, and what task 7a fed the
    mask model and stored as the training input. The statistics are taken
    over all 512 bins - see the module docstring.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"expected (channels, bins, frames), got {raw.shape}")
    mean = raw.mean(axis=(1, 2))[:, None, None]
    std = raw.std(axis=(1, 2))[:, None, None]
    return ((raw - mean) / (std + STD_EPS)).astype(np.float32)


def restrict_to_band(spec: np.ndarray) -> np.ndarray:
    """Bins 164:512 - the 348-bin AE band the network's input axis is."""
    spec = np.asarray(spec)
    if spec.shape[-2] != N_BINS:
        raise ValueError(f"expected {N_BINS} bins, got {spec.shape[-2]}")
    return spec[..., BAND_LO_BIN:BAND_HI_BIN, :]


def band_freqs_khz(fs_hz: float = NOMINAL_FS_HZ) -> np.ndarray:
    """Centre frequency of each of the 348 band bins, in kHz."""
    fs_khz = float(fs_hz) / 1e3
    bins = np.arange(BAND_LO_BIN, BAND_HI_BIN, dtype=np.float64) + 1.0
    return bins * fs_khz / N_FFT


def model_input(signals: np.ndarray) -> np.ndarray:
    """`(C, N)` CO2 waveforms -> the `(C, 348, frames)` float32 the model eats.

    The whole transform in one call: STFT, standardise, band-restrict.
    """
    return restrict_to_band(standardise(spectrogram(signals)))
