"""
Signal processing utilities for fusion data transformations.

This module contains functions for signal resampling and transformation.

NOTE: All transforms should take in a shape of (channels, time) and return a
shape of (channels, features, time).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


AVAILABLE_TRANSFORMS = [
    "transform_identity",
    "transform_stft",
]

AVAILABLE_RESAMPLINGS = [
    "resample_linear",
    "resample_polyphase",
    "resample_fourier",
]


def transform_identity(
    x: np.ndarray,
) -> np.ndarray:
    """
    Identity transform.

    Args:
        x: Input signal

    Returns:
        Input signal
    """
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=1)
    return x


def transform_stft(
    x: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """
    Apply STFT transformation to an individual sample.

    Transforms time-domain signal to frequency-domain representation using
    Short-Time Fourier Transform with logarithmic magnitude scaling.

    Args:
        x: Input time-domain signal

    Returns:
        Log-magnitude STFT representation
    """
    # TODO: make this modular pipeline-ish
    # TODO: parameterize window type
    # TODO: parameterize window size (check if gives warning or allowed)
    import torch

    x = x.astype(np.float32)
    x_tensor = torch.from_numpy(x).float()
    y = torch.stft(
        x_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    y = torch.abs(y)
    return y.numpy()


def resample_fourier(
    x: np.ndarray,
    new_len: int,
) -> np.ndarray:
    """
    Resample a signal to a new length using scipy.signal.resample.

    Args:
        x: Input signal
        new_len: Target length for resampled signal

    Returns:
        Resampled signal as numpy array
    """
    from scipy.signal import resample

    x = x.astype(np.float32)
    target_length = x.shape[-1]
    x = resample(x, target_length)
    return np.expand_dims(x, axis=1)


def resample_polyphase(
    x: np.ndarray,
    new_len: int,
) -> np.ndarray:
    """
    Resample a signal to a new length using scipy.signal.resample.

    Args:
        x: Input signal
        new_len: Target length for resampled signal

    Returns:
        Resampled signal as numpy array
    """
    from scipy.signal import resample_poly

    x = x.astype(np.float32)
    x = resample_poly(x, new_len, x.shape[-1])
    return np.expand_dims(x, axis=1)


def resample_linear(
    x: np.ndarray,
    ref_shape: tuple,
) -> np.ndarray:
    """
    Resample a signal to match a reference shape.
    """
    x = x.astype(np.float32)
    target_length = ref_shape[-1]
    lold = np.linspace(0, 1, len(x[-1]))
    lnew = np.linspace(0, 1, target_length)
    x = [np.interp(lnew, lold, x_) for x_ in x]
    x = np.expand_dims(x, axis=1)
    return x
