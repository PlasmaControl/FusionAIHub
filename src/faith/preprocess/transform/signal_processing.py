"""
Signal processing utilities for fusion dataset preparation.

This module contains functions for signal resampling and transformation,
including STFT transformations and nearest-neighbor resampling.
"""

import numpy as np
import torch
from scipy.signal import resample
from scipy.interpolate import interp1d
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


def resample_fn(y: np.ndarray, new_len: int) -> np.ndarray:
    """
    Resample a signal to a new length using scipy.signal.resample.
    
    Args:
        y: Input signal array
        new_len: Target length for resampled signal
        
    Returns:
        Resampled signal as numpy array
    """
    # orig_len = len(y)
    # gcd = np.gcd(orig_len, new_len)
    # up = new_len // gcd
    # down = orig_len // gcd
    # return resample_poly(y, up, down)
    resampled = resample(y, new_len)
    return np.asarray(resampled)


def stft_transform(
    x: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
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


def resample_transform(
    x: np.ndarray,
    ref_shape: tuple,
) -> np.ndarray:
    """
    Resample a signal to match a reference shape.
    
    Args:
        x: Input signal
        ref_shape: Reference shape (tuple from STFT result)

    Returns:
        Resampled signal to match reference time dimension
    """
    x = x.astype(np.float32)
    target_length = ref_shape[-1] # assuming time dimension is last dimension
    y = [resample_fn(x_, target_length) for x_ in x]
    y = np.expand_dims(y, axis=1)
    return np.array(y)

def resample_linear_transform(
    x: np.ndarray,
    ref_shape: tuple,
) -> np.ndarray:
    """
    Resample a signal to match a reference shape.
    """
    x = x.astype(np.float32)
    target_length = ref_shape[-1] # assuming time dimension is last dimension
    lold = np.linspace(0, 1, len(x[-1]))
    lnew = np.linspace(0, 1, target_length)
    y = [np.interp(lnew, lold, x_) for x_ in x]
    y = np.expand_dims(y, axis=1)
    return np.array(y)


def wav_like_transform(
    x: np.ndarray,
    max_lim: float = 40,
    cutoff_freq: float = 1.0,
    fs: float = 500,
) -> np.ndarray:
    """
    Transform a signal to a WAV-like format.
    """
    from scipy.signal import butter, filtfilt
    def highpass_filter(data, cutoff_freq=1.0, fs=44100, order=4):
        nyquist = fs / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    signal = df[df['mhr_4_state'] == True]['mhr_4'].values
    max_lim = 40
    signal = signal[~np.isnan(signal)]
    signal = highpass_filter(signal, cutoff_freq=1.0, fs=500)

    # Remove outliers beyond 3 standard deviations
    signal_std = signal.std()
    signal_mean = np.mean(signal)
    outlier_mask = np.abs(signal - signal_mean) <= 4 * signal_std
    signal = signal[outlier_mask]

    # Normalize signal to [-1, 1]
    amplitude = np.max(np.abs(signal))
    if amplitude > max_lim: print(f"Amplitude {amplitude} is greater than {max_lim}")
    signal = signal / max_lim

    # Convert to WAV format
    wav_vals = np.int16(signal * 32767)

def identity_transform(x: np.ndarray) -> np.ndarray:
    """
    Identity transform.
    
    Args:
        x: Input signal
        
    Returns:
        Input signal
    """
    y = x.astype(np.float32)
    y = np.expand_dims(y, axis=1)
    return y