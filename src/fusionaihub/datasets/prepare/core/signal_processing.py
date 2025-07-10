"""
Signal processing utilities for fusion dataset preparation.

This module contains functions for signal resampling and transformation,
including STFT transformations and nearest-neighbor resampling.
"""

import numpy as np
import torch
from scipy.signal import resample


def resample_nearest(y: np.ndarray, new_len: int) -> np.ndarray:
    """
    Resample a signal to a new length using scipy.signal.resample.
    
    Args:
        y: Input signal array
        new_len: Target length for resampled signal
        
    Returns:
        Resampled signal as numpy array
    """
    orig_len = len(y)
    gcd = np.gcd(orig_len, new_len)
    up = new_len // gcd
    down = orig_len // gcd
    # return resample_poly(y, up, down)
    resampled = resample(y, new_len)
    return np.asarray(resampled)


def transform_individual_sample(x: np.ndarray) -> np.ndarray:
    """
    Apply STFT transformation to an individual sample.
    
    Transforms time-domain signal to frequency-domain representation using
    Short-Time Fourier Transform with logarithmic magnitude scaling.
    
    Args:
        x: Input time-domain signal
        
    Returns:
        Log-magnitude STFT representation
    """
    x_tensor = torch.from_numpy(x).float()
    y = torch.stft(
        x_tensor, 
        n_fft=1024, 
        hop_length=256, 
        window=torch.hann_window(1024), 
        return_complex=True
    )
    y = torch.log(torch.abs(y))
    # y = torch.clip(y, min=-10, max=5)
    return y.numpy() 