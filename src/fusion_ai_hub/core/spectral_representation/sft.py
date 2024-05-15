import numpy as np
from scipy import signal
from typing import Any, Literal, Optional, Tuple, Union


__all__ = ['spectrogram', 'stft']


def spectrogram(y: np.ndarray, *, fs: float = 48000, n_fft: int = 2048,
                hop_length: int = 256, win_length: int = 2048,
                window: Union[str, float, Tuple[str, Any, ...]] = "hamming",
                scaling: Literal["magnitude", "psd"] = "magnitude",
                detrend: Literal["linear", "constant"] = "constant",
                pad_mode: Literal["zeros", "edge", "even", "odd"] = 'zeros',
                return_t_f: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    fft_window = signal.get_window(window, win_length, fftbins=True)

    sft = signal.ShortTimeFFT(fft_window, hop_length, fs, fft_mode='onesided',
                              mfft=n_fft, dual_win=None, scale_to=scaling,
                              phase_shift=None)

    t = sft.t(y.shape[0])
    f = sft.f

    spectrogram_matrix = sft.spectrogram(y, detr=detrend, padding=pad_mode)

    if return_t_f:
        return spectrogram_matrix, t, f
    return spectrogram_matrix


def stft(y: np.ndarray, *, fs: float = 48000, n_fft: int = 2048,
         hop_length: Optional[int] = None, win_length: Optional[int] = None,
         window: Union[str, float, Tuple[str, Any, ...]] = "hamming",
         scaling: Literal["magnitude", "psd"] = "magnitude",
         pad_mode: Literal["zeros", "edge", "even", "odd"] = 'zeros',
         return_t_f: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    fft_window = signal.get_window(window, win_length, fftbins=True)

    sft = signal.ShortTimeFFT(fft_window, hop_length, fs, fft_mode='onesided',
                              mfft=n_fft, dual_win=None, scale_to=scaling,
                              phase_shift=None)

    t = sft.t(y.shape[0])
    f = sft.f

    stft_matrix = sft.stft(y, padding=pad_mode)

    if return_t_f:
        return stft_matrix, t, f
    return stft_matrix
