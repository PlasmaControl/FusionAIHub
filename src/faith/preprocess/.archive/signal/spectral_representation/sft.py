from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
from scipy import signal

__all__ = ['spectrogram', 'stft']


def spectrogram(y: np.ndarray, *, fs: float = 48000, n_fft: int = 2048,
                hop_length: int = 256, win_length: int = 2048,
                window: Union[str, float, Tuple[str, Any, ...]] = "hamming",
                scaling: Literal["magnitude", "psd"] = "magnitude",
                detrend: Literal["linear", "constant"] = "constant",
                pad_mode: Literal["zeros", "edge", "even", "odd"] = 'zeros',
                return_t_f: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Spectrogram.

    The spectrogram is the absolute square of the STFT, and thus is always
    non-negative. This is a convenience function for calling ``stft`` /
    ``stft_detrend``. It represents a signal in the time-frequency domain by
    computing discrete Fourier transforms (DFT) over short overlapping windows.

    This function returns a real-valued matrix D such that it is the squared
    absolute value of the complex STFT coefficients.

    Parameters
    ----------
    y : np.ndarray [shape=(…, n)], real-valued
        The input signal. Multi-channel supported.
    fs : float
        The sampling frequency in Hertz
    n_fft : int
        Length of the windowed signal after padding with zeros. The number of
        rows in the matrix ``D`` is ``(1 + n_fft/2)``. It is always recommended
        setting n_fft to a power of two for optimizing the speed of the fast
        Fourier transform (FFT) algorithm.
    hop_length : int
        Number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to ``win_length // 4`` (see below).
    win_length : int
        Each frame of the input signal is windowed by window of length
        ``win_length`` and then padded with zeros to match ``n_fft``.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time) at
        the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal y.

        If unspecified, defaults to ``win_length = n_fft``.
    window : str, float, tuple
        Either:
            - a window specification (string, tuple, or number); see
            ``scipy.signal.get_window`` for details
            - a window function, such as ``scipy.signal.windows.hann``

        Defaults to a raised cosine window (‘hamming’), which is adequate for
        most applications in signal processing.
    scaling : str
        Normalization applied to the window function (‘magnitude’, ‘psd’ or
        None).

        If not None, the FFTs can be either interpreted as a magnitude or a
        power spectral density spectrum.

        The window function can be scaled by calling the ``scale_to`` method,
        or it is set by the initializer parameter ``scale_to``.
    detrend : str
        If detr is set to ‘constant’, the mean is subtracted, if set to
        “linear”, the linear trend is removed. This is achieved by calling
        ``scipy.signal.detrend``. If detr is a function, detr is applied to
        each segment. All other parameters have the same meaning as in stft.

        Note that due to the detrending, the original signal cannot be
        reconstructed by the istft.
    pad_mode : str
        Kind of values which are added, when the sliding window sticks out on
        either the lower or upper end of the input x. Zeros are added if the
        default ‘zeros’ is set. For ‘edge’ either the first or the last value
        of x is used. ‘even’ pads by reflecting the signal on the first or last
        sample and ‘odd’ additionally multiplies it with -1.
    return_t_f :
        If ``return_t_f=True``, the time and frequency bins are also returned.

    Returns
    -------
    spectrogram_matrix : np.ndarray [shape=(..., n_samples,)]
        Real-valued matrix of magnitude spectrogram.
    t : np.ndarray
        If ``return_t_f=True``, the time steps are returned.
    f : np.ndarray
        If ``return_t_f=True``, the frequency bins are returned.
    """
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
    """
    STFT.

    The STFT represents a signal in the time-frequency domain by computing
    discrete Fourier transforms (DFT) over short overlapping windows.

    This function returns a complex-valued matrix D such that

    - ``np.abs(D[..., f, t])`` is the magnitude of frequency bin ``f``
      at frame ``t``, and

    - ``np.angle(D[..., f, t])`` is the phase of frequency bin ``f``
      at frame ``t``.

    Parameters
    ----------
    y : np.ndarray [shape=(…, n)], real-valued
        The input signal. Multi-channel supported.
    fs : float
        The sampling frequency in Hertz
    n_fft : int
        Length of the windowed signal after padding with zeros. The number of
        rows in the matrix ``D`` is ``(1 + n_fft/2)``. It is always recommended
        setting n_fft to a power of two for optimizing the speed of the fast
        Fourier transform (FFT) algorithm.
    hop_length : int
        Number of audio samples between adjacent STFT columns.

        Smaller values increase the number of columns in ``D`` without
        affecting the frequency resolution of the STFT.

        If unspecified, defaults to ``win_length // 4`` (see below).
    win_length : int
        Each frame of the input signal is windowed by window of length
        ``win_length`` and then padded with zeros to match ``n_fft``.

        Smaller values improve the temporal resolution of the STFT (i.e. the
        ability to discriminate impulses that are closely spaced in time) at
        the expense of frequency resolution (i.e. the ability to discriminate
        pure tones that are closely spaced in frequency). This effect is known
        as the time-frequency localization trade-off and needs to be adjusted
        according to the properties of the input signal y.

        If unspecified, defaults to ``win_length = n_fft``.
    window : str, float, tuple
        Either:
            - a window specification (string, tuple, or number); see
            ``scipy.signal.get_window`` for details
            - a window function, such as ``scipy.signal.windows.hann``

        Defaults to a raised cosine window (‘hamming’), which is adequate for
        most applications in signal processing.
    scaling : str
        Normalization applied to the window function (‘magnitude’, ‘psd’ or
        None).

        If not None, the FFTs can be either interpreted as a magnitude or a
        power spectral density spectrum.

        The window function can be scaled by calling the ``scale_to`` method,
        or it is set by the initializer parameter ``scale_to``.
    pad_mode : str
        Kind of values which are added, when the sliding window sticks out on
        either the lower or upper end of the input x. Zeros are added if the
        default ‘zeros’ is set. For ‘edge’ either the first or the last value
        of x is used. ‘even’ pads by reflecting the signal on the first or last
        sample and ‘odd’ additionally multiplies it with -1.
    return_t_f :
        If ``return_t_f=True``, the time and frequency bins are also returned.

    Returns
    -------
    spectrogram_matrix : np.ndarray [shape=(..., n_samples,)]
        Real-valued matrix of magnitude spectrogram.
    t : np.ndarray
        If ``return_t_f=True``, the time steps are returned.
    f : np.ndarray
        If ``return_t_f=True``, the frequency bins are returned.
    """
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
