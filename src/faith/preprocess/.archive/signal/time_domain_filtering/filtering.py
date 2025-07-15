"""
Filtering
=========

Time-domain filters
-------------------
.. autosummary::
    :toctree: generated/

    mel
    chroma
    wavelet
    semitone_filterbank

Window functions
----------------
.. autosummary::
    :toctree: generated/

    window_bandwidth
    get_window
"""
import numpy as np
from scipy import signal
from typing import Optional, Tuple, Union
from numpy.typing import ArrayLike
from scipy.signal import get_window


__all__ = ["lfilter", "filtfilt"]


def lfilter(y: np.ndarray, *, b: ArrayLike, a: Optional[ArrayLike] = None,
            zi: Optional[ArrayLike] = None, return_zf: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Filter a signal with a FIR or an IIR digital filter:

        y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                         - a[1]*y[n-1] - ... - a[N]*y[n-N]


    Parameters
    ----------
    y : np.ndarray [shape=(..., n_samples,)]
        The signal on which to apply the digital filter.
        Can be a multichannel signal.

    b : ArrayLike
        The numerator coefficient vector in a 1-D sequence.

    a : ArrayLike
        The denominator coefficient vector in a 1-D sequence. Note that only
        the coefficients beginning from ``a[1]`` need to be supplied. If ``a``
        is ``None``, then the resulting filter is a FIR filter.

    zi : ArrayLike
        Initial filter state. When making successive calls to non-overlapping
        frames, this can be set to the ``zf`` returned from the previous call.

    return_zf :
        If ``True``, return the final filter state.
        If ``False``, only return the pre-emphasized signal.

    Returns
    -------
    y_out : np.ndarray [shape=(..., n_samples,)]
        Filtered signal.
    zf : np.ndarray
        If ``return_zf=True``, the final filter state is also returned.
    """
    b = np.asarray(b, dtype=y.dtype)

    if a is None:
        a = [1.0]
    else:
        a = np.insert(a, 0, 1.0)

    a = np.asarray(a, dtype=y.dtype)

    if zi is None:
        zi = signal.lfilter_zi(b, a)

    zi = np.atleast_1d(zi)

    y_out, zf = signal.lfilter(b, a, y, zi=np.asarray(zi, dtype=y.dtype))

    if return_zf:
        return y_out, zf

    return y_out


def filtfilt(y: np.ndarray, *, b: ArrayLike, a: Optional[ArrayLike] = None) \
        -> np.ndarray:
    """
    Filter a signal with a FIR or IIR digital filter forward and backward.

    This function applies a linear digital filter twice, once forward and once
    backwards. The combined filter has zero phase and a filter order twice that
    of the original.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n_samples,)]
        The signal on which to apply the digital filter.
        Can be a multichannel signal.

    b : ArrayLike
        The numerator coefficient vector in a 1-D sequence.

    a : ArrayLike
        The denominator coefficient vector in a 1-D sequence. Note that only
        the coefficients beginning from ``a[1]`` need to be supplied. If ``a``
        is ``None``, then the resulting filter is a FIR filter.

    Returns
    -------
    y_out : np.ndarray [shape=(..., n_samples)]
        Filtered signal.
    """
    b = np.asarray(b, dtype=y.dtype)

    if a is None:
        a = [1.0]
    else:
        a = np.insert(a, 0, 1.0)

    a = np.asarray(a, dtype=y.dtype)
    y_out = signal.filtfilt(b, a, y)
    return y_out
