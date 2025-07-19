"""
Preemphasis
===========

.. autosummary::
    :toctree: generated/

    preemphasis
    deemphasis
"""
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .filtering import lfilter

__all__ = ['preemphasis', 'deemphasis']


def preemphasis(y: np.ndarray, *, coef: float = 0.97,
                zi: Optional[ArrayLike] = None, return_zf: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Pre-emphasis a signal with a first-order differencing filter:

        y[n] -> y[n] - coef * y[n-1]

    This function is taken from librosa and modified for own purposes.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n_samples,)]
        The signal on which to apply the pre-emphasis filter.
        Can be a multichannel signal.

    coef : float, default=0.97
        The pre-emphasis coefficient. It should be between 0 and 1.

        At the limit ``coef=0``, the signal is unchanged.

        At the limit ``coef=1``, the result is the first-order difference of
        the signal.

    zi : ArrayLike
        Initial filter state. When making successive calls to non-overlapping
        frames, this can be set to the ``zf`` returned from the previous call.

        By default, ``zi`` is initialized as ``2*y[0] - y[1]``.

    return_zf : bool, default=False
        If ``True``, return the final filter state.
        If ``False``, only return the pre-emphasized signal.

    Returns
    -------
    y_out : np.ndarray [shape=(..., n_samples,)]
        pre-emphasized signal
    zf : np.ndarray
        If ``return_zf=True``, the final filter state is also returned.
    """
    b = np.asarray([1.0, -coef], dtype=y.dtype)

    if zi is None:
        zi = 2 * y[..., 0:1] - y[..., 1:2]

    y_out, zf = lfilter(y, b=b, zi=zi, return_zf=True)

    if return_zf:
        return y_out, zf

    return y_out


def deemphasis(y: np.ndarray, *, coef: float = 0.97,
               zi: Optional[ArrayLike] = None, return_zf: bool = False) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    De-emphasize an audio signal with the inverse operation of preemphasis.

    If y = preemphasis(x, coef=coef, zi=zi), the deemphasis is:

    >>> x[i] = y[i] + coef * x[i-1]
    >>> x = deemphasis(y, coef=coef, zi=zi)

    Parameters
    ----------
    y : np.ndarray [shape=(..., n_samples,)]
        The signal on which to apply the pre-emphasis filter.
        Can be a multichannel signal.

    coef : float, default=0.97
        The pre-emphasis coefficient. It should be between 0 and 1.

        At the limit ``coef=0``, the signal is unchanged.

        At the limit ``coef=1``, the result is the first-order difference of
        the signal.

    zi : number
        Initial filter state. If inverting a previous preemphasis(), the same
        value should be used.

        By default, ``zi`` is initialized as
        ``((2 - coef) * y[0] - y[1]) / (3 - coef)``. This value corresponds to
        the transformation of the default initialization of ``zi`` in
        ``preemphasis()``, ``2*x[0] - x[1]``.

    return_zf : boolean
        If ``True``, return the final filter state.
        If ``False``, only return the pre-emphasized signal.

    Returns
    -------
    y_out : np.ndarray [shape=(..., n_samples)]
        pre-emphasized signal
    zf : np.ndarray
        If ``return_zf=True``, the final filter state is also returned.
    """
    b = np.array([1.0, -coef], dtype=y.dtype)

    if zi is None:
        # initialize with all zeros
        zi = np.zeros(list(y.shape[:-1]) + [1], dtype=y.dtype)
        y_out, zf = lfilter(y, b=b, zi=zi, return_zf=True)

        # factor in the linear extrapolation
        y_out -= (((2 - coef) * y[..., 0:1] - y[..., 1:2]) / (3 - coef)
                  * (coef ** np.arange(y.shape[-1]))
                  )

    else:
        y_out, zf = lfilter(y, b=b, zi=zi, return_zf=True)

    if return_zf:
        return y_out, zf
    else:
        return y_out
