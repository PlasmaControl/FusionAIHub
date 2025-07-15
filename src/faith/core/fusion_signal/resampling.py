import numpy as np
from scipy import signal


def resample(y: np.ndarray, *, orig_fs: float, target_fs: float,
             resample_type: str = "scipy", scale: bool = False,
             axis: int = -1) -> np.ndarray:
    """
    Resample a time series from orig_fs to target_fs.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n, ...)]
        time series, with `n` samples along the specified axis.

    orig_fs : number > 0 [scalar]
        original sampling frequency of ``y``

    target_fs : number > 0 [scalar]
        target sampling frequency

    resample_type : str (default: `scipy`)
        resample type

        'fft' or 'scipy'
            `scipy.signal.resample` Fourier method.
        'polyphase'
            `scipy.signal.resample_poly` polyphase filtering. (fast)
        'linear'
            `samplerate` linear interpolation. (very fast, but not bandlimited)

        .. note::
            Not all options yield a bandlimited interpolator. If you use
            `polyphase`, `linear`, or `zero_order_hold`, you need to be aware
            of possible aliasing effects.

        .. note::
            When using ``res_type='polyphase'``, only integer sampling rates
            are supported.

    scale : bool
        Scale the resampled signal so that ``y`` and ``y_hat`` have
        approximately equal total energy.

    axis : int
        The target axis along which to resample. Defaults to the trailing axis.

    Returns
    -------
    y_hat : np.ndarray [shape=(..., n * target_sr / orig_sr, ...)]
        ``y`` resampled from ``orig_sr`` to ``target_sr`` along the target axis
    """
    if orig_fs == target_fs:
        return y

    ratio = float(target_fs) / orig_fs

    n_samples = int(np.ceil(y.shape[axis] * ratio))

    if resample_type in ("scipy", "fft"):
        y_hat = signal.resample(y, n_samples, axis=axis)
    elif resample_type == "polyphase":
        if int(orig_fs) != orig_fs or int(target_fs) != target_fs:
            raise ValueError("polyphase resampling is only supported for "
                             "integer-valued sampling rates.")

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_fs = int(orig_fs)
        target_fs = int(target_fs)
        gcd = np.gcd(orig_fs, target_fs)
        y_hat = signal.resample_poly(y, target_fs // gcd, orig_fs // gcd,
                                     axis=axis)
    else:
        raise NameError("Unknown resampling type.")

    if scale:
        y_hat /= np.sqrt(ratio)

    # Match dtypes
    return np.asarray(y_hat, dtype=y.dtype)
