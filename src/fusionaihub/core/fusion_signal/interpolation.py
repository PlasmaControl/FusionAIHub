import numpy as np
from scipy import interpolate
from typing import Optional


def interpolate_signal(y: np.ndarray, t_ori: np.ndarray, t_new: np.ndarray,
                       axis: Optional[int] = 0) -> np.ndarray:
    """
    Resample and interpolate missing values in a multivariate time-series.

    Parameters
    ----------
    y : np.ndarray, shape = (..., n)
        The input signal to be interpolated.
    t_ori : np.ndarray, shape = (n):
        Array of original times. If None, assumes evenly spaced intervals.
    t_new : np.ndarray:
        Array of target times for resampling.
    axis : int
        Axis along which to interpolate. 0 for rows, 1 for columns.

    Returns
    -------
    np.ndarray: Resampled and interpolated time-series.
    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")

    # Transpose data if we are interpolating along columns
    if axis == 1:
        y = y.T

    num_samples, num_features = y.shape
    if t_ori is None:
        t_ori = np.arange(num_samples)

    # Prepare the resampled array
    resampled_data = np.empty((len(t_new), num_features))

    # Resample and interpolate each feature (row in transposed data)
    for i in range(num_features):
        feature_data = y[:, i]
        nans = np.isnan(feature_data)
        known_times = t_ori[~nans]
        known_values = feature_data[~nans]
        resampled_data[:, i] = np.interp(t_new, known_times, known_values)

    # Transpose back if needed
    if axis == 1:
        resampled_data = resampled_data.T

    return resampled_data

