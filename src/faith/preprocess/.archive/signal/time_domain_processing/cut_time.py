import numpy as np


def cut_time(
    t: np.ndarray, data: np.ndarray, t_min: float, t_max: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cut the time-series data between two specified times.

    Parameters
    ----------
    t : np.ndarray, shape = (n,)
        Array of times.
    data : np.ndarray, shape = (..., n)
        The input data to be cut.
    t_min : float
        Minimum time to cut data.
    t_max : float
        Maximum time to cut data.

    Returns
    -------
    np.ndarray: Cut time array.
    np.ndarray: Cut data array.
    """
    t_indx_min = np.argmin(abs(np.array(t) - t_min))
    t_indx_max = np.argmin(abs(np.array(t) - t_max))

    return t[t_indx_min:t_indx_max], data[..., t_indx_min:t_indx_max]


# reference
# @staticmethod
# def cut_time(time, data, t_min, t_max):
#     t_indx_min=np.argmin(abs(np.array(time)-t_min))
#     t_indx_max=np.argmin(abs(np.array(time)-t_max))

#     return time[t_indx_min:t_indx_max], data[...,t_indx_min:t_indx_max]
