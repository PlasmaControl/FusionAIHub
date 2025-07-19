from typing import Callable, Literal, Union

import numpy as np

# actually just use sklearn.preprocessing

def signal_optimize(
    signal: Union[str, list],
    apply: bool = True,
) -> dict:
    """
    The idea is to have a user input a list of files and then the function will read through a library to see if the channel has any special scaling rules. If it will automatically apply the scaling. There's also an option to only output the scaling rule considerations without applying them. I'm thinking there could be some dictionary file with this.
    
    For example, the signal 'dalpha' needs to be log transformed, but no other signals need to be log transformed. The signal 'density' needs to be multiplied by 10^-19.

    Parameters
    ----------
    signal : Union[str, list]
        _description_

    Returns
    -------
    dict
        _description_
    """

    return None


def get_scaling_factor(
    data: dict,
    scaling: Union[Literal["mean", "std", "norm", "oom", "min", "max"], Callable] = "mean",
) -> dict:
    """
    Apply a scaling operation specified by the `scaling` parameter to each array in the provided `data` dictionary.
    Supports standard operations like 'mean', 'std', 'min', and 'max', or any function that operates over a numpy array.
    Also supports getting order of magnitude (oom).
    
    Parameters
    ----------
    data : dict
        Dictionary where keys are channel identifiers and values are dicts with key 'zdata' pointing to numpy arrays.
    scaling : Union[Literal["mean", "std", "min", "max"], Callable], optional
        The scaling operation to apply. Can be one of 'mean', 'std', 'min', 'max', or a function that accepts a numpy array.
        Defaults to 'mean'.

    Returns
    -------
    dict
        Dictionary with the same keys as `data`, where each value is the result of the scaling operation applied to `data[key]['zdata']`.

    Examples
    --------
    >>> data = {'channel1': {'zdata': np.array([1, 2, 3])}}
    >>> print(get_scaling_factor(data, scaling='max'))
    {'channel1': 3}
    """

    # Map string identifiers to numpy functions
    scaling_functions = {
        "mean": np.mean,
        "std": np.std,
        "norm": np.linalg.norm,
        "oom": lambda x: np.floor(np.log10(abs(x))),
        "min": np.min,
        "max": np.max,
    }

    # If the scaling argument is a string, use the corresponding numpy function
    if isinstance(scaling, str):
        scaling_function = scaling_functions.get(scaling)
        if scaling_function is None:
            raise ValueError(f"Unsupported scaling operation: {scaling}")
    elif callable(scaling):
        scaling_function = scaling
    else:
        raise TypeError("Scaling must be either a string key for predefined functions or a callable.")

    scaled_values = {}
    for key, value in data.items():
        try:
            data_array = value['zdata']
            scaled_value = scaling_function(data_array)
            if np.isnan(scaled_value).any():
                scaled_value = 0  # Handle NaN values, if any
            scaled_values[key] = scaled_value
        except Exception as e:
            raise RuntimeError(f"Error processing {key}: {str(e)}")

    return scaled_values

def normalize(
    data: dict,
    norm: dict,
    std: dict,
) -> dict:

    return None

def standardize(
    data: dict,
    mean: dict,
    std: dict,
) -> dict:

    return None
