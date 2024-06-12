import numpy as np

import h5py
import os
from pathlib import Path

from typing import Any, Union


def load_data(
    path: Union[str, int, Any[os.Pathlike]],
    ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    path : Union[str, int, Any[os.Pathlike]]
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    
    with h5py.File(path, 'r') as f:
        data = f['data'][()]
    return data


def load_time(
    path: Union[str, int, Any[os.Pathlike]],
    ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    path : Union[str, int, Any[os.Pathlike]]
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    
    with h5py.File(path, 'r') as f:
        time = f['time'][()]
    return time


def load_attributes(
    path: Union[str, int, Any[os.Pathlike]],
    ) -> list:
    """_summary_

    Parameters
    ----------
    path : Union[str, int, Any[os.Pathlike]]
        _description_

    Returns
    -------
    list
        _description_
    """
    
    with h5py.File(path, 'r') as f:
        attributes = list(f.attrs.keys())
    return attributes


def load(
    path: Union[str, int, Any[os.Pathlike]],
    ) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    path : Union[str, int, Any[os.Pathlike]]
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """
    
    with h5py.File(path, 'r') as f:
        data = f['data'][()]
        time = f['time'][()]
        attributes = list(f.attrs.keys())

    return {'data': data,
            'time': time,
            'attributes': attributes,
            }


def dict_to_hdf5(
    dictionary: dict,
    h5file: h5py.File,
    compression: str = None,
    ) -> None:
    """_summary_

    Parameters
    ----------
    dictionary : dict
        _description_
    h5file : h5py.File
        _description_
    compression : str, optional
        _description_, by default None
    """
    
    for key, value in dictionary.items():
        if isinstance(value, dict):
            group = h5file.create_group(key)
            dict_to_hdf5(value, group, compression)
        else:
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            h5file.create_dataset(key,
                                  data=value,
                                  compression=compression,
                                  chunks=True,
                                  )


def save(
    dictionary: dict,
    path: Union[str, int, Any[os.Pathlike]],
    compression: str = None,
    ) -> None:
    """_summary_

    Parameters
    ----------
    dictionary : dict
        _description_
    path : Union[str, int, Any[os.Pathlike]]
        _description_
    compression : str, optional
        _description_, by default None
    """
    
    if not path.endswith('.h5'):
        path += '.h5'
    
    path.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        dict_to_hdf5(dictionary, f, compression)

    print(f'Saved to {path}')


def merge(
    path_1: Union[str, int, Any[os.Pathlike]],
    path_2: Union[str, int, Any[os.Pathlike]],
    path_out: Union[str, int, Any[os.Pathlike]],
    ) -> None:
    """_summary_

    Parameters
    ----------
    path_1 : Union[str, int, Any[os.Pathlike]]
        _description_
    path_2 : Union[str, int, Any[os.Pathlike]]
        _description_
    path_out : Union[str, int, Any[os.Pathlike]]
        _description_

    Returns
    -------
    _type_
        _description_
    """

    return None