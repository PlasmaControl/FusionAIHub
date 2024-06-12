import numpy as np

import h5py
import os
from pathlib import Path

from typing import Any, Union

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