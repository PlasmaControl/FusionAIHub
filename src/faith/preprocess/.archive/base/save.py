import warnings
from pathlib import Path

import h5py
import numpy as np


def dict_to_hdf5(
    dictionary: dict,
    h5file: h5py.File,
    compression: str | None = None,
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
    warnings.warn(
        "dict_to_hdf5 is deprecated. Use the new function save instead",
        DeprecationWarning,
        stacklevel=2,
    )

    for key, value in dictionary.items():
        if isinstance(value, dict):
            group = h5file.create_group(key)
            dict_to_hdf5(value, group, compression)
        else:
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            h5file.create_dataset(
                key,
                data=value,
                compression=compression,
                chunks=True,
            )


def save(
    dictionary: dict,
    path: str | Path,
    file_format: str = "h5",
    compression: str | None = None,
) -> None:
    if file_format == "h5":
        if not isinstance(path, Path):
            path = Path(path)
        if not path.suffix == ".h5":
            path = path.with_suffix(".h5")

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            dict_to_hdf5(dictionary, f, compression)

    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    print(f"Saved to {path}")
