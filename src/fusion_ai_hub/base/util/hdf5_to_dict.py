import h5py
from pathlib import Path

def hdf5_to_dict(file_path: str) -> dict:
    """
    Convert an HDF5 file to a dictionary.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing the contents of the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        result = {}
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                result[key] = f[key][()]
            elif isinstance(f[key], h5py.Group):
                result[key] = hdf5_to_dict(f[key])
    return result

# reference
# def hdf5_to_dict(self, group):
#     result = {}
#     for key in group.keys():
#         if isinstance(group[key], h5py.Dataset):
#             result[key] = group[key][()]
#         elif isinstance(group[key], h5py.Group):
#             result[key] = self.hdf5_to_dict(group[key])
#     return result