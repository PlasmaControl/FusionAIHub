import numpy as np
import h5py

def read_data(file_path: str) -> np.ndarray:
    """
    Read data from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    np.ndarray
        The data stored in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        data = f['data'][()]
    return data

def read_time(file_path: str) -> np.ndarray:
    """
    Read time from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    np.ndarray
        The time stored in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        time = f['time'][()]
    return time

def read_attributes(file_path: str) -> list:
    """
    Read attributes from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    list
        A list of attributes stored in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        attributes = list(f.attrs.keys())
    return attributes

def read_file(file_path: str) -> dict:
    """
    Read data, time, and attributes from an HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing the data, time, and attributes stored in the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        data = f['data'][()]
        time = f['time'][()]
        attributes = list(f.attrs.keys())
    return {'data': data, 'time': time, 'attributes': attributes}

# reference
# def read_file(discharge, file_suffix, df_time):

#     path = find_path(discharge)
#     file = h5py.File(f'{path}{discharge}_{file_suffix}.h5', 'r')
#     keys = file.keys()

#     for i, key in enumerate(keys):
#         dict_tmp = {'xdata': file[key]['xdata']}
#         if len(file[key]['zdata'].shape) == 2:
#             for j in range(file[key]['zdata'].shape[0]):
#                 dict_tmp[key+str(j)] = file[key]['zdata'][j, :]
#         elif len(file[key]['zdata'].shape) == 1:
#             dict_tmp[key] = file[key]['zdata']
            
#         df_tmp = pd.DataFrame(dict_tmp).astype('float32')
#         if i == 0:
#             df = pd.merge_asof(df_time, df_tmp, on='xdata',
#                                 direction='nearest')
#         else:
#             df = pd.merge_asof(df, df_tmp, on='xdata',
#                                 direction='nearest')
#     file.close()
#     return df