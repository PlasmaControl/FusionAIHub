"""
Data extraction utilities for fusion dataset preparation.

This module contains functions for extracting data from HDF5 files,
determining plasma running time, and aligning signals to a common timebase.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample


def extract(shot: int, directory: Path, signal: str) -> pd.DataFrame:
    """
    Extract signal data from HDF5 file for a given shot.
    
    Args:
        shot: Shot number
        directory: Directory containing HDF5 files
        signal: Signal name to extract
        
    Returns:
        DataFrame containing the signal data
    """
    path = (directory / str(shot)).with_suffix(".h5")
    df = pd.read_hdf(path, key=signal)
    return pd.DataFrame(df)


def running_time(directory: Path, shot: int, ip_threshold: float) -> tuple[float, float]:
    """
    Determine the plasma running time for a shot based on plasma current threshold.
    
    Args:
        directory: Directory containing HDF5 files
        shot: Shot number
        ip_threshold: Plasma current threshold
        
    Returns:
        Tuple of (start_time, end_time) in milliseconds
    """
    path = (directory / str(shot)).with_suffix(".h5")
    with pd.HDFStore(path, 'r') as store:
        df = store['ip']['ipsip']
    df = df.loc[df > ip_threshold]
    start_time = df.index[0]
    end_time = df.index[-1]
    return start_time, end_time


def align(df: pd.DataFrame, start_time: float, end_time: float, fs: float) -> pd.DataFrame:
    """
    Align signal data to a common timebase and sampling frequency.
    
    Crops the signal to the specified time window, resamples to the target
    sampling frequency, and adds padding and state information.
    
    Args:
        df: Input DataFrame with signal data
        start_time: Start time for alignment (ms)
        end_time: End time for alignment (ms)
        fs: Target sampling frequency (kHz)
        
    Returns:
        Aligned DataFrame with data and state columns
    """
    # get sampling frequency
    fs_raw = len(df) / (df.index[-1] - df.index[0])
    
    # crop time
    df = df.loc[(df.index >= start_time) & (df.index <= end_time)]
    
    # resample
    num = len(df)
    num = int(num * fs / fs_raw)
    
    df = pd.DataFrame(
        {col: resample(df[col].values, num) for col in df.columns},
        index=np.linspace(df.index[0], df.index[-1], num)
    )
    
    # mark on-off states
    start_nan = (df.index[0] - start_time) * fs
    end_nan = (end_time - df.index[-1]) * fs
    start_pad = pd.DataFrame(
        0, index=pd.RangeIndex(start=int(start_nan)), columns=df.columns)
    end_pad = pd.DataFrame(
        0, index=pd.RangeIndex(start=int(len(df) + start_nan), stop=int(len(df) + start_nan + end_nan)), columns=df.columns)
    
    df_state = pd.DataFrame(True, index=df.index, columns=df.columns)
    start_pad_state = pd.DataFrame(False, index=start_pad.index, columns=df.columns)
    end_pad_state = pd.DataFrame(False, index=end_pad.index, columns=df.columns)
    
    df = pd.concat([start_pad, df, end_pad], ignore_index=True)
    df_state = pd.concat([start_pad_state, df_state, end_pad_state], ignore_index=True)
    df_state.columns = [f"{col}_state" for col in df.columns]
    
    # combine data with state
    df = df.astype(np.float32)
    df_state = df_state.astype(np.bool)
    df = pd.concat([df, df_state], axis=1)

    # convert time to ms
    df = df.rename_axis("time")

    return df 