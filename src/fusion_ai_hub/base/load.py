import numpy as np

import h5py
import os
from pathlib import Path

from typing import Any, Union

def list_signals(
    path: Union[str, int, Any[os.Pathlike]],
    ) -> list:
    
    with h5py.File(path, 'r') as f:
        signals = list(f.keys())
        
    return signals


def load_sample(
    path: Union[str, int, Any[os.Pathlike]],
    signal: Union[str, list[str]],
    ) -> np.ndarray:
    
    with h5py.File(path, 'r') as f:
        data = f[signal]
        sample = data["data"][()]
        
    return sample


def load_time(
    path: Union[str, int, Any[os.Pathlike]],
    signal: Union[str, list[str]],
    ) -> np.ndarray:
    
    with h5py.File(path, 'r') as f:
        data = f[signal]
        time = data["time"][()]
        
    return time


def load_attributes(
    path: Union[str, int, Any[os.Pathlike]],
    signal: Union[str, list[str]],
    ) -> list:
    
    with h5py.File(path, 'r') as f:
        data = f[signal]
        attributes = list(data.attrs.keys())
        
    return attributes


def load_channels(
    path: Union[str, int, Any[os.Pathlike]],
    signal: Union[str, list[str]],
    ) -> list:
    
    with h5py.File(path, 'r') as f:
        data = f[signal]
        channels = data.attrs["channel_ids"]
        
    return channels


def load(
    path: Union[str, int, Any[os.Pathlike]],
    signal: Union[str, list[str]],
    ) -> np.ndarray:
    
    with h5py.File(path, 'r') as f:
        data = f[signal]

    return data