from pathlib import Path

import h5py


def list_signals(
    path: str | Path,
) -> list[str]:
    with h5py.File(path, "r") as f:
        signals = list(f.keys())

    return signals


def load_sample(
    path: str | Path,
    signal_name: str | list[str] | None = None,
) -> dict:
    samples = {}
    with h5py.File(path, "r") as f:
        if signal_name is None:
            signal_name = list(f.keys())
        if isinstance(signal_name, str):
            signal_name = [signal_name]
        for signal in signal_name:
            data = f[signal]
            samples[signal] = data["data"][()]

    return samples


def load_time(
    path: str | Path,
    signal_name: str | list[str] | None = None,
) -> dict:
    times = {}
    with h5py.File(path, "r") as f:
        if signal_name is None:
            signal_name = list(f.keys())
        if isinstance(signal_name, str):
            signal_name = [signal_name]
        for signal in signal_name:
            data = f[signal]
            times[signal] = data["time"][()]

    return times


def load_attributes(
    path: str | Path,
    signal_name: str | list[str] | None = None,
) -> dict:
    attributes = {}
    with h5py.File(path, "r") as f:
        if signal_name is None:
            signal_name = list(f.keys())
        if isinstance(signal_name, str):
            signal_name = [signal_name]
        for signal in signal_name:
            data = f[signal]
            attributes[signal] = list(data.attrs.keys())

    return attributes


def load_channels(
    path: str | Path,
    signal_name: str | list[str] | None = None,
) -> dict:
    channels = {}
    with h5py.File(path, "r") as f:
        if signal_name is None:
            signal_name = list(f.keys())
        if isinstance(signal_name, str):
            signal_name = [signal_name]
        for signal in signal_name:
            data = f[signal]
            channels[signal] = data.attrs["channel_ids"]

    return channels


def load(
    path: str | Path,
    signal_name: str | list[str] | None = None,
) -> dict:
    loaded_data = {}
    with h5py.File(path, "r") as f:
        if signal_name is None:
            signal_name = list(f.keys())
        if isinstance(signal_name, str):
            signal_name = [signal_name]
        for signal in signal_name:
            data = f[signal]
            loaded_data[signal] = data

    return loaded_data
