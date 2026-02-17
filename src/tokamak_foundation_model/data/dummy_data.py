import numpy as np
import h5py
from dataclasses import dataclass
import typing
import random
from pathlib import Path


@dataclass
class Signal:
    abbr: str
    """Abbreviated name, e.g. `mhr`."""
    name: str
    """HDF5 group name, e.g. `magnetics_high_resolution`."""
    stft: bool
    """Whether to apply Short-Time Fourier Transform (STFT)."""
    num_channels: int
    """Number of channels."""


_DEFAULT_SIGNALS: typing.Final[list[Signal]] = [
    Signal(abbr="mhr", name="magnetics_high_resolution", stft=True, num_channels=8),
    Signal(abbr="ece", name="ece_cali", stft=True, num_channels=48),
    Signal(abbr="co2", name="co2_density", stft=True, num_channels=4),
    Signal(abbr="gas", name="gas", stft=False, num_channels=5),
    Signal(abbr="ech", name="ech", stft=False, num_channels=11),
    Signal(abbr="pin", name="p_inj", stft=False, num_channels=8),
    Signal(abbr="tin", name="t_inj", stft=False, num_channels=8),
]

# Define sampling rates for specific signals
_SAMPLING_RATES = {
    "mhr": 500e3,  # 500 kHz
    "ece": 500e3,  # 500 kHz
    "co2": 500e3,  # 500 kHz
    "gas": 1e3,  # 1 kHz
    "ech": 1e3,  # 1 kHz
    "pin": 1e3,  # 1 kHz
    "tin": 1e3,  # 1 kHz
}


def _get_signal_data(duration_s: float, sampling_rate: float, num_channels: int):
    """
    Generates time series data for a single signal.

    Args:
        duration_s: Duration of the signal in seconds.
        sampling_rate: Sampling rate of the signal in Hz.
        num_channels: Number of channels for the signal.

    Returns:
        A tuple (xdata, ydata) where xdata is the time vector and ydata is the signal data.
    """
    num_points = int(duration_s * sampling_rate)
    xdata = np.linspace(0, duration_s, num_points)
    ydata = (
        np.random.rand(num_points, num_channels) * 2 - 1
    )  # Random data between -1 and 1
    return xdata, ydata


def generate_multi_modal_dummy_sample(duration_s: float = 1.0):
    """
    Generates a single multi-modal dummy data sample with a new nested structure.

    Args:
        duration_s: The duration of the time-series signals in seconds.

    Returns:
        A dictionary containing the generated multi-modal dummy data.
    """
    sample_data = {}

    # Generate time-series signals using _DEFAULT_SIGNALS and _SAMPLING_RATES
    for signal_def in _DEFAULT_SIGNALS:
        sampling_rate = _SAMPLING_RATES.get(
            signal_def.abbr, 1e3
        )  # Default to 1kHz if not specified
        xdata, ydata = _get_signal_data(
            duration_s, sampling_rate, signal_def.num_channels
        )
        sample_data[signal_def.abbr] = {
            "xdata": xdata.astype(np.float32),
            "ydata": ydata.astype(np.float32),
        }

    # Generate movie data (e.g., 10 frames of 3-channel 64x64 video)
    sample_data["movie"] = np.random.randint(
        0, 256, size=(10, 3, 64, 64), dtype=np.uint8
    )

    # Generate image data (e.g., a single 3-channel 128x128 image)
    sample_data["image"] = np.random.randint(0, 256, size=(3, 128, 128), dtype=np.uint8)

    # Generate profile data (e.g., a simple parabolic profile)
    profile_len = 50
    x = np.linspace(0, 10, profile_len)
    profile = -0.1 * (x - 5) ** 2 + 10  # Parabolic shape
    sample_data["profile"] = profile.astype(np.float32)

    # Generate log data as text
    num_log_entries = random.randint(5, 15)
    log_entries = []
    for i in range(num_log_entries):
        timestamp = f"2026-02-06 {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}.{random.randint(0, 999):03d}"
        log_level = random.choice(["INFO", "WARNING", "ERROR", "DEBUG"])
        message = random.choice(
            [
                f"Processing step {i}. Data integrity check: {random.choice(['passed', 'failed'])}",
                f"System startup complete. Version {random.randint(0, 9)}",
                f"Potential issue detected. {random.choice(['High temperature', 'Low flux', 'Pressure fluctuation'])}",
                f"Critical error at module X. {random.choice(['Connection lost', 'Memory overflow', 'Sensor failure'])}",
            ]
        )
        log_entries.append(f"{timestamp} - {log_level} - {message}")
    sample_data["log"] = "\n".join(log_entries)

    return sample_data


def create_multi_sample_hdf5(
    hdf5_path: str,
    num_samples: int,
    min_duration_s: float = 0.5,
    max_duration_s: float = 2.0,
):
    """
    Generates multiple multi-modal dummy data samples and saves them to an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file to be created.
        num_samples: Number of dummy samples to generate and save.
        min_duration_s: Minimum duration for randomly generated time series in seconds.
        max_duration_s: Maximum duration for randomly generated time series in seconds.
    """
    with h5py.File(hdf5_path, "w") as f:
        print(f"Creating HDF5 file: {hdf5_path} with {num_samples} samples...")
        for i in range(num_samples):
            duration_s = random.uniform(min_duration_s, max_duration_s)
            sample = generate_multi_modal_dummy_sample(duration_s=duration_s)

            sample_group = f.create_group(f"sample_{i:03d}")

            for signal_abbr, signal_data in sample.items():
                if signal_abbr not in ["movie", "image", "profile", "log"]:
                    signal_group = sample_group.create_group(signal_abbr)
                    signal_group.create_dataset("xdata", data=signal_data["xdata"])
                    signal_group.create_dataset("ydata", data=signal_data["ydata"])
                else:
                    # Store movie, image, profile, log directly in sample group
                    if signal_abbr == "log":
                        sample_group.create_dataset(
                            signal_abbr,
                            data=np.array(
                                signal_data, dtype=h5py.string_dtype(encoding="utf-8")
                            ),
                        )
                    else:
                        sample_group.create_dataset(signal_abbr, data=signal_data)
        print(f"Successfully created {num_samples} samples in {hdf5_path}")


def create_single_sample_hdf5():
    data_path = Path("C:\\Temp")
    shot = 182620
    with h5py.File(data_path / f"{shot}.h5", "r") as f:
        bes = (
            f["bes"]["axis1"][:],
            f["bes"]["block0_values"][:],
        )
        dalpha = (
            f["d_alpha"]["axis1"][:],
            f["d_alpha"]["block0_values"][:],
        )
        mse = (
            f["mse"]["axis1"][:],
            f["mse"]["block0_values"][:],
        )
        ts_core_density = (
            f["ts_core_density"]["axis1"][:],
            f["ts_core_density"]["block0_values"][:],
        )
        magnetics_high_resolution = (
            f["magnetics_high_resolution"]["axis1"][:],
            f["magnetics_high_resolution"]["block0_values"][:],
        )
        ece = (
            f["ece_cali"]["axis1"][:],
            f["ece_cali"]["block0_values"][:],
        )
        co2 = (
            f["co2_density"]["axis1"][:],
            f["co2_density"]["block0_values"][:],
        )
        gas = (
            f["gas"]["axis1"][:],
            f["gas"]["block0_values"][:],
        )
        ech = (
            f["ech"]["axis1"][:],
            f["ech"]["block0_values"][:],
        )
        pin = (
            f["p_inj"]["axis1"][:],
            f["p_inj"]["block0_values"][:],
        )
        tin = (
            f["t_inj"]["axis1"][:],
            f["t_inj"]["block0_values"][:],
        )

    with h5py.File(data_path / f"{shot}_image.h5", "r") as f:
        bolo = (
            f["bolo"]["time"][:],
            f["bolo"]["data"][:],
        )
        irtv = (
            f["irtv"]["time"][:],
            f["irtv"]["data"][:],
        )
        tangtv = (
            f["tangtv"]["time"][:],
            f["tangtv"]["data"][:],
        )

    with open(data_path / f"{shot}.txt", "r") as f:
        logfile = f.read()

    with h5py.File(data_path / f"{shot}_processed.h5", "w") as f:
        signal_group = f.create_group("bes")
        signal_group.create_dataset("xdata", data=bes[0])
        signal_group.create_dataset("ydata", data=bes[1])
        signal_group = f.create_group("dalpha")
        signal_group.create_dataset("xdata", data=dalpha[0])
        signal_group.create_dataset("ydata", data=dalpha[1])
        signal_group = f.create_group("mse")
        signal_group.create_dataset("xdata", data=mse[0])
        signal_group.create_dataset("ydata", data=mse[1])
        signal_group = f.create_group("ts_core_density")
        signal_group.create_dataset("xdata", data=ts_core_density[0])
        signal_group.create_dataset("ydata", data=ts_core_density[1])
        signal_group = f.create_group("mhr")
        signal_group.create_dataset("xdata", data=magnetics_high_resolution[0])
        signal_group.create_dataset("ydata", data=magnetics_high_resolution[1])
        signal_group = f.create_group("ece")
        signal_group.create_dataset("xdata", data=ece[0])
        signal_group.create_dataset("ydata", data=ece[1])
        signal_group = f.create_group("co2")
        signal_group.create_dataset("xdata", data=co2[0])
        signal_group.create_dataset("ydata", data=co2[1])
        signal_group = f.create_group("gas")
        signal_group.create_dataset("xdata", data=gas[0])
        signal_group.create_dataset("ydata", data=gas[1])
        signal_group = f.create_group("ech")
        signal_group.create_dataset("xdata", data=ech[0])
        signal_group.create_dataset("ydata", data=ech[1])
        signal_group = f.create_group("pin")
        signal_group.create_dataset("xdata", data=pin[0])
        signal_group.create_dataset("ydata", data=pin[1])
        signal_group = f.create_group("tin")
        signal_group.create_dataset("xdata", data=tin[0])
        signal_group.create_dataset("ydata", data=tin[1])
        signal_group = f.create_group("bolo")
        signal_group.create_dataset("xdata", data=bolo[0])
        signal_group.create_dataset("ydata", data=bolo[1])
        signal_group = f.create_group("irtv")
        signal_group.create_dataset("xdata", data=irtv[0])
        signal_group.create_dataset("ydata", data=irtv[1])
        signal_group = f.create_group("tangtv")
        signal_group.create_dataset("xdata", data=tangtv[0])
        signal_group.create_dataset("ydata", data=tangtv[1])
        signal_group = f.create_group("log")
        signal_group.create_dataset(
            "data", data=np.array(logfile, dtype=h5py.string_dtype(encoding="utf-8"))
        )
