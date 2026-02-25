import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch.nn.functional as F
import copy


# TODO: implement this for calculation
class WelfordTensor:
    """
    Welford algorithm for computing running statistics on batched multi-channel tensors.

    Computes per-channel statistics by aggregating across batch and all other dimensions.

    For signals (B, C, F, T) or (B, C, 1, T): computes stats per channel → shape (C,)
    For profiles (B, S, T): computes stats per spatial point → shape (S,)
    For videos (B, T, H, W): computes global stats → shape (1,)
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        self.n = 0
        self.M2 = None
        self.initialized = False

    def _initialize(self, value: torch.Tensor):
        """Initialize arrays based on first tensor's shape."""
        # Determine number of channels based on tensor shape (excluding batch dim)
        if value.ndim == 4:
            # (batch, channels, freq_bins, time) or (batch, channels, 1, time)
            n_channels = value.shape[1]
        elif value.ndim == 3:
            # (batch, spatial_points, time) or (batch, time, height) - ambiguous
            # Assume spatial/channel dim is second
            n_channels = value.shape[1]
        elif value.ndim == 2:
            # (batch, time) - single channel
            n_channels = 1
        else:
            # Shouldn't happen, but treat as single channel
            n_channels = 1

        self.mean = torch.zeros(n_channels, dtype=torch.float64)
        self.M2 = torch.zeros(n_channels, dtype=torch.float64)
        self.min_val = torch.full((n_channels,), float('inf'), dtype=torch.float64)
        self.max_val = torch.full((n_channels,), float('-inf'), dtype=torch.float64)
        self.initialized = True

    def update(self, value: torch.Tensor):
        """
        Update statistics with new batched tensor.

        Parameters
        ----------
        value : torch.Tensor
            Input tensor of shape:
            - (batch, channels, freq_bins, time) for spectrograms
            - (batch, channels, 1, time) for time series
            - (batch, spatial_points, time) for profiles
            - (batch, time, height, width) for videos
        """
        # Skip if contains NaN
        if torch.isnan(value).any():
            return

        # Initialize on first call
        if not self.initialized:
            self._initialize(value)

        # Convert to float64 for numerical stability
        value = value.to(dtype=torch.float64)

        # Compute per-channel statistics by flattening batch and all non-channel dims
        if value.ndim == 4 and value.shape[1] == self.mean.shape[0]:
            # (batch, channels, freq_bins, time) → flatten batch, freq, time
            # (B, C, F, T) → (C, B*F*T)
            batch_size = value.shape[0]
            n_channels = value.shape[1]
            value_flat = value.permute(1, 0, 2, 3).reshape(n_channels, -1)  # (C, B*F*T)

            # Per-channel mean, min, max
            batch_mean = value_flat.mean(dim=1)
            batch_min = value_flat.min(dim=1).values
            batch_max = value_flat.max(dim=1).values
            n_samples = value_flat.shape[1]

            # For variance, we need sum of squared deviations
            batch_var = value_flat.var(dim=1, unbiased=False)
            batch_M2 = batch_var * n_samples

        elif value.ndim == 3:
            # (batch, spatial_points, time) → flatten batch, time
            # (B, S, T) → (S, B*T)
            n_channels = value.shape[1]
            value_flat = value.permute(1, 0, 2).reshape(n_channels, -1)  # (S, B*T)

            batch_mean = value_flat.mean(dim=1)
            batch_min = value_flat.min(dim=1).values
            batch_max = value_flat.max(dim=1).values
            n_samples = value_flat.shape[1]

            batch_var = value_flat.var(dim=1, unbiased=False)
            batch_M2 = batch_var * n_samples

        else:
            # Video (batch, time, height, width) → global statistics
            value_flat = value.flatten()

            batch_mean = torch.tensor([value_flat.mean()], dtype=torch.float64)
            batch_min = torch.tensor([value_flat.min()], dtype=torch.float64)
            batch_max = torch.tensor([value_flat.max()], dtype=torch.float64)
            n_samples = value_flat.shape[0]

            batch_var = value_flat.var(unbiased=False)
            batch_M2 = batch_var * n_samples

        # Parallel Welford's algorithm for combining batches
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        n_old = self.n
        n_new = n_samples
        n_total = n_old + n_new

        # Update mean
        delta = batch_mean - self.mean
        self.mean = (n_old * self.mean + n_new * batch_mean) / n_total

        # Update M2 (sum of squared deviations)
        # M2_total = M2_old + M2_new + delta^2 * n_old * n_new / n_total
        self.M2 = self.M2 + batch_M2 + delta * delta * n_old * n_new / n_total

        self.n = n_total

        # Update min/max
        self.min_val = torch.minimum(self.min_val, batch_min)
        self.max_val = torch.maximum(self.max_val, batch_max)

    def _compute_std(self):
        """Compute standard deviation from M2."""
        if self.n > 1:
            self.std = torch.sqrt(self.M2 / (self.n - 1))
        else:
            self.std = torch.zeros_like(self.mean)

    def compute(self):
        """
        Compute final statistics.

        Returns
        -------
        dict
            Dictionary with numpy arrays:
            - 'mean': per-channel mean
            - 'std': per-channel standard deviation
            - 'min_val': per-channel minimum
            - 'max_val': per-channel maximum
        """
        self._compute_std()

        return {
            "mean": self.mean.numpy(),
            "std": self.std.numpy(),
            "min_val": self.min_val.numpy(),
            "max_val": self.max_val.numpy(),
        }


def compute_preprocessing_stats(
        datasets,
        output_path="preprocessing_stats.pt",
        num_samples=1000
):
    """Compute preprocessing statistics across multiple datasets.

    Args:
        datasets: List of TokamakH5Dataset instances
        output_path: Where to save statistics
        num_samples: Number of samples per dataset to use
    """
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm

    combined = ConcatDataset(datasets)
    dataloader = DataLoader(combined, batch_size=32, collate_fn=collate_fn, num_workers=1)

    # Get signal names from first dataset
    signal_configs = datasets[0].SIGNAL_CONFIGS
    movie_configs = datasets[0].MOVIE_CONFIGS

    welford_stats = {cfg.name: WelfordTensor() for cfg in signal_configs + movie_configs}

    for batch in tqdm(dataloader):
        for modality_name, tensor in batch.items():
            # Update statistics
            welford_stats[modality_name].update(tensor)

    # Compute final statistics
    final_stats = {
        modality: tracker.compute()
        for modality, tracker in welford_stats.items()
    }
    torch.save(final_stats, output_path)

    print(f"Saved statistics to {output_path}")
    return final_stats


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""

    method: str = "none"  # "none", "standardize", "normalize", "log_standardize"
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    eps: float = 1e-8


@dataclass
class SignalConfig:
    """Configuration for a single signal/diagnostic."""

    name: str
    hdf5_keys: list[str]
    num_channels: int
    target_fs: float
    apply_stft: bool
    channels_to_use: slice = field(default_factory=lambda: slice(0, -1))  # Optional slice to select specific channels
    preprocess: PreprocessConfig = None  # Add preprocessing config

    def __post_init__(self):
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()


@dataclass
class MovieConfig:
    """Configuration for a movie/video diagnostic."""

    name: str  # Key in output dict
    hdf5_keys: list[str]  # Possible HDF5 paths to search
    channels: int  # Color channels (e.g., 3 for RGB)
    target_fps: int  # Target frames per second after resampling
    height: int  # Frame height
    width: int  # Frame width
    preprocess: PreprocessConfig = None  # Add preprocessing config

    def __post_init__(self):
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()
   
            
@dataclass
class ValueConfig:
    """Configuration for dataloader numericals (maybe a another description)"""
    
    rdcc_nbytes: int # Number of bytes for the chunk cache. Adjust based on dataset size and memory constraints.
    rdcc_nslots: int # Number of chunk slots in the cache. Adjust based on dataset size and access patterns.
    ms_to_s: float = 1/1000 # Conversion factor from seconds to milliseconds for time calculations

class TokamakH5Dataset(Dataset):
    """
    Dataset for loading multi-modal tokamak data from HDF5 files.

    Processing pipeline:
    1. Load raw data at native sampling rate
    2. Apply processing (STFT or nothing)
    3. Resample to target time frames

    For prediction mode:
    - Loads extended window (input_duration + prediction_horizon)
    - Processes entire window jointly
    - Splits into input and target frames
    """

    # Define all signal configurations with preprocessing
    SIGNAL_CONFIGS = [
        SignalConfig(
            name = "mhr",
            hdf5_keys=["mhr"],
            num_channels=8, # change to 6?, and then later specify which ones
            target_fs=500e3,
            apply_stft=True,
            channels_to_use=slice(2, 8),  # Use only the first 8 channels
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "ece",
            ["ece"],
            48, # change to 40?, and then later specify which ones
            500e3,
            apply_stft=True,
            channels_to_use=slice(0, 40),  # Use only the first 40 channels
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "co2",
            ["co2"],
            4,
            500e3,
            apply_stft=True,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "d_alpha",
            ["dalpha"],
            6,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "gas",
            ["gas"],
            5,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "ech",
            ["ech"],
            11,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "pin",
            ["pin"],
            8,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "tin",
            ["tin"],
            8,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        # TODO: Include Gas as additional actuator!!!
        SignalConfig(
            "mse",
            ["mse"],
            69,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "ts_core_density",
            ["ts_core_density"],
            44,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
    ]

    MOVIE_CONFIGS = [
        MovieConfig("bolo", ["bolo"], 1, 50, 80, 120),
        MovieConfig("irtv", ["irtv"], 1, 50, 513, 640),
        MovieConfig("tangtv", ["tangtv"], 1, 50, 240, 720),
    ]
    
    VALUE_CONFIG = ValueConfig(
        rdcc_nbytes=1024**2 * 32,  # 32 MB chunk cache
        rdcc_nslots=300,  # Number of chunk slots
        ms_to_s=1/1000,  # Conversion factor from milliseconds to seconds
    )

    def __init__(
            self,
            hdf5_path: str,
            chunk_duration_s: float = 0.5,
            n_fft: int = 1024,
            hop_length: int = 256,
            preprocessing_stats: Optional[dict] = None,
            prediction_mode: bool = False,
            prediction_horizon_s: float = 0.2,
            input_signals: Optional[list[str]] = None,
            target_signals: Optional[list[str]] = None,
    ):
        # Make instance-level copies to avoid class-level mutation
        self.signal_configs = copy.deepcopy(self.SIGNAL_CONFIGS)
        self.movie_configs = copy.deepcopy(self.MOVIE_CONFIGS)

        self.hdf5_path = Path(hdf5_path)
        self.chunk_duration_s = chunk_duration_s
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preprocessing_stats = preprocessing_stats or {}

        # Prediction settings
        self.prediction_mode = prediction_mode
        self.prediction_horizon_s = prediction_horizon_s
        self.input_signals = input_signals or ["ece", "co2", "mhr"]
        self.target_signals = target_signals or ["d_alpha", "mse", "ts_core_density"]

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        self._update_preprocessing_stats()
        self.h5_file = None
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                self.duration, self.t0_indices = self._compute_duration_and_t0_indices(f)
        except OSError as e:
            print(self.hdf5_path)
            raise e
        # In prediction mode, reduce length to ensure extended window fits
        if self.prediction_mode:
            total_window = self.chunk_duration_s + self.prediction_horizon_s
            max_time = self.duration - total_window
            self.length = max(1, int(np.floor(max_time / self.chunk_duration_s)))
        else:
            self.length = max(1, int(np.ceil(self.duration / self.chunk_duration_s)))

        self.n_freq_bins = n_fft // 2 + 1
        self.stft_window = torch.hann_window(n_fft)

    def _find_t0_index(self, xdata_ms: np.ndarray) -> tuple[int, float]:
        """
        Find the index and exact time of t=0 in xdata.

        Parameters
        ----------
        xdata_ms : np.ndarray
            Array of timestamps in milliseconds

        Returns
        -------
        tuple[int, float]
            (index, actual_time_ms) where:
            - index: Index closest to t=0, or -1 if all data is before t=0
            - actual_time_ms: The actual timestamp at that index
        """
        if len(xdata_ms) == 0:
            return -1, 0.0

        if len(xdata_ms) == 1:
            # Single sample - use it if >= 0, else -1
            if xdata_ms[0] >= 0:
                return 0, xdata_ms[0]
            else:
                return -1, xdata_ms[0]

        # All data before t=0
        if xdata_ms[-1] < 0:
            return -1, xdata_ms[-1]

        # All data after t=0 (first sample is already past t=0)
        if xdata_ms[0] > 0:
            return 0, xdata_ms[0]

        # t=0 is within range - find nearest index using binary search
        idx = np.searchsorted(xdata_ms, 0)

        # searchsorted returns insertion point
        # Check if previous index is closer to 0
        if idx > 0 and idx < len(xdata_ms):
            if abs(xdata_ms[idx - 1]) < abs(xdata_ms[idx]):
                idx = idx - 1
        elif idx >= len(xdata_ms):
            idx = len(xdata_ms) - 1

        return idx, xdata_ms[idx]

    def _compute_duration_and_t0_indices(self, f: h5py.File) -> tuple[float, dict]:
        """
        Compute duration from t=0 and store info about where t=0 occurs for each signal.

        Returns
        -------
        tuple[float, dict]
            (max_duration_from_t0, {signal_name: {'index': int, 'time_s': float}})
            where:
            - 'index': first index where xdata >= 0
            - 'time_s': actual time value (in seconds) at that index
        """
        max_duration = 0.0
        t0_indices = {}

        # Process signals
        for config in self.signal_configs:
            for key_path in config.hdf5_keys:
                try:
                    parts = key_path.split("/")
                    curr = f
                    for part in parts:
                        curr = curr[part]

                    xdata_ms = curr["xdata"][:]

                    if len(xdata_ms) < 2:
                        continue

                    # Find first index where t >= 0
                    t0_idx = np.searchsorted(xdata_ms, 0, side="left")

                    # If all data is before t=0, skip
                    if t0_idx >= len(xdata_ms):
                        continue

                    # Store both index and actual time at that index
                    t0_indices[config.name] = {
                        "index": int(t0_idx),
                        "time_s": float(xdata_ms[t0_idx]) * self.VALUE_CONFIG.ms_to_s,
                    }

                    # Duration from t=0 to end
                    duration_s = (xdata_ms[-1] - 0.0) * self.VALUE_CONFIG.ms_to_s
                    max_duration = max(max_duration, duration_s)

                    break

                except (KeyError, ValueError):
                    continue

        # Process movies
        for movie_config in self.movie_configs:
            for key_path in movie_config.hdf5_keys:
                try:
                    parts = key_path.split("/")
                    curr = f
                    for part in parts:
                        curr = curr[part]

                    xdata_ms = curr["xdata"][:]

                    if len(xdata_ms) < 2:
                        continue

                    t0_idx = np.searchsorted(xdata_ms, 0, side="left")

                    if t0_idx >= len(xdata_ms):
                        continue

                    t0_indices[movie_config.name] = {
                        "index": int(t0_idx),
                        "time_s": float(xdata_ms[t0_idx]) * self.VALUE_CONFIG.ms_to_s,
                    }

                    duration_s = (xdata_ms[-1] - 0.0) * self.VALUE_CONFIG.ms_to_s
                    max_duration = max(max_duration, duration_s)

                    break

                except (KeyError, ValueError):
                    continue

        return max(max_duration, 1.0), t0_indices

    def _update_preprocessing_stats(self):
        """Update preprocessing configs with loaded statistics."""
        for config in self.signal_configs:
            if config.name in self.preprocessing_stats:
                stats = self.preprocessing_stats[config.name]
                if "mean" in stats:
                    config.preprocess.mean = stats["mean"]
                if "std" in stats:
                    config.preprocess.std = stats["std"]
                if "min_val" in stats:
                    config.preprocess.min_val = stats["min_val"]
                if "max_val" in stats:
                    config.preprocess.max_val = stats["max_val"]

    def _apply_preprocessing(
        self, tensor: torch.Tensor, config: PreprocessConfig
    ) -> torch.Tensor:
        """Apply preprocessing transformation.

        Args:
            tensor: Can be:
                - Spectrogram: (channels, freq_bins, time_frames)
                - Timeseries: (channels, 1, time_frames)
        """
        if config.method == "none":
            return tensor

        # Determine how to reshape statistics based on tensor dimensions
        # For (C, F, T) spectrograms, we want (C, 1, 1) for per-channel stats
        # For (C, 1, T) timeseries, we want (C, 1, 1) for per-channel stats
        if tensor.ndim == 3:
            # Reshape to (channels, 1, 1) for proper broadcasting
            reshape_dims = (tensor.shape[0], 1, 1)
        elif tensor.ndim == 2:
            # Reshape to (channels, 1)
            reshape_dims = (tensor.shape[0], 1)
        else:
            reshape_dims = None

        if config.method == "standardize":
            if config.mean is None or config.std is None:
                print("Warning: standardize requested but no statistics provided")
                return tensor

            # Convert to tensor and reshape for broadcasting
            mean = torch.as_tensor(
                config.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.as_tensor(
                config.std, dtype=tensor.dtype, device=tensor.device)

            if reshape_dims is not None:
                mean = mean.reshape(reshape_dims)
                std = std.reshape(reshape_dims)

            return (tensor - mean) / (std + config.eps)

        elif config.method == "normalize":
            if config.min_val is None or config.max_val is None:
                print("Warning: normalize requested but no statistics provided")
                return tensor

            min_val = torch.tensor(
                config.min_val, dtype=tensor.dtype, device=tensor.device
            )
            max_val = torch.tensor(
                config.max_val, dtype=tensor.dtype, device=tensor.device
            )

            # These are scalars, no reshape needed
            return (tensor - min_val) / (max_val - min_val + config.eps)

        elif config.method == "log_standardize":
            tensor_log = torch.log10(tensor + 1)

            if config.mean is None or config.std is None:
                print("Warning: log_standardize requested but no statistics provided")
                return tensor_log

            # Convert to tensor and reshape for broadcasting
            mean = torch.as_tensor(
                config.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.as_tensor(
                config.std, dtype=tensor.dtype, device=tensor.device)

            if reshape_dims is not None:
                mean = mean.reshape(reshape_dims)
                std = std.reshape(reshape_dims)

            return (tensor_log - mean) / (std + config.eps)

        elif config.method == "log":
            tensor_log = torch.log10(tensor + 1)
            return tensor_log

        return tensor

    def _open_hdf5(self):
        """Open HDF5 file for this worker with optimized cache settings."""
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.hdf5_path,
                "r",
                rdcc_nbytes=self.VALUE_CONFIG.rdcc_nbytes,
                rdcc_nslots=self.VALUE_CONFIG.rdcc_nslots,
            )

    def _load_signal_raw(
            self,
            f: h5py.File,
            config: SignalConfig,
            t_start: float,
            t_end: float
    ) -> torch.Tensor:
        """
        Load raw signal at native sampling rate within time window.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle
        config : SignalConfig
            Signal configuration
        t_start : float
            Start time in seconds (relative to t=0)
        t_end : float
            End time in seconds (relative to t=0)

        Returns
        -------
        torch.Tensor
            Array of shape (time_samples, channels) at native sampling rate
        """
        duration_s = t_end - t_start

        # Find the signal in HDF5
        data_group = None
        for key_path in config.hdf5_keys:
            try:
                parts = key_path.split("/")
                curr = f
                for part in parts:
                    curr = curr[part]
                data_group = curr
                break
            except KeyError:
                continue

        if data_group is None:
            return torch.zeros(
                (round(duration_s * config.target_fs), config.num_channels)
            )

        ydata_ds = data_group["ydata"]
        xdata_ds = data_group["xdata"]

        # Get time range and sample count
        xdata_start_s = xdata_ds[0] / 1000.0
        xdata_end_s = xdata_ds[-1] / 1000.0
        n_samples = xdata_ds.shape[0]

        if n_samples < 2 or xdata_end_s == xdata_start_s:
            return torch.zeros(
                (round(duration_s * config.target_fs), config.num_channels)
            )

        # Compute actual sampling frequency from the data
        actual_fs = (n_samples - 1) / (xdata_end_s - xdata_start_s)

        # Step 1: Initialize output array with zeros
        output = np.zeros(
            (round(duration_s * actual_fs), config.num_channels),
            dtype=np.float32
        )

        # Step 2: Calculate which HDF5 indices correspond to [t_start, t_end]
        # xdata[i] = xdata_start_s + i / actual_fs
        # Solving for i: i = (t - xdata_start_s) * actual_fs
        hdf5_start = round((t_start - xdata_start_s) * actual_fs)
        hdf5_end = round((t_end - xdata_start_s) * actual_fs)

        # Clamp to valid HDF5 range [0, n_samples]
        hdf5_start_clamped = max(0, min(hdf5_start, n_samples))
        hdf5_end_clamped = max(0, min(hdf5_end, n_samples))

        # Step 3: Load data if there's any overlap
        if hdf5_start_clamped < hdf5_end_clamped:
            data = ydata_ds[hdf5_start_clamped:hdf5_end_clamped]
            np.nan_to_num(data, copy=False, nan=0.0)

            # Step 4: Calculate where to insert in output array
            # The loaded data starts at time: xdata_start_s + hdf5_start_clamped / actual_fs
            # This corresponds to output index: (that_time - t_start) * actual_fs
            output_start = hdf5_start_clamped - hdf5_start
            output_end = output_start + data.shape[0]

            # Clamp to output bounds
            src_start = 0
            src_end = data.shape[0]

            if output_start < 0:
                src_start = -output_start
                output_start = 0
            if output_end > output.shape[0]:
                src_end -= output_end - output.shape[0]
                output_end = output.shape[0]

            # Insert data into output
            if src_start < src_end and output_start < output_end:
                if data.shape[1] == config.num_channels:
                    output[output_start:output_end] = data[src_start:src_end]
                elif data.shape[1] > config.num_channels:
                    output[output_start:output_end] = data[src_start:src_end, :config.num_channels]
                else:
                    output[output_start:output_end, :data.shape[1]] = data[src_start:src_end]

        # Step 6: Convert to tensor and resample to target frequency
        tensor = torch.from_numpy(output).float()

        tensor = (
            F.interpolate(
                tensor.unsqueeze(0).permute(0, 2, 1),
                size=round(duration_s * config.target_fs),
                mode="linear",
                align_corners=False,
            )
            .permute(0, 2, 1)
            .squeeze(0)
        )

        return tensor

    def _compute_stft(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude spectrogram.

        Args:
            signal: (channels, time_samples) at native sampling rate

        Returns:
            Magnitude spectrogram (channels, freq_bins, time_frames)
        """
        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window,
            return_complex=True,
        )
        spec = spec[:, 1:, :]  # Remove DC component (extreme values)
        return torch.abs(spec)

    def _load_metadata(self, f: h5py.File) -> dict:
        """Load text data."""
        metadata = {}

        # Text
        if "log" in f:
            raw_log = f["log"]["data"][()]
        else:
            raw_log = ""

        metadata["text"] = (
            raw_log.decode("utf-8") if isinstance(raw_log, bytes) else raw_log
        )

        return metadata

    def __len__(self):
        return self.length

    def __getstate__(self):
        """Prepare state for pickling - exclude HDF5 file handle."""
        state = self.__dict__.copy()
        state['h5_file'] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)

    def _process_signal(
        self, data: torch.Tensor, config: SignalConfig
    ) -> torch.Tensor:
        """Process signal for extended window (input + prediction horizon).

        Args:
            data: Raw signal data
            config: Signal configuration

        Returns:
            STFT signals: (channels, freq_bins, extended_frames)
            Non-STFT signals: (channels, 1, extended_frames)
        """
        # Step 1: Convert to torch and transpose to (channels, time)
        tensor = data.T

        # Step 2: Process (STFT or nothing)
        if config.apply_stft:
            processed = self._compute_stft(tensor)
        else:
            processed = tensor

        # Step 3: Apply preprocessing
        processed = self._apply_preprocessing(processed, config.preprocess)

        return processed

    def _load_movie_raw(
            self,
            f: h5py.File,
            config: MovieConfig,
            t_start: float,
            t_end: float
    ) -> torch.Tensor:
        """Load raw movie data without resampling (for prediction mode).

        Returns:
            Raw movie array at native frame rate, shape (time, height, width)
        """
        duration_s = t_end - t_start

        # Find the movie in HDF5
        data_group = None
        for key_path in config.hdf5_keys:
            try:
                parts = key_path.split("/")
                curr = f
                for part in parts:
                    curr = curr[part]
                data_group = curr
                break
            except KeyError:
                continue

        ydata_ds = data_group["ydata"]
        xdata_ds = data_group["xdata"]

        if ydata_ds.size == 0:
            return torch.zeros(
                (round(duration_s * config.target_fps), config.height, config.width)
            )

        # Get time range and frame count
        xdata_start_s = xdata_ds[0] / 1000.0
        xdata_end_s = xdata_ds[-1] / 1000.0
        n_frames = xdata_ds.shape[0]

        if n_frames < 2 or xdata_end_s == xdata_start_s:
            return torch.zeros(
                (round(duration_s * config.target_fps), config.height, config.width)
            )

        # Compute actual frame rate from the data
        actual_fps = (n_frames - 1) / (xdata_end_s - xdata_start_s)

        # Get actual dimensions from data
        raw_height, raw_width = ydata_ds.shape[1], ydata_ds.shape[2]

        # Step 1: Initialize output array with zeros at actual fps
        output = np.zeros(
            (round(duration_s * actual_fps), raw_height, raw_width),
            dtype=np.float32
        )

        # Step 2: Calculate which HDF5 indices correspond to [t_start, t_end]
        # xdata[i] = xdata_start_s + i / actual_fps
        # Solving for i: i = (t - xdata_start_s) * actual_fps
        hdf5_start = round((t_start - xdata_start_s) * actual_fps)
        hdf5_end = round((t_end - xdata_start_s) * actual_fps)

        # Clamp to valid HDF5 range [0, n_frames]
        hdf5_start_clamped = max(0, min(hdf5_start, n_frames))
        hdf5_end_clamped = max(0, min(hdf5_end, n_frames))

        # Step 3: Load data if there's any overlap
        if hdf5_start_clamped < hdf5_end_clamped:
            data = ydata_ds[hdf5_start_clamped:hdf5_end_clamped]
            data[np.isnan(data)] = 0.0

            # Step 4: Calculate where to insert in output array
            # The loaded data starts at time: xdata_start_s + hdf5_start_clamped / actual_fps
            # This corresponds to output index: (that_time - t_start) * actual_fps
            output_start = hdf5_start_clamped - hdf5_start
            output_end = output_start + data.shape[0]

            # Clamp to output bounds
            src_start = 0
            src_end = data.shape[0]

            if output_start < 0:
                src_start = -output_start
                output_start = 0
            if output_end > output.shape[0]:
                src_end -= output_end - output.shape[0]
                output_end = output.shape[0]

            # Insert data into output
            if src_start < src_end and output_start < output_end:
                output[output_start:output_end] = data[src_start:src_end]

        # Step 5: Convert to tensor and resample to target fps and dimensions
        tensor = torch.from_numpy(output).float()

        # Resample using trilinear interpolation
        # Input: (time, height, width) → add batch and channel dims
        # Output: (batch=1, channels=1, time, height, width)
        tensor = (
            F.interpolate(tensor.unsqueeze(0).unsqueeze(0),
                          size=(round(duration_s * config.target_fps),
                                config.height,
                                config.width,
                                ),
                          mode="trilinear",
                          align_corners=False,
                          ).squeeze(0).squeeze(0)
        )

        return tensor

    def __getitem__(self, idx):
        self._open_hdf5()

        if self.prediction_mode:
            return self._getitem_prediction(idx)
        else:
            return self._getitem_standard(idx)

    def _getitem_standard(self, idx):
        """Original __getitem__ logic."""
        t_start = idx * self.chunk_duration_s
        t_end = t_start + self.chunk_duration_s

        # Load and process all signals
        all_signals = {}
        for config in self.signal_configs:
            if config.name in self.input_signals:
                raw_data = self._load_signal_raw(self.h5_file, config, t_start, t_end)
                all_signals[config.name] = self._process_signal(raw_data, config)

        # Load and process movies
        all_movies = {}
        for movie_config in self.movie_configs:
            if movie_config.name in self.input_signals:
                raw_movie = self._load_movie_raw(
                    self.h5_file, movie_config, t_start, t_end
                )
                all_movies[movie_config.name] = raw_movie

        # Load metadata
        if "text" in self.input_signals:
            all_metadata = self._load_metadata(self.h5_file)
        else:
            all_metadata = {}

        return {**all_signals, **all_movies, **all_metadata}

    def _getitem_prediction(self, idx):
        """Load extended window, process jointly, then split into input/target."""
        # Extended window: from t to t + chunk_duration + prediction_horizon
        t_start = idx * self.chunk_duration_s
        t_end = t_start + self.chunk_duration_s + self.prediction_horizon_s

        signals_to_load = set(self.input_signals) | set(self.target_signals)

        # Load and process all signals with extended window
        all_signals = {}
        for config in self.signal_configs:
            if config.name not in signals_to_load:
                continue
            raw_data = self._load_signal_raw(self.h5_file, config, t_start, t_end)
            all_signals[config.name] = self._process_signal(raw_data, config)

        # Load and process movies
        all_movies = {}
        for movie_config in self.movie_configs:
            if movie_config.name not in signals_to_load:
                continue
            # Load raw movie data
            raw_movie = self._load_movie_raw(self.h5_file, movie_config, t_start, t_end)
            all_movies[movie_config.name] = raw_movie

        # Load metadata
        all_metadata = self._load_metadata(self.h5_file)

        # Split into inputs and targets
        inputs = {}
        targets = {}

        # For signals: split at input_frames
        for config in self.signal_configs:
            if config.name not in signals_to_load:
                continue
            signal = all_signals[config.name]

            if config.apply_stft:
                n_training_frames = round(
                    self.chunk_duration_s * config.target_fs / self.hop_length
                )
            else:
                n_training_frames = round(self.chunk_duration_s * config.target_fs)

            if config.name in self.input_signals:
                inputs[config.name] = signal[..., :n_training_frames]

            if config.name in self.target_signals:
                targets[config.name] = signal[..., n_training_frames:]

        # Movies: split along time dimension
        for movie_config in self.movie_configs:
            if movie_config.name not in signals_to_load:
                continue
            movie_name = movie_config.name
            movie_data = all_movies[movie_name]
            n_training_frames = round(self.chunk_duration_s * movie_config.target_fps)
            # movie_data shape: (extended_movie_frames, height, width)
            if movie_name in self.input_signals:
                inputs[movie_name] = movie_data[:n_training_frames]

            # Include movies in targets if specified
            if movie_name in self.target_signals:
                targets[movie_name] = movie_data[n_training_frames:]

        # Metadata (text) only goes to inputs
        if "text" in self.input_signals:
            inputs.update(all_metadata)

        return {"inputs": inputs, "targets": targets}

    def __del__(self):
        """Close file when dataset is deleted."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass


def collate_fn(batch):
    """Custom collate function for batching."""
    elem = batch[0]

    # Check if prediction mode (has 'inputs' and 'targets' keys)
    if "inputs" in elem and "targets" in elem:
        return collate_fn_prediction(batch)

    # Standard mode
    collated = {}
    for key in elem:
        if key == "text":
            collated[key] = [d[key] for d in batch]
        else:
            collated[key] = torch.stack([d[key] for d in batch])
    return collated


def collate_fn_prediction(batch):
    """Collate function for prediction mode."""
    inputs_batch = []
    targets_batch = []

    for item in batch:
        inputs_batch.append(item["inputs"])
        targets_batch.append(item["targets"])

    # Collate inputs
    inputs_collated = {}
    for key in inputs_batch[0]:
        if key == "text":
            inputs_collated[key] = [d[key] for d in inputs_batch]
        else:
            inputs_collated[key] = torch.stack([d[key] for d in inputs_batch])

    # Collate targets
    targets_collated = {}
    for key in targets_batch[0]:
        targets_collated[key] = torch.stack([d[key] for d in targets_batch])

    return {"inputs": inputs_collated, "targets": targets_collated}