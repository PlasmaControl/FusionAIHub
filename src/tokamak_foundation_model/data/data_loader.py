import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F

# TODO: implement this for calculation
class Welford:
    def __init__(self):
        self.mean = 0
        self.std = 0
        self.min_val = 0
        self.max_val = 0
        self.n = 0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.std += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def get_stats(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

def compute_preprocessing_stats(
    datasets, output_path="preprocessing_stats.pt", num_samples=1000
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
    stats = {}

    # Get signal names from first dataset
    signal_configs = datasets[0].SIGNAL_CONFIGS

    for config in signal_configs:
        print(f"Computing statistics for {config.name}...")

        # Collect values
        values = []
        indices = torch.randperm(len(combined))[:num_samples]

        for idx in tqdm(indices):
            batch = combined[int(idx)]
            if config.name in batch['inputs']:
                values.append(batch['inputs'][config.name])
                values.append(batch['targets'][config.name])

        if not values:
            continue

        # Stack and compute statistics
        if values[0].ndim == 2:
            all_values = torch.cat(values, dim=1)  # (channels, time)
        elif values[0].ndim == 3:
            all_values = torch.cat(values, dim=2)  # (channels, freq_bins, time)

        # Compute per-channel statistics
        # Reduce over all dimensions except channel dimension (dim=1)
        dims_to_reduce = list(range(all_values.ndim))
        dims_to_reduce.remove(0)  # Keep channel dimension

        mean = all_values.mean(dim=dims_to_reduce)
        std = all_values.std(dim=dims_to_reduce)
        min_val = all_values.min()
        max_val = all_values.max()

        stats[config.name] = {
            "mean": mean,
            "std": std,
            "min_val": min_val.item(),
            "max_val": max_val.item(),
        }

    torch.save(stats, output_path)
    print(f"Saved statistics to {output_path}")
    return stats


@dataclass
class MovieConfig:
    """Configuration for a movie/video diagnostic."""

    name: str  # Key in output dict
    hdf5_keys: list[str]  # Possible HDF5 paths to search
    channels: int  # Color channels (e.g., 3 for RGB)
    target_fps: int  # Target frames per second after resampling
    height: int  # Frame height
    width: int  # Frame width


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
    preprocess: PreprocessConfig = None  # Add preprocessing config

    def __post_init__(self):
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()


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
            "mhr",
            ["mhr"],
            8,
            500e3,
            apply_stft=True,
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "ece",
            ["ece"],
            48,
            500e3,
            apply_stft=True,
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
            preprocess=PreprocessConfig(method="none"),
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

    def __init__(
        self,
        hdf5_path: str,
        chunk_duration_s: float = 0.5,
        n_fft: int = 1024,
        hop_length: int = 256,
        preprocessing_stats: Optional[dict] = None,
        prediction_mode: bool = True,
        prediction_horizon_s: float = 0.2,
        input_signals: Optional[list[str]] = None,
        target_signals: Optional[list[str]] = None,
    ):
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

        with h5py.File(self.hdf5_path, "r") as f:
            self.duration = self._compute_duration_from_handle(f)

        # In prediction mode, reduce length to ensure extended window fits
        if self.prediction_mode:
            total_window = self.chunk_duration_s + self.prediction_horizon_s
            max_time = self.duration - total_window
            self.length = max(1, int(np.floor(max_time / self.chunk_duration_s)))
        else:
            self.length = max(1, int(np.ceil(self.duration / self.chunk_duration_s)))

        self.n_freq_bins = n_fft // 2 + 1
        self.stft_window = torch.hann_window(n_fft)

    def _update_preprocessing_stats(self):
        """Update preprocessing configs with loaded statistics."""
        for config in self.SIGNAL_CONFIGS:
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

    def _compute_duration_from_handle(self, f: h5py.File) -> float:
        """Compute total duration from an open HDF5 file handle."""
        try:
            for key_path in ["mhr/xdata", "ece/xdata", "co2/xdata"]:
                try:
                    parts = key_path.split("/")
                    data = f
                    for part in parts:
                        data = data[part]
                    xdata = data[:]
                    return (xdata[-1] - xdata[0]) / 1000.0
                except (KeyError, ValueError):
                    continue
        except Exception as e:
            print(f"Warning: Could not determine duration from {self.hdf5_path}: {e}")

        return 1.0  # Default fallback

    def _open_hdf5(self):
        """Open HDF5 file for this worker with optimized cache settings."""
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.hdf5_path,
                "r",
                rdcc_nbytes=1024**2 * 256,  # 256 MB chunk cache
                rdcc_nslots=10000,  # Number of chunk slots
            )

    def _load_signal_raw(
        self, f: h5py.File, config: SignalConfig, t_start: float, t_end: float
    ) -> torch.Tensor:
        """Load raw signal at native sampling rate within time window.

        Returns:
            Array of shape (time, channels) at native sampling rate
        """
        # Try to find the signal in HDF5
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

        # Extract data with time slicing
        ydata_ds = data_group["ydata"]
        xdata_ds = data_group["xdata"]

        # Load only first and last timestamp
        t0 = xdata_ds[0] / 1000.0
        t1 = xdata_ds[-1] / 1000.0
        n_samples = xdata_ds.shape[0]

        duration_s = t_end - t_start

        fs_raw = (n_samples - 1) / (t1 - t0)

        ydata = np.zeros(
            (max(1, round(duration_s * fs_raw)), config.num_channels), dtype=np.float32
        )

        start_idx = max(0, int((t_start - t0) * fs_raw))
        end_idx = min(n_samples, int((t_end - t0) * fs_raw))

        if end_idx > start_idx:
            data = ydata_ds[start_idx:end_idx]
            np.nan_to_num(data, copy=False, nan=0.0)

            # Compute offset based on actual start time
            actual_t_start = t0 + start_idx / fs_raw
            idx_1 = round((actual_t_start - t_start) * fs_raw)
            idx_2 = idx_1 + data.shape[0]

            # Clamp to array bounds
            src_start = 0
            src_end = data.shape[0]

            if idx_1 < 0:
                src_start = -idx_1
                idx_1 = 0
            if idx_2 > ydata.shape[0]:
                src_end -= idx_2 - ydata.shape[0]
                idx_2 = ydata.shape[0]

            if (idx_1 == 0 and idx_2 == ydata.shape[0]
                    and src_start == 0 and src_end == data.shape[0]):
                ydata = data  # No copy needed
            else:
                ydata[idx_1:idx_2] = data[src_start:src_end]

        tensor = torch.from_numpy(ydata).float()

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
        spec = spec[:, 1:, :] # Remove DC component (extreme values)
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
        self, f: h5py.File, config: MovieConfig, t_start: float, t_end: float
    ) -> torch.Tensor:
        """Load raw movie data without resampling (for prediction mode).

        Returns:
            Raw movie array at native frame rate, shape (time, height, width)
        """
        # Try to find the movie in HDF5
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
        
        # Extract data with time slicing
        ydata_ds = data_group["ydata"]
        xdata_ds = data_group["xdata"]

        # Load only first and last timestamp
        t0 = xdata_ds[0] / 1000.0
        t1 = xdata_ds[-1] / 1000.0
        n_samples = xdata_ds.shape[0]

        fps_raw = (n_samples - 1) / (t1 - t0)
        duration_s = t_end - t_start

        if n_samples < 2 or t1 == t0:
            n_frames = round(duration_s * config.target_fps)
            return torch.zeros(max(n_frames, 1), config.height, config.width)

        raw_height, raw_width = ydata_ds.shape[1], ydata_ds.shape[2]
        ydata = np.zeros(
            (max(1, round(duration_s * fps_raw)), raw_height, raw_width), dtype=np.float32
        )
        
        # Compute indices directly (no full xdata load)
        start_idx = max(0, int((t_start - t0) * fps_raw))
        end_idx = min(n_samples, int((t_end - t0) * fps_raw))

        if end_idx > start_idx:
            data = ydata_ds[start_idx:end_idx]
            data[np.isnan(data)] = 0.0
            # Compute offset based on actual start time
            actual_t_start = t0 + start_idx / fps_raw
            idx_1 = round((actual_t_start - t_start) * fps_raw)
            idx_2 = idx_1 + data.shape[0]

            # Clamp to array bounds
            src_start = 0
            src_end = data.shape[0]

            if idx_1 < 0:
                src_start = -idx_1
                idx_1 = 0
            if idx_2 > ydata.shape[0]:
                src_end -= idx_2 - ydata.shape[0]
                idx_2 = ydata.shape[0]

            if (idx_1 == 0 and idx_2 == ydata.shape[0] and
                    src_start == 0 and src_end == data.shape[0]):
                ydata = data  # No copy needed
            else:
                ydata[idx_1:idx_2] = data[src_start:src_end]

        tensor = torch.from_numpy(ydata).float()

        tensor = (
            F.interpolate(
                tensor.unsqueeze(0).unsqueeze(0),
                size=(
                    round(duration_s * config.target_fps),
                    config.height,
                    config.width,
                ),
                mode="trilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
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
        for config in self.SIGNAL_CONFIGS:
            if config.name in self.input_signals:
                raw_data = self._load_signal_raw(self.h5_file, config, t_start, t_end)
                all_signals[config.name] = self._process_signal(raw_data, config)

        # Load and process movies
        all_movies = {}
        for movie_config in self.MOVIE_CONFIGS:
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
        for config in self.SIGNAL_CONFIGS:
            if config.name not in signals_to_load:
                continue
            raw_data = self._load_signal_raw(self.h5_file, config, t_start, t_end)
            all_signals[config.name] = self._process_signal(raw_data, config)

        # Load and process movies
        all_movies = {}
        for movie_config in self.MOVIE_CONFIGS:
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
        for config in self.SIGNAL_CONFIGS:
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
        for movie_config in self.MOVIE_CONFIGS:
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
