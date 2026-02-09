import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F


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
        print(f"\nComputing statistics for {config.name}...")

        # Collect values
        values = []
        indices = torch.randperm(len(combined))[:num_samples]

        for idx in tqdm(indices):
            batch = combined[int(idx)]
            if config.name in batch:
                values.append(batch[config.name])

        if not values:
            continue

        # Stack and compute statistics
        all_values = torch.stack(values)

        # Compute per-channel statistics
        # Reduce over all dimensions except channel dimension (dim=1)
        dims_to_reduce = list(range(all_values.ndim))
        dims_to_reduce.remove(1)  # Keep channel dimension

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
    print(f"\nSaved statistics to {output_path}")
    return stats


@dataclass
class SignalConfig:
    """Configuration for a single signal/diagnostic."""

    name: str  # Key in output dict
    hdf5_keys: list[str]  # Possible HDF5 paths to search
    num_channels: int  # Number of channels
    apply_stft: bool  # Whether to compute STFT
    target_time_frames: int  # Target number of time frames after processing


@dataclass
class MovieConfig:
    """Configuration for a movie/video diagnostic."""

    name: str  # Key in output dict
    hdf5_keys: list[str]  # Possible HDF5 paths to search
    target_frames: int  # Target number of frames
    channels: int  # Color channels (e.g., 3 for RGB)
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
    apply_stft: bool
    target_time_frames: int
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
            apply_stft=True,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "ece",
            ["ece"],
            48,
            apply_stft=True,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "co2",
            ["co2"],
            4,
            apply_stft=True,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "d_alpha",
            ["dalpha"],
            6,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "gas",
            ["gas"],
            5,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "ech",
            ["ech"],
            11,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "pin",
            ["pin"],
            8,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "tin",
            ["tin"],
            8,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "mse",
            ["mse"],
            69,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "ts_core_density",
            ["ts_core_density"],
            44,
            apply_stft=False,
            target_time_frames=100,
            preprocess=PreprocessConfig(method="none"),
        ),
    ]

    MOVIE_CONFIGS = [
        MovieConfig("bolo", ["bolo"], 10, 3, 64, 64),
        MovieConfig("irtv", ["irtv"], 10, 3, 64, 64),
        MovieConfig("tangtv", ["tangtv"], 10, 3, 64, 64),
    ]

    def __init__(
        self,
        hdf5_path: str,
        chunk_duration_s: float = 0.2,
        n_fft: int = 512,
        hop_length: int = 256,
        preprocessing_stats: Optional[dict] = None,
        prediction_mode: bool = True,
        prediction_horizon_s: float = 0.1,
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
            mean = torch.tensor(config.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.tensor(config.std, dtype=tensor.dtype, device=tensor.device)

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
            tensor_log = torch.log(tensor + 1)

            if config.mean is None or config.std is None:
                print("Warning: log_standardize requested but no statistics provided")
                return tensor_log

            # Convert to tensor and reshape for broadcasting
            mean = torch.tensor(config.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.tensor(config.std, dtype=tensor.dtype, device=tensor.device)

            if reshape_dims is not None:
                mean = mean.reshape(reshape_dims)
                std = std.reshape(reshape_dims)

            return (tensor_log - mean) / (std + config.eps)

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
    ) -> np.ndarray:
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

        # Handle missing data
        if data_group is None:
            return np.zeros((0, config.num_channels), dtype=np.float32)

        # Extract data with time slicing
        ydata = None
        if isinstance(data_group, h5py.Group) and "ydata" in data_group:
            ydata_ds = data_group["ydata"]
            if "xdata" in data_group:
                xdata = data_group["xdata"][:] / 1000.0  # Convert to seconds
                mask = (xdata >= t_start) & (xdata < t_end)
                if np.any(mask):
                    ydata = ydata_ds[mask]
                    ydata[np.isnan(ydata)] = 0.0
                else:
                    ydata = np.zeros((0,) + ydata_ds.shape[1:], dtype=np.float32)
            else:
                ydata = ydata_ds[:]
        elif isinstance(data_group, h5py.Dataset):
            ydata = data_group[:]

        if ydata is None:
            return np.zeros((0, config.num_channels), dtype=np.float32)

        # Ensure 2D shape (time, channels)
        if ydata.ndim == 1:
            ydata = ydata[:, np.newaxis]

        return ydata.astype(np.float32)

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
        return torch.abs(spec)

    def _resample_spectrogram(
        self, spec: torch.Tensor, target_time_frames: int
    ) -> torch.Tensor:
        """Resample spectrogram to target number of time frames.

        Args:
            spec: (channels, freq_bins, time_frames)
            target_time_frames: Target number of time frames

        Returns:
            Resampled spectrogram (channels, freq_bins, target_time_frames)
        """
        if spec.shape[2] == target_time_frames:
            return spec

        # Use 2D interpolation (freq, time)
        # Add batch dimension: (1, channels, freq, time)
        spec = spec.unsqueeze(0)

        # Interpolate
        resampled = F.interpolate(
            spec,
            size=(spec.shape[2], target_time_frames),  # Keep freq_bins, resample time
            mode="bilinear",
            align_corners=False,
        )

        return resampled.squeeze(0)

    def _resample_timeseries(
        self, signal: torch.Tensor, target_time_frames: int
    ) -> torch.Tensor:
        """Resample time series to target number of time frames.

        Args:
            signal: (channels, time_samples)
            target_time_frames: Target number of time frames

        Returns:
            Resampled signal (channels, 1, target_time_frames) with singleton freq dim
        """
        if signal.shape[1] == target_time_frames:
            return signal.unsqueeze(1)

        # Resample along time dimension
        resampled = F.interpolate(
            signal.unsqueeze(0),
            size=target_time_frames,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        # Add singleton frequency dimension
        return resampled.unsqueeze(1)  # (channels, 1, target_time_frames)

    def _load_movie(
        self, f: h5py.File, config: MovieConfig, t_start: float, t_end: float
    ) -> np.ndarray:
        """Load and resample a single movie from HDF5."""
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

        # Handle missing data
        if data_group is None:
            fallback_shape = (
                config.target_frames,
                config.channels,
                config.height,
                config.width,
            )
            return np.zeros(fallback_shape, dtype=np.uint8)

        # Extract data with time slicing
        ydata = None
        if isinstance(data_group, h5py.Group) and "ydata" in data_group:
            ydata_ds = data_group["ydata"]
            if "xdata" in data_group:
                xdata = data_group["xdata"][:] / 1000.0
                mask = (xdata >= t_start) & (xdata < t_end)
                if np.any(mask):
                    ydata = ydata_ds[mask]
                else:
                    ydata = np.zeros((0,) + ydata_ds.shape[1:], dtype=np.uint8)
            else:
                ydata = ydata_ds[:]
        elif isinstance(data_group, h5py.Dataset):
            ydata = data_group[:]

        if ydata is None or len(ydata) == 0:
            fallback_shape = (
                config.target_frames,
                config.channels,
                config.height,
                config.width,
            )
            return np.zeros(fallback_shape, dtype=np.uint8)

        # Resample video to target number of frames
        return self._resample_video(ydata, config.target_frames)

    def _resample_video(self, data: np.ndarray, target_frames: int) -> np.ndarray:
        """Resample video data along time dimension."""
        if len(data) == 0:
            return np.zeros((target_frames,) + data.shape[1:], dtype=data.dtype)

        tensor = torch.from_numpy(data).float()

        # Handle different possible input formats
        if tensor.ndim == 4:
            # Check if channels are last or second dimension
            if tensor.shape[-1] in [1, 3, 4]:
                # (time, height, width, channels) -> (time, channels, height, width)
                tensor = tensor.permute(0, 3, 1, 2)

            # Now tensor is (time, channels, height, width)
            T, C, H, W = tensor.shape

            # Reshape to (channels, time, height, width) for interpolation
            tensor = tensor.permute(1, 0, 2, 3)  # (C, T, H, W)

            # Add batch dimension: (1, C, T, H, W)
            tensor = tensor.unsqueeze(0)

            # Interpolate along time dimension
            out = F.interpolate(
                tensor,
                size=(target_frames, H, W),
                mode="trilinear",
                align_corners=False,
            )

            # Remove batch dimension and permute back: (C, T, H, W) -> (T, C, H, W)
            out = out.squeeze(0).permute(1, 0, 2, 3)

            return out.numpy().astype(data.dtype)

        elif tensor.ndim == 3:
            # (time, height, width) - grayscale video
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
            out = F.interpolate(
                tensor,
                size=(target_frames, *tensor.shape[3:]),
                mode="trilinear",
                align_corners=False,
            )
            return out.squeeze(0).squeeze(0).numpy().astype(data.dtype)

        return data

    def _load_movies(self, f: h5py.File, t_start: float, t_end: float) -> dict:
        """Load all movie data."""
        movies = {}
        for config in self.MOVIE_CONFIGS:
            data = self._load_movie(f, config, t_start, t_end)
            movies[config.name] = torch.from_numpy(data).float()

        return movies

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

    def _process_signal_extended(
        self, data: np.ndarray, config: SignalConfig, extended_frames: int
    ) -> torch.Tensor:
        """Process signal for extended window (input + prediction horizon).

        Args:
            data: Raw signal data
            config: Signal configuration
            extended_frames: Total frames for input+target window

        Returns:
            STFT signals: (channels, freq_bins, extended_frames)
            Non-STFT signals: (channels, 1, extended_frames)
        """
        # Handle empty data
        if len(data) == 0:
            if config.apply_stft:
                return torch.zeros(
                    config.num_channels, self.n_freq_bins, extended_frames
                )
            else:
                return torch.zeros(config.num_channels, 1, extended_frames)

        # Step 1: Convert to torch and transpose to (channels, time)
        tensor = torch.from_numpy(data).float().T

        # Step 2: Process (STFT or nothing)
        if config.apply_stft:
            processed = self._compute_stft(tensor)
            processed = self._resample_spectrogram(processed, extended_frames)
        else:
            processed = self._resample_timeseries(tensor, extended_frames)

        # Step 3: Apply preprocessing
        processed = self._apply_preprocessing(processed, config.preprocess)

        return processed

    def _load_movie_raw(
        self, f: h5py.File, config: MovieConfig, t_start: float, t_end: float
    ) -> np.ndarray:
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

        # Handle missing data
        if data_group is None:
            return np.zeros((0, config.height, config.width), dtype=np.uint8)

        # Extract data with time slicing
        ydata = None
        if isinstance(data_group, h5py.Group) and "ydata" in data_group:
            ydata_ds = data_group["ydata"]
            if "xdata" in data_group:
                xdata = data_group["xdata"][:] / 1000.0
                mask = (xdata >= t_start) & (xdata < t_end)
                if np.any(mask):
                    ydata = ydata_ds[mask]
                else:
                    ydata = np.zeros((0, config.height, config.width), dtype=np.uint8)
            else:
                ydata = ydata_ds[:]
        elif isinstance(data_group, h5py.Dataset):
            ydata = data_group[:]

        if ydata is None or len(ydata) == 0:
            return np.zeros((0, config.height, config.width))

        # Ensure shape is (time, height, width) for grayscale
        if ydata.ndim == 4:
            # (time, height, width, channels) -> take first channel
            if ydata.shape[-1] in [1, 3, 4]:
                ydata = ydata[..., 0]  # Take first channel
            elif ydata.shape[1] in [1, 3, 4]:
                # (time, channels, height, width) -> take first channel
                ydata = ydata[:, 0, :, :]

        # Should now be (time, height, width)
        return ydata

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
        signals = {}
        for config in self.SIGNAL_CONFIGS:
            raw_data = self._load_signal_raw(self.h5_file, config, t_start, t_end)
            signals[config.name] = self._process_signal(raw_data, config)

        # Load movies and metadata
        movies = self._load_movies(self.h5_file, t_start, t_end)
        metadata = self._load_metadata(self.h5_file)

        return {**signals, **movies, **metadata}

    def _getitem_prediction(self, idx):
        """Load extended window, process jointly, then split into input/target."""
        # Extended window: from t to t + chunk_duration + prediction_horizon
        t_start = idx * self.chunk_duration_s
        t_end = t_start + self.chunk_duration_s + self.prediction_horizon_s

        # Calculate extended number of frames based on time duration
        total_duration = self.chunk_duration_s + self.prediction_horizon_s
        base_frames = self.SIGNAL_CONFIGS[0].target_time_frames

        # Extended frames = base_frames scaled by total duration
        extended_frames = int(base_frames * total_duration / self.chunk_duration_s)

        # Calculate split point based on time proportion
        # Input gets chunk_duration_s proportion, target gets prediction_horizon_s proportion
        input_frames = int(extended_frames * self.chunk_duration_s / total_duration)
        target_frames = extended_frames - input_frames

        print(f"Debug: total_duration={total_duration}, base_frames={base_frames}")
        print(
            f"Debug: extended_frames={extended_frames}, input_frames={input_frames}, target_frames={target_frames}"
        )

        # Load and process all signals with extended window
        all_signals = {}
        for config in self.SIGNAL_CONFIGS:
            raw_data = self._load_signal_raw(self.h5_file, config, t_start, t_end)
            # Process with extended frames
            all_signals[config.name] = self._process_signal_extended(
                raw_data, config, extended_frames
            )

        # Load and process movies with extended window
        all_movies = {}
        for movie_config in self.MOVIE_CONFIGS:
            # Load raw movie data for extended window
            raw_movie = self._load_movie_raw(self.h5_file, movie_config, t_start, t_end)

            # Calculate extended movie frames
            extended_movie_frames = int(
                movie_config.target_frames * total_duration / self.chunk_duration_s
            )

            # Resample to extended frames
            resampled_movie = self._resample_video(raw_movie, extended_movie_frames)
            all_movies[movie_config.name] = torch.from_numpy(resampled_movie).float()

        # Load metadata
        all_metadata = self._load_metadata(self.h5_file)

        # Split into inputs and targets
        inputs = {}
        targets = {}

        # For signals: split at input_frames
        for config in self.SIGNAL_CONFIGS:
            signal = all_signals[config.name]
            # signal shape: (channels, freq_bins, extended_frames) or (channels, 1, extended_frames)
            inputs[config.name] = signal[..., :input_frames]

            # Only save targets for specified signals
            if config.name in self.target_signals:
                targets[config.name] = signal[..., input_frames:]

        # Movies: split along time dimension
        for movie_config in self.MOVIE_CONFIGS:
            movie_name = movie_config.name
            movie_data = all_movies[movie_name]

            # Calculate split point for movies based on same time proportion
            extended_movie_frames = int(
                movie_config.target_frames * total_duration / self.chunk_duration_s
            )
            movie_input_frames = int(
                extended_movie_frames * self.chunk_duration_s / total_duration
            )

            # movie_data shape: (extended_movie_frames, height, width)
            inputs[movie_name] = movie_data[:movie_input_frames]

            # Include movies in targets if specified
            if movie_name in self.target_signals:
                targets[movie_name] = movie_data[movie_input_frames:]

        # Metadata (text) only goes to inputs
        inputs.update(all_metadata)

        return {"inputs": inputs, "targets": targets}

    def _process_signal(self, data: np.ndarray, config: SignalConfig) -> torch.Tensor:
        """Process signal: load → process → resample (for standard mode).

        Returns:
            STFT signals: (channels, freq_bins, target_time_frames)
            Non-STFT signals: (channels, 1, target_time_frames)
        """
        # Handle empty data
        if len(data) == 0:
            if config.apply_stft:
                return torch.zeros(
                    config.num_channels, self.n_freq_bins, config.target_time_frames
                )
            else:
                return torch.zeros(config.num_channels, 1, config.target_time_frames)

        # Step 1: Convert to torch and transpose to (channels, time)
        tensor = torch.from_numpy(data).float().T

        # Step 2: Process (STFT or nothing)
        if config.apply_stft:
            processed = self._compute_stft(tensor)
            processed = self._resample_spectrogram(processed, config.target_time_frames)
        else:
            processed = self._resample_timeseries(tensor, config.target_time_frames)

        # Step 3: Apply preprocessing
        processed = self._apply_preprocessing(processed, config.preprocess)

        return processed

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
