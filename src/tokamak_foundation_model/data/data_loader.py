import torch
from torch.utils.data import Dataset
import numpy as np
import h5py  # type: ignore
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
import copy


@dataclass
class PreprocessConfig:
    """
    Configuration for a signal preprocessing transformation.

    Specifies which normalisation strategy to apply to a tensor before it is
    fed into the model.  Statistics (*mean*, *std*, *min_val*, *max_val*)
    are populated at runtime from pre-computed dataset statistics (see
    :func:`compute_preprocessing_stats`).

    Parameters
    ----------
    method : str, optional
        Transformation to apply.  One of:

        ``'none'``
            Pass the tensor through unchanged.
        ``'standardize'``
            Zero-mean, unit-variance scaling:
            ``(x - mean) / (std + eps)``.
        ``'normalize'``
            Min-max scaling to ``[0, 1]``:
            ``(x - min_val) / (max_val - min_val + eps)``.
        ``'log_standardize'``
            Apply ``log10(x + 1)``, then standardize.
        ``'log'``
            Apply ``log10(x + 1)`` only.

        Default is ``'none'``.
    mean : float or None, optional
        Per-channel mean used by ``'standardize'`` and
        ``'log_standardize'``.  Default is ``None``.
    std : float or None, optional
        Per-channel standard deviation used by ``'standardize'`` and
        ``'log_standardize'``.  Default is ``None``.
    min_val : float or None, optional
        Per-channel minimum used by ``'normalize'``.  Default is ``None``.
    max_val : float or None, optional
        Per-channel maximum used by ``'normalize'``.  Default is ``None``.
    eps : float, optional
        Small constant added to denominators for numerical stability.
        Default is ``1e-8``.
    """

    method: str = "none"
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    eps: float = 1e-8


@dataclass
class SignalConfig:
    """
    Configuration for a single time-series or spectrogram diagnostic.

    Collects all parameters needed to load, resample, and preprocess one
    modality from an HDF5 file produced by the data-preparation pipeline.

    Parameters
    ----------
    name : str
        Unique identifier for this modality; used as the dictionary key
        in the batch returned by :class:`TokamakH5Dataset`.
    hdf5_keys : list of str
        Ordered list of HDF5 group paths to search for the signal data.
        The first path that exists in the file is used.
    num_channels : int
        Number of output channels after applying *channels_to_use*.  Must
        equal ``len(range(*channels_to_use.indices(N)))`` when
        *channels_to_use* is not ``None``.
    target_fs : float
        Target sampling frequency in Hz.  The raw signal is resampled to
        this rate before being returned.
    apply_stft : bool
        If ``True``, compute an STFT magnitude spectrogram after loading,
        yielding output shape ``(C, F, T)``.  If ``False``, the signal is
        returned as ``(C, T)``.
    channels_to_use : slice or None, optional
        Slice applied to the HDF5 channel axis before writing to the output
        buffer.  ``None`` (default) passes all available channels through,
        truncating or zero-padding to *num_channels* as needed.
    preprocess : PreprocessConfig, optional
        Preprocessing transformation applied after the STFT (or
        pass-through).  Defaults to :class:`PreprocessConfig` with
        ``method='none'``.
    """

    name: str
    hdf5_keys: list[str]
    num_channels: int
    target_fs: float
    apply_stft: bool
    channels_to_use: Optional[slice] = None
    preprocess: PreprocessConfig | None = None
    zero_is_missing: bool = False

    def __post_init__(self):
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()


@dataclass
class MovieConfig:
    """
    Configuration for a video / camera diagnostic.

    Collects all parameters needed to load, resample, and preprocess one
    movie modality from an HDF5 file produced by the data-preparation
    pipeline.

    Parameters
    ----------
    name : str
        Unique identifier for this modality; used as the dictionary key
        in the batch returned by :class:`TokamakH5Dataset`.
    hdf5_keys : list of str
        Ordered list of HDF5 group paths to search for the movie data.
        The first path that exists in the file is used.
    channels : int
        Number of colour channels (e.g. ``1`` for grayscale, ``3`` for
        RGB).
    target_fps : int
        Target frame rate in frames per second.  The raw video is
        resampled to this rate via trilinear interpolation.
    height : int
        Output frame height in pixels after spatial resampling.
    width : int
        Output frame width in pixels after spatial resampling.
    channels_to_use : slice or None, optional
        Slice selecting a subset of channels from the raw data.
        ``None`` (default) uses all channels.
    preprocess : PreprocessConfig, optional
        Preprocessing transformation applied to the video tensor.
        Defaults to :class:`PreprocessConfig` with ``method='none'``.
    """

    name: str  # Key in output dict
    hdf5_keys: list[str]  # Possible HDF5 paths to search
    channels: int  # Color channels (e.g., 3 for RGB)
    target_fps: int  # Target frames per second after resampling
    height: int  # Frame height
    width: int  # Frame width
    channels_to_use: Optional[slice] = None
    preprocess: PreprocessConfig | None = None

    def __post_init__(self):
        if self.preprocess is None:
            self.preprocess = PreprocessConfig()


class TokamakH5Dataset(Dataset):
    """
    PyTorch Dataset for multi-modal tokamak plasma diagnostics stored in HDF5.

    Each item corresponds to a fixed-duration time window (chunk) drawn from a
    single shot file.  The processing pipeline for every chunk is:

    1. Load raw signal / movie data at the native sampling rate from HDF5.
    2. Optionally compute an STFT magnitude spectrogram (signals only).
    3. Resample to the modality's target frequency via linear or trilinear
       interpolation.
    4. Apply the configured preprocessing transformation
       (see :class:`PreprocessConfig`).

    Two operating modes are supported:

    **Standard mode** (``prediction_mode=False``)
        Returns a flat dictionary ``{modality_name: tensor}`` covering the
        half-open interval ``[t_start, t_start + chunk_duration_s)``.

    **Prediction mode** (``prediction_mode=True``)
        Loads an extended window of
        ``chunk_duration_s + prediction_horizon_s`` seconds, processes it
        jointly, then splits into
        ``{"inputs": {…}, "targets": {…}}``.

    Parameters
    ----------
    hdf5_path : str | Path
        Path to a preprocessed HDF5 shot file (output of the
        data-preparation pipeline).
    chunk_duration_s : float, optional
        Duration of each time window in seconds.  Default is ``0.5``.
    max_duration_s : float, optional
        Maximum duration of a shot to be considered.
    n_fft : int, optional
        FFT size used for STFT computation.  Determines the number of
        frequency bins: ``n_fft // 2 + 1``.  Default is ``1024``.
    hop_length : int, optional
        STFT hop size in samples.  Default is ``256``.
    preprocessing_stats : dict or None, optional
        Nested statistics dictionary as returned by
        :func:`compute_preprocessing_stats`.  When provided, the per-modality
        statistics are injected into the corresponding
        :class:`PreprocessConfig` instances.  Default is ``None``
        (no statistics applied).
    prediction_mode : bool, optional
        If ``True``, operate in prediction mode.  Default is ``False``.
    prediction_horizon_s : float, optional
        Duration of the prediction target window in seconds.  Only used
        when ``prediction_mode=True``.  Default is ``0.2``.
    input_signals : list of str or None, optional
        Modality names to include in the returned batch (or in the
        ``'inputs'`` dict in prediction mode).  Defaults to
        ``['ece', 'co2', 'mhr']``.
    target_signals : list of str or None, optional
        Modality names to include in the ``'targets'`` dict in prediction
        mode.  Defaults to ``['d_alpha', 'mse', 'ts_core_density']``.

    Attributes
    ----------
    signal_configs : list of SignalConfig
        Per-instance deep copy of :attr:`SIGNAL_CONFIGS`, updated with
        any statistics from *preprocessing_stats*.
    movie_configs : list of MovieConfig
        Per-instance deep copy of :attr:`MOVIE_CONFIGS`.
    hdf5_path : Path
        Resolved path to the HDF5 file.
    duration : float
        Total shot duration from t = 0 in seconds, as inferred from the
        HDF5 time axes.
    length : int
        Number of non-overlapping chunks available (i.e. ``__len__``).
    n_freq_bins : int
        Number of STFT frequency bins: ``n_fft // 2 + 1``.
    stft_window : torch.Tensor
        Hann window tensor of length ``n_fft`` used for STFT computation.

    Notes
    -----
    The class-level :attr:`SIGNAL_CONFIGS` and :attr:`MOVIE_CONFIGS` lists
    define the full set of supported diagnostics:

    **Signals** (``SIGNAL_CONFIGS``)

    ==========================  ========  ==========  =====  ==================
    Name                        Channels  Target fs   STFT   Preprocessing
    ==========================  ========  ==========  =====  ==================
    ``mhr``                     6         500 kHz     yes    log
    ``ece``                     40        500 kHz     yes    log
    ``co2``                     4         500 kHz     yes    log
    ``ech_power``               12        10 kHz      no     none
    ``ech_tor_angle``           12        10 kHz      no     none
    ``ech_pol_angle``           12        10 kHz      no     none
    ``ech_polarization``        12        10 kHz      no     none
    ``pin``                     8         10 kHz      no     standardize
    ``beam_voltage``            8         10 kHz      no     none
    ``tin``                     8         10 kHz      no     none
    ``mse``                     69        100 Hz      no     standardize
    ``filterscopes``            104       10 kHz      yes    log
    ``cer_ti``                  48        100 Hz      no     standardize
    ``cer_rot``                 48        100 Hz      no     standardize
    ``sxr``                     320       10 kHz      no     log
    ``neutron_rate``            4         40 kHz      no     log
    ``ts_core_density``         44        100 Hz      no     log_standardize
    ``ts_tangential_density``   10        100 Hz      no     log_standardize
    ``ts_core_temp``            44        100 Hz      no     log_standardize
    ``ts_tangential_temp``      10        100 Hz      no     log_standardize
    ``vib``                     24        50 Hz       yes    log
    ``bolo_raw``                48        10 kHz      no     log
    ``gas_flow``                11        10 kHz      no     none
    ``gas_raw``                 11        10 kHz      no     none
    ``ich``                     1         10 kHz      no     none
    ``rmp``                     12        10 kHz      no     none
    ``mirnov``                  29        500 kHz     yes    log
    ``langmuir``                72        500 kHz     yes    log
    ``i_coil``                  18        50 kHz      no     none
    ``bes``                     64        500 kHz     yes    log
    ==========================  ========  ==========  =====  ==================

    **Movies** (``MOVIE_CONFIGS``)

    ===========  ===  =======  =========
    Name         FPS  Height   Width
    ===========  ===  =======  =========
    ``irtv``     50   513      640
    ``tangtv``   50   240      720
    ===========  ===  =======  =========
    """

    # Define all signal configurations with preprocessing
    SIGNAL_CONFIGS = [
        SignalConfig(
            name = "mhr",
            hdf5_keys=["mhr"],
            num_channels=8,
            target_fs=500e3,
            apply_stft=True,
            channels_to_use=slice(2, 8),  # Skip first 2 channels
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "ece",
            ["ece"],
            48,
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
            preprocess=PreprocessConfig(method="log_standardize"),
        ),
        SignalConfig(
            "ech_power",
            ["ech_power"],
            12,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "ech_tor_angle",
            ["ech_tor_angle"],
            12,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "ech_pol_angle",
            ["ech_pol_angle"],
            12,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "ech_polarization",
            ["ech_polarization"],
            12,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "pin",
            ["pinj"],
            8,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "beam_voltage",
            ["beam_voltage"],
            8,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "tin",
            ["tinj"],
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
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "ts_core_density",
            ["ts_core_density"],
            44,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log_standardize"),
            zero_is_missing=True,
        ),
        SignalConfig(
            "filterscopes",
            ["filterscopes"],
            104,
            10e3,
            channels_to_use=slice(0, 8),  # Use only the first 8 channels
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "cer_ti",
            ["cer_ti"],
            48,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "cer_rot",
            ["cer_rot"],
            48,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="standardize"),
        ),
        SignalConfig(
            "sxr",
            ["sxr"],
            320,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "neutron_rate",
            ["neutron_rate"],
            4,
            40e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "ts_tangential_density",
            ["ts_tangential_density"],
            10,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log_standardize"),
            zero_is_missing=True,
        ),
        SignalConfig(
            "ts_core_temp",
            ["ts_core_temp"],
            44,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log_standardize"),
            zero_is_missing=True,
        ),
        SignalConfig(
            "ts_tangential_temp",
            ["ts_tangential_temp"],
            10,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log_standardize"),
            zero_is_missing=True,
        ),
        SignalConfig(
            "vib",
            ["vib"],
            24,
            50,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "bolo_raw",
            ["bolo"],
            48,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "gas_flow",
            ["gas_flow"],
            11,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "gas_raw",
            ["gas_raw"],
            11,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "ich",
            ["ich"],
            1,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "rmp",
            ["rmp"],
            12,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "mirnov",
            ["mirnov"],
            29,
            500e3,
            apply_stft=True,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "langmuir",
            ["langmuir"],
            72,
            500e3,
            apply_stft=True,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "i_coil",
            ["i_coil"],
            18,
            50e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
        ),
        SignalConfig(
            "bes",
            ["bes"],
            64,
            500e3,
            apply_stft=True,
            preprocess=PreprocessConfig(method="log"),
        ),
    ]

    MOVIE_CONFIGS = [
        MovieConfig("irtv", ["irtv"], 7, 50, 513, 640),
        MovieConfig("tangtv", ["tangtv"], 7, 50, 240, 720),
    ]

    def __init__(
            self,
            hdf5_path: str | Path,
            chunk_duration_s: float = 0.5,
            max_duration_s: float = 12.0,
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

        if isinstance(hdf5_path, str):
            self.hdf5_path = Path(hdf5_path)
        else:
            self.hdf5_path = hdf5_path
        self.chunk_duration_s = chunk_duration_s
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preprocessing_stats = preprocessing_stats or {}

        # Prediction settings
        self.prediction_mode = prediction_mode
        self.prediction_horizon_s = prediction_horizon_s
        self.input_signals = input_signals or ["ece", "co2", "mhr"]
        self.target_signals = (
                target_signals or ["mse", "ts_core_density"])

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        self._update_preprocessing_stats()
        self.h5_file = None
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                duration = self._compute_duration(f)
        except OSError as e:
            print(self.hdf5_path)
            raise e
        self.duration = min(duration, max_duration_s)
        # In prediction mode, reduce length to ensure extended window fits
        if self.prediction_mode:
            total_window = self.chunk_duration_s + self.prediction_horizon_s
            max_time = self.duration - total_window
            self.length = max(
                1, int(np.floor(max_time / self.chunk_duration_s)))
        else:
            self.length = max(
                1, int(np.ceil(self.duration / self.chunk_duration_s)))

        self.n_freq_bins = n_fft // 2 + 1
        self.stft_window = torch.hann_window(n_fft)

    def _compute_duration(
            self,
            f: h5py.File,
    ) -> float:
        """
        Compute shot duration from t=0.

        Iterates over all signal and movie configurations, reads the
        ``xdata`` timestamps from the HDF5 file, and accumulates the
        maximum duration across all available diagnostics.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle for the shot.

        Returns
        -------
        max_duration : float
            Duration in seconds from t=0 to the last sample, across all
            signals and movies.  Guaranteed to be at least 1.0 s.
        """
        max_duration = 0.0

        # Process signals
        for config in self.signal_configs:
            for key_path in config.hdf5_keys:
                try:
                    parts = key_path.split("/")
                    curr = f
                    for part in parts:
                        curr = curr[part]

                    xdata_s = curr["xdata"][:]

                    if len(xdata_s) < 2:
                        continue

                    # Duration from t=0 to end
                    duration_s = (xdata_s[-1] - 0.0)
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

                    duration_s = (xdata_ms[-1] - 0.0)
                    max_duration = max(max_duration, duration_s)
                    break

                except (KeyError, ValueError):
                    continue

        return max_duration

    def _update_preprocessing_stats(self):
        """
        Propagate loaded statistics into each signal's preprocessing config.

        Reads ``self.preprocessing_stats`` — a mapping from signal name to
        a dict with ``'raw'`` and ``'log'`` sub-dicts, each containing
        ``'mean'``, ``'std'``, ``'min_val'``, and ``'max_val'``.

        The appropriate sub-dict is selected based on the preprocessing
        method: ``log_standardize`` uses ``'log'`` stats, all others use
        ``'raw'`` stats.

        Also supports the legacy flat format (no ``'raw'``/``'log'`` keys)
        for backwards compatibility.

        Returns
        -------
        None
        """
        _LOG_METHODS = {"log_standardize", "log_normalize"}

        for config in self.signal_configs + self.movie_configs:
            if config.name not in self.preprocessing_stats:
                continue
            entry = self.preprocessing_stats[config.name]

            # New format: entry has 'raw' and/or 'log' sub-dicts
            if "raw" in entry or "log" in entry:
                key = "log" if config.preprocess.method in _LOG_METHODS else "raw"
                stats = entry.get(key, {})
            else:
                # Legacy flat format
                stats = entry

            if "mean" in stats:
                val = np.array(stats["mean"], dtype=np.float64)
                val[np.isnan(val)] = 0.0
                config.preprocess.mean = val
            if "std" in stats:
                val = np.array(stats["std"], dtype=np.float64)
                val[np.isnan(val)] = 1.0
                config.preprocess.std = val
            if "min_val" in stats:
                val = np.array(stats["min_val"], dtype=np.float64)
                val[np.isnan(val)] = 0.0
                config.preprocess.min_val = val
            if "max_val" in stats:
                val = np.array(stats["max_val"], dtype=np.float64)
                val[np.isnan(val)] = 1.0
                config.preprocess.max_val = val

    def _apply_preprocessing(
            self,
            tensor: torch.Tensor,
            config: SignalConfig
    ) -> torch.Tensor:
        """
        Apply the configured preprocessing transformation to a tensor.

        Statistics stored on *config* (mean, std, min_val, max_val) are
        reshaped to ``(C, 1, 1)`` or ``(C, 1)`` as needed so they broadcast
        correctly over time and frequency dimensions.

        Parameters
        ----------
        tensor : torch.Tensor
            Input data; one of:

            - spectrogram ``(C, F, T)``
            - time-series ``(C, T)``
            - video ``(C, T, H, W)``
        config : SignalConfig
            Signal configuration specifying ``method`` and the optional
            statistical parameters.

        Returns
        -------
        torch.Tensor
            Transformed tensor with the same shape as *tensor*.
        """
        preprocessing_config: PreprocessConfig = config.preprocess
        if preprocessing_config.method == "none":
            return tensor

        ch = config.channels_to_use

        # Reshape per-channel statistics for correct broadcasting.
        # Stats have shape (C,); we add trailing singleton dims to match ndim.
        reshape_dims: tuple[int, ...] | None
        if tensor.ndim == 4:
            # (C, T, H, W) — video
            reshape_dims = (tensor.shape[0], 1, 1, 1)
        elif tensor.ndim == 3:
            # (C, F, T) — spectrogram
            reshape_dims = (tensor.shape[0], 1, 1)
        elif tensor.ndim == 2:
            # (C, T) — time-series
            reshape_dims = (tensor.shape[0], 1)
        else:
            reshape_dims = None

        if preprocessing_config.method == "standardize":
            if preprocessing_config.mean is None or preprocessing_config.std is None:
                print("Warning: "
                      "standardize requested but no statistics provided")
                return tensor

            mean = torch.as_tensor(
                preprocessing_config.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.as_tensor(
                preprocessing_config.std, dtype=tensor.dtype, device=tensor.device)
            if ch is not None:
                mean = mean[ch]
                std = std[ch]
            if reshape_dims is not None:
                mean = mean.reshape(reshape_dims)
                std = std.reshape(reshape_dims)

            tensor -= mean
            tensor /= std.clamp(min=1e-3)
            return tensor

        elif preprocessing_config.method == "normalize":
            if preprocessing_config.min_val is None or preprocessing_config.max_val is None:
                print("Warning: "
                      "normalize requested but no statistics provided")
                return tensor

            min_val = torch.as_tensor(
                preprocessing_config.min_val, dtype=tensor.dtype, device=tensor.device)
            max_val = torch.as_tensor(
                preprocessing_config.max_val, dtype=tensor.dtype, device=tensor.device)
            if ch is not None:
                min_val = min_val[ch]
                max_val = max_val[ch]
            if reshape_dims is not None:
                min_val = min_val.reshape(reshape_dims)
                max_val = max_val.reshape(reshape_dims)

            return (tensor - min_val) / (max_val - min_val + preprocessing_config.eps)

        elif preprocessing_config.method == "log_standardize":
            arr = tensor.numpy()
            arr = np.clip(arr, a_min=-.99, a_max=None, out=arr)
            arr += 1
            np.log10(arr, out=arr)

            if preprocessing_config.mean is None or preprocessing_config.std is None:
                print("Warning: "
                      "log_standardize requested but no statistics provided")
                return tensor

            mean = torch.as_tensor(
                preprocessing_config.mean, dtype=tensor.dtype, device=tensor.device)
            std = torch.as_tensor(
                preprocessing_config.std, dtype=tensor.dtype, device=tensor.device)
            if ch is not None:
                mean = mean[ch]
                std = std[ch]
            if reshape_dims is not None:
                mean = mean.reshape(reshape_dims)
                std = std.reshape(reshape_dims)

            # In-place to avoid allocating temporary tensors in worker
            # processes. With large batch sizes and many workers, out-of-place
            # `(tensor - mean) / std` fragments each worker's heap enough to
            # cause CPU OOM after several epochs.
            tensor -= mean
            tensor /= std.clamp(min=1e-3)
            return tensor

        elif preprocessing_config.method == "log_normalize":
            arr = tensor.numpy()
            arr = np.clip(arr, a_min=-.99, a_max=None, out=arr)
            arr += 1
            np.log10(arr, out=arr)

            if preprocessing_config.min_val is None or preprocessing_config.max_val is None:
                print("Warning: "
                      "log_normalize requested but no statistics provided")
                return tensor

            min_val = torch.as_tensor(
                preprocessing_config.min_val, dtype=tensor.dtype, device=tensor.device)
            max_val = torch.as_tensor(
                preprocessing_config.max_val, dtype=tensor.dtype, device=tensor.device)
            if ch is not None:
                min_val = min_val[ch]
                max_val = max_val[ch]
            if reshape_dims is not None:
                min_val = min_val.reshape(reshape_dims)
                max_val = max_val.reshape(reshape_dims)

            tensor -= min_val
            tensor /= (max_val - min_val + preprocessing_config.eps)
            return tensor

        elif preprocessing_config.method == "log":
            arr = tensor.numpy()
            arr = np.clip(arr, a_min=-.99, a_max=None, out=arr)
            arr += 1
            np.log10(arr, out=arr)
            return tensor

        return tensor

    def _open_hdf5(self):
        """
        Open the HDF5 file for the current worker, if not already open.

        Uses a large chunk cache (256 MB, 10 000 slots) to amortise
        repeated random-access reads during training.  The open file handle
        is stored in ``self.h5_file`` and reused across subsequent calls.

        Returns
        -------
        None
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")

    def _load_signal_raw(
            self,
            f: h5py.File,
            config: SignalConfig,
            t_start: float,
            t_end: float
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
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
        tensor : torch.Tensor
            Array of shape ``(C, T)`` at target sampling rate.
            Positions beyond the actual signal end are zero-padded;
            positions that were NaN in the raw data are replaced with 0.
        valid_length : int
            Number of valid (non-padded) samples in the time dimension,
            expressed in terms of ``config.target_fs``.
        nan_mask : torch.Tensor
            Float tensor of shape ``(C, T)`` where ``1.0`` marks positions
            that were NaN in the raw HDF5 data and ``0.0`` marks valid
            positions.
        """
        duration_s = t_end - t_start
        T_target = round(duration_s * config.target_fs)

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
            if config.channels_to_use:
                num_channels = len(
                    range(*config.channels_to_use.indices(config.num_channels))
                )
            else:
                num_channels = config.num_channels
            nan_mask = torch.ones((num_channels, T_target))
            return torch.zeros((num_channels, T_target)), 0, nan_mask

        ydata_ds = data_group["ydata"]
        xdata_ds = data_group["xdata"]

        # Get time range and sample count
        xdata_start_s = xdata_ds[0]
        xdata_end_s = xdata_ds[-1]

        n_samples = xdata_ds.shape[0]

        if n_samples < 2 or xdata_end_s == xdata_start_s:
            if config.channels_to_use:
                num_channels = len(
                    range(*config.channels_to_use.indices(config.num_channels))
                )
            else:
                num_channels = config.num_channels
            nan_mask = torch.ones((num_channels, T_target))
            return torch.zeros((num_channels, T_target)), 0, nan_mask

        # Compute actual sampling frequency from the data
        actual_fs = (n_samples - 1) / (xdata_end_s - xdata_start_s)

        # Step 1: Initialize output array (C, T) — matches HDF5 storage layout,
        # avoiding a transpose and keeping all copies between contiguous arrays
        if config.channels_to_use:
            num_channels = len(
                range(*config.channels_to_use.indices(config.num_channels))
            )
        else:
            num_channels = config.num_channels
        output = np.zeros(
            (num_channels, round(duration_s * actual_fs)),
            dtype=np.float32
        )
        self._nan_mask_buf = np.zeros_like(output, dtype=bool)

        # Step 2: Calculate which HDF5 indices correspond to [t_start, t_end]
        # xdata[i] = xdata_start_s + i / actual_fs
        # Solving for i: i = (t - xdata_start_s) * actual_fs
        hdf5_start = round((t_start - xdata_start_s) * actual_fs)
        hdf5_end = round((t_end - xdata_start_s) * actual_fs)

        # Clamp to valid HDF5 range [0, n_samples]
        hdf5_start_clamped = max(0, min(hdf5_start, n_samples))
        hdf5_end_clamped = max(0, min(hdf5_end, n_samples))

        # Step 3: Load data if there's any overlap.
        # Clip channels at read time so HDF5 transfers, isnan scan, and copy
        # all operate on the minimum number of channels needed.
        if hdf5_start_clamped < hdf5_end_clamped:
            ch_slice = (
                config.channels_to_use
                if config.channels_to_use is not None
                else slice(None, config.num_channels)
            )
            data = ydata_ds[ch_slice, hdf5_start_clamped:hdf5_end_clamped]

            # Step 4: Calculate where to insert in output array
            # The loaded data starts at time:
            # xdata_start_s + hdf5_start_clamped / actual_fs
            # This corresponds to output index:
            # (that_time - t_start) * actual_fs
            output_start = hdf5_start_clamped - hdf5_start
            output_end = output_start + data.shape[1]

            # Clamp to output bounds
            src_start = 0
            src_end = data.shape[1]

            if output_start < 0:
                src_start = -output_start
                output_start = 0
            if output_end > output.shape[1]:
                src_end -= output_end - output.shape[1]
                output_end = output.shape[1]

            if src_start < src_end and output_start < output_end:
                chunk = data[:, src_start:src_end]
                nan_mask = np.isnan(chunk)
                chunk[nan_mask] = 0
                self._nan_mask_buf[:chunk.shape[0],
                                   output_start:output_end] |= \
                    nan_mask[:, :output_end - output_start]

                if chunk.shape[0] == config.num_channels:
                    output[:, output_start:output_end] = chunk
                else:
                    output[:chunk.shape[0], output_start:output_end] = chunk

        # Step 5: Compute valid_length — how many target-rate samples correspond
        # to real data.  The HDF5 data ends at hdf5_end_clamped (native index),
        # which maps to time xdata_start_s + hdf5_end_clamped / actual_fs.
        t_data_end = xdata_start_s + hdf5_end_clamped / actual_fs
        valid_length = min(T_target, max(0, round((t_data_end - t_start) * config.target_fs)))

        # Step 6: Convert to tensor and resample to target frequency.
        # tensor is already (C, T), so no permute is needed around interpolate.
        tensor = torch.from_numpy(output)

        # Build NaN mask before resampling
        nan_mask = torch.from_numpy(self._nan_mask_buf.copy()).float()
        del self._nan_mask_buf

        if tensor.shape[1] != T_target:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=T_target,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            if nan_mask is not None:
                # Resample mask: nearest-neighbor to avoid blurring
                nan_mask = F.interpolate(
                    nan_mask.unsqueeze(0),
                    size=T_target,
                    mode="nearest",
                ).squeeze(0)

        return tensor, valid_length, nan_mask

    def _compute_stft(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute the STFT magnitude spectrogram of a multi-channel signal.

        Applies a Hann-windowed STFT and discards the DC component (bin 0)
        to avoid extreme values from the signal offset.

        Parameters
        ----------
        signal : torch.Tensor
            Multi-channel time-series of shape ``(C, T)`` at the signal's
            native sampling rate.

        Returns
        -------
        torch.Tensor
            Magnitude spectrogram of shape ``(C, n_fft // 2, time_frames)``.
        """
        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window,
            return_complex=True,
        )
        # spec = spec[:, 1:, :]  # Remove DC component (extreme values)
        return torch.abs(spec)[:, 1:, :]  # Remove DC component (extreme value)

    def _load_metadata(self, f: h5py.File) -> dict:
        """
        Load shot metadata from the HDF5 file.

        Extracts the operator log stored under ``f['log']['data']`` as a
        UTF-8 string.  Returns an empty string for the ``'text'`` key when
        the ``'log'`` group is absent.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle for the shot.

        Returns
        -------
        dict
            Dictionary with a single key ``'text'`` mapping to the decoded
            log string.
        """
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

    def __len__(self) -> int:
        """
        Return the number of non-overlapping chunks in the shot.

        Returns
        -------
        int
            ``ceil(duration / chunk_duration_s)`` in standard mode, or
            ``floor((duration - prediction_horizon_s) / chunk_duration_s)``
            in prediction mode; at least 1.
        """
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
            self,
            data: torch.Tensor,
            config: SignalConfig,
            valid_length: int,
    ) -> tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Transpose, optionally compute STFT, and preprocess a raw signal.

        Parameters
        ----------
        data : torch.Tensor
            Raw signal of shape ``(C, T)`` as returned by
            :meth:`_load_signal_raw`.
        config : SignalConfig
            Configuration for the signal, including ``apply_stft`` and
            ``preprocess`` settings.
        valid_length : int
            Number of valid (non-padded) samples in ``data``, as returned by
            :meth:`_load_signal_raw`.

        Returns
        -------
        processed : torch.Tensor
            Processed tensor:

            - ``(C, n_fft // 2, time_frames)`` when
              ``config.apply_stft`` is ``True``.
            - ``(C, T)`` otherwise.
        valid_length_out : int
            Number of valid entries in the time (last) dimension of the
            processed tensor.  For STFT signals this is expressed in frames;
            for raw signals it equals ``valid_length``.
        element_mask : torch.Tensor or None
            Boolean mask of shape matching *processed* where ``True``
            indicates a valid (non-missing) element.  Only returned when
            ``config.zero_is_missing`` is ``True``; otherwise ``None``.
        """
        # Build per-element mask before any transformation
        if config.zero_is_missing:
            element_mask = data != 0.0
        else:
            element_mask = None

        if config.apply_stft:
            processed = self._compute_stft(data)
            # With torch.stft default center=True: n_frames = T // hop_length + 1
            valid_length_out = min(
                processed.shape[-1],
                valid_length // self.hop_length + 1,
            )
        else:
            processed = data
            valid_length_out = valid_length

        processed = self._apply_preprocessing(processed, config)

        if element_mask is not None:
            # Fill missing positions with 0 after preprocessing so they
            # don't pollute neighbours but remain numerically benign.
            processed[~element_mask] = 0.0

        return processed, valid_length_out, element_mask

    def _load_movie_raw(
            self,
            f: h5py.File,
            config: MovieConfig,
            t_start: float,
            t_end: float
    ) -> torch.Tensor:
        """
        Load, window, and resample a raw movie to the target resolution.

        Reads frame data from the HDF5 file (stored as ``(C, W, H, T)``),
        clips to the requested time window, collapses channels via
        ``nanmean``, and resamples with trilinear interpolation to the
        target frame rate and spatial dimensions defined in *config*.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle for the shot.
        config : MovieConfig
            Camera configuration specifying target FPS, height, and width.
        t_start : float
            Start time in seconds (relative to t=0).
        t_end : float
            End time in seconds (relative to t=0).

        Returns
        -------
        torch.Tensor
            Resampled movie of shape
            ``(config.channels,
            round((t_end - t_start) * config.target_fps),
            config.height, config.width)``.
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

        if data_group is None:
            return torch.zeros(
                (config.channels, round(duration_s * config.target_fps),
                 config.height, config.width)
            )

        ydata_ds = data_group["ydata"]
        xdata_ds = data_group["xdata"]

        if ydata_ds.size == 0:
            return torch.zeros(
                (config.channels, round(duration_s * config.target_fps),
                 config.height, config.width)
            )

        # Get time range and frame count
        xdata_start_s = xdata_ds[0]
        xdata_end_s = xdata_ds[-1]
        n_frames = xdata_ds.shape[0]

        if n_frames < 2 or xdata_end_s == xdata_start_s:
            return torch.zeros(
                (config.channels, round(duration_s * config.target_fps),
                 config.height, config.width)
            )

        # Compute actual frame rate from the data
        actual_fps = (n_frames - 1) / (xdata_end_s - xdata_start_s)

        # ydata layout: (C, W, H, T) — time is the last axis
        raw_channels = ydata_ds.shape[0]
        raw_height = ydata_ds.shape[2]  # H
        raw_width = ydata_ds.shape[3]  # W

        # Step 1: Initialize output array with zeros at actual fps
        # (T, C, H, W)
        output = np.zeros(
            (
                raw_channels, round(duration_s * actual_fps),
                raw_height,
                raw_width
            ),
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
            data = ydata_ds[:, hdf5_start_clamped:hdf5_end_clamped, :, :]
            data[np.isnan(data)] = 0

            # Step 4: Calculate where to insert in output array
            # The loaded data starts at time:
            # xdata_start_s + hdf5_start_clamped / actual_fps
            # This corresponds to output index:
            # (that_time - t_start) * actual_fps
            output_start = hdf5_start_clamped - hdf5_start
            output_end = output_start + data.shape[1]

            # Clamp to output bounds
            src_start = 0
            src_end = data.shape[1]

            if output_start < 0:
                src_start = -output_start
                output_start = 0
            if output_end > output.shape[1]:
                src_end -= output_end - output.shape[1]
                output_end = output.shape[1]

            # Insert data into output
            if src_start < src_end and output_start < output_end:
                output[:, output_start:output_end] = data[:, src_start:src_end]

        # Step 5: Convert to tensor and resample to target fps and dimensions
        tensor = torch.from_numpy(output)

        # Resample using trilinear interpolation within channels independently.
        # F.interpolate treats dim-1 as channels (not interpolated across);
        # the 3D kernel blends only within each channel's (T, H, W) volume.
        # (C, T, H, W) → (1, C, T, H, W) → trilinear → (C, T', H', W')
        target_size = (
            round(duration_s * config.target_fps),
            config.height,
            config.width
        )
        if tensor.shape[1:] != torch.Size(target_size):
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=target_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        return tensor

    def __getitem__(self, idx: int) -> dict:
        """
        Return the data chunk at position *idx*.

        Opens the HDF5 file on the first call (lazy initialisation) and
        delegates to :meth:`_getitem_standard` or
        :meth:`_getitem_prediction` depending on ``self.prediction_mode``.

        Parameters
        ----------
        idx : int
            Chunk index in ``[0, len(self))``.

        Returns
        -------
        dict
            In standard mode: flat mapping from signal/movie/metadata name
            to processed tensor or string.
            In prediction mode: ``{'inputs': dict, 'targets': dict}``.
        """
        self._open_hdf5()

        if self.prediction_mode:
            return self._getitem_prediction(idx)
        else:
            return self._getitem_standard(idx)

    def _getitem_standard(self, idx: int) -> dict:
        """
        Load and return the data chunk at *idx* in standard mode.

        Computes the time window
        ``[idx * chunk_duration_s, (idx + 1) * chunk_duration_s]``, loads
        all active signals, movies, and metadata, and returns them as a
        flat dictionary.

        Parameters
        ----------
        idx : int
            Chunk index in ``[0, len(self))``.

        Returns
        -------
        dict[str, torch.Tensor | str]
            Keys are signal/movie names plus ``'text'`` (when ``'text'``
            is in ``self.input_signals``).  Tensor shapes follow the rules
            in :meth:`_process_signal` and :meth:`_load_movie_raw`.
        """
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        t_start = idx * step
        t_end = t_start + self.chunk_duration_s

        # Load and process all signals
        all_signals = {}
        for config in self.signal_configs:
            if config.name in self.input_signals:
                raw_data, valid_length, nan_mask = self._load_signal_raw(
                    self.h5_file,
                    config, t_start,
                    t_end
                )
                tensor, valid_length_out, element_mask = self._process_signal(
                    raw_data, config, valid_length
                )
                # Combine zero_is_missing and NaN masks
                valid_mask = nan_mask < 0.5  # True = valid (not NaN)
                if element_mask is not None:
                    element_mask = element_mask & valid_mask
                else:
                    element_mask = valid_mask

                # Zero out masked positions so the model never sees
                # bogus values (e.g. standardized NaN-replaced zeros).
                tensor[~element_mask] = 0.0

                all_signals[config.name] = tensor
                all_signals[f"{config.name}_valid"] = valid_length_out
                if element_mask is not None:
                    all_signals[f"{config.name}_mask"] = element_mask

        # Load and process movies
        all_movies = {}
        for movie_config in self.movie_configs:
            if movie_config.name in self.input_signals:
                raw_movie = self._load_movie_raw(
                    self.h5_file, movie_config, t_start, t_end
                )
                all_movies[movie_config.name] = self._apply_preprocessing(
                    raw_movie, movie_config)

        # Load metadata
        if "text" in self.input_signals:
            all_metadata = self._load_metadata(self.h5_file)
        else:
            all_metadata = {}

        return {**all_signals, **all_movies, **all_metadata}

    def _getitem_prediction(self, idx: int) -> dict:
        """
        Load an extended window and split it into input and target chunks.

        The extended window spans
        ``[idx * chunk_duration_s,
        idx * chunk_duration_s + chunk_duration_s + prediction_horizon_s]``.
        All configured signals are processed over this window and then split
        at ``chunk_duration_s`` frames into the input and target portions.

        Parameters
        ----------
        idx : int
            Chunk index in ``[0, len(self))``.

        Returns
        -------
        dict
            ``{'inputs': dict[str, torch.Tensor | str],
            'targets': dict[str, torch.Tensor]}``.
            Each inner dict maps signal names to the corresponding slice of
            the processed tensor.
        """
        # Extended window: from t to t + chunk_duration + prediction_horizon
        step = getattr(self, "step_size_s", self.chunk_duration_s)
        t_start = idx * step
        t_end = t_start + self.chunk_duration_s + self.prediction_horizon_s

        signals_to_load = set(self.input_signals) | set(self.target_signals)

        # Load and process all signals with extended window
        all_signals = {}
        for config in self.signal_configs:
            if config.name not in signals_to_load:
                continue
            raw_data, valid_length, nan_mask = self._load_signal_raw(
                self.h5_file, config, t_start, t_end
            )
            tensor, valid_length_out, element_mask = self._process_signal(
                raw_data, config, valid_length
            )
            if nan_mask is not None:
                valid_mask = nan_mask < 0.5
                if element_mask is not None:
                    element_mask = element_mask & valid_mask
                else:
                    element_mask = valid_mask

            # Zero out masked positions so the model never sees
            # bogus values (e.g. standardized NaN-replaced zeros).
            if element_mask is not None:
                tensor[~element_mask] = 0.0

            all_signals[config.name] = tensor
            all_signals[f"{config.name}_valid"] = valid_length_out
            if element_mask is not None:
                all_signals[f"{config.name}_mask"] = element_mask

        # Load and process movies
        all_movies = {}
        for movie_config in self.movie_configs:
            if movie_config.name not in signals_to_load:
                continue
            raw_movie = self._load_movie_raw(
                self.h5_file, movie_config, t_start, t_end
            )
            all_movies[movie_config.name] = self._apply_preprocessing(
                raw_movie, movie_config
            )

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
                n_training_frames = round(
                    self.chunk_duration_s * config.target_fs
                )

            if config.name in self.input_signals:
                inputs[config.name] = signal[..., :n_training_frames]

            if config.name in self.target_signals:
                targets[config.name] = signal[..., n_training_frames:]

        # Movies: split along the time dimension (dim 1 of (C, T, H, W))
        for movie_config in self.movie_configs:
            if movie_config.name not in signals_to_load:
                continue
            movie_name = movie_config.name
            movie_data = all_movies[movie_name]
            n_training_frames = round(
                self.chunk_duration_s * movie_config.target_fps
            )
            # movie_data shape: (C, extended_movie_frames, height, width)
            if movie_name in self.input_signals:
                inputs[movie_name] = movie_data[:, :n_training_frames]

            if movie_name in self.target_signals:
                targets[movie_name] = movie_data[:, n_training_frames:]

        # Metadata (text) only goes to inputs
        if "text" in self.input_signals:
            inputs.update(all_metadata)

        return {"inputs": inputs, "targets": targets}

    def __del__(self):
        """
        Close the HDF5 file handle when the dataset is garbage-collected.

        Silently ignores errors that may occur if the file was already
        closed or if Python is shutting down.

        Returns
        -------
        None
        """
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass


def _collate_dict(samples: list[dict]) -> dict:
    """Collate a list of sample dicts into a batched dict.

    Keys ending in ``'_valid'`` hold plain Python ints and are stacked into a
    ``[B]`` long tensor.  ``'text'`` keys are kept as a list.  All other keys
    are assumed to hold tensors and are stacked normally.
    """
    collated = {}
    for key in samples[0]:
        if key == "text":
            collated[key] = [d[key] for d in samples]
        elif key.endswith("_valid"):
            collated[key] = torch.tensor([d[key] for d in samples], dtype=torch.long)
        else:
            collated[key] = torch.stack([d[key] for d in samples])
    return collated


def collate_fn(batch):
    """Custom collate function for batching."""
    elem = batch[0]

    # Check if prediction mode (has 'inputs' and 'targets' keys)
    if "inputs" in elem and "targets" in elem:
        return collate_fn_prediction(batch)

    return _collate_dict(batch)


def collate_fn_prediction(batch):
    """Collate function for prediction mode."""
    inputs_batch = [item["inputs"] for item in batch]
    targets_batch = [item["targets"] for item in batch]

    return {
        "inputs": _collate_dict(inputs_batch),
        "targets": _collate_dict(targets_batch),
    }
