import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch.nn.functional as F
import copy


class WelfordTensor:
    """
    Online Welford algorithm for per-channel statistics on batched tensors.

    Accumulates running mean, variance, minimum, and maximum over an arbitrary
    number of :meth:`update` calls without storing the full dataset in memory.
    Statistics are computed along the channel axis (axis 1 for 3-D and 4-D
    tensors) by aggregating across the batch dimension and all remaining
    non-channel dimensions.  Batches that contain any ``NaN`` value are
    silently skipped.

    The shape of the statistics vectors depends on the input rank:

    =========  ===================================  ===========
    ``ndim``   Interpretation                       Stats shape
    =========  ===================================  ===========
    4          ``(B, C, F, T)`` — spectrograms /    ``(C,)``
               time series
    3          ``(B, S, T)`` — profiles             ``(S,)``
    ≤ 2        ``(B, T)`` or scalar — video /       ``(1,)``
               fallback
    =========  ===================================  ===========

    Attributes
    ----------
    mean : torch.Tensor or None
        Running per-channel mean, shape ``(C,)``.  ``None`` before the first
        :meth:`update` call.
    std : torch.Tensor or None
        Per-channel sample standard deviation, shape ``(C,)``.  Populated
        only after :meth:`compute` is called.
    min_val : torch.Tensor or None
        Running per-channel minimum, shape ``(C,)``.  ``None`` before the
        first :meth:`update` call.
    max_val : torch.Tensor or None
        Running per-channel maximum, shape ``(C,)``.  ``None`` before the
        first :meth:`update` call.
    n : int
        Total number of scalar samples seen so far (summed over all
        non-channel dimensions across all batches).
    M2 : torch.Tensor or None
        Running sum of squared deviations from the mean (Welford
        accumulator), shape ``(C,)``.  ``None`` before the first
        :meth:`update` call.
    initialized : bool
        ``True`` once the internal buffers have been allocated on the first
        :meth:`update` call.

    Notes
    -----
    The parallel (batch) variant of Welford's algorithm is used to combine
    each incoming batch with the accumulated state in a single pass
    [1]_.  All accumulation is done in ``float64`` regardless of the input
    dtype to minimise floating-point cancellation errors.

    References
    ----------
    .. [1] Welford, B. P. (1962). Note on a method for calculating corrected
       sums of squares and products. *Technometrics*, 4(3), 419–420.
       https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    Examples
    --------
    >>> import torch
    >>> tracker = WelfordTensor()
    >>> for _ in range(10):
    ...     batch = torch.randn(32, 8, 512, 200)  # (B, C, F, T)
    ...     tracker.update(batch)
    >>> stats = tracker.compute()
    >>> stats['mean'].shape
    (8,)
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
        """
        Allocate accumulator buffers sized to match *value*.

        Called automatically by :meth:`update` on the first non-NaN batch.
        Derives the number of channels from the input rank:

        * ``ndim == 4``: channel axis is 1 (spectrograms / time series).
        * ``ndim == 3``: channel axis is 1 (profiles / spatial signals).
        * ``ndim <= 2``: treated as single-channel (``n_channels = 1``).

        Parameters
        ----------
        value : torch.Tensor
            First batch tensor, used only to infer ``n_channels``.
            Shape must be ``(B, C, ...)`` for 3-D or 4-D inputs.

        Returns
        -------
        None
        """
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
        self.min_val = torch.full(
            (n_channels,), float('inf'), dtype=torch.float64)
        self.max_val = torch.full(
            (n_channels,), float('-inf'), dtype=torch.float64)
        self.initialized = True

    def update(self, value: torch.Tensor):
        """
        Incorporate a new batch into the running statistics.

        Batches that contain any ``NaN`` element are silently skipped.  On
        the first valid call the accumulator buffers are allocated via
        :meth:`_initialize`.  Subsequent calls merge the incoming batch
        statistics with the accumulated state using the parallel Welford
        update rule.

        Parameters
        ----------
        value : torch.Tensor
            Batched input tensor.  Supported shapes:

            * ``(B, C, F, T)`` — spectrograms or multi-channel time series.
            * ``(B, C, 1, T)`` — single-frequency time series.
            * ``(B, S, T)``    — spatial profiles.
            * ``(B, T, H, W)`` — video frames (global statistics).

        Returns
        -------
        None
        """
        # Skip if contains NaN
        if torch.isnan(value).any():
            return

        # Initialize on first call
        if not self.initialized:
            self._initialize(value)

        # Convert to float64 for numerical stability
        value = value.to(dtype=torch.float64)

        # Compute per-channel statistics by flattening batch
        # and all non-channel dims
        if value.ndim == 4 and value.shape[1] == self.mean.shape[0]:
            # (batch, channels, freq_bins, time) → flatten batch, freq, time
            # (B, C, F, T) → (C, B*F*T)
            n_channels = value.shape[1]
            value_flat = value.permute(1, 0, 2, 3).reshape(n_channels, -1)

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
            value_flat = value.permute(1, 0, 2).reshape(n_channels, -1)

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
        """
        Derive sample standard deviation from the Welford M2 accumulator.

        Uses Bessel's correction (``n - 1``) when more than one sample has
        been seen; falls back to zeros when ``n <= 1`` to avoid division by
        zero.  The result is written to :attr:`std` in-place.

        Returns
        -------
        None
        """
        if self.n > 1:
            self.std = torch.sqrt(self.M2 / (self.n - 1))
        else:
            self.std = torch.zeros_like(self.mean)

    def compute(self):
        """
        Finalise and return all accumulated statistics as NumPy arrays.

        Calls :meth:`_compute_std` internally to derive the standard
        deviation from the Welford M2 accumulator before returning.
        Returns ``None`` if :meth:`update` was never called.

        Returns
        -------
        dict or None
            ``None`` if no data was ever seen.  Otherwise a dictionary
            with the following keys, each mapping to a
            ``numpy.ndarray`` of shape ``(C,)``:

            ``'mean'``
                Per-channel arithmetic mean.
            ``'std'``
                Per-channel sample standard deviation (Bessel-corrected).
            ``'min_val'``
                Per-channel minimum value seen across all batches.
            ``'max_val'``
                Per-channel maximum value seen across all batches.
        """
        if not self.initialized:
            return None

        self._compute_std()

        return {
            "mean": self.mean.numpy(),
            "std": self.std.numpy(),
            "min_val": self.min_val.numpy(),
            "max_val": self.max_val.numpy(),
        }


def compute_preprocessing_stats(
        datasets: "list[TokamakH5Dataset]",
        output_path: str | Path = "preprocessing_stats.pt",
        batch_size: int = 32,
        num_workers: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute per-modality preprocessing statistics over a collection of
    datasets.

    Iterates over all chunks in every dataset, accumulates running statistics
    with :class:`WelfordTensor`, and saves the result to *output_path* via
    :func:`torch.save`.  Only modalities that appear in the loaded batches
    are included in the output.

    Parameters
    ----------
    datasets : list of TokamakH5Dataset
        One or more dataset instances whose data will be concatenated.
        Signal and movie configurations are read from ``datasets[0]``.
    output_path : str or Path, optional
        Filesystem path for the saved ``.pt`` statistics file.
        Default is ``"preprocessing_stats.pt"``.
    batch_size : int, optional
        Batch size for the internal DataLoader.  Default is ``32``.
    num_workers : int, optional
        Number of DataLoader worker processes.  Default is ``1``.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        Nested dictionary ``{modality_name: stats}``, where *stats* is the
        dictionary returned by :meth:`WelfordTensor.compute`:

        ``'mean'``
            Per-channel arithmetic mean, shape ``(C,)``.
        ``'std'``
            Per-channel sample standard deviation, shape ``(C,)``.
        ``'min_val'``
            Per-channel minimum, shape ``(C,)``.
        ``'max_val'``
            Per-channel maximum, shape ``(C,)``.
    """
    from torch.utils.data import ConcatDataset
    from tqdm import tqdm

    combined = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined, batch_size=batch_size, collate_fn=collate_fn,
        num_workers=num_workers)

    # Use instance-level configs (deep copies that may have been modified).
    signal_configs = datasets[0].signal_configs
    movie_configs = datasets[0].movie_configs

    welford_stats = {
        cfg.name: WelfordTensor()
        for cfg in signal_configs + movie_configs}

    for batch in tqdm(dataloader):
        for modality_name, tensor in batch.items():
            if modality_name not in welford_stats:
                continue
            # Movies arrive as (B, C, T, H, W); flatten spatial/temporal dims
            # to (B, C, T*H*W) so WelfordTensor computes per-channel stats.
            if tensor.ndim == 5:
                B, C, T, H, W = tensor.shape
                tensor = tensor.reshape(B, C, T * H * W)
            welford_stats[modality_name].update(tensor)

    # Only include trackers that received data
    final_stats = {
        modality: tracker.compute()
        for modality, tracker in welford_stats.items()
        if tracker.initialized
    }
    torch.save(final_stats, output_path)

    print(f"Saved statistics to {output_path}")
    return final_stats


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
    preprocess: PreprocessConfig = None

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
    hdf5_path : str
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
    t0_indices : dict
        Mapping ``{modality_name: {'index': int, 'time_s': float}}``
        giving the HDF5 array index and exact timestamp (seconds) of
        t = 0 for each modality.
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
    ``ech``                     12        10 kHz      no     none
    ``pin``                     8         10 kHz      no     standardize
    ``tin``                     8         10 kHz      no     none
    ``mse``                     69        100 Hz      no     none
    ``ts_core_density``         44        100 Hz      no     log
    ``filterscopes``            104       10 kHz      yes    log
    ``cer_ti``                  48        100 Hz      no     log
    ``cer_rot``                 48        100 Hz      no     none
    ``sxr``                     320       10 kHz      no     log
    ``neutron_rate``            4         40 kHz      no     log
    ``ts_tangential_density``   10        100 Hz      no     log
    ``ts_core_temp``            44        100 Hz      no     log
    ``ts_tangential_temp``      10        100 Hz      no     log
    ``vib``                     24        50 Hz       yes    log
    ``bolo_raw``                48        10 kHz      no     log
    ``gas_flow``                11        10 kHz      no     none
    ``gas_raw``                 11        10 kHz      no     none
    ``ich``                     1         10 kHz      no     none
    ``mirnov``                  29        500 kHz     no     log
    ``langmuir``                72        500 kHz     no     log
    ``i_coil``                  18        50 kHz      no     none
    ``bes``                     64        500 kHz     no     log
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
            preprocess=PreprocessConfig(method="log"),
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
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "ech",
            ["ech"],
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
        # --- groups below added from modalities.yaml ---
        SignalConfig(
            "filterscopes",
            ["filterscopes"],
            104,
            10e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "cer_ti",
            ["cer_ti"],
            48,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "cer_rot",
            ["cer_rot"],
            48,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="none"),
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
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "ts_core_temp",
            ["ts_core_temp"],
            44,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "ts_tangential_temp",
            ["ts_tangential_temp"],
            10,
            1e2,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
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
            "mirnov",
            ["mirnov"],
            29,
            500e3,
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
        SignalConfig(
            "langmuir",
            ["langmuir"],
            72,
            500e3,
            apply_stft=False,
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
            apply_stft=False,
            preprocess=PreprocessConfig(method="log"),
        ),
    ]

    MOVIE_CONFIGS = [
        MovieConfig("irtv", ["irtv"], 6, 50, 513, 640),
        MovieConfig("tangtv", ["tangtv"], 7, 50, 240, 720),
    ]

    VALUE_CONFIG = ValueConfig(
        rdcc_nbytes=1024**2 * 256,  # 256 MB chunk cache
        rdcc_nslots=10000,  # Number of chunk slots
        ms_to_s=1/1000,  # Conversion factor from milliseconds to seconds
    )

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
                self.duration, self.t0_indices = \
                    self._compute_duration_and_t0_indices(f, max_duration_s)
        except OSError as e:
            print(self.hdf5_path)
            raise e
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

    def _find_t0_index(self, xdata_ms: np.ndarray) -> tuple[int, float]:
        """
        Find the index and exact time of t=0 in xdata.

        Parameters
        ----------
        xdata_ms : np.ndarray
            Array of timestamps in milliseconds, assumed sorted ascending.

        Returns
        -------
        index : int
            Index closest to t=0, or ``-1`` if all data is before t=0.
        actual_time_ms : float
            The actual timestamp at that index, in milliseconds.
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

    def _compute_duration_and_t0_indices(
            self,
            f: h5py.File,
            max_duration_s: float | None = None,
    ) -> tuple[float, dict]:
        """
        Compute shot duration from t=0 and locate the t=0 index per signal.

        Iterates over all signal and movie configurations, reads the
        ``xdata`` timestamps from the HDF5 file, finds the first sample at
        or after t=0, and accumulates the maximum duration across all
        available diagnostics.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle for the shot.

        Returns
        -------
        max_duration : float
            Duration in seconds from t=0 to the last sample, across all
            signals and movies.  Guaranteed to be at least 1.0 s.
        t0_indices : dict[str, dict[str, int | float]]
            Mapping from signal/movie name to a dict with keys:

            - ``'index'``: first HDF5 sample index where ``xdata >= 0``.
            - ``'time_s'``: actual timestamp at that index, in seconds.
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
                        "time_s": float(xdata_ms[t0_idx]) / 1000.0,
                    }

                    # Duration from t=0 to end
                    duration_s = (xdata_ms[-1] - 0.0) / 1000.0
                    max_duration = max(
                        max_duration, min(duration_s, max_duration_s)
                    )

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
                        "time_s": float(xdata_ms[t0_idx]) / 1000.0,
                    }

                    duration_s = (xdata_ms[-1] - 0.0) / 1000.0
                    max_duration = max(
                        max_duration, min(max_duration_s, duration_s)
                    )

                    break

                except (KeyError, ValueError):
                    continue

        return max(max_duration, 1.0), t0_indices

    def _update_preprocessing_stats(self):
        """
        Propagate loaded statistics into each signal's preprocessing config.

        Reads ``self.preprocessing_stats`` — a mapping from signal name to
        a dict of arrays keyed by ``'mean'``, ``'std'``, ``'min_val'``, and
        ``'max_val'`` — and writes found values into the corresponding
        :class:`PreprocessConfig` objects in ``self.signal_configs``.
        Signals not present in ``self.preprocessing_stats`` are unchanged.

        Returns
        -------
        None
        """
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
            self,
            tensor: torch.Tensor,
            config: PreprocessConfig
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
        config : PreprocessConfig
            Preprocessing configuration specifying ``method`` and the
            optional statistical parameters.

        Returns
        -------
        torch.Tensor
            Transformed tensor with the same shape as *tensor*.
        """
        if config.method == "none":
            return tensor

        # Reshape per-channel statistics for correct broadcasting.
        # Stats have shape (C,); we add trailing singleton dims to match ndim.
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
        xdata_start_s = xdata_ds[0]
        xdata_end_s = xdata_ds[-1]

        n_samples = xdata_ds.shape[0]

        if n_samples < 2 or xdata_end_s == xdata_start_s or np.isnan(xdata_start_s) or np.isnan(xdata_end_s):
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
            data = ydata_ds[:, hdf5_start_clamped:hdf5_end_clamped].T
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
                chunk = data[src_start:src_end]

                # Apply channel selection if specified
                if config.channels_to_use is not None:
                    chunk = chunk[:, config.channels_to_use]

                if chunk.shape[1] == config.num_channels:
                    output[output_start:output_end] = chunk
                elif chunk.shape[1] > config.num_channels:
                    output[output_start:output_end] = chunk[:, :config.num_channels]
                else:
                    output[output_start:output_end, :chunk.shape[1]] = chunk

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
        spec = spec[:, 1:, :]  # Remove DC component (extreme values)
        return torch.abs(spec)

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
        self, data: torch.Tensor, config: SignalConfig
    ) -> torch.Tensor:
        """
        Transpose, optionally compute STFT, and preprocess a raw signal.

        Parameters
        ----------
        data : torch.Tensor
            Raw signal of shape ``(T, C)`` as returned by
            :meth:`_load_signal_raw`.
        config : SignalConfig
            Configuration for the signal, including ``apply_stft`` and
            ``preprocess`` settings.

        Returns
        -------
        torch.Tensor
            Processed tensor:

            - ``(C, n_fft // 2, time_frames)`` when
              ``config.apply_stft`` is ``True``.
            - ``(C, T)`` otherwise.
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
            (raw_channels, round(duration_s * actual_fps), raw_height, raw_width),
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
            chunk = ydata_ds[:, hdf5_start_clamped:hdf5_end_clamped, :, :]
            data = np.nan_to_num(chunk, nan=0.0)

            # Step 4: Calculate where to insert in output array
            # The loaded data starts at time: xdata_start_s + hdf5_start_clamped / actual_fps
            # This corresponds to output index: (that_time - t_start) * actual_fps
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
        tensor = torch.from_numpy(output).float()

        # Resample using trilinear interpolation.
        # (C, T, H, W) → (1, C, T, H, W)
        # → interpolate → (1, C, T', H', W') → (C, T', H', W')
        tensor = (
            F.interpolate(
                tensor.unsqueeze(0),  # (1, C, T, H, W)
                size=(round(duration_s * config.target_fps), config.height, config.width),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)  # (C, T', H', W')
        )

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
                all_movies[movie_config.name] = self._apply_preprocessing(
                    raw_movie, movie_config.preprocess)

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
            raw_movie = self._load_movie_raw(self.h5_file, movie_config, t_start, t_end)
            all_movies[movie_config.name] = self._apply_preprocessing(
                raw_movie, movie_config.preprocess)

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

        # Movies: split along the time dimension (dim 1 of (C, T, H, W))
        for movie_config in self.movie_configs:
            if movie_config.name not in signals_to_load:
                continue
            movie_name = movie_config.name
            movie_data = all_movies[movie_name]
            n_training_frames = round(self.chunk_duration_s * movie_config.target_fps)
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
