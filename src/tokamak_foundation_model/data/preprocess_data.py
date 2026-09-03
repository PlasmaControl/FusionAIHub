import torch
import numpy as np
from pathlib import Path
from typing import Optional


def _safe_sum_f64(x: torch.Tensor) -> torch.Tensor:
    """Per-channel sum along the last dim, accumulated in float64."""
    return x.sum(dim=1).to(torch.float64)


def _safe_sum_sq_f64(x: torch.Tensor) -> torch.Tensor:
    """Per-channel sum-of-squares along the last dim, guaranteed finite.

    Tries the cheap float32 path first; if any per-channel result is
    non-finite (possible when raw values have magnitudes ~1e19, e.g.
    ts_core_density, whose squares overflow float32), recomputes by
    upcasting the whole row to float64 before squaring.
    """
    out = (x * x).sum(dim=1, dtype=torch.float64)
    if torch.isfinite(out).all():
        return out
    xf = x.to(torch.float64)
    return (xf * xf).sum(dim=1)


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
        # Determine number of channels based on tensor shape
        # (excluding batch dim)
        if value.ndim == 4:
            # (batch, channels, freq_bins, time) or (batch, channels, 1, time)
            n_channels = value.shape[1]
        elif value.ndim == 3:
            # (batch, spatial_points, time) or (batch, time, height)
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
        # Initialize on first call
        if not self.initialized:
            self._initialize(value)

        # Compute per-channel statistics by flattening batch
        # and all non-channel dims, ignoring NaNs
        if value.ndim == 4 and value.shape[1] == self.mean.shape[0]:
            # (B, C, F, T) → (C, B*F*T)
            n_channels = value.shape[1]
            value_flat = value.permute(1, 0, 2, 3).reshape(n_channels, -1)

        elif value.ndim == 3:
            # (B, S, T) → (S, B*T)
            n_channels = value.shape[1]
            value_flat = value.permute(1, 0, 2).reshape(n_channels, -1)

        else:
            # Video (batch, time, height, width) → global statistics
            value_flat = value.flatten().unsqueeze(0)  # (1, N)

        # NaN-aware reductions.  The previous implementation made three
        # full-tensor `.clone()` calls plus a squared temporary, i.e.
        # ~4× the input size in transient allocations per update() —
        # dominated by memcpy cost for the GB-scale STFT magnitudes
        # (e.g. langmuir: 72 × ~3M = 0.87 GB).  We sniff once whether
        # the batch actually contains any NaN; for the STFT signals
        # (which never do) this lets us skip the clones, the bool mask,
        # and the bool `.sum()` entirely.
        C, N = value_flat.shape

        if torch.isnan(value_flat).any().item():
            # Slow path: some NaNs present.  Use ONE clone and rewrite
            # it in place for each of the three reductions (sum, min,
            # max) instead of re-cloning, saving two full-tensor copies.
            nan_mask = torch.isnan(value_flat)
            n_valid = (~nan_mask).sum(dim=1)

            if (n_valid == 0).all():
                return

            safe = value_flat.clone()
            safe[nan_mask] = 0.0
            batch_sum = _safe_sum_f64(safe)
            batch_sum_sq = _safe_sum_sq_f64(safe)
            # reuse safe buffer for min/max sentinels instead of
            # re-cloning value_flat twice
            safe.copy_(value_flat)
            safe[nan_mask] = float('inf')
            batch_min = safe.amin(dim=1)
            safe[nan_mask] = float('-inf')  # +inf positions → -inf
            batch_max = safe.amax(dim=1)
        else:
            # Fast path: no NaNs — work directly on value_flat.
            n_valid = torch.full((C,), N, dtype=torch.int64)
            batch_sum = _safe_sum_f64(value_flat)
            batch_sum_sq = _safe_sum_sq_f64(value_flat)
            batch_min = value_flat.amin(dim=1)
            batch_max = value_flat.amax(dim=1)

        safe_n = n_valid.clamp(min=1).to(torch.float64)
        batch_mean = batch_sum / safe_n
        batch_mean_sq = batch_sum_sq / safe_n
        batch_var = (batch_mean_sq - batch_mean * batch_mean).clamp(min=0)

        # Parallel Welford's algorithm for combining batches
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        # Use per-channel valid counts instead of a single n_samples
        n_old = self.n if isinstance(self.n, torch.Tensor) else torch.full_like(n_valid, self.n)
        n_new = n_valid
        n_total = n_old + n_new
        batch_M2 = batch_var * n_new

        # Update mean (per-channel, guarded against zero counts)
        safe_total = n_total.clamp(min=1)
        delta = batch_mean - self.mean
        self.mean = (n_old * self.mean + n_new * batch_mean) / safe_total

        # Update M2
        self.M2 = self.M2 + batch_M2 + delta * delta * n_old * n_new / safe_total

        self.n = n_total

        # Update min/max (only where we had valid data)
        has_data = n_valid > 0
        self.min_val[has_data] = torch.minimum(
            self.min_val[has_data], batch_min[has_data])
        self.max_val[has_data] = torch.maximum(
            self.max_val[has_data], batch_max[has_data])

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
        if isinstance(self.n, torch.Tensor):
            denom = (self.n - 1).clamp(min=1)
            self.std = torch.sqrt(self.M2 / denom)
        elif self.n > 1:
            self.std = torch.sqrt(self.M2 / (self.n - 1))
        else:
            self.std = torch.zeros_like(self.mean)

    def merge(self, other: "WelfordTensor"):
        """
        Merge another WelfordTensor into this one using the parallel
        Welford algorithm.

        Parameters
        ----------
        other : WelfordTensor
            Tracker to merge in.  Left unchanged.
        """
        if not other.initialized:
            return
        if not self.initialized:
            self.mean = other.mean.clone()
            self.M2 = other.M2.clone()
            self.min_val = other.min_val.clone()
            self.max_val = other.max_val.clone()
            self.n = other.n
            self.initialized = True
            return

        n_a, n_b = self.n, other.n
        n_total = n_a + n_b
        delta = other.mean - self.mean

        if isinstance(n_total, torch.Tensor):
            safe_total = n_total.clamp(min=1)
        else:
            safe_total = max(n_total, 1)
        self.mean = (n_a * self.mean + n_b * other.mean) / safe_total
        self.M2 = self.M2 + other.M2 + delta * delta * n_a * n_b / safe_total
        self.n = n_total
        self.min_val = torch.minimum(self.min_val, other.min_val)
        self.max_val = torch.maximum(self.max_val, other.max_val)

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


class WelfordTensorPerBin:
    """Welford accumulator for 4-D (B, C, F, T) spec tensors, tracking
    mean / std / min / max **per (channel, frequency-bin)** instead of
    per channel.

    Use this when the loss should weight each frequency bin equally —
    in the channel-only WelfordTensor, the high-dynamic-range bins (the
    low-frequency band) dominate the channel-wide mean / std, making
    quiet mode bins effectively invisible to a downstream MSE / MAE
    objective.
    """

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None     # (C, F)
        self.M2: Optional[torch.Tensor] = None       # (C, F)
        self.n: Optional[torch.Tensor] = None        # (C, F)  per-bin count
        self.min_val: Optional[torch.Tensor] = None  # (C, F)
        self.max_val: Optional[torch.Tensor] = None  # (C, F)
        self.initialized: bool = False

    def _initialize(self, n_channels: int, n_freq_bins: int) -> None:
        self.mean = torch.zeros(n_channels, n_freq_bins, dtype=torch.float64)
        self.M2 = torch.zeros(n_channels, n_freq_bins, dtype=torch.float64)
        self.n = torch.zeros(n_channels, n_freq_bins, dtype=torch.int64)
        self.min_val = torch.full(
            (n_channels, n_freq_bins), float("inf"), dtype=torch.float64,
        )
        self.max_val = torch.full(
            (n_channels, n_freq_bins), float("-inf"), dtype=torch.float64,
        )
        self.initialized = True

    def update(self, value: torch.Tensor) -> None:
        """Accept ``(B, C, F, T)`` or ``(C, F, T)``."""
        if value.ndim == 3:
            value = value.unsqueeze(0)
        if value.ndim != 4:
            return
        B, C, F, T = value.shape
        if not self.initialized:
            self._initialize(C, F)
        v = value.permute(1, 2, 0, 3).reshape(C, F, B * T).to(torch.float64)

        if torch.isnan(v).any().item():
            nan_mask = torch.isnan(v)
            n_valid = (~nan_mask).sum(dim=2).to(torch.int64)
            if (n_valid == 0).all():
                return
            safe = v.clone()
            safe[nan_mask] = 0.0
            batch_sum = safe.sum(dim=2)
            batch_sum_sq = (safe * safe).sum(dim=2)
            safe.copy_(v)
            safe[nan_mask] = float("inf")
            batch_min = safe.amin(dim=2)
            safe[nan_mask] = float("-inf")
            batch_max = safe.amax(dim=2)
        else:
            n_valid = torch.full((C, F), B * T, dtype=torch.int64)
            batch_sum = v.sum(dim=2)
            batch_sum_sq = (v * v).sum(dim=2)
            batch_min = v.amin(dim=2)
            batch_max = v.amax(dim=2)

        n_valid_f = n_valid.to(torch.float64)
        safe_n_f = n_valid_f.clamp(min=1)
        batch_mean = batch_sum / safe_n_f
        batch_var = (
            batch_sum_sq / safe_n_f - batch_mean * batch_mean
        ).clamp(min=0)
        batch_M2 = batch_var * n_valid_f

        n_old_f = self.n.to(torch.float64)
        n_total_f = (self.n + n_valid).to(torch.float64).clamp(min=1)
        delta = batch_mean - self.mean
        self.mean = (n_old_f * self.mean + n_valid_f * batch_mean) / n_total_f
        self.M2 = (
            self.M2 + batch_M2
            + delta * delta * n_old_f * n_valid_f / n_total_f
        )
        self.n = self.n + n_valid

        has_data = n_valid > 0
        self.min_val[has_data] = torch.minimum(
            self.min_val[has_data], batch_min[has_data]
        )
        self.max_val[has_data] = torch.maximum(
            self.max_val[has_data], batch_max[has_data]
        )

    def merge(self, other: "WelfordTensorPerBin") -> None:
        if not other.initialized:
            return
        if not self.initialized:
            self.mean = other.mean.clone()
            self.M2 = other.M2.clone()
            self.n = other.n.clone()
            self.min_val = other.min_val.clone()
            self.max_val = other.max_val.clone()
            self.initialized = True
            return
        n_old_f = self.n.to(torch.float64)
        n_new_f = other.n.to(torch.float64)
        n_total_f = (self.n + other.n).to(torch.float64).clamp(min=1)
        delta = other.mean - self.mean
        self.mean = (n_old_f * self.mean + n_new_f * other.mean) / n_total_f
        self.M2 = (
            self.M2 + other.M2
            + delta * delta * n_old_f * n_new_f / n_total_f
        )
        self.n = self.n + other.n
        self.min_val = torch.minimum(self.min_val, other.min_val)
        self.max_val = torch.maximum(self.max_val, other.max_val)

    def compute(self) -> Optional[dict]:
        if not self.initialized or int(self.n.max().item()) < 2:
            return None
        denom = (self.n - 1).clamp(min=1).to(torch.float64)
        std = torch.sqrt(self.M2 / denom)
        return {
            "mean": self.mean.numpy(),
            "std": std.numpy(),
            "min_val": self.min_val.numpy(),
            "max_val": self.max_val.numpy(),
        }


_shared_counter = None
_worker_args = {}


def _init_worker(counter, args):
    global _shared_counter, _worker_args
    _shared_counter = counter
    _worker_args = args


def _worker_fn(chunk):
    return _process_file_chunk(chunk, **_worker_args, counter=_shared_counter)


def _process_file_chunk(
        paths: list[Path],
        signal_names: list[str],
        stft_signals: set[str],
        n_fft: int,
        hop_length: int,
        hdf5_key_map: Optional[dict[str, str]] = None,
        zero_is_missing_signals: Optional[set[str]] = None,
        compute_per_bin_for_stft: bool = False,
        counter=None,
) -> dict[str, dict[str, "WelfordTensor"]]:
    """Process a chunk of HDF5 files, returning per-signal Welford trackers.

    When ``compute_per_bin_for_stft`` is True, an additional per-bin
    Welford tracker is also accumulated for every signal in
    ``stft_signals`` and returned under the ``'log_per_bin'`` key.
    Returned dict keys are ``'raw'``, ``'log'``, optionally
    ``'log_per_bin'``.
    """
    import h5py

    if hdf5_key_map is None:
        hdf5_key_map = {}
    if zero_is_missing_signals is None:
        zero_is_missing_signals = set()

    stft_window = torch.hann_window(n_fft)
    raw_trackers = {name: WelfordTensor() for name in signal_names}
    log_trackers = {name: WelfordTensor() for name in signal_names}
    log_per_bin_trackers: dict[str, WelfordTensorPerBin] = {}
    if compute_per_bin_for_stft:
        log_per_bin_trackers = {
            name: WelfordTensorPerBin() for name in signal_names
            if name in stft_signals
        }

    for path in paths:
        try:
            f = h5py.File(path, "r")
        except OSError:
            continue

        with f:
            for name in signal_names:
                hdf5_key = hdf5_key_map.get(name, name)
                if hdf5_key not in f:
                    continue
                group = f[hdf5_key]
                if "ydata" not in group:
                    continue

                ydata = group["ydata"]
                if ydata.size == 0 or ydata.shape[-1] <= 1:
                    continue

                # For large arrays (videos), subsample via HDF5 slicing
                if ydata.ndim >= 3:
                    data = torch.from_numpy(
                        ydata[::1, ::4, ::4, ::10]).float()
                    data = data.reshape(1, 1, -1)     # (1, 1, N)
                else:
                    # For STFT signals, read only a 1s window to avoid
                    # loading hundreds of MB per file.
                    max_stft_samples = 1_500_000  # ~3s at 500kHz
                    if name in stft_signals and ydata.shape[-1] > max_stft_samples:
                        data = torch.from_numpy(
                            ydata[:, :max_stft_samples]).float()
                    else:
                        data = torch.from_numpy(ydata[:]).float()
                    # HDF5 stores time-series as (C, T) or (T,)
                    if data.ndim == 1:
                        data = data.unsqueeze(0)      # (1, T)
                    data = data.unsqueeze(0)           # (1, C, T)

                    # Compute STFT for spectrogram signals
                    if name in stft_signals:
                        C, T = data.shape[1], data.shape[2]
                        if T >= n_fft:
                            spec = torch.stft(
                                data.squeeze(0),
                                n_fft=n_fft,
                                hop_length=hop_length,
                                window=stft_window,
                                return_complex=True,
                            )
                            data = torch.abs(spec)[:, 1:, :]
                            data = data.unsqueeze(0)
                        else:
                            continue

                if name in zero_is_missing_signals:
                    # Mask positions where the raw value is exactly 0 — these
                    # are "missing data" markers at training time and must
                    # not contribute to mean/std (otherwise they drag the
                    # log-mean down and inflate the log-std dramatically).
                    data = data.clone()
                    data[data == 0] = float('nan')

                raw_trackers[name].update(data)
                log_data = torch.log10(data.clamp(min=-0.99) + 1)
                log_trackers[name].update(log_data)
                # Per-(channel, freq_bin) tracker for STFT signals when
                # requested. log_data here is (B, C, F, T) for STFT
                # signals (constructed above in the STFT branch) — the
                # PerBin accumulator handles its 4-D shape natively.
                if name in log_per_bin_trackers and log_data.ndim == 4:
                    log_per_bin_trackers[name].update(log_data)

        if counter is not None:
            with counter.get_lock():
                counter.value += 1

    out: dict[str, dict[str, object]] = {}
    for name in signal_names:
        entry: dict[str, object] = {
            "raw": raw_trackers[name],
            "log": log_trackers[name],
        }
        if name in log_per_bin_trackers:
            entry["log_per_bin"] = log_per_bin_trackers[name]
        out[name] = entry
    return out


def compute_preprocessing_stats(
        hdf5_paths: list[Path],
        signal_names: list[str],
        output_path: str | Path = "preprocessing_stats.pt",
        max_files: Optional[int] = None,
        stft_signals: Optional[set[str]] = None,
        hdf5_key_map: Optional[dict[str, str]] = None,
        zero_is_missing_signals: Optional[set[str]] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_workers: int = 1,
        compute_per_bin_for_stft: bool = False,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Compute per-modality preprocessing statistics directly from HDF5 files.

    Opens each HDF5 file once, reads the raw data for every requested
    signal, and feeds it to :class:`WelfordTensor` trackers for both raw
    and log-space statistics.  This bypasses the Dataset/DataLoader
    pipeline entirely, avoiding chunking, resampling, and multi-process
    overhead.

    For signals in *stft_signals*, the STFT magnitude spectrogram is
    computed before collecting statistics, matching what the data loader
    produces at training time.

    Parameters
    ----------
    hdf5_paths : list of Path
        Paths to preprocessed HDF5 shot files.
    signal_names : list of str
        Signal names to compute statistics for.
    output_path : str or Path, optional
        Filesystem path for the saved ``.pt`` statistics file.
    max_files : int or None, optional
        Maximum number of files to process.  ``None`` processes all files.
    stft_signals : set of str or None, optional
        Signal names that require STFT before stats computation.
    n_fft : int, optional
        FFT size for STFT computation.  Default is ``1024``.
    hop_length : int, optional
        Hop length for STFT computation.  Default is ``256``.
    num_workers : int, optional
        Number of parallel worker processes.  Default is ``1`` (no
        parallelism).  Each worker processes a disjoint subset of files.

    Returns
    -------
    dict[str, dict[str, dict[str, numpy.ndarray]]]
        Nested dictionary ``{signal_name: {"raw": stats, "log": stats}}``,
        where each *stats* dict contains ``'mean'``, ``'std'``,
        ``'min_val'``, and ``'max_val'`` arrays of shape ``(C,)``.
    """
    from tqdm import tqdm

    if stft_signals is None:
        stft_signals = set()
    if zero_is_missing_signals is None:
        zero_is_missing_signals = set()

    paths = list(hdf5_paths)
    if max_files is not None and max_files < len(paths):
        indices = torch.randperm(len(paths))[:max_files].tolist()
        paths = [paths[i] for i in indices]
        print(f"Subsampling {max_files:,} / {len(hdf5_paths):,} files.")

    # Split files into chunks, one per worker
    num_workers = max(1, num_workers)
    chunk_size = max(1, len(paths) // num_workers)
    file_chunks = [
        paths[i:i + chunk_size]
        for i in range(0, len(paths), chunk_size)
    ]

    if num_workers == 1:
        # Single-process: run with progress bar
        results = []
        for path in tqdm(paths, desc="Files"):
            r = _process_file_chunk(
                [path], signal_names, stft_signals, n_fft, hop_length,
                hdf5_key_map,
                zero_is_missing_signals=zero_is_missing_signals,
                compute_per_bin_for_stft=compute_per_bin_for_stft)
            results.append(r)
    else:
        import multiprocessing as mp
        import time

        _counter = mp.Value("i", 0)
        worker_args = dict(
            signal_names=signal_names,
            stft_signals=stft_signals,
            n_fft=n_fft,
            hop_length=hop_length,
            hdf5_key_map=hdf5_key_map,
            zero_is_missing_signals=zero_is_missing_signals,
            compute_per_bin_for_stft=compute_per_bin_for_stft,
        )

        total = len(paths)
        print(f"Processing {total} files with {len(file_chunks)} workers...")

        pool = mp.Pool(
            num_workers,
            initializer=_init_worker,
            initargs=(_counter, worker_args),
        )
        async_results = [pool.apply_async(_worker_fn, (chunk,))
                         for chunk in file_chunks]

        pbar = tqdm(total=total, desc="Files")
        while not all(r.ready() for r in async_results):
            with _counter.get_lock():
                pbar.n = _counter.value
            pbar.refresh()
            time.sleep(1.0)
        pbar.n = total
        pbar.refresh()
        pbar.close()

        results = [r.get() for r in async_results]
        pool.close()
        pool.join()

    # Merge all worker results. Each worker returns a dict of dicts:
    # ``{name: {'raw': WelfordTensor, 'log': WelfordTensor,
    #          'log_per_bin': WelfordTensorPerBin (optional)}}``.
    raw_merged = {name: WelfordTensor() for name in signal_names}
    log_merged = {name: WelfordTensor() for name in signal_names}
    log_per_bin_merged: dict[str, WelfordTensorPerBin] = {}
    if compute_per_bin_for_stft:
        log_per_bin_merged = {
            name: WelfordTensorPerBin() for name in signal_names
            if stft_signals is not None and name in stft_signals
        }
    for partial in results:
        for name in signal_names:
            if name not in partial:
                continue
            entry = partial[name]
            raw_merged[name].merge(entry["raw"])
            log_merged[name].merge(entry["log"])
            if "log_per_bin" in entry and name in log_per_bin_merged:
                log_per_bin_merged[name].merge(entry["log_per_bin"])

    # Build final stats dict
    final_stats = {}
    for name in signal_names:
        raw_ok = raw_merged[name].initialized
        log_ok = log_merged[name].initialized
        if not raw_ok and not log_ok:
            continue
        final_stats[name] = {}
        if raw_ok:
            final_stats[name]["raw"] = raw_merged[name].compute()
        if log_ok:
            final_stats[name]["log"] = log_merged[name].compute()
        per_bin_tracker = log_per_bin_merged.get(name)
        if per_bin_tracker is not None and per_bin_tracker.initialized:
            per_bin = per_bin_tracker.compute()
            if per_bin is not None:
                # ``log_per_bin`` lives next to ``raw`` / ``log`` —
                # training code reads either depending on its own
                # config; channel-wise (``log``) remains the default.
                final_stats[name]["log_per_bin"] = per_bin

    torch.save(final_stats, output_path)
    print(f"Saved statistics for {len(final_stats)} modalities to {output_path}")
    return final_stats
