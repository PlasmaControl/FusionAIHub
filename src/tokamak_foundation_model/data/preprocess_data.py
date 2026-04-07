import torch
import numpy as np
from pathlib import Path
from typing import Optional


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

        self.mean = (n_a * self.mean + n_b * other.mean) / n_total
        self.M2 = self.M2 + other.M2 + delta * delta * n_a * n_b / n_total
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
        counter=None,
) -> dict[str, tuple[WelfordTensor, WelfordTensor]]:
    """Process a chunk of HDF5 files, returning per-signal Welford trackers."""
    import h5py

    stft_window = torch.hann_window(n_fft)
    raw_trackers = {name: WelfordTensor() for name in signal_names}
    log_trackers = {name: WelfordTensor() for name in signal_names}

    for path in paths:
        try:
            f = h5py.File(path, "r")
        except OSError:
            continue

        with f:
            for name in signal_names:
                if name not in f:
                    continue
                group = f[name]
                if "ydata" not in group:
                    continue

                ydata = group["ydata"]
                if ydata.size == 0:
                    continue

                # For large arrays (videos), subsample via HDF5 slicing
                if ydata.ndim >= 3:
                    data = torch.from_numpy(
                        ydata[::1, ::2, ::2, ::5]).float()
                    data = data.reshape(1, 1, -1)     # (1, 1, N)
                else:
                    data = torch.from_numpy(ydata[:]).float()
                    if data.ndim == 1:
                        data = data.unsqueeze(1)      # (T, 1)
                    data = data.T.unsqueeze(0)        # (1, C, T)

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

                if torch.isnan(data).any():
                    continue

                raw_trackers[name].update(data)
                log_data = torch.log10(data.clamp(min=-0.99) + 1)
                log_trackers[name].update(log_data)

        if counter is not None:
            with counter.get_lock():
                counter.value += 1

    return {name: (raw_trackers[name], log_trackers[name])
            for name in signal_names}


def compute_preprocessing_stats(
        hdf5_paths: list[Path],
        signal_names: list[str],
        output_path: str | Path = "preprocessing_stats.pt",
        max_files: Optional[int] = None,
        stft_signals: Optional[set[str]] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_workers: int = 1,
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
                [path], signal_names, stft_signals, n_fft, hop_length)
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

    # Merge all worker results
    raw_merged = {name: WelfordTensor() for name in signal_names}
    log_merged = {name: WelfordTensor() for name in signal_names}
    for partial in results:
        for name in signal_names:
            if name in partial:
                raw_merged[name].merge(partial[name][0])
                log_merged[name].merge(partial[name][1])

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

    torch.save(final_stats, output_path)
    print(f"Saved statistics for {len(final_stats)} modalities to {output_path}")
    return final_stats
