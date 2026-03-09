import torch
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from .multi_file_dataset import TokamakMultiFileDataset
from .data_loader import collate_fn, collate_fn_prediction


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
        dataset: TokamakMultiFileDataset,
        output_path: str | Path = "preprocessing_stats.pt",
        batch_size: int = 1,
        num_workers: int = 0,
        max_chunks: Optional[int] = 10_000,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute per-modality preprocessing statistics over a dataset.

    Accumulates running statistics with :class:`WelfordTensor` and saves the
    result to *output_path* via :func:`torch.save`.  Only modalities that
    appear in the loaded batches are included in the output.

    Parameters
    ----------
    dataset : TokamakMultiFileDataset
        Dataset to compute statistics over.
    output_path : str or Path, optional
        Filesystem path for the saved ``.pt`` statistics file.
        Default is ``"preprocessing_stats.pt"``.
    batch_size : int, optional
        Batch size for the internal DataLoader.  Default is ``1``.
    num_workers : int, optional
        Number of DataLoader worker processes.  Default is ``0`` (main
        process only).  Workers add IPC overhead that outweighs any benefit
        for this CPU-only, I/O-bound task.
    max_chunks : int or None, optional
        Maximum number of chunks to sample from the dataset.  A random
        subset of this size is drawn without replacement.  ``None`` means
        use the full dataset.  Default is ``10_000``, which gives accurate
        statistics in ~1-2 hours instead of hundreds of hours.

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
    from tqdm import tqdm

    # Use instance-level configs (deep copies that may have been modified).
    signal_configs = dataset.signal_configs
    movie_configs = dataset.movie_configs

    welford_stats = {
        cfg.name: WelfordTensor()
        for cfg in signal_configs + movie_configs}

    n_total = len(dataset)
    if max_chunks is not None and max_chunks < n_total:
        indices = torch.randperm(n_total)[:max_chunks].tolist()
        print(f"Subsampling {max_chunks:,} / {n_total:,} chunks for statistics.")
    else:
        indices = list(range(n_total))

    collate = collate_fn_prediction if dataset.prediction_mode else collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(indices),
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
    )

    for batch in tqdm(dataloader, total=len(indices) // batch_size):
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
