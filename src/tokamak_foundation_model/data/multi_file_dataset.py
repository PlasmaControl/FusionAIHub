"""
Multi-file dataset for large-scale tokamak model training with HDF5 files.

Design goals
------------
* **Bounded file descriptors**: an LRU cache keeps at most *max_open_files*
  HDF5 handles open per worker, regardless of how many files exist.
* **Sequential I/O**: :class:`TwoLevelSampler` shuffles file order each epoch
  but accesses chunks sequentially *within* each file, maximising OS page-cache
  hit rate and minimising seek overhead.
* **Fast startup**: file lengths (number of chunks) are written to a sidecar
  ``.pt`` file on the first run and reloaded instantly on subsequent runs —
  no HDF5 opens at init time after the first run.
* **No code duplication**: :class:`TokamakMultiFileDataset` subclasses
  :class:`~tokamak_foundation_model.data.data_loader.TokamakH5Dataset` and
  reuses all signal / movie loading methods unchanged.

Typical usage
-------------
>>> from tokamak_foundation_model.data.multi_file_dataset import (
...     TokamakMultiFileDataset, TwoLevelSampler, make_dataloader)
>>> from torch.utils.data import DataLoader
>>>
>>> dataset = TokamakMultiFileDataset(
...     hdf5_paths=sorted(Path("data/").glob("*_processed.h5")),
...     input_signals=["ece", "mhr", "co2"],
...     target_signals=["ece", "mhr", "co2"],
...     lengths_cache_path="dataset_lengths.pt",
...     max_open_files=100,
... )
>>> loader = make_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True)
>>> for batch in loader:
...     ...
"""

from __future__ import annotations

import collections
import copy
import os
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from tokamak_foundation_model.data.data_loader import (
    TokamakH5Dataset,
    collate_fn,
    collate_fn_prediction,
)


class TokamakMultiFileDataset(TokamakH5Dataset):
    """
    Torch Dataset spanning many HDF5 shot files with an LRU file handle cache.

    Subclasses :class:`TokamakH5Dataset` and inherits all data loading logic.
    The key differences are:

    * A single dataset object covers **all** files instead of one per file.
    * Open file handles are managed via a per-worker LRU cache bounded by
      *max_open_files*, so file descriptor usage stays constant regardless of dataset size.
    * File lengths can be persisted to a sidecar file to avoid re-scanning
      HDF5 files at startup.

    Parameters
    ----------
    hdf5_paths : list of str or Path
        Ordered list of HDF5 shot files to include.
    chunk_duration_s : float, optional
        Duration of each time window in seconds.  Default ``0.5``.
    max_duration_s : float, optional
        Maximum shot duration to consider.  Default ``12.0``.
    n_fft : int, optional
        FFT size for STFT computation.  Default ``1024``.
    hop_length : int, optional
        STFT hop size in samples.  Default ``256``.
    preprocessing_stats : dict or None, optional
        Statistics dict as returned by
        :func:`~tokamak_foundation_model.data.data_loader.compute_preprocessing_stats`.
    prediction_mode : bool, optional
        If ``True``, return ``{'inputs': …, 'targets': …}`` pairs.
    prediction_horizon_s : float, optional
        Target window duration in prediction mode.  Default ``0.2``.
    input_signals : list of str or None, optional
        Modality names to include as inputs.
    target_signals : list of str or None, optional
        Modality names to include as targets (prediction mode only).
    lengths_cache_path : str or Path or None, optional
        Path to a ``.pt`` sidecar file used to cache per-file chunk counts.
        On the first call the lengths are computed and written here; on
        subsequent calls they are loaded instantly.  ``None`` disables caching.
    max_open_files : int, optional
        Maximum number of HDF5 file handles kept open simultaneously **per
        worker**.  Default ``100``.  Limits file descriptor usage; datasets are
        stored contiguously so there is no active HDF5 chunk cache.

    Attributes
    ----------
    hdf5_paths : list of Path
        All file paths passed at construction.
    _valid_indices : list of int
        Indices into *hdf5_paths* for files that were successfully read.
    _valid_lengths : list of int
        Number of chunks in each valid file.
    _cumulative_lengths : numpy.ndarray
        Prefix-sum of *_valid_lengths*, used for O(log N) index mapping.
    """

    def __init__(
            self,
            hdf5_paths: list[str | Path],
            chunk_duration_s: float = 0.5,
            max_duration_s: float = 12.0,
            n_fft: int = 1024,
            hop_length: int = 256,
            preprocessing_stats: Optional[dict] = None,
            prediction_mode: bool = False,
            prediction_horizon_s: float = 0.2,
            input_signals: Optional[list[str]] = None,
            target_signals: Optional[list[str]] = None,
            lengths_cache_path: Optional[str | Path] = None,
            max_open_files: int = 512,
            step_size_s: Optional[float] = None,
            warmup_s: float = 0.0,
    ):
        # Set up all instance attributes that parent methods rely on.
        # We deliberately skip super().__init__() because it expects a single
        # hdf5_path and opens that file — neither applies here.
        self.signal_configs = copy.deepcopy(self.SIGNAL_CONFIGS)
        self.movie_configs = copy.deepcopy(self.MOVIE_CONFIGS)

        self.chunk_duration_s = chunk_duration_s
        self.step_size_s = step_size_s if step_size_s is not None else chunk_duration_s
        self.warmup_s = warmup_s
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preprocessing_stats = preprocessing_stats or {}
        self.prediction_mode = prediction_mode
        self.prediction_horizon_s = prediction_horizon_s
        self.input_signals = input_signals or ["ece", "co2", "mhr"]
        self.target_signals = target_signals or ["mse", "ts_core_density"]
        self.n_freq_bins = n_fft // 2 + 1
        self.stft_window = torch.hann_window(n_fft)
        # h5_file is not kept persistently; it is set in __getitem__ via the
        # LRU cache so that the parent's _getitem_standard / _getitem_prediction
        # methods find it on self.
        self.h5_file = None

        self._update_preprocessing_stats()

        # --- multi-file state ------------------------------------------------
        self.hdf5_paths = [Path(p) for p in hdf5_paths]
        self.max_open_files = max_open_files
        # LRU cache: key = index into hdf5_paths, value = open h5py.File.
        # OrderedDict provides O(1) move_to_end for LRU bookkeeping.
        self._file_handles: collections.OrderedDict[int, h5py.File] = (
            collections.OrderedDict()
        )
        # Per-worker profiling counters (reset in __setstate__).
        self._prof_hits = 0
        self._prof_opens = 0
        self._prof_open_s = 0.0
        self._prof_close_s = 0.0
        self._prof_getitem_calls = 0
        self._prof_getitem_s = 0.0
        self._prof_load_s = 0.0
        self._prof_process_s = 0.0
        self._prof_movie_s = 0.0
        self._prof_log_every = 50

        # --- lengths ---------------------------------------------------------
        file_lengths = self._load_or_compute_lengths(
            max_duration_s=max_duration_s,
            lengths_cache_path=lengths_cache_path,
        )

        valid = [
            (i, length) for i, length in enumerate(file_lengths) if length > 0
        ]
        n_skipped = len(self.hdf5_paths) - len(valid)
        if n_skipped:
            print(
                f"Warning: {n_skipped} file(s) skipped (unreadable or empty)."
            )

        self._valid_indices: list[int] = [i for i, _ in valid]
        self._valid_lengths: list[int] = [length for _, length in valid]
        self._cumulative_lengths = np.concatenate(
            [[0], np.cumsum(self._valid_lengths)]
        ).astype(np.int64)

    # -------------------------------------------------------------------------
    # Length caching
    # -------------------------------------------------------------------------

    def _load_or_compute_lengths(
            self,
            max_duration_s: float,
            lengths_cache_path: Optional[Path],
    ) -> list[int]:
        """
        Return per-file chunk counts, loading from cache when available.

        Under DDP only rank 0 reads/computes/writes the cache; all other
        ranks receive the result via ``dist.broadcast_object_list``. This
        avoids 8 ranks hammering the Lustre MDS with redundant scans and
        prevents concurrent ``torch.save`` calls from corrupting the
        sidecar zip file.

        Parameters
        ----------
        max_duration_s : float
            Cap on shot duration used when computing chunk counts.
        lengths_cache_path : Path or None
            Path to the sidecar cache file.  If the file exists *and* its
            stored path list matches the current ``hdf5_paths``, the cached
            lengths are returned directly without opening any HDF5 file.
            Otherwise lengths are computed and written to this path
            atomically (``.tmp`` + ``replace``).

        Returns
        -------
        list of int
            Number of chunks for each path in ``self.hdf5_paths``.
            Files that could not be opened have length ``0``.
        """
        import torch.distributed as dist
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0

        paths_as_str = [str(p) for p in self.hdf5_paths]
        lengths: Optional[list[int]] = None

        if rank == 0:
            if lengths_cache_path is not None:
                cache_path = Path(lengths_cache_path)
                if cache_path.exists():
                    try:
                        cache = torch.load(cache_path, weights_only=False)
                        if cache.get("paths") == paths_as_str:
                            print(f"Loaded file lengths from cache: {cache_path}")
                            lengths = cache["lengths"]
                    except Exception as e:
                        print(
                            f"Warning: lengths cache at {cache_path} is "
                            f"unreadable ({e}); recomputing."
                        )

            if lengths is None:
                lengths = []
                for path in tqdm(self.hdf5_paths, desc="Computing file lengths"):
                    try:
                        with h5py.File(path, "r") as f:
                            duration = min(self._compute_duration(f), max_duration_s)
                        # Subtract warmup: usable duration starts after warmup_s
                        duration = duration - self.warmup_s
                        if duration <= 0.0:
                            length = 0
                        elif self.prediction_mode:
                            total_window = (
                                    self.chunk_duration_s + self.prediction_horizon_s
                            )
                            length = max(0, int(np.floor(
                                (duration - total_window) / self.step_size_s
                            )) + 1)
                        else:
                            if duration < self.chunk_duration_s:
                                length = 0
                            else:
                                length = int(np.floor(
                                    (duration - self.chunk_duration_s) / self.step_size_s
                                )) + 1
                    except OSError as e:
                        print(f"Warning: could not open {path}: {e}")
                        length = 0
                    lengths.append(length)

                if lengths_cache_path is not None:
                    # Atomic write: write to .tmp then rename, so a crashed
                    # write never leaves a half-written zip that the next
                    # torch.load would barf on.
                    tmp_path = Path(str(lengths_cache_path) + ".tmp")
                    torch.save(
                        {"paths": paths_as_str, "lengths": lengths}, tmp_path,
                    )
                    tmp_path.replace(Path(lengths_cache_path))
                    print(f"Saved file lengths to cache: {lengths_cache_path}")

        if distributed:
            payload = [lengths] if rank == 0 else [None]
            dist.broadcast_object_list(payload, src=0)
            lengths = payload[0]

        return lengths

    # -------------------------------------------------------------------------
    # LRU file handle cache
    # -------------------------------------------------------------------------

    def _get_file_handle(self, file_idx: int) -> h5py.File:
        """
        Return an open HDF5 handle for *file_idx*, managing an LRU cache.

        If the handle is already cached it is promoted to most-recently-used.
        If the cache is full the least-recently-used handle is closed and
        evicted before opening the new file.

        Parameters
        ----------
        file_idx : int
            Index into ``self.hdf5_paths``.

        Returns
        -------
        h5py.File
            Open, ready-to-read file handle.
        """
        if file_idx in self._file_handles:
            self._file_handles.move_to_end(file_idx)
            self._prof_hits += 1
            return self._file_handles[file_idx]

        # Evict LRU entry when at capacity
        if len(self._file_handles) >= self.max_open_files:
            _, lru_handle = self._file_handles.popitem(last=False)
            t0 = time.perf_counter()
            lru_handle.close()
            self._prof_close_s += time.perf_counter() - t0

        # rdcc_nbytes=0 disables the per-file HDF5 chunk cache (default 1 MB).
        # Sequential reads don't benefit from it, and keeping it enabled with
        # many open files wastes significant CPU RAM.
        t0 = time.perf_counter()
        handle = h5py.File(
            self.hdf5_paths[file_idx], "r", rdcc_nbytes=0, rdcc_nslots=0
        )
        self._prof_open_s += time.perf_counter() - t0
        self._prof_opens += 1
        self._file_handles[file_idx] = handle
        return handle

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def _open_hdf5(self) -> None:
        """No-op: file handles are opened on demand via the LRU cache."""
        pass

    def __len__(self) -> int:
        return int(self._cumulative_lengths[-1])

    def __getitem__(self, idx: int) -> dict:
        """
        Return the data chunk at global position *idx*.

        Maps *idx* to a ``(file, chunk)`` pair via binary search on the
        cumulative length array, retrieves the file handle from the LRU cache,
        and delegates to the parent's standard or prediction loader.
        """
        t_call_start = time.perf_counter()
        # O(log N) mapping: global idx → position in valid-file list
        pos = int(np.searchsorted(self._cumulative_lengths, idx + 1) - 1)
        file_idx = self._valid_indices[pos]
        chunk_idx = idx - int(self._cumulative_lengths[pos])

        # Expose the handle on self so parent methods (_getitem_standard,
        # _getitem_prediction, _load_signal_raw, …) can find it.
        # Safe: each DataLoader worker owns its own copy of this object.
        self.h5_file = self._get_file_handle(file_idx)

        if self.prediction_mode:
            result = self._getitem_prediction(chunk_idx)
        else:
            result = self._getitem_standard(chunk_idx)

        self._prof_getitem_calls += 1
        self._prof_getitem_s += time.perf_counter() - t_call_start
        if self._prof_getitem_calls % self._prof_log_every == 0:
            n = self._prof_getitem_calls
            total_io = self._prof_open_s + self._prof_close_s
            print(
                f"[w-pid{os.getpid()}] prof_worker calls={n} "
                f"avg_getitem_ms={1000*self._prof_getitem_s/n:.1f} "
                f"hits={self._prof_hits} cold_opens={self._prof_opens} "
                f"avg_open_ms={1000*self._prof_open_s/max(self._prof_opens,1):.1f} "
                f"avg_close_ms={1000*self._prof_close_s/max(self._prof_opens,1):.1f} "
                f"sum_open_s={self._prof_open_s:.2f} "
                f"sum_close_s={self._prof_close_s:.2f} "
                f"sum_load_s={self._prof_load_s:.2f} "
                f"sum_process_s={self._prof_process_s:.2f} "
                f"sum_movie_s={self._prof_movie_s:.2f} "
                f"cache_size={len(self._file_handles)}",
                flush=True,
            )
        return result

    # -------------------------------------------------------------------------
    # Pickling (DataLoader worker processes)
    # -------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Close all open handles before the object is pickled to a worker."""
        state = self.__dict__.copy()
        for handle in state["_file_handles"].values():
            handle.close()
        state["_file_handles"] = collections.OrderedDict()
        state["h5_file"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """
        Restore state in the worker process (file handles re-opened on demand).
        """
        self.__dict__.update(state)
        self._prof_hits = 0
        self._prof_opens = 0
        self._prof_open_s = 0.0
        self._prof_close_s = 0.0
        self._prof_getitem_calls = 0
        self._prof_getitem_s = 0.0
        self._prof_load_s = 0.0
        self._prof_process_s = 0.0
        self._prof_movie_s = 0.0
        self._prof_log_every = 50


# =============================================================================
# Two-level sampler
# =============================================================================

class TwoLevelSampler(Sampler):
    """
    Epoch-level sampler that maximises sequential HDF5 access.

    Each epoch the list of files is shuffled (or kept in order when
    ``shuffle=False``), and then the chunk indices for each file are yielded
    **sequentially**.  This means the DataLoader sees a different global order
    each epoch while each individual file is always read front-to-back,
    keeping HDF5 chunk cache utilisation high and the LRU file handle cache
    effective.

    Parameters
    ----------
    dataset : TokamakMultiFileDataset
        The dataset to sample from.
    shuffle : bool, optional
        If ``True`` (default), shuffle file order at each iteration.
    """

    def __init__(self, dataset: TokamakMultiFileDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        n_files = len(self.dataset._valid_lengths)
        file_order = (
            torch.randperm(n_files).tolist() if self.shuffle
            else list(range(n_files))
        )
        for pos in file_order:
            start = int(self.dataset._cumulative_lengths[pos])
            end = int(self.dataset._cumulative_lengths[pos + 1])
            yield from range(start, end)


# =============================================================================
# DDP-aware two-level sampler (file-level sharding)
# =============================================================================


class DistributedTwoLevelSampler(Sampler):
    """
    DDP-aware file-level sharding with sequential intra-file iteration.

    Combines :class:`TwoLevelSampler`'s file-sequential locality with
    DDP-aware sharding. The file list is partitioned across ranks **once**
    at construction (round-robin: rank ``r`` owns positions
    ``r, r + N, r + 2N, …``). Each rank then iterates **its own** files,
    front-to-back within each file, with per-epoch shuffling of the
    rank's own file order via :meth:`set_epoch`.

    Why this matters
    ----------------
    PyTorch's :class:`~torch.utils.data.distributed.DistributedSampler`
    shards *chunk indices* across ranks, which scatters each rank's
    accesses across the entire file pool and defeats the per-worker LRU
    file-handle cache in :class:`TokamakMultiFileDataset`. On the live
    DIII-D dataset (~7900 shots, LRU=100) this collapses cache hit rate
    to ~1 % and makes HDF5 ``open()`` the dominant per-step cost under
    DDP (observed ~12 s/step on a 2-GPU DDP run vs. ~1 s/step single-GPU
    at the same batch size).

    Static (vs. rotated) sharding
    -----------------------------
    The file-to-rank assignment is fixed for the lifetime of the
    sampler. Each rank only ever sees its own subset of files. This
    keeps the LRU file-handle cache warm across epochs (especially with
    ``persistent_workers=True``). For many-epoch training the cross-rank
    data diversity that rotated sharding would buy is dominated by
    within-rank re-exposure; use PyTorch's ``DistributedSampler`` if
    you'd rather have every rank eventually see every file at the cost
    of cache locality.

    Length parity across ranks
    --------------------------
    File sizes vary; per-rank totals may differ. Every rank truncates to
    the minimum per-rank chunk count so DDP all-reduce stays in
    lockstep. Padding (``drop_last=False``) is not supported.

    Parameters
    ----------
    dataset : TokamakMultiFileDataset
        Dataset with ``_valid_lengths`` and ``_cumulative_lengths``.
    num_replicas : int
        World size.
    rank : int
        This rank's index in ``[0, num_replicas)``.
    shuffle : bool, optional
        Per-epoch shuffle of the rank's own file order. Default
        ``True``.
    seed : int, optional
        RNG seed. The per-epoch RNG uses ``seed + epoch``. Default ``0``.
    drop_last : bool, optional
        Must be ``True``. Present for API compatibility with
        ``DistributedSampler``. Default ``True``.
    """

    def __init__(
        self,
        dataset: "TokamakMultiFileDataset",
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas < 1:
            raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
        if not (0 <= rank < num_replicas):
            raise ValueError(
                f"rank {rank} not in [0, num_replicas={num_replicas})"
            )
        n_files = len(dataset._valid_lengths)
        if num_replicas > n_files:
            raise ValueError(
                f"num_replicas={num_replicas} exceeds n_files={n_files}; "
                f"cannot shard."
            )
        if not drop_last:
            raise NotImplementedError(
                "drop_last=False (padded sampling) is not supported. "
                "Pass drop_last=True so every rank sees the same number "
                "of samples per epoch."
            )

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = True
        self.epoch = 0

        # Static round-robin partition of the *valid* file list.
        self._rank_file_positions: list[int] = list(
            range(self.rank, n_files, self.num_replicas)
        )

        # Pre-compute equal per-rank chunk count = min over ranks.
        per_rank_totals = [
            sum(int(dataset._valid_lengths[p])
                for p in range(r, n_files, self.num_replicas))
            for r in range(self.num_replicas)
        ]
        self._num_samples = min(per_rank_totals)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed per-epoch shuffles. Mirrors
        :meth:`torch.utils.data.distributed.DistributedSampler.set_epoch`.
        Call once per training epoch before iterating."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(
                len(self._rank_file_positions), generator=g,
            ).tolist()
            file_order = [self._rank_file_positions[i] for i in perm]
        else:
            file_order = list(self._rank_file_positions)

        yielded = 0
        for pos in file_order:
            start = int(self.dataset._cumulative_lengths[pos])
            end = int(self.dataset._cumulative_lengths[pos + 1])
            for chunk_idx in range(start, end):
                if yielded >= self._num_samples:
                    return
                yield chunk_idx
                yielded += 1


# =============================================================================
# Convenience factory
# =============================================================================

def make_dataloader(
        dataset: TokamakMultiFileDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
) -> DataLoader:
    """
    Build a DataLoader wired with :class:`TwoLevelSampler`.

    Parameters
    ----------
    dataset : TokamakMultiFileDataset
        Dataset to wrap.
    batch_size : int, optional
        Samples per batch.  Default ``32``.
    num_workers : int, optional
        Number of DataLoader worker processes.  Default ``4``.
    shuffle : bool, optional
        Whether to shuffle file order each epoch.  Default ``True``.
    pin_memory : bool, optional
        Pin CPU tensors to accelerate CPU→GPU transfer.  Default ``True``.
    prefetch_factor : int, optional
        Batches to prefetch per worker, overlapping I/O with GPU work.
        Default ``2``.

    Returns
    -------
    DataLoader
    """
    sampler = TwoLevelSampler(dataset, shuffle=shuffle)
    fn = collate_fn_prediction if dataset.prediction_mode else collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=fn,
        pin_memory=pin_memory,
        persistent_workers=False,  # TODO: validate if this affects the performance.
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def filter_video_present_files(
    paths: list[Path],
    camera_names: list[str],
    cache_path: Optional[Path] = None,
) -> list[Path]:
    """Return only paths whose HDF5 has non-empty data for any camera.

    Used at trainer startup to drop shots without video data when
    training with ``--use_video``. The TwoLevelSampler accesses chunks
    sequentially within each file, so a batch is effectively one
    file's chunks; if that file has no tangtv, every sample's
    ``tangtv_valid=0`` and the masked video loss reports 0 with no
    gradient signal. Filtering up-front guarantees every batch
    contributes to video learning.

    Parameters
    ----------
    paths : list of Path
        HDF5 shot files to filter.
    camera_names : list of str
        Camera names (e.g. ``["tangtv"]``) to check for. A shot is
        kept if **any** requested camera has non-empty ``ydata`` and
        a sufficiently long ``xdata`` (>=2 timestamps).
    cache_path : Path or None, optional
        If given, the result is keyed by ``(paths, sorted cameras)``
        and persisted as a sidecar ``.pt`` file. On the next call
        with the same ``(paths, cameras)``, no HDF5 files are opened.

    Returns
    -------
    list of Path
        The subset of ``paths`` with at least one camera present.
        Order is preserved.
    """
    import torch.distributed as dist
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    paths_key = tuple(str(p) for p in paths)
    cameras_key = tuple(sorted(camera_names))
    video_present: Optional[list[str]] = None

    if rank == 0:
        if cache_path is not None and cache_path.exists():
            try:
                cache = torch.load(cache_path, weights_only=False)
                if (
                    cache.get("paths_key") == paths_key
                    and cache.get("cameras_key") == cameras_key
                ):
                    video_present = list(cache["video_present"])
            except Exception:
                # Corrupt or unreadable cache — fall through to rescan.
                video_present = None

        if video_present is None:
            print(
                f"Scanning {len(paths)} files for {cameras_key} video presence "
                "(cache miss)..."
            )
            video_present = []
            for p in tqdm(paths, desc="Video presence scan"):
                try:
                    with h5py.File(p, "r") as f:
                        for cam in camera_names:
                            if cam not in f or "ydata" not in f[cam]:
                                continue
                            yd = f[cam]["ydata"]
                            xd = f[cam].get("xdata")
                            if (
                                yd.size > 0
                                and yd.ndim == 4
                                and xd is not None
                                and xd.size >= 2
                            ):
                                video_present.append(str(p))
                                break
                except Exception as e:
                    print(f"  skipping {p.name}: {e}")

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = Path(str(cache_path) + ".tmp")
                torch.save(
                    {
                        "paths_key": paths_key,
                        "cameras_key": cameras_key,
                        "video_present": video_present,
                    },
                    tmp_path,
                )
                tmp_path.replace(Path(cache_path))
                print(f"Saved video-presence cache to {cache_path}")

    if distributed:
        payload = [video_present] if rank == 0 else [None]
        dist.broadcast_object_list(payload, src=0)
        video_present = payload[0]

    present = set(video_present)
    return [p for p in paths if str(p) in present]
