"""Base class for datasets that defer file opening until worker processes."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch.utils.data import Dataset, get_worker_info


class LazyFileDataset(Dataset, ABC):
    """Abstract base class for datasets that defer file opening.

    This class provides the foundation for datasets that need to:
    - Defer file opening until worker processes are initialized
    - Handle memory-mapped files safely across multiple workers
    - Manage file handles efficiently in multiprocessing environments

    The pattern ensures that each DataLoader worker has its own file handles,
    preventing issues with shared file descriptors and memory mapping.
    """

    def __init__(
        self,
        file_paths: Union[str, list[str]],
        validate_on_init: bool = True,
        max_open_files: Optional[int] = None,
    ) -> None:
        """Initialize the lazy file dataset.

        Parameters
        ----------
        file_paths : Union[str, list[str]]
            Path or list of paths to data files.
        validate_on_init : bool, optional
            Whether to validate file existence and basic format on
            initialization, by default True.
        max_open_files : Optional[int], optional
            Maximum number of files to keep open simultaneously. If None,
            no limit is imposed, by default None.
        """
        super().__init__()

        # Normalize file paths to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = file_paths

        # Configuration
        self.validate_on_init = validate_on_init
        self.max_open_files = max_open_files

        # Worker-specific state (None until worker_init is called)
        self._opened_files = None
        self._worker_id = None
        self._is_initialized = False

        # File metadata (populated during construction)
        self.file_metadata = []

        # Initialize metadata and validate files if requested
        if self.validate_on_init:
            self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initialize file metadata without opening files permanently.

        This method should inspect files to gather necessary metadata
        (like shapes, sample counts, etc.) while minimizing resource usage.
        """
        self.file_metadata = []

        for file_path in self.file_paths:
            try:
                metadata = self._inspect_file(file_path)
                self.file_metadata.append(metadata)
            except Exception as e:
                raise ValueError(
                    f"Failed to inspect file {file_path}: {e}"
                ) from e

    @abstractmethod
    def _inspect_file(self, file_path: str) -> dict[str, Any]:
        """Inspect a single file to extract metadata.

        This method should be implemented by subclasses to extract
        necessary metadata from files without keeping them open.
        The implementation should examine the actual data available
        in the file and extract relevant information like shapes,
        available keys, and any other format-specific metadata.

        Parameters
        ----------
        file_path : str
            Path to the file to inspect.

        Returns
        -------
        dict[str, Any]
            Dictionary containing file metadata. Should include at minimum:
            - 'path': str - The file path
            - 'valid': bool - Whether the file is valid
            - 'available_keys': list[str] - Keys/datasets available in the file
            Additional keys depend on the specific file format and available
            data.
        """
        pass

    @abstractmethod
    def _open_file(self, file_path: str) -> Any:
        """Open a file for reading.

        This method should be implemented by subclasses to open files
        in the appropriate format (joblib, hdf5, etc.) with proper
        memory mapping or lazy loading.

        Parameters
        ----------
        file_path : str
            Path to the file to open.

        Returns
        -------
        Any
            Opened file object or data structure.
        """
        pass

    @abstractmethod
    def _close_file(self, file_handle: Any) -> None:
        """Close a file handle.

        This method should be implemented by subclasses to properly
        close file handles and free resources.

        Parameters
        ----------
        file_handle : Any
            File handle to close.
        """
        pass

    def worker_init(self) -> None:
        """Initialize the dataset in a worker process.

        This method should be called once in each DataLoader worker process
        to open files with worker-specific handles. It's typically called
        from a worker_init_fn passed to DataLoader.
        """
        if self._is_initialized:
            warnings.warn(
                "Dataset already initialized in this worker. "
                "Skipping re-initialization."
            )
            return

        # Get worker info if available
        worker_info = get_worker_info()
        self._worker_id = worker_info.id if worker_info else 0

        # Open files in this worker
        self._opened_files = []

        try:
            for file_path in self.file_paths:
                file_handle = self._open_file(file_path)
                self._opened_files.append(file_handle)
        except Exception as e:
            # Clean up any partially opened files
            self._cleanup_files()
            raise RuntimeError(
                f"Failed to open files in worker {self._worker_id}: {e}"
            ) from e

        self._is_initialized = True

    def _cleanup_files(self) -> None:
        """Clean up opened file handles."""
        if self._opened_files is not None:
            for file_handle in self._opened_files:
                if file_handle is not None:
                    try:
                        self._close_file(file_handle)
                    except Exception as e:
                        warnings.warn(f"Failed to close file handle: {e}")
            self._opened_files = None
        self._is_initialized = False

    def __del__(self) -> None:
        """Cleanup when dataset is garbage collected."""
        self._cleanup_files()

    def _ensure_initialized(self) -> None:
        """Ensure the dataset is initialized in the current worker.

        Raises
        ------
        RuntimeError
            If the dataset has not been initialized in the current worker.
        """
        if not self._is_initialized or self._opened_files is None:
            raise RuntimeError(
                f"Dataset not initialized in this worker. Ensure you call "
                f"dataset.worker_init() in your worker_init_fn. "
                f"Worker ID: {self._worker_id}"
            )

    def get_file_handle(
        self,
        file_index: int,
    ) -> Any:
        """Get the file handle for a specific file index.

        Parameters
        ----------
        file_index : int
            Index of the file in self.file_paths.

        Returns
        -------
        Any
            Opened file handle.

        Raises
        ------
        RuntimeError
            If dataset not initialized or invalid file index.
        """
        self._ensure_initialized()

        if not (0 <= file_index < len(self._opened_files)):
            raise IndexError(
                f"File index {file_index} out of range. "
                f"Available files: 0-{len(self._opened_files) - 1}"
            )

        return self._opened_files[file_index]

    def get_file_metadata(
        self,
        file_index: int,
    ) -> dict[str, Any]:
        """Get metadata for a specific file.

        Parameters
        ----------
        file_index : int
            Index of the file in self.file_paths.

        Returns
        -------
        dict[str, Any]
            File metadata dictionary.
        """
        if not (0 <= file_index < len(self.file_metadata)):
            raise IndexError(
                f"File index {file_index} out of range. "
                f"Available files: 0-{len(self.file_metadata) - 1}"
            )

        return self.file_metadata[file_index]

    @property
    def num_files(self) -> int:
        """Get the number of files in the dataset.

        Returns
        -------
        int
            Number of files.
        """
        return len(self.file_paths)

    @property
    def is_initialized(self) -> bool:
        """Check if the dataset is initialized in the current worker.

        Returns
        -------
        bool
            True if initialized, False otherwise.
        """
        return self._is_initialized

    @property
    def worker_id(self) -> Optional[int]:
        """Get the current worker ID.

        Returns
        -------
        Optional[int]
            Worker ID if in a worker process, None otherwise.
        """
        return self._worker_id

    def get_sample_info(self, sample_idx: int = 0) -> dict[str, Any]:
        """Get information about a sample without worker initialization.

        Parameters
        ----------
        sample_idx : int, optional
            Global sample index, by default 0.

        Returns
        -------
        dict[str, Any]
            Dictionary with sample information including shapes.
        """
        if hasattr(self, "peek_sample"):
            # For file-based datasets, use peek_sample
            if sample_idx >= len(self):
                raise IndexError(f"Sample index {sample_idx} out of range")

            # Find which file and subsequence this sample belongs to
            file_idx, start_idx, end_idx = self.subseq_index[sample_idx]

            # Get shapes using peek method
            try:
                sample_input, sample_target = self.peek_sample(
                    file_idx=file_idx, subseq_idx=0
                )

                if isinstance(sample_input, dict):
                    input_shape = {
                        key: tensor.shape
                        for key, tensor in sample_input.items()
                    }
                else:
                    input_shape = sample_input.shape

                if isinstance(sample_target, dict):
                    target_shape = {
                        key: tensor.shape
                        for key, tensor in sample_target.items()
                    }
                else:
                    target_shape = sample_target.shape

                return {
                    "sample_idx": sample_idx,
                    "file_idx": file_idx,
                    "input_shape": input_shape,
                    "target_shape": target_shape,
                    "subseq_start": start_idx,
                    "subseq_end": end_idx,
                    "is_multi_input": getattr(self, "is_multi_input", False),
                    "is_multi_target": getattr(self, "is_multi_target", False),
                }
            except Exception as e:
                return {
                    "sample_idx": sample_idx,
                    "error": f"Could not peek sample: {e}",
                }
        else:
            return {
                "sample_idx": sample_idx,
                "error": "Dataset does not support peeking",
            }

    def validate_files(self) -> list[tuple[str, bool, Optional[str]]]:
        """Validate all files in the dataset.

        Returns
        -------
        list[tuple[str, bool, Optional[str]]]
            List of tuples containing (file_path, is_valid, error_message).
            error_message is None if the file is valid.
        """
        results = []

        for file_path in self.file_paths:
            try:
                metadata = self._inspect_file(file_path)
                is_valid = metadata.get("valid", False)
                results.append((file_path, is_valid, None))
            except Exception as e:
                results.append((file_path, False, str(e)))

        return results

    def summary(self) -> dict[str, Any]:
        """Get a summary of the dataset.

        Returns
        -------
        dict[str, Any]
            Dictionary containing dataset summary information.
        """
        return {
            "num_files": self.num_files,
            "file_paths": self.file_paths,
            "is_initialized": self.is_initialized,
            "worker_id": self.worker_id,
            "validate_on_init": self.validate_on_init,
            "max_open_files": self.max_open_files,
            "file_metadata": self.file_metadata,
        }

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"num_files={self.num_files}, "
            f"initialized={self.is_initialized}, "
            f"worker_id={self.worker_id})"
        )


def create_worker_init_fn(dataset: LazyFileDataset) -> callable:
    """Create a worker initialization function for a LazyFileDataset.

    This is a convenience function that creates the worker_init_fn
    needed by PyTorch's DataLoader to properly initialize lazy datasets
    in each worker process.

    Parameters
    ----------
    dataset : LazyFileDataset
        The dataset that needs worker initialization.

    Returns
    -------
    callable
        Worker initialization function suitable for DataLoader.

    Examples
    --------
    >>> dataset = MyLazyDataset(file_paths=['file1.dat', 'file2.dat'])
    >>> worker_init_fn = create_worker_init_fn(dataset)
    >>> loader = DataLoader(dataset, batch_size=32, num_workers=4,
    ...                     worker_init_fn=worker_init_fn)
    """

    def worker_init_fn(worker_id: int) -> None:
        """Initialize dataset in worker process.

        Parameters
        ----------
        worker_id : int
            ID of the current worker process.
        """
        # Get the dataset from worker info
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_dataset = worker_info.dataset
            if isinstance(worker_dataset, LazyFileDataset):
                worker_dataset.worker_init()
            else:
                warnings.warn(
                    f"Dataset in worker {worker_id} is not a LazyFileDataset. "
                    f"Got {type(worker_dataset)}."
                )

    return worker_init_fn


class SequentialDataset(LazyFileDataset, ABC):
    """Abstract base class for sequential/time-series datasets with chunking.

    This class extends LazyFileDataset to handle sequential data that needs to
    be divided into subsequences or chunks. It provides the infrastructure for:
    - Building indices of available subsequences across files
    - Different chunking strategies (non-overlapping, sliding window, etc.)
    - Flexible subsequence length handling
    - Sample indexing and retrieval

    The class maintains an index of all available subsequences without loading
    the actual data, enabling efficient random access to chunks across multiple
    files.
    """

    def __init__(
        self,
        file_paths: str | list[str],
        subseq_len: int,
        chunking_strategy: str = "non_overlapping",
        overlap: int = 0,
        min_seq_len: int | None = None,
        validate_on_init: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the sequential dataset.

        Parameters
        ----------
        file_paths : str | list[str]
            Path or list of paths to data files.
        subseq_len : int
            Length of subsequences to extract. Use -1 to use entire sequences.
        chunking_strategy : str, optional
            Strategy for creating chunks ("non_overlapping", "sliding_window",
            "random_crop"), by default "non_overlapping".
        overlap : int, optional
            Number of samples to overlap between consecutive chunks
            (for sliding_window), by default 0.
        min_seq_len : int | None, optional
            Minimum sequence length required to include a file. If None,
            uses subseq_len, by default None.
        validate_on_init : bool, optional
            Whether to validate files and build index on initialization,
            by default True.
        **kwargs
            Additional arguments passed to LazyFileDataset.
        """
        # Store chunking parameters before calling super().__init__
        self.subseq_len = subseq_len
        self.chunking_strategy = chunking_strategy
        self.overlap = overlap
        self.min_seq_len = min_seq_len or (subseq_len if subseq_len > 0 else 1)

        # Validate chunking parameters
        self._validate_chunking_params()

        # Subsequence index: list of (file_idx, start_idx, end_idx) tuples
        self.subseq_index = []

        # Initialize parent class
        super().__init__(
            file_paths=file_paths, validate_on_init=validate_on_init, **kwargs
        )

        # Build subsequence index after metadata is available
        if validate_on_init:
            self._build_subsequence_index()

    def _validate_chunking_params(self) -> None:
        """Validate chunking parameters.

        Raises
        ------
        ValueError
            If chunking parameters are invalid.
        """
        valid_strategies = ["non_overlapping", "sliding_window", "random_crop"]
        if self.chunking_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid chunking_strategy '{self.chunking_strategy}'. "
                f"Must be one of {valid_strategies}"
            )

        if self.subseq_len <= 0 and self.subseq_len != -1:
            raise ValueError(
                f"subseq_len must be positive or -1 (for full sequences), "
                f"got {self.subseq_len}"
            )

        if self.overlap < 0:
            raise ValueError(
                f"overlap must be non-negative, got {self.overlap}"
            )

        if (
            self.chunking_strategy == "sliding_window"
            and self.overlap >= self.subseq_len > 0
        ):
            raise ValueError(
                f"overlap ({self.overlap}) must be less than subseq_len "
                f"({self.subseq_len}) for sliding_window strategy"
            )

    @abstractmethod
    def _get_sequence_length(self, file_metadata: dict[str, Any]) -> int:
        """Get the sequence length from file metadata.

        This method should be implemented by subclasses to extract the
        sequence length (number of time steps) from the file metadata.
        The implementation should examine the actual data shapes available
        in the file and infer the time dimension.

        Parameters
        ----------
        file_metadata : dict[str, Any]
            Metadata dictionary for a file.

        Returns
        -------
        int
            Number of time steps in the sequence.
        """
        pass

    def _build_subsequence_index(self) -> None:
        """Build an index of all available subsequences across files.

        This creates a flat index where each entry represents one subsequence
        that can be accessed via __getitem__. The index contains tuples of
        (file_index, start_sample, end_sample).
        """
        self.subseq_index = []

        for file_idx, metadata in enumerate(self.file_metadata):
            seq_len = self._get_sequence_length(metadata)

            # Skip files that are too short
            if seq_len < self.min_seq_len:
                warnings.warn(
                    f"Skipping file {metadata.get('path', file_idx)}: "
                    f"sequence length {seq_len} < minimum {self.min_seq_len}"
                )
                continue

            # Generate subsequences based on strategy
            subsequences = self._generate_subsequences(seq_len, file_idx)
            self.subseq_index.extend(subsequences)

    def _generate_subsequences(
        self, seq_len: int, file_idx: int
    ) -> list[tuple[int, int, int]]:
        """Generate subsequence indices for a single file.

        Parameters
        ----------
        seq_len : int
            Total length of the sequence in the file.
        file_idx : int
            Index of the file.

        Returns
        -------
        list[tuple[int, int, int]]
            List of (file_idx, start_idx, end_idx) tuples.
        """
        subsequences = []

        # Handle full sequence case
        if self.subseq_len == -1:
            subsequences.append((file_idx, 0, seq_len))
            return subsequences

        # Handle different chunking strategies
        if self.chunking_strategy == "non_overlapping":
            subsequences = self._generate_non_overlapping(seq_len, file_idx)
        elif self.chunking_strategy == "sliding_window":
            subsequences = self._generate_sliding_window(seq_len, file_idx)
        elif self.chunking_strategy == "random_crop":
            subsequences = self._generate_random_crop(seq_len, file_idx)

        return subsequences

    def _generate_non_overlapping(
        self, seq_len: int, file_idx: int
    ) -> list[tuple[int, int, int]]:
        """Generate non-overlapping subsequences.

        Parameters
        ----------
        seq_len : int
            Total sequence length.
        file_idx : int
            File index.

        Returns
        -------
        list[tuple[int, int, int]]
            List of non-overlapping subsequences.
        """
        subsequences = []

        if seq_len >= self.subseq_len:
            n_chunks = seq_len // self.subseq_len
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.subseq_len
                end_idx = start_idx + self.subseq_len
                subsequences.append((file_idx, start_idx, end_idx))

        return subsequences

    def _generate_sliding_window(
        self, seq_len: int, file_idx: int
    ) -> list[tuple[int, int, int]]:
        """Generate sliding window subsequences.

        Parameters
        ----------
        seq_len : int
            Total sequence length.
        file_idx : int
            File index.

        Returns
        -------
        list[tuple[int, int, int]]
            List of sliding window subsequences.
        """
        subsequences = []

        if seq_len >= self.subseq_len:
            step_size = self.subseq_len - self.overlap
            start_idx = 0

            while start_idx + self.subseq_len <= seq_len:
                end_idx = start_idx + self.subseq_len
                subsequences.append((file_idx, start_idx, end_idx))
                start_idx += step_size

        return subsequences

    def _generate_random_crop(
        self, seq_len: int, file_idx: int
    ) -> list[tuple[int, int, int]]:
        """Generate random crop positions (one per file for now).

        For random cropping, we typically generate crop positions at runtime,
        but we still need to register that this file can provide crops.

        Parameters
        ----------
        seq_len : int
            Total sequence length.
        file_idx : int
            File index.

        Returns
        -------
        list[tuple[int, int, int]]
            List containing one entry representing the file's crop potential.
        """
        subsequences = []

        if seq_len >= self.subseq_len:
            # For random crop, we store (file_idx, -1, seq_len) to indicate
            # that this file can provide random crops
            subsequences.append((file_idx, -1, seq_len))

        return subsequences

    def __len__(self) -> int:
        """Get the total number of subsequences.

        Returns
        -------
        int
            Total number of available subsequences across all files.
        """
        return len(self.subseq_index)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor],
        torch.Tensor | dict[str, torch.Tensor],
    ]:
        """Get a subsequence by index.

        Parameters
        ----------
        idx : int
            Index of the subsequence to retrieve.

        Returns
        -------
        tuple[torch.Tensor | dict[str, torch.Tensor],
              torch.Tensor | dict[str, torch.Tensor],
        ]
            Tuple of (input_tensor, target_tensor). Each can be either a single
            tensor or a dictionary of tensors depending on whether multiple
            keys were specified.

        Raises
        ------
        IndexError
            If index is out of range.
        RuntimeError
            If dataset is not properly initialized.
        """
        if not (0 <= idx < len(self.subseq_index)):
            raise IndexError(
                f"Index {idx} out of range. Dataset has "
                f"{len(self.subseq_index)} subsequences."
            )

        self._ensure_initialized()

        file_idx, start_idx, end_idx = self.subseq_index[idx]

        # Handle random crop case
        if start_idx == -1:
            start_idx, end_idx = self._generate_random_crop_indices(end_idx)

        # Get data from file
        input_tensor, target_tensor = self._extract_subsequence(
            file_idx, start_idx, end_idx
        )

        return input_tensor, target_tensor

    def _generate_random_crop_indices(
        self,
        seq_len: int,
    ) -> tuple[int, int]:
        """Generate random crop start and end indices.

        Parameters
        ----------
        seq_len : int
            Total sequence length available for cropping.

        Returns
        -------
        tuple[int, int]
            Start and end indices for the random crop.
        """
        if seq_len < self.subseq_len:
            raise ValueError(
                f"Sequence length {seq_len} < subsequence length "
                f"{self.subseq_len}"
            )

        max_start = seq_len - self.subseq_len
        start_idx = torch.randint(0, max_start + 1, (1,)).item()
        end_idx = start_idx + self.subseq_len

        return start_idx, end_idx

    @abstractmethod
    def _extract_subsequence(
        self,
        file_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor],
        torch.Tensor | dict[str, torch.Tensor],
    ]:
        """Extract a subsequence from a file.

        This method should be implemented by subclasses to extract the actual
        data subsequence from the opened file.

        Parameters
        ----------
        file_idx : int
            Index of the file to read from.
        start_idx : int
            Start index of the subsequence.
        end_idx : int
            End index of the subsequence.

        Returns
        -------
        tuple[Union[torch.Tensor, dict[str, torch.Tensor]],
        Union[torch.Tensor, dict[str, torch.Tensor]]]
            Tuple of (input_tensor, target_tensor). Each can be either a single
            tensor or a dictionary of tensors depending on whether multiple
            keys were specified.
        """
        pass

    def get_subsequence_info(self, idx: int) -> dict[str, Any]:
        """Get information about a specific subsequence.

        Parameters
        ----------
        idx : int
            Index of the subsequence.

        Returns
        -------
        dict[str, Any]
            Dictionary containing subsequence information.
        """
        if not (0 <= idx < len(self.subseq_index)):
            raise IndexError(f"Index {idx} out of range")

        file_idx, start_idx, end_idx = self.subseq_index[idx]
        file_metadata = self.get_file_metadata(file_idx)

        return {
            "subsequence_idx": idx,
            "file_idx": file_idx,
            "file_path": file_metadata.get("path", "unknown"),
            "start_idx": start_idx,
            "end_idx": end_idx,
            "length": end_idx - start_idx
            if start_idx != -1
            else self.subseq_len,
            "is_random_crop": start_idx == -1,
        }

    def get_file_subsequences(self, file_idx: int) -> list[int]:
        """Get all subsequence indices that belong to a specific file.

        Parameters
        ----------
        file_idx : int
            Index of the file.

        Returns
        -------
        list[int]
            List of subsequence indices from the specified file.
        """
        return [
            idx
            for idx, (f_idx, _, _) in enumerate(self.subseq_index)
            if f_idx == file_idx
        ]

    def summary(self) -> dict[str, Any]:
        """Get a summary of the dataset.

        Returns
        -------
        dict[str, Any]
            Dictionary containing dataset summary information.
        """
        base_summary = super().summary()

        # Add sequential-specific information
        sequential_info = {
            "subseq_len": self.subseq_len,
            "chunking_strategy": self.chunking_strategy,
            "overlap": self.overlap,
            "min_seq_len": self.min_seq_len,
            "total_subsequences": len(self.subseq_index),
            "subsequences_per_file": [
                len(self.get_file_subsequences(i))
                for i in range(self.num_files)
            ],
        }

        base_summary.update(sequential_info)
        return base_summary

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns
        -------
        str
            String representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"num_files={self.num_files}, "
            f"total_subsequences={len(self.subseq_index)}, "
            f"subseq_len={self.subseq_len}, "
            f"strategy={self.chunking_strategy})"
        )


class MultiFileDataset(SequentialDataset, ABC):
    """
    Abstract base class for datasets spanning multiple files with advanced
    management.

    This class extends SequentialDataset to provide sophisticated multi-file
    handling:
    - File discovery and pattern matching
    - File sorting and organization
    - Load balancing across files
    - File filtering and selection
    - Memory management for large file collections

    The class is designed to handle datasets where:
    - Data is distributed across many files
    - Files may have different sizes
    - You want to control which files are used
    - You need efficient access patterns across files
    """

    def __init__(
        self,
        file_paths: str | list[str] | Path,
        subseq_len: int,
        file_pattern: str | None = None,
        file_filter: callable | None = None,
        sort_files: bool = True,
        max_files: int | None = None,
        balance_files: bool = False,
        file_weights: dict[str, float] | None = None,
        cache_metadata: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the multi-file dataset.

        Parameters
        ----------
        file_paths : str | list[str] | Path
            Path(s) to data files. Can be:
            - Single file path
            - List of file paths
            - Directory path (will search for files)
            - Glob pattern
        subseq_len : int
            Length of subsequences to extract.
        file_pattern : str | None, optional
            Glob pattern for file discovery when file_paths is a directory,
            by default None (uses "*").
        file_filter : callable | None, optional
            Function to filter files. Should take file path and return bool,
            by default None.
        sort_files : bool, optional
            Whether to sort files by name, by default True.
        max_files : int | None, optional
            Maximum number of files to use, by default None (use all).
        balance_files : bool, optional
            Whether to balance subsequences across files, by default False.
        file_weights : dict[str, float] | None, optional
            Weights for sampling from different files, by default None.
        cache_metadata : bool, optional
            Whether to cache file metadata to disk, by default True.
        **kwargs
            Additional arguments passed to SequentialDataset.
        """
        # Store multi-file parameters
        self.file_pattern = file_pattern or "*"
        self.file_filter = file_filter
        self.sort_files = sort_files
        self.max_files = max_files
        self.balance_files = balance_files
        self.file_weights = file_weights or {}
        self.cache_metadata = cache_metadata

        # File management state
        self.file_stats = []
        self.file_groups = {}
        self.balanced_indices = None
        self._metadata_cache = {}

        # Discover and process files
        resolved_file_paths = self._discover_files(file_paths)

        # Initialize parent with resolved file paths
        super().__init__(
            file_paths=resolved_file_paths, subseq_len=subseq_len, **kwargs
        )

        # Build file statistics and balancing if requested
        if self.validate_on_init:
            self._build_file_stats()
            if self.balance_files:
                self._build_balanced_indices()

    def _discover_files(
        self,
        file_paths: str | list[str] | Path,
    ) -> list[str]:
        """Discover and process file paths.

        Parameters
        ----------
        file_paths : str | list[str] | Path
            Input file paths specification.

        Returns
        -------
        list[str]
            List of resolved file paths.
        """
        if isinstance(file_paths, (str, Path)):
            file_paths = Path(file_paths)

            if file_paths.is_dir():
                # Directory: search for files
                discovered_files = list(file_paths.glob(self.file_pattern))
                resolved_paths = [
                    str(f) for f in discovered_files if f.is_file()
                ]
            elif "*" in str(file_paths) or "?" in str(file_paths):
                # Glob pattern
                from glob import glob

                resolved_paths = glob(str(file_paths))
            else:
                # Single file
                resolved_paths = [str(file_paths)]
        else:
            # List of paths
            resolved_paths = [str(p) for p in file_paths]

        # Apply file filter if provided
        if self.file_filter:
            resolved_paths = [f for f in resolved_paths if self.file_filter(f)]

        # Sort files if requested
        if self.sort_files:
            resolved_paths.sort()

        # Limit number of files
        if self.max_files and len(resolved_paths) > self.max_files:
            resolved_paths = resolved_paths[: self.max_files]

        if not resolved_paths:
            raise ValueError(
                f"No files found matching criteria. "
                f"Input: {file_paths}, pattern: {self.file_pattern}"
            )

        return resolved_paths

    def _build_file_stats(self) -> None:
        """Build statistics for each file."""
        self.file_stats = []

        for file_idx, metadata in enumerate(self.file_metadata):
            file_path = metadata.get("path", f"file_{file_idx}")
            seq_len = self._get_sequence_length(metadata)

            # Count subsequences for this file
            file_subseqs = self.get_file_subsequences(file_idx)
            num_subseqs = len(file_subseqs)

            # Calculate file weight
            weight = self.file_weights.get(file_path, 1.0)

            stats = {
                "file_idx": file_idx,
                "file_path": file_path,
                "sequence_length": seq_len,
                "num_subsequences": num_subseqs,
                "weight": weight,
                "subsequence_indices": file_subseqs,
            }

            self.file_stats.append(stats)

    def _build_balanced_indices(self) -> None:
        """Build balanced indices for equal representation across files.

        This creates a new indexing scheme where each file contributes
        roughly the same number of samples, regardless of file size.
        """
        if not self.file_stats:
            return

        # Find minimum number of subsequences across files
        min_subseqs = min(
            stats["num_subsequences"]
            for stats in self.file_stats
            if stats["num_subsequences"] > 0
        )

        if min_subseqs == 0:
            warnings.warn("No files have valid subsequences for balancing")
            return

        # Build balanced index
        self.balanced_indices = []

        for stats in self.file_stats:
            if stats["num_subsequences"] > 0:
                # Sample evenly from this file's subsequences
                file_subseqs = stats["subsequence_indices"]

                if len(file_subseqs) >= min_subseqs:
                    # Sample min_subseqs indices evenly
                    step = len(file_subseqs) / min_subseqs
                    selected_indices = [
                        file_subseqs[int(i * step)] for i in range(min_subseqs)
                    ]
                else:
                    # Use all available indices
                    # (shouldn't happen due to min calculation)
                    selected_indices = file_subseqs

                self.balanced_indices.extend(selected_indices)

    def __len__(self) -> int:
        """Get the total number of subsequences.

        Returns
        -------
        int
            Total number of subsequences (balanced or unbalanced).
        """
        if self.balance_files and self.balanced_indices is not None:
            return len(self.balanced_indices)
        else:
            return super().__len__()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a subsequence by index.

        Parameters
        ----------
        idx : int
            Index of the subsequence to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of (input_tensor, target_tensor).
        """
        # Handle balanced indexing
        if self.balance_files and self.balanced_indices is not None:
            if not (0 <= idx < len(self.balanced_indices)):
                raise IndexError(
                    f"Balanced index {idx} out of range. Dataset has "
                    f"{len(self.balanced_indices)} balanced subsequences."
                )
            # Map to actual subsequence index
            actual_idx = self.balanced_indices[idx]
            return super().__getitem__(actual_idx)
        else:
            return super().__getitem__(idx)

    def get_file_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all files.

        Returns
        -------
        list[dict[str, Any]]
            List of file statistics dictionaries.
        """
        return self.file_stats.copy()

    def get_largest_files(self, n: int = 5) -> list[dict[str, Any]]:
        """Get the n largest files by subsequence count.

        Parameters
        ----------
        n : int, optional
            Number of files to return, by default 5.

        Returns
        -------
        list[dict[str, Any]]
            List of file statistics sorted by subsequence count.
        """
        sorted_files = sorted(
            self.file_stats, key=lambda x: x["num_subsequences"], reverse=True
        )
        return sorted_files[:n]

    def get_files_by_pattern(self, pattern: str) -> list[dict[str, Any]]:
        """Get files matching a pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match against file paths.

        Returns
        -------
        list[dict[str, Any]]
            List of matching file statistics.
        """
        import re

        regex = re.compile(pattern)

        return [
            stats
            for stats in self.file_stats
            if regex.search(stats["file_path"])
        ]

    def filter_files_by_size(
        self,
        min_subsequences: int = 1,
        max_subsequences: int | None = None,
    ) -> MultiFileDataset:
        """Create a new dataset with files filtered by subsequence count.

        Parameters
        ----------
        min_subsequences : int, optional
            Minimum number of subsequences required, by default 1.
        max_subsequences : int | None, optional
            Maximum number of subsequences allowed, by default None.

        Returns
        -------
        MultiFileDataset
            New dataset with filtered files.
        """
        filtered_paths = []

        for stats in self.file_stats:
            num_subseqs = stats["num_subsequences"]
            if num_subseqs >= min_subsequences:
                if max_subsequences is None or num_subseqs <= max_subsequences:
                    filtered_paths.append(stats["file_path"])

        # Create new dataset with same parameters but filtered files
        return self.__class__(
            file_paths=filtered_paths,
            subseq_len=self.subseq_len,
            chunking_strategy=self.chunking_strategy,
            overlap=self.overlap,
            balance_files=self.balance_files,
            file_weights=self.file_weights,
            validate_on_init=self.validate_on_init,
        )

    def split_by_files(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        random_seed: int | None = None,
    ) -> tuple[MultiFileDataset, MultiFileDataset]:
        """Split the dataset by files (not by samples).

        This ensures that samples from the same file don't appear in both
        training and validation sets, which is important for proper evaluation.

        Parameters
        ----------
        train_ratio : float, optional
            Fraction of files for training, by default 0.8.
        val_ratio : float, optional
            Fraction of files for validation, by default 0.1.
        random_seed : int | None, optional
            Random seed for reproducible splits, by default None.

        Returns
        -------
        tuple[MultiFileDataset, MultiFileDataset, MultiFileDataset]
            Train, validation, and test datasets.
        """
        if abs(train_ratio + val_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Get all file paths
        all_paths = [stats["file_path"] for stats in self.file_stats]

        # Shuffle files
        if random_seed is not None:
            import random

            random.seed(random_seed)
            random.shuffle(all_paths)

        # Calculate split points
        n_files = len(all_paths)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files - n_train)

        # Split file paths
        train_paths = all_paths[:n_train]
        val_paths = all_paths[n_train:]

        if n_train != len(train_paths):
            raise ValueError(
                "Training set size does not match expected number of files."
            )
        if n_val != len(val_paths):
            raise ValueError(
                "Validation set size does not match expected number of files."
            )

        # Create split datasets
        common_params = {
            "subseq_len": self.subseq_len,
            "chunking_strategy": self.chunking_strategy,
            "overlap": self.overlap,
            "balance_files": self.balance_files,
            "validate_on_init": self.validate_on_init,
        }

        train_dataset = self.__class__(file_paths=train_paths, **common_params)
        val_dataset = self.__class__(file_paths=val_paths, **common_params)

        return train_dataset, val_dataset

    def get_memory_usage_estimate(self) -> dict[str, float]:
        """Estimate memory usage for different scenarios.

        Returns
        -------
        dict[str, float]
            Dictionary with memory estimates in GB.
        """
        total_subseqs = len(self)

        # Estimate based on first file (rough approximation)
        if self.file_metadata:
            # This would need to be implemented by subclasses
            # based on their specific data types and sizes
            try:
                sample_input, sample_target = self[0]

                input_size = sample_input.numel() * sample_input.element_size()
                target_size = (
                    sample_target.numel() * sample_target.element_size()
                )

                per_sample_bytes = input_size + target_size

                return {
                    "per_sample_mb": per_sample_bytes / (1024 * 1024),
                    "total_dataset_gb": (total_subseqs * per_sample_bytes)
                    / (1024**3),
                    "single_batch_mb": (32 * per_sample_bytes) / (1024 * 1024),
                    # Assume batch=32
                }
            except Exception:
                pass

        return {"error": "No data available for estimation"}

    def summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the dataset.

        Returns
        -------
        dict[str, Any]
            Dictionary containing dataset summary information.
        """
        base_summary = super().summary()

        # Add multi-file specific information
        file_sizes = [stats["num_subsequences"] for stats in self.file_stats]

        multifile_info = {
            "file_pattern": self.file_pattern,
            "max_files": self.max_files,
            "balance_files": self.balance_files,
            "actual_num_files": len(self.file_stats),
            "file_size_stats": {
                "min_subsequences": min(file_sizes) if file_sizes else 0,
                "max_subsequences": max(file_sizes) if file_sizes else 0,
                "avg_subsequences": sum(file_sizes) / len(file_sizes)
                if file_sizes
                else 0,
                "total_subsequences": sum(file_sizes),
            },
            "balanced_length": len(self.balanced_indices)
            if self.balanced_indices
            else None,
            "has_file_weights": bool(self.file_weights),
        }

        base_summary.update(multifile_info)
        return base_summary

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns
        -------
        str
            String representation.
        """
        balance_info = ""
        if self.balance_files:
            balanced = (
                len(self.balanced_indices) if self.balanced_indices else 0
            )
            balance_info = f", balanced={balanced}"

        return (
            f"{self.__class__.__name__}(files={len(self.file_stats)}, "
            f"subsequences={len(self)}{balance_info}, "
            f"subseq_len={self.subseq_len})"
        )
