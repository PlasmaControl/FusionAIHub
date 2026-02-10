"""Concrete implementations of file-based datasets."""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch

from .base import MultiFileDataset


class JoblibDataset(MultiFileDataset):
    """Dataset for joblib files with memory-mapped loading.

    Supports flexible key configuration for different file formats.
    Can work with input-only data (autoencoders) or input-target pairs.
    """

    def __init__(
        self,
        file_paths: str | list[str] | Path,
        subseq_len: int,
        input_key: str | list[str] | None = None,
        target_key: str | list[str] | None = None,
        target_slice: tuple | None = None,
        auto_detect_keys: bool = True,
        **kwargs,
    ) -> None:
        """Initialize joblib dataset.

        Parameters
        ----------
        file_paths : str | list[str] | Path
            Path(s) to joblib files.
        subseq_len : int
            Length of subsequences to extract.
        input_key : str | list[str] | None, optional
            Key(s) for input data in joblib files. If None, will auto-detect.
            If list, returns dictionary of inputs, by default None.
        target_key : str | list[str] | None, optional
            Key(s) for target data in joblib files. If None, uses input as
            target (autoencoder mode). If list, returns dictionary of targets,
            by default None.
        target_slice : tuple | None, optional
            Slice to apply to target tensor
            (e.g., (slice(None, 48), slice(None), slice(None))
            for [:48, :, :]), by default None.
        auto_detect_keys : bool, optional
            Whether to automatically detect keys from first valid file,
            by default True.
        **kwargs
            Additional arguments passed to MultiFileDataset.
        """
        self.input_key = input_key
        self.target_key = target_key
        self.target_slice = target_slice

        # Auto-detect if no input key provided
        self.auto_detect_keys = auto_detect_keys or (input_key is None)
        self.is_autoencoder_mode = target_key is None

        # Handle multiple keys
        self.is_multi_input = isinstance(input_key, list)
        self.is_multi_target = isinstance(target_key, list)

        # Store original keys for auto-detection fallback
        self._original_input_key = input_key
        self._original_target_key = target_key

        # Track if keys have been detected
        self._keys_detected = False

        super().__init__(file_paths, subseq_len, **kwargs)

    def _detect_keys(
        self,
        data_dict: dict[str, Any],
        file_path: str,
    ) -> None:
        """Auto-detect keys from file contents."""
        available_keys = list(data_dict.keys())

        print(f"Auto-detecting keys from {file_path}")
        print(f"Available keys: {available_keys}")

        # Skip auto-detection if keys are already provided as lists
        if self.is_multi_input or self.is_multi_target:
            print("Skipping auto-detection since list of keys provided")
            return

        # Try to find input key
        if not self.is_multi_input and self.input_key is None:
            input_candidates = [
                "input",
                "data",
                "x",
                "features",
                "spectrogram",
                "signal",
            ]
            for candidate in input_candidates:
                if candidate in available_keys:
                    self.input_key = candidate
                    print(f"Detected input_key: {self.input_key}")
                    break
            else:
                # If no standard key found, use the first array-like key
                for key in available_keys:
                    if (
                        hasattr(data_dict[key], "shape")
                        and len(data_dict[key].shape) >= 2
                    ):
                        self.input_key = key
                        print(
                            f"Using first array key as input_key: "
                            f"{self.input_key}"
                        )
                        break

        # Try to find target key
        # (only if not in autoencoder mode and not multi-target)
        if (
            not self.is_autoencoder_mode
            and not self.is_multi_target
            and self.target_key is None
        ):
            target_candidates = [
                "target",
                "label",
                "y",
                "output",
                "ground_truth",
            ]
            for candidate in target_candidates:
                if candidate in available_keys:
                    self.target_key = candidate
                    print(f"Detected target_key: {self.target_key}")
                    break
            else:
                # If no standard target key found,
                # look for second array-like key
                array_keys = [
                    k
                    for k in available_keys
                    if hasattr(data_dict[k], "shape")
                    and len(data_dict[k].shape) >= 2
                ]
                if len(array_keys) >= 2:
                    # Use second array key as target (first is input)
                    target_candidates_from_arrays = [
                        k for k in array_keys if k != self.input_key
                    ]
                    if target_candidates_from_arrays:
                        self.target_key = target_candidates_from_arrays[0]
                        print(
                            f"Using second array key as target_key: "
                            f"{self.target_key}"
                        )

    def _infer_sequence_length_from_data(
        self,
        data_dict: dict[str, Any],
    ) -> int:
        """Infer sequence length by examining data shapes.

        Parameters
        ----------
        data_dict : dict[str, Any]
            Dictionary containing the loaded data.

        Returns
        -------
        int
            Inferred sequence length.
        """
        # Handle multiple input keys
        if self.is_multi_input:
            sequence_lengths = []
            for key in self.input_key:
                if key in data_dict:
                    input_data = data_dict[key]
                    if hasattr(input_data, "shape"):
                        shape = input_data.shape
                        # Assume time dimension is the longest dimension or
                        # axis 2 if >= 3D
                        if len(shape) >= 3:
                            # Conventional: (channels, features, time)
                            seq_len = shape[-1]
                        elif len(shape) == 2:
                            # Use the larger dimension as time
                            seq_len = max(shape)
                        elif len(shape) == 1:
                            seq_len = shape[0]  # 1D: (time,)
                        else:
                            continue  # Skip scalar values

                        sequence_lengths.append((key, seq_len))

            if sequence_lengths:
                # Check if all keys have the same sequence length
                unique_lengths = set(length for _, length in sequence_lengths)
                if len(unique_lengths) == 1:
                    return sequence_lengths[0][1]  # All keys have same length
                else:
                    # Different sequence lengths - use the most common one or
                    # smallest
                    length_counts = Counter(
                        length for _, length in sequence_lengths
                    )
                    most_common_length = length_counts.most_common(1)[0][0]

                    # Warn about inconsistent lengths
                    warnings.warn(
                        f"Inconsistent sequence lengths across input keys: "
                        f"{dict(sequence_lengths)}. Using most common length: "
                        f"{most_common_length}"
                    )
                    return most_common_length

        # Handle single input key (original logic)
        elif self.input_key is not None and self.input_key in data_dict:
            input_data = data_dict[self.input_key]
            if hasattr(input_data, "shape"):
                shape = input_data.shape
                # Assume time dimension is longest dimension or axis 1 if >= 3D
                if len(shape) >= 3:
                    return shape[
                        -1
                    ]  # Conventional: (channels, features, time)
                elif len(shape) == 2:
                    # Use the larger dimension as time
                    return max(shape)
                elif len(shape) == 1:
                    return shape[0]  # 1D: (time,)

        # Fallback: find the largest array and use its time dimension
        max_samples = 0
        for key, value in data_dict.items():
            if hasattr(value, "shape") and len(value.shape) >= 1:
                shape = value.shape
                # Try different conventions for time dimension
                if len(shape) >= 3:
                    # Try axis 2 first (channels, features, time)
                    candidate_time = shape[-1]
                    max_samples = max(max_samples, candidate_time)
                elif len(shape) >= 2:
                    # Use the larger dimension as time
                    candidate_time = max(shape)
                    max_samples = max(max_samples, candidate_time)
                elif len(shape) == 1:
                    # 1D array: assume it's all time
                    max_samples = max(max_samples, shape[0])

        return max_samples

    def _validate_keys_in_file(
        self,
        data_dict: dict[str, Any],
        file_path: str,
    ) -> tuple[bool, str]:
        """Validate that required keys exist in the file.

        Parameters
        ----------
        data_dict : dict[str, Any]
            Dictionary containing the loaded data.
        file_path : str
            Path to the file being validated.

        Returns
        -------
        tuple[bool, str]
            Tuple of (is_valid, error_message). Error_message empty if valid.
        """
        available_keys = list(data_dict.keys())

        # Validate input keys
        if self.is_multi_input:
            missing_input_keys = [
                key for key in self.input_key if key not in data_dict
            ]
            if missing_input_keys:
                return (
                    False,
                    f"Missing input keys {missing_input_keys}. "
                    f"Available keys: {available_keys}",
                )
        elif self.input_key is not None and self.input_key not in data_dict:
            return (
                False,
                f"Missing input key '{self.input_key}'. "
                f"Available keys: {available_keys}",
            )

        # Validate target keys (if not in autoencoder mode)
        if not self.is_autoencoder_mode:
            if self.is_multi_target:
                missing_target_keys = [
                    key for key in self.target_key if key not in data_dict
                ]
                if missing_target_keys:
                    return (
                        False,
                        f"Missing target keys {missing_target_keys}. "
                        f"Available keys: {available_keys}",
                    )
            elif (
                self.target_key is not None
                and self.target_key not in data_dict
            ):
                return (
                    False,
                    f"Missing target key '{self.target_key}'. "
                    f"Available keys: {available_keys}",
                )

        return True, ""

    def _inspect_file(
        self,
        file_path: str,
    ) -> dict[str, Any]:
        """Inspect joblib file to extract metadata."""
        try:
            from joblib import load
        except ImportError:
            raise ImportError(
                "joblib is required for JoblibDataset. "
                "Install with: pip install joblib"
            )

        try:
            data_dict = load(file_path, mmap_mode="r")

            # Auto-detect keys from first file if requested
            if self.auto_detect_keys and not self._keys_detected:
                self._detect_keys(data_dict, file_path)
                self._keys_detected = True

            # Get available keys
            available_keys = list(data_dict.keys())

            # Validate keys exist in file
            is_valid, error_msg = self._validate_keys_in_file(
                data_dict,
                file_path,
            )
            if not is_valid:
                return {
                    "path": file_path,
                    "valid": False,
                    "error": error_msg,
                    "n_samples": 0,
                    "available_keys": available_keys,
                }

            # Infer sequence length from data shapes
            n_samples = self._infer_sequence_length_from_data(
                data_dict,
            )

            # Get data shapes for metadata
            input_shape = None
            target_shape = None

            if self.is_multi_input:
                input_shape = {
                    key: data_dict[key].shape
                    for key in self.input_key
                    if key in data_dict
                }
            elif self.input_key is not None and self.input_key in data_dict:
                input_shape = data_dict[self.input_key].shape

            if not self.is_autoencoder_mode:
                if self.is_multi_target:
                    target_shape = {
                        key: data_dict[key].shape
                        for key in self.target_key
                        if key in data_dict
                    }
                elif (
                    self.target_key is not None
                    and self.target_key in data_dict
                ):
                    target_shape = data_dict[self.target_key].shape

            metadata = {
                "path": file_path,
                "valid": True,
                "n_samples": n_samples,
                "input_shape": input_shape,
                "target_shape": target_shape,
                "available_keys": available_keys,
                "is_autoencoder_mode": self.is_autoencoder_mode,
                "inferred_input_key": self.input_key,
                "inferred_target_key": self.target_key,
            }

            del data_dict  # Close file handle
            return metadata

        except Exception as e:
            return {
                "path": file_path,
                "valid": False,
                "error": str(e),
                "n_samples": 0,
            }

    def _get_sequence_length(
        self,
        file_metadata: dict[str, Any],
    ) -> int:
        """Get sequence length from metadata."""
        return file_metadata.get("n_samples", 0)

    def _open_file(
        self,
        file_path: str,
    ) -> Any:
        """Open joblib file with memory mapping."""
        try:
            from joblib import load
        except ImportError:
            raise ImportError(
                "joblib is required for JoblibDataset. "
                "Install with: pip install joblib"
            )

        return load(file_path, mmap_mode="r")

    def _close_file(
        self,
        file_handle: Any,
    ) -> None:
        """Close joblib file handle."""
        del file_handle  # Joblib handles cleanup automatically

    def _extract_tensor_from_array(
        self,
        array: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract a tensor from an array with flexible shape handling.

        Parameters
        ----------
        array : numpy.ndarray
            Input array to extract from.
        start_idx : int
            Start index for extraction.
        end_idx : int
            End index for extraction.

        Returns
        -------
        torch.Tensor
            Extracted tensor.
        """
        # Handle different array shapes flexibly
        if len(array.shape) >= 3:
            # Assume (channels, features, time) or similar
            sub_array = array[:, :, start_idx:end_idx]
        elif len(array.shape) == 2:
            # Assume (features, time) or (time, features)
            # Check which dimension is longer to determine time axis
            if array.shape[0] > array.shape[1]:
                # Assume (time, features)
                sub_array = array[start_idx:end_idx, :]
            else:
                # Assume (features, time)
                sub_array = array[:, start_idx:end_idx]
        else:
            # 1D array: (time,)
            sub_array = array[start_idx:end_idx]

        return torch.from_numpy(np.array(sub_array)).float()

    def _extract_subsequence(
        self,
        file_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor],
        torch.Tensor | dict[str, torch.Tensor],
    ]:
        """Extract subsequence from joblib file."""
        file_handle = self.get_file_handle(file_idx)

        # Get the keys from metadata (they might have been auto-detected)
        metadata = self.get_file_metadata(file_idx)

        # Extract input data
        if self.is_multi_input:
            input_data = {}
            for key in self.input_key:
                if key in file_handle:
                    input_data[key] = self._extract_tensor_from_array(
                        file_handle[key], start_idx, end_idx
                    )
                else:
                    raise ValueError(f"Input key '{key}' not found in file")
        else:
            input_key = metadata.get("inferred_input_key") or self.input_key
            if input_key and input_key in file_handle:
                input_data = self._extract_tensor_from_array(
                    file_handle[input_key], start_idx, end_idx
                )
            else:
                raise ValueError(f"Input key '{input_key}' not found in file")

        # Extract target data
        if self.is_autoencoder_mode:
            # Autoencoder mode: target = input
            if self.is_multi_input:
                target_data = {
                    key: tensor.clone() for key, tensor in input_data.items()
                }
            else:
                target_data = input_data.clone()
        else:
            # Supervised mode: use separate target
            if self.is_multi_target:
                target_data = {}
                for key in self.target_key:
                    if key in file_handle:
                        target_data[key] = self._extract_tensor_from_array(
                            file_handle[key], start_idx, end_idx
                        )
                    else:
                        raise ValueError(
                            f"Target key '{key}' not found in file"
                        )
            else:
                target_key = (
                    metadata.get("inferred_target_key") or self.target_key
                )
                if target_key and target_key in file_handle:
                    target_data = self._extract_tensor_from_array(
                        file_handle[target_key], start_idx, end_idx
                    )
                else:
                    # Fallback to input if no target found
                    if self.is_multi_input:
                        target_data = {
                            key: tensor.clone()
                            for key, tensor in input_data.items()
                        }
                    else:
                        target_data = input_data.clone()

        # Apply target slice if specified (only for single tensor targets)
        if self.target_slice is not None and not self.is_multi_target:
            target_data = target_data[self.target_slice]

        return input_data, target_data

    def get_sample_shape(
        self,
        file_idx: int = 0,
        start_idx: int = 0,
        end_idx: int = 10,
    ) -> tuple[Any, Any]:
        """Get sample input and target shapes without initializing workers.

        This method temporarily opens a file to inspect data shapes, then
        closes it. Useful for model configuration before training.

        Parameters
        ----------
        file_idx : int, optional
            Index of the file to sample from, by default 0.
        start_idx : int, optional
            Start index for extraction, by default 0.
        end_idx : int, optional
            End index for extraction, by default 10.

        Returns
        -------
        tuple[Any, Any]
            Tuple of (input_shape, target_shape). For multi-key scenarios,
            returns dict of shapes.
        """
        if file_idx >= len(self.file_paths):
            raise IndexError(f"File index {file_idx} out of range")

        # Temporarily open the file
        file_handle = self._open_file(self.file_paths[file_idx])

        try:
            # Get a small subsequence to determine shapes
            start_idx = 0
            end_idx = min(self.subseq_len, 10) if self.subseq_len > 0 else 10

            # Temporarily store the opened file for _extract_subsequence
            original_opened_files = self._opened_files
            original_initialized = self._is_initialized

            self._opened_files = [None] * len(self.file_paths)
            self._opened_files[file_idx] = file_handle
            self._is_initialized = True

            # Extract a sample to get shapes
            sample_input, sample_target = self._extract_subsequence(
                file_idx, start_idx, end_idx
            )

            # Extract shapes
            if isinstance(sample_input, dict):
                input_shape = {
                    key: tensor.shape for key, tensor in sample_input.items()
                }
            else:
                input_shape = sample_input.shape

            if isinstance(sample_target, dict):
                target_shape = {
                    key: tensor.shape for key, tensor in sample_target.items()
                }
            else:
                target_shape = sample_target.shape

            return input_shape, target_shape

        finally:
            # Always close the file
            self._opened_files = original_opened_files
            self._is_initialized = original_initialized
            self._close_file(file_handle)

    def peek_sample(
        self,
        file_idx: int = 0,
        subseq_idx: int = 0,
        start_idx: int = 0,
        end_idx: int = 10,
    ) -> tuple[
        Union[torch.Tensor, dict[str, torch.Tensor]],
        Union[torch.Tensor, dict[str, torch.Tensor]],
    ]:
        """Peek at a sample without worker initialization.

        This method temporarily opens a file, extracts one sample, then closes
        it. Useful for data inspection and debugging.

        Parameters
        ----------
        file_idx : int, optional
            Index of the file to sample from, by default 0.
        subseq_idx : int, optional
            Index of the subsequence within the file, by default 0.
        start_idx : int, optional
            Start index for extraction, by default 0.
        end_idx : int, optional
            End index for extraction, by default 10.

        Returns
        -------
        tuple[Union[torch.Tensor, dict[str, torch.Tensor]],
              Union[torch.Tensor, dict[str, torch.Tensor]]]
            Sample input and target data.
        """
        if file_idx >= len(self.file_paths):
            raise IndexError(f"File index {file_idx} out of range")

        # Get file metadata to determine valid subsequence range
        file_metadata = self.get_file_metadata(file_idx)
        seq_len = self._get_sequence_length(file_metadata)

        # Calculate valid subsequence bounds
        if self.subseq_len == -1:
            start_idx = 0
            end_idx = seq_len
        else:
            max_start = max(0, seq_len - self.subseq_len)
            start_idx = min(subseq_idx * self.subseq_len, max_start)
            end_idx = min(start_idx + self.subseq_len, seq_len)

        # Temporarily open the file
        file_handle = self._open_file(self.file_paths[file_idx])

        try:
            # Temporarily store the opened file for _extract_subsequence
            original_opened_files = self._opened_files
            original_initialized = self._is_initialized

            self._opened_files = [None] * len(self.file_paths)
            self._opened_files[file_idx] = file_handle
            self._is_initialized = True

            # Extract the sample
            sample_input, sample_target = self._extract_subsequence(
                file_idx, start_idx, end_idx
            )

            return sample_input, sample_target

        finally:
            # Restore original state
            self._opened_files = original_opened_files
            self._is_initialized = original_initialized

            # Close the temporary file handle
            self._close_file(file_handle)


class HDF5Dataset(MultiFileDataset):
    """Dataset for HDF5 files with configurable keys.

    Supports flexible key configuration for different file formats.
    Can work with input-only data (autoencoders) or input-target pairs.
    """

    def __init__(
        self,
        file_paths: Union[str, list[str], Path],
        subseq_len: int,
        input_key: Union[str, list[str]] = "input",
        target_key: Optional[Union[str, list[str]]] = None,
        target_slice: Optional[tuple] = None,
        auto_detect_keys: bool = False,
        **kwargs,
    ) -> None:
        """Initialize HDF5 dataset.

        Parameters
        ----------
        file_paths : Union[str, list[str], Path]
            Path(s) to HDF5 files.
        subseq_len : int
            Length of subsequences to extract.
        input_key : Union[str, list[str]], optional
            Key(s) for input data in HDF5 files. If list, returns dictionary of
            inputs, by default 'input'.
        target_key : Optional[Union[str, list[str]]], optional
            Key(s) for target data in HDF5 files. If None, uses input as target
            (autoencoder mode). If list, returns dictionary of targets,
            by default None.
        target_slice : Optional[tuple], optional
            Slice to apply to target tensor, by default None.
        auto_detect_keys : bool, optional
            Whether to automatically detect keys from first valid file,
            by default False.
        **kwargs
            Additional arguments passed to MultiFileDataset.
        """
        self.input_key = input_key
        self.target_key = target_key
        self.target_slice = target_slice
        self.auto_detect_keys = auto_detect_keys
        self.is_autoencoder_mode = target_key is None

        # Handle multiple keys
        self.is_multi_input = isinstance(input_key, list)
        self.is_multi_target = isinstance(target_key, list)

        # Store original keys for auto-detection fallback
        self._original_input_key = input_key
        self._original_target_key = target_key

        # Track if keys have been detected
        self._keys_detected = False

        super().__init__(file_paths, subseq_len, **kwargs)

    def _detect_keys(self, file_handle, file_path: str) -> None:
        """Auto-detect keys from HDF5 file contents."""
        available_keys = list(file_handle.keys())

        print(f"Auto-detecting keys from {file_path}")
        print(f"Available keys: {available_keys}")

        # Skip auto-detection if keys are already provided as lists
        if self.is_multi_input or self.is_multi_target:
            print("Skipping auto-detection since list of keys provided")
            return

        # Try to find input key
        if not self.is_multi_input:
            input_candidates = ["input", "data", "x", "features"]
            for candidate in input_candidates:
                if candidate in available_keys:
                    self.input_key = candidate
                    print(f"Detected input_key: {self.input_key}")
                    break

        # Try to find target key
        # (only if not in autoencoder mode and not multi-target)
        if not self.is_autoencoder_mode and not self.is_multi_target:
            target_candidates = ["target", "label", "y", "output"]
            for candidate in target_candidates:
                if candidate in available_keys:
                    self.target_key = candidate
                    print(f"Detected target_key: {self.target_key}")
                    break

    def _inspect_file(self, file_path: str) -> dict[str, Any]:
        """Inspect HDF5 file to extract metadata."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5Dataset. Install with: pip install h5py"
            )

        try:
            with h5py.File(file_path, "r") as f:
                # Auto-detect keys from first file if requested
                if self.auto_detect_keys and not self._keys_detected:
                    self._detect_keys(f, file_path)
                    self._keys_detected = True

                # Validate required keys exist
                if self.is_multi_input:
                    missing_input_keys = [
                        key for key in self.input_key if key not in f
                    ]
                    if missing_input_keys:
                        available_keys = list(f.keys())
                        return {
                            "path": file_path,
                            "valid": False,
                            "error": f"Missing input keys {missing_input_keys}. "
                            f"Available keys: {available_keys}",
                            "n_samples": 0,
                            "available_keys": available_keys,
                        }
                elif self.input_key not in f:
                    available_keys = list(f.keys())
                    return {
                        "path": file_path,
                        "valid": False,
                        "error": f"Missing input key '{self.input_key}'. "
                        f"Available keys: {available_keys}",
                        "n_samples": 0,
                        "available_keys": available_keys,
                    }

                # Check target key if not in autoencoder mode
                if not self.is_autoencoder_mode:
                    if self.is_multi_target:
                        missing_target_keys = [
                            key for key in self.target_key if key not in f
                        ]
                        if missing_target_keys:
                            available_keys = list(f.keys())
                            return {
                                "path": file_path,
                                "valid": False,
                                "error": f"Missing target keys "
                                f"{missing_target_keys}. "
                                f"Available keys: {available_keys}",
                                "n_samples": 0,
                                "available_keys": available_keys,
                            }
                    elif self.target_key not in f:
                        available_keys = list(f.keys())
                        return {
                            "path": file_path,
                            "valid": False,
                            "error": f"Missing target key '{self.target_key}'. "
                            f"Available keys: {available_keys}",
                            "n_samples": 0,
                            "available_keys": available_keys,
                        }

                # Get sequence length from input shape
                # (assume time dimension is axis 1)
                if self.is_multi_input:
                    # Use first input key to determine sequence length
                    first_key = self.input_key[0]
                    input_shape = f[first_key].shape
                else:
                    input_shape = f[self.input_key].shape
                n_samples = (
                    input_shape[1] if len(input_shape) >= 2 else input_shape[0]
                )

                # Get target shape if available
                target_shape = None
                if not self.is_autoencoder_mode:
                    if self.is_multi_target:
                        target_shape = {
                            key: f[key].shape
                            for key in self.target_key
                            if key in f
                        }
                    elif self.target_key in f:
                        target_shape = f[self.target_key].shape

                metadata = {
                    "path": file_path,
                    "valid": True,
                    "n_samples": n_samples,
                    "input_shape": input_shape,
                    "target_shape": target_shape,
                    "available_keys": list(f.keys()),
                    "is_autoencoder_mode": self.is_autoencoder_mode,
                }

            return metadata
        except Exception as e:
            return {
                "path": file_path,
                "valid": False,
                "error": str(e),
                "n_samples": 0,
            }

    def _get_sequence_length(self, file_metadata: dict[str, Any]) -> int:
        """Get sequence length from metadata."""
        return file_metadata.get("n_samples", 0)

    def _open_file(self, file_path: str) -> Any:
        """Open HDF5 file."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5Dataset. "
                "Install with: pip install h5py"
            )

        return h5py.File(file_path, "r")

    def _close_file(self, file_handle: Any) -> None:
        """Close HDF5 file handle."""
        file_handle.close()

    def _extract_subsequence(
        self,
        file_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> tuple[
        torch.Tensor | dict[str, torch.Tensor],
        torch.Tensor | dict[str, torch.Tensor],
    ]:
        """Extract subsequence from HDF5 file."""
        file_handle = self.get_file_handle(file_idx)

        # Extract input data
        if self.is_multi_input:
            input_data = {}
            for key in self.input_key:
                input_arr = file_handle[key][:, :, start_idx:end_idx]
                input_data[key] = torch.from_numpy(input_arr).float()
        else:
            input_arr = file_handle[self.input_key][:, :, start_idx:end_idx]
            input_data = torch.from_numpy(input_arr).float()

        # Extract target data
        if self.is_autoencoder_mode:
            # Autoencoder mode: target = input
            if self.is_multi_input:
                target_data = {
                    key: tensor.clone() for key, tensor in input_data.items()
                }
            else:
                target_data = input_data.clone()
        else:
            # Supervised mode: use separate target
            if self.is_multi_target:
                target_data = {}
                for key in self.target_key:
                    target_arr = file_handle[key][:, :, start_idx:end_idx]
                    target_data[key] = torch.from_numpy(target_arr).float()
            else:
                target_arr = file_handle[self.target_key][
                    :, :, start_idx:end_idx
                ]
                target_data = torch.from_numpy(target_arr).float()

        # Apply target slice if specified (only for single tensor targets)
        if self.target_slice is not None and not self.is_multi_target:
            target_data = target_data[self.target_slice]

        return input_data, target_data


class NumpyDataset(MultiFileDataset):
    """Dataset for NumPy .npz files with configurable keys.

    Supports flexible key configuration for different file formats.
    Can work with input-only data (autoencoders) or input-target pairs.
    """

    def __init__(
        self,
        file_paths: Union[str, list[str], Path],
        subseq_len: int,
        input_key: Union[str, list[str]] = "input",
        target_key: Optional[Union[str, list[str]]] = None,
        target_slice: Optional[tuple] = None,
        auto_detect_keys: bool = False,
        **kwargs,
    ) -> None:
        """Initialize NumPy dataset.

        Parameters
        ----------
        file_paths : Union[str, list[str], Path]
            Path(s) to .npz files.
        subseq_len : int
            Length of subsequences to extract.
        input_key : Union[str, list[str]], optional
            Key(s) for input data in .npz files. If list,
            returns dictionary of inputs, by default 'input'.
        target_key : Optional[Union[str, list[str]]], optional
            Key(s) for target data in .npz files. If None, uses input as target
            (autoencoder mode). If list, returns dictionary of targets,
            by default None.
        target_slice : Optional[tuple], optional
            Slice to apply to target tensor, by default None.
        auto_detect_keys : bool, optional
            Whether to automatically detect keys from first valid file,
            by default False.
        **kwargs
            Additional arguments passed to MultiFileDataset.
        """
        self.input_key = input_key
        self.target_key = target_key
        self.target_slice = target_slice
        self.auto_detect_keys = auto_detect_keys
        self.is_autoencoder_mode = target_key is None

        # Handle multiple keys
        self.is_multi_input = isinstance(input_key, list)
        self.is_multi_target = isinstance(target_key, list)

        # Store original keys for auto-detection fallback
        self._original_input_key = input_key
        self._original_target_key = target_key

        # Track if keys have been detected
        self._keys_detected = False

        super().__init__(file_paths, subseq_len, **kwargs)

    def _detect_keys(self, data_dict: dict[str, Any], file_path: str) -> None:
        """Auto-detect keys from NumPy file contents."""
        available_keys = list(data_dict.keys())

        print(f"Auto-detecting keys from {file_path}")
        print(f"Available keys: {available_keys}")

        # Skip auto-detection if keys are already provided as lists
        if self.is_multi_input or self.is_multi_target:
            print("Skipping auto-detection since list of keys provided")
            return

        # Try to find input key
        if not self.is_multi_input:
            input_candidates = ["input", "data", "x", "features"]
            for candidate in input_candidates:
                if candidate in available_keys:
                    self.input_key = candidate
                    print(f"Detected input_key: {self.input_key}")
                    break

        # Try to find target key
        # (only if not in autoencoder mode and not multi-target)
        if not self.is_autoencoder_mode and not self.is_multi_target:
            target_candidates = ["target", "label", "y", "output"]
            for candidate in target_candidates:
                if candidate in available_keys:
                    self.target_key = candidate
                    print(f"Detected target_key: {self.target_key}")
                    break

    def _inspect_file(self, file_path: str) -> dict[str, Any]:
        """Inspect NumPy file to extract metadata."""
        try:
            with np.load(file_path, mmap_mode="r") as data:
                # Auto-detect keys from first file if requested
                if self.auto_detect_keys and not self._keys_detected:
                    self._detect_keys(data, file_path)
                    self._keys_detected = True

                # Validate required keys exist
                if self.is_multi_input:
                    missing_input_keys = [
                        key for key in self.input_key if key not in data
                    ]
                    if missing_input_keys:
                        available_keys = list(data.keys())
                        return {
                            "path": file_path,
                            "valid": False,
                            "error": f"Missing input keys {missing_input_keys}. "
                            f"Available keys: {available_keys}",
                            "n_samples": 0,
                            "available_keys": available_keys,
                        }
                elif self.input_key not in data:
                    available_keys = list(data.keys())
                    return {
                        "path": file_path,
                        "valid": False,
                        "error": f"Missing input key '{self.input_key}'. "
                        f"Available keys: {available_keys}",
                        "n_samples": 0,
                        "available_keys": available_keys,
                    }

                # Check target key if not in autoencoder mode
                if not self.is_autoencoder_mode:
                    if self.is_multi_target:
                        missing_target_keys = [
                            key for key in self.target_key if key not in data
                        ]
                        if missing_target_keys:
                            available_keys = list(data.keys())
                            return {
                                "path": file_path,
                                "valid": False,
                                "error": f"Missing target keys "
                                f"{missing_target_keys}. "
                                f"Available keys: {available_keys}",
                                "n_samples": 0,
                                "available_keys": available_keys,
                            }
                    elif self.target_key not in data:
                        available_keys = list(data.keys())
                        return {
                            "path": file_path,
                            "valid": False,
                            "error": f"Missing target key '{self.target_key}'. "
                            f"Available keys: {available_keys}",
                            "n_samples": 0,
                            "available_keys": available_keys,
                        }

                # Get sequence length from input shape
                # (assume time dimension is axis 1)
                if self.is_multi_input:
                    # Use first input key to determine sequence length
                    first_key = self.input_key[0]
                    input_arr = data[first_key]
                else:
                    input_arr = data[self.input_key]
                input_shape = input_arr.shape
                n_samples = (
                    input_shape[1] if len(input_shape) >= 2 else input_shape[0]
                )

                # Get target shape if available
                target_shape = None
                if not self.is_autoencoder_mode:
                    if self.is_multi_target:
                        target_shape = {
                            key: data[key].shape
                            for key in self.target_key
                            if key in data
                        }
                    elif self.target_key in data:
                        target_shape = data[self.target_key].shape

                metadata = {
                    "path": file_path,
                    "valid": True,
                    "n_samples": n_samples,
                    "input_shape": input_shape,
                    "target_shape": target_shape,
                    "available_keys": list(data.keys()),
                    "is_autoencoder_mode": self.is_autoencoder_mode,
                }

            return metadata
        except Exception as e:
            return {
                "path": file_path,
                "valid": False,
                "error": str(e),
                "n_samples": 0,
            }

    def _get_sequence_length(self, file_metadata: dict[str, Any]) -> int:
        """Get sequence length from metadata."""
        return file_metadata.get("n_samples", 0)

    def _open_file(self, file_path: str) -> Any:
        """Open NumPy file with memory mapping."""
        return np.load(file_path, mmap_mode="r")

    def _close_file(self, file_handle: Any) -> None:
        """Close NumPy file handle."""
        file_handle.close()

    def _extract_subsequence(
        self, file_idx: int, start_idx: int, end_idx: int
    ) -> tuple[
        Union[torch.Tensor, dict[str, torch.Tensor]],
        Union[torch.Tensor, dict[str, torch.Tensor]],
    ]:
        """Extract subsequence from NumPy file."""
        file_handle = self.get_file_handle(file_idx)

        # Extract input data
        if self.is_multi_input:
            input_data = {}
            for key in self.input_key:
                input_arr = file_handle[key][:, :, start_idx:end_idx]
                input_data[key] = torch.from_numpy(input_arr).float()
        else:
            input_arr = file_handle[self.input_key][:, :, start_idx:end_idx]
            input_data = torch.from_numpy(input_arr).float()

        # Extract target data
        if self.is_autoencoder_mode:
            # Autoencoder mode: target = input
            if self.is_multi_input:
                target_data = {
                    key: tensor.clone() for key, tensor in input_data.items()
                }
            else:
                target_data = input_data.clone()
        else:
            # Supervised mode: use separate target
            if self.is_multi_target:
                target_data = {}
                for key in self.target_key:
                    target_arr = file_handle[key][:, :, start_idx:end_idx]
                    target_data[key] = torch.from_numpy(target_arr).float()
            else:
                target_arr = file_handle[self.target_key][
                    :, :, start_idx:end_idx
                ]
                target_data = torch.from_numpy(target_arr).float()

        # Apply target slice if specified (only for single tensor targets)
        if self.target_slice is not None and not self.is_multi_target:
            target_data = target_data[self.target_slice]

        return input_data, target_data


# Helper function for creating worker init function
def worker_init_fn(worker_id: int) -> None:
    """Worker initialization function for LazyFileDataset subclasses."""
    from torch.utils.data import get_worker_info

    worker_info = get_worker_info()
    if worker_info is not None:
        worker_dataset = worker_info.dataset
        if hasattr(worker_dataset, "worker_init"):
            worker_dataset.worker_init()
        else:
            warnings.warn(
                f"Dataset in worker {worker_id} does not have worker_init "
                f"method. Got {type(worker_dataset)}."
            )


# Helper function for getting the file paths for an indexed joblib dataset
def get_file_paths(
    dataset_name: str,
    base_path: str | Path = "",
) -> list[str]:
    """Get the file paths for an indexed joblib dataset."""
    import pandas as pd

    if base_path == "":
        base_path = Path("/scratch/gpfs/EKOLEMEN/hackathon/foundation25/")
    elif isinstance(base_path, str):
        base_path = Path(base_path)
    file_path = base_path / dataset_name / "index.csv"

    if not file_path.exists():
        available_datasets = [path.name for path in base_path.glob("*")]
        raise ValueError(
            f"File '{dataset_name}' does not exist.\n"
            f"Available datasets: {available_datasets}"
        )

    df = pd.read_csv(file_path)
    return df.values[:, 0].sort()
