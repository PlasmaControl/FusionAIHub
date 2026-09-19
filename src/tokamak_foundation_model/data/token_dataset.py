"""Token dataset for Stage 3 multimodal prediction training.

Loads pre-computed tokens and observations from HDF5 files
produced by ``generate_token_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class TokenDataset(Dataset):
    """Loads pre-computed tokens from a single token HDF5 file.

    Each sample returns input tokens at time ``t`` and target tokens +
    observations at time ``t + prediction_horizon_chunks``.

    Parameters
    ----------
    token_h5_path : str or Path
        Path to the token HDF5 file.
    input_signals : list of str
        Signal names to include as inputs.
    target_signals : list of str
        Signal names to include as targets.
    prediction_horizon_chunks : int
        Number of chunks into the future to predict.
    """

    def __init__(
        self,
        token_h5_path: str | Path,
        input_signals: list[str],
        target_signals: list[str],
        prediction_horizon_chunks: int = 1,
    ):
        self.token_h5_path = str(token_h5_path)
        self.input_signals = input_signals
        self.target_signals = target_signals
        self.prediction_horizon_chunks = prediction_horizon_chunks
        self.h5_file: Optional[h5py.File] = None

        # Determine length from file
        with h5py.File(self.token_h5_path, "r") as f:
            # Use first available signal to get chunk count
            available = [s for s in input_signals if s in f]
            if not available:
                raise ValueError(
                    f"No input signals found in {token_h5_path}. "
                    f"Available: {list(f.keys())}"
                )
            self.n_chunks = f[available[0]]["tokens"].shape[0]
            self._available_inputs = [s for s in input_signals if s in f]
            self._available_targets = [s for s in target_signals if s in f]

    def _open_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.token_h5_path, "r")

    def __len__(self):
        return max(0, self.n_chunks - self.prediction_horizon_chunks)

    def __getitem__(self, idx):
        self._open_h5()
        target_idx = idx + self.prediction_horizon_chunks

        inputs = {}
        for sig in self._available_inputs:
            inputs[sig] = torch.from_numpy(
                self.h5_file[sig]["tokens"][idx].astype(np.float32)
            )

        targets = {}
        observations = {}
        for sig in self._available_targets:
            targets[sig] = torch.from_numpy(
                self.h5_file[sig]["tokens"][target_idx].astype(np.float32)
            )
            observations[sig] = torch.from_numpy(
                self.h5_file[sig]["observations"][target_idx].astype(np.float32)
            )

        return {
            "inputs": inputs,
            "targets": targets,
            "observations": observations,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


def collate_token_fn(batch):
    """Collate function for TokenDataset batches."""
    inputs = {}
    targets = {}
    observations = {}

    all_input_keys = batch[0]["inputs"].keys()
    all_target_keys = batch[0]["targets"].keys()
    all_obs_keys = batch[0]["observations"].keys()

    for key in all_input_keys:
        inputs[key] = torch.stack([b["inputs"][key] for b in batch])
    for key in all_target_keys:
        targets[key] = torch.stack([b["targets"][key] for b in batch])
    for key in all_obs_keys:
        observations[key] = torch.stack([b["observations"][key] for b in batch])

    return {
        "inputs": inputs,
        "targets": targets,
        "observations": observations,
    }


def make_token_dataloader(
    token_dir: str | Path,
    input_signals: list[str],
    target_signals: list[str],
    prediction_horizon_chunks: int = 1,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    val_split: float = 0.0,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train (and optionally val) dataloaders from a token directory.

    Parameters
    ----------
    token_dir : path
        Directory containing ``*_tokens.h5`` files.
    input_signals, target_signals : list of str
        Signal names.
    prediction_horizon_chunks : int
        Chunks ahead to predict.
    batch_size, num_workers : int
        DataLoader settings.
    shuffle : bool
        Shuffle training data.
    val_split : float
        Fraction of files for validation (0 = no val).

    Returns
    -------
    tuple
        ``(train_dataloader, val_dataloader)`` where val may be None.
    """
    token_dir = Path(token_dir)
    h5_files = sorted(token_dir.glob("*_tokens.h5"))

    if not h5_files:
        raise FileNotFoundError(f"No token HDF5 files found in {token_dir}")

    # Split files
    if val_split > 0:
        rng = torch.Generator().manual_seed(42)
        n_val = max(1, int(len(h5_files) * val_split))
        perm = torch.randperm(len(h5_files), generator=rng)
        val_files = [h5_files[i] for i in perm[:n_val]]
        train_files = [h5_files[i] for i in perm[n_val:]]
    else:
        train_files = h5_files
        val_files = []

    def _make_concat(files):
        datasets = []
        for f in files:
            try:
                ds = TokenDataset(
                    f, input_signals, target_signals, prediction_horizon_chunks
                )
                datasets.append(ds)
            except (ValueError, OSError) as e:
                print(f"Skipping {f}: {e}")
        return ConcatDataset(datasets) if datasets else None

    train_ds = _make_concat(train_files)
    val_ds = _make_concat(val_files) if val_files else None

    def _worker_init(worker_id):
        info = torch.utils.data.get_worker_info()
        if info is not None:
            ds = info.dataset
            if hasattr(ds, "datasets"):
                for d in ds.datasets:
                    d.h5_file = None
            else:
                ds.h5_file = None

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_token_fn,
        worker_init_fn=_worker_init,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        shuffle=shuffle,
    )

    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=collate_token_fn,
            worker_init_fn=_worker_init,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            shuffle=False,
        )

    return train_dl, val_dl
