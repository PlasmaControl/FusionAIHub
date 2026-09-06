"""Load a torch `state_dict` checkpoint written by this repo's own training.

Unlike `keras_h5` and `dsm_pickle`, which reconstruct somebody else's graph
from a serialized description, this runner loads a network whose class lives
in labelmaker (`labelmaker.ae.model.AeSeldNet`). The checkpoint carries the
constructor's arguments under `config`, so the file is self-describing and the
adapter never hard-codes an architecture that a retrained checkpoint might
change.

`torch.load(..., weights_only=False)` is used because the checkpoint holds
plain dicts and floats beside the tensors (`config`, `loss_config`,
`valid_metrics`). That is unpickling, which is only safe on a file whose bytes
are known - and they are: `registry.verify_artifacts` checks the card's sha256
before `run.py` ever calls `load`, exactly as it does for the pickled DSM
weights. Never point this at a checkpoint the card does not digest.

CPU only. The pixi env carries a CPU torch build; the GPU venv is for
training. `load_module` pins one intra-op thread inside a forked worker for
the same reason `keras_h5.load_ensemble` does - `run.py` forks a pool and
torch would otherwise start one thread per core in each of them.
"""
from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


def read_checkpoint(path) -> dict:
    """The raw checkpoint dict, on CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def load_module(
    path,
    build: Callable[[dict], torch.nn.Module],
) -> tuple[torch.nn.Module, dict]:
    """Rebuild a network from its checkpoint and return it in eval mode.

    `build` receives the checkpoint's `config` mapping and returns the
    constructed (untrained) module; this function loads `state_dict` into it
    strictly, so a checkpoint whose config does not describe its own weights
    is an error here rather than a silently half-initialised network.

    Returns the module and the checkpoint, since the metadata beside the
    weights - the frequency-head normalisation, the training metrics - is
    what a card and a label's attributes are written from.
    """
    if multiprocessing.parent_process() is not None:
        torch.set_num_threads(1)
    blob = read_checkpoint(path)
    for key in ("state_dict", "config"):
        if key not in blob:
            raise KeyError(f"{path}: checkpoint has no {key!r}")
    module = build(blob["config"])
    module.load_state_dict(blob["state_dict"], strict=True)
    module.eval()
    return module, blob


def run_windows(
    module: torch.nn.Module,
    x: np.ndarray,
    *,
    window: int,
    context: int,
) -> np.ndarray:
    """Evaluate a `(C, T, F)` record in overlapping time windows -> `(T, out)`.

    The network pools nothing along time, so a whole record is one legal
    forward pass - but a 23,000-frame CO2 record would need tens of GB for the
    first convolution's activations, so it is cut into `window`-frame pieces.
    Each piece is evaluated with `context` extra frames on both sides and only
    its interior is kept, so the recurrent layers see the same neighbourhood
    they would in a single pass everywhere except within `context` frames of
    the record's own ends. The seam is therefore an approximation with a
    bounded, stated width, not an invisible one.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected (channels, frames, bins), got {x.shape}")
    n = x.shape[1]
    if window <= 0 or context < 0:
        raise ValueError(f"window must be positive and context non-negative; got {window}, {context}")
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, window):
            stop = min(start + window, n)
            lo = max(start - context, 0)
            hi = min(stop + context, n)
            chunk = torch.from_numpy(np.ascontiguousarray(x[:, lo:hi, :])).unsqueeze(0)
            got = module(chunk)[0].numpy()
            out.append(got[start - lo: (start - lo) + (stop - start)])
    stacked = np.concatenate(out, axis=0)
    if stacked.shape[0] != n:
        raise RuntimeError(f"windowed output has {stacked.shape[0]} frames, expected {n}")
    return stacked


def checkpoint_meta(blob: dict, keys) -> dict[str, Any]:
    """The scalar metadata a card or a label attribute quotes, when present."""
    return {k: blob[k] for k in keys if k in blob}
