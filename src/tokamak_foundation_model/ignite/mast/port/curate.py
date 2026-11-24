"""Oversample TokEye mode-active windows in dev-peter's dynamics training (off unless asked).

dev-peter's ``FrameCodeDataset`` already oversamples windows around actuator steps
(``event_windows_per_shot``) so the model sees more of the plasma responding to its
actuators. This does the same for MHD modes: a training window whose predicted frames hold at
least ``min_active`` frames of TokEye coherent activity on a kept Mirnov spectrogram
(``port.specquality``: mean over the probe-level ``ok`` probes, at or above
``specquality.MODE_COH``) is added ``factor`` extra times to the epoch's index. Validation
shots are never touched, so the trainer's fixed validation windows are unchanged.

Applied by wrapping ``train_dynamics.FrameCodeDataset.__init__`` and ``set_epoch`` from
outside, like ``port.layout.patch_ignite``; `unpatch_mode_oversample` restores both.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from . import specquality
from .layout import td

_ORIG = {"init": td.FrameCodeDataset.__init__, "set_epoch": td.FrameCodeDataset.set_epoch}


def mode_active_by_shot(frames_parquet: Path | str, mode_coh: float = specquality.MODE_COH) -> dict[str, np.ndarray]:
    """``{shot: (n,) bool}``: frame is mode-active on either kept spectrogram signal.

    ``frames_parquet`` is ``tokeye_keep.py``'s ``frames.parquet`` (shot, sig, frame, mode).
    """
    fr = pd.read_parquet(frames_parquet)
    out: dict[str, np.ndarray] = {}
    for shot, d in fr.groupby("shot"):
        n = int(d.frame.max()) + 1
        act = np.zeros(n, dtype=bool)
        hit = d[d["mode"] >= mode_coh].frame.to_numpy()
        act[hit] = True
        out[str(shot)] = act
    return out


def active_starts(active: np.ndarray, n_frames: int, win: int, k0: int, min_active: int) -> list[int]:
    """Window starts whose predicted frames ``[s + k0, s + win)`` hold ``>= min_active`` active frames."""
    a = np.zeros(int(n_frames), dtype=np.int64)
    m = min(active.size, a.size)
    a[:m] = active[:m]
    c = np.r_[0, np.cumsum(a)]
    return [s for s in range(0, int(n_frames) - int(win) + 1)
            if c[s + win] - c[s + k0] >= int(min_active)]


def patch_mode_oversample(
    frames_parquet: Path | str,
    factor: int,
    train_shots: Iterable[str],
    *,
    min_active: int = 4,
    mode_coh: float = specquality.MODE_COH,
    log=print,
) -> None:
    """Make every ``FrameCodeDataset`` over training shots repeat mode-active windows ``factor`` times."""
    if int(factor) < 1:
        return
    active = mode_active_by_shot(frames_parquet, mode_coh)
    train = {str(s) for s in train_shots}

    def __init__(self, cache_dir, shots, cfg, *args, **kwargs):
        _ORIG["init"](self, cache_dir, shots, cfg, *args, **kwargs)
        k0 = int(getattr(cfg, "k0_seed", 0))
        extra = []
        for shot_index, shot in enumerate(self.shot_ids):
            if shot not in train or shot not in active:
                continue
            n = int(self.shots[shot_index]["n_frames"])
            starts = active_starts(active[shot], n, self.win, k0, min_active)
            extra.extend((shot_index, s) for s in starts for _ in range(int(factor)))
        self._tokeye_extra = extra
        self.index.extend(extra)
        if extra:
            log(f"[tokeye] oversample: +{len(extra)} windows (x{int(factor)} of {len(extra) // int(factor)} "
                f"mode-active starts; min_active={min_active}, mode_coh={mode_coh}) -> {len(self.index)} total")

    def set_epoch(self, epoch: int) -> None:
        _ORIG["set_epoch"](self, epoch)
        if self.windows_per_shot:                  # the original redraws self.index from scratch
            self.index.extend(getattr(self, "_tokeye_extra", []))

    td.FrameCodeDataset.__init__ = __init__
    td.FrameCodeDataset.set_epoch = set_epoch


def unpatch_mode_oversample() -> None:
    """Restore dev-peter's ``FrameCodeDataset``."""
    td.FrameCodeDataset.__init__ = _ORIG["init"]
    td.FrameCodeDataset.set_epoch = _ORIG["set_epoch"]


def train_shots_from_splits(cache_dir: Path | str) -> list[str]:
    """The ``train`` list of the cache's ``splits.json``."""
    import json

    doc: Mapping = json.loads((Path(cache_dir) / "splits.json").read_text())
    return [str(s) for s in doc["train"]]
