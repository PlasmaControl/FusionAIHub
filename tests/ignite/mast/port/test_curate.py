"""TokEye mode-window oversampling (`port.curate`) on the synthetic three-shot frame cache."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from tokamak_foundation_model.ignite.mast.port import curate, layout
from tokamak_foundation_model.ignite.mast.port.layout import td

from .test_port import _make_synthetic_three_shot_fixture


def test_active_starts_counts_predicted_frames_only():
    act = np.zeros(14, dtype=bool)
    act[6:12] = True
    # win 8, k0 4: predicted frames [s+4, s+8) must hold >= 4 active frames
    assert curate.active_starts(act, 14, 8, 4, 4) == [2, 3, 4]
    assert curate.active_starts(act, 14, 8, 4, 5) == []
    assert curate.active_starts(np.zeros(3, dtype=bool), 14, 8, 4, 1) == []   # short array = inactive


def test_patch_repeats_active_train_windows_only(tmp_path):
    _, fc_dir, *_ = _make_synthetic_three_shot_fixture(tmp_path)
    cfg = SimpleNamespace(modalities=layout.modality_specs_from_manifest(fc_dir), max_frames=8, k0_seed=4)
    rows = [("1001", "omv", f, 0.05 if 6 <= f < 12 else 0.0) for f in range(14)]
    rows += [("1003", "omv", f, 0.05) for f in range(14)]          # val shot: never oversampled
    pq = tmp_path / "frames.parquet"
    pd.DataFrame(rows, columns=["shot", "sig", "frame", "mode"]).to_parquet(pq)
    shots = ["1001", "1002", "1003"]
    base = td.FrameCodeDataset(fc_dir, shots, cfg)
    logs = []
    curate.patch_mode_oversample(pq, 2, ["1001", "1002"], min_active=4, log=logs.append)
    try:
        ds = td.FrameCodeDataset(fc_dir, shots, cfg)
        ds_epoch = td.FrameCodeDataset(fc_dir, shots, cfg, windows_per_shot=2)
        ds_epoch.set_epoch(3)
    finally:
        curate.unpatch_mode_oversample()
    assert td.FrameCodeDataset.__init__ is curate._ORIG["init"]
    extra = ds.index[len(base.index):]
    assert ds.index[: len(base.index)] == base.index
    assert sorted(extra) == [(0, 2), (0, 2), (0, 3), (0, 3), (0, 4), (0, 4)]
    assert len(logs) == 2 and "+6 windows" in logs[0]
    # with per-epoch redraws the extra windows are appended again after every redraw
    assert ds_epoch.index[-6:] == extra and len(ds_epoch.index) == 3 * 2 + 6
    # without a factor nothing is patched
    curate.patch_mode_oversample(pq, 0, ["1001"])
    assert td.FrameCodeDataset.__init__ is curate._ORIG["init"]
