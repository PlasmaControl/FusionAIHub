"""Checkpoint failure modes and training seeds independent of shot partitions."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from labeler.elm import inputs, labels, net, train


def _shot(shot, present=False):
    spans = pd.DataFrame(
        [
            (shot, 0.0, 100.0, "absent"),
            (shot, 100.0, 200.0, "crowd" if present else "absent"),
        ],
        columns=["shot", "t_start", "t_end", "kind"],
    )
    x = np.zeros((inputs.N_CHANNELS, net.UNIT), dtype=np.float32)
    x[inputs.VALID] = 1.0
    return train.ShotData(
        shot,
        x,
        labels.dense(spans, 256),
        256,
        spans,
        np.array([0.0]),
        np.array([200.0]),
        labels.scored_bins(spans, [0.0], [200.0]),
    )


def test_all_nan_validation_auprc_has_clear_error():
    data = {1: _shot(1, present=True), 2: _shot(2)}
    cfg = train.Config(epochs=1, iters=2, batch=1, crop_ms=256)
    with pytest.raises(ValueError, match="no finite inner-validation AUPRC"):
        train.train_fold(data, [1], [2], cfg, torch.device("cpu"), log=lambda _: None)


def test_finite_validation_selects_a_usable_checkpoint():
    data = {1: _shot(1, present=True), 2: _shot(2, present=True)}
    cfg = train.Config(epochs=1, iters=2, batch=1, crop_ms=256)
    state, history, best = train.train_fold(
        data, [1], [2], cfg, torch.device("cpu"), log=lambda _: None
    )
    assert state and len(history) == 1
    assert np.isfinite(best["val_auprc"])
    assert np.isfinite(best["val_threshold"])


def test_training_seed_changes_model_seed_without_changing_shot_splits(
    tmp_path, monkeypatch
):
    tables = tmp_path / "tables" / "edge_localized_mode" / "review"
    tables.mkdir(parents=True)
    rows = [(s, 0.0, 100.0, 0, None) for s in range(1, 7)]
    pd.DataFrame(
        rows, columns=["shot", "t_start", "t_end", "category", "attrs"]
    ).to_csv(tables / "labels.csv", index=False)
    (tmp_path / "catalog").mkdir()
    pd.DataFrame({"shot": range(1, 7), "split": ["train"] * 6}).to_csv(
        tmp_path / "catalog" / "cohort.csv", index=False
    )
    prepared = tmp_path / "round4" / "elm" / "inputs"
    prepared.mkdir(parents=True)
    for shot in range(1, 7):
        np.save(prepared / f"{shot}.npy", _shot(shot).x)
    monkeypatch.setenv("LABELER_ROOT", str(tmp_path))
    monkeypatch.setenv("LABELER_LABEL_TABLES", str(tmp_path / "tables"))
    fitted = []
    state = net.ElmUNet().state_dict()
    best = {"val_threshold": 0.5, "epoch": 0, "val_auprc": 0.8}

    def fitting(data, fit_shots, val_shots, cfg, device):
        fitted.append((fit_shots, val_shots, cfg.seed))
        return state, [best], best

    monkeypatch.setattr(train, "train_fold", fitting)
    monkeypatch.setattr(train, "onset_threshold", lambda *args: (0.4, 0.0))
    monkeypatch.setattr(
        train, "predict", lambda model, x, device: np.zeros((2, x.shape[1] // 10))
    )
    records = []
    for training_seed in (21, 22):
        args = SimpleNamespace(
            run=f"seed{training_seed}",
            epochs=1,
            iters=1,
            batch=1,
            lr=0.002,
            seed=12,
            training_seed=training_seed,
            device="cpu",
            folds=2,
            inner_val=1,
            only=None,
        )
        train.run(args)
        path = tmp_path / "round4" / "elm" / "cv" / args.run / "run.json"
        records.append(json.loads(path.read_text()))
    assert records[0]["folds"] == records[1]["folds"]
    assert records[0]["partition_seed"] == records[1]["partition_seed"] == 12
    assert [(t, v) for t, v, _ in fitted[:2]] == [(t, v) for t, v, _ in fitted[2:]]
    assert [s for _, _, s in fitted] == [21, 121, 22, 122]
