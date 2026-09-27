"""FrameCNN and its training loop, on shapes and a toy problem it must solve."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from labeler.ae.xpower import data, train
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig


def test_one_logit_per_frame_for_any_length():
    model = FrameCNN().eval()
    for frames in (1, 7, 64):
        out = model(torch.zeros(2, 3, 175, data.SUBS * frames))
        assert out.shape == (2, frames)
    assert sum(p.numel() for p in model.parameters()) < 100_000


def test_sub_frames_must_make_whole_frames():
    with pytest.raises(ValueError, match="not whole frames"):
        FrameCNN()(torch.zeros(1, 3, 175, data.SUBS * 4 + 1))


def test_the_config_round_trips():
    config = FrameCNNConfig(width=8, dilations=(1, 3))
    assert FrameCNNConfig.from_dict(json.loads(json.dumps(config.as_dict()))) == config


def test_weights_score_owner_calls_and_up_weight_mhd_absent_frames():
    states = np.array([-1, 0, 1, 2, 3, 0, 0])
    mhd = np.array([0, 1, 1, 0, 0, 0, 1], bool)
    observed = np.array([1, 1, 1, 1, 1, 1, 0], bool)
    weights = train.frame_weights(states, mhd, 3.0, observed)
    assert weights.tolist() == [0, 3, 1, 0, 0, 1, 0]
    assert train.frame_weights(states, mhd, 3.0).tolist() == [0, 3, 1, 0, 0, 1, 3]


def test_cells_and_f1_count_only_present_and_absent_frames():
    states = np.array([1, 1, 0, 0, 2, -1])
    prob = np.array([0.9, 0.2, 0.7, 0.1, 0.9, 0.9])
    assert train.frame_cells(prob, states, 0.5).tolist() == [1, 1, 1, 1]
    assert train.f1_of([1, 1, 1, 1]) == pytest.approx(0.5)
    assert np.isnan(train.f1_of([0, 0, 0, 5]))


def test_the_threshold_is_the_best_f1_and_ties_go_to_the_middle():
    states = [np.array([1, 1, 0, 0])]
    # Only 0.30 and 0.35 separate these; the tie goes to 0.35, nearer 0.5.
    assert train.pick_threshold([np.array([0.35, 0.4, 0.2, 0.25])], states) == 0.35
    # None separates these; 0.15 and 0.20 both give the best F1, 0.8.
    assert train.pick_threshold([np.array([0.2, 0.4, 0.1, 0.3])], states) == 0.2
    # Every threshold from 0.10 to 0.90 separates these: 0.5 wins the tie.
    assert train.pick_threshold([np.array([0.95, 0.95, 0.05, 0.05])], states) == 0.5


def _toy(n_shots, rng, n=60, bins=16):
    """Shots whose AE frames carry a bright line somewhere in the band."""
    shots = []
    for i in range(n_shots):
        states = np.zeros(n, dtype=np.int8)
        start = rng.integers(5, 30)
        states[start : start + rng.integers(10, 25)] = 1
        x = rng.uniform(0, 0.2, (3, bins, data.SUBS * n)).astype(np.float32)
        line = rng.integers(0, bins)
        for k in np.flatnonzero(states == 1):
            x[:, line, data.SUBS * k : data.SUBS * (k + 1)] += 0.7
        mhd = np.zeros(n, bool)
        shots.append(
            data.Shot(i, 0, x.astype(np.float16), states, mhd, np.ones(n, bool))
        )
    return shots


def test_crops_skip_short_shots_and_carry_targets_and_weights():
    rng = np.random.default_rng(0)
    shots = _toy(2, rng)
    x, y, w = train.sample_crops(shots, rng, k=3, length=32, mhd_weight=3.0)
    assert x.shape == (6, 3, 16, data.SUBS * 32) and y.shape == w.shape == (6, 32)
    assert set(np.unique(y)) <= {0.0, 1.0} and (w == 1).all()
    with pytest.raises(ValueError, match="no shot"):
        train.sample_crops(shots, rng, k=3, length=61, mhd_weight=3.0)


def test_fit_learns_a_toy_band_line_and_saves_what_it_learned(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    shots_train, shots_val = _toy(8, rng), _toy(3, rng)
    config = train.TrainConfig(
        epochs=12, batch=8, crop_frames=32, crops_per_shot=4, patience=12
    )
    model, history, threshold = train.fit(
        shots_train, shots_val, config, log=lambda m: None
    )
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    kept = train.best_epoch(history)
    assert history[kept - 1]["val_f1"] >= 0.9
    assert 0.1 <= threshold <= 0.9
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text("shot\n")
    split = {0: "train", 1: "val", 2: "test"}
    train.save(
        tmp_path / "m",
        model,
        threshold=threshold,
        split=split,
        history=history,
        config=config,
        band_khz=data.BAND_KHZ,
        labels_file=labels_file,
        candidate="band80-mhd3",
    )
    loaded, blob = train.load(tmp_path / "m" / "model.pt")
    x = shots_val[0].x
    assert np.allclose(train.predict(loaded, x), train.predict(model, x), atol=1e-6)
    assert blob["threshold"] == threshold and blob["best_epoch"] == kept
    assert blob["band_khz"] == [80.0, 250.0] and blob["train"]["epochs"] == 12
    assert train.read_split(tmp_path / "m" / "split.csv") == split
    assert (tmp_path / "m" / "review" / "labels.csv").read_text() == "shot\n"
    record = json.loads((tmp_path / "m" / "training.json").read_text())
    assert record["counts"] == {"test": 1, "train": 1, "val": 1}
    assert record["candidate"] == blob["candidate"] == "band80-mhd3"
    assert record["band_khz"] == blob["band_khz"] == [80.0, 250.0]
    assert record["git_sha"] == blob["git_sha"]
    assert record["labels_sha256"] == blob["labels_sha256"]


@pytest.mark.parametrize("n", ["5", "21"])
def test_a_pilot_is_six_to_twenty_shots(n):
    with pytest.raises(SystemExit):
        train.main(["--candidate", "band80-mhd3", "--pilot", n])
