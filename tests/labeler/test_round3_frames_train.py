"""Round three, Task 2.9: `RowsCNN` and its training."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from labeler import frames
from labeler.events.review import build as review_build
from labeler.frames import prepare, train
from labeler.frames import shots as frames_shots
from labeler.frames.model import RowsCNN, bin_logits
from labeler.frames.targets import ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN

from . import editor_tree, frames_tree
from .frames_tree import ELM, IP_SHOT, LABELS_SHOT, SHOTS

#: Each spec's feature channels (Task 2.6's widths).
WIDTHS = {"elm_frames": 2, "hmode_frames": 5, "ntm_frames": 42, "sawtooth_frames": 41}
KEYS = {"state_dict", "config", "threshold", "channels", "subs", "spec", "split_sha256"}


def _subs(method: str) -> int:
    return round(10 / frames.SPECS[method].sub_ms)


@pytest.mark.parametrize("method", sorted(WIDTHS))
def test_the_model_is_under_40k_parameters(method):
    model = RowsCNN(WIDTHS[method], _subs(method))
    assert sum(p.numel() for p in model.parameters()) < 40_000


@pytest.mark.parametrize("method", sorted(WIDTHS))
def test_one_logit_a_frame(method):
    subs = _subs(method)
    model = RowsCNN(WIDTHS[method], subs).eval()
    with torch.no_grad():
        out = model(torch.rand(3, WIDTHS[method], subs * 37))
    assert out.shape == (3, 37)
    if subs > 1:
        with pytest.raises(ValueError, match="whole frames"):
            model(torch.rand(1, WIDTHS[method], subs * 37 + 1))


def test_bin_logits_pool_frames_by_max_or_mean():
    logits = torch.tensor([[0.0, 1.0, -2.0, 3.0, 0.5, -1.0, -1.0, -1.0, -1.0, 4.0]])
    assert bin_logits(logits, 5, "max").tolist() == [[3.0, 4.0]]
    assert torch.allclose(bin_logits(logits, 5, "mean"), torch.tensor([[0.5, 0.0]]))
    assert torch.equal(bin_logits(logits, 1, "mean"), logits)
    with pytest.raises(ValueError):
        bin_logits(logits, 3, "max")
    with pytest.raises(ValueError):
        bin_logits(logits, 5, "median")


def test_masked_bins_carry_no_loss():
    states = np.array([ABSENT, PRESENT_T, UNKNOWN, UNCERTAIN_T, PRESENT_T], np.int8)
    observed = np.array([True, True, True, True, False])
    weights = train.bin_weights(states, observed, pos_weight=3.0)
    assert weights.tolist() == [1.0, 3.0, 0.0, 0.0, 0.0]
    logits = torch.zeros(1, 25, requires_grad=True)
    loss = train.bin_loss(logits, states[None], weights[None], 5, "mean")
    loss.backward()
    grad = logits.grad.reshape(5, 5)
    assert torch.all(grad[2:] == 0) and torch.all(grad[:2] != 0)
    # Whatever the masked bins' logits, the loss is the same.
    moved = torch.zeros(1, 25)
    moved[0, 10:] = 7.0
    again = train.bin_loss(moved, states[None], weights[None], 5, "mean")
    assert torch.isclose(loss.detach(), again)


def test_the_positive_weight_is_the_ratio_up_to_its_cap():
    absent = np.full(40, ABSENT, np.int8)
    assert train.positive_weight([np.r_[absent, [PRESENT_T] * 4]], 5.0) == 5.0
    assert train.positive_weight([np.r_[absent[:8], [PRESENT_T] * 4]], 5.0) == 2.0
    assert train.positive_weight([np.r_[absent[:2], [PRESENT_T] * 4]], 5.0) == 1.0
    assert train.positive_weight([absent], 5.0) == 1.0


def test_the_threshold_has_the_best_f1_ties_nearest_a_half():
    prob = np.array([0.95, 0.85, 0.3, 0.2, 0.1])
    states = np.array([PRESENT_T, PRESENT_T, ABSENT, ABSENT, ABSENT], np.int8)
    scored = np.ones(5, bool)
    assert train.pick_threshold([prob], [states], [scored]) == 0.5
    states = np.array([PRESENT_T, PRESENT_T, PRESENT_T, ABSENT, ABSENT], np.int8)
    assert train.pick_threshold([prob], [states], [scored]) == 0.3


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """The ELM method's split and features on the tree, with a val shot: the
    tree's ELM split has two train shots and the owner's, so the owner's is
    trained on here and IP_SHOT is val."""
    paths = frames_tree.build(tmp_path / "tree")
    for event in frames_tree.STORE_ROWS:
        monkeypatch.setitem(review_build.BUILDERS, event, frames_tree.builder)
    frames_shots.make(paths, "elm_frames")
    shots = [SHOTS[ELM], IP_SHOT, LABELS_SHOT]
    prepare.build_stores(paths, "elm_frames", shots)
    assert prepare.prepare(paths, "elm_frames", shots)["written"] == sorted(shots)
    path = frames.shots_file(paths, "elm_frames")
    frame = pd.read_csv(path)
    frame["split"] = frame.shot.map({SHOTS[ELM]: "train", IP_SHOT: "val"}).fillna(
        "train"
    )
    frame.to_csv(path, index=False)
    return paths


def test_fit_saves_a_model_with_its_threshold(prepared):
    path = train.fit("elm_frames", prepared, train.TrainConfig(epochs=2), log=None)
    assert path == frames.model_dir(prepared, "elm_frames") / "model.pt"
    model, blob = train.load(path)
    assert KEYS <= set(blob)
    assert 0.10 <= blob["threshold"] <= 0.90
    assert blob["threshold"] in train.THRESHOLDS
    assert (blob["channels"], blob["subs"]) == (2, 5)
    split = frames.shots_file(prepared, "elm_frames").read_bytes()
    assert blob["split_sha256"] == hashlib.sha256(split).hexdigest()
    assert blob["config"]["epochs"] == 2 and blob["spec"]["method"] == "elm_frames"
    assert blob["shots"] == {
        "train": sorted([SHOTS[ELM], LABELS_SHOT]),
        "val": [IP_SHOT],
    }
    assert len(blob["history"]) <= 2 and not model.training
    assert (path.parent / "training.json").is_file()
    # A model is trained once on its split.
    with pytest.raises(FileExistsError):
        train.fit("elm_frames", prepared, train.TrainConfig(epochs=1), log=None)


def test_a_pilot_writes_under_runs_and_may_be_replaced(prepared):
    out = prepared.runs / "frames" / "pilot" / "elm_frames"
    for _ in range(2):
        path = train.fit(
            "elm_frames", prepared, train.TrainConfig(epochs=1), out=out, log=None
        )
        assert path == out / "model.pt"
    assert not (frames.model_dir(prepared, "elm_frames") / "model.pt").exists()


def test_torch_takes_slurm_cpus_per_task_threads(prepared, monkeypatch, capsys):
    editor_tree.use_env(monkeypatch, prepared)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    seen = {}

    def fake_fit(method, paths, config, **kwargs):
        seen.update(threads=torch.get_num_threads(), epochs=config.epochs, **kwargs)
        return kwargs["out"] / "model.pt"

    monkeypatch.setattr(train, "fit", fake_fit)
    threads = torch.get_num_threads()
    try:
        assert train.main(["--method", "elm_frames", "--pilot", "--limit", "4"]) == 0
    finally:
        torch.set_num_threads(threads)
    assert seen["threads"] == 3 and seen["epochs"] == 2 and seen["limit"] == 4
    assert seen["out"] == prepared.runs / "frames" / "pilot" / "elm_frames"
