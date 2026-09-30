"""Round three, Task 2.9: `RowsCNN` and its training."""

from __future__ import annotations

import dataclasses
import hashlib
import json

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
    both = train.bin_weights(states, observed, 1.0, absent_weight=2.5)
    assert both.tolist() == [2.5, 1.0, 0.0, 0.0, 0.0]
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


def test_the_minority_class_weighs_the_ratio_up_to_its_cap():
    # F3: whichever class is the fewer, present or absent, weighs the ratio.
    absent = np.full(40, ABSENT, np.int8)
    present = np.full(40, PRESENT_T, np.int8)
    ignored = np.array([UNKNOWN, UNCERTAIN_T] * 10, np.int8)  # never counted

    def weights(*parts):
        return train.class_weights([np.r_[parts]], 5.0)

    assert weights(absent, present[:4], ignored) == {"absent": 1.0, "present": 5.0}
    assert weights(absent[:8], present[:4]) == {"absent": 1.0, "present": 2.0}
    # H-mode's case: H the majority, so L (absent) weighs the ratio.
    assert weights(absent[:2], present[:4]) == {"absent": 2.0, "present": 1.0}
    assert weights(absent[:4], present) == {"absent": 5.0, "present": 1.0}
    assert weights(absent[:4], present[:4]) == {"absent": 1.0, "present": 1.0}
    assert weights(absent) == {"absent": 1.0, "present": 1.0}
    assert weights(present) == {"absent": 1.0, "present": 1.0}
    assert train.class_weights([], 5.0) == {"absent": 1.0, "present": 1.0}


def _shot(states) -> train.Shot:
    states = np.asarray(states, np.int8)
    return train.Shot(1, np.zeros((1, len(states))), states, np.ones(len(states), bool))


def test_balanced_crops_centre_half_on_absent_and_half_on_present():
    # F11: 30 bins, absent from 12 to 17; a present-only shot; an unscored one.
    mixed = _shot([PRESENT_T] * 12 + [ABSENT] * 6 + [PRESENT_T] * 12)
    only = _shot([PRESENT_T] * 25 + [UNCERTAIN_T] * 5)
    unscored = _shot([UNKNOWN] * 30)
    shots = [mixed, only, unscored]
    rng = np.random.default_rng(0)
    windows = train.crop_windows(shots, rng, 8, balance=True)
    of = {i: [w for w in windows if w[0] == i] for i in range(3)}
    assert train.centre_counts(of[0]) == {"absent": 2, "present": 2}
    assert train.centre_counts(of[1]) == {"present": train.CROPS_PER_SHOT}
    assert of[2] == []  # no scored bin, no crop
    for i, start, name in windows:
        assert 0 <= start <= 30 - 8
        assert (shots[i].states[start : start + 8] == train.CENTRES[name]).any()
    # An absent bin is never near an edge here, so its crop is centred on it.
    assert all(mixed.states[w[1] + 4] == ABSENT for w in of[0] if w[2] == "absent")
    # Over the train shots, as many crops centre on absent time as on present.
    many = train.crop_windows([mixed] * 5, rng, 8, balance=True)
    assert train.centre_counts(many) == {"absent": 10, "present": 10}
    # The crops' scored bins give the class weights, not the shots' (F11).
    scored = train.crop_states([mixed], of[0], 8)
    assert [len(x) for x in scored] == [8] * 4
    assert all(np.isin(x, (ABSENT, PRESENT_T)).all() for x in scored)
    assert train.class_weights(scored, 5.0) == train.class_weights(
        [np.concatenate(scored)], 5.0
    )
    # Unbalanced, each crop starts at random, a short shot's at 0.
    windows = train.crop_windows([mixed, _shot([ABSENT] * 5)], rng, 8)
    assert train.centre_counts(windows) == {"random": 2 * train.CROPS_PER_SHOT}
    assert {w[1] for w in windows if w[0] == 1} == {0}
    x, states, weights = train.crops(
        [mixed, _shot([ABSENT] * 5)], windows, 8, 1, 1, {"absent": 2.0, "present": 1.0}
    )
    assert x.shape == (8, 1, 8) and states.shape == weights.shape == (8, 8)
    assert weights[-1].tolist() == [2.0] * 5 + [0.0] * 3  # the padding unscored


def test_the_threshold_has_the_best_f1_ties_nearest_a_half():
    prob = np.array([0.95, 0.85, 0.3, 0.2, 0.1])
    states = np.array([PRESENT_T, PRESENT_T, ABSENT, ABSENT, ABSENT], np.int8)
    scored = np.ones(5, bool)
    assert train.pick_threshold([prob], [states], [scored]) == 0.5
    states = np.array([PRESENT_T, PRESENT_T, PRESENT_T, ABSENT, ABSENT], np.int8)
    assert train.pick_threshold([prob], [states], [scored]) == 0.3


def test_the_macro_threshold_weighs_the_absent_class_too():
    # F3: eight H bins and two L bins, one L bin at P(H) 0.3 and one H bin at
    # 0.25. Calling both H is F1(H)'s best (0.941 from 0.15 to 0.25, 0.25 the
    # nearest 0.5), but it costs F1(L) a third; the macro rule's best (0.867,
    # from 0.35 to 0.90) calls both L.
    prob = np.array([0.9] * 7 + [0.25, 0.3, 0.1])
    states = np.array([PRESENT_T] * 8 + [ABSENT] * 2, np.int8)
    scored = np.ones(10, bool)
    assert train.pick_threshold([prob], [states], [scored], "f1") == 0.25
    assert train.pick_threshold([prob], [states], [scored], "macro_f1") == 0.5
    cells = train.bin_cells(prob, states, 0.5, scored)  # tp 7, fp 0, fn 1, tn 2
    assert cells.tolist() == [7, 0, 1, 2]
    f1_h, f1_l = 14 / 15, 4 / 5
    assert train.rule_score(cells, "macro_f1") == pytest.approx((f1_h + f1_l) / 2)
    assert train.rule_score(cells, "f1") == pytest.approx(f1_h)
    with pytest.raises(ValueError, match="rule"):
        train.rule_score(cells, "accuracy")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """The ELM method's split and features on the tree, with a val shot: the
    tree's ELM split has three train shots, the owner's among them, so IP_SHOT
    is made val here."""
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
    assert blob["threshold_rule"] == "f1"
    # Both classes' weights (F3): the scored train bins hold fewer present.
    assert set(blob["weights"]) == {"absent", "present"}
    assert blob["weights"]["absent"] == 1.0 and blob["weights"]["present"] > 1.0
    assert blob["balance_crops"] is False
    # F17: the scored train bins by class the weights came from.
    shots = [train.read_shot(prepared, "elm_frames", s) for s in blob["shots"]["train"]]
    bins = train.class_bins([s.states[s.observed] for s in shots])
    assert blob["weight_bins"] == {"from": "train", "train": bins, "crops": None}
    assert bins["present"] < bins["absent"]
    ratio = bins["absent"] / bins["present"]
    assert blob["weights"]["present"] == pytest.approx(min(ratio, 5.0))
    per_epoch = [h["centres"] for h in blob["history"]]
    assert per_epoch == [{"random": 2 * train.CROPS_PER_SHOT}] * len(per_epoch)
    assert blob["crop_centres"] == {"random": 2 * 4 * len(per_epoch)}
    assert (
        json.loads((path.parent / "training.json").read_text())["weights"]
        == (blob["weights"])
    )
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


def test_balanced_crops_weigh_the_crops_drawn(prepared, monkeypatch):
    balanced = [m for m, s in frames.SPECS.items() if s.balance_crops]
    assert balanced == ["sawtooth_frames"], "the sawteeth's alone (F11)"
    spec = dataclasses.replace(frames.SPECS["elm_frames"], balance_crops=True)
    monkeypatch.setitem(frames.SPECS, "elm_frames", spec)
    out = prepared.runs / "frames" / "pilot" / "elm_frames"
    config = train.TrainConfig(epochs=1)
    path = train.fit("elm_frames", prepared, config, out=out, log=None)
    record = json.loads((path.parent / "training.json").read_text())
    assert record["balance_crops"] is True and record["spec"]["balance_crops"]
    # The first epoch's crops, drawn at the seed, give the weights (F11).
    shots = [
        train.read_shot(prepared, "elm_frames", s) for s in record["shots"]["train"]
    ]
    crop_bins = int(train.CROP_MS // spec.bin_ms)
    rng = np.random.default_rng(train.SEED)
    windows = train.crop_windows(shots, rng, crop_bins, balance=True)
    drawn = train.crop_states(shots, windows, crop_bins)
    assert record["weights"] == train.class_weights(drawn, config.pos_weight_max)
    # F17: the crops' scored bins by class, beside the whole train shots'.
    whole = train.class_bins([s.states[s.observed] for s in shots])
    assert record["weight_bins"] == {
        "from": "crops",
        "train": whole,
        "crops": train.class_bins(drawn),
    }
    assert record["crop_centres"] == train.centre_counts(windows)
    assert record["history"][0]["centres"] == record["crop_centres"]
    assert sum(record["crop_centres"].values()) == 2 * train.CROPS_PER_SHOT


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
