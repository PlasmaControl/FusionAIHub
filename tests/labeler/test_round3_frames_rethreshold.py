"""Round three, T1: a frame model's threshold re-chosen on its val shots, once."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

from labeler import frames
from labeler.events.review import build as review_build
from labeler.frames import prepare, rethreshold, train
from labeler.frames import shots as frames_shots
from labeler.frames.targets import ABSENT, PRESENT_T

from . import editor_tree, frames_tree
from .frames_tree import HMODE, SHOTS
from .test_round3_frames_evaluate import _resplit, _save_model, rechoose

METHOD = "hmode_frames"
FINE = rethreshold.FINE_THRESHOLDS


def test_the_fine_grid_holds_the_trained_one():
    assert len(FINE) == 99 and (FINE[0], FINE[-1]) == (0.01, 0.99)
    assert np.allclose(np.diff(FINE), 0.01)
    assert set(train.THRESHOLDS.tolist()) <= set(FINE.tolist())
    assert rethreshold.grid_of(FINE) == {
        "from": 0.01,
        "to": 0.99,
        "step": 0.01,
        "points": 99,
    }
    assert rethreshold.grid_of(train.THRESHOLDS)["points"] == 17


def test_the_fine_search_keeps_the_rule_and_its_tie():
    # F1 is 1 from 0.31 to 0.33, which the trained grid cannot reach (its best,
    # 6/7, runs from 0.15 to 0.30); the fine grid takes 0.33, the nearest 0.5.
    prob = np.array([0.95, 0.85, 0.33, 0.3, 0.1])
    states = np.array([PRESENT_T] * 3 + [ABSENT] * 2, np.int8)
    scored = np.ones(5, bool)
    assert train.pick_threshold([prob], [states], [scored]) == 0.3
    assert train.pick_threshold([prob], [states], [scored], grid=FINE) == 0.33
    # A best run across 0.5 still gives 0.5.
    states = np.array([PRESENT_T] * 2 + [ABSENT] * 3, np.int8)
    assert train.pick_threshold([prob], [states], [scored], grid=FINE) == 0.5
    # H-mode's macro rule: both classes right only from 0.38 to 0.39; the
    # trained grid's best, 0.867 from 0.40 to 0.90, gives 0.5.
    prob = np.array([0.9] * 7 + [0.39, 0.37, 0.1])
    states = np.array([PRESENT_T] * 8 + [ABSENT] * 2, np.int8)
    scored = np.ones(10, bool)
    assert train.pick_threshold([prob], [states], [scored], "macro_f1") == 0.5
    assert train.pick_threshold([prob], [states], [scored], "macro_f1", FINE) == 0.39


@pytest.fixture
def tree(tmp_path, monkeypatch):
    paths = frames_tree.build(tmp_path / "tree")
    for event in frames_tree.STORE_ROWS:
        monkeypatch.setitem(review_build.BUILDERS, event, frames_tree.builder)
    monkeypatch.setattr(rethreshold, "git_dirty", lambda: False)
    editor_tree.use_env(monkeypatch, paths)
    return paths


def _trained(paths, *, off: float = 0.0, val=True):
    """H-mode's one shot, prepared and made val, and a random model stopped on
    it (so its record says, unless not `val`) at the threshold the trained grid
    picks on it, plus `off`: the model's folder, its val P and the shots."""
    shot = SHOTS[HMODE]
    frames_shots.make(paths, METHOD)
    assert prepare.prepare(paths, METHOD, [shot])["written"] == [shot]
    _resplit(paths, METHOD, {shot: "val"})
    shots = [train.read_shot(paths, METHOD, shot)]
    out = _save_model(paths, METHOD, val_shots=[shot])
    model, _ = train.load(out / "model.pt")
    spec = frames.SPECS[METHOD]
    per = train.frames_per_bin(spec)
    probs = [train.bin_probs(model, s.x, per, spec.pool) for s in shots]
    picked = train.pick_threshold(
        probs, [s.states for s in shots], [s.observed for s in shots], "macro_f1"
    )
    _save_model(
        paths,
        METHOD,
        threshold=round(picked + off, 2),
        val_shots=[shot] if val else [],
        model=model,
    )
    return out, probs, shots


def test_the_threshold_is_re_chosen_once_on_the_val_shots(tree):
    out, probs, shots = _trained(tree)
    states, observed = [s.states for s in shots], [s.observed for s in shots]
    trained = train.pick_threshold(probs, states, observed, "macro_f1")
    fine = train.pick_threshold(probs, states, observed, "macro_f1", FINE)
    record = rethreshold.rethreshold(tree, METHOD)
    path = out / train.THRESHOLD_FILE
    assert json.loads(path.read_text()) == record
    assert (record["threshold"], record["trained_threshold"]) == (fine, trained)
    assert (record["threshold_rule"], record["reproduced"]) == ("macro_f1", trained)
    assert record["grid"] == rethreshold.grid_of(FINE)
    assert record["trained_grid"] == rethreshold.grid_of(train.THRESHOLDS)
    val = record["val"]
    assert val["shots"] == [SHOTS[HMODE]]
    bins = train.class_bins([s.states[s.observed] for s in shots])
    assert val["bins"] == bins and bins["absent"] and bins["present"]
    for key, threshold in (("at_threshold", fine), ("at_trained_threshold", trained)):
        cells = sum(
            train.bin_cells(p, s, threshold, o)
            for p, s, o in zip(probs, states, observed, strict=True)
        )
        assert val[key]["threshold"] == threshold
        assert val[key]["cells"] == cells.tolist()
        assert val[key]["score"] == pytest.approx(train.rule_score(cells, "macro_f1"))
        assert val[key]["f1(H)"] == pytest.approx(train.f1_of(cells))
        assert val[key]["f1(L)"] == pytest.approx(train.f1_of(cells[::-1]))
    assert val["at_threshold"]["score"] >= val["at_trained_threshold"]["score"]
    model = out / "model.pt"
    assert record["model"] == {
        "path": str(model),
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
    }
    assert record["why"] == rethreshold.WHY
    assert "after the v2 test scores were seen" in record["why"]
    assert "the test shots did not choose it" in record["why"]
    assert record["git_sha"]
    # Once.
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="once"):
        rethreshold.rethreshold(tree, METHOD)
    assert path.read_bytes() == before


@pytest.mark.parametrize("dirty", [True, None])
def test_no_threshold_is_written_from_uncommitted_code(tree, monkeypatch, dirty):
    out, _, _ = _trained(tree)
    monkeypatch.setattr(rethreshold, "git_dirty", lambda: dirty)
    with pytest.raises(RuntimeError, match="commit first"):
        rethreshold.rethreshold(tree, METHOD)
    assert not (out / train.THRESHOLD_FILE).exists()


def test_the_trained_grid_must_give_the_trained_threshold(tree, capsys):
    out, _, _ = _trained(tree, off=0.05)
    with pytest.raises(rethreshold.NotReproduced, match="the trained grid gives"):
        rethreshold.rethreshold(tree, METHOD)
    assert not (out / train.THRESHOLD_FILE).exists()
    with pytest.raises(SystemExit) as refused:
        rethreshold.main(["--method", METHOD])
    assert refused.value.code == 2
    assert "the trained grid gives" in capsys.readouterr().err
    assert not (out / train.THRESHOLD_FILE).exists()


def test_the_val_shots_must_be_the_model_s(tree):
    out, _, _ = _trained(tree, val=False)
    with pytest.raises(ValueError, match="val shots"):
        rethreshold.rethreshold(tree, METHOD)
    assert not (out / train.THRESHOLD_FILE).exists()


def test_only_the_two_named_methods(tree, capsys):
    assert rethreshold.METHODS == ("hmode_frames", "ntm_frames")
    with pytest.raises(ValueError, match="hmode_frames and ntm_frames"):
        rethreshold.rethreshold(tree, "elm_frames")
    with pytest.raises(SystemExit):
        rethreshold.main(["--method", "sawtooth_frames"])
    out, _, _ = _trained(tree)
    capsys.readouterr()
    assert rethreshold.main(["--method", METHOD]) == 0
    said = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    record = json.loads((out / train.THRESHOLD_FILE).read_text())
    assert said["threshold"] == record["threshold"]
    assert said["trained_threshold"] == record["trained_threshold"]


def test_load_gives_the_re_chosen_threshold_and_its_source(tree):
    out, _, _ = _trained(tree)
    model = out / "model.pt"
    _, blob = train.load(model)
    trained = blob["threshold"]
    assert blob["trained_threshold"] == trained
    assert (blob["threshold_source"], blob["threshold_sha256"]) == ("training", None)
    path = rechoose(out, 0.37)
    _, blob = train.load(model)
    assert (blob["threshold"], blob["trained_threshold"]) == (0.37, trained)
    assert blob["threshold_source"] == "threshold.json"
    assert blob["threshold_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # A model read from bytes names its folder.
    data = model.read_bytes()
    assert train.load(io.BytesIO(data), folder=out)[1]["threshold"] == 0.37
    with pytest.raises(TypeError, match="folder"):
        train.load(io.BytesIO(data))
    # Or it gives threshold.json's bytes as a caller pinned them (the paper
    # build): those apply, and the file on disk is not read.
    pinned = path.read_bytes()
    rechoose(out, 0.61)
    _, blob = train.load(io.BytesIO(data), threshold_json=pinned)
    assert (blob["threshold"], blob["trained_threshold"]) == (0.37, trained)
    assert blob["threshold_source"] == "threshold.json"
    assert blob["threshold_sha256"] == hashlib.sha256(pinned).hexdigest()
    _, blob = train.load(model, threshold_json=None)  # pinned as absent
    assert (blob["threshold"], blob["threshold_source"]) == (trained, "training")
    assert blob["threshold_sha256"] is None
    # A threshold.json made for another model is refused, on disk or pinned.
    other = json.dumps({"threshold": 0.37, "model": {"sha256": "0" * 64}})
    path.write_text(other)
    with pytest.raises(ValueError, match="another model"):
        train.load(model)
    with pytest.raises(ValueError, match="another model"):
        train.load(io.BytesIO(data), threshold_json=other.encode())
