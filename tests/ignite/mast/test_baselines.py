"""Tests of the mean and persistence baselines (`baselines.py` needs no `tokamark`)."""

import json

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import baselines, tasks


def test_mean_predictor_fills_with_the_signal_mean(toy_task, toy_batch):
    p = baselines.mean_predictor(toy_task)(toy_batch(n=3, seed=0))
    s = toy_task.outputs[0]
    assert p[s.name].shape == (3, *s.values_shape, toy_task.n_out(s))
    assert p[s.name].dtype == np.float32 and np.all(p[s.name] == np.float32(s.mean))


def test_persistence_repeats_last_valid_and_falls_back_to_mean(toy_task, toy_batch):
    b = toy_batch(n=2, seed=0)
    s = toy_task.outputs[0]
    b.last_valid[s.name][0] = 7.0
    b.last_valid[s.name][1] = np.nan
    p = baselines.persistence_predictor(toy_task)(b)[s.name]
    assert p.dtype == np.float32 and p.shape == b.targets[s.name].shape
    assert np.all(p[0] == 7.0) and np.all(p[1] == np.float32(s.mean))


def test_persistence_needs_outputs_among_inputs():
    pytest.importorskip("tokamark")
    with pytest.raises(ValueError):
        baselines.persistence_predictor(tasks.load_task("task_1-1"))


def test_persistence_needs_outputs_among_inputs_hand_built(toy_task):
    coil = toy_task.actuators[0]
    no_overlap = tasks.TaskSpec(**{**toy_task.__dict__, "outputs": (coil,),
                                   "steps": {**toy_task.steps, "output": {coil.name: 100}}})
    with pytest.raises(ValueError, match="pf_active-coil_voltage"):
        baselines.persistence_predictor(no_overlap)


def test_persistence_fills_only_the_channel_without_a_finite_input(toy_task, toy_batch):
    """A NaN in last_valid for one channel of one window is the realistic case."""
    flux = toy_task.inputs[1]
    task = tasks.TaskSpec(**{**toy_task.__dict__, "outputs": (flux,),
                             "steps": {**toy_task.steps, "output": {flux.name: 100}}})
    b = toy_batch(n=2, seed=0)
    b.targets = {flux.name: np.zeros((2, *flux.values_shape, 100), np.float32)}
    last = b.last_valid[flux.name].copy()
    b.last_valid[flux.name][1, 3] = np.nan
    p = baselines.persistence_predictor(task)(b)[flux.name]
    assert p.shape == (2, *flux.values_shape, 100) and p.dtype == np.float32
    assert np.all(p[1, 3] == np.float32(flux.mean))
    keep = np.ones(p.shape[:2], bool)
    keep[1, 3] = False
    for i, c in zip(*np.nonzero(keep)):
        assert np.all(p[i, c] == last[i, c])
    assert np.isfinite(p).all()


def test_the_predictors_do_not_modify_the_batch(toy_task, toy_batch):
    b = toy_batch(n=2, seed=0)
    b.last_valid["summary-ip"][0] = np.nan
    before = b.last_valid["summary-ip"].copy()
    baselines.persistence_predictor(toy_task)(b)
    assert np.array_equal(b.last_valid["summary-ip"], before, equal_nan=True)


def test_write_score_json(tmp_path):
    import pandas as pd
    from tokamak_foundation_model.ignite.mast.harness import ScoreResult

    r = ScoreResult(task="task_x", task_nrmse=0.5, per_signal=pd.DataFrame(), n_windows=12,
                    per_shot=pd.DataFrame({"shot_id": [1, 2, 3], "NRMSE": [0.4, 0.5, 0.6]}),
                    out_dir=tmp_path)
    path = baselines.write_score_json(r, "mean", "test")
    assert path == tmp_path / "score.json"
    d = json.loads(path.read_text())
    assert (d["task"], d["predictor"], d["split"]) == ("task_x", "mean", "test")
    assert d["nrmse"] == 0.5 and d["n_shots"] == 3 and d["n_windows"] == 12
    assert len(d["git_revision"]) == 40 and d["date"].endswith("+00:00")
