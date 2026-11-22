"""Tests of the mean and persistence baselines (`baselines.py` needs no `tokamark`)."""

import json
import re

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
    # `_git_revision` answers "unknown" where git is absent, so the form is checked, not the length.
    assert d["git_revision"] == baselines._GIT_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", d["git_revision"])
    assert d["date"].endswith("+00:00")


# ---------------------------------------------------------------- the gate's verdict and kept scores

@pytest.mark.parametrize("ours", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("published", [None, 1.027])
def test_a_nonfinite_score_is_outside_tolerance(ours, published):
    assert baselines.outside_tolerance(ours, published, 0.002)


def test_outside_tolerance_compares_with_the_published_value():
    assert not baselines.outside_tolerance(1.0271, 1.027, 0.002)
    assert baselines.outside_tolerance(1.0300, 1.027, 0.002)
    assert not baselines.outside_tolerance(0.5, None, 0.002)


def _score_result(out_dir):
    import pandas as pd
    from tokamak_foundation_model.ignite.mast.harness import ScoreResult

    return ScoreResult(task="task_x", task_nrmse=0.5, per_signal=pd.DataFrame(), n_windows=12,
                       per_shot=pd.DataFrame({"shot_id": [1, 2], "NRMSE": [0.4, 0.6]}), out_dir=out_dir)


def test_scoring_code_sha256_is_a_stable_hex_digest():
    h = baselines.scoring_code_sha256()
    assert len(h) == 64 and int(h, 16) >= 0 and h == baselines.scoring_code_sha256()


def test_write_score_json_records_the_scoring_code(tmp_path):
    d = json.loads(baselines.write_score_json(_score_result(tmp_path), "mean", "test").read_text())
    assert d["code_sha256"] == baselines.scoring_code_sha256() and len(d) == 9


def test_a_score_written_now_is_kept(tmp_path):
    baselines.write_score_json(_score_result(tmp_path), "mean", "test")
    stored, reason = baselines.read_kept_score(tmp_path, "task_x", "mean", "test")
    assert reason == "" and stored["nrmse"] == 0.5 and stored["n_windows"] == 12


def _rewrite(path, **changes):
    d = json.loads(path.read_text())
    for k, v in changes.items():
        if v is None:
            d.pop(k)
        else:
            d[k] = v
    path.write_text(json.dumps(d))


@pytest.mark.parametrize("case", ["missing", "truncated", "other_task", "other_predictor", "other_split", "no_hash",
                                  "other_hash", "right_hash_missing_key"])
def test_a_stored_score_that_is_not_the_one_asked_for_is_not_kept(tmp_path, case):
    path = baselines.write_score_json(_score_result(tmp_path), "mean", "test")
    if case == "missing":
        path.unlink()
    elif case == "truncated":
        path.write_text(path.read_text()[:40])
    elif case == "other_task":
        _rewrite(path, task="task_y")
    elif case == "other_predictor":
        _rewrite(path, predictor="persistence")
    elif case == "other_split":
        _rewrite(path, split="val_report")
    elif case == "no_hash":
        _rewrite(path, code_sha256=None)
    elif case == "right_hash_missing_key":
        # The hash is the current one but the record lacks a key a reader needs (here the score itself).
        assert json.loads(path.read_text())["code_sha256"] == baselines.scoring_code_sha256()
        _rewrite(path, nrmse=None)
    else:
        _rewrite(path, code_sha256="0" * 64)
    stored, reason = baselines.read_kept_score(tmp_path, "task_x", "mean", "test")
    assert stored is None and reason
    if case == "other_task":
        assert "task_y/mean/test" in reason
    if case in ("no_hash", "other_hash"):
        assert "scoring code changed since" in reason


# ---------------------------------------------------------------- stamped with the code as loaded

def test_the_hash_is_the_digest_of_the_files_of_an_unedited_tree():
    assert baselines.scoring_code_sha256() == baselines._hash_scoring_files()


def test_the_hash_is_taken_at_import_not_when_a_score_is_written(tmp_path, monkeypatch):
    """A score computed by the code as loaded must not be stamped with files edited since."""
    loaded = baselines.scoring_code_sha256()
    before = json.loads(baselines.write_score_json(_score_result(tmp_path), "mean", "test").read_text())
    monkeypatch.setattr(baselines, "_hash_scoring_files", lambda: "0" * 64)
    assert baselines.scoring_code_sha256() == loaded
    after = json.loads(baselines.write_score_json(_score_result(tmp_path), "mean", "test").read_text())
    assert before["code_sha256"] == after["code_sha256"] == loaded
    kept, reason = baselines.read_kept_score(tmp_path, "task_x", "mean", "test")
    assert reason == "" and kept["code_sha256"] == loaded


def test_the_git_revision_is_taken_at_import_not_when_a_score_is_written(tmp_path, monkeypatch):
    loaded = json.loads(baselines.write_score_json(_score_result(tmp_path), "mean", "test").read_text())["git_revision"]
    monkeypatch.setattr(baselines, "_git_revision", lambda: "f" * 40)
    again = json.loads(baselines.write_score_json(_score_result(tmp_path), "mean", "test").read_text())
    assert again["git_revision"] == loaded != "f" * 40
