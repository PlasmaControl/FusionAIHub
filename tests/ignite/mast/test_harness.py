"""Tests of the scoring harness.

The toy-batch tests need `tokamark` for the evaluator but no data; the `real_data` tests
need TokaMark's Zarr store too (skipped when either is missing, see conftest.py).
"""

import dataclasses

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import harness, tasks


# Python 3.12 warns when a multi-threaded process forks; torch's DataLoader forks its workers
# (third-party, harmless here: the workers only read Zarr files). Only this message is silenced,
# only on the tests that start workers.
FORK_WARNING = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks in the child"
    ":DeprecationWarning"
)


def _const_pred(s, value):
    return lambda bb: {s.name: np.full_like(bb.targets[s.name], value)}


# ---------------------------------------------------------------- the scoring path (evaluator)

def test_known_answer_nrmse_is_one(toy_task, toy_batch, tmp_path):
    pytest.importorskip("tokamark")
    b = toy_batch(n=8, seed=0)
    s = toy_task.outputs[0]
    b.targets[s.name][:] = s.mean + s.std
    r = harness.score_batches("task_4-4", [b], _const_pred(s, s.mean), tmp_path)
    assert abs(r.task_nrmse - 1.0) < 1e-6 and r.n_windows == 8
    assert r.task == "task_4-4"
    assert list(r.per_signal["feature_name"]) == [s.name]
    assert abs(r.per_signal["NRMSE"].iloc[0] - 1.0) < 1e-6
    assert set(r.per_shot["shot_id"]) == {1000, 1001}
    assert (tmp_path / "task_4-4" / "task_metrics.csv").exists()
    assert (tmp_path / "task_4-4" / "shots_metrics.csv").exists()


def test_nan_targets_are_ignored(toy_task, toy_batch, tmp_path):
    pytest.importorskip("tokamark")
    b = toy_batch(n=8, seed=0)
    s = toy_task.outputs[0]
    b.targets[s.name][:] = s.mean
    b.targets[s.name][:, ..., ::2] = np.nan
    r = harness.score_batches("task_4-4", [b], _const_pred(s, s.mean), tmp_path)
    assert r.task_nrmse < 1e-9


def test_nonfinite_prediction_is_fine_where_the_target_is_nan(toy_task, toy_batch, tmp_path):
    pytest.importorskip("tokamark")
    b = toy_batch(n=4, seed=0)
    s = toy_task.outputs[0]
    b.targets[s.name][:] = s.mean
    b.targets[s.name][:, ..., ::2] = np.nan

    def predict(bb):
        p = np.full_like(bb.targets[s.name], s.mean)
        p[:, ..., ::2] = np.nan
        return {s.name: p}

    assert harness.score_batches("task_4-4", [b], predict, tmp_path).task_nrmse < 1e-9


def test_score_batches_raises_on_a_nonfinite_task_score(toy_task, toy_batch, tmp_path, monkeypatch):
    """A NaN task score must be an error: `abs(nan - x) > tol` is False, so it would read as agreement.

    The real evaluator gives no NaN from a toy batch that passes the prediction checks, so the
    evaluator's `compute_metrics` is replaced by one returning a NaN task row (the harness
    calls it as `tokamark.evaluator.compute_metrics`).
    """
    pytest.importorskip("tokamark")
    import pandas as pd
    import tokamark.evaluator

    s = toy_task.outputs[0]
    monkeypatch.setattr(
        tokamark.evaluator, "compute_metrics",
        lambda **kw: pd.DataFrame({"NRMSE_mean": [np.nan, np.nan]}, index=["task_4-4", s.name]),
    )
    with pytest.raises(RuntimeError, match=r"task_4-4.*not finite.*summary-ip"):
        harness.score_batches("task_4-4", [toy_batch(n=4, seed=0)], _const_pred(s, s.mean), tmp_path)


def test_score_rejects_nonfinite_prediction(toy_task, toy_batch, tmp_path):
    b = toy_batch(n=4, seed=0)
    s = toy_task.outputs[0]
    with pytest.raises(ValueError, match="non-finite"):
        harness.score_batches("task_4-4", [b], _const_pred(s, np.nan), tmp_path)


def test_score_rejects_missing_output_and_wrong_shape(toy_task, toy_batch, tmp_path):
    b = toy_batch(n=4, seed=0)
    s = toy_task.outputs[0]
    with pytest.raises(ValueError, match="missing"):
        harness.score_batches("task_4-4", [b], lambda bb: {}, tmp_path)
    with pytest.raises(ValueError, match="shape"):
        harness.score_batches(
            "task_4-4", [b], lambda bb: {s.name: np.zeros_like(bb.targets[s.name])[..., 1:]}, tmp_path
        )


def test_score_rejects_a_window_delivered_twice(toy_task, toy_batch, tmp_path):
    pytest.importorskip("tokamark")
    b = toy_batch(n=4, seed=0)
    s = toy_task.outputs[0]
    with pytest.raises(ValueError, match="twice"):
        harness.score_batches("task_4-4", [b, b], _const_pred(s, s.mean), tmp_path)


def test_score_rejects_no_windows(tmp_path):
    with pytest.raises(ValueError, match="no windows"):
        harness.score_batches("task_4-4", [], lambda bb: {}, tmp_path)


# ---------------------------------------------------------------- the window transform (no data)

def _window(toy_task, history, rng, *, with_nan_tail=False):
    """One TokaMark window as the dataset delivers it, for the non-markovian variant of the toy task."""
    ip, flux = toy_task.inputs
    coil = toy_task.actuators[0]
    ip_hist = rng.standard_normal((history,))  # a scalar signal arrives squeezed to (T,)
    if with_nan_tail:
        ip_hist[-5:] = np.nan
    return {
        "shot_id": 7, "window_index": 3, "t_cut": 0.1,
        "input": {
            ip.name: {"time": np.arange(history), "values": ip_hist},
            flux.name: {"time": np.arange(history), "values": rng.standard_normal((15, history))},
        },
        "actuator": {coil.name: {"time": np.arange(history + 100),
                                 "values": rng.standard_normal((4, history + 100))}},
        "output": {ip.name: {"time": np.arange(100), "values": rng.standard_normal((100,))}},
    }


@pytest.fixture
def toy_nonmarkov(toy_task):
    """Non-markovian toy task: 5 ms of history by default (20 ip samples, 25 flux), 100 output samples."""
    return dataclasses.replace(toy_task, markovian=False)


def test_transform_keeps_the_last_history_samples_and_last_valid_sees_all(toy_nonmarkov):
    t = toy_nonmarkov
    rng = np.random.default_rng(0)
    w = _window(t, history=200, rng=rng, with_nan_tail=True)
    out = harness._WindowTransform(t, history_s=None)(w)
    ip = t.inputs[0]
    full = np.asarray(w["input"][ip.name]["values"], dtype=np.float32)
    assert out["inputs"][ip.name].shape == (1, 20)
    # the last 5 samples are NaN, so the crop is NaN-tailed but last_valid reaches before the crop
    np.testing.assert_array_equal(out["inputs"][ip.name][0], full[-20:])
    assert out["last_valid"][ip.name].shape == (1,)
    assert out["last_valid"][ip.name][0] == full[-6]
    # the scalar signal came squeezed; the others keep their channels
    assert out["inputs"][t.inputs[1].name].shape == (15, 25)
    assert out["targets"][ip.name].shape == (1, 100)
    assert out["inputs"][ip.name].dtype == np.float32


def test_transform_crops_actuators_to_history_plus_future(toy_nonmarkov):
    t = toy_nonmarkov
    w = _window(t, history=200, rng=np.random.default_rng(1))
    coil = t.actuators[0]
    # 0.00025 s per sample: 5 ms of history is 20 samples, plus the 100 future samples
    assert harness._WindowTransform(t, None)(w)["actuators"][coil.name].shape == (4, 120)
    # 0.0025 s of history is 10 samples
    assert harness._WindowTransform(t, 0.0025)(w)["actuators"][coil.name].shape == (4, 110)


def test_transform_left_pads_short_history_with_nan_and_last_valid_ignores_the_pad(toy_nonmarkov):
    t = toy_nonmarkov
    w = _window(t, history=8, rng=np.random.default_rng(2))
    ip = t.inputs[0]
    out = harness._WindowTransform(t, None)(w)
    a = out["inputs"][ip.name][0]
    assert a.shape == (20,) and np.isnan(a[:12]).all() and np.isfinite(a[12:]).all()
    assert out["last_valid"][ip.name][0] == a[-1]


def test_transform_last_valid_is_nan_when_nothing_is_finite(toy_nonmarkov):
    t = toy_nonmarkov
    w = _window(t, history=30, rng=np.random.default_rng(3))
    w["input"][t.inputs[0].name]["values"][:] = np.nan
    out = harness._WindowTransform(t, None)(w)
    assert np.isnan(out["last_valid"][t.inputs[0].name]).all()


def test_last_finite_matches_a_plain_scan_whatever_the_tail():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 3, 40))
    a[rng.random(a.shape) < 0.7] = np.nan
    a[0, 0, :] = np.nan            # nothing finite at all
    a[1, 1, 10:] = np.nan          # finite only before any short tail
    a[2, 2, 39] = np.inf           # inf is not finite
    for tail in (1, 5, 40, 100):
        got = harness._last_finite(a, tail=tail)
        want = np.full(a.shape[:-1], np.nan, dtype=np.float32)
        for idx in np.ndindex(*a.shape[:-1]):
            f = np.flatnonzero(np.isfinite(a[idx]))
            if len(f):
                want[idx] = a[idx][f[-1]]
        np.testing.assert_array_equal(got, want)
        assert got.dtype == np.float32


def test_transform_markovian_requires_the_exact_length(toy_task):
    w = _window(toy_task, history=19, rng=np.random.default_rng(4))
    with pytest.raises(ValueError, match="summary-ip"):
        harness._WindowTransform(toy_task, None)(w)


def test_collate_stacks_one_window_batches(toy_nonmarkov):
    t = toy_nonmarkov
    items = [
        {"shot_id": 5 + i, "window_index": i,
         **harness._WindowTransform(t, None)(_window(t, 50, np.random.default_rng(i)))}
        for i in range(3)
    ]
    b = harness._to_window_batch(harness._collate(items))
    assert b.shot_ids.dtype == np.int64 and list(b.shot_ids) == [5, 6, 7]
    assert list(b.window_indices) == [0, 1, 2]
    assert b.inputs[t.inputs[0].name].shape == (3, 1, 20)
    assert b.last_valid[t.inputs[1].name].shape == (3, 15)
    assert b.actuators[t.actuators[0].name].shape == (3, 4, 120)
    assert b.targets[t.outputs[0].name].shape == (3, 1, 100)


def test_iter_windows_rejects_stride_in_test_mode(toy_task):
    with pytest.raises(ValueError, match="stride_s"):
        harness.iter_windows(toy_task, [1], test_mode=True, stride_s=0.005)


# ---------------------------------------------------------------- shot lists (need tokamark, no data)

def test_train_windows_refuse_non_training_shots():
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_1-1")
    with pytest.raises(ValueError, match="not in the training split"):
        next(iter(harness.train_windows(t, shots=[tasks.split_shots("test")[0]])))


def test_score_refuses_shots_outside_the_split(tmp_path):
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_1-1")
    with pytest.raises(ValueError, match="not in the .* split"):
        harness.score(t, lambda b: {}, tmp_path, split="val", shots=[tasks.split_shots("test")[0]])
    with pytest.raises(ValueError, match="split"):
        harness.score(t, lambda b: {}, tmp_path, split="train")


def test_open_mast_dataset_refuses_shots_outside_the_splits():
    """The shots in none of the three splits are never loaded; the check precedes any data access."""
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_1-1")
    assert 1 not in tasks.split_shots("train") + tasks.split_shots("val") + tasks.split_shots("test")
    with pytest.raises(ValueError, match="not in any split"):
        harness.open_mast_dataset({"task": t.name}, [tasks.split_shots("train")[0], 1])


# ---------------------------------------------------------------- real data

@pytest.mark.real_data
def test_real_windows_shapes_and_history_crop():
    t = tasks.load_task("task_4-4")
    shot = tasks.split_shots("val")[0]
    b = next(iter(harness.iter_windows(t, [shot], test_mode=True, batch_size=4, num_workers=0)))
    for s in t.inputs:
        assert b.inputs[s.name].shape == (4, *s.values_shape, t.n_in(s, 0.15))
        assert b.last_valid[s.name].shape == (4, *s.values_shape)
    for s in t.actuators:
        assert b.actuators[s.name].shape == (4, *s.values_shape, t.n_act(s))
    for s in t.outputs:
        assert b.targets[s.name].shape == (4, *s.values_shape, t.n_out(s))


@pytest.mark.real_data
@pytest.mark.parametrize("name", ["task_3-1", "task_2-1", "task_4-5"])
def test_real_signal_that_is_input_and_output_keeps_the_input_length(name):
    """TokaMark aliases one dict for such a signal; neither the windows nor the TaskSpec may take the output's length."""
    t = tasks.load_task(name)
    shots = tasks.split_shots("val")[:40]  # not every shot has every diagnostic: take the first window found
    b = next(iter(harness.iter_windows(t, shots, test_mode=False, batch_size=2, num_workers=0)))
    both = {s.name for s in t.inputs} & {s.name for s in t.outputs}
    assert both
    for s in t.inputs:
        assert b.inputs[s.name].shape == (2, *s.values_shape, round(t.input_length / s.dt))
    for s in t.outputs:
        assert b.targets[s.name].shape == (2, *s.values_shape, round(t.output_length / s.dt))


@pytest.mark.real_data
def test_real_shorter_history_is_the_tail_of_the_full_one():
    t = tasks.load_task("task_4-4")
    shot = tasks.split_shots("val")[0]
    kw = dict(test_mode=True, batch_size=8, num_workers=0)
    full = next(iter(harness.iter_windows(t, [shot], **kw)))
    short = next(iter(harness.iter_windows(t, [shot], history_s=0.05, **kw)))
    assert list(full.window_indices) == list(short.window_indices)
    for s in t.inputs:
        n = t.n_in(s, 0.05)
        assert short.inputs[s.name].shape[-1] == n
        np.testing.assert_array_equal(short.inputs[s.name], full.inputs[s.name][..., -n:])
        np.testing.assert_array_equal(short.last_valid[s.name], full.last_valid[s.name])


@pytest.mark.real_data
def test_real_task_without_actuators_has_an_empty_actuator_dict():
    t = tasks.load_task("task_1-1")
    shot = tasks.split_shots("val")[0]
    b = next(iter(harness.iter_windows(t, [shot], test_mode=True, batch_size=4, num_workers=0)))
    assert b.actuators == {}
    for s in t.outputs:
        assert b.targets[s.name].shape == (4, *s.values_shape, t.n_out(s))


@pytest.mark.real_data
def test_training_stride_thins_windows():
    t = tasks.load_task("task_1-1")
    shot = tasks.split_shots("train")[0]
    n1 = sum(len(b.shot_ids) for b in harness.train_windows(t, [shot], num_workers=0))
    n5 = sum(len(b.shot_ids) for b in harness.train_windows(t, [shot], num_workers=0, stride_s=0.005))
    assert 4.0 < n1 / n5 < 6.0


@FORK_WARNING
@pytest.mark.real_data
def test_no_duplicate_windows_with_workers():
    t = tasks.load_task("task_1-1")
    shots = tasks.split_shots("val")[:3]

    def pairs(num_workers):
        out = []
        for b in harness.iter_windows(t, shots, test_mode=True, batch_size=64, num_workers=num_workers):
            out += list(zip(b.shot_ids.tolist(), b.window_indices.tolist()))
        return out

    serial, parallel = pairs(0), pairs(4)
    assert len(serial) == len(set(serial)) > 0
    assert len(parallel) == len(set(parallel))
    assert set(parallel) == set(serial)


@FORK_WARNING
@pytest.mark.real_data
def test_persistence_scores_through_the_real_evaluator(tmp_path):
    """End to end on two validation shots: a persistence predictor gives a finite task score."""
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_4-4")
    shots = tasks.split_shots("val_select")[:2]

    def persistence(b):
        out = {}
        for s in t.outputs:
            last = b.last_valid[s.name]  # summary-ip is also an input of task_4-4
            out[s.name] = np.broadcast_to(np.nan_to_num(last, nan=s.mean)[..., None],
                                          b.targets[s.name].shape).astype(np.float32)
        return out

    r = harness.score(t, persistence, tmp_path, split="val_select", shots=shots, num_workers=2)
    assert np.isfinite(r.task_nrmse) and r.n_windows > 0
    assert set(r.per_shot["shot_id"]) <= set(shots)
