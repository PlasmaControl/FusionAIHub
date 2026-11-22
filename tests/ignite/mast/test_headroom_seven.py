"""The stage-0 predictors on the seven tasks stage 0 did not run. CPU only; no scoring, no training."""

import dataclasses

import numpy as np
import pytest
import torch

pytest.importorskip("tokamark")
from tokamak_foundation_model.ignite.mast import tasks                      # noqa: E402
from tokamak_foundation_model.ignite.mast.harness import WindowBatch        # noqa: E402
from tokamak_foundation_model.ignite.mast.headroom import cheap, direct, floor   # noqa: E402

SEVEN = ("task_1-3", "task_2-2", "task_2-3", "task_3-2", "task_3-3", "task_4-2", "task_4-3")
pytestmark = pytest.mark.real_data


def synthetic_batch(task, n=2, seed=0):
    rng = np.random.default_rng(seed)

    def draw(specs, length):
        return {s.name: (s.mean + s.std * rng.standard_normal((n, *s.values_shape, length(s)))).astype(np.float32)
                for s in specs}

    inputs = draw(task.inputs, task.n_in)
    return WindowBatch(shot_ids=np.full(n, 30000, np.int64), window_indices=np.arange(n, dtype=np.int64),
                       inputs=inputs, actuators=draw(task.actuators, task.n_act),
                       targets=draw(task.outputs, task.n_out),
                       last_valid={k: a[..., -1].copy() for k, a in inputs.items()})


def test_the_seven_are_the_tasks_stage0_left():
    assert set(SEVEN) == set(tasks.ALL_TASKS) - set(tasks.STAGE0_TASKS)


@pytest.mark.parametrize("name", SEVEN)
def test_the_floor_builds_and_keeps_the_shape_for_each_of_the_seven_tasks(name):
    task = tasks.load_task(name)
    predict = floor.floor_predictor(task, floor.tokamind_output_indices(name), floor.tokamind_nan_imputation(name))
    batch = synthetic_batch(task)
    out = predict(batch)
    for s in task.outputs:
        assert out[s.name].shape == batch.targets[s.name].shape and np.isfinite(out[s.name]).all()


@pytest.mark.parametrize("name", SEVEN)
def test_the_direct_model_builds_and_runs_for_each_of_the_seven_tasks(name):
    task = tasks.load_task(name)
    cfg = dataclasses.replace(direct.default_config(task), d_model=32, depth=1, heads=2)
    model = direct.DirectModel(task, cfg)
    inputs, actuators, targets = direct.to_tensors(synthetic_batch(task), task, "cpu")
    pred = model(inputs, actuators)
    for s in task.outputs:
        assert pred[s.name].shape == targets[s.name].shape
    loss = direct.masked_mse(pred, targets)
    loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("name", SEVEN)
def test_the_cheap_predictors_fit_and_predict_for_each_of_the_seven_tasks(name):
    """Fitted the way `cheap.obtain` does: `fit` on the training batches, and for ridge `select` on held-out ones."""
    task = tasks.load_task(name)
    train = [synthetic_batch(task, n=8, seed=i) for i in range(3)]
    held_out = synthetic_batch(task, n=8, seed=3)
    shrunk = cheap.ShrunkPersistence(task).fit(train)
    ridge = cheap.RidgeFrames(task).fit(train)
    ridge.select([held_out])
    for predictor in (shrunk, ridge):
        out = predictor.predict(held_out)
        assert set(out) == {s.name for s in task.outputs}
        for s in task.outputs:
            assert out[s.name].shape == held_out.targets[s.name].shape, (predictor.NAME, s.name)
            assert np.isfinite(out[s.name]).all(), (predictor.NAME, s.name)
