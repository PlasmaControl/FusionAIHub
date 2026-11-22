"""The cheap headroom predictors: shrunk persistence and ridge regression on frame means."""

import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import harness
from tokamak_foundation_model.ignite.mast.headroom import cheap, direct

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "run_headroom.py"
N_FEATURES = 2 * (1 + 15) + 4 * 6 + 1  # the toy task: ip and 15 flux loops, 4 coils in 6 frames, a constant
N_FRAMES = 5                            # 25 ms of output in frames of 5 ms


def _run_headroom():
    spec = importlib.util.spec_from_file_location("run_headroom_cheap_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ frames

def test_frame_means_and_hold_roundtrip():
    x = np.repeat(np.arange(4.0), 25)[None]            # 4 frames of 25 samples at dt = 0.2 ms
    m = cheap.frame_means(x, 0.0002); assert m.shape == (1, 4) and np.allclose(m, [[0, 1, 2, 3]])
    assert np.array_equal(cheap.hold_frames(m, 0.0002, 100), x)


def test_frame_means_partial_last_frame_and_nans():
    x = np.ones((1, 98)); x[0, :5] = np.nan            # 98 stamps at 1.024 ms: frames of 5 stamps -> 20 frames, the last has 3
    assert cheap.frame_len(0.001024) == 5
    m = cheap.frame_means(x, 0.001024); assert m.shape[-1] == 20 and np.isnan(m[0, 0]) and m[0, -1] == 1.0


def test_hold_frames_when_n_is_not_a_multiple_of_the_frame_length():
    m = np.array([[0.0, 1.0, 2.0, 3.0]])
    held = cheap.hold_frames(m, 0.0002, 90)             # frames of 25 samples: the last frame is cut to 15
    assert held.shape == (1, 90)
    np.testing.assert_array_equal(held[0], np.repeat([0.0, 1.0, 2.0, 3.0], [25, 25, 25, 15]))
    with pytest.raises(ValueError, match="frames"):
        cheap.hold_frames(m, 0.0002, 101)               # four frames cannot cover 101 samples


def test_frame_means_does_not_warn_on_an_all_nan_frame():
    x = np.full((2, 3, 10), np.nan)
    assert np.isnan(cheap.frame_means(x, 0.0005)).all()  # -W error: no "mean of empty slice"


# ------------------------------------------------------------------ features

def test_window_features_shape_and_finiteness(toy_task, toy_batch):
    b = toy_batch(n=5, seed=0); b.inputs["summary-ip"][0] = np.nan
    f = cheap.window_features(b, toy_task)
    assert f.shape == (5, 2 * (1 + 15) + 4 * 6 + 1) and np.isfinite(f).all() and np.all(f[:, -1] == 1.0)


def test_the_validity_flags_follow_the_inputs(toy_task, toy_batch):
    b = toy_batch(n=4, seed=0)
    b.inputs["summary-ip"][1] = np.nan
    b.inputs["magnetics-flux_loop_flux"][2, 3] = np.nan  # one loop of window 2
    f = cheap.window_features(b, toy_task)
    ip_block, flux_block = f[:, :2], f[:, 2:2 + 30]
    assert ip_block[:, 1].tolist() == [1, 0, 1, 1] and ip_block[1, 0] == 0.0
    assert flux_block[2, 15 + 3] == 0 and flux_block[2, 3] == 0 and flux_block[2, 15:].sum() == 14
    assert flux_block[[0, 1, 3], 15:].min() == 1


def test_a_spectrogram_input_gets_one_flag_and_per_bin_values():
    from tokamak_foundation_model.ignite.mast import tasks
    name = tasks.SPECTROGRAM_SIGNALS[0]
    spec = tasks.SignalSpec(name, 0.001024, (3, 4), 0.0, 2.0)
    task = tasks.TaskSpec(name="task_spec", markovian=False, input_length=0.02, output_length=0.01, delta=0.0,
                          stride=0.01, inputs=(spec,), actuators=(), outputs=(spec,),
                          steps={"input": {name: 20}, "actuator": {}, "output": {name: 10}})
    x = np.full((2, 3, 4, 20), 2.0, dtype=np.float32); x[1] = np.nan; x[0, :, :, :10] = np.nan
    b = harness.WindowBatch(np.arange(2), np.zeros(2, dtype=np.int64), {name: x}, {}, {name: x[..., :10] * 0},
                            {name: np.full((2, 3, 4), np.nan, dtype=np.float32)})
    f = cheap.window_features(b, task)
    assert f.shape == (2, 12 + 1 + 1)
    np.testing.assert_allclose(f[0, :12], 1.0)          # (2 - 0) / 2 over the kept history, all stamps that are finite
    assert f[0, 12] == 1 and f[1, 12] == 0 and np.all(f[1, :12] == 0)


# ------------------------------------------------------------------ shrunk persistence

def _slope_batch(toy_task, toy_batch, n, seed, slope=0.6, noise=0.02):
    spec = toy_task.outputs[0]
    rng = np.random.default_rng(seed)
    b = toy_batch(n=n, seed=seed)
    anchor = spec.standardize(b.last_valid["summary-ip"].astype(np.float64))        # (n, 1)
    z = slope * anchor[..., None] + noise * rng.standard_normal((n, 1, toy_task.n_out(spec)))
    b.targets["summary-ip"] = spec.unstandardize(z).astype(np.float32)
    return b


def test_shrunk_persistence_recovers_the_slope(toy_task, toy_batch):
    b = _slope_batch(toy_task, toy_batch, 4000, 0)
    sp = cheap.ShrunkPersistence(toy_task).fit([b])
    a = sp.slope["summary-ip"]
    assert a.shape == (N_FRAMES,) and np.all(np.abs(a - 0.6) < 0.02)
    # a window whose last valid sample is missing is predicted by the training mean
    spec = toy_task.outputs[0]
    t = toy_batch(n=3, seed=9); t.last_valid["summary-ip"][:] = np.nan
    pred = sp.predict(t)["summary-ip"]
    assert pred.shape == t.targets["summary-ip"].shape and np.isfinite(pred).all()
    train_mean = np.nanmean(b.targets["summary-ip"].astype(np.float64))
    np.testing.assert_allclose(pred, train_mean, rtol=0, atol=spec.std * 1e-3)
    # and a window with a finite anchor is shrunk toward it by the slope
    good = toy_batch(n=3, seed=10)
    out = sp.predict(good)["summary-ip"][:, 0, 0]
    z = spec.standardize(good.last_valid["summary-ip"][:, 0].astype(np.float64))
    np.testing.assert_allclose(spec.standardize(out), a[0] * z + (1 - a[0]) * sp.mean["summary-ip"][0], atol=1e-4)


def test_shrunk_contains_the_mean_and_persistence_as_special_cases(toy_task, toy_batch):
    """With targets that ignore the anchor the slope is 0; with targets equal to it, 1."""
    flat = cheap.ShrunkPersistence(toy_task).fit([_slope_batch(toy_task, toy_batch, 3000, 1, slope=0.0)])
    same = cheap.ShrunkPersistence(toy_task).fit([_slope_batch(toy_task, toy_batch, 3000, 2, slope=1.0, noise=0.0)])
    assert np.all(np.abs(flat.slope["summary-ip"]) < 0.05)
    np.testing.assert_allclose(same.slope["summary-ip"], 1.0, atol=1e-6)


# ------------------------------------------------------------------ ridge

def _linear_batch(toy_task, toy_batch, n, seed, w, rng):
    spec = toy_task.outputs[0]
    b = toy_batch(n=n, seed=seed)
    z = cheap.window_features(b, toy_task) @ w + 0.01 * rng.standard_normal((n, N_FRAMES))
    held = cheap.hold_frames(z[:, None, :], spec.dt, toy_task.n_out(spec))
    b.targets["summary-ip"] = spec.unstandardize(held).astype(np.float32)
    return b, z


def test_ridge_recovers_a_linear_map(toy_task, toy_batch):
    rng = np.random.default_rng(1)
    w = 0.3 * rng.standard_normal((N_FEATURES, N_FRAMES))
    train = [_linear_batch(toy_task, toy_batch, 1000, s, w, rng)[0] for s in (1, 2, 3)]
    val = [_linear_batch(toy_task, toy_batch, 500, 4, w, rng)[0]]
    rf = cheap.RidgeFrames(toy_task).fit(train)
    lam = rf.select(val)
    assert lam in cheap.RidgeFrames.lambdas
    test, z = _linear_batch(toy_task, toy_batch, 400, 5, w, rng)
    spec = toy_task.outputs[0]
    pred = spec.standardize(rf.predict(test)["summary-ip"].astype(np.float64))
    held = spec.standardize(test.targets["summary-ip"].astype(np.float64))
    assert np.sqrt(np.mean((pred - held) ** 2)) < 0.05


def test_ridge_leaves_the_constant_unpenalized(toy_task, toy_batch):
    """A target that is a constant is found exactly even with the largest penalty."""
    spec = toy_task.outputs[0]
    b = toy_batch(n=600, seed=0)
    b.targets["summary-ip"] = np.full_like(b.targets["summary-ip"], spec.unstandardize(2.5))
    rf = cheap.RidgeFrames(toy_task).fit([b])
    rf.lam = max(cheap.RidgeFrames.lambdas)
    rf.W = rf._solve(rf.lam)
    out = rf.predict(toy_batch(n=3, seed=1))["summary-ip"]
    np.testing.assert_allclose(spec.standardize(out.astype(np.float64)), 2.5, atol=1e-3)


# ------------------------------------------------------------------ missing data

def _fitted(toy_task, toy_batch):
    train, val = [toy_batch(n=200, seed=0)], [toy_batch(n=100, seed=1)]
    rf = cheap.RidgeFrames(toy_task).fit(train)
    rf.select(val)
    return cheap.ShrunkPersistence(toy_task).fit(train), rf


def test_a_batch_with_an_entirely_missing_input_signal_is_predicted_finitely(toy_task, toy_batch):
    b = toy_batch(n=6, seed=2)
    for name in b.inputs:                                   # every input and actuator missing
        b.inputs[name][:] = np.nan; b.last_valid[name][:] = np.nan
    for name in b.actuators:
        b.actuators[name][:] = np.nan
    for predictor in _fitted(toy_task, toy_batch):
        out = predictor.predict(b)
        assert set(out) == {"summary-ip"} and out["summary-ip"].shape == b.targets["summary-ip"].shape
        assert np.isfinite(out["summary-ip"]).all()
    only_flux = toy_batch(n=6, seed=3)
    only_flux.inputs["magnetics-flux_loop_flux"][:] = np.nan
    for predictor in _fitted(toy_task, toy_batch):
        assert np.isfinite(predictor.predict(only_flux)["summary-ip"]).all()


def test_entirely_missing_targets_neither_raise_nor_break_the_fit(toy_task, toy_batch):
    nan_targets = toy_batch(n=50, seed=4)
    nan_targets.targets["summary-ip"][:] = np.nan
    for cls in (cheap.ShrunkPersistence, cheap.RidgeFrames):
        p = cls(toy_task).fit([nan_targets, toy_batch(n=150, seed=5)])
        if cls is cheap.RidgeFrames:
            p.select([toy_batch(n=50, seed=6)])
        assert np.isfinite(p.predict(toy_batch(n=4, seed=7))["summary-ip"]).all()
        assert p.n_channels_without_target == 0


def test_a_channel_that_never_has_a_target_gets_tokamarks_mean(toy_task, toy_batch):
    nan_targets = toy_batch(n=50, seed=4)
    nan_targets.targets["summary-ip"][:] = np.nan
    spec = toy_task.outputs[0]
    for cls in (cheap.ShrunkPersistence, cheap.RidgeFrames):
        p = cls(toy_task).fit([nan_targets])
        if cls is cheap.RidgeFrames:
            p.select([toy_batch(n=20, seed=6)])
        assert p.n_channels_without_target == 1
        out = p.predict(toy_batch(n=3, seed=8))["summary-ip"]
        np.testing.assert_allclose(out, spec.mean, rtol=1e-5)


def test_predict_does_not_mutate_the_batch(toy_task, toy_batch):
    b = toy_batch(n=4, seed=1); b.inputs["summary-ip"][0] = np.nan
    before = {n: a.copy() for n, a in b.inputs.items()}
    for predictor in _fitted(toy_task, toy_batch):
        predictor.predict(b)
    for n, a in before.items():
        np.testing.assert_array_equal(b.inputs[n], a)


# ------------------------------------------------------------------ the fit file

def test_the_fit_file_round_trips(toy_task, toy_batch, tmp_path):
    shrunk, ridge = _fitted(toy_task, toy_batch)
    test = toy_batch(n=7, seed=11)
    for name, predictor in (("shrunk", shrunk), ("ridge", ridge)):
        path = tmp_path / "cheap" / "task_toy" / f"{name}.npz"
        cheap.save_fit(path, predictor, {"extra": 3})
        assert not list(path.parent.glob("*.tmp*")), "the temporary file is renamed away"
        arrays, meta = cheap.load_fit(path)
        loaded = type(predictor).from_arrays(toy_task, arrays)
        np.testing.assert_array_equal(loaded.predict(test)["summary-ip"], predictor.predict(test)["summary-ip"])
        assert meta["extra"] == 3 and meta["code_sha256"] == cheap.PREDICTOR_SHA256
        assert meta["n_windows"] == 200 and meta["n_channels_without_target"] == 0
    assert meta["lambda"] == ridge.lam and cheap.load_fit(tmp_path / "cheap/task_toy/shrunk.npz")[1]["lambda"] is None


# ------------------------------------------------------------------ fit or reuse

@pytest.fixture
def stub_data(monkeypatch, toy_task, toy_batch):
    """Training and validation windows from the toy generator; counts how often each is read.

    Each split has its own shots, so a test can tell which split a selection was drawn from."""
    calls = {"train": 0, "val": 0, "shots": [], "val_shots": [], "splits": []}
    splits = {"train": list(range(1, 6001)), "val_select": list(range(7000, 7400))}

    def split_shots(split):
        calls["splits"].append(split)
        return list(splits.get(split, range(9000, 9400)))

    monkeypatch.setattr(cheap.tasks, "split_shots", split_shots)

    def train_windows(task, shots, **kw):
        calls["train"] += 1; calls["shots"].append(list(shots)); calls["stride"] = kw.get("stride_s")
        return iter([toy_batch(n=300, seed=0), toy_batch(n=300, seed=1)])

    def iter_windows(task, shots, *, test_mode, **kw):
        assert test_mode
        calls["val"] += 1
        calls["val_shots"] = list(shots)
        return iter([toy_batch(n=100, seed=2)])

    monkeypatch.setattr(cheap.harness, "train_windows", train_windows)
    monkeypatch.setattr(cheap.harness, "iter_windows", iter_windows)
    return calls


def test_the_training_shots_are_a_fixed_sorted_sample(monkeypatch):
    monkeypatch.setattr(cheap.tasks, "split_shots", lambda split: list(range(1, 6001)))
    shots = cheap.train_shots()
    assert len(shots) == 3000 and shots == sorted(shots) and len(set(shots)) == 3000
    assert shots == cheap.train_shots()
    assert shots == sorted(np.random.default_rng(0).choice(np.arange(1, 6001), 3000, replace=False).tolist())


@pytest.mark.parametrize("name", ["shrunk", "ridge"])
def test_obtain_fits_once_and_reuses_the_file(stub_data, toy_task, tmp_path, name, capsys):
    kw = dict(num_workers=2, batch_size=8, force=False)
    p1, digest1 = cheap.obtain(name, toy_task, tmp_path, **kw)
    path = cheap.fit_path(tmp_path, toy_task.name, name)
    assert path.exists() and stub_data["train"] == 1 and stub_data["stride"] == 0.005
    assert stub_data["val"] == (1 if name == "ridge" else 0)
    assert len(stub_data["shots"][0]) == 3000
    digest = cheap.current_fit_digest(path, name, toy_task.name)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    p2, digest2 = cheap.obtain(name, toy_task, tmp_path, **kw)    # reused
    assert digest1 == digest2 == digest
    assert stub_data["train"] == 1 and "reusing" in capsys.readouterr().out
    assert cheap.current_fit_digest(path, name, toy_task.name) == digest          # a reuse does not rewrite the file
    p3, _ = cheap.obtain(name, toy_task, tmp_path, **{**kw, "force": True})
    assert stub_data["train"] == 2 and p1 is not p2 and p3 is not None


@pytest.mark.parametrize("change", ["code", "shots", "stride", "task"])
def test_obtain_fits_again_when_what_defines_the_fit_changed(stub_data, toy_task, tmp_path, monkeypatch, change):
    cheap.obtain("shrunk", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    path = cheap.fit_path(tmp_path, toy_task.name, "shrunk")
    assert cheap.current_fit_digest(path, "shrunk", toy_task.name) != cheap.NO_CURRENT_FIT
    task = toy_task
    if change == "code":
        monkeypatch.setattr(cheap, "PREDICTOR_SHA256", "0" * 64)
    elif change == "shots":
        monkeypatch.setattr(cheap.tasks, "split_shots", lambda split: list(range(2, 6002)))
    elif change == "stride":
        monkeypatch.setattr(cheap, "TRAIN_STRIDE_S", 0.01)
    else:
        task = dataclasses.replace(toy_task, name="task_other")
        path = cheap.fit_path(tmp_path, "task_other", "shrunk")
        assert cheap.current_fit_digest(path, "shrunk", "task_other") == cheap.NO_CURRENT_FIT   # no file for that task
    if change != "task":
        assert cheap.current_fit_digest(path, "shrunk", toy_task.name) == cheap.NO_CURRENT_FIT   # the file on disk is stale
    cheap.obtain("shrunk", task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert stub_data["train"] == 2
    assert cheap.current_fit_digest(cheap.fit_path(tmp_path, task.name, "shrunk"), "shrunk", task.name) != cheap.NO_CURRENT_FIT


def test_an_unreadable_fit_file_is_fitted_again(stub_data, toy_task, tmp_path):
    path = cheap.fit_path(tmp_path, toy_task.name, "shrunk")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an npz")
    assert cheap.current_fit_digest(path, "shrunk", toy_task.name) == cheap.NO_CURRENT_FIT
    cheap.obtain("shrunk", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert stub_data["train"] == 1 and cheap.load_fit(path)[1]["predictor"] == "shrunk"


# ------------------------------------------------------------------ the command

@pytest.fixture
def command(monkeypatch, tmp_path, toy_task, stub_data):
    """run_headroom with the data and the scorer stubbed; returns (module, scored, stage0 root)."""
    rh = _run_headroom()
    root = tmp_path / "stage0"
    monkeypatch.setattr(rh.paths, "stage0_dir", lambda: root)
    monkeypatch.setattr(rh.paths, "prepare_write_root", lambda p: Path(p))
    monkeypatch.setattr(rh.tasks, "load_task", lambda name: dataclasses.replace(toy_task, name=name))
    scored = []

    def fake_score(task, predict_fn, out_dir, *, split, **kwargs):
        from tokamak_foundation_model.ignite.mast import harness as h
        scored.append(split)
        b = _a_batch(toy_task)
        assert np.isfinite(predict_fn(b)["summary-ip"]).all()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return h.ScoreResult(task=task.name, task_nrmse=0.25,
                             per_signal=pd.DataFrame({"feature_name": ["summary-ip"], "NRMSE": [0.25]}),
                             per_shot=pd.DataFrame({"shot_id": [1], "NRMSE": [0.25]}), n_windows=7,
                             out_dir=Path(out_dir))

    monkeypatch.setattr(rh.harness, "score", fake_score)
    return rh, scored, root


def _a_batch(task):
    rng = np.random.default_rng(0)

    def draw(specs, length):
        return {s.name: (s.mean + s.std * rng.standard_normal((3, *s.values_shape, length(s)))).astype(np.float32)
                for s in specs}

    inputs = draw(task.inputs, task.n_in)
    return harness.WindowBatch(np.arange(3), np.zeros(3, dtype=np.int64), inputs, draw(task.actuators, task.n_act),
                               draw(task.outputs, task.n_out), {n: a[..., -1].copy() for n, a in inputs.items()})


@pytest.mark.parametrize("name", ["shrunk", "ridge"])
def test_the_command_ties_the_stored_score_to_the_fit_file(command, stub_data, name, capsys):
    rh, scored, root = command
    argv = ["--task", "task_1-2", "--predictor", name, "--split", "val_report"]
    out_dir = root / "scores" / "task_1-2" / name / "val_report"
    fit = root / "cheap" / "task_1-2" / f"{name}.npz"

    assert rh.main(argv) == 0
    stored = json.loads((out_dir / "score.json").read_text())
    assert fit.exists() and stub_data["train"] == 1 and scored == ["val_report"]
    assert stored["predictor_sha256"] == cheap.PREDICTOR_SHA256
    assert stored["fit_sha256"] == hashlib.sha256(fit.read_bytes()).hexdigest()

    assert rh.main(argv) == 0                                      # kept: nothing is fitted or scored
    assert stub_data["train"] == 1 and scored == ["val_report"]
    assert "kept task_1-2" in capsys.readouterr().out

    predictor, _ = cheap.obtain(name, rh.tasks.load_task("task_1-2"), root, num_workers=1, batch_size=1, force=False)
    assert stub_data["train"] == 1                                 # reused
    cheap.save_fit(fit, _perturbed(predictor), cheap.training_meta())                 # another fit file, with valid metadata
    assert rh.main(argv) == 0                                      # the fit on disk is not the one scored
    assert "re-scoring task_1-2" in capsys.readouterr().out
    assert scored == ["val_report"] * 2 and stub_data["train"] == 1
    assert json.loads((out_dir / "score.json").read_text())["fit_sha256"] == hashlib.sha256(fit.read_bytes()).hexdigest()

    fit.unlink()                                                   # a missing fit is fitted again, then scored again
    assert rh.main(argv) == 0
    assert stub_data["train"] == 2 and len(scored) == 3
    assert json.loads((out_dir / "score.json").read_text())["fit_sha256"] == hashlib.sha256(fit.read_bytes()).hexdigest()

    assert rh.main([*argv, "--force"]) == 0                        # --force fits again and scores again
    assert stub_data["train"] == 3 and len(scored) == 4


def _perturbed(predictor):
    """A predictor of the same kind with slightly different fitted arrays (so another file digest)."""
    arrays = predictor.arrays()
    key = next(k for k in arrays if k.startswith("mean/"))
    arrays[key] = arrays[key] + 0.5
    return type(predictor).from_arrays(predictor.task, arrays)


def test_the_command_refits_when_the_code_changed(command, stub_data, monkeypatch):
    rh, scored, root = command
    argv = ["--task", "task_1-2", "--predictor", "shrunk", "--split", "val_report"]
    assert rh.main(argv) == 0 and stub_data["train"] == 1
    monkeypatch.setattr(cheap, "PREDICTOR_SHA256", "1" * 64)
    assert rh.main(argv) == 0
    assert stub_data["train"] == 2 and len(scored) == 2            # never scored from a fit made by other code
    stored = json.loads((root / "scores/task_1-2/shrunk/val_report/score.json").read_text())
    assert stored["predictor_sha256"] == "1" * 64
    assert cheap.load_fit(root / "cheap/task_1-2/shrunk.npz")[1]["code_sha256"] == "1" * 64


# ------------------------------------------------------------------ fix round 1

def test_shrunk_predicts_a_multi_frame_output_that_is_not_an_input(toy_task, toy_batch):
    """Every output without an anchor is the training mean, in as many frames as the output spans."""
    task = dataclasses.replace(toy_task, inputs=toy_task.inputs[1:])      # summary-ip is no longer an input
    spec = task.outputs[0]
    assert task.n_out(spec) > cheap.frame_len(spec.dt)                    # more than one frame: the case that failed
    assert spec.name not in {s.name for s in task.inputs}
    b = toy_batch(n=300, seed=3)
    sp = cheap.ShrunkPersistence(task).fit([b])
    pred = sp.predict(b)["summary-ip"]
    assert pred.shape == b.targets["summary-ip"].shape and np.isfinite(pred).all()
    expected = spec.unstandardize(sp.mean["summary-ip"])[0]
    np.testing.assert_allclose(pred, expected, rtol=1e-5)
    np.testing.assert_allclose(expected, np.nanmean(b.targets["summary-ip"].astype(np.float64)),
                               atol=spec.std * 1e-3)


def test_a_fit_is_reused_only_under_the_scoring_code_it_was_made_under(stub_data, toy_task, tmp_path, monkeypatch,
                                                                     capsys):
    kw = dict(num_workers=2, batch_size=8, force=False)
    monkeypatch.setattr(cheap.baselines, "scoring_code_sha256", lambda: "a" * 64)
    cheap.obtain("shrunk", toy_task, tmp_path, **kw)
    path = cheap.fit_path(tmp_path, toy_task.name, "shrunk")
    assert cheap.load_fit(path)[1]["scoring_code_sha256"] == "a" * 64
    digest = cheap.current_fit_digest(path, "shrunk", toy_task.name)
    assert digest != cheap.NO_CURRENT_FIT
    cheap.obtain("shrunk", toy_task, tmp_path, **kw)                       # same hash: reused
    assert stub_data["train"] == 1

    monkeypatch.setattr(cheap.baselines, "scoring_code_sha256", lambda: "b" * 64)   # harness.py or tasks.py changed
    assert cheap.current_fit_digest(path, "shrunk", toy_task.name) == cheap.NO_CURRENT_FIT
    capsys.readouterr()
    cheap.obtain("shrunk", toy_task, tmp_path, **kw)
    assert stub_data["train"] == 2 and "scoring_code_sha256 changed" in capsys.readouterr().out
    assert cheap.load_fit(path)[1]["scoring_code_sha256"] == "b" * 64


def test_a_fit_file_without_the_scoring_code_hash_is_not_current(stub_data, toy_task, tmp_path, capsys):
    cheap.obtain("shrunk", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    path = cheap.fit_path(tmp_path, toy_task.name, "shrunk")
    arrays, meta = cheap.load_fit(path)
    del meta["scoring_code_sha256"]
    with open(path, "wb") as f:
        np.savez(f, **arrays, meta=np.array(json.dumps(meta)))
    assert cheap.current_fit_digest(path, "shrunk", toy_task.name) == cheap.NO_CURRENT_FIT
    capsys.readouterr()
    cheap.obtain("shrunk", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert stub_data["train"] == 2 and "scoring_code_sha256 changed" in capsys.readouterr().out


# ------------------------------------------------------------------ final fix wave: C1

def _decades_selection(monkeypatch, shots):
    monkeypatch.setattr(cheap.tasks, "split_shots", lambda split: list(shots) if split == "val_select" else [])


@pytest.mark.parametrize("n", [1, 2, 7, 200, 399, 400, 401, 5000])
def test_select_shots_are_the_direct_models_validation_shots(monkeypatch, n):
    """The rule exists twice (`cheap` cannot import `direct`, which imports it); this pins them equal."""
    shots = [7000 + 3 * i for i in range(400)]
    _decades_selection(monkeypatch, shots)
    chosen = cheap.select_shots(n)
    assert chosen == direct.validation_shots(n)
    assert len(chosen) == len(set(chosen)) == min(n, 400) and chosen == sorted(chosen)
    assert chosen[0] == shots[0] and (n == 1 or chosen[-1] == shots[-1])
    if n >= 400:
        assert chosen == shots


def test_select_shots_spread_over_the_whole_sorted_list(monkeypatch):
    shots = [7000 + 3 * i for i in range(1000)]
    _decades_selection(monkeypatch, shots)
    assert cheap.select_shots(4) == [shots[0], shots[333], shots[666], shots[999]]
    assert cheap.N_SELECT_SHOTS == 200 and len(cheap.select_shots()) == 200
    assert cheap.select_shots()[-1] == shots[-1] and cheap.select_shots() != shots[:200]


def test_ridges_penalty_is_chosen_on_evenly_spaced_val_select_shots(stub_data, toy_task, tmp_path):
    cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    expected = [7000 + int(i) for i in np.linspace(0, 399, 200).round()]
    assert stub_data["val_shots"] == expected == cheap.select_shots()
    assert expected[0] == 7000 and expected[-1] == 7399 and expected != list(range(7000, 7200))
    assert set(stub_data["splits"]) == {"train", "val_select"}          # never val_report or test


def test_the_penalty_grid_is_the_ten_decades():
    assert cheap.RidgeFrames.lambdas == (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _selecting(toy_task, toy_batch, best, *, raises=(), not_finite=()):
    """A fitted ridge whose solves are scripted: `best` gives the weights that fit the validation windows
    best, every other penalty a worse set, those in `raises` a Cholesky failure, those in `not_finite` NaN."""
    rf = cheap.RidgeFrames(toy_task).fit([toy_batch(n=300, seed=0)])
    good = rf._solve(1e-3)

    def solve(lam):
        if lam in raises:
            raise np.linalg.LinAlgError("the matrix is not positive definite")
        if lam in not_finite:
            w = good.copy(); w[0, 0] = np.nan
            return w
        return good if lam == best else good + 1.0

    rf._solve = solve
    return rf, [toy_batch(n=150, seed=1)]


@pytest.mark.parametrize("best, edge", [(1e-8, "lower"), (1e-5, None), (1e-2, None), (10.0, "upper")])
def test_the_selection_flags_a_choice_at_an_edge_of_the_grid(toy_task, toy_batch, best, edge):
    rf, val = _selecting(toy_task, toy_batch, best)
    assert rf.select(val) == best and rf.lam == best
    assert rf.at_edge == edge and rf.candidates == cheap.RidgeFrames.lambdas


def test_a_penalty_whose_solve_is_not_finite_is_not_a_candidate_and_is_printed(toy_task, toy_batch, capsys):
    rf, val = _selecting(toy_task, toy_batch, 1e-5, raises=(1e-8, 1e-7), not_finite=(1e-6,))
    assert rf.select(val) == 1e-5
    assert rf.candidates == (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    assert rf.at_edge == "lower"                  # relative to the candidates that solved, not to the grid
    out = capsys.readouterr().out
    for lam in ("1e-08", "1e-07", "1e-06"):
        assert lam in out
    assert "not a candidate" in out


def test_a_selection_with_no_penalty_that_solves_is_an_error(toy_task, toy_batch):
    rf, val = _selecting(toy_task, toy_batch, 1e-3, raises=cheap.RidgeFrames.lambdas)
    with pytest.raises(ValueError, match="no penalty"):
        rf.select(val)


def test_a_real_selection_has_finite_weights_for_every_candidate(toy_task, toy_batch):
    rf = cheap.RidgeFrames(toy_task).fit([toy_batch(n=400, seed=0)])
    rf.select([toy_batch(n=200, seed=1)])
    assert rf.lam in rf.candidates and set(rf.candidates) <= set(cheap.RidgeFrames.lambdas)
    assert np.isfinite(rf.W).all() and rf.at_edge in (None, "lower", "upper")


def test_the_fit_file_stores_the_penalty_the_candidates_and_the_edge(stub_data, toy_task, tmp_path, capsys):
    cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    path = cheap.fit_path(tmp_path, toy_task.name, "ridge")
    arrays, meta = cheap.load_fit(path)
    assert meta["lambda"] == float(arrays["lam"]) and meta["lambda"] in meta["lambda_candidates"]
    assert meta["lambda_grid"] == list(cheap.RidgeFrames.lambdas)
    assert set(meta["lambda_candidates"]) <= set(meta["lambda_grid"])
    assert meta["lambda_at_edge"] in (None, "lower", "upper")
    out = capsys.readouterr().out
    assert f"lambda {meta['lambda']:g}" in out and "candidates" in out
    # a reused fit carries the same record
    capsys.readouterr()
    reused, _ = cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert reused.lam == meta["lambda"] and reused.at_edge == meta["lambda_at_edge"]
    assert list(reused.candidates) == meta["lambda_candidates"]
    assert reused.fit_record() == {"fit_lambda": meta["lambda"], "fit_lambda_at_edge": meta["lambda_at_edge"],
                                   "fit_n_windows": 600, "fit_n_channels_without_target": 0}


def test_obtain_prints_when_the_choice_is_at_an_edge(stub_data, toy_task, tmp_path, monkeypatch, capsys):
    real = cheap.RidgeFrames.select

    def at_the_lower_edge(self, val_batches):
        real(self, val_batches)
        self.lam, self.W, self.at_edge = self.candidates[0], self._solve(self.candidates[0]), "lower"
        return self.lam

    monkeypatch.setattr(cheap.RidgeFrames, "select", at_the_lower_edge)
    cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert "lower edge" in capsys.readouterr().out
    assert cheap.load_fit(cheap.fit_path(tmp_path, toy_task.name, "ridge"))[1]["lambda_at_edge"] == "lower"


def test_the_shrunk_record_has_the_two_counts_only(toy_task, toy_batch):
    shrunk, ridge = _fitted(toy_task, toy_batch)
    assert shrunk.fit_record() == {"fit_n_windows": 200, "fit_n_channels_without_target": 0}
    assert set(ridge.fit_record()) == {"fit_n_windows", "fit_n_channels_without_target", "fit_lambda",
                                       "fit_lambda_at_edge"}


def test_ridge_has_no_lazy_solve(toy_task, toy_batch):
    """`use_lambda` and the solve inside `predict` are gone: weights come from `select`, or `predict` says so."""
    rf = cheap.RidgeFrames(toy_task).fit([toy_batch(n=100, seed=0)])
    assert not hasattr(rf, "use_lambda")
    with pytest.raises(ValueError, match="select"):
        rf.predict(toy_batch(n=2, seed=1))


# ---- the digest and the old fits

def test_save_fit_returns_the_digest_of_the_bytes_it_wrote(toy_task, toy_batch, tmp_path):
    shrunk, _ = _fitted(toy_task, toy_batch)
    path = tmp_path / "cheap" / "task_toy" / "shrunk.npz"
    digest = cheap.save_fit(path, shrunk, {})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["shrunk", "ridge"])
def test_obtain_digests_the_bytes_it_loaded_from_one_read(stub_data, toy_task, tmp_path, monkeypatch, name):
    cheap.obtain(name, toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    path = cheap.fit_path(tmp_path, toy_task.name, name)
    reads = []
    real = Path.read_bytes

    def counting(self):
        data = real(self)
        if self == path:
            reads.append(hashlib.sha256(data).hexdigest())
        return data

    monkeypatch.setattr(Path, "read_bytes", counting)
    _, digest = cheap.obtain(name, toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert reads == [digest] and stub_data["train"] == 1       # one read, and the digest is of what it returned
    assert digest != cheap.NO_CURRENT_FIT


def test_a_forced_fit_returns_the_digest_of_the_file_it_wrote(stub_data, toy_task, tmp_path):
    _, digest = cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=True)
    path = cheap.fit_path(tmp_path, toy_task.name, "ridge")
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def _rewritten(path, *, drop_array=None, drop_meta=None):
    arrays, meta = cheap.load_fit(path)
    arrays.pop(drop_array, None)
    meta.pop(drop_meta, None)
    with open(path, "wb") as f:
        np.savez(f, **arrays, meta=np.array(json.dumps(meta)))


@pytest.mark.parametrize("damage", [dict(drop_meta="lambda_at_edge"), dict(drop_meta="lambda_candidates"),
                                    dict(drop_array="W"), dict(drop_array="lam")])
def test_a_ridge_fit_from_before_the_grid_change_is_fitted_again_not_an_error(stub_data, toy_task, tmp_path, damage,
                                                                           capsys):
    cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    path = cheap.fit_path(tmp_path, toy_task.name, "ridge")
    _rewritten(path, **damage)                  # the code hash is the present one: only the record is old
    assert cheap.current_fit_digest(path, "ridge", toy_task.name) == cheap.NO_CURRENT_FIT
    capsys.readouterr()
    predictor, digest = cheap.obtain("ridge", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert stub_data["train"] == 2 and predictor.at_edge in (None, "lower", "upper")
    assert digest == cheap.current_fit_digest(path, "ridge", toy_task.name)


def test_a_fit_that_passes_the_record_check_but_cannot_be_restored_is_fitted_again(stub_data, toy_task, tmp_path,
                                                                               monkeypatch):
    cheap.obtain("shrunk", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    path = cheap.fit_path(tmp_path, toy_task.name, "shrunk")
    _rewritten(path, drop_array="slope/summary-ip")
    assert cheap.current_fit_digest(path, "shrunk", toy_task.name) != cheap.NO_CURRENT_FIT   # the name-free check
    predictor, digest = cheap.obtain("shrunk", toy_task, tmp_path, num_workers=2, batch_size=8, force=False)
    assert stub_data["train"] == 2 and "slope/summary-ip" in cheap.load_fit(path)[0]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the features and the predictions

def test_a_signal_with_a_zero_std_gives_finite_features_and_a_zero_flag(toy_task, toy_batch):
    ip = dataclasses.replace(toy_task.inputs[0], std=0.0)
    coil = dataclasses.replace(toy_task.actuators[0], std=0.0)
    task = dataclasses.replace(toy_task, inputs=(ip, toy_task.inputs[1]), actuators=(coil,))
    f = cheap.window_features(toy_batch(n=5, seed=0), task)          # -W error: no divide warning
    assert f.shape == (5, N_FEATURES) and np.isfinite(f).all()
    assert np.all(f[:, :2] == 0)                                      # value 0, flag 0: the signal is unusable
    assert f[:, 17:32].min() == 1                                     # the flux flags are intact
    assert np.all(f[:, 32:-1] == 0)                                   # the actuator block


def test_predictions_do_not_depend_on_the_batch_targets(toy_task, toy_batch):
    b = toy_batch(n=6, seed=3)
    no_targets = dataclasses.replace(b, targets={})
    nan_targets = dataclasses.replace(b, targets={k: np.full_like(v, np.nan) for k, v in b.targets.items()})
    for predictor in _fitted(toy_task, toy_batch):
        expected = predictor.predict(b)["summary-ip"]
        for other in (no_targets, nan_targets):
            np.testing.assert_array_equal(predictor.predict(other)["summary-ip"], expected)


# ---- spectrogram outputs: the path behind the task 4-5 numbers

def _spectrogram_task():
    from tokamak_foundation_model.ignite.mast import tasks
    name = tasks.SPECTROGRAM_SIGNALS[0]
    spec = tasks.SignalSpec(name, 0.001024, (3, 4), 1.0, 2.0)
    return spec, tasks.TaskSpec(name="task_spec", markovian=False, input_length=0.02, output_length=0.01,
                                delta=0.0, stride=0.01, inputs=(spec,), actuators=(), outputs=(spec,),
                                steps={"input": {spec.name: 20}, "actuator": {}, "output": {spec.name: 10}})


def _spectrogram_batch(spec, n, seed, gain=3.0):
    """Inputs random; each target is `gain` times the standardized per-bin input mean, held over the output."""
    rng = np.random.default_rng(seed)
    x = (spec.mean + spec.std * rng.standard_normal((n, 3, 4, 20))).astype(np.float32)
    anchor = spec.standardize(x.astype(np.float64).mean(axis=-1))               # (n, 3, 4)
    y = spec.unstandardize(gain * anchor[..., None] + np.zeros((n, 3, 4, 10))).astype(np.float32)
    return harness.WindowBatch(np.arange(n), np.zeros(n, dtype=np.int64), {spec.name: x}, {}, {spec.name: y},
                               {spec.name: x[..., -1].copy()})


def test_shrunk_and_ridge_with_a_spectrogram_output_hold_one_frame_over_the_window():
    spec, task = _spectrogram_task()
    train = [_spectrogram_batch(spec, 400, 0), _spectrogram_batch(spec, 400, 1)]
    test = _spectrogram_batch(spec, 50, 2)
    assert cheap.target_count(task) == 12 and cheap.feature_count(task) == 12 + 1 + 1

    shrunk = cheap.ShrunkPersistence(task).fit(train)
    assert shrunk.slope[spec.name].shape == (1,)
    assert abs(float(shrunk.slope[spec.name][0]) - 3.0) < 0.15                  # the slope of target on anchor
    ridge = cheap.RidgeFrames(task).fit(train)
    ridge.select([_spectrogram_batch(spec, 100, 3)])
    for predictor in (shrunk, ridge):
        out = predictor.predict(test)[spec.name]
        assert out.shape == test.targets[spec.name].shape and out.dtype == np.float32
        assert np.isfinite(out).all()
        assert np.all(out == out[..., :1])                                     # one value per bin, held in time
        z = spec.standardize(out[..., 0].astype(np.float64))
        want = spec.standardize(test.targets[spec.name][..., 0].astype(np.float64))
        assert np.sqrt(np.mean((z - want) ** 2)) < 0.12                        # the anchor is the information


def test_ridge_with_a_spectrogram_output_recovers_a_map_that_shrunk_cannot():
    """Targets that depend on the neighbouring bin: the per-bin slope of shrunk cannot, ridge can."""
    spec, task = _spectrogram_task()

    def batch(n, seed):
        b = _spectrogram_batch(spec, n, seed, gain=0.0)
        anchor = spec.standardize(b.inputs[spec.name].astype(np.float64).mean(axis=-1))
        z = 4.0 * np.roll(anchor, 1, axis=-1)
        b.targets[spec.name] = spec.unstandardize(z[..., None] + np.zeros((n, 3, 4, 10))).astype(np.float32)
        return b

    ridge = cheap.RidgeFrames(task).fit([batch(800, 0), batch(800, 1)])
    ridge.select([batch(200, 2)])
    shrunk = cheap.ShrunkPersistence(task).fit([batch(800, 0)])
    test = batch(100, 3)
    want = spec.standardize(test.targets[spec.name][..., 0].astype(np.float64))
    err = {}
    for name, p in (("ridge", ridge), ("shrunk", shrunk)):
        got = spec.standardize(p.predict(test)[spec.name][..., 0].astype(np.float64))
        err[name] = float(np.sqrt(np.mean((got - want) ** 2)))
    assert err["ridge"] < 0.1 and err["shrunk"] > 0.5


# ---- the streamed centring against a two-pass fit

def test_streamed_centring_equals_a_two_pass_fit_when_targets_are_missing(toy_task, toy_batch):
    spec = toy_task.outputs[0]
    rng = np.random.default_rng(5)
    batches = []
    for seed in (0, 1, 2):
        b = toy_batch(n=120, seed=seed)
        t = b.targets["summary-ip"]
        t[rng.random(t.shape) < 0.15] = np.nan                      # scattered missing samples
        t[:10, :, 20:40] = np.nan                                   # a whole frame missing in ten windows
        t[10:15] = np.nan                                           # ten windows with no target at all
        batches.append(b)
    rf = cheap.RidgeFrames(toy_task).fit(batches)

    # two passes: the mean first, then the centred moments
    z = np.concatenate([spec.standardize(b.targets["summary-ip"].astype(np.float64)) for b in batches])
    m = np.nansum(z) / np.isfinite(z).sum()
    np.testing.assert_allclose(rf.mean["summary-ip"], [m], rtol=1e-12)
    x = np.concatenate([cheap.window_features(b, toy_task) for b in batches])
    frames = z.reshape(len(z), 1, N_FRAMES, 20)
    finite = np.isfinite(frames)
    y = np.where(finite.any(axis=-1), np.where(finite, frames, 0.0).sum(axis=-1) / np.maximum(finite.sum(axis=-1), 1),
                 np.nan).reshape(len(z), N_FRAMES)
    valid = np.isfinite(y)
    yc = np.where(valid, y - m, 0.0)
    xtx_ref, xty_ref = x.T @ x, x.T @ yc
    xtx, xty, n = rf._moments
    assert n == len(z)
    np.testing.assert_allclose(xtx, xtx_ref, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(xty, xty_ref, rtol=1e-9, atol=1e-8)
    lam = 1e-3
    penalty = np.eye(x.shape[1]) * lam * n
    penalty[-1, -1] = 0.0
    np.testing.assert_allclose(rf._solve(lam), np.linalg.solve(xtx_ref + penalty, xty_ref), rtol=1e-7, atol=1e-9)


# ---- the direct model's two imports (C1-M3)

def test_the_values_the_direct_model_imports_are_pinned():
    """`direct.py` imports `FRAME_S` and `frame_len`, and its hash does not cover this file: a change here
    would change the model's patches and every direct score silently."""
    assert cheap.FRAME_S == 0.005
    # the distinct sampling intervals of the 14 benchmark tasks' signals (read from TokaMark 2026-11-22)
    assert {dt: cheap.frame_len(dt) for dt in (2e-05, 0.0002, 0.00025, 0.001024, 0.005)} == {
        2e-05: 250, 0.0002: 25, 0.00025: 20, 0.001024: 5, 0.005: 1}
    assert cheap.frame_len(0.02) == 1 and cheap.frame_len(1.0) == 1          # never below one sample
    assert direct.FRAME_S is cheap.FRAME_S and direct.frame_len is cheap.frame_len


# ---- the command: what the fit knows reaches score.json, and the digest is of the bytes that were used

@pytest.mark.parametrize("name", ["shrunk", "ridge"])
def test_score_json_carries_what_the_fit_knows(command, stub_data, name):
    rh, scored, root = command
    assert rh.main(["--task", "task_1-2", "--predictor", name, "--split", "val_report"]) == 0
    stored = json.loads((root / "scores" / "task_1-2" / name / "val_report" / "score.json").read_text())
    assert stored["fit_n_windows"] == 600 and stored["fit_n_channels_without_target"] == 0
    if name == "ridge":
        assert stored["fit_lambda"] in cheap.RidgeFrames.lambdas
        assert "fit_lambda_at_edge" in stored and stored["fit_lambda_at_edge"] in (None, "lower", "upper")
        meta = cheap.load_fit(root / "cheap" / "task_1-2" / "ridge.npz")[1]
        assert stored["fit_lambda"] == meta["lambda"] and stored["fit_lambda_at_edge"] == meta["lambda_at_edge"]
    else:
        assert not {"fit_lambda", "fit_lambda_at_edge"} & set(stored)


def test_a_reused_fit_gives_the_same_record(command, stub_data):
    rh, scored, root = command
    argv = ["--task", "task_1-2", "--predictor", "ridge", "--split", "val_report"]
    out = root / "scores" / "task_1-2" / "ridge" / "val_report" / "score.json"
    assert rh.main(argv) == 0
    first = json.loads(out.read_text())
    out.unlink()                                              # score again from the reused fit file
    assert rh.main(argv) == 0 and stub_data["train"] == 1
    again = json.loads(out.read_text())
    for key in ("fit_n_windows", "fit_n_channels_without_target", "fit_lambda", "fit_lambda_at_edge", "fit_sha256"):
        assert again[key] == first[key]


def test_the_stored_digest_is_that_of_the_bytes_the_predictor_was_made_from(command, stub_data, monkeypatch):
    """A writer that replaces the fit file after it was read must not change the digest that is stored."""
    rh, scored, root = command
    real = cheap.obtain
    seen = {}

    def obtain_then_replace(*a, **kw):
        predictor, digest = real(*a, **kw)
        path = cheap.fit_path(root, "task_1-2", "shrunk")
        cheap.save_fit(path, _perturbed(predictor), cheap.training_meta())      # another writer, after the read
        seen["digest"] = digest
        return predictor, digest

    monkeypatch.setattr(cheap, "obtain", obtain_then_replace)
    assert rh.main(["--task", "task_1-2", "--predictor", "shrunk", "--split", "val_report"]) == 0
    fit = root / "cheap" / "task_1-2" / "shrunk.npz"
    stored = json.loads((root / "scores/task_1-2/shrunk/val_report/score.json").read_text())
    assert stored["fit_sha256"] == seen["digest"] != hashlib.sha256(fit.read_bytes()).hexdigest()


def test_the_no_current_fit_sentinel_is_never_stored_as_a_digest(command, stub_data, monkeypatch, capsys):
    rh, scored, root = command
    real = cheap.obtain
    monkeypatch.setattr(cheap, "obtain", lambda *a, **kw: (real(*a, **kw)[0], cheap.NO_CURRENT_FIT))
    assert rh.main(["--task", "task_1-2", "--predictor", "ridge", "--split", "val_report"]) == 1
    assert "no current fit" in capsys.readouterr().err
    assert scored == [] and not (root / "scores" / "task_1-2" / "ridge").exists()


# ---- A-R3: the fitted predictors are scored blind to the targets

def test_the_cheap_predictors_give_the_same_output_blinded_or_not(command, toy_task, toy_batch):
    rh, scored, root = command
    b = toy_batch(n=6, seed=21)
    for predictor in _fitted(toy_task, toy_batch):
        np.testing.assert_array_equal(rh._blind_to_targets(predictor.predict)(b)["summary-ip"],
                                      predictor.predict(b)["summary-ip"])


@pytest.mark.parametrize("name", ["shrunk", "ridge"])
def test_the_command_scores_the_cheap_predictors_through_the_blind_wrapper(command, stub_data, monkeypatch, name):
    rh, scored, root = command
    seen = []
    cls = cheap.PREDICTOR_CLASSES[name]
    real = cls.predict

    def spy(self, batch):
        seen.append([bool(np.isnan(v).all()) for v in batch.targets.values()])
        return real(self, batch)

    monkeypatch.setattr(cls, "predict", spy)
    assert rh.main(["--task", "task_1-2", "--predictor", name, "--split", "val_report"]) == 0
    assert seen and all(all(flags) for flags in seen)          # the fake scorer passes a batch with finite targets
