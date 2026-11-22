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
from tokamak_foundation_model.ignite.mast.headroom import cheap

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
    """Training and validation windows from the toy generator; counts how often each is read."""
    calls = {"train": 0, "val": 0, "shots": []}
    monkeypatch.setattr(cheap.tasks, "split_shots",
                        lambda split: list(range(1, 6001)) if split == "train" else list(range(7000, 7400)))

    def train_windows(task, shots, **kw):
        calls["train"] += 1; calls["shots"].append(list(shots)); calls["stride"] = kw.get("stride_s")
        return iter([toy_batch(n=300, seed=0), toy_batch(n=300, seed=1)])

    def iter_windows(task, shots, *, test_mode, **kw):
        assert test_mode and len(shots) == 200
        calls["val"] += 1
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
    p1 = cheap.obtain(name, toy_task, tmp_path, **kw)
    path = cheap.fit_path(tmp_path, toy_task.name, name)
    assert path.exists() and stub_data["train"] == 1 and stub_data["stride"] == 0.005
    assert stub_data["val"] == (1 if name == "ridge" else 0)
    assert len(stub_data["shots"][0]) == 3000
    digest = cheap.current_fit_digest(path, name, toy_task.name)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    p2 = cheap.obtain(name, toy_task, tmp_path, **kw)             # reused
    assert stub_data["train"] == 1 and "reusing" in capsys.readouterr().out
    assert cheap.current_fit_digest(path, name, toy_task.name) == digest          # a reuse does not rewrite the file
    p3 = cheap.obtain(name, toy_task, tmp_path, **{**kw, "force": True})
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

    predictor = cheap.obtain(name, rh.tasks.load_task("task_1-2"), root, num_workers=1, batch_size=1, force=False)
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
