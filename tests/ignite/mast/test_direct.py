"""The direct supervised model (headroom column C): model, loss, trainer and the command's handling of it.

Everything runs on the CPU with tiny configurations and the toy task. No test opens real data, uses a
GPU or writes under the real stage-0 root: data access (`harness.train_windows`, `harness.score`,
`tasks.split_shots`) and `nvidia-smi` are stubbed and the write root is `tmp_path`.
"""

import csv
import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tokamak_foundation_model.ignite.mast import harness, tasks  # noqa: E402
from tokamak_foundation_model.ignite.mast.headroom import FITTED, direct  # noqa: E402

TINY = direct.DirectConfig(d_model=32, depth=2, heads=4, batch_size=16, max_steps=300, warmup=10)
SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "run_headroom.py"


@pytest.fixture(autouse=True)
def _few_threads():
    """The tiny models gain nothing from the many intra-op threads of this machine, which fight the other jobs (measured 2026-11-22: 25 ms per tiny matmul with the default threads, 0.5 ms with one)."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


# ------------------------------------------------------------------ the model

def test_output_shapes(toy_task, toy_batch):
    m = direct.DirectModel(toy_task, TINY)
    x, a, y = direct.to_tensors(toy_batch(n=3, seed=0), toy_task, "cpu")
    out = m(x, a)
    s = toy_task.outputs[0]
    assert out[s.name].shape == (3, *s.values_shape, toy_task.n_out(s))
    assert y[s.name].shape == out[s.name].shape


def test_partial_last_patch_for_spectrogram_outputs(toy_task):
    # a (3, 256) output at dt = 1.024 ms with 98 stamps: 20 patches of 5 stamps, the last holds 3
    spec = tasks.SignalSpec("magnetics-b_field_tor_probe_cc_field", 0.001024, (3, 256), 0.0, 1.0)
    ip = toy_task.inputs[0]
    task = dataclasses.replace(toy_task, inputs=(ip,), actuators=(), outputs=(spec,),
                               steps={"input": {ip.name: 20}, "actuator": {}, "output": {spec.name: 98}})
    out = direct.DirectModel(task, TINY)({ip.name: torch.randn(2, 1, 20)}, {})
    assert out[spec.name].shape == (2, 3, 256, 98)
    counts = direct.token_counts(task, TINY)
    assert counts["query"] == 3 * 20            # one token per probe and patch, all 256 bins in each


def test_direct_model_is_finite_with_all_nan_inputs(toy_task, toy_batch):
    b = toy_batch(n=2, seed=0)
    for v in b.inputs.values():
        v[:] = np.nan
    out = direct.DirectModel(toy_task, TINY)(*direct.to_tensors(b, toy_task, "cpu")[:2])
    assert all(torch.isfinite(o).all() for o in out.values())


def test_token_counts_follow_the_patch_rule(toy_task):
    # inputs: ip 20 samples at 0.25 ms (p = 20 -> 1 patch), 15 flux loops 25 samples at 0.2 ms (p = 25 -> 1 patch)
    # actuators: 4 coils, 120 samples (6 patches); queries: ip, 100 samples (5 patches)
    assert direct.token_counts(toy_task, TINY) == {"input": 1 + 15, "actuator": 4 * 6, "query": 5}


def test_time_enters_in_seconds_relative_to_the_cut(toy_task):
    inputs, actuators, outputs = direct.layouts(toy_task, TINY)
    ip = next(l for l in inputs if l.name == "summary-ip")
    assert ip.centres == pytest.approx((-0.0025,))                       # 20 samples of 0.25 ms end at the cut
    coil = actuators[0]                                                   # ends at delta + output length = 25 ms
    assert coil.centres[-1] == pytest.approx(0.025 - 0.0025) and coil.centres[0] == pytest.approx(-0.0025 - 0.0)
    assert outputs[0].centres == pytest.approx(tuple(0.0025 + 0.005 * k for k in range(5)))


# ------------------------------------------------------------------ the loss

def test_masked_mse_ignores_nan_targets_and_has_finite_gradients(toy_task, toy_batch):
    s = toy_task.outputs[0]
    shape = (4, *s.values_shape, toy_task.n_out(s))
    gen = torch.Generator().manual_seed(0)
    pred = torch.randn(shape, generator=gen, requires_grad=True)
    target = torch.randn(shape, generator=gen)
    target[:2] = float("nan")
    loss = direct.masked_mse({s.name: pred}, {s.name: target})
    assert torch.isfinite(loss)
    expected = ((pred[2:] - target[2:]) ** 2).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad[:2] == 0).all() and (pred.grad[2:] != 0).any()


def test_masked_mse_weights_signals_equally():
    a = torch.zeros(2, 1, 4); b = torch.zeros(2, 100, 4)
    ta = torch.ones_like(a); tb = 3 * torch.ones_like(b)
    loss = direct.masked_mse({"a": a, "b": b}, {"a": ta, "b": tb})
    assert torch.allclose(loss, torch.tensor((1.0 + 9.0) / 2))          # not weighted by element count


def test_masked_mse_with_nothing_finite_is_zero_with_a_gradient_path():
    p = torch.randn(2, 3, requires_grad=True)
    loss = direct.masked_mse({"a": p}, {"a": torch.full((2, 3), float("nan"))})
    loss.backward()
    assert loss.item() == 0.0 and torch.isfinite(p.grad).all()


def test_overfits_a_linear_rule(toy_task, toy_batch):
    torch.manual_seed(0)
    b = toy_batch(n=64, seed=1)
    x, a, _ = direct.to_tensors(b, toy_task, "cpu")
    s = toy_task.outputs[0]
    target = {s.name: (2.0 * x["summary-ip"][..., -1:] + 1.0).expand(64, 1, toy_task.n_out(s)).contiguous()}
    model = direct.DirectModel(toy_task, TINY)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # 0.0034 at step 300 (measured 2026-11-22); 2e-3 oscillates at 0.002 to 0.02
    for _ in range(300):
        opt.zero_grad()
        loss = direct.masked_mse(model(x, a), target)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-2


# ------------------------------------------------------------------ checkpoint and predictor

def _save(toy_task, tmp_path, cfg=TINY):
    model = direct.DirectModel(toy_task, cfg)
    ckpt = tmp_path / "best.pt"
    direct.write_config(tmp_path, toy_task, cfg)
    direct.save_checkpoint(ckpt, model, cfg, toy_task, step=0, val_nrmse=None)
    return ckpt


def test_predictor_returns_physical_units(toy_task, toy_batch, tmp_path):
    ckpt = _save(toy_task, tmp_path)
    batch = toy_batch(n=3, seed=0)
    pred = direct.direct_predictor(toy_task, ckpt, "cpu")(batch)
    s = toy_task.outputs[0]
    assert set(pred) == {s.name}
    assert pred[s.name].shape == batch.targets[s.name].shape and pred[s.name].dtype == np.float32
    assert np.isfinite(pred[s.name]).all()
    # a standardized output of about unit size is a physical output of about one std around the mean
    assert abs(float(pred[s.name].mean()) - s.mean) < 3 * s.std


def test_predictor_rebuilds_the_model_from_the_config_beside_the_checkpoint(toy_task, toy_batch, tmp_path):
    ckpt = _save(toy_task, tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(
        {"task": toy_task.name, "cfg": {**dataclasses.asdict(TINY), "depth": 3}}))
    with pytest.raises(ValueError, match="config"):
        direct.direct_predictor(toy_task, ckpt, "cpu")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        direct.direct_predictor(toy_task, ckpt, "cpu")


def test_the_predictor_exposes_the_history_the_scorer_must_use(toy_task, tmp_path):
    cfg = dataclasses.replace(TINY, history_s=0.01)   # markovian: the toy task ignores it in n_in
    ckpt = _save(toy_task, tmp_path, cfg)
    assert direct.direct_predictor(toy_task, ckpt, "cpu").history_s == 0.01


# ------------------------------------------------------------------ the trainer, with data access stubbed

@pytest.fixture
def stub_training(monkeypatch, toy_task, toy_batch):
    """Fake loader (four small batches per epoch, 4 shots each) and scorer; returns the call records."""
    record = {"loader": [], "scored": []}

    def fake_train_windows(task, shots=None, **kw):
        record["loader"].append(kw)

        def gen():
            for i in range(4):
                b = toy_batch(n=kw["batch_size"], seed=i)
                b.shot_ids = b.shot_ids + 100 * i
                yield b
        return gen()

    def fake_score(task, predict_fn, out_dir, *, split, shots=None, history_s=None, **kw):
        record["scored"].append({"split": split, "shots": shots, "history_s": history_s, "out_dir": Path(out_dir)})
        pred = predict_fn(toy_batch(n=4, seed=99))["summary-ip"]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        nrmse = float(np.abs(pred - toy_task.outputs[0].mean).mean() / toy_task.outputs[0].std)
        return harness.ScoreResult(
            task=task.name, task_nrmse=nrmse,
            per_signal=pd.DataFrame({"feature_name": ["summary-ip"], "NRMSE": [nrmse]}),
            per_shot=pd.DataFrame({"shot_id": [1], "NRMSE": [nrmse]}), n_windows=4, out_dir=Path(out_dir))

    monkeypatch.setattr(harness, "train_windows", fake_train_windows)
    monkeypatch.setattr(harness, "score", fake_score)
    monkeypatch.setattr(tasks, "split_shots", lambda split: list(range(1000, 1010)))
    return record


def test_train_direct_writes_checkpoints_config_and_log(toy_task, stub_training, tmp_path, capsys):
    cfg = dataclasses.replace(TINY, max_steps=7, val_every=3, warmup=2, val_shots=4)
    out = direct.train_direct(toy_task, cfg, tmp_path / "run", "cpu", num_workers=2)
    assert out == tmp_path / "run" / "best.pt"
    for name in ("best.pt", "last.pt", "config.json", "train_log.csv"):
        assert (tmp_path / "run" / name).exists(), name
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["task"] == toy_task.name and config["cfg"] == dataclasses.asdict(cfg)
    assert "git_revision" in config
    rows = list(csv.DictReader((tmp_path / "run" / "train_log.csv").open()))
    assert list(rows[0]) == ["step", "train_loss", "val_nrmse", "seconds", "windows_per_s", "shots_per_batch"]
    assert [int(r["step"]) for r in rows] == [3, 6, 7]          # every val_every steps and the last step
    assert all(np.isfinite(float(r["val_nrmse"])) and float(r["windows_per_s"]) > 0 for r in rows)
    assert float(rows[0]["shots_per_batch"]) >= 1
    scored = stub_training["scored"]
    assert [s["split"] for s in scored] == ["val_select"] * 3
    assert scored[0]["shots"] == list(range(1000, 1004)) and scored[0]["out_dir"].name == "step_000003"
    assert stub_training["loader"][0]["stride_s"] == 0.005 and stub_training["loader"][0]["shuffle"] is True
    state = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert state["cfg"] == dataclasses.asdict(cfg) and state["step"] in (3, 6, 7)
    assert torch.load(tmp_path / "run" / "last.pt", weights_only=True)["step"] == 7
    # best.pt is the lowest validation
    best_row = min(rows, key=lambda r: float(r["val_nrmse"]))
    assert state["step"] == int(best_row["step"])


def test_train_direct_restarts_the_loader_and_refuses_to_overwrite(toy_task, stub_training, tmp_path):
    cfg = dataclasses.replace(TINY, max_steps=6, val_every=6, warmup=1, batch_size=8)
    direct.train_direct(toy_task, cfg, tmp_path / "run", "cpu", num_workers=2)
    # the fake loader ends after 4 batches; filling the pool and 6 steps takes more than one epoch
    assert len(stub_training["loader"]) >= 2
    with pytest.raises(FileExistsError):
        direct.train_direct(toy_task, cfg, tmp_path / "run", "cpu", num_workers=2)


def test_a_training_batch_mixes_the_shots_of_the_pool(toy_task, stub_training, tmp_path):
    cfg = dataclasses.replace(TINY, max_steps=4, val_every=4, warmup=1, batch_size=8)
    direct.train_direct(toy_task, cfg, tmp_path / "run", "cpu", num_workers=2)
    rows = list(csv.DictReader((tmp_path / "run" / "train_log.csv").open()))
    # one fake loader batch holds 2 distinct shots (4 windows per shot); a pool of 4 batches mixes more
    assert float(rows[-1]["shots_per_batch"]) > 2


def test_train_direct_aborts_on_a_loss_that_is_not_finite(toy_task, stub_training, tmp_path, monkeypatch):
    cfg = dataclasses.replace(TINY, max_steps=30, val_every=30, warmup=1, batch_size=8, lr=float("inf"))
    with pytest.raises(RuntimeError, match="not finite"):
        direct.train_direct(toy_task, cfg, tmp_path / "run", "cpu", num_workers=2)


# ------------------------------------------------------------------ the command

def _run_headroom():
    spec = importlib.util.spec_from_file_location("run_headroom_direct_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = ["--task", "task_1-2", "--predictor", "direct"]
TRAIN = ["--train", "--gpu", "0", "--max-steps", "4", "--d-model", "32", "--depth", "1", "--heads", "4",
         "--train-batch-size", "8", "--val-every", "2", "--num-workers", "2"]


@pytest.fixture
def command(monkeypatch, tmp_path, toy_task, stub_training):
    """run_headroom with data, scorer, load_task, device and nvidia-smi stubbed; returns (module, root)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")           # restored at teardown; main() sets it
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    rh = _run_headroom()
    root = tmp_path / "stage0"
    monkeypatch.setattr(rh.paths, "stage0_dir", lambda: root)
    monkeypatch.setattr(rh.tasks, "load_task", lambda name: dataclasses.replace(toy_task, name=name))
    monkeypatch.setattr(rh, "DEVICE", "cpu")
    rh.smi = {"gpus": "0, GPU-aaa\n1, GPU-bbb\n", "apps": "GPU-bbb, 4242\n"}
    monkeypatch.setattr(rh, "_nvidia_smi", lambda *query: rh.smi["gpus" if "index" in " ".join(query) else "apps"])
    return rh, root


def test_a_busy_gpu_is_found_by_its_uuid():
    rh = _run_headroom()
    gpus = "0, GPU-aaa\n1, GPU-bbb\n3, GPU-ddd\n"
    apps = "GPU-ddd, 29029\nGPU-ddd, 29030\nGPU-bbb, 7\n"
    assert rh.busy_pids(3, gpus, apps) == [29029, 29030]
    assert rh.busy_pids(1, gpus, apps) == [7]
    assert rh.busy_pids(0, gpus, apps) == []
    assert rh.busy_pids(0, gpus, "") == []                    # nothing running: an empty listing


@pytest.mark.parametrize("gpus, apps", [("", ""), ("not csv at all", ""), ("0, GPU-aaa\n", "GPU-aaa, notapid\n"),
                                        ("0, GPU-aaa\n", "garbage\n")])
def test_unparsable_nvidia_smi_output_is_an_error(gpus, apps):
    rh = _run_headroom()
    with pytest.raises(ValueError):
        rh.busy_pids(0, gpus, apps)


def test_a_gpu_that_nvidia_smi_does_not_list_is_an_error():
    rh = _run_headroom()
    with pytest.raises(ValueError, match="no GPU with index 7"):
        rh.busy_pids(7, "0, GPU-aaa\n", "")


def test_the_command_refuses_a_busy_gpu_before_training(command, capsys, monkeypatch):
    rh, root = command
    rh.smi["apps"] = "GPU-aaa, 4242\nGPU-aaa, 4243\n"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 2
    err = capsys.readouterr().err
    assert "4242" in err and "4243" in err
    assert not (root / "direct").exists()


def test_the_command_refuses_when_nvidia_smi_cannot_be_read(command, capsys):
    rh, root = command
    rh.smi["gpus"] = "garbage"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 2
    assert "nvidia-smi" in capsys.readouterr().err
    assert not (root / "direct").exists()


def test_the_command_selects_the_gpu_before_torch_is_used(command, monkeypatch):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_direct_needs_a_gpu_argument(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report"]) == 2
    assert "--gpu" in capsys.readouterr().err


def test_direct_train_and_score_route_the_base_variant(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    run = root / "direct" / "task_1-2" / "base"
    for name in ("best.pt", "last.pt", "config.json", "train_log.csv"):
        assert (run / name).exists(), name
    cfg = json.loads((run / "config.json").read_text())["cfg"]
    assert cfg["max_steps"] == 4 and cfg["d_model"] == 32 and cfg["batch_size"] == 8 and cfg["val_every"] == 2
    score = json.loads((root / "scores" / "task_1-2" / "direct" / "val_report" / "score.json").read_text())
    assert score["predictor"] == "direct"
    assert score["checkpoint_sha256"] == hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    assert score["predictor_sha256"] == direct.PREDICTOR_SHA256


def test_a_variant_has_its_own_directory_and_predictor_name(command):
    rh, root = command
    assert rh.main([*BASE, "--variant", "wide", "--split", "val_report", *TRAIN]) == 0
    assert (root / "direct" / "task_1-2" / "wide" / "best.pt").exists()
    assert not (root / "direct" / "task_1-2" / "base").exists()
    score = json.loads((root / "scores" / "task_1-2" / "direct.wide" / "val_report" / "score.json").read_text())
    assert score["predictor"] == "direct.wide"


@pytest.mark.parametrize("variant", ["base", "wide"])
def test_no_direct_predictor_is_scored_on_the_test_split(command, monkeypatch, capsys, variant):
    rh, root = command

    def forbidden(*a, **k):
        raise AssertionError("data opened")

    monkeypatch.setattr(rh.harness, "score", forbidden)
    monkeypatch.setattr(rh.tasks, "load_task", forbidden)
    assert rh.main([*BASE, "--variant", variant, "--split", "test", "--gpu", "0"]) == 2
    assert "validation" in capsys.readouterr().err


def test_a_missing_checkpoint_says_train_first(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", "--gpu", "0"]) == 2
    assert "train first" in capsys.readouterr().err
    assert not (root / "direct" / "task_1-2" / "base" / "best.pt").exists()


def test_configuration_options_without_train_are_refused(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", "--gpu", "0", "--depth", "3"]) == 2
    assert "--train" in capsys.readouterr().err


def test_a_variant_belongs_to_direct_only(command, capsys):
    rh, root = command
    assert rh.main(["--task", "task_1-2", "--predictor", "floor", "--variant", "x", "--split", "val_report"]) == 2
    assert "--variant" in capsys.readouterr().err


def test_training_refuses_to_overwrite_a_checkpoint_unless_forced(command, capsys):
    rh, root = command
    args = [*BASE, "--split", "val_report", *TRAIN]
    assert rh.main(args) == 0
    first = (root / "direct" / "task_1-2" / "base" / "best.pt").read_bytes()
    assert rh.main(args) == 2
    assert "--force" in capsys.readouterr().err
    assert (root / "direct" / "task_1-2" / "base" / "best.pt").read_bytes() == first
    assert rh.main([*args, "--force"]) == 0


def test_a_stored_direct_score_is_kept_until_the_checkpoint_or_the_code_changes(command, monkeypatch, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    capsys.readouterr()
    scored = []
    real_score = rh.harness.score

    def counting(*a, **k):
        scored.append(1)
        return real_score(*a, **k)

    monkeypatch.setattr(rh.harness, "score", counting)
    score_only = [*BASE, "--split", "val_report", "--gpu", "0"]
    assert rh.main(score_only) == 0
    assert not scored and "kept task_1-2 direct val_report" in capsys.readouterr().out

    best = root / "direct" / "task_1-2" / "base" / "best.pt"
    state = torch.load(best, weights_only=True)
    state["step"] += 1                                       # other bytes, same weights
    torch.save(state, best)
    assert rh.main(score_only) == 0
    assert len(scored) == 1 and "re-scoring task_1-2 direct val_report: checkpoint_sha256 changed" in capsys.readouterr().out
    assert rh.main(score_only) == 0 and len(scored) == 1

    monkeypatch.setattr(direct, "PREDICTOR_SHA256", "0" * 64)
    assert rh.main(score_only) == 0
    assert len(scored) == 2 and "predictor code changed" in capsys.readouterr().out

    assert rh.main([*score_only, "--force"]) == 0 and len(scored) == 3


def test_scoring_uses_the_history_of_the_checkpoints_config(command, stub_training):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--history-s", "0.01"]) == 0
    assert stub_training["scored"][-1]["split"] == "val_report"
    assert stub_training["scored"][-1]["history_s"] == 0.01
