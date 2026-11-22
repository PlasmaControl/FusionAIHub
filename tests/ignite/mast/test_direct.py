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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tokamak_foundation_model.ignite.mast import baselines, harness, tasks  # noqa: E402
from tokamak_foundation_model.ignite.mast.headroom import direct  # noqa: E402

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

def _log_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:      # closed: an unclosed file is a ResourceWarning under -W error
        return list(csv.DictReader(f))


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
    rows = _log_rows(tmp_path / "run" / "train_log.csv")
    assert list(rows[0]) == ["step", "train_loss", "val_nrmse", "seconds", "windows_per_s", "shots_per_batch"]
    assert [int(r["step"]) for r in rows] == [3, 6, 7]          # every val_every steps and the last step
    assert all(np.isfinite(float(r["val_nrmse"])) and float(r["windows_per_s"]) > 0 for r in rows)
    assert float(rows[0]["shots_per_batch"]) >= 1
    scored = stub_training["scored"]
    assert [s["split"] for s in scored] == ["val_select"] * 3
    # the stub's val_select is 1000..1009: four shots spread over it, first and last included (not the first four)
    assert scored[0]["shots"] == [1000, 1003, 1006, 1009] and scored[0]["out_dir"].name == "step_000003"
    assert all(s["shots"] == direct.validation_shots(4) for s in scored)
    assert stub_training["loader"][0]["stride_s"] == 0.005 and stub_training["loader"][0]["shuffle"] is True
    state = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    assert state["cfg"] == dataclasses.asdict(cfg) and state["step"] in (3, 6, 7)
    assert torch.load(tmp_path / "run" / "last.pt", weights_only=True)["step"] == 7
    # best.pt is the lowest validation
    best_row = min(rows, key=lambda r: float(r["val_nrmse"]))
    assert state["step"] == int(best_row["step"])


def test_validation_shots_are_spread_over_the_sorted_list(monkeypatch):
    shots = [7000 + 3 * i for i in range(1000)]                       # ascending, like split_shots
    monkeypatch.setattr(tasks, "split_shots", lambda split: shots if split == "val_select" else [])
    assert direct.validation_shots(4) == [shots[0], shots[333], shots[666], shots[999]]
    chosen = direct.validation_shots(200)
    assert len(chosen) == len(set(chosen)) == 200 and chosen == sorted(chosen)
    assert chosen[0] == shots[0] and chosen[-1] == shots[-1]
    assert direct.validation_shots(1) == [shots[0]]


@pytest.mark.parametrize("val_shots", [10, 11, 5000])
def test_validation_shots_with_enough_budget_are_every_shot_once_in_order(monkeypatch, val_shots):
    shots = list(range(100, 110))
    monkeypatch.setattr(tasks, "split_shots", lambda split: shots)
    assert direct.validation_shots(val_shots) == shots


def test_validation_shots_are_deterministic_and_come_from_val_select(monkeypatch):
    asked = []
    monkeypatch.setattr(tasks, "split_shots", lambda split: asked.append(split) or list(range(50)))
    assert direct.validation_shots(7) == direct.validation_shots(7)
    assert set(asked) == {"val_select"}


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
    rows = _log_rows(tmp_path / "run" / "train_log.csv")
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
    monkeypatch.setattr(rh, "_cuda_initialized", lambda: False)   # whatever the pytest process did to CUDA
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


def test_the_command_selects_the_gpu_by_the_uuid_nvidia_smi_prints_for_its_index(command):
    rh, root = command
    rh.smi["apps"] = ""
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-aaa"      # the form nvidia-smi prints, prefix included
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--gpu", "1", "--variant", "other"]) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-bbb"      # index 1, not index 0 or the number 1


def test_one_nvidia_smi_lookup_serves_the_selection_and_the_busy_check(command):
    rh, root = command
    rh.smi["apps"] = ""
    calls = []
    real = rh._nvidia_smi
    rh._nvidia_smi = lambda *q: (calls.append(q), real(*q))[1]
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    assert len(calls) == 2                                      # the GPU listing and the compute apps, once each


def test_a_gpu_index_nvidia_smi_does_not_list_is_refused(command, capsys, monkeypatch):
    rh, root = command
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unchanged")
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--gpu", "7"]) == 2
    assert "no GPU with index 7" in capsys.readouterr().err
    assert not (root / "direct").exists()


def test_the_command_refuses_when_cuda_is_already_initialized(command, capsys, monkeypatch):
    rh, root = command
    monkeypatch.setattr(rh, "_cuda_initialized", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unchanged")

    def forbidden(*a, **k):
        raise AssertionError("opened")

    monkeypatch.setattr(rh, "_nvidia_smi", forbidden)
    monkeypatch.setattr(rh.tasks, "load_task", forbidden)
    monkeypatch.setattr(rh.harness, "score", forbidden)
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 2
    assert "CUDA is already initialized" in capsys.readouterr().err
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "unchanged"
    assert not (root / "direct").exists()


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


def test_scoring_uses_the_history_of_the_checkpoints_config(command, stub_training, toy_task, monkeypatch):
    """A non-markovian task: --history-s is recorded in the checkpoint's config and the scorer cuts with it. The
    fake loader delivers windows of the task's own length, so the history asked for equals it (0.005 s)."""
    rh, root = command
    monkeypatch.setattr(rh.tasks, "load_task", lambda name: dataclasses.replace(toy_task, name=name, markovian=False))
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--history-s", "0.005"]) == 0
    assert stub_training["scored"][-1]["split"] == "val_report"
    assert stub_training["scored"][-1]["history_s"] == 0.005


# ------------------------------------------------------------------ scores follow the checkpoint's life

def _score_json(root, name="direct"):
    return root / "scores" / "task_1-2" / name / "val_report" / "score.json"


def test_a_score_records_the_steps_of_a_complete_run(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    record = json.loads(_score_json(root).read_text())
    assert (record["checkpoint_step"], record["trained_steps"], record["max_steps"]) == (4, 4, 4)
    assert "warning" not in capsys.readouterr().err


def test_a_run_that_stopped_early_is_scored_with_a_warning_and_its_steps(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--max-steps", "6"]) == 0
    run = root / "direct" / "task_1-2" / "base"
    # as a run that died after its second validation leaves the files: best.pt and last.pt of step 2, max_steps 6
    for name in ("best.pt", "last.pt"):
        state = torch.load(run / name, weights_only=True)
        state["step"] = 2
        torch.save(state, run / name)
    capsys.readouterr()
    assert rh.main([*BASE, "--split", "val_report", "--gpu", "0"]) == 0
    record = json.loads(_score_json(root).read_text())
    assert (record["checkpoint_step"], record["trained_steps"], record["max_steps"]) == (2, 2, 6)
    err = capsys.readouterr().err
    assert err.count("warning") == 1 and "2 of 6" in err


def test_a_run_whose_best_is_older_than_its_last_validation_records_both(command):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    run = root / "direct" / "task_1-2" / "base"
    state = torch.load(run / "best.pt", weights_only=True)
    state["step"] = 2
    torch.save(state, run / "best.pt")
    assert rh.main([*BASE, "--split", "val_report", "--gpu", "0"]) == 0
    record = json.loads(_score_json(root).read_text())
    assert (record["checkpoint_step"], record["trained_steps"], record["max_steps"]) == (2, 4, 4)


def test_a_run_without_last_checkpoint_has_null_trained_steps_and_a_warning(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    (root / "direct" / "task_1-2" / "base" / "last.pt").unlink()
    capsys.readouterr()
    assert rh.main([*BASE, "--split", "val_report", "--gpu", "0", "--force"]) == 0
    assert json.loads(_score_json(root).read_text())["trained_steps"] is None
    assert capsys.readouterr().err.count("warning") == 1


def test_the_steps_are_records_and_not_provenance(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    path = _score_json(root)
    record = json.loads(path.read_text())
    for key in ("checkpoint_step", "trained_steps", "max_steps"):
        record[key] = 999
    path.write_text(json.dumps(record))
    capsys.readouterr()
    assert rh.main([*BASE, "--split", "val_report", "--gpu", "0"]) == 0
    assert "kept task_1-2 direct val_report" in capsys.readouterr().out


def test_a_forced_retraining_removes_the_stored_score_before_it_trains(command, monkeypatch):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    assert _score_json(root).exists()

    def dies(*a, **k):
        raise RuntimeError("the training died")

    monkeypatch.setattr(direct, "train_direct", dies)
    with pytest.raises(RuntimeError, match="died"):
        rh.main([*BASE, "--split", "val_report", *TRAIN, "--force"])
    assert not _score_json(root).parent.exists()                  # no number for a checkpoint that is gone
    assert not (root / "direct" / "task_1-2" / "base" / "best.pt").exists()


def test_a_busy_gpu_is_refused_before_anything_is_deleted(command):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    rh.smi["apps"] = "GPU-aaa, 4242\n"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--force"]) == 2
    assert _score_json(root).exists() and (root / "direct" / "task_1-2" / "base" / "best.pt").exists()


def test_a_forced_retraining_of_one_variant_leaves_the_other_variants_scores(command):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    assert rh.main([*BASE, "--variant", "wide", "--split", "val_report", *TRAIN]) == 0
    assert rh.main([*BASE, "--variant", "wide", "--split", "val_report", *TRAIN, "--force"]) == 0
    assert _score_json(root).exists() and _score_json(root, "direct.wide").exists()


# ------------------------------------------------------------------ --share-gpu

def test_share_gpu_trains_on_a_busy_gpu_and_names_the_pids(command, capsys):
    rh, root = command
    rh.smi["apps"] = "GPU-aaa, 4242\nGPU-aaa, 4243\n"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--share-gpu"]) == 0
    out = capsys.readouterr().out
    assert "sharing GPU 0 with compute processes (pids 4242, 4243)" in out
    assert (root / "direct" / "task_1-2" / "base" / "best.pt").exists()
    assert _score_json(root).exists()


def test_share_gpu_prints_nothing_extra_on_a_free_gpu(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--share-gpu"]) == 0       # GPU 0 is free in the fixture
    assert "sharing" not in capsys.readouterr().out


def test_share_gpu_scores_on_a_busy_gpu_without_training(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    rh.smi["apps"] = "GPU-aaa, 777\n"
    score_again = [*BASE, "--split", "val_report", "--gpu", "0", "--force"]
    capsys.readouterr()
    assert rh.main(score_again) == 2                                   # the default did not move
    assert "777" in capsys.readouterr().err
    assert rh.main([*score_again, "--share-gpu"]) == 0
    assert "sharing GPU 0 with compute processes (pids 777)" in capsys.readouterr().out


def test_share_gpu_still_refuses_when_nvidia_smi_cannot_be_read(command, capsys):
    rh, root = command
    rh.smi["gpus"] = "garbage"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--share-gpu"]) == 2
    assert "nvidia-smi" in capsys.readouterr().err
    assert not (root / "direct").exists()


def test_share_gpu_still_refuses_an_index_nvidia_smi_does_not_list(command, capsys):
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--gpu", "7", "--share-gpu"]) == 2
    assert "no GPU with index 7" in capsys.readouterr().err


def test_share_gpu_still_refuses_when_cuda_is_already_initialized(command, capsys, monkeypatch):
    rh, root = command
    monkeypatch.setattr(rh, "_cuda_initialized", lambda: True)
    rh.smi["apps"] = "GPU-aaa, 4242\n"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--share-gpu"]) == 2
    assert "CUDA is already initialized" in capsys.readouterr().err
    assert not (root / "direct").exists()


def test_share_gpu_does_not_delete_before_the_decision_is_made(command):
    """With --share-gpu a forced retraining proceeds as usual (deletes, trains); the default path keeps refusing first."""
    rh, root = command
    assert rh.main([*BASE, "--split", "val_report", *TRAIN]) == 0
    rh.smi["apps"] = "GPU-aaa, 4242\n"
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--force"]) == 2
    assert _score_json(root).exists()
    assert rh.main([*BASE, "--split", "val_report", *TRAIN, "--force", "--share-gpu"]) == 0


@pytest.mark.parametrize("predictor", ["floor", "shrunk", "ridge"])
def test_share_gpu_belongs_to_direct_only(command, capsys, predictor):
    rh, root = command
    assert rh.main(["--task", "task_1-2", "--predictor", predictor, "--split", "val_report", "--share-gpu"]) == 2
    assert "--share-gpu" in capsys.readouterr().err



# ------------------------------------------------------------------ final fix wave: C2

def _run(root, variant="base"):
    return root / "direct" / "task_1-2" / variant


def _sidecar(root, variant="base"):
    return json.loads((_run(root, variant) / "trained_by.json").read_text())


def _the_code_now() -> dict:
    return {"predictor_sha256": direct.PREDICTOR_SHA256, "code_sha256": baselines.scoring_code_sha256()}


SCORE_ONLY = [*BASE, "--split", "val_report", "--gpu", "0"]
TRAIN_ARGV = [*BASE, "--split", "val_report", *TRAIN]


# ---- C2-I1: the checkpoint is tied to the code that trained it

def test_the_training_leaves_a_sidecar_with_the_code_that_trained_it(command):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    assert _sidecar(root) == _the_code_now()
    assert set(_sidecar(root)) == {"predictor_sha256", "code_sha256"}
    assert not list(_run(root).glob("*.tmp"))


def test_the_sidecar_exists_before_the_first_training_step_and_survives_a_stopped_training(command, monkeypatch):
    rh, root = command
    seen = {}

    def dies(task, cfg, run_dir, device, **kw):
        seen["at_start"] = json.loads((Path(run_dir) / "trained_by.json").read_text())
        raise RuntimeError("the training died")

    monkeypatch.setattr(direct, "train_direct", dies)
    with pytest.raises(RuntimeError, match="died"):
        rh.main(TRAIN_ARGV)
    assert seen["at_start"] == _the_code_now() == _sidecar(root)


def test_a_forced_retraining_replaces_the_sidecar(command, monkeypatch):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    monkeypatch.setattr(direct, "PREDICTOR_SHA256", "1" * 64)
    assert rh.main([*TRAIN_ARGV, "--force"]) == 0
    assert _sidecar(root)["predictor_sha256"] == "1" * 64


@pytest.mark.parametrize("sidecar", ["absent", "not json", "wrong keys", "wrong type"])
def test_the_score_records_the_code_that_trained_the_checkpoint_or_null(command, sidecar):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    record = json.loads(_score_json(root).read_text())
    assert record["trained_predictor_sha256"] == direct.PREDICTOR_SHA256
    assert record["trained_code_sha256"] == baselines.scoring_code_sha256()
    path = _run(root) / "trained_by.json"
    if sidecar == "absent":
        path.unlink()
    else:
        path.write_text({"not json": "{not json", "wrong keys": json.dumps({"predictor_sha256": "a"}),
                         "wrong type": json.dumps({"predictor_sha256": 1, "code_sha256": 2})}[sidecar])
    assert rh.main([*SCORE_ONLY, "--force"]) == 0
    record = json.loads(_score_json(root).read_text())
    assert record["trained_predictor_sha256"] is None and record["trained_code_sha256"] is None


def test_the_trained_by_hashes_are_a_mark_and_not_part_of_the_rule_that_keeps_a_score(command, capsys):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    assert rh.stale_reason(root, "task_1-2", "direct", "val_report") == ""
    capsys.readouterr()

    (_run(root) / "trained_by.json").write_text(json.dumps({**_the_code_now(), "predictor_sha256": "0" * 64}))
    assert rh.main(SCORE_ONLY) == 0                                 # scoring again cannot cure a training by other code
    assert "kept task_1-2 direct val_report" in capsys.readouterr().out
    assert json.loads(_score_json(root).read_text())["trained_predictor_sha256"] == "0" * 64   # the mark follows the run
    assert rh.stale_reason(root, "task_1-2", "direct", "val_report") == \
        "trained by other code than the code now loaded"

    (_run(root) / "trained_by.json").write_text(json.dumps({**_the_code_now(), "code_sha256": "0" * 64}))
    assert rh.main(SCORE_ONLY) == 0
    assert rh.stale_reason(root, "task_1-2", "direct", "val_report") == \
        "trained by other code than the code now loaded"

    (_run(root) / "trained_by.json").unlink()
    assert rh.main(SCORE_ONLY) == 0
    assert rh.stale_reason(root, "task_1-2", "direct", "val_report") == "training code not recorded"

    (_run(root) / "trained_by.json").write_text(json.dumps(_the_code_now()))
    assert rh.main(SCORE_ONLY) == 0
    assert rh.stale_reason(root, "task_1-2", "direct", "val_report") == ""


def test_the_table_command_marks_a_direct_score_by_the_code_that_trained_it(command, capsys):
    rh, root = command

    def notes():
        capsys.readouterr()
        assert rh.main(["--table"]) == 0
        return [line for line in capsys.readouterr().out.splitlines() if line.startswith("- 1-2 direct_val")]

    assert rh.main(TRAIN_ARGV) == 0
    assert notes() == []                                            # trained by the code now loaded: no mark
    (_run(root) / "trained_by.json").write_text(json.dumps({**_the_code_now(), "code_sha256": "0" * 64}))
    assert rh.main(SCORE_ONLY) == 0                                 # kept; the record follows the sidecar
    assert notes() == ["- 1-2 direct_val: trained by other code than the code now loaded (stale)"]
    (_run(root) / "trained_by.json").unlink()
    assert rh.main(SCORE_ONLY) == 0
    assert notes() == ["- 1-2 direct_val: training code not recorded (stale)"]


def test_a_score_from_before_the_sidecar_has_no_training_code_recorded(command):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    path = _score_json(root)
    record = json.loads(path.read_text())
    del record["trained_predictor_sha256"], record["trained_code_sha256"]
    path.write_text(json.dumps(record))
    assert rh.stale_reason(root, "task_1-2", "direct", "val_report") == "training code not recorded"


def test_a_stale_score_keeps_its_own_reason_before_the_trained_by_one(command):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    (_run(root) / "trained_by.json").unlink()
    best = _run(root) / "best.pt"
    state = torch.load(best, weights_only=True)
    state["step"] += 1
    torch.save(state, best)
    assert "checkpoint_sha256 changed since" in rh.stale_reason(root, "task_1-2", "direct", "val_report")


@pytest.mark.parametrize("variant, name", [("base", "direct"), ("wide", "direct.wide")])
def test_stale_reason_of_a_direct_checkpoint_follows_its_bytes(command, variant, name):
    rh, root = command
    assert rh.main([*TRAIN_ARGV, "--variant", variant]) == 0
    assert rh.stale_reason(root, "task_1-2", name, "val_report") == ""
    best = _run(root, variant) / "best.pt"
    state = torch.load(best, weights_only=True)
    state["step"] += 1                                                # other bytes
    torch.save(state, best)
    assert rh.stale_reason(root, "task_1-2", name, "val_report").startswith("checkpoint_sha256 changed since")


# ---- C2-M1: a kept score refreshes what the files say about the run

def test_a_kept_direct_score_refreshes_the_three_run_keys(command, capsys):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    path = _score_json(root)
    record = json.loads(path.read_text())
    record.update(checkpoint_step=2, trained_steps=2, max_steps=4)     # a score taken in the middle of the training
    path.write_text(json.dumps(record))
    capsys.readouterr()
    assert rh.main(SCORE_ONLY) == 0
    out = capsys.readouterr().out
    assert "kept task_1-2 direct val_report" in out and "refreshed" in out
    refreshed = json.loads(path.read_text())
    assert (refreshed["checkpoint_step"], refreshed["trained_steps"], refreshed["max_steps"]) == (4, 4, 4)
    assert {k: v for k, v in refreshed.items() if not k.startswith(("trained_", "checkpoint_step"))} == \
        {k: v for k, v in record.items() if not k.startswith(("trained_", "checkpoint_step"))}   # nothing else moved
    assert not list(path.parent.glob("*.tmp"))
    capsys.readouterr()
    assert rh.main(SCORE_ONLY) == 0 and "refreshed" not in capsys.readouterr().out        # nothing to refresh again


def test_a_kept_score_whose_run_cannot_be_read_is_still_kept(command, capsys):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    (_run(root) / "config.json").unlink()                              # the record keys cannot be refreshed
    capsys.readouterr()
    assert rh.main(SCORE_ONLY) == 0
    captured = capsys.readouterr()
    assert "kept task_1-2 direct val_report" in captured.out and "cannot refresh" in captured.err


# ---- C2-M2: one checkpoint for the scores of one call

def test_a_checkpoint_that_changes_while_the_predictor_is_built_is_refused(command, monkeypatch, capsys):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    before = _score_json(root).read_text()
    build, code, provenance = rh.PREDICTORS["direct"]

    def build_then_replace(ctx):
        predictor = build(ctx)
        best = _run(root) / "best.pt"
        state = torch.load(best, weights_only=True)
        state["step"] += 1                                             # a concurrent writer
        torch.save(state, best)
        return predictor

    monkeypatch.setitem(rh.PREDICTORS, "direct", (build_then_replace, code, provenance))
    scored = []
    monkeypatch.setattr(rh.harness, "score", lambda *a, **k: scored.append(1))
    capsys.readouterr()
    assert rh.main([*SCORE_ONLY, "--force"]) == 1
    assert "changed while" in capsys.readouterr().err and not scored
    assert _score_json(root).read_text() == before


# ---- C2-M4: the configuration is validated before anything is deleted

@pytest.mark.parametrize("bad", [["--heads", "3"], ["--max-steps", "0"], ["--val-every", "0"], ["--lr", "0"]])
def test_an_invalid_configuration_is_refused_before_a_forced_retraining_deletes_anything(command, capsys, bad):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    files = {n: (_run(root) / n).read_bytes() for n in ("best.pt", "last.pt", "trained_by.json", "config.json")}
    score = _score_json(root).read_text()
    capsys.readouterr()
    assert rh.main([*TRAIN_ARGV, "--force", *bad]) == 2
    err = capsys.readouterr().err
    assert "refused" in err and bad[0].lstrip("-").replace("-", "_") in err      # it names the field
    assert {n: (_run(root) / n).read_bytes() for n in files} == files
    assert _score_json(root).read_text() == score
    assert not (_run(root) / "train.lock").exists()


def test_an_invalid_configuration_creates_no_run_directory(command, capsys):
    rh, root = command
    assert rh.main([*TRAIN_ARGV, "--variant", "bad", "--heads", "3"]) == 2
    assert "heads" in capsys.readouterr().err
    assert not _run(root, "bad").exists()


def test_the_help_says_what_force_means_with_and_without_train(command, capsys):
    rh, root = command
    with pytest.raises(SystemExit):
        rh.main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "--force" in text and "without --train: score again" in text and "With `direct --train`: train again" in text


# ---- C2-M5: a score that cannot be removed stops the command

def test_a_stored_score_that_cannot_be_removed_stops_the_command(command, monkeypatch, capsys):
    import types
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    best = (_run(root) / "best.pt").read_bytes()
    score = _score_json(root).read_text()

    def denied(path, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(rh, "shutil", types.SimpleNamespace(rmtree=denied))
    trained = []
    monkeypatch.setattr(direct, "train_direct", lambda *a, **k: trained.append(1))
    capsys.readouterr()
    assert rh.main([*TRAIN_ARGV, "--force"]) == 2
    err = capsys.readouterr().err
    assert "cannot remove" in err and str(_score_json(root).parent) in err
    assert not trained and (_run(root) / "best.pt").read_bytes() == best      # nothing was deleted or trained
    assert _score_json(root).read_text() == score
    assert not (_run(root) / "train.lock").exists()


# ---- C2-M7 and the guard of A-M1: --history-s

def _non_markovian(rh, toy_task, monkeypatch, **changes):
    monkeypatch.setattr(rh.tasks, "load_task",
                        lambda name: dataclasses.replace(toy_task, name=name, markovian=False, **changes))


def test_history_is_refused_for_a_markovian_task_before_anything_is_written(command, capsys):
    rh, root = command
    assert rh.main([*TRAIN_ARGV, "--history-s", "0.01"]) == 2
    err = capsys.readouterr().err
    assert "--history-s" in err and "markovian" in err and "task_1-2" in err
    assert not root.exists()


@pytest.mark.parametrize("history", ["0.0001", "0", "-0.01", "nan"])
def test_a_history_shorter_than_half_a_sample_is_refused_naming_the_signal(command, toy_task, monkeypatch, capsys,
                                                                          history):
    rh, root = command
    _non_markovian(rh, toy_task, monkeypatch)
    assert rh.main([*TRAIN_ARGV, "--history-s", history]) == 2
    err = capsys.readouterr().err
    assert "--history-s" in err and "summary-ip" in err and "magnetics-flux_loop_flux" in err
    assert not root.exists()                                              # nothing was written


def test_the_guard_names_an_actuator_that_is_too_coarse_for_the_history(command, toy_task, monkeypatch, capsys):
    rh, root = command
    coarse = dataclasses.replace(toy_task.actuators[0], dt=0.01)           # 10 ms samples: 4 ms history rounds to 0
    _non_markovian(rh, toy_task, monkeypatch, actuators=(coarse,))
    assert rh.main([*TRAIN_ARGV, "--history-s", "0.004"]) == 2
    err = capsys.readouterr().err
    assert "pf_active-coil_voltage" in err and "summary-ip" not in err
    assert not root.exists()


def test_a_history_of_at_least_half_a_sample_is_accepted(command, toy_task, monkeypatch):
    rh, root = command
    _non_markovian(rh, toy_task, monkeypatch)
    assert rh.main([*TRAIN_ARGV, "--history-s", "0.005"]) == 0              # round(0.005 / dt) >= 1 for every signal


# ---- L278: one training of a run at a time

def _alive_pid_not_ours():
    return os.getppid()


def test_the_lock_is_held_during_the_training_and_removed_after_it(command, monkeypatch):
    rh, root = command
    seen = {}
    real = direct.train_direct

    def checking(task, cfg, run_dir, device, **kw):
        seen["lock"] = (Path(run_dir) / "train.lock").read_text().strip()
        seen["checkpoint_looks_present"] = (Path(run_dir) / "best.pt").exists()
        return real(task, cfg, run_dir, device, **kw)

    monkeypatch.setattr(direct, "train_direct", checking)
    assert rh.main(TRAIN_ARGV) == 0
    assert seen == {"lock": str(os.getpid()), "checkpoint_looks_present": False}   # a lock is not a checkpoint
    assert not (_run(root) / "train.lock").exists()


def test_the_lock_is_removed_when_the_training_fails(command, monkeypatch):
    rh, root = command

    def dies(*a, **k):
        raise RuntimeError("the training died")

    monkeypatch.setattr(direct, "train_direct", dies)
    with pytest.raises(RuntimeError):
        rh.main(TRAIN_ARGV)
    assert not (_run(root) / "train.lock").exists()


def test_a_second_launch_of_a_run_that_is_training_is_refused_naming_the_pid(command, monkeypatch, capsys):
    rh, root = command
    pid = _alive_pid_not_ours()
    _run(root).mkdir(parents=True)
    (_run(root) / "train.lock").write_text(f"{pid}\n")
    trained = []
    monkeypatch.setattr(direct, "train_direct", lambda *a, **k: trained.append(1))
    assert rh.main(TRAIN_ARGV) == 2
    err = capsys.readouterr().err
    assert f"pid {pid}" in err and "train.lock" in err and not trained
    assert (_run(root) / "train.lock").read_text().strip() == str(pid)    # the other training's lock is untouched
    assert not (_run(root) / "best.pt").exists()


def test_a_refused_launch_deletes_nothing(command, capsys):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    best, score = (_run(root) / "best.pt").read_bytes(), _score_json(root).read_text()
    (_run(root) / "train.lock").write_text(f"{_alive_pid_not_ours()}\n")
    assert rh.main([*TRAIN_ARGV, "--force"]) == 2
    assert (_run(root) / "best.pt").read_bytes() == best and _score_json(root).read_text() == score


def test_a_lock_of_a_pid_that_is_not_running_is_taken_over_with_one_line(command, capsys):
    import subprocess
    rh, root = command
    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True,
                          check=True)
    dead = int(done.stdout)
    _run(root).mkdir(parents=True)
    (_run(root) / "train.lock").write_text(f"{dead}\n")
    capsys.readouterr()
    assert rh.main(TRAIN_ARGV) == 0
    out = capsys.readouterr().out
    assert out.count("taking over") == 1 and f"pid {dead}" in out
    assert not (_run(root) / "train.lock").exists() and (_run(root) / "best.pt").exists()


def test_a_lock_without_a_readable_pid_is_not_taken_over(command, capsys):
    rh, root = command
    _run(root).mkdir(parents=True)
    (_run(root) / "train.lock").write_text("")
    assert rh.main(TRAIN_ARGV) == 2
    assert "no readable pid" in capsys.readouterr().err and (_run(root) / "train.lock").exists()


def test_the_old_validation_directories_are_left_alone(command):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    leftovers = sorted(p.name for p in (_run(root) / "val").iterdir())
    assert leftovers and rh.main([*TRAIN_ARGV, "--force"]) == 0
    assert set(leftovers) <= {p.name for p in (_run(root) / "val").iterdir()}


# ---- C2-M9: the second write of score.json is atomic

def test_the_second_write_of_score_json_is_atomic(command, monkeypatch):
    rh, root = command
    real_replace = os.replace

    def failing(src, dst):
        if str(dst).endswith("score.json"):
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(rh.os, "replace", failing)
    with pytest.raises(OSError, match="disk full"):
        rh.main(TRAIN_ARGV)
    path = _score_json(root)
    first = json.loads(path.read_text())                                   # the first write is whole, not truncated
    assert first["predictor"] == "direct" and "predictor_sha256" not in first
    assert not list(path.parent.glob("*.tmp*"))
    monkeypatch.setattr(rh.os, "replace", real_replace)


# ---- A-R3: a fitted predictor never sees the targets

def test_blinding_gives_a_predictor_that_reads_its_targets_nan_arrays_of_the_same_shape(command, toy_batch):
    rh, root = command
    batch = toy_batch(n=3, seed=0)
    original = {k: v.copy() for k, v in batch.targets.items()}

    def peeking(b):
        return {k: v.copy() for k, v in b.targets.items()}

    peeking.history_s = 0.02
    peeking.other = "kept"
    out = rh._blind_to_targets(peeking)(batch)
    assert set(out) == set(original)
    for k, v in out.items():
        assert v.shape == original[k].shape and v.dtype == original[k].dtype and np.isnan(v).all()
    for k, v in batch.targets.items():
        np.testing.assert_array_equal(v, original[k])                      # the harness still has the real targets
    blind = rh._blind_to_targets(peeking)
    assert blind.history_s == 0.02 and blind.other == "kept"
    assert not hasattr(rh._blind_to_targets(lambda b: {}), "history_s")


def test_blinding_a_bound_method_keeps_its_history_attribute(command, toy_task, tmp_path):
    rh, root = command
    ckpt = _save(toy_task, tmp_path, dataclasses.replace(TINY, history_s=0.005))
    predict = direct.direct_predictor(toy_task, ckpt, "cpu")
    assert rh._blind_to_targets(predict).history_s == 0.005


def test_the_direct_predictor_gives_the_same_output_blinded_or_not(command, toy_task, toy_batch, tmp_path):
    rh, root = command
    ckpt = _save(toy_task, tmp_path)
    predict = direct.direct_predictor(toy_task, ckpt, "cpu")
    batch = toy_batch(n=4, seed=5)
    np.testing.assert_array_equal(rh._blind_to_targets(predict)(batch)["summary-ip"], predict(batch)["summary-ip"])
    np.testing.assert_array_equal(predict(dataclasses.replace(batch, targets={}))["summary-ip"],
                                  predict(batch)["summary-ip"])           # L275: predictions do not read the targets


def test_the_command_scores_a_direct_model_through_the_blind_wrapper(command, monkeypatch):
    rh, root = command
    assert rh.main(TRAIN_ARGV) == 0
    seen = []
    real = direct._inputs_to_tensors
    monkeypatch.setattr(direct, "_inputs_to_tensors",
                        lambda batch, *a, **k: (seen.append([bool(np.isnan(v).all()) for v in batch.targets.values()]),
                                                real(batch, *a, **k))[1])
    assert rh.main([*SCORE_ONLY, "--force"]) == 0
    assert seen and all(all(flags) for flags in seen)                      # every call during scoring saw NaN targets


# ---- L275: what is validated and saved is the EMA

def test_the_ema_weights_are_what_is_validated_and_saved(toy_task, stub_training, tmp_path, monkeypatch):
    cfg = dataclasses.replace(TINY, max_steps=6, val_every=3, warmup=2, val_shots=4, batch_size=8)
    validated = []
    real = direct._predict_fn

    def spy(model, task, cfg_, device):
        validated.append((model, {k: v.detach().clone() for k, v in model.state_dict().items()},
                          [p.requires_grad for p in model.parameters()]))
        return real(model, task, cfg_, device)

    monkeypatch.setattr(direct, "_predict_fn", spy)
    direct.train_direct(toy_task, cfg, tmp_path / "run", "cpu", num_workers=2)
    assert len(validated) == 2
    for _, _, grads in validated:
        assert not any(grads)                                              # the EMA copy: it is never trained
    saved = torch.load(tmp_path / "run" / "last.pt", weights_only=True)
    last = validated[-1][1]
    assert set(saved["state"]) == set(last) and all(torch.equal(saved["state"][k], last[k]) for k in last)
    best = torch.load(tmp_path / "run" / "best.pt", weights_only=True)
    by_step = {3: validated[0][1], 6: validated[1][1]}[best["step"]]
    assert all(torch.equal(best["state"][k], by_step[k]) for k in by_step)
    # and it is not the raw weights: the EMA of 6 steps is still near the initial weights, the raw ones moved
    assert not all(torch.equal(validated[0][1][k], validated[1][1][k]) for k in last)
