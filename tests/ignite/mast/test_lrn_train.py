"""Examples for and the training of a network recoder (plan 1c)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import registry
from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, nets, recode, store, train

RNG = np.random.default_rng(0)
TABLE = identity.StreamTable({"m": (0, 3)}, RNG.standard_normal((3, identity.N_FEAT)).astype(np.float32))
CONFIG = dict(d_in=16, n_values=8, n_streams=3, d_model=64, depth=1, p_drop_ident=0.1, concat=False, n_tok=1, n_val=8)
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "mast"


def test_examples_per_stream_and_concatenated():
    values = RNG.standard_normal((11, 3, 4)).astype(np.float32)
    values[0, 1] = np.nan                                  # one stream missing in one frame
    values[10] = np.nan                                    # a frame without any stream
    stores = {"m": store.Store("m", values, np.zeros(11, dtype=np.int64))}
    norms = {"m": store.Norm(np.zeros((3, 4)), np.full(3, 2.0))}
    e, mask, ident = train.examples(stores, norms, TABLE, concat=False)
    assert e.shape == (29, 4) and mask.all() and np.isfinite(e).all() and e.dtype == np.float32
    assert np.bincount(ident).tolist() == [10, 9, 10] and np.allclose(e[0], values[0, 0] / 2.0)
    e, mask, ident = train.examples(stores, norms, TABLE, concat=True)
    assert e.shape == (10, 12) and mask.sum() == 116 and (ident == -1).all()
    assert not mask[0, 4:8].any() and (e[0, 4:8] == 0).all() and np.allclose(e[1, 4:8], values[1, 1] / 2.0)
    with pytest.raises(ValueError):
        train.examples({**stores, "n": stores["m"]}, {**norms, "n": norms["m"]}, TABLE, concat=True)
    assert train.prefix_lengths(16) == [1, 2, 4, 8, 16] and train.prefix_lengths(24) == [1, 2, 4, 8, 16, 24]


def rank_two(m):
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((16, 16)))
    e = (RNG.standard_normal((m, 2)) * [1.0, 0.5]) @ q[:, :2].T + 0.01 * RNG.standard_normal((m, 16))
    e = (e / 0.28).astype(np.float32)                      # about unit variance per entry
    return e, np.ones_like(e, dtype=bool), RNG.integers(0, 3, m)


def prefix_error(recoder, e, ident, k):
    v = recoder.encode(e, ident)
    v[:, k:] = 0
    return float(np.mean((recoder.decode(v, ident) - e) ** 2))


def test_a_short_training_learns_a_low_rank_signal_and_keeps_its_best_weights(tmp_path):
    cfg = train.TrainConfig(d_model=64, depth=1, steps=400, batch=256, lr=1e-3, warmup=20, val_every=100)
    train_ex, val_ex = rank_two(4000), rank_two(1000)
    res = train.train(CONFIG, cfg, train_ex, val_ex, TABLE, tmp_path, log=lambda line: None)
    variance = float(np.mean(val_ex[0] ** 2))
    assert res["best_val_mse"] < 0.5 * variance and res["steps"] == 400
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert log["step"].tolist() == [100, 200, 300, 400] and list(log.columns) == ["step", "train_loss", "val_mse", "val_full"]
    assert res["best_step"] == int(log.loc[log["val_mse"].idxmin(), "step"])
    r = nets.load_net_recoder(tmp_path, TABLE)
    assert prefix_error(r, val_ex[0], val_ex[2], 8) == pytest.approx(res["best_val_full"], rel=1e-3)
    assert prefix_error(r, val_ex[0], val_ex[2], 1) > prefix_error(r, val_ex[0], val_ex[2], 8)   # a prefix is coarser


def test_a_loss_that_is_not_finite_stops_the_training(tmp_path):
    e, mask, ident = rank_two(500)
    cfg = train.TrainConfig(d_model=64, depth=1, steps=50, batch=64, val_every=50)
    with pytest.raises(FloatingPointError):
        train.train(CONFIG, cfg, (np.full_like(e, np.inf), mask, ident), (e, mask, ident), TABLE, tmp_path,
                    log=lambda line: None)


def test_plateau_reads_the_last_quarter():
    def log(values):
        return pd.DataFrame({"step": range(1, len(values) + 1), "val_mse": values})
    assert train.plateau(log([1.0, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.399]))
    assert not train.plateau(log([1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33]))
    assert not train.plateau(log([1.0, 0.5, 0.4]))


def test_every_optimizer_step_runs_in_training_mode_also_after_a_validation(tmp_path, monkeypatch):
    """Validation puts the net in eval mode, which switches off the identity drop and the rounding of the
    values; a net trained in eval mode still learns, so only a record of `training` shows the mistake."""
    seen = []
    forward = nets.TokenNet.forward

    def recording(self, *args, **kwargs):
        seen.append(self.training)
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(nets.TokenNet, "forward", recording)
    cfg = train.TrainConfig(d_model=32, depth=1, steps=6, batch=32, warmup=2, val_every=2)
    train.train(CONFIG, cfg, rank_two(200), rank_two(100), TABLE, tmp_path, log=lambda line: None)
    assert len(seen) == 6 and all(seen)
    assert len(pd.read_csv(tmp_path / "train_log.csv")) == 3                 # validations after steps 2, 4 and 6


def test_the_same_seed_gives_the_same_training(tmp_path):
    cfg = train.TrainConfig(d_model=32, depth=1, steps=20, batch=32, warmup=2, val_every=10, seed=3)
    ex = rank_two(300), rank_two(100)
    a = train.train(CONFIG, cfg, ex[0], ex[1], TABLE, tmp_path / "a", log=lambda line: None)
    b = train.train(CONFIG, cfg, ex[0], ex[1], TABLE, tmp_path / "b", log=lambda line: None)
    assert a["best_val_mse"] == b["best_val_mse"] and a["best_step"] == b["best_step"]
    assert (tmp_path / "a" / "train_log.csv").read_text() == (tmp_path / "b" / "train_log.csv").read_text()


# ------------------------------------------------------------------------------- the sub-command

NAME = "summary-ip"


def load_script(monkeypatch):
    spec = importlib.util.spec_from_file_location("train_tokenizer_under_test", SCRIPTS / "train_tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    arm = module.Arm("net.series", (NAME,), False, True, 1, 16)
    monkeypatch.setattr(module.registry, "all_signals", lambda: [])
    monkeypatch.setattr(module, "catalog", lambda specs: ([], [], [], {NAME: 1}))
    monkeypatch.setattr(module, "arms", lambda *a: {
        "net.series": arm, "klt.series": module.Arm("klt.series", (NAME,), False, False, 1, 16),
        "net.cat": module.Arm("net.cat", (NAME,), True, True, 2, 4)})
    return module


def shrink(monkeypatch):
    """A small net and a validation every 3 steps for the command; returns the real `TrainConfig`."""
    real = train.TrainConfig
    monkeypatch.setattr(train, "TrainConfig", lambda **kw: real(**{**dict(d_model=32, depth=1, batch=64, warmup=2,
                                                                         val_every=3), **kw}))
    return real


def toy_stage1(tmp_path, monkeypatch):
    """A stage1 root with a store of one 1-stream signal (60 train and 30 validation frames of 16 values)."""
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    cache_root = tmp_path / "cache" / "v1"
    cache_root.mkdir(parents=True)
    reg = pd.DataFrame([{"device": "mast", "signal": NAME, "channel": 0, "channel_name": "ip", "family": "slow",
                         "dt": 0.00025, "units": "A", "imas_path": "", "r": np.nan, "z": np.nan, "tm_mean": 0.0,
                         "tm_std": 1.0, "roles": "inp+out", "sat_abs": np.nan, "valid_fraction": 1.0,
                         "dead_fraction": 0.0, "compress": "none", "scale": 1.0, "mean": 0.0, "std": 1.0}])
    registry.save_registry(reg, cache_root / "registry.csv")
    meta = {"signals": {NAME: {"kind": "frame_dct", "domain": "linear", "n_streams": 1, "values_per_token": 1}},
            "cache_dir": str(cache_root), "assignment_sha256": "a" * 64, "fit_sha256": "f" * 64,
            "stats_sha256": "s" * 64, "learned_code_sha256": "old"}
    for key, m in (("train", 60), ("val", 30)):
        values = (rank_two(m)[0] * 3.0 + 1.0).reshape(m, 1, 16)
        store.save_store({NAME: store.Store(NAME, values, np.zeros(m, dtype=np.int64))},
                         tmp_path / "stage1" / "store" / "v1" / key, meta)
    return tmp_path / "stage1" / "recoders"


@pytest.mark.parametrize("argv", [
    ["--arm", "nope", "--cpu"],                                             # unknown arm
    ["--arm", "klt.series", "--cpu"],                                       # a linear arm
    ["--arm", "net.series"],                                                # neither --gpu nor --cpu
    ["--arm", "net.series", "--cpu", "--gpu", "1"],                         # both
    ["--arm", "net.series", "--cpu", "--share-gpu"],                        # a GPU flag without a GPU
    ["--arm", "net.series", "--cpu", "--lr", "1e-3"],                       # a non-default setting without --variant
    ["--arm", "net.series", "--cpu", "--steps", "10"],
    ["--arm", "net.series", "--cpu", "--steps", "0", "--variant", "x"],
    ["--arm", "net.series", "--cpu", "--variant", "a/b"],
])
def test_the_command_refuses_before_it_reads_the_store(tmp_path, monkeypatch, capsys, argv):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    recoders = toy_stage1(tmp_path, monkeypatch)
    tt = load_script(monkeypatch)
    monkeypatch.setattr(tt.store, "load_store", lambda *a, **k: pytest.fail("the store was read"))
    assert tt.main(["train", *argv]) == 2
    assert capsys.readouterr().err.startswith("refused: ")
    assert not recoders.exists()


def test_the_command_trains_writes_the_run_and_refuses_to_overwrite_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    recoders = toy_stage1(tmp_path, monkeypatch)
    tt = load_script(monkeypatch)
    small = shrink(monkeypatch)
    assert (tt.DEFAULT_LR, tt.DEFAULT_STEPS) == (small().lr, small().steps)         # the script's defaults are the config's
    argv = ["train", "--arm", "net.series", "--cpu", "--steps", "6", "--variant", "t"]
    rc = tt.main(argv)
    out = capsys.readouterr().out.strip().splitlines()
    run = recoders / "net.series.s0.t"
    result = json.loads((run / "result.json").read_text())
    assert rc == (3 if result["collapsed"] else 0) and out[-1].startswith("net.series.s0.t: best val_mse ")
    assert f"at step {result['best_step']} of 6" in out[-1] and f"collapsed {'yes' if result['collapsed'] else 'no'}" in out[-1]
    assert sorted(p.name for p in run.iterdir() if not p.name.startswith("train.lock")) == [
        "best.pt", "config.json", "norm.npz", "recoder.json", "result.json", "table.npz", "train_log.csv", "trained_by.json"]
    assert not (run / "train.lock").exists()
    by = json.loads((run / "trained_by.json").read_text())
    assert by["assignment_sha256"] == "a" * 64 and by["fit_sha256"] == "f" * 64 and by["stats_sha256"] == "s" * 64
    assert set(by) == {"learned_code_sha256", "tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256",
                       "argv", "date"} and by["argv"] == argv and len(by["learned_code_sha256"]) == 64
    config = json.loads((run / "config.json").read_text())
    assert config["arm"] == "net.series" and config["seed"] == 0 and config["variant"] == "t"
    assert config["train"]["steps"] == 6 and config["net"]["d_in"] == 16 and config["net"]["n_values"] == 16
    assert config["net"]["n_streams"] == 1 and config["net"]["d_model"] == 32 and config["net"]["concat"] is False
    assert {"best_step", "best_val_mse", "best_val_full", "steps", "seconds", "plateau", "collapsed"} <= set(result)
    assert result["steps"] == 6 and result["usage"][NAME]["n_examples"] == 30
    loaded = recode.load_recoder(run)
    assert loaded.recoder.kind == "net" and loaded.ranges == {} and loaded.meta["signals"][NAME]["n_streams"] == 1

    assert tt.main(argv) == 2 and "result.json" in capsys.readouterr().err        # a finished run is not overwritten
    assert (run / "result.json").read_text() == json.dumps(result, indent=2) + "\n"
    assert tt.main([*argv, "--force"]) in (0, 3)                                   # --force trains it again
    assert json.loads((run / "result.json").read_text())["best_val_mse"] == result["best_val_mse"]   # same seed


def test_the_command_trains_a_concatenated_arm_without_stream_identity(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    recoders = toy_stage1(tmp_path, monkeypatch)
    tt = load_script(monkeypatch)
    shrink(monkeypatch)
    rc = tt.main(["train", "--arm", "net.cat", "--cpu", "--steps", "4", "--variant", "t"])
    run = recoders / "net.cat.s0.t"
    result = json.loads((run / "result.json").read_text())
    assert rc == (3 if result["collapsed"] else 0)
    net = json.loads((run / "config.json").read_text())["net"]
    assert net["concat"] is True and net["n_streams"] == 0 and net["d_in"] == 16 and net["n_values"] == 8
    assert recode.load_recoder(run).recoder.concat is True


def test_the_command_refuses_a_run_that_another_process_holds(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    recoders = toy_stage1(tmp_path, monkeypatch)
    tt = load_script(monkeypatch)
    run = recoders / "net.series.s0"
    run.mkdir(parents=True)
    (run / "train.lock").write_text("1\n")                                         # pid 1 is alive
    assert tt.main(["train", "--arm", "net.series", "--cpu"]) == 2
    assert "another training" in capsys.readouterr().err and not (run / "config.json").exists()
