"""shot_design simulate: the CLI's orchestration and its status-file contract.

The heavy steps (the dynamics checkpoint, the rollouts, the codecs) are small fakes; each
has its own suite. `report.write` is real, so status.json, metrics.json and the h5 file
are read back from disk.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from shot_design import cli
from shot_design.design import program as program_mod
from shot_design.design import program_reference as program_reference_mod
from shot_design.shotdb import ignite as shotdb_ignite
from shot_design.simulate import core as core_mod
from shot_design.simulate import decode as decode_mod

IDENT = "a" * 32
N_FRAMES = 100  # k0 (20) + n_predict (80), exactly the design seed


def _design_seed(n_frames: int = N_FRAMES) -> dict:
    return {
        "codes": {"ece": torch.zeros(n_frames, 4, dtype=torch.int32)},
        "actuators": torch.zeros(n_frames, 88, dtype=torch.float16),
        "n_frames": n_frames,
        "vocabs": {"ece": 1000},
    }


class _Config(SimpleNamespace):
    @property
    def max_frames(self):
        return self.k0_seed + self.n_predict


def _cfg(trained: int = 100) -> _Config:
    return _Config(
        k0_seed=20,
        n_predict=trained - 20,
        maskgit_decode_steps=10,
        modalities=(SimpleNamespace(name="ece"),),
    )


@pytest.fixture
def fake_bundle(paths):
    """A real (tiny) codecs/MANIFEST.json, so `bundle_identity` hashes real bytes."""
    bundle = shotdb_ignite.bundle_dir(paths)
    (bundle / "codecs").mkdir(parents=True)
    (bundle / "codecs" / "MANIFEST.json").write_text(json.dumps({"modalities": {}}))
    return bundle


@pytest.fixture
def fakes(monkeypatch, paths, fake_bundle, tmp_path):
    """Install the fakes; tests flip ``raise_in`` to take the failure path and read
    back what each step was called with."""
    calls: dict = {"mid_state": None, "raise_in": None, "cfg": _cfg()}
    seed_path = tmp_path / "seed.pt"
    torch.save(_design_seed(), seed_path)

    def step(name, result):
        def fake(*args, **kwargs):
            if calls["raise_in"] == name:
                raise ValueError(f"boom in {name}")
            calls[name] = (args, kwargs)
            return result(*args, **kwargs)

        return fake

    def load_program(ident, paths_arg):
        # status.json already says "running" when the first real step runs
        [status_path] = tmp_path.rglob("status.json")
        calls["mid_state"] = json.loads(status_path.read_text())["state"]
        return SimpleNamespace(id=IDENT, reference_shot=1, start_s=2.0, end_s=6.0)

    def reference(shot, paths_arg):
        return SimpleNamespace(cache={"actuators": torch.ones(200, 88)})

    def run_ensemble(model, cfg, codes, arms, members, **kw):
        total = cfg.k0_seed + cfg.n_predict
        gt = codes["ece"][:total].long()
        return core_mod.Ensemble(
            k0=cfg.k0_seed,
            gt={"ece": gt},
            arms={a: {"ece": gt.expand(members, -1, -1).clone()} for a in arms},
            seeds={a: seed for a, (_, seed) in arms.items()},
            held=(),
            batch=kw.get("batch") or 1,
        )

    def decode_ensemble(codecs, ens):
        rng = np.random.default_rng(0)
        f, m = ens.gt["ece"].shape[0], ens.arms["real"]["ece"].shape[0]
        per = {"gt": rng.normal(size=(f, 2)).astype(np.float32)}
        return {"ece": per | {a: rng.normal(size=(m, f, 2)).astype(np.float32)
                              for a in ens.arms}}

    monkeypatch.setattr(program_mod, "load_program", step("load_program", load_program))
    monkeypatch.setattr(program_mod, "export_ignite", step("export_ignite", lambda *a: seed_path))
    monkeypatch.setattr(program_reference_mod, "reference", step("reference", reference))
    monkeypatch.setattr(
        core_mod, "load_dynamics",
        step("load_dynamics", lambda *a: (object(), calls["cfg"], 4242)),
    )
    monkeypatch.setattr(core_mod, "run_ensemble", step("run_ensemble", run_ensemble))
    monkeypatch.setattr(
        shotdb_ignite, "load_codecs",
        step("load_codecs", lambda *a: {"ece": (None, None, "spectro")}),
    )
    monkeypatch.setattr(decode_mod, "decode_ensemble", step("decode_ensemble", decode_ensemble))
    return calls


def _out(paths, ident=IDENT):
    return paths.data_root / "outputs" / ident / "simulation"


def _status(paths):
    return json.loads((_out(paths) / "status.json").read_text())


def _metrics(paths):
    return json.loads((_out(paths) / "metrics.json").read_text())


def test_simulate_writes_running_then_complete_status_and_its_outputs(paths, fakes):
    assert cli.main(["simulate", IDENT]) == 0
    assert fakes["mid_state"] == "running"
    status = _status(paths)
    assert status["state"] == "complete" and status["report"] == "report.md"
    assert status["error"] is None and status["started"] and status["finished"]
    for name in ("report.md", "metrics.json", "simulation.h5", "panels/ece.png"):
        assert (_out(paths) / name).is_file(), name


@pytest.mark.parametrize(
    "raise_in",
    ["load_program", "export_ignite", "reference", "load_dynamics", "run_ensemble",
     "load_codecs", "decode_ensemble"],
)  # fmt: skip
def test_simulate_leaves_a_failed_status_and_exits_1(paths, fakes, raise_in):
    fakes["raise_in"] = raise_in
    assert cli.main(["simulate", IDENT]) == 1
    status = _status(paths)
    assert status["state"] == "failed" and f"boom in {raise_in}" in status["error"]
    assert status["report"] is None and status["started"] and status["finished"]


def test_simulate_prints_a_traceback_to_stderr_on_failure(paths, fakes, capsys):
    """A Slurm job leaves only stderr and status.json; status["error"] is str(exc)."""
    fakes["raise_in"] = "load_program"
    assert cli.main(["simulate", IDENT]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err and "boom in load_program" in err


def test_real_and_proposed_share_the_seed_and_null_takes_fresh_random_numbers(
    paths, fakes
):
    assert cli.main(["simulate", IDENT, "--members", "3", "--seed", "5"]) == 0
    (_, _, _, arms, members), _ = fakes["run_ensemble"]
    assert members == 3
    assert {a: seed for a, (_, seed) in arms.items()} == {"real": 5, "proposed": 5, "null": 8}
    assert torch.equal(arms["real"][0], arms["null"][0])  # the same measured actuators
    assert (arms["real"][0][20:] == 1).all() and (arms["proposed"][0][20:] == 0).all()
    assert torch.equal(arms["proposed"][0][:20], arms["real"][0][:20])


def test_sampling_flags_reach_the_rollouts_and_the_records(paths, fakes):
    argv = ["--decode-steps", "4", "--temperature", "0.7", "--batch", "2", "--bf16"]
    assert cli.main(["simulate", IDENT, *argv]) == 0
    _, kw = fakes["run_ensemble"]
    assert kw["sampler"].temperature == 0.7 and kw["batch"] == 2 and kw["bf16"]
    assert fakes["cfg"].maskgit_decode_steps == 4
    doc = _metrics(paths)
    assert doc["decode_steps"] == 4 and doc["temperature"] == 0.7
    with h5py.File(_out(paths) / "simulation.h5") as f:
        assert f.attrs["precision"] == "bf16" and f.attrs["members"] == 8


def test_every_modality_is_decoded_unless_some_are_named(paths, fakes):
    assert cli.main(["simulate", IDENT]) == 0
    assert fakes["load_codecs"][0][1] == ["ece"]
    assert cli.main(["simulate", IDENT, "--decode", "mhr, ece"]) == 0
    assert fakes["load_codecs"][0][1] == ["mhr", "ece"]


def test_the_h5_records_the_bundle_the_window_and_the_actuators(paths, fakes):
    assert cli.main(["simulate", IDENT]) == 0
    with h5py.File(_out(paths) / "simulation.h5") as f:
        assert "actuators/real" in f and "actuators/proposed" in f
        assert f.attrs["design_id"] == IDENT and f.attrs["codec_generation"]
        assert f.attrs["frame_origin_s"] == 1.0 and f.attrs["frame_s"] == 0.05
        assert f.attrs["dynamics_step"] == 4242
        assert json.loads(f.attrs["seeds"]) == {"real": 0, "proposed": 0, "null": 8}


@pytest.mark.parametrize(("k0", "t0_s"), [(20, 2.0), (10, 1.5)])
def test_t0_is_the_shot_time_of_the_first_predicted_frame(paths, fakes, k0, t0_s):
    # the seed starts 20 frames before the 2.0 s window, at the 1.0 s frame origin
    assert cli.main(["simulate", IDENT, "--k0", str(k0), "--n-predict", "80"]) == 0
    assert _metrics(paths)["t0_s"] == pytest.approx(t0_s)


def test_simulate_out_flag_overrides_the_default_directory(paths, fakes, tmp_path):
    custom = tmp_path / "custom_out"
    assert cli.main(["simulate", IDENT, "--out", str(custom)]) == 0
    assert (custom / "status.json").is_file() and (custom / "metrics.json").is_file()


@pytest.mark.parametrize(
    ("n_frames", "trained", "expected"), [(60, 100, 40), (200, 100, 80), (100, 70, 50)]
)
def test_default_prediction_fits_seed_and_checkpoint(
    paths, fakes, monkeypatch, tmp_path, n_frames, trained, expected
):
    seed_path = tmp_path / "sized_seed.pt"
    torch.save(_design_seed(n_frames), seed_path)
    monkeypatch.setattr(program_mod, "export_ignite", lambda *a: seed_path)
    fakes["cfg"] = _cfg(trained)
    assert cli.main(["simulate", IDENT]) == 0
    assert fakes["cfg"].n_predict == expected
    with h5py.File(_out(paths) / "simulation.h5") as f:
        assert f.attrs["n_predict"] == expected
        assert f["actuators/real"].shape[0] == expected + 20


@pytest.mark.parametrize(
    "argv",
    [
        ["--k0", "50", "--n-predict", "80"],  # past the design seed's 100 frames
        ["--n-predict", "0"],
        ["--n-predict", "-1"],
        ["--k0", "100"],  # no room left to predict
    ],
)
def test_a_horizon_that_does_not_fit_fails(paths, fakes, argv):
    assert cli.main(["simulate", IDENT, *argv]) == 1
    assert "does not fit" in _status(paths)["error"]


def test_a_horizon_past_the_checkpoints_is_refused(paths, fakes, monkeypatch, tmp_path):
    """The design seed (200 frames) has room; the checkpoint (100) does not."""
    seed_path = tmp_path / "big_seed.pt"
    torch.save(_design_seed(200), seed_path)
    monkeypatch.setattr(program_mod, "export_ignite", lambda *a: seed_path)
    assert cli.main(["simulate", IDENT, "--n-predict", "100"]) == 1
    assert "checkpoint's 100" in _status(paths)["error"]
