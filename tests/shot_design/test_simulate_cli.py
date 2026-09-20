"""shot_design simulate: the CLI's own orchestration and status-file contract.

Every heavy dependency (D1/D2's functions, and program/program_reference's real
corpus + IGNITE reads) is a small fake here -- the dynamics rollout and the
decode/report pipeline each have their own suite (test_simulate_core.py,
test_simulate_decode.py, test_simulate_report.py). `report.write` itself is NOT
faked: status.json and report.md are real files, read back from tmp_path.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from shot_design import cli
from shot_design.design import program as program_mod
from shot_design.design import program_reference as program_reference_mod
from shot_design.shotdb import ignite as shotdb_ignite
from shot_design.simulate import core as core_mod
from shot_design.simulate import decode as decode_mod

IDENT = "a" * 32
N_FRAMES = 100  # k0(20) + n_predict(80) default, exactly matching the design seed


def _design_seed() -> dict:
    return {
        "codes": {"ece": torch.zeros(N_FRAMES, 4, dtype=torch.int32)},
        "actuators": torch.zeros(N_FRAMES, 88, dtype=torch.float16),
        "n_frames": N_FRAMES,
        "vocabs": {"ece": 1000},
    }


def _program() -> SimpleNamespace:
    # start_s=1.0 -> start_frame 20 -> context 0 (SEED_FRAMES=20);
    # end_s=5.0 -> display_end 100 == N_FRAMES, so the windowed reference cache
    # lines up with the design seed exactly.
    return SimpleNamespace(id=IDENT, reference_shot=1, start_s=1.0, end_s=5.0)


@pytest.fixture
def fake_bundle(paths):
    """A real (tiny) codecs/MANIFEST.json -- `bundle_identity` hashes real bytes
    off it, without needing an actual codec checkpoint."""
    bundle = shotdb_ignite.bundle_dir(paths)
    (bundle / "codecs").mkdir(parents=True)
    (bundle / "codecs" / "MANIFEST.json").write_text(json.dumps({"modalities": {}}))
    return bundle


@pytest.fixture
def fakes(monkeypatch, paths, fake_bundle, tmp_path):
    """Install the brief's Step-1 fakes; return a dict of mutable knobs the
    individual tests flip (`raise_in`) to exercise the failure path."""
    calls: dict = {"mid_state": None, "raise_in": None}

    seed_path = tmp_path / "seed.pt"
    torch.save(_design_seed(), seed_path)

    def fake_load_program(ident, paths_arg):
        if calls["raise_in"] == "load_program":
            raise ValueError("boom in load_program")
        # status.json must already say "running" by the time the first real
        # step runs -- wherever `run()` decided to put it (default or --out,
        # which may sit outside data_root entirely).
        [status_path] = tmp_path.rglob("status.json")
        calls["mid_state"] = json.loads(status_path.read_text())["state"]
        return _program()

    def fake_export_ignite(prog, paths_arg):
        if calls["raise_in"] == "export_ignite":
            raise ValueError("boom in export_ignite")
        return seed_path

    def fake_reference(shot, paths_arg):
        if calls["raise_in"] == "reference":
            raise ValueError("boom in reference")
        cache = {"actuators": torch.ones(N_FRAMES, 88, dtype=torch.float16)}
        return SimpleNamespace(cache=cache)

    def fake_load_dynamics(paths_arg, device):
        if calls["raise_in"] == "load_dynamics":
            raise ValueError("boom in load_dynamics")
        cfg = SimpleNamespace(k0_seed=20, n_predict=80, maskgit_decode_steps=10)
        return SimpleNamespace(), cfg, 4242

    def fake_run_paired(model, cfg, codes, real_act, prop_act, *, seed, **kw):
        if calls["raise_in"] == "run_paired":
            raise ValueError("boom in run_paired")
        f = codes["ece"]
        return core_mod.SimulationArms(
            seed_frames=cfg.k0_seed,
            predict_frames=cfg.n_predict,
            real={"ece": f.clone()},
            proposed={"ece": f.clone()},
            gt={"ece": f.clone()},
            divergence_vs_real={"ece": 0.0},
            token_accuracy={"ece": 1.0},
            persistence_accuracy={"ece": 1.0},
        )

    def fake_decode_modalities(codecs, arms, names):
        if calls["raise_in"] == "decode_modalities":
            raise ValueError("boom in decode_modalities")
        return {}

    def fake_load_codecs(ckpt_dir, names, device):
        return {}

    monkeypatch.setattr(program_mod, "load_program", fake_load_program)
    monkeypatch.setattr(program_mod, "export_ignite", fake_export_ignite)
    monkeypatch.setattr(program_reference_mod, "reference", fake_reference)
    monkeypatch.setattr(core_mod, "load_dynamics", fake_load_dynamics)
    monkeypatch.setattr(core_mod, "run_paired", fake_run_paired)
    monkeypatch.setattr(decode_mod, "decode_modalities", fake_decode_modalities)
    monkeypatch.setattr(shotdb_ignite, "load_codecs", fake_load_codecs)
    return calls


def _status(paths, ident=IDENT):
    out_dir = paths.data_root / "outputs" / ident / "simulation"
    return out_dir, json.loads((out_dir / "status.json").read_text())


def test_simulate_writes_running_then_complete_status_and_a_report(paths, fakes):
    rc = cli.main(["simulate", IDENT])
    assert rc == 0
    assert fakes["mid_state"] == "running"
    out_dir, status = _status(paths)
    assert status["state"] == "complete"
    assert status["report"] == "report.md"
    assert status["error"] is None
    assert status["started"] and status["finished"]
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "simulation.h5").is_file()


@pytest.mark.parametrize(
    "raise_in",
    ["load_program", "export_ignite", "reference", "load_dynamics", "run_paired",
     "decode_modalities"],
)  # fmt: skip
def test_simulate_leaves_a_failed_status_and_exits_1(paths, fakes, raise_in):
    fakes["raise_in"] = raise_in
    rc = cli.main(["simulate", IDENT])
    assert rc == 1
    out_dir, status = _status(paths)
    assert status["state"] == "failed"
    assert f"boom in {raise_in}" in status["error"]
    assert status["report"] is None
    assert status["started"] and status["finished"]


def test_simulate_rejects_k0_plus_n_predict_over_the_seeds_frame_count(paths, fakes):
    rc = cli.main(["simulate", IDENT, "--k0", "50", "--n-predict", "80"])
    assert rc == 1
    _, status = _status(paths)
    assert status["state"] == "failed"
    assert "130" in status["error"] and "100" in status["error"]


def test_simulate_writes_actuators_into_the_h5(paths, fakes):
    assert cli.main(["simulate", IDENT]) == 0
    import h5py

    out_dir = paths.data_root / "outputs" / IDENT / "simulation"
    with h5py.File(out_dir / "simulation.h5") as f:
        assert "actuators/real" in f and "actuators/proposed" in f
        assert f.attrs["design_id"] == IDENT
        assert f.attrs["codec_generation"]
        assert f.attrs["dynamics_step"] == 4242


def test_simulate_default_out_dir_is_data_root_outputs_ident_simulation(paths, fakes):
    assert cli.main(["simulate", IDENT]) == 0
    out_dir = paths.data_root / "outputs" / IDENT / "simulation"
    assert (out_dir / "status.json").is_file()


def test_simulate_out_flag_overrides_the_default_directory(paths, fakes, tmp_path):
    custom = tmp_path / "custom_out"
    assert cli.main(["simulate", IDENT, "--out", str(custom)]) == 0
    assert (custom / "status.json").is_file()
    assert (custom / "report.md").is_file()
