import dataclasses

import pytest
import torch

from tokamak_foundation_model.ignite import dynamics_config as dc, maskgit
from shot_design.shotdb import ignite as shotdb_ignite
from shot_design.simulate import core

MODS = (
    dc.ModalitySpec("ece", "spectro", 4, 16),
    dc.ModalitySpec("mse", "slowts", 2, 16),
)


def _model():
    cfg = dc.DynamicsConfig(
        modalities=MODS,
        d_model=32,
        depth=1,
        n_heads=2,
        actuator_dim=88,
        grad_checkpointing=False,
        k0_seed=2,
        n_predict=3,
    )
    return maskgit.MaskGITDynamics(cfg).eval(), cfg


def _codes(F):
    return {
        "ece": torch.randint(0, 16, (F, 4), dtype=torch.int32),
        "mse": torch.randint(0, 16, (F, 2), dtype=torch.int32),
    }


def test_actuator_arms_share_the_seed_frames():
    ref = {"actuators": torch.randn(10, 88).half()}
    des = {"actuators": torch.randn(5, 88).half()}
    real, prop = core.actuator_arms(ref, des, k0=2, n_predict=3)
    assert real.shape == prop.shape == (5, 88)
    assert torch.equal(real[:2], prop[:2])
    assert not torch.equal(real[2:], prop[2:])


def test_run_paired_is_deterministic_and_reports_token_metrics():
    model, cfg = _model()
    codes = _codes(5)
    real = torch.zeros(5, 88)
    prop = torch.ones(5, 88)
    a = core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    b = core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    assert torch.equal(a.real["ece"], b.real["ece"])
    assert a.real["ece"].shape == (5, 4) and a.proposed["mse"].shape == (5, 2)
    assert set(a.divergence_vs_real) == {"ece", "mse"}
    assert 0 <= a.token_accuracy["ece"] <= 1
    # seed frames are copied through
    assert torch.equal(a.real["ece"][:2], codes["ece"][:2])


def test_run_paired_restores_the_shared_cfgs_decode_steps():
    model, cfg = _model()
    codes = _codes(5)
    real = torch.zeros(5, 88)
    prop = torch.ones(5, 88)
    original = cfg.maskgit_decode_steps
    core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    assert cfg.maskgit_decode_steps == original


def test_run_paired_rejects_code_windows_shorter_than_k0_plus_n_predict():
    model, cfg = _model()
    codes = _codes(4)  # k0+n_predict is 5; one frame short
    real = torch.zeros(5, 88)
    prop = torch.ones(5, 88)
    with pytest.raises(ValueError) as exc:
        core.run_paired(model, cfg, codes, real, prop, seed=1, decode_steps=2)
    assert "4" in str(exc.value) and "5" in str(exc.value)


def test_actuator_arms_rejects_missing_reference_actuators():
    des = {"actuators": torch.randn(5, 88).half()}
    with pytest.raises(ValueError, match="reference_cache"):
        core.actuator_arms({"actuators": None}, des, k0=2, n_predict=3)


def test_actuator_arms_rejects_missing_design_actuators():
    ref = {"actuators": torch.randn(5, 88).half()}
    with pytest.raises(ValueError, match="design_seed"):
        core.actuator_arms(ref, {"actuators": None}, k0=2, n_predict=3)


def test_load_dynamics_reads_the_pinned_bundle_checkpoint(paths):
    model, cfg = _model()
    ck = {
        "model": model.state_dict(),
        "cfg_depth": cfg.depth,
        "cfg_d_model": cfg.d_model,
        "cfg_n_heads": cfg.n_heads,
        "cfg_k0": cfg.k0_seed,
        "cfg_n_predict": cfg.n_predict,
        "modalities": [dataclasses.astuple(m) for m in cfg.modalities],
        "step": 7,
    }
    bundle = shotdb_ignite.bundle_dir(paths)
    bundle.mkdir(parents=True, exist_ok=True)
    torch.save(ck, bundle / shotdb_ignite.model_cfg()["dynamics_file"])
    _, loaded_cfg, step = core.load_dynamics(paths, "cpu")
    assert loaded_cfg.actuator_dim == 88 and step == 7
