import dataclasses

import pytest
import torch

from shot_design.shotdb import ignite as shotdb_ignite
from shot_design.simulate import core
from tokamak_foundation_model.ignite import dynamics_config as dc
from tokamak_foundation_model.ignite import maskgit

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
        maskgit_decode_steps=2,
    )
    torch.manual_seed(0)
    return maskgit.MaskGITDynamics(cfg).eval(), cfg


def _codes(F):
    g = torch.Generator().manual_seed(1)
    return {
        "ece": torch.randint(0, 16, (F, 4), dtype=torch.int32, generator=g),
        "mse": torch.randint(0, 16, (F, 2), dtype=torch.int32, generator=g),
    }


def _arms(**seeds):
    return {arm: (torch.zeros(5, 88), seed) for arm, seed in seeds.items()}


def test_actuator_arms_share_the_seed_frames():
    ref = {"actuators": torch.randn(10, 88).half()}
    des = {"actuators": torch.randn(5, 88).half()}
    real, prop = core.actuator_arms(ref, des, k0=2, n_predict=3)
    assert real.shape == prop.shape == (5, 88)
    assert torch.equal(real[:2], prop[:2])
    assert not torch.equal(real[2:], prop[2:])


def test_an_ensemble_has_every_member_of_every_arm_from_the_measured_seed():
    model, cfg = _model()
    codes = _codes(6)
    ens = core.run_ensemble(model, cfg, codes, _arms(real=0, null=4), members=3)
    assert set(ens.arms) == {"real", "null"} and ens.k0 == 2 and ens.batch == 1
    assert ens.seeds == {"real": 0, "null": 4} and ens.held == ()
    for tokens in ens.arms.values():
        assert tokens["ece"].shape == (3, 5, 4) and tokens["mse"].shape == (3, 5, 2)
        assert (tokens["ece"][:, :2] == codes["ece"][:2]).all()
    assert torch.equal(ens.gt["ece"], codes["ece"][:5].long())


def test_arms_on_one_seed_share_random_numbers_and_the_null_arm_does_not():
    model, cfg = _model()
    ens = core.run_ensemble(model, cfg, _codes(5), _arms(real=0, same=0, null=3), 3)
    assert torch.equal(ens.arms["real"]["ece"], ens.arms["same"]["ece"])
    assert not torch.equal(ens.arms["real"]["ece"], ens.arms["null"]["ece"])


def test_members_differ_within_an_arm_and_batching_keeps_the_shapes():
    model, cfg = _model()
    ens = core.run_ensemble(model, cfg, _codes(5), _arms(real=0), 3, batch=2)
    ece = ens.arms["real"]["ece"]
    assert ece.shape == (3, 5, 4) and ens.batch == 2
    assert not torch.equal(ece[0], ece[1])


def test_a_diagnostic_absent_from_the_seed_is_held_at_its_placeholder():
    model, cfg = _model()
    codes = _codes(5)
    codes["mse"][:] = 7  # the codec's constant null code in every seed frame
    ens = core.run_ensemble(model, cfg, codes, _arms(real=0), 2)
    assert ens.held == ("mse",)
    assert (ens.arms["real"]["mse"] == 7).all()


def test_windows_shorter_than_k0_plus_n_predict_are_refused():
    model, cfg = _model()
    with pytest.raises(ValueError, match="has 4 frames"):
        core.run_ensemble(model, cfg, _codes(4), _arms(real=0), 1)
    short = {"real": (torch.zeros(4, 88), 0)}
    with pytest.raises(ValueError, match="real actuators cover 4 frames"):
        core.run_ensemble(model, cfg, _codes(5), short, 1)


def test_one_member_per_pass_on_the_cpu():
    _, cfg = _model()
    assert core.members_per_pass(cfg, "cpu") == 1


def test_actuator_arms_rejects_actuator_windows_shorter_than_k0_plus_n_predict():
    ref = {"actuators": torch.randn(5, 88).half()}
    des = {"actuators": torch.randn(4, 88).half()}  # one frame short
    with pytest.raises(ValueError, match="design_seed actuators has 4 frames"):
        core.actuator_arms(ref, des, k0=2, n_predict=3)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_an_ensemble_on_the_gpu_returns_host_tensors_and_leaves_its_inputs():
    model, cfg = _model()
    codes = _codes(5)
    ens = core.run_ensemble(model.cuda(), cfg, codes, _arms(real=0), 2, bf16=True)
    for tokens in (ens.gt, *ens.arms.values()):
        assert all(t.device.type == "cpu" for t in tokens.values())
    assert all(v.device.type == "cpu" for v in codes.values())
