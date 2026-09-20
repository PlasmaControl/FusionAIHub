import torch

from tokamak_foundation_model.ignite import dynamics_config as dc, maskgit
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
