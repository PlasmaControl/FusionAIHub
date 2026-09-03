"""Actuator dropout (training) and classifier-free guidance (inference)."""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.sampling import SamplerConfig


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def test_actuator_dropout_defaults_to_off():
    assert _tiny().actuator_dropout_p == 0.0


def test_drop_actuators_makes_encode_ignore_the_actuators():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    codes = {m.name: torch.randint(0, m.codebook_size, (1, 4, m.n_tok)) for m in cfg.modalities}
    a1 = torch.randn(1, 4, cfg.actuator_dim)
    a2 = torch.randn(1, 4, cfg.actuator_dim)
    h1 = mg.backbone.encode(codes, a1, drop_actuators=True)
    h2 = mg.backbone.encode(codes, a2, drop_actuators=True)
    assert torch.allclose(h1, h2), "dropped actuators must not influence the hidden states"
    assert not torch.allclose(mg.backbone.encode(codes, a1), h1)


def test_cfg_scale_one_is_identical_to_no_guidance():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    base = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4))
    same = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4),
                             sampler=SamplerConfig(cfg_scale=1.0))
    for m in cfg.modalities:
        assert torch.equal(base[m.name], same[m.name])


def test_cfg_scale_above_one_changes_the_frame():
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    past = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok))
            for m in cfg.modalities}
    act = torch.randn(1, cfg.k0_seed + 1, cfg.actuator_dim)
    base = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4))
    guided = mg.generate_frame(past, act, generator=torch.Generator().manual_seed(4),
                               sampler=SamplerConfig(cfg_scale=3.0))
    assert any(not torch.equal(base[m.name], guided[m.name]) for m in cfg.modalities)
