"""Complete-context (CTF) training: match the conditional the rollout actually uses."""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def _codes(cfg, B, F):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok)) for m in cfg.modalities}


def test_ctf_defaults_to_off():
    assert _tiny().ctf_frac == 0.0


def test_ctf_loss_is_finite_and_backprops():
    cfg = _tiny(ctf_frac=1.0)
    mg = MaskGITDynamics(cfg).train()
    # F == cfg.max_frames (k0_seed 2 + n_predict 3): the longest legal frame layout
    codes = _codes(cfg, B=2, F=cfg.max_frames)
    act = torch.randn(2, cfg.max_frames, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_default_path_draws_no_extra_rng():
    """The ctf_frac == 0 gate must not consume a single draw.

    The golden fixture (tests/ignite/test_phaseb_compat.py) pins the default loss VALUE, which
    only holds while the default path's RNG stream is exactly a bare _random_mask run. This pins
    the stream itself, so a future gate that draws unconditionally fails here with a clear cause
    instead of showing up as a mystery fixture mismatch.
    """
    cfg = _tiny()
    assert cfg.ctf_frac == 0.0
    mg = MaskGITDynamics(cfg)
    codes = _codes(cfg, B=2, F=cfg.max_frames)
    act = torch.randn(2, cfg.max_frames, cfg.actuator_dim)
    g1 = torch.Generator().manual_seed(5)
    mg.training_loss(codes, act, generator=g1)
    g2 = torch.Generator().manual_seed(5)
    mg._random_mask(codes, g2)                     # the ONLY draws the default path may make
    assert torch.equal(g1.get_state(), g2.get_state()), "default path consumed extra RNG"


def test_default_loss_weight_is_uniform():
    assert _tiny().modality_loss_weight == "uniform"
