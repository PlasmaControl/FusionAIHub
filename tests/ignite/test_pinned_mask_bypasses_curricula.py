"""A caller-pinned mask (``mask_ratio`` given) bypasses the CTF and Self-Forcing arms.

``_gen_val_loss`` pins ``mask_ratio=1.0, history_frames=k0_seed, gen_mask_p=0.0`` to
score the cold-start conditional a rollout actually performs. Under ``--ctf_frac>0`` /
``--sf_frames>0`` the training curricula used to win that branch and silently turn the
diagnostic into a random CTF/SF loss (inert only at the defaults ctf_frac=0,
sf_frames=0).
"""
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5),
                    ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=3,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def _codes(cfg, B, F):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok))
            for m in cfg.modalities}


PINNED = dict(mask_ratio=1.0, history_frames=2, gen_mask_p=0.0)


def _model(cfg):
    torch.manual_seed(0)   # ctf_frac / sf_frames add no parameters -> identical weights
    return MaskGITDynamics(cfg).eval()


def test_pinned_mask_ignores_ctf_and_sf_curricula():
    base = _tiny()
    curr = _tiny(ctf_frac=1.0, sf_frames=1, sf_prob=1.0)
    codes = _codes(base, B=2, F=base.max_frames)
    act = torch.randn(2, base.max_frames, base.actuator_dim)
    with torch.no_grad():
        l_base = _model(base).training_loss(
            codes, act, generator=torch.Generator().manual_seed(7), **PINNED)
        l_curr = _model(curr).training_loss(
            codes, act, generator=torch.Generator().manual_seed(7), **PINNED)
    assert torch.equal(l_base, l_curr), "curricula changed a caller-pinned mask layout"


def test_pinned_mask_draws_only_the_random_mask_rng():
    """The bypass must not consume a draw: no CTF gate, no SF gate, no boundary draw."""
    cfg = _tiny(ctf_frac=1.0, sf_frames=1, sf_prob=0.5)
    mg = _model(cfg)
    codes = _codes(cfg, B=2, F=cfg.max_frames)
    act = torch.randn(2, cfg.max_frames, cfg.actuator_dim)
    g1 = torch.Generator().manual_seed(5)
    with torch.no_grad():
        mg.training_loss(codes, act, generator=g1, **PINNED)
    g2 = torch.Generator().manual_seed(5)
    mg._random_mask(codes, g2, ratio_override=1.0, history_frames=2, gen_mask_p=0.0)
    assert torch.equal(g1.get_state(), g2.get_state()), "pinned path consumed extra RNG"


def test_unpinned_call_still_uses_the_ctf_curriculum():
    """Regression guard: the bypass is keyed on the pin, not on ctf_frac."""
    cfg = _tiny(ctf_frac=1.0)
    mg = _model(cfg)
    codes = _codes(cfg, B=2, F=cfg.max_frames)
    act = torch.randn(2, cfg.max_frames, cfg.actuator_dim)
    g1 = torch.Generator().manual_seed(5)
    with torch.no_grad():
        mg.training_loss(codes, act, generator=g1)
    g2 = torch.Generator().manual_seed(5)
    mg._random_mask(codes, g2)
    assert not torch.equal(g1.get_state(), g2.get_state()), \
        "ctf_frac=1.0 took the random path"
