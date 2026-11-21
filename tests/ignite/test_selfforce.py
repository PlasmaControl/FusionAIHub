"""Rollout-context fine-tune: train on the model's OWN context distribution."""
import pytest
import torch

from tokamak_foundation_model.ignite import maskgit as maskgit_mod
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics
from tokamak_foundation_model.ignite.selfforce import rollout_context


def _tiny(**kw):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 4, 5), ModalitySpec("b", "slowts", 3, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=4,
        maskgit_decode_steps=3, actuator_dim=6, **kw)


def _codes(cfg, B, F):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok)) for m in cfg.modalities}


def test_sf_defaults_to_off():
    c = _tiny()
    assert c.sf_frames == 0 and c.sf_prob == 1.0 and c.sf_decode_steps == 4


def test_training_loss_with_sf_frames_backprops():
    cfg = _tiny(sf_frames=2, ctf_frac=1.0)
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=6)
    act = torch.randn(2, 6, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_rollout_context_rejects_a_negative_boundary():
    """A negative boundary must raise, not silently duplicate frames.

    ``v[:, :-1]`` is a perfectly legal slice, so without the guard the window would be built on a
    truncated prefix and real frames would be duplicated into the output — a silently wrong
    training context rather than an error.
    """
    cfg = _tiny()
    torch.manual_seed(0)
    mg = MaskGITDynamics(cfg).eval()
    codes = _codes(cfg, B=1, F=6)
    act = torch.randn(1, 6, cfg.actuator_dim)
    with pytest.raises(ValueError, match="boundary"):
        rollout_context(mg, codes, act, boundary=-1, n_roll=2)


def test_sf_backprops_under_autocast_with_checkpointing():
    """The no-grad rollout must not leak grad-untracked casts into the autocast weight cache.

    Needs all three ingredients together, which is why the plain-CPU tests above never caught it:
    an ambient autocast region with its cast cache live (what the trainer runs), the ``no_grad``
    self-forcing rollout inside it, and gradient checkpointing on the supervised forward. The
    rollout's weight casts land in the cache detached from the autograd graph; the checkpointed
    recompute then reuses them and the ``addmm`` grad path dies with "mat1 and mat2 shapes cannot
    be multiplied (NxD and 1xN)". With checkpointing OFF the same stale casts fail SILENTLY
    instead — the backbone parameters get no gradient at all — so both are asserted here.

    Reproduced on GPU first (Task 11, bf16 autocast at d512); this is the CPU-autocast equivalent.
    """
    for gc in (True, False):
        cfg = _tiny(sf_frames=2, ctf_frac=1.0, grad_checkpointing=gc)
        torch.manual_seed(0)
        mg = MaskGITDynamics(cfg).train()
        codes = _codes(cfg, B=2, F=6)
        act = torch.randn(2, 6, cfg.actuator_dim)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
        loss.backward()
        g = mg.backbone.blocks[0].spatial.qkv.weight.grad
        assert g is not None, f"grad_checkpointing={gc}: no gradient reached the backbone"
        assert g.abs().sum() > 0, f"grad_checkpointing={gc}: gradient is all zero"
