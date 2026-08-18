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


def test_boundary_mask_leaves_a_clean_prefix_and_masks_the_suffix():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg)
    codes = _codes(cfg, B=4, F=6)
    masked, mask = mg._boundary_mask(codes, gen=torch.Generator().manual_seed(0))
    for b in range(4):
        per_frame = torch.stack([mask[m.name][b].any(dim=-1) for m in cfg.modalities]).any(0)
        first_masked = int(per_frame.float().argmax())
        assert per_frame[first_masked:].all(), "every frame at/after the boundary is supervised"
        assert not per_frame[:first_masked].any(), "the prefix is COMPLETE (unmasked) context"
    for m in cfg.modalities:                       # unmasked positions keep the true code
        keep = ~mask[m.name]
        assert torch.equal(masked[m.name][keep], codes[m.name][keep])


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
