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
    B, Fr = 4, 6
    codes = _codes(cfg, B=B, F=Fr)
    masked, mask = mg._boundary_mask(codes, gen=torch.Generator().manual_seed(0))
    supervised = torch.stack([mask[m.name].any(dim=-1) for m in cfg.modalities])   # (M, B, F)
    for b in range(B):
        per_frame = supervised[:, b].any(0)
        first_masked = int(per_frame.float().argmax())
        # The boundary must be a REAL split. Deriving it from the mask makes the layout asserts
        # below vacuous for a degenerate row (c == 0, or nothing masked at all -> argmax 0), so
        # pin it independently: >=1 complete context frame AND >=1 supervised frame.
        assert per_frame.any(), f"sample {b}: no frame is supervised"
        assert 1 <= first_masked <= Fr - 1, f"sample {b}: degenerate boundary {first_masked}"
        # PER MODALITY, not just their union: every modality independently supervises every
        # frame at/after the boundary and leaves every frame before it untouched.
        for mi, m in enumerate(cfg.modalities):
            pf = supervised[mi, b]
            assert pf[first_masked:].all(), f"{m.name}: frames at/after the boundary supervised"
            assert not pf[:first_masked].any(), f"{m.name}: prefix is COMPLETE (unmasked) context"
    for m in cfg.modalities:
        keep = ~mask[m.name]
        assert torch.equal(masked[m.name][keep], codes[m.name][keep])   # kept -> true code
        assert (masked[m.name][mask[m.name]] == m.codebook_size).all()  # masked -> MASK id


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


def test_loss_weighting_modes_change_the_loss_but_stay_finite():
    import math
    cfg_u = _tiny()
    cfg_t = _tiny(modality_loss_weight="tokens")
    torch.manual_seed(0)
    mg_u = MaskGITDynamics(cfg_u).train()
    torch.manual_seed(0)
    mg_t = MaskGITDynamics(cfg_t).train()
    codes = _codes(cfg_u, B=2, F=5)
    act = torch.randn(2, 5, cfg_u.actuator_dim)
    lu = mg_u.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    lt = mg_t.training_loss(codes, act, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(lu) and torch.isfinite(lt)
    assert not math.isclose(float(lu), float(lt), rel_tol=1e-9)


def test_default_loss_weight_is_uniform():
    assert _tiny().modality_loss_weight == "uniform"
