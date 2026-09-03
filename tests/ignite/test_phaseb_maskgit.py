"""Phase-B MaskGIT: masked-token training loss, valid iterative decode, commit-code rollout."""

import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics, _cosine_keep_fractions


def _tiny(depth=2, decode=4):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4)),
        d_model=16, depth=depth, n_heads=2, ffn_mult=2,
        k0_seed=2, n_predict=3, maskgit_decode_steps=decode, actuator_dim=6,
    )


def _codes(cfg, B, F, gen=None):
    return {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok), generator=gen)
            for m in cfg.modalities}


def test_cosine_schedule_reveals_everything():
    fr = _cosine_keep_fractions(6)
    assert fr[0] < 1.0 and abs(fr[-1]) < 1e-6          # ends fully revealed
    assert all(a >= b - 1e-9 for a, b in zip(fr, fr[1:]))  # monotone non-increasing


def test_random_mask_uses_mask_id_and_at_least_one_per_frame():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg)
    codes = _codes(cfg, B=2, F=4)
    masked, mask = mg._random_mask(codes, gen=None)
    for m in cfg.modalities:
        mid = cfg_mask_id = m.codebook_size
        assert (masked[m.name][mask[m.name]] == mid).all()          # masked -> MASK id
        assert (masked[m.name][~mask[m.name]] == codes[m.name][~mask[m.name]]).all()  # rest kept
        assert mask[m.name].any(dim=-1).all()                        # >=1 masked per frame


def test_training_loss_finite_and_grads_flow():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=5)
    act = torch.randn(2, 5, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    g = mg.backbone.blocks[0].spatial.qkv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_generate_frame_commits_valid_codes():
    cfg = _tiny(decode=4)
    mg = MaskGITDynamics(cfg).eval()
    past = _codes(cfg, B=2, F=cfg.k0_seed)
    act = torch.randn(2, cfg.k0_seed + 1, cfg.actuator_dim)
    gen = torch.Generator().manual_seed(1)
    nxt = mg.generate_frame(past, act, generator=gen)
    for m in cfg.modalities:
        assert nxt[m.name].shape == (2, m.n_tok)
        assert (nxt[m.name] >= 0).all() and (nxt[m.name] < m.codebook_size).all()  # NEVER the MASK id


def test_rollout_shapes_seed_preserved_and_valid():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    seed = _codes(cfg, B=2, F=cfg.k0_seed)
    act = torch.randn(2, cfg.k0_seed + cfg.n_predict, cfg.actuator_dim)
    traj = mg.rollout(seed, act, generator=torch.Generator().manual_seed(2))
    for m in cfg.modalities:
        assert traj[m.name].shape == (2, cfg.k0_seed + cfg.n_predict, m.n_tok)
        # seed frames untouched
        assert torch.equal(traj[m.name][:, : cfg.k0_seed], seed[m.name])
        # generated frames are valid codes
        gen_part = traj[m.name][:, cfg.k0_seed:]
        assert (gen_part >= 0).all() and (gen_part < m.codebook_size).all()


def test_rollout_is_deterministic_with_generator():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    seed = _codes(cfg, B=1, F=cfg.k0_seed)
    act = torch.randn(1, cfg.k0_seed + cfg.n_predict, cfg.actuator_dim)
    t1 = mg.rollout(seed, act, generator=torch.Generator().manual_seed(7))
    t2 = mg.rollout(seed, act, generator=torch.Generator().manual_seed(7))
    for m in cfg.modalities:
        assert torch.equal(t1[m.name], t2[m.name])


def test_ss_fraction_ramps_from_zero_to_final():
    mg = MaskGITDynamics(_tiny())
    assert mg.ss_fraction(0) == 0.0
    assert mg.ss_fraction(mg.cfg.ss_ramp_steps) == pytest.approx(mg.cfg.ss_ramp_final_frac)
    mid = mg.ss_fraction(mg.cfg.ss_ramp_steps // 2)
    assert 0.0 < mid < mg.cfg.ss_ramp_final_frac


def test_training_loss_with_scheduled_sampling_finite_and_trains():
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).train()
    codes = _codes(cfg, B=2, F=5)
    act = torch.randn(2, 5, cfg.actuator_dim)
    loss = mg.training_loss(codes, act, generator=torch.Generator().manual_seed(3), ss_frac=0.5)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert mg.backbone.blocks[0].spatial.qkv.weight.grad is not None


def test_rollout_future_actuator_does_not_change_earlier_frames():
    # committing frame t must not depend on actuator_{>t+1} (causal control)
    cfg = _tiny()
    mg = MaskGITDynamics(cfg).eval()
    seed = _codes(cfg, B=1, F=cfg.k0_seed)
    act = torch.randn(1, cfg.k0_seed + cfg.n_predict, cfg.actuator_dim)
    g = 11
    base = mg.rollout(seed, act, generator=torch.Generator().manual_seed(g))
    act2 = act.clone(); act2[:, -1, :] += 5.0                        # perturb the LAST actuator
    pert = mg.rollout(seed, act2, generator=torch.Generator().manual_seed(g))
    # the FIRST generated frame (t=K0) cannot see the last actuator
    for m in cfg.modalities:
        assert torch.equal(base[m.name][:, cfg.k0_seed], pert[m.name][:, cfg.k0_seed])
