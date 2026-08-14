"""Phase-B backbone: shapes, depth knob, and the temporal + actuator CAUSALITY the
controllability claim rests on (frame t must not see frame/actuator > t)."""

import torch

from tokamak_foundation_model.ignite.dynamics import DynamicsBackbone, FactorizedSTBlock
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec


def _tiny(depth=2):
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4)),
        d_model=16, depth=depth, n_heads=2, ffn_mult=2,
        k0_seed=2, n_predict=4, actuator_dim=6,
    )


def _inputs(cfg, B=2, F=5):
    codes = {m.name: torch.randint(0, m.codebook_size, (B, F, m.n_tok)) for m in cfg.modalities}
    act = torch.randn(B, F, cfg.actuator_dim)
    return codes, act


def test_forward_shapes():
    cfg = _tiny()
    model = DynamicsBackbone(cfg).eval()
    codes, act = _inputs(cfg, B=2, F=5)
    logits = model(codes, act)
    assert set(logits) == {"a", "b"}
    assert logits["a"].shape == (2, 5, 3, 5)
    assert logits["b"].shape == (2, 5, 2, 4)


def test_depth_is_configurable():
    for k in (1, 3, 6):
        m = DynamicsBackbone(_tiny(depth=k))
        assert len(m.blocks) == k and all(isinstance(b, FactorizedSTBlock) for b in m.blocks)


def test_temporal_attention_is_causal():
    cfg = _tiny()
    model = DynamicsBackbone(cfg).eval()
    codes, act = _inputs(cfg, B=2, F=5)
    with torch.no_grad():
        base = model(codes, act)
        codes2 = {k: v.clone() for k, v in codes.items()}
        codes2["a"][:, -1, :] = (codes2["a"][:, -1, :] + 1) % 5   # perturb the LAST frame
        pert = model(codes2, act)
    # frame 0 cannot see the future -> unchanged; the perturbed frame itself DID change
    assert torch.allclose(base["a"][:, 0], pert["a"][:, 0], atol=1e-5)
    assert torch.allclose(base["b"][:, 0], pert["b"][:, 0], atol=1e-5)
    assert not torch.allclose(base["a"][:, -1], pert["a"][:, -1], atol=1e-5)


def test_actuator_conditioning_is_causal_and_effective():
    cfg = _tiny()
    model = DynamicsBackbone(cfg).eval()
    codes, act = _inputs(cfg, B=2, F=5)
    with torch.no_grad():
        base = model(codes, act)
        # perturb the LAST actuator -> earlier frames unaffected (causal)
        a_future = act.clone(); a_future[:, -1, :] += 3.0
        pert_f = model(codes, a_future)
        assert torch.allclose(base["a"][:, 0], pert_f["a"][:, 0], atol=1e-5)
        # perturb frame-0 actuator -> frame-0 output DOES change (conditioning is effective)
        a_now = act.clone(); a_now[:, 0, :] += 3.0
        pert_n = model(codes, a_now)
        assert not torch.allclose(base["a"][:, 0], pert_n["a"][:, 0], atol=1e-5)


def test_gradient_flows_end_to_end():
    cfg = _tiny()
    model = DynamicsBackbone(cfg).train()
    codes, act = _inputs(cfg, B=2, F=5)
    logits = model(codes, act)
    loss = sum(l.float().pow(2).mean() for l in logits.values())
    loss.backward()
    g = model.blocks[0].spatial.qkv.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # actuator embedding receives gradient (conditioning is trained)
    ga = model.act_embed.weight.grad
    assert ga is not None and torch.isfinite(ga).all() and ga.abs().sum() > 0
