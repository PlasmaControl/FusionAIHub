"""Flag-gated per-shot text conditioning: additive like actuators, off by default.

``cfg.text_embed_dim == 0`` (the default) must reproduce today's behaviour exactly — no
``text_embed`` module, no RNG draws, bit-identical to the golden fixture (see
test_phaseb_compat.py). These tests cover the flag turned ON: shapes, errors, that the
conditioning actually changes outputs, that ``drop_text`` / ``text_dropout_p`` produce the
same null branch, and that a checkpoint payload with the new keys round-trips.
"""
from pathlib import Path

import pytest
import torch

from tests.ignite.fixtures._make_golden import build, CFG_KW, MODS
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics

FIXTURE = Path(__file__).parent / "fixtures" / "phaseb_golden.pt"


def _ref():
    return torch.load(FIXTURE, weights_only=False)


def _build_text_model(seed=0, **cfg_overrides):
    kw = dict(CFG_KW)
    kw.update(cfg_overrides)
    cfg = DynamicsConfig(modalities=MODS, **kw)
    torch.manual_seed(seed)
    model = MaskGITDynamics(cfg)
    return cfg, model


def test_off_by_default_state_dict_and_loss():
    ref = _ref()
    cfg, model = build()
    assert not any("text_embed" in k for k in model.state_dict().keys())

    g = torch.Generator().manual_seed(1)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok), generator=g)
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim, generator=g)
    roll = model.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(7))
    for name, want in ref["rollout_codes"].items():
        assert torch.equal(roll[name], want), f"{name}: default rollout changed"

    codes = {m.name: torch.randint(0, m.codebook_size, (2, 5, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    tact = torch.randn(2, 5, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    loss = model.training_loss(codes, tact, generator=torch.Generator().manual_seed(5))
    assert float(loss) == ref["train_loss"], "default training loss changed"


def test_error_cases():
    _, model_on = _build_text_model(text_embed_dim=8)
    _, model_off = build()
    B, Fr = 2, 5
    codes = {m.name: torch.randint(0, m.codebook_size, (B, Fr, m.n_tok))
             for m in model_on.cfg.modalities}
    act = torch.randn(B, Fr, model_on.cfg.actuator_dim)

    with pytest.raises(ValueError):
        model_on.backbone.encode(codes, act, text=None)

    with pytest.raises(ValueError):
        model_off.backbone.encode(codes, act, text=torch.randn(B, 8))

    with pytest.raises(ValueError):
        model_on.backbone.encode(codes, act, text=torch.randn(B, 3))  # wrong dim
    with pytest.raises(ValueError):
        model_on.backbone.encode(codes, act, text=torch.randn(B + 1, 8))  # wrong batch


def test_text_conditioning_effective():
    B, Fr = 2, 5
    cfg, model = _build_text_model(seed=42, text_embed_dim=8)
    codes = {m.name: torch.randint(0, m.codebook_size, (B, Fr, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    act = torch.randn(B, Fr, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))

    t1 = torch.nn.functional.normalize(torch.randn(B, 8, generator=torch.Generator().manual_seed(11)), dim=-1)
    t2 = torch.nn.functional.normalize(torch.randn(B, 8, generator=torch.Generator().manual_seed(12)), dim=-1)

    model.eval()
    loss1 = model.training_loss(codes, act, text=t1, generator=torch.Generator().manual_seed(5))
    loss2 = model.training_loss(codes, act, text=t2, generator=torch.Generator().manual_seed(5))
    loss1b = model.training_loss(codes, act, text=t1, generator=torch.Generator().manual_seed(5))
    assert float(loss1) != float(loss2)
    assert float(loss1) == float(loss1b)

    seed_codes = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok),
                                        generator=torch.Generator().manual_seed(1))
                  for m in cfg.modalities}
    ract = torch.randn(1, cfg.max_frames, cfg.actuator_dim, generator=torch.Generator().manual_seed(2))
    rt1 = t1[:1]
    rt2 = t2[:1]
    roll1 = model.rollout(seed_codes, ract, n_predict=3, text=rt1,
                          generator=torch.Generator().manual_seed(7))
    roll2 = model.rollout(seed_codes, ract, n_predict=3, text=rt2,
                          generator=torch.Generator().manual_seed(7))
    roll1b = model.rollout(seed_codes, ract, n_predict=3, text=rt1,
                           generator=torch.Generator().manual_seed(7))
    differ = any(not torch.equal(roll1[n], roll2[n]) for n in roll1)
    assert differ
    for n in roll1:
        assert torch.equal(roll1[n], roll1b[n])


def test_null_consistency():
    """``drop_text`` must fully ignore the text value passed in (like drop_actuators does),
    and ``text_dropout_p == 1.0`` in train mode must do the same regardless of RNG stream —
    every keep-mask draw comes out False, so which text was passed cannot matter.

    ``text_embed`` is bias=False (controller ruling), so all three null paths coincide
    EXACTLY at t=0: a zeros-INPUT encode, a drop_text=True encode, and a real-text encode
    under text_dropout_p==1.0 must all be bit-identical to each other."""
    B, Fr = 2, 5
    cfg, model = _build_text_model(seed=7, text_embed_dim=8)
    codes = {m.name: torch.randint(0, m.codebook_size, (B, Fr, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    act = torch.randn(B, Fr, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    t1 = torch.randn(B, 8, generator=torch.Generator().manual_seed(9))
    t2 = torch.randn(B, 8, generator=torch.Generator().manual_seed(10))

    model.eval()
    h_drop1 = model.backbone.encode(codes, act, text=t1, drop_text=True)
    h_drop2 = model.backbone.encode(codes, act, text=t2, drop_text=True)
    h_zero_input = model.backbone.encode(codes, act, text=torch.zeros(B, 8))
    assert torch.equal(h_drop1, h_drop2), "dropped text must not influence the hidden states"
    assert not torch.equal(model.backbone.encode(codes, act, text=t1), h_drop1)
    # bias=False: a zeros-INPUT encode (what the dataset feeds for undocumented shots) must
    # be bit-identical to the trained null (drop_text=True) — no bias term to distinguish them.
    assert torch.equal(h_zero_input, h_drop1)

    # text_dropout_p == 1.0 in train mode zeroes text for EVERY sample regardless of which
    # text was passed: seed the generator identically before each call so the keep-mask draw
    # (the only RNG the text path touches) lines up, then vary only the text argument.
    cfg2, model2 = _build_text_model(seed=7, text_embed_dim=8, text_dropout_p=1.0)
    model2.train()
    loss_t1 = model2.training_loss(codes, act, text=t1,
                                   generator=torch.Generator().manual_seed(5))
    loss_t2 = model2.training_loss(codes, act, text=t2,
                                   generator=torch.Generator().manual_seed(5))
    assert float(loss_t1) == float(loss_t2)


def test_cfg_uncond_drops_text():
    """The CFG unconditional branch (drop_actuators=True) must ALSO drop text: fully dropping
    both must be invariant to the text value, and must differ from the conditional pass and
    from an actuators-only-dropped pass (which still lets text through). With bias=False the
    fully-unconditional branch must also coincide exactly with a zeros-text-INPUT encode."""
    B, Fr = 2, 5
    cfg, model = _build_text_model(seed=3, text_embed_dim=8)
    model.eval()
    codes = {m.name: torch.randint(0, m.codebook_size, (B, Fr, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    act = torch.randn(B, Fr, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    t1 = torch.randn(B, 8, generator=torch.Generator().manual_seed(9))
    t2 = torch.randn(B, 8, generator=torch.Generator().manual_seed(10))

    fully_uncond1 = model.backbone.encode(codes, act, drop_actuators=True, text=t1, drop_text=True)
    fully_uncond2 = model.backbone.encode(codes, act, drop_actuators=True, text=t2, drop_text=True)
    cond = model.backbone.encode(codes, act, text=t1)
    actuators_only_dropped = model.backbone.encode(codes, act, drop_actuators=True, text=t1)
    uncond_zero_input = model.backbone.encode(codes, act, drop_actuators=True, text=torch.zeros(B, 8))

    assert torch.equal(fully_uncond1, fully_uncond2), "text value must not leak through drop_text"
    assert not torch.equal(fully_uncond1, cond)
    assert not torch.equal(fully_uncond1, actuators_only_dropped)
    assert torch.equal(fully_uncond1, uncond_zero_input)


def test_checkpoint_roundtrip():
    cfg, model = _build_text_model(seed=1, text_embed_dim=8, text_dropout_p=0.1)
    payload = {
        "model": model.state_dict(),
        "cfg_depth": cfg.depth,
        "cfg_d_model": cfg.d_model,
        "cfg_n_heads": cfg.n_heads,
        "cfg_k0": cfg.k0_seed,
        "cfg_n_predict": cfg.n_predict,
        "modalities": cfg.modalities,
        "cfg_text_embed_dim": 8,
        "cfg_text_dropout_p": 0.1,
    }
    rebuilt_cfg = DynamicsConfig(
        modalities=payload["modalities"],
        depth=payload["cfg_depth"], d_model=payload["cfg_d_model"],
        n_heads=payload["cfg_n_heads"], k0_seed=payload["cfg_k0"],
        n_predict=payload["cfg_n_predict"],
        actuator_dim=cfg.actuator_dim, ffn_mult=cfg.ffn_mult,
        maskgit_decode_steps=cfg.maskgit_decode_steps,
        text_embed_dim=payload["cfg_text_embed_dim"],
        text_dropout_p=payload["cfg_text_dropout_p"],
    )
    rebuilt = MaskGITDynamics(rebuilt_cfg)
    rebuilt.load_state_dict(payload["model"], strict=True)

    # a payload WITHOUT text keys builds a dim=0 model whose strict load still succeeds
    cfg0, model0 = build()
    payload0 = {
        "model": model0.state_dict(),
        "cfg_depth": cfg0.depth, "cfg_d_model": cfg0.d_model,
        "cfg_n_heads": cfg0.n_heads, "cfg_k0": cfg0.k0_seed,
        "cfg_n_predict": cfg0.n_predict, "modalities": cfg0.modalities,
    }
    rebuilt_cfg0 = DynamicsConfig(
        modalities=payload0["modalities"],
        depth=payload0["cfg_depth"], d_model=payload0["cfg_d_model"],
        n_heads=payload0["cfg_n_heads"], k0_seed=payload0["cfg_k0"],
        n_predict=payload0["cfg_n_predict"],
        actuator_dim=cfg0.actuator_dim, ffn_mult=cfg0.ffn_mult,
        maskgit_decode_steps=cfg0.maskgit_decode_steps,
        text_embed_dim=payload0.get("cfg_text_embed_dim", 0),
        text_dropout_p=payload0.get("cfg_text_dropout_p", 0.0),
    )
    rebuilt0 = MaskGITDynamics(rebuilt_cfg0)
    rebuilt0.load_state_dict(payload0["model"], strict=True)
