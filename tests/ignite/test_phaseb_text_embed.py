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


def _eval_loadable_model(seed=0, **cfg_overrides):
    """A model whose config uses ONLY the fields eval_dynamics.load_model actually
    reconstructs from a checkpoint (depth/d_model/n_heads/k0_seed/n_predict/modalities
    [+ text_embed_dim/text_dropout_p]) — everything else at DynamicsConfig's own default,
    so a real load_model() roundtrip does not hit an unrelated shape mismatch."""
    kw = dict(d_model=16, depth=2, n_heads=2, k0_seed=2, n_predict=3)
    kw.update(cfg_overrides)
    cfg = DynamicsConfig(modalities=MODS, **kw)
    torch.manual_seed(seed)
    model = MaskGITDynamics(cfg)
    return cfg, model
