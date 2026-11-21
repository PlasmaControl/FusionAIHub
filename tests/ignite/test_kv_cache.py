"""The KV cache changes what a rollout costs, never what it produces; the clamp holds absent
diagnostics at their placeholder code."""

import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
from tokamak_foundation_model.ignite.maskgit import MaskGITDynamics, placeholders
from tokamak_foundation_model.ignite.sampling import SamplerConfig


def _model(**kw):
    cfg = DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4)),
        d_model=16, depth=2, n_heads=2, ffn_mult=2, k0_seed=2, n_predict=4,
        maskgit_decode_steps=4, actuator_dim=6, **kw)
    torch.manual_seed(0)
    return MaskGITDynamics(cfg).eval(), cfg


def _inputs(cfg, batch=2):
    g = torch.Generator().manual_seed(1)
    codes = {m.name: torch.randint(0, m.codebook_size, (batch, cfg.k0_seed, m.n_tok), generator=g)
             for m in cfg.modalities}
    return codes, torch.randn(batch, cfg.max_frames, cfg.actuator_dim, generator=g)


def _rollout(model, codes, act, **kw):
    return model.rollout(codes, act, generator=torch.Generator().manual_seed(7), **kw)


def test_placeholders_are_the_rows_that_never_change():
    codes = {"a": torch.tensor([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 4]]]),
             "b": torch.tensor([[[0, 1], [1, 0]], [[2, 2], [3, 3]]])}
    held = placeholders(codes)
    assert set(held) == {"a"}
    assert held["a"].tolist() == [True, False]
