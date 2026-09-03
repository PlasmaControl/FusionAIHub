"""Byte-identical behaviour guard: Peter's model must keep running unchanged.

Every feature added by the rollout-quality work is flag-gated. With no flag set,
rollout and training_loss must reproduce the recorded reference EXACTLY, including
the RNG draw sequence. If this test fails, a default changed — that is a bug, not
a fixture to regenerate.
"""
from pathlib import Path

import torch

from tests.ignite.fixtures._make_golden import build

FIXTURE = Path(__file__).parent / "fixtures" / "phaseb_golden.pt"


def _ref():
    return torch.load(FIXTURE, weights_only=False)


def test_default_rollout_is_bit_identical():
    ref = _ref()
    cfg, model = build()
    g = torch.Generator().manual_seed(1)
    seed = {m.name: torch.randint(0, m.codebook_size, (1, cfg.k0_seed, m.n_tok), generator=g)
            for m in cfg.modalities}
    act = torch.randn(1, cfg.max_frames, cfg.actuator_dim, generator=g)
    roll = model.rollout(seed, act, n_predict=3, generator=torch.Generator().manual_seed(7))
    for name, want in ref["rollout_codes"].items():
        assert torch.equal(roll[name], want), f"{name}: default rollout changed"


def test_default_training_loss_is_bit_identical():
    ref = _ref()
    cfg, model = build()
    codes = {m.name: torch.randint(0, m.codebook_size, (2, 5, m.n_tok),
                                   generator=torch.Generator().manual_seed(3))
             for m in cfg.modalities}
    act = torch.randn(2, 5, cfg.actuator_dim, generator=torch.Generator().manual_seed(4))
    loss = model.training_loss(codes, act, generator=torch.Generator().manual_seed(5))
    assert float(loss) == ref["train_loss"], "default training loss changed"


def test_old_checkpoint_config_still_constructs():
    """A checkpoint saved before the new fields exist must still build a config."""
    from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
    old_payload_kw = dict(d_model=512, depth=8, n_heads=16, k0_seed=20, n_predict=80)
    cfg = DynamicsConfig(modalities=(ModalitySpec("mhr", "spectro", 192, 1000),),
                         **old_payload_kw)
    assert cfg.max_frames == 100 and cfg.tokens_per_frame == 192
