"""`eval_dynamics.load_model` must read the checkpoint's own actuator width, never a
`DynamicsConfig` default: `_ACT_SPEC` grew 70 -> 88 channels when `i_coil` was added,
and a default of 70 fails to load an 88-channel checkpoint with a
`backbone.act_embed.weight` size mismatch (job 5329757, all 4 checkpoints, 2026-08-23).
This pins the fix by building a tiny checkpoint with an 88-channel backbone and checking
it round-trips through `load_model`.
"""

from __future__ import annotations

import dataclasses

import torch

from tokamak_foundation_model.ignite import dynamics_config as dc
from tokamak_foundation_model.ignite import eval_dynamics, maskgit


def test_load_model_builds_the_checkpoints_actuator_width(tmp_path):
    mods = (
        dc.ModalitySpec("ece", "spectro", 4, 16),
        dc.ModalitySpec("mse", "slowts", 2, 16),
    )
    cfg = dc.DynamicsConfig(
        modalities=mods,
        d_model=32,
        depth=1,
        n_heads=2,
        actuator_dim=88,
        grad_checkpointing=False,
    )
    m = maskgit.MaskGITDynamics(cfg)
    ck = {
        "model": m.state_dict(),
        "cfg_depth": 1,
        "cfg_d_model": 32,
        "cfg_n_heads": 2,
        "modalities": [dataclasses.astuple(x) for x in mods],
        "step": 7,
    }
    p = tmp_path / "d.pt"
    torch.save(ck, p)
    model, cfg2, step = eval_dynamics.load_model(p, "cpu")
    assert cfg2.actuator_dim == 88 and step == 7
