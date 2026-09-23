"""`eval_dynamics.load_model` reads the architecture from the weights and every saved
training knob from the payload, never from a `DynamicsConfig` default: `_ACT_SPEC` grew
70 -> 88 channels when `i_coil` was added, and a default of 70 failed to load an 88-channel
checkpoint (job 5329757, all 4 checkpoints, 2026-08-23); `lag_embed_k` and `act_cross_attn`
failed the same way.
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


def _save(tmp_path, cfg, **payload):
    m = maskgit.MaskGITDynamics(cfg)
    ck = {"model": m.state_dict(), "cfg_depth": cfg.depth, "cfg_d_model": cfg.d_model,
          "cfg_n_heads": cfg.n_heads,
          "modalities": [dataclasses.astuple(x) for x in cfg.modalities], **payload}
    p = tmp_path / "d.pt"
    torch.save(ck, p)
    return p


def test_load_model_rebuilds_the_shapes_the_weights_have(tmp_path):
    mods = (dc.ModalitySpec("ece", "spectro", 4, 16),)
    cfg = dc.DynamicsConfig(modalities=mods, d_model=32, depth=1, n_heads=2, actuator_dim=6,
                            lag_embed_k=2, act_cross_attn=True, text_embed_dim=5)
    _, got, _ = eval_dynamics.load_model(_save(tmp_path, cfg), "cpu")
    assert (got.actuator_dim, got.lag_embed_k, got.act_cross_attn, got.text_embed_dim) == (
        6, 2, True, 5)


def test_load_model_restores_every_saved_training_knob(tmp_path):
    mods = (dc.ModalitySpec("ece", "spectro", 4, 16),)
    cfg = dc.DynamicsConfig(modalities=mods, d_model=32, depth=1, n_heads=2, k0_seed=3,
                            n_predict=4)
    p = _save(tmp_path, cfg, cfg_ctf_frac=0.3, cfg_sf_frames=2, cfg_ss_final_frac=0.5,
              cfg_ss_ramp_steps=100, cfg_k0=3, cfg_n_predict=4,
              cfg_modality_loss_weight="tokens", cfg_dropout=0.2)
    _, got, _ = eval_dynamics.load_model(p, "cpu")
    assert (got.ctf_frac, got.sf_frames, got.ss_ramp_final_frac, got.ss_ramp_steps) == (
        0.3, 2, 0.5, 100)
    assert (got.k0_seed, got.n_predict, got.modality_loss_weight) == (3, 4, "tokens")
    assert got.dropout == 0.0 and not got.grad_checkpointing       # inference settings
