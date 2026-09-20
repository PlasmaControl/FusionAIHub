"""Seed assembly and paired real/proposed IGNITE rollout.

A design proposal only changes actuators; the plasma-state seed (the K0 real frames a
rollout starts from) and the ground truth to score against both come from the shot's own
frame-code cache. This module pairs two rollouts of the SAME dynamics model from the
SAME seed frames, one conditioned on the reference shot's measured actuators (`real`)
and one on the design's proposed actuators (`proposed`), so any divergence between them
is attributable to the actuator edit alone rather than to sampling noise: both draw from
an identical RNG stream (`torch.manual_seed(seed)` reset before each rollout), which
`MaskGITDynamics.rollout` consumes an identical number of draws from regardless of the
actuator conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tokamak_foundation_model.ignite import eval_dynamics
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig

from ..shotdb import ignite as shotdb_ignite


@dataclass
class SimulationArms:
    seed_frames: int  # k0, default 20
    predict_frames: int  # default 80
    real: dict[str, torch.Tensor]  # {m: (F, n_tok)}, predicted with reference actuators
    proposed: dict[str, torch.Tensor]  # same shape, proposed actuators
    gt: dict[str, torch.Tensor]  # reference cache codes over the same frames
    divergence_vs_real: dict[str, float]  # fraction of differing tokens, predicted only
    token_accuracy: dict[str, float]  # vs gt, predicted region only
    persistence_accuracy: dict[str, float]  # last seed frame repeated, vs gt


def load_dynamics(paths, device) -> tuple[torch.nn.Module, DynamicsConfig, int]:
    """Load the pinned bundle's dynamics checkpoint via the repo's own eval harness."""
    cfg = shotdb_ignite.model_cfg()
    ckpt = Path(shotdb_ignite.bundle_dir(paths)) / cfg["dynamics_file"]
    return eval_dynamics.load_model(ckpt, device)


def actuator_arms(
    reference_cache: dict, design_seed: dict, k0: int, n_predict: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (real, proposed) actuator trajectories over [0, k0+n_predict), float32.

    The seed frames [0, k0) are measured controls -- identical for both arms whatever
    the design proposes there -- so `proposed` is forced to agree with `real` over
    that prefix even though `design_seed["actuators"]` already carries the reference
    values there too; only [k0, k0+n_predict) can differ.
    """
    total = k0 + n_predict
    real = reference_cache["actuators"][:total].float()
    proposed = design_seed["actuators"][:total].float().clone()
    proposed[:k0] = real[:k0]
    return real, proposed


def _token_fraction_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a == b).float().mean().item()


def run_paired(
    model,
    cfg: DynamicsConfig,
    codes: dict,
    real_act: torch.Tensor,
    prop_act: torch.Tensor,
    *,
    seed: int,
    temperature: float = 1.0,
    decode_steps: int = 10,
) -> SimulationArms:
    """Roll out the real and proposed actuator arms from the same seed frames + RNG.

    Both rollouts consume an identical number of RNG draws (`rollout`'s decode loop is a
    fixed number of multinomial calls per frame), so resetting `torch.manual_seed(seed)`
    immediately before each call makes the two trajectories differ ONLY through the
    actuator conditioning -- a paired comparison, not one confounded by sampling noise.
    """
    k0, n_predict = cfg.k0_seed, cfg.n_predict
    names = [m.name for m in cfg.modalities]
    # generate_frame reads cfg.maskgit_decode_steps off the model's own bound config;
    # rollout() takes no decode-step argument, so this is the only way in.
    cfg.maskgit_decode_steps = decode_steps

    gt = {n: codes[n][: k0 + n_predict].long() for n in names}
    seed_codes = {n: codes[n][:k0].long().unsqueeze(0) for n in names}
    real_act = real_act.float().unsqueeze(0)
    prop_act = prop_act.float().unsqueeze(0)

    torch.manual_seed(seed)
    real_traj = model.rollout(
        seed_codes, real_act, n_predict=n_predict, temperature=temperature
    )
    torch.manual_seed(seed)
    prop_traj = model.rollout(
        seed_codes, prop_act, n_predict=n_predict, temperature=temperature
    )
    real = {n: real_traj[n][0] for n in names}
    proposed = {n: prop_traj[n][0] for n in names}

    divergence_vs_real, token_accuracy, persistence_accuracy = {}, {}, {}
    for n in names:
        r_pred, p_pred, g_pred = real[n][k0:], proposed[n][k0:], gt[n][k0:]
        divergence_vs_real[n] = 1.0 - _token_fraction_equal(r_pred, p_pred)
        token_accuracy[n] = _token_fraction_equal(r_pred, g_pred)
        persisted = gt[n][k0 - 1].unsqueeze(0).expand_as(g_pred)
        persistence_accuracy[n] = _token_fraction_equal(persisted, g_pred)

    return SimulationArms(
        seed_frames=k0,
        predict_frames=n_predict,
        real=real,
        proposed=proposed,
        gt=gt,
        divergence_vs_real=divergence_vs_real,
        token_accuracy=token_accuracy,
        persistence_accuracy=persistence_accuracy,
    )
