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

    `reference_cache` must already be windowed to the SAME `[context:display_end]`
    slice the design seed was cut from (`program.py`'s `_evaluate`/`export_ignite`) --
    this function does not re-align two caches taken over different windows, it only
    truncates both to `k0+n_predict` frames from index 0.
    """
    total = k0 + n_predict
    real_full = reference_cache.get("actuators")
    if real_full is None:
        raise ValueError(
            "reference_cache has no actuators; the real arm needs measured controls"
        )
    prop_full = design_seed.get("actuators")
    if prop_full is None:
        raise ValueError(
            "design_seed has no actuators; the proposed arm needs measured controls"
        )
    for label, arr in (("reference_cache", real_full), ("design_seed", prop_full)):
        if arr.shape[0] < total:
            raise ValueError(
                f"{label} actuators has {arr.shape[0]} frames; need at least "
                f"k0+n_predict={total}"
            )
    real = real_full[:total].float()
    proposed = prop_full[:total].float().clone()
    proposed[:k0] = real[:k0]
    return real, proposed


def _token_fraction_equal(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a == b).float().mean().item()


def _check_frame_counts(codes: dict, names: list[str], needed: int) -> None:
    for n in names:
        if n not in codes:
            raise ValueError(
                f"codes is missing modality {n!r} required by cfg.modalities"
            )
        have = codes[n].shape[0]
        if have < needed:
            raise ValueError(
                f"codes[{n!r}] has {have} frames; need at least k0+n_predict={needed}"
            )


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
    total = k0 + n_predict
    names = [m.name for m in cfg.modalities]
    _check_frame_counts(codes, names, total)
    if real_act.shape[0] < total or prop_act.shape[0] < total:
        raise ValueError(
            f"actuators must cover k0+n_predict={total} frames; got "
            f"real={real_act.shape[0]}, proposed={prop_act.shape[0]}"
        )

    gt = {n: codes[n][:total].long() for n in names}
    seed_codes = {n: codes[n][:k0].long().unsqueeze(0) for n in names}
    real_act = real_act.float().unsqueeze(0)
    prop_act = prop_act.float().unsqueeze(0)

    # generate_frame reads cfg.maskgit_decode_steps off the model's own bound config
    # (model.cfg IS this cfg object); rollout() takes no decode-step argument, so this
    # is the only way in. Restored in `finally` -- a shared model/cfg (e.g. one loaded
    # in a long-lived route) must not carry one caller's decode_steps into the next.
    original_decode_steps = cfg.maskgit_decode_steps
    cfg.maskgit_decode_steps = decode_steps
    try:
        torch.manual_seed(seed)
        real_traj = model.rollout(
            seed_codes, real_act, n_predict=n_predict, temperature=temperature
        )
        torch.manual_seed(seed)
        prop_traj = model.rollout(
            seed_codes, prop_act, n_predict=n_predict, temperature=temperature
        )
    finally:
        cfg.maskgit_decode_steps = original_decode_steps

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
