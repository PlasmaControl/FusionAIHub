"""Seed assembly and ensembles of IGNITE rollouts for one design.

A design only changes actuators: the plasma-state seed (the K0 measured frames a rollout
starts from) and the measurement to score against both come from the shot's own frame-code
cache. `run_ensemble` rolls the dynamics model out several times per arm from those seed
frames.

Arms given the same seed share their random numbers. MaskGIT samples by Gumbel-max over
exponential variates, so two such arms draw the same token wherever their distributions are
close, and they part only where the actuators moved a distribution far enough. After that
the rollout carries any difference forward and amplifies it, so arms can also diverge for
reasons an edit did not cause. The null arm, the real actuators on fresh random numbers,
measures how far that alone moves an ensemble; an edit is resolved only when it moves the
ensemble mean well beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tokamak_foundation_model.ignite import eval_dynamics
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig
from tokamak_foundation_model.ignite.maskgit import placeholders

from ..shotdb import ignite as shotdb_ignite


@dataclass
class Ensemble:
    k0: int  # seed frames; frames [k0, F) are predicted
    gt: dict[str, torch.Tensor]  # {m: (F, n_tok)} the measured codes
    arms: dict[str, dict[str, torch.Tensor]]  # {arm: {m: (M, F, n_tok)}}
    seeds: dict[str, int]  # {arm: seed}; arms with one seed share random numbers
    held: tuple[str, ...]  # modalities held at their placeholder code
    batch: int  # members per batched rollout


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


def members_per_pass(cfg: DynamicsConfig, device, bf16: bool = False) -> int:
    """How many members one batched rollout can hold on ``device``.

    The KV cache dominates: 2 x depth x tokens per frame x frames x d_model values per
    member (15 GiB for v4 in fp32), next to 1-2 GiB of weights and activations.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return 1
    per = 2 * cfg.depth * cfg.tokens_per_frame * cfg.max_frames * cfg.d_model
    per *= 2 if bf16 else 4
    free, _ = torch.cuda.mem_get_info(device)
    return max(1, int(0.9 * free / (1.05 * per)))


def run_ensemble(
    model,
    cfg: DynamicsConfig,
    codes: dict,
    arms: dict[str, tuple[torch.Tensor, int]],
    members: int = 8,
    *,
    sampler=None,
    batch: int | None = None,
    bf16: bool = False,
) -> Ensemble:
    """``members`` rollouts per arm from the same K0 seed frames.

    ``arms`` maps a name to ``(actuators (F, A), seed)``. Members go through the model
    ``batch`` at a time, the j-th batch on a generator seeded ``seed + j * batch``, so arms
    with the same seed and batch share every random number. Modalities whose seed frames
    never change are held at that placeholder code (`maskgit.placeholders`). The rollouts
    use the exact KV cache; ``bf16`` runs them under bfloat16 autocast.
    """
    k0, n_predict = cfg.k0_seed, cfg.n_predict
    total = k0 + n_predict
    names = [m.name for m in cfg.modalities]
    _check_frame_counts(codes, names, total)
    dev = _model_device(model)
    seed_codes = {n: codes[n][:k0].long().unsqueeze(0).to(dev) for n in names}
    held = placeholders(seed_codes)
    batch = batch or min(members, members_per_pass(cfg, dev, bf16))
    out = {}
    for arm, (act, seed) in arms.items():
        if act.shape[0] < total:
            raise ValueError(
                f"the {arm} actuators cover {act.shape[0]} frames; need k0+n_predict={total}"
            )
        act = act[:total].float().unsqueeze(0).to(dev)
        rolls = []
        for i0 in range(0, members, batch):
            b = min(batch, members - i0)
            gen = torch.Generator(dev).manual_seed(seed + i0)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                traj = model.rollout(
                    {n: c.expand(b, -1, -1) for n, c in seed_codes.items()},
                    act.expand(b, -1, -1),
                    n_predict=n_predict,
                    generator=gen,
                    sampler=sampler,
                    kv_cache=True,
                    hold={n: r.expand(b) for n, r in held.items()},
                )
            rolls.append({n: traj[n].cpu() for n in names})
        out[arm] = {n: torch.cat([r[n] for r in rolls]) for n in names}
    return Ensemble(
        k0=k0,
        gt={n: codes[n][:total].long() for n in names},
        arms=out,
        seeds={arm: seed for arm, (_, seed) in arms.items()},
        held=tuple(held),
        batch=batch,
    )


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


def _model_device(model) -> torch.device:
    """The device the model's weights live on; cpu for a model with no parameters."""
    for p in model.parameters():
        return p.device
    return torch.device("cpu")
