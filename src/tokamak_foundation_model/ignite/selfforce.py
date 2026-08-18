"""Self-Forcing Stage A: build training context from the model's OWN rollout.

Self-Forcing (arXiv 2506.08009) isolates the active ingredient with an ablation that
holds the loss fixed and varies only where the context comes from: teacher-forced 82.32,
diffusion-forced 82.76, self-rollout 84.31 (VBench). Self-Forcing++ (arXiv 2510.02283)
then shows synthetic corruption of the context is NOT a substitute — real rollout errors
have structured statistics (drift toward stasis) that random noise does not reproduce.

The rollout here runs under no_grad with a cheap few-step decode and every returned
tensor is detached: gradients flow only through the supervised prediction that follows,
exactly as Self-Forcing does by detaching its KV cache. Cost is therefore ~one extra
forward per rolled frame, not a backprop-through-time.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from .sampling import SamplerConfig


@torch.no_grad()
def rollout_context(model, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                    boundary: int, n_roll: int,
                    sampler: Optional[SamplerConfig] = None,
                    generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """Replace frames ``[boundary, boundary+n_roll)`` with the model's own committed codes.

    Frames before ``boundary`` stay ground truth (the real seed); frames after the rolled
    window are left untouched so the caller can still supervise against them.
    """
    cfg = model.cfg
    ref = codes[cfg.modalities[0].name]
    Fr = ref.shape[1]
    if boundary < 0:
        # A negative boundary would not error: `v[:, :-1]` is a legal slice, so the window would
        # silently be built on a truncated prefix and DUPLICATE real frames into the output.
        raise ValueError(f"boundary must be >= 0; got {boundary}")
    n_roll = max(0, min(n_roll, Fr - boundary))
    if n_roll == 0:
        return {k: v.detach() for k, v in codes.items()}

    was_training = model.training
    prev_steps = cfg.maskgit_decode_steps
    # Resolve the cheap-decode override BEFORE flipping the mode: the try/finally below only
    # covers what follows it, so a bad sf_decode_steps raising here would otherwise strand a
    # training model in eval().
    steps = max(1, int(getattr(cfg, "sf_decode_steps", 4)))
    model.eval()                                   # no dropout inside the rollout
    cfg.maskgit_decode_steps = steps
    try:
        # cache_enabled=False is load-bearing, and the reason is invisible from here: autocast's
        # weight cache x this no_grad rollout x gradient checkpointing break each other. Casts made
        # inside no_grad land in the ambient autocast region's cache DETACHED from the autograd
        # graph; the checkpointed supervised forward then recomputes against those stale casts and
        # the addmm grad path dies ("mat1 and mat2 shapes cannot be multiplied (NxD and 1xN)"), or,
        # with checkpointing off, SILENTLY drops the gradient of every parameter the rollout
        # touched. All three ingredients are required, so only the trainer's configuration hits it.
        # The nested context mirrors the ambient one so numerics are unchanged — it disables the
        # cache and nothing else, and the supervised pass outside keeps its cache.
        dev_type = ref.device.type
        ac_enabled = torch.is_autocast_enabled(dev_type)
        ac_dtype = torch.get_autocast_dtype(dev_type) if ac_enabled else None
        with torch.autocast(device_type=dev_type, enabled=ac_enabled,
                            dtype=ac_dtype, cache_enabled=False):
            ctx = {n: v[:, :boundary].clone() for n, v in codes.items()}
            for t in range(n_roll):
                nxt = model.generate_frame(ctx, actuators[:, : boundary + t + 1],
                                           generator=generator, sampler=sampler)
                ctx = {n: torch.cat([ctx[n], nxt[n].unsqueeze(1)], dim=1) for n in ctx}
    finally:
        cfg.maskgit_decode_steps = prev_steps
        if was_training:
            model.train()

    out = {}
    for n, v in codes.items():
        out[n] = torch.cat([ctx[n], v[:, boundary + n_roll:]], dim=1).detach()
    return out
