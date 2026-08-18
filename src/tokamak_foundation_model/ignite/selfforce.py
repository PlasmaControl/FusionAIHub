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
    n_roll = max(0, min(n_roll, Fr - boundary))
    if n_roll == 0:
        return {k: v.detach() for k, v in codes.items()}

    was_training = model.training
    model.eval()                                   # no dropout inside the rollout
    prev_steps = cfg.maskgit_decode_steps
    cfg.maskgit_decode_steps = max(1, int(getattr(cfg, "sf_decode_steps", 4)))
    try:
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
