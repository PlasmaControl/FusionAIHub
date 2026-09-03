"""Self-consistency scoring for generated trajectories.

The discrete counterpart of "the teacher scores this window": re-mask a fraction of a
trajectory's OWN tokens and measure the frozen model's cross-entropy at reproducing
them. It needs no ground truth, so it works at inference time (best-of-N reranking)
and doubles as a reward signal for post-training.

Lower is better. Averaging several independent mask draws keeps the estimate stable —
a single draw is noisy because the mask decides which tokens are being asked about.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .sampling import SamplerConfig


@torch.no_grad()
def masked_pseudo_likelihood(model, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                             frames: Optional[slice] = None, mask_frac: float = 0.3,
                             n_draws: int = 4,
                             generator: Optional[torch.Generator] = None,
                             text: Optional[torch.Tensor] = None) -> float:
    """Mean masked cross-entropy of ``codes`` under ``model``. Lower = more self-consistent.

    ``frames`` restricts scoring to a window (e.g. the predicted region only); the whole
    sequence is still fed as context so the score is conditioned on the real seed.
    """
    cfg = model.cfg
    ref = codes[cfg.modalities[0].name]
    Fr = ref.shape[1]
    sel = torch.zeros(Fr, dtype=torch.bool, device=ref.device)
    sel[frames if frames is not None else slice(0, Fr)] = True
    total = 0.0
    for _ in range(max(1, n_draws)):
        masked, mask = {}, {}
        for m in cfg.modalities:
            c = codes[m.name]
            r = torch.rand(c.shape, generator=generator, device=c.device)
            mk = (r < mask_frac) & sel.view(1, Fr, 1)
            mk[:, :, 0] |= ~mk.any(dim=-1) & sel.view(1, Fr)   # >=1 target per scored frame
            masked[m.name] = torch.where(mk, torch.full_like(c, model.backbone.tok.mask_ids[m.name]), c)
            mask[m.name] = mk
        h = model.backbone.encode(masked, actuators, text=text)
        mlog = model.backbone.tok.masked_logits(h, mask)
        ce, n = 0.0, 0
        for m in cfg.modalities:
            if not bool(mask[m.name].any()):
                continue
            ce += float(F.cross_entropy(mlog[m.name], codes[m.name][mask[m.name]]))
            n += 1
        total += ce / max(n, 1)
    return total / max(1, n_draws)


@torch.no_grad()
def best_of_n(model, seed_codes: Dict[str, torch.Tensor], actuators: torch.Tensor, n: int,
              n_predict: Optional[int] = None, sampler: Optional[SamplerConfig] = None,
              generator: Optional[torch.Generator] = None,
              score_frames: Optional[slice] = None,
              text: Optional[torch.Tensor] = None
              ) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """Roll out ``n`` candidates and return the most self-consistent one plus all scores."""
    cfg = model.cfg
    n_predict = cfg.n_predict if n_predict is None else n_predict
    K0 = seed_codes[cfg.modalities[0].name].shape[1]
    win = score_frames if score_frames is not None else slice(K0, K0 + n_predict)
    best, best_score, scores = None, float("inf"), []
    for _ in range(n):
        traj = model.rollout(seed_codes, actuators, n_predict=n_predict,
                             generator=generator, sampler=sampler, text=text)
        s = masked_pseudo_likelihood(model, traj, actuators[:, : K0 + n_predict],
                                     frames=win, generator=generator, text=text)
        scores.append(s)
        if s < best_score:
            best, best_score = traj, s
    return best, scores
