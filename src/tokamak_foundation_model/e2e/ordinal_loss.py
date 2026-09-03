"""Soft / ordinal cross-entropy for FSQ code prediction (Step-3 candidate loss).

Motivation (mode-audit 2026-07-13): FSQ code jitter is dominated by ADJACENT-LEVEL
dither — ±1-level-tolerant stability ~0.90 vs exact ~0.6 on the s16 codec. Standard
hard CE penalizes a k±1 prediction as fully as a k±8 one, so it fights the (irreducible,
quantization-boundary) dither. An ORDINAL target gives partial credit to neighbours:

    q(k)   = 1 - 2*eps        (true level)
    q(k±1) = eps              (each neighbour)
  edge handling: the out-of-range neighbour's mass is CLAMPED onto the true level
  (k=0 -> q=[1-eps, eps, ...]; k=L-1 -> q=[..., eps, 1-eps]); then renormalized (no-op
  in the standard case, kept for safety). Loss = CE(logits, q) = -sum_j q_j log p_j.

Also provides ``tol1_codeacc`` = the training-log metric the audit gate reads:
fraction of (token, dim) whose argmax is within +-1 level of the target.

Pure, dependency-light (torch only), unit-tested in analysis/mode_audit/test_ordinal_ce.py.
NOT yet wired into the trainer — Step 3 imports it under whichever branch fires.
"""
from typing import Optional

import torch
import torch.nn.functional as F


def build_ordinal_target(codes: torch.Tensor, num_levels: int, eps: float = 0.1) -> torch.Tensor:
    """codes (..., ) int in [0, L) -> soft target (..., L). [eps, 1-2eps, eps] on
    [k-1, k, k+1]; out-of-range neighbour mass clamped onto k; renormalized."""
    L = int(num_levels)
    idx = codes.long().clamp(0, L - 1)
    q = torch.zeros(*idx.shape, L, device=codes.device, dtype=torch.float32)
    e = torch.full_like(idx, eps, dtype=torch.float32).unsqueeze(-1)
    q.scatter_add_(-1, idx.unsqueeze(-1), torch.full_like(idx, 1.0 - 2.0 * eps, dtype=torch.float32).unsqueeze(-1))
    q.scatter_add_(-1, (idx - 1).clamp(0, L - 1).unsqueeze(-1), e)   # k=0 -> folds onto level 0
    q.scatter_add_(-1, (idx + 1).clamp(0, L - 1).unsqueeze(-1), e)   # k=L-1 -> folds onto level L-1
    return q / q.sum(-1, keepdim=True)


def soft_ordinal_ce(
    logits: torch.Tensor,          # (..., L)
    codes: torch.Tensor,           # (...,) int target levels
    eps: float = 0.1,
    weight: Optional[torch.Tensor] = None,   # (...,) per-element weight (e.g. class weight at true k)
    reduction: str = "mean",
) -> torch.Tensor:
    """Ordinal (soft-neighbour) cross-entropy. Reduces to hard CE as eps->0."""
    L = logits.shape[-1]
    q = build_ordinal_target(codes, L, eps).to(logits.dtype)
    logp = F.log_softmax(logits, dim=-1)
    per = -(q * logp).sum(-1)                                        # (...,)
    if weight is not None:
        per = per * weight
        if reduction == "mean":
            return per.sum() / (weight.sum() + 1e-8)
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    return per


@torch.no_grad()
def tol1_codeacc(logits: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """Fraction of elements whose argmax level is within +-1 of the target (the gate metric)."""
    pred = logits.argmax(dim=-1)
    return ((pred - codes.long()).abs() <= 1).float().mean()


@torch.no_grad()
def exact_codeacc(logits: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    return (pred == codes.long()).float().mean()
