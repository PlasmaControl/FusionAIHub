"""Per-sample masked error metrics, trainer-faithful.

``per_sample_masked_mae`` is the per-window analogue of
``train_e2e_stage1.masked_mae`` (NaN-clean both sides, combine masks, sum /
mask-count). It additionally returns the mask weight so window-level values
re-aggregate *exactly* to the training-time batch metric:
``MAE_pool = Σ(mae_i · w_i) / Σ w_i``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _clean(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(t)
    return torch.where(finite, t, torch.zeros_like(t)), finite.float()


@torch.no_grad()
def per_sample_masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(mae, weight)`` per batch element; ``mae`` is NaN where weight == 0.

    Shapes must already agree — upstream code (trainer-faithful forward)
    truncates spectro targets to ``trunc_t``, so a mismatch here is a bug,
    not something to crop away silently.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {tuple(pred.shape)} vs "
                         f"target {tuple(target.shape)}")
    cleaned_pred, pred_mask = _clean(pred)
    cleaned_target, target_mask = _clean(target)
    if mask is not None:
        target_mask = target_mask * mask.float()
    combined = pred_mask * target_mask
    num = ((cleaned_pred - cleaned_target).abs() * combined).flatten(1).sum(1)
    den = combined.flatten(1).sum(1)
    mae = num / den.clamp_min(1.0)
    mae = torch.where(den > 0, mae, torch.full_like(mae, float("nan")))
    return mae, den


@torch.no_grad()
def aggregate_masked_mae(
    mae: torch.Tensor, weight: torch.Tensor
) -> float:
    """Pool per-window ``(mae, weight)`` back to the trainer's batch MAE."""
    ok = torch.isfinite(mae) & (weight > 0)
    if not ok.any():
        return float("nan")
    return float((mae[ok] * weight[ok]).sum() / weight[ok].sum())
