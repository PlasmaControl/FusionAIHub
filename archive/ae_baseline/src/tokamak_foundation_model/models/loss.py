import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedL1Loss(nn.Module):
    """L1 loss that ignores zero-padded time steps and optionally missing elements.

    Expects tensors of shape ``(B, C, T)`` (time-series) or
    ``(B, C, F, T)`` (spectrograms).  For each sample in the batch the last
    dimension is masked to ``valid_lengths[b]`` frames; positions beyond that
    are excluded from the mean.
    """

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None,
            element_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_lengths is None and element_mask is None:
            return F.l1_loss(output, target)

        mask = torch.ones_like(output)

        if valid_lengths is not None:
            T = output.shape[-1]
            t_idx = torch.arange(T, device=output.device)
            time_mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()
            for _ in range(output.dim() - 2):
                time_mask = time_mask.unsqueeze(1)
            mask = mask * time_mask

        if element_mask is not None:
            mask = mask * element_mask.float()

        return ((output - target).abs() * mask).sum() / mask.sum().clamp(min=1)

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores zero-padded time steps and optionally missing elements.

    Supports two complementary masking modes that can be used together:

    * **valid_lengths** — ``[B]`` long tensor: masks out padding at the end
      of the time axis (last dim).
    * **element_mask** — bool tensor broadcastable to ``(B, C, ..., T)``:
      ``True`` marks valid elements, ``False`` marks missing data (e.g.
      zero-valued measurements that should be excluded from the loss).
    """

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None,
            element_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_lengths is None and element_mask is None:
            return F.mse_loss(output, target)

        # Start with an all-ones mask
        mask = torch.ones_like(output)

        # Apply time-padding mask from valid_lengths
        if valid_lengths is not None:
            T = output.shape[-1]
            t_idx = torch.arange(T, device=output.device)
            time_mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()  # [B, T]
            for _ in range(output.dim() - 2):
                time_mask = time_mask.unsqueeze(1)
            mask = mask * time_mask

        # Apply per-element mask (e.g. zero_is_missing)
        if element_mask is not None:
            mask = mask * element_mask.float()

        return ((output - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


class MaskedHuberLoss(nn.Module):
    """Huber loss that ignores zero-padded time steps. Same interface as MaskedMSELoss.

    Parameters
    ----------
    delta : float
        Threshold between quadratic and linear regimes. Default ``1.0``.
    """

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None,
            element_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_lengths is None and element_mask is None:
            return F.huber_loss(output, target, delta=self.delta)

        mask = torch.ones_like(output)

        if valid_lengths is not None:
            T = output.shape[-1]
            t_idx = torch.arange(T, device=output.device)
            time_mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()
            for _ in range(output.dim() - 2):
                time_mask = time_mask.unsqueeze(1)
            mask = mask * time_mask

        if element_mask is not None:
            mask = mask * element_mask.float()

        loss = F.huber_loss(output, target, reduction="none", delta=self.delta)
        return (loss * mask).sum() / mask.sum().clamp(min=1)


class MaskedRelativeMSELoss(nn.Module):
    """Relative MSE loss that upweights high-amplitude samples.

    Computes ``(recon - target)² / (|target| + eps)²`` so the error is
    normalised by the local target magnitude.  High-amplitude targets
    contribute proportionally more to the gradient, counteracting the
    amplitude compression from BatchNorm in the encoder bottleneck.

    Parameters
    ----------
    eps : float
        Stability constant added to the denominator to avoid division by
        zero near flat regions.  Default ``1.0`` keeps the loss close to
        plain MSE for small target values while rescaling large ones.
    """

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None,
            element_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sq_err = (output - target) ** 2
        weight = 1.0 / (target.abs() + self.eps) ** 2

        if valid_lengths is None and element_mask is None:
            return (sq_err * weight).mean()

        mask = torch.ones_like(output)

        if valid_lengths is not None:
            T = output.shape[-1]
            t_idx = torch.arange(T, device=output.device)
            time_mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()
            for _ in range(output.dim() - 2):
                time_mask = time_mask.unsqueeze(1)
            mask = mask * time_mask

        if element_mask is not None:
            mask = mask * element_mask.float()

        return (sq_err * weight * mask).sum() / mask.sum().clamp(min=1)


class DictMSELoss(nn.Module):
    """MSE loss for dict outputs: averages MSE across all target keys."""

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        losses = []
        for key in outputs:
            if key in targets:
                losses.append(F.mse_loss(outputs[key], targets[key]))
        return torch.stack(losses).mean()

class WeightedMSELoss(nn.Module): # For video reconstruction
    def __init__(self, reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B,T,H,W) or broadcast-compatible
        weight:       broadcast-compatible with pred (e.g., (B,T,H,W), (1,T,1,1), (B,1,1,1), etc.)
        """
        weight = 1 + (target * 10)
        err2 = (pred - target) ** 2
        w = weight.to(err2.dtype).to(err2.device)

        weighted = err2 * w

        if self.reduction == "none":
            return weighted

        if self.reduction == "sum":
            return weighted.sum()
        
        return torch.mean(weighted) # Or "weighted.sum() / (w.sum() + self.eps)" to normalize by sum of weights (not by number of elements)
