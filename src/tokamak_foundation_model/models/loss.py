import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskedL1Loss(nn.Module):
    """L1 loss that ignores zero-padded time steps.

    Expects tensors of shape ``(B, C, T)`` (time-series) or
    ``(B, C, F, T)`` (spectrograms).  For each sample in the batch the last
    dimension is masked to ``valid_lengths[b]`` frames; positions beyond that
    are excluded from the mean.

    Parameters
    ----------
    valid_lengths : torch.Tensor
        Long tensor of shape ``[B]`` holding the number of valid time steps
        per sample.  Passed to :meth:`forward`.
    """

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        output : torch.Tensor
            Model predictions, shape ``(B, ..., T)``.
        target : torch.Tensor
            Ground truth, same shape as *output*.
        valid_lengths : torch.Tensor or None
            Long tensor of shape ``[B]``.  When ``None``, falls back to plain
            L1 over all positions.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if valid_lengths is None:
            return F.l1_loss(output, target)

        T = output.shape[-1]
        # Build float mask [B, T]: 1.0 where position is valid
        t_idx = torch.arange(T, device=output.device)                    # [T]
        mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()  # [B, T]

        # Broadcast mask to full tensor shape (B, ..., T)
        for _ in range(output.dim() - 2):
            mask = mask.unsqueeze(1)                                      # [B, 1, ..., T]

        # Divide by the total number of valid elements across ALL dimensions
        # (B, C, ..., T), not just (B, T).  mask is [B, 1, ..., T] so
        # mask.sum() only counts B×T — without this correction the loss is
        # inflated by a factor of C (number of channels).
        # expand() returns a view (no copy), so this is memory-efficient.
        return ((output - target).abs() * mask).sum() / mask.expand_as(output).sum().clamp(min=1)

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores zero-padded time steps. Same interface as MaskedL1Loss."""

    def forward(
            self,
            output: torch.Tensor,
            target: torch.Tensor,
            valid_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_lengths is None:
            return F.mse_loss(output, target)

        T = output.shape[-1]
        t_idx = torch.arange(T, device=output.device)
        mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()  # [B, T]

        for _ in range(output.dim() - 2):
            mask = mask.unsqueeze(1)

        return ((output - target) ** 2 * mask).sum() / mask.expand_as(output).sum().clamp(min=1)


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
    ) -> torch.Tensor:
        if valid_lengths is None:
            return F.huber_loss(output, target, delta=self.delta)

        T = output.shape[-1]
        t_idx = torch.arange(T, device=output.device)
        mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()  # [B, T]

        for _ in range(output.dim() - 2):
            mask = mask.unsqueeze(1)

        loss = F.huber_loss(output, target, reduction="none", delta=self.delta)
        return (loss * mask).sum() / mask.expand_as(output).sum().clamp(min=1)


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
    ) -> torch.Tensor:
        sq_err = (output - target) ** 2
        weight = 1.0 / (target.abs() + self.eps) ** 2

        if valid_lengths is None:
            return (sq_err * weight).mean()

        T = output.shape[-1]
        t_idx = torch.arange(T, device=output.device)
        mask = (t_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)).float()  # [B, T]

        for _ in range(output.dim() - 2):
            mask = mask.unsqueeze(1)

        return (sq_err * weight * mask).sum() / mask.expand_as(output).sum().clamp(min=1)


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
