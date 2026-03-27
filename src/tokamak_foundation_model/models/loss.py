import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MultiScaleSpectrogramLoss(nn.Module):
    """Multi-scale L1 loss on 2D spectrograms via average pooling.

    Computes L1 at multiple spatial resolutions by downsampling both pred
    and target with adaptive_avg_pool2d, then averages across scales.
    Forces the model to reconstruct both coarse structure and fine detail.

    Parameters
    ----------
    scales : tuple of float
        Fraction of original resolution at each scale.
        Default (1.0, 0.5, 0.25) = full-res, half-res, quarter-res.
    weights : tuple of float or None
        Per-scale weights. None = equal weighting (1/n each).
    """

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 0.5, 0.25),
        weights: tuple[float, ...] | None = None,
    ):
        super().__init__()
        if not scales:
            raise ValueError("scales must be non-empty")
        if weights is None:
            weights = tuple(1.0 / len(scales) for _ in scales)
        if len(weights) != len(scales):
            raise ValueError("weights must have same length as scales")
        total = sum(weights)
        self.scales = scales
        self.weights = tuple(w / total for w in weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : (B, C, F, T)

        Returns
        -------
        Scalar loss.
        """
        H, W = pred.shape[-2], pred.shape[-1]
        total = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        for scale, w in zip(self.scales, self.weights):
            if scale == 1.0:
                total = total + w * F.l1_loss(pred, target)
            else:
                size = (max(1, round(H * scale)), max(1, round(W * scale)))
                total = total + w * F.l1_loss(
                    F.adaptive_avg_pool2d(pred, size),
                    F.adaptive_avg_pool2d(target, size),
                )
        return total

    def multi_scale_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Multi-scale L1 with optional per-pixel spatial weighting.

        The weight tensor is downsampled alongside pred/target at each scale.
        Used by the variance-weighted loss path in FSQUnimodalTrainer.
        """
        H, W = pred.shape[-2], pred.shape[-1]
        total = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        for scale, sw in zip(self.scales, self.weights):
            if scale == 1.0:
                err = (pred - target).abs()
                if weight is not None:
                    err = weight * err
                total = total + sw * err.mean()
            else:
                size = (max(1, round(H * scale)), max(1, round(W * scale)))
                pred_s = F.adaptive_avg_pool2d(pred, size)
                target_s = F.adaptive_avg_pool2d(target, size)
                err = (pred_s - target_s).abs()
                if weight is not None:
                    err = F.adaptive_avg_pool2d(weight, size) * err
                total = total + sw * err.mean()
        return total
