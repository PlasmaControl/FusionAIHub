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
