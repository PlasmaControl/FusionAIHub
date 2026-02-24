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


class SparseVideoWeightedMSE(nn.Module):
    """
    Weighted MSE Loss optimized for sparse bright spots in dark movies.
    Applies a fixed multiplier to errors on pixels exceeding a brightness threshold.
    """
    def __init__(
        self, 
        reduction: str = "mean", 
        threshold: float = 0.2, 
        bright_weight: float = 15.0, 
        eps: float = 1e-12
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        
        self.reduction = reduction
        self.threshold = threshold
        self.bright_weight = bright_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, T, H, W) or (B, C, T, H, W)
        """
        # Calculate standard squared error
        err2 = (pred - target) ** 2
        
        # Create a weight map: 
        # 1.0 for dark areas, bright_weight for spots > threshold
        # This is the logic from my original suggestion
        weight_map = torch.where(target > self.threshold, self.bright_weight, 1.0)
        
        # Ensure weight map is on the correct device and dtype
        weight_map = weight_map.to(err2.dtype).to(err2.device)

        weighted_err = err2 * weight_map

        if self.reduction == "none":
            return weighted_err

        if self.reduction == "sum":
            return weighted_err.sum()
        
        # Default to mean reduction across all elements
        return torch.mean(weighted_err)