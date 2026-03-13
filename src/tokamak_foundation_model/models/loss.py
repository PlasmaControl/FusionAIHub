import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

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
    Weighted MSE Loss optimized for multi-channel video.
    Allows for per-channel thresholds and bright weights to handle 
    mixed modalities (e.g., one sparse channel, one dense channel).
    """
    def __init__(
        self, 
        reduction: str = "mean", 
        l1l2: str = "l2", 
        threshold: Union[float, List[float]] = 0.1, 
        bright_weight: Union[float, List[float]] = 50.0, 
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of: mean, sum, none")
        
        self.reduction = reduction
        self.threshold = threshold
        self.bright_weight = bright_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: Expected shape (B, C, T, H, W)
        """
        if l1l2 == 'l1':
            print('l1 runs...')
            err = torch.abs(pred - target) # L1 instead of (pred-target)**2
        else:
            print('l2 runs...')
            err = (pred - target) ** 2
        
        err
        # Scenario 1: Apply the exact same rules globally to all channels
        if isinstance(self.threshold, float) and isinstance(self.bright_weight, float):
            weight_map = torch.where(target > self.threshold, self.bright_weight, 1.0)
            
        # Scenario 2: Apply specific rules per channel
        else:
            C = target.shape[1] # Assumes channel is index 1: (B, C, T, H, W)
            weight_map = torch.ones_like(target)
            
            # Normalize inputs to lists matching the channel count
            thresh_list = [self.threshold] * C if isinstance(self.threshold, float) else self.threshold
            weight_list = [self.bright_weight] * C if isinstance(self.bright_weight, float) else self.bright_weight
            
            for c in range(C):
                chan_target = target[:, c:c+1, ...] # Keep dimensions using slice
                chan_weight = torch.where(chan_target > thresh_list[c], weight_list[c], 1.0)
                weight_map[:, c:c+1, ...] = chan_weight

        # Ensure weight map matches the device and dtype of the errors
        weight_map = weight_map.to(err.dtype).to(err.device)

        weighted_err = err * weight_map

        if self.reduction == "none":
            return weighted_err
        if self.reduction == "sum":
            return weighted_err.sum()
        
        return torch.mean(weighted_err)