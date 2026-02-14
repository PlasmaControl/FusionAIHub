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