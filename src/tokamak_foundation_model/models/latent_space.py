import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion layer."""

    def __init__(self, feature_dim, num_modalities, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        """
        Args:
            features: List of tensors, each (batch, feature_dim)
        Returns:
            fused: Tensor (batch, feature_dim)
        """
        # Stack features: (batch, num_modalities, feature_dim)
        stacked = torch.stack(features, dim=1)

        # Self-attention across modalities
        attended, _ = self.attention(stacked, stacked, stacked)

        # Residual connection
        attended = self.norm(attended + stacked)

        # Aggregate (mean pooling across modalities)
        fused = attended.mean(dim=1)  # (batch, feature_dim)

        return fused
