import torch
import torch.nn as nn

class CrossAttentionBaselineModel(nn.Module):
    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_modalities, batch_first=True)

    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        attended, _ = self.attn(stacked, stacked, stacked)
        return attended.mean(dim=1)


class ConcatenationBaselineModel(nn.Module):
    def __init__(self, feature_dim: int, num_modalities: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.fc = nn.Linear(feature_dim * num_modalities, feature_dim)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.fc(torch.cat(features, dim=1))