import torch
import torch.nn as nn
from .base import ModalityEncoder, ModalityDecoder


class FastTimeSeriesEncoder(ModalityEncoder):

    def __init__(self, in_channels, out_features=64, hidden_dim=128):
        super().__init__(in_channels, out_features)
        self.conv_layers = nn.Sequential(
            # Layer 1: (B, C, T) -> (B, 64, T//5)
            nn.Conv1d(in_channels, 64, kernel_size=10, stride=5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Layer 2: -> (B, 128, T//15)
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            # Layer 3: -> (B, 256, T//30)
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.GELU(),
            # Layer 4: -> (B, 256, T//60)
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, hidden_dim * 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.proj(self.pool(self.conv_layers(x)))


class FastTimeSeriesDecoder(ModalityDecoder):

    def __init__(self, in_features=64, out_channels=1, target_length=5000, hidden_dim=128):
        super().__init__(in_features, out_channels)
        self.target_length = target_length
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim * 2, 1)),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, 64, kernel_size=5, stride=3, padding=1, output_padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(64, out_channels, kernel_size=10, stride=5, padding=2, output_padding=4),
        )
        self.resample = nn.AdaptiveAvgPool1d(target_length)

    def forward(self, z):
        return self.resample(self.deconv_layers(self.proj(z)))
