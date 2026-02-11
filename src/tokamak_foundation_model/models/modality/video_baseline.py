import torch
import torch.nn as nn
from .base import ModalityEncoder, ModalityDecoder


class VideoEncoder(ModalityEncoder):
    def __init__(self, in_channels=1, out_features=64):
        super().__init__(in_channels, out_features)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1),
            nn.Flatten(), nn.Linear(64, out_features), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class VideoDecoder(ModalityDecoder):
    def __init__(self, in_features=64, out_channels=1, target_size=(10, 64, 64)):
        super().__init__(in_features, out_channels)
        self.target_size = target_size  # (T, H, W)
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Unflatten(1, (64, 1, 1, 1)),
            nn.ConvTranspose3d(64, 32, (2, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), nn.ReLU(),
            nn.ConvTranspose3d(32, 16, (2, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)), nn.ReLU(),
            nn.ConvTranspose3d(16, out_channels, 3, padding=1),
        )
        self.resample = nn.AdaptiveAvgPool3d(target_size)

    def forward(self, z):
        x = self.resample(self.net(z))
        return x.squeeze(1)  # remove channel dim -> (B, T, H, W)
