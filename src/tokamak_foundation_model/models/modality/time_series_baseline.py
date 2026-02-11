import torch
import torch.nn as nn
from .base import ModalityEncoder, ModalityDecoder


class TimeSeriesEncoder(ModalityEncoder):
    def __init__(self, in_channels, out_features=64):
        super().__init__(in_channels, out_features)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(64, out_features), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TimeSeriesDecoder(ModalityDecoder):
    def __init__(self, in_features=64, out_channels=1, target_length=100):
        super().__init__(in_features, out_channels)
        self.target_length = target_length
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Unflatten(1, (64, 1)),
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, out_channels, 4, stride=2, padding=1),
        )
        self.resample = nn.AdaptiveAvgPool1d(target_length)

    def forward(self, z):
        return self.resample(self.net(z))
