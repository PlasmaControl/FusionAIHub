import torch
import torch.nn as nn
from .base import ModalityEncoder, ModalityDecoder


class SpectrogramEncoder(ModalityEncoder):
    def __init__(self, in_channels, out_features=64):
        super().__init__(in_channels, out_features)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(128, out_features), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class SpectrogramDecoder(ModalityDecoder):
    def __init__(self, in_features=64, out_channels=1, target_size=(33, 100)):
        super().__init__(in_features, out_channels)
        self.target_size = target_size
        self.net = nn.Sequential(
            nn.Linear(in_features, 128 * 2 * 2), nn.ReLU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 3, padding=1),
        )
        self.resample = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, z):
        return self.resample(self.net(z))
