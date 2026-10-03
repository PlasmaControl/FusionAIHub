"""Paper-based CNN+LSTM regime classifier and PhaseNet-style crash trace model.

No external implementation is imported. OuYang's ambiguous layer count is
resolved to one bidirectional 32-unit LSTM, following the text. PhaseNet's
stride-four, kernel-seven topology uses smaller channel widths for CPU training.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F


class HL3(nn.Module):
    """20 ms / 200 sample / four-channel three-regime classifier."""

    def __init__(self, channels=4):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv1d(channels, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU()
        )
        self.second = nn.Sequential(
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.BatchNorm1d(32)
        )
        self.skip = nn.Conv1d(16, 32, 1, stride=2)
        self.channel_attention = nn.Sequential(
            nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 32), nn.Sigmoid()
        )
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.sequence_attention = nn.Linear(64, 1)
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.first(x)
        x = F.relu(self.second(x) + self.skip(x))
        x = x * self.channel_attention(x.mean(dim=-1))[..., None]
        x, _ = self.lstm(x.transpose(1, 2))
        weights = self.sequence_attention(x).softmax(dim=1)
        return self.head((x * weights).sum(dim=1))


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(cin, cout, 7, padding=3),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
            nn.Conv1d(cout, cout, 7, padding=3),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class PhasePicker(nn.Module):
    """48 ECE channels -> crash/noise and train-presence logits at every sample."""

    def __init__(self, channels=48, widths=(8, 11, 16, 22, 32)):
        super().__init__()
        self.stem = Block(channels, widths[0])
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(a, b, 7, stride=4, padding=3), nn.ReLU(), Block(b, b)
                )
                for a, b in itertools.pairwise(widths)
            ]
        )
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose1d(b, a, 7, stride=4, padding=3, output_padding=3)
                for a, b in reversed(list(itertools.pairwise(widths)))
            ]
        )
        self.combine = nn.ModuleList([Block(2 * a, a) for a in reversed(widths[:-1])])
        self.head = nn.Conv1d(widths[0], 3, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = [x]
        for layer in self.down:
            x = layer(x)
            skips.append(x)
        for layer, combine, skip in zip(
            self.up, self.combine, reversed(skips[:-1]), strict=True
        ):
            x = layer(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(
                    x, size=skip.shape[-1], mode="linear", align_corners=False
                )
            x = combine(torch.cat([x, skip], dim=1))
        return self.head(x)


def soft_crash_target(t_s, crashes_s, sigma_s=0.0005):
    """PhaseNet Gaussian picks, truncated at +/- 3 sigma, independently of spans."""
    import numpy as np

    target = np.zeros(len(t_s), dtype=np.float32)
    for time in crashes_s:
        distance = np.abs(np.asarray(t_s) - time)
        target = np.maximum(
            target,
            np.where(
                distance <= 3 * sigma_s, np.exp(-0.5 * (distance / sigma_s) ** 2), 0
            ),
        )
    return target
