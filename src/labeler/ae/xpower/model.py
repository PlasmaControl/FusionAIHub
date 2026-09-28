"""FrameCNN: three cross-power spectrograms in, one AE logit per 10 ms frame out.

Four 3x3 convolution blocks over (frequency, time), halving frequency three
times; a max over what is left of frequency, so a mode anywhere in 80-250 kHz
counts the same; four dilated convolutions over time (about +-40 ms of context
in all); the mean of each frame's five sub-frames; a 1x1 convolution to the
logit. Fully convolutional in time, so a whole shot runs in one call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FrameCNNConfig:
    channels: int = 3
    width: int = 32
    subs: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8)

    def as_dict(self) -> dict:
        return {**asdict(self), "dilations": list(self.dilations)}

    @classmethod
    def from_dict(cls, raw: dict) -> FrameCNNConfig:
        return cls(**{**raw, "dilations": tuple(raw["dilations"])})


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()
    )


class FrameCNN(nn.Module):
    def __init__(self, config: FrameCNNConfig | None = None):
        super().__init__()
        config = config or FrameCNNConfig()
        self.config = config
        w = config.width
        self.features = nn.Sequential(
            _block(config.channels, 16),
            nn.MaxPool2d((2, 1)),
            _block(16, w),
            nn.MaxPool2d((2, 1)),
            _block(w, w),
            nn.MaxPool2d((2, 1)),
            _block(w, w),
        )
        self.temporal = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(w, w, 3, padding=d, dilation=d), nn.BatchNorm1d(w), nn.ReLU()
            )
            for d in config.dilations
        )
        self.head = nn.Conv1d(w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, C, n_bins, subs * n)` -> `(B, n)` logits."""
        if x.shape[-1] % self.config.subs:
            raise ValueError(f"{x.shape[-1]} sub-frames is not whole frames")
        h = self.features(x).amax(dim=2)
        for layer in self.temporal:
            h = h + layer(h)
        h = F.avg_pool1d(h, self.config.subs)
        return self.head(h)[:, 0]
