"""A small U-Net over the review page's AE rows.

Input `(B, 3, H, W)`: the three cross-power rows (R0 x V1, V2, V3), 0-250 kHz
on the page's 0.977 kHz bins, at the store's level 8 (2.048 ms columns), each
in [0, 1]. The whole band goes in, so the network sees an MHD mode's 0-60 kHz
fundamental beside its harmonics; the loss only counts 80-250 kHz. Output
`(B, 1, H, W)` logits, one per pixel. `H` and `W` need not be multiples of
anything: the input is padded to one of `2 ** depth` and the output cropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SegNetConfig:
    channels: int = 3
    width: int = 12
    depth: int = 3

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> SegNetConfig:
        return cls(**values)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(min(4, c_out), c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(min(4, c_out), c_out),
        nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):
    def __init__(self, config: SegNetConfig | None = None):
        super().__init__()
        self.config = config = config or SegNetConfig()
        widths = [config.width * 2**k for k in range(config.depth + 1)]
        self.down = nn.ModuleList(
            [_block(config.channels, widths[0])]
            + [_block(widths[k], widths[k + 1]) for k in range(config.depth)]
        )
        self.up = nn.ModuleList(
            [_block(widths[k + 1] + widths[k], widths[k]) for k in range(config.depth)]
        )
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        step = 2**self.config.depth
        x = F.pad(x, (0, -w % step, 0, -h % step))
        skips = []
        for k, block in enumerate(self.down):
            x = block(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for k in reversed(range(self.config.depth)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.up[k](torch.cat([x, skips[k]], dim=1))
        return self.head(x)[..., :h, :w]
