"""RowsCNN: a shot's review-store rows in, one logit per 10 ms frame out (D37).

The rows' sub-frame features (`features.features`, `(C, subs * n)`) go through:
- a 1x1 stem to `width` channels;
- three dilated residual blocks over the sub-frames (dilations 1, 2, 4);
- the mean over each 10 ms frame's `subs` sub-frames;
- five dilated residual blocks over the frames (1, 2, 4, 8, 16): about +-310 ms
  of context in all;
- a 1x1 convolution to one logit a frame.

It is fully convolutional in time, so a whole shot runs in one call, and has
under 40 k parameters at width 32 for every spec's channels, so it trains on
CPU. `bin_logits` pools the frames' logits to a spec's bins, where its loss
and its decisions are: their maximum for onsets (ELM), their mean for states.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SUB_DILATIONS = (1, 2, 4)
FRAME_DILATIONS = (1, 2, 4, 8, 16)
POOLS = ("max", "mean")


def _block(width: int, dilation: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
        nn.BatchNorm1d(width),
        nn.ReLU(),
    )


class RowsCNN(nn.Module):
    def __init__(self, channels: int, subs: int, width: int = 32):
        super().__init__()
        if channels < 1 or subs < 1:
            raise ValueError(f"channels {channels} and subs {subs} must be positive")
        self.channels, self.subs, self.width = int(channels), int(subs), int(width)
        self.stem = nn.Sequential(
            nn.Conv1d(channels, width, 1), nn.BatchNorm1d(width), nn.ReLU()
        )
        self.sub_blocks = nn.ModuleList(_block(width, d) for d in SUB_DILATIONS)
        self.frame_blocks = nn.ModuleList(_block(width, d) for d in FRAME_DILATIONS)
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, channels, subs * n)` -> `(B, n)` logits."""
        if x.shape[-1] % self.subs:
            raise ValueError(
                f"{x.shape[-1]} sub-frames is not whole frames of {self.subs}"
            )
        h = self.stem(x)
        for block in self.sub_blocks:
            h = h + block(h)
        if self.subs > 1:
            h = F.avg_pool1d(h, self.subs)
        for block in self.frame_blocks:
            h = h + block(h)
        return self.head(h)[:, 0]


def bin_logits(frame_logits: torch.Tensor, frames_per_bin: int, pool: str):
    """`(B, n)` frame logits -> `(B, n // frames_per_bin)` bin logits: each bin's
    frames' maximum ("max", onsets) or mean ("mean", states)."""
    if pool not in POOLS:
        raise ValueError(f"pool {pool!r} is not one of {POOLS}")
    n = frame_logits.shape[-1]
    if n % frames_per_bin:
        raise ValueError(f"{n} frames is not whole bins of {frames_per_bin}")
    blocks = frame_logits.reshape(*frame_logits.shape[:-1], -1, frames_per_bin)
    return blocks.amax(dim=-1) if pool == "max" else blocks.mean(dim=-1)
