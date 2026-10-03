"""A 1-D U-Net over the ELM inputs: ELMy time and ELM onsets, one value per millisecond.

The architecture is the PhaseNet / U-Time kind used for dense segmentation of
physiological and seismic series (Zhu and Beroza 2019; Perslev et al. 2019): a
strided stem takes the 10 kHz input to 1 kHz, an encoder of strided
convolutions widens the context at a quarter of the rate each level, and a
decoder with skip connections returns to 1 kHz. `STEM_STRIDE` is the ratio of
the input grid (`inputs.CELLS_PER_MS`) to the output; the receptive field at the
bottom is `STRIDE ** DEPTH` ms (256 ms) times the kernel, enough to see a whole
ELM train and the baseline it rises from.

Two logits per millisecond: `event`, ELMy time (the review's present spans,
crowds included), and `onset`, the start of a non-crowd present span. `forward`
returns `(batch, 2, n_ms)`. The input length must be a multiple of
`CELLS_PER_MS * STRIDE ** DEPTH`; `pad_to` gives the padded length.
"""

from __future__ import annotations

import torch
from torch import nn

from . import inputs

STRIDE = 4
DEPTH = 4
KERNEL = 7
WIDTHS = (24, 32, 48, 64, 96)  # stem, then one per encoder level
UNIT = inputs.CELLS_PER_MS * STRIDE**DEPTH  # input cells the length must divide by
HEADS = ("event", "onset")


def pad_to(n_cells: int) -> int:
    """The smallest valid input length (cells) that is at least `n_cells`."""
    return -(-n_cells // UNIT) * UNIT


def _groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


def block(c_in: int, c_out: int, *, stride: int = 1, kernel: int = KERNEL) -> nn.Module:
    """Convolution, group norm, GELU; twice, the first one strided."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
        nn.GroupNorm(_groups(c_out), c_out),
        nn.GELU(),
        nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2),
        nn.GroupNorm(_groups(c_out), c_out),
        nn.GELU(),
    )


class ElmUNet(nn.Module):
    """The dense detector. `widths[0]` is the stem, `widths[1:]` the encoder levels."""

    def __init__(
        self,
        in_channels: int = inputs.N_CHANNELS,
        widths: tuple[int, ...] = WIDTHS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(widths) != DEPTH + 1:
            raise ValueError(f"widths must have {DEPTH + 1} entries")
        s = inputs.CELLS_PER_MS
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], 2 * s, stride=s, padding=s // 2),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.GELU(),
            nn.Conv1d(widths[0], widths[0], KERNEL, padding=KERNEL // 2),
            nn.GroupNorm(_groups(widths[0]), widths[0]),
            nn.GELU(),
        )
        self.down = nn.ModuleList(
            block(widths[i], widths[i + 1], stride=STRIDE) for i in range(DEPTH)
        )
        self.up = nn.ModuleList(
            block(widths[i + 1] + widths[i], widths[i]) for i in range(DEPTH)
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(widths[0], len(HEADS), 1)
        # start the onset head rare, as its target is
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[1] = -4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % UNIT:
            raise ValueError(f"input length {x.shape[-1]} is not a multiple of {UNIT}")
        h = self.stem(x)
        skips = [h]
        for down in self.down:
            h = down(h)
            skips.append(h)
        h = self.drop(h)
        for up, skip in zip(reversed(self.up), reversed(skips[:-1])):
            h = nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = up(torch.cat([h, skip], dim=1))
        return self.head(h)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
