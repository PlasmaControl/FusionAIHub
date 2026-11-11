"""Frame PatchGAN discriminator for the tangtv video codec.

The video counterpart of ``discriminator.FreqAwarePatchGAN``, using only ``torch.nn``
and ``config.py`` (docs/IGNITE_DESIGN.md §4.2, §4.3).

- Frames are judged by a standard image PatchGAN. Video has no frequency axis: height
  and width are translation-equivariant, which suits a fully convolutional PatchGAN, so
  the frequency-position channels of ``FreqAwarePatchGAN`` are left out.
- The same body is applied at two scales (full frame and a downsampled copy) to judge
  fine texture and coarse structure.
- A ``(B, C, T, H, W)`` video is reshaped to ``(B*T, C, H, W)`` and each frame is scored
  on its own (as in MagViT and VQGAN), so the hinge loss averages over all frames.
- The heads return raw patch scores (no sigmoid) for the hinge loss.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import VideoCodecConfig


class _FramePatchGANBody(nn.Module):
    """Small conv PatchGAN: (B, C, H, W) -> one raw score map."""

    def __init__(self, in_channels: int, base: int = 32, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        out = base
        for li in range(n_layers):
            stride = 2 if li < n_layers - 1 else 1
            layers.append(nn.Conv2d(ch, out, kernel_size=4, stride=stride, padding=1))
            if li > 0:
                layers.append(nn.GroupNorm(num_groups=min(8, out), num_channels=out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out
            out = min(out * 2, base * 4)
        # Final conv to a single-channel raw score map, no activation.
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the body and also return its intermediate activations.

        Returns ``(score, feats)``. ``score`` equals :meth:`forward`; ``feats`` holds
        the output of every LeakyReLU, not the input or the final score map. The
        generator's feature-matching loss compares these. Same as
        ``_PatchGANBody.forward_features``.
        """
        feats: List[torch.Tensor] = []
        h = x
        n = len(self.net)
        for i, layer in enumerate(self.net):
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU) and i < n - 1:
                feats.append(h)
        return h, feats


class FramePatchGAN(nn.Module):
    """Multi-scale frame PatchGAN discriminator for the tangtv video codec.

    ``forward(x)`` maps ``(B, C, T, H, W)`` to a list of raw patch-score maps, one per
    scale, each ``(B*T, 1, h, w)``. Every frame is scored on its own.
    ``forward(x, return_features=True)`` also returns the intermediate activations of all
    bodies as one flat list, for feature matching.

    Same structure as ``discriminator.FreqAwarePatchGAN`` without the frequency-position
    channels and with the per-frame reshape.
    """

    def __init__(self, cfg: VideoCodecConfig, base: int = 32, scales: int = 2):
        super().__init__()
        self.cfg = cfg
        self.scales = scales
        in_ch = cfg.channels  # no frequency-position channels
        self.bodies = nn.ModuleList(
            [_FramePatchGANBody(in_ch, base=base) for _ in range(scales)]
        )

    @staticmethod
    def _to_frames(x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B*T, C, H, W), every frame on the batch axis."""
        if x.dim() != 5:
            raise ValueError(f"FramePatchGAN expects (B, C, T, H, W); got {tuple(x.shape)}")
        return rearrange(x, "b c t h w -> (b t) c h w")

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """``(B, C, T, H, W) -> list[score]``, or ``(list[score], list[feat])`` if requested."""
        frames = self._to_frames(x)  # (B*T, C, H, W)
        maps: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        cur = frames
        for si, body in enumerate(self.bodies):
            if si > 0:
                # Downsample the frames for the coarser scale.
                cur = F.avg_pool2d(cur, kernel_size=2, ceil_mode=True)
            if return_features:
                score, body_feats = body.forward_features(cur)
                maps.append(score)
                feats.extend(body_feats)
            else:
                maps.append(body(cur))
        if return_features:
            return maps, feats
        return maps
