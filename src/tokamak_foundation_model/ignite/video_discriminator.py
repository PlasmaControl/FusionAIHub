"""Frame PatchGAN discriminator (IGNITE Phase-A tangtv video codec).

The video analogue of ``discriminator.FreqAwarePatchGAN``. Fresh code (no FAITH model
reuse); uses only ``torch.nn`` + ``config.py``.

Design (docs/IGNITE_DESIGN.md §4.2/§4.3):
- Video is Genie-native: smooth frames judged by a standard image PatchGAN. Unlike the
  spectrogram case, there is **no frequency axis** — the two spatial axes (height, width)
  ARE translation-equivariant (a divertor feature is the same feature wherever it appears),
  which is exactly the inductive bias a fully-convolutional PatchGAN wants. So we DROP the
  freq-PE channel that ``FreqAwarePatchGAN`` needed to break the spectrogram's freq symmetry.
- Multi-scale: the same PatchGAN body is applied at a couple of input scales (full frame +
  a spatially-downsampled copy) so both fine texture and coarse structure are judged.
- Operates per-FRAME: a ``(B, C, T, H, W)`` video is reshaped to ``(B*T, C, H, W)`` and each
  frame is scored independently (image-GAN over frames — the standard tokenizer-GAN choice,
  MagViT/VQGAN). Score maps keep the batch axis ``B*T`` so the hinge loss averages over all
  frames of all clips.
- Hinge-ready: the heads emit RAW patch scores (no final sigmoid).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import VideoCodecConfig


class _FramePatchGANBody(nn.Module):
    """A small conv PatchGAN producing a single raw score map from (B, C, H, W)."""

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
        # final conv -> single-channel raw score map (no activation / no sigmoid)
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the body, capturing the intermediate conv activations before the final score.

        Returns ``(score, feats)``: ``score`` is byte-identical to :meth:`forward`; ``feats``
        is the list of activations right after each LeakyReLU (every conv-block output that
        feeds forward), EXCLUDING the raw input and the final score map. These are the tensors
        the generator's feature-matching loss matches (VGG-style perceptual, via the disc's
        own features — no external VGG needed). Mirrors ``_PatchGANBody.forward_features``.
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

    Forward: ``(B, C, T, H, W) -> list[Tensor]`` of raw patch-score maps (one per scale),
    each ``(B*T, 1, h, w)``. Hinge-ready (no final sigmoid). Each video frame is scored
    independently (per-frame image GAN).

    ``forward(x, return_features=True) -> (list[score], list[feat])`` additionally returns a
    flat list of intermediate conv activations (both scales' bodies), for the generator's
    feature-matching loss. The default ``forward(x)`` path is byte-identical to before.

    Structurally identical to ``discriminator.FreqAwarePatchGAN`` (same body, multi-scale
    avg-pool, hinge-raw scores, feature-matching hook), MINUS the freq-PE channel (video has
    no frequency axis) and PLUS the per-frame reshape.
    """

    def __init__(self, cfg: VideoCodecConfig, base: int = 32, scales: int = 2):
        super().__init__()
        self.cfg = cfg
        self.scales = scales
        in_ch = cfg.channels  # no freq-PE channels: video axes are translation-equivariant
        self.bodies = nn.ModuleList(
            [_FramePatchGANBody(in_ch, base=base) for _ in range(scales)]
        )

    @staticmethod
    def _to_frames(x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B*T, C, H, W): stack every frame of every clip on the batch axis."""
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
                # downsample the RAW frames spatially for the coarse-scale judge.
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
