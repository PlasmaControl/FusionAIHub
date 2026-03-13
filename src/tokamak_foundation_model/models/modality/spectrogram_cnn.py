"""Simple CNN autoencoder for tokamak spectrogram diagnostics.

A lightweight 2D convolutional autoencoder with residual blocks and
strided downsampling / nearest-neighbour upsampling.  No attention, no
patching, no quantization — just convolutions.  Designed as a strong
baseline that can easily overfit small datasets.

Architecture (default dims=[64, 128])
--------------------------------------
Encoder:
  Stem:  Conv2d(C, 64, 3, pad=1) + GN + GELU
  Down1: ResBlock(64→64, stride=2)
  Down2: ResBlock(64→128, stride=2)

Decoder (mirror):
  Up1:   Upsample(2x) + ResBlock(128→64)
  Up2:   Upsample(2x) + ResBlock(64→64)
  Head:  Conv2d(64, C, 1)

Return contract
---------------
Training : (reconstructed, z) — z is bottleneck feature map for monitoring.
Eval     : reconstructed      — shape (B, C, F, T) matching input.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with min(32, channels) groups."""
    return nn.GroupNorm(min(32, channels), channels)


class _ResBlock(nn.Module):
    """Two-conv residual block with optional stride/channel change."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.gn1 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = _gn(out_ch)
        self.act = nn.GELU()

        # Shortcut: 1x1 conv when dimensions change
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                _gn(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class _UpBlock(nn.Module):
    """Upsample(2x, nearest) + ResBlock — avoids checkerboard artifacts."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = _ResBlock(in_ch, out_ch, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.up(x))


class _CNNEncoder(nn.Module):
    """CNN encoder that pads to multiples of 2^n_stages before encoding."""

    def __init__(self, n_channels: int, dims: list[int]) -> None:
        super().__init__()
        self.n_stages = len(dims)
        self.stride = 2 ** self.n_stages

        # Stem
        layers: list[nn.Module] = [
            nn.Conv2d(n_channels, dims[0], 3, padding=1),
            _gn(dims[0]),
            nn.GELU(),
        ]
        # Downsampling blocks
        in_ch = dims[0]
        for out_ch in dims:
            layers.append(_ResBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Pad to multiples of stride
        _, _, H, W = x.shape
        pad_h = (self.stride - H % self.stride) % self.stride
        pad_w = (self.stride - W % self.stride) % self.stride
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
        return self.net(x)


class _CNNDecoder(nn.Module):
    """CNN decoder — mirror of encoder with nearest-neighbour upsampling."""

    def __init__(self, n_channels: int, dims: list[int]) -> None:
        super().__init__()
        # Upsampling blocks (reverse order)
        layers: list[nn.Module] = []
        reversed_dims = list(reversed(dims))
        in_ch = reversed_dims[0]
        for out_ch in reversed_dims[1:]:
            layers.append(_UpBlock(in_ch, out_ch))
            in_ch = out_ch
        # Final up block back to stem dim
        layers.append(_UpBlock(in_ch, dims[0]))
        # Head
        layers.append(nn.Conv2d(dims[0], n_channels, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class SpectrogramCNNAutoEncoder(ModalityAutoEncoder):
    """Simple CNN autoencoder for spectrogram signals.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Unused; kept for interface compatibility with build_model().
    n_tokens : int
        Unused; kept for interface compatibility with build_model().
    dims : list[int] | None
        Channel dimensions per downsampling stage (default [64, 128]).
        Each stage halves spatial resolution, so total downsampling = 2^len(dims).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        dims: list[int] | None = None,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        if dims is None:
            dims = [64, 128]
        self.dims = dims

        self.encoder = _CNNEncoder(n_channels, dims)
        self.decoder = _CNNDecoder(n_channels, dims)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        B, C, F_orig, T_orig = x.shape

        z = self.encoder(x)
        reconstructed = self.decoder(z)

        # Crop to original spatial dims (encoder may have padded)
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed
        return reconstructed, z
