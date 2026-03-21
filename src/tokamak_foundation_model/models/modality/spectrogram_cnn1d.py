"""1D ConvNeXt autoencoder with hierarchical frequency-reducing stem.

The stem gradually collapses the frequency axis through multiple 2D conv stages
(each reducing frequency by 4x), then 1D ConvNeXt blocks process temporal
features.  This design:
  - Preserves spectral detail through gradual compression (not one-shot)
  - Scales to any n_channels: only the first conv layer depends on C
  - Produces T/frame_width tokens with full spectral information each

Architecture
------------
Encoder:
  Conv2d stages (freq stride 4 each)     -> (B, dim, 1, T') -> squeeze
  N x ConvNeXtV2Block1d                  -> (B, dim, T')
  Conv1d(dim, bottleneck_dim, 1)         -> (B, bn, T')

Decoder:
  Conv1d(bottleneck_dim, dim, 1)         -> (B, dim, T')
  N x ConvNeXtV2Block1d                  -> (B, dim, T')
  ConvTranspose2d stages (mirror)        -> (B, C, F, T)

Return contract
---------------
Training : (reconstructed, z_tokens) -- z_tokens is (B, T', bottleneck_dim)
Eval     : reconstructed             -- shape (B, C, F, T) matching input
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder


# ---------------------------------------------------------------------------
# 1D building blocks
# ---------------------------------------------------------------------------


class _GRN1d(nn.Module):
    """Global Response Normalization for 1D features (channels-last layout)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C) channels-last
        gx = torch.norm(x, p=2, dim=1, keepdim=True)  # (B, 1, C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class _ConvNeXtV2Block1d(nn.Module):
    """ConvNeXt V2 block for 1D temporal sequences.

    Depthwise Conv1d -> LayerNorm -> Linear -> GELU -> GRN -> Linear + residual.
    """

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.grn = _GRN1d(dim * 4)
        self.pwconv2 = nn.Linear(dim * 4, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T) channels-first
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, C, T)
        return residual + x


# ---------------------------------------------------------------------------
# Encoder wrapper (for .encoder attribute used in tests)
# ---------------------------------------------------------------------------


class _CNN1dEncoder(nn.Module):
    """Hierarchical frequency-reducing encoder + 1D ConvNeXt + bottleneck."""

    def __init__(
        self,
        stem: nn.Sequential,
        blocks: nn.ModuleList,
        bottleneck_proj: nn.Conv1d,
        frame_width: int,
        pad_freq: int,
    ) -> None:
        super().__init__()
        self.stem = stem
        self.blocks = blocks
        self.bottleneck_proj = bottleneck_proj
        self.frame_width = frame_width
        self.pad_freq = pad_freq

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, F, T) -> (B, T', bottleneck_dim) token sequence."""
        B, C, F, T = x.shape
        fw = self.frame_width
        pad_t = (fw - T % fw) % fw
        if pad_t:
            x = Fnn.pad(x, (0, pad_t))
        if self.pad_freq > 0:
            x = Fnn.pad(x, (0, 0, 0, self.pad_freq))
        z = self.stem(x).squeeze(2)  # (B, dim, T')
        for block in self.blocks:
            z = block(z)
        z = self.bottleneck_proj(z)  # (B, bottleneck_dim, T')
        return z.transpose(1, 2)  # (B, T', bottleneck_dim)


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class SpectrogramCNN1dAutoEncoder(ModalityAutoEncoder):
    """1D ConvNeXt autoencoder with hierarchical frequency-reducing stem.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels (4 for CO2, 40+ for ECE).
    d_model : int
        Output token dimension (= bottleneck_dim if not overridden).
    n_tokens : int
        Unused; kept for interface compatibility.
    freq_bins : int
        Frequency dimension of the input spectrogram.
    frame_width : int
        Time steps grouped per stem position (default 2).
    dim : int
        Hidden channel dimension for 1D ConvNeXt blocks (default 256).
    depth : int
        Number of 1D ConvNeXt blocks in encoder and decoder (default 6).
    stem_dims : list[int] | None
        Channel widths for intermediate frequency-reduction stages.  Each
        stage reduces frequency by 4x.  A final conv collapses any remaining
        frequency bins.  Default ``[64, 128]`` gives two 4x stages.
    bottleneck_dim : int | None
        Channel bottleneck dimension (default: d_model).
    kernel_size : int
        Depthwise conv kernel size for 1D blocks (default 7).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        freq_bins: int = 128,
        frame_width: int = 2,
        dim: int = 256,
        depth: int = 6,
        stem_dims: list[int] | None = None,
        bottleneck_dim: int | None = None,
        kernel_size: int = 7,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        if stem_dims is None:
            stem_dims = [64, 128]
        self.freq_bins = freq_bins
        self.frame_width = frame_width
        if bottleneck_dim is None:
            bottleneck_dim = d_model
        self.bottleneck_dim = bottleneck_dim

        # -- Compute frequency padding / strides -----------------------------
        intermediate_stride = 4 ** len(stem_dims)
        padded_freq = int(math.ceil(freq_bins / intermediate_stride)) * intermediate_stride
        self._pad_freq = padded_freq - freq_bins
        remaining_freq = padded_freq // intermediate_stride  # freq bins after intermediate stages

        # -- Encoder stem (2D, frequency-reducing) ---------------------------
        enc_stem_layers: list[nn.Module] = []
        in_ch = n_channels
        for i, out_ch in enumerate(stem_dims):
            s_t = frame_width if i == 0 else 1
            enc_stem_layers.extend([
                nn.Conv2d(in_ch, out_ch, (4, s_t), stride=(4, s_t)),
                nn.GELU(),
                _gn(out_ch),
            ])
            in_ch = out_ch
        # Final collapse: remaining freq bins -> 1
        s_t_final = frame_width if len(stem_dims) == 0 else 1
        enc_stem_layers.append(
            nn.Conv2d(in_ch, dim, (remaining_freq, s_t_final),
                      stride=(remaining_freq, s_t_final)),
        )
        self.enc_stem = nn.Sequential(*enc_stem_layers)

        # -- 1D temporal processing ------------------------------------------
        self.enc_blocks = nn.ModuleList(
            [_ConvNeXtV2Block1d(dim, kernel_size) for _ in range(depth)]
        )
        self.bottleneck_down = nn.Conv1d(dim, bottleneck_dim, 1)

        # -- Decoder ---------------------------------------------------------
        self.bottleneck_up = nn.Conv1d(bottleneck_dim, dim, 1)
        self.dec_blocks = nn.ModuleList(
            [_ConvNeXtV2Block1d(dim, kernel_size) for _ in range(depth)]
        )

        # -- Decoder stem (mirror of encoder stem) ---------------------------
        dec_stem_layers: list[nn.Module] = []
        # First: expand from freq=1 to remaining_freq
        out_ch_first = stem_dims[-1] if stem_dims else n_channels
        dec_stem_layers.append(
            nn.ConvTranspose2d(dim, out_ch_first, (remaining_freq, s_t_final),
                               stride=(remaining_freq, s_t_final)),
        )
        if stem_dims:
            dec_stem_layers.extend([nn.GELU(), _gn(out_ch_first)])
        # Reverse through intermediate stages
        for i in range(len(stem_dims) - 1, 0, -1):
            in_ch_dec = stem_dims[i]
            out_ch_dec = stem_dims[i - 1]
            dec_stem_layers.extend([
                nn.ConvTranspose2d(in_ch_dec, out_ch_dec, (4, 1), stride=(4, 1)),
                nn.GELU(),
                _gn(out_ch_dec),
            ])
        # Final: restore original channels (+ time upsampling if needed)
        if stem_dims:
            s_t_last = frame_width
            dec_stem_layers.append(
                nn.ConvTranspose2d(stem_dims[0], n_channels,
                                   (4, s_t_last), stride=(4, s_t_last)),
            )
        self.dec_stem = nn.Sequential(*dec_stem_layers)

        # -- Encoder wrapper for tests ---------------------------------------
        self.encoder = _CNN1dEncoder(
            self.enc_stem, self.enc_blocks, self.bottleneck_down,
            frame_width, self._pad_freq,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        B, C, F_orig, T_orig = x.shape
        fw = self.frame_width

        # -- 1. Pad time and frequency ---------------------------------------
        pad_t = (fw - T_orig % fw) % fw
        if pad_t:
            x = Fnn.pad(x, (0, pad_t))
        if self._pad_freq > 0:
            x = Fnn.pad(x, (0, 0, 0, self._pad_freq))

        # -- 2. Encoder stem (hierarchical freq reduction) -------------------
        z = self.enc_stem(x).squeeze(2)  # (B, dim, T')
        for block in self.enc_blocks:
            z = block(z)

        # -- 3. Channel bottleneck -------------------------------------------
        z_bn = self.bottleneck_down(z)  # (B, bottleneck_dim, T')
        z_tokens = z_bn.transpose(1, 2)  # (B, T', bottleneck_dim)

        # -- 4. Decode -------------------------------------------------------
        z_dec = self.bottleneck_up(z_bn)  # (B, dim, T')
        for block in self.dec_blocks:
            z_dec = block(z_dec)

        # -- 5. Decoder stem (mirror: reconstruct F and T) -------------------
        z_dec = z_dec.unsqueeze(2)  # (B, dim, 1, T')
        reconstructed = self.dec_stem(z_dec)  # (B, C, F_padded, T_padded)
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed
        return reconstructed, z_tokens
