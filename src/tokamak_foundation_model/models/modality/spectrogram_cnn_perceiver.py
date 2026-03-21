"""ConvNeXt autoencoder with two-stage bottleneck for tokamak spectrograms.

Uses the ConvNeXt V2 encoder/decoder from spectrogram_convnext_fsq.py.
Stage 1 (channel bottleneck) compresses channels via 1×1 conv while preserving
the full spatial grid.  Stage 2 (spatial compressor) further reduces the spatial
dimensions via strided convolutions to produce a compact token set for the
fusion transformer.

Architecture
------------
Encoder:
  _ConvNeXtV2Encoder (stem+stages)  → (B, dims[-1], H', W')   deep features
  Conv2d 1×1 (dims[-1] → bn_dim)   → (B, bn_dim, H', W')     channel bottleneck
  [if compress_stride > 1]:
    strided Conv2d(s)               → (B, d_model, H'', W'')   spatial compression
  flatten + reshape                 → (B, N, d_model)          token output

Decoder:
  reshape → (B, d_model, H'', W'')
  [if compress_stride > 1]:
    Upsample + Conv2d(s)            → (B, bn_dim, H', W')      spatial decompression
  Conv2d 1×1 (bn_dim → dims[-1])   → (B, dims[-1], H', W')
  _ConvNeXtV2Decoder (stages+head)  → (B, C, F, T)

Return contract
---------------
Training : (reconstructed, z_tokens) — z_tokens (B, N, d_model) for monitoring
Eval     : reconstructed             — shape (B, C, F, T) matching input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_convnext_fsq import (
    _ConvNeXtV2Encoder,
    _ConvNeXtV2Decoder,
)


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with min(32, channels) groups."""
    return nn.GroupNorm(min(32, channels), channels)


# ---------------------------------------------------------------------------
# Standalone encoder (for test_encoder_output_is_finite)
# ---------------------------------------------------------------------------

class _ConvNeXtBottleneckEncoder(nn.Module):
    """ConvNeXt encoder + channel bottleneck + optional spatial compressor.

    Returns (B, N, d_model) token sequence where N depends on spatial dims
    and compress_stride.
    """

    def __init__(
        self,
        convnext_encoder: _ConvNeXtV2Encoder,
        bottleneck_proj: nn.Module,
        spatial_compressor: nn.Module | None,
    ) -> None:
        super().__init__()
        self.convnext_encoder = convnext_encoder
        self.bottleneck_proj = bottleneck_proj
        self.spatial_compressor = spatial_compressor

    def forward(self, x: Tensor) -> Tensor:
        z = self.convnext_encoder(x)        # (B, dims[-1], H', W')
        z = self.bottleneck_proj(z)         # (B, bn_dim, H', W')
        if self.spatial_compressor is not None:
            z = self.spatial_compressor(z)  # (B, d_model, H'', W'')
        B, C, H, W = z.shape
        return z.flatten(2).transpose(1, 2) # (B, N, d_model)


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class SpectrogramCNNPerceiverAutoEncoder(ModalityAutoEncoder):
    """ConvNeXt V2 autoencoder with two-stage bottleneck.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Token dimension after spatial compression.
    n_tokens : int
        Unused; token count is determined by spatial dims and strides.
        Kept for interface compatibility.
    dims : list[int] | None
        ConvNeXt channel dims per stage (default [256] for single stage).
    depths : list[int] | None
        ConvNeXt blocks per stage (default [6]).
    stem_stride : int
        Stride of the patchify stem convolution (default 4).
    bottleneck_dim : int | None
        Channel dimension at the first bottleneck.  If None, defaults to d_model.
    compress_stride : int
        Additional spatial compression after channel bottleneck.
        1 = no extra compression (full spatial grid as tokens).
        4 = 4× spatial reduction via two stride-2 conv layers.
    kernel_size : int
        Depthwise conv kernel size (default 7).
    n_heads, n_self_layers, n_dec_self_layers, dropout : —
        Unused; kept for CLI compatibility.
    fsq_levels : list[int] | None
        Unused; kept for CLI compatibility.
    max_freq_patches, max_time_patches : int
        Unused; kept for CLI compatibility.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        dims: list[int] | None = None,
        depths: list[int] | None = None,
        stem_stride: int = 4,
        bottleneck_dim: int | None = None,
        compress_stride: int = 1,
        kernel_size: int = 7,
        # Kept for CLI compatibility — unused
        n_heads: int = 4,
        n_self_layers: int = 2,
        n_dec_self_layers: int = 2,
        dropout: float = 0.1,
        fsq_levels: list[int] | None = None,
        max_freq_patches: int = 64,
        max_time_patches: int = 512,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)

        if dims is None:
            dims = [256]
        if depths is None:
            depths = [6]
        if bottleneck_dim is None:
            bottleneck_dim = d_model

        assert len(dims) == len(depths), "dims and depths must have the same length"
        assert compress_stride in (1, 2, 4, 8), (
            f"compress_stride must be 1, 2, 4, or 8, got {compress_stride}"
        )

        self.dims = dims
        self.depths = depths
        self.stem_stride = stem_stride
        self.bottleneck_dim = bottleneck_dim
        self.compress_stride = compress_stride
        n_stages = len(dims)
        self.total_stride = stem_stride * (2 ** (n_stages - 1))

        # --- ConvNeXt Encoder ---
        self.convnext_encoder = _ConvNeXtV2Encoder(
            in_channels=n_channels,
            dims=dims,
            depths=depths,
            stem_stride=stem_stride,
            kernel_size=kernel_size,
        )

        # --- Stage 1: Channel bottleneck (dims[-1] → bottleneck_dim) ---
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(dims[-1], bottleneck_dim, 1),
            nn.GELU(),
        )

        # --- Stage 2: Spatial compressor (strided convs) ---
        if compress_stride > 1:
            # Build a chain of stride-2 conv layers
            n_compress_layers = {2: 1, 4: 2, 8: 3}[compress_stride]
            compress_layers: list[nn.Module] = []
            in_ch = bottleneck_dim
            for i in range(n_compress_layers):
                out_ch = d_model if i == n_compress_layers - 1 else bottleneck_dim
                compress_layers.extend([
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                    _gn(out_ch),
                    nn.GELU(),
                ])
                in_ch = out_ch
            self.spatial_compressor = nn.Sequential(*compress_layers)

            # Mirror: spatial decompressor (upsample + conv)
            decompress_layers: list[nn.Module] = []
            in_ch = d_model
            for i in range(n_compress_layers):
                out_ch = bottleneck_dim
                decompress_layers.extend([
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    _gn(out_ch),
                    nn.GELU(),
                ])
                in_ch = out_ch
            self.spatial_decompressor = nn.Sequential(*decompress_layers)
        else:
            self.spatial_compressor = None
            self.spatial_decompressor = None

        # --- Decoder: expand bottleneck + ConvNeXt decode ---
        self.bottleneck_expand = nn.Sequential(
            nn.Conv2d(bottleneck_dim, dims[-1], 1),
            nn.GELU(),
        )

        self.convnext_decoder = _ConvNeXtV2Decoder(
            out_channels=n_channels,
            dims=list(reversed(dims)),
            depths=list(reversed(depths)),
            stem_stride=stem_stride,
            kernel_size=kernel_size,
        )

        # --- Expose standalone encoder for tests ---
        self.encoder = _ConvNeXtBottleneckEncoder(
            convnext_encoder=self.convnext_encoder,
            bottleneck_proj=self.bottleneck_proj,
            spatial_compressor=self.spatial_compressor,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        B, C, F_orig, T_orig = x.shape

        # ── 1. Pad to align with total_stride * compress_stride ───────
        full_stride = self.total_stride * self.compress_stride
        pad_f = (full_stride - F_orig % full_stride) % full_stride
        pad_t = (full_stride - T_orig % full_stride) % full_stride
        if pad_f > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_f))

        # ── 2. ConvNeXt encode ────────────────────────────────────────
        z = self.convnext_encoder(x)       # (B, dims[-1], H', W')

        # ── 3. Channel bottleneck ─────────────────────────────────────
        z_bn = self.bottleneck_proj(z)     # (B, bn_dim, H', W')

        # ── 4. Spatial compression (if enabled) ───────────────────────
        if self.spatial_compressor is not None:
            z_compressed = self.spatial_compressor(z_bn)  # (B, d_model, H'', W'')
        else:
            z_compressed = z_bn

        # ── 5. Token representation for monitoring ────────────────────
        z_tokens = z_compressed.flatten(2).transpose(1, 2)  # (B, N, d_model)

        # ── 6. Spatial decompression (if enabled) ─────────────────────
        if self.spatial_decompressor is not None:
            z_dec = self.spatial_decompressor(z_compressed)  # (B, bn_dim, H', W')
        else:
            z_dec = z_bn

        # ── 7. Expand + ConvNeXt decode ───────────────────────────────
        z_dec = self.bottleneck_expand(z_dec)  # (B, dims[-1], H', W')
        reconstructed = self.convnext_decoder(z_dec)

        # ── 8. Crop to original spatial dims ──────────────────────────
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed
        return reconstructed, z_tokens
