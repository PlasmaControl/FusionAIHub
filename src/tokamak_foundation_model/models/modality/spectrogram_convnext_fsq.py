"""ConvNeXt V2 + FSQ convolutional autoencoder for tokamak spectrogram diagnostics.

Uses a ConvNeXt V2-style encoder-decoder with Global Response Normalization (GRN)
and Finite Scalar Quantization (FSQ) bottleneck. CNNs have stronger spatial
inductive biases (local connectivity, translation equivariance, multi-scale
receptive fields) that better preserve fine spectral details compared to
ViT-based approaches.

Return contract (same as SpectrogramFSQVAEAutoEncoder)
------------------------------------------------------
Training : (reconstructed, indices) — indices are LongTensor (B, N) of scalar
           mixed-radix codebook indices, useful for monitoring utilisation.
Eval     : reconstructed             — shape (B, C, F, T) matching input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_fsq_vae import FSQ


# ---------------------------------------------------------------------------
# GRN — Global Response Normalization (ConvNeXt V2)
# ---------------------------------------------------------------------------

class GRN(nn.Module):
    """Global Response Normalization (Woo et al. 2023, ConvNeXt V2).

    Computes per-channel spatial L2 norms, then inter-channel contrast,
    and applies a learnable gating with residual connection.

    Operates on channels-last layout: (..., C).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C) — channels-last
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)   # (B, 1, 1, C)
        return self.gamma * (x * nx) + self.beta + x


# ---------------------------------------------------------------------------
# ConvNeXt V2 Block
# ---------------------------------------------------------------------------

class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: DwConv -> LN -> Linear -> GELU -> GRN -> Linear + skip.

    Parameters
    ----------
    dim : int
        Number of input/output channels.
    kernel_size : int
        Depthwise convolution kernel size (default 7).
    """

    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x + residual


# ---------------------------------------------------------------------------
# Down / Up sampling modules
# ---------------------------------------------------------------------------

class _Downsample2d(nn.Module):
    """LayerNorm + strided Conv2d for spatial downsampling."""

    def __init__(self, in_dim: int, out_dim: int, stride: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return self.conv(x)


class _Upsample2d(nn.Module):
    """Nearest-neighbor upsample + LayerNorm + Conv2d.

    Avoids checkerboard artifacts from transposed convolutions.
    """

    def __init__(self, in_dim: int, out_dim: int, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class _ConvNeXtV2Encoder(nn.Module):
    """ConvNeXt V2 encoder: Stem -> [Stage -> Downsample] x (S-1) -> Stage.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    dims : list[int]
        Channel dimensions per stage.
    depths : list[int]
        Number of ConvNeXt blocks per stage.
    stem_stride : int
        Stride of the patchify stem convolution.
    kernel_size : int
        Depthwise conv kernel size in each block.
    """

    def __init__(
        self,
        in_channels: int,
        dims: list[int],
        depths: list[int],
        stem_stride: int = 4,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        assert len(dims) == len(depths)

        # Patchify stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=stem_stride, stride=stem_stride),
            nn.LayerNorm([dims[0], 1, 1]),  # placeholder, replaced by permute-norm
        )
        # Replace stem norm with proper channels-last LayerNorm
        self.stem_norm = nn.LayerNorm(dims[0])

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(dims)):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dims[i], kernel_size) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            if i < len(dims) - 1:
                self.downsamples.append(_Downsample2d(dims[i], dims[i + 1], stride=2))

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.stem[0](x)  # Conv2d
        x = x.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        x = self.stem_norm(x)
        x = x.permute(0, 3, 1, 2)  # -> (B, C, H, W)

        # Stages + downsamples
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        return x


class _ConvNeXtV2Decoder(nn.Module):
    """ConvNeXt V2 decoder: Stage -> [Upsample -> Stage] x (S-1) -> Head.

    Mirror of the encoder. The head upsamples by stem_stride and projects
    back to the original number of channels.

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    dims : list[int]
        Channel dimensions per stage (reversed from encoder order).
    depths : list[int]
        Number of ConvNeXt blocks per stage (reversed from encoder order).
    stem_stride : int
        Scale factor for the output head (inverse of encoder stem).
    kernel_size : int
        Depthwise conv kernel size in each block.
    """

    def __init__(
        self,
        out_channels: int,
        dims: list[int],
        depths: list[int],
        stem_stride: int = 4,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        assert len(dims) == len(depths)

        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(len(dims)):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dims[i], kernel_size) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            if i < len(dims) - 1:
                self.upsamples.append(_Upsample2d(dims[i], dims[i + 1], scale=2))

        # Output head: upsample by stem_stride + project to output channels
        self.head = nn.Sequential(
            _Upsample2d(dims[-1], dims[-1], scale=stem_stride),
            nn.Conv2d(dims[-1], out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.upsamples):
                x = self.upsamples[i](x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class SpectrogramConvNeXtFSQAutoEncoder(ModalityAutoEncoder):
    """ConvNeXt V2 + FSQ autoencoder for multichannel spectrogram signals.

    Architecture
    ------------
    Pad -> ConvNeXt Encoder -> Flatten spatial -> pre_fsq Linear ->
    FSQ -> post_fsq Linear -> Reshape -> ConvNeXt Decoder -> Crop

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Kept for interface compatibility (unused by CNN; encoder output
        dim is ``dims[-1]``).
    n_tokens : int
        Kept for interface compatibility (unused; token count is
        determined by spatial dimensions and total stride).
    stem_stride : int
        Stride of the patchify stem convolution (default 4).
    dims : list[int] | None
        Channel dimensions per encoder stage (default [64, 128, 256]).
    depths : list[int] | None
        ConvNeXt blocks per encoder stage (default [2, 2, 6]).
    fsq_levels : list[int] | None
        FSQ quantization levels per dimension (default [8, 5, 5, 5, 5]).
    kernel_size : int
        Depthwise convolution kernel size in each block (default 7).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        stem_stride: int = 4,
        dims: list[int] | None = None,
        depths: list[int] | None = None,
        fsq_levels: list[int] | None = None,
        kernel_size: int = 7,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)

        if dims is None:
            dims = [64, 128, 256]
        if depths is None:
            depths = [2, 2, 6]
        if fsq_levels is None:
            fsq_levels = [8, 5, 5, 5, 5]

        assert len(dims) == len(depths), "dims and depths must have the same length"

        self.stem_stride = stem_stride
        self.dims = dims
        self.depths = depths
        n_stages = len(dims)
        # Total spatial downsampling: stem_stride * 2^(n_stages - 1)
        self.total_stride = stem_stride * (2 ** (n_stages - 1))
        fsq_dim = len(fsq_levels)
        bottleneck_dim = dims[-1]

        # Encoder
        self.encoder = _ConvNeXtV2Encoder(
            in_channels=n_channels,
            dims=dims,
            depths=depths,
            stem_stride=stem_stride,
            kernel_size=kernel_size,
        )

        # FSQ bottleneck
        self.pre_fsq = nn.Linear(bottleneck_dim, fsq_dim)
        nn.init.normal_(self.pre_fsq.weight, std=0.02)
        nn.init.zeros_(self.pre_fsq.bias)
        self.fsq = FSQ(fsq_levels)
        self.post_fsq = nn.Linear(fsq_dim, bottleneck_dim)

        # Decoder (reversed dims/depths)
        self.decoder = _ConvNeXtV2Decoder(
            out_channels=n_channels,
            dims=list(reversed(dims)),
            depths=list(reversed(depths)),
            stem_stride=stem_stride,
            kernel_size=kernel_size,
        )

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, LongTensor] | Tensor:
        """Forward pass.

        Training : returns (reconstructed, indices).
        Eval     : returns reconstructed — shape (B, C, F, T) matching input.
        """
        B, C, F_orig, T_orig = x.shape

        # ── 1. Pad to align with total_stride ─────────────────────────────
        pad_f = (self.total_stride - F_orig % self.total_stride) % self.total_stride
        pad_t = (self.total_stride - T_orig % self.total_stride) % self.total_stride
        if pad_f > 0 or pad_t > 0:
            x = F.pad(x, (0, pad_t, 0, pad_f))

        # ── 2. Encode ─────────────────────────────────────────────────────
        z = self.encoder(x)  # (B, dims[-1], F', T')
        B, C_enc, H, W = z.shape

        # ── 3. FSQ bottleneck ──────────────────────────────────────────────
        z_flat = z.permute(0, 2, 3, 1).reshape(B, H * W, C_enc)  # (B, N, dims[-1])
        z_proj = self.pre_fsq(z_flat)         # (B, N, fsq_dim)
        z_q, indices = self.fsq(z_proj)       # (B, N, fsq_dim), (B, N)
        z_dec = self.post_fsq(z_q)            # (B, N, dims[-1])
        z_dec = z_dec.reshape(B, H, W, C_enc).permute(0, 3, 1, 2)  # (B, dims[-1], H, W)

        # ── 4. Decode ─────────────────────────────────────────────────────
        reconstructed = self.decoder(z_dec)   # (B, C, F_padded, T_padded)

        # ── 5. Crop to original spatial dims ───────────────────────────────
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed
        return reconstructed, indices
