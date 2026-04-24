"""Channel-Attention AST autoencoder for tokamak spectrogram diagnostics.

Uses **per-channel frame embedding** (``Linear(F*fw, d_model)``) and
**transformer attention across channels** to capture inter-channel
correlations.  Physics is local in time, so temporal context uses local 1D
ConvNeXt convolutions instead of full attention.

This avoids the per-token ``C*F*fw → d_model`` compression of the original
AST-FSQ, which becomes unworkable for high-channel-count signals (ECE C=40+).

Architecture
------------
Encoder:
  Per-channel frame embed: (B, C, N, F*fw) → Linear → (B, C, N, d_model)
  + channel_pos_embed + time_pos_embed
  n_enc_layers × ChannelTimeBlock:
    1. Channel attn: (B*N, C, D) → TransformerEncoderLayer
    2. Time conv:    (B*C, D, N) → ConvNeXtV2Block1d
  Flatten → (B, C*N, d_model)

Decoder:
  Reshape → (B, C, N, d_model)
  + decoder channel_pos_embed + time_pos_embed
  n_dec_layers × ChannelTimeBlock
  Frame unembed: Linear(d_model → F*fw)

Return contract
---------------
Training : (reconstructed, z_tokens) — z_tokens is (B, C*N, d_model) encoder
           output, useful for downstream latent-space work.
Eval     : reconstructed             — shape (B, C, F, T) matching input.
"""

import torch
import torch.nn as nn
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder

# ---------------------------------------------------------------------------
# 1D ConvNeXt building blocks (inlined for self-containment)
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
            dim,
            dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=dim,
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
# Building block: channel attention + temporal convolution
# ---------------------------------------------------------------------------


class _ChannelTimeBlock(nn.Module):
    """Channel attention followed by temporal ConvNeXt convolution.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads for channel attention.
    dropout : float
        Dropout rate.
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        time_conv_kernel: int,
    ) -> None:
        super().__init__()
        self.channel_attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_conv = _ConvNeXtV2Block1d(d_model, time_conv_kernel)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, N, D) → (B, C, N, D)."""
        B, C, N, D = x.shape

        # 1. Channel attention: merge batch and time → (B*N, C, D)
        x_ch = x.permute(0, 2, 1, 3).reshape(B * N, C, D)
        x_ch = self.channel_attn(x_ch)
        x = x_ch.reshape(B, N, C, D).permute(0, 2, 1, 3)  # (B, C, N, D)

        # 2. Time conv: merge batch and channels → (B*C, D, N)
        x_t = x.reshape(B * C, N, D).permute(0, 2, 1)  # (B*C, D, N)
        x_t = self.time_conv(x_t)
        x = x_t.permute(0, 2, 1).reshape(B, C, N, D)  # (B, C, N, D)

        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class _ChannelASTEncoder(nn.Module):
    """Per-channel frame encoder with channel attention + temporal conv.

    Parameters
    ----------
    freq_bins : int
        Frequency dimension (F).
    frame_width : int
        Number of time steps per frame token.
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads for channel attention.
    n_layers : int
        Number of ChannelTimeBlocks.
    dropout : float
        Dropout rate.
    max_channels : int
        Capacity of the channel positional embedding table.
    max_time_frames : int
        Capacity of the time positional embedding table.
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt blocks.
    """

    def __init__(
        self,
        freq_bins: int,
        frame_width: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_channels: int,
        max_time_frames: int,
        time_conv_kernel: int,
    ) -> None:
        super().__init__()
        self.freq_bins = freq_bins
        self.frame_width = frame_width

        self.frame_proj = nn.Linear(freq_bins * frame_width, d_model)

        self.channel_pos_embed = nn.Parameter(torch.zeros(1, max_channels, 1, d_model))
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 1, max_time_frames, d_model))
        nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _ChannelTimeBlock(d_model, n_heads, dropout, time_conv_kernel)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, F, T) → (B, C*N, d_model).

        Pads T to a multiple of frame_width before framing.
        """
        B, C, F, T = x.shape
        fw = self.frame_width

        # Pad T to multiple of frame_width
        pad_t = (fw - T % fw) % fw
        if pad_t > 0:
            x = nn.functional.pad(x, (0, pad_t))
        T_padded = T + pad_t
        n_frames = T_padded // fw

        # Per-channel frame embed: (B, C, F, N, fw) → (B, C, N, F*fw) → Linear
        frames = (
            x.reshape(B, C, F, n_frames, fw)
            .permute(0, 1, 3, 2, 4)  # (B, C, N, F, fw)
            .reshape(B, C, n_frames, F * fw)
        )
        tokens = self.frame_proj(frames)  # (B, C, N, d_model)

        # Add positional embeddings
        tokens = (
            tokens
            + self.channel_pos_embed[:, :C]
            + self.time_pos_embed[:, :, :n_frames]
        )

        # ChannelTimeBlocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        # Flatten to (B, C*N, d_model)
        return tokens.reshape(B, C * n_frames, tokens.shape[-1])


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class _ChannelASTDecoder(nn.Module):
    """Per-channel frame decoder with channel attention + temporal conv.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads.
    n_layers : int
        Number of ChannelTimeBlocks.
    dropout : float
        Dropout rate.
    max_channels : int
        Capacity of the channel positional embedding table.
    max_time_frames : int
        Capacity of the time positional embedding table.
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt blocks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_channels: int,
        max_time_frames: int,
        time_conv_kernel: int,
    ) -> None:
        super().__init__()
        self.channel_pos_embed = nn.Parameter(torch.zeros(1, max_channels, 1, d_model))
        self.time_pos_embed = nn.Parameter(torch.zeros(1, 1, max_time_frames, d_model))
        nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _ChannelTimeBlock(d_model, n_heads, dropout, time_conv_kernel)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, n_channels: int, n_frames: int) -> Tensor:
        """(B, C*N, d_model) → (B, C, N, d_model).

        Reshapes flat token sequence back to (B, C, N, D), adds decoder
        positional embeddings, runs blocks, and returns (B, C, N, D).
        """
        B = tokens.shape[0]
        D = tokens.shape[-1]
        tokens = tokens.reshape(B, n_channels, n_frames, D)

        tokens = (
            tokens
            + self.channel_pos_embed[:, :n_channels]
            + self.time_pos_embed[:, :, :n_frames]
        )

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Full Channel-AST autoencoder
# ---------------------------------------------------------------------------


class SpectrogramChannelASTAutoEncoder(ModalityAutoEncoder):
    """Channel-Attention AST autoencoder for multichannel spectrograms.

    Each token spans the full frequency axis for a **single channel** and
    ``frame_width`` time steps.  Channel correlations are captured by
    transformer attention; temporal context by local ConvNeXt convolutions.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Hidden dimension.
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    freq_bins : int
        Frequency dimension of the input spectrogram.
    frame_width : int
        Number of time steps per frame token (default 2).
    n_enc_layers, n_dec_layers : int
        Depth for encoder and decoder (default 4 each).
    n_heads : int
        Attention heads (default 4).
    dropout : float
        Dropout rate (default 0.1).
    max_channels : int
        Channel positional embedding table capacity (default 64).
    max_time_frames : int
        Time positional embedding table capacity (default 2048).
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt blocks (default 7).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        freq_bins: int = 512,
        frame_width: int = 2,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_channels: int = 64,
        max_time_frames: int = 2048,
        time_conv_kernel: int = 7,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.frame_width = frame_width

        # Encoder
        self.encoder = _ChannelASTEncoder(
            freq_bins=freq_bins,
            frame_width=frame_width,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
            max_channels=max_channels,
            max_time_frames=max_time_frames,
            time_conv_kernel=time_conv_kernel,
        )

        # Decoder
        self.decoder = _ChannelASTDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            dropout=dropout,
            max_channels=max_channels,
            max_time_frames=max_time_frames,
            time_conv_kernel=time_conv_kernel,
        )

        # Frame unembed
        self.frame_unembed = nn.Linear(d_model, freq_bins * frame_width)

    # ------------------------------------------------------------------
    # Encode / Decode / Forward
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> tuple[Tensor, int, int, int]:
        """Encode a spectrogram into latent tokens.

        Parameters
        ----------
        x : Tensor
            Input spectrogram, shape ``(B, C, F, T)``.

        Returns
        -------
        z_tokens : Tensor
            Latent tokens, shape ``(B, C*N, d_model)`` where
            ``N = ceil(T / frame_width)``.
        n_channels : int
            Number of channels (C), needed by :meth:`decode`.
        n_frames : int
            Number of time frames (N), needed by :meth:`decode`.
        T_orig : int
            Original time length before padding, needed by :meth:`decode`
            to crop the reconstruction.
        """
        B, C, F, T_orig = x.shape
        fw = self.frame_width

        pad_t = (fw - T_orig % fw) % fw
        if pad_t > 0:
            x = nn.functional.pad(x, (0, pad_t))
        n_frames = (T_orig + pad_t) // fw

        frames = (
            x.reshape(B, C, F, n_frames, fw)
            .permute(0, 1, 3, 2, 4)  # (B, C, N, F, fw)
            .reshape(B, C, n_frames, F * fw)
        )
        tokens = self.encoder.frame_proj(frames)
        tokens = (
            tokens
            + self.encoder.channel_pos_embed[:, :C]
            + self.encoder.time_pos_embed[:, :, :n_frames]
        )

        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)  # (B, C, N, d_model)

        z_tokens = tokens.reshape(B, C * n_frames, -1)
        return z_tokens, C, n_frames, T_orig

    def decode(
        self,
        z_tokens: Tensor,
        n_channels: int,
        n_frames: int,
        T_orig: int,
    ) -> Tensor:
        """Decode latent tokens back to a spectrogram.

        Parameters
        ----------
        z_tokens : Tensor
            Latent tokens, shape ``(B, C*N, d_model)``.
        n_channels : int
            Number of channels (C).
        n_frames : int
            Number of time frames (N).
        T_orig : int
            Original time length; the output is cropped to this size.

        Returns
        -------
        Tensor
            Reconstructed spectrogram, shape ``(B, C, F, T_orig)``.
        """
        B = z_tokens.shape[0]
        F = self.freq_bins
        fw = self.frame_width

        decoded = self.decoder(z_tokens, n_channels, n_frames)  # (B, C, N, d_model)
        pixels = self.frame_unembed(decoded)  # (B, C, N, F*fw)
        reconstructed = (
            pixels.reshape(B, n_channels, n_frames, F, fw)
            .permute(0, 1, 3, 2, 4)  # (B, C, F, N, fw)
            .reshape(B, n_channels, F, n_frames * fw)
        )
        return reconstructed[:, :, :, :T_orig]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full encode-decode pass.

        Returns (reconstructed, z_tokens):
          - reconstructed: ``(B, C, F, T)`` matching input shape.
          - z_tokens:      ``(B, C*N, d_model)`` encoder latent tokens.
        """
        z_tokens, C, n_frames, T_orig = self.encode(x)
        reconstructed = self.decode(z_tokens, C, n_frames, T_orig)
        return reconstructed, z_tokens
