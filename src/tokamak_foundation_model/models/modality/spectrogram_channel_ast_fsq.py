"""Channel-Attention AST + FSQ autoencoder for tokamak spectrogram diagnostics.

Uses **per-channel frame embedding** (``Linear(F*fw, d_model)`` — nearly 1:1 for
CO2) and **transformer attention across channels** to capture inter-channel
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

Bottleneck:
  pre_fsq → FSQ → post_fsq

Decoder:
  Reshape → (B, C, N, d_model)
  + decoder channel_pos_embed + time_pos_embed
  n_dec_layers × ChannelTimeBlock
  Frame unembed: Linear(d_model → F*fw)

Return contract
---------------
Training : (reconstructed, indices) — indices are LongTensor (B, C*N) of scalar
           mixed-radix codebook indices, useful for monitoring utilisation.
Eval     : reconstructed             — shape (B, C, F, T) matching input.
"""

import torch
import torch.nn as nn
from torch import Tensor, LongTensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_fsq_vae import FSQ
from tokamak_foundation_model.models.modality.spectrogram_cnn1d import _ConvNeXtV2Block1d


# ---------------------------------------------------------------------------
# Channel merge: cross-attention C→1 per time frame
# ---------------------------------------------------------------------------

class _ChannelMerge(nn.Module):
    """Cross-attention that merges C channel tokens into 1 per time frame.

    A single learned query attends into the C channel key-value tokens at
    each time step independently.  Pre-norm on both query and key/value.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.merge_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.merge_query, std=0.02)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, N, D) → (B, N, D)."""
        B, C, N, D = x.shape
        kv = x.permute(0, 2, 1, 3).reshape(B * N, C, D)  # (B*N, C, D)
        q = self.merge_query.expand(B * N, -1, -1)        # (B*N, 1, D)
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        merged, _ = self.cross_attn(q, kv, kv)            # (B*N, 1, D)
        return merged.reshape(B, N, D)


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

        self.channel_pos_embed = nn.Parameter(
            torch.zeros(1, max_channels, 1, d_model)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, 1, max_time_frames, d_model)
        )
        nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _ChannelTimeBlock(d_model, n_heads, dropout, time_conv_kernel)
            for _ in range(n_layers)
        ])
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
            .permute(0, 1, 3, 2, 4)       # (B, C, N, F, fw)
            .reshape(B, C, n_frames, F * fw)
        )
        tokens = self.frame_proj(frames)   # (B, C, N, d_model)

        # Add positional embeddings
        tokens = tokens + self.channel_pos_embed[:, :C] + self.time_pos_embed[:, :, :n_frames]

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
        self.channel_pos_embed = nn.Parameter(
            torch.zeros(1, max_channels, 1, d_model)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, 1, max_time_frames, d_model)
        )
        nn.init.trunc_normal_(self.channel_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            _ChannelTimeBlock(d_model, n_heads, dropout, time_conv_kernel)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, n_channels: int, n_frames: int) -> Tensor:
        """(B, C*N, d_model) → (B, C, N, d_model).

        Reshapes flat token sequence back to (B, C, N, D), adds decoder
        positional embeddings, runs blocks, and returns (B, C, N, D).
        """
        B = tokens.shape[0]
        D = tokens.shape[-1]
        tokens = tokens.reshape(B, n_channels, n_frames, D)

        tokens = tokens + self.channel_pos_embed[:, :n_channels] + self.time_pos_embed[:, :, :n_frames]

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Full Channel-AST-FSQ autoencoder
# ---------------------------------------------------------------------------

class SpectrogramChannelASTFSQAutoEncoder(ModalityAutoEncoder):
    """Channel-Attention AST + FSQ autoencoder for multichannel spectrograms.

    Each token spans the full frequency axis for a **single channel** and
    ``frame_width`` time steps.  Channel correlations are captured by
    transformer attention; temporal context by local ConvNeXt convolutions.
    FSQ provides the discrete bottleneck.

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
    fsq_levels : list[int] | None
        FSQ quantization levels per dimension (default [8, 5, 5, 5, 5]).
        Pass an empty list ``[]`` to disable quantization entirely
        (continuous bottleneck — encoder feeds directly into decoder).
    channel_merge : bool
        If True, insert a cross-attention layer after the encoder that
        merges C channel tokens into 1 per time frame.  The decoder
        reverses this by replicating + channel positional embeddings,
        relying on ChannelTimeBlock attention to re-separate.
        Reduces token count from C*N to N (default False).
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
        fsq_levels: list[int] | None = None,
        channel_merge: bool = False,
        max_channels: int = 64,
        max_time_frames: int = 2048,
        time_conv_kernel: int = 7,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.frame_width = frame_width

        if fsq_levels is None:
            fsq_levels = [8, 5, 5, 5, 5]
        self.use_fsq = len(fsq_levels) > 0
        self.use_channel_merge = channel_merge

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

        # Channel merge (optional): C tokens per frame → 1 token per frame
        if self.use_channel_merge:
            self.channel_merge = _ChannelMerge(d_model, n_heads, dropout)

        # FSQ bottleneck (optional)
        if self.use_fsq:
            fsq_dim = len(fsq_levels)
            self.pre_fsq = nn.Linear(d_model, fsq_dim)
            nn.init.normal_(self.pre_fsq.weight, std=0.02)
            nn.init.zeros_(self.pre_fsq.bias)
            self.fsq = FSQ(fsq_levels)
            self.post_fsq = nn.Linear(fsq_dim, d_model)

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
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, LongTensor] | Tensor:
        """Forward pass.

        Training : returns (reconstructed, indices) where indices is a
                   LongTensor (B, C*N) of per-token codebook indices.
        Eval     : returns reconstructed — shape (B, C, F, T) matching input.
        """
        B, C, F, T_orig = x.shape
        fw = self.frame_width

        # ── 1. Pad T to multiple of frame_width ──────────────────────────
        pad_t = (fw - T_orig % fw) % fw
        if pad_t > 0:
            x = nn.functional.pad(x, (0, pad_t))
        T_padded = T_orig + pad_t
        n_frames = T_padded // fw

        # ── 2. Per-channel frame embedding + encoder pos embed ────────────
        frames = (
            x.reshape(B, C, F, n_frames, fw)
            .permute(0, 1, 3, 2, 4)       # (B, C, N, F, fw)
            .reshape(B, C, n_frames, F * fw)
        )
        tokens = self.encoder.frame_proj(frames)  # (B, C, N, d_model)
        tokens = (
            tokens
            + self.encoder.channel_pos_embed[:, :C]
            + self.encoder.time_pos_embed[:, :, :n_frames]
        )

        # ── 3. Encoder blocks ─────────────────────────────────────────────
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens_enc = self.encoder.norm(tokens)  # (B, C, N, d_model)

        # ── 4. Optional channel merge + optional FSQ bottleneck ─────────
        if self.use_channel_merge:
            # Merge: (B, C, N, D) → (B, N, D)
            tokens_merged = self.channel_merge(tokens_enc)
            if self.use_fsq:
                z = self.pre_fsq(tokens_merged)
                z_q, indices = self.fsq(z)
                tokens_latent = self.post_fsq(z_q)
            else:
                tokens_latent = tokens_merged
                indices = None
            # Expand: replicate merged token across C channels
            # Decoder's channel_pos_embed + ChannelTimeBlocks re-separate
            tokens_dec = (
                tokens_latent
                .unsqueeze(1)
                .expand(-1, C, -1, -1)
                .contiguous()
                .reshape(B, C * n_frames, -1)
            )
        else:
            tokens_flat = tokens_enc.reshape(B, C * n_frames, -1)
            if self.use_fsq:
                z = self.pre_fsq(tokens_flat)
                z_q, indices = self.fsq(z)
                tokens_dec = self.post_fsq(z_q)
            else:
                tokens_dec = tokens_flat
                indices = None

        # ── 5. Decoder ────────────────────────────────────────────────────
        decoded = self.decoder(tokens_dec, C, n_frames)  # (B, C, N, d_model)

        # ── 6. Frame unembed + crop to original T ─────────────────────────
        pixels = self.frame_unembed(decoded)  # (B, C, N, F*fw)
        reconstructed = (
            pixels
            .reshape(B, C, n_frames, F, fw)
            .permute(0, 1, 3, 2, 4)           # (B, C, F, N, fw)
            .reshape(B, C, F, T_padded)
        )
        reconstructed = reconstructed[:, :, :, :T_orig]

        if not self.training:
            return reconstructed
        if indices is not None:
            return reconstructed, indices
        return reconstructed
