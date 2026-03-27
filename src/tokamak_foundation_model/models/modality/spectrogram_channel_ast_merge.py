"""Channel-Attention AST with multi-query channel pool/expand for tokamak spectrograms.

Builds on the Channel-AST architecture (per-channel frame embedding, channel
self-attention, temporal ConvNeXt) but adds a **channel pool/expand bottleneck**
that compresses C channel tokens into k pool tokens per time frame, making the
latent token count **k×N regardless of C**.

This solves the scaling problem: CO2 (C=4) and ECE (C=40) produce the same
number of tokens for the fusion transformer.

Architecture
------------
Encoder:
  _ChannelASTEncoder (reused from spectrogram_channel_ast_fsq):
    Per-channel frame embed → channel/time pos embeds → ChannelTimeBlocks
    → (B, C, N, d_model)

Channel pool:
  k learned queries cross-attend into C channel tokens per time frame
  → (B, k, N, d_model)

Latent: flatten to (B, k*N, d_model) — fusion transformer sees this

Channel expand:
  C learned channel queries cross-attend into k pool tokens per time frame
  → (B, C, N, d_model)

Decoder:
  _ChannelASTDecoder (reused from spectrogram_channel_ast_fsq):
    channel/time pos embeds → ChannelTimeBlocks → (B, C, N, d_model)

Frame unembed → reshape → crop to original T

Return contract
---------------
Training and eval both return ``reconstructed`` (no FSQ, no indices).
"""

import torch
import torch.nn as nn
from torch import Tensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_channel_ast_fsq import (
    _ChannelASTEncoder,
    _ChannelASTDecoder,
)


# ---------------------------------------------------------------------------
# Channel pool: cross-attention C → k per time frame
# ---------------------------------------------------------------------------

class _PoolLayer(nn.Module):
    """Single cross-attention layer: q attends to kv, with pre-norm + residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Pre-norm cross-attention with residual on q."""
        out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        return q + out


class _ChannelPool(nn.Module):
    """Cross-attention that pools C channel tokens into k per time frame.

    Stacks ``n_layers`` cross-attention layers with residual connections
    for deeper information transfer from C channels to k queries.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_merge_queries : int
        Number of pool queries (k).
    n_heads : int
        Attention heads.
    dropout : float
        Dropout rate.
    n_layers : int
        Number of cross-attention layers (default 1).
    """

    def __init__(
        self,
        d_model: int,
        n_merge_queries: int,
        n_heads: int,
        dropout: float,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.pool_queries = nn.Parameter(
            torch.zeros(1, n_merge_queries, d_model)
        )
        nn.init.trunc_normal_(self.pool_queries, std=0.02)
        self.layers = nn.ModuleList([
            _PoolLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, N, D) → (B, k, N, D)."""
        B, C, N, D = x.shape
        kv = x.permute(0, 2, 1, 3).reshape(B * N, C, D)    # (B*N, C, D)
        q = self.pool_queries.expand(B * N, -1, -1)          # (B*N, k, D)
        for layer in self.layers:
            q = layer(q, kv)
        q = self.norm(q)
        k = self.pool_queries.shape[1]
        return q.reshape(B, N, k, D).permute(0, 2, 1, 3)    # (B, k, N, D)


# ---------------------------------------------------------------------------
# Channel expand: cross-attention k → C per time frame
# ---------------------------------------------------------------------------

class _ChannelExpand(nn.Module):
    """Cross-attention that expands k pool tokens to C channel tokens per frame.

    Stacks ``n_layers`` cross-attention layers with residual connections
    for deeper reconstruction from k pool tokens to C channels.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Attention heads.
    dropout : float
        Dropout rate.
    max_channels : int
        Capacity of the channel query table.
    n_layers : int
        Number of cross-attention layers (default 1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_channels: int = 64,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.channel_queries = nn.Parameter(
            torch.zeros(1, max_channels, d_model)
        )
        nn.init.trunc_normal_(self.channel_queries, std=0.02)
        self.layers = nn.ModuleList([
            _PoolLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, n_channels: int) -> Tensor:
        """(B, k, N, D) → (B, C, N, D) where C = n_channels."""
        B, k, N, D = x.shape
        kv = x.permute(0, 2, 1, 3).reshape(B * N, k, D)                  # (B*N, k, D)
        q = self.channel_queries[:, :n_channels].expand(B * N, -1, -1)    # (B*N, C, D)
        for layer in self.layers:
            q = layer(q, kv)
        q = self.norm(q)
        return q.reshape(B, N, n_channels, D).permute(0, 2, 1, 3)         # (B, C, N, D)


# ---------------------------------------------------------------------------
# Encoder wrapper for test harness (model.encoder(x))
# ---------------------------------------------------------------------------

class _MergeEncoder(nn.Module):
    """Wraps _ChannelASTEncoder + _ChannelPool as a single module.

    Exposes the full encode pipeline (input → latent tokens) for the test
    harness which calls ``model.encoder(x)`` and checks finiteness.
    Shares parameters with the autoencoder — no duplication.
    """

    def __init__(
        self,
        ast_encoder: _ChannelASTEncoder,
        channel_pool: _ChannelPool,
        n_channels: int,
    ) -> None:
        super().__init__()
        self.ast_encoder = ast_encoder
        self.channel_pool = channel_pool
        self.n_channels = n_channels

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, F, T) → (B, k*N, D)."""
        tokens_flat = self.ast_encoder(x)  # (B, C*N, D)
        B, CN, D = tokens_flat.shape
        C = self.n_channels
        N = CN // C
        tokens_4d = tokens_flat.reshape(B, C, N, D)
        pooled = self.channel_pool(tokens_4d)  # (B, k, N, D)
        k = pooled.shape[1]
        return pooled.reshape(B, k * N, D)


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class SpectrogramChannelASTMergeAutoEncoder(ModalityAutoEncoder):
    """Channel-AST with multi-query channel pool/expand bottleneck.

    Uses per-channel frame embedding and channel attention (from the
    Channel-AST architecture) with a cross-attention bottleneck that pools
    C channels into k queries per time frame.  The latent token count is
    k × N_frames regardless of C, making it uniform across modalities.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Hidden dimension.
    n_tokens : int
        Unused; kept for ModalityAutoEncoder interface compatibility.
    freq_bins : int
        Frequency dimension of the input spectrogram.
    frame_width : int
        Number of time steps per frame token.
    n_enc_layers, n_dec_layers : int
        Depth for encoder and decoder.
    n_heads : int
        Attention heads.
    dropout : float
        Dropout rate.
    n_merge_queries : int
        Number of pool queries (k). Token count = k × ceil(T / frame_width).
    n_pool_layers : int
        Cross-attention depth for the channel pool (default 1).
    n_expand_layers : int
        Cross-attention depth for the channel expand (default 1).
    max_channels : int
        Channel positional embedding / query table capacity.
    max_time_frames : int
        Time positional embedding table capacity.
    time_conv_kernel : int
        Kernel size for temporal ConvNeXt blocks.
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
        n_merge_queries: int = 4,
        n_pool_layers: int = 1,
        n_expand_layers: int = 1,
        max_channels: int = 64,
        max_time_frames: int = 2048,
        time_conv_kernel: int = 7,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.frame_width = frame_width
        self.n_merge_queries = n_merge_queries

        # Encoder (reused from spectrogram_channel_ast_fsq)
        self._ast_encoder = _ChannelASTEncoder(
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

        # Channel pool: C → k per time frame
        self.channel_pool = _ChannelPool(
            d_model, n_merge_queries, n_heads, dropout, n_pool_layers,
        )

        # Channel expand: k → C per time frame
        self.channel_expand = _ChannelExpand(
            d_model, n_heads, dropout, max_channels, n_expand_layers,
        )

        # Decoder (reused from spectrogram_channel_ast_fsq)
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

        # Encoder wrapper for test harness
        self.encoder = _MergeEncoder(
            self._ast_encoder, self.channel_pool, n_channels,
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, F, T_orig = x.shape
        fw = self.frame_width

        # ── 1. Pad T to multiple of frame_width ──────────────────────────
        pad_t = (fw - T_orig % fw) % fw
        if pad_t > 0:
            x = nn.functional.pad(x, (0, pad_t))
        T_padded = T_orig + pad_t
        n_frames = T_padded // fw

        # ── 2. Per-channel frame embedding + encoder pos embeds ──────────
        frames = (
            x.reshape(B, C, F, n_frames, fw)
            .permute(0, 1, 3, 2, 4)       # (B, C, N, F, fw)
            .reshape(B, C, n_frames, F * fw)
        )
        tokens = self._ast_encoder.frame_proj(frames)  # (B, C, N, d_model)
        tokens = (
            tokens
            + self._ast_encoder.channel_pos_embed[:, :C]
            + self._ast_encoder.time_pos_embed[:, :, :n_frames]
        )

        # ── 3. Encoder ChannelTimeBlocks ─────────────────────────────────
        for block in self._ast_encoder.blocks:
            tokens = block(tokens)
        tokens_enc = self._ast_encoder.norm(tokens)  # (B, C, N, d_model)

        # ── 4. Channel pool: (B, C, N, D) → (B, k, N, D) ───────────────
        tokens_pooled = self.channel_pool(tokens_enc)

        # ── 5. Channel expand: (B, k, N, D) → (B, C, N, D) ─────────────
        tokens_expanded = self.channel_expand(tokens_pooled, C)

        # ── 6. Decoder ───────────────────────────────────────────────────
        tokens_dec = tokens_expanded.reshape(B, C * n_frames, -1)
        decoded = self.decoder(tokens_dec, C, n_frames)  # (B, C, N, d_model)

        # ── 7. Frame unembed + crop to original T ────────────────────────
        pixels = self.frame_unembed(decoded)  # (B, C, N, F*fw)
        reconstructed = (
            pixels
            .reshape(B, C, n_frames, F, fw)
            .permute(0, 1, 3, 2, 4)           # (B, C, F, N, fw)
            .reshape(B, C, F, T_padded)
        )
        return reconstructed[:, :, :, :T_orig]
