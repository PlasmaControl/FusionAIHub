"""Frame-based AST + FSQ autoencoder for tokamak spectrogram diagnostics.

Uses frame-based tokenization (MAE-AST style): each token spans the **full
frequency axis** and only a small time window (``frame_width`` time steps).
This is more natural for spectrograms where the frequency axis represents a
fixed set of physical measurements and each token should carry complete
spectral information for its time window.

Combined with Finite Scalar Quantization (FSQ, Mentzer et al. 2023) as the
discrete bottleneck.

Positional embeddings are 1D (time-only) since each token already spans
the full frequency range.

Return contract
---------------
Training : (reconstructed, indices) — indices are LongTensor (B, N) of scalar
           mixed-radix codebook indices, useful for monitoring utilisation.
Eval     : reconstructed             — shape (B, C, F, T) matching input.
"""

import torch
import torch.nn as nn
from torch import Tensor, LongTensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_fsq_vae import FSQ


# ---------------------------------------------------------------------------
# AST Encoder
# ---------------------------------------------------------------------------

class _ASTEncoder(nn.Module):
    """Frame-based encoder: each token = full frequency × frame_width time steps.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    freq_bins : int
        Frequency dimension (F) baked into the Linear input size.
    frame_width : int
        Number of time steps per frame token.
    d_model : int
        Transformer hidden dimension.
    n_heads, n_layers, dropout : —
    max_time_frames : int
        Capacity of the 1D time positional embedding table.
    """

    def __init__(
        self,
        n_channels: int,
        freq_bins: int,
        frame_width: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_time_frames: int,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.frame_width = frame_width

        self.frame_proj = nn.Linear(n_channels * freq_bins * frame_width, d_model)

        # 1D time-only positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_time_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, F, T) → (B, N, d_model) where N = ceil(T / frame_width).

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

        # Reshape to frames: (B, C, F, N, fw) → (B, N, C*F*fw)
        frames = (
            x.reshape(B, C, F, n_frames, fw)
            .permute(0, 3, 1, 2, 4)
            .reshape(B, n_frames, C * F * fw)
        )

        tokens = self.frame_proj(frames)  # (B, N, d_model)
        tokens = tokens + self.pos_embed[:, :n_frames]
        return self.transformer(tokens)


# ---------------------------------------------------------------------------
# AST Decoder
# ---------------------------------------------------------------------------

class _ASTDecoder(nn.Module):
    """Lightweight transformer decoder with separate 1D time positional embeddings.

    Parameters
    ----------
    d_model, n_heads, n_layers, dropout : —
    max_time_frames : int
        Capacity of the decoder 1D time positional embedding table.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_time_frames: int,
    ) -> None:
        super().__init__()
        # Separate decoder positional embeddings
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, max_time_frames, d_model))
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, tokens: Tensor, n_frames: int) -> Tensor:
        """(B, N, d_model) → (B, N, d_model) after adding pos embed + transformer."""
        tokens = tokens + self.dec_pos_embed[:, :n_frames]
        return self.transformer(tokens)


# ---------------------------------------------------------------------------
# Full AST-FSQ autoencoder
# ---------------------------------------------------------------------------

class SpectrogramASTFSQAutoEncoder(ModalityAutoEncoder):
    """Frame-based AST + FSQ autoencoder for multichannel spectrogram signals.

    Each token spans the full frequency axis and ``frame_width`` time steps,
    giving complete spectral information per token. FSQ provides the discrete
    bottleneck. Positional embeddings are 1D (time-only).

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Transformer hidden dimension.
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    freq_bins : int
        Frequency dimension of the input spectrogram.
    frame_width : int
        Number of time steps per frame token (default 2).
    n_enc_layers, n_dec_layers : int
        Transformer depth for encoder and decoder (default 4 each).
    n_heads : int
        Attention heads (default 4).
    dropout : float
        Dropout rate (default 0.1).
    fsq_levels : list[int] | None
        FSQ quantization levels per dimension (default [8, 5, 5, 5, 5]).
    max_time_frames : int
        Positional embedding table capacity (default 512).
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
        max_time_frames: int = 2048,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.frame_width = frame_width

        if fsq_levels is None:
            fsq_levels = [8, 5, 5, 5, 5]
        fsq_dim = len(fsq_levels)

        # Encoder
        self.encoder = _ASTEncoder(
            n_channels=n_channels,
            freq_bins=freq_bins,
            frame_width=frame_width,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
            max_time_frames=max_time_frames,
        )

        # FSQ bottleneck
        self.pre_fsq = nn.Linear(d_model, fsq_dim)
        nn.init.normal_(self.pre_fsq.weight, std=0.02)
        nn.init.zeros_(self.pre_fsq.bias)
        self.fsq = FSQ(fsq_levels)
        self.post_fsq = nn.Linear(fsq_dim, d_model)

        # Decoder
        self.decoder = _ASTDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            dropout=dropout,
            max_time_frames=max_time_frames,
        )

        # Frame unembed
        self.frame_unembed = nn.Linear(d_model, n_channels * freq_bins * frame_width)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, LongTensor] | Tensor:
        """Forward pass.

        Training : returns (reconstructed, indices) where indices is a
                   LongTensor (B, N) of per-token codebook indices.
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

        # ── 2. Frame embedding + encoder pos embed ───────────────────────
        frames = (
            x.reshape(B, C, F, n_frames, fw)
            .permute(0, 3, 1, 2, 4)
            .reshape(B, n_frames, C * F * fw)
        )
        tokens = self.encoder.frame_proj(frames)
        tokens = tokens + self.encoder.pos_embed[:, :n_frames]

        # ── 3. Encoder transformer ───────────────────────────────────────
        tokens_enc = self.encoder.transformer(tokens)

        # ── 4. FSQ bottleneck ────────────────────────────────────────────
        z = self.pre_fsq(tokens_enc)
        z_q, indices = self.fsq(z)
        tokens_dec = self.post_fsq(z_q)

        # ── 5. Decoder transformer + pos embed ───────────────────────────
        tokens_out = self.decoder(tokens_dec, n_frames)

        # ── 6. Frame unembed + crop to original T ────────────────────────
        pixels = self.frame_unembed(tokens_out)  # (B, N, C*F*fw)
        reconstructed = (
            pixels
            .reshape(B, n_frames, C, F, fw)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, C, F, T_padded)
        )
        reconstructed = reconstructed[:, :, :, :T_orig]

        if not self.training:
            return reconstructed
        return reconstructed, indices
