"""ViT-based FSQ-VAE Autoencoder for tokamak spectrogram diagnostics.

Uses Finite Scalar Quantization (Mentzer et al. 2023) as the bottleneck instead
of the masked-autoencoder training objective.  The result is a reconstruction-
first autoencoder with a fully deterministic, single-pass encode → quantize →
decode pipeline suitable for inference-time use in a simulation pipeline.

FSQ replaces the discrete codebook lookup of VQ-VAE with per-dimension tanh-
bounded rounding and a straight-through gradient estimator.  There is no
commitment loss, no posterior collapse, and no codebook reset heuristics.

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
from tokamak_foundation_model.models.modality.spectrogram_mae import (
    FlatPatchEmbed2d,
    FlatPatchUnembed2d,
    PerChannelPatchEmbed2d,
    PerChannelPatchUnembed2d,
    _build_2d_pos_embed,
    _ViTEncoder,
)


# ---------------------------------------------------------------------------
# FSQ module
# ---------------------------------------------------------------------------

class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023).

    Each dimension i of the L-dimensional code vector is independently
    quantized to ``levels[i]`` uniformly-spaced bins via tanh-bounded
    rounding with a straight-through gradient estimator.

    Parameters
    ----------
    levels : list[int]
        Number of quantization levels per dimension.  The total codebook
        size is ``prod(levels)``.  Typical choice: ``[8, 5, 5, 5, 5]``
        gives 5 000 codes; ``[4, 3, 3]`` gives 36.
    """

    def __init__(self, levels: list[int]) -> None:
        super().__init__()
        half_levels = torch.tensor([l // 2 for l in levels], dtype=torch.float32)

        # Mixed-radix strides: stride[i] = product of levels[i+1:]
        strides = torch.ones(len(levels), dtype=torch.float32)
        for i in range(len(levels) - 2, -1, -1):
            strides[i] = strides[i + 1] * levels[i + 1]

        self.register_buffer("half_levels", half_levels)
        self.register_buffer("strides", strides)
        self.n_codes: int = 1
        for l in levels:
            self.n_codes *= l

    def forward(self, z: Tensor) -> tuple[Tensor, LongTensor]:
        """Quantize ``z`` using tanh-bounded rounding.

        Parameters
        ----------
        z : Tensor (..., L)

        Returns
        -------
        z_q : Tensor (..., L)
            Quantized code vector; gradient flows through straight-through.
        idx : LongTensor (...,)
            Scalar mixed-radix codebook index for each token.
        """
        # Bound to [-half_l, half_l] per dimension
        z_bounded = torch.tanh(z) * self.half_levels
        z_rounded = z_bounded.round()
        # Straight-through estimator: forward uses rounded, backward uses bounded
        z_q = z_bounded + (z_rounded - z_bounded).detach()
        # Shift to [0, level_i - 1] and compute scalar index
        shifted = z_rounded + self.half_levels
        idx = (shifted * self.strides).sum(-1).long()
        return z_q, idx


# ---------------------------------------------------------------------------
# ViT Decoder (symmetric to _ViTEncoder)
# ---------------------------------------------------------------------------

class _ViTDecoder(nn.Module):
    """ViT decoder with separate 2D factored positional embeddings.

    Accepts a token sequence (B, N, d_model), adds decoder-specific
    positional embeddings, and runs a transformer.  The patch unembed
    projection lives in the parent autoencoder.

    Parameters
    ----------
    d_model, n_heads, n_layers, dropout : —
    max_freq_patches, max_time_patches : int
        Capacity of the decoder frequency- and time-axis embedding tables.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_freq_patches: int,
        max_time_patches: int,
    ) -> None:
        super().__init__()
        # Separate decoder positional embeddings (not shared with encoder)
        self.dec_freq_embed = nn.Parameter(torch.zeros(1, max_freq_patches, d_model // 2))
        self.dec_time_embed = nn.Parameter(torch.zeros(1, max_time_patches, d_model // 2))
        nn.init.trunc_normal_(self.dec_freq_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_time_embed, std=0.02)

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

    def build_pos_embed(self, n_h: int, n_w: int) -> Tensor:
        """Return (1, n_h*n_w, d_model) decoder positional embedding."""
        return _build_2d_pos_embed(self.dec_freq_embed, self.dec_time_embed, n_h, n_w)

    def forward(self, tokens: Tensor, n_h: int, n_w: int) -> Tensor:
        """(B, N, d_model) → (B, N, d_model) after adding pos embed + transformer."""
        tokens = tokens + self.build_pos_embed(n_h, n_w)
        return self.transformer(tokens)


# ---------------------------------------------------------------------------
# Full FSQ-VAE autoencoder
# ---------------------------------------------------------------------------

class SpectrogramFSQVAEAutoEncoder(ModalityAutoEncoder):
    """ViT + FSQ autoencoder for multichannel spectrogram signals.

    Unlike the MAE variant, this model trains with a reconstruction loss over
    ALL patches (no masking) and uses FSQ as the information bottleneck.  The
    decoder is used at inference time, making it suitable for the FAITH
    simulation pipeline (encode → forward predictor → decode).

    Architecture
    ------------
    FlatPatchEmbed2d → Transformer Encoder → Linear(d_model→L) →
    FSQ([8,5,5,5,5]) → Linear(L→d_model) → Transformer Decoder →
    FlatPatchUnembed2d

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Transformer hidden dimension.  Must be even.
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    patch_h, patch_w : int
        Patch size in pixels (default 16 × 16).
    n_enc_layers, n_dec_layers : int
        Transformer depth for encoder and decoder (default 4 each).
    n_heads : int
        Attention heads; must divide d_model (default 4).
    dropout : float
        Dropout rate (default 0.1).
    fsq_levels : list[int] | None
        FSQ quantization levels per dimension (default [8,5,5,5,5]).
    per_channel_patch : bool
        Use per-channel patch embed/unembed (C independent Linear heads,
        summed on embed, independent on unembed) instead of the flat
        Linear(C*ph*pw ↔ d_model) projection.  Avoids rank-d_model
        bottleneck for signals with many decorrelated channels (e.g. ECE).
        Default False.
    max_freq_patches, max_time_patches : int
        Positional embedding table capacity (default 64, 512).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        patch_h: int = 16,
        patch_w: int = 16,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        fsq_levels: list[int] | None = None,
        per_channel_patch: bool = False,
        max_freq_patches: int = 64,
        max_time_patches: int = 512,
    ) -> None:
        assert d_model % 2 == 0, (
            f"d_model must be even for 2D concat positional embeddings, got {d_model}"
        )
        super().__init__(n_channels, d_model, n_tokens)
        self.n_channels = n_channels
        self.patch_h = patch_h
        self.patch_w = patch_w

        if fsq_levels is None:
            fsq_levels = [8, 5, 5, 5, 5]
        fsq_dim = len(fsq_levels)

        # Encoder (also exposed as self.encoder for finiteness tests)
        self.encoder = _ViTEncoder(
            n_channels=n_channels,
            d_model=d_model,
            patch_h=patch_h,
            patch_w=patch_w,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
            max_freq_patches=max_freq_patches,
            max_time_patches=max_time_patches,
            per_channel_patch=per_channel_patch,
        )

        # FSQ bottleneck
        self.pre_fsq = nn.Linear(d_model, fsq_dim)
        nn.init.normal_(self.pre_fsq.weight, std=0.02)  # prevent tanh saturation
        nn.init.zeros_(self.pre_fsq.bias)
        self.fsq = FSQ(fsq_levels)
        self.post_fsq = nn.Linear(fsq_dim, d_model)

        # Decoder
        self.decoder = _ViTDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_dec_layers,
            dropout=dropout,
            max_freq_patches=max_freq_patches,
            max_time_patches=max_time_patches,
        )

        # Decode head (per-channel or flat, symmetric to encoder embed)
        if per_channel_patch:
            self.patch_unembed = PerChannelPatchUnembed2d(
                n_channels, d_model, patch_h, patch_w
            )
        else:
            self.patch_unembed = FlatPatchUnembed2d(
                n_channels, d_model, patch_h, patch_w
            )

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
        B, C, F_orig, T_orig = x.shape
        ph, pw = self.patch_h, self.patch_w

        # ── 1. Pad to patch-aligned dimensions ────────────────────────────
        pad_f = (ph - F_orig % ph) % ph
        pad_t = (pw - T_orig % pw) % pw
        if pad_f > 0 or pad_t > 0:
            x = nn.functional.pad(x, (0, pad_t, 0, pad_f))

        # ── 2. Patch embedding + encoder positional embeddings ─────────────
        tokens, (n_h, n_w) = self.encoder.patch_embed(x)  # (B, N, d_model)
        tokens = tokens + self.encoder.build_pos_embed(n_h, n_w)

        # ── 3. Encoder transformer ─────────────────────────────────────────
        tokens_enc = self.encoder.transformer(tokens)  # (B, N, d_model)

        # ── 4. FSQ bottleneck ──────────────────────────────────────────────
        z = self.pre_fsq(tokens_enc)          # (B, N, L)
        z_q, indices = self.fsq(z)            # (B, N, L), (B, N)
        tokens_dec = self.post_fsq(z_q)       # (B, N, d_model)

        # ── 5. Decoder transformer + positional embeddings ─────────────────
        tokens_out = self.decoder(tokens_dec, n_h, n_w)  # (B, N, d_model)

        # ── 6. Unembed + crop to original spatial dims ─────────────────────
        reconstructed = self.patch_unembed(tokens_out, n_h, n_w)
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed
        return reconstructed, indices
