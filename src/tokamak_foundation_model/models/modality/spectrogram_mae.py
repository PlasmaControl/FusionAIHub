"""ViT-based Masked Autoencoder for tokamak spectrogram diagnostics.

Implements He et al., 2022 (MAE) with a fully attention-based architecture:
the encoder processes only the visible (unmasked) patch tokens, so masked
patches have zero influence on the latent representation (no CNN receptive-
field leakage). The decoder restores the full token sequence by inserting a
shared learnable mask_token at masked positions before the lighter decoder
transformer.

Return contract
---------------
Training  : (reconstructed, mask_pixel) — mask_pixel is BoolTensor (B,1,F,T)
Eval      : reconstructed               — shape (B, C, F, T) matching input
"""

import torch
import torch.nn as nn

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_baseline import (
    PatchEmbed2d,
    PatchUnembed2d,
)


class _ViTEncoder(nn.Module):
    """ViT encoder that processes all patch tokens without masking.

    Used as ``SpectrogramMAEAutoEncoder.encoder`` so that
    ``model.encoder(x)`` returns a finite token sequence for shape tests.

    Parameters
    ----------
    n_channels : int
    d_model : int
    patch_h, patch_w : int
    n_heads : int
    n_layers : int
    dropout : float
    max_patches : int
        Upper bound on number of patches; sets positional embedding size.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        patch_h: int,
        patch_w: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_patches: int,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed2d(n_channels, d_model, patch_h, patch_w)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) → (B, N, d_model), all tokens, no masking."""
        tokens, (n_h, n_w) = self.patch_embed(x)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N]
        return self.transformer(tokens)


class SpectrogramMAEAutoEncoder(ModalityAutoEncoder):
    """ViT-based Masked Autoencoder for multichannel spectrogram signals.

    The encoder operates on *visible* tokens only — there is no spatial
    neighbourhood so masked patches cannot leak positional information through
    a CNN receptive field. A single learnable ``mask_token`` vector fills
    masked positions before the lightweight decoder transformer.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels (e.g. 4 for CO2, 48 for ECE).
    d_model : int
        Transformer hidden dimension.
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    mask_ratio : float
        Fraction of patches to mask during training (default 0.75).
    patch_h, patch_w : int
        Patch height and width in pixels (default 16 × 16).
    n_enc_layers : int
        Transformer layers in the encoder (default 4).
    n_dec_layers : int
        Transformer layers in the decoder (default 2).
    n_heads : int
        Attention heads; must divide d_model (default 4).
    dropout : float
        Dropout rate (default 0.1).
    max_patches : int
        Maximum patches; sets positional embedding capacity (default 1024).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        mask_ratio: float = 0.75,
        patch_h: int = 16,
        patch_w: int = 16,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_patches: int = 1024,
    ) -> None:
        super().__init__(n_channels, d_model, n_tokens)
        self.mask_ratio = mask_ratio
        self.patch_h = patch_h
        self.patch_w = patch_w

        # Encoder — also exposed as self.encoder for test_encoder_output_is_finite
        self.encoder = _ViTEncoder(
            n_channels=n_channels,
            d_model=d_model,
            patch_h=patch_h,
            patch_w=patch_w,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            dropout=dropout,
            max_patches=max_patches,
        )

        # Decoder positional embeddings (separate from encoder's)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Learnable fill vector for masked positions (full d_model embedding)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_layers = nn.TransformerEncoder(
            decoder_layer,
            num_layers=n_dec_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Maps decoded tokens back to pixel patches
        self.patch_unembed = PatchUnembed2d(n_channels, d_model, patch_h, patch_w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_tokens(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop a fraction of patch tokens.

        Parameters
        ----------
        tokens : Tensor (B, N, d_model)
            Position-encoded patch tokens.

        Returns
        -------
        tokens_visible : Tensor (B, N_vis, d_model)
        ids_keep       : LongTensor (B, N_vis) — original indices kept visible
        ids_restore    : LongTensor (B, N)     — inverse permutation of shuffle
        mask_bool      : BoolTensor (B, N)     — True where token is masked
                         in the *original* (unshuffled) token order
        """
        B, N, D = tokens.shape
        N_vis = max(1, int(N * (1 - self.mask_ratio)))

        # Per-sample random shuffle; first N_vis positions are "visible"
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)        # (B, N)
        ids_restore = ids_shuffle.argsort(dim=1)  # (B, N), inverse permutation
        ids_keep = ids_shuffle[:, :N_vis]          # (B, N_vis)

        # Gather visible tokens from original order
        tokens_visible = tokens.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, N_vis, D)

        # Boolean mask in original order: True = masked
        mask_bool = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
        mask_bool.scatter_(1, ids_keep, False)

        return tokens_visible, ids_keep, ids_restore, mask_bool

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        Training : returns (reconstructed, mask_pixel) so the trainer can
                   compute loss only on masked patches.
        Eval     : returns reconstructed — shape (B, C, F, T) matching input.
        """
        B, C, F_orig, T_orig = x.shape
        ph, pw = self.patch_h, self.patch_w

        # ── 1. Pad to patch-aligned dimensions ────────────────────────────
        pad_f = (ph - F_orig % ph) % ph
        pad_t = (pw - T_orig % pw) % pw
        if pad_f > 0 or pad_t > 0:
            x = torch.nn.functional.pad(x, (0, pad_t, 0, pad_f))
        B, C, F, T = x.shape

        # ── 2. Patch embedding ────────────────────────────────────────────
        tokens, (n_h, n_w) = self.encoder.patch_embed(x)  # (B, N, d_model)
        N = tokens.shape[1]

        # ── 3. Encoder positional embeddings (applied to all N positions) ──
        tokens = tokens + self.encoder.pos_embed[:, :N]

        # ── 4. Mask & encode ─────────────────────────────────────────────
        if self.training:
            tokens_vis, ids_keep, _, mask_bool = self._mask_tokens(tokens)
            tokens_enc = self.encoder.transformer(tokens_vis)  # (B, N_vis, D)
        else:
            tokens_enc = self.encoder.transformer(tokens)      # (B, N, D)

        # ── 5. Restore full sequence (training only) ──────────────────────
        if self.training:
            D = tokens_enc.shape[-1]
            # Fill all slots with mask_token, then overwrite visible positions
            x_full = self.mask_token.expand(B, N, D).clone()
            x_full.scatter_(
                1,
                ids_keep.unsqueeze(-1).expand(-1, -1, D),
                tokens_enc,
            )  # (B, N, D)
        else:
            x_full = tokens_enc  # (B, N, D), all visible

        # ── 6. Decoder positional embeddings ──────────────────────────────
        x_full = x_full + self.decoder_pos_embed[:, :N]

        # ── 7. Decode ─────────────────────────────────────────────────────
        x_dec = self.decoder_layers(x_full)

        # ── 8. Pixel reconstruction ───────────────────────────────────────
        reconstructed = self.patch_unembed(x_dec, n_h, n_w)  # (B, C, n_h*ph, n_w*pw)
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed

        # ── 9. Upsample token mask → pixel mask ───────────────────────────
        mask_pixel = (
            mask_bool
            .reshape(B, 1, n_h, n_w)
            .repeat_interleave(ph, dim=2)
            .repeat_interleave(pw, dim=3)
        )[:, :, :F_orig, :T_orig]

        return reconstructed, mask_pixel
