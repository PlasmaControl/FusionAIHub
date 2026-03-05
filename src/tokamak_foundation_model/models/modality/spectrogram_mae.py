"""ViT-based Masked Autoencoder for tokamak spectrogram diagnostics.

Implements He et al., 2022 (MAE) with a fully attention-based architecture:
the encoder processes only the visible (unmasked) patch tokens, so masked
patches have zero influence on the latent representation (no CNN receptive-
field leakage). The decoder restores the full token sequence by inserting a
shared learnable mask_token at masked positions before the lighter decoder
transformer.

Patch embedding / unembedding
------------------------------
Each channel has its own independent linear projection for embedding
(PerChannelPatchEmbed2d) and its own independent linear head for decoding
(PerChannelPatchUnembed2d).  The per-channel embeddings are *summed* into a
single d_model token so the sequence length stays at N = n_h × n_w regardless
of the number of channels.  This lets the model specialise to channel-specific
spectral statistics while keeping the transformer architecture unchanged.

Positional encoding
-------------------
Uses 2D factored learned embeddings: separate frequency-axis and time-axis
embedding tables, each of dimension d_model//2, concatenated to form the full
d_model position vector for each patch.  This gives the model an explicit
signal for both axes (critical for spectrograms where the frequency axis is
typically far shorter than the time axis) and avoids the entanglement that
summed 1D embeddings produce.  d_model must therefore be even.

Return contract
---------------
Training  : (reconstructed, mask_pixel) — mask_pixel is BoolTensor (B,1,F,T)
Eval      : reconstructed               — shape (B, C, F, T) matching input
"""

import torch
import torch.nn as nn

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder


# ---------------------------------------------------------------------------
# Per-channel patch embed / unembed
# ---------------------------------------------------------------------------

class PerChannelPatchEmbed2d(nn.Module):
    """Patch embedding with independent linear projections per channel.

    Each channel is projected separately from ``patch_h × patch_w`` to
    ``d_model`` and the results are *summed*, yielding one token per patch.
    This gives every channel its own kernel while keeping sequence length
    at N = n_h × n_w.

    Returns ``(tokens, (n_h, n_w))`` — same interface as ``PatchEmbed2d``.
    """

    def __init__(
        self, n_channels: int, d_model: int, patch_h: int, patch_w: int
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels
        self.projs = nn.ModuleList(
            [nn.Linear(patch_h * patch_w, d_model) for _ in range(n_channels)]
        )

    def forward(self, x: torch.Tensor):
        B, C, Fr, T = x.shape
        ph, pw = self.patch_h, self.patch_w
        n_h, n_w = Fr // ph, T // pw
        # (B, C, n_h, ph, n_w, pw) → (B, N, C, ph*pw)
        patches = (
            x.reshape(B, C, n_h, ph, n_w, pw)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B, n_h * n_w, C, ph * pw)
        )
        # Sum per-channel projections → (B, N, d_model)
        tokens = sum(self.projs[c](patches[:, :, c]) for c in range(C))
        return tokens, (n_h, n_w)


class PerChannelPatchUnembed2d(nn.Module):
    """Patch unembedding with independent linear decode heads per channel.

    Applies a separate ``d_model → patch_h × patch_w`` projection for each
    channel, then assembles the results into a ``(B, C, Fr, T)`` tensor.
    """

    def __init__(
        self, n_channels: int, d_model: int, patch_h: int, patch_w: int
    ) -> None:
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.n_channels = n_channels
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, patch_h * patch_w) for _ in range(n_channels)]
        )

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        B = x.shape[0]
        ph, pw = self.patch_h, self.patch_w
        # (B, N, C, ph*pw) → (B, C, n_h*ph, n_w*pw)
        per_ch = torch.stack([h(x) for h in self.heads], dim=2)
        return (
            per_ch
            .reshape(B, n_h, n_w, self.n_channels, ph, pw)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, self.n_channels, n_h * ph, n_w * pw)
        )


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------

def _build_2d_pos_embed(
    freq_embed: torch.Tensor,   # (1, max_F, d_model//2)
    time_embed: torch.Tensor,   # (1, max_T, d_model//2)
    n_h: int,
    n_w: int,
) -> torch.Tensor:
    """Concatenate factored freq/time embeddings into a full 2D pos embedding.

    Returns
    -------
    Tensor of shape (1, n_h * n_w, d_model) in raster (row-major) order
    matching the patch layout produced by PerChannelPatchEmbed2d.
    """
    freq = freq_embed[:, :n_h].unsqueeze(2).expand(-1, -1, n_w, -1)
    time = time_embed[:, :n_w].unsqueeze(1).expand(-1, n_h, -1, -1)
    return torch.cat([freq, time], dim=-1).reshape(1, n_h * n_w, -1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _ViTEncoder(nn.Module):
    """ViT encoder with per-channel patch embedding and 2D factored pos embeds.

    Processes all patch tokens without masking; used as
    ``SpectrogramMAEAutoEncoder.encoder`` so that ``model.encoder(x)``
    returns a finite token sequence for shape / finiteness tests.

    Parameters
    ----------
    n_channels, d_model, patch_h, patch_w, n_heads, n_layers, dropout : —
    max_freq_patches : int
        Capacity of the frequency-axis embedding table.
    max_time_patches : int
        Capacity of the time-axis embedding table.
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
        max_freq_patches: int,
        max_time_patches: int,
    ) -> None:
        super().__init__()
        self.patch_embed = PerChannelPatchEmbed2d(n_channels, d_model, patch_h, patch_w)

        # Factored 2D positional embeddings (concatenated along d_model)
        self.freq_embed = nn.Parameter(torch.zeros(1, max_freq_patches, d_model // 2))
        self.time_embed = nn.Parameter(torch.zeros(1, max_time_patches, d_model // 2))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

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

    def build_pos_embed(self, n_h: int, n_w: int) -> torch.Tensor:
        """Return (1, n_h*n_w, d_model) positional embedding."""
        return _build_2d_pos_embed(self.freq_embed, self.time_embed, n_h, n_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) → (B, N, d_model), all tokens, no masking."""
        tokens, (n_h, n_w) = self.patch_embed(x)
        tokens = tokens + self.build_pos_embed(n_h, n_w)
        return self.transformer(tokens)


# ---------------------------------------------------------------------------
# Full MAE autoencoder
# ---------------------------------------------------------------------------

class SpectrogramMAEAutoEncoder(ModalityAutoEncoder):
    """ViT-based Masked Autoencoder for multichannel spectrogram signals.

    The encoder operates on *visible* tokens only — there is no spatial
    neighbourhood so masked patches cannot leak positional information through
    a CNN receptive field.  A single learnable ``mask_token`` vector fills
    masked positions before the lightweight decoder transformer.

    Each channel has independent patch embedding kernels and an independent
    decode head, allowing the model to specialise to per-channel spectral
    statistics.

    Positional embeddings are 2D factored (freq ⊕ time, concatenated) so
    the model receives an unambiguous signal for each spatial axis.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels (e.g. 4 for CO2, 48 for ECE).
    d_model : int
        Transformer hidden dimension.  Must be even (split equally between
        freq and time embedding halves).
    n_tokens : int
        Unused; kept for interface compatibility with ModalityAutoEncoder.
    mask_ratio : float
        Fraction of patches to mask during training (default 0.5).
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
    max_freq_patches : int
        Capacity of the frequency positional embedding table (default 64).
    max_time_patches : int
        Capacity of the time positional embedding table (default 512).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 0,
        *,
        mask_ratio: float = 0.5,
        patch_h: int = 16,
        patch_w: int = 16,
        n_enc_layers: int = 4,
        n_dec_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_freq_patches: int = 64,
        max_time_patches: int = 512,
    ) -> None:
        assert d_model % 2 == 0, (
            f"d_model must be even for 2D concat positional embeddings, got {d_model}"
        )
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
            max_freq_patches=max_freq_patches,
            max_time_patches=max_time_patches,
        )

        # Decoder 2D factored positional embeddings (separate from encoder's)
        self.dec_freq_embed = nn.Parameter(torch.zeros(1, max_freq_patches, d_model // 2))
        self.dec_time_embed = nn.Parameter(torch.zeros(1, max_time_patches, d_model // 2))
        nn.init.trunc_normal_(self.dec_freq_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_time_embed, std=0.02)

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

        # Per-channel decode heads
        self.patch_unembed = PerChannelPatchUnembed2d(n_channels, d_model, patch_h, patch_w)

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

        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :N_vis]

        tokens_visible = tokens.gather(
            1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

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

        # ── 2. Per-channel patch embedding ────────────────────────────────
        tokens, (n_h, n_w) = self.encoder.patch_embed(x)  # (B, N, d_model)

        # ── 3. 2D encoder positional embeddings ───────────────────────────
        tokens = tokens + self.encoder.build_pos_embed(n_h, n_w)

        # ── 4. Mask & encode ──────────────────────────────────────────────
        if self.training:
            tokens_vis, ids_keep, _, mask_bool = self._mask_tokens(tokens)
            tokens_enc = self.encoder.transformer(tokens_vis)  # (B, N_vis, D)
        else:
            tokens_enc = self.encoder.transformer(tokens)      # (B, N, D)

        # ── 5. Restore full sequence (training only) ──────────────────────
        if self.training:
            D = tokens_enc.shape[-1]
            x_full = self.mask_token.expand(B, tokens.shape[1], D).clone()
            x_full.scatter_(
                1,
                ids_keep.unsqueeze(-1).expand(-1, -1, D),
                tokens_enc,
            )
        else:
            x_full = tokens_enc

        # ── 6. 2D decoder positional embeddings ───────────────────────────
        x_full = x_full + _build_2d_pos_embed(
            self.dec_freq_embed, self.dec_time_embed, n_h, n_w
        )

        # ── 7. Decode ─────────────────────────────────────────────────────
        x_dec = self.decoder_layers(x_full)

        # ── 8. Per-channel pixel reconstruction ───────────────────────────
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
