"""CNN + Perceiver bottleneck autoencoder for tokamak spectrogram diagnostics.

Combines a CNN backbone for local feature extraction with a Perceiver-style
cross-attention bottleneck that compresses the spatial feature map into a small
set of learned latent tokens.  Optionally quantizes those tokens with FSQ to
create a discrete bottleneck.

Architecture
------------
Encoder:
  _CNNEncoder(C → dims)         → (B, dims[-1], H', W')     local features
  flatten + Linear(dims[-1], d_model) → (B, H'*W', d_model)  project to token dim
  + 2D factored pos embed       → (B, H'*W', d_model)        spatial-aware tokens
  cross-attn: N queries attend to H'*W' spatial tokens        global compression
  self-attn layers on N tokens  → (B, N, d_model)             refined latent
  [optional] FSQ                → (B, N, d_model)             discrete bottleneck

Decoder:
  [optional] post-FSQ linear    → (B, N, d_model)
  H'*W' spatial queries + 2D pos embed cross-attend to N latent tokens
  self-attn layers on H'*W' tokens → (B, H'*W', d_model)     spatial reconstruction
  Linear(d_model, dims[-1]) + reshape → (B, dims[-1], H', W')
  _CNNDecoder(dims → C)         → (B, C, F, T)               pixel reconstruction

Return contract
---------------
Training, no FSQ  : (reconstructed, latent)  — latent for monitoring compression
Training, with FSQ: (reconstructed, indices)  — indices for codebook utilisation
Eval              : reconstructed             — shape (B, C, F, T) matching input
"""

import torch
import torch.nn as nn
from torch import Tensor, LongTensor

from tokamak_foundation_model.models.modality.base import ModalityAutoEncoder
from tokamak_foundation_model.models.modality.spectrogram_cnn import (
    _CNNEncoder,
    _CNNDecoder,
)
from tokamak_foundation_model.models.modality.spectrogram_fsq_vae import FSQ
from tokamak_foundation_model.models.modality.spectrogram_mae import _build_2d_pos_embed


# ---------------------------------------------------------------------------
# Cross-attention block
# ---------------------------------------------------------------------------

class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        normed_q = self.norm_q(q)
        normed_kv = self.norm_kv(kv)
        q = q + self.cross_attn(normed_q, normed_kv, normed_kv)[0]
        q = q + self.ffn(self.norm_ffn(q))
        return q


# ---------------------------------------------------------------------------
# Standalone encoder (for test_encoder_output_is_finite)
# ---------------------------------------------------------------------------

class _CNNPerceiverEncoder(nn.Module):
    """CNN encoder + Perceiver cross-attention compression.

    Wraps the full encode path so that ``model.encoder(x)`` returns
    ``(B, n_tokens, d_model)`` latent tokens for shape / finiteness tests.
    """

    def __init__(
        self,
        cnn_encoder: _CNNEncoder,
        project_in: nn.Linear,
        freq_embed: nn.Parameter,
        time_embed: nn.Parameter,
        latent_tokens: nn.Parameter,
        enc_cross_attn: _CrossAttentionBlock,
        enc_self_attn: nn.TransformerEncoder,
    ) -> None:
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.project_in = project_in
        self.freq_embed = freq_embed
        self.time_embed = time_embed
        self.latent_tokens = latent_tokens
        self.enc_cross_attn = enc_cross_attn
        self.enc_self_attn = enc_self_attn

    def forward(self, x: Tensor) -> Tensor:
        z = self.cnn_encoder(x)  # (B, D, H', W')
        B, _D, H, W = z.shape
        tokens = self.project_in(z.flatten(2).transpose(1, 2))  # (B, H'*W', d_model)
        pos = _build_2d_pos_embed(self.freq_embed, self.time_embed, H, W)
        tokens = tokens + pos
        queries = self.latent_tokens.expand(B, -1, -1)  # (B, N, d_model)
        latent = self.enc_cross_attn(queries, tokens)  # (B, N, d_model)
        latent = self.enc_self_attn(latent)  # (B, N, d_model)
        return latent


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class SpectrogramCNNPerceiverAutoEncoder(ModalityAutoEncoder):
    """CNN + Perceiver bottleneck autoencoder for spectrogram signals.

    Parameters
    ----------
    n_channels : int
        Number of spectrogram channels.
    d_model : int
        Token / latent dimension.  Must be even (for 2D factored pos embeds).
    n_tokens : int
        Number of learned latent query tokens (compression factor).
    dims : list[int] | None
        CNN stage dims (default [64, 128]).  Each stage halves spatial resolution.
    n_heads : int
        Attention heads for cross- and self-attention.
    n_self_layers : int
        Self-attention layers on latent tokens (encoder side).
    n_dec_self_layers : int
        Self-attention layers on spatial queries (decoder side).
    dropout : float
        Dropout rate.
    fsq_levels : list[int] | None
        FSQ quantization levels.  None = continuous bottleneck.
    max_freq_patches : int
        Capacity of frequency-axis positional embedding table.
    max_time_patches : int
        Capacity of time-axis positional embedding table.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 256,
        n_tokens: int = 16,
        *,
        dims: list[int] | None = None,
        n_heads: int = 4,
        n_self_layers: int = 2,
        n_dec_self_layers: int = 2,
        dropout: float = 0.1,
        fsq_levels: list[int] | None = None,
        max_freq_patches: int = 64,
        max_time_patches: int = 512,
    ) -> None:
        assert d_model % 2 == 0, (
            f"d_model must be even for 2D concat positional embeddings, got {d_model}"
        )
        super().__init__(n_channels, d_model, n_tokens)
        if dims is None:
            dims = [64, 128]
        self.dims = dims

        # --- CNN backbone ---
        self.cnn_encoder = _CNNEncoder(n_channels, dims)
        self.cnn_decoder = _CNNDecoder(n_channels, dims)

        # --- Encoder: spatial tokens → latent tokens ---
        self.project_in = nn.Linear(dims[-1], d_model)

        # 2D factored positional embeddings (encoder)
        self.freq_embed = nn.Parameter(torch.zeros(1, max_freq_patches, d_model // 2))
        self.time_embed = nn.Parameter(torch.zeros(1, max_time_patches, d_model // 2))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        # Learned latent query tokens
        self.latent_tokens = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)

        # Cross-attention: N queries attend to H'*W' spatial tokens
        self.enc_cross_attn = _CrossAttentionBlock(d_model, n_heads, dropout)

        # Self-attention on N latent tokens
        enc_self_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc_self_attn = nn.TransformerEncoder(
            enc_self_layer,
            num_layers=n_self_layers,
            norm=nn.LayerNorm(d_model),
        )

        # --- Optional FSQ ---
        if fsq_levels is not None:
            fsq_dim = len(fsq_levels)
            self.pre_fsq = nn.Linear(d_model, fsq_dim)
            nn.init.normal_(self.pre_fsq.weight, std=0.02)
            nn.init.zeros_(self.pre_fsq.bias)
            self.fsq = FSQ(fsq_levels)
            self.post_fsq = nn.Linear(fsq_dim, d_model)
        else:
            self.pre_fsq = None
            self.fsq = None
            self.post_fsq = None

        # --- Decoder: latent tokens → spatial tokens ---
        # 2D factored positional embeddings (decoder, separate from encoder)
        self.dec_freq_embed = nn.Parameter(torch.zeros(1, max_freq_patches, d_model // 2))
        self.dec_time_embed = nn.Parameter(torch.zeros(1, max_time_patches, d_model // 2))
        nn.init.trunc_normal_(self.dec_freq_embed, std=0.02)
        nn.init.trunc_normal_(self.dec_time_embed, std=0.02)

        # Cross-attention: H'*W' spatial queries attend to N latent tokens
        self.dec_cross_attn = _CrossAttentionBlock(d_model, n_heads, dropout)

        # Self-attention on H'*W' decoded spatial tokens
        dec_self_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dec_self_attn = nn.TransformerEncoder(
            dec_self_layer,
            num_layers=n_dec_self_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.project_out = nn.Linear(d_model, dims[-1])

        # --- Expose standalone encoder for tests ---
        self.encoder = _CNNPerceiverEncoder(
            cnn_encoder=self.cnn_encoder,
            project_in=self.project_in,
            freq_embed=self.freq_embed,
            time_embed=self.time_embed,
            latent_tokens=self.latent_tokens,
            enc_cross_attn=self.enc_cross_attn,
            enc_self_attn=self.enc_self_attn,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, LongTensor]:
        B, C, F_orig, T_orig = x.shape

        # ── 1. CNN encode ─────────────────────────────────────────────
        z_cnn = self.cnn_encoder(x)  # (B, dims[-1], H', W')
        _, _, H, W = z_cnn.shape

        # ── 2. Flatten + project to token dim ─────────────────────────
        tokens = self.project_in(z_cnn.flatten(2).transpose(1, 2))  # (B, H'*W', d_model)
        tokens = tokens + _build_2d_pos_embed(self.freq_embed, self.time_embed, H, W)

        # ── 3. Cross-attention compression: H'*W' → N ─────────────────
        queries = self.latent_tokens.expand(B, -1, -1)
        latent = self.enc_cross_attn(queries, tokens)  # (B, N, d_model)
        latent = self.enc_self_attn(latent)  # (B, N, d_model)

        # ── 4. Optional FSQ ───────────────────────────────────────────
        if self.fsq is not None:
            z_fsq = self.pre_fsq(latent)  # (B, N, L)
            z_q, indices = self.fsq(z_fsq)  # (B, N, L), (B, N)
            latent = self.post_fsq(z_q)  # (B, N, d_model)

        # ── 5. Decode: spatial queries cross-attend to latent tokens ──
        spatial_queries = torch.zeros(B, H * W, self.d_model, device=x.device)
        spatial_queries = spatial_queries + _build_2d_pos_embed(
            self.dec_freq_embed, self.dec_time_embed, H, W,
        )
        decoded = self.dec_cross_attn(spatial_queries, latent)  # (B, H'*W', d_model)
        decoded = self.dec_self_attn(decoded)  # (B, H'*W', d_model)

        # ── 6. Reshape back to spatial + CNN decode ───────────────────
        z_dec = self.project_out(decoded).transpose(1, 2).reshape(B, -1, H, W)
        reconstructed = self.cnn_decoder(z_dec)
        reconstructed = reconstructed[:, :, :F_orig, :T_orig]

        if not self.training:
            return reconstructed
        if self.fsq is not None:
            return reconstructed, indices
        return reconstructed, latent
