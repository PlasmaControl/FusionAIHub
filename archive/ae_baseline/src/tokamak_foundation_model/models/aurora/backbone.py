"""
Latent backbone for Aurora-inspired tokamak foundation model.

Replaces the lightweight recurrent dynamics (MLP + 1 self-attention layer)
with a deep Transformer stack that processes the full latent state at
every rollout step.  Analogous to Aurora's 3D Swin U-Net backbone, but
using global self-attention (our latent tokens have no spatial structure).

Each :class:`BackboneBlock` consists of:
    1. Pre-norm self-attention (inter-token interaction)
    2. Pre-norm cross-attention to actuator tokens (control conditioning)
    3. Pre-norm FFN

The :class:`LatentBackbone` stacks N blocks with optional U-Net skip
connections and adds Fourier step conditioning so the model can
distinguish rollout step 0 from step 7.
"""

import torch
import torch.nn as nn

from tokamak_foundation_model.models.latent_feature_space.modality_tokenizer import (
    sinusoidal_time_encoding,
)


class BackboneBlock(nn.Module):
    """Single pre-norm Transformer block with self-attn + cross-attn + FFN.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        FFN hidden dim = ``d_model * mlp_ratio``.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Self-attention: latent tokens interact
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # Cross-attention: latent tokens attend to actuator tokens.
        # Only normalize queries, not KV — actuator tokens are already
        # LayerNormed by ActuatorTokenizer, and per-token LN on context
        # kills uniform-value tokens.
        self.norm_xa_q = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

        # Feed-forward
        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, latent: torch.Tensor, actuator_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape ``[B, N_L, D]``.
        actuator_tokens : torch.Tensor
            Shape ``[B, N_act, D]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_L, D]``.
        """
        # Self-attention (pre-norm)
        x = self.norm_sa(latent)
        latent = latent + self.self_attn(x, x, x)[0]

        # Cross-attention to actuators (pre-norm on queries only)
        q = self.norm_xa_q(latent)
        latent = latent + self.cross_attn(q, actuator_tokens, actuator_tokens)[0]

        # FFN (pre-norm)
        latent = latent + self.ffn(self.norm_ffn(latent))

        return latent


class LatentBackbone(nn.Module):
    """Deep Transformer backbone operating on the Perceiver latent array.

    Conditioned on actuator tokens (via cross-attention in each block)
    and rollout step index (via Fourier embedding added to all tokens).

    Optional U-Net skip connections: the first ``n_blocks // 2`` blocks
    save their output, and the corresponding later blocks add it back.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_blocks : int
        Number of :class:`BackboneBlock` layers.
    n_heads : int
        Number of attention heads per block.
    mlp_ratio : float
        FFN hidden dim = ``d_model * mlp_ratio``.
    dropout : float
        Dropout rate.
    use_skips : bool
        If ``True``, add U-Net style skip connections between the first
        and second halves of the backbone.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_blocks: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_skips: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.use_skips = use_skips

        # Fourier step embedding + MLP
        self.step_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Backbone blocks
        self.blocks = nn.ModuleList([
            BackboneBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_blocks)
        ])

        # Final LayerNorm (standard for pre-norm architectures)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        latent: torch.Tensor,
        actuator_tokens: torch.Tensor,
        step_index: int,
        offset_ms: float = 0.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape ``[B, N_L, D]`` — encoded plasma state.
        actuator_tokens : torch.Tensor
            Shape ``[B, N_act, D]`` — tokenized actuator signals.
        step_index : int
            Rollout step (0, 1, 2, ...).  Fourier-encoded and added to
            all latent tokens so the backbone can distinguish steps.
        offset_ms : float
            Absolute time in ms (alternative to integer step_index for
            continuous time encoding).  Uses ``offset_ms`` if > 0,
            otherwise falls back to ``step_index``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_L, D]`` — predicted next latent state.
        """
        B = latent.shape[0]
        device = latent.device

        # Step conditioning: Fourier encode + MLP, add to all tokens
        t_val = offset_ms if offset_ms > 0 else float(step_index)
        t_ms = torch.tensor(
            [[t_val]], device=device, dtype=torch.float32,
        ).expand(B, 1)
        step_enc = sinusoidal_time_encoding(t_ms, self.d_model)  # [B,1,D]
        step_embed = self.step_mlp(step_enc.squeeze(1))  # [B, D]
        latent = latent + step_embed.unsqueeze(1)  # broadcast to all tokens

        # Forward through backbone blocks with optional skips
        half = self.n_blocks // 2
        skips = []

        for i, block in enumerate(self.blocks):
            if self.use_skips and i < half:
                skips.append(latent)

            latent = block(latent, actuator_tokens)

            if self.use_skips and i >= half and skips:
                latent = latent + skips.pop()

        return self.final_norm(latent)
