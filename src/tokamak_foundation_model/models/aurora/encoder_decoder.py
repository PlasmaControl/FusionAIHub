"""
Pre-norm Perceiver encoder and decoder for the Aurora-inspired model.

All attention blocks use pre-norm (normalize inputs, not outputs) for
stable processing.  The encoder compresses variable-length diagnostic
+ actuator tokens into a fixed-size latent array.  The decoder expands
the latent back to per-modality AE token sequences.
"""

from typing import Optional

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────


class PreNormCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention with query residual + FFN.

    Used in the Perceiver encoder and decoder where the query residual
    is desired (queries = latent queries or output queries that should
    be refined, not replaced).

    Only the queries are LayerNormed before attention, NOT the context.
    The context comes from heterogeneous input tokens whose scale
    carries information — normalizing it per-token kills uniform-value
    tokens (LayerNorm maps constant vectors to zero).
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, queries: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        queries : torch.Tensor
            Shape ``[B, N_q, D]``.
        context : torch.Tensor
            Shape ``[B, N_c, D]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_q, D]``.
        """
        q = self.norm_q(queries)
        queries = queries + self.cross_attn(q, context, context)[0]
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class PreNormSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, N, D]``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N, D]``.
        """
        h = self.norm_sa(x)
        x = x + self.self_attn(h, h, h)[0]
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────
# Perceiver Encoder
# ─────────────────────────────────────────────────────────────────────


class PerceiverEncoder(nn.Module):
    """Compress variable-length token sequence into fixed-size latent array.

    Learned latent queries cross-attend to the concatenated diagnostic
    + actuator tokens, then self-attend for refinement.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_latent_queries : int
        Number of latent queries (compressed state size).
    n_cross_layers : int
        Number of cross-attention layers.
    n_self_layers : int
        Number of self-attention processing layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_latent_queries: int = 128,
        n_cross_layers: int = 2,
        n_self_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_queries = nn.Parameter(
            torch.randn(n_latent_queries, d_model) * 0.02,
        )
        self.cross_blocks = nn.ModuleList([
            PreNormCrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])
        self.self_blocks = nn.ModuleList([
            PreNormSelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_tokens : torch.Tensor
            Concatenated diagnostic + actuator tokens,
            shape ``[B, N_input, d_model]``.

        Returns
        -------
        torch.Tensor
            Latent array, shape ``[B, N_latent, d_model]``.
        """
        B = input_tokens.shape[0]
        latent = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        for block in self.cross_blocks:
            latent = block(queries=latent, context=input_tokens)

        for block in self.self_blocks:
            latent = block(latent)

        return self.final_norm(latent)


# ─────────────────────────────────────────────────────────────────────
# Perceiver Decoder
# ─────────────────────────────────────────────────────────────────────


class PerceiverDecoder(nn.Module):
    """Decode latent array to per-modality AE token sequences.

    Each modality has its own set of learned output queries.  Each
    decoder layer consists of cross-attention to the latent followed
    by self-attention among the output queries.

    Parameters
    ----------
    d_model : int
        Model dimension.
    output_queries_config : dict
        ``{modality_name: n_tokens}``.
    n_layers : int
        Number of interleaved (cross-attn + self-attn) layers.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        output_queries_config: Optional[dict] = None,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if output_queries_config is None:
            output_queries_config = {}

        self.d_model = d_model
        self.n_layers = n_layers

        self.output_queries = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(n_tok, d_model) * 0.02)
            for mod, n_tok in output_queries_config.items()
        })
        self.cross_blocks = nn.ModuleDict({
            mod: nn.ModuleList([
                PreNormCrossAttentionBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])
            for mod in output_queries_config
        })
        self.self_blocks = nn.ModuleDict({
            mod: nn.ModuleList([
                PreNormSelfAttentionBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])
            for mod in output_queries_config
        })
        self.final_norms = nn.ModuleDict({
            mod: nn.LayerNorm(d_model)
            for mod in output_queries_config
        })

    def _decode_modality(
        self, mod: str, latent: torch.Tensor,
    ) -> torch.Tensor:
        B = latent.shape[0]
        tokens = self.output_queries[mod].unsqueeze(0).expand(B, -1, -1)
        for cross_blk, self_blk in zip(
            self.cross_blocks[mod], self.self_blocks[mod],
        ):
            tokens = cross_blk(queries=tokens, context=latent)
            tokens = self_blk(tokens)
        return self.final_norms[mod](tokens)

    def forward(
        self,
        latent: torch.Tensor,
        modality: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape ``[B, N_latent, d_model]``.
        modality : str or None
            Decode this modality only, or all if ``None``.

        Returns
        -------
        dict or torch.Tensor
            ``{mod: [B, N_m, d_model]}`` if *modality* is ``None``,
            otherwise ``[B, N_m, d_model]``.
        """
        if modality is not None:
            return self._decode_modality(modality, latent)
        return {
            mod: self._decode_modality(mod, latent)
            for mod in self.output_queries
        }
