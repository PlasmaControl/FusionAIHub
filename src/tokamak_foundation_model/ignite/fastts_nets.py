"""Phase-A fast-TS (filterscopes) encoder / decoder over the ELM ACTIVITY ENVELOPE.

Mirrors ``nets.py`` (spectrogram) and ``video_nets.py`` (video) but tokenizes the 1-D ELM
envelope ``(B, C, E)`` (E = envelope-time bins) instead of a 2-D spectrogram / 3-D video
volume. The envelope is split into non-overlapping ``patch_e``-bin patches along the TIME
(envelope-bin) axis; each patch carries all ``C`` channels as a feature dimension. Patches
are linearly embedded to ``d_model`` with a learned per-patch position embedding, run through
an ``x_transformers.Encoder`` (bidirectional — a codec frame is a token set, no causal mask),
FSQ-quantized, and the decoder mirrors the path back to ``(B, C, E)``.

Only external libs (torch, x_transformers, einops) + config.py are used; **no** FAITH model
code is imported (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import FastTSCodecConfig


class _EnvPatchPosEmb(nn.Module):
    """Additive learned position embedding over the envelope-time patches.

    Token order is the envelope-patch index (row-major over the single time axis). Returns
    ``(1, n_tok, d_model)`` broadcastable.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.n_e = cfg.n_env_patch
        self.env_pe = nn.Parameter(torch.zeros(self.n_e, cfg.d_model))
        nn.init.normal_(self.env_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.env_pe.unsqueeze(0)  # (1, n_tok, d_model)


class FastTSEncoder(nn.Module):
    """(B, C, E) envelope -> (B, n_tok, d_model)."""

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.patch_e
        self.to_tokens = nn.Linear(patch_dim, cfg.d_model)
        self.pos_emb = _EnvPatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.enc_depth,
            heads=cfg.heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # patchify along the envelope-time axis: n_tok = E // patch_e; each patch is
        # (channels * patch_e) features.
        patches = rearrange(
            x,
            "b c (ne pe) -> b ne (c pe)",
            pe=cfg.patch_e,
        )
        tokens = self.to_tokens(patches) + self.pos_emb()
        return self.transformer(tokens)


class FastTSDecoder(nn.Module):
    """(B, n_tok, d_model) -> (B, C, E).

    Generative (adversarial) decoder: a transformer over the quantized tokens, a linear
    projection to per-patch envelope values, then a parameter-free unpatchify. ``to_pixels``
    is the last layer that hallucinates the envelope realization from the statistic-codes
    (§4.1) and is the VQGAN adaptive-adversarial-weight anchor.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.patch_e
        self.pos_emb = _EnvPatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.dec_depth,
            heads=cfg.heads,
        )
        self.to_pixels = nn.Linear(cfg.d_model, patch_dim)

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight ``Parameter`` of the final layer producing the (B, C, E) output.

        This is ``to_pixels.weight`` — the last linear before the (parameter-free)
        unpatchify rearrange. The VQGAN adaptive adversarial weight balances the
        reconstruction and adversarial gradients at this tensor ("Taming Transformers" §3.3),
        exactly as ``nets.SpectroDecoder.last_layer`` / ``video_nets.VideoDecoder.last_layer``.
        """
        return self.to_pixels.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_pixels(h)
        return rearrange(
            patches,
            "b ne (c pe) -> b c (ne pe)",
            ne=cfg.n_env_patch,
            c=cfg.channels,
            pe=cfg.patch_e,
        )
