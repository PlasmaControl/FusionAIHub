"""Phase-A spectrogram encoder / decoder (statistics-first codec).

Both built on ``x_transformers.Encoder`` attention primitives (no causal mask;
a codec frame is processed as a bidirectional token set). The spectrogram
``(B, C, F, T)`` is patchified into ``(patch_f x patch_t)`` non-overlapping
patches, linearly embedded to ``d_model``, and given learned freq/time
patch-position embeddings. The decoder mirrors the path: transformer over the
quantized tokens, then a linear unpatchify back to ``(B, C, F, T)``.

Only external libs (torch, x_transformers, einops) + config.py are used; no
FAITH model code is imported.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import SpectroCodecConfig


class _PatchPosEmb(nn.Module):
    """Additive learned position embedding factorized over freq / time patches.

    Token order is ``(freq_patch, time_patch)`` row-major (freq outer), matching
    the patchify rearrange below. Returns ``(1, n_tok, d_model)`` broadcastable.
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.n_f = cfg.n_freq_patch
        self.n_t = cfg.n_time_patch
        self.freq_pe = nn.Parameter(torch.zeros(self.n_f, cfg.d_model))
        self.time_pe = nn.Parameter(torch.zeros(self.n_t, cfg.d_model))
        nn.init.normal_(self.freq_pe, std=0.02)
        nn.init.normal_(self.time_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        # (n_f, 1, d) + (1, n_t, d) -> (n_f, n_t, d) -> (1, n_tok, d)
        pe = self.freq_pe[:, None, :] + self.time_pe[None, :, :]
        return rearrange(pe, "f t d -> 1 (f t) d")


class SpectroEncoder(nn.Module):
    """(B, C, F, T) -> (B, n_tok, d_model)."""

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.patch_f * cfg.patch_t
        self.to_tokens = nn.Linear(patch_dim, cfg.d_model)
        self.pos_emb = _PatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.enc_depth,
            heads=cfg.heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # patchify: freq outer, time inner -> n_tok = n_f * n_t
        patches = rearrange(
            x,
            "b c (nf pf) (nt pt) -> b (nf nt) (c pf pt)",
            pf=cfg.patch_f,
            pt=cfg.patch_t,
        )
        tokens = self.to_tokens(patches) + self.pos_emb()
        return self.transformer(tokens)


class SpectroDecoder(nn.Module):
    """(B, n_tok, d_model) -> (B, C, F, T)."""

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.patch_f * cfg.patch_t
        self.pos_emb = _PatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.dec_depth,
            heads=cfg.heads,
        )
        self.to_pixels = nn.Linear(cfg.d_model, patch_dim)

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_pixels(h)
        # unpatchify: inverse of the encoder rearrange.
        return rearrange(
            patches,
            "b (nf nt) (c pf pt) -> b c (nf pf) (nt pt)",
            nf=cfg.n_freq_patch,
            nt=cfg.n_time_patch,
            c=cfg.channels,
            pf=cfg.patch_f,
            pt=cfg.patch_t,
        )
