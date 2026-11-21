"""Slow-TS encoder and decoder.

Mirrors ``nets.py`` (the spectrogram codec) but patchifies the slow-TS window
``(B, C, T)``, ``C`` profile positions by ``T`` time samples, instead of a spectrogram
``(B, C, F, T)``. Both sides are ``x_transformers.Encoder`` stacks without a causal
mask, since a codec frame is a bidirectional token set. The window is cut into non-overlapping
``(patch_c, patch_t)`` patches, embedded linearly to ``d_model`` and given learned
factorized position embeddings. The decoder runs a transformer over the quantized tokens
and unpatchifies linearly back to ``(B, C, T)``.

The slow-TS codec has no adversarial term (docs/IGNITE_DESIGN.md §4.3, see
``config.SlowTSCodecConfig``), so the final linear is a plain reconstruction projection.
``last_layer`` is still exposed for symmetry with the other decoders and for
diagnostics. No FAITH model code is imported (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import SlowTSCodecConfig


class _SlowTSPatchPosEmb(nn.Module):
    """Learned additive position embedding, factorized over position and time patches.

    Token order is ``(pos_patch, time_patch)``, position outer and time inner, matching
    the patchify below. Returns ``(1, n_tok, d_model)``.
    """

    def __init__(self, cfg: SlowTSCodecConfig) -> None:
        super().__init__()
        self.n_c = cfg.n_pos_patch
        self.n_t = cfg.n_time_patch
        self.pos_pe = nn.Parameter(torch.zeros(self.n_c, cfg.d_model))
        self.time_pe = nn.Parameter(torch.zeros(self.n_t, cfg.d_model))
        nn.init.normal_(self.pos_pe, std=0.02)
        nn.init.normal_(self.time_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        # (n_c, 1, d) + (1, n_t, d) -> (n_c, n_t, d) -> (1, n_tok, d)
        pe = self.pos_pe[:, None, :] + self.time_pe[None, :, :]
        return rearrange(pe, "c t d -> 1 (c t) d")


class SlowTSEncoder(nn.Module):
    """(B, C, T) -> (B, n_tok, d_model)."""

    def __init__(self, cfg: SlowTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.patch_c * cfg.patch_t
        self.to_tokens = nn.Linear(patch_dim, cfg.d_model)
        self.pos_emb = _SlowTSPatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.enc_depth,
            heads=cfg.heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # Patchify, position outer and time inner, so n_tok = n_c * n_t. `x` has
        # `padded_channels` positions (n_pos_patch * patch_c): the loader pads the
        # profile up to that count and masks the padding
        # (SlowTSCodecPairDataset._fit_window).
        patches = rearrange(
            x,
            "b (nc pc) (nt pt) -> b (nc nt) (pc pt)",
            pc=cfg.patch_c,
            pt=cfg.patch_t,
        )
        tokens = self.to_tokens(patches) + self.pos_emb()
        return self.transformer(tokens)


class SlowTSDecoder(nn.Module):
    """(B, n_tok, d_model) -> (B, C, T).

    A transformer over the quantized tokens, a linear projection to per-patch values and
    a parameter-free unpatchify.
    """

    def __init__(self, cfg: SlowTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.patch_c * cfg.patch_t
        self.pos_emb = _SlowTSPatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.dec_depth,
            heads=cfg.heads,
        )
        self.to_signal = nn.Linear(cfg.d_model, patch_dim)

    @property
    def last_layer(self) -> nn.Parameter:
        """Weight of the last layer before the unpatchify (``to_signal.weight``).

        Kept for symmetry with ``nets.SpectroDecoder.last_layer``. With no adversarial
        term, no adaptive weight uses it.
        """
        return self.to_signal.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_signal(h)
        # Inverse of the encoder patchify.
        return rearrange(
            patches,
            "b (nc nt) (pc pt) -> b (nc pc) (nt pt)",
            nc=cfg.n_pos_patch,
            nt=cfg.n_time_patch,
            pc=cfg.patch_c,
            pt=cfg.patch_t,
        )
