"""Phase-A slow-TS encoder / decoder (statistics-first, "lightest touch" codec).

Mirrors ``nets.py`` (the spectrogram codec) but patchifies over the slow-TS window
``(B, C, T)`` — ``C`` profile positions × ``T`` time samples — instead of the spectrogram
``(B, C, F, T)``. Both are built on ``x_transformers.Encoder`` attention primitives (no
causal mask — a codec frame is a bidirectional token set). The window is split into
non-overlapping ``(patch_c × patch_t)`` position-time patches, linearly embedded to
``d_model``, and given learned factorized position/time patch-position embeddings. The
decoder mirrors the path: transformer over the quantized tokens, then a linear unpatchify
back to ``(B, C, T)``.

Unlike the spectro / video decoders there is **no adversarial head** (docs/IGNITE_DESIGN.md
§4.3 "lightest touch"; see ``config.SlowTSCodecConfig``), so the decoder's final linear is a
plain reconstruction projection. ``last_layer`` is still exposed (the ``to_signal`` weight)
for symmetry / diagnostics, but the slow-TS codec's ``generator_losses`` uses a masked
reconstruction + entropy objective and never invokes the VQGAN adaptive-adversarial balance.

Only external libs (torch, x_transformers, einops) + config.py are used; **no** FAITH model
code is imported (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import SlowTSCodecConfig


class _SlowTSPatchPosEmb(nn.Module):
    """Additive learned position embedding factorized over position / time patches.

    Token order is ``(pos_patch, time_patch)`` row-major (position outer, time inner),
    matching the patchify rearrange below. Returns ``(1, n_tok, d_model)`` broadcastable.
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
        # patchify: position outer, time inner -> n_tok = n_c * n_t. Each patch is a flat
        # (patch_c * patch_t) vector of the smooth profile slice.
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

    Plain (non-adversarial) reconstruction decoder: a transformer over the quantized tokens
    followed by a linear projection back to per-patch values and a parameter-free unpatchify.
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
        """The weight ``Parameter`` of the final layer producing the (B, C, T) output.

        This is ``to_signal.weight`` — the last linear before the (parameter-free) unpatchify.
        Exposed for symmetry with ``nets.SpectroDecoder.last_layer`` / diagnostics; the
        slow-TS codec has no adversarial term, so (unlike spectro/video) it is NOT used by an
        adaptive-adversarial weight.
        """
        return self.to_signal.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_signal(h)
        # unpatchify: inverse of the encoder rearrange.
        return rearrange(
            patches,
            "b (nc nt) (pc pt) -> b (nc pc) (nt pt)",
            nc=cfg.n_pos_patch,
            nt=cfg.n_time_patch,
            pc=cfg.patch_c,
            pt=cfg.patch_t,
        )
