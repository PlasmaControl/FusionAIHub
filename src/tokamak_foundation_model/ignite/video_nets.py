"""Phase-A tangtv **video** encoder / decoder (statistics-first, generative codec).

Mirrors ``nets.py`` (the spectrogram codec) but patchifies over the video volume
``(B, C, T, H, W)`` instead of the spectrogram ``(B, C, F, T)``. Both are built on
``x_transformers.Encoder`` attention primitives (no causal mask — a codec frame is a
bidirectional token set). The video window is split into non-overlapping
``(patch_t × patch_h × patch_w)`` space-time patches, linearly embedded to ``d_model``,
and given learned factorized time/height/width patch-position embeddings. The decoder
mirrors the path: transformer over the quantized tokens, then a linear unpatchify back to
``(B, C, T, H, W)``.

Only external libs (torch, x_transformers, einops) + config.py are used; **no** FAITH
model code is imported (docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import VideoCodecConfig


class _VideoPatchPosEmb(nn.Module):
    """Additive learned position embedding factorized over time / height / width patches.

    Token order is ``(time_patch, height_patch, width_patch)`` row-major (time outer,
    width inner), matching the patchify rearrange below. Returns ``(1, n_tok, d_model)``.
    """

    def __init__(self, cfg: VideoCodecConfig) -> None:
        super().__init__()
        self.n_t = cfg.n_time_patch
        self.n_h = cfg.n_height_patch
        self.n_w = cfg.n_width_patch
        self.time_pe = nn.Parameter(torch.zeros(self.n_t, cfg.d_model))
        self.height_pe = nn.Parameter(torch.zeros(self.n_h, cfg.d_model))
        self.width_pe = nn.Parameter(torch.zeros(self.n_w, cfg.d_model))
        nn.init.normal_(self.time_pe, std=0.02)
        nn.init.normal_(self.height_pe, std=0.02)
        nn.init.normal_(self.width_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        # (n_t,1,1,d) + (1,n_h,1,d) + (1,1,n_w,d) -> (n_t, n_h, n_w, d) -> (1, n_tok, d)
        pe = (
            self.time_pe[:, None, None, :]
            + self.height_pe[None, :, None, :]
            + self.width_pe[None, None, :, :]
        )
        return rearrange(pe, "t h w d -> 1 (t h w) d")


class VideoEncoder(nn.Module):
    """(B, C, T, H, W) -> (B, n_tok, d_model)."""

    def __init__(self, cfg: VideoCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.patch_t * cfg.patch_h * cfg.patch_w
        self.to_tokens = nn.Linear(patch_dim, cfg.d_model)
        self.pos_emb = _VideoPatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.enc_depth,
            heads=cfg.heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # patchify: time outer, width inner -> n_tok = n_t * n_h * n_w
        patches = rearrange(
            x,
            "b c (nt pt) (nh ph) (nw pw) -> b (nt nh nw) (c pt ph pw)",
            pt=cfg.patch_t,
            ph=cfg.patch_h,
            pw=cfg.patch_w,
        )
        tokens = self.to_tokens(patches) + self.pos_emb()
        return self.transformer(tokens)


class VideoDecoder(nn.Module):
    """(B, n_tok, d_model) -> (B, C, T, H, W).

    Generative (adversarial) decoder: a transformer over the quantized tokens followed by a
    linear projection to per-patch pixels and a parameter-free unpatchify. The
    ``to_pixels`` linear is the analogue of a deconv stack's final layer — it is what
    hallucinates the frame's realization from the statistic-codes (§4.1).
    """

    def __init__(self, cfg: VideoCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.channels * cfg.patch_t * cfg.patch_h * cfg.patch_w
        self.pos_emb = _VideoPatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.dec_depth,
            heads=cfg.heads,
        )
        self.to_pixels = nn.Linear(cfg.d_model, patch_dim)
        # Optional RESIDUAL per-frame conv refinement head (cfg.refine_depth > 0): stride-1
        # kernel-3 2D convs that blend the linear head's independently-rendered patches across
        # their 20x20 seams (the v6 GAN-free checkerboard fix — see the config note). The final
        # conv is ZERO-INIT so the head starts as an exact identity; depth 0 (and old pickled
        # configs, which lack the field entirely — hence the getattr) builds NOTHING, keeping
        # the state_dict byte-identical to pre-refine checkpoints.
        depth = int(getattr(cfg, "refine_depth", 0))
        self.refine: nn.Sequential | None = None
        if depth > 0:
            hidden = int(getattr(cfg, "refine_hidden", 64))
            layers: list[nn.Module] = []
            for i in range(depth - 1):
                layers += [
                    nn.Conv2d(cfg.channels if i == 0 else hidden, hidden, 3, padding=1),
                    nn.GELU(),
                ]
            last = nn.Conv2d(hidden if depth > 1 else cfg.channels, cfg.channels, 3, padding=1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            layers.append(last)
            self.refine = nn.Sequential(*layers)

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight ``Parameter`` of the final layer producing the (B,C,T,H,W) output.

        Without the refinement head this is ``to_pixels.weight`` — the last linear before the
        (parameter-free) unpatchify rearrange; with ``cfg.refine_depth > 0`` it is the final
        refinement conv's weight (the last parameterized layer on the output path). The VQGAN
        adaptive adversarial weight balances the reconstruction and adversarial gradients at
        this tensor (see ``video_codec.VideoCodec.generator_losses`` and "Taming Transformers"
        §3.3), exactly as ``nets.SpectroDecoder.last_layer`` does for the spectro codec.
        """
        if self.refine is not None:
            return self.refine[-1].weight
        return self.to_pixels.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_pixels(h)
        # unpatchify: inverse of the encoder rearrange.
        x = rearrange(
            patches,
            "b (nt nh nw) (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)",
            nt=cfg.n_time_patch,
            nh=cfg.n_height_patch,
            nw=cfg.n_width_patch,
            c=cfg.channels,
            pt=cfg.patch_t,
            ph=cfg.patch_h,
            pw=cfg.patch_w,
        )
        if self.refine is not None:
            frames = rearrange(x, "b c t h w -> (b t) c h w")
            frames = frames + self.refine(frames)
            x = rearrange(frames, "(b t) c h w -> b c t h w", b=x.shape[0])
        return x
