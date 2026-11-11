"""tangtv video encoder and decoder.

Mirrors ``nets.py`` (the spectrogram codec) but patchifies the video volume
``(B, C, T, H, W)`` instead of a spectrogram ``(B, C, F, T)``. Both sides are
``x_transformers.Encoder`` stacks without a causal mask, since a codec frame is a
bidirectional token set. The window is cut into non-overlapping
``(patch_t, patch_h, patch_w)`` patches, embedded linearly to ``d_model`` and given
learned factorized time, height and width embeddings. The decoder runs a transformer over the quantized tokens
and unpatchifies linearly back to ``(B, C, T, H, W)``. No FAITH model code is used
(docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import VideoCodecConfig


class _VideoPatchPosEmb(nn.Module):
    """Learned additive position embedding, factorized over time, height and width.

    Token order is ``(time_patch, height_patch, width_patch)``, time outer and width
    inner, matching the patchify below. Returns ``(1, n_tok, d_model)``.
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
        # Patchify, time outer and width inner, so n_tok = n_t * n_h * n_w.
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

    Adversarially trained decoder: a transformer over the quantized tokens, a linear
    projection to per-patch pixels and a parameter-free unpatchify. ``to_pixels`` plays
    the role of a deconv stack's last layer and renders the frame from the codes (§4.1).
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
        # Optional residual per-frame conv refinement (cfg.refine_depth > 0): stride-1,
        # kernel-3 convs that blend the separately rendered 20x20 patches across their
        # seams (against the checkerboard; see the config). The last conv is
        # zero-initialised, so the head starts as the identity. Depth 0, and old pickled
        # configs without the field (hence getattr), build nothing and keep the old
        # state_dict.
        depth = int(getattr(cfg, "refine_depth", 0))
        self.refine: nn.Sequential | None = None
        if depth > 0:
            hidden = int(getattr(cfg, "refine_hidden", 64))
            # With cfg.refine_dilated (default False, all dilations 1) layer i uses
            # dilation 2**i, so the receptive field is 2^(depth+1)-1 instead of
            # 2*depth+1. The head has to see further than one 20x20 patch to remove
            # seams: depth 4 dilated (RF 31) does, depth 4 undilated (RF 9) only smooths
            # inside a patch.
            dilated = bool(getattr(cfg, "refine_dilated", False))
            layers: list[nn.Module] = []
            for i in range(depth - 1):
                d = (2 ** i) if dilated else 1
                layers += [
                    nn.Conv2d(cfg.channels if i == 0 else hidden, hidden, 3,
                              padding=d, dilation=d),
                    nn.GELU(),
                ]
            d_last = (2 ** (depth - 1)) if dilated else 1
            last = nn.Conv2d(hidden if depth > 1 else cfg.channels, cfg.channels, 3,
                             padding=d_last, dilation=d_last)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            layers.append(last)
            self.refine = nn.Sequential(*layers)

    @property
    def last_layer(self) -> nn.Parameter:
        """Weight of the last parameterized layer on the output path.

        ``to_pixels.weight`` without the refinement head, otherwise the last refinement
        conv's weight. The VQGAN adaptive adversarial weight balances the reconstruction
        and adversarial gradients at this tensor
        (``video_codec.VideoCodec.generator_losses``, "Taming Transformers" §3.3), as
        ``nets.SpectroDecoder.last_layer`` does.
        """
        if self.refine is not None:
            return self.refine[-1].weight
        return self.to_pixels.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_pixels(h)
        # Inverse of the encoder patchify.
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
