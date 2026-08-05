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

from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from x_transformers import Encoder

from .config import SpectroCodecConfig


class _PatchPosEmb(nn.Module):
    """Additive learned position embedding factorized over freq / time patches.

    Token order is ``(freq_patch, time_patch)`` row-major (freq outer), matching
    the patchify rearrange below. Returns ``(1, n_tok, d_model)`` broadcastable.

    With ``cfg.channel_groups > 1`` (channel-factorized tokens) a learned per-group
    embedding is added and the token order becomes ``(group, freq_patch, time_patch)``
    (group outer), matching the grouped patchify rearrange. At the default
    ``channel_groups == 1`` no group parameter is created and the module is
    byte-identical to before (state_dict-compatible).
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.n_f = cfg.n_freq_patch
        self.n_t = cfg.n_time_patch
        self.n_g = int(getattr(cfg, "channel_groups", 1))
        self.freq_pe = nn.Parameter(torch.zeros(self.n_f, cfg.d_model))
        self.time_pe = nn.Parameter(torch.zeros(self.n_t, cfg.d_model))
        nn.init.normal_(self.freq_pe, std=0.02)
        nn.init.normal_(self.time_pe, std=0.02)
        if self.n_g > 1:
            self.group_pe = nn.Parameter(torch.zeros(self.n_g, cfg.d_model))
            nn.init.normal_(self.group_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        # (n_f, 1, d) + (1, n_t, d) -> (n_f, n_t, d)
        pe = self.freq_pe[:, None, :] + self.time_pe[None, :, :]
        if self.n_g > 1:
            pe = self.group_pe[:, None, None, :] + pe[None]   # (g, n_f, n_t, d)
            return rearrange(pe, "g f t d -> 1 (g f t) d")
        return rearrange(pe, "f t d -> 1 (f t) d")


class SpectroEncoder(nn.Module):
    """(B, C, F, T) -> (B, n_tok, d_model)."""

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        g = int(getattr(cfg, "channel_groups", 1))
        assert cfg.channels % g == 0, (
            f"channels ({cfg.channels}) must divide by channel_groups ({g})"
        )
        patch_dim = (cfg.channels // g) * cfg.patch_f * cfg.patch_t
        self.to_tokens = nn.Linear(patch_dim, cfg.d_model)
        self.pos_emb = _PatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.enc_depth,
            heads=cfg.heads,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # patchify: group outer, then freq, time inner -> n_tok = g * n_f * n_t.
        # At g=1 this is the identical (c pf pt) mapping as before (byte-identical).
        patches = rearrange(
            x,
            "b (g gc) (nf pf) (nt pt) -> b (g nf nt) (gc pf pt)",
            g=int(getattr(cfg, "channel_groups", 1)),
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
        g = int(getattr(cfg, "channel_groups", 1))
        patch_dim = (cfg.channels // g) * cfg.patch_f * cfg.patch_t
        self.pos_emb = _PatchPosEmb(cfg)
        self.transformer = Encoder(
            dim=cfg.d_model,
            depth=cfg.dec_depth,
            heads=cfg.heads,
        )
        self.to_pixels = nn.Linear(cfg.d_model, patch_dim)

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight ``Parameter`` of the final layer producing the (B,C,F,T) output.

        This is ``to_pixels.weight`` — the last conv/linear before the (parameter-free)
        unpatchify rearrange. The VQGAN adaptive adversarial weight balances the
        reconstruction and adversarial gradients at this tensor (see
        ``codec.SpectroCodec.generator_losses`` and "Taming Transformers" §3.3).
        """
        return self.to_pixels.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        h = self.transformer(quant + self.pos_emb())
        patches = self.to_pixels(h)
        # unpatchify: inverse of the encoder rearrange (g=1 -> identical to before).
        g = int(getattr(cfg, "channel_groups", 1))
        return rearrange(
            patches,
            "b (g nf nt) (gc pf pt) -> b (g gc) (nf pf) (nt pt)",
            g=g,
            nf=cfg.n_freq_patch,
            nt=cfg.n_time_patch,
            gc=cfg.channels // g,
            pf=cfg.patch_f,
            pt=cfg.patch_t,
        )


class _ResBlock2d(nn.Module):
    """VQGAN/HiFi-GAN-style pre-activation residual conv block (2D).

    ``Conv2d -> LeakyReLU -> Conv2d`` with a skip connection, all 3×3 same-padding so the
    spatial size is preserved. Synthesizes local texture on top of the upsampled feature map
    (the very thing a single linear ``to_pixels`` map cannot do). A 1×1 projection is used on
    the skip when ``in_ch != out_ch`` so the block can also change channel count.
    """

    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(slope)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(x))
        h = self.conv2(self.act(h))
        return h + self.skip(x)


class SpectroConvDecoder(nn.Module):
    """(B, n_tok, d_model) -> (B, C, F, T), HiFi-GAN/VQGAN-style 2D conv upsampler.

    Drop-in replacement for :class:`SpectroDecoder` (identical ``forward(quant) -> (B,C,F,T)``
    contract and ``last_layer`` property) whose ``to_pixels`` linear can only produce SMOOTH
    patches. Here each FSQ token is placed on a coarse ``(n_freq_patch, n_time_patch)`` grid and
    a stack of transposed-conv upsample + residual-conv blocks *synthesizes* the fine turbulent
    texture (bes/mhr/ece broadband) that a per-patch linear map cannot.

    Structure (NeMo HiFiGAN generator, 2D analogue; cf. VQGAN "Taming Transformers" decoder):
        tokens (B, n_tok, d)                        [freq-outer / time-inner order]
          -> reshape to feature map (B, d, n_freq_patch, n_time_patch)
          -> proj_in: 1×1 Conv2d  d_model -> base_ch
          -> [ ConvTranspose2d(stride=(sf, st))     upsample freq/time by 2 while needed
               + LeakyReLU
               + res_blocks × _ResBlock2d ] × n_up  channels halved every 2 blocks (>= min_ch)
          -> (if a patch size is not a power of 2) F.interpolate to the exact (F, T)
          -> to_out: 3×3 Conv2d  ch -> C           (no activation) => (B, C, F, T)

    The per-axis upsample factors are ``patch_f`` (freq) and ``patch_t`` (time). We do
    ``n_up = max(log2(patch_f), log2(patch_t))`` blocks, each doubling an axis by 2 *only while
    that axis still needs it* (else stride 1 on that axis). For the power-of-2 patch sizes used
    in production (16/32/64) this reaches ``(F, T)`` exactly with no interpolation.
    """

    def __init__(self, cfg: SpectroCodecConfig) -> None:
        super().__init__()
        if int(getattr(cfg, "channel_groups", 1)) != 1:
            raise NotImplementedError(
                "SpectroConvDecoder does not support channel_groups > 1; "
                "use decoder='linear' for channel-factorized tokens"
            )
        self.cfg = cfg
        base_ch = int(cfg.conv_dec_base_ch)
        n_res = int(cfg.conv_dec_res_blocks)
        min_ch = 64

        # per-axis upsample counts (how many ×2 steps each axis needs to reach the patch size).
        self._pow2_f = (cfg.patch_f & (cfg.patch_f - 1)) == 0 and cfg.patch_f > 0
        self._pow2_t = (cfg.patch_t & (cfg.patch_t - 1)) == 0 and cfg.patch_t > 0
        n_freq_ups = int(round(log2(cfg.patch_f))) if self._pow2_f else 0
        n_time_ups = int(round(log2(cfg.patch_t))) if self._pow2_t else 0
        n_up = max(n_freq_ups, n_time_ups, 0)

        self.proj_in = nn.Conv2d(cfg.d_model, base_ch, kernel_size=1)

        blocks: list[nn.Module] = []
        ch = base_ch
        for i in range(n_up):
            sf = 2 if i < n_freq_ups else 1   # still upsample freq?
            st = 2 if i < n_time_ups else 1   # still upsample time?
            # halve channels every 2 upsample blocks, floored at min_ch (last block also halves).
            out_ch = max(min_ch, base_ch // (2 ** ((i // 2) + 1)))
            # ConvTranspose2d out = (in-1)*stride - 2*pad + kernel + output_pad. Per axis:
            #   stride 2 (upsample ×2): kernel 4, pad 1, output_pad 0 -> out = 2*in  (exact).
            #   stride 1 (keep size):   kernel 3, pad 1, output_pad 0 -> out = in     (exact).
            blocks.append(
                nn.ConvTranspose2d(
                    ch, out_ch,
                    kernel_size=(4 if sf == 2 else 3, 4 if st == 2 else 3),
                    stride=(sf, st),
                    padding=1,
                    output_padding=0,
                )
            )
            blocks.append(nn.LeakyReLU(0.2))
            for _ in range(max(1, n_res)):
                blocks.append(_ResBlock2d(out_ch, out_ch))
            ch = out_ch
        self.up = nn.ModuleList(blocks)
        self._final_ch = ch

        # final projection to (B, C, F, T). 3×3 same-pad conv, NO activation. This is the
        # `last_layer` the VQGAN adaptive-adversarial weight balances gradients at.
        self.to_out = nn.Conv2d(ch, cfg.channels, kernel_size=3, padding=1)

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight ``Parameter`` of the final conv producing the (B,C,F,T) output.

        This is ``to_out.weight`` — the last conv before the reconstruction, mirroring
        ``SpectroDecoder.last_layer = to_pixels.weight``. The VQGAN adaptive adversarial weight
        balances the reconstruction and adversarial gradients at this tensor (see
        ``codec.SpectroCodec.generator_losses`` and "Taming Transformers" §3.3); it MUST be the
        last conv before the output.
        """
        return self.to_out.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # tokens are (freq-outer, time-inner) -> place on the coarse 2D grid.
        # inverse of the encoder's "b c (nf pf) (nt pt) -> b (nf nt) (c pf pt)" ordering:
        # here we only undo the (nf nt) flattening into a (d, nf, nt) feature map.
        h = rearrange(
            quant,
            "b (nf nt) d -> b d nf nt",
            nf=cfg.n_freq_patch,
            nt=cfg.n_time_patch,
        )
        h = self.proj_in(h)
        for layer in self.up:
            h = layer(h)
        # If a patch size was not a power of 2 (upsample stack under-shoots), snap to (F, T).
        if h.shape[-2] != cfg.freq_bins or h.shape[-1] != cfg.time_frames:
            h = F.interpolate(
                h, size=(cfg.freq_bins, cfg.time_frames),
                mode="bilinear", align_corners=False,
            )
        return self.to_out(h)
