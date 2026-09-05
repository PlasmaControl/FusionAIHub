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


_GAIN_EPS: float = 1e-6


def spectro_gain_shape_split(x: torch.Tensor, use_scale: bool):
    """Split a spectrogram window into (envelope gain, normalized shape).

    The spectro port of ``fastts_nets.gain_shape_split``, with the statistics taken per
    (channel, FREQUENCY BIN) along TIME instead of per channel along the envelope bins --
    because that is where the structure the world model needs lives: a mode is a coherent
    track at one frequency, and its amplitude in time is exactly ``sigma`` below.

    ``x`` is ``(B, C, F, T)``. Returns ``(gain, shape, level, sigma)``::

        level (B, C, F)      x.mean(-1)                      the static envelope
        sigma (B, C, F)      (x - level).std(-1)             the TEMPORAL amplitude per bin
        shape (B, C, F, T)   (x - level) / sigma  when ``use_scale`` (zero mean, unit std
                             along T), else just ``x - level`` (zero mean, free scale)
        gain  (B, 2C or C, F)  ``[level ; log1p(sigma)]`` -- the gain encoder's input

    ``log1p`` on sigma because sigma is non-negative and spans orders of magnitude across
    quiet and active frequency bins; the log makes it an O(1) regression target, and
    ``expm1`` in the decoder inverts it exactly.
    """
    level = x.mean(dim=-1)                                          # (B, C, F)
    res = x - level.unsqueeze(-1)
    sigma = res.std(dim=-1, unbiased=False)                         # (B, C, F)
    if use_scale:
        shape = res / sigma.clamp_min(_GAIN_EPS).unsqueeze(-1)
        gain = torch.cat([level, torch.log1p(sigma.clamp_min(0.0))], dim=1)
    else:
        shape = res
        gain = level
    return gain, shape, level, sigma


def _gain_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    """Per-patch 3-layer MLP. Shared across gain tokens (applied on the last axis)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


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
        # --- ENVELOPE/SHAPE SPLIT (cfg.gain_shape; see SpectroCodecConfig) ------------- #
        # Built ONLY when enabled, so an unchanged config has exactly the pre-2026-09-04
        # parameter set and every existing checkpoint still loads strictly.
        # The gain tokens BYPASS the transformer: the split is meant to be HARD, and
        # self-attention between gain and shape tokens would let the shape path leak back
        # into the envelope code -- which is the failure the split exists to remove.
        self.gain_shape = cfg.n_gain_tok > 0
        if self.gain_shape:
            self.gain_to_tokens = _gain_mlp(
                cfg.gain_values_per_tok, cfg.gain_hidden, cfg.d_model)
            # n_tok shape patches -> n_shape_tok tokens: a learned mix along the TOKEN axis,
            # so the patchify/unpatchify geometry (and therefore the decoder) is unchanged.
            self.shape_mix = nn.Linear(cfg.n_tok, cfg.n_shape_tok)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if self.gain_shape:
            gain, x = spectro_gain_shape_split(x, cfg.uses_gain_scale)[:2]
            # gain (B, G, F) -> one token per contiguous frequency band, all channels.
            g_patch = rearrange(gain, "b g (ng pf) -> b ng (g pf)", pf=cfg.gain_patch_f)
            g_tok = self.gain_to_tokens(g_patch)                # (B, n_gain_tok, d)
        # patchify: group outer, then freq, time inner -> n_tok = g * n_f * n_t.
        # At g=1 this is the identical (c pf pt) mapping as before (byte-identical).
        patches = rearrange(
            x,
            "b (g gc) (nf pf) (nt pt) -> b (g nf nt) (gc pf pt)",
            g=int(getattr(cfg, "channel_groups", 1)),
            pf=cfg.patch_f,
            pt=cfg.patch_t,
        )
        if not self.gain_shape:
            tokens = self.to_tokens(patches) + self.pos_emb()
            return self.transformer(tokens)
        s_tok = self.transformer(self.to_tokens(patches))       # (B, n_tok, d)
        s_tok = self.shape_mix(s_tok.transpose(1, 2)).transpose(1, 2)   # (B, n_shape_tok, d)
        return torch.cat([g_tok, s_tok], dim=1) + self.pos_emb()


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
        # Optional RESIDUAL conv refinement head over the ASSEMBLED (F, T) spectrogram
        # (cfg.refine_depth > 0): stride-1 kernel-3 2D convs that give the decoder
        # full-resolution CROSS-PATCH context, so its texture is no longer forced to be one
        # shared basis tiled on a (patch_f x patch_t) lattice — the measured checkerboard
        # (gate.patch_lattice_metrics; see the SpectroCodecConfig note). The final conv is
        # ZERO-INIT so the head starts as an exact identity; depth 0 (and old pickled configs,
        # which lack the field entirely — hence the getattr) builds NOTHING, keeping the
        # state_dict byte-identical to pre-refine checkpoints.
        depth = int(getattr(cfg, "refine_depth", 0))
        self.refine: nn.Sequential | None = None
        if depth > 0:
            hidden = int(getattr(cfg, "refine_hidden", 64))
            dilated = bool(getattr(cfg, "refine_dilated", False))
            ch = cfg.channels
            layers: list[nn.Module] = []
            for i in range(depth - 1):
                d = (2 ** i) if dilated else 1
                layers += [
                    nn.Conv2d(ch if i == 0 else hidden, hidden, 3, padding=d, dilation=d),
                    nn.GELU(),
                ]
            d_last = (2 ** (depth - 1)) if dilated else 1
            last = nn.Conv2d(
                hidden if depth > 1 else ch, ch, 3, padding=d_last, dilation=d_last,
            )
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            layers.append(last)
            self.refine = nn.Sequential(*layers)
        # Optional StyleGAN-style per-pixel NOISE input with a learned PER-CHANNEL scale
        # (cfg.decoder_noise). Zero-init => an exact identity at step 0; see the
        # SpectroCodecConfig note for why a deterministic decoder cannot satisfy the
        # adversarial term without tiling a fixed texture. Applied at FULL resolution AFTER
        # the refinement head, in train AND eval (this is a generative decoder — the decoded
        # spectrogram is a sample, not a conditional mean).
        self.noise_scale: nn.Parameter | None = (
            nn.Parameter(torch.zeros(cfg.channels))
            if bool(getattr(cfg, "decoder_noise", False)) else None
        )
        # --- ENVELOPE/SHAPE SPLIT heads (cfg.gain_shape) ------------------------------- #
        # `gain_head` is the ONLY thing that produces the reconstruction's per-(channel,
        # frequency) time-mean AND its temporal std -- the shape branch is mean-removed and
        # (when gain_scale) std-normalized in `_forward_gain_shape` below, so it can carry
        # neither. That is what makes a flat plate unrepresentable.
        self.gain_shape = cfg.n_gain_tok > 0
        if self.gain_shape:
            self.gain_head = _gain_mlp(
                cfg.d_model, cfg.gain_hidden, cfg.gain_values_per_tok)
            self.shape_unmix = nn.Linear(cfg.n_shape_tok, cfg.n_tok)

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight ``Parameter`` of the final layer producing the (B,C,F,T) output.

        Without the refinement head this is ``to_pixels.weight`` — the last conv/linear before
        the (parameter-free) unpatchify rearrange; with ``cfg.refine_depth > 0`` it is the final
        refinement conv's weight (the last parameterized layer on the output path). The VQGAN
        adaptive adversarial weight balances the reconstruction and adversarial gradients at
        this tensor (see ``codec.SpectroCodec.generator_losses`` and "Taming Transformers"
        §3.3); it MUST be the last parameterized layer before the output.
        """
        if self.refine is not None:
            return self.refine[-1].weight
        return self.to_pixels.weight

    def forward(self, quant: torch.Tensor, return_aux: bool = False):
        """Quantized tokens -> reconstruction.

        ``return_aux`` (gain-shape only) additionally returns the decoded gain
        ``(B, 2C or C, F)`` so the codec can put a DIRECT loss on it; the default False keeps
        the pre-2026-09-04 single-tensor contract for every existing caller.
        """
        if self.gain_shape:
            recon, gain_pred = self._forward_gain_shape(quant)
            return (recon, gain_pred) if return_aux else recon
        x = self._shape_branch(quant)
        return (x, None) if return_aux else x

    def _shape_branch(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n_tok, d)`` -> ``(B, C, F, T)``: transformer + unpatchify + refine + noise.

        This is the ENTIRE pre-gain-shape decoder body, factored out unchanged so both paths
        share it bit-for-bit (the gain-shape path feeds it the un-mixed shape tokens).
        """
        cfg = self.cfg
        h = self.transformer(tokens + self.pos_emb())
        patches = self.to_pixels(h)
        # unpatchify: inverse of the encoder rearrange (g=1 -> identical to before).
        g = int(getattr(cfg, "channel_groups", 1))
        x = rearrange(
            patches,
            "b (g nf nt) (gc pf pt) -> b (g gc) (nf pf) (nt pt)",
            g=g,
            nf=cfg.n_freq_patch,
            nt=cfg.n_time_patch,
            gc=cfg.channels // g,
            pf=cfg.patch_f,
            pt=cfg.patch_t,
        )
        if self.refine is not None:
            x = x + self.refine(x)
        if self.noise_scale is not None:
            x = x + self.noise_scale.view(1, -1, 1, 1) * torch.randn_like(x)
        return x

    def _forward_gain_shape(self, quant: torch.Tensor):
        """``recon = level_hat + sigma_hat * unit_shape`` — the STRUCTURAL half of the split.

        Two guarantees are enforced HERE rather than hoped for from the loss (both pinned by
        tests/test_spectro_gain_shape.py):

        1. The shape branch's output is MEAN-REMOVED along the TIME axis, so it cannot carry
           any envelope. The reconstruction's per-(window, channel, frequency) time-mean is
           therefore exactly ``level_hat``, produced by the gain head from the gain tokens
           alone. Zero the shape path and the codec degenerates EXACTLY to a quantized
           per-(channel, frequency) envelope coder.
        2. When ``gain_scale`` is on the shape is also STD-NORMALIZED along time, so the
           reconstruction's temporal std IS the transmitted ``sigma_hat`` and NOT an
           nRMSE-minimising conditional mean. A flat plate is then unrepresentable: this is
           the structural attack on std_ratio 0.15-0.28 / hf_ratio 0.008.

        NOTE the refinement head and the noise injection run INSIDE the shape branch, i.e.
        BEFORE the mean-removal and normalization. Applying them to the assembled
        reconstruction instead would let them re-introduce DC and rescale the amplitude,
        destroying both guarantees — the patch-lattice fix and the amplitude guarantee are
        compatible only in this order.
        """
        cfg = self.cfg
        n_g = cfg.n_gain_tok
        # gain: per-patch MLP -> (B, n_gain_tok, G*gain_patch_f) -> (B, G, F)
        g_out = self.gain_head(quant[:, :n_g])
        gain_pred = rearrange(g_out, "b ng (g pf) -> b g (ng pf)", pf=cfg.gain_patch_f)
        level_hat = gain_pred[:, : cfg.channels]                        # (B, C, F)
        h = self.shape_unmix(quant[:, n_g:].transpose(1, 2)).transpose(1, 2)
        shape = self._shape_branch(h)                                   # (B, C, F, T)
        shape = shape - shape.mean(dim=-1, keepdim=True)                # (1) carries NO level
        if cfg.uses_gain_scale:
            shape = shape / shape.std(dim=-1, unbiased=False,
                                      keepdim=True).clamp_min(_GAIN_EPS)
            sigma_hat = torch.expm1(gain_pred[:, cfg.channels:].clamp(0.0, 30.0))
            recon = level_hat.unsqueeze(-1) + sigma_hat.unsqueeze(-1) * shape   # (2)
        else:
            recon = level_hat.unsqueeze(-1) + shape
        return recon, gain_pred


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
        # Channel floor for the halving schedule. Configurable so the decoder can be sized to
        # the NVIDIA Spectral Codec's 5.5:1 decoder:encoder ratio (arXiv 2406.05298 section 4:
        # "HiFi-GAN V1 decoder with upsample rates [8,8,4,2] and 1024 initial channels",
        # 55 M decoder vs 10 M encoder). Default 64 = the prior hard-coded value.
        min_ch = int(getattr(cfg, "conv_dec_min_ch", 64))

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
        if h.shape[-2] != cfg.eff_freq_bins or h.shape[-1] != cfg.time_frames:
            h = F.interpolate(
                h, size=(cfg.eff_freq_bins, cfg.time_frames),
                mode="bilinear", align_corners=False,
            )
        return self.to_out(h)
