"""Spectrogram encoder and decoder.

Both are ``x_transformers.Encoder`` stacks without a causal mask, since a codec frame is
a bidirectional token set. The spectrogram ``(B, C, F, T)`` is cut into non-overlapping
``(patch_f, patch_t)`` patches, embedded linearly to ``d_model`` and given learned
frequency and time position embeddings. The decoder runs a transformer over the
quantized tokens and unpatchifies linearly back to ``(B, C, F, T)``. No FAITH model code
is used.
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
    """Learned additive position embedding, factorized over frequency and time patches.

    Token order is ``(freq_patch, time_patch)``, frequency outer, matching the patchify
    below. Returns ``(1, n_tok, d_model)``.

    With ``cfg.channel_groups > 1`` (channel-factorized tokens) a learned per-group
    embedding is added and the order becomes ``(group, freq_patch, time_patch)``. At the
    default of 1 no group parameter exists, so old state_dicts still load.
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

    The spectrogram version of ``fastts_nets.gain_shape_split``. Statistics are taken
    per (channel, frequency bin) along time rather than per channel along the envelope
    bins, because a mode is a coherent track at one frequency and its amplitude in time
    is ``sigma`` below.

    ``x`` is ``(B, C, F, T)``. Returns ``(gain, shape, level, sigma)``::

        level (B, C, F)      x.mean(-1)                      static envelope
        sigma (B, C, F)      (x - level).std(-1)             temporal amplitude per bin
        shape (B, C, F, T)   (x - level) / sigma  when ``use_scale`` (zero mean, unit std
                             along T), else ``x - level`` (zero mean, free scale)
        gain  (B, 2C or C, F)  ``[level ; log1p(sigma)]``, the gain encoder's input

    sigma is non-negative and spans orders of magnitude between quiet and active bins.
    ``log1p`` makes it an O(1) regression target; ``expm1`` in the decoder inverts it.
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
    """Per-patch 3-layer MLP, shared across gain tokens (acts on the last axis)."""
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
        # Envelope/shape split (cfg.gain_shape, see SpectroCodecConfig). Built only when
        # enabled, so other configs keep their parameter set and old checkpoints load
        # strictly. The gain tokens bypass the transformer: attention between gain and
        # shape tokens would let the shape path leak back into the envelope code, which
        # the split is there to prevent.
        self.gain_shape = cfg.n_gain_tok > 0
        if self.gain_shape:
            self.gain_to_tokens = _gain_mlp(
                cfg.gain_values_per_tok, cfg.gain_hidden, cfg.d_model)
            # n_tok shape patches -> n_shape_tok tokens by a learned mix along the token
            # axis, so the patch geometry and the decoder stay the same.
            self.shape_mix = nn.Linear(cfg.n_tok, cfg.n_shape_tok)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if self.gain_shape:
            gain, x = spectro_gain_shape_split(x, cfg.uses_gain_scale)[:2]
            # gain (B, G, F) -> one token per contiguous frequency band, all channels.
            g_patch = rearrange(gain, "b g (ng pf) -> b ng (g pf)", pf=cfg.gain_patch_f)
            g_tok = self.gain_to_tokens(g_patch)                # (B, n_gain_tok, d)
        # Patchify: group outer, then frequency, time inner; n_tok = g * n_f * n_t.
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
        # Optional residual conv refinement over the assembled (F, T) spectrogram
        # (cfg.refine_depth > 0): stride-1, kernel-3 convs that give the decoder
        # cross-patch context, so its texture is not one basis tiled on the (patch_f,
        # patch_t) lattice (the checkerboard measured by gate.patch_lattice_metrics).
        # The last conv is zero-initialised, so the head starts as the identity. Depth
        # 0, and old pickled configs without the field (hence getattr), build nothing
        # and keep the old state_dict.
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
        # Optional StyleGAN-style per-pixel noise with a learned per-channel scale
        # (cfg.decoder_noise), zero-initialised. A deterministic decoder can only
        # satisfy the adversarial term by tiling a fixed texture (see
        # SpectroCodecConfig). Applied at full resolution after the refinement head, in
        # training and eval, so the decoded spectrogram is a sample, not a conditional
        # mean.
        self.noise_scale: nn.Parameter | None = (
            nn.Parameter(torch.zeros(cfg.channels))
            if bool(getattr(cfg, "decoder_noise", False)) else None
        )
        # Envelope/shape split heads. `gain_head` alone sets the reconstruction's
        # per-(channel, frequency) time mean and temporal std: the shape branch is
        # mean-removed and (with gain_scale) std-normalized in `_forward_gain_shape`, so
        # it can carry neither, and a flat plate cannot be represented.
        self.gain_shape = cfg.n_gain_tok > 0
        if self.gain_shape:
            self.gain_head = _gain_mlp(
                cfg.d_model, cfg.gain_hidden, cfg.gain_values_per_tok)
            self.shape_unmix = nn.Linear(cfg.n_shape_tok, cfg.n_tok)

    @property
    def last_layer(self) -> nn.Parameter:
        """Weight of the last parameterized layer on the output path.

        ``to_pixels.weight`` without the refinement head, otherwise the last refinement
        conv's weight. The VQGAN adaptive adversarial weight balances the reconstruction
        and adversarial gradients at this tensor
        (``codec.SpectroCodec.generator_losses``, "Taming Transformers" §3.3), so it has
        to be the last layer before the output.
        """
        if self.refine is not None:
            return self.refine[-1].weight
        return self.to_pixels.weight

    def forward(self, quant: torch.Tensor, return_aux: bool = False):
        """Quantized tokens -> reconstruction.

        With ``return_aux`` (gain-shape only) the decoded gain ``(B, 2C or C, F)`` is
        returned too, so the codec can put a loss on it directly.
        """
        if self.gain_shape:
            recon, gain_pred = self._forward_gain_shape(quant)
            return (recon, gain_pred) if return_aux else recon
        x = self._shape_branch(quant)
        return (x, None) if return_aux else x

    def _shape_branch(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n_tok, d)`` -> ``(B, C, F, T)``: transformer, unpatchify, refine, noise.

        The whole decoder body without the gain-shape split; the gain-shape path feeds
        it the unmixed shape tokens.
        """
        cfg = self.cfg
        h = self.transformer(tokens + self.pos_emb())
        patches = self.to_pixels(h)
        # Inverse of the encoder patchify.
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
        """``recon = level_hat + sigma_hat * unit_shape``.

        Two properties are enforced here rather than left to the loss (tested in
        tests/test_spectro_gain_shape.py):

        1. The shape branch output has its time mean removed, so it carries no envelope.
           The reconstruction's per-(window, channel, frequency) time mean is exactly
           ``level_hat``, from the gain tokens alone. With the shape path zeroed the
           codec becomes a quantized per-(channel, frequency) envelope coder.
        2. With ``gain_scale`` the shape is also normalized to unit std along time, so
           the reconstruction's temporal std is the transmitted ``sigma_hat`` rather
           than an nRMSE-minimising conditional mean. A flat plate cannot be
           represented; this targets the observed std_ratio 0.15-0.28 and hf_ratio
           0.008.

        The refinement head and the noise run inside the shape branch, before the mean
        removal and normalization. On the assembled reconstruction they could add DC and
        rescale the amplitude, breaking both properties.
        """
        cfg = self.cfg
        n_g = cfg.n_gain_tok
        # Gain: per-patch MLP -> (B, n_gain_tok, G*gain_patch_f) -> (B, G, F)
        g_out = self.gain_head(quant[:, :n_g])
        gain_pred = rearrange(g_out, "b ng (g pf) -> b g (ng pf)", pf=cfg.gain_patch_f)
        level_hat = gain_pred[:, : cfg.channels]                        # (B, C, F)
        h = self.shape_unmix(quant[:, n_g:].transpose(1, 2)).transpose(1, 2)
        shape = self._shape_branch(h)                                   # (B, C, F, T)
        shape = shape - shape.mean(dim=-1, keepdim=True)                # (1) no level
        if cfg.uses_gain_scale:
            shape = shape / shape.std(dim=-1, unbiased=False,
                                      keepdim=True).clamp_min(_GAIN_EPS)
            sigma_hat = torch.expm1(gain_pred[:, cfg.channels:].clamp(0.0, 30.0))
            recon = level_hat.unsqueeze(-1) + sigma_hat.unsqueeze(-1) * shape   # (2)
        else:
            recon = level_hat.unsqueeze(-1) + shape
        return recon, gain_pred


class _ResBlock2d(nn.Module):
    """Pre-activation residual conv block (2D), as in VQGAN and HiFi-GAN.

    ``Conv2d -> LeakyReLU -> Conv2d`` with a skip connection, 3x3 same padding so the
    size is kept. It adds local texture to the upsampled feature map, which a linear
    ``to_pixels`` cannot. The skip uses a 1x1 projection when ``in_ch != out_ch``.
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
    """(B, n_tok, d_model) -> (B, C, F, T) with a HiFi-GAN/VQGAN-style 2D conv upsampler.

    Alternative to :class:`SpectroDecoder` with the same ``forward(quant)`` and
    ``last_layer``. The linear ``to_pixels`` can only produce smooth patches. Here each
    FSQ token is placed on a coarse ``(n_freq_patch, n_time_patch)`` grid and transposed
    conv and residual blocks synthesize the fine turbulent texture (bes/mhr/ece
    broadband).

    Structure (2D version of the NeMo HiFi-GAN generator; cf. the VQGAN decoder)::

        tokens (B, n_tok, d)                        frequency outer, time inner
          -> feature map (B, d, n_freq_patch, n_time_patch)
          -> proj_in: 1x1 Conv2d  d_model -> base_ch
          -> [ ConvTranspose2d(stride=(sf, st))     x2 per axis while needed
               + LeakyReLU
               + res_blocks x _ResBlock2d ] x n_up  channels halved every 2 blocks
          -> F.interpolate to (F, T) if a patch size is not a power of 2
          -> to_out: 3x3 Conv2d  ch -> C           no activation, (B, C, F, T)

    The upsample factors are ``patch_f`` and ``patch_t``. There are
    ``n_up = max(log2(patch_f), log2(patch_t))`` blocks, each doubling an axis only
    while it still needs it (stride 1 otherwise). Power-of-2 patch sizes (16, 32, 64)
    reach ``(F, T)`` exactly without interpolation.
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
        # Channel floor of the halving schedule. Configurable so the decoder can match
        # the NVIDIA Spectral Codec's 5.5:1 decoder:encoder size (arXiv 2406.05298 §4:
        # HiFi-GAN V1 decoder, upsample rates [8,8,4,2], 1024 initial channels, 55 M vs
        # 10 M parameters).
        min_ch = int(getattr(cfg, "conv_dec_min_ch", 64))

        # Number of x2 steps each axis needs to reach the patch size.
        self._pow2_f = (cfg.patch_f & (cfg.patch_f - 1)) == 0 and cfg.patch_f > 0
        self._pow2_t = (cfg.patch_t & (cfg.patch_t - 1)) == 0 and cfg.patch_t > 0
        n_freq_ups = int(round(log2(cfg.patch_f))) if self._pow2_f else 0
        n_time_ups = int(round(log2(cfg.patch_t))) if self._pow2_t else 0
        n_up = max(n_freq_ups, n_time_ups, 0)

        self.proj_in = nn.Conv2d(cfg.d_model, base_ch, kernel_size=1)

        blocks: list[nn.Module] = []
        ch = base_ch
        for i in range(n_up):
            sf = 2 if i < n_freq_ups else 1   # upsample frequency?
            st = 2 if i < n_time_ups else 1   # upsample time?
            # Halve channels every 2 blocks, down to min_ch.
            out_ch = max(min_ch, base_ch // (2 ** ((i // 2) + 1)))
            # ConvTranspose2d out = (in-1)*stride - 2*pad + kernel + output_pad:
            #   stride 2: kernel 4, pad 1 -> out = 2*in
            #   stride 1: kernel 3, pad 1 -> out = in
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

        # Final 3x3 conv to (B, C, F, T), no activation. This is `last_layer`.
        self.to_out = nn.Conv2d(ch, cfg.channels, kernel_size=3, padding=1)

    @property
    def last_layer(self) -> nn.Parameter:
        """Weight of the final conv, ``to_out.weight``.

        Counterpart of ``SpectroDecoder.last_layer``; the VQGAN adaptive adversarial
        weight balances the reconstruction and adversarial gradients here
        (``codec.SpectroCodec.generator_losses``, "Taming Transformers" §3.3).
        """
        return self.to_out.weight

    def forward(self, quant: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        # Tokens are frequency outer, time inner; undo only the (nf nt) flattening into
        # a (d, nf, nt) feature map.
        h = rearrange(
            quant,
            "b (nf nt) d -> b d nf nt",
            nf=cfg.n_freq_patch,
            nt=cfg.n_time_patch,
        )
        h = self.proj_in(h)
        for layer in self.up:
            h = layer(h)
        # Snap to (F, T) if a patch size was not a power of 2.
        if h.shape[-2] != cfg.eff_freq_bins or h.shape[-1] != cfg.time_frames:
            h = F.interpolate(
                h, size=(cfg.eff_freq_bins, cfg.time_frames),
                mode="bilinear", align_corners=False,
            )
        return self.to_out(h)
