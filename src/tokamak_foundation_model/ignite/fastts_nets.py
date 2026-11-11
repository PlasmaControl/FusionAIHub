"""Phase-A fast-TS (filterscopes) encoder and decoder.

``cfg.target`` selects the mode. Envelope mode (the default) tokenizes a 5-bin ELM
activity envelope ``(B, C, E)``. Raw mode tokenizes the waveform ``(B, C, W)``, with
``C = 8`` channels and ``W = 500`` samples (50 ms at ``FASTTS_FS`` = 10 kHz), and
reconstructs the same array sample by sample, with no pooling, rectification or log
compression.

Raw-mode encoder
----------------
    (B, C, W)
      -> conv stem at the full 10 kHz rate                (B, stem_ch, W)
      -> non-overlapping patches along time, ``cfg.patch_w`` samples each
      -> linear to d_model + learned per-patch position embedding
      -> bidirectional ``x_transformers.Encoder``         (B, n_tok, d_model)

The decoder mirrors this back through a conv head to ``(B, C, W)``.

Why a conv stem
---------------
Weighting the loss toward spikes did not work for this modality; a conv stem before
patching (or overlapping patches) was the alternative. The stem is both: it runs at
stride 1 with an odd kernel (default 15) and 'same' padding, so each output sample mixes
+-7 raw samples and each patch sees about ``stem_kernel//2 * stem_layers`` samples past
its boundary on either side. A spike on a patch boundary is seen whole by both
neighbours. ``cfg.stem_layers = 0`` disables the stem (the plain linear-patch control).

Uses only torch, x_transformers, einops and config.py; no FAITH model code
(docs/IGNITE_DESIGN.md §7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import FastTSCodecConfig


class _EnvPatchPosEmb(nn.Module):
    """Envelope-mode position embedding.

    Do not rename ``env_pe``: envelope checkpoints store ``encoder.pos_emb.env_pe`` and
    ``decoder.pos_emb.env_pe``, and a strict ``load_state_dict`` fails if the name
    changes.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.n_e = cfg.n_env_patch
        self.env_pe = nn.Parameter(torch.zeros(self.n_e, cfg.d_model))
        nn.init.normal_(self.env_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.env_pe.unsqueeze(0)  # (1, n_tok, d_model)


class _PatchPosEmb(nn.Module):
    """Raw-mode additive learned position embedding over the sample patches.

    Tokens are ordered by patch index along time. Returns a broadcastable
    ``(1, n_tok, d_model)``.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.n_tok = cfg.n_tok
        self.pe = nn.Parameter(torch.zeros(self.n_tok, cfg.d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.pe.unsqueeze(0)  # (1, n_tok, d_model)


def _conv_stem(in_ch: int, out_ch: int, kernel: int, layers: int) -> nn.Module:
    """``layers`` stride-1 'same'-padded Conv1d + GELU blocks (identity when ``layers == 0``).

    Stride 1 keeps the full 10 kHz rate, so receptive fields of neighbouring patches
    overlap and there is no hard patch boundary.
    """
    if layers <= 0:
        return nn.Identity()
    pad = kernel // 2
    mods: list[nn.Module] = []
    ch = in_ch
    for _ in range(layers):
        mods.append(nn.Conv1d(ch, out_ch, kernel_size=kernel, stride=1, padding=pad))
        mods.append(nn.GELU())
        ch = out_ch
    return nn.Sequential(*mods)


_GAIN_EPS: float = 1e-6


def gain_shape_split(x: torch.Tensor, use_scale: bool):
    """Split an envelope window into gain and unit shape (analytic half of gain-shape).

    ``x`` is ``(B, C, E)``. Returns ``(gain, shape, level, sigma)`` with

        level  (B, C)      mean over the E envelope bins (96% of the variance)
        sigma  (B, C)      std over E of the mean-removed rest (within-window amplitude)
        shape  (B, C, E)   (x - level) / sigma (zero mean, unit std) when ``use_scale``,
                           else ``x - level`` (zero mean, free scale)
        gain   (B, C or 2C)  ``[level]`` or ``[level, log1p(sigma)]``, the gain input

    sigma goes through ``log1p`` because the envelope is already a log1p-compressed RMS:
    sigma spans ~1e-3 to ~10 across windows, and the log makes it an O(1) target.
    """
    level = x.mean(dim=-1)                                      # (B, C)
    res = x - level.unsqueeze(-1)
    sigma = res.std(dim=-1, unbiased=False)                     # (B, C)
    if use_scale:
        shape = res / sigma.clamp_min(_GAIN_EPS).unsqueeze(-1)
        gain = torch.cat([level, torch.log1p(sigma.clamp_min(0.0))], dim=-1)
    else:
        shape = res
        gain = level
    return gain, shape, level, sigma


def _gain_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class FastTSEncoder(nn.Module):
    """``(B, C, E)`` envelope or ``(B, C, W)`` samples -> ``(B, n_tok, d_model)``.

    ``cfg.target`` selects the mode. Envelope mode keeps the original submodule names
    (``to_tokens``, ``pos_emb.env_pe``) and shapes and has no conv stem, so old
    checkpoints load strictly. Raw mode adds the conv stem.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.raw = cfg.is_raw
        if self.raw:
            self.stem = _conv_stem(cfg.channels, cfg.stem_channels,
                                   cfg.stem_kernel, cfg.stem_layers)
            feat_ch = cfg.stem_channels if cfg.stem_layers > 0 else cfg.channels
            patch_dim = feat_ch * cfg.patch_w
            self.pos_emb = _PatchPosEmb(cfg)
        else:
            feat_ch = cfg.channels
            patch_dim = cfg.channels * cfg.patch_e
            self.pos_emb = _EnvPatchPosEmb(cfg)
        self.feat_ch = feat_ch
        self.to_tokens = nn.Linear(patch_dim, cfg.d_model)
        self.transformer = Encoder(dim=cfg.d_model, depth=cfg.enc_depth, heads=cfg.heads)
        # Gain-shape branch (envelope mode, opt-in). Built only when enabled, so other
        # configs keep the original parameter set. The gain tokens bypass the
        # transformer: self-attention between gain and shape tokens would let the shape
        # path leak into the level code, which is what the split is meant to prevent.
        self.gain_shape = bool(cfg.n_gain_tok > 0)
        if self.gain_shape:
            self.gain_to_tokens = _gain_mlp(
                cfg.gain_values, cfg.gain_hidden, cfg.n_gain_tok * cfg.d_model)
            # 5 shape patches -> n_shape_tok tokens, a learned mix along the token axis.
            self.shape_mix = (nn.Linear(cfg.n_env_patch, cfg.n_shape_tok)
                              if cfg.n_shape_tok > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if not self.raw:
            if not self.gain_shape:
                patches = rearrange(x, "b c (ne pe) -> b ne (c pe)", pe=cfg.patch_e)
                return self.transformer(self.to_tokens(patches) + self.pos_emb())
            gain, shape, _, _ = gain_shape_split(x, cfg.uses_gain_scale)
            g_tok = self.gain_to_tokens(gain).reshape(
                x.shape[0], cfg.n_gain_tok, cfg.d_model)          # (B, n_gain, d)
            if cfg.n_shape_tok == 0:
                return g_tok + self.pos_emb()
            patches = rearrange(shape, "b c (ne pe) -> b ne (c pe)", pe=cfg.patch_e)
            s_tok = self.transformer(self.to_tokens(patches))     # (B, n_env_patch, d)
            s_tok = self.shape_mix(s_tok.transpose(1, 2)).transpose(1, 2)
            return torch.cat([g_tok, s_tok], dim=1) + self.pos_emb()
        if x.dim() != 3 or x.shape[-1] != cfg.window:
            raise ValueError(
                f"FastTSEncoder expects (B, {cfg.channels}, {cfg.window}); got {tuple(x.shape)}"
            )
        h = self.stem(x)                                       # (B, feat_ch, W), full rate
        patches = rearrange(h, "b d (n p) -> b n (d p)", p=cfg.patch_w)
        return self.transformer(self.to_tokens(patches) + self.pos_emb())


class FastTSDecoder(nn.Module):
    """``(B, n_tok, d_model)`` -> ``(B, C, E)`` envelope or ``(B, C, W)`` samples.

    Envelope mode is a transformer over the quantized tokens, a linear ``to_pixels``
    projection and a parameter-free unpatchify. Checkpoints store the names
    ``to_pixels`` and ``pos_emb.env_pe``.

    Raw mode replaces ``to_pixels`` with ``to_features`` and a stride-1 conv ``head`` at
    the full 10 kHz rate, so the reconstruction is continuous across patch boundaries
    instead of showing a seam every ``patch_w`` samples.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.raw = cfg.is_raw
        self.transformer = Encoder(dim=cfg.d_model, depth=cfg.dec_depth, heads=cfg.heads)
        self.gain_shape = bool(cfg.n_gain_tok > 0)
        if not self.raw:
            self.pos_emb = _EnvPatchPosEmb(cfg)
            self.to_pixels = nn.Linear(cfg.d_model, cfg.channels * cfg.patch_e)
            if self.gain_shape:
                # The gain head alone produces the reconstruction's per-(window,
                # channel) mean (the shape output is mean-removed), which makes the
                # quantized window-mean encoder a floor on quality.
                self.gain_head = _gain_mlp(
                    cfg.n_gain_tok * cfg.d_model, cfg.gain_hidden, cfg.gain_values)
                self.shape_unmix = (nn.Linear(cfg.n_shape_tok, cfg.n_env_patch)
                                    if cfg.n_shape_tok > 0 else None)
            return
        self.pos_emb = _PatchPosEmb(cfg)
        feat_ch = cfg.stem_channels if cfg.stem_layers > 0 else cfg.channels
        self.feat_ch = feat_ch
        self.to_features = nn.Linear(cfg.d_model, feat_ch * cfg.patch_w)
        if cfg.stem_layers > 0:
            pad = cfg.stem_kernel // 2
            head: list[nn.Module] = []
            for _ in range(max(0, cfg.stem_layers - 1)):
                head.append(nn.Conv1d(feat_ch, feat_ch, cfg.stem_kernel, 1, pad))
                head.append(nn.GELU())
            head.append(nn.Conv1d(feat_ch, cfg.channels, cfg.stem_kernel, 1, pad))
            self.head = nn.Sequential(*head)
        else:
            self.head = nn.Identity()

    @property
    def last_layer(self) -> nn.Parameter:
        """The weight ``Parameter`` of the final layer producing the output.

        Envelope mode: ``to_pixels.weight``. Raw mode: the last ``Conv1d`` of the head,
        or ``to_features.weight`` when the stem is disabled. The VQGAN adaptive
        adversarial weight compares reconstruction and adversarial gradients here
        ("Taming Transformers" 3.3), as with ``nets.SpectroDecoder.last_layer``.
        """
        if not self.raw:
            if self.gain_shape and self.cfg.n_shape_tok == 0:
                # Level-only codec: to_pixels is unused and the gain MLP is the output.
                return [m for m in self.gain_head.modules() if isinstance(m, nn.Linear)][-1].weight
            return self.to_pixels.weight
        if isinstance(self.head, nn.Identity):
            return self.to_features.weight
        return [m for m in self.head.modules() if isinstance(m, nn.Conv1d)][-1].weight

    def forward(self, quant: torch.Tensor, return_aux: bool = False):
        """Quantized tokens -> reconstruction.

        With ``return_aux`` it returns ``(recon, gain)``, where ``gain`` is the decoded
        ``(B, C or 2C)`` gain in gain-shape mode (for a direct loss) and None otherwise.
        The default returns the reconstruction alone.
        """
        cfg = self.cfg
        if not self.raw and self.gain_shape:
            recon, gain_pred = self._forward_gain_shape(quant)
            return (recon, gain_pred) if return_aux else recon
        h = self.transformer(quant + self.pos_emb())
        if not self.raw:
            out = rearrange(self.to_pixels(h), "b ne (c pe) -> b c (ne pe)",
                            ne=cfg.n_env_patch, c=cfg.channels, pe=cfg.patch_e)
            return (out, None) if return_aux else out
        sig = rearrange(self.to_features(h), "b n (d p) -> b d (n p)",
                        d=self.feat_ch, p=cfg.patch_w)
        out = self.head(sig)                                   # (B, C, W)
        return (out, None) if return_aux else out

    def _forward_gain_shape(self, quant: torch.Tensor):
        """``recon = level_hat + sigma_hat * unit_shape``, the structural split.

        Two properties are enforced here rather than left to the loss:

        1. The shape output is mean-removed along the bin axis, so it carries no DC
           level and the reconstruction's per-(window, channel) mean is exactly
           ``level_hat``, from the gain tokens alone. With the shape path zeroed the
           codec is the quantized window-mean encoder (nRMSE ~0.19), so that score is a
           floor.
        2. With ``gain_scale`` the shape is also std-normalized, so the within-window
           amplitude is the transmitted ``sigma_hat`` rather than a shrunken
           nRMSE-optimal conditional mean.
        """
        cfg = self.cfg
        n_g = cfg.n_gain_tok
        gain_pred = self.gain_head(quant[:, :n_g].reshape(quant.shape[0], -1))
        level_hat = gain_pred[:, : cfg.channels]
        if cfg.n_shape_tok == 0:
            return level_hat.unsqueeze(-1).expand(-1, -1, cfg.env_bins).contiguous(), gain_pred
        h = self.shape_unmix(quant[:, n_g:].transpose(1, 2)).transpose(1, 2)
        h = self.transformer(h + self.pos_emb())
        shape = rearrange(self.to_pixels(h), "b ne (c pe) -> b c (ne pe)",
                          ne=cfg.n_env_patch, c=cfg.channels, pe=cfg.patch_e)
        shape = shape - shape.mean(dim=-1, keepdim=True)          # (1) no level
        if cfg.uses_gain_scale:
            shape = shape / shape.std(dim=-1, unbiased=False, keepdim=True).clamp_min(_GAIN_EPS)
            sigma_hat = torch.expm1(gain_pred[:, cfg.channels:].clamp(0.0, 30.0))
            recon = level_hat.unsqueeze(-1) + sigma_hat.unsqueeze(-1) * shape   # (2)
        else:
            recon = level_hat.unsqueeze(-1) + shape
        return recon, gain_pred
