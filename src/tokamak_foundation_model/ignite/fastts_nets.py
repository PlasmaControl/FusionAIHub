"""Phase-A fast-TS (filterscopes) encoder / decoder over the RAW 10 kHz SAMPLES.

2026-09-03 REDESIGN. This module used to tokenize a 5-bin ELM ACTIVITY ENVELOPE. It now
tokenizes the RAW waveform itself: ``(B, C, W)`` with ``C = 8`` filterscope channels and
``W = 500`` raw samples (50 ms at ``FASTTS_FS`` = 10 kHz), and reconstructs the SAME array
sample by sample. Nothing is pooled, rectified or log-compressed anywhere in the path.

Path (encoder)
--------------
    (B, C, W)
      -> PRE-PATCH CONV STEM at the full 10 kHz rate  (B, stem_ch, W)
      -> non-overlapping patchify over the SAMPLE axis, patch = ``cfg.patch_w`` samples
      -> linear to d_model + learned per-patch position embedding
      -> bidirectional ``x_transformers.Encoder``            (B, n_tok, d_model)

Path (decoder) mirrors it back through a conv head to ``(B, C, W)``.

WHY A CONV STEM (and not a spike-weighted loss)
-----------------------------------------------
The recorded finding for this modality is that loss-weighting for fast-TS spikes is a DEAD
END, and that the alternative to try is a **pre-patch conv stem or overlapping patches**. The
stem is both at once: it runs at stride 1 with an odd kernel (default 15) and 'same' padding,
so every stem output sample mixes +-7 raw samples and therefore each ``patch_w``-sample patch
sees ~``stem_kernel//2 * stem_layers`` samples PAST its own boundary on each side. A spike
sitting on a patch seam is seen whole by both neighbours instead of being split by a hard
patch edge. ``cfg.stem_layers = 0`` disables the stem and reproduces the plain
linear-patch path (the control arm).

Only external libs (torch, x_transformers, einops) + config.py are used; **no** FAITH model
code is imported (docs/IGNITE_DESIGN.md 7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder

from .config import FastTSCodecConfig


class _EnvPatchPosEmb(nn.Module):
    """ENVELOPE-mode position embedding. Parameter name ``env_pe`` — DO NOT RENAME.

    Every fast-TS checkpoint written before 2026-09-03 stores ``encoder.pos_emb.env_pe`` /
    ``decoder.pos_emb.env_pe``; a strict ``load_state_dict`` fails the moment this name moves.
    """

    def __init__(self, cfg: FastTSCodecConfig) -> None:
        super().__init__()
        self.n_e = cfg.n_env_patch
        self.env_pe = nn.Parameter(torch.zeros(self.n_e, cfg.d_model))
        nn.init.normal_(self.env_pe, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.env_pe.unsqueeze(0)  # (1, n_tok, d_model)


class _PatchPosEmb(nn.Module):
    """RAW-mode additive learned position embedding over the raw-sample patches.

    Token order is the sample-patch index along the single time axis. Returns
    ``(1, n_tok, d_model)`` broadcastable.
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

    Stride 1 keeps the FULL 10 kHz rate, which is the point: the receptive-field overlap
    between neighbouring patches is what removes the hard patch seam.
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
    """Split an ENVELOPE window into (gain, unit shape) — the analytic half of gain-shape VQ.

    ``x`` is ``(B, C, E)``. Returns ``(gain, shape, level, sigma)`` with

        level  (B, C)      mean over the E envelope bins        -- 96.2% of the variance
        sigma  (B, C)      std over E of the mean-removed rest  -- the within-window AMPLITUDE
        shape  (B, C, E)   (x - level) / sigma   (zero mean, unit std) when ``use_scale``,
                           else just ``x - level`` (zero mean, free scale)
        gain   (B, C or 2C)  ``[level]`` or ``[level, log1p(sigma)]`` -- the gain encoder's input

    ``log1p`` on sigma because the envelope is already a log1p-compressed RMS: sigma spans
    ~1e-3 to ~10 across windows, and the log makes that an O(1) regression target.
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
    """``(B, C, E)`` envelope (DEFAULT) or ``(B, C, W)`` raw samples -> ``(B, n_tok, d_model)``.

    ``cfg.target`` selects the mode. ENVELOPE mode is byte-identical to the pre-2026-09-03
    build — same submodule names (``to_tokens``, ``pos_emb.env_pe``), same shapes, no conv
    stem — so old checkpoints load strictly. RAW mode adds the pre-patch conv stem.
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
        # --- GAIN-SHAPE branch (envelope mode, opt-in). Built ONLY when enabled, so an
        # unchanged config has exactly the pre-2026-09-03 parameter set and old checkpoints
        # load strictly. The gain tokens BYPASS the transformer: the split is meant to be
        # hard, and self-attention between gain and shape tokens would let the shape path
        # leak back into the level code (which is the failure gain-shape exists to remove).
        self.gain_shape = bool(cfg.n_gain_tok > 0)
        if self.gain_shape:
            self.gain_to_tokens = _gain_mlp(
                cfg.gain_values, cfg.gain_hidden, cfg.n_gain_tok * cfg.d_model)
            # 5 shape patches -> n_shape_tok tokens, a learned mix along the TOKEN axis.
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
    """``(B, n_tok, d_model)`` -> ``(B, C, E)`` envelope (DEFAULT) or ``(B, C, W)`` raw samples.

    ENVELOPE mode is byte-identical to the pre-2026-09-03 build: a transformer over the
    quantized tokens, a linear ``to_pixels`` projection and a parameter-free unpatchify.
    ``to_pixels`` / ``pos_emb.env_pe`` are the names every existing checkpoint stores.

    RAW mode replaces ``to_pixels`` with ``to_features`` + a stride-1 conv ``head`` at the full
    10 kHz rate, so the reconstruction is continuous ACROSS patch boundaries instead of
    showing a seam every ``patch_w`` samples.
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
                # The gain HEAD is the whole reason the 0.1919 per-window-mean anchor becomes a
                # FLOOR: it, and nothing else, produces the reconstruction's per-(window,
                # channel) mean (the shape branch's output is mean-removed below).
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

        Envelope mode: ``to_pixels.weight`` (unchanged). Raw mode: the last ``Conv1d`` of the
        head, or ``to_features.weight`` when the stem is disabled. The VQGAN adaptive
        adversarial weight balances reconstruction against adversarial gradients here
        ("Taming Transformers" 3.3), exactly as ``nets.SpectroDecoder.last_layer`` does.
        """
        if not self.raw:
            if self.gain_shape and self.cfg.n_shape_tok == 0:
                # LEVEL-ONLY codec: to_pixels is unused, the gain MLP is the output layer.
                return [m for m in self.gain_head.modules() if isinstance(m, nn.Linear)][-1].weight
            return self.to_pixels.weight
        if isinstance(self.head, nn.Identity):
            return self.to_features.weight
        return [m for m in self.head.modules() if isinstance(m, nn.Conv1d)][-1].weight

    def forward(self, quant: torch.Tensor, return_aux: bool = False):
        """Quantized tokens -> reconstruction.

        ``return_aux`` (gain-shape only) additionally returns the decoded gain
        ``(B, C or 2C)`` so the codec can put a DIRECT loss on it; the default False keeps the
        pre-2026-09-03 single-tensor contract for every existing caller.
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
        """``recon = level_hat + sigma_hat * unit_shape`` — the STRUCTURAL half of gain-shape.

        Two guarantees are enforced here rather than hoped for from the loss:

        1. The shape branch's output is MEAN-REMOVED along the bin axis, so it cannot carry
           any DC level. The reconstruction's per-(window, channel) mean is therefore exactly
           ``level_hat``, produced by the gain head from the gain tokens alone. Zero the shape
           path and the codec degenerates EXACTLY to the quantised-window-mean encoder whose
           measured pooled nRMSE is 0.1919 — i.e. that number is a floor, not a bar.
        2. When ``gain_scale`` is on the shape is also STD-NORMALIZED, so the within-window
           amplitude is set by the transmitted ``sigma_hat`` and not by an nRMSE-minimising
           conditional mean. This is the structural attack on the 36% burst-height retention.
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
        shape = shape - shape.mean(dim=-1, keepdim=True)          # (1) carries NO level
        if cfg.uses_gain_scale:
            shape = shape / shape.std(dim=-1, unbiased=False, keepdim=True).clamp_min(_GAIN_EPS)
            sigma_hat = torch.expm1(gain_pred[:, cfg.channels:].clamp(0.0, 30.0))
            recon = level_hat.unsqueeze(-1) + sigma_hat.unsqueeze(-1) * shape   # (2)
        else:
            recon = level_hat.unsqueeze(-1) + shape
        return recon, gain_pred
