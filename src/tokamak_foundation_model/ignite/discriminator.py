"""Frequency-aware multi-scale PatchGAN discriminator (IGNITE Phase A).

Fresh code (no FAITH model reuse). Uses only `torch.nn` + `einops` + `config.py`.

Design (see docs/IGNITE_DESIGN.md §4.2):
- The spectrogram frequency axis is NOT translation-invariant (a mode at 50 kHz is
  physically different from one at 150 kHz). A vanilla PatchGAN is fully convolutional
  and therefore freq-shift-equivariant, which is the wrong inductive bias. We break that
  symmetry by concatenating a *frequency positional-encoding* channel to the input.
- Multi-scale: the same freq-aware PatchGAN body is applied at a couple of input scales
  (full resolution + a freq/time-downsampled copy) so both fine mode texture and coarse
  band structure are judged.
- Hinge-ready: the heads emit RAW patch scores (no final sigmoid) so a hinge GAN loss can
  be applied directly.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectroCodecConfig


def _sinusoidal_freq_pe(freq_bins: int, n_channels: int, device, dtype) -> torch.Tensor:
    """Return a (n_channels, freq_bins) sinusoidal positional encoding over the freq axis.

    Deterministic, parameter-free. Constant along time (broadcast later).
    """
    pos = torch.arange(freq_bins, device=device, dtype=dtype).unsqueeze(1)  # (F, 1)
    i = torch.arange(n_channels, device=device, dtype=dtype).unsqueeze(0)   # (1, C)
    # frequency of each PE channel, geometric spacing
    div = torch.pow(
        torch.tensor(10_000.0, device=device, dtype=dtype),
        (2.0 * torch.floor(i / 2.0)) / max(n_channels, 1),
    )
    angles = pos / div  # (F, C)
    pe = torch.where(i.long() % 2 == 0, torch.sin(angles), torch.cos(angles))
    return pe.transpose(0, 1).contiguous()  # (C, F)


class _PatchGANBody(nn.Module):
    """A small conv PatchGAN producing a single raw score map from (B, C_in, F, T)."""

    def __init__(self, in_channels: int, base: int = 32, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        out = base
        for li in range(n_layers):
            stride = 2 if li < n_layers - 1 else 1
            layers.append(
                nn.Conv2d(ch, out, kernel_size=4, stride=stride, padding=1)
            )
            if li > 0:
                layers.append(nn.GroupNorm(num_groups=min(8, out), num_channels=out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = out
            out = min(out * 2, base * 4)
        # final 1x1-ish conv -> single-channel raw score map (no activation / no sigmoid)
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the body, capturing the intermediate conv activations before the final score conv.

        Returns ``(score, feats)`` where ``score`` is byte-identical to :meth:`forward`
        (the final-conv output) and ``feats`` is the list of activations emitted right after
        each LeakyReLU (i.e. every conv-block output that feeds forward), EXCLUDING the raw
        input and the final score map. These are the tensors matched by the generator's
        feature-matching loss (HiFi-GAN/MelGAN-style vocoder feature matching).
        """
        feats: List[torch.Tensor] = []
        h = x
        n = len(self.net)
        for i, layer in enumerate(self.net):
            h = layer(h)
            # collect after each activation (LeakyReLU), but never the final score conv output
            if isinstance(layer, nn.LeakyReLU) and i < n - 1:
                feats.append(h)
        return h, feats


class FreqAwarePatchGAN(nn.Module):
    """Multi-scale, frequency-aware PatchGAN discriminator.

    Forward: ``(B, C, F, T) -> list[Tensor]`` of raw patch-score maps (one per scale),
    each ``(B, 1, h, w)``. Hinge-ready (no final sigmoid).

    ``forward(x, return_features=True) -> (list[score], list[feat])`` additionally returns a
    flat list of intermediate conv activations (both scales' bodies), for the generator's
    feature-matching loss. The default ``forward(x)`` path is byte-identical to before.
    """

    def __init__(self, cfg: SpectroCodecConfig, base: int = 32, n_pe_channels: int = 4,
                 scales: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_pe_channels = n_pe_channels
        self.scales = scales
        in_ch = cfg.channels + n_pe_channels
        # one body per scale (independent judges of full vs downsampled input)
        self.bodies = nn.ModuleList(
            [_PatchGANBody(in_ch, base=base) for _ in range(scales)]
        )

    def freq_pe(self, x: torch.Tensor) -> torch.Tensor:
        """Build the freq positional-encoding channels for input ``x`` (B, C, F, T).

        Returns (B, n_pe_channels, F, T): constant across time, varying across freq.
        """
        b, _, freq_bins, t = x.shape
        pe = _sinusoidal_freq_pe(freq_bins, self.n_pe_channels, x.device, x.dtype)  # (Cpe, F)
        pe = pe.unsqueeze(0).unsqueeze(-1)  # (1, Cpe, F, 1)
        pe = pe.expand(b, self.n_pe_channels, freq_bins, t)
        return pe

    def _apply_body(self, body: _PatchGANBody, x: torch.Tensor) -> torch.Tensor:
        pe = self.freq_pe(x)
        xin = torch.cat([x, pe], dim=1)
        return body(xin)

    def _apply_body_features(
        self, body: _PatchGANBody, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        pe = self.freq_pe(x)
        xin = torch.cat([x, pe], dim=1)
        return body.forward_features(xin)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """``(B, C, F, T) -> list[score]``, or ``(list[score], list[feat])`` if requested.

        With ``return_features=False`` (default) the behavior is byte-identical to the prior
        implementation. With ``return_features=True`` the second element is a FLAT list of the
        intermediate conv activations across BOTH scales' bodies (each scale contributes the
        activations before its final score conv), which the feature-matching loss consumes.
        """
        maps: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        cur = x
        for si, body in enumerate(self.bodies):
            if si > 0:
                # downsample the RAW input, then re-attach a freq-PE at the new resolution
                cur = F.avg_pool2d(cur, kernel_size=2, ceil_mode=True)
            if return_features:
                score, body_feats = self._apply_body_features(body, cur)
                maps.append(score)
                feats.extend(body_feats)
            else:
                maps.append(self._apply_body(body, cur))
        if return_features:
            return maps, feats
        return maps


class MultiScaleSpectroGAN(nn.Module):
    """WHOLE-SPECTROGRAM multi-scale critic — the anti-patch-lattice discriminator.

    Why this exists. :class:`FreqAwarePatchGAN` emits a *map* of per-patch scores, so the
    generator can satisfy it by learning ONE convincing patch texture and tiling it. That is
    not a hypothetical: it is the measured artifact — ``patch_lattice_ratio`` 61.02 on the
    reconstruction against a ground-truth control of 1.14, with 92-94 % of the reconstruction's
    high-frequency energy sitting exactly on the ``(patch_f, patch_t)`` lattice, and the
    adversarial objective is what sustains it.

    This critic instead reduces the whole spectrogram to a SINGLE score per scale, evaluated at
    1x / 2x / 4x frequency-time downsampling. A tiled texture is globally implausible at every
    scale, so it cannot buy a high score. Structure mirrors the reference's design intent — a
    multi-scale critic that sees the signal globally rather than per patch (arXiv 2406.05298
    section 4 pairs a multi-period discriminator with a multi-scale complex-STFT one).

    Contract is IDENTICAL to ``FreqAwarePatchGAN`` so it is a drop-in: ``forward(x)`` returns a
    list of raw score tensors (one per scale, shape ``(B, 1, 1, 1)``), and
    ``forward(x, return_features=True)`` also returns the intermediate activations used by the
    feature-matching term. The frequency positional-encoding channel is kept — frequency is
    not translation-invariant here either.
    """

    def __init__(self, cfg: SpectroCodecConfig, base: int = 32, n_layers: int = 4,
                 pe_channels: int = 4, scales: Tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.cfg = cfg
        self.pe_channels = int(pe_channels)
        self.scales = tuple(int(s) for s in scales)
        in_ch = cfg.channels + self.pe_channels
        self.bodies = nn.ModuleList()
        self.heads = nn.ModuleList()
        for _s in self.scales:
            layers: List[nn.Module] = []
            ch, out = in_ch, base
            for li in range(n_layers):
                layers.append(nn.Conv2d(ch, out, kernel_size=4, stride=2, padding=1))
                if li > 0:
                    layers.append(nn.GroupNorm(num_groups=min(8, out), num_channels=out))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                ch, out = out, min(out * 2, 256)
            self.bodies.append(nn.Sequential(*layers))
            # GLOBAL score: 1x1 conv then mean over the whole map -> one number per sample.
            self.heads.append(nn.Conv2d(ch, 1, kernel_size=1))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        import torch.nn.functional as _F
        pe = _sinusoidal_freq_pe(x.shape[-2], self.pe_channels, x.device, x.dtype)
        scores: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        for si, s in enumerate(self.scales):
            z = x if s == 1 else _F.avg_pool2d(x, s)
            p = pe if s == 1 else _sinusoidal_freq_pe(
                z.shape[-2], self.pe_channels, x.device, x.dtype)
            z = torch.cat([z, p[None, :, :, None].expand(z.shape[0], -1, -1, z.shape[-1])], 1)
            h = z
            for layer in self.bodies[si]:
                h = layer(h)
                if return_features and isinstance(layer, nn.LeakyReLU):
                    feats.append(h)
            # mean over (F, T) => a single global score, keepdim so the shape stays 4-D
            scores.append(self.heads[si](h).mean(dim=(-2, -1), keepdim=True))
        return (scores, feats) if return_features else scores
