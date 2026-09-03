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
