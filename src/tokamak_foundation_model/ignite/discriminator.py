"""Frequency-aware multi-scale PatchGAN discriminator for the spectrogram codec.

Uses only ``torch.nn``, ``einops`` and ``config.py`` (docs/IGNITE_DESIGN.md §4.2).

- The frequency axis of a spectrogram is not translation-invariant: a mode at 50 kHz is
  physically different from one at 150 kHz. A plain PatchGAN is fully convolutional and
  so equivariant to frequency shifts. Concatenating frequency positional-encoding
  channels to the input breaks that symmetry.
- The same body is applied at two scales (full resolution and a downsampled copy) to
  judge fine mode texture and coarse band structure.
- The heads return raw patch scores (no sigmoid) for a hinge loss.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectroCodecConfig


def _sinusoidal_freq_pe(freq_bins: int, n_channels: int, device, dtype) -> torch.Tensor:
    """Sinusoidal positional encoding over frequency, shape (n_channels, freq_bins).

    Parameter-free; the caller broadcasts it along time.
    """
    pos = torch.arange(freq_bins, device=device, dtype=dtype).unsqueeze(1)  # (F, 1)
    i = torch.arange(n_channels, device=device, dtype=dtype).unsqueeze(0)   # (1, C)
    # Geometrically spaced frequency per encoding channel.
    div = torch.pow(
        torch.tensor(10_000.0, device=device, dtype=dtype),
        (2.0 * torch.floor(i / 2.0)) / max(n_channels, 1),
    )
    angles = pos / div  # (F, C)
    pe = torch.where(i.long() % 2 == 0, torch.sin(angles), torch.cos(angles))
    return pe.transpose(0, 1).contiguous()  # (C, F)


class _PatchGANBody(nn.Module):
    """Small conv PatchGAN: (B, C_in, F, T) -> one raw score map."""

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
        # Final conv to a single-channel raw score map, no activation.
        layers.append(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the body and also return its intermediate activations.

        Returns ``(score, feats)``. ``score`` equals :meth:`forward`; ``feats`` holds
        the output of every LeakyReLU, not the input or the final score map. The
        generator's feature-matching loss (as in HiFi-GAN and MelGAN) compares these.
        """
        feats: List[torch.Tensor] = []
        h = x
        n = len(self.net)
        for i, layer in enumerate(self.net):
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU) and i < n - 1:
                feats.append(h)
        return h, feats


class FreqAwarePatchGAN(nn.Module):
    """Multi-scale, frequency-aware PatchGAN discriminator.

    ``forward(x)`` maps ``(B, C, F, T)`` to a list of raw patch-score maps, one per
    scale, each ``(B, 1, h, w)``. ``forward(x, return_features=True)`` also returns the
    intermediate activations of all bodies as one flat list, for feature matching.
    """

    def __init__(self, cfg: SpectroCodecConfig, base: int = 32, n_pe_channels: int = 4,
                 scales: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_pe_channels = n_pe_channels
        self.scales = scales
        in_ch = cfg.channels + n_pe_channels
        # One independent body per scale.
        self.bodies = nn.ModuleList(
            [_PatchGANBody(in_ch, base=base) for _ in range(scales)]
        )

    def freq_pe(self, x: torch.Tensor) -> torch.Tensor:
        """Frequency-position channels for ``x`` (B, C, F, T).

        Returns (B, n_pe_channels, F, T), constant in time.
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

        With ``return_features=True`` the second element is a flat list of the
        intermediate activations of every scale's body, for the feature-matching loss.
        """
        maps: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        cur = x
        for si, body in enumerate(self.bodies):
            if si > 0:
                # Downsample the input; freq_pe is rebuilt at the new resolution.
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
    """Multi-scale critic that scores the whole spectrogram, against patch tiling.

    :class:`FreqAwarePatchGAN` returns a map of per-patch scores, so the generator can
    satisfy it by learning one convincing patch texture and tiling it. That happened:
    the reconstruction had ``patch_lattice_ratio`` 61.02 against 1.14 for the ground
    truth, with 92-94% of its high-frequency energy on the ``(patch_f, patch_t)``
    lattice, sustained by the adversarial objective.

    This critic reduces the whole spectrogram to one score per scale, at 1x, 2x and 4x
    frequency-time downsampling. A tiled texture looks wrong globally at every scale.
    The idea follows arXiv 2406.05298 §4, which pairs a multi-period discriminator with
    a multi-scale complex-STFT one.

    Same interface as ``FreqAwarePatchGAN``: ``forward(x)`` returns one raw score per
    scale, shape ``(B, 1, 1, 1)``, and ``forward(x, return_features=True)`` also returns
    the intermediate activations for feature matching. The frequency-position channels
    are kept, since frequency is not translation-invariant.
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
            # Global score: 1x1 conv, then the mean over the whole map in forward().
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
            # Mean over (F, T) gives one score; keepdim keeps the shape 4-D.
            scores.append(self.heads[si](h).mean(dim=(-2, -1), keepdim=True))
        return (scores, feats) if return_features else scores
