"""1-D PatchGAN discriminator for the Phase-A fast-TS (filterscopes) codec.

The fast-TS counterpart of ``discriminator.FreqAwarePatchGAN`` (docs/IGNITE_DESIGN.md
§4.2/§4.3).

- It is a multi-scale 1-D PatchGAN over the time axis of ``(B, C, W)``, with the ``C``
  filterscope channels as conv input channels. In raw mode W = cfg.window = 500 samples;
  in envelope mode the input is the E = 5 bin envelope. The body adapts its stride-2
  stack to the input length.
- As for spectrograms, where a burst sits in the 50 ms window carries meaning, so the
  translation equivariance of a plain convolutional PatchGAN is the wrong bias. A small
  sinusoidal time positional-encoding channel is concatenated to the input to break it.
- In raw mode ``FastTSCodecConfig.adversarial_weight`` and ``fm_weight`` are 0.0, since
  a GAN adds uncorrelated high-frequency detail that raises sample-wise nRMSE. The
  module stays wired so the trade-off can be measured with ``--adversarial_weight``.
- Multi-scale: the body runs on the input and on a time-downsampled copy, so both fine
  burst texture and coarse activity are judged.
- Heads emit raw patch scores (no sigmoid) for the hinge loss. ``return_features=True``
  also returns intermediate activations for the generator's feature-matching loss.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FastTSCodecConfig


def _sinusoidal_time_pe(env_bins: int, n_channels: int, device, dtype) -> torch.Tensor:
    """Return a (n_channels, env_bins) sinusoidal positional encoding over the time axis.

    Deterministic and parameter-free; the 1-D version of
    ``discriminator._sinusoidal_freq_pe``.
    """
    pos = torch.arange(env_bins, device=device, dtype=dtype).unsqueeze(1)   # (E, 1)
    i = torch.arange(n_channels, device=device, dtype=dtype).unsqueeze(0)   # (1, C)
    div = torch.pow(
        torch.tensor(10_000.0, device=device, dtype=dtype),
        (2.0 * torch.floor(i / 2.0)) / max(n_channels, 1),
    )
    angles = pos / div  # (E, C)
    pe = torch.where(i.long() % 2 == 0, torch.sin(angles), torch.cos(angles))
    return pe.transpose(0, 1).contiguous()  # (C, E)


def _conv_out_len(length: int, kernel: int, stride: int, pad: int) -> int:
    """Output length of a conv1d, (L + 2*pad - kernel) // stride + 1 (<= 0 if empty)."""
    return (length + 2 * pad - kernel) // stride + 1


class _Env1DPatchGANBody(nn.Module):
    """A small 1-D conv PatchGAN producing one raw score map from (B, C_in, E).

    ``n_layers`` is the requested number of kernel-4 conv blocks. Stride and the actual
    block count follow the input ``length``, because a kernel-4/pad-1 conv needs an
    input of length >= 2 and the 5-bin envelope is too short for a fixed 3-layer
    stride-2 stack. Each block uses stride 2 except the last requested one, falls back
    to stride 1 when a stride-2 conv would not fit, and the loop stops once a kernel-4
    conv would underflow (L < 2). A kernel-3 stride-1 score conv (length >= 1) always
    ends the stack. At E = 50 this is the original body (two stride-2 and one stride-1
    kernel-4 blocks, then the kernel-3 conv).
    """

    def __init__(self, in_channels: int, length: int, base: int = 32, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        out = base
        L = int(length)
        placed = 0
        for li in range(n_layers):
            if L < 2:  # a kernel-4 conv no longer fits
                break
            want_stride = 2 if li < n_layers - 1 else 1
            # Fall back to stride 1 if a stride-2 conv would underflow at this length.
            stride = want_stride if _conv_out_len(L, 4, want_stride, 1) >= 1 else 1
            layers.append(nn.Conv1d(ch, out, kernel_size=4, stride=stride, padding=1))
            if placed > 0:
                layers.append(nn.GroupNorm(num_groups=min(8, out), num_channels=out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            L = _conv_out_len(L, 4, stride, 1)
            ch = out
            out = min(out * 2, base * 4)
            placed += 1
        # Final conv to a one-channel raw score map (no activation, no sigmoid). Kernel
        # 3 with pad 1 needs only L >= 1, which always holds after the loop.
        layers.append(nn.Conv1d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the body, capturing intermediate conv activations before the final score conv.

        Returns ``(score, feats)``: ``score`` equals :meth:`forward`, and ``feats``
        holds the activation after each LeakyReLU (not the input or the score map),
        which the generator's feature-matching loss compares. Mirrors
        ``_PatchGANBody.forward_features``.
        """
        feats: List[torch.Tensor] = []
        h = x
        n = len(self.net)
        for i, layer in enumerate(self.net):
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU) and i < n - 1:
                feats.append(h)
        return h, feats


class Env1DPatchGAN(nn.Module):
    """Multi-scale, time-aware 1-D PatchGAN discriminator for the fast-TS codec.

    ``forward(x)`` maps ``(B, C, W)`` to a list of raw patch-score maps, one ``(B, 1,
    e)`` per scale, with no sigmoid. ``forward(x, return_features=True)`` also returns a
    flat list of intermediate activations from all scales, for feature matching.

    Same structure as ``discriminator.FreqAwarePatchGAN`` (body, multi-scale average
    pooling, raw hinge scores, feature hook, sinusoidal PE channel), over the time axis
    instead of the (freq, time) plane.
    """

    def __init__(self, cfg: FastTSCodecConfig, base: int = 32, n_pe_channels: int = 4,
                 scales: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_pe_channels = n_pe_channels
        in_ch = cfg.channels + n_pe_channels
        # Scale si sees the input average-pooled si times with k=2 (ceil_mode). Keep
        # only scales whose pooled length stays >= 2 (a length-1 input has no texture to
        # judge), and size each body for its own length so no conv empties out.
        lengths: List[int] = []
        # Envelope mode judges an E=5 curve, raw mode a W=500 waveform.
        L = int(cfg.window if cfg.is_raw else cfg.env_bins)
        for si in range(scales):
            if si > 0:
                L = -(-L // 2)  # ceil(L / 2), matching F.avg_pool1d(kernel=2, ceil_mode=True)
            if si > 0 and L < 2:
                break
            lengths.append(L)
        self.scales = len(lengths)
        self.bodies = nn.ModuleList(
            [_Env1DPatchGANBody(in_ch, length=Ls, base=base) for Ls in lengths]
        )

    def time_pe(self, x: torch.Tensor) -> torch.Tensor:
        """Build the time positional-encoding channels for input ``x`` (B, C, W) -> (B, Cpe, W)."""
        b, _, env_bins = x.shape
        pe = _sinusoidal_time_pe(env_bins, self.n_pe_channels, x.device, x.dtype)  # (Cpe, E)
        return pe.unsqueeze(0).expand(b, self.n_pe_channels, env_bins)

    def _apply_body(self, body: _Env1DPatchGANBody, x: torch.Tensor) -> torch.Tensor:
        xin = torch.cat([x, self.time_pe(x)], dim=1)
        return body(xin)

    def _apply_body_features(
        self, body: _Env1DPatchGANBody, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        xin = torch.cat([x, self.time_pe(x)], dim=1)
        return body.forward_features(xin)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """``(B, C, W) -> list[score]``, or ``(list[score], list[feat])`` if requested."""
        if x.dim() != 3:
            raise ValueError(f"Env1DPatchGAN expects (B, C, W); got {tuple(x.shape)}")
        maps: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        cur = x
        for si, body in enumerate(self.bodies):
            if si > 0:
                cur = F.avg_pool1d(cur, kernel_size=2, ceil_mode=True)
            if return_features:
                score, body_feats = self._apply_body_features(body, cur)
                maps.append(score)
                feats.extend(body_feats)
            else:
                maps.append(self._apply_body(body, cur))
        if return_features:
            return maps, feats
        return maps
