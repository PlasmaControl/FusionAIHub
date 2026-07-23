"""1-D PatchGAN discriminator (IGNITE Phase-A fast-TS / filterscopes envelope codec).

The fast-TS analogue of ``discriminator.FreqAwarePatchGAN``. Fresh code (no FAITH model
reuse); uses only ``torch.nn`` + ``config.py``.

Design (docs/IGNITE_DESIGN.md §4.2/§4.3):
- The codec target is the 1-D ELM activity ENVELOPE ``(B, C, E)`` (E = envelope-time bins),
  so the discriminator is a 1-D (over the envelope-time axis) multi-scale PatchGAN with the
  ``C`` filterscope channels as conv input channels.
- Like the spectrogram (and unlike video), the envelope-time axis is NOT purely
  translation-invariant — WHEN in the 50 ms window an ELM burst sits carries physical meaning
  (ramp / flat-top phase). A fully-convolutional PatchGAN is time-shift-equivariant, the wrong
  bias, so we break that symmetry with a small sinusoidal TIME positional-encoding channel
  concatenated to the input (the 1-D analogue of ``FreqAwarePatchGAN``'s freq-PE).
- Multi-scale: the body is applied at a couple of input scales (full + a time-downsampled
  copy) so both fine burst texture and coarse activity structure are judged.
- Hinge-ready: heads emit RAW patch scores (no final sigmoid). ``return_features=True`` also
  returns intermediate conv activations for the generator's feature-matching loss.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FastTSCodecConfig


def _sinusoidal_time_pe(env_bins: int, n_channels: int, device, dtype) -> torch.Tensor:
    """Return a (n_channels, env_bins) sinusoidal positional encoding over the time axis.

    Deterministic, parameter-free — the 1-D analogue of ``discriminator._sinusoidal_freq_pe``.
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
    """Output length of a conv1d: floor((L + 2*pad - kernel)/stride) + 1 (0 if underflows)."""
    return (length + 2 * pad - kernel) // stride + 1


class _Env1DPatchGANBody(nn.Module):
    """A small 1-D conv PatchGAN producing a single raw score map from (B, C_in, E).

    ``n_layers`` is the REQUESTED number of kernel-4 conv blocks (as before). The stride and the
    ACTUAL block count are chosen ADAPTIVELY from the input ``length``: a kernel-4/pad-1 conv
    needs an input of length >= 2, so on the COARSE 10 ms envelope (only E=5 bins, far shorter
    than the old 50-bin grid) the fixed 3-layer / stride-2 stack underflows. Each block uses
    stride 2 for all but the last requested block (as originally), but downgrades to stride 1
    when a stride-2 conv would not fit, and the loop stops early once a kernel-4 conv would
    underflow (L < 2). It always finishes with the kernel-3 / stride-1 score conv (needs
    length >= 1). At the old E=50 this reproduces the original 3×kernel-4 (2 stride-2 + 1
    stride-1) + kernel-3 body EXACTLY; at E=5 it degrades gracefully to a valid short stack.
    """

    def __init__(self, in_channels: int, length: int, base: int = 32, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        out = base
        L = int(length)
        placed = 0
        for li in range(n_layers):
            if L < 2:  # a kernel-4 conv can no longer fit — stop adding downsampling blocks.
                break
            want_stride = 2 if li < n_layers - 1 else 1
            # downgrade to stride 1 if a stride-2 conv would underflow at this length.
            stride = want_stride if _conv_out_len(L, 4, want_stride, 1) >= 1 else 1
            layers.append(nn.Conv1d(ch, out, kernel_size=4, stride=stride, padding=1))
            if placed > 0:
                layers.append(nn.GroupNorm(num_groups=min(8, out), num_channels=out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            L = _conv_out_len(L, 4, stride, 1)
            ch = out
            out = min(out * 2, base * 4)
            placed += 1
        # final conv -> single-channel raw score map (no activation / no sigmoid); kernel 3
        # (pad 1) needs only L >= 1, which always holds after the loop.
        layers.append(nn.Conv1d(ch, 1, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the body, capturing intermediate conv activations before the final score conv.

        Returns ``(score, feats)`` where ``score`` is byte-identical to :meth:`forward` and
        ``feats`` is the list of activations after each LeakyReLU (every conv-block output that
        feeds forward), EXCLUDING the raw input and the final score map — the tensors matched
        by the generator's feature-matching loss. Mirrors ``_PatchGANBody.forward_features``.
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
    """Multi-scale, time-aware 1-D PatchGAN discriminator for the fast-TS envelope codec.

    Forward: ``(B, C, E) -> list[Tensor]`` of raw patch-score maps (one per scale), each
    ``(B, 1, e)``. Hinge-ready (no final sigmoid).

    ``forward(x, return_features=True) -> (list[score], list[feat])`` additionally returns a
    flat list of intermediate conv activations (both scales' bodies), for the generator's
    feature-matching loss. The default ``forward(x)`` path is byte-identical to before.

    Structurally the 1-D analogue of ``discriminator.FreqAwarePatchGAN`` (same body pattern,
    multi-scale avg-pool, hinge-raw scores, feature-matching hook, sinusoidal PE channel), over
    the envelope-time axis instead of the (freq, time) spectrogram plane.
    """

    def __init__(self, cfg: FastTSCodecConfig, base: int = 32, n_pe_channels: int = 4,
                 scales: int = 2):
        super().__init__()
        self.cfg = cfg
        self.n_pe_channels = n_pe_channels
        in_ch = cfg.channels + n_pe_channels
        # Per-scale input length: scale si sees the envelope avg-pooled si times by k=2
        # (ceil_mode). The COARSE envelope is only E=5 bins, so cap the number of scales to
        # those whose pooled length stays >= 2 (a length-1 input has no multi-scale texture and
        # would collapse the body); this keeps 2 scales at the old 50-bin grid but folds to 1 at
        # E=5. Each body is then sized for its OWN pooled length so no conv empties out.
        lengths: List[int] = []
        L = int(cfg.env_bins)
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
        """Build the time positional-encoding channels for input ``x`` (B, C, E) -> (B, Cpe, E)."""
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
        """``(B, C, E) -> list[score]``, or ``(list[score], list[feat])`` if requested."""
        if x.dim() != 3:
            raise ValueError(f"Env1DPatchGAN expects (B, C, E); got {tuple(x.shape)}")
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
