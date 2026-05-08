"""Tube-patch video tokenizer for the tangtv camera.

Each spatiotemporal patch ``(n_channels, T_p, H_p, W_p)`` of the input
becomes one token. With patch shape ``(3, 12, 12)`` over input
``(7, 3, 120, 360)`` this gives ``(120/12) * (360/12) = 300`` tokens
per camera per 50 ms window. Each token has a bounded receptive field
of one patch (``7 x 3 x 12 x 12 = 3024`` pixels), unlike the earlier
Perceiver-pool design where each token's content was a global average
over all patches.

This local-patch property is the structural reason per-patch
reconstruction can preserve plasma fine structure: the decoder only
needs to map each token to its own ``(C, T_p, H_p, W_p)`` region, and
each region is small enough (864 floats compressed to 256 ≈ 3.4x for
the tangtv 2-channel config; channels 4 and 6 only — see SignalConfig)
to be reproducible. The Perceiver-pool design plateaued at ratio
~0.62 on plasma channels regardless of token count or decoder depth
because global pooling cannot encode unbounded local structure into
a bounded number of global tokens.

Forward contract:
* ``x``: ``(B, n_channels, n_frames, H, W)``.
* ``mask``: optional ``(B,)`` bool. ``True`` rows encoded normally;
  ``False`` rows replaced by the learned ``missing_token``. ``None``
  is equivalent to all-True.
* output: ``(B, n_tokens, d_model)`` where ``n_tokens = n_h * n_w``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VideoTokenizer(nn.Module):
    """Tube-patch video tokenizer.

    Parameters
    ----------
    n_channels : int, optional
        Number of optical-filter / colour channels in the input. The
        tangtv default is ``2`` (filters 4 and 6 — the only two that
        carry plasma data on this camera; ch0–3, ch5 are background /
        calibration / dim). Selection happens in ``MovieConfig.channels_to_use``.
    n_frames : int, optional
        Number of time samples per window. Default ``3`` (3 evenly
        spaced frames per 50 ms half-window).
    patch_size : tuple of int, optional
        ``(T_p, H_p, W_p)``. Each patch becomes one token. Must
        satisfy ``n_frames % T_p == 0`` and ``H % H_p == 0`` and
        ``W % W_p == 0`` (i.e. the patch grid tiles the input). Default
        ``(3, 12, 12)``.
    d_model : int, optional
        Backbone token dimension. Default ``256``.
    spatial_size : tuple of int, optional
        Input spatial size ``(H, W)``. Default ``(120, 360)`` (tangtv
        after 2x bilinear downsample).

    Notes
    -----
    Initial weights:
    * Patch embedding ``Conv3d``: PyTorch default (Kaiming-ish).
    * ``spatial_pe``, ``modality_emb``, ``missing_token``: std=0.02.
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_frames: int = 3,
        patch_size: tuple[int, int, int] = (3, 12, 12),
        d_model: int = 256,
        spatial_size: tuple[int, int] = (120, 360),
    ) -> None:
        super().__init__()
        T_p, H_p, W_p = (int(p) for p in patch_size)
        H, W = int(spatial_size[0]), int(spatial_size[1])
        if n_frames % T_p:
            raise ValueError(
                f"n_frames={n_frames} must be divisible by patch T_p={T_p}."
            )
        if H % H_p:
            raise ValueError(
                f"spatial H={H} must be divisible by patch H_p={H_p}."
            )
        if W % W_p:
            raise ValueError(
                f"spatial W={W} must be divisible by patch W_p={W_p}."
            )

        self.n_channels = n_channels
        self.n_frames = n_frames
        self.patch_size = (T_p, H_p, W_p)
        self.d_model = d_model
        self.spatial_size = (H, W)
        self.n_h = H // H_p
        self.n_w = W // W_p
        self.n_t = n_frames // T_p
        self.n_tokens = self.n_h * self.n_w * self.n_t

        # Patch embedding: kernel and stride both equal to the patch
        # size, so each output element is a learned linear projection
        # of one disjoint patch.
        self.patch_embed = nn.Conv3d(
            n_channels,
            d_model,
            kernel_size=(T_p, H_p, W_p),
            stride=(T_p, H_p, W_p),
        )

        # Per-token spatial position embedding. ``n_t`` is folded into
        # the token sequence after the conv by reshape; we keep one PE
        # per (t, h, w) cell so each token knows its full position.
        self.spatial_pe = nn.Parameter(
            torch.randn(1, self.n_tokens, d_model) * 0.02
        )

        # Modality embedding (one per camera) and learned
        # missing-camera replacement.
        self.modality_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_token = nn.Parameter(
            torch.randn(1, self.n_tokens, d_model) * 0.02
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of present-camera frames to ``(B, n_tokens, d_model)``."""
        # x: (B, C, T, H, W)
        feat = self.patch_embed(x)             # (B, d_model, n_t, n_h, n_w)
        # (B, d_model, n_t, n_h, n_w) → (B, n_tokens, d_model)
        feat = feat.flatten(2).transpose(1, 2)
        feat = feat + self.spatial_pe
        feat = feat + self.modality_emb
        return feat

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B = x.shape[0]
        if mask is None or mask.all():
            return self._encode(x)
        out = self.missing_token.expand(B, -1, -1).clone()
        if mask.any():
            out[mask] = self._encode(x[mask])
        return out
