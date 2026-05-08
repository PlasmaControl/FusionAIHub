"""Patch-based spectrogram tokenizer for ECE / CO2 / BES.

Each ``(C, F_p, T_p)`` patch of the STFT magnitude spectrogram becomes
one token via a single ``Conv2d`` with kernel and stride equal to the
patch size. With patch ``(F_p, T_p) = (32, 8)`` on input
``(40, 512, 98)`` (truncated to 98 → 96 internally), this yields
``(512/32) * (96/8) = 16 * 12 = 192`` tokens per ECE window. Each
token has a bounded receptive field of one patch, mirroring the
Phase C tube-patch video tokenizer's local-patch property.

The Perceiver-pool alternative (a small fixed set of global queries)
was abandoned for video because bounded global tokens cannot encode
unbounded local spatial structure. The same argument applies to
spectrograms.

Forward contract:
* ``x``: ``(B, n_channels, freq_bins, time_frames)`` — STFT magnitude
  in ``(C, F, T)`` axis order with DC bin already removed by the data
  loader. ``freq_bins=512``, ``time_frames=98`` for the project's
  default ``n_fft=1024, hop=256`` on a 50 ms 500 kHz window.
* ``mask``: optional ``(B,)`` bool. ``True`` rows encoded normally;
  ``False`` rows replaced by the learned ``missing_token``. ``None``
  is equivalent to all-True. Mirrors the Phase C ``VideoTokenizer``
  contract — used when a modality is absent for a given shot
  (``<name>_valid == 0`` from the data loader).
* output: ``(B, n_tokens, d_model)`` where ``n_tokens = n_patches_f
  * n_patches_t``. Time is truncated to the largest multiple of
  ``patch_t`` ≤ ``time_frames`` (98 → 96 by default); the discarded
  tail represents <2.1% of the window.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectrogramTokenizer(nn.Module):
    """Patch-based spectrogram tokenizer.

    Parameters
    ----------
    n_channels : int
        Number of input channels (40 for ECE, 4 for CO2, 16 for BES).
    d_model : int
        Token embedding dimension.
    patch_f : int
        Frequency-axis patch size. Must divide ``freq_bins`` cleanly.
    patch_t : int
        Time-axis patch size. ``time_frames`` is truncated to the
        largest multiple of ``patch_t`` ≤ ``time_frames``.
    freq_bins : int
        Number of STFT frequency bins (DC dropped by the data loader).
        Default project value is 512.
    time_frames : int
        Number of STFT time frames in the input window. Default project
        value is 98 (a 50 ms window at 500 kHz with hop=256, center=True).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        patch_f: int,
        patch_t: int,
        freq_bins: int,
        time_frames: int,
    ) -> None:
        super().__init__()
        if freq_bins % patch_f != 0:
            raise ValueError(
                f"freq_bins ({freq_bins}) must be divisible by patch_f "
                f"({patch_f})."
            )

        self.n_channels = n_channels
        self.d_model = d_model
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        # Truncate time to the largest multiple of patch_t.
        self.trunc_t = (time_frames // patch_t) * patch_t

        self.n_patches_f = freq_bins // patch_f
        self.n_patches_t = self.trunc_t // patch_t
        self.n_tokens = self.n_patches_f * self.n_patches_t

        # Conv2d kernel_size=(F_p, T_p) matches data layout (B, C, F, T).
        self.proj = nn.Conv2d(
            in_channels=n_channels,
            out_channels=d_model,
            kernel_size=(patch_f, patch_t),
            stride=(patch_f, patch_t),
        )
        self.spatial_pe = nn.Parameter(torch.empty(self.n_tokens, d_model))
        self.modality_embed = nn.Parameter(torch.empty(d_model))
        # Learned replacement used when a sample has the modality absent
        # (per-batch ``mask=False``). Same pattern as VideoTokenizer.
        self.missing_token = nn.Parameter(torch.empty(self.n_tokens, d_model))

        nn.init.normal_(self.spatial_pe, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)
        nn.init.normal_(self.missing_token, std=0.02)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of present-modality spectrograms to
        ``(B, n_tokens, d_model)``."""
        x = x[..., : self.trunc_t]                      # (B, C, F, T_trunc)
        tokens = self.proj(x)                           # (B, d_model, n_f, n_t)
        tokens = tokens.flatten(2).transpose(1, 2)      # (B, n_tokens, d_model)
        return tokens + self.spatial_pe + self.modality_embed

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Tokenize one batch of spectrograms.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, n_channels, freq_bins, time_frames)``.
        mask : torch.Tensor, optional
            ``(B,)`` bool tensor. ``True`` rows go through the normal
            Conv2d path; ``False`` rows are replaced by the learned
            ``missing_token``. ``None`` is equivalent to all-True.

        Returns
        -------
        torch.Tensor
            Tokens of shape ``(B, n_tokens, d_model)``.
        """
        B = x.shape[0]
        if mask is None or mask.all():
            return self._encode(x)
        out = self.missing_token.expand(B, -1, -1).clone()
        if mask.any():
            out[mask] = self._encode(x[mask])
        return out
