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
import torch.nn.functional as F


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
        enable_freq_stem: bool = False,
        freq_stem_hidden: int = 128,
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

        # Pre-backbone per-token MLP refiners (stacked ViT-style residual MLP
        # blocks). Each block is independently applied with a residual at the
        # call site so adding/removing blocks is a single-line change.
        # 2026-05-19: bumped 4 → 12 to add capacity around the d_model=256
        # bottleneck for fine-pattern reconstruction (harmonics + transients).
        # train_e2e_stage1.py's resume path auto-detects the extra refine
        # blocks as missing keys and auto-applies a 1-epoch (1180-step)
        # freeze on backbone/ts/video while the new blocks 4..11 settle.
        n_refine_blocks = 12
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_refine_blocks)
        ])

        nn.init.normal_(self.spatial_pe, std=0.02)
        nn.init.normal_(self.modality_embed, std=0.02)
        nn.init.normal_(self.missing_token, std=0.02)

        # OPT-IN full-frequency encoder stem (2026-06-13). The patch
        # Conv2d below has a receptive field of only patch_f (32) freq
        # bins, so a token cannot encode where a mode peak sits relative
        # to the WHOLE spectrum — information lost before the bottleneck.
        # This stem mixes across all `freq_bins` bins BEFORE patching, as
        # a zero-init residual so the encoder is bit-identical at load
        # (exact warm-start) and only diverges as it trains. Expressed as
        # a Linear over the frequency axis (a full-freq filter == a dense
        # freq->freq map) rather than a (freq_bins, 1) Conv2d: matmuls run
        # on rocBLAS with no per-shape MIOpen tuning, sidestepping the
        # kernel-tuning/fallback pathology that novel conv shapes hit at
        # batch 32 (jobs 4802391/4803320). Weights shared across channels
        # and time (folded into the matmul batch dim).
        self.enable_freq_stem = bool(enable_freq_stem)
        if self.enable_freq_stem:
            self.fs_lin1 = nn.Linear(freq_bins, freq_stem_hidden)
            self.fs_lin2 = nn.Linear(freq_stem_hidden, freq_bins)
            nn.init.zeros_(self.fs_lin2.weight)
            nn.init.zeros_(self.fs_lin2.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of present-modality spectrograms to
        ``(B, n_tokens, d_model)``."""
        x = x[..., : self.trunc_t]                      # (B, C, F, T_trunc)
        if self.enable_freq_stem:
            # Full-frequency residual mixing before patching. Operate
            # with frequency as the last (feature) dim so the Linears
            # mix across all freq bins; zero-init fs_lin2 → no-op at
            # construction → exact warm-start.
            h = x.transpose(2, 3)                       # (B, C, T, F)
            h = self.fs_lin2(F.gelu(self.fs_lin1(h)))   # (B, C, T, F)
            x = x + h.transpose(2, 3)                   # (B, C, F, T_trunc)
        tokens = self.proj(x)                           # (B, d_model, n_f, n_t)
        tokens = tokens.flatten(2).transpose(1, 2)      # (B, n_tokens, d_model)
        tokens = tokens + self.spatial_pe + self.modality_embed
        for block in self.refine:
            tokens = tokens + block(tokens)
        return tokens

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
        # Always invoke _encode and reference missing_token so the autograd
        # graph for proj / spatial_pe / modality_embed / missing_token is
        # data-independent. Lets us run DDP without `find_unused_parameters`
        # (RCCL bucket rebuilds on a per-batch-changing unused-set were
        # causing GPU memory faults on Frontier). Extra cost: a Conv2d on
        # the masked-out rows; small relative to the backbone transformer.
        B = x.shape[0]
        encoded = self._encode(x)
        missing = self.missing_token.expand(B, -1, -1)
        if mask is None:
            return encoded + 0.0 * missing.sum()
        return torch.where(mask.view(B, 1, 1), encoded, missing)
