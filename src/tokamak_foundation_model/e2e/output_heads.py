"""Per-modality output heads.

Each head is an approximate inverse of its sibling tokenizer. They fire only
to compute the training loss against ground-truth raw signals — during
autoregressive rollout the backbone's token output is fed directly to the
next step, bypassing the heads (``ResearchPlan.MD`` §3.5, §3.6, §5.7).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt


class SlowTimeSeriesHead(nn.Module):
    """Linear head reconstructing a slow time-series modality.

    Parameters
    ----------
    d_model
        Token embedding dimension.
    n_channels
        Number of diagnostic channels.
    window_samples
        Samples per channel in one 50 ms window (``5`` at 100 Hz).

    Notes
    -----
    Approximate inverse of :class:`SlowTimeSeriesTokenizer`: a single shared
    ``Linear(d_model, window_samples)`` unprojects each per-channel token back
    to raw signal samples.
    """

    def __init__(
        self, d_model: int, n_channels: int, window_samples: int
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.proj = nn.Linear(d_model, window_samples)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw signal.

        Parameters
        ----------
        tokens
            ``(batch, n_channels, d_model)`` — per-channel tokens from the
            backbone for this modality.

        Returns
        -------
        torch.Tensor
            ``(batch, n_channels, window_samples)`` raw-signal reconstruction.
        """
        return self.proj(tokens)


class FastTimeSeriesHead(nn.Module):
    """ConvTranspose1d head reconstructing a fast time-series modality.

    Parameters
    ----------
    d_model
        Token embedding dimension.
    n_channels
        Number of diagnostic channels.
    window_samples
        Samples per channel in one 50 ms window (``500`` at 10 kHz).
    patch_size
        Patch length matching the sibling tokenizer (``50`` by default). Must
        divide ``window_samples``.

    Notes
    -----
    Approximate inverse of :class:`FastTimeSeriesTokenizer`. Channels are
    reshaped into the batch axis so a single shared
    ``ConvTranspose1d(in=d_model, out=1, k=s=patch_size)`` unpacks each
    per-channel patch sequence back to raw samples.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        window_samples: int,
        patch_size: int = 50,
    ) -> None:
        super().__init__()
        if window_samples % patch_size != 0:
            raise ValueError(
                f"window_samples ({window_samples}) must be a multiple of "
                f"patch_size ({patch_size})"
            )
        self.d_model = d_model
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.patch_size = patch_size
        self.n_patches = window_samples // patch_size

        # Post-deconv inverse-stem at sample resolution, mirroring the
        # tokenizer's pre-patch stem. The deconv first lifts each token back
        # to ``stem_channels × patch_size`` samples; the inverse stem then
        # refines the per-sample reconstruction with two small-kernel convs,
        # giving the head the capacity to recover sharp features (spikes,
        # bursts) the linear deconv alone smooths over.
        stem_channels = 64
        self.deconv = nn.ConvTranspose1d(
            in_channels=d_model,
            out_channels=stem_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.inv_stem = nn.Sequential(
            nn.Conv1d(stem_channels, stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(stem_channels, 1, kernel_size=3, padding=1),
        )

        # Pre-unembed per-token MLP refiners (mirror of the tokenizer's).
        # 2026-05-19: bumped 2 → 4 alongside FastTimeSeriesTokenizer.
        n_refine_blocks = 4
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(n_refine_blocks)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct raw signal.

        Parameters
        ----------
        tokens
            ``(batch, n_channels * n_patches, d_model)`` in channel-major
            order (matching :class:`FastTimeSeriesTokenizer`).

        Returns
        -------
        torch.Tensor
            ``(batch, n_channels, window_samples)`` raw-signal reconstruction.
        """
        batch = tokens.shape[0]
        for block in self.refine:
            tokens = tokens + block(tokens)
        t = tokens.reshape(batch, self.n_channels, self.n_patches, self.d_model)
        t = t.reshape(batch * self.n_channels, self.n_patches, self.d_model)
        t = t.transpose(1, 2)  # (B*C, d_model, n_patches)
        out = self.deconv(t)  # (B*C, stem_channels, window_samples)
        out = self.inv_stem(out)  # (B*C, 1, window_samples)
        return out.reshape(batch, self.n_channels, self.window_samples)


class VideoOutputHead(nn.Module):
    """Per-patch reconstruction head — exact inverse of the tube-patch
    :class:`VideoTokenizer`.

    Tokens arrive as ``(B, n_tokens, d_model)`` where
    ``n_tokens = (n_frames / T_p) * (H / H_p) * (W / W_p)``. They are
    reshaped to a 5-D feature volume ``(B, d_model, n_t, n_h, n_w)`` and
    passed through a single ``ConvTranspose3d`` whose kernel and stride
    both equal the patch shape. Each token thus reconstructs its own
    ``(n_channels, T_p, H_p, W_p)`` region without any global mixing.
    Output shape ``(B, n_frames, n_channels, H, W)`` matches the input
    layout permuted from ``(C, T, H, W)`` to ``(T, C, H, W)``.

    Parameters
    ----------
    n_channels : int, optional
        Number of optical filters reconstructed. Default ``7``.
    n_frames : int, optional
        Number of time samples per output window. Default ``3``.
    patch_size : tuple of int, optional
        ``(T_p, H_p, W_p)`` — must match the tokenizer.
        Default ``(3, 12, 12)``.
    d_model : int, optional
        Backbone token dimension. Default ``256``.
    spatial_size : tuple of int, optional
        Output spatial size ``(H, W)``. Default ``(120, 360)``.

    Notes
    -----
    No bilinear upsampling and no MLP. ``ConvTranspose3d`` with
    ``kernel = stride = patch_size`` exactly inverts the tokenizer's
    patch ``Conv3d`` and is the standard ViT/VideoMAE inverse. Param
    count is ``d_model * n_channels * prod(patch_size) + n_channels``,
    e.g. 256 * 2 * 3 * 12 * 12 + 2 ≈ 221 k for the tangtv 2-channel
    config (channels 4 and 6 only).
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_frames: int = 3,
        patch_size: tuple[int, int, int] = (3, 12, 12),
        d_model: int = 256,
        spatial_size: tuple[int, int] = (120, 360),
        enable_seam_refine: bool = False,
        seam_refine_hidden_ch: int = 16,
        seam_refine_kernel: tuple[int, int, int] = (1, 3, 3),
        decoder: str = "deconv",
        resize_conv_hidden_ch: int = 64,
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

        self.decoder = str(decoder)
        if self.decoder not in ("deconv", "resize_conv"):
            raise ValueError(
                f"decoder must be 'deconv' or 'resize_conv', got {decoder!r}."
            )

        if self.decoder == "resize_conv":
            # Checkerboard-free upsampler. The per-patch ConvTranspose3d
            # (kernel=stride=patch) decodes each token's region INDEPENDENTLY
            # → hard 12×12 patch seams (the "checkerboard"). Instead, upsample
            # to full resolution then convolve, so overlapping receptive fields
            # straddle patch boundaries and the seam structure never forms
            # (Odena et al., "Deconvolution and Checkerboard Artifacts").
            #
            # Memory/compute-efficient ordering: reduce d_model → hidden at the
            # LOW (patch-grid) resolution FIRST, then upsample only the
            # `hidden`-channel volume. This keeps the upsampled tensor at
            # `hidden` (e.g. 64) channels instead of d_model (e.g. 512) —
            # ~d_model/hidden× less activation memory — and moves the heavy
            # d_model→hidden conv off the full-resolution grid (~hundreds× fewer
            # spatial positions). The full-res `hidden→…` convs still provide
            # the overlapping receptive fields that kill the checkerboard.
            # Supersedes the seam-refine workaround below.
            h = int(resize_conv_hidden_ch)
            self.resize_proj = nn.Conv3d(d_model, h, kernel_size=3, padding=1)
            self.resize_block = nn.Sequential(
                nn.Conv3d(h, h, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(h, n_channels, kernel_size=3, padding=1),
            )
        else:
            # Inverse of the tokenizer's patch_embed Conv3d (per-patch).
            self.patch_unembed = nn.ConvTranspose3d(
                d_model,
                n_channels,
                kernel_size=(T_p, H_p, W_p),
                stride=(T_p, H_p, W_p),
            )

        # OPT-IN zero-initialised residual seam-refinement block. The
        # ConvTranspose3d above decodes each patch INDEPENDENTLY,
        # producing the visible 12×12 patch-grid checkerboard at
        # Stage 2's autoregressive K-step round-trip. The refine
        # block is a tiny spatial Conv3d → GELU → Conv3d stack
        # operating on the post-decoder (B, C, T, H, W) tensor;
        # because it crosses patch boundaries (3×3 kernel) it can
        # see and correct the seam discontinuities the per-patch
        # decoder cannot. Last conv is zero-initialised so the
        # residual contribution is 0 at module init —
        # bit-identical output to the pre-patch model when an
        # existing checkpoint is loaded, then training (with
        # --video_smoothness_weight > 0) shapes the module to
        # cancel the checkerboard. Param cost: ~600 weights for
        # the tangtv 2-channel, hidden=16 config.
        #
        # Default OFF so Stage 1 (K=1, no autoregressive checkerboard)
        # and any other consumer of VideoOutputHead don't pay the
        # param-count cost — only the Stage 2 trainers should pass
        # `enable_seam_refine=True` when constructing the head.
        # seam_refine_hidden_ch / seam_refine_kernel are constructor
        # parameters (2026-06-12) so a fine-tune can request a wider
        # refine block (e.g. 64 ch, (3,5,5) kernel) without changing
        # the default architecture that running Stage 2 chains resume
        # into — defaults (16, (1,3,3)) match the original block
        # bit-for-bit.
        # resize_conv already crosses patch seams, so seam-refine is moot there.
        self.enable_seam_refine = (
            bool(enable_seam_refine) and self.decoder == "deconv"
        )
        if self.enable_seam_refine:
            k = tuple(int(x) for x in seam_refine_kernel)
            pad = tuple(x // 2 for x in k)
            self.refine_block = nn.Sequential(
                nn.Conv3d(
                    n_channels, seam_refine_hidden_ch,
                    kernel_size=k, padding=pad,
                ),
                nn.GELU(),
                nn.Conv3d(
                    seam_refine_hidden_ch, n_channels,
                    kernel_size=k, padding=pad,
                ),
            )
            nn.init.zeros_(self.refine_block[-1].weight)
            nn.init.zeros_(self.refine_block[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n_tokens, d_model) -> (B, n_frames, n_channels, H, W)``."""
        B = tokens.shape[0]
        # (B, n_tokens, d_model) -> (B, d_model, n_t, n_h, n_w)
        x = tokens.transpose(1, 2).reshape(
            B, self.d_model, self.n_t, self.n_h, self.n_w
        )
        if self.decoder == "resize_conv":
            # Reduce channels at the LOW patch-grid resolution, THEN upsample
            # only the `hidden`-channel volume, THEN convolve at full res
            # (overlapping receptive fields across patch seams → no
            # checkerboard). Keeps the big upsampled tensor at `hidden` ch.
            x = self.resize_proj(x)              # (B, hidden, n_t, n_h, n_w)
            x = F.interpolate(
                x, scale_factor=self.patch_size,
                mode="trilinear", align_corners=False,
            )                                    # (B, hidden, T, H, W)
            out = self.resize_block(x)           # (B, n_channels, T, H, W)
        else:
            out = self.patch_unembed(x)          # (B, n_channels, T, H, W)
            if self.enable_seam_refine:
                out = out + self.refine_block(out)   # zero at init → no-op
        return out.permute(0, 2, 1, 3, 4)        # (B, T, C, H, W)


class SpectrogramOutputHead(nn.Module):
    """Per-patch reconstruction head — exact inverse of
    :class:`SpectrogramTokenizer`.

    Tokens arrive as ``(B, n_tokens, d_model)`` where
    ``n_tokens = n_patches_f * n_patches_t``. They are reshaped to a
    4-D feature map ``(B, d_model, n_patches_f, n_patches_t)`` and
    passed through a single ``ConvTranspose2d`` whose kernel and
    stride both equal the patch shape ``(F_p, T_p)``. Each token
    reconstructs its own ``(n_channels, F_p, T_p)`` region without
    global mixing. Output shape ``(B, n_channels, freq_bins,
    n_patches_t * T_p)`` matches the tokenizer's input layout
    ``(C, F, T)`` after the time-axis truncation that the tokenizer
    applies internally — the original 2 dropped time frames are not
    recovered.

    Parameters
    ----------
    n_channels : int
        Number of input/output channels (40 for ECE, 4 for CO2,
        16 for BES).
    d_model : int
        Backbone token dimension.
    patch_f : int
        Frequency-axis patch size. Must match the tokenizer.
    patch_t : int
        Time-axis patch size. Must match the tokenizer.
    n_patches_f : int
        Number of frequency patches (``freq_bins // patch_f``).
    n_patches_t : int
        Number of time patches (``trunc_t // patch_t``).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        patch_f: int,
        patch_t: int,
        n_patches_f: int,
        n_patches_t: int,
        enable_seam_refine: bool = False,
        seam_refine_hidden_ch: int = 16,
        seam_refine_kernel: int = 3,
        enable_inv_stem: bool = False,
        inv_stem_ch: int = 64,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.patch_f = patch_f
        self.patch_t = patch_t
        self.n_patches_f = n_patches_f
        self.n_patches_t = n_patches_t

        # Pre-unembed per-token MLP refiners (mirror of the tokenizer's).
        # 2026-05-19: bumped 4 → 12 to widen the per-modality reconstruction
        # capacity around the d_model=256 bottleneck — see
        # SpectrogramTokenizer comment in tokenizers/spectrogram.py.
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

        # Inverse of the tokenizer's patch Conv2d.
        self.patch_unembed = nn.ConvTranspose2d(
            d_model,
            n_channels,
            kernel_size=(patch_f, patch_t),
            stride=(patch_f, patch_t),
        )

        # Optional zero-init seam-refine convolution applied to the
        # post-unembed spectrogram. Matches VideoOutputHead's
        # enable_seam_refine in shape and intent: every kernel can
        # reach across the (patch_f, patch_t) patch boundary to
        # smooth out the patch-grid checkerboard that the K-step
        # rollout amplifies in Stage 2. Last conv zero-init keeps
        # the head an exact identity at construction so a Stage 1
        # checkpoint loads + behaves identically.
        #
        # Default OFF so Stage 1 and any consumer that doesn't want
        # the extra params is unaffected — only Stage 2 trainers
        # should pass `enable_seam_refine=True`.
        # Parameterized (2026-06-12) — defaults (16, 3) match the
        # original block exactly so running Stage 2 chains are
        # unaffected; the spec-fix fine-tune passes wider values.
        self.enable_seam_refine = bool(enable_seam_refine)
        if self.enable_seam_refine:
            k = int(seam_refine_kernel)
            self.refine_block = nn.Sequential(
                nn.Conv2d(
                    n_channels, seam_refine_hidden_ch,
                    kernel_size=k, padding=k // 2,
                ),
                nn.GELU(),
                nn.Conv2d(
                    seam_refine_hidden_ch, n_channels,
                    kernel_size=k, padding=k // 2,
                ),
            )
            nn.init.zeros_(self.refine_block[-1].weight)
            nn.init.zeros_(self.refine_block[-1].bias)

        # OPT-IN inv_stem branch (2026-06-12) — the fast-TS head's
        # deconv→inv_stem pattern ported to spectrograms. Motivation:
        # FastTimeSeriesHead lifts each token to a 64-ch feature map at
        # SAMPLE resolution and lets two small convs compute the final
        # value from a local feature neighbourhood — and fast-TS does
        # NOT mean-collapse. The spectrogram head's single per-patch
        # linear map (ConvTranspose2d straight to n_channels) must
        # instead encode all within-patch structure in one projection,
        # and demonstrably collapses to the per-bin mean. This branch
        # adds the missing feature-space decode:
        #   tokens → ConvTranspose2d(d_model → inv_stem_ch, k=stride=patch)
        #          → GELU → Conv2d(3×3) → GELU → Conv2d(3×3) → n_channels
        # applied RESIDUALLY on top of the existing patch_unembed output.
        # The 3×3 convs operate at (freq-bin, time-frame) resolution and
        # straddle patch boundaries at feature level. Last conv
        # zero-init → exact identity at load; old checkpoints
        # warm-start bit-identically.
        self.enable_inv_stem = bool(enable_inv_stem)
        if self.enable_inv_stem:
            self.inv_stem_unembed = nn.ConvTranspose2d(
                d_model,
                inv_stem_ch,
                kernel_size=(patch_f, patch_t),
                stride=(patch_f, patch_t),
            )
            self.inv_stem = nn.Sequential(
                nn.GELU(),
                nn.Conv2d(inv_stem_ch, inv_stem_ch, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(inv_stem_ch, n_channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.inv_stem[-1].weight)
            nn.init.zeros_(self.inv_stem[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n_tokens, d_model) -> (B, n_channels, freq_bins,
        n_patches_t * patch_t)``."""
        B = tokens.shape[0]
        for block in self.refine:
            tokens = tokens + block(tokens)
        # (B, n_tokens, d_model) -> (B, d_model, n_patches_f, n_patches_t).
        # The flatten order in the tokenizer is (n_patches_f, n_patches_t)
        # row-major (n_patches_f slow, n_patches_t fast), so we reshape
        # back into the same order here.
        x = tokens.transpose(1, 2).reshape(
            B, self.d_model, self.n_patches_f, self.n_patches_t
        )
        out = self.patch_unembed(x)              # (B, C, F, T_trunc)
        if self.enable_inv_stem:
            # Feature-space decode at bin/frame resolution, residual.
            out = out + self.inv_stem(self.inv_stem_unembed(x))
        if self.enable_seam_refine:
            out = out + self.refine_block(out)   # zero at init → no-op
        return out


class SpectroFreqWarpHead(nn.Module):
    """Predict a smooth frequency-axis shift field and *slide* the input
    spectrogram along frequency by it (mode-DRIFT fix for the persistence
    anchor).

    Motivation. A persistence anchor ``pred = input + delta`` copies the input
    window's mode ridge, so on a *drifting* mode the ridge stays at the OLD
    frequency — the visible "shifted mode" artifact. Moving a sharp ridge with
    an additive ``delta`` is expensive under MAE (broadband dominates the loss,
    so the head parks at ``delta≈0`` = persistence). Instead this head predicts
    a small, smooth per-``(channel, freq, time)`` shift field ``s`` and warps
    the input ridge to its forecast frequency via ``grid_sample`` — a cheap,
    bounded prediction that actually moves the peak. Amplitude comes from the
    real input (VISIBLE by construction, ``tvr≈1``). The within-window time
    tilt of the input ridge carries the drift rate, so a single window suffices
    to extrapolate one step ahead.

    Zero-init predictor → ``s=0`` → identity warp → exact persistence at
    construction (warm-start / rollout safe). The shift is predicted at
    patch resolution and bilinearly upsampled to ``(freq_bins, T)`` so the
    field is smooth (no per-bin jitter that would smear the ridge), then
    bounded to ``±max_shift_bins`` by ``tanh``.

    Parameters mirror :class:`SpectrogramOutputHead`'s patch layout so the
    token slice reshape matches the tokenizer's ``(n_patches_f, n_patches_t)``
    row-major flatten.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        n_patches_f: int,
        n_patches_t: int,
        freq_bins: int,
        trunc_t: int,
        max_shift_bins: float = 8.0,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_patches_f = n_patches_f
        self.n_patches_t = n_patches_t
        self.freq_bins = freq_bins
        self.trunc_t = trunc_t
        self.max_shift = float(max_shift_bins)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_channels)
        # Zero-init → shift field is 0 → identity warp → exact persistence at
        # construction (a from-scratch or warm-started model starts at the
        # visible-correct-frequency floor and learns the drift correction).
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _shift_field(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, n_tok, d_model) -> (B, C, freq_bins, trunc_t)`` in bins."""
        B = tokens.shape[0]
        s = self.proj(self.norm(tokens))          # (B, n_tok, C)
        # n_tok flatten is (n_patches_f slow, n_patches_t fast) — mirror it.
        s = s.transpose(1, 2).reshape(
            B, self.n_channels, self.n_patches_f, self.n_patches_t
        )
        s = F.interpolate(
            s, size=(self.freq_bins, self.trunc_t),
            mode="bilinear", align_corners=False,
        )
        return self.max_shift * torch.tanh(s)     # (B, C, F, T)

    def forward(
        self, tokens: torch.Tensor, input_spectro: torch.Tensor
    ) -> torch.Tensor:
        """Warp ``input_spectro`` ``(B, C, F, T)`` along freq by the predicted
        shift field. ``output[f] = input[f - s(f)]`` (positive shift → ridge
        moves to higher frequency)."""
        B, C, Fb, T = input_spectro.shape
        shift = self._shift_field(tokens)                    # (B, C, F, T)
        # grid_sample in fp32 (bf16 grids are unsupported / lossy on ROCm).
        x = input_spectro.reshape(B * C, 1, Fb, T).float()
        s = shift.reshape(B * C, Fb, T).float()
        ff = torch.arange(Fb, device=x.device, dtype=torch.float32)
        tt = torch.arange(T, device=x.device, dtype=torch.float32)
        src_f = ff[None, :, None] - s                        # (B*C, F, T)
        gy = src_f / max(Fb - 1, 1) * 2.0 - 1.0
        gx = (tt[None, None, :] / max(T - 1, 1) * 2.0 - 1.0).expand(B * C, Fb, T)
        grid = torch.stack([gx, gy], dim=-1)                 # (B*C, F, T, 2)
        warped = F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        return warped.reshape(B, C, Fb, T).to(input_spectro.dtype)


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a flow time ``t in [0, 1]``.

    ``t`` is ``(B,)``; returns ``(B, dim)``. ``t`` is scaled by 1000 so the
    [0, 1] interval spans a useful range of the sinusoidal frequencies.
    """
    half = max(dim // 2, 1)
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * 1000.0 * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class _AdaGNResBlock2d(nn.Module):
    """``GroupNorm → SiLU → Conv3×3 → AdaGN(cond) → SiLU → Conv3×3`` residual.

    The conditioning vector (timestep + global-token embedding) drives the
    second norm's per-channel scale/shift (adaptive group-norm), injecting
    flow-time + window context at every spatial resolution.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(math.gcd(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(math.gcd(8, out_ch), out_ch, affine=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(cond_dim, 2 * out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


def _sinusoidal_freq_pe(n_ch: int, freq_bins: int) -> torch.Tensor:
    """Fixed sinusoidal positional embedding over the FREQUENCY axis →
    ``(n_ch, freq_bins)``. The velocity net's 2-D convs are translation-
    equivariant in frequency, so without an absolute-frequency coordinate they
    cannot place a mode at a specific frequency. Concatenating this PE at the
    stem supplies that coordinate. Deterministic → regenerated at construction
    (not persisted), so eval reconstructs it from ``freq_pe_ch`` + ``freq_bins``.
    """
    if n_ch <= 0 or freq_bins <= 0:
        return torch.zeros(max(0, n_ch), max(0, freq_bins))
    pos = torch.arange(freq_bins, dtype=torch.float32).unsqueeze(0)   # (1, F)
    idx = torch.arange(n_ch // 2, dtype=torch.float32).unsqueeze(1)   # (n_ch//2, 1)
    div = torch.exp(-math.log(10000.0) * (2.0 * idx) / n_ch)          # (n_ch//2, 1)
    ang = pos * div                                                   # (n_ch//2, F)
    pe = torch.zeros(n_ch, freq_bins)
    pe[0::2] = torch.sin(ang)
    pe[1::2] = torch.cos(ang)
    return pe


class _CondVelocityUNet(nn.Module):
    """Small 2-downsample U-Net predicting the flow-matching velocity.

    Input is the noisy residual ``(B, C, F, T)`` concatenated with a
    spatial conditioning map; ``cond_vec`` (timestep + global tokens) drives
    AdaGN in every block. Output (same ``(B, C, F, T)``) is the velocity;
    the final conv is zero-initialised so the head is an identity (velocity
    0 → sample == mean) at construction.

    ``freq_pe_ch`` / ``time_pe_ch`` sinusoidal positional channels are
    concatenated at the stem (over the frequency / time axis respectively) so
    the convs gain absolute-frequency and absolute-time awareness — needed to
    PLACE coherent modes (freq) and time-localized bursts/chirps (time). Both
    default to 0 (legacy behaviour; checkpoints without the PE load unchanged).
    Time awareness matters most when the patch reallocation leaves few time
    tokens (e.g. (64,32) -> 3 time tokens), so the cond_map is coarse in time.
    """

    def __init__(
        self, n_channels: int, cond_ch: int, base_ch: int, cond_dim: int,
        freq_bins: int = 0, freq_pe_ch: int = 0,
        time_bins: int = 0, time_pe_ch: int = 0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        # Recompute the ResBlock activations in the backward pass instead of
        # storing them (same trade as --backbone_grad_checkpoint). Essential for
        # the VIDEO flow head, whose velocity U-Net convolves the full 120×360
        # grid (huge activations); harmless-off for the small spectro F×T nets.
        self.grad_checkpoint = bool(grad_checkpoint)
        self.freq_pe_ch = int(freq_pe_ch)
        self.time_pe_ch = int(time_pe_ch)
        if self.freq_pe_ch > 0:
            if self.freq_pe_ch % 2 != 0:
                raise ValueError(f"freq_pe_ch must be even, got {freq_pe_ch}")
            self.register_buffer(
                "freq_pe", _sinusoidal_freq_pe(self.freq_pe_ch, int(freq_bins)),
                persistent=False,
            )
        if self.time_pe_ch > 0:
            if self.time_pe_ch % 2 != 0:
                raise ValueError(f"time_pe_ch must be even, got {time_pe_ch}")
            self.register_buffer(
                "time_pe", _sinusoidal_freq_pe(self.time_pe_ch, int(time_bins)),
                persistent=False,
            )
        c1, c2 = base_ch, base_ch * 2
        self.stem = nn.Conv2d(
            n_channels + cond_ch + self.freq_pe_ch + self.time_pe_ch,
            c1, 3, padding=1,
        )
        self.enc0 = _AdaGNResBlock2d(c1, c1, cond_dim)
        self.down0 = nn.Conv2d(c1, c1, 4, stride=2, padding=1)
        self.enc1 = _AdaGNResBlock2d(c1, c2, cond_dim)
        self.down1 = nn.Conv2d(c2, c2, 4, stride=2, padding=1)
        self.mid0 = _AdaGNResBlock2d(c2, c2, cond_dim)
        self.mid1 = _AdaGNResBlock2d(c2, c2, cond_dim)
        self.up1 = _AdaGNResBlock2d(c2 + c2, c1, cond_dim)
        self.up0 = _AdaGNResBlock2d(c1 + c1, c1, cond_dim)
        self.out_norm = nn.GroupNorm(math.gcd(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, n_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x: torch.Tensor, cond_map: torch.Tensor, cond_vec: torch.Tensor
    ) -> torch.Tensor:
        parts = [x, cond_map]
        if self.freq_pe_ch > 0 or self.time_pe_ch > 0:
            B, _, Fdim, T = x.shape
        if self.freq_pe_ch > 0:
            pe = self.freq_pe                          # (freq_pe_ch, freq_bins)
            if pe.shape[-1] != Fdim:                   # defensive: freq mismatch
                pe = F.interpolate(pe.unsqueeze(0), size=Fdim,
                                   mode="linear", align_corners=False).squeeze(0)
            parts.append(
                pe.to(x.dtype)[None, :, :, None].expand(B, self.freq_pe_ch, Fdim, T)
            )
        if self.time_pe_ch > 0:
            pt = self.time_pe                          # (time_pe_ch, time_bins)
            if pt.shape[-1] != T:                      # defensive: time mismatch
                pt = F.interpolate(pt.unsqueeze(0), size=T,
                                   mode="linear", align_corners=False).squeeze(0)
            parts.append(
                pt.to(x.dtype)[None, :, None, :].expand(B, self.time_pe_ch, Fdim, T)
            )
        def _cp(mod, *a):                                  # checkpoint ResBlocks
            if self.grad_checkpoint and self.training:
                return torch_ckpt.checkpoint(mod, *a, use_reentrant=False)
            return mod(*a)
        h = self.stem(torch.cat(parts, dim=1))
        e0 = _cp(self.enc0, h, cond_vec)
        e1 = _cp(self.enc1, self.down0(e0), cond_vec)
        m = _cp(self.mid1, _cp(self.mid0, self.down1(e1), cond_vec), cond_vec)
        u1 = F.interpolate(m, size=e1.shape[-2:], mode="nearest")
        u1 = _cp(self.up1, torch.cat([u1, e1], dim=1), cond_vec)
        u0 = F.interpolate(u1, size=e0.shape[-2:], mode="nearest")
        u0 = _cp(self.up0, torch.cat([u0, e0], dim=1), cond_vec)
        return self.out_conv(F.silu(self.out_norm(u0)))


class SpectrogramFlowHead(nn.Module):
    """Generative spectrogram head — rectified flow over a deterministic mean.

    The existing :class:`SpectrogramOutputHead` is reused verbatim as the
    deterministic **mean** branch ``μ(tokens)``; a conditioned velocity U-Net
    models the **residual** ``r = target − μ`` (standardised per freq-bin) via
    conditional flow matching. Deterministic regression converges to the
    conditional mean of partly-stochastic mode structure → blurry envelope;
    a generative residual can be sharp in any single draw, recovering modes.

    Training (``self.training``): ``forward(tokens)`` returns ``μ`` only; the
    flow loss is computed separately by :meth:`flow_loss` (which exercises the
    velocity net every step → all params get grads, DDP-safe). Eval:
    ``forward(tokens)`` returns ``μ + σ·sample`` via a short Euler ODE, the
    SAME ``(B, C, F, T)`` contract as the deterministic head — a drop-in for
    ``decode()`` / rollout / the animation eval path.

    ``sigma_pb`` (C, F) is a buffer the trainer fills once from the per-bin
    stats (so quiet, mode-carrying bins get unit-scale velocity targets and
    the residual cannot collapse to zero); it is saved in the checkpoint, so
    eval reconstructs it automatically. Defaults to ones (no standardisation).
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        patch_f: int,
        patch_t: int,
        n_patches_f: int,
        n_patches_t: int,
        flow_base_ch: int = 64,
        flow_sample_steps: int = 6,
        flow_lambda: float = 1.0,
        flow_freq_pe_ch: int = 0,
        flow_time_pe_ch: int = 0,
        enable_seam_refine: bool = False,
        seam_refine_hidden_ch: int = 16,
        seam_refine_kernel: int = 3,
        enable_inv_stem: bool = False,
        inv_stem_ch: int = 64,
        enable_mask: bool = False,
        mask_hidden_ch: int = 64,
        enable_input_cond: bool = False,
        enable_input_feat: bool = False,
        residual_anchor: bool = False,
    ) -> None:
        super().__init__()
        # residual_anchor: the "mean" is the REAL input window (supplied by the
        # model's persistence-anchor block: pred = input + head_output), NOT a
        # learned mean_head. The flow then models the sampled residual
        # (target - input) — the mode's stochastic amplitude growth — so a single
        # draw is SHARP and FULL-amplitude (input floor guarantees tvr~1; the
        # sample adds the growth). Fixes the prior flow's dampening (tvr 0.23),
        # where the flow modelled residual-over-a-DAMPENED-mean and a broadband-
        # dominated velocity loss produced smooth samples. No mean_head → no
        # unused DDP params. The trainer must set spec_persistence_anchor so the
        # anchor block adds the input.
        self.residual_anchor = bool(residual_anchor)
        self.mean_head = None if self.residual_anchor else SpectrogramOutputHead(
            n_channels=n_channels,
            d_model=d_model,
            patch_f=patch_f,
            patch_t=patch_t,
            n_patches_f=n_patches_f,
            n_patches_t=n_patches_t,
            enable_seam_refine=enable_seam_refine,
            seam_refine_hidden_ch=seam_refine_hidden_ch,
            seam_refine_kernel=seam_refine_kernel,
            enable_inv_stem=enable_inv_stem,
            inv_stem_ch=inv_stem_ch,
        )
        self.n_channels = n_channels
        self.d_model = d_model
        self.n_patches_f = n_patches_f
        self.n_patches_t = n_patches_t
        self.freq_bins = patch_f * n_patches_f
        self.trunc_t = patch_t * n_patches_t
        self.flow_sample_steps = int(flow_sample_steps)
        self.flow_lambda = float(flow_lambda)

        cond_ch = int(flow_base_ch)
        cond_dim = int(flow_base_ch) * 4
        self.cond_dim = cond_dim
        self.cond_proj = nn.Conv2d(d_model, cond_ch, 1)
        self.token_global = nn.Sequential(
            nn.Linear(d_model, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.velocity = _CondVelocityUNet(
            n_channels, cond_ch, int(flow_base_ch), cond_dim,
            freq_bins=self.freq_bins, freq_pe_ch=int(flow_freq_pe_ch),
            time_bins=self.trunc_t, time_pe_ch=int(flow_time_pe_ch),
        )
        self.register_buffer(
            "sigma_pb", torch.ones(n_channels, self.freq_bins)
        )

        # ── Mode-MASK branch (Plan B, 2026-06-30) ────────────────────────────
        # A SEPARATE decode that predicts the binarized mode field (per
        # (C, F, T) sigmoid). The overfit-prediction test proved BOTH μ (MAE)
        # and the flow sample (velocity MSE) collapse to a smooth envelope —
        # because both are L2, whose optimum is the conditional mean of
        # partly-stochastic modes. A segmentation target trained with
        # soft-Dice/BCE has NO mean-seeking optimum, and mode PRESENCE is the
        # predictable quantity (survival τ½≈201 ms). The μ head keeps doing the
        # (smooth, correct) magnitude regression; this branch supplies the sharp
        # mode locations; render fuses μ with the PREDICTED mask (no GT → genuine
        # forecast). Mirrors the inv_stem feature-space decode (ConvTranspose to
        # bin/frame resolution → 3×3 convs straddling patch seams). Independent of
        # μ/flow → no objective conflict (the struct-loss-on-μ approach failed
        # precisely because it asked μ to be smooth AND mode-shaped at once).
        # enable_input_feat: give the mask decode the INPUT-window mode mask as
        # LEARNABLE feature channels (concat), NOT a fixed logit-add (that was
        # enable_input_cond, which collapsed). The convs learn to combine the
        # observed input modes with the forecast-token features → can copy
        # persistence AND predict changes. Bypasses the forecast-token
        # mode-collapse bottleneck (the tokens may not carry modes; the input
        # does). AR-compatible: in rollout the "input mask" is the previous
        # step's predicted mask.
        self.enable_input_feat = bool(enable_input_feat)
        self.enable_mask = bool(enable_mask)
        if self.enable_mask:
            mh = int(mask_hidden_ch)
            self.mask_unembed = nn.ConvTranspose2d(
                d_model, mh,
                kernel_size=(patch_f, patch_t),
                stride=(patch_f, patch_t),
            )
            if self.enable_input_feat:
                # input_feat: split the GELU out so mask_logits can concat the
                # input-mode features between it and the decode convs (first conv
                # takes mh + n_channels).
                self.mask_pre = nn.GELU()
                self.mask_decode = nn.Sequential(
                    nn.Conv2d(mh + n_channels, mh, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(mh, n_channels, kernel_size=3, padding=1),
                )
            else:
                # ORIGINAL structure — kept identical so pre-input_feat mask
                # checkpoints (e.g. the objective-fix POC) load without key
                # mismatch. GELU is inside mask_decode here.
                self.mask_decode = nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(mh, mh, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(mh, n_channels, kernel_size=3, padding=1),
                )
            # Final-logit bias:
            #  • WITHOUT input-conditioning → sparse prior (modes ~5 % of bins):
            #    bias −3 (p≈0.047) so the initial mask is near-empty and doesn't
            #    flood the Dice/BCE before it learns.
            #  • WITH input-conditioning → the persistence `prior` provides the
            #    baseline (logits += gain·logit(prior)); a −3 bias would FIGHT a
            #    soft prior (~0.7 → logit 0.85 can't overcome −3 → mask stays
            #    empty). Bias 0 → at init logits ≈ logit(prior) → predicted mask
            #    ≈ the input modes (persistence), which is the whole point.
            if enable_input_cond:
                # ZERO-INIT the final conv (weight + bias): at init decode=0 → the
                # persistence prior FULLY controls the mask → maskdice starts at the
                # ~0.66 persistence ceiling (measured: frame-aligned input↔output
                # mode overlap). Without this, the fresh decode on real d1024 tokens
                # produces large logits that OVERRIDE the ±6.9 prior → maskdice stuck
                # ~0.08 (job 4923342). Zero-init is safe: the final conv's WEIGHT grad
                # ∝ the input activation (nonzero) → it turns on and learns the
                # residual (fades/drift), then the earlier convs follow.
                nn.init.zeros_(self.mask_decode[-1].weight)
                nn.init.zeros_(self.mask_decode[-1].bias)
            else:
                # No input-cond → sparse prior (modes ~5 % of bins): bias −3
                # (p≈0.047) so the initial mask is near-empty and doesn't flood the
                # Dice/BCE. Weight default init → earlier convs get gradients from
                # step 0.
                nn.init.constant_(self.mask_decode[-1].bias, -3.0)

        # ── Input-conditioning / persistence (recurrence-ready, 2026-06-30) ──
        # The overfit tests showed the backbone forecast tokens do NOT carry the
        # specific mode locations (mask-from-tokens plateaus at chance, λ-inde-
        # pendent). But ECE modes PERSIST (survival τ½≈201 ms ≫ the 50 ms
        # horizon), so the mode field of the PAST (input window, or in Stage-2
        # rollout the PREVIOUS predicted mask) is a strong prior for the future.
        # mask_logits adds `gain · logit(prior)` — the head only learns the
        # RESIDUAL (how modes change) on top of persistence. `prior` is pluggable:
        #   Stage 1  → the input-window mode mask (the trainer computes it);
        #   Stage 2  → the previous rollout step's predicted mask (a mask
        #              recurrence with the learnable decay `gain` → modes
        #              propagate near-horizon, fade past τ½ = physically correct).
        # This is a HEAD-level skip from observed data — NO extra backbone tokens,
        # NO token-count change, warm-start-safe (gain inits fresh).
        self.enable_input_cond = bool(enable_input_cond)
        if self.enable_input_cond:
            # scalar persistence gain, init 1.0 (strong copy-forward at start;
            # training/rollout can down-weight it). NOT tied to enable_mask below
            # — asserted at construction that mask is on (input-cond needs it).
            self.mask_prior_gain = nn.Parameter(torch.ones(1))

    def mask_logits(
        self, tokens: torch.Tensor, prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(B, n_tok, d_model) → (B, n_channels, freq_bins, trunc_t)`` mode-mask
        logits. Parallel decode from the same backbone tokens as μ; the trainer
        turns these into a probability and scores them against the
        production-binarized GT with soft-Dice + BCE.

        ``prior`` (optional, broadcastable to the logits, values in [0, 1]) is the
        persistence prior — the input-window mode mask (Stage 1) or the previous
        rollout step's predicted mask (Stage 2). When input-conditioning is on and
        a prior is given, ``gain · logit(prior)`` is added so the head defaults to
        copying modes forward and learns only the residual change."""
        B = tokens.shape[0]
        x = tokens.transpose(1, 2).reshape(
            B, self.d_model, self.n_patches_f, self.n_patches_t
        )
        if self.enable_input_feat:
            # concat the input-window mode mask as LEARNABLE feature channels →
            # the decode convs learn to combine observed modes + token dynamics
            # (copy persistence and/or predict changes). Bypasses the collapsed
            # forecast tokens. bf16-safe (no logit; a plain concat).
            feat = self.mask_pre(self.mask_unembed(x))      # (B, mh, F, T)
            if prior is not None:
                feat = torch.cat([feat, prior.to(feat.dtype)], dim=1)
            else:                                            # DDP/robustness: pad
                feat = torch.cat(
                    [feat, feat.new_zeros(feat.shape[0], self.n_channels,
                                          *feat.shape[2:])], dim=1)
            logits = self.mask_decode(feat)
        else:
            logits = self.mask_decode(self.mask_unembed(x))  # original path
        if self.enable_input_cond and prior is not None:
            # Compute the prior logit in fp32 with a SAFE margin. Under bf16/fp16
            # autocast, 1−1e-4 rounds to 1.0 → logit(1.0)=+inf → NaN (this NaN'd
            # job 4922044). 1e-3 margin → logit bounded ±6.9 (still flips the
            # sigmoid fully); fp32 keeps it finite, then cast back to logits.dtype.
            p = prior.float().clamp(1e-3, 1.0 - 1e-3)
            logits = logits + (self.mask_prior_gain * torch.logit(p)).to(
                logits.dtype
            )
        return logits

    def mask_prob(
        self, tokens: torch.Tensor, prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sigmoid of :meth:`mask_logits` — the predicted mode-mask probability
        ``(B, C, F, T)`` used at render (fuse with μ) and as the Stage-2 rollout
        recurrence's fed-forward prior."""
        return torch.sigmoid(self.mask_logits(tokens, prior=prior))

    def set_sigma_pb(self, sigma: torch.Tensor) -> None:
        """Set the per-(channel, freq) residual std (C, F). Called once by the
        trainer from the per-bin stats; persisted in the checkpoint."""
        with torch.no_grad():
            self.sigma_pb.copy_(sigma.to(self.sigma_pb).reshape_as(self.sigma_pb))

    def _cond(self, tokens: torch.Tensor, t: torch.Tensor):
        """tokens (B, n_tok, d_model), t (B,) → (cond_map (B,cond_ch,F,T),
        cond_vec (B,cond_dim))."""
        B = tokens.shape[0]
        tmap = tokens.transpose(1, 2).reshape(
            B, self.d_model, self.n_patches_f, self.n_patches_t
        )
        cond_map = self.cond_proj(tmap)
        cond_map = F.interpolate(
            cond_map, size=(self.freq_bins, self.trunc_t),
            mode="bilinear", align_corners=False,
        )
        g = self.token_global(tokens.mean(dim=1))
        cond_vec = self.t_mlp(_timestep_embedding(t, self.cond_dim).to(g.dtype)) + g
        return cond_map, cond_vec

    def flow_loss(
        self,
        tokens: torch.Tensor,
        mu: torch.Tensor,
        target: torch.Tensor,
        mask,
        band_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Rectified-flow velocity MSE on the standardised residual. ``mu`` is
        detached (the mean is anchored by its own MAE term, or — in
        ``residual_anchor`` mode — is the real input window supplied by the
        anchor block, so the residual is ``target - input``).

        ``band_weight`` (C, F) up-weights the coherent-mode band in the velocity
        MSE (broadcast (1,C,F,1)). Without it the velocity loss is broadband-
        dominated → smooth velocity → smooth (dampened) samples; weighting the
        mode band forces the flow to model the sharp mode residual so a draw is
        sharp, not blurred."""
        B = mu.shape[0]
        sigma = self.sigma_pb[None, :, :, None].float()
        x1 = (target.float() - mu.detach().float()) / sigma          # residual
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=mu.device, dtype=torch.float32)
        tb = t[:, None, None, None]
        xt = (1.0 - tb) * x0 + tb * x1
        cond_map, cond_vec = self._cond(tokens, t)
        v = self.velocity(xt.to(cond_map.dtype), cond_map, cond_vec)
        err = (v.float() - (x1 - x0)) ** 2                           # target u
        if band_weight is not None:
            bw = band_weight.to(err.dtype)
            err = err * bw.view(1, bw.shape[0], bw.shape[1], 1)
        if mask is not None:
            m = mask.to(err.dtype).expand_as(err)
            return (err * m).sum() / m.sum().clamp_min(1.0)
        return err.mean()

    @torch.no_grad()
    def sample(
        self, tokens: torch.Tensor, mu: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """μ + σ · Euler-ODE(noise → residual). Returns (B, C, F, T).

        ``noise`` (optional, shape == ``mu``): the initial x0. Pass the SAME
        draw across all K rollout steps for temporally coherent block-mode
        frames (the residual then evolves only with the conditioning, not the
        noise). ``None`` → a fresh independent draw (can flicker frame-to-frame).
        """
        sigma = self.sigma_pb[None, :, :, None].float()
        if noise is None:
            x = torch.randn(mu.shape, device=mu.device, dtype=torch.float32)
        else:
            x = noise.to(device=mu.device, dtype=torch.float32)
        n = max(1, self.flow_sample_steps)
        for i in range(n):
            t = torch.full((mu.shape[0],), i / n,
                           device=mu.device, dtype=torch.float32)
            cond_map, cond_vec = self._cond(tokens, t)
            v = self.velocity(x.to(cond_map.dtype), cond_map, cond_vec).float()
            x = x + (1.0 / n) * v
        return mu + (sigma * x).to(mu.dtype)

    def forward(
        self, tokens: torch.Tensor, noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Training: returns μ (flow loss computed via :meth:`flow_loss`).
        Eval: returns a sampled spectrogram (μ + σ·residual); ``noise`` lets
        the caller share one draw across rollout steps for coherence.

        In ``residual_anchor`` mode there is no mean_head: the head returns the
        RESIDUAL only (0 in train; σ·Euler sample in eval), and the model's
        persistence-anchor block adds the real input window as the mean →
        ``pred = input + sampled_residual``."""
        if self.residual_anchor:
            B = tokens.shape[0]
            zero = tokens.new_zeros(
                B, self.n_channels, self.freq_bins, self.trunc_t
            )
            if self.training or getattr(self, "render_mean", False):
                return zero               # anchor → pred = input (+ grain only at eval)
            return self.sample(tokens, zero, noise=noise)   # σ·Euler residual
        mu = self.mean_head(tokens)
        if self.training or getattr(self, "render_mean", False):
            return mu                                   # render_mean → deterministic μ (no grain)
        return self.sample(tokens, mu, noise=noise)


class VideoFlowHead(nn.Module):
    """Generative video head — rectified flow over a deterministic mean.

    Video analog of :class:`SpectrogramFlowHead` for the tangtv camera. The
    deterministic :class:`VideoOutputHead` (resize-conv, option B) is the mean
    branch ``μ(tokens)`` → ``(B, T, C, H, W)``; a conditioned velocity net models
    the residual ``r = target − μ`` (per-folded-channel standardised) via
    conditional flow matching, sampled at eval (option A). ``n_frames`` is small
    (3), so ``(C, T)`` are FOLDED into ``C·T`` channels and the existing 2-D
    :class:`_CondVelocityUNet` runs over the ``(H, W)`` frame grid — its freq/time
    positional embeddings become the spatial ``(H, W)`` PE (option D). Same
    ``(B, T, C, H, W)`` contract as VideoOutputHead, so it's a drop-in decoder.
    """

    def __init__(
        self,
        n_channels: int = 2,
        n_frames: int = 3,
        patch_size: tuple[int, int, int] = (3, 12, 12),
        d_model: int = 256,
        spatial_size: tuple[int, int] = (120, 360),
        flow_base_ch: int = 48,
        flow_sample_steps: int = 6,
        flow_lambda: float = 1.0,
        flow_h_pe_ch: int = 16,
        flow_w_pe_ch: int = 16,
        resize_conv_hidden_ch: int = 64,
        sigma_spatial: bool = False,
    ) -> None:
        super().__init__()
        T_p, H_p, W_p = (int(p) for p in patch_size)
        H, W = int(spatial_size[0]), int(spatial_size[1])
        self.n_channels = n_channels
        self.n_frames = n_frames
        self.H, self.W = H, W
        self.ct = n_channels * n_frames              # folded velocity channels
        self.n_t = n_frames // T_p
        self.n_h = H // H_p
        self.n_w = W // W_p
        self.d_model = d_model
        self.flow_sample_steps = int(flow_sample_steps)
        self.flow_lambda = float(flow_lambda)
        self.mean_head = VideoOutputHead(
            n_channels=n_channels, n_frames=n_frames, patch_size=patch_size,
            d_model=d_model, spatial_size=spatial_size, decoder="resize_conv",
            resize_conv_hidden_ch=resize_conv_hidden_ch,
        )
        cond_ch = int(flow_base_ch)
        cond_dim = int(flow_base_ch) * 4
        self.cond_dim = cond_dim
        self.cond_proj = nn.Conv2d(d_model, cond_ch, 1)
        self.token_global = nn.Sequential(
            nn.Linear(d_model, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.velocity = _CondVelocityUNet(
            self.ct, cond_ch, int(flow_base_ch), cond_dim,
            freq_bins=H, freq_pe_ch=int(flow_h_pe_ch),     # H-axis spatial PE
            time_bins=W, time_pe_ch=int(flow_w_pe_ch),     # W-axis spatial PE
            grad_checkpoint=True,                          # full-res grid → essential
        )
        # Residual scale for the flow. Default per-folded-channel scalar
        # (C·T,1,1). sigma_spatial → per-pixel (C·T,H,W): noise is injected only
        # where the residual actually varies (quiet background stays clean),
        # the video analog of the spectrogram's per-frequency σ.
        if sigma_spatial:
            self.register_buffer("sigma_pb", torch.ones(self.ct, H, W))
        else:
            self.register_buffer("sigma_pb", torch.ones(self.ct, 1, 1))

    def set_sigma_pb(self, sigma: torch.Tensor) -> None:
        """Per-folded-channel residual std → (C·T,1,1) or (C·T,H,W) buffer (fold
        order T-major: index t*C + c). Persisted in the checkpoint."""
        with torch.no_grad():
            self.sigma_pb.copy_(sigma.to(self.sigma_pb).reshape_as(self.sigma_pb))

    def _fold(self, v: torch.Tensor) -> torch.Tensor:      # (B,T,C,H,W)->(B,T*C,H,W)
        B = v.shape[0]
        return v.reshape(B, self.n_frames * self.n_channels, self.H, self.W)

    def _unfold(self, v: torch.Tensor) -> torch.Tensor:    # (B,T*C,H,W)->(B,T,C,H,W)
        B = v.shape[0]
        return v.reshape(B, self.n_frames, self.n_channels, self.H, self.W)

    def _cond(self, tokens: torch.Tensor, t: torch.Tensor):
        """tokens (B,n_tok,d), t (B,) → cond_map (B,cond_ch,H,W), cond_vec."""
        B = tokens.shape[0]
        tmap = tokens.transpose(1, 2).reshape(
            B, self.d_model, self.n_t, self.n_h, self.n_w
        )[:, :, 0]                                   # n_t==1 → (B, d, n_h, n_w)
        cond_map = self.cond_proj(tmap)
        cond_map = F.interpolate(
            cond_map, size=(self.H, self.W), mode="bilinear", align_corners=False,
        )
        g = self.token_global(tokens.mean(dim=1))
        cond_vec = self.t_mlp(_timestep_embedding(t, self.cond_dim).to(g.dtype)) + g
        return cond_map, cond_vec

    def flow_loss(self, tokens, mu, target, mask=None):
        """Rectified-flow velocity MSE on the standardised residual. mu/target:
        (B,T,C,H,W). mask: (B,T,C) present-flags (missing channels/frames) or None."""
        muf = self._fold(mu).detach().float()
        tgf = self._fold(target).float()
        B = muf.shape[0]
        sigma = self.sigma_pb[None].float()          # (1, C·T, 1, 1)
        x1 = (tgf - muf) / sigma
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=mu.device, dtype=torch.float32)
        tb = t[:, None, None, None]
        xt = (1.0 - tb) * x0 + tb * x1
        cond_map, cond_vec = self._cond(tokens, t)
        v = self.velocity(xt.to(cond_map.dtype), cond_map, cond_vec)
        err = (v.float() - (x1 - x0)) ** 2
        if mask is not None:
            m = mask.reshape(B, self.n_frames * self.n_channels, 1, 1).to(err.dtype)
            m = m.expand_as(err)
            return (err * m).sum() / m.sum().clamp_min(1.0)
        return err.mean()

    @torch.no_grad()
    def sample(self, tokens, mu, noise=None):
        """μ + σ·Euler-ODE(noise → residual). Returns (B,T,C,H,W)."""
        muf = self._fold(mu).float()
        sigma = self.sigma_pb[None].float()
        x = torch.randn_like(muf) if noise is None else self._fold(noise).float()
        n = max(1, self.flow_sample_steps)
        for i in range(n):
            t = torch.full((muf.shape[0],), i / n,
                           device=mu.device, dtype=torch.float32)
            cond_map, cond_vec = self._cond(tokens, t)
            v = self.velocity(x.to(cond_map.dtype), cond_map, cond_vec).float()
            x = x + (1.0 / n) * v
        return self._unfold(muf + sigma * x).to(mu.dtype)

    def forward(self, tokens, noise=None):
        """Training: returns μ (flow loss via :meth:`flow_loss`). Eval: a sampled
        video (μ + σ·residual), same (B,T,C,H,W) contract as VideoOutputHead."""
        mu = self.mean_head(tokens)
        if self.training or getattr(self, "render_mean", False):
            return mu                                   # render_mean → deterministic μ (no grain)
        return self.sample(tokens, mu, noise=noise)


class SlowTimeSeriesCodeHead(nn.Module):
    """Discrete-code slow-TS (Thomson/CER/MSE) head — analog of
    :class:`FastTimeSeriesCodeHead`. A FROZEN adversarial-FSQ codec + a NEW per-token
    code-prediction head over the backbone tokens. One token per channel; ``decode``/
    ``forward`` return ``(B, C, WIN)``. The codec is trained in the dataset-standardized
    space, so ``encode_target`` takes the dataset target as-is (no re-normalization).
    Codec INJECTED already-frozen (``load_frozen_slowts_codec``) → no ``quantizers``
    import here (avoids the circular import with ``SlowTimeSeriesHead``)."""

    def __init__(
        self,
        codec: nn.Module,
        d_model: int,
        pred_hidden: int = 512,
        pred_layers: int = 2,
        sample_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.n_tok = codec.n_tok
        self.dim = codec.dim
        self.levels = codec.levels
        self.sample_temperature = float(sample_temperature)
        trunk: list = []
        h = d_model
        for _ in range(max(1, pred_layers)):
            trunk += [nn.Linear(h, pred_hidden), nn.GELU()]
            h = pred_hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.ModuleList([nn.Linear(h, self.levels) for _ in range(self.dim)])

    def code_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tok, d_model) -> (B, n_tok, dim, levels). Trainable path."""
        h = self.trunk(tokens)
        return torch.stack([hd(h) for hd in self.heads], dim=2)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, WIN) dataset-standardized -> per-dim int codes (B, n_tok, dim)."""
        return self.codec.encode_codes(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> (B, C, WIN). Frozen decoder."""
        return self.codec.decode_codes(codes)

    def sample_codes(
        self, logits: torch.Tensor, temperature: Optional[float] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        if hard:
            return logits.argmax(dim=-1)
        t = self.sample_temperature if temperature is None else float(temperature)
        probs = torch.softmax(logits / max(t, 1e-6), dim=-1)
        B, n, d, L = probs.shape
        return torch.multinomial(probs.reshape(-1, L), 1).reshape(B, n, d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode predicted codes -> (B, C, WIN). code_logits runs WITH grad (AMP fix)."""
        logits = self.code_logits(tokens)
        with torch.no_grad():
            codes = self.sample_codes(logits, hard=self.training)
            return self.codec.decode_codes(codes)


class FastTimeSeriesCodeHead(nn.Module):
    """Discrete-code fast-TS (filterscope/ELM) head — 1-D analog of
    :class:`SpectrogramCodeHead`. A FROZEN adversarial-FSQ codec (Phase 1a) + a NEW
    per-token code-prediction head over the backbone tokens.

    Why: deterministic MAE on filterscopes smooths the sharp, sparse ELM spikes
    (the signal). Here prediction is CATEGORICAL over the FSQ code levels; a sampled
    code decodes through the FROZEN codec, which renders sharp spikes from any
    plausible codes — so imperfect code-accuracy still yields spikes.

    The codec is INJECTED already-built and frozen (``load_frozen_fastts_codec``) so
    this module never imports ``quantizers`` (the codec depends on this file's
    ``FastTimeSeriesHead`` → avoids a circular import). Same contracts as
    ``SpectrogramCodeHead`` except ``decode``/``forward`` return ``(B, C, WIN)``.
    """

    def __init__(
        self,
        codec: nn.Module,
        d_model: int,
        pred_hidden: int = 512,
        pred_layers: int = 2,
        sample_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.codec = codec                       # params already requires_grad_(False)
        self.n_tok = codec.n_tok
        self.dim = codec.dim
        self.levels = codec.levels
        self.sample_temperature = float(sample_temperature)
        trunk: list = []
        h = d_model
        for _ in range(max(1, pred_layers)):
            trunk += [nn.Linear(h, pred_hidden), nn.GELU()]
            h = pred_hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.ModuleList([nn.Linear(h, self.levels) for _ in range(self.dim)])

    def code_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tok, d_model) -> (B, n_tok, dim, levels). Trainable path."""
        h = self.trunk(tokens)
        return torch.stack([hd(h) for hd in self.heads], dim=2)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, WIN) per-window-z-scored -> per-dim int codes (B, n_tok, dim)."""
        return self.codec.encode_codes(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> (B, C, WIN). Frozen decoder."""
        return self.codec.decode_codes(codes)

    def sample_codes(
        self, logits: torch.Tensor, temperature: Optional[float] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        if hard:
            return logits.argmax(dim=-1)
        t = self.sample_temperature if temperature is None else float(temperature)
        probs = torch.softmax(logits / max(t, 1e-6), dim=-1)
        B, n, d, L = probs.shape
        return torch.multinomial(probs.reshape(-1, L), 1).reshape(B, n, d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode predicted codes -> (B, C, WIN). Train: argmax; eval: sample.

        ``code_logits`` runs WITH grad here (AMP+DDP fix — see SpectrogramCodeHead)."""
        logits = self.code_logits(tokens)
        with torch.no_grad():
            codes = self.sample_codes(logits, hard=self.training)
            return self.codec.decode_codes(codes)


class SpectrogramCodeHead(nn.Module):
    """Discrete-code spectrogram head — a FROZEN adversarial-FSQ codec (Phase 1a)
    + a NEW code-prediction head over the backbone tokens.

    Why: deterministic regression (MAE / even flow's pixel-MAE anchor) collapses
    thin spectrogram modes to the conditional mean. Here prediction is
    **categorical** — the backbone predicts a distribution over the per-dim FSQ
    code levels and we sample; a sampled code decodes through the FROZEN
    adversarial decoder, which renders SHARP modes from ANY plausible codes (the
    VQ-GAN property validated in the POC). So imperfect code-accuracy still yields
    mode-bearing output — the failure mode of the continuous heads is structurally
    removed.

    The codec (encoder + FSQ + decoder) is **injected** already-built and frozen
    (``load_frozen_codec``) so this module never imports ``quantizers`` (avoids a
    circular import: the codec depends on this file's ``SpectrogramOutputHead``).

    Contracts:
      * ``forward(tokens) -> (B, C, F, T)`` — drop-in for ``model.decode()`` /
        predictions / rollout / eval. Train mode: argmax-decode (deterministic,
        gives a real reconstruction for logging-MAE + the collapse-aware TVR
        metric). Eval mode: multinomial sample (mode-diverse viz).
      * ``code_logits(tokens) -> (B, n_tok, dim, levels)`` — the TRAINABLE path.
        The class-weighted-CE loss (in the trainer) calls this on the backbone
        token slice; every prediction-head param is exercised each step → DDP-safe
        (mirrors ``SpectrogramFlowHead.flow_loss``). The frozen codec params are
        ``requires_grad=False`` → excluded from the reducer.
      * ``encode_target(spectro) -> (B, n_tok, dim)`` int — frozen; the CE targets.
      * ``decode(codes) -> (B, C, F, T)`` — frozen decoder; viz / rollout.
    """

    def __init__(
        self,
        codec: nn.Module,
        d_model: int,
        pred_hidden: int = 512,
        pred_layers: int = 2,
        sample_temperature: float = 1.0,
        bg_subtract: bool = False,
        bg_sigma: float = 8.0,
    ) -> None:
        super().__init__()
        # frozen codec (params already requires_grad_(False) by load_frozen_codec)
        self.codec = codec
        self.n_tok = codec.n_tok
        self.dim = codec.dim
        self.levels = codec.levels
        self.sample_temperature = float(sample_temperature)
        # Residual-codec marker: when the frozen codec was trained on the
        # baseline-subtracted residual (cfg["bg_subtract"]), the whole spectro
        # pathway (backbone input tokenizer + code targets + decode) must run in
        # residual space. ``forward_batch`` reads these to split the spectro
        # input/target into R = S - B before anything touches the codec. The
        # sigma matches the codec's training operator (default 8.0).
        self.bg_subtract = bool(bg_subtract)
        self.bg_sigma = float(bg_sigma)

        # Prediction head: backbone token (d_model) -> per-dim code logits.
        # Per-token MLP trunk + one linear classifier per FSQ dim (each over
        # ``levels`` classes). Kept small — the sharpness comes from the frozen
        # decoder, not this head.
        trunk: list = []
        h = d_model
        for _ in range(max(1, pred_layers)):
            trunk += [nn.Linear(h, pred_hidden), nn.GELU()]
            h = pred_hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.ModuleList(
            [nn.Linear(h, self.levels) for _ in range(self.dim)]
        )

    def code_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tok, d_model) -> (B, n_tok, dim, levels). Trainable path."""
        h = self.trunk(tokens)
        return torch.stack([hd(h) for hd in self.heads], dim=2)

    @torch.no_grad()
    def encode_target(self, spectro: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) -> per-dim int codes (B, n_tok, dim). Frozen; CE targets."""
        return self.codec.encode_codes(spectro)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> spectrogram (B, C, F, T). Frozen."""
        return self.codec.decode_codes(codes)

    def sample_codes(
        self, logits: torch.Tensor, temperature: Optional[float] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        """(B, n_tok, dim, levels) -> per-dim int codes (B, n_tok, dim)."""
        if hard:
            return logits.argmax(dim=-1)
        t = self.sample_temperature if temperature is None else float(temperature)
        probs = torch.softmax(logits / max(t, 1e-6), dim=-1)
        B, n, d, L = probs.shape
        s = torch.multinomial(probs.reshape(-1, L), 1).reshape(B, n, d)
        return s

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode predicted codes to a spectrogram (viz/logging/TVR path). Train:
        argmax-decode; eval: sample-decode.

        ``code_logits`` MUST run WITH grad here (only sampling + the frozen decode
        are ``no_grad``), even though this path's output is detached and the real
        training gradient flows through the SEPARATE ``code_logits`` call in the
        CE loss. Reason: under AMP autocast the fp32->bf16 weight cast is CACHED
        on first use; a ``no_grad`` first use (this forward, which runs before the
        loss) caches a DETACHED cast, so the with-grad ``code_logits`` in the loss
        reuses it and the fp32 pred-head params receive NO gradient -> DDP
        'unused parameter' crash under AMP. Running it with grad here makes the
        cached cast grad-connected (mirrors how the flow head's velocity net is
        first used with grad). Cheap: the orphaned graph is freed at return."""
        logits = self.code_logits(tokens)
        with torch.no_grad():
            codes = self.sample_codes(logits, hard=self.training)
            return self.codec.decode_codes(codes)


class SpectrogramMaskGITHead(nn.Module):
    """JOINT (MaskGIT-style) spectrogram code head — FROZEN FSQ codec + a
    bidirectional transformer that predicts the window's patch-codes JOINTLY,
    conditioned on the backbone tokens.

    Fixes the collapse of the *independent* per-patch head (argmax->blocky,
    sample->speckle): predicting the n_tok patch-codes independently cannot make a
    COHERENT mode (a ridge spanning patches). A bidirectional transformer over the
    patch grid makes each patch's code depend on the others. Training = BERT-style
    random masking, one parallel pass, CE on the masked patches. Inference =
    MaskGIT iterative parallel unmasking (~decode_steps passes, low temperature for
    PRECISION). Keeps the frozen codec (encode_target/decode unchanged) and the
    per-window / token-space rollout contract (this only maps backbone tokens ->
    spectrogram, exactly where SpectrogramCodeHead sits).
    """

    def __init__(
        self,
        codec: nn.Module,
        d_model: int,
        mg_dim: int = 512,
        mg_layers: int = 4,
        mg_heads: int = 8,
        decode_steps: int = 10,
        decode_temperature: float = 0.5,
        sample_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.codec = codec                       # frozen (requires_grad_(False))
        self.n_tok = codec.n_tok
        self.dim = codec.dim
        self.levels = codec.levels
        self.decode_steps = int(decode_steps)
        self.decode_temperature = float(decode_temperature)
        self.sample_temperature = float(sample_temperature)
        d = int(mg_dim)
        self.cond_proj = nn.Linear(d_model, d)   # backbone token -> conditioning
        self.code_embed = nn.Embedding(self.dim * self.levels, d)  # factored code emb
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tok, d))
        nn.init.normal_(self.pos_emb, std=0.02)
        # per-dim offset so dim d indexes its own [d*levels, (d+1)*levels) block
        self.register_buffer(
            "_dim_offset",
            (torch.arange(self.dim) * self.levels).view(1, 1, self.dim),
            persistent=False,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=mg_heads, dim_feedforward=4 * d, dropout=0.0,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(mg_layers))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleList(
            [nn.Linear(d, self.levels) for _ in range(self.dim)]
        )

    @torch.no_grad()
    def encode_target(self, spectro: torch.Tensor) -> torch.Tensor:
        """(B, C, F, T) -> per-dim int codes (B, n_tok, dim). Frozen; CE targets."""
        return self.codec.encode_codes(spectro)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> spectrogram (B, C, F, T). Frozen."""
        return self.codec.decode_codes(codes)

    def masked_logits(
        self, tokens: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """tokens (B,N,d_model) conditioning; codes (B,N,dim) known/true codes;
        mask (B,N) True where MASKED -> logits (B,N,dim,levels). Bidirectional."""
        B, N, _ = tokens.shape
        cond = self.cond_proj(tokens)
        emb = self.code_embed(codes + self._dim_offset).sum(dim=2)      # (B,N,d)
        me = self.mask_token.view(1, 1, -1)
        x = torch.where(mask.unsqueeze(-1), me, emb) + cond + self.pos_emb[:, :N]
        x = self.norm(self.transformer(x))                              # (B,N,d)
        return torch.stack([hd(x) for hd in self.heads], dim=2)         # (B,N,dim,L)

    def sample_mask(self, B: int, N: int, device) -> torch.Tensor:
        """MaskGIT cosine mask schedule -> (B,N) bool, ~cos(pi/2 u)*N masked."""
        u = torch.rand(B, device=device)
        ratio = torch.cos(0.5 * math.pi * u).clamp(1e-3, 1.0)
        n_mask = (ratio * N).round().clamp(min=1).long()               # (B,)
        r = torch.rand(B, N, device=device)
        kth = torch.gather(r.sort(dim=1).values, 1, (n_mask - 1).unsqueeze(1))
        return r <= kth                                                # (B,N) ~n_mask True

    @torch.no_grad()
    def iterative_decode(
        self, tokens: torch.Tensor, n_steps: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """MaskGIT parallel decode: all-masked -> keep most-confident -> repeat."""
        n_steps = self.decode_steps if n_steps is None else int(n_steps)
        t = self.decode_temperature if temperature is None else float(temperature)
        B, N, _ = tokens.shape
        dev = tokens.device
        codes = torch.zeros(B, N, self.dim, dtype=torch.long, device=dev)
        mask = torch.ones(B, N, dtype=torch.bool, device=dev)
        for step in range(n_steps):
            logits = self.masked_logits(tokens, codes, mask)
            probs = torch.softmax(logits / max(t, 1e-6), dim=-1)
            if t <= 1e-3:
                pred = logits.argmax(-1)
            else:
                pred = torch.multinomial(
                    probs.reshape(-1, self.levels), 1).reshape(B, N, self.dim)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1).mean(-1)  # (B,N)
            codes = torch.where(mask.unsqueeze(-1), pred, codes)
            conf = torch.where(mask, conf, torch.full_like(conf, float("inf")))
            frac = math.cos(0.5 * math.pi * (step + 1) / n_steps)      # still-masked
            n_keep_masked = int(math.floor(frac * N)) if step < n_steps - 1 else 0
            if n_keep_masked <= 0:
                mask = torch.zeros_like(mask)
            else:
                kth = conf.sort(dim=1).values[:, n_keep_masked - 1].unsqueeze(1)
                mask = conf <= kth                                     # least conf stay masked
        return codes

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Backbone tokens -> decoded spectrogram (viz/logging path). Train: cheap
        single all-masked pass (WITH grad so the AMP fp32->bf16 cast is grad-
        connected, mirroring SpectrogramCodeHead.forward) + argmax-decode. Eval:
        full MaskGIT iterative decode."""
        B, N, _ = tokens.shape
        if self.training:
            mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
            codes0 = torch.zeros(B, N, self.dim, dtype=torch.long, device=tokens.device)
            logits = self.masked_logits(tokens, codes0, mask)          # WITH grad
            with torch.no_grad():
                return self.codec.decode_codes(logits.argmax(-1))
        with torch.no_grad():
            return self.codec.decode_codes(self.iterative_decode(tokens))


class SpectrogramDescriptorHead(nn.Module):
    """FACTORIZATION (mode-descriptor) head — AUXILIARY to the main spectro head.

    Forecasts the shift-stable MODE DESCRIPTOR (channel-max, 5-40 kHz band-power
    profile, time-pooled to TCOL columns) rather than the exact spectrogram
    realization. Measured facts (2026-07-13): the descriptor persists t->t+1 at
    0.75-0.95 (vs shuffled 0.16-0.47), while exact FSQ codes do not (0.10-0.34) —
    so a model trained on the descriptor CAN forecast modes, unlike the code /
    continuous heads that mean-collapse. The main spectro head still produces the
    (B,C,F,T) texture; this auxiliary head carries the modes (rendered + gated).

      forward(tokens (B,n_tok,d))     -> descriptor (B, NF, TCOL)
      descriptor_target(x (B,C,F,T))  -> (B, NF, TCOL) target profile (no grad)

    NF = mode_hi-mode_lo (mode-band freq bins); TCOL = time columns per window.
    """

    def __init__(self, d_model: int, n_tok: int, mode_lo: int, mode_hi: int,
                 tcol: int = 6, hidden: int = 512, proj: int = 8, dropout: float = 0.1,
                 horizons=(1,)):
        super().__init__()
        self.mode_lo = int(mode_lo)
        self.mode_hi = int(mode_hi)
        self.nf = self.mode_hi - self.mode_lo
        self.tcol = int(tcol)
        # MULTI-HORIZON: one descriptor readout per forecast horizon (in WINDOWS ahead).
        # (1,) = legacy single-step (t+1), shape-compatible with pre-2026-07-14 checkpoints
        # (n_horizons*nf*tcol == nf*tcol). e.g. (2, 4) = predict t+2 AND t+4 from the SAME
        # tokens; multi-target anchors the mode's oscillation phase (Gate 2b probe: t+4 drift
        # is anti-momentum, learnable as a short path far better than as one 200 ms jump).
        self.horizons = tuple(int(h) for h in horizons)
        self.n_horizons = len(self.horizons)
        self.tp = nn.Linear(d_model, proj)
        self.mlp = nn.Sequential(
            nn.Linear(n_tok * proj, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.n_horizons * self.nf * self.tcol),
        )
        # ZERO-INIT the final layer so the head output starts at 0. Under the
        # persistence-ANCHOR (pred_logit = anchor_logit + head), this means the model
        # starts EXACTLY at persistence (the current-window mode) and can only learn a
        # residual drift on top — it structurally cannot collapse to a flat output.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tok, d) -> (B, n_horizons, NF, TCOL) per-horizon descriptor residual."""
        B = tokens.shape[0]
        out = self.mlp(self.tp(tokens).reshape(B, -1))
        return out.reshape(B, self.n_horizons, self.nf, self.tcol)

    @torch.no_grad()
    def descriptor_target(self, spectro: torch.Tensor) -> torch.Tensor:
        """(B,C,F,T) standardized magnitude -> (B, NF, TCOL) channel-max mode-band
        residual profile, time-pooled. Baseline = box-25 avg over freq (approx the
        gaussian-sigma-6 baseline the audit detectors use)."""
        x = spectro.abs()
        B, C, Fq, T = x.shape
        xf = x.permute(0, 1, 3, 2).reshape(B * C * T, 1, Fq)
        base = F.avg_pool1d(xf, kernel_size=25, stride=1, padding=12)
        r = (xf - base).reshape(B, C, T, Fq).permute(0, 1, 3, 2)
        cmax = r.max(dim=1).values                              # (B,F,T) channel-max residual
        band = cmax[:, self.mode_lo:self.mode_hi, :]            # (B,NF,T)
        return F.adaptive_avg_pool1d(band, self.tcol)           # (B,NF,TCOL)


class VideoCodeHead(nn.Module):
    """Discrete-code VIDEO head — video analog of :class:`SpectrogramCodeHead`.

    A FROZEN adversarial-FSQ video codec (:class:`VideoFSQCodec`, Phase 1a) + a NEW
    code-prediction head over the backbone tokens. The backbone predicts the per-dim
    FSQ code levels (class-weighted CE); a sampled code decodes through the FROZEN
    resize-conv decoder → sharp, checkerboard-free frames. Injected already-frozen
    (``load_frozen_video_codec``) so this file never imports ``quantizers`` (the
    codec depends on this file's ``VideoOutputHead``).

    Contracts (shapes differ from spectro — video is (B,T,C,H,W) at the head):
      * ``forward(tokens) -> (B, T, C, H, W)`` — matches ``VideoOutputHead`` so the
        trainer's ``.permute(0,2,1,3,4)`` gives (B,C,T,H,W). Train: argmax-decode;
        eval: sample-decode. ``code_logits`` runs WITH grad (AMP cache fix).
      * ``code_logits(tokens) -> (B, n_tok, dim, levels)`` — TRAINABLE (class-weighted CE).
      * ``encode_target(video (B,C,T,H,W)) -> (B, n_tok, dim)`` int — frozen; CE targets.
      * ``decode(codes) -> (B, T, C, H, W)`` — frozen decoder; viz / rollout.
    """

    def __init__(
        self,
        codec: nn.Module,
        d_model: int,
        pred_hidden: int = 512,
        pred_layers: int = 2,
        sample_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.codec = codec                       # frozen (requires_grad False)
        self.n_tok = codec.n_tok
        self.dim = codec.dim
        self.levels = codec.levels
        self.sample_temperature = float(sample_temperature)
        trunk: list = []
        h = d_model
        for _ in range(max(1, pred_layers)):
            trunk += [nn.Linear(h, pred_hidden), nn.GELU()]
            h = pred_hidden
        self.trunk = nn.Sequential(*trunk)
        self.heads = nn.ModuleList(
            [nn.Linear(h, self.levels) for _ in range(self.dim)]
        )

    def code_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, n_tok, d_model) -> (B, n_tok, dim, levels). Trainable path."""
        h = self.trunk(tokens)
        return torch.stack([hd(h) for hd in self.heads], dim=2)

    @torch.no_grad()
    def encode_target(self, video: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> per-dim int codes (B, n_tok, dim). Frozen; CE targets."""
        return self.codec.encode_codes(video)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """per-dim int codes (B, n_tok, dim) -> video (B, T, C, H, W). Frozen decoder."""
        return self.codec.decode_codes(codes)

    def sample_codes(
        self, logits: torch.Tensor, temperature: Optional[float] = None,
        hard: bool = False,
    ) -> torch.Tensor:
        if hard:
            return logits.argmax(dim=-1)
        t = self.sample_temperature if temperature is None else float(temperature)
        probs = torch.softmax(logits / max(t, 1e-6), dim=-1)
        B, n, d, L = probs.shape
        return torch.multinomial(probs.reshape(-1, L), 1).reshape(B, n, d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode predicted codes to video (B, T, C, H, W). Viz/logging path; the
        training gradient flows through the SEPARATE ``code_logits`` call in the CE
        loss. ``code_logits`` runs WITH grad here (only sampling + frozen decode are
        no_grad) — the AMP fp32->bf16 cast-cache fix (see SpectrogramCodeHead)."""
        logits = self.code_logits(tokens)
        with torch.no_grad():
            codes = self.sample_codes(logits, hard=self.training)
            return self.codec.decode_codes(codes)
