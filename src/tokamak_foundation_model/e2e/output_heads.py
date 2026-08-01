"""Per-modality output heads.

Each head is an approximate inverse of its sibling tokenizer. They fire only
to compute the training loss against ground-truth raw signals — during
autoregressive rollout the backbone's token output is fed directly to the
next step, bypassing the heads (``ResearchPlan.MD`` §3.5, §3.6, §5.7).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _CondVelocityUNet(nn.Module):
    """Small 2-downsample U-Net predicting the flow-matching velocity.

    Input is the noisy residual ``(B, C, F, T)`` concatenated with a
    spatial conditioning map; ``cond_vec`` (timestep + global tokens) drives
    AdaGN in every block. Output (same ``(B, C, F, T)``) is the velocity;
    the final conv is zero-initialised so the head is an identity (velocity
    0 → sample == mean) at construction.
    """

    def __init__(
        self, n_channels: int, cond_ch: int, base_ch: int, cond_dim: int
    ) -> None:
        super().__init__()
        c1, c2 = base_ch, base_ch * 2
        self.stem = nn.Conv2d(n_channels + cond_ch, c1, 3, padding=1)
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
        h = self.stem(torch.cat([x, cond_map], dim=1))
        e0 = self.enc0(h, cond_vec)
        e1 = self.enc1(self.down0(e0), cond_vec)
        m = self.mid1(self.mid0(self.down1(e1), cond_vec), cond_vec)
        u1 = F.interpolate(m, size=e1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, e1], dim=1), cond_vec)
        u0 = F.interpolate(u1, size=e0.shape[-2:], mode="nearest")
        u0 = self.up0(torch.cat([u0, e0], dim=1), cond_vec)
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
        enable_seam_refine: bool = False,
        seam_refine_hidden_ch: int = 16,
        seam_refine_kernel: int = 3,
        enable_inv_stem: bool = False,
        inv_stem_ch: int = 64,
    ) -> None:
        super().__init__()
        self.mean_head = SpectrogramOutputHead(
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
            n_channels, cond_ch, int(flow_base_ch), cond_dim
        )
        self.register_buffer(
            "sigma_pb", torch.ones(n_channels, self.freq_bins)
        )

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
    ) -> torch.Tensor:
        """Rectified-flow velocity MSE on the standardised residual. ``mu`` is
        detached (the mean is anchored by its own MAE term)."""
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
        the caller share one draw across rollout steps for coherence."""
        mu = self.mean_head(tokens)
        if self.training:
            return mu
        return self.sample(tokens, mu, noise=noise)
