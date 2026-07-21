"""End-to-end foundation model assembly.

Ties per-modality tokenizers and output heads to the shared backbone. Tokens
for all modalities plus actuator commands are concatenated along the token
axis, fed through the backbone in one pass, and split back out to each head
for loss computation (``ResearchPlan.MD`` §3–§5.8).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import SharedBackbone, MultiWindowBackbone
from .output_heads import (
    FastTimeSeriesCodeHead,
    FastTimeSeriesHead,
    SlowTimeSeriesCodeHead,
    SlowTimeSeriesHead,
    SpectrogramCodeHead,
    SpectrogramMaskGITHead,
    SpectrogramDescriptorHead,
    SpectrogramFlowHead,
    SpectrogramOutputHead,
    SpectroFreqWarpHead,
    VideoCodeHead,
    VideoFlowHead,
    VideoOutputHead,
)
from .quantizers import (
    load_frozen_codec,
    load_frozen_fastts_codec,
    load_frozen_slowts_codec,
    load_frozen_video_codec,
)
from .tokenizers.actuator import ActuatorTokenizer
from .tokenizers.fast_time_series import FastTimeSeriesTokenizer
from .tokenizers.slow_time_series import SlowTimeSeriesTokenizer
from .tokenizers.spectrogram import SpectrogramTokenizer
from .tokenizers.video import VideoTokenizer


@dataclass(frozen=True)
class DiagnosticConfig:
    """Config for one diagnostic modality.

    Parameters
    ----------
    name
        Unique identifier used as the key in forward-pass input/output dicts.
    kind
        One of ``"slow_ts"`` (Linear-per-channel tokenization), ``"fast_ts"``
        (Conv1d patching tokenization), ``"video"`` (tube-patch tokenization
        for camera diagnostics), or ``"spectrogram"`` (2D patch tokenization
        of an STFT magnitude spectrogram).
    n_channels
        Channel count. For video, the number of optical filters / colour
        channels. For spectrogram, the number of input STFT channels.
    window_samples
        Time-axis length of one 50 ms window. For ``"slow_ts"`` /
        ``"fast_ts"`` this is samples per channel; for ``"video"`` it
        is ``n_frames``; for ``"spectrogram"`` it is the number of STFT
        time frames (e.g. 98 for a 50 ms 500 kHz window with hop=256).
    patch_size
        Conv1d stride; required for ``"fast_ts"``, ignored otherwise.
    height
        Spatial frame height. Required for ``"video"``, ignored otherwise.
    width
        Spatial frame width. Required for ``"video"``, ignored otherwise.
    video_patch_size
        Tube patch shape ``(T_p, H_p, W_p)`` — kernel and stride of the
        ``Conv3d`` patch embedding. Required for ``"video"``, ignored
        otherwise. ``window_samples``, ``height``, ``width`` must each be
        divisible by the corresponding axis of this tuple.
    freq_bins
        STFT frequency-axis length (DC dropped by the data loader; e.g.
        512 for ``n_fft=1024``). Required for ``"spectrogram"``, ignored
        otherwise.
    spectrogram_patch_size
        2D patch ``(F_p, T_p)`` — kernel and stride of the ``Conv2d``
        patch embedding for spectrograms. Required for ``"spectrogram"``,
        ignored otherwise. ``freq_bins`` must be divisible by ``F_p``;
        ``window_samples`` is truncated to the largest multiple of ``T_p``.
    """

    name: str
    kind: str
    n_channels: int
    window_samples: int
    patch_size: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    video_patch_size: Optional[tuple[int, int, int]] = None
    freq_bins: Optional[int] = None
    spectrogram_patch_size: Optional[tuple[int, int]] = None

    def n_tokens(self) -> int:
        if self.kind == "slow_ts":
            return self.n_channels
        if self.kind == "fast_ts":
            if self.patch_size is None:
                raise ValueError(f"{self.name}: fast_ts requires patch_size")
            return self.n_channels * (self.window_samples // self.patch_size)
        if self.kind == "video":
            if (
                self.video_patch_size is None
                or self.height is None
                or self.width is None
            ):
                raise ValueError(
                    f"{self.name}: video requires height, width, "
                    "video_patch_size"
                )
            T_p, H_p, W_p = self.video_patch_size
            return (
                (self.window_samples // T_p)
                * (self.height // H_p)
                * (self.width // W_p)
            )
        if self.kind == "spectrogram":
            if (
                self.freq_bins is None
                or self.spectrogram_patch_size is None
            ):
                raise ValueError(
                    f"{self.name}: spectrogram requires freq_bins and "
                    "spectrogram_patch_size"
                )
            F_p, T_p = self.spectrogram_patch_size
            if self.freq_bins % F_p != 0:
                raise ValueError(
                    f"{self.name}: freq_bins={self.freq_bins} must be "
                    f"divisible by F_p={F_p}"
                )
            trunc_t = (self.window_samples // T_p) * T_p
            return (self.freq_bins // F_p) * (trunc_t // T_p)
        raise ValueError(f"Unknown diagnostic kind: {self.kind}")


@dataclass(frozen=True)
class ActuatorConfig:
    """Config for one actuator group (e.g. NBI, ECH, gas, RMP)."""

    name: str
    n_channels: int
    window_samples: int
    n_tokens: int = 3


@dataclass
class TokenSlice:
    """Where a modality's tokens live in the backbone's flat token sequence."""

    name: str
    slice_: slice
    is_diagnostic: bool


class E2EFoundationModel(nn.Module):
    """End-to-end multi-modal foundation model (Phase A: time-series only).

    Parameters
    ----------
    diagnostics
        Ordered list of :class:`DiagnosticConfig`.
    actuators
        Ordered list of :class:`ActuatorConfig`.
    d_model
        Token dimension (``256`` in the full config).
    n_heads
        Attention heads.
    n_layers
        Transformer blocks.
    mlp_ratio
        MLP hidden-dim ratio.
    dropout
        Dropout fraction inside attention and MLP.
    """

    def __init__(
        self,
        diagnostics: List[DiagnosticConfig],
        actuators: List[ActuatorConfig],
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        backbone_grad_checkpoint: bool = False,
        video_seam_refine: bool = False,
        spectro_seam_refine: bool = False,
        seam_refine_hidden_ch: int = 16,
        spectro_refine_kernel: int = 3,
        video_refine_kernel: tuple = (1, 3, 3),
        spectro_inv_stem: bool = False,
        spectro_inv_stem_ch: int = 64,
        spectro_freq_stem: bool = False,
        spectro_freq_stem_hidden: int = 128,
        video_resize_conv: bool = False,
        video_resize_conv_hidden: int = 64,
        video_generative: bool = False,
        video_flow_base_ch: int = 64,
        video_flow_sample_steps: int = 6,
        video_flow_lambda: float = 1.0,
        video_flow_pe_ch: int = 16,
        video_sigma_spatial: bool = False,
        spectro_generative: bool = False,
        spectro_flow_base_ch: int = 64,
        spectro_flow_sample_steps: int = 6,
        spectro_flow_lambda: float = 1.0,
        spectro_flow_freq_pe_ch: int = 0,
        spectro_flow_time_pe_ch: int = 0,
        spectro_mask: bool = False,
        spectro_mask_hidden: int = 64,
        spectro_input_cond: bool = False,
        spectro_input_feat: bool = False,
        spectro_flow_residual_anchor: bool = False,
        spectro_fsq: bool = False,
        spectro_fsq_codec_dir: str = "",
        spectro_code_pred_hidden: int = 512,
        spectro_code_pred_layers: int = 2,
        spectro_code_temperature: float = 1.0,
        spectro_maskgit: bool = False,
        spectro_maskgit_dim: int = 512,
        spectro_maskgit_layers: int = 4,
        spectro_maskgit_heads: int = 8,
        spectro_maskgit_decode_steps: int = 10,
        spectro_maskgit_decode_temp: float = 0.5,
        video_fsq: bool = False,
        video_fsq_codec_dir: str = "",
        video_code_pred_hidden: int = 512,
        video_code_pred_layers: int = 2,
        video_code_temperature: float = 1.0,
        fastts_fsq: bool = False,
        fastts_fsq_codec_dir: str = "",
        fastts_code_pred_hidden: int = 512,
        fastts_code_pred_layers: int = 2,
        fastts_code_temperature: float = 1.0,
        slow_ts_fsq: bool = False,
        slow_ts_fsq_codec_dir: str = "",
        slow_ts_code_pred_hidden: int = 512,
        slow_ts_code_pred_layers: int = 2,
        slow_ts_code_temperature: float = 1.0,
        backbone_input_skip: bool = False,
        spec_persistence_anchor: bool = False,
        spec_warp_anchor: bool = False,
        spec_warp_max_bins: float = 8.0,
        spec_descriptor: bool = False,
        spec_descriptor_tcol: int = 6,
        spec_descriptor_hidden: int = 512,
        spec_descriptor_horizons=(1,),
        history_windows: int = 1,
        use_actuator_film: bool = False,
    ) -> None:
        super().__init__()
        # Phase-1b FSQ code head config (recorded so eval can rebuild the head).
        self.spectro_fsq = bool(spectro_fsq)
        self.spectro_fsq_codec_dir = str(spectro_fsq_codec_dir)
        self.video_fsq = bool(video_fsq)
        self.video_fsq_codec_dir = str(video_fsq_codec_dir)
        self.fastts_fsq = bool(fastts_fsq)
        self.fastts_fsq_codec_dir = str(fastts_fsq_codec_dir)
        self.slow_ts_fsq = bool(slow_ts_fsq)
        self.slow_ts_fsq_codec_dir = str(slow_ts_fsq_codec_dir)
        self.diagnostics = list(diagnostics)
        self.actuators = list(actuators)
        self.d_model = d_model
        # Gated input->output residual skip AROUND the backbone. The transformer
        # attention dilutes the localized spectro mode tokens (autoencode-of-the-
        # current-window only reaches 0.41 vs the codec's 0.70), so route the input
        # token representation to the head via a LayerScale-style gated residual:
        #   out = tokens + gate * backbone(tokens)
        # The gate (init 0.2) keeps the added transformer term BOUNDED so the skip
        # is stable under autoregressive rollout (raw add would accumulate across
        # K steps and drift); it also lets the input-window mode be carried to the
        # head and sustained across rollout steps (persistence prior in token space).
        self.backbone_input_skip = bool(backbone_input_skip)
        if self.backbone_input_skip:
            self.backbone_skip_gate = nn.Parameter(torch.tensor(0.2))
        # Explicit output-level persistence anchor: spectro prediction =
        # input-window spectrogram + head(tokens). The amplitude comes from the
        # REAL input mode (full amplitude → the ridge is visible, not dampened);
        # the head learns only the delta (target - input), which is small because
        # modes persist. Default (delta→0) = persistence = a visible, correct-freq
        # mode. Rollout-ready: in Stage-2 the "input" is the previous predicted
        # window, so the mode is carried forward step to step.
        self.spec_persistence_anchor = bool(spec_persistence_anchor)
        # Frequency-WARP anchor: spectro prediction = warp(input-window, s) +
        # head(tokens), where the head predicts a smooth freq-axis shift field s
        # (SpectroFreqWarpHead) that SLIDES the real input ridge to its forecast
        # frequency. Unlike the additive persistence anchor (ridge stuck at the
        # input frequency → visible "shift" on a drifting mode), the warp can
        # MOVE the peak forward — a cheap, bounded prediction — while amplitude
        # still comes from the real input (visible, tvr≈1). Zero-init warp head
        # → identity warp → exact persistence at start. Implies the anchor.
        self.spec_warp_anchor = bool(spec_warp_anchor)
        self.spec_warp_max_bins = float(spec_warp_max_bins)
        self.spec_warp_heads = nn.ModuleDict()
        # FACTORIZATION: auxiliary mode-descriptor heads (forecast the shift-stable
        # band-power profile the codes can't carry). Called directly by the trainer
        # on the backbone token slices; does not touch model.forward outputs.
        self.spec_descriptor = bool(spec_descriptor)
        self.spec_descriptor_tcol = int(spec_descriptor_tcol)
        self.spec_descriptor_hidden = int(spec_descriptor_hidden)
        self.spec_descriptor_horizons = tuple(int(h) for h in spec_descriptor_horizons)
        self.spec_descriptor_heads = nn.ModuleDict()

        self.diag_tokenizers = nn.ModuleDict()
        self.diag_heads = nn.ModuleDict()
        self.act_tokenizers = nn.ModuleDict()
        self.token_layout: List[TokenSlice] = []

        offset = 0
        for d_cfg in diagnostics:
            n = d_cfg.n_tokens()
            if d_cfg.kind == "slow_ts":
                self.diag_tokenizers[d_cfg.name] = SlowTimeSeriesTokenizer(
                    d_cfg.n_channels, d_cfg.window_samples, d_model
                )
                if slow_ts_fsq:
                    # Phase 1b: predict the FROZEN slow-TS FSQ codec's discrete codes.
                    # Input tokenization above is unchanged (warm-start safe).
                    codec_path = os.path.join(
                        slow_ts_fsq_codec_dir, f"slowts_codec_{d_cfg.name}.pt"
                    )
                    scodec, _scfg = load_frozen_slowts_codec(codec_path)
                    if scodec.n_tok != d_cfg.n_channels:
                        raise ValueError(
                            f"slow-TS codec n_tok {scodec.n_tok} != backbone token "
                            f"count {d_cfg.n_channels} for '{d_cfg.name}' (one token/"
                            f"channel) — channel mismatch with codec ({codec_path})."
                        )
                    self.diag_heads[d_cfg.name] = SlowTimeSeriesCodeHead(
                        codec=scodec,
                        d_model=d_model,
                        pred_hidden=slow_ts_code_pred_hidden,
                        pred_layers=slow_ts_code_pred_layers,
                        sample_temperature=slow_ts_code_temperature,
                    )
                else:
                    self.diag_heads[d_cfg.name] = SlowTimeSeriesHead(
                        d_model, d_cfg.n_channels, d_cfg.window_samples
                    )
            elif d_cfg.kind == "fast_ts":
                assert d_cfg.patch_size is not None
                self.diag_tokenizers[d_cfg.name] = FastTimeSeriesTokenizer(
                    d_cfg.n_channels, d_cfg.window_samples, d_model, d_cfg.patch_size
                )
                if fastts_fsq:
                    # Phase 1b: predict the FROZEN fast-TS FSQ codec's discrete codes
                    # (categorical → keeps sharp ELM spikes). Codec loaded frozen;
                    # input tokenization above is unchanged (warm-start safe).
                    codec_path = os.path.join(
                        fastts_fsq_codec_dir, f"fastts_codec_{d_cfg.name}.pt"
                    )
                    if not os.path.exists(codec_path):
                        codec_path = os.path.join(fastts_fsq_codec_dir, "fastts_codec.pt")
                    fcodec, _fcfg = load_frozen_fastts_codec(codec_path)
                    exp_tok = d_cfg.n_channels * (d_cfg.window_samples // d_cfg.patch_size)
                    if fcodec.n_tok != exp_tok:
                        raise ValueError(
                            f"fastts codec n_tok {fcodec.n_tok} != backbone token "
                            f"count {exp_tok} for '{d_cfg.name}' — channel/patch "
                            f"mismatch between codec ({codec_path}) and diag config."
                        )
                    if fcodec.C != d_cfg.n_channels:
                        raise ValueError(
                            f"fastts codec channels {fcodec.C} != diag "
                            f"'{d_cfg.name}' n_channels {d_cfg.n_channels} ({codec_path})."
                        )
                    self.diag_heads[d_cfg.name] = FastTimeSeriesCodeHead(
                        codec=fcodec,
                        d_model=d_model,
                        pred_hidden=fastts_code_pred_hidden,
                        pred_layers=fastts_code_pred_layers,
                        sample_temperature=fastts_code_temperature,
                    )
                else:
                    self.diag_heads[d_cfg.name] = FastTimeSeriesHead(
                        d_model, d_cfg.n_channels, d_cfg.window_samples, d_cfg.patch_size
                    )
            elif d_cfg.kind == "video":
                assert d_cfg.video_patch_size is not None
                assert d_cfg.height is not None and d_cfg.width is not None
                self.diag_tokenizers[d_cfg.name] = VideoTokenizer(
                    n_channels=d_cfg.n_channels,
                    n_frames=d_cfg.window_samples,
                    patch_size=d_cfg.video_patch_size,
                    d_model=d_model,
                    spatial_size=(d_cfg.height, d_cfg.width),
                )
                if video_fsq:
                    # Phase 1b: predict a FROZEN adversarial-FSQ video codec's
                    # discrete codes (categorical) instead of regressing pixels.
                    # Frozen resize-conv decoder → sharp, checkerboard-free frames.
                    # Per-divertor codec loaded from video_fsq_codec_dir. Input
                    # tokenization stays continuous (VideoTokenizer above) →
                    # warm-start-friendly; only the OUTPUT path becomes code-based.
                    codec_path = os.path.join(
                        video_fsq_codec_dir, f"video_codec_{d_cfg.name}.pt"
                    )
                    vcodec, _vcfg = load_frozen_video_codec(codec_path)
                    if vcodec.n_tok != d_cfg.n_tokens():
                        raise ValueError(
                            f"video codec n_tok {vcodec.n_tok} != backbone video "
                            f"token count {d_cfg.n_tokens()} for '{d_cfg.name}' "
                            f"({codec_path})."
                        )
                    if vcodec.C != d_cfg.n_channels:
                        raise ValueError(
                            f"video codec channels {vcodec.C} != diag '{d_cfg.name}' "
                            f"n_channels {d_cfg.n_channels} ({codec_path})."
                        )
                    self.diag_heads[d_cfg.name] = VideoCodeHead(
                        codec=vcodec,
                        d_model=d_model,
                        pred_hidden=video_code_pred_hidden,
                        pred_layers=video_code_pred_layers,
                        sample_temperature=video_code_temperature,
                    )
                elif video_generative:
                    # Generative VideoFlowHead: resize-conv mean (no patch
                    # checkerboard) + rectified-flow residual (sharp, no
                    # mean-collapse) + spatial PE. Robust to imperfect backbone
                    # tokens (see eval_runs/video_test/SUMMARY.md). The mean
                    # branch is resize-conv internally, so --video_resize_conv
                    # is implied/irrelevant here.
                    self.diag_heads[d_cfg.name] = VideoFlowHead(
                        n_channels=d_cfg.n_channels,
                        n_frames=d_cfg.window_samples,
                        patch_size=d_cfg.video_patch_size,
                        d_model=d_model,
                        spatial_size=(d_cfg.height, d_cfg.width),
                        flow_base_ch=video_flow_base_ch,
                        flow_sample_steps=video_flow_sample_steps,
                        flow_lambda=video_flow_lambda,
                        flow_h_pe_ch=video_flow_pe_ch,
                        flow_w_pe_ch=video_flow_pe_ch,
                        resize_conv_hidden_ch=video_resize_conv_hidden,
                        sigma_spatial=video_sigma_spatial,
                    )
                else:
                    self.diag_heads[d_cfg.name] = VideoOutputHead(
                        n_channels=d_cfg.n_channels,
                        n_frames=d_cfg.window_samples,
                        patch_size=d_cfg.video_patch_size,
                        d_model=d_model,
                        spatial_size=(d_cfg.height, d_cfg.width),
                        enable_seam_refine=video_seam_refine,
                        seam_refine_hidden_ch=seam_refine_hidden_ch,
                        seam_refine_kernel=tuple(video_refine_kernel),
                        decoder="resize_conv" if video_resize_conv else "deconv",
                        resize_conv_hidden_ch=video_resize_conv_hidden,
                    )
            elif d_cfg.kind == "spectrogram":
                assert d_cfg.freq_bins is not None
                assert d_cfg.spectrogram_patch_size is not None
                F_p, T_p = d_cfg.spectrogram_patch_size
                trunc_t = (d_cfg.window_samples // T_p) * T_p
                self.diag_tokenizers[d_cfg.name] = SpectrogramTokenizer(
                    n_channels=d_cfg.n_channels,
                    d_model=d_model,
                    patch_f=F_p,
                    patch_t=T_p,
                    freq_bins=d_cfg.freq_bins,
                    time_frames=d_cfg.window_samples,
                    enable_freq_stem=spectro_freq_stem,
                    freq_stem_hidden=spectro_freq_stem_hidden,
                )
                spec_head_kwargs = dict(
                    n_channels=d_cfg.n_channels,
                    d_model=d_model,
                    patch_f=F_p,
                    patch_t=T_p,
                    n_patches_f=d_cfg.freq_bins // F_p,
                    n_patches_t=trunc_t // T_p,
                    enable_seam_refine=spectro_seam_refine,
                    seam_refine_hidden_ch=seam_refine_hidden_ch,
                    seam_refine_kernel=spectro_refine_kernel,
                    enable_inv_stem=spectro_inv_stem,
                    inv_stem_ch=spectro_inv_stem_ch,
                )
                if spectro_fsq:
                    # Phase 1b: predict the FROZEN adversarial-FSQ codec's discrete
                    # codes (categorical, cannot mean-collapse). The per-modality
                    # codec (Phase 1a) is loaded frozen from spectro_fsq_codec_dir.
                    codec_path = os.path.join(
                        spectro_fsq_codec_dir, f"spectro_codec_{d_cfg.name}.pt"
                    )
                    codec, _codec_cfg = load_frozen_codec(codec_path)
                    if codec.n_tok != (d_cfg.freq_bins // F_p) * (trunc_t // T_p):
                        raise ValueError(
                            f"codec n_tok {codec.n_tok} != backbone spectro token "
                            f"count for '{d_cfg.name}' — patch/freq mismatch between "
                            f"codec ({codec_path}) and diag config."
                        )
                    if codec.C != d_cfg.n_channels:
                        raise ValueError(
                            f"codec channels {codec.C} != diag '{d_cfg.name}' "
                            f"n_channels {d_cfg.n_channels} ({codec_path})."
                        )
                    if spectro_maskgit:
                        # JOINT (MaskGIT) code head — bidirectional transformer over
                        # the patch grid; fixes the independent-head collapse.
                        self.diag_heads[d_cfg.name] = SpectrogramMaskGITHead(
                            codec=codec,
                            d_model=d_model,
                            mg_dim=spectro_maskgit_dim,
                            mg_layers=spectro_maskgit_layers,
                            mg_heads=spectro_maskgit_heads,
                            decode_steps=spectro_maskgit_decode_steps,
                            decode_temperature=spectro_maskgit_decode_temp,
                            sample_temperature=spectro_code_temperature,
                        )
                    else:
                        self.diag_heads[d_cfg.name] = SpectrogramCodeHead(
                            codec=codec,
                            d_model=d_model,
                            pred_hidden=spectro_code_pred_hidden,
                            pred_layers=spectro_code_pred_layers,
                            sample_temperature=spectro_code_temperature,
                            # residual-codec: baseline subtraction is baked into
                            # the codec cfg, so the head self-declares it and the
                            # trainer/eval forward runs this modality in R-space.
                            bg_subtract=bool(_codec_cfg.get("bg_subtract", False)),
                            bg_sigma=float(_codec_cfg.get("bg_sigma", 8.0)),
                        )
                elif spectro_generative:
                    self.diag_heads[d_cfg.name] = SpectrogramFlowHead(
                        flow_base_ch=spectro_flow_base_ch,
                        flow_sample_steps=spectro_flow_sample_steps,
                        flow_lambda=spectro_flow_lambda,
                        flow_freq_pe_ch=spectro_flow_freq_pe_ch,
                        flow_time_pe_ch=spectro_flow_time_pe_ch,
                        enable_mask=spectro_mask,
                        mask_hidden_ch=spectro_mask_hidden,
                        enable_input_cond=spectro_input_cond,
                        enable_input_feat=spectro_input_feat,
                        residual_anchor=spectro_flow_residual_anchor,
                        **spec_head_kwargs,
                    )
                else:
                    self.diag_heads[d_cfg.name] = SpectrogramOutputHead(
                        **spec_head_kwargs
                    )
                if self.spec_warp_anchor:
                    # Predicts the freq-shift field that slides the input ridge
                    # to its forecast frequency (drift fix; see ctor comment).
                    self.spec_warp_heads[d_cfg.name] = SpectroFreqWarpHead(
                        d_model=d_model,
                        n_channels=d_cfg.n_channels,
                        n_patches_f=d_cfg.freq_bins // F_p,
                        n_patches_t=trunc_t // T_p,
                        freq_bins=d_cfg.freq_bins,
                        trunc_t=trunc_t,
                        max_shift_bins=self.spec_warp_max_bins,
                    )
                if self.spec_descriptor:
                    _fb = int(d_cfg.freq_bins)
                    _mlo = int(round(5.0 / 250.0 * _fb))   # 5 kHz over 0-250 kHz Nyquist
                    _mhi = int(round(40.0 / 250.0 * _fb))  # 40 kHz  (=10..82 at Fq=512)
                    self.spec_descriptor_heads[d_cfg.name] = SpectrogramDescriptorHead(
                        d_model=d_model,
                        n_tok=(d_cfg.freq_bins // F_p) * (trunc_t // T_p),
                        mode_lo=_mlo,
                        mode_hi=_mhi,
                        tcol=self.spec_descriptor_tcol,
                        hidden=self.spec_descriptor_hidden,
                        horizons=self.spec_descriptor_horizons,
                    )
            else:
                raise ValueError(f"Unknown diagnostic kind: {d_cfg.kind}")
            self.token_layout.append(
                TokenSlice(d_cfg.name, slice(offset, offset + n), is_diagnostic=True)
            )
            offset += n

        # Capture the diagnostic-prefix length before actuators are
        # appended; ``rollout.py`` slices ``[:, :n_diag_tokens]`` to
        # propagate diagnostic outputs autoregressively.
        self.n_diag_tokens = offset

        for a_cfg in actuators:
            self.act_tokenizers[a_cfg.name] = ActuatorTokenizer(
                a_cfg.n_channels, a_cfg.window_samples, d_model, a_cfg.n_tokens
            )
            self.token_layout.append(
                TokenSlice(
                    a_cfg.name,
                    slice(offset, offset + a_cfg.n_tokens),
                    is_diagnostic=False,
                )
            )
            offset += a_cfg.n_tokens

        self.n_total_tokens = offset
        # history_windows K>1 → true multi-window temporal backbone: K past 50 ms
        # windows in, next-window out, with causal temporal attention across
        # windows (so the model sees mode VELOCITY). K=1 → the original
        # single-window backbone, byte-identical (production path untouched).
        self.history_windows = int(history_windows)
        if self.history_windows > 1:
            self.backbone = MultiWindowBackbone(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                grad_checkpoint=backbone_grad_checkpoint,
            )
        else:
            self.backbone = SharedBackbone(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                grad_checkpoint=backbone_grad_checkpoint,
            )
        # GATE-4/FiLM: inject actuators by conditioning-by-construction — FiLM (γ,β) per backbone block
        # from the pooled actuator embedding — INSTEAD of the actuator-token pathway (removed from the
        # sequence when enabled). Reuses the banked act_tokenizers to encode actuators → pooled embedding.
        # Final layer ZERO-init ⇒ γ=β=0 at start ⇒ identity FiLM ⇒ warm-start byte-identical, then learns.
        self.use_actuator_film = bool(use_actuator_film)
        if self.use_actuator_film:
            self._film_n_layers = int(n_layers)
            _fh = d_model * 4
            self.actuator_film = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, _fh), nn.GELU(),
                nn.Linear(_fh, n_layers * 2 * d_model),
            )
            nn.init.zeros_(self.actuator_film[-1].weight)
            nn.init.zeros_(self.actuator_film[-1].bias)

    def _actuator_film_params(self, act_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Pooled actuator embedding → per-block (γ,β): (B, n_layers, 2, d_model)."""
        pieces = [self.act_tokenizers[a.name](act_inputs[a.name]) for a in self.actuators]
        emb = torch.cat(pieces, dim=1).mean(dim=1)                       # (B, d) mean over actuator tokens
        return self.actuator_film(emb).view(emb.shape[0], self._film_n_layers, 2, -1)

    def tokenize(
        self,
        diag_inputs: Dict[str, torch.Tensor],
        act_inputs: Dict[str, torch.Tensor],
        actuators_as_film: bool = False,
    ) -> torch.Tensor:
        """Tokenize all modalities and concatenate along the token axis.

        For ``kind="video"`` and ``kind="spectrogram"`` diagnostics, an
        optional per-modality validity mask is read from
        ``diag_inputs[f"{name}_valid"]`` (a ``(B,)`` long tensor;
        zero-rows trigger the tokenizer's learned ``missing_token``).
        If absent, the modality is treated as always present. The TS
        path is unchanged for backwards compatibility.
        """
        pieces: List[torch.Tensor] = []
        for d_cfg in self.diagnostics:
            if d_cfg.kind in ("video", "spectrogram"):
                x = diag_inputs[d_cfg.name]
                valid = diag_inputs.get(f"{d_cfg.name}_valid")
                mask = valid.bool() if valid is not None else None
                pieces.append(
                    self.diag_tokenizers[d_cfg.name](x, mask=mask)
                )
            else:
                pieces.append(
                    self.diag_tokenizers[d_cfg.name](diag_inputs[d_cfg.name])
                )
        if not actuators_as_film:   # FiLM path injects actuators as backbone modulation, not as tokens
            for a_cfg in self.actuators:
                pieces.append(
                    self.act_tokenizers[a_cfg.name](act_inputs[a_cfg.name])
                )
        return torch.cat(pieces, dim=1)

    def decode(
        self, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run per-modality heads on backbone output tokens."""
        outputs: Dict[str, torch.Tensor] = {}
        for layout in self.token_layout:
            if not layout.is_diagnostic:
                continue
            outputs[layout.name] = self.diag_heads[layout.name](
                tokens[:, layout.slice_]
            )
        return outputs

    def forward(
        self,
        diag_inputs: Dict[str, torch.Tensor],
        act_inputs: Dict[str, torch.Tensor],
        step_index: torch.Tensor,
        time_offset_s: torch.Tensor,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full tokenize → backbone → per-modality-decode pipeline.

        Returns a dict of reconstructed raw signals, one per diagnostic
        modality, keyed by ``DiagnosticConfig.name``. When ``return_tokens``
        is set, returns ``(predictions, diag_token_slices)`` where
        ``diag_token_slices[name]`` is the backbone output slice fed to that
        modality's head — needed by generative heads (e.g.
        :class:`SpectrogramFlowHead`) to compute their conditioning-dependent
        loss against the targets (which the head's forward never sees).
        """
        if self.history_windows > 1:
            # Multi-window: inputs arrive as (B, K, ...) with the K history axis
            # at dim 1. Tokenize each window with the shared tokenizers → stack
            # (B, K, N, d); the spatiotemporal backbone attends across windows
            # (causal) and returns the LAST window's tokens (B, N, d). The anchor
            # then uses the LAST (most recent) input window.
            K = self.history_windows
            # Only the K-windowed diagnostic tensors carry a window axis at dim 1
            # (shape (B, K, C, ...), dim>=4). Actuators are single-window
            # (from targets, (B, C, ...), dim 3) and per-modality `_valid` scalars
            # are (B,) — both are passed UNSLICED to every window's tokenize.
            def _win(v, k):
                return v[:, k] if (torch.is_tensor(v) and v.dim() >= 4) else v
            win_tokens = []
            for k in range(K):
                dk = {key: _win(v, k) for key, v in diag_inputs.items()}
                win_tokens.append(self.tokenize(dk, act_inputs))    # (B, N, d)
            tokens = torch.stack(win_tokens, dim=1)                 # (B, K, N, d)
            out_tokens = self.backbone(tokens, step_index, time_offset_s)  # (B, N, d)
            if self.backbone_input_skip:
                out_tokens = tokens[:, -1] + self.backbone_skip_gate * out_tokens
            anchor_inputs = {name: _win(v, -1) for name, v in diag_inputs.items()}
        else:
            _film = self._actuator_film_params(act_inputs) if self.use_actuator_film else None
            tokens = self.tokenize(diag_inputs, act_inputs, actuators_as_film=self.use_actuator_film)
            out_tokens = self.backbone(tokens, step_index, time_offset_s, film_params=_film)
            if self.backbone_input_skip:
                # Gated residual around the backbone (rollout-safe): the head sees the
                # input tokens + a bounded transformer correction, so the localized
                # mode survives attention and is carried across rollout steps.
                out_tokens = tokens + self.backbone_skip_gate * out_tokens
            anchor_inputs = diag_inputs
        predictions = self.decode(out_tokens)
        if self.spec_warp_anchor:
            # spectro prediction = warp(input window, predicted shift) + head
            # delta. The warp SLIDES the real input ridge to its forecast
            # frequency (drift fix); amplitude still comes from the real input
            # (visible). Head learns the residual delta on top of the warped
            # persistence. Zero-init warp head → identity → persistence at start.
            slice_by_name = {
                layout.name: layout.slice_
                for layout in self.token_layout
                if layout.is_diagnostic
            }
            for cfg in self.diagnostics:
                if (
                    cfg.kind == "spectrogram"
                    and cfg.name in predictions
                    and cfg.name in self.spec_warp_heads
                ):
                    inp = anchor_inputs.get(cfg.name)
                    if inp is not None:
                        t = predictions[cfg.name].shape[-1]
                        tok = out_tokens[:, slice_by_name[cfg.name]]
                        warped = self.spec_warp_heads[cfg.name](tok, inp[..., :t])
                        predictions[cfg.name] = predictions[cfg.name] + warped
        elif self.spec_persistence_anchor:
            # spectro prediction = input window (persistence, full amplitude) +
            # head delta. Head is thereby trained on (target - input) via the
            # standard MAE loss; the visible mode comes from the real input.
            for cfg in self.diagnostics:
                if cfg.kind == "spectrogram" and cfg.name in predictions:
                    inp = anchor_inputs.get(cfg.name)
                    if inp is not None:
                        t = predictions[cfg.name].shape[-1]
                        predictions[cfg.name] = predictions[cfg.name] + inp[..., :t]
        if return_tokens:
            diag_token_slices = {
                layout.name: out_tokens[:, layout.slice_]
                for layout in self.token_layout
                if layout.is_diagnostic
            }
            return predictions, diag_token_slices
        return predictions
