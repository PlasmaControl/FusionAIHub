"""End-to-end foundation model assembly.

Ties per-modality tokenizers and output heads to the shared backbone. Tokens
for all modalities plus actuator commands are concatenated along the token
axis, fed through the backbone in one pass, and split back out to each head
for loss computation (``ResearchPlan.MD`` §3–§5.8).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import SharedBackbone
from .output_heads import (
    FastTimeSeriesHead,
    SlowTimeSeriesHead,
    SpectrogramOutputHead,
    VideoOutputHead,
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
        attn_impl: str = "standard",
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.diagnostics = list(diagnostics)
        self.actuators = list(actuators)
        self.d_model = d_model

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
                self.diag_heads[d_cfg.name] = SlowTimeSeriesHead(
                    d_model, d_cfg.n_channels, d_cfg.window_samples
                )
            elif d_cfg.kind == "fast_ts":
                assert d_cfg.patch_size is not None
                self.diag_tokenizers[d_cfg.name] = FastTimeSeriesTokenizer(
                    d_cfg.n_channels, d_cfg.window_samples, d_model, d_cfg.patch_size
                )
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
                self.diag_heads[d_cfg.name] = VideoOutputHead(
                    n_channels=d_cfg.n_channels,
                    n_frames=d_cfg.window_samples,
                    patch_size=d_cfg.video_patch_size,
                    d_model=d_model,
                    spatial_size=(d_cfg.height, d_cfg.width),
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
                )
                self.diag_heads[d_cfg.name] = SpectrogramOutputHead(
                    n_channels=d_cfg.n_channels,
                    d_model=d_model,
                    patch_f=F_p,
                    patch_t=T_p,
                    n_patches_f=d_cfg.freq_bins // F_p,
                    n_patches_t=trunc_t // T_p,
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
        self.backbone = SharedBackbone(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attn_impl=attn_impl,
            gradient_checkpoint=gradient_checkpoint,
        )

    def tokenize(
        self,
        diag_inputs: Dict[str, torch.Tensor],
        act_inputs: Dict[str, torch.Tensor],
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
    ) -> Dict[str, torch.Tensor]:
        """Full tokenize → backbone → per-modality-decode pipeline.

        Returns a dict of reconstructed raw signals, one per diagnostic
        modality, keyed by ``DiagnosticConfig.name``.
        """
        tokens = self.tokenize(diag_inputs, act_inputs)
        out_tokens = self.backbone(tokens, step_index, time_offset_s)
        return self.decode(out_tokens)
