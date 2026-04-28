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
from .output_heads import FastTimeSeriesHead, SlowTimeSeriesHead
from .tokenizers.actuator import ActuatorTokenizer
from .tokenizers.fast_time_series import FastTimeSeriesTokenizer
from .tokenizers.slow_time_series import SlowTimeSeriesTokenizer


@dataclass(frozen=True)
class DiagnosticConfig:
    """Config for one diagnostic modality.

    Parameters
    ----------
    name
        Unique identifier used as the key in forward-pass input/output dicts.
    kind
        Either ``"slow_ts"`` (Linear-per-channel tokenization) or ``"fast_ts"``
        (Conv1d patching tokenization).
    n_channels
        Channel count.
    window_samples
        Samples per channel in one 50 ms window.
    patch_size
        Conv1d stride; required for ``"fast_ts"``, ignored for ``"slow_ts"``.
    """

    name: str
    kind: str
    n_channels: int
    window_samples: int
    patch_size: Optional[int] = None

    def n_tokens(self) -> int:
        if self.kind == "slow_ts":
            return self.n_channels
        if self.kind == "fast_ts":
            if self.patch_size is None:
                raise ValueError(f"{self.name}: fast_ts requires patch_size")
            return self.n_channels * (self.window_samples // self.patch_size)
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
            else:
                raise ValueError(f"Unknown diagnostic kind: {d_cfg.kind}")
            self.token_layout.append(
                TokenSlice(d_cfg.name, slice(offset, offset + n), is_diagnostic=True)
            )
            offset += n

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
        )

    def tokenize(
        self,
        diag_inputs: Dict[str, torch.Tensor],
        act_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Tokenize all modalities and concatenate along the token axis."""
        pieces: List[torch.Tensor] = []
        for d_cfg in self.diagnostics:
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