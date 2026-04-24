"""Token-space autoregressive rollout.

At each step ``k``, the diagnostic-token slice output by the backbone at
step ``k-1`` is fed directly as the diagnostic-token input at step ``k``
(no detokenize-then-retokenize). Actuator tokens are recomputed from fresh
per-step actuator commands. Output heads fire only so a loss can be computed
against raw ground truth — their output is never fed back (``ResearchPlan.MD``
§3.6, §5.9).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .model import E2EFoundationModel


@dataclass
class RolloutResult:
    """Everything the training loop or a §5.9 test needs from one rollout.

    Attributes
    ----------
    predictions
        Length ``K`` list; entry ``k`` is a ``{modality_name: raw_signal}``
        dict of head-decoded predictions for step ``k+1``.
    diagnostic_tokens
        Length ``K + 1`` list of ``(batch, n_diag_tokens, d_model)`` tensors.
        Index 0 is the tokenized initial state; index ``k + 1`` is the
        diagnostic slice of the backbone output after step ``k``.
    backbone_outputs
        Length ``K`` list of full ``(batch, n_total_tokens, d_model)``
        backbone outputs, covering diagnostic and actuator slots, one per
        step.
    """

    predictions: List[Dict[str, torch.Tensor]]
    diagnostic_tokens: List[torch.Tensor]
    backbone_outputs: List[torch.Tensor]


class TokenSpaceRollout(nn.Module):
    """Autoregressive rollout wrapper around :class:`E2EFoundationModel`.

    Parameters
    ----------
    model
        The end-to-end foundation model providing tokenizers, backbone, and
        heads.
    dt_s
        Per-step time increment passed into the step-conditioning MLP.
        Defaults to 0.05 (50 ms, matching the Phase A window).
    """

    def __init__(self, model: E2EFoundationModel, dt_s: float = 0.05) -> None:
        super().__init__()
        self.model = model
        self.dt_s = dt_s
        self.n_diag_tokens = sum(
            layout.slice_.stop - layout.slice_.start
            for layout in model.token_layout
            if layout.is_diagnostic
        )

    def _tokenize_diagnostics(
        self, diag_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for cfg in self.model.diagnostics:
            pieces.append(self.model.diag_tokenizers[cfg.name](diag_inputs[cfg.name]))
        return torch.cat(pieces, dim=1)

    def _tokenize_actuators(
        self, act_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for cfg in self.model.actuators:
            pieces.append(self.model.act_tokenizers[cfg.name](act_inputs[cfg.name]))
        return torch.cat(pieces, dim=1)

    def _decode_diagnostics(
        self, diag_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        offset = 0
        for cfg in self.model.diagnostics:
            n = cfg.n_tokens()
            out[cfg.name] = self.model.diag_heads[cfg.name](
                diag_tokens[:, offset : offset + n]
            )
            offset += n
        return out

    def forward(
        self,
        initial_diag_inputs: Dict[str, torch.Tensor],
        act_inputs_per_step: List[Dict[str, torch.Tensor]],
        *,
        start_time_s: Optional[torch.Tensor] = None,
    ) -> RolloutResult:
        """Run a ``K``-step rollout.

        Parameters
        ----------
        initial_diag_inputs
            Ground-truth raw signals at step 0, one entry per diagnostic.
        act_inputs_per_step
            Length-``K`` list of actuator-input dicts, one per rollout step.
        start_time_s
            Optional ``(batch,)`` absolute-time tensor for step 0. Defaults
            to zeros.

        Returns
        -------
        RolloutResult
        """
        batch = next(iter(initial_diag_inputs.values())).shape[0]
        device = next(iter(initial_diag_inputs.values())).device
        n_steps = len(act_inputs_per_step)
        if start_time_s is None:
            start_time_s = torch.zeros(batch, device=device)

        diag_tokens = self._tokenize_diagnostics(initial_diag_inputs)
        diagnostic_tokens_history: List[torch.Tensor] = [diag_tokens]
        predictions: List[Dict[str, torch.Tensor]] = []
        backbone_outputs: List[torch.Tensor] = []

        for k in range(n_steps):
            act_tokens = self._tokenize_actuators(act_inputs_per_step[k])
            all_tokens = torch.cat([diag_tokens, act_tokens], dim=1)
            step_idx = torch.full(
                (batch,), k, dtype=torch.long, device=device
            )
            time_s = start_time_s + k * self.dt_s
            out_tokens = self.model.backbone(all_tokens, step_idx, time_s)
            backbone_outputs.append(out_tokens)

            diag_tokens = out_tokens[:, : self.n_diag_tokens]
            diagnostic_tokens_history.append(diag_tokens)
            predictions.append(self._decode_diagnostics(diag_tokens))

        return RolloutResult(
            predictions=predictions,
            diagnostic_tokens=diagnostic_tokens_history,
            backbone_outputs=backbone_outputs,
        )