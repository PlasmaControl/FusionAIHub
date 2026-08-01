"""Token-space autoregressive rollout.

At each step ``k``, the diagnostic-token slice output by the backbone at
step ``k-1`` is fed directly as the diagnostic-token input at step ``k``
(no detokenize-then-retokenize). Actuator tokens are recomputed from fresh
per-step actuator commands. Output heads fire only so a loss can be computed
against raw ground truth — their output is never fed back (``ResearchPlan.MD``
§3.6, §5.9).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .model import E2EFoundationModel
from .output_heads import SpectrogramFlowHead


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
    # Length-``K`` list; entry ``k`` maps ``modality_name -> (batch, n_tokens,
    # d_model)`` backbone token slice fed to that modality's head at step
    # ``k`` — the conditioning a generative head (SpectrogramFlowHead) needs
    # to compute its per-step flow loss. Empty unless ``collect_token_slices``.
    diag_token_slices: List[Dict[str, torch.Tensor]] = field(default_factory=list)


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
            x = diag_inputs[cfg.name]
            if cfg.kind == "video":
                # Video tokenizers honour a per-row camera-validity mask
                # (False rows are replaced with the learned missing_token).
                # Mirrors E2EFoundationModel.tokenize so missing-camera
                # samples don't get encoded as if a real camera frame
                # were present during step-0 init or TF re-tokenisation.
                valid = diag_inputs.get(f"{cfg.name}_valid")
                mask = valid.bool() if valid is not None else None
                pieces.append(
                    self.model.diag_tokenizers[cfg.name](x, mask=mask)
                )
            else:
                pieces.append(self.model.diag_tokenizers[cfg.name](x))
        return torch.cat(pieces, dim=1)

    def _tokenize_actuators(
        self, act_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for cfg in self.model.actuators:
            pieces.append(self.model.act_tokenizers[cfg.name](act_inputs[cfg.name]))
        return torch.cat(pieces, dim=1)

    def _decode_diagnostics(
        self,
        diag_tokens: torch.Tensor,
        *,
        flow_noise: Optional[Dict[str, torch.Tensor]] = None,
        return_slices: bool = False,
    ):
        """Decode per-modality predictions from the diagnostic token slice.

        ``flow_noise`` (eval only): a per-modality noise tensor passed to a
        generative head's sampler so the SAME draw can be reused across all
        K rollout steps → temporally coherent block-mode frames (no per-step
        flicker). ``return_slices``: also return the per-modality token slice
        (the conditioning the per-step flow loss needs). Both default off →
        byte-identical to the original predictions-only path.
        """
        out: Dict[str, torch.Tensor] = {}
        slices: Dict[str, torch.Tensor] = {}
        offset = 0
        for cfg in self.model.diagnostics:
            n = cfg.n_tokens()
            sl = diag_tokens[:, offset : offset + n]
            head = self.model.diag_heads[cfg.name]
            if flow_noise is not None and isinstance(head, SpectrogramFlowHead):
                out[cfg.name] = head(sl, noise=flow_noise.get(cfg.name))
            else:
                out[cfg.name] = head(sl)
            if return_slices:
                slices[cfg.name] = sl
            offset += n
        if return_slices:
            return out, slices
        return out

    def forward(
        self,
        initial_diag_inputs: Dict[str, torch.Tensor],
        act_inputs_per_step: List[Dict[str, torch.Tensor]],
        *,
        start_time_s: Optional[torch.Tensor] = None,
        collect_history: bool = True,
        gt_target_per_step: Optional[
            List[Dict[str, torch.Tensor]]
        ] = None,
        p_tf: float = 0.0,
        collect_token_slices: bool = False,
        flow_noise: Optional[Dict[str, torch.Tensor]] = None,
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
        collect_history
            When ``False``, skip appending to ``diagnostic_tokens`` and
            ``backbone_outputs`` (returned lists are empty). Saves ~4 GB of
            GPU memory at K=80, batch=128. Default ``True`` preserves prior
            §5.9 test behaviour.
        gt_target_per_step
            Optional length-``K`` list of ground-truth diagnostic dicts;
            ``gt_target_per_step[k]`` is the GT state at ``t = (k+1)*dt_s``
            (i.e. the rollout target of step ``k``). Required when
            ``p_tf > 0``; ignored otherwise. Predictions and history are
            unaffected — they always reflect the model's actual outputs.
        p_tf
            Teacher-forcing probability at each step ``k >= 1``. With
            probability ``p_tf`` the next-step diagnostic input is the
            re-tokenized GT state; otherwise it is the backbone's
            previous output (the default free-rollout behaviour). The
            coin is flipped per ``(rollout-step, training-step)`` and
            applies uniformly across the batch. Default ``0.0`` (pure
            free-rollout, byte-identical to prior behaviour).

        Returns
        -------
        RolloutResult
        """
        batch = next(iter(initial_diag_inputs.values())).shape[0]
        device = next(iter(initial_diag_inputs.values())).device
        n_steps = len(act_inputs_per_step)
        if start_time_s is None:
            start_time_s = torch.zeros(batch, device=device)

        # Teacher-forcing setup. ``use_tf`` is gated on both inputs being
        # supplied AND p_tf being non-zero, so the TF code path is fully
        # dormant when the trainer doesn't ask for it (preserves
        # byte-identity for existing tests / Aurora trainer / impulse
        # tests, none of which pass these args).
        use_tf = (
            p_tf > 0.0
            and gt_target_per_step is not None
            and len(gt_target_per_step) >= n_steps
        )

        diag_tokens = self._tokenize_diagnostics(initial_diag_inputs)
        diagnostic_tokens_history: List[torch.Tensor] = (
            [diag_tokens] if collect_history else []
        )
        predictions: List[Dict[str, torch.Tensor]] = []
        backbone_outputs: List[torch.Tensor] = []
        diag_token_slices_history: List[Dict[str, torch.Tensor]] = []

        for k in range(n_steps):
            act_tokens = self._tokenize_actuators(act_inputs_per_step[k])
            all_tokens = torch.cat([diag_tokens, act_tokens], dim=1)
            step_idx = torch.full(
                (batch,), k, dtype=torch.long, device=device
            )
            time_s = start_time_s + k * self.dt_s
            out_tokens = self.model.backbone(all_tokens, step_idx, time_s)
            if collect_history:
                backbone_outputs.append(out_tokens)

            # Predictions are always the model's real backbone output —
            # the TF decision below only affects what flows into the
            # *next* iteration's backbone, not what's scored.
            pred_diag_tokens = out_tokens[:, : self.n_diag_tokens]
            if collect_token_slices:
                preds_k, slices_k = self._decode_diagnostics(
                    pred_diag_tokens, flow_noise=flow_noise, return_slices=True,
                )
                predictions.append(preds_k)
                diag_token_slices_history.append(slices_k)
            else:
                predictions.append(
                    self._decode_diagnostics(
                        pred_diag_tokens, flow_noise=flow_noise,
                    )
                )

            # Decide what to feed into iteration k+1. On the last
            # iteration there's no next step; fall through to recording
            # ``pred_diag_tokens`` in history.
            if (
                k + 1 < n_steps
                and use_tf
                and torch.rand((), device=device).item() < p_tf
            ):
                # Teacher-force: re-tokenize the GT state at
                # ``t = (k+1) * dt_s`` (= rollout target of step k).
                diag_tokens = self._tokenize_diagnostics(
                    gt_target_per_step[k]
                )
            else:
                diag_tokens = pred_diag_tokens

            if collect_history:
                diagnostic_tokens_history.append(diag_tokens)

        return RolloutResult(
            predictions=predictions,
            diagnostic_tokens=diagnostic_tokens_history,
            backbone_outputs=backbone_outputs,
            diag_token_slices=diag_token_slices_history,
        )