"""Loss functions for Stage 3 multimodal prediction training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedPredictionLoss(nn.Module):
    """Combined token-space and observation-space loss.

    Parameters
    ----------
    frozen_decoders : dict[str, nn.Module]
        Per-signal frozen decoders for observation-space loss.
    token_weight : float
        Weight for the token-space L1 loss.
    obs_weight : float
        Weight for the observation-space L1 loss.
    """

    def __init__(
        self,
        frozen_decoders: dict[str, nn.Module],
        token_weight: float = 1.0,
        obs_weight: float = 1.0,
    ):
        super().__init__()
        self._frozen_decoders = frozen_decoders
        self.token_weight = token_weight
        self.obs_weight = obs_weight

    def forward(
        self,
        predicted_tokens: dict[str, torch.Tensor],
        target_tokens: dict[str, torch.Tensor],
        target_observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predicted_tokens : dict
            ``{signal: (B, n_tokens, d_model)}`` from forecasting heads.
        target_tokens : dict
            ``{signal: (B, n_tokens, d_model)}`` actual future tokens.
        target_observations : dict
            ``{signal: (B, ...)}`` actual future observations.

        Returns
        -------
        torch.Tensor
            Scalar combined loss.
        """
        token_losses = []
        obs_losses = []

        for sig in predicted_tokens:
            pred = predicted_tokens[sig]

            # Token-space loss
            if sig in target_tokens:
                token_losses.append(F.l1_loss(pred, target_tokens[sig]))

            # Observation-space loss
            # Gradients flow through decoder computation graph back to
            # forecasting heads. Decoder params have requires_grad=False
            # so they won't be updated.
            if sig in target_observations and sig in self._frozen_decoders:
                decoder = self._frozen_decoders[sig]
                decoded = decoder(pred)
                obs_losses.append(F.l1_loss(decoded, target_observations[sig]))

        loss = torch.tensor(0.0, device=pred.device)

        if token_losses:
            loss = loss + self.token_weight * torch.stack(token_losses).mean()
        if obs_losses:
            loss = loss + self.obs_weight * torch.stack(obs_losses).mean()

        return loss
