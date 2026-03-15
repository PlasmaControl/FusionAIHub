"""Stage 3: Multimodal prediction model.

Combines a fusion transformer with per-modality forecasting heads
and frozen decoders for observation-space loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tokamak_foundation_model.models.latent_feature_space.baseline_fusion_transformer import (
    BaselineFusionTransformer,
)


class ForecastingHead(nn.Module):
    """Maps fused tokens to predicted next-step tokens for one modality.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_tokens : int
        Number of tokens for this modality.
    """

    def __init__(self, d_model: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fused_tokens : torch.Tensor
            ``(B, n_tokens, d_model)`` slice from fusion transformer output.

        Returns
        -------
        torch.Tensor
            ``(B, n_tokens, d_model)`` predicted tokens.
        """
        return self.mlp(fused_tokens)


class MultimodalPredictionModel(nn.Module):
    """Multimodal fusion + forecasting model for Stage 3.

    Takes pre-computed tokens from multiple modalities at time ``t``,
    fuses them via a transformer, and predicts tokens at time ``t+1``
    for each output modality.

    Parameters
    ----------
    fusion_transformer : BaselineFusionTransformer
        Trainable fusion transformer.
    forecasting_heads : nn.ModuleDict
        Per-signal ``ForecastingHead`` modules (trainable).
    frozen_decoders : dict
        Per-signal frozen decoders for observation-space loss.
        Stored as plain dict (not ``nn.ModuleDict``) to exclude
        from optimizer.
    modality_token_ranges : dict
        ``{signal_name: (start_idx, end_idx)}`` for slicing fusion output.
    modality_ids : dict
        ``{signal_name: int}`` for modality embeddings.
    """

    def __init__(
        self,
        fusion_transformer: BaselineFusionTransformer,
        forecasting_heads: nn.ModuleDict,
        frozen_decoders: dict[str, nn.Module],
        modality_token_ranges: dict[str, tuple[int, int]],
        modality_ids: dict[str, int],
    ):
        super().__init__()
        self.fusion_transformer = fusion_transformer
        self.forecasting_heads = forecasting_heads
        self._frozen_decoders = frozen_decoders
        self.modality_token_ranges = modality_token_ranges
        self.modality_ids = modality_ids

    @property
    def frozen_decoders(self) -> dict[str, nn.Module]:
        return self._frozen_decoders

    def forward(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : dict[str, Tensor]
            ``{signal_name: (B, n_tokens, d_model)}`` input tokens.

        Returns
        -------
        dict[str, Tensor]
            ``{signal_name: (B, n_tokens, d_model)}`` predicted tokens.
        """
        # Build token_list for fusion transformer
        token_list = []
        for signal_name in sorted(inputs.keys()):
            tokens = inputs[signal_name]
            modality_id = self.modality_ids[signal_name]
            token_list.append((tokens, modality_id))

        # Fused output: (B, total_tokens, d_model)
        fused = self.fusion_transformer(token_list)

        # Per-modality forecasting
        predicted_tokens = {}
        for signal_name, head in self.forecasting_heads.items():
            start, end = self.modality_token_ranges[signal_name]
            predicted_tokens[signal_name] = head(fused[:, start:end])

        return predicted_tokens

    def decode(
        self,
        predicted_tokens: dict[str, torch.Tensor],
        output_shapes: dict[str, tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode predicted tokens back to observation space.

        Uses frozen decoders. Gradients flow through the token inputs
        but decoder parameters are not updated.

        Parameters
        ----------
        predicted_tokens : dict[str, Tensor]
            ``{signal_name: (B, n_tokens, d_model)}``.
        output_shapes : dict, optional
            Per-signal target shapes for the decoder.

        Returns
        -------
        dict[str, Tensor]
            ``{signal_name: decoded_observation}``.
        """
        decoded = {}
        for sig, tokens in predicted_tokens.items():
            if sig in self._frozen_decoders:
                shape = None
                if output_shapes and sig in output_shapes:
                    shape = output_shapes[sig]
                decoded[sig] = self._frozen_decoders[sig](tokens, output_shape=shape)
        return decoded


def build_prediction_model(
    input_signals: list[str],
    target_signals: list[str],
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    token_counts: dict[str, int],
    frozen_decoders: dict[str, nn.Module],
) -> MultimodalPredictionModel:
    """Factory function for building the Stage 3 model.

    Parameters
    ----------
    input_signals : list of str
        All input signal names.
    target_signals : list of str
        Signal names to predict.
    d_model : int
        Token dimension.
    n_heads, n_layers : int
        Transformer hyperparameters.
    dropout : float
        Dropout rate.
    token_counts : dict
        ``{signal_name: n_tokens}`` for each input signal.
    frozen_decoders : dict
        ``{signal_name: frozen_decoder_module}`` for target signals.

    Returns
    -------
    MultimodalPredictionModel
    """
    # Compute token ranges and modality IDs
    offset = 0
    modality_token_ranges = {}
    modality_ids = {}
    for i, sig in enumerate(sorted(input_signals)):
        n_tok = token_counts[sig]
        modality_token_ranges[sig] = (offset, offset + n_tok)
        modality_ids[sig] = i
        offset += n_tok

    total_tokens = offset

    fusion_transformer = BaselineFusionTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        n_modalities=len(input_signals),
        max_tokens=total_tokens,
    )

    forecasting_heads = nn.ModuleDict(
        {
            sig: ForecastingHead(d_model, token_counts[sig])
            for sig in target_signals
            if sig in modality_token_ranges
        }
    )

    return MultimodalPredictionModel(
        fusion_transformer=fusion_transformer,
        forecasting_heads=forecasting_heads,
        frozen_decoders=frozen_decoders,
        modality_token_ranges=modality_token_ranges,
        modality_ids=modality_ids,
    )
