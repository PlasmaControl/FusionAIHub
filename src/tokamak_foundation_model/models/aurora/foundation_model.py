"""
Aurora-inspired tokamak foundation model.

The model takes AE tokens as input ("observation space") and predicts
AE tokens at the next timestep.  A full encode → backbone → decode pass
runs at every rollout step.  Predictions are fed back as input in
AE token space — no latent accumulation, no distribution drift.

Frozen AEs sit outside this model as preprocessing/postprocessing.
"""

from typing import Optional

import torch
import torch.nn as nn

from tokamak_foundation_model.models.latent_feature_space.modality_tokenizer import (
    ActuatorTokenizer,
    ModalityTokenizer,
)

from .backbone import LatentBackbone
from .encoder_decoder import PerceiverDecoder, PerceiverEncoder


class TokamakFoundationModel(nn.Module):
    """Aurora-inspired foundation model for tokamak plasma prediction.

    Each call to :meth:`forward` runs the full pipeline:
    tokenize → encode → backbone → decode → project.  During rollout,
    the output AE tokens are fed back as input — the model never
    accumulates deltas in a compressed latent space.

    Parameters
    ----------
    modality_configs : dict
        ``{name: {"d_lat": int, "n_tokens": int}}``.
    d_model : int
        Common model dimension.
    n_latent : int
        Number of Perceiver latent queries.
    n_heads : int
        Attention heads throughout.
    encoder_cross_layers : int
        Cross-attention layers in the Perceiver encoder.
    encoder_self_layers : int
        Self-attention layers in the Perceiver encoder.
    backbone_blocks : int
        Number of Transformer blocks in the latent backbone.
    decoder_layers : int
        Interleaved (cross + self) layers in the Perceiver decoder.
    mlp_ratio : float
        FFN hidden dim = ``d_model * mlp_ratio``.
    dropout : float
        Dropout rate.
    actuator_configs : dict or None
        ``{name: {"n_channels": int, "patch_len": int, "target_fs": float}}``.
    window_ms : float
        Context window duration in milliseconds.
    use_skips : bool
        U-Net skip connections in the backbone.
    """

    def __init__(
        self,
        modality_configs: dict,
        d_model: int = 256,
        n_latent: int = 128,
        n_heads: int = 8,
        encoder_cross_layers: int = 2,
        encoder_self_layers: int = 2,
        backbone_blocks: int = 8,
        decoder_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        actuator_configs: Optional[dict] = None,
        window_ms: float = 500.0,
        use_skips: bool = True,
    ):
        super().__init__()

        # Tokenizers (reused from latent_feature_space)
        self.modality_tokenizer = ModalityTokenizer(
            modality_configs=modality_configs,
            d_model=d_model,
            window_ms=window_ms,
        )
        self.actuator_tokenizer: Optional[ActuatorTokenizer] = None
        if actuator_configs is not None:
            self.actuator_tokenizer = ActuatorTokenizer(
                actuator_configs, d_model,
            )

        # Perceiver encoder
        self.encoder = PerceiverEncoder(
            d_model=d_model,
            n_latent_queries=n_latent,
            n_cross_layers=encoder_cross_layers,
            n_self_layers=encoder_self_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Deep backbone (the main capacity)
        self.backbone = LatentBackbone(
            d_model=d_model,
            n_blocks=backbone_blocks,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_skips=use_skips,
        )

        # Perceiver decoder
        output_queries_config = {
            name: cfg["n_tokens"]
            for name, cfg in modality_configs.items()
        }
        self.decoder = PerceiverDecoder(
            d_model=d_model,
            output_queries_config=output_queries_config,
            n_layers=decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Project from d_model back to each modality's d_lat
        self.output_projections = nn.ModuleDict({
            name: nn.Linear(d_model, cfg["d_lat"], bias=False)
            for name, cfg in modality_configs.items()
        })

    def forward(
        self,
        ae_tokens: dict,
        act_curr_signals: dict,
        act_fut_signals: dict,
        step_index: int = 0,
        offset_ms: float = 0.0,
        dt_ms: float = 500.0,
    ) -> dict:
        """Single-step forward: AE tokens in → AE tokens out.

        Parameters
        ----------
        ae_tokens : dict
            ``{modality: Tensor[B, N_m, d_lat_m]}`` — current state
            in AE token space.
        act_curr_signals : dict
            ``{name: Tensor[B, C, T_samples]}`` — raw actuator signals
            for the current DT_S window.
        act_fut_signals : dict
            ``{name: Tensor[B, C, T_samples]}`` — raw actuator signals
            for the next DT_S window.
        step_index : int
            Rollout step (0, 1, 2, ...).
        offset_ms : float
            Absolute time offset in ms.
        dt_ms : float
            Duration of one dynamics step in ms.

        Returns
        -------
        dict
            ``{modality: Tensor[B, N_m, d_lat_m]}`` — predicted AE
            tokens at the next timestep.
        """
        # 1. Tokenize diagnostics
        diag_tokens = self.modality_tokenizer(ae_tokens)

        # 2. Tokenize actuators (current + future windows)
        if self.actuator_tokenizer is not None:
            act_curr_tok = self.actuator_tokenizer(
                act_curr_signals, offset_ms=offset_ms)
            act_fut_tok = self.actuator_tokenizer(
                act_fut_signals, offset_ms=offset_ms + dt_ms)
            act_tokens = torch.cat([act_curr_tok, act_fut_tok], dim=1)
            encoder_input = torch.cat([diag_tokens, act_tokens], dim=1)
        else:
            act_tokens = torch.zeros(
                diag_tokens.shape[0], 0, diag_tokens.shape[2],
                device=diag_tokens.device)
            encoder_input = diag_tokens

        # 3. Encode: compress into fixed-size latent
        latent = self.encoder(encoder_input)

        # 4. Backbone: predict next latent state
        latent_next = self.backbone(
            latent, act_tokens, step_index=step_index, offset_ms=offset_ms)

        # 5. Decode: expand back to per-modality tokens
        decoded = self.decoder(latent_next)

        # 6. Project to AE latent dimensions
        return {
            name: self.output_projections[name](tokens)
            for name, tokens in decoded.items()
        }

    @torch.no_grad()
    def rollout(
        self,
        ae_tokens_context: dict,
        actuator_step_pairs: list,
        n_steps: Optional[int] = None,
        window_ms: float = 500.0,
        dt_ms: float = 500.0,
    ) -> list:
        """Autoregressive rollout in AE token space.

        The full model runs at every step.  Predictions are fed back
        as input — no latent accumulation.

        Parameters
        ----------
        ae_tokens_context : dict
            ``{modality: Tensor[B, N_m, d_lat_m]}`` — initial state.
        actuator_step_pairs : list
            ``[(act_curr_dict, act_fut_dict), ...]`` per rollout step.
        n_steps : int or None
            Number of steps (defaults to ``len(actuator_step_pairs)``).
        window_ms : float
            Context window duration in ms.
        dt_ms : float
            Step duration in ms.

        Returns
        -------
        list of dict
            One ``{modality: Tensor[B, N_m, d_lat_m]}`` per step.
        """
        if n_steps is None:
            n_steps = len(actuator_step_pairs)

        current = ae_tokens_context
        predictions = []

        for k in range(n_steps):
            act_curr, act_fut = actuator_step_pairs[k]
            offset_ms = window_ms + k * dt_ms
            current = self.forward(
                ae_tokens=current,
                act_curr_signals=act_curr,
                act_fut_signals=act_fut,
                step_index=k,
                offset_ms=offset_ms,
                dt_ms=dt_ms,
            )
            predictions.append(current)

        return predictions
