import copy
from typing import Optional

import torch
import torch.nn as nn

from .modality_tokenizer import ActuatorTokenizer, ModalityTokenizer
from .perceiver_components import (
    CrossAttentionDynamics,
    GRUDynamics,
    PerceiverEncoder,
    LatentProcessor,
    DynamicsModelWithFuture,
    PerceiverDecoder,
)


class PerceiverFoundationModel(nn.Module):
    """
    Multi-modal foundation model for autoregressive tokamak state prediction.

    Combines Perceiver IO (Jaegle et al., 2022) for multi-modal
    encode/decode, action-conditioned latent dynamics (Hafner et al., 2019),
    and JEPA-style EMA target encoding (Assran et al., 2023).

    Training objective (JEPA)
    -------------------------
    Given a 500 ms context window (shifted windows differ by ``dt`` ms):

    .. code-block:: text

        latent_ctx    = online_encode(ae_latents of context at t)
        latent_pred   = dynamics(latent_ctx, act_t, act_{t+dt})
        latent_target = ema_encode(ae_latents of target at t+dt)   # no grad
        loss = MSE(latent_pred, latent_target)

    The EMA (exponential moving average) target encoder is a slowly-updated
    copy of the online encoder.  This prevents representation collapse
    without needing contrastive negatives (cf. BYOL, I-JEPA).

    Inference (autoregressive rollout)
    -----------------------------------
    The online encoder is called once on the initial context; subsequent
    steps propagate the latent forward via the dynamics model only.

    Parameters
    ----------
    modality_configs : dict
        ``{name: {"d_lat": int, "n_tokens": int}}`` — passed to
        :class:`ModalityTokenizer`.
    d_model : int
        Model dimension for the Perceiver.  Default 512.
    n_latent : int
        Number of latent queries (compressed state size).  Default 256.
    n_actuators : int
        Dimensionality of the actuator vector fed to the dynamics model.
        Default 32.
    encoder_layers : int
        Number of cross-attention layers in :class:`PerceiverEncoder`.
        Default 2.
    processor_layers : int
        Number of self-attention layers in :class:`LatentProcessor`.
        Default 4.
    decoder_layers : int
        Number of interleaved (cross-attn + self-attn) blocks in
        :class:`PerceiverDecoder`.  Default 2.
    dynamics_layers : int
        Number of MLP layers in :class:`DynamicsModelWithFuture`.  Default 3.
    n_heads : int
        Number of attention heads.  Default 8.
    dropout : float
        Dropout rate.  Default 0.1.
    dynamics_mode : str
        ``'residual'`` (predict delta) or ``'direct'`` (predict absolute).
        Default ``'residual'``.
    window_ms : float
        Duration of the context window in milliseconds.  Default 500.0.
    ema_decay : float
        EMA decay rate for the target encoder.  Default 0.996.
    """

    def __init__(
        self,
        modality_configs: dict,
        d_model: int = 512,
        n_latent: int = 256,
        n_actuators: int = 32,
        encoder_layers: int = 2,
        processor_layers: int = 4,
        decoder_layers: int = 2,
        decoder_self_attn_layers: int = 0,
        dynamics_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
        dynamics_mode: str = "residual",
        dynamics_type: str = "mlp",
        actuator_configs: Optional[dict] = None,
        window_ms: float = 500.0,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.dynamics_type = dynamics_type

        # --- Online encoder (receives gradients) ---
        self.tokenizer = ModalityTokenizer(
            modality_configs=modality_configs,
            d_model=d_model,
            window_ms=window_ms,
        )
        self.encoder = PerceiverEncoder(
            d_model=d_model,
            n_latent_queries=n_latent,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.processor = LatentProcessor(
            d_model=d_model,
            n_layers=processor_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- Actuator tokenizer (for encoder context) ---
        if actuator_configs is not None and dynamics_type in ("cross_attention", "gru"):
            self.actuator_tokenizer: Optional[ActuatorTokenizer] = (
                ActuatorTokenizer(actuator_configs, d_model)
            )
        else:
            self.actuator_tokenizer = None

        # --- EMA target encoder (no gradients, slowly tracks online) ---
        self.ema_tokenizer = copy.deepcopy(self.tokenizer)
        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_processor = copy.deepcopy(self.processor)
        if self.actuator_tokenizer is not None:
            self.ema_actuator_tokenizer: Optional[ActuatorTokenizer] = (
                copy.deepcopy(self.actuator_tokenizer)
            )
        else:
            self.ema_actuator_tokenizer = None
        for p in self.ema_parameters():
            p.requires_grad_(False)

        # --- Dynamics model ---
        if dynamics_type == "cross_attention":
            if actuator_configs is None:
                raise ValueError(
                    "actuator_configs required for cross_attention dynamics"
                )
            self.dynamics = CrossAttentionDynamics(
                d_model=d_model,
                actuator_configs=actuator_configs,
                n_cross_layers=dynamics_layers,
                n_self_layers=1,
                n_heads=n_heads,
                n_latent=n_latent,
                dropout=dropout,
                mode=dynamics_mode,
            )
        elif dynamics_type == "gru":
            if actuator_configs is None:
                raise ValueError(
                    "actuator_configs required for gru dynamics"
                )
            self.dynamics = GRUDynamics(
                d_model=d_model,
                actuator_configs=actuator_configs,
                n_latent=n_latent,
                dropout=dropout,
            )
        else:
            self.dynamics = DynamicsModelWithFuture(
                d_model=d_model,
                n_actuators=n_actuators,
                n_layers=dynamics_layers,
                dropout=dropout,
                mode=dynamics_mode,
            )

        # --- Decoder: Perceiver latent → per-modality AE latent tokens ---
        output_queries_config = {
            name: cfg["n_tokens"] for name, cfg in modality_configs.items()
        }
        self.decoder = PerceiverDecoder(
            d_model=d_model,
            output_queries_config=output_queries_config,
            n_layers=decoder_layers,
            n_heads=n_heads,
            dropout=dropout,
            n_self_attn_layers=decoder_self_attn_layers,
        )
        # Project from Perceiver d_model back to each modality's d_lat
        self.output_projections = nn.ModuleDict({
            name: nn.Linear(d_model, cfg["d_lat"], bias=False)
            for name, cfg in modality_configs.items()
        })

    def ema_parameters(self):
        """Iterate over all EMA target encoder parameters."""
        yield from self.ema_tokenizer.parameters()
        yield from self.ema_encoder.parameters()
        yield from self.ema_processor.parameters()
        if self.ema_actuator_tokenizer is not None:
            yield from self.ema_actuator_tokenizer.parameters()

    @torch.no_grad()
    def update_ema(self):
        """Update EMA target encoder weights toward the online encoder."""
        tau = self.ema_decay
        for p_online, p_ema in zip(self.tokenizer.parameters(),
                                   self.ema_tokenizer.parameters()):
            p_ema.data.lerp_(p_online.data, 1 - tau)
        for p_online, p_ema in zip(self.encoder.parameters(),
                                   self.ema_encoder.parameters()):
            p_ema.data.lerp_(p_online.data, 1 - tau)
        for p_online, p_ema in zip(self.processor.parameters(),
                                   self.ema_processor.parameters()):
            p_ema.data.lerp_(p_online.data, 1 - tau)
        if (self.actuator_tokenizer is not None
                and self.ema_actuator_tokenizer is not None):
            for p_online, p_ema in zip(
                self.actuator_tokenizer.parameters(),
                self.ema_actuator_tokenizer.parameters(),
            ):
                p_ema.data.lerp_(p_online.data, 1 - tau)

    def encode(
        self,
        latents: dict,
        actuator_context: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Encode multi-modal AE latents using the **online** encoder.

        Parameters
        ----------
        latents : dict
            ``{modality: Tensor[B, T_mod, d_lat]}``
        actuator_context : dict or None
            ``{name: Tensor[B, C, T_samples]}`` — raw actuator signals
            covering the context window.  Only used when
            ``dynamics_type='cross_attention'``.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_latent, d_model]``.
        """
        tokens = self.tokenizer(latents)   # [B, N_total, d_model]
        if actuator_context is not None and self.actuator_tokenizer is not None:
            act_tokens = self.actuator_tokenizer(actuator_context)
            tokens = torch.cat([tokens, act_tokens], dim=1)
        latent = self.encoder(tokens)
        return self.processor(latent)      # [B, N_latent, d_model]

    @torch.no_grad()
    def ema_encode(
        self,
        latents: dict,
        actuator_context: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Encode multi-modal AE latents using the **EMA target** encoder.

        No gradients flow through this path.

        Parameters
        ----------
        latents : dict
            ``{modality: Tensor[B, T_mod, d_lat]}``
        actuator_context : dict or None
            Same as in :meth:`encode`.

        Returns
        -------
        torch.Tensor
            Shape ``[B, N_latent, d_model]``.
        """
        tokens = self.ema_tokenizer(latents)
        if actuator_context is not None and self.ema_actuator_tokenizer is not None:
            act_tokens = self.ema_actuator_tokenizer(actuator_context)
            tokens = torch.cat([tokens, act_tokens], dim=1)
        latent = self.ema_encoder(tokens)
        return self.ema_processor(latent)

    def decode(self, latent: torch.Tensor) -> dict:
        """
        Decode a Perceiver latent array to per-modality AE latent tokens.

        Parameters
        ----------
        latent : torch.Tensor
            Shape ``[B, N_latent, d_model]``.

        Returns
        -------
        dict
            ``{modality: Tensor[B, n_tokens, d_lat]}``, matching the shape
            produced by the per-modality AE encoders.
        """
        decoded = self.decoder(latent)  # {name: [B, n_tokens, d_model]}
        return {
            name: self.output_projections[name](tokens)
            for name, tokens in decoded.items()
        }

    def forward(
        self,
        latents_context: dict,
        actuators_current,
        actuators_future,
        actuator_context: Optional[dict] = None,
        offset_ms: float = 0.0,
        dt_ms: float = 50.0,
    ) -> torch.Tensor:
        """
        Predict the next latent state from the current context and actuators.

        Parameters
        ----------
        latents_context : dict
            AE latents of the 500 ms context window.
            ``{modality: Tensor[B, T_mod, d_lat]}``
        actuators_current
            MLP mode: ``Tensor[B, n_actuators]``.
            Cross-attention mode: ``dict {name: Tensor[B, C, T_step]}``.
        actuators_future
            Same type as *actuators_current*.
        actuator_context : dict or None
            Raw actuator signals for the context window (cross-attention
            mode only).
        offset_ms : float
            Absolute time offset for the dynamics step (cross-attention
            mode only).
        dt_ms : float
            Duration of one dynamics step in ms (cross-attention mode only).

        Returns
        -------
        torch.Tensor
            Predicted latent at ``t + dt``, shape ``[B, N_latent, d_model]``.
        """
        latent = self.encode(latents_context, actuator_context)
        if self.dynamics_type in ("cross_attention", "gru"):
            return self.dynamics(
                latent, actuators_current, actuators_future,
                offset_ms=offset_ms, dt_ms=dt_ms,
            )
        return self.dynamics(latent, actuators_current, actuators_future)

    def predict_signals(
        self,
        latents_context: dict,
        actuators_current: torch.Tensor,
        actuators_future: torch.Tensor,
        ae_decoders: dict,
    ) -> dict:
        """
        Full prediction pipeline: encode → dynamics → decode → AE decode.

        Parameters
        ----------
        latents_context : dict
            AE latents of the context window.
            ``{modality: Tensor[B, T_mod, d_lat]}``
        actuators_current : torch.Tensor
            Shape ``[B, n_actuators]``.
        actuators_future : torch.Tensor
            Shape ``[B, n_actuators]``.
        ae_decoders : dict
            ``{modality: nn.Module}`` — frozen AE decoders.

        Returns
        -------
        dict
            ``{modality: Tensor}`` — predicted signals in original space.
        """
        lat_pred = self.forward(latents_context, actuators_current, actuators_future)
        ae_tokens = self.decode(lat_pred)
        return {
            name: ae_decoders[name](tokens)
            for name, tokens in ae_tokens.items()
            if name in ae_decoders
        }

    def rollout_signals(
        self,
        initial_latents: dict,
        actuators_sequence: torch.Tensor,
        ae_decoders: dict,
        n_steps: Optional[int] = None,
    ) -> dict:
        """
        Autoregressive rollout with full signal decoding at each step.

        Parameters
        ----------
        initial_latents : dict
            AE latents of the initial context window.
        actuators_sequence : torch.Tensor
            Shape ``[B, n_steps + 1, n_actuators]``.
        ae_decoders : dict
            ``{modality: nn.Module}`` — frozen AE decoders.
        n_steps : int or None
            Number of prediction steps.

        Returns
        -------
        dict
            ``{modality: Tensor[B, n_steps, ...]}``.
        """
        if n_steps is None:
            n_steps = actuators_sequence.shape[1] - 1

        latent = self.encode(initial_latents)
        all_signals = {name: [] for name in ae_decoders}

        for k in range(n_steps):
            latent = self.dynamics(
                latent,
                actuators_sequence[:, k, :],
                actuators_sequence[:, k + 1, :],
            )
            ae_tokens = self.decode(latent)
            for name, tokens in ae_tokens.items():
                if name in ae_decoders:
                    all_signals[name].append(ae_decoders[name](tokens))

        return {
            name: torch.stack(sigs, dim=1)
            for name, sigs in all_signals.items()
            if sigs
        }

    def rollout(
        self,
        initial_latents: dict,
        actuators_sequence: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively predict ``n_steps`` future latent states.

        The Perceiver encoder is called only once (on the initial context);
        all subsequent steps propagate the latent via the dynamics model.

        Parameters
        ----------
        initial_latents : dict
            AE latents of the initial 500 ms context window.
        actuators_sequence : torch.Tensor
            Shape ``[B, n_steps + 1, n_actuators]``.
            ``actuators_sequence[:, k, :]`` is the actuator vector at step
            ``k``; the dynamics model uses pairs ``(k, k+1)`` at each step.
        n_steps : int or None
            Number of prediction steps.  Inferred from ``actuators_sequence``
            if ``None``.

        Returns
        -------
        torch.Tensor
            Stacked predicted latents, shape ``[B, n_steps, N_latent, d_model]``.
        """
        if n_steps is None:
            n_steps = actuators_sequence.shape[1] - 1

        latent = self.encode(initial_latents)
        predictions = []
        for k in range(n_steps):
            latent = self.dynamics(
                latent,
                actuators_sequence[:, k, :],
                actuators_sequence[:, k + 1, :],
            )
            predictions.append(latent)

        return torch.stack(predictions, dim=1)  # [B, n_steps, N_latent, D]