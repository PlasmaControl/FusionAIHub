from typing import Optional

import torch
import torch.nn as nn


class PerceiverCrossAttentionBlock(nn.Module):
    """
    Cross-attention block for Perceiver architecture.
    Queries attend to context via cross-attention.
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries, context):
        """
        Parameters
        ----------
        queries : torch.Tensor
            Shape [batch, n_queries, d_model]
        context : torch.Tensor
            Shape [batch, n_context, d_model]

        Returns
        -------
        torch.Tensor
            Shape [batch, n_queries, d_model]
        """
        # Cross-attention: queries attend to context
        attn_out, _ = self.cross_attn(
            query=queries,
            key=context,
            value=context,
        )
        queries = self.norm1(queries + attn_out)

        # Feed-forward
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)

        return queries


class PerceiverSelfAttentionBlock(nn.Module):
    """
    Self-attention block for processing latent array.
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, n_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Shape [batch, n_tokens, d_model]
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PerceiverEncoder(nn.Module):
    """
    Encodes input tokens to fixed-size latent array via cross-attention.

    Parameters
    ----------
    d_model : int
        Model dimension
    n_latent_queries : int
        Number of latent queries (size of bottleneck)
    n_layers : int
        Number of cross-attention layers
    n_heads : int
        Number of attention heads
    dropout : float
        Dropout rate
    """

    def __init__(
            self,
            d_model=512,
            n_latent_queries=256,
            n_layers=2,
            n_heads=8,
            dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latent_queries = n_latent_queries

        # Learned latent queries (the "plasma state")
        self.latent_queries = nn.Parameter(
            torch.randn(n_latent_queries, d_model)
        )

        # Stack of cross-attention blocks
        self.cross_attn_blocks = nn.ModuleList([
            PerceiverCrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input_tokens):
        """
        Encode input tokens to latent array.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Concatenated tokens from all modalities
            Shape [batch, n_input_tokens, d_model]

        Returns
        -------
        torch.Tensor
            Latent array, shape [batch, n_latent_queries, d_model]
        """
        batch_size = input_tokens.shape[0]

        # Initialize latent with learned queries
        latent = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attend to input tokens
        for block in self.cross_attn_blocks:
            latent = block(queries=latent, context=input_tokens)

        return latent


class LatentProcessor(nn.Module):
    """
    Processes latent array with self-attention.

    Parameters
    ----------
    d_model : int
        Model dimension
    n_layers : int
        Number of self-attention layers
    n_heads : int
        Number of attention heads
    dropout : float
        Dropout rate
    """

    def __init__(
            self,
            d_model=512,
            n_layers=4,
            n_heads=8,
            dropout=0.1
    ):
        super().__init__()

        self.self_attn_blocks = nn.ModuleList([
            PerceiverSelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, latent):
        """
        Process latent array.

        Parameters
        ----------
        latent : torch.Tensor
            Shape [batch, n_latent, d_model]

        Returns
        -------
        torch.Tensor
            Processed latent, shape [batch, n_latent, d_model]
        """
        for block in self.self_attn_blocks:
            latent = block(latent)

        return latent


class DynamicsModel(nn.Module):
    """
    Predicts future latent state from current latent state and actuators.

    Parameters
    ----------
    d_model : int
        Model dimension
    n_actuators : int
        Number of actuator inputs
    n_layers : int
        Number of MLP layers
    dropout : float
        Dropout rate
    mode : str
        'residual' - predict delta (latent_future = latent_current + delta)
        'direct' - predict future directly
    """

    def __init__(
            self,
            d_model=512,
            n_actuators=32,
            n_layers=3,
            dropout=0.1,
            mode='residual'
    ):
        super().__init__()

        self.mode = mode

        layers = []
        input_dim = d_model + n_actuators

        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            ])

        self.dynamics_net = nn.Sequential(*layers)

    def forward(self, latent_current, actuators):
        """
        Predict future latent state.

        Parameters
        ----------
        latent_current : torch.Tensor
            Current latent state, shape [batch, n_latent, d_model]
        actuators : torch.Tensor
            Actuator values, shape [batch, n_actuators]

        Returns
        -------
        torch.Tensor
            Future latent state, shape [batch, n_latent, d_model]
        """
        batch_size, n_latent, d_model = latent_current.shape

        # Flatten latent for processing
        latent_flat = latent_current.reshape(batch_size * n_latent, d_model)

        # Expand actuators to match latent dimension
        actuators_expanded = actuators.unsqueeze(1).expand(-1, n_latent, -1)
        actuators_flat = actuators_expanded.reshape(batch_size * n_latent, -1)

        # Concatenate and process
        combined = torch.cat([latent_flat, actuators_flat], dim=1)

        if self.mode == 'residual':
            # Predict delta
            delta = self.dynamics_net(combined)
            delta = delta.reshape(batch_size, n_latent, d_model)
            latent_future = latent_current + delta
        else:
            # Predict future directly
            latent_future = self.dynamics_net(combined)
            latent_future = latent_future.reshape(
                batch_size, n_latent, d_model
            )

        return latent_future


class DynamicsModelWithFuture(nn.Module):
    """
    Predicts future latent state from:
    - Current latent state
    - Current actuator values
    - Future actuator values

    Parameters
    ----------
    d_model : int
        Model dimension
    n_actuators : int
        Number of actuator inputs
    n_layers : int
        Number of MLP layers
    dropout : float
        Dropout rate
    mode : str
        'residual' - predict delta (latent_future = latent_current + delta)
        'direct' - predict future directly
    """

    def __init__(
            self,
            d_model=512,
            n_actuators=32,
            n_layers=3,
            dropout=0.1,
            mode='residual'
    ):
        super().__init__()

        self.mode = mode

        # Input: latent + current_actuators + future_actuators
        input_dim = d_model + 2 * n_actuators

        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
            else:
                layers.extend([
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])

        self.dynamics_net = nn.Sequential(*layers)

    def forward(self, latent_current, actuators_current, actuators_future):
        """
        Predict future latent state.

        Parameters
        ----------
        latent_current : torch.Tensor
            Current latent state [B, N_L, D]
        actuators_current : torch.Tensor
            Current actuator values [B, D_act]
        actuators_future : torch.Tensor
            Future actuator values [B, D_act]

        Returns
        -------
        torch.Tensor
            Future latent state [B, N_L, D]
        """
        B, N_L, D = latent_current.shape

        # Flatten latent
        latent_flat = latent_current.reshape(B * N_L, D)

        # Expand actuators to match each latent query
        act_curr_exp = actuators_current.unsqueeze(1).expand(-1, N_L, -1)
        act_curr_flat = act_curr_exp.reshape(B * N_L, -1)

        act_fut_exp = actuators_future.unsqueeze(1).expand(-1, N_L, -1)
        act_fut_flat = act_fut_exp.reshape(B * N_L, -1)

        # Concatenate: [latent, act_current, act_future]
        combined = torch.cat([latent_flat, act_curr_flat, act_fut_flat], dim=1)

        # MLP
        if self.mode == 'residual':
            delta = self.dynamics_net(combined)
            delta = delta.reshape(B, N_L, D)
            latent_future = latent_current + delta
        else:
            latent_future = self.dynamics_net(combined)
            latent_future = latent_future.reshape(B, N_L, D)

        return latent_future


class _DeltaCrossAttentionBlock(nn.Module):
    """Cross-attention block **without** internal residual connections.

    Used in the dynamics delta network so that the output is computed
    entirely from the cross-attention to the context (actuators + state).
    There is no skip connection that would let the input pass through
    unchanged, forcing the block to use the context.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, context: torch.Tensor):
        x, _ = self.cross_attn(query=queries, key=context, value=context)
        x = self.norm1(x)
        x = self.norm2(self.ffn(x))
        return x


class CrossAttentionDynamics(nn.Module):
    """
    Predicts future latent state as ``latent_current + delta``.

    The delta is computed by cross-attending to both the current latent
    and the actuator tokens.  The delta network uses blocks **without**
    internal residual connections, so there is no free identity path —
    the model must actively use the actuator context to produce each
    output element.

    Parameters
    ----------
    d_model : int
        Model dimension.
    actuator_configs : dict
        ``{name: {"n_channels": int, "patch_len": int, "target_fs": float}}``.
        Passed to :class:`ActuatorTokenizer`.
    n_cross_layers : int
        Number of cross-attention layers in the delta network.
    n_self_layers : int
        Number of self-attention layers after cross-attention.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    mode : str
        Kept for checkpoint compatibility; ignored.
    """

    def __init__(
        self,
        d_model: int = 512,
        actuator_configs: Optional[dict] = None,
        n_cross_layers: int = 2,
        n_self_layers: int = 1,
        n_heads: int = 8,
        n_latent: int = 128,
        dropout: float = 0.1,
        mode: str = "residual",
    ):
        super().__init__()
        from .modality_tokenizer import ActuatorTokenizer

        if actuator_configs is None:
            actuator_configs = {}

        self.actuator_tokenizer = ActuatorTokenizer(
            actuator_configs, d_model,
        )

        # Delta network: no internal residuals → no free copy path.
        # Queries cross-attend to (latent_current ⊕ actuator_tokens)
        # so the delta is informed by both state and control.
        self.delta_cross_blocks = nn.ModuleList([
            _DeltaCrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        self.delta_self_blocks = nn.ModuleList([
            PerceiverSelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

        # Learned delta queries — NOT initialized from latent_current,
        # so the delta network starts from a neutral state and must
        # extract everything from the context.
        self.delta_queries = nn.Parameter(
            torch.randn(1, n_latent, d_model) * 0.02
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        latent_current: torch.Tensor,
        act_curr_signals: dict,
        act_fut_signals: dict,
        offset_ms: float = 0.0,
        dt_ms: float = 50.0,
    ) -> torch.Tensor:
        """
        Predict future latent state via ``latent_current + delta``.

        The delta is computed by learned queries that cross-attend to
        the concatenation of ``latent_current`` and actuator tokens.

        Parameters
        ----------
        latent_current : torch.Tensor
            Current latent state ``[B, N_L, D]``.
        act_curr_signals : dict
            ``{name: [B, C, T_step]}`` — raw actuator signals for the
            current ``DT_S`` window.
        act_fut_signals : dict
            ``{name: [B, C, T_step]}`` — raw actuator signals for the
            next ``DT_S`` window.
        offset_ms : float
            Absolute time offset (for sinusoidal time PE).
        dt_ms : float
            Duration of one dynamics step in milliseconds.

        Returns
        -------
        torch.Tensor
            Predicted future latent ``[B, N_L, D]``.
        """
        B = latent_current.shape[0]

        # Tokenize current and future actuator windows
        act_curr_tokens = self.actuator_tokenizer(
            act_curr_signals, offset_ms=offset_ms,
        )
        act_fut_tokens = self.actuator_tokenizer(
            act_fut_signals, offset_ms=offset_ms + dt_ms,
        )

        # Context = current latent ⊕ current actuators ⊕ future actuators
        context = torch.cat(
            [latent_current, act_curr_tokens, act_fut_tokens], dim=1,
        )

        # Delta queries cross-attend to context (no residual → must
        # use context to produce every output element)
        delta = self.delta_queries.expand(B, -1, -1)
        for block in self.delta_cross_blocks:
            delta = block(queries=delta, context=context)

        # Self-attention for inter-query communication
        for block in self.delta_self_blocks:
            delta = block(delta)

        return self.output_norm(latent_current + delta)


class PerceiverDecoder(nn.Module):
    """
    Decodes latent array to output tokens via interleaved cross- and
    self-attention (Perceiver IO style).

    Each decoder layer consists of a cross-attention block (output queries
    attend to the latent) followed by a self-attention block (output tokens
    exchange information).  Interleaving allows iterative refinement: later
    layers can query the latent with refined, context-aware queries rather
    than only seeing it once.

    Parameters
    ----------
    d_model : int
        Model dimension.
    output_queries_config : dict
        ``{modality_name: n_tokens}`` — learned output queries per modality.
    n_layers : int
        Number of interleaved (cross-attn + self-attn) blocks per modality.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    n_self_attn_layers : int
        Ignored (kept for backward compat).  Each layer always includes
        one self-attention block after the cross-attention.
    """

    def __init__(
            self,
            d_model=512,
            output_queries_config=None,
            n_layers=2,
            n_heads=8,
            dropout=0.1,
            n_self_attn_layers=0,
    ):
        super().__init__()

        if output_queries_config is None:
            output_queries_config = {
                'ts': 50,
                'prof': 10,
                'vid': 30,
                'spec': 30
            }

        self.d_model = d_model
        self.n_layers = n_layers

        # Learned output queries per modality
        self.output_queries = nn.ParameterDict({
            modality: nn.Parameter(torch.randn(n_tokens, d_model))
            for modality, n_tokens in output_queries_config.items()
        })

        # Interleaved (cross-attn, self-attn) blocks per modality
        self.cross_attn_blocks = nn.ModuleDict({
            modality: nn.ModuleList([
                PerceiverCrossAttentionBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])
            for modality in output_queries_config.keys()
        })
        self.self_attn_blocks = nn.ModuleDict({
            modality: nn.ModuleList([
                PerceiverSelfAttentionBlock(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ])
            for modality in output_queries_config.keys()
        })

    def _decode_modality(self, mod: str, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        tokens = self.output_queries[mod].unsqueeze(0).expand(
            batch_size, -1, -1
        )
        for cross_blk, self_blk in zip(
            self.cross_attn_blocks[mod],
            self.self_attn_blocks[mod],
        ):
            tokens = cross_blk(queries=tokens, context=latent)
            tokens = self_blk(tokens)
        return tokens

    def forward(self, latent, modality=None):
        """
        Decode latent to output tokens.

        Parameters
        ----------
        latent : torch.Tensor
            Latent array, shape ``[batch, n_latent, d_model]``.
        modality : str or None
            If specified, only decode this modality.
            If ``None``, decode all modalities.

        Returns
        -------
        dict or torch.Tensor
            If *modality* is ``None``: dict mapping modality names to output
            tokens.  Otherwise: output tokens for that modality.
            Each output has shape ``[batch, n_output_tokens, d_model]``.
        """
        if modality is not None:
            return self._decode_modality(modality, latent)

        return {
            mod: self._decode_modality(mod, latent)
            for mod in self.output_queries.keys()
        }


class PerceiverComponents(nn.Module):
    """
    Complete Perceiver architecture with future actuator support.
    """
    def __init__(
            self,
            d_model=512,
            n_latent_queries=256,
            n_actuators=32,
            output_queries_config=None,
            encoder_layers=2,
            processor_layers=4,
            decoder_layers=2,
            dynamics_layers=3,
            n_heads=8,
            dropout=0.1,
            dynamics_mode='residual'
    ):
        super().__init__()

        self.encoder = PerceiverEncoder(
            d_model=d_model,
            n_latent_queries=n_latent_queries,
            n_layers=encoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        self.processor = LatentProcessor(
            d_model=d_model,
            n_layers=processor_layers,
            n_heads=n_heads,
            dropout=dropout
        )

        # Updated dynamics with future actuators
        self.dynamics = DynamicsModelWithFuture(
            d_model=d_model,
            n_actuators=n_actuators,
            n_layers=dynamics_layers,
            dropout=dropout,
            mode=dynamics_mode
        )

        self.decoder = PerceiverDecoder(
            d_model=d_model,
            output_queries_config=output_queries_config,
            n_layers=decoder_layers,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, input_tokens, actuators_current, actuators_future):
        """
        Full forward pass through Perceiver.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Concatenated input tokens [B, N_in, D]
        actuators_current : torch.Tensor
            Current actuator values [B, D_act]
        actuators_future : torch.Tensor
            Future actuator values [B, D_act]

        Returns
        -------
        tuple
            (output_tokens, latent_current, latent_future)
        """
        # Encode to latent
        latent_current = self.encoder(input_tokens)

        # Process latent
        latent_current = self.processor(latent_current)

        # Predict future latent (using both current and future actuators)
        latent_future = self.dynamics(
            latent_current,
            actuators_current,
            actuators_future
        )

        # Decode to output tokens
        output_tokens = self.decoder(latent_future)

        return output_tokens, latent_current, latent_future


# Example usage
if __name__ == "__main__":
    # Configuration
    d_model = 512
    batch_size = 4
    n_input_tokens = 200  # Total from all modalities
    n_actuators = 32

    # Create Perceiver components
    perceiver = PerceiverComponents(
        d_model=d_model,
        n_latent_queries=256,
        n_actuators=n_actuators,
        output_queries_config={
            'ts': 50,
            'prof': 10,
            'vid': 30,
            'spec': 30
        },
        encoder_layers=2,
        processor_layers=4,
        decoder_layers=2,
        n_heads=8,
        dropout=0.1
    )

    # Dummy inputs
    input_tokens = torch.randn(batch_size, n_input_tokens, d_model)
    actuators = torch.randn(batch_size, n_actuators)

    # Forward pass
    output_tokens, latent_current, latent_future = perceiver(
        input_tokens, actuators
    )

    print(f"Input tokens:     {input_tokens.shape}")
    print(f"Latent current:   {latent_current.shape}")
    print(f"Latent future:    {latent_future.shape}")
    print(f"Output tokens:")
    for modality, tokens in output_tokens.items():
        print(f"  {modality}: {tokens.shape}")
