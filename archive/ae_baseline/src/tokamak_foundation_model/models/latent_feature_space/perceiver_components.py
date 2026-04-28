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


class _DynamicsCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block **without** query residual.

    Uses pre-norm (normalize inputs, not outputs) so the residual stream
    is unbounded across recurrent rollout steps.  Post-norm would cap
    ``delta_k`` at ~sqrt(d_model) every step, causing the dynamics to
    converge to a fixed point.

    The output is derived entirely from cross-attention to the actuator
    context (values).  There is no skip connection from queries to output,
    so the block cannot pass queries through unchanged.  The queries
    (from ``latent_current``) determine *what* to attend to via Q-K
    alignment, but the output is built from values only.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor):
        # Pre-norm on queries only. Context (actuator tokens) is already
        # LayerNormed by ActuatorTokenizer — per-token LN here would
        # kill uniform-value tokens.
        q_norm = self.norm_q(queries)
        attn_out, _ = self.cross_attn(
            query=q_norm, key=context, value=context)
        # NO residual from queries — output is pure attention
        # FFN with pre-norm residual (from attn_out, not queries)
        x = attn_out + self.ffn(self.norm_ffn(attn_out))
        return x


class _DynamicsPreNormSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block for the dynamics recurrent path.

    Unlike :class:`PerceiverSelfAttentionBlock` (post-norm), this
    normalizes *inputs* rather than *outputs*.  In a recurrent path
    the delta is added to a growing latent, so post-norm's bounded
    output would shrink delta relative to the latent over rollout
    steps.  Pre-norm keeps the residual stream unbounded.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionDynamics(nn.Module):
    """
    Predicts future latent state as ``latent_current + delta``.

    1. **Cross-attention** (no query residual) extracts actuator
       information routed by the current plasma state.
    2. **Fusion MLP** combines this actuator info with the current
       latent state token-wise, enabling ``delta = f(state, actuators)``
       instead of ``delta = g(actuators)``.
    3. **Self-attention** allows inter-token communication.
    4. **Residual** output: ``latent_current + del``.

    The cross-attention blocks still have no query residual, so the
    actuator path can never be bypassed.  The fusion MLP provides
    state-dependent modulation of the actuator-derived signal.

    Parameters
    ----------
    d_model : int
        Model dimension.
    actuator_configs : dict
        ``{name: {"n_channels": int, "patch_len": int, "target_fs": float}}``.
        Passed to :class:`ActuatorTokenizer`.
    n_cross_layers : int
        Number of cross-attention layers.
    n_self_layers : int
        Number of self-attention layers after cross-attention.
    n_heads : int
        Number of attention heads.
    n_latent : int
        Kept for checkpoint compatibility; ignored.
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

        self.d_model = d_model

        if actuator_configs is None:
            actuator_configs = {}

        self.actuator_tokenizer = ActuatorTokenizer(
            actuator_configs, d_model,
        )

        # Pre-norm cross-attention: latent_current queries attend to
        # actuator tokens.  No query residual — output is purely
        # actuator-derived.  Pre-norm keeps the residual stream
        # unbounded across rollout steps.
        self.cross_blocks = nn.ModuleList([
            _DynamicsCrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_cross_layers)
        ])

        # Gated query residual: allows state information to leak through
        # the cross-attention when actuators are slowly varying.
        # Initialized near-closed (bias=-3 → sigmoid≈0.05) so the model
        # starts with minimal state leakage and learns to open the gate.
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate_proj.bias, -3.0)

        # Step embedding: Fourier-encode offset_ms through an MLP so
        # the dynamics can distinguish step 1 from step 15.  Without
        # this, the model receives near-identical inputs at every step
        # and copy is the expected result.
        self.step_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Token-wise fusion: combines actuator info, current state,
        # previous state (velocity info), and step embedding.
        # Input dim is 4*d_model:
        #   [act_info; latent_current; latent_prev; step_embed]
        self.fusion_net = nn.Sequential(
            nn.Linear(4 * d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # Pre-norm self-attention for inter-query communication.
        # Pre-norm keeps delta magnitude unbounded.
        self.self_blocks = nn.ModuleList([
            _DynamicsPreNormSelfAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_self_layers)
        ])

    def forward(
        self,
        latent_current: torch.Tensor,
        act_curr_signals: dict,
        act_fut_signals: dict,
        offset_ms: float = 0.0,
        dt_ms: float = 50.0,
        latent_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict future latent state.

        Cross-attention extracts actuator info (no query residual),
        then a fusion MLP combines it with ``latent_current``,
        ``latent_prev`` (implicit velocity), and a step embedding
        to compute a state-dependent delta.

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
        latent_prev : torch.Tensor or None
            Previous latent state ``[B, N_L, D]``.  Provides implicit
            velocity information.  If ``None`` (first step), uses
            ``latent_current`` (zero velocity assumption).

        Returns
        -------
        torch.Tensor
            Predicted future latent ``[B, N_L, D]``.
        """
        from .modality_tokenizer import sinusoidal_time_encoding

        B, N_L, D = latent_current.shape
        device = latent_current.device

        if latent_prev is None:
            latent_prev = latent_current

        # Tokenize current and future actuator windows
        act_curr_tokens = self.actuator_tokenizer(
            act_curr_signals, offset_ms=offset_ms,
        )
        act_fut_tokens = self.actuator_tokenizer(
            act_fut_signals, offset_ms=offset_ms + dt_ms,
        )

        # Context = current actuators ⊕ future actuators
        # (latent_current is NOT in the context — it IS the queries)
        context = torch.cat(
            [act_curr_tokens, act_fut_tokens], dim=1,
        )

        # State-dependent cross-attention WITHOUT query residual.
        # The output is in the span of actuator value vectors —
        # latent_current only affects attention routing (Q-K alignment).
        act_info = latent_current
        for block in self.cross_blocks:
            act_info = block(queries=act_info, context=context)

        # Gated query residual: blend act_info with latent_current.
        # When actuators change slowly, act_info is near-identical at
        # every step.  The gate lets state information leak through.
        gate = torch.sigmoid(self.gate_proj(latent_current))  # [B,N_L,1]
        act_info = (1 - gate) * act_info + gate * latent_current

        # Step embedding: Fourier-encode absolute time so the dynamics
        # can distinguish different rollout steps.
        t_ms = torch.tensor(
            [[offset_ms]], device=device, dtype=torch.float32,
        ).expand(B, 1)
        step_enc = sinusoidal_time_encoding(t_ms, self.d_model)  # [B,1,D]
        step_embed = self.step_mlp(step_enc.squeeze(1))  # [B, D]
        step_embed = step_embed.unsqueeze(1).expand(-1, N_L, -1)  # [B,N_L,D]

        # Token-wise fusion: combine actuator info, current state,
        # previous state (velocity), and step embedding.
        delta = self.fusion_net(
            torch.cat([act_info, latent_current, latent_prev, step_embed],
                      dim=-1))

        # Pre-norm self-attention for inter-query communication
        for block in self.self_blocks:
            delta = block(delta)

        return latent_current + delta


class GRUDynamics(nn.Module):
    """
    GRU-based dynamics for autoregressive latent prediction.

    A GRU cell is applied independently to each latent query, with
    actuator signals as the input at each step.  The hidden state IS
    the latent query — it evolves naturally through rollout steps,
    giving the model temporal memory that feedforward dynamics lacks.

    Actuator signals are tokenized via :class:`ActuatorTokenizer`,
    mean-pooled to a fixed-size embedding, and projected to the GRU
    input dimension.

    Parameters
    ----------
    d_model : int
        Model dimension (= latent query dimension).
    actuator_configs : dict
        Passed to :class:`ActuatorTokenizer`.
    n_latent : int
        Number of latent queries (kept for API compatibility).
    dropout : float
        Dropout rate.
    mode : str
        Kept for API compatibility; ignored.
    """

    def __init__(
        self,
        d_model: int = 256,
        actuator_configs: Optional[dict] = None,
        n_latent: int = 128,
        dropout: float = 0.1,
        mode: str = "residual",
        **kwargs,
    ):
        super().__init__()
        from .modality_tokenizer import ActuatorTokenizer

        if actuator_configs is None:
            actuator_configs = {}

        self.actuator_tokenizer = ActuatorTokenizer(
            actuator_configs, d_model,
        )

        # Project current + future actuator embeddings → GRU input
        self.act_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )

        # GRU cell: input = actuator embedding, hidden = latent query
        self.gru = nn.GRUCell(input_size=d_model, hidden_size=d_model)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        latent_current: torch.Tensor,
        act_curr_signals: dict,
        act_fut_signals: dict,
        offset_ms: float = 0.0,
        dt_ms: float = 100.0,
    ) -> torch.Tensor:
        """
        One-step GRU dynamics update.

        Parameters
        ----------
        latent_current : torch.Tensor
            Current latent state ``[B, N_L, D]``.  Used as GRU hidden
            state (each query independently).
        act_curr_signals : dict
            ``{name: [B, C, T_step]}`` — current actuator window.
        act_fut_signals : dict
            ``{name: [B, C, T_step]}`` — future actuator window.
        offset_ms : float
            Absolute time offset for actuator PE.
        dt_ms : float
            Duration of one dynamics step in ms.

        Returns
        -------
        torch.Tensor
            Next latent state ``[B, N_L, D]``.
        """
        B, N_L, D = latent_current.shape

        # Tokenize and mean-pool actuators → fixed-size embeddings
        act_curr_tokens = self.actuator_tokenizer(
            act_curr_signals, offset_ms=offset_ms,
        )  # [B, N_act, D]
        act_fut_tokens = self.actuator_tokenizer(
            act_fut_signals, offset_ms=offset_ms + dt_ms,
        )  # [B, N_act, D]

        act_curr_embed = act_curr_tokens.mean(dim=1)  # [B, D]
        act_fut_embed = act_fut_tokens.mean(dim=1)     # [B, D]

        # Project to GRU input
        act_input = self.act_proj(
            torch.cat([act_curr_embed, act_fut_embed], dim=-1)
        )  # [B, D]

        # Expand to each latent query and flatten
        act_input = act_input.unsqueeze(1).expand(-1, N_L, -1)
        act_flat = act_input.reshape(B * N_L, D)   # [B*N_L, D]
        h_flat = latent_current.reshape(B * N_L, D) # [B*N_L, D]

        # GRU step
        h_next = self.gru(act_flat, h_flat)  # [B*N_L, D]

        return self.output_norm(h_next.reshape(B, N_L, D))


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
