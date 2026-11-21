"""Factorized spatio-temporal transformer backbone over the frozen codes.

One block is spatial self-attention within a frame, causal temporal self-attention
across frames at each token position, then an FFN, all pre-norm with residuals.
Actuators enter additively per frame, so with the causal temporal mask frame ``t``
is conditioned only on actuators up to ``t``. See ``docs/IGNITE_DESIGN.md``
sections 5.2 and 5.4.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .dynamics_config import DynamicsConfig
from .frame_layout import FrameTokenizer


def _attend_earlier_and_self(q, history_k, history_v, self_k, self_v):
    """Attention of position ``i`` over history positions ``< i`` and its own key.

    One fused ``scaled_dot_product_attention`` over the keys ``[history | self]``
    with the mask ``[earlier | diagonal]``, so the ``(L, 2L)`` scores are never
    stored for backward. On ROCm the masked kernel's backward fails with "HIP
    error: invalid argument" (Frontier), so there it falls back to
    :func:`_attend_earlier_and_self_explicit`.

    Parameters
    ----------
    q, self_k, self_v : torch.Tensor of shape (B, heads, L, head_dim)
        Query-stream projections.
    history_k, history_v : torch.Tensor of shape (B, heads, L, head_dim)
        History-stream keys and values.

    Returns
    -------
    attended : torch.Tensor of shape (B, heads, L, head_dim)
    """
    if torch.version.hip:
        return _attend_earlier_and_self_explicit(q, history_k, history_v, self_k, self_v)
    length = q.shape[-2]
    position = torch.arange(length, device=q.device)
    earlier = position.view(1, -1) < position.view(-1, 1)
    mask = torch.cat([earlier, position.view(1, -1) == position.view(-1, 1)], dim=-1)
    return F.scaled_dot_product_attention(
        q,
        torch.cat([history_k, self_k], dim=-2),
        torch.cat([history_v, self_v], dim=-2),
        attn_mask=mask,
    )


def _attend_earlier_and_self_explicit(q, history_k, history_v, self_k, self_v):
    """:func:`_attend_earlier_and_self` without ``scaled_dot_product_attention``.

    The own key enters as one extra column, so the scores are ``(L, L + 1)``
    instead of ``(L, 2L)``; the softmax runs in fp32.

    Parameters
    ----------
    q, self_k, self_v : torch.Tensor of shape (B, heads, L, head_dim)
        Query-stream projections.
    history_k, history_v : torch.Tensor of shape (B, heads, L, head_dim)
        History-stream keys and values.

    Returns
    -------
    attended : torch.Tensor of shape (B, heads, L, head_dim)
    """
    length = q.shape[-2]
    scale = q.shape[-1] ** -0.5
    history_scores = (q @ history_k.transpose(-1, -2)) * scale
    position = torch.arange(length, device=q.device)
    earlier = position.view(1, -1) < position.view(-1, 1)
    history_scores = history_scores.masked_fill(~earlier, float("-inf"))
    self_scores = (q * self_k).sum(-1, keepdim=True) * scale
    weights = torch.cat([history_scores, self_scores], dim=-1).float().softmax(-1)
    weights = weights.to(q.dtype)
    return weights[..., :length] @ history_v + weights[..., length:] * self_v


class _MHA(nn.Module):
    """Multi-head self-attention over the middle axis of a (B, L, d) tensor.

    Notes
    -----
    Dropout is applied to the projected output, not through the ``dropout_p`` of
    ``scaled_dot_product_attention``, which aborts with a HIP "invalid argument"
    error on the ROCm build used on Frontier.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor):
        B, L, _d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _output(self, attended: torch.Tensor, shape) -> torch.Tensor:
        return self.attn_drop(self.proj(attended.transpose(1, 2).reshape(shape)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self._project(x)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self._output(attended, x.shape)

    def forward_with_history(self, history: torch.Tensor, query: torch.Tensor):
        """Causal attention of a history sequence, plus a query sequence on it.

        Position ``i`` of ``query`` attends to the history positions before ``i``
        and to itself, never to history position ``i``: it predicts frame ``i``
        from strictly earlier frames, as a rollout does.

        Parameters
        ----------
        history, query : torch.Tensor of shape (B, L, d)
            The two streams, aligned position by position.

        Returns
        -------
        history_out, query_out : torch.Tensor of shape (B, L, d)
            Attention outputs of both streams.
        """
        history_q, history_k, history_v = self._project(history)
        query_q, query_k, query_v = self._project(query)
        history_out = self._output(
            F.scaled_dot_product_attention(
                history_q, history_k, history_v, is_causal=self.causal
            ),
            history.shape,
        )
        query_out = self._output(
            _attend_earlier_and_self(query_q, history_k, history_v, query_k, query_v),
            query.shape,
        )
        return history_out, query_out


class _FFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class FactorizedSTBlock(nn.Module):
    """Spatial attention, then causal temporal attention, then FFN.

    Parameters
    ----------
    cfg : DynamicsConfig
        Model configuration.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.sn = nn.LayerNorm(d)
        self.spatial = _MHA(d, h, causal=False, dropout=cfg.dropout)
        self.tn = nn.LayerNorm(d)
        self.temporal = _MHA(d, h, causal=True, dropout=cfg.dropout)
        self.fn = nn.LayerNorm(d)
        self.ffn = _FFN(d, cfg.ffn_mult, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` of shape (B, F, N, d)."""
        B, Fr, N, d = x.shape
        per_frame = x.reshape(B * Fr, N, d)
        per_frame = per_frame + self.spatial(self.sn(per_frame))
        x = per_frame.reshape(B, Fr, N, d)
        per_token = x.permute(0, 2, 1, 3).reshape(B * N, Fr, d)
        per_token = per_token + self.temporal(self.tn(per_token))
        x = per_token.reshape(B, N, Fr, d).permute(0, 2, 1, 3)
        # The FFN runs on a 2-D view: a 4-D Linear input breaks its backward under
        # DDP with non-reentrant checkpoint recompute.
        flat = x.reshape(B * Fr * N, d)
        flat = flat + self.ffn(self.fn(flat))
        return flat.reshape(B, Fr, N, d)

    def forward_keep_temporal_kv(self, x: torch.Tensor):
        """:meth:`forward`, also returning the temporal attention keys and values.

        Inference only (the history cache, :class:`HistoryCache`). The output equals
        :meth:`forward` up to floating-point rounding.

        Returns
        -------
        x : torch.Tensor of shape (B, F, N, d)
        keys, values : torch.Tensor of shape (B * N, heads, F, head_dim)
        """
        B, Fr, N, d = x.shape
        per_frame = x.reshape(B * Fr, N, d)
        per_frame = per_frame + self.spatial(self.sn(per_frame))
        x = per_frame.reshape(B, Fr, N, d)
        per_token = x.permute(0, 2, 1, 3).reshape(B * N, Fr, d)
        q, k, v = self.temporal._project(self.tn(per_token))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        per_token = per_token + self.temporal._output(attended, per_token.shape)
        x = per_token.reshape(B, N, Fr, d).permute(0, 2, 1, 3)
        flat = x.reshape(B * Fr * N, d)
        flat = flat + self.ffn(self.fn(flat))
        return flat.reshape(B, Fr, N, d), k, v

    def forward_next_frame(self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        """Apply the block to one new frame, given the cached history.

        Spatial attention and the FFN act on the frame alone; in temporal attention
        each of its tokens attends to the cached keys and values of the same token
        in every earlier frame and to itself. Because temporal attention is causal,
        that is exactly what :meth:`forward` computes for the last frame.

        Parameters
        ----------
        x : torch.Tensor of shape (B, 1, N, d)
        past_k, past_v : torch.Tensor of shape (B * N, heads, P, head_dim)

        Returns
        -------
        x : torch.Tensor of shape (B, 1, N, d)
        keys, values : torch.Tensor of shape (B * N, heads, 1, head_dim)
            This frame's temporal keys and values, for appending to the cache.
        """
        B, Fr, N, d = x.shape
        per_frame = x.reshape(B, N, d)
        per_frame = per_frame + self.spatial(self.sn(per_frame))
        per_token = per_frame.reshape(B * N, 1, d)
        q, k, v = self.temporal._project(self.tn(per_token))
        attended = F.scaled_dot_product_attention(
            q, torch.cat([past_k, k], dim=-2), torch.cat([past_v, v], dim=-2)
        )
        per_token = per_token + self.temporal._output(attended, per_token.shape)
        flat = per_token.reshape(B * N, d)
        flat = flat + self.ffn(self.fn(flat))
        return flat.reshape(B, 1, N, d), k, v

    def forward_two_stream(self, history: torch.Tensor, query: torch.Tensor):
        """Apply the block to a history and a query stream of shape (B, F, N, d).

        The streams share every weight. Spatial attention and the FFN act on each
        stream on its own; in temporal attention the query frame ``t`` attends to
        history frames before ``t`` and to itself (see
        :meth:`_MHA.forward_with_history`).
        """
        B, Fr, N, d = history.shape
        both = torch.cat([history, query], dim=0)
        per_frame = both.reshape(2 * B * Fr, N, d)
        per_frame = per_frame + self.spatial(self.sn(per_frame))
        both = per_frame.reshape(2 * B, Fr, N, d)
        per_token = both.permute(0, 2, 1, 3).reshape(2 * B * N, Fr, d)
        normed = self.tn(per_token)
        history_out, query_out = self.temporal.forward_with_history(
            normed[: B * N], normed[B * N :]
        )
        per_token = per_token + torch.cat([history_out, query_out], dim=0)
        both = per_token.reshape(2 * B, N, Fr, d).permute(0, 2, 1, 3)
        flat = both.reshape(2 * B * Fr * N, d)
        flat = flat + self.ffn(self.fn(flat))
        both = flat.reshape(2 * B, Fr, N, d)
        return both[:B], both[B:]


class ActuatorCrossAttention(nn.Module):
    """Per-token cross-attention from frame tokens to individual actuator channels.

    Each channel is its own key/value token (value embedding plus channel
    identity), so a token can select the channels relevant to it rather than
    receiving one pooled vector. The output projection is zero-initialised, so the
    module is an exact no-op at initialisation.

    Parameters
    ----------
    cfg : DynamicsConfig
        Model configuration; uses ``d_model``, ``n_heads`` and ``actuator_dim``.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        d, self.n_act = cfg.d_model, cfg.actuator_dim
        self.n_head = max(1, cfg.n_heads // 2)
        self.chan_val = nn.Linear(1, d)
        self.chan_id = nn.Parameter(torch.randn(cfg.actuator_dim, d) * 0.02)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)
        self.qn = nn.LayerNorm(d)
        self.kn = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, actuators: torch.Tensor) -> torch.Tensor:
        """Attend from ``x`` (B, F, N, d) to ``actuators`` (B, F, n_act).

        Attention stays within a frame, so causality is untouched.
        """
        B, Fr, N, d = x.shape
        channels = self.chan_val(actuators.unsqueeze(-1)) + self.chan_id.view(
            1, 1, self.n_act, d
        )
        channels = self.kn(channels)
        k, v = self.kv(channels).chunk(2, dim=-1)
        q = self.q(self.qn(x))
        n_heads = self.n_head
        q = q.reshape(B * Fr, N, n_heads, d // n_heads).transpose(1, 2)
        k = k.reshape(B * Fr, self.n_act, n_heads, d // n_heads).transpose(1, 2)
        v = v.reshape(B * Fr, self.n_act, n_heads, d // n_heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.o(attended.transpose(1, 2).reshape(B, Fr, N, d))


class HistoryCache:
    """Temporal keys and values of the committed frames, per block (inference only).

    Built by :meth:`DynamicsBackbone.start_history_cache`, extended by one frame at a
    time with :meth:`DynamicsBackbone.extend_history_cache`. Storage is allocated
    once for ``max_frames``; at d1024 it is ``2 * depth * B * tokens * max_frames *
    d_model`` floats (16 GB for 16 blocks, one sample, 100 frames in float32).
    """

    def __init__(self, keys, values, length: int):
        self.keys = keys
        self.values = values
        self.length = length

    def past(self, block: int):
        """Keys and values of block ``block`` over the committed frames."""
        n = self.length
        return self.keys[block][:, :, :n], self.values[block][:, :, :n]


class DynamicsBackbone(nn.Module):
    """Map codes and actuators to per-modality head outputs.

    Parameters
    ----------
    cfg : DynamicsConfig
        Model configuration.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = FrameTokenizer(cfg)
        self.act_embed = nn.Linear(cfg.actuator_dim, cfg.d_model)
        self.act_cross = ActuatorCrossAttention(cfg) if cfg.act_cross_attn else None
        self.blocks = nn.ModuleList([FactorizedSTBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def encode(
        self,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        act_null: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return hidden states of shape (B, F, tokens_per_frame, d_model).

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Input codes.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Actuators of each frame.
        act_null : torch.BoolTensor of shape (B,), default=None
            Samples run unconditionally: no actuator term at all (see :meth:`_embed`).
        """
        x = self._embed(codes, actuators, act_null)
        recompute = self.training and self.cfg.grad_checkpointing and x.requires_grad
        for block in self.blocks:
            x = self._checkpointed(block, recompute, x)
        return self.out_norm(x)

    def encode_block_forcing(
        self,
        history_codes: Dict[str, torch.Tensor],
        query_codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        act_null: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return query-stream hidden states for block teacher forcing.

        Every query frame is predicted from history frames strictly before it, so
        all frames of the window are one-step forecasts in a single pass, under
        the condition of a rollout.

        Parameters
        ----------
        history_codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Fully visible history frames.
        query_codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            The frames to predict, with masked positions set to ``[MASK]``.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Actuators of each frame, shared by both streams.
        act_null : torch.BoolTensor of shape (B,), default=None
            Samples run unconditionally, in both streams.

        Returns
        -------
        hidden : torch.Tensor of shape (B, F, tokens_per_frame, d_model)
            Query-stream hidden states.
        """
        history = self._embed(history_codes, actuators, act_null)
        query = self._embed(query_codes, actuators, act_null)
        recompute = (
            self.training and self.cfg.grad_checkpointing and query.requires_grad
        )
        for block in self.blocks:
            history, query = self._checkpointed(
                block.forward_two_stream, recompute, history, query
            )
        return self.out_norm(query)

    def _embed(self, codes, actuators, act_null=None, frame_offset: int = 0) -> torch.Tensor:
        """Token embeddings plus the actuator term.

        ``act_null`` (B,) drops the whole actuator term of those samples, the
        unconditional branch of classifier-free guidance. That is distinct from
        actuators at their normalised mean, which still add the projection bias, so
        no new parameter is needed and older checkpoints load unchanged.
        """
        x = self.tok.embed(codes, frame_offset=frame_offset)
        B, Fr, N, d = x.shape
        if actuators.shape != (B, Fr, self.cfg.actuator_dim):
            raise ValueError(
                f"actuators expected {(B, Fr, self.cfg.actuator_dim)}; "
                f"got {tuple(actuators.shape)}"
            )
        keep = None if act_null is None else (~act_null).to(x.dtype).view(B, 1, 1, 1)
        conditioning = self.act_embed(actuators).unsqueeze(2)
        if self.act_cross is not None:
            conditioning = conditioning + self.act_cross(x, actuators)
        return x + (conditioning if keep is None else conditioning * keep)

    def supports_history_cache(self) -> bool:
        """Whether :meth:`start_history_cache` applies (no lag embedding)."""
        return int(getattr(self.tok, "lag_k", 0)) == 0

    @torch.no_grad()
    def start_history_cache(
        self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor
    ) -> HistoryCache:
        """Run the committed frames once and keep every block's temporal keys/values.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, P, n_tok)
            Committed frames, as the model should see them (e.g. blind modalities
            already set to ``[MASK]``).
        actuators : torch.Tensor of shape (B, P, actuator_dim)
            Their actuators.
        """
        if not self.supports_history_cache():
            raise ValueError("the history cache does not support lag_embed_k > 0")
        x = self._embed(codes, actuators)
        B, P, N, _d = x.shape
        size = self.cfg.max_frames
        keys, values = [], []
        for block in self.blocks:
            x, k, v = block.forward_keep_temporal_kv(x)
            store_k = k.new_empty(k.shape[0], k.shape[1], size, k.shape[3])
            store_v = torch.empty_like(store_k)
            store_k[:, :, :P] = k
            store_v[:, :, :P] = v
            keys.append(store_k)
            values.append(store_v)
        return HistoryCache(keys, values, P)

    @torch.no_grad()
    def next_frame_logits(
        self,
        cache: HistoryCache,
        frame_codes: Dict[str, torch.Tensor],
        frame_actuators: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Head outputs of one new frame after the cached history.

        Equals the last frame of :meth:`forward` on the history plus this frame, up
        to floating-point rounding, at the cost of one frame instead of all of them.

        Parameters
        ----------
        frame_codes : dict of str to torch.LongTensor of shape (B, 1, n_tok)
        frame_actuators : torch.Tensor of shape (B, 1, actuator_dim)

        Returns
        -------
        logits : dict of str to torch.Tensor of shape (B, 1, n_tok, n_out)
        """
        x, _ = self._cached_frame(cache, frame_codes, frame_actuators, keep=False)
        return self.tok.logits(self.out_norm(x))

    @torch.no_grad()
    def extend_history_cache(
        self,
        cache: HistoryCache,
        frame_codes: Dict[str, torch.Tensor],
        frame_actuators: torch.Tensor,
    ) -> None:
        """Append a committed frame (shapes as :meth:`next_frame_logits`) to ``cache``."""
        if cache.length >= self.cfg.max_frames:
            raise ValueError(f"history cache full at max_frames={self.cfg.max_frames}")
        self._cached_frame(cache, frame_codes, frame_actuators, keep=True)
        cache.length += 1

    def _cached_frame(self, cache, frame_codes, frame_actuators, keep: bool):
        x = self._embed(frame_codes, frame_actuators, frame_offset=cache.length)
        n = cache.length
        for index, block in enumerate(self.blocks):
            past_k, past_v = cache.past(index)
            x, k, v = block.forward_next_frame(x, past_k, past_v)
            if keep:
                cache.keys[index][:, :, n : n + 1] = k
                cache.values[index][:, :, n : n + 1] = v
        return x, None

    def _checkpointed(self, function, recompute: bool, *inputs):
        if not recompute:
            return function(*inputs)
        # The trainer all-reduces gradients by hand: DDP hooks corrupt the
        # recompute. The RNG state must be preserved whenever dropout is on, or the
        # recompute draws a different mask than the forward pass.
        return torch.utils.checkpoint.checkpoint(
            function,
            *inputs,
            use_reentrant=False,
            determinism_check="none",
            preserve_rng_state=float(self.cfg.dropout) > 0.0,
        )

    def forward(
        self,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        act_null: Optional[torch.Tensor] = None,
    ):
        """Return per-modality head outputs for every position."""
        return self.tok.logits(self.encode(codes, actuators, act_null))
