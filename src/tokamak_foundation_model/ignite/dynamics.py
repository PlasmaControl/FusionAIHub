"""Factorized spatio-temporal transformer backbone over the frozen codes.

One block is spatial self-attention within a frame, causal temporal self-attention
across frames at each token position, then an FFN, all pre-norm with residuals.
Actuators enter additively per frame, so with the causal temporal mask frame ``t``
is conditioned only on actuators up to ``t``. See ``docs/IGNITE_DESIGN.md``
sections 5.2 and 5.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .dynamics_config import DynamicsConfig
from .frame_layout import FrameTokenizer


def _attend_earlier_and_self(q, history_k, history_v, self_k, self_v):
    """Attention of position ``i`` over history positions ``< i`` and its own key.

    Computed explicitly rather than with a masked ``scaled_dot_product_attention``,
    whose backward fails with "HIP error: invalid argument" on Frontier's ROCm
    build. The own key enters as one extra column, so the scores are
    ``(L, L + 1)`` instead of ``(L, 2L)``; the softmax runs in fp32.

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

    def prefill(self, x: torch.Tensor):
        """Apply :meth:`forward` and also return the temporal keys and values.

        Returns
        -------
        x : torch.Tensor of shape (B, F, N, d)
        k, v : torch.Tensor of shape (B * N, heads, F, head_dim)
        """
        B, Fr, N, d = x.shape
        per_frame = x.reshape(B * Fr, N, d)
        per_frame = per_frame + self.spatial(self.sn(per_frame))
        per_token = per_frame.reshape(B, Fr, N, d).permute(0, 2, 1, 3)
        per_token = per_token.reshape(B * N, Fr, d)
        q, k, v = self.temporal._project(self.tn(per_token))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        per_token = per_token + self.temporal._output(attended, per_token.shape)
        flat = per_token.reshape(B, N, Fr, d).permute(0, 2, 1, 3).reshape(B * Fr * N, d)
        return (flat + self.ffn(self.fn(flat))).reshape(B, Fr, N, d), k, v

    def step(
        self, x: torch.Tensor, k_buf: torch.Tensor, v_buf: torch.Tensor, n: int
    ) -> torch.Tensor:
        """Apply the block to one new frame ``x`` of shape (B, N, d) at index ``n``.

        Its keys and values go to slot ``n`` of the buffers, and it attends over
        slots ``[0, n]``.
        """
        B, N, d = x.shape
        x = x + self.spatial(self.sn(x))
        per_token = x.reshape(B * N, 1, d)
        q, k, v = self.temporal._project(self.tn(per_token))
        k_buf[:, :, n : n + 1], v_buf[:, :, n : n + 1] = k, v
        attended = F.scaled_dot_product_attention(
            q, k_buf[:, :, : n + 1], v_buf[:, :, : n + 1]
        )
        flat = (per_token + self.temporal._output(attended, per_token.shape)).reshape(
            B * N, d
        )
        return (flat + self.ffn(self.fn(flat))).reshape(B, N, d)


@dataclass
class KVCache:
    """Every block's temporal keys and values for the frames committed so far.

    Incremental decoding with it is exact: spatial attention stays inside a frame,
    temporal attention is causal, and actuators and text are added per frame, so a
    committed frame's keys and values never change when a later frame arrives.

    Attributes
    ----------
    k, v : list of torch.Tensor of shape (B * N, heads, max_frames, head_dim)
        Each block's buffers; slots ``[0, n)`` hold the committed frames.
    n : int
        Number of committed frames.
    """

    k: List[torch.Tensor]
    v: List[torch.Tensor]
    n: int


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
        if cfg.text_embed_dim > 0:
            # No bias: a zero text input, drop_text and text_dropout_p == 1 then coincide
            # exactly, so missing text is indistinguishable from the trained null.
            self.text_embed = nn.Linear(cfg.text_embed_dim, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([FactorizedSTBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def encode(
        self,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        drop_actuators: bool = False,
        text: Optional[torch.Tensor] = None,
        drop_text: bool = False,
    ) -> torch.Tensor:
        """Return hidden states of shape (B, F, tokens_per_frame, d_model).

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Input codes.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Actuators of each frame.
        drop_actuators : bool, default=False
            Zero the actuator conditioning of the whole batch: the unconditional
            branch of classifier-free guidance.
        text : torch.Tensor of shape (B, text_embed_dim), optional
            Per-shot text embedding, required when ``cfg.text_embed_dim > 0``.
        drop_text : bool, default=False
            Zero the text conditioning of the whole batch.
        """
        x = self._embed(
            codes,
            actuators,
            drop_actuators=drop_actuators,
            text=text,
            drop_text=drop_text,
        )
        recompute = self.training and self.cfg.grad_checkpointing and x.requires_grad
        for block in self.blocks:
            x = self._checkpointed(block, recompute, x)
        return self.out_norm(x)

    def encode_block_forcing(
        self,
        history_codes: Dict[str, torch.Tensor],
        query_codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        text: Optional[torch.Tensor] = None,
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
        text : torch.Tensor of shape (B, text_embed_dim), optional
            Per-shot text embedding, shared by both streams.

        Returns
        -------
        hidden : torch.Tensor of shape (B, F, tokens_per_frame, d_model)
            Query-stream hidden states.
        """
        # One conditioning-dropout draw for both streams: they are the same samples.
        keep = self._conditioning_keep(actuators.shape[0], actuators.device)
        history = self._embed(history_codes, actuators, text=text, keep=keep)
        query = self._embed(query_codes, actuators, text=text, keep=keep)
        recompute = (
            self.training and self.cfg.grad_checkpointing and query.requires_grad
        )
        for block in self.blocks:
            history, query = self._checkpointed(
                block.forward_two_stream, recompute, history, query
            )
        return self.out_norm(query)

    def _conditioning_keep(
        self,
        batch: int,
        device: torch.device,
        drop_actuators: bool = False,
        drop_text: bool = False,
    ):
        """Draw the per-sample actuator and text dropout masks of a training batch.

        A whole trajectory is conditional or unconditional, as guidance is applied
        at inference. Nothing is drawn when the dropout is off or in eval mode.

        Returns
        -------
        actuator_keep : torch.BoolTensor of shape (B, 1, 1) or None
        text_keep : torch.BoolTensor of shape (B, 1) or None
        """
        actuator_keep = text_keep = None
        if self.training and not drop_actuators and self.cfg.actuator_dropout_p > 0.0:
            actuator_keep = (
                torch.rand((batch, 1, 1), device=device) >= self.cfg.actuator_dropout_p
            )
        if (
            self.training
            and not drop_text
            and self.cfg.text_embed_dim > 0
            and self.cfg.text_dropout_p > 0.0
        ):
            text_keep = torch.rand((batch, 1), device=device) >= self.cfg.text_dropout_p
        return actuator_keep, text_keep

    def _embed(
        self,
        codes,
        actuators,
        frame_offset: int = 0,
        drop_actuators: bool = False,
        text: Optional[torch.Tensor] = None,
        drop_text: bool = False,
        keep=None,
    ) -> torch.Tensor:
        x = self.tok.embed(codes, frame_offset=frame_offset)
        B, Fr, N, d = x.shape
        if actuators.shape != (B, Fr, self.cfg.actuator_dim):
            raise ValueError(
                f"actuators expected {(B, Fr, self.cfg.actuator_dim)}; "
                f"got {tuple(actuators.shape)}"
            )
        if keep is None:
            keep = self._conditioning_keep(B, x.device, drop_actuators, drop_text)
        actuator_keep, text_keep = keep
        # Additive and causal: frame f gets actuator f, and temporal attention is causal.
        actuator_embedding = self.act_embed(actuators)
        if drop_actuators:
            actuator_embedding = torch.zeros_like(actuator_embedding)
        elif actuator_keep is not None:
            actuator_embedding = actuator_embedding * actuator_keep.to(
                actuator_embedding.dtype
            )
        x = x + actuator_embedding.unsqueeze(2)
        if self.act_cross is not None and not drop_actuators:
            # A second actuator pathway, so it obeys the same conditioning drop:
            # otherwise the unconditional branch would still see the actuators.
            cross = self.act_cross(x, actuators)
            if actuator_keep is not None:
                cross = cross * actuator_keep.to(cross.dtype).unsqueeze(-1)
            x = x + cross
        if self.cfg.text_embed_dim > 0:
            if text is None:
                raise ValueError("cfg.text_embed_dim > 0 but no text embedding passed")
            if text.shape != (B, self.cfg.text_embed_dim):
                raise ValueError(
                    f"text expected {(B, self.cfg.text_embed_dim)}; "
                    f"got {tuple(text.shape)}"
                )
            text_embedding = self.text_embed(text)
            if drop_text:
                text_embedding = torch.zeros_like(text_embedding)
            elif text_keep is not None:
                text_embedding = text_embedding * text_keep.to(text_embedding.dtype)
            x = x + text_embedding.view(B, 1, 1, d)
        elif text is not None:
            raise ValueError("text passed but cfg.text_embed_dim == 0")
        return x

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
        text: Optional[torch.Tensor] = None,
    ):
        """Return per-modality head outputs for every position."""
        return self.tok.logits(self.encode(codes, actuators, text=text))

    def prefill(
        self,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        max_frames: int,
        text: Optional[torch.Tensor] = None,
    ) -> KVCache:
        """Run the committed frames once and keep every block's temporal keys and values.

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Committed frames.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Their actuators.
        max_frames : int
            Capacity of the cache buffers.
        text : torch.Tensor of shape (B, text_embed_dim), optional
            Per-shot text embedding.
        """
        if self.tok.lag_k:
            raise ValueError("the KV cache does not support lag_embed_k > 0")
        x = self._embed(codes, actuators, text=text)
        n = x.shape[1]
        keys, values = [], []
        for block in self.blocks:
            x, k, v = block.prefill(x)
            pad = (*k.shape[:2], max_frames - n, k.shape[-1])
            keys.append(torch.cat([k, k.new_empty(pad)], dim=2))
            values.append(torch.cat([v, v.new_empty(pad)], dim=2))
        return KVCache(keys, values, n)

    def step(
        self,
        cache: KVCache,
        codes: Dict[str, torch.Tensor],
        actuators: torch.Tensor,
        text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Hidden states (B, tokens_per_frame, d_model) of frame ``cache.n``.

        ``codes`` (``{name: (B, n_tok)}``) and ``actuators`` (B, actuator_dim) are that
        frame's. Its keys and values are written to slot ``cache.n``; the caller
        commits the frame by running this once more with the final codes and then
        advancing ``cache.n``.
        """
        x = self._embed(
            {name: c.unsqueeze(1) for name, c in codes.items()},
            actuators.unsqueeze(1),
            frame_offset=cache.n,
            text=text,
        )[:, 0]
        for block, k_buf, v_buf in zip(self.blocks, cache.k, cache.v):
            x = block.step(x, k_buf, v_buf, cache.n)
        return self.out_norm(x)
