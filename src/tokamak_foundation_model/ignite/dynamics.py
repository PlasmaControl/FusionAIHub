"""Factorized spatio-temporal transformer backbone over the frozen codes.

One block is spatial self-attention within a frame, causal temporal self-attention
across frames at each token position, then an FFN, all pre-norm with residuals.
Actuators enter additively per frame, so with the causal temporal mask frame ``t``
is conditioned only on actuators up to ``t``. See ``docs/IGNITE_DESIGN.md``
sections 5.2 and 5.4.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .dynamics_config import DynamicsConfig
from .frame_layout import FrameTokenizer


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.attn_drop(self.proj(attended.transpose(1, 2).reshape(B, L, d)))


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
        self.blocks = nn.ModuleList([FactorizedSTBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def encode(
        self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor
    ) -> torch.Tensor:
        """Return hidden states of shape (B, F, tokens_per_frame, d_model).

        Parameters
        ----------
        codes : dict of str to torch.LongTensor of shape (B, F, n_tok)
            Input codes.
        actuators : torch.Tensor of shape (B, F, actuator_dim)
            Actuators of each frame.
        """
        x = self.tok.embed(codes)
        B, Fr, N, d = x.shape
        if actuators.shape != (B, Fr, self.cfg.actuator_dim):
            raise ValueError(
                f"actuators expected {(B, Fr, self.cfg.actuator_dim)}; "
                f"got {tuple(actuators.shape)}"
            )
        x = x + self.act_embed(actuators).unsqueeze(2)
        if self.act_cross is not None:
            x = x + self.act_cross(x, actuators)
        recompute = self.training and self.cfg.grad_checkpointing and x.requires_grad
        for block in self.blocks:
            if recompute:
                # The trainer all-reduces gradients by hand: DDP hooks corrupt the
                # recompute. The RNG state must be preserved whenever dropout is on,
                # or the recompute draws a different mask than the forward pass.
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                    determinism_check="none",
                    preserve_rng_state=float(self.cfg.dropout) > 0.0,
                )
            else:
                x = block(x)
        return self.out_norm(x)

    def forward(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor):
        """Return per-modality head outputs for every position."""
        return self.tok.logits(self.encode(codes, actuators))
