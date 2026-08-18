"""Phase-B factorized ST-transformer backbone over the frozen codes (fresh model code).

One block = spatial self-attention (within a frame, over all 1012 multi-modal tokens — cross-
modal + within-modal mixing of one 50 ms state) → causal temporal self-attention (across frames,
per token position; frame *t* attends only to ≤ *t*) → FFN, all pre-norm with residuals.
``depth`` blocks stacked (a first-class knob for the depth study). Actuator conditioning is
**additive and causal**: continuous 70-ch ``actuator_t`` → linear embedding → added to frame
*t*'s tokens; combined with the causal temporal mask, frame *t* is conditioned only on actuators
≤ *t* (docs/IGNITE_DESIGN.md §5.2, §5.4).

Self-contained attention (torch SDPA) rather than an external transformer stack — keeps the
causality wiring explicit and testable (the controllability claim rests on it), and satisfies the
fresh-model-code rule.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from .dynamics_config import DynamicsConfig
from .frame_layout import FrameTokenizer


class _MHA(nn.Module):
    """Multi-head self-attention over the last-but-one axis of a (B, L, d) tensor."""

    def __init__(self, d_model: int, n_heads: int, causal: bool, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.causal = causal
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d)
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, h, L, dh)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(o.transpose(1, 2).reshape(B, L, d))


class _FFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class FactorizedSTBlock(nn.Module):
    """Spatial (within-frame, full) → temporal (across-frame, causal) → FFN. Pre-norm residuals."""

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.n_heads
        self.sn = nn.LayerNorm(d)
        self.spatial = _MHA(d, h, causal=False, dropout=cfg.dropout)
        self.tn = nn.LayerNorm(d)
        self.temporal = _MHA(d, h, causal=True, dropout=cfg.dropout)
        self.fn = nn.LayerNorm(d)
        self.ffn = _FFN(d, cfg.ffn_mult, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F, N, d)
        B, Fr, N, d = x.shape
        # spatial: attend within each frame over the N=1012 tokens
        xs = x.reshape(B * Fr, N, d)
        xs = xs + self.spatial(self.sn(xs))
        x = xs.reshape(B, Fr, N, d)
        # temporal: attend across frames per token position, causal
        xt = x.permute(0, 2, 1, 3).reshape(B * N, Fr, d)
        xt = xt + self.temporal(self.tn(xt))
        x = xt.reshape(B, N, Fr, d).permute(0, 2, 1, 3)
        # FFN — flatten to 2D (B*F*N, d) so the Linear backward always takes the standard 2D path.
        # On the 4D (B,F,N,d) tensor the Linear grad-weight path could hit `t()` on a 4D input under
        # DDP + non-reentrant checkpoint recompute ("t() expects <=2D, self is 4D"); 2D avoids it.
        xf = x.reshape(B * Fr * N, d)
        xf = xf + self.ffn(self.fn(xf))
        x = xf.reshape(B, Fr, N, d)
        return x


class DynamicsBackbone(nn.Module):
    """codes {name:(B,F,n_tok)} + actuators (B,F,70) → per-modality logits {name:(B,F,n_tok,vocab)}."""

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = FrameTokenizer(cfg)
        self.act_embed = nn.Linear(cfg.actuator_dim, cfg.d_model)
        self.blocks = nn.ModuleList([FactorizedSTBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def encode(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
               drop_actuators: bool = False) -> torch.Tensor:
        """→ hidden states (B, F, tokens_per_frame, d_model).

        ``drop_actuators`` zeroes the actuator contribution for the whole batch — the
        unconditional branch used by classifier-free guidance at inference.
        """
        x = self.tok.embed(codes)                              # (B, F, N, d)
        B, Fr, N, d = x.shape
        if actuators.shape != (B, Fr, self.cfg.actuator_dim):
            raise ValueError(
                f"actuators expected {(B, Fr, self.cfg.actuator_dim)}; got {tuple(actuators.shape)}"
            )
        # additive, causal (actuator_f added to frame f; temporal attn is causal)
        a = self.act_embed(actuators)                          # (B, F, d)
        if drop_actuators:
            a = torch.zeros_like(a)
        else:
            p = getattr(self.cfg, "actuator_dropout_p", 0.0)
            if self.training and p > 0.0:
                # per-SAMPLE dropout: a whole trajectory is conditional or unconditional,
                # matching how guidance is applied at inference.
                keep = (torch.rand((B, 1, 1), device=a.device) >= p).to(a.dtype)
                a = a * keep
        x = x + a.unsqueeze(2)                                 # (B, F, 1, d) broadcast over tokens
        use_ckpt = self.training and getattr(self.cfg, "grad_checkpointing", False) and x.requires_grad
        for blk in self.blocks:
            if use_ckpt:
                # Non-reentrant checkpoint, run BARE (no DDP wrapper — the trainer syncs grads with a
                # manual all-reduce). DDP's autograd hooks are what corrupt the checkpoint recompute
                # (t()-on-4D / addmm / lost grads on out-of-checkpoint params); bare + manual sync
                # avoids that entirely. determinism_check off: SDPA saves different valid tensors on
                # recompute. preserve_rng_state off: blocks are RNG-free (dropout=0).
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False, determinism_check="none",
                    preserve_rng_state=False)
            else:
                x = blk(x)
        return self.out_norm(x)

    def forward(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor):
        return self.tok.logits(self.encode(codes, actuators))
