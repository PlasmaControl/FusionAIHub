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

from typing import Dict, Optional

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
        # OUTPUT dropout, deliberately NOT SDPA's dropout_p. Passing dropout_p > 0 to
        # scaled_dot_product_attention aborts on this ROCm build with
        # "torch.AcceleratorError: HIP error: invalid argument" -- the flash kernel rejects it
        # (killed job 5322227, 2026-08-21, all 64 ranks in the first 40 s). Dropping the mask on
        # the projected output regularises the same path, keeps the fast attention kernel, and
        # adds NO parameters, so existing checkpoints load unchanged.
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, d)
        B, L, d = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, h, L, dh)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)   # dropout_p=0: see __init__
        return self.attn_drop(self.proj(o.transpose(1, 2).reshape(B, L, d)))


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


class ActuatorCrossAttention(nn.Module):
    """Per-token cross-attention from frame tokens (queries) to the actuator channels (keys/values).

    WHY THIS REPLACES `x + act_embed(actuators)`. MEASURED 2026-08-18 on the best model, 24 held-out
    shots: swapping in a DIFFERENT shot's control program moves the forecast by only 0.263 of its
    own spread; zeroing the actuators entirely, 0.240. Three quarters of the forecast is
    independent of the actuators -- for a model whose purpose is "forecast diagnostics from
    actuator trajectories" that is the whole task missing.

    The old pathway was 70 channels -> ONE Linear -> ONE vector added identically to all ~1280
    token embeddings of the frame. No gating, no per-token selectivity, no per-modality
    modulation: the control signal competes additively against 1280 learned embeddings and is
    swamped. Here each token ATTENDS over the 70 actuator channels individually, so a co2 band
    token can weight `pinj` while a magnetics token weights `i_coil`.

    Channels are embedded individually (value * per-channel embedding + per-channel identity) so
    the key set is 70 tokens, not one pooled vector -- otherwise attention has nothing to select
    between and degenerates to the additive path it replaces.

    Zero-init on the output projection makes this an EXACT no-op at initialisation, so a run
    starts byte-identical to the additive baseline and the cross-attention has to earn its
    contribution. cfg.act_cross_attn = False restores the old path entirely.
    """

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        d, self.n_act = cfg.d_model, cfg.actuator_dim
        self.n_head = max(1, cfg.n_heads // 2)
        self.chan_val = nn.Linear(1, d)                     # scalar channel value -> d
        self.chan_id = nn.Parameter(torch.randn(cfg.actuator_dim, d) * 0.02)
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.o.weight)                       # exact no-op at init
        self.qn = nn.LayerNorm(d)
        self.kn = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, actuators: torch.Tensor) -> torch.Tensor:
        """x: (B, F, N, d) frame tokens. actuators: (B, F, n_act) -> same shape as x."""
        B, Fr, N, d = x.shape
        # per-CHANNEL key/value tokens: (B, F, n_act, d)
        a = self.chan_val(actuators.unsqueeze(-1)) + self.chan_id.view(1, 1, self.n_act, d)
        a = self.kn(a)
        k, v = self.kv(a).chunk(2, dim=-1)
        q = self.q(self.qn(x))
        h = self.n_head
        # fold (B,F) into the batch axis: attention is WITHIN a frame, so no information crosses
        # frames here and causality is untouched (the temporal blocks still own that).
        q = q.reshape(B * Fr, N, h, d // h).transpose(1, 2)
        k = k.reshape(B * Fr, self.n_act, h, d // h).transpose(1, 2)
        v = v.reshape(B * Fr, self.n_act, h, d // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B * Fr, N, d).reshape(B, Fr, N, d)
        return self.o(o)


class DynamicsBackbone(nn.Module):
    """codes {name:(B,F,n_tok)} + actuators (B,F,70) → per-modality logits {name:(B,F,n_tok,vocab)}."""

    def __init__(self, cfg: DynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = FrameTokenizer(cfg)
        self.act_embed = nn.Linear(cfg.actuator_dim, cfg.d_model)
        self.act_cross = (ActuatorCrossAttention(cfg)
                          if getattr(cfg, 'act_cross_attn', False) else None)
        if getattr(cfg, "text_embed_dim", 0) > 0:
            # bias=False: makes zeros-text input, drop_text, and text_dropout_p==1.0 coincide
            # exactly at t=0 — missing text is indistinguishable from the trained null.
            self.text_embed = nn.Linear(cfg.text_embed_dim, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([FactorizedSTBlock(cfg) for _ in range(cfg.depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def encode(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
               drop_actuators: bool = False, text: Optional[torch.Tensor] = None,
               drop_text: bool = False) -> torch.Tensor:
        """→ hidden states (B, F, tokens_per_frame, d_model).

        ``drop_actuators`` zeroes the actuator contribution for the whole batch — the
        unconditional branch used by classifier-free guidance at inference.

        ``text`` is a per-shot embedding (B, text_embed_dim), added to every token of every
        frame (time-invariant conditioning) when ``cfg.text_embed_dim > 0``. ``drop_text``
        zeroes it (the CFG unconditional branch).
        """
        x = self.tok.embed(codes)                              # (B, F, N, d)
        B, Fr, N, d = x.shape
        if actuators.shape != (B, Fr, self.cfg.actuator_dim):
            raise ValueError(
                f"actuators expected {(B, Fr, self.cfg.actuator_dim)}; got {tuple(actuators.shape)}"
            )
        # additive, causal (actuator_f added to frame f; temporal attn is causal)
        a = self.act_embed(actuators)                          # (B, F, d)
        keep = None
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
        if self.act_cross is not None and not drop_actuators:
            # per-token attention over the actuator channels. It is a SECOND actuator pathway,
            # so it obeys the same conditioning drop as the additive one -- otherwise the
            # "unconditional" CFG branch would still see the control program.
            c = self.act_cross(x, actuators)
            if keep is not None:
                c = c * keep.unsqueeze(-1)
            x = x + c
        tdim = getattr(self.cfg, "text_embed_dim", 0)
        if tdim > 0:
            if text is None:
                raise ValueError("cfg.text_embed_dim > 0 but no text embedding passed")
            if text.shape != (B, tdim):
                raise ValueError(f"text expected {(B, tdim)}; got {tuple(text.shape)}")
            t = self.text_embed(text)                                  # (B, d)
            if drop_text:
                t = torch.zeros_like(t)
            else:
                p = getattr(self.cfg, "text_dropout_p", 0.0)
                if self.training and p > 0.0:
                    keep_t = (torch.rand((B, 1), device=t.device) >= p).to(t.dtype)
                    t = t * keep_t
            x = x + t.view(B, 1, 1, d)                                 # broadcast over frames and tokens
        elif text is not None:
            raise ValueError("text passed but cfg.text_embed_dim == 0")
        use_ckpt = self.training and getattr(self.cfg, "grad_checkpointing", False) and x.requires_grad
        for blk in self.blocks:
            if use_ckpt:
                # Non-reentrant checkpoint, run BARE (no DDP wrapper — the trainer syncs grads with a
                # manual all-reduce). DDP's autograd hooks are what corrupt the checkpoint recompute
                # (t()-on-4D / addmm / lost grads on out-of-checkpoint params); bare + manual sync
                # avoids that entirely. determinism_check off: SDPA saves different valid tensors on
                # recompute. preserve_rng_state: OFF is sound ONLY while the blocks are RNG-free.
                # With dropout > 0 they are NOT -- the recompute would draw a DIFFERENT dropout
                # mask than the forward pass, so gradients would be silently wrong (no error,
                # just a corrupted run). Tie the flag to dropout so enabling regularisation
                # cannot quietly break checkpointing. Verified numerically 2026-08-21.
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, use_reentrant=False, determinism_check="none",
                    preserve_rng_state=float(getattr(self.cfg, "dropout", 0.0)) > 0.0)
            else:
                x = blk(x)
        return self.out_norm(x)

    def forward(self, codes: Dict[str, torch.Tensor], actuators: torch.Tensor,
                text: Optional[torch.Tensor] = None):
        return self.tok.logits(self.encode(codes, actuators, text=text))
