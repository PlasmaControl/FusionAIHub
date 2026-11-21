"""Attention kernels for the dynamics backbone: FlashAttention-4 where it runs, else SDPA.

FlashAttention-4 (`flash-attn-4`, the CuTe DSL kernels in `flash_attn.cute`) runs on
compute capability 9.x (Hopper) and 10.x/11.x (Blackwell, Vera Rubin) for fp16/bf16
inputs with head dimension 8-128. Everything else (CPU tests, fp32, ROCm on Frontier,
the A100s on Stellar) goes through ``torch.nn.functional.scaled_dot_product_attention``.

Measured on node13 (Vera Rubin, compute capability 10.7, torch 2.15+cu132,
flash-attn-4 4.0.0b40, 2026-11-21), causal bf16 B=8 H=16 L=4096 D=128, forward and
backward: FA4 1.04 ms, cuDNN SDPA 1.07 ms, SDPA's default choice 2.52 ms, SDPA's
FlashAttention-2 backend 3.9 ms. FA4 compiles each kernel configuration on first use
(about 4.5 s).

``IGNITE_ATTENTION`` selects the path: ``auto`` (default) uses FA4 where it can,
``sdpa`` never uses it, ``fa4`` raises where it cannot be used.

All functions take and return the backbone's layout, (B, heads, L, head_dim).
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import torch.nn.functional as F

#: Environment variable selecting the attention path: auto, sdpa or fa4.
ENV = "IGNITE_ATTENTION"


@lru_cache(maxsize=None)
def _fa4():
    """The FlashAttention-4 entry point, or None when it is not installed."""
    try:
        from flash_attn.cute import flash_attn_func
    except Exception:  # not installed, or its CuTe DSL runtime does not load
        return None
    return flash_attn_func


def fa4_reason(q: torch.Tensor) -> str | None:
    """Why FlashAttention-4 cannot run on ``q``, or None when it can."""
    if not q.is_cuda or torch.version.hip:
        return "not a CUDA tensor"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype} (FA4 takes fp16/bf16)"
    major = torch.cuda.get_device_capability(q.device)[0]
    if major not in (9, 10, 11):
        return f"compute capability {major}.x (FA4 needs 9.x, 10.x or 11.x)"
    head_dim = q.shape[-1]
    if not 8 <= head_dim <= 128 or head_dim % 8:
        return f"head_dim {head_dim} (FA4 takes 8-128, a multiple of 8)"
    if _fa4() is None:
        return "flash-attn-4 is not installed"
    return None


def use_fa4(q: torch.Tensor) -> bool:
    """Whether to run FlashAttention-4 on ``q`` under ``IGNITE_ATTENTION``."""
    mode = os.environ.get(ENV, "auto").strip().lower()
    if mode == "sdpa":
        return False
    if mode not in ("auto", "fa4"):
        raise ValueError(f"{ENV}={mode!r}: expected auto, sdpa or fa4")
    reason = fa4_reason(q)
    if reason is not None and mode == "fa4":
        raise RuntimeError(f"{ENV}=fa4 but FlashAttention-4 cannot run here: {reason}")
    return reason is None


def _fa4_call(q, k, v, causal: bool, return_lse: bool = False):
    """FlashAttention-4 on (B, heads, L, head_dim) tensors, in the same layout."""
    out = _fa4()(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=causal,
        return_lse=return_lse,
    )
    if isinstance(out, tuple):
        out, lse = out[0], out[1]
    else:
        lse = None
    out = out.transpose(1, 2)
    return (out, lse) if return_lse else out


def attention(q, k, v, causal: bool = False):
    """Self- or cross-attention, ``softmax(q k^T / sqrt(d)) v``.

    With ``causal`` the query and key lengths must be equal: SDPA aligns its causal
    mask top-left and FA4 bottom-right, and the two agree only then.

    Parameters
    ----------
    q : torch.Tensor of shape (B, heads, Lq, head_dim)
    k, v : torch.Tensor of shape (B, heads, Lk, head_dim)
    causal : bool, default=False

    Returns
    -------
    attended : torch.Tensor of shape (B, heads, Lq, head_dim)
    """
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError(f"causal attention needs Lq == Lk, got {q.shape[-2]} and {k.shape[-2]}")
    if use_fa4(q):
        return _fa4_call(q, k, v, causal)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def attend_earlier_and_self_fa4(q, history_k, history_v, self_k, self_v):
    """Attention over history positions ``< i`` and the own key, with FlashAttention-4.

    Exact, without a dense mask: position ``i >= 1`` attends causally to history
    ``0 .. i-1`` (queries ``1..L-1`` against keys ``0..L-2``, a plain causal call),
    and that partial softmax is merged with the own-key score through the
    log-sum-exp FA4 returns (its backward carries the gradient of the
    log-sum-exp). Position 0 has no earlier history, so it is its own value.

    Parameters
    ----------
    q, self_k, self_v : torch.Tensor of shape (B, heads, L, head_dim)
    history_k, history_v : torch.Tensor of shape (B, heads, L, head_dim)

    Returns
    -------
    attended : torch.Tensor of shape (B, heads, L, head_dim)
    """
    length = q.shape[-2]
    if length == 1:
        return self_v
    scale = q.shape[-1] ** -0.5
    history_out, history_lse = _fa4_call(
        q[..., 1:, :], history_k[..., :-1, :], history_v[..., :-1, :], causal=True,
        return_lse=True,
    )
    self_score = (q[..., 1:, :].float() * self_k[..., 1:, :].float()).sum(-1) * scale
    peak = torch.maximum(history_lse, self_score)
    history_w = torch.exp(history_lse - peak).unsqueeze(-1)
    self_w = torch.exp(self_score - peak).unsqueeze(-1)
    rest = (history_w * history_out.float() + self_w * self_v[..., 1:, :].float()) / (
        history_w + self_w
    )
    return torch.cat([self_v[..., :1, :], rest.to(q.dtype)], dim=-2)
