"""Shared Transformer backbone with rollout-step conditioning.

Pre-norm Transformer encoder (LayerNorm → attention → residual, LayerNorm →
MLP → residual), with a Fourier-feature MLP encoding of ``(step_index,
time_offset_s)`` broadcast-added to all tokens before the first block.
See ``ResearchPlan.MD`` §3.4 and §5.6.
"""

import math
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn.modules.mha import MHA as _FlashMHA
except ImportError:
    _FlashMHA = None


def _fourier_features(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Map ``x`` of shape ``(B,)`` to ``(B, 2*n_freq)`` sin/cos features."""
    phase = x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class StepConditioning(nn.Module):
    """Fourier features of ``(step_index, time_offset_s)`` → ``d_model`` MLP.

    ``step_freqs`` cover typical 0–80-step rollouts; ``time_freqs`` cover
    absolute offsets on the ~0–10 s shot timescale. Frequencies are fixed
    buffers; only the 2-layer MLP is learned.
    """

    def __init__(
        self, d_model: int, n_freq: int = 16, hidden: Optional[int] = None
    ) -> None:
        super().__init__()
        if hidden is None:
            hidden = 4 * d_model
        step_freqs = 2 * math.pi * torch.logspace(-3, 0, n_freq)
        time_freqs = 2 * math.pi * torch.logspace(-1, 2, n_freq)
        self.register_buffer("step_freqs", step_freqs)
        self.register_buffer("time_freqs", time_freqs)
        self.mlp = nn.Sequential(
            nn.Linear(4 * n_freq, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        # Default PyTorch init on the output layer gives embed std ≈ 0.1,
        # too weak to visibly condition the token stream at init (cos_sim
        # between step=0 and step=40 stays > 0.98 through 2 blocks). Scale
        # up so step embed has per-element std ≈ 0.5 at init — same order
        # as post-tokenizer tokens — which is the level §5.6 requires.
        nn.init.normal_(self.mlp[-1].weight, std=0.3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, step_index: torch.Tensor, time_offset_s: torch.Tensor
    ) -> torch.Tensor:
        """Return a per-batch conditioning vector of shape ``(B, d_model)``."""
        step_feats = _fourier_features(
            step_index.float(), cast(torch.Tensor, self.step_freqs)
        )
        time_feats = _fourier_features(
            time_offset_s.float(), cast(torch.Tensor, self.time_freqs)
        )
        return self.mlp(torch.cat([step_feats, time_feats], dim=-1))


class FlashSelfAttention(nn.Module):
    """flash_attn MHA wrapped to match nn.MultiheadAttention's self-attn call.

    BackboneBlock calls ``self.attn(h, h, h, need_weights=False)`` and
    unpacks ``attn_out, _``. We mimic that signature; only self-attention
    (q is k is v) is supported. Requires fp16/bf16 inputs at runtime —
    the training script's bf16 autocast satisfies this.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if _FlashMHA is None:
            raise ImportError(
                "flash_attn not installed; build it via "
                "`pixi run -e frontier setup-flash-attn`"
            )
        self.mha = _FlashMHA(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            causal=False,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        del k, v, need_weights
        return self.mha(q), None


class SDPASelfAttention(nn.Module):
    """Self-attention via ``F.scaled_dot_product_attention``.

    Drop-in for ``nn.MultiheadAttention(h, h, h, need_weights=False)`` but
    routes through PyTorch's SDPA, which on ROCm 7.x dispatches to AOTriton
    flash-attention. Empirical wins over ``nn.MultiheadAttention`` on MI250X:
    1.4-5× attention speedup, 2-3× lower attention memory.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model={d_model} must be divisible by n_heads={n_heads}"
        )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Fused QKV projection — single matmul, matches what nn.MultiheadAttention
        # does internally but keeps the weight name distinct so a switch
        # between attn_impls never silently loads a wrong-shaped checkpoint.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        # Self-attention path: BackboneBlock calls self.attn(h, h, h, ...)
        del k, v, need_weights
        B, S, D = q.shape
        # (B, S, 3*D) -> (B, S, 3, H, D_head) -> (3, B, H, S, D_head)
        qkv = self.qkv(q).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q_, k_, v_ = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        # (B, H, S, D_head) -> (B, S, D)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out), None


class BackboneBlock(nn.Module):
    """Pre-norm Transformer encoder block: norm→attn→residual, norm→MLP→residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_impl: str = "standard",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if attn_impl == "flash":
            self.attn = FlashSelfAttention(d_model, n_heads, dropout=dropout)
        elif attn_impl == "sdpa":
            self.attn = SDPASelfAttention(d_model, n_heads, dropout=dropout)
        elif attn_impl == "standard":
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        else:
            raise ValueError(
                f"attn_impl must be 'standard', 'sdpa', or 'flash', got "
                f"{attn_impl!r}"
            )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SharedBackbone(nn.Module):
    """Stack of :class:`BackboneBlock` with step conditioning.

    Parameters
    ----------
    d_model
        Token embedding dimension (``256`` in the full config, smaller for
        tests).
    n_heads
        Number of attention heads.
    n_layers
        Number of stacked blocks (``8`` in the full config).
    mlp_ratio
        MLP hidden-dim ratio (``4.0``).
    dropout
        Dropout applied inside attention and MLP.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_impl: str = "standard",
        gradient_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.gradient_checkpoint = gradient_checkpoint
        self.step_cond = StepConditioning(d_model)
        self.blocks = nn.ModuleList(
            [
                BackboneBlock(d_model, n_heads, mlp_ratio, dropout, attn_impl=attn_impl)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        step_index: torch.Tensor,
        time_offset_s: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Run tokens through the stack.

        Parameters
        ----------
        tokens
            Input of shape ``(batch, n_tokens, d_model)``.
        step_index
            Integer-valued tensor of shape ``(batch,)``.
        time_offset_s
            Float tensor of shape ``(batch,)`` with absolute time in seconds.
        return_intermediates
            If ``True``, return a list of length ``n_layers + 2`` containing
            the post-conditioning input, each block's output, and the
            final-norm output (for §5.6 progressive-mixing tests).
        """
        step_embed = self.step_cond(step_index, time_offset_s).unsqueeze(1)
        x = tokens + step_embed
        if return_intermediates:
            # Intermediates path keeps every block's output anyway, so
            # checkpointing would defeat its purpose — disable here.
            intermediates: List[torch.Tensor] = [x]
            for block in self.blocks:
                x = block(x)
                intermediates.append(x)
            intermediates.append(self.final_norm(x))
            return intermediates
        # Gradient checkpointing recomputes each block's activations during
        # backward instead of storing them. Only active during training
        # (no-op under inference / no_grad) so eval cost is unchanged.
        use_ckpt = self.gradient_checkpoint and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.final_norm(x)