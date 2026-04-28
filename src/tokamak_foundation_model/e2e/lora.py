"""Handrolled LoRA adapters for the shared backbone's attention layers.

Used in Stage 3 (``ResearchPlan.MD`` §4.3) to fine-tune the model for long
autoregressive rollouts without perturbing the Stage 2 weights. The base
``nn.MultiheadAttention`` modules are frozen; only low-rank ``B @ A`` deltas
on the Q/K/V input projection and the output projection are trained.

Zero-initialising ``B`` guarantees that at t=0 the LoRA-wrapped module is
numerically identical to the base module, so loading a Stage 2 checkpoint
into a LoRA-adapted model does not change its predictions.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BackboneBlock, SharedBackbone


class LoRAMultiheadAttention(nn.Module):
    """Drop-in replacement for self-attention with frozen base + rank-``r`` LoRA.

    Wraps an existing ``nn.MultiheadAttention`` (its parameters are frozen on
    construction) and adds a learnable rank-``r`` low-rank delta to both the
    fused Q/K/V input projection weight and the output projection weight.

    Only self-attention is supported; our backbone always calls
    ``self.attn(h, h, h)``. The forward signature mirrors
    ``nn.MultiheadAttention.__call__`` so the wrapper is a literal drop-in
    inside :class:`BackboneBlock`. The returned ``attn_weights`` is always
    ``None`` since ``need_weights=True`` is not used anywhere in the E2E code
    path.

    Parameters
    ----------
    base
        The pretrained ``nn.MultiheadAttention`` whose weights are to be
        frozen. Must have been constructed with ``batch_first=True``.
    rank
        Rank ``r`` of the LoRA delta (typically 4–16). Paper's default is 16.
    alpha
        LoRA scaling factor; the effective delta is ``(alpha / r) · (B @ A)``.
        Follows the convention in Hu et al. (2022). Default ``alpha = r`` →
        scale = 1.0.
    """

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int = 16,
        alpha: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not getattr(base, "batch_first", False):
            raise ValueError(
                "LoRAMultiheadAttention requires base to have batch_first=True"
            )
        self.base = base
        self.embed_dim = base.embed_dim
        self.num_heads = base.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.rank = rank
        self.scale = (alpha if alpha is not None else float(rank)) / rank

        # Freeze base parameters.
        for p in self.base.parameters():
            p.requires_grad = False

        # Match the base module's device so wrapping a GPU-resident MHA
        # produces a GPU-resident wrapper. Default tensor creation is on
        # CPU, which would break when Stage 3 calls apply_lora_to_backbone
        # after model.to(device).
        device = self.base.in_proj_weight.device
        dtype = self.base.in_proj_weight.dtype

        # LoRA deltas:
        #   - input-projection delta for Q, K, V independently, each (d, d)
        #     parameterised as B @ A with A: (r, d), B: (d, r). Stack the
        #     three ``(B, A)`` pairs along a leading dim for a single bmm.
        #   - output-projection delta (d, d), parameterised the same way.
        self.lora_A_qkv = nn.Parameter(
            torch.empty(3, rank, self.embed_dim, device=device, dtype=dtype)
        )
        self.lora_B_qkv = nn.Parameter(
            torch.zeros(3, self.embed_dim, rank, device=device, dtype=dtype)
        )
        self.lora_A_out = nn.Parameter(
            torch.empty(rank, self.embed_dim, device=device, dtype=dtype)
        )
        self.lora_B_out = nn.Parameter(
            torch.zeros(self.embed_dim, rank, device=device, dtype=dtype)
        )
        # Initialise ``A`` with Kaiming uniform (the LoRA-paper default);
        # ``B`` is zero so the initial delta is exactly zero → wrapper
        # matches base at construction.
        nn.init.kaiming_uniform_(self.lora_A_qkv, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))

    def _delta_in_proj(self) -> torch.Tensor:
        """Compute the (3·d, d) delta for the fused Q/K/V input projection."""
        delta = torch.bmm(self.lora_B_qkv, self.lora_A_qkv)  # (3, d, d)
        delta = delta * self.scale
        return delta.reshape(3 * self.embed_dim, self.embed_dim)

    def _delta_out_proj(self) -> torch.Tensor:
        return (self.lora_B_out @ self.lora_A_out) * self.scale

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """Self-attention forward pass with LoRA-perturbed projections.

        Expects ``query is key is value`` (self-attention). Input shape is
        ``(B, N, d)``; returns ``(attn_output, None)`` — the ``None`` mirrors
        ``nn.MultiheadAttention``'s second return when weights are discarded.
        """
        if key is None:
            key = query
        if value is None:
            value = query
        if not (query is key and query is value):
            raise NotImplementedError(
                "LoRAMultiheadAttention only supports self-attention"
            )

        h = query
        batch, n_tokens, _ = h.shape

        in_weight = self.base.in_proj_weight + self._delta_in_proj()
        qkv = F.linear(h, in_weight, self.base.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, N, d) → (B, H, N, head_dim)
        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).reshape(batch, n_tokens, self.embed_dim)

        out_weight = self.base.out_proj.weight + self._delta_out_proj()
        out = F.linear(attn, out_weight, self.base.out_proj.bias)
        return out, None


def apply_lora_to_backbone(
    backbone: SharedBackbone,
    rank: int = 16,
    alpha: Optional[float] = None,
) -> SharedBackbone:
    """In-place wrap every ``BackboneBlock``'s ``.attn`` with :class:`LoRAMultiheadAttention`.

    After this call:
      - Every base attention parameter has ``requires_grad = False``.
      - The new LoRA parameters (``lora_A_{qkv,out}``, ``lora_B_{qkv,out}``)
        have ``requires_grad = True``.
      - MLPs, LayerNorms, step-conditioning MLP, and tokenizer/head weights
        are *not* modified. Freeze them separately if you only want LoRA
        to train.

    Returns the same ``backbone`` for chaining convenience.
    """
    for block in backbone.blocks:
        assert isinstance(block, BackboneBlock)
        # Intentional duck-typed drop-in; LoRA wrapper matches the subset of
        # nn.MultiheadAttention's forward signature that BackboneBlock uses.
        block.attn = LoRAMultiheadAttention(  # type: ignore[assignment]
            block.attn, rank=rank, alpha=alpha
        )
    return backbone


def freeze_non_lora_parameters(module: nn.Module) -> None:
    """Set ``requires_grad = False`` on every parameter whose name does not
    start with ``lora_``.

    Stage 3 freezes everything outside the LoRA adapters (backbone MLPs,
    LayerNorms, step conditioning, tokenizers, output heads).
    """
    for name, param in module.named_parameters():
        if ".lora_" in name or name.startswith("lora_"):
            param.requires_grad = True
        else:
            param.requires_grad = False