"""Explicit checkpoint loading for the E2E foundation model.

Replaces the default ``model.load_state_dict(state, strict=True)`` call
in the trainers with a structured key check that:

* **Always raises on unexpected keys** — silently dropping them would
  mask renamed / removed TS keys, the exact regression Phase C edits
  could introduce.
* **Allows missing keys whose names start with one of
  ``allowed_missing_prefixes``** — e.g. when loading a TS-only Phase A
  checkpoint into a TS+video model, the freshly-initialised
  ``diag_tokenizers.tangtv.*`` and ``diag_heads.tangtv.*`` keys are
  expected to be missing from the saved state.
* **Otherwise raises on missing keys** — partial loads should be
  explicit, not the default.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence, Tuple

import torch
import torch.nn as nn


def load_state_dict_explicit(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    allowed_missing_prefixes: Sequence[str] = (),
) -> None:
    """Load ``state_dict`` into ``model`` with explicit key checks.

    Parameters
    ----------
    model : nn.Module
        Target model. Must already have its final architecture (e.g.
        already include video modules if a TS+video state is loaded).
    state_dict : mapping
        Dict of ``name -> Tensor`` to load.
    allowed_missing_prefixes : sequence of str
        If non-empty, missing keys are allowed only when their name
        starts with one of these prefixes. Use this to permit fresh
        init of new modules that didn't exist in the saved state.

    Raises
    ------
    RuntimeError
        If the state contains any unexpected keys, or if any missing
        key falls outside ``allowed_missing_prefixes``.
    """
    result = model.load_state_dict(state_dict, strict=False)

    if result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in checkpoint (state contains keys the "
            f"model does not have): {result.unexpected_keys}"
        )

    disallowed_missing = [
        k
        for k in result.missing_keys
        if not any(k.startswith(p) for p in allowed_missing_prefixes)
    ]
    if disallowed_missing:
        raise RuntimeError(
            "Missing keys in checkpoint not covered by "
            f"allowed_missing_prefixes={tuple(allowed_missing_prefixes)}: "
            f"{disallowed_missing}"
        )


_BACKBONE_BLOCK_RE = re.compile(r"^backbone\.blocks\.(\d+)\.")


def warm_start_extend_backbone(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    target_n_layers: int,
) -> Tuple[int, Tuple[str, ...]]:
    """Near-identity init for extra backbone layers added on top of a
    shallower checkpoint.

    Inspects ``state_dict`` for ``backbone.blocks.<i>.*`` keys to infer
    the checkpoint's depth ``old_n_layers``. If ``old_n_layers <
    target_n_layers``, mutates ``model`` in place: for each new block
    index ``i in [old_n_layers, target_n_layers)`` it zeros the
    attention output projection and the MLP's final linear so the new
    block contributes zero to its residual stream at init — making the
    deeper model produce the same outputs as the shallow source until
    training wakes the new layers up.

    Returns ``(old_n_layers, allowed_missing_prefixes)`` where the
    prefixes are the ``backbone.blocks.<i>.`` strings the caller must
    add to ``allowed_missing_prefixes`` of
    :func:`load_state_dict_explicit`. When the checkpoint already
    matches or exceeds the target depth, returns
    ``(target_n_layers, ())`` and no init is performed.

    Assumes the BackboneBlock layout from
    ``e2e/backbone.py`` (``norm1`` → ``attn`` → ``norm2`` →
    ``mlp = Sequential(Linear, GELU, Dropout, Linear, Dropout)``).
    """
    indices = {
        int(m.group(1))
        for k in state_dict
        for m in (_BACKBONE_BLOCK_RE.match(k),)
        if m is not None
    }
    if not indices:
        # No backbone in this checkpoint at all — nothing to extend.
        return target_n_layers, ()
    old_n_layers = max(indices) + 1
    if old_n_layers >= target_n_layers:
        return old_n_layers, ()

    backbone = model.backbone
    for i in range(old_n_layers, target_n_layers):
        block = backbone.blocks[i]
        # Zero attention output projection → attn_out contributes 0 to
        # the residual at init.
        nn.init.zeros_(block.attn.out_proj.weight)
        if block.attn.out_proj.bias is not None:
            nn.init.zeros_(block.attn.out_proj.bias)
        # MLP final linear (index 3 in the Sequential) → MLP contributes
        # 0 to the residual at init.
        nn.init.zeros_(block.mlp[3].weight)
        if block.mlp[3].bias is not None:
            nn.init.zeros_(block.mlp[3].bias)
        # norm1 / norm2 keep PyTorch default (gamma=1, beta=0) — identity.

    allowed_prefixes = tuple(
        f"backbone.blocks.{i}." for i in range(old_n_layers, target_n_layers)
    )
    return old_n_layers, allowed_prefixes