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

from typing import Mapping, Sequence

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