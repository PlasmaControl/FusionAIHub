"""Checkpoint loading for e2e evaluation.

Single source of truth for rebuilding an :class:`E2EFoundationModel` from a
training checkpoint — replaces the loader block previously copy-pasted across
``eval_e2e_stage1.py`` / ``eval_e2e_stage2.py`` / ``debug_*`` scripts.

The full architecture spec lives *inside* the ``.pt`` (``diagnostics`` /
``actuators`` config dicts + ``args``); never rebuild from ``build_configs()``
— the registry has drifted since training (e.g. tangtv channel count).
"""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional, Tuple

import torch

from tokamak_foundation_model.e2e.lora import apply_lora_to_backbone
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

DEFAULT_CKPT = (
    "/lustre/orion/proj-shared/fus187/models/e2e_stage1_d1024_48L/"
    "e2e_stage1_best.pt"
)

# (model kwarg, ckpt["args"] key, default) — mirrors the construction block in
# scripts/training/train_e2e_stage1.py:1365-1388, with the trainer's argparse
# defaults. ``.get`` keeps older checkpoints (fewer args keys) loadable.
_MODEL_ARG_MAP = [
    ("backbone_grad_checkpoint", "backbone_grad_checkpoint", False),
    ("video_seam_refine", "video_seam_refine", False),
    ("spectro_seam_refine", "spectro_seam_refine", False),
    ("seam_refine_hidden_ch", "seam_refine_hidden_ch", 16),
    ("spectro_refine_kernel", "spectro_refine_kernel", 3),
    ("video_refine_kernel", "video_refine_kernel", (1, 3, 3)),
    ("spectro_inv_stem", "spec_inv_stem", False),
    ("spectro_inv_stem_ch", "spec_inv_stem_ch", 64),
    ("spectro_freq_stem", "spec_freq_stem", False),
    ("spectro_freq_stem_hidden", "spec_freq_stem_hidden", 128),
    ("video_resize_conv", "video_resize_conv", False),
    ("video_resize_conv_hidden", "video_resize_conv_hidden", 64),
    ("spectro_generative", "spec_generative", False),
    ("spectro_flow_base_ch", "spec_flow_base_ch", 64),
    ("spectro_flow_sample_steps", "spec_flow_steps", 6),
    ("spectro_flow_lambda", "spec_flow_lambda", 1.0),
]


def load_ckpt(path: str = DEFAULT_CKPT, drop_optimizer: bool = True) -> Dict[str, Any]:
    """Load a training checkpoint CPU-side.

    ``mmap=True`` keeps the 16 GB file from being read into RSS up front;
    optimizer/scheduler state (2/3 of the file) is dropped immediately unless
    a resume actually needs it.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if drop_optimizer:
        ckpt.pop("optimizer_state_dict", None)
        ckpt.pop("scheduler_state_dict", None)
        gc.collect()
    return ckpt


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    dropout: float = 0.0,
    device: Optional[str] = None,
) -> Tuple[E2EFoundationModel, list, list]:
    """Rebuild the model from checkpoint config and strictly load weights.

    Returns ``(model.eval(), diagnostics, actuators)``. ``attn_impl`` is
    pinned to ``"standard"`` — SDPA/flash variants use different parameter
    names and can never load a standard-attention checkpoint.
    """
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    args = ckpt.get("args", {}) or {}

    kwargs: Dict[str, Any] = {}
    for model_key, args_key, default in _MODEL_ARG_MAP:
        val = args.get(args_key, default)
        if model_key == "video_refine_kernel" and val is not None:
            val = tuple(val)
        kwargs[model_key] = val

    model = E2EFoundationModel(
        diagnostics=diagnostics,
        actuators=actuators,
        d_model=args.get("d_model", 256),
        n_heads=args.get("n_heads", 8),
        n_layers=args.get("n_layers", 8),
        dropout=dropout,
        attn_impl="standard",
        **kwargs,
    )

    state_dict = ckpt["model_state_dict"]
    if any(".lora_" in k for k in state_dict):
        apply_lora_to_backbone(
            model.backbone,
            rank=args.get("lora_rank", 16),
            alpha=args.get("lora_alpha", 16.0),
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if device is not None:
        model = model.to(device)
    return model, diagnostics, actuators
