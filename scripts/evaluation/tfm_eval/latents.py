"""Trainer-faithful K=1 forward that also exposes pooled backbone latents.

``model.forward(return_tokens=True)`` only returns *diagnostic* token
slices, and ``eval_e2e.forward_one_batch`` deviates from training in two
ways (no spectro input truncation, no spectro ``_valid`` passthrough). This
module instead mirrors ``train_e2e_stage1.forward_batch`` exactly and routes
``tokenize → backbone → decode`` manually, so the full (B, 1178, d_model)
backbone output — including actuator token slices — is available for
pooling.

Pooled latent keys: one per ``token_layout`` entry (12 diagnostics + 9
actuators, mean over that modality's tokens) plus ``diag_global`` /
``act_global`` (mean over the diagnostic / actuator prefix). Each is
(B, d_model) fp32; drivers downcast to fp16 at save time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_e2e import (  # noqa: E402
    _clean_and_mask,
    _spectro_loss_gate,
    _spectro_trunc_t,
    _video_loss_gate,
    _video_standardize_per_bc,
)

POOLED_GLOBAL_KEYS = ("diag_global", "act_global")


@torch.no_grad()
def forward_with_latents(
    model: Any,
    batch: Dict,
    device: torch.device,
) -> Dict[str, Any]:
    """One K=1 forward; returns predictions, targets, masks, pooled latents.

    Returns a dict with keys:
      ``predictions`` — {name: tensor}, video already (B, C, T, H, W)
      ``targets``     — {name: tensor}, video z-scored with *input* stats,
                        spectro truncated to trunc_t (matches head output)
      ``masks``       — {name: tensor|None}, gates/masks as in training loss
      ``diag_inputs`` — cleaned inputs actually fed to the model (the copy
                        baseline in the same normalized space as targets)
      ``pooled``      — {layout_name|diag_global|act_global: (B, d_model)}
    """
    diag_inputs: Dict[str, torch.Tensor] = {}
    norm_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for cfg in model.diagnostics:
        raw = batch["inputs"][cfg.name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if cfg.kind == "video":
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            norm_stats[cfg.name] = (mu, sd)
        elif cfg.kind == "spectrogram":
            cleaned = cleaned[..., : _spectro_trunc_t(cfg)]
        diag_inputs[cfg.name] = cleaned
        if cfg.kind in ("video", "spectrogram"):
            valid_key = f"{cfg.name}_valid"
            if valid_key in batch["inputs"]:
                diag_inputs[valid_key] = batch["inputs"][valid_key].to(
                    device, non_blocking=True
                )

    act_inputs: Dict[str, torch.Tensor] = {}
    for cfg in model.actuators:
        raw = batch["targets"][cfg.name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        act_inputs[cfg.name] = cleaned

    batch_size = next(iter(act_inputs.values())).shape[0]
    step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    time_offset = torch.zeros(batch_size, device=device)

    tokens = model.tokenize(diag_inputs, act_inputs)
    out_tokens = model.backbone(tokens, step_idx, time_offset)
    predictions = model.decode(out_tokens)
    for cfg in model.diagnostics:
        if cfg.kind == "video":
            predictions[cfg.name] = predictions[cfg.name].permute(0, 2, 1, 3, 4)

    pooled: Dict[str, torch.Tensor] = {}
    for layout in model.token_layout:
        pooled[layout.name] = out_tokens[:, layout.slice_].mean(dim=1)
    pooled["diag_global"] = out_tokens[:, : model.n_diag_tokens].mean(dim=1)
    pooled["act_global"] = out_tokens[:, model.n_diag_tokens :].mean(dim=1)

    targets: Dict[str, torch.Tensor] = {}
    masks: Dict[str, Optional[torch.Tensor]] = {}
    for cfg in model.diagnostics:
        tgt = batch["targets"][cfg.name].to(device, non_blocking=True).float()
        if cfg.kind == "video":
            mu, sd = norm_stats[cfg.name]
            targets[cfg.name] = (tgt - mu) / sd
            masks[cfg.name] = _video_loss_gate(cfg, batch, device)
        elif cfg.kind == "spectrogram":
            targets[cfg.name] = tgt[..., : _spectro_trunc_t(cfg)]
            masks[cfg.name] = _spectro_loss_gate(cfg.name, batch, device)
        else:
            targets[cfg.name] = tgt
            mask_key = f"{cfg.name}_mask"
            masks[cfg.name] = (
                batch["targets"][mask_key].to(device, non_blocking=True).float()
                if mask_key in batch["targets"]
                else None
            )

    return {
        "predictions": predictions,
        "targets": targets,
        "masks": masks,
        "diag_inputs": diag_inputs,
        "pooled": pooled,
    }


@torch.no_grad()
def copy_prediction(
    cfg: Any, result: Dict[str, Any]
) -> torch.Tensor:
    """Persistence baseline in the same normalized space as ``targets``.

    The cleaned input window *is* the copy prediction: video is already
    z-scored with the shared input stats, spectro already truncated — so
    shapes and normalization line up with ``result['targets']`` untouched.
    """
    x = result["diag_inputs"][cfg.name]
    tgt = result["targets"][cfg.name]
    if x.shape != tgt.shape:
        raise ValueError(
            f"{cfg.name}: copy pred {tuple(x.shape)} vs target "
            f"{tuple(tgt.shape)} — window grid mismatch"
        )
    return x
