"""Shared eval helpers used by Phase 1/2/3 scripts (Stage 1 + Stage 2).

This module is helpers-only — there is no ``main()`` here. The eval
pipeline runs through driver scripts that all import from this file:

  * ``eval_e2e_phase1.py``           — metrics + PASS/FAIL gates
  * ``eval_e2e_phase2_per_shot.py``  — per-shot trajectory plots
  * ``eval_e2e_phase2_plots.py``     — aggregate-scatter plots
  * ``eval_e2e_phase3_stitched.py``  — stitched-segment plots
  * ``eval_e2e_phase3_1_video.py``   — video grid + mp4

What lives here:
  - Input cleaning / video standardisation / mask helpers.
  - ``forward_one_batch``           — single-step (K=1) forward.
  - ``rollout_forward_one_batch``   — unified K-step forward; K=1 falls
                                       through to the fast model.forward
                                       path, K>1 uses TokenSpaceRollout.
  - ``detect_stage_K``              — Stage 1 (K=1) vs Stage 2 (K_max)
                                       autodetection from ckpt['args'].
  - Per-step split helpers for slow_ts / fast_ts / actuator, video,
    spectrogram targets.
  - ``copy_baseline_for_modality``  — persistence baseline.

The standalone single-step ``main()`` that used to live here was
superseded by ``eval_e2e_phase1.py`` and removed when the pipeline was
unified across Stage 1 and Stage 2.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from tokamak_foundation_model.e2e.checkpoint import load_state_dict_explicit
from tokamak_foundation_model.e2e.model import (
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout

logger = logging.getLogger("eval_e2e")


def load_checkpoint_with_refine_tolerance(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> None:
    """Load checkpoint into model, allowing the model's spectro / fast_ts
    refine-MLP stacks to be deeper than the checkpoint's.

    2026-05-19: SpectrogramTokenizer/Head bumped 4 → 12 refine blocks and
    FastTimeSeriesTokenizer/Head bumped 2 → 4. Eval scripts must tolerate
    the extra refine.<i> indices being absent from older checkpoints.
    """
    allowed_missing: List[str] = []
    for d_cfg in model.diagnostics:
        # spec / fast_ts: refine MLP stack length grew over time;
        # let older checkpoints miss the extra refine.<i> indices.
        if d_cfg.kind in ("spectrogram", "fast_ts"):
            for mod_path in (
                f"diag_tokenizers.{d_cfg.name}",
                f"diag_heads.{d_cfg.name}",
            ):
                try:
                    mod = model.get_submodule(mod_path)
                except AttributeError:
                    continue
                if not hasattr(mod, "refine"):
                    continue
                n_model = len(mod.refine)
                prefix = f"{mod_path}.refine."
                ckpt_indices = set()
                for k in state_dict:
                    if k.startswith(prefix):
                        head, _, _ = k[len(prefix):].partition(".")
                        if head.isdigit():
                            ckpt_indices.add(int(head))
                n_ckpt = (max(ckpt_indices) + 1) if ckpt_indices else 0
                for i in range(n_ckpt, n_model):
                    allowed_missing.append(f"{mod_path}.refine.{i}.")
        # video + spectrogram: VideoOutputHead and SpectrogramOutputHead
        # both gained a zero-init `refine_block` residual for the
        # patch-grid checkerboard fix — permit it as missing when
        # loading pre-patch checkpoints.
        if d_cfg.kind in ("video", "spectrogram"):
            mod_path = f"diag_heads.{d_cfg.name}"
            try:
                mod = model.get_submodule(mod_path)
            except AttributeError:
                continue
            if hasattr(mod, "refine_block"):
                prefix = f"{mod_path}.refine_block."
                if not any(k.startswith(prefix) for k in state_dict):
                    allowed_missing.append(prefix)
            # Spec heads may carry the 2026-06-12 inv_stem branch
            # (zero-init residual) — permit it as missing when the
            # checkpoint predates it.
            for sub in ("inv_stem", "inv_stem_unembed"):
                if hasattr(mod, sub):
                    prefix = f"{mod_path}.{sub}."
                    if not any(k.startswith(prefix) for k in state_dict):
                        allowed_missing.append(prefix)
        # Spec TOKENIZERS may carry the 2026-06-13 freq stem (zero-init
        # residual full-freq mixing) — permit fs_lin* as missing.
        if d_cfg.kind == "spectrogram":
            tok_path = f"diag_tokenizers.{d_cfg.name}"
            try:
                tok = model.get_submodule(tok_path)
            except AttributeError:
                tok = None
            if tok is not None and getattr(tok, "enable_freq_stem", False):
                prefix = f"{tok_path}.fs_lin"
                if not any(k.startswith(prefix) for k in state_dict):
                    allowed_missing.append(prefix)
    load_state_dict_explicit(
        model, state_dict, allowed_missing_prefixes=tuple(allowed_missing)
    )


# ── Helpers (inlined from train_e2e_stage1.py for stability) ─────────


def _clean_and_mask(
    tensor: torch.Tensor, existing_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(tensor)
    cleaned = torch.where(finite, tensor, torch.zeros_like(tensor))
    mask = finite.float()
    if existing_mask is not None:
        mask = mask * existing_mask
    return cleaned, mask


def _video_standardize_per_bc(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
    return (x - mu) / sd, mu, sd


def _video_loss_gate(
    cfg: DiagnosticConfig, batch: Dict, device: torch.device
) -> torch.Tensor:
    name = cfg.name
    chan_mask = batch["targets"][f"{name}_channel_mask"].to(
        device, non_blocking=True
    ).float()
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()
    return (
        valid[:, None, None, None, None]
        * chan_mask[:, :, None, None, None]
    )


def _ts_mask(
    cfg: DiagnosticConfig, batch: Dict, device: torch.device
) -> Optional[torch.Tensor]:
    mask_key = f"{cfg.name}_mask"
    if mask_key in batch["targets"]:
        return (
            batch["targets"][mask_key]
            .to(device, non_blocking=True)
            .float()
        )
    return None


# ── K-step rollout helpers (unify Stage 1 K=1 and Stage 2 K>1 paths) ──
# Constants + split helpers are inlined from train_e2e_stage2_delta.py
# so the eval pipeline can serve both stages without importing from
# the trainer (which is a script, not a library).

SLOW_FS = 100.0
FAST_FS = 10_000.0

_SLOW_TS_NAMES = {
    "ts_core_density", "ts_core_temp",
    "ts_tangential_density", "ts_tangential_temp",
    "cer_ti", "cer_rot", "mse",
}
_FAST_TS_NAMES = {"filterscopes"}
_ACTUATOR_NAMES = {
    "pin", "beam_voltage", "tin", "ech_power", "ech_tor_angle", "ech_pol_angle",
    "ech_polarization", "gas_flow", "gas_raw", "rmp",
}

SAMPLE_RATES_HZ: Dict[str, float] = {
    **{n: SLOW_FS for n in _SLOW_TS_NAMES},
    **{n: FAST_FS for n in _FAST_TS_NAMES},
    **{n: FAST_FS for n in _ACTUATOR_NAMES},
}


def samples_per_step(name: str, chunk_duration_s: float) -> int:
    return round(chunk_duration_s * SAMPLE_RATES_HZ[name])


def split_target_by_step(
    tensor: torch.Tensor, name: str, k_steps: int, chunk_duration_s: float
) -> List[torch.Tensor]:
    """Split a (B, C, K*per_step) slow_ts / fast_ts / actuator target."""
    per = samples_per_step(name, chunk_duration_s)
    return [
        tensor[..., k * per : (k + 1) * per].contiguous() for k in range(k_steps)
    ]


def split_video_target_by_step(
    target: torch.Tensor, k_steps: int, n_per_step: int,
) -> List[torch.Tensor]:
    """Split (B, C, K*n_per_step, H, W) video target into K windows."""
    return [
        target[:, :, k * n_per_step : (k + 1) * n_per_step].contiguous()
        for k in range(k_steps)
    ]


def split_spectro_target_by_step(
    target: torch.Tensor, k_steps: int, trunc_t: int,
) -> List[torch.Tensor]:
    """Split (B, C, F, K*trunc_t) spectrogram target into K windows.

    ``trunc_t`` must match ``SpectrogramTokenizer.trunc_t`` — i.e.
    ``(cfg.window_samples // T_p) * T_p``. Trailing frames past
    ``K * trunc_t`` (typically <2%) are dropped to match the head output.
    """
    return [
        target[:, :, :, k * trunc_t : (k + 1) * trunc_t].contiguous()
        for k in range(k_steps)
    ]


def _spectro_loss_gate(
    name: str, batch: Dict, device: torch.device
) -> torch.Tensor:
    """Per-batch (B, 1, 1, 1) loss gate from ``<name>_valid``."""
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()
    return valid[:, None, None, None]


def _spectro_trunc_t(cfg: DiagnosticConfig) -> int:
    """Match ``SpectrogramTokenizer.trunc_t`` per cfg.window_samples / T_p."""
    assert cfg.kind == "spectrogram" and cfg.spectrogram_patch_size is not None
    _, T_p = cfg.spectrogram_patch_size
    return (cfg.window_samples // T_p) * T_p


def detect_stage_K(ckpt: Dict) -> int:
    """Return the natural K for this checkpoint: 1 for Stage 1, K_max for
    Stage 2 (delta-rollout). Stage 2 checkpoints carry ``K_max`` in
    ``ckpt['args']``; Stage 1 checkpoints don't.
    """
    args = ckpt.get("args", {}) or {}
    K = args.get("K_max", 1)
    return int(K) if K else 1


def make_rollout_if_needed(
    model: E2EFoundationModel, K: int, chunk_duration_s: float,
) -> Optional[TokenSpaceRollout]:
    """Build a TokenSpaceRollout for K>1, else None (K=1 uses model.forward)."""
    if K <= 1:
        return None
    return TokenSpaceRollout(model, dt_s=chunk_duration_s)


@torch.no_grad()
def rollout_forward_one_batch(
    model: E2EFoundationModel,
    rollout: Optional[TokenSpaceRollout],
    batch: Dict,
    device: torch.device,
    K: int,
    chunk_duration_s: float,
) -> Tuple[
    List[Dict[str, torch.Tensor]],            # predictions_per_k (length K)
    Dict[str, torch.Tensor],                   # diag_initial (step-0 inputs)
    List[Dict[str, torch.Tensor]],            # targets_per_k (length K)
    List[Dict[str, Optional[torch.Tensor]]],  # masks_per_k (length K)
]:
    """Unified K-step forward for Stage 1 (K=1) and Stage 2 (K>1).

    For K=1, ``rollout`` may be None: takes the fast model.forward()
    path, matching the byte-exact behaviour of ``forward_one_batch`` on
    Stage 1 checkpoints. For K>1, uses TokenSpaceRollout with per-step
    target/mask splitting (slow_ts/fast_ts/actuator via sample count;
    video by frame count; spectrogram by trunc_t).
    """
    video_diags = [c.name for c in model.diagnostics if c.kind == "video"]
    spectro_diags = [c.name for c in model.diagnostics if c.kind == "spectrogram"]
    cfg_by_name = {c.name: c for c in model.diagnostics}
    act_names = [c.name for c in model.actuators]

    # Step-0 diagnostic inputs (with video standardization stats stashed
    # so the targets can use the same per-(B, C) z-score).
    video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    diag_initial: Dict[str, torch.Tensor] = {}
    for cfg in model.diagnostics:
        name = cfg.name
        raw = batch["inputs"][name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if cfg.kind == "video":
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            video_stats[name] = (mu, sd)
        diag_initial[name] = cleaned
        if cfg.kind in ("video", "spectrogram"):
            valid_key = f"{name}_valid"
            if valid_key in batch["inputs"]:
                diag_initial[valid_key] = batch["inputs"][valid_key].to(
                    device, non_blocking=True
                )

    # Build full-horizon target + gate tensors for video / spectro.
    video_target_full: Dict[str, torch.Tensor] = {}
    video_gate: Dict[str, torch.Tensor] = {}
    spectro_target_full: Dict[str, torch.Tensor] = {}
    spectro_gate: Dict[str, torch.Tensor] = {}
    spectro_trunc: Dict[str, int] = {}
    for name in video_diags:
        raw = batch["targets"][name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        mu, sd = video_stats[name]
        video_target_full[name] = (cleaned - mu) / sd
        video_gate[name] = _video_loss_gate(cfg_by_name[name], batch, device)
    for name in spectro_diags:
        raw = batch["targets"][name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        spectro_target_full[name] = cleaned
        spectro_gate[name] = _spectro_loss_gate(name, batch, device)
        spectro_trunc[name] = _spectro_trunc_t(cfg_by_name[name])

    # Per-step act, target, mask dicts (length K).
    act_per_step: List[Dict[str, torch.Tensor]] = []
    target_per_step: List[Dict[str, torch.Tensor]] = []
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
    for k in range(K):
        act_k: Dict[str, torch.Tensor] = {}
        for name in act_names:
            raw = batch["targets"][name].to(device, non_blocking=True).float()
            slc = split_target_by_step(raw, name, K, chunk_duration_s)[k]
            cleaned, _ = _clean_and_mask(slc, None)
            act_k[name] = cleaned
        act_per_step.append(act_k)

        tgt_k: Dict[str, torch.Tensor] = {}
        mk_k: Dict[str, Optional[torch.Tensor]] = {}
        for cfg in model.diagnostics:
            name = cfg.name
            if cfg.kind == "video":
                n_per = video_target_full[name].shape[2] // K
                tgt_k[name] = split_video_target_by_step(
                    video_target_full[name], K, n_per
                )[k]
                mk_k[name] = video_gate[name]
            elif cfg.kind == "spectrogram":
                tgt_k[name] = split_spectro_target_by_step(
                    spectro_target_full[name], K, spectro_trunc[name]
                )[k]
                mk_k[name] = spectro_gate[name]
            else:
                raw = batch["targets"][name].to(device, non_blocking=True).float()
                tgt_k[name] = split_target_by_step(raw, name, K, chunk_duration_s)[k]
                mask_key = f"{name}_mask"
                if mask_key in batch["targets"]:
                    raw_mask = batch["targets"][mask_key].to(
                        device, non_blocking=True
                    ).float()
                    mk_k[name] = split_target_by_step(
                        raw_mask, name, K, chunk_duration_s
                    )[k]
                else:
                    mk_k[name] = None
        target_per_step.append(tgt_k)
        mask_per_step.append(mk_k)

    # Forward.
    if rollout is not None and K > 1:
        result = rollout(diag_initial, act_per_step, collect_history=False)
        predictions_per_k = result.predictions
    else:
        batch_size = next(iter(diag_initial.values())).shape[0]
        step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        time_offset = torch.zeros(batch_size, device=device)
        predictions_per_k = [
            model(diag_initial, act_per_step[0], step_idx, time_offset)
        ]

    # Video predictions come out (B, T, C, H, W); flip to (B, C, T, H, W)
    # so downstream consumers see one shape contract.
    for k in range(len(predictions_per_k)):
        for name in video_diags:
            if name in predictions_per_k[k]:
                predictions_per_k[k][name] = (
                    predictions_per_k[k][name].permute(0, 2, 1, 3, 4)
                )

    return predictions_per_k, diag_initial, target_per_step, mask_per_step


@torch.no_grad()
def forward_one_batch(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],  # predictions (post permute for video)
    Dict[str, torch.Tensor],  # diag_inputs (cleaned, video standardized)
    Dict[str, torch.Tensor],  # targets (raw or standardized for video)
    Dict[str, Optional[torch.Tensor]],  # masks
]:
    """Single forward pass mirroring trainer.forward_batch behaviour."""
    diag_inputs: Dict[str, torch.Tensor] = {}
    video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for cfg in model.diagnostics:
        raw = batch["inputs"][cfg.name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if cfg.kind == "video":
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            video_stats[cfg.name] = (mu, sd)
        diag_inputs[cfg.name] = cleaned
        if cfg.kind == "video":
            valid_key = f"{cfg.name}_valid"
            if valid_key in batch["inputs"]:
                diag_inputs[valid_key] = (
                    batch["inputs"][valid_key].to(device, non_blocking=True)
                )

    act_inputs: Dict[str, torch.Tensor] = {}
    for cfg in model.actuators:
        raw = batch["targets"][cfg.name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        act_inputs[cfg.name] = cleaned

    batch_size = next(iter(diag_inputs.values())).shape[0]
    step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    time_offset = torch.zeros(batch_size, device=device)
    predictions = model(diag_inputs, act_inputs, step_idx, time_offset)

    for cfg in model.diagnostics:
        if cfg.kind == "video":
            predictions[cfg.name] = predictions[cfg.name].permute(0, 2, 1, 3, 4)

    targets: Dict[str, torch.Tensor] = {}
    masks: Dict[str, Optional[torch.Tensor]] = {}
    for cfg in model.diagnostics:
        targets[cfg.name] = (
            batch["targets"][cfg.name].to(device, non_blocking=True).float()
        )
        if cfg.kind == "video":
            mu, sd = video_stats[cfg.name]
            targets[cfg.name] = (targets[cfg.name] - mu) / sd
            masks[cfg.name] = _video_loss_gate(cfg, batch, device)
        else:
            masks[cfg.name] = _ts_mask(cfg, batch, device)
    return predictions, diag_inputs, targets, masks


@torch.no_grad()
def copy_baseline_for_modality(
    cfg: DiagnosticConfig,
    batch: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(copy_pred, target, mask)`` for one diagnostic modality.

    ``copy_pred`` is the input echoed into the target shape; for video the
    same per-(B, C) z-score is applied as in training so the number lives in
    the same normalised space as the model's prediction.
    """
    name = cfg.name
    pred = batch["inputs"][name].to(device, non_blocking=True).float()
    target = batch["targets"][name].to(device, non_blocking=True).float()
    if cfg.kind == "video":
        pred, mu, sd = _video_standardize_per_bc(pred)
        target = (target - mu) / sd
        mask = _video_loss_gate(cfg, batch, device)
    else:
        mask = _ts_mask(cfg, batch, device)
    return pred, target, mask
