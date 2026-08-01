"""Stage 2b: displacement-loss fine-tuning of the E2E foundation model.

Replaces Stage 2's pure masked-MAE objective with a mixed loss that directly
rewards predicting the *displacement* (pred − ctx) in both direction and
magnitude. Motivated by §5.9 test 5 showing Stage 2's best checkpoint moves
predictions *away* from target at mid-rollout (direction_cos negative) — a
diagnostic that MAE alone does not penalise.

Loss (summed over rollout steps and modalities)::

    L_k_m = α · masked_mae(pred, target)
          + β · (1 − cos_sim(pred − ctx, target − ctx))     on samples with
          + γ · |log‖pred − ctx‖ − log‖target − ctx‖|        ‖target − ctx‖ > min_disp_norm

Defaults: α=1.0, β=0.3, γ=0.1, min_disp_norm=0.01.

Context semantics (teacher-forced for scoring displacement):
  - step k=0: ctx = diag_initial (the true state at window 0)
  - step k≥1: ctx = target_{k-1}  (the true state at window k)

The token rollout itself still feeds the model's predicted diag tokens
forward — Stage 2b is a *loss change*, not a data-flow change.

Smoke test::

    pixi run python scripts/training/train_e2e_stage2_delta.py \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --checkpoint_dir /tmp/e2e_stage2_delta_smoke \
        --max_files 4 --max_steps 50 --batch_size 2 --num_workers 0 \
        --K_max 3 --curriculum_steps 30 --val_every 1000 --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt
import yaml
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    DistributedTwoLevelSampler,
    TokamakMultiFileDataset,
    TwoLevelSampler,
    filter_video_present_files,
)
from tokamak_foundation_model.e2e.checkpoint import (
    load_state_dict_explicit,
    warm_start_extend_backbone,
)
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout
from tokamak_foundation_model.e2e.output_heads import SpectrogramFlowHead
from tokamak_foundation_model.utils.distributed import DistributedManager

# Specfix helpers (2026-06-12) — shared with the Stage 1 trainer rather
# than duplicated. Sibling-module import works because this script's own
# directory is on sys.path when launched as `python scripts/training/...`.
from train_e2e_stage1 import (  # noqa: E402
    weighted_masked_mae,
    build_spec_per_bin_weights,
    build_spec_per_bin_sigma,
    _apply_module_freeze as _stage1_apply_module_freeze,
)

from tokamak_foundation_model.e2e.multimodal import (
    SPECTROGRAM_MODALITIES,
    VIDEO_MODALITIES,
    append_multimodal_diagnostics,
    spectro_loss_gate as _spectro_loss_gate,
    spectro_trunc_t as _spectro_trunc_t,
    split_spectro_target_by_step,
    split_video_target_by_step,
    video_loss_gate as _video_loss_gate,
    video_standardize_per_bc as _video_standardize_per_bc,
)

from tokamak_foundation_model.e2e.multimodal import (
    SPECTROGRAM_MODALITIES,
    VIDEO_MODALITIES,
    append_multimodal_diagnostics,
    spectro_loss_gate as _spectro_loss_gate,
    spectro_trunc_t as _spectro_trunc_t,
    split_spectro_target_by_step,
    split_video_target_by_step,
    video_loss_gate as _video_loss_gate,
    video_standardize_per_bc as _video_standardize_per_bc,
)


def _core(module):
    return module.module if hasattr(module, "module") else module

logger = logging.getLogger("e2e_stage2_delta")


# ── Modality inventory (duplicated from Stage 1/2 by design) ─────────────

SLOW_TS_MODALITIES: List[Tuple[str, int]] = [
    ("ts_core_density", 44),
    ("ts_core_temp", 44),
    ("ts_tangential_density", 10),
    ("ts_tangential_temp", 10),
    ("cer_ti", 48),
    ("cer_rot", 48),
    ("mse", 69),
]
FAST_TS_MODALITIES: List[Tuple[str, int, int]] = [("filterscopes", 8, 50)]
ACTUATOR_MODALITIES: List[Tuple[str, int]] = [
    ("pin", 8),
    ("beam_voltage", 8),
    ("tin", 8),
    ("ech_power", 12),
    ("ech_tor_angle", 12),
    ("ech_pol_angle", 12),
    ("ech_polarization", 12),
    ("gas_flow", 11),
    ("gas_raw", 11),
    ("rmp", 12),
]
SLOW_FS = 100.0
FAST_FS = 10_000.0
SAMPLE_RATES_HZ: Dict[str, float] = {
    **{name: SLOW_FS for name, _ in SLOW_TS_MODALITIES},
    **{name: FAST_FS for name, _, _ in FAST_TS_MODALITIES},
    **{name: FAST_FS for name, _ in ACTUATOR_MODALITIES},
}

def build_configs(
    chunk_duration_s: float,
    use_video: Optional[List[str]] = None,
    use_spectro: Optional[List[str]] = None,
    spectro_patch_f: Optional[int] = None,
    spectro_patch_t: Optional[int] = None,
) -> Tuple[List[DiagnosticConfig], List[ActuatorConfig]]:
    slow_samples = round(chunk_duration_s * SLOW_FS)
    fast_samples = round(chunk_duration_s * FAST_FS)
    diagnostics: List[DiagnosticConfig] = [
        DiagnosticConfig(n, "slow_ts", c, slow_samples)
        for n, c in SLOW_TS_MODALITIES
    ] + [
        DiagnosticConfig(n, "fast_ts", c, fast_samples, p)
        for n, c, p in FAST_TS_MODALITIES
    ]
    # Order locked at [slow_ts | fast_ts | spectrogram | video | actuators]
    # so the rollout's diagnostic-prefix slice stays contiguous (Guard G1).
    diagnostics = append_multimodal_diagnostics(
        diagnostics, use_video=use_video, use_spectro=use_spectro,
        spectro_patch_f=spectro_patch_f, spectro_patch_t=spectro_patch_t,
    )
    actuators: List[ActuatorConfig] = [
        ActuatorConfig(n, c, fast_samples, n_tokens=5)
        for n, c in ACTUATOR_MODALITIES
    ]
    return diagnostics, actuators


def _load_shot_yaml(path: Path) -> List[int]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    shots = data.get("shots", []) if isinstance(data, dict) else (data or [])
    return [int(s) for s in shots]


def _shot_to_h5(data_dir: Path, shot: int) -> Path:
    return data_dir / f"{shot}_processed.h5"


def resolve_shot_files(
    data_dir: Path, train_yaml: Optional[Path], val_yaml: Optional[Path],
    max_files: Optional[int], val_fraction: float, seed: int,
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    if train_yaml is not None:
        train_files = [_shot_to_h5(data_dir, s) for s in _load_shot_yaml(train_yaml)]
        train_files = [p for p in train_files if p.exists()]
        if val_yaml is not None:
            val_files = [_shot_to_h5(data_dir, s) for s in _load_shot_yaml(val_yaml)]
            val_files = [p for p in val_files if p.exists()]
        else:
            rng.shuffle(train_files)
            n_val = max(1, int(val_fraction * len(train_files)))
            val_files = train_files[:n_val]
            train_files = train_files[n_val:]
    else:
        all_files = sorted(data_dir.glob("*_processed.h5"))
        rng.shuffle(all_files)
        n_val = max(1, int(val_fraction * len(all_files)))
        val_files = all_files[:n_val]
        train_files = all_files[n_val:]
    if max_files is not None:
        train_files = train_files[:max_files]
        val_files = val_files[: max(1, max_files // 4)]
    return train_files, val_files


# ── Target splitting (time-based, per-modality) ──────────────────────────


def samples_per_step(name: str, chunk_duration_s: float) -> int:
    return round(chunk_duration_s * SAMPLE_RATES_HZ[name])


def split_target_by_step(
    tensor: torch.Tensor, name: str, k_steps: int, chunk_duration_s: float,
) -> List[torch.Tensor]:
    per = samples_per_step(name, chunk_duration_s)
    expected = per * k_steps
    if tensor.shape[-1] < expected:
        raise ValueError(
            f"{name}: target length {tensor.shape[-1]} < expected {expected}"
        )
    return [
        tensor[..., k * per : (k + 1) * per].contiguous()
        for k in range(k_steps)
    ]


def _clean_and_mask(
    tensor: torch.Tensor, existing_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(tensor)
    cleaned = torch.where(finite, tensor, torch.zeros_like(tensor))
    mask = finite.float()
    if existing_mask is not None:
        mask = mask * existing_mask
    return cleaned, mask


def masked_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    cleaned_pred, pm = _clean_and_mask(pred, None)
    cleaned_target, tm = _clean_and_mask(target, mask)
    combined = pm * tm
    diff = (cleaned_pred - cleaned_target).abs() * combined
    return diff.sum() / combined.sum().clamp_min(1.0)


def _video_standardize_per_bc(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(B, C) z-score over (T, H, W). Returns ``(x_norm, mu, sd)``.

    ``sd.clamp(min=1.0)`` keeps off-channels (zero-filled) finite. Same
    convention as train_e2e_stage1.py / standalone video AE.
    """
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
    return (x - mu) / sd, mu, sd


def _video_loss_gate(
    name: str, batch: Dict, device: torch.device,
) -> torch.Tensor:
    """Per-element loss gate combining camera-validity scalar with the
    per-channel availability mask. Shape ``(B, C, 1, 1, 1)`` broadcasts
    cleanly over ``(B, C, T, H, W)``. Per-shot, not per-step."""
    chan = batch["targets"][f"{name}_channel_mask"].to(
        device, non_blocking=True
    ).float()
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()
    return valid[:, None, None, None, None] * chan[:, :, None, None, None]


def split_video_target_by_step(
    target: torch.Tensor, k_steps: int, n_per_step: int,
) -> List[torch.Tensor]:
    """Split (B, C, K * n_per_step, H, W) into K windows of (B, C, n_per_step, H, W).

    Pairs with the K-window emission added to ``data_loader._getitem_prediction``.
    """
    expected = k_steps * n_per_step
    if target.shape[2] < expected:
        raise ValueError(
            f"video target T={target.shape[2]} < expected K*n={expected}"
        )
    return [
        target[:, :, k * n_per_step : (k + 1) * n_per_step].contiguous()
        for k in range(k_steps)
    ]


def _spectro_loss_gate(
    name: str, batch: Dict, device: torch.device,
) -> torch.Tensor:
    """Per-sample loss gate from per-modality presence ``<name>_valid``.

    Spectrograms have no per-channel runtime availability mask; the
    gate is just a per-batch scalar broadcast over ``(B, C, F, T)``.
    """
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()
    return valid[:, None, None, None]                # (B, 1, 1, 1)


def split_spectro_target_by_step(
    target: torch.Tensor, k_steps: int, trunc_t: int,
) -> List[torch.Tensor]:
    """Split (B, C, F, T) into K windows of ``trunc_t`` frames each.

    ``trunc_t`` must equal the spectrogram tokenizer's truncated time
    length — i.e. ``(DiagnosticConfig.window_samples // T_p) * T_p``,
    typically 96 for the standard 98-frame, T_p=8 config. The
    spectrogram head emits exactly ``trunc_t`` frames per step, so the
    target is sliced to the same length to match shapes for the
    masked-MAE loss. Frames past ``K * trunc_t`` are discarded — STFT
    over the full extended (input+prediction) window with
    ``center=True`` doesn't produce a frame count that divides cleanly
    by K, so a handful of trailing frames are dropped (typically <2%
    of the window).
    """
    needed = k_steps * trunc_t
    if target.shape[3] < needed:
        raise ValueError(
            f"spectro target T={target.shape[3]} < K * trunc_t = {needed}"
        )
    return [
        target[:, :, :, k * trunc_t : (k + 1) * trunc_t].contiguous()
        for k in range(k_steps)
    ]


def _spectro_trunc_t(cfg: "DiagnosticConfig") -> int:
    """Return the per-step time-axis truncation for a spectrogram cfg.

    Mirrors ``SpectrogramTokenizer.trunc_t`` so trainer-side target
    slicing and the head's ``patch_unembed`` output stay in lockstep.
    """
    assert cfg.kind == "spectrogram" and cfg.spectrogram_patch_size is not None
    _, T_p = cfg.spectrogram_patch_size
    return (cfg.window_samples // T_p) * T_p


def displacement_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    existing_mask: Optional[torch.Tensor],
    min_disp_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-modality-per-step cos + log-mag displacement losses.

    Returns five scalar tensors on the input device:
    ``(cos_loss, mag_loss, dir_cos_summary, mag_ratio_summary, n_valid)``.

    Gradients flow through ``cos_loss`` and ``mag_loss``. The last three are
    detached scalars suitable for logging; they remain on-device so callers
    can batch them into a single ``.cpu()`` transfer at the end of the
    forward pass instead of forcing a sync per (step, modality).

    Implementation notes — the prior version called ``valid.sum().item()``
    and two ``.item()`` calls per invocation, and used boolean indexing
    ``dp_flat[valid]`` which creates dynamic-shape gathers. At K=10 with 8
    modalities that added up to ~320 CUDA syncs per training step and was
    the main source of the observed 25× slowdown vs. pure-MAE Stage 2.
    This version uses **mask-weighted means on static shapes**: cos and
    mag are computed for the full batch and then reduced with the
    ``valid.float()`` weights.
    """
    cleaned_pred, pm = _clean_and_mask(pred, None)
    cleaned_tgt, tm = _clean_and_mask(target, existing_mask)
    cleaned_ctx, cm = _clean_and_mask(ctx, None)
    joint = pm * tm * cm
    disp_pred = (cleaned_pred - cleaned_ctx) * joint
    disp_tgt = (cleaned_tgt - cleaned_ctx) * joint

    batch = disp_pred.shape[0]
    dp_flat = disp_pred.reshape(batch, -1)
    dt_flat = disp_tgt.reshape(batch, -1)
    tgt_norm = dt_flat.norm(dim=1)
    pred_norm = dp_flat.norm(dim=1)

    # Static-shape validity mask; no boolean indexing anywhere downstream.
    valid_f = (tgt_norm > min_disp_norm).float()
    n_valid = valid_f.sum()
    denom = n_valid.clamp_min(1.0)

    # Whole-batch per-sample cosine + log-mag diff; select with the mask.
    cos_per = F.cosine_similarity(dp_flat, dt_flat, dim=1, eps=1e-8)
    cos_loss = ((1.0 - cos_per) * valid_f).sum() / denom

    eps = 1e-6
    log_pred = torch.log(pred_norm.clamp_min(eps))
    log_tgt = torch.log(tgt_norm.clamp_min(eps))
    mag_per = (log_pred - log_tgt).abs()
    mag_loss = (mag_per * valid_f).sum() / denom

    # Scalar-tensor summaries (no .item() — batched to CPU by caller).
    dir_cos_summary = (cos_per.detach() * valid_f).sum() / denom
    mag_ratio_summary = (
        (pred_norm.detach() / tgt_norm.detach().clamp_min(eps)) * valid_f
    ).sum() / denom

    return cos_loss, mag_loss, dir_cos_summary, mag_ratio_summary, n_valid.detach()


# ── Curriculum ───────────────────────────────────────────────────────────


def current_K(step: int, curriculum_steps: int, K_max: int) -> int:
    block = max(1, curriculum_steps // K_max)
    return min(K_max, step // block + 1)


# ── Rollout forward + per-step loss ──────────────────────────────────────


def _patch_grid_smoothness_loss(
    pred: torch.Tensor,
    patch_size: Tuple[int, int, int],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalize per-pixel discontinuities at the video patch-grid boundaries.

    VideoOutputHead uses ``ConvTranspose3d(kernel=stride=patch_size)`` —
    each token decodes its own (T_p, H_p, W_p) patch INDEPENDENTLY, so
    neighbouring patches have no pixel-level continuity by construction.
    The autoregressive Stage 2 round-trip bakes the patch grid into the
    model's representation, producing the visible 12×12 checkerboard.
    This loss penalises the L2 of differences across each patch-grid
    boundary along T/H/W. Within-patch gradients (real plasma features)
    are untouched.

    pred shape: ``(B, T, C, H, W)``; patch_size = (T_p, H_p, W_p).
    """
    T_p, H_p, W_p = patch_size
    loss = pred.new_zeros(())
    n_terms = 0
    if mask is not None:
        pred = pred * mask
    H = pred.shape[-2]
    if H_p > 0 and H > H_p:
        b = torch.arange(H_p, H, H_p, device=pred.device)
        if b.numel() > 0:
            diff = pred[..., b, :] - pred[..., b - 1, :]
            loss = loss + (diff ** 2).mean()
            n_terms += 1
    W = pred.shape[-1]
    if W_p > 0 and W > W_p:
        b = torch.arange(W_p, W, W_p, device=pred.device)
        if b.numel() > 0:
            diff = pred[..., :, b] - pred[..., :, b - 1]
            loss = loss + (diff ** 2).mean()
            n_terms += 1
    T_dim = pred.shape[1]
    if T_p > 0 and T_dim > T_p:
        b = torch.arange(T_p, T_dim, T_p, device=pred.device)
        if b.numel() > 0:
            diff = pred[:, b, ...] - pred[:, b - 1, ...]
            loss = loss + (diff ** 2).mean()
            n_terms += 1
    return loss / max(n_terms, 1)


def rollout_forward_loss_delta(
    rollout: TokenSpaceRollout,
    batch: Dict,
    diagnostic_names: List[str],
    actuator_names: List[str],
    k_steps: int,
    chunk_duration_s: float,
    device: torch.device,
    mae_weight: float,
    cos_weight: float,
    mag_weight: float,
    min_disp_norm: float,
    video_diag_names: Optional[List[str]] = None,
    video_n_frames: Optional[Dict[str, int]] = None,
    spectro_diag_names: Optional[List[str]] = None,
    grad_checkpoint_every: int = 0,
    video_smoothness_weight: float = 0.0,
    spec_pb_weights: Optional[Dict[str, torch.Tensor]] = None,
    keep_displacement_for_flow: bool = False,
) -> Tuple[torch.Tensor, List[Dict[str, Dict[str, float]]]]:
    """Tokenise step-0, split targets/actuators, run K-step rollout with full
    backprop, and return (summed loss, per-step per-modality metrics).

    Per-step, per-modality metrics dict contains::

        {"mae": float, "dir_cos": float, "mag_ratio": float}

    Video and spectrogram modalities use plain MAE only (no displacement
    loss). Video has per-batch (B, C) z-score applied to inputs/targets;
    spectrograms keep the data loader's ``log_standardize`` and skip
    per-batch z-score (resolved Open Decision #6 in the spectrogram plan).
    """
    video_diag_names = video_diag_names or []
    video_n_frames = video_n_frames or {}
    spectro_diag_names = spectro_diag_names or []
    video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    diag_initial: Dict[str, torch.Tensor] = {}
    for name in diagnostic_names:
        raw = batch["inputs"][name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if name in video_diag_names:
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            video_stats[name] = (mu, sd)
        diag_initial[name] = cleaned
        if name in video_diag_names or name in spectro_diag_names:
            # Route per-modality presence so the model's tokenize() can
            # substitute the learned ``missing_token`` for absent samples.
            valid_key = f"{name}_valid"
            if valid_key in batch["inputs"]:
                diag_initial[valid_key] = batch["inputs"][valid_key].to(
                    device, non_blocking=True
                )

    act_per_step: List[Dict[str, torch.Tensor]] = []
    target_per_step: List[Dict[str, torch.Tensor]] = []
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
    video_target_full: Dict[str, torch.Tensor] = {}
    video_gate: Dict[str, torch.Tensor] = {}
    for name in video_diag_names:
        raw = batch["targets"][name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        mu, sd = video_stats[name]
        video_target_full[name] = (cleaned - mu) / sd
        video_gate[name] = _video_loss_gate(name, batch, device)
    spectro_target_full: Dict[str, torch.Tensor] = {}
    spectro_gate: Dict[str, torch.Tensor] = {}
    spectro_trunc_t: Dict[str, int] = {}
    # Use _core(rollout) for the metadata read so this works whether the
    # rollout is DDP-wrapped (training) or already unwrapped (validate()).
    # DDP only proxies forward(); arbitrary attribute access like .model
    # raises AttributeError on the DDP wrapper.
    cfg_by_name = {c.name: c for c in _core(rollout).model.diagnostics}
    for name in spectro_diag_names:
        raw = batch["targets"][name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        spectro_target_full[name] = cleaned                # no standardization
        spectro_gate[name] = _spectro_loss_gate(name, batch, device)
        spectro_trunc_t[name] = _spectro_trunc_t(cfg_by_name[name])

    for k in range(k_steps):
        act_k: Dict[str, torch.Tensor] = {}
        for name in actuator_names:
            raw = batch["targets"][name].to(device).float()
            slc = split_target_by_step(raw, name, k_steps, chunk_duration_s)[k]
            cleaned, _ = _clean_and_mask(slc, None)
            act_k[name] = cleaned
        act_per_step.append(act_k)

        tgt_k: Dict[str, torch.Tensor] = {}
        mk_k: Dict[str, Optional[torch.Tensor]] = {}
        for name in diagnostic_names:
            if name in video_diag_names:
                n_per = video_n_frames[name]
                tgt_k[name] = split_video_target_by_step(
                    video_target_full[name], k_steps, n_per
                )[k]
                mk_k[name] = video_gate[name]   # per-shot, broadcast over T
                continue
            if name in spectro_diag_names:
                tgt_k[name] = split_spectro_target_by_step(
                    spectro_target_full[name], k_steps,
                    trunc_t=spectro_trunc_t[name],
                )[k]
                mk_k[name] = spectro_gate[name]   # per-shot, broadcast over (F, T)
                continue
            raw = batch["targets"][name].to(device).float()
            tgt_k[name] = split_target_by_step(raw, name, k_steps, chunk_duration_s)[k]
            mask_key = f"{name}_mask"
            if mask_key in batch["targets"]:
                raw_mask = batch["targets"][mask_key].to(device).float()
                mk_k[name] = split_target_by_step(
                    raw_mask, name, k_steps, chunk_duration_s
                )[k]
            else:
                mk_k[name] = None
        target_per_step.append(tgt_k)
        mask_per_step.append(mk_k)

    # Gradient checkpointing on the rollout (ported from stage 2 extended).
    # When grad_checkpoint_every >= k_steps the entire K-step rollout is one
    # checkpoint group: forward activations are discarded; recomputed during
    # backward → ~K-fold less activation memory at ~33% step-time penalty.
    # Per-group chunking (0 < g < k_steps) needs the chunk_fn pattern from
    # stage 2 extended — not ported here.
    #
    # Bypass DDP inside the checkpointed function (use _core(rollout))
    # to avoid DDP forward hooks firing twice (first forward + recompute
    # backward), which on MI250X produces "Memory access fault by GPU".
    # DDP's gradient all_reduce still works correctly because the hooks
    # are registered on parameters and fire when grads are populated,
    # independent of which forward path produced the gradient.
    inner_rollout = _core(rollout)
    model_ref = inner_rollout.model
    # Collect per-step backbone token slices only when a generative spectro
    # head is present (it needs them as conditioning for its flow loss);
    # otherwise the rollout is byte-identical to before.
    has_flow = any(
        isinstance(model_ref.diag_heads[n], SpectrogramFlowHead)
        for n in spectro_diag_names
    )

    def _checkpointed_rollout(diag_init, act):
        r = inner_rollout(diag_init, act, collect_token_slices=has_flow)
        return r.predictions, r.diag_token_slices

    token_slices: List[Dict[str, torch.Tensor]] = []
    if grad_checkpoint_every <= 0:
        res = rollout(diag_initial, act_per_step, collect_token_slices=has_flow)
        predictions = res.predictions
        token_slices = res.diag_token_slices
    elif grad_checkpoint_every >= k_steps:
        predictions, token_slices = torch_ckpt.checkpoint(
            _checkpointed_rollout, diag_initial, act_per_step,
            use_reentrant=False,
        )
    else:
        raise NotImplementedError(
            f"grad_checkpoint_every={grad_checkpoint_every} < "
            f"k_steps={k_steps}: per-group chunking is not ported to "
            "stage 2 delta. Pass 0 (off) or a value >= k_steps "
            f"(single group). Current k_steps={k_steps}."
        )
    # Video heads emit (B, T, C, H, W); permute per step to (B, C, T, H, W)
    # so loss / metric paths see a single shape contract.
    for k in range(k_steps):
        for name in video_diag_names:
            predictions[k][name] = predictions[k][name].permute(0, 2, 1, 3, 4)

    # Accumulate per-(step, modality) metrics as on-device scalar tensors;
    # transfer them to CPU once at the end of the forward pass instead of
    # 4 .item() calls per (step, modality) — which was the dominant cost
    # in the pre-refactor path (320 syncs/training step at K=10).
    total_loss = torch.zeros((), device=device)
    mae_grid: List[List[torch.Tensor]] = []
    dcos_grid: List[List[torch.Tensor]] = []
    mr_grid: List[List[torch.Tensor]] = []
    nvalid_grid: List[List[torch.Tensor]] = []
    for k in range(k_steps):
        mae_row: List[torch.Tensor] = []
        dcos_row: List[torch.Tensor] = []
        mr_row: List[torch.Tensor] = []
        nv_row: List[torch.Tensor] = []
        for name in diagnostic_names:
            pred = predictions[k][name]
            target = target_per_step[k][name]
            mask = mask_per_step[k][name]
            if name in video_diag_names:
                # Video: MAE only — cosine in ~900k pixels is meaningless
                # (project_phase_c_video_design memory). dir_cos and
                # mag_ratio reported as NaN / 0 for the metric grid.
                mae = masked_mae(pred, target, mask)
                total_loss = total_loss + mae_weight * mae
                # Patch-grid smoothness — fights the 12×12 checkerboard
                # baked into Stage 2's representation by the autoregressive
                # round-trip through patch_unembed → re-tokenize.
                if video_smoothness_weight > 0.0:
                    cfg = cfg_by_name[name]
                    patch_size = cfg.video_patch_size
                    smooth = _patch_grid_smoothness_loss(pred, patch_size, mask)
                    total_loss = total_loss + video_smoothness_weight * smooth
                mae_row.append(mae.detach())
                zero = torch.zeros((), device=pred.device)
                dcos_row.append(zero)
                mr_row.append(zero)
                nv_row.append(zero)
                continue
            # Context: teacher-forced — ground-truth state at step k-1
            # (= window index k in the pool). At k=0, ctx is the rollout
            # input (diag_initial). Spectrogram diag_initial holds the
            # full STFT output (e.g. 98 frames) while pred/target are
            # already truncated to trunc_t (e.g. 96), so at k=0 we slice
            # diag_initial to match. At k>=1 ctx comes from
            # target_per_step[k-1] which is already trunc_t-sized.
            # 2026-06-01: spectrograms now get displacement loss too —
            # previously gated off (Open Decision #3), but the gating
            # let the spectrogram head collapse to outputting a near-
            # constant ≈ dataset mean (see project-spectrogram-mean-
            # collapse memory). Cosine + magnitude regularization is the
            # missing signal.
            if k == 0:
                if name in spectro_diag_names:
                    ctx = diag_initial[name][..., : spectro_trunc_t[name]]
                else:
                    ctx = diag_initial[name]
            else:
                ctx = target_per_step[k - 1][name]

            if (spec_pb_weights is not None
                    and name in spectro_diag_names
                    and name in spec_pb_weights):
                # Per-bin weighted MAE (anti mean-collapse) — same
                # semantics as the Stage 1 specfix trainer.
                mae = weighted_masked_mae(
                    pred, target, mask, spec_pb_weights[name]
                )
            else:
                mae = masked_mae(pred, target, mask)
            cos_loss, mag_loss, dcos_t, mr_t, nv_t = displacement_losses(
                pred, target, ctx, mask, min_disp_norm
            )
            head = model_ref.diag_heads[name]
            if isinstance(head, SpectrogramFlowHead):
                # Generative spectro: pred == μ (train mode); the flow loss on
                # the residual owns the mode structure. Replace the cos+mag
                # displacement (the failed deterministic mode-fix) unless
                # explicitly kept for ablation.
                flow = head.flow_loss(
                    token_slices[k][name], pred, target, mask
                )
                step_loss = mae_weight * mae + head.flow_lambda * flow
                if keep_displacement_for_flow:
                    step_loss = (
                        step_loss
                        + cos_weight * cos_loss + mag_weight * mag_loss
                    )
            else:
                step_loss = (
                    mae_weight * mae
                    + cos_weight * cos_loss + mag_weight * mag_loss
                )
            total_loss = total_loss + step_loss
            mae_row.append(mae.detach())
            dcos_row.append(dcos_t)
            mr_row.append(mr_t)
            nv_row.append(nv_t)
        mae_grid.append(mae_row)
        dcos_grid.append(dcos_row)
        mr_grid.append(mr_row)
        nvalid_grid.append(nv_row)

    # Single cross-device transfer of (k_steps × n_modalities) scalars.
    mae_cpu = torch.stack([torch.stack(r) for r in mae_grid]).detach().cpu()
    dcos_cpu = torch.stack([torch.stack(r) for r in dcos_grid]).detach().cpu()
    mr_cpu = torch.stack([torch.stack(r) for r in mr_grid]).detach().cpu()
    nv_cpu = torch.stack([torch.stack(r) for r in nvalid_grid]).detach().cpu()

    per_step: List[Dict[str, Dict[str, float]]] = []
    for k in range(k_steps):
        per_mod: Dict[str, Dict[str, float]] = {}
        for j, name in enumerate(diagnostic_names):
            nv = float(nv_cpu[k, j].item())
            per_mod[name] = {
                "mae": float(mae_cpu[k, j].item()),
                "dir_cos": float(dcos_cpu[k, j].item()) if nv > 0 else float("nan"),
                "mag_ratio": float(mr_cpu[k, j].item()) if nv > 0 else float("nan"),
                "n_valid": int(nv),
            }
        per_step.append(per_mod)
    return total_loss, per_step


# ── Validation ───────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    rollout: TokenSpaceRollout,
    loader: DataLoader,
    device: torch.device,
    diagnostic_names: List[str],
    actuator_names: List[str],
    chunk_duration_s: float,
    K_max: int,
    min_disp_norm: float,
    max_batches: Optional[int] = None,
    video_diag_names: Optional[List[str]] = None,
    video_n_frames: Optional[Dict[str, int]] = None,
    spectro_diag_names: Optional[List[str]] = None,
    step: int = 0,
    ckpt_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Full K=K_max rollout; return per-step per-modality averaged metrics.

    Each modality's dict carries: ``model_mae, copy_mae, dir_cos, mag_ratio``.
    Copy baseline is the step-0 input echoed to every step.

    Video and spectrogram modalities use MAE-only metrics; ``dir_cos`` /
    ``mag_ratio`` are reported as NaN. Video gets per-(B, C) z-score;
    spectrograms keep the data loader's ``log_standardize`` only.
    """
    video_diag_names = video_diag_names or []
    video_n_frames = video_n_frames or {}
    spectro_diag_names = spectro_diag_names or []
    rollout.model.eval()
    keys = ("model_mae", "copy_mae", "dir_cos", "mag_ratio",
            "pred_var", "gt_var")
    sums = {
        k: {n: {m: 0.0 for m in keys} for n in diagnostic_names}
        for k in range(K_max)
    }
    counts = {
        k: {n: {"mae": 0, "disp": 0} for n in diagnostic_names}
        for k in range(K_max)
    }
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        diag_initial: Dict[str, torch.Tensor] = {}
        for name in diagnostic_names:
            raw = batch["inputs"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            if name in video_diag_names:
                cleaned, mu, sd = _video_standardize_per_bc(cleaned)
                video_stats[name] = (mu, sd)
            diag_initial[name] = cleaned
            if name in video_diag_names or name in spectro_diag_names:
                vk = f"{name}_valid"
                if vk in batch["inputs"]:
                    diag_initial[vk] = batch["inputs"][vk].to(device, non_blocking=True)
        # Pre-build full-horizon video targets in standardised space; gates
        # are per-shot (broadcast over T).
        video_target_full: Dict[str, torch.Tensor] = {}
        video_gate: Dict[str, torch.Tensor] = {}
        for name in video_diag_names:
            raw = batch["targets"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            mu, sd = video_stats[name]
            video_target_full[name] = (cleaned - mu) / sd
            video_gate[name] = _video_loss_gate(name, batch, device)
        # Spectrogram targets stay in data-loader-normalized space
        # (log_standardize only); per-batch z-score deliberately
        # skipped (Open Decision #6).
        spectro_target_full: Dict[str, torch.Tensor] = {}
        spectro_gate: Dict[str, torch.Tensor] = {}
        spectro_trunc_t: Dict[str, int] = {}
        cfg_by_name = {c.name: c for c in rollout.model.diagnostics}
        for name in spectro_diag_names:
            raw = batch["targets"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            spectro_target_full[name] = cleaned
            spectro_gate[name] = _spectro_loss_gate(name, batch, device)
            spectro_trunc_t[name] = _spectro_trunc_t(cfg_by_name[name])

        act_per_step: List[Dict[str, torch.Tensor]] = []
        target_per_step: List[Dict[str, torch.Tensor]] = []
        mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
        for k in range(K_max):
            ak: Dict[str, torch.Tensor] = {}
            for name in actuator_names:
                raw = batch["targets"][name].to(device).float()
                ak[name], _ = _clean_and_mask(
                    split_target_by_step(raw, name, K_max, chunk_duration_s)[k],
                    None,
                )
            act_per_step.append(ak)
            tk: Dict[str, torch.Tensor] = {}
            mk: Dict[str, Optional[torch.Tensor]] = {}
            for name in diagnostic_names:
                if name in video_diag_names:
                    n_per = video_n_frames[name]
                    tk[name] = split_video_target_by_step(
                        video_target_full[name], K_max, n_per
                    )[k]
                    mk[name] = video_gate[name]
                    continue
                if name in spectro_diag_names:
                    tk[name] = split_spectro_target_by_step(
                        spectro_target_full[name], K_max,
                        trunc_t=spectro_trunc_t[name],
                    )[k]
                    mk[name] = spectro_gate[name]
                    continue
                raw = batch["targets"][name].to(device).float()
                tk[name] = split_target_by_step(raw, name, K_max, chunk_duration_s)[k]
                mask_key = f"{name}_mask"
                mk[name] = (
                    split_target_by_step(
                        batch["targets"][mask_key].to(device).float(),
                        name, K_max, chunk_duration_s,
                    )[k]
                    if mask_key in batch["targets"]
                    else None
                )
            target_per_step.append(tk)
            mask_per_step.append(mk)

        result = rollout(diag_initial, act_per_step)
        # Permute video predictions (B, T, C, H, W) -> (B, C, T, H, W).
        for k in range(K_max):
            for name in video_diag_names:
                result.predictions[k][name] = (
                    result.predictions[k][name].permute(0, 2, 1, 3, 4)
                )
        for k in range(K_max):
            for name in diagnostic_names:
                pred = result.predictions[k][name].float()
                target = target_per_step[k][name]
                mask = mask_per_step[k][name]
                if name in video_diag_names:
                    # Video: MAE only — no displacement metrics.
                    mae = masked_mae(pred, target, mask).item()
                    copy_mae = masked_mae(diag_initial[name], target, mask).item()
                    sums[k][name]["model_mae"] += mae
                    sums[k][name]["copy_mae"] += copy_mae
                    counts[k][name]["mae"] += 1
                    continue
                # Spectrogram diag_initial holds the full STFT output (e.g.
                # 98 frames) while pred/target are trunc_t (e.g. 96), so
                # slice diag_initial for both copy_mae and ctx so shapes
                # match. Slow_ts uses diag_initial directly (no trunc).
                if name in spectro_diag_names:
                    baseline_input = diag_initial[name][
                        ..., : spectro_trunc_t[name]
                    ]
                else:
                    baseline_input = diag_initial[name]
                ctx = baseline_input if k == 0 else target_per_step[k - 1][name]
                mae = masked_mae(pred, target, mask).item()
                copy_mae = masked_mae(baseline_input, target, mask).item()
                _, _, dir_cos, mag_ratio, n_valid = displacement_losses(
                    pred, target, ctx, mask, min_disp_norm
                )
                sums[k][name]["model_mae"] += mae
                sums[k][name]["copy_mae"] += copy_mae
                counts[k][name]["mae"] += 1
                if n_valid > 0:
                    sums[k][name]["dir_cos"] += dir_cos
                    sums[k][name]["mag_ratio"] += mag_ratio
                    counts[k][name]["disp"] += 1
                # Temporal-variance ratio for spectro modalities (collapse
                # diagnostic): var over the time axis, summed over valid bins.
                # pred/gt summed over the SAME mask → the count cancels in the
                # ratio, so no separate count is needed.
                if name in spectro_diag_names:
                    mv = (
                        (mask[..., 0] > 0).float() if mask is not None
                        else torch.ones(pred.shape[:3], device=pred.device)
                    )
                    sums[k][name]["pred_var"] += float(
                        (pred.var(dim=-1) * mv).sum()
                    )
                    sums[k][name]["gt_var"] += float(
                        (target.var(dim=-1) * mv).sum()
                    )

    rollout.model.train()

    # Aggregate metrics across DDP ranks. Tier 4 approach (2026-05-24):
    # NO DDP collectives inside validate(). Each rank writes its per-rank
    # sums/counts to a small .pt file in ckpt_dir; rank 0 polls for those
    # files (with a 5-min deadline for stragglers), merges available ones
    # into its own sums/counts, and cleans up. Non-rank-0 ranks return
    # with their rank-LOCAL metrics — that's fine because only rank 0
    # logs val_loss and saves best.pt.
    #
    # Earlier collective-based attempts (all_reduce, monitored_barrier
    # + gloo sync, etc.) hung because val pipeline rank skew was larger
    # than NCCL's 10-min watchdog (cold val NFS reads). Files + polling
    # tolerate arbitrary skew up to the deadline. The downstream
    # dm.barrier() in the training loop is the only post-val collective,
    # and all ranks reach it asynchronously after their own val loop +
    # file write completes.
    if (dist.is_available() and dist.is_initialized()
            and ckpt_dir is not None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        ckpt_dir_p = Path(ckpt_dir)
        my_file = ckpt_dir_p / f"_val_metrics_step{step}_rank{rank:03d}.pt"
        # Atomic write: write to .tmp then rename so rank 0 never sees a
        # half-written file.
        tmp_file = my_file.with_suffix(".pt.tmp")
        try:
            torch.save({"sums": sums, "counts": counts}, tmp_file)
            tmp_file.rename(my_file)
        except Exception as e:
            logger.warning(
                f"[rank{rank}] Tier 4 val metrics write failed: {e!r}"
            )
        if rank == 0:
            import time as _time
            deadline = _time.time() + 300.0  # 5 min for stragglers
            collected = {0}  # already have own sums/counts in memory
            while _time.time() < deadline and len(collected) < world_size:
                for r in range(world_size):
                    if r in collected:
                        continue
                    target = ckpt_dir_p / f"_val_metrics_step{step}_rank{r:03d}.pt"
                    if not target.exists():
                        continue
                    try:
                        other = torch.load(target, weights_only=False)
                    except Exception as e:
                        # Partial file? Will retry on next loop iteration.
                        logger.warning(
                            f"[rank0] failed to load rank {r} metrics "
                            f"(may be partial): {e!r}"
                        )
                        continue
                    for k in range(K_max):
                        for n in diagnostic_names:
                            for m in keys:
                                sums[k][n][m] += other["sums"][k][n][m]
                            for m in ("mae", "disp"):
                                counts[k][n][m] += other["counts"][k][n][m]
                    collected.add(r)
                if len(collected) < world_size:
                    _time.sleep(1.0)
            if len(collected) == world_size:
                logger.info(
                    f"[rank0] val merge: collected all {world_size} "
                    f"rank files at step {step}"
                )
            else:
                missing = [r for r in range(world_size) if r not in collected]
                logger.warning(
                    f"[rank0] val merge: collected {len(collected)}/"
                    f"{world_size} rank files at step {step} (5-min "
                    f"deadline hit). Missing ranks: {missing}. "
                    "val_loss reflects partial sample."
                )
            # Cleanup: remove per-rank files for this step.
            for r in range(world_size):
                target = ckpt_dir_p / f"_val_metrics_step{step}_rank{r:03d}.pt"
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(
                        f"[rank0] failed to unlink {target}: {e!r}"
                    )

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in range(K_max):
        out[k] = {}
        for name in diagnostic_names:
            mae_n = max(counts[k][name]["mae"], 1)
            disp_n = max(counts[k][name]["disp"], 1)
            gt_var = sums[k][name]["gt_var"]
            out[k][name] = {
                "model_mae": sums[k][name]["model_mae"] / mae_n,
                "copy_mae": sums[k][name]["copy_mae"] / mae_n,
                "dir_cos": sums[k][name]["dir_cos"] / disp_n
                if counts[k][name]["disp"] else float("nan"),
                "mag_ratio": sums[k][name]["mag_ratio"] / disp_n
                if counts[k][name]["disp"] else float("nan"),
                "tvr": (sums[k][name]["pred_var"] / gt_var
                        if gt_var > 1e-8 else float("nan")),
            }
    return out


def build_scheduler(
    opt: torch.optim.Optimizer, max_steps: int, warmup_steps: int, min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=max(warmup_steps, 1)
    )
    cosine_steps = max(max_steps - warmup_steps, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cosine_steps, eta_min=min_lr
    )
    return torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[max(warmup_steps, 1)]
    )


def head_weight_l2(model: E2EFoundationModel) -> Dict[str, float]:
    """L2 norm of each diagnostic head's main projection weight — monitored
    for head unstuck-ness. If these don't move after 5k steps, heads are
    in a flat region.

    Picks the conventional weight tensor per head kind:
    * slow_ts (``SlowTimeSeriesHead``)         -> ``head.proj.weight``
    * fast_ts (``FastTimeSeriesHead``)         -> ``head.deconv.weight``
    * spectrogram (``SpectrogramOutputHead``)  -> ``head.patch_unembed.weight``
    * video (``VideoOutputHead``)              -> ``head.patch_unembed.weight``

    Falls back to the head's first parameter for unknown kinds so future
    additions surface without a code edit.
    """
    out: Dict[str, float] = {}
    for cfg in model.diagnostics:
        head = model.diag_heads[cfg.name]
        if hasattr(head, "proj"):                # slow_ts
            w = head.proj.weight
        elif hasattr(head, "deconv"):            # fast_ts
            w = head.deconv.weight
        elif hasattr(head, "patch_unembed"):     # spectrogram, video
            w = head.patch_unembed.weight
        else:
            params = list(head.parameters())
            if not params:
                continue
            w = params[0]
        out[cfg.name] = w.detach().float().norm().item()
    return out


# ── Driver ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument(
        "--lengths_cache_dir",
        type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model_meta"),
        help="Directory for TokamakMultiFileDataset length-cache sidecar "
        "files (lengths_e2e_stage2_delta_{train,val}.pt) and the "
        "video-presence cache (video_present_{train,val}.pt). Defaults "
        "to the same shared dir Stage 1 uses so the video-presence "
        "cache is reused — it only depends on (paths, camera_names), "
        "not the stage. Kept separate from --checkpoint_dir so cache "
        "files survive checkpoint-dir cleanups.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=Path,
        default=None,
        help="Stage 1 best checkpoint to initialise from. Random init if omitted "
        "(smoke-test only — real Stage 2b must warm-start from Stage 1 best).",
    )
    parser.add_argument("--train_shots_yaml", type=Path, default=None)
    parser.add_argument("--val_shots_yaml", type=Path, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--chunk_duration_s", type=float, default=0.05)
    parser.add_argument("--step_size_s", type=float, default=0.01)
    parser.add_argument("--warmup_s", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--backbone_grad_checkpoint", action="store_true",
        help="Per-block gradient checkpointing in the shared backbone. "
        "Required at d_model=1024+ where activations don't fit per-GCD VRAM.",
    )

    parser.add_argument(
        "--use_video", nargs="*", default=[],
        choices=[entry[0] for entry in VIDEO_MODALITIES],
        help="Camera names (e.g. tangtv). Empty (default) reproduces "
             "TS-only Stage 2b byte-for-byte.",
    )
    parser.add_argument(
        "--use_spectro", nargs="*", default=[],
        choices=[entry[0] for entry in SPECTROGRAM_MODALITIES],
        help="Spectrogram modality names (e.g. ece co2 bes). Empty "
             "(default) keeps Stage 2b TS-only / TS+video byte-for-byte. "
             "Spectrograms train under MAE-only loss (displacement "
             "deferred per the spectrogram plan's Open Decision #3).",
    )
    parser.add_argument(
        "--spectro_patch_f", type=int, default=None,
        help="Override spectro freq-patch size for all spectro modalities "
             "(default: registry). 512 = full-frequency patch. Must match "
             "the Stage-1 init checkpoint's patch shape.",
    )
    parser.add_argument(
        "--spectro_patch_t", type=int, default=None,
        help="Override spectro time-patch size (default: registry). Must "
             "match the Stage-1 init checkpoint's patch shape.",
    )
    parser.add_argument("--K_max", type=int, default=10)
    parser.add_argument("--curriculum_steps", type=int, default=25_000)
    parser.add_argument(
        "--grad_checkpoint_every", type=int, default=10,
        help="Gradient checkpointing group size for the K-step rollout. "
        "0 = disabled (full activation memory). >= k_steps = single "
        "checkpoint group covering the entire rollout (recommended for "
        "K_max=10: pass 10). Activations within the group are discarded "
        "after forward and recomputed during backward (~33%% step-time "
        "penalty in exchange for ~K-fold less activation memory). "
        "Values 0 < g < k_steps would need per-group chunking (matching "
        "stage 2 extended); not yet supported here.",
    )

    # Loss weights — Stage 2b specific.
    parser.add_argument("--mae_weight", type=float, default=1.0)
    parser.add_argument("--cos_weight", type=float, default=0.3)
    parser.add_argument("--mag_weight", type=float, default=0.1)
    parser.add_argument(
        "--min_disp_norm",
        type=float,
        default=0.01,
        help="Minimum target-displacement norm (per-sample) below which the "
        "cosine and magnitude terms do not contribute. Prevents wasting "
        "gradient on samples where copy is the correct prediction.",
    )
    parser.add_argument(
        "--video_smoothness_weight",
        type=float, default=0.0,
        help="Weight on the patch-grid smoothness loss for video predictions. "
        "Suppresses the 12×12 checkerboard baked into Stage 2's "
        "representation by the autoregressive round-trip through "
        "ConvTranspose3d(kernel=stride=patch_size). 0.0 disables; "
        "0.1 is a reasonable starting point. Applied per-step for each "
        "video modality with the same loss gate as the per-step MAE.",
    )

    # ── Specfix flags (2026-06-12) — all default-off / legacy-shaped so
    # production chains are unaffected. Mirror the Stage 1 trainer.
    parser.add_argument(
        "--spec_per_bin_loss", action="store_true",
        help="Spectrogram MAE terms use the per-(channel, freq-bin) "
             "weighted MAE (anti mean-collapse). Requires 'log_per_bin' "
             "in preprocessing_stats.",
    )
    parser.add_argument(
        "--spec_per_bin_weight_clamp", type=float, default=10.0,
        help="Upper clamp on the per-bin weight (lower fixed at 1.0).",
    )
    parser.add_argument(
        "--spec_inv_stem", action="store_true",
        help="Build spec heads with the inv_stem feature-space decode "
             "branch (zero-init residual). Shape-mismatched or absent "
             "keys in the init checkpoint are re-initialized.",
    )
    parser.add_argument("--spec_inv_stem_ch", type=int, default=64)
    parser.add_argument("--spec_freq_stem", action="store_true",
                        help="Full-frequency encoder stem on spec "
                             "tokenizers (zero-init residual).")
    parser.add_argument("--spec_freq_stem_hidden", type=int, default=128)
    parser.add_argument("--spec_per_bin_weight_power", type=float, default=1.0,
                        help="Exponent on (sigma_c/sigma_pb) before clamp.")
    # ── Generative-head / checkerboard-fix flags (2026-06-21) ──
    parser.add_argument("--video_resize_conv", action="store_true",
                        help="Resize-conv video decoder (kills checkerboard); "
                             "supersedes seam-refine. From-scratch only.")
    parser.add_argument("--video_resize_conv_hidden", type=int, default=64)
    parser.add_argument("--spec_generative", action="store_true",
                        help="Generative SpectrogramFlowHead (rectified flow "
                             "over a deterministic mean). From-scratch only.")
    parser.add_argument("--spec_flow_base_ch", type=int, default=64)
    parser.add_argument("--spec_flow_steps", type=int, default=6)
    parser.add_argument("--spec_flow_lambda", type=float, default=1.0)
    parser.add_argument("--spec_gen_keep_displacement", action="store_true",
                        help="Keep cos+mag displacement loss on a generative "
                             "spectro head's mean (default off — the flow "
                             "loss owns mode structure). Ablation knob.")
    parser.add_argument("--collapse_aware_best", action="store_true",
                        help="Penalise low TVR (mean-collapse) in best.pt "
                             "selection so it can't win on MAE alone.")
    parser.add_argument("--collapse_aware_lambda", type=float, default=1.0)
    parser.add_argument(
        "--seam_refine_hidden_ch", type=int, default=16,
        help="Hidden channels of seam-refine blocks (16 = legacy).",
    )
    parser.add_argument("--spectro_refine_kernel", type=int, default=3)
    parser.add_argument(
        "--video_refine_kernel", type=int, nargs=3, default=[1, 3, 3],
    )
    parser.add_argument(
        "--freeze_categories", nargs="*", default=[],
        choices=["slow_ts", "fast_ts", "video", "spectro", "backbone"],
        help="Module categories frozen for the ENTIRE run, applied "
             "BEFORE the DDP wrap (reducer excludes them — no "
             "unused-parameter crash, no release-divergence). Empty "
             "(default) = everything trainable, as in production.",
    )

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=50_000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_max_batches", type=int, default=20)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--resume_checkpoint", type=Path, default=None,
        help="Resume from a *_latest.pt or *_final.pt, restoring model + "
        "optimizer + scheduler + step + best_val_loss. Overrides the "
        "--init_checkpoint path. Intended for SLURM resubmission.",
    )
    args = parser.parse_args()

    dm = DistributedManager()

    logging.basicConfig(
        level=logging.INFO if dm.is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{dm.rank}] %(message)s",
    )

    # OOM mitigation. Stage-1 chained jobs 4581026/27/28 OOM'd at
    # ~9h45m / ~5850 steps with num_workers=6. Clamp here so already-
    # queued stage-2 jobs that read this Python source at start-time
    # inherit the cap without needing re-submission.
    if args.num_workers > 4:
        logger.warning(
            f"Capping --num_workers {args.num_workers} → 4 (OOM mitigation; "
            "see persistent_workers comment in this file)."
        )
        args.num_workers = 4

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if dm.distributed:
        device = dm.device
    else:
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
    logger.info(
        f"Device: {device}  distributed={dm.distributed} "
        f"rank={dm.rank}/{dm.world_size}"
    )
    if dm.is_main:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args.lengths_cache_dir.mkdir(parents=True, exist_ok=True)
    dm.barrier()

    train_files, val_files = resolve_shot_files(
        args.data_dir, args.train_shots_yaml, args.val_shots_yaml,
        args.max_files, args.val_fraction, args.seed,
    )
    logger.info(f"Files — train: {len(train_files)}  val: {len(val_files)}")
    if not train_files or not val_files:
        raise SystemExit("No train or val files resolved; aborting.")
    if args.use_video:
        n_train_pre, n_val_pre = len(train_files), len(val_files)
        train_files = filter_video_present_files(
            train_files, args.use_video,
            cache_path=args.lengths_cache_dir / "video_present_train.pt",
        )
        val_files = filter_video_present_files(
            val_files, args.use_video,
            cache_path=args.lengths_cache_dir / "video_present_val.pt",
        )
        logger.info(
            f"Video-presence filter ({args.use_video}): "
            f"train {n_train_pre} -> {len(train_files)} "
            f"({100 * len(train_files) / max(1, n_train_pre):.1f}%); "
            f"val {n_val_pre} -> {len(val_files)} "
            f"({100 * len(val_files) / max(1, n_val_pre):.1f}%)"
        )
        if not train_files or not val_files:
            raise SystemExit(
                f"Video-presence filter dropped all files. Check that "
                f"{args.use_video} HDF5 groups exist in the data dir."
            )
    stats = torch.load(args.stats_path, weights_only=False)

    diagnostics, actuators = build_configs(
        args.chunk_duration_s,
        use_video=args.use_video,
        use_spectro=args.use_spectro,
        spectro_patch_f=args.spectro_patch_f,
        spectro_patch_t=args.spectro_patch_t,
    )
    diagnostic_names = [c.name for c in diagnostics]
    actuator_names = [c.name for c in actuators]
    video_diag_names = [c.name for c in diagnostics if c.kind == "video"]
    video_n_frames = {c.name: c.window_samples for c in diagnostics if c.kind == "video"}
    spectro_diag_names = [c.name for c in diagnostics if c.kind == "spectrogram"]
    logger.info(f"Diagnostics ({len(diagnostics)}): " + ", ".join(diagnostic_names))
    logger.info(f"Actuators ({len(actuators)}): " + ", ".join(actuator_names))

    # Stage 2 enables the VideoOutputHead + SpectrogramOutputHead
    # seam-refine blocks — fights the patch-grid checkerboard that
    # the autoregressive K-step rollout amplifies. Stage 1
    # (single-step) leaves them off.
    video_seam_refine_flag = True
    spectro_seam_refine_flag = True
    logger.info(
        f"Seam-refine: video_seam_refine={video_seam_refine_flag} "
        f"spectro_seam_refine={spectro_seam_refine_flag}"
    )
    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
        backbone_grad_checkpoint=args.backbone_grad_checkpoint,
        video_seam_refine=video_seam_refine_flag,
        spectro_seam_refine=spectro_seam_refine_flag,
        seam_refine_hidden_ch=args.seam_refine_hidden_ch,
        spectro_refine_kernel=args.spectro_refine_kernel,
        video_refine_kernel=tuple(args.video_refine_kernel),
        spectro_inv_stem=args.spec_inv_stem,
        spectro_inv_stem_ch=args.spec_inv_stem_ch,
        spectro_freq_stem=args.spec_freq_stem,
        spectro_freq_stem_hidden=args.spec_freq_stem_hidden,
        video_resize_conv=args.video_resize_conv,
        video_resize_conv_hidden=args.video_resize_conv_hidden,
        spectro_generative=args.spec_generative,
        spectro_flow_base_ch=args.spec_flow_base_ch,
        spectro_flow_sample_steps=args.spec_flow_steps,
        spectro_flow_lambda=args.spec_flow_lambda,
    ).to(device)
    # Confirm the head modules actually built the refine_block — a
    # safety net so an accidental rename of the param (or a stale
    # cached .pyc) is visible at runtime instead of silently OFF.
    for name in spectro_diag_names:
        head = model.diag_heads[name]
        logger.info(
            f"  spectro head '{name}': enable_seam_refine="
            f"{getattr(head, 'enable_seam_refine', 'MISSING')}, "
            f"has refine_block={hasattr(head, 'refine_block')}"
        )

    if args.init_checkpoint is not None:
        ckpt = torch.load(
            args.init_checkpoint, weights_only=False, map_location=device
        )
        # When --use_video / --use_spectro is set and the init checkpoint
        # lacks those modules (e.g. Phase A Stage 1 best, or B/C-Stage 1
        # best with one modality only), allow the corresponding
        # tokenizer/head keys to be absent in the source state_dict. When
        # init already has them (BC-Stage 1 best with everything), all
        # keys match and the same call still works.
        allowed = tuple(
            f"diag_{kind}.{n}."
            for n in (*args.use_video, *args.use_spectro)
            for kind in ("tokenizers", "heads")
        )
        old_n_layers, backbone_extra_allowed = warm_start_extend_backbone(
            model, ckpt["model_state_dict"], args.n_layers,
        )
        if backbone_extra_allowed:
            logger.info(
                f"Warm-start: extending backbone from {old_n_layers} to "
                f"{args.n_layers} layers; new blocks "
                f"[{old_n_layers}, {args.n_layers}) initialised as "
                "near-identity (zero attn.out_proj + mlp final linear)."
            )
            allowed = allowed + backbone_extra_allowed
        # Specfix: a resized seam-refine block (e.g. 64ch/5x5 vs the
        # checkpoint's trained 16ch/3x3) makes the checkpoint's
        # refine_block keys SHAPE-MISMATCHED — torch raises on those
        # even with strict=False. Drop them explicitly (they fall under
        # the diag_heads.<name>. allowed-missing prefixes and re-init
        # zero → identity residual, retrained by this run). Logged so
        # the drop is never silent.
        init_sd = ckpt["model_state_dict"]
        model_sd = model.state_dict()
        mismatched = [
            k for k in init_sd
            if k in model_sd and init_sd[k].shape != model_sd[k].shape
        ]
        if mismatched:
            not_allowed = [
                k for k in mismatched
                if not any(k.startswith(p) for p in allowed)
            ]
            if not_allowed:
                raise RuntimeError(
                    "Shape-mismatched init keys outside allowed prefixes "
                    f"(refusing silent re-init): {not_allowed[:8]}"
                )
            logger.info(
                f"Init: dropping {len(mismatched)} shape-mismatched keys "
                f"(re-initialized fresh): {sorted(mismatched)[:6]} ..."
            )
            init_sd = {
                k: v for k, v in init_sd.items() if k not in mismatched
            }
        load_state_dict_explicit(
            model, init_sd, allowed_missing_prefixes=allowed
        )
        logger.info(
            f"Initialised from {args.init_checkpoint.name} "
            f"(val_loss={ckpt.get('val_loss', 'n/a')} "
            f"step={ckpt.get('step', 'n/a')})"
        )
    else:
        # The warning is only meaningful when neither warm-start path
        # will be used. Chained jobs in production pass --resume_checkpoint
        # (latest.pt), which loads weights ~50 lines below; emitting the
        # "random weights" warning before that resume happens is just
        # noise. Only fire it when both init AND resume paths are absent.
        will_resume = (
            args.resume_checkpoint is not None
            and args.resume_checkpoint.exists()
        )
        if not will_resume:
            logger.warning(
                "No --init_checkpoint and no --resume_checkpoint; random "
                "weights. Smoke-test only — real Stage 2b must warm-start "
                "from Stage 1 best, not Stage 2 best."
            )

    # Whole-run category freezes, applied BEFORE the DDP wrap so the
    # reducer is built without the frozen params (post-wrap flips crash
    # DDP; post-wrap releases silently diverge ranks — see Stage 1
    # trainer's --freeze_whole_run for the same pattern).
    if args.freeze_categories:
        labels = _stage1_apply_module_freeze(
            model,
            freeze_slow_ts="slow_ts" in args.freeze_categories,
            freeze_fast_ts="fast_ts" in args.freeze_categories,
            freeze_video="video" in args.freeze_categories,
            freeze_spectro="spectro" in args.freeze_categories,
            freeze_backbone="backbone" in args.freeze_categories,
        )
        n_total = sum(p.numel() for p in model.parameters())
        n_train = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            f"freeze_categories={args.freeze_categories}: frozen labels "
            f"= {labels}; trainable {n_train / 1e6:.2f}M / "
            f"{n_total / 1e6:.2f}M"
        )

    rollout = TokenSpaceRollout(model, dt_s=args.chunk_duration_s)
    rollout = dm.wrap(rollout)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model — d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads}  tokens={model.n_total_tokens}  "
        f"params={n_params / 1e6:.2f}M"
    )
    logger.info(
        f"Loss weights: α(mae)={args.mae_weight} β(cos)={args.cos_weight} "
        f"γ(mag)={args.mag_weight}  min_disp_norm={args.min_disp_norm}"
    )

    prediction_horizon_s = args.K_max * args.chunk_duration_s
    # Video diagnostic names are already in diagnostic_names; passing them
    # in input_signals + target_signals lets the dataset emit per-shot
    # input + K-window target frames (data_loader._getitem_prediction).
    shared = dict(
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=prediction_horizon_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diagnostic_names,
        target_signals=diagnostic_names + actuator_names,
    )
    train_ds = TokamakMultiFileDataset(
        train_files,
        lengths_cache_path=args.lengths_cache_dir / "lengths_e2e_stage2_delta_train.pt",
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=args.lengths_cache_dir / "lengths_e2e_stage2_delta_val.pt",
        **shared,
    )
    logger.info(
        f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}  "
        f"prediction_horizon_s={prediction_horizon_s:.3f} (K_max={args.K_max})"
    )

    # Per-bin spec loss weights (specfix). None = plain MAE (production).
    spec_pb_weights: Optional[Dict[str, torch.Tensor]] = None
    if args.spec_per_bin_loss:
        spec_pb_weights = build_spec_per_bin_weights(
            stats, model.diagnostics, train_ds.signal_configs,
            device=device, clamp_max=args.spec_per_bin_weight_clamp,
            power=args.spec_per_bin_weight_power,
        )
        if not spec_pb_weights:
            logger.warning(
                "--spec_per_bin_loss requested but preprocessing_stats "
                "lacks 'log_per_bin' for one or more spectrogram "
                "modalities — falling back to plain MAE."
            )
            spec_pb_weights = None
        else:
            for n, w in spec_pb_weights.items():
                logger.info(
                    f"Per-bin spec loss [{n}]: weight shape "
                    f"{tuple(w.shape)}, range [{float(w.min()):.2f}, "
                    f"{float(w.max()):.2f}], mean {float(w.mean()):.2f}"
                )

    # Generative spectro heads: set the per-(channel, freq) residual-std
    # buffer from the per-bin stats (persisted in the checkpoint). Falls back
    # to ones (no standardisation) if log_per_bin stats are unavailable.
    if args.spec_generative:
        sigma_pb_map = build_spec_per_bin_sigma(
            stats, model.diagnostics, train_ds.signal_configs,
        )
        if not sigma_pb_map:
            logger.warning(
                "--spec_generative: preprocessing_stats lacks 'log_per_bin' "
                "— flow heads use unit residual std (no per-bin scaling)."
            )
        for name in spectro_diag_names:
            head = model.diag_heads[name]
            if isinstance(head, SpectrogramFlowHead) and name in sigma_pb_map:
                head.set_sigma_pb(sigma_pb_map[name].to(device))
                logger.info(
                    f"Flow head [{name}]: per-bin residual std set, "
                    f"shape {tuple(sigma_pb_map[name].shape)}"
                )

    # Per-worker OMP_NUM_THREADS enforcement: with --cpus-per-task=7 in
    # the SLURM script and 6 DataLoader workers per rank, default torch
    # thread heuristics can oversubscribe (each worker spawning 7 OMP
    # threads → 42 threads competing for 7 cores). Match the value the
    # parent process saw via OMP_NUM_THREADS (set to 1 in
    # _frontier_settings.sh).
    def _worker_init(_worker_id: int) -> None:
        import os as _os
        n = int(_os.environ.get("OMP_NUM_THREADS", "1"))
        torch.set_num_threads(n)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        # TwoLevelSampler: shuffle file order per epoch, sequential
        # within each file. Keeps the per-worker LRU file-handle
        # cache (max_open_files=100) nearly always hitting.
        # RandomSampler across 7878 files gave ~1% hit rate and
        # spent ~10% of worker time on HDF5 file opens (observed
        # via py-spy on Stage 1 job 2719669).
        # DistributedTwoLevelSampler is the DDP-aware sibling: each
        # rank owns a fixed slice of the file list and iterates its
        # own files front-to-back, so the per-worker LRU stays warm
        # across epochs. PyTorch's DistributedSampler shards chunk
        # indices instead and was observed to push step time from
        # ~1 s to ~12 s under 2-GPU DDP on Stage 1.
        sampler=(
            DistributedTwoLevelSampler(
                train_ds,
                num_replicas=dm.world_size,
                rank=dm.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            if dm.distributed
            else TwoLevelSampler(train_ds, shuffle=True)
        ),
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
        # prefetch_factor=3 + val_num_workers=4 is the v9-validated config
        # at batch=8 (RAM ~68% steady, ~75% val-overlap peak — comfortable
        # under the 502 GB cap). Larger batch needs revisiting via the
        # empirical model: variable cost ≈ num_workers × prefetch ×
        # batch × ~1.3 GB.
        prefetch_factor=3,
        pin_memory=device.type == "cuda",
        # persistent_workers=False: stage-1 chained jobs 4581026/27/28
        # OOM'd at ~9h45m / ~5850 steps with persistent train workers —
        # slow per-worker leak (h5py metadata / PyTorch caches) fills
        # 502 GB per node. Tearing workers down at end-of-epoch
        # releases the state; spin-up cost (~5–10 s) is negligible vs
        # the multi-hour epoch wall time at K_max=10. Same fix applied
        # to train_e2e_stage1.py.
        persistent_workers=False,
        worker_init_fn=_worker_init,
    )
    # Val sampler mirrors the train sampler's DDP pattern: shard files
    # across ranks so each rank evaluates ~1/world_size of the val set,
    # then sums + counts are all_reduce'd inside validate() (see below).
    if dm.distributed:
        val_sampler = DistributedTwoLevelSampler(
            val_ds, num_replicas=dm.world_size, rank=dm.rank,
            shuffle=False, seed=args.seed, drop_last=True,
        )
    else:
        val_sampler = TwoLevelSampler(val_ds, shuffle=False)
    # Val loader memory budget (ported from Stage 1 OOM testing):
    # train workers stay alive during val (persistent=True on train) and
    # hold their prefetched batches. Capping val to
    # num_workers=min(4, args.num_workers), prefetch_factor=1, and
    # persistent_workers=False keeps the combined in-flight footprint
    # under the 502 GB node budget. Without this we OOM'd at 97% host
    # RAM on 2-node smokes when val workers spun up alongside the train
    # 6×2 prefetch pool.
    val_num_workers = min(4, args.num_workers)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=val_num_workers, collate_fn=collate_fn, drop_last=True,
        prefetch_factor=1,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=_worker_init,
    )

    # Pre-warm val NFS cache. The Stage 2 val 1 has been the recurring
    # failure point: val files are mostly disjoint from train, so when
    # the first val event fires after ~10h of training, a subset of
    # ranks hit cold NFS reads and lag the rest. monitored_barrier's
    # 2-min window can't tolerate this, and downstream collectives hang
    # on rank skew (see 4628766, 4634071, 4635053, 4641898). By having
    # each rank's main process briefly open every val file at startup,
    # the per-node NFS metadata+page cache is warm before val 1 fires.
    # ~1-2 min added to startup; eliminates the cold-val-cache hazard.
    import time as _time
    import h5py as _h5py
    if dm.distributed:
        n_files = len(val_ds.hdf5_paths)
        if dm.is_main:
            logger.info(f"Pre-warming val NFS cache ({n_files} files)...")
        t0 = _time.time()
        for path in val_ds.hdf5_paths:
            try:
                with _h5py.File(path, "r"):
                    pass  # open + close warms the per-node NFS cache
            except Exception:
                pass  # skip unreadable files; dataset handles them
        dm.barrier()  # all ranks finish warming before training starts
        if dm.is_main:
            logger.info(
                f"  val NFS cache pre-warm: {n_files} files in "
                f"{_time.time()-t0:.1f}s"
            )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        opt, args.max_steps, args.warmup_steps, args.min_lr
    )

    use_amp = (not args.no_amp) and device.type == "cuda"

    def amp_ctx_factory():
        if use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    logger.info(
        f"Starting Stage 2b — K_max={args.K_max} curriculum_steps="
        f"{args.curriculum_steps} lr={args.lr}→{args.min_lr} "
        f"warmup={args.warmup_steps} amp={'bf16' if use_amp else 'off'}"
    )

    # Initial head weights snapshot (monitored for stuck-ness).
    initial_head_norms = head_weight_l2(model)
    logger.info("Initial head weight L2:")
    for n, v in initial_head_norms.items():
        logger.info(f"  {n:<25s} {v:.4f}")

    best_val_loss = float("inf")
    best_step = 0
    resume_start_step = 0
    first_val_done = False
    if args.resume_checkpoint is not None and args.resume_checkpoint.exists():
        resume_ckpt = torch.load(
            args.resume_checkpoint, weights_only=False, map_location=device
        )
        # VideoOutputHead and SpectrogramOutputHead each gained a
        # zero-init residual refine_block for the patch-grid
        # checkerboard fix; old checkpoints lack those keys. Permit
        # them as missing — the module's zero init makes the load
        # bit-identical until training writes into the refine weights.
        resume_allowed_missing = tuple(
            f"diag_heads.{n}.{sub}"
            for n in (list(args.use_video) + list(args.use_spectro))
            for sub in ("refine_block.", "inv_stem.", "inv_stem_unembed.")
        ) + tuple(
            f"diag_tokenizers.{n}.fs_lin"
            for n in list(args.use_spectro)
        )
        load_state_dict_explicit(
            model, resume_ckpt["model_state_dict"],
            allowed_missing_prefixes=resume_allowed_missing,
        )
        if "optimizer_state_dict" in resume_ckpt:
            # Generic param-count guard: when the model gains/loses
            # params between submit and resume (e.g. the VideoOutputHead
            # `refine_block` added 2026-06-08 made the optimizer's
            # param-list longer than the saved state's), `load_state_dict`
            # raises an unrecoverable ValueError that kills the whole
            # chain. Detect the mismatch and fall back to fresh AdamW.
            saved_n = sum(
                len(g["params"])
                for g in resume_ckpt["optimizer_state_dict"]["param_groups"]
            )
            cur_n = sum(len(g["params"]) for g in opt.param_groups)
            if saved_n != cur_n:
                logger.warning(
                    f"Skipping optimizer state restore: saved had "
                    f"{saved_n} params, model has {cur_n}. AdamW "
                    f"starts fresh — momentum re-accumulates over the "
                    f"first few hundred steps. Loss may wobble briefly."
                )
            else:
                opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        resume_start_step = int(resume_ckpt.get("step", 0))
        best_val_loss = float(resume_ckpt.get(
            "best_val_loss", resume_ckpt.get("val_loss", float("inf"))
        ))
        best_step = int(resume_ckpt.get("best_step", resume_start_step))
        first_val_done = True
        logger.info(
            f"RESUMED from {args.resume_checkpoint.name}: starting at step "
            f"{resume_start_step}; best_val_loss={best_val_loss:.4f} at step "
            f"{best_step}"
        )
    step = resume_start_step
    running = 0.0
    running_count = 0
    prev_K = -1
    train_sampler = train_loader.sampler
    epoch_counter = 0
    if dm.distributed and hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(epoch_counter)
    train_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch_counter += 1
            if dm.distributed and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch_counter)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        K = current_K(step, args.curriculum_steps, args.K_max)
        if K != prev_K:
            logger.info(f"Curriculum: step {step} → K = {K}")
            prev_K = K

        opt.zero_grad()
        with amp_ctx_factory():
            loss, per_step_per_mod = rollout_forward_loss_delta(
                rollout, batch, diagnostic_names, actuator_names,
                k_steps=K, chunk_duration_s=args.chunk_duration_s, device=device,
                mae_weight=args.mae_weight, cos_weight=args.cos_weight,
                mag_weight=args.mag_weight, min_disp_norm=args.min_disp_norm,
                video_diag_names=video_diag_names,
                video_n_frames=video_n_frames,
                spectro_diag_names=spectro_diag_names,
                grad_checkpoint_every=args.grad_checkpoint_every,
                video_smoothness_weight=args.video_smoothness_weight,
                spec_pb_weights=spec_pb_weights,
                keep_displacement_for_flow=args.spec_gen_keep_displacement,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        scheduler.step()
        running += loss.item()
        running_count += 1
        step += 1

        if step % args.log_every == 0:
            avg = running / running_count
            lr_now = opt.param_groups[0]["lr"]
            # Compact training log: average direction_cos across steps/modalities.
            all_dir_cos = [
                per_step_per_mod[k][n]["dir_cos"]
                for k in range(K)
                for n in diagnostic_names
                if not (per_step_per_mod[k][n]["dir_cos"] != per_step_per_mod[k][n]["dir_cos"])  # not nan
            ]
            mean_dir_cos = sum(all_dir_cos) / max(1, len(all_dir_cos))
            logger.info(
                f"step {step}/{args.max_steps}  K={K}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}  mean_dir_cos={mean_dir_cos:+.4f}"
            )
            # Host-RAM trajectory (psutil): chained jobs OOM'd reproducibly
            # with no sampler logs. Inline reading gives per-log-step RAM%
            # in the .err file so we can diagnose without re-submitting.
            vm = psutil.virtual_memory()
            logger.info(
                f"  host_ram used={vm.used/1e9:.1f}GB "
                f"available={vm.available/1e9:.1f}GB "
                f"percent={vm.percent:.1f}%"
            )
            running = 0.0
            running_count = 0

        if step % args.val_every == 0 or step == args.max_steps:
            metrics = validate(
                _core(rollout), val_loader, device,
                diagnostic_names, actuator_names,
                chunk_duration_s=args.chunk_duration_s,
                K_max=args.K_max,
                min_disp_norm=args.min_disp_norm,
                max_batches=args.val_max_batches,
                video_diag_names=video_diag_names,
                video_n_frames=video_n_frames,
                spectro_diag_names=spectro_diag_names,
                step=step,
                ckpt_dir=args.checkpoint_dir,
            )
            highlight = sorted({0, min(4, args.K_max - 1), args.K_max - 1})
            hdr = (
                "FIRST VALIDATION — direction_cos is the Stage 2b success metric"
                if not first_val_done
                else f"Validation @ step {step}"
            )
            logger.info("")
            logger.info(
                f"{hdr} — per-modality metrics at steps "
                + ", ".join(str(k + 1) for k in highlight) + ":"
            )
            for name in diagnostic_names:
                parts = []
                for k in highlight:
                    m = metrics[k][name]
                    parts.append(
                        f"k{k + 1}: mae={m['model_mae']:.3f} "
                        f"dcos={m['dir_cos']:+.3f} "
                        f"mrat={m['mag_ratio']:.2f}"
                    )
                logger.info(f"  {name:<25s} " + " | ".join(parts))
            val_loss = sum(
                metrics[k][name]["model_mae"]
                for k in range(args.K_max)
                for name in diagnostic_names
            )
            # Direction-cos summary line
            all_dc = [
                metrics[k][name]["dir_cos"]
                for k in range(args.K_max)
                for name in diagnostic_names
                if metrics[k][name]["dir_cos"] == metrics[k][name]["dir_cos"]
            ]
            mean_dir_cos_val = sum(all_dc) / max(1, len(all_dc))
            logger.info(
                f"  [sum model MAE] {val_loss:.4f}   "
                f"[mean direction_cos across K×modalities] {mean_dir_cos_val:+.4f}"
            )
            # Head weight monitoring
            cur_head_norms = head_weight_l2(model)
            # head_weight_l2 only reports TS head norms (slow_ts/fast_ts);
            # video heads have a different shape and are skipped there.
            # Iterate over what the function actually returned.
            head_delta = max(
                abs(cur_head_norms[n] - initial_head_norms[n])
                for n in initial_head_norms
            )
            logger.info(
                f"  [head-weight L2 max |Δ| from init] {head_delta:.5f}"
            )
            if step >= 5000 and head_delta < 1e-4:
                logger.warning(
                    "  Head weights have not moved in 5k+ steps — heads may be "
                    "stuck in a flat region. Consider a head-only LR warmup."
                )

            first_val_done = True
            # Collapse-aware selection: penalise spectro modalities whose
            # temporal-variance ratio is below 1 (mean-collapsed) so a low-MAE
            # collapse can't win best.pt. Default off → plain sum(MAE).
            sel_loss = val_loss
            if args.collapse_aware_best:
                tvr_pen = sum(
                    max(0.0, 1.0 - metrics[k][name]["tvr"])
                    for k in range(args.K_max)
                    for name in diagnostic_names
                    if not math.isnan(metrics[k][name].get("tvr", float("nan")))
                )
                sel_loss = val_loss + args.collapse_aware_lambda * tvr_pen
                logger.info(
                    f"  [collapse-aware] sel_loss={sel_loss:.4f} "
                    f"(tvr_penalty={tvr_pen:.4f}, "
                    f"lambda={args.collapse_aware_lambda})"
                )
            is_new_best = sel_loss < best_val_loss
            if is_new_best:
                best_val_loss = sel_loss
                best_step = step
            if dm.is_main:
                ckpt_state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "mean_dir_cos": mean_dir_cos_val,
                    "metrics": metrics,
                    "diagnostics": [asdict(c) for c in diagnostics],
                    "actuators": [asdict(c) for c in actuators],
                    "args": vars(args),
                }
                latest_path = args.checkpoint_dir / "e2e_stage2_delta_latest.pt"
                torch.save(ckpt_state, latest_path)
                if is_new_best:
                    best_path = args.checkpoint_dir / "e2e_stage2_delta_best.pt"
                    torch.save(ckpt_state, best_path)
                    logger.info(
                        f"  ✓ new best val_loss={val_loss:.4f}  saved {best_path.name}"
                    )
            dm.barrier()

    if dm.is_main:
        final_path = args.checkpoint_dir / "e2e_stage2_delta_final.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "diagnostics": [asdict(c) for c in diagnostics],
                "actuators": [asdict(c) for c in actuators],
                "args": vars(args),
            },
            final_path,
        )
        logger.info(
            f"Saved final checkpoint: {final_path}. "
            f"Best val_loss={best_val_loss:.4f} at step {best_step}."
        )
    dm.barrier()


if __name__ == "__main__":
    main()