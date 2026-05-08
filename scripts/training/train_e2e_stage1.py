"""Stage 1 single-step pretraining for the end-to-end foundation model.

Implements ``ResearchPlan.MD`` §4.1: the backbone learns to predict the next
50 ms of every diagnostic modality, conditioned on actuator commands for
that step.

Key data-pipeline choices (all configurable via CLI):
  - ``chunk_duration_s = 0.05`` (input 50 ms window)
  - ``prediction_horizon_s = 0.05`` (target 50 ms window)
  - ``step_size_s = 0.01`` (10 ms stride between chunks → diverse starts)
  - ``warmup_s = 1.0`` (skip first 1 s of each shot)
  - ``prediction_mode = True`` (dataset emits ``{inputs, targets}`` dicts;
    diagnostics live in both lists so we get the input and target halves;
    actuators live in ``target_signals`` only so the dataset gives us the
    actuator commands driving the step-1 transition)

Debug smoke test::

    pixi run python scripts/training/train_e2e_stage1.py \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --train_shots_yaml src/tokamak_foundation_model/data/config/shot_list/train_debug.yaml \
        --max_files 4 --max_steps 50 --batch_size 4 --num_workers 2 \
        --checkpoint_dir runs/e2e_stage1_debug
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
    TwoLevelSampler,
    filter_video_present_files,
)
from tokamak_foundation_model.e2e.checkpoint import load_state_dict_explicit
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.utils.distributed import DistributedManager

logger = logging.getLogger("e2e_stage1")


def _core(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module for DDP-wrapped or plain models."""
    return model.module if hasattr(model, "module") else model


# ── Modality inventory ───────────────────────────────────────────────────
#
# Channel counts match ``TokamakH5Dataset.SIGNAL_CONFIGS`` in
# ``src/tokamak_foundation_model/data/data_loader.py``. Filterscopes is
# downselected from 104 → 8 inside the dataset
# (``channels_to_use=slice(0, 8)``).

SLOW_TS_MODALITIES: List[Tuple[str, int]] = [
    ("ts_core_density", 44),
    ("ts_core_temp", 44),
    ("ts_tangential_density", 10),
    ("ts_tangential_temp", 10),
    ("cer_ti", 48),
    ("cer_rot", 48),
    ("mse", 69),
]
FAST_TS_MODALITIES: List[Tuple[str, int, int]] = [
    # (name, n_channels, patch_size)
    ("filterscopes", 8, 50),
]
ACTUATOR_MODALITIES: List[Tuple[str, int]] = [
    ("pin", 8),
    ("beam_voltage", 8),
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


# Per-camera video modality registry. Each entry is
# ``(name, n_channels, n_frames, (height, width), (T_p, H_p, W_p))``.
# Only included when the user passes ``--use_video <name> [<name> ...]``;
# otherwise behaviour is byte-identical to Phase A pre-Step-5 (G2/G3).
VIDEO_MODALITIES: List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int, int]]] = [
    ("tangtv", 2, 3, (120, 360), (3, 12, 12)),
]

# Per-modality spectrogram registry. Each entry is
# ``(name, n_channels, (F_p, T_p))``. STFT shape is fixed by the data
# loader (n_fft=1024, hop=256, fs=500 kHz) so freq_bins=512, time_frames=98
# for the canonical 50 ms window. Only included when the user passes
# ``--use_spectro <name> [<name> ...]``; empty default keeps Phase A
# byte-identical (G2/G3).
SPECTRO_FREQ_BINS = 512
SPECTRO_TIME_FRAMES = 98
SPECTROGRAM_MODALITIES: List[Tuple[str, int, Tuple[int, int]]] = [
    ("ece", 40, (32, 8)),
    ("co2", 4, (64, 8)),
    ("bes", 16, (32, 8)),
]


def build_configs(
    chunk_duration_s: float,
    use_video: Optional[List[str]] = None,
    use_spectro: Optional[List[str]] = None,
) -> Tuple[List[DiagnosticConfig], List[ActuatorConfig]]:
    slow_samples = round(chunk_duration_s * SLOW_FS)
    fast_samples = round(chunk_duration_s * FAST_FS)
    diagnostics: List[DiagnosticConfig] = []
    for name, n_channels in SLOW_TS_MODALITIES:
        diagnostics.append(
            DiagnosticConfig(name, "slow_ts", n_channels, slow_samples)
        )
    for name, n_channels, patch in FAST_TS_MODALITIES:
        diagnostics.append(
            DiagnosticConfig(name, "fast_ts", n_channels, fast_samples, patch)
        )
    # Token ordering inside the diagnostic prefix:
    #   [slow_ts | fast_ts | spectrogram | video | actuators]
    # Spectrograms go before video so adding either does not perturb the
    # other's layout in the backbone token sequence.
    if use_spectro:
        registry = {entry[0]: entry for entry in SPECTROGRAM_MODALITIES}
        for spec_name in use_spectro:
            if spec_name not in registry:
                raise SystemExit(
                    f"--use_spectro {spec_name!r}: unknown modality; known: "
                    f"{sorted(registry.keys())}"
                )
            (_, n_channels, patch_size) = registry[spec_name]
            diagnostics.append(
                DiagnosticConfig(
                    name=spec_name,
                    kind="spectrogram",
                    n_channels=n_channels,
                    window_samples=SPECTRO_TIME_FRAMES,
                    freq_bins=SPECTRO_FREQ_BINS,
                    spectrogram_patch_size=patch_size,
                )
            )
    # Video diagnostics go in the diagnostic prefix AFTER all TS configs and
    # spectrograms, BEFORE the actuators, so the ``rollout.py`` slice
    # ``[:, :n_diag_tokens]`` keeps propagating diagnostic tokens contiguously.
    if use_video:
        registry = {entry[0]: entry for entry in VIDEO_MODALITIES}
        for cam_name in use_video:
            if cam_name not in registry:
                raise SystemExit(
                    f"--use_video {cam_name!r}: unknown camera; known: "
                    f"{sorted(registry.keys())}"
                )
            (_, n_channels, n_frames, (height, width), patch_size) = registry[cam_name]
            diagnostics.append(
                DiagnosticConfig(
                    name=cam_name,
                    kind="video",
                    n_channels=n_channels,
                    window_samples=n_frames,
                    height=height,
                    width=width,
                    video_patch_size=patch_size,
                )
            )
    # n_tokens=5 at 10 kHz × 50 ms → patch_size=100 (= 10 ms of history per
    # token). n_tokens=3 from the plan table doesn't divide 500; 5 is the
    # nearest divisor ≥ 3 that covers the window cleanly.
    actuators: List[ActuatorConfig] = [
        ActuatorConfig(name, n_channels, fast_samples, n_tokens=5)
        for name, n_channels in ACTUATOR_MODALITIES
    ]
    return diagnostics, actuators


# ── Shot-list resolution ─────────────────────────────────────────────────


def _load_shot_yaml(path: Path) -> List[int]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, dict):
        shots = data.get("shots", [])
    else:
        shots = data or []
    return [int(s) for s in shots]


def _shot_to_h5(data_dir: Path, shot: int) -> Path:
    return data_dir / f"{shot}_processed.h5"


def resolve_shot_files(
    data_dir: Path,
    train_shots_yaml: Optional[Path],
    val_shots_yaml: Optional[Path],
    max_files: Optional[int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """Return ``(train_files, val_files)`` as existing HDF5 paths.

    If ``train_shots_yaml`` is given, use it for training. Same for
    ``val_shots_yaml``. If only training is given and ``val_shots_yaml`` is
    not, split off ``val_fraction`` of the training files for validation.
    If neither is given, glob the directory and random-split.
    """
    rng = random.Random(seed)

    def _existing(paths: List[Path]) -> List[Path]:
        kept = [p for p in paths if p.exists()]
        missing = len(paths) - len(kept)
        if missing:
            logger.warning(f"{missing} shots from YAML not found in {data_dir}")
        return kept

    if train_shots_yaml is not None:
        train_shots = _load_shot_yaml(train_shots_yaml)
        train_files = _existing([_shot_to_h5(data_dir, s) for s in train_shots])
        if val_shots_yaml is not None:
            val_shots = _load_shot_yaml(val_shots_yaml)
            val_files = _existing([_shot_to_h5(data_dir, s) for s in val_shots])
        else:
            rng.shuffle(train_files)
            n_val = max(1, int(val_fraction * len(train_files)))
            val_files = train_files[:n_val]
            train_files = train_files[n_val:]
    else:
        all_files = sorted(data_dir.glob("*_processed.h5"))
        rng.shuffle(all_files)
        n = len(all_files)
        n_val = max(1, int(val_fraction * n))
        val_files = all_files[:n_val]
        train_files = all_files[n_val:]

    if max_files is not None:
        train_files = train_files[:max_files]
        val_files = val_files[: max(1, max_files // 4)]
    return train_files, val_files


# ── Dataset construction ─────────────────────────────────────────────────


def build_datasets(
    data_dir: Path,
    train_files: List[Path],
    val_files: List[Path],
    preprocessing_stats: dict,
    chunk_duration_s: float,
    prediction_horizon_s: float,
    step_size_s: float,
    warmup_s: float,
    diagnostic_names: List[str],
    actuator_names: List[str],
    lengths_cache_dir: Path,
) -> Tuple[TokamakMultiFileDataset, TokamakMultiFileDataset]:
    """Construct Stage 1 train + val datasets.

    Diagnostics are in both ``input_signals`` and ``target_signals`` so the
    loader returns input (t) and target (t+50 ms) halves. Actuators are in
    ``target_signals`` only so we receive the actuator commands driving
    the step-1 transition.
    """
    input_signals = diagnostic_names
    target_signals = diagnostic_names + actuator_names

    lengths_cache_dir.mkdir(parents=True, exist_ok=True)
    shared = dict(
        chunk_duration_s=chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=prediction_horizon_s,
        step_size_s=step_size_s,
        warmup_s=warmup_s,
        preprocessing_stats=preprocessing_stats,
        input_signals=input_signals,
        target_signals=target_signals,
        max_open_files=1024,
    )
    train_ds = TokamakMultiFileDataset(
        train_files,
        lengths_cache_path=lengths_cache_dir / "lengths_e2e_stage1_train.pt",
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=lengths_cache_dir / "lengths_e2e_stage1_val.pt",
        **shared,
    )
    return train_ds, val_ds


# ── Loss ─────────────────────────────────────────────────────────────────


def _clean_and_mask(
    tensor: torch.Tensor, existing_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace NaN/Inf with 0 and combine with an optional upstream mask.

    Returns ``(cleaned_tensor, mask)`` where mask is ``1`` for positions that
    are both finite in ``tensor`` and valid under ``existing_mask``. The
    data loader only zero-fills missing values for modalities with
    ``zero_is_missing=True`` or that carry an explicit ``nan_mask``;
    ``mse`` / ``cer_*`` have neither and arrive with NaN entries in some
    shots, so the loop applies this guard on every tensor it touches.
    """
    finite = torch.isfinite(tensor)
    cleaned = torch.where(finite, tensor, torch.zeros_like(tensor))
    mask = finite.float()
    if existing_mask is not None:
        mask = mask * existing_mask
    return cleaned, mask


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean absolute error with a combined NaN + upstream mask."""
    cleaned_pred, pred_mask = _clean_and_mask(pred, None)
    cleaned_target, target_mask = _clean_and_mask(target, mask)
    combined = pred_mask * target_mask
    diff = (cleaned_pred - cleaned_target).abs() * combined
    return diff.sum() / combined.sum().clamp_min(1.0)


def _video_standardize_per_bc(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(B, C) z-score over (T, H, W) for a video tensor.

    Returns ``(x_norm, mu, sd)`` so the same statistics can be applied
    to the target half-window without re-computing.

    Why this is needed: tangtv targets are raw pixel values
    (mean ~50, std ~17, range 0..235). With AdamW at ``lr=1e-4`` the
    output head's last-layer bias would need ~5×10⁵ steps to learn a
    constant offset of 50; the whole training is 3.36×10⁵. Without
    standardization the video loss simply does not move and TS losses
    drift only because of batch-content variability. The standalone AE
    (``train_video_ae.py``) hit exactly this and was rescued with the
    identical operation; until precomputed per-channel stats land in
    ``preprocessing_stats.pt`` we apply the same fix in-line here.

    ``sd.clamp(min=1.0)`` keeps off-channels (NaN-filled to zeros, std
    exactly 0) finite — they remain at zero post-standardize, and the
    channel-mask gate excludes them from the loss anyway.
    """
    mu = x.mean(dim=(2, 3, 4), keepdim=True)
    sd = x.std(dim=(2, 3, 4), keepdim=True).clamp(min=1.0)
    return (x - mu) / sd, mu, sd


def _video_loss_gate(
    cfg: DiagnosticConfig, batch: Dict, device: torch.device
) -> torch.Tensor:
    """Per-element loss gate for a video modality.

    Combines the per-batch camera-availability scalar
    ``f"{name}_valid"`` with the per-channel availability mask
    ``f"{name}_channel_mask"``. Returned shape ``(B, C, 1, 1, 1)``
    broadcasts cleanly to ``(B, C, T, H, W)`` — matches both target
    and (post-permute) prediction shapes for video.
    """
    name = cfg.name
    chan_mask = batch["targets"][f"{name}_channel_mask"].to(
        device, non_blocking=True
    ).float()                                        # (B, C)
    valid = batch["targets"][f"{name}_valid"].to(
        device, non_blocking=True
    ).float()                                        # (B,)
    return (
        valid[:, None, None, None, None]
        * chan_mask[:, :, None, None, None]
    )                                                # (B, C, 1, 1, 1)


def _spectro_loss_gate(
    cfg: DiagnosticConfig, batch: Dict, device: torch.device
) -> torch.Tensor:
    """Per-element loss gate for a spectrogram modality.

    Spectrograms have no per-channel runtime availability mask
    (campaign-dependent dead channels are tolerated; ``log_standardize``
    flattens amplitude differences). The gate is just the per-batch
    presence scalar broadcast over ``(B, C, F, T)``.
    """
    valid = batch["targets"][f"{cfg.name}_valid"].to(
        device, non_blocking=True
    ).float()                                        # (B,)
    return valid[:, None, None, None]                # (B, 1, 1, 1)


def forward_batch(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],  # predictions
    Dict[str, torch.Tensor],  # diag_inputs (cleaned)
    Dict[str, torch.Tensor],  # targets (raw; loss/metrics handle NaN)
    Dict[str, Optional[torch.Tensor]],  # existing per-modality target masks
]:
    """Forward pass with NaN-cleaned inputs; return predictions + tensors needed for metrics."""
    diag_inputs: Dict[str, torch.Tensor] = {}
    # Per-(B, C) z-score statistics for video and spectrogram modalities.
    # Computed from the *input* window and reused for the corresponding
    # target so prediction and ground truth live in the same normalized
    # frame. Empty when no such diagnostics are configured.
    norm_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for cfg in _core(model).diagnostics:
        raw = batch["inputs"][cfg.name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if cfg.kind == "video":
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            norm_stats[cfg.name] = (mu, sd)
        elif cfg.kind == "spectrogram":
            assert cfg.spectrogram_patch_size is not None
            _, T_p = cfg.spectrogram_patch_size
            trunc_t = (cfg.window_samples // T_p) * T_p
            cleaned = cleaned[..., :trunc_t]
        diag_inputs[cfg.name] = cleaned
        if cfg.kind in ("video", "spectrogram"):
            valid_key = f"{cfg.name}_valid"
            if valid_key in batch["inputs"]:
                diag_inputs[valid_key] = batch["inputs"][valid_key].to(
                    device, non_blocking=True
                )
    act_inputs: Dict[str, torch.Tensor] = {}
    for cfg in _core(model).actuators:
        raw = batch["targets"][cfg.name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        act_inputs[cfg.name] = cleaned

    batch_size = next(iter(diag_inputs.values())).shape[0]
    step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    time_offset = torch.zeros(batch_size, device=device)

    predictions = model(diag_inputs, act_inputs, step_idx, time_offset)

    # Normalise video predictions to (B, C, T, H, W) — VideoOutputHead
    # emits (B, T, C, H, W) but the data loader produces video targets
    # in (B, C, T, H, W) order (matching the (B, C, T) TS convention).
    # Doing the permute here means downstream loss / metric code can
    # treat all modalities under a single shape contract.
    for cfg in _core(model).diagnostics:
        if cfg.kind == "video":
            predictions[cfg.name] = predictions[cfg.name].permute(0, 2, 1, 3, 4)

    targets: Dict[str, torch.Tensor] = {}
    masks: Dict[str, Optional[torch.Tensor]] = {}
    for cfg in _core(model).diagnostics:
        targets[cfg.name] = batch["targets"][cfg.name].to(device, non_blocking=True).float()
        if cfg.kind == "video":
            mu, sd = norm_stats[cfg.name]
            targets[cfg.name] = (targets[cfg.name] - mu) / sd
            masks[cfg.name] = _video_loss_gate(cfg, batch, device)
        elif cfg.kind == "spectrogram":
            assert cfg.spectrogram_patch_size is not None
            _, T_p = cfg.spectrogram_patch_size
            trunc_t = (cfg.window_samples // T_p) * T_p
            targets[cfg.name] = targets[cfg.name][..., :trunc_t]
            masks[cfg.name] = _spectro_loss_gate(cfg, batch, device)
        else:
            mask_key = f"{cfg.name}_mask"
            masks[cfg.name] = (
                batch["targets"][mask_key].to(device, non_blocking=True).float()
                if mask_key in batch["targets"]
                else None
            )
    return predictions, diag_inputs, targets, masks


def compute_step_loss(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one forward pass and return ``(total_loss, per-modality MAE dict)``."""
    predictions, _, targets, masks = forward_batch(model, batch, device)
    per_modality: Dict[str, float] = {}
    total_loss = torch.zeros((), device=device)
    for cfg in _core(model).diagnostics:
        loss = masked_mae(predictions[cfg.name], targets[cfg.name], masks[cfg.name])
        per_modality[cfg.name] = loss.item()
        total_loss = total_loss + loss
    return total_loss, per_modality


@torch.no_grad()
def copy_baseline_mae(
    batch: Dict,
    diagnostics: List[DiagnosticConfig],
    device: torch.device,
) -> Dict[str, float]:
    """MAE of the trivial ``prediction = input`` baseline (target-sized).

    For video and spectrogram modalities the same per-(B, C) z-score
    applied during training is applied here too, so the copy-baseline
    number is in the same normalized space as the model's training
    MAE and they can be compared directly.
    """
    out: Dict[str, float] = {}
    for cfg in diagnostics:
        name = cfg.name
        pred = batch["inputs"][name].to(device).float()
        target = batch["targets"][name].to(device).float()
        if cfg.kind == "video":
            pred, mu, sd = _video_standardize_per_bc(pred)
            target = (target - mu) / sd
            mask = _video_loss_gate(cfg, batch, device)
        elif cfg.kind == "spectrogram":
            # No per-batch z-score; data loader's log_standardize is
            # the only normalization (see forward_batch comment).
            # Match the time-axis truncation applied in forward_batch
            # so the copy baseline lives in the same shape as the
            # model's predictions.
            assert cfg.spectrogram_patch_size is not None
            _, T_p = cfg.spectrogram_patch_size
            trunc_t = (cfg.window_samples // T_p) * T_p
            pred = pred[..., :trunc_t]
            target = target[..., :trunc_t]
            mask = _spectro_loss_gate(cfg, batch, device)
        else:
            mask_key = f"{name}_mask"
            mask = (
                batch["targets"][mask_key].to(device).float()
                if mask_key in batch["targets"]
                else None
            )
        out[name] = masked_mae(pred, target, mask).item()
    return out


# ── Validation ───────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    model: E2EFoundationModel,
    loader: DataLoader,
    device: torch.device,
    diagnostic_names: List[str],
    max_batches: Optional[int] = None,
    use_amp: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Return per-modality validation metrics.

    ``out[name]`` has keys ``model_mae``, ``copy_mae``, ``pred_delta``,
    ``tgt_delta``, ``delta_ratio``.

    ``pred_delta`` and ``tgt_delta`` are displacement-magnitude metrics
    (``ResearchPlan.MD`` §7): ``||pred - input||`` and ``||target - input||``
    respectively, both masked. A model that copies its input has
    ``pred_delta ≈ 0``; a model predicting the true dynamics has
    ``delta_ratio = pred_delta / tgt_delta ∈ [0.8, 1.2]``.
    """
    model.eval()
    keys = ("model_mae", "copy_mae", "pred_delta", "tgt_delta")
    sums = {k: {n: 0.0 for n in diagnostic_names} for k in keys}
    n_batches = 0

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with amp_ctx:
            predictions, diag_inputs, targets, masks = forward_batch(
                model, batch, device
            )
        copy_mod = copy_baseline_mae(batch, model.diagnostics, device)
        for name in diagnostic_names:
            pred = predictions[name]
            inp = diag_inputs[name]
            tgt = targets[name]
            existing = masks[name]

            cleaned_pred, mask_p = _clean_and_mask(pred, None)
            cleaned_tgt, mask_t = _clean_and_mask(tgt, existing)
            combined = mask_p * mask_t
            denom = combined.sum().clamp_min(1.0)

            model_mae_v = (
                (cleaned_pred - cleaned_tgt).abs() * combined
            ).sum() / denom
            pred_delta = (
                (cleaned_pred - inp).abs() * combined
            ).sum() / denom
            tgt_delta = (
                (cleaned_tgt - inp).abs() * combined
            ).sum() / denom

            sums["model_mae"][name] += model_mae_v.item()
            sums["copy_mae"][name] += copy_mod[name]
            sums["pred_delta"][name] += pred_delta.item()
            sums["tgt_delta"][name] += tgt_delta.item()
        n_batches += 1

    denom = max(n_batches, 1)
    model.train()
    out: Dict[str, Dict[str, float]] = {}
    for name in diagnostic_names:
        model_mae = sums["model_mae"][name] / denom
        copy_mae = sums["copy_mae"][name] / denom
        pred_d = sums["pred_delta"][name] / denom
        tgt_d = sums["tgt_delta"][name] / denom
        ratio = pred_d / tgt_d if tgt_d > 1e-8 else float("nan")
        out[name] = {
            "model_mae": model_mae,
            "copy_mae": copy_mae,
            "pred_delta": pred_d,
            "tgt_delta": tgt_d,
            "delta_ratio": ratio,
        }
    return out


def _build_scheduler(
    opt: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup 1e-3·base_lr → base_lr over ``warmup_steps``, then cosine
    decay to ``min_lr`` over the remaining steps.
    """
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


# ── Warm-start module freeze ─────────────────────────────────────────────


_TS_KINDS = ("slow_ts", "fast_ts")


def _module_param_iter(
    model: E2EFoundationModel,
    *,
    freeze_ts: bool,
    freeze_video: bool,
    freeze_spectro: bool,
    freeze_backbone: bool,
) -> List[Tuple[str, torch.nn.Parameter]]:
    """Return ``[(label, param), ...]`` for every parameter the caller
    asked to freeze. ``label`` is a short string identifying the source
    (e.g. ``"ts:ts_core_density"``, ``"backbone"``) for log output.

    No-op categories return no params, so passing ``freeze_video=True``
    on a model without video modules is harmless.
    """
    out: List[Tuple[str, torch.nn.Parameter]] = []
    for cfg in model.diagnostics:
        is_ts = cfg.kind in _TS_KINDS
        if is_ts and freeze_ts:
            label = f"ts:{cfg.name}"
        elif cfg.kind == "video" and freeze_video:
            label = f"video:{cfg.name}"
        elif cfg.kind == "spectrogram" and freeze_spectro:
            label = f"spectro:{cfg.name}"
        else:
            continue
        for p in model.diag_tokenizers[cfg.name].parameters():
            out.append((label, p))
        for p in model.diag_heads[cfg.name].parameters():
            out.append((label, p))
    if freeze_backbone:
        for p in model.backbone.parameters():
            out.append(("backbone", p))
    return out


def _apply_module_freeze(
    model: E2EFoundationModel,
    *,
    freeze_ts: bool,
    freeze_video: bool,
    freeze_spectro: bool,
    freeze_backbone: bool,
) -> List[str]:
    """Freeze the per-module parameters indicated by the four flags.

    Each flag is independent; pass ``True`` for any subset. Actuator
    tokenizers stay trainable in all cases (they are tiny and
    inseparable from the dynamics the model learns).

    Returns the deduplicated list of frozen labels (for log output).
    """
    pairs = _module_param_iter(
        model,
        freeze_ts=freeze_ts,
        freeze_video=freeze_video,
        freeze_spectro=freeze_spectro,
        freeze_backbone=freeze_backbone,
    )
    seen_labels: List[str] = []
    seen_params: set[int] = set()
    for label, p in pairs:
        if id(p) in seen_params:
            continue
        seen_params.add(id(p))
        p.requires_grad = False
        if label not in seen_labels:
            seen_labels.append(label)
    return seen_labels


def _release_module_freeze(
    model: E2EFoundationModel,
    *,
    freeze_ts: bool,
    freeze_video: bool,
    freeze_spectro: bool,
    freeze_backbone: bool,
) -> int:
    """Release the freeze applied by :func:`_apply_module_freeze` with
    the same flags; return the number of parameter tensors unfrozen
    (for log output)."""
    pairs = _module_param_iter(
        model,
        freeze_ts=freeze_ts,
        freeze_video=freeze_video,
        freeze_spectro=freeze_spectro,
        freeze_backbone=freeze_backbone,
    )
    seen_params: set[int] = set()
    n_unfrozen = 0
    for _, p in pairs:
        if id(p) in seen_params:
            continue
        seen_params.add(id(p))
        if not p.requires_grad:
            n_unfrozen += 1
        p.requires_grad = True
    return n_unfrozen


# ── Training driver ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--train_shots_yaml", type=Path, default=None)
    parser.add_argument("--val_shots_yaml", type=Path, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Data windowing
    parser.add_argument("--chunk_duration_s", type=float, default=0.05)
    parser.add_argument("--prediction_horizon_s", type=float, default=0.05)
    parser.add_argument("--step_size_s", type=float, default=0.01)
    parser.add_argument("--warmup_s", type=float, default=1.0)

    # Model (debug-scale defaults per user)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Optim
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--val_max_batches", type=int, default=20)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--resume_checkpoint", type=Path, default=None,
        help="Resume from a *_latest.pt or *_final.pt, restoring model + "
        "optimizer + scheduler + step + best_val_loss. Overrides the "
        "fresh-init path. Intended for SLURM resubmission after the 24 h wall.",
    )
    parser.add_argument(
        "--init_checkpoint", type=Path, default=None,
        help="Load model weights from a checkpoint at the start of "
        "training, but do NOT restore optimizer / scheduler / step. "
        "Used by Phase C Stage 1 to warm-start from Phase A Stage 1 "
        "best (TS+actuator weights) while leaving any video modules "
        "freshly initialised. Ignored when --resume_checkpoint is "
        "provided AND the resume file exists.",
    )
    parser.add_argument(
        "--use_video", nargs="*", default=[],
        choices=[entry[0] for entry in VIDEO_MODALITIES],
        help="Camera names to include as video modalities.",
    )
    parser.add_argument(
        "--use_spectro", nargs="*", default=[],
        choices=[entry[0] for entry in SPECTROGRAM_MODALITIES],
        help="Spectrogram modality names to include.",
    )
    parser.add_argument(
        "--freeze_ts_steps", type=int, default=0,
        help="Warm-start: freeze TS tokenizers + heads for N steps.",
    )
    parser.add_argument(
        "--freeze_video_steps", type=int, default=0,
        help="Warm-start: freeze video tokenizers + heads for N steps.",
    )
    parser.add_argument(
        "--freeze_spectro_steps", type=int, default=0,
        help="Warm-start: freeze spectrogram tokenizers + heads for N steps.",
    )
    parser.add_argument(
        "--freeze_backbone_steps", type=int, default=0,
        help="Warm-start: freeze the shared backbone for N steps.",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable bf16 mixed precision (default: AMP on when CUDA).",
    )
    args = parser.parse_args()

    dm = DistributedManager()

    logging.basicConfig(
        level=logging.INFO if dm.is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{dm.rank}] %(message)s",
    )

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
    dm.barrier()

    # ── Resolve files + stats ────────────────────────────────────────────
    train_files, val_files = resolve_shot_files(
        args.data_dir,
        args.train_shots_yaml,
        args.val_shots_yaml,
        args.max_files,
        args.val_fraction,
        args.seed,
    )
    logger.info(f"Files — train: {len(train_files)}  val: {len(val_files)}")
    if not train_files or not val_files:
        raise SystemExit("No train or val files resolved; aborting.")

    # Phase C: when training with video, filter the file lists to shots
    # whose HDF5 actually contains non-empty data for the requested
    # camera(s). Without this, TwoLevelSampler's "one-batch-per-file"
    # property combined with ~45% of shots lacking tangtv (Step 0) means
    # roughly half of all batches give zero gradient signal for the
    # video path. Per-modality validity masking still works at the
    # sample level for batches that mix tangtv-present with
    # tangtv-absent samples — but TwoLevelSampler doesn't mix.
    # No-op when args.use_video is empty (G2/G3 stay byte-identical).
    if args.use_video:
        n_train_before = len(train_files)
        n_val_before = len(val_files)
        train_files = filter_video_present_files(
            train_files,
            args.use_video,
            cache_path=args.checkpoint_dir / "video_present_train.pt",
        )
        val_files = filter_video_present_files(
            val_files,
            args.use_video,
            cache_path=args.checkpoint_dir / "video_present_val.pt",
        )
        logger.info(
            f"Video-presence filter ({args.use_video}): "
            f"train {n_train_before} -> {len(train_files)} "
            f"({100 * len(train_files) / max(n_train_before, 1):.1f}%); "
            f"val {n_val_before} -> {len(val_files)} "
            f"({100 * len(val_files) / max(n_val_before, 1):.1f}%)"
        )
        if not train_files or not val_files:
            raise SystemExit(
                "Video-presence filter dropped every file. "
                f"Check that {args.use_video} HDF5 groups exist + are "
                "non-empty in the data dir."
            )

    stats = torch.load(args.stats_path, weights_only=False)

    # ── Model + configs ─────────────────────────────────────────────────
    diagnostics, actuators = build_configs(
        args.chunk_duration_s,
        use_video=args.use_video,
        use_spectro=args.use_spectro,
    )
    diagnostic_names = [c.name for c in diagnostics]
    actuator_names = [c.name for c in actuators]
    logger.info(
        f"Diagnostics ({len(diagnostics)}): " + ", ".join(diagnostic_names)
    )
    logger.info(
        f"Actuators ({len(actuators)}): " + ", ".join(actuator_names)
    )

    model = E2EFoundationModel(
        diagnostics=diagnostics,
        actuators=actuators,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_total_tokens = model.n_total_tokens
    model = dm.wrap(model)
    logger.info(
        f"Model — d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads}  tokens={n_total_tokens}  "
        f"params={n_params / 1e6:.2f}M  ddp={dm.distributed}"
    )

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(
        args.data_dir,
        train_files,
        val_files,
        preprocessing_stats=stats,
        chunk_duration_s=args.chunk_duration_s,
        prediction_horizon_s=args.prediction_horizon_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
        lengths_cache_dir=args.checkpoint_dir,
    )
    logger.info(f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}")

    # PyTorch's _worker_loop pins each DataLoader worker to a single
    # torch thread regardless of OMP_NUM_THREADS, so we override here to
    # let CPU-side STFT actually use the threads OMP_NUM_THREADS exposes.
    def _worker_init(_worker_id: int) -> None:
        import os as _os
        n = int(_os.environ.get("OMP_NUM_THREADS", "1"))
        torch.set_num_threads(n)

    if dm.distributed:
        # DistributedSampler shards chunk indices across ranks. Loses the
        # file-sequential cache locality of TwoLevelSampler — revisit if
        # HDF5 open() time becomes a bottleneck under DDP.
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dm.world_size,
            rank=dm.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
    else:
        train_sampler = TwoLevelSampler(train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=2,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init,
    )

    # ── Optim + schedule ───────────────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = _build_scheduler(
        opt, args.max_steps, args.warmup_steps, args.min_lr
    )

    # bf16 mixed precision. bf16 has the same dynamic range as fp32 so
    # no GradScaler is required; matches train_e2e_stage2_delta.py.
    use_amp = (not args.no_amp) and device.type == "cuda"

    def amp_ctx_factory():
        if use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ── Train ──────────────────────────────────────────────────────────
    logger.info(
        f"Starting training — lr schedule: linear warmup "
        f"{args.warmup_steps} steps → cosine → min_lr {args.min_lr}; "
        f"amp={'bf16' if use_amp else 'off'}."
    )
    best_val_loss = float("inf")
    best_step = 0

    # ── Optional resume (restores step / optimizer / scheduler / best_val_loss) ──
    resume_start_step = 0
    if args.resume_checkpoint is not None and args.resume_checkpoint.exists():
        resume_ckpt = torch.load(
            args.resume_checkpoint, weights_only=False, map_location=device
        )
        # Allow video/spectro keys to be missing from older TS-only checkpoints
        # (e.g. resuming a Phase A Stage 1 checkpoint into a TS+tangtv model).
        allowed_missing = tuple(
            f"{prefix}{name}." for prefix in (
                "diag_tokenizers.", "diag_heads."
            )
            for name in (*args.use_video, *args.use_spectro)
        )
        load_state_dict_explicit(
            _core(model),
            resume_ckpt["model_state_dict"],
            allowed_missing_prefixes=allowed_missing,
        )
        if "optimizer_state_dict" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        resume_start_step = int(resume_ckpt.get("step", 0))
        best_val_loss = float(resume_ckpt.get(
            "best_val_loss", resume_ckpt.get("val_loss", float("inf"))
        ))
        best_step = int(resume_ckpt.get("best_step", resume_start_step))
        logger.info(
            f"RESUMED from {args.resume_checkpoint.name}: starting at step "
            f"{resume_start_step}; best_val_loss={best_val_loss:.4f} at step "
            f"{best_step}"
        )
    elif args.init_checkpoint is not None:
        # Cold start with weights warm-loaded from another checkpoint.
        init_ckpt = torch.load(
            args.init_checkpoint, weights_only=False, map_location=device
        )
        allowed_missing = tuple(
            f"{prefix}{name}." for prefix in (
                "diag_tokenizers.", "diag_heads."
            )
            for name in (*args.use_video, *args.use_spectro)
        )
        load_state_dict_explicit(
            _core(model),
            init_ckpt["model_state_dict"],
            allowed_missing_prefixes=allowed_missing,
        )
        logger.info(
            f"INIT from {args.init_checkpoint.name} "
            f"(val_loss={init_ckpt.get('val_loss', 'n/a')} "
            f"step={init_ckpt.get('step', 'n/a')}); "
            "optimizer/scheduler/step start fresh."
        )
    step = resume_start_step

    # ── Per-category warm-start freezes ──────────────────────────────
    freeze_specs = [
        ("ts", args.freeze_ts_steps),
        ("video", args.freeze_video_steps),
        ("spectro", args.freeze_spectro_steps),
        ("backbone", args.freeze_backbone_steps),
    ]
    active_freezes: Dict[str, int] = {}
    for cat, n_steps in freeze_specs:
        if n_steps > 0 and step < n_steps:
            kwargs = {f"freeze_{c}": (c == cat) for c, _ in freeze_specs}
            labels = _apply_module_freeze(_core(model), **kwargs)
            if labels:
                active_freezes[cat] = n_steps
                logger.info(
                    f"Freeze({cat}) active until step {n_steps}; "
                    f"frozen labels = {labels}. Currently at step {step}."
                )
            else:
                logger.info(
                    f"Freeze({cat}) requested for {n_steps} steps but no "
                    f"matching modules — skipped."
                )
        elif n_steps > 0:
            logger.info(
                f"Freeze({cat}) past its release step {n_steps} "
                f"(currently {step}); category fully trainable."
            )
    running_total = 0.0
    running_count = 0
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

        opt.zero_grad()
        with amp_ctx_factory():
            loss, per_mod = compute_step_loss(model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        scheduler.step()
        running_total += loss.item()
        running_count += 1
        step += 1

        # Release each warm-start freeze when its step budget elapses.
        # Categories act independently so two can release at different
        # times if their step counts differ.
        for cat in list(active_freezes.keys()):
            if step >= active_freezes[cat]:
                kwargs = {f"freeze_{c}": (c == cat) for c, _ in freeze_specs}
                n_unfrozen = _release_module_freeze(model, **kwargs)
                logger.info(
                    f"Freeze({cat}) released at step {step}; "
                    f"{n_unfrozen} parameter tensors now trainable."
                )
                del active_freezes[cat]

        if step % args.log_every == 0:
            avg = running_total / running_count
            lr_now = opt.param_groups[0]["lr"]
            per_mod_str = ", ".join(
                f"{n}={per_mod[n]:.4f}" for n in diagnostic_names
            )
            logger.info(
                f"step {step}/{args.max_steps}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}  | {per_mod_str}"
            )
            running_total = 0.0
            running_count = 0

        if step % args.val_every == 0 or step == args.max_steps:
            # All ranks run validate() in lockstep — DDP forward broadcasts
            # buffers (broadcast_buffers=True default), so rank-0-only val
            # would deadlock. Each rank computes the same metrics from the
            # replicated (non-distributed) val_loader; only rank 0 logs/saves.
            metrics = validate(
                model,
                val_loader,
                device,
                diagnostic_names,
                max_batches=args.val_max_batches,
                use_amp=use_amp,
            )
            logger.info(
                "Validation (MAE model vs copy; delta-ratio pred/tgt):"
            )
            for n in diagnostic_names:
                m = metrics[n]
                delta = m["model_mae"] - m["copy_mae"]
                marker = "↓" if delta < 0 else "↑"
                logger.info(
                    f"  {n:<25s} "
                    f"model={m['model_mae']:.4f}  copy={m['copy_mae']:.4f}  "
                    f"{marker} {abs(delta):.4f}  | "
                    f"pred_d={m['pred_delta']:.4f}  tgt_d={m['tgt_delta']:.4f}  "
                    f"ratio={m['delta_ratio']:.3f}"
                )
            val_loss = sum(metrics[n]["model_mae"] for n in diagnostic_names)
            logger.info(f"  [sum model MAE] {val_loss:.4f}")
            # Decide best-update first so both `latest` and `best` share the
            # same final best_val_loss / best_step values — otherwise resume
            # from `latest` would see a stale best.
            is_new_best = val_loss < best_val_loss
            if is_new_best:
                best_val_loss = val_loss
                best_step = step

            if dm.is_main:
                ckpt_state = {
                    "model_state_dict": _core(model).state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                    "metrics": metrics,
                    "diagnostics": [asdict(c) for c in diagnostics],
                    "actuators": [asdict(c) for c in actuators],
                    "args": vars(args),
                }
                latest_path = args.checkpoint_dir / "e2e_stage1_latest.pt"
                torch.save(ckpt_state, latest_path)
                if is_new_best:
                    best_path = args.checkpoint_dir / "e2e_stage1_best.pt"
                    torch.save(ckpt_state, best_path)
                    logger.info(
                        f"  ✓ new best val_loss={val_loss:.4f}  saved {best_path.name}"
                    )
            dm.barrier()

    if dm.is_main:
        ckpt_path = args.checkpoint_dir / "e2e_stage1_final.pt"
        torch.save(
            {
                "model_state_dict": _core(model).state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "best_step": best_step,
                "diagnostics": [asdict(c) for c in diagnostics],
                "actuators": [asdict(c) for c in actuators],
                "args": vars(args),
            },
            ckpt_path,
        )
        logger.info(
            f"Saved final checkpoint: {ckpt_path}. "
            f"Best val_loss={best_val_loss:.4f} at step {best_step}."
        )
    dm.barrier()


if __name__ == "__main__":
    main()