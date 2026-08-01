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
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from tokamak_foundation_model.e2e.output_heads import SpectrogramFlowHead
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


# Per-camera video modality registry. Each entry is
# ``(name, n_channels, n_frames, (height, width), (T_p, H_p, W_p))``.
# Only included when the user passes ``--use_video <name> [<name> ...]``;
# otherwise behaviour is byte-identical to Phase A pre-Step-5 (G2/G3).
VIDEO_MODALITIES: List[Tuple[str, int, int, Tuple[int, int], Tuple[int, int, int]]] = [
    ("tangtv", 7, 3, (120, 360), (3, 12, 12)),
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
    ("mhr", 6, (32, 8)),
]


def build_configs(
    chunk_duration_s: float,
    use_video: Optional[List[str]] = None,
    use_spectro: Optional[List[str]] = None,
    spectro_patch_f: Optional[int] = None,
    spectro_patch_t: Optional[int] = None,
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
            if spectro_patch_f is not None or spectro_patch_t is not None:
                pf, pt = patch_size
                patch_size = (spectro_patch_f or pf, spectro_patch_t or pt)
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


def weighted_masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    """Masked MAE with a per-(channel, freq-bin) weight tensor.

    For spectrogram modalities, ``weight[c, f] = sigma_channel[c] /
    sigma_per_bin[c, f]`` makes this equivalent to MAE in per-bin
    standardized space — every freq bin contributes equally to the loss
    instead of loud (low-freq, broadband) bins dominating. Goal: counter
    spec mean-collapse by giving quiet, mode-carrying bins the same
    loss-budget pressure as loud background bins.

    The weight is broadcast as (1, C, F, 1) against (B, C, F, T)
    pred/target tensors. Plain MAE is recovered when ``weight ≡ 1``.
    """
    cleaned_pred, pred_mask = _clean_and_mask(pred, None)
    cleaned_target, target_mask = _clean_and_mask(target, mask)
    combined = pred_mask * target_mask
    w = weight.view(1, weight.shape[0], weight.shape[1], 1)
    diff = (cleaned_pred - cleaned_target).abs() * combined * w
    return diff.sum() / combined.sum().clamp_min(1.0)


def build_spec_per_bin_weights(
    stats: Dict,
    diagnostics: List[DiagnosticConfig],
    signal_configs: List,
    device: torch.device,
    clamp_min: float = 1.0,
    clamp_max: float = 10.0,
    power: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Per-modality (C_sliced, F) weight tensors for the per-bin MAE.

    ``w[c, f] = sigma_channel[c] / sigma_per_bin[c, f]`` (clamped). Quiet
    bins (small ``sigma_per_bin``) get larger weight so the model can't
    cheaply mean-collapse them. ``sigma_channel`` is the standard
    log-standardize std stored under ``stats[name]["log"]["std"]``;
    ``sigma_per_bin`` comes from the new ``stats[name]["log_per_bin"]
    ["std"]`` sub-key (computed by
    ``scripts/data_preparation/make_processing_stats.py`` with
    ``compute_per_bin_for_stft=True``).

    Returns ``{}`` (and the caller falls back to plain MAE) if ANY
    spectrogram modality lacks ``log_per_bin`` stats. Channel slicing
    matches each ``SignalConfig.channels_to_use`` so the weight shape
    aligns with the model's actual input channel count.
    """
    cfg_by_name = {c.name: c for c in signal_configs}
    out: Dict[str, torch.Tensor] = {}
    for cfg in diagnostics:
        if cfg.kind != "spectrogram":
            continue
        entry = stats.get(cfg.name, {})
        if "log" not in entry or "log_per_bin" not in entry:
            return {}
        sigma_c = torch.as_tensor(entry["log"]["std"]).to(torch.float32)
        sigma_pb = torch.as_tensor(entry["log_per_bin"]["std"]).to(torch.float32)
        sigma_c = torch.where(torch.isnan(sigma_c), torch.ones_like(sigma_c), sigma_c)
        sigma_pb = torch.where(torch.isnan(sigma_pb), torch.ones_like(sigma_pb), sigma_pb)
        sig_cfg = cfg_by_name.get(cfg.name)
        sl = sig_cfg.channels_to_use if sig_cfg is not None else None
        if sl is not None:
            sigma_c = sigma_c[sl]
            sigma_pb = sigma_pb[sl]
        w = sigma_c[:, None] / sigma_pb.clamp(min=1e-6)
        if power != 1.0:
            w = w ** power
        w = w.clamp(min=clamp_min, max=clamp_max).to(device)
        out[cfg.name] = w
    return out


def build_spec_per_bin_sigma(
    stats: Dict,
    diagnostics: List[DiagnosticConfig],
    signal_configs: List,
) -> Dict[str, torch.Tensor]:
    """Per-modality ``(C_sliced, F)`` per-bin std tensors for the generative
    head's residual standardisation. Reads ``stats[name]["log_per_bin"]
    ["std"]`` (same source as :func:`build_spec_per_bin_weights`), sliced to
    the model's channels. Returns ``{}`` if any spectro modality lacks the
    ``log_per_bin`` stats → the flow head keeps its default ones (no
    standardisation). NaNs and tiny values are floored to 1.0 / 1e-3.
    """
    cfg_by_name = {c.name: c for c in signal_configs}
    out: Dict[str, torch.Tensor] = {}
    for cfg in diagnostics:
        if cfg.kind != "spectrogram":
            continue
        entry = stats.get(cfg.name, {})
        if "log_per_bin" not in entry:
            return {}
        sigma_pb = torch.as_tensor(entry["log_per_bin"]["std"]).to(torch.float32)
        sigma_pb = torch.where(
            torch.isnan(sigma_pb), torch.ones_like(sigma_pb), sigma_pb
        )
        sig_cfg = cfg_by_name.get(cfg.name)
        sl = sig_cfg.channels_to_use if sig_cfg is not None else None
        if sl is not None:
            sigma_pb = sigma_pb[sl]
        out[cfg.name] = sigma_pb.clamp(min=1e-3)
    return out


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
    Dict[str, torch.Tensor],  # per-modality backbone token slices (conditioning)
]:
    """Forward pass with NaN-cleaned inputs; return predictions + tensors needed for metrics.

    The 5th return value maps each diagnostic name to its backbone output
    token slice (the conditioning a generative head needs to compute its
    loss against the target, which the head's own forward never sees).
    """
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

    predictions, diag_token_slices = model(
        diag_inputs, act_inputs, step_idx, time_offset, return_tokens=True
    )

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
    return predictions, diag_inputs, targets, masks, diag_token_slices


def compute_step_loss(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
    spec_pb_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one forward pass and return ``(total_loss, per-modality MAE dict)``.

    ``spec_pb_weights`` (default ``None``) enables the per-bin weighted
    MAE for spectrogram modalities. When ``None`` (the historical
    default), every modality uses plain ``masked_mae`` — identical to
    pre-2026-06-12 behavior. When a dict ``{name: weight_tensor(C, F)}``,
    each named spectrogram is scored via ``weighted_masked_mae`` to
    counter spec mean-collapse.
    """
    predictions, _, targets, masks, token_slices = forward_batch(
        model, batch, device
    )
    per_modality: Dict[str, float] = {}
    total_loss = torch.zeros((), device=device)
    core = _core(model)
    for cfg in core.diagnostics:
        head = core.diag_heads[cfg.name]
        use_pb = (
            spec_pb_weights is not None
            and cfg.kind == "spectrogram"
            and cfg.name in spec_pb_weights
        )
        if use_pb:
            mae = weighted_masked_mae(
                predictions[cfg.name], targets[cfg.name],
                masks[cfg.name], spec_pb_weights[cfg.name],
            )
        else:
            mae = masked_mae(
                predictions[cfg.name], targets[cfg.name], masks[cfg.name]
            )
        if isinstance(head, SpectrogramFlowHead):
            # predictions[name] == μ in train mode (head.forward returns the
            # deterministic mean); add the rectified-flow velocity loss on the
            # residual. The velocity net runs every step here → all its params
            # get grads (DDP-safe, no unused parameters).
            flow = head.flow_loss(
                token_slices[cfg.name], predictions[cfg.name],
                targets[cfg.name], masks[cfg.name],
            )
            loss = mae + head.flow_lambda * flow
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_flow"] = flow.item()
        else:
            loss = mae
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
    """Return per-modality validation metrics, computed in a
    distribution-aware way.

    The val_loader is assumed to be sharded across ranks (via a
    ``DistributedTwoLevelSampler`` with ``shuffle=False``). Each rank
    accumulates partial sums on its shard; the totals are all-reduced
    once at the end so every rank ends up with the same global metric
    values. This replaces the previous "every rank validates everything"
    behaviour, which caused host-memory OOMs at 64+ ranks because each
    rank held the full val workload in flight independently.

    ``out[name]`` has keys ``model_mae``, ``copy_mae``, ``pred_delta``,
    ``tgt_delta``, ``delta_ratio``.

    ``pred_delta`` and ``tgt_delta`` are displacement-magnitude metrics
    (``ResearchPlan.MD`` §7): ``||pred - input||`` and ``||target - input||``
    respectively, both masked. A model that copies its input has
    ``pred_delta ≈ 0``; a model predicting the true dynamics has
    ``delta_ratio = pred_delta / tgt_delta ∈ [0.8, 1.2]``.
    """
    import torch.distributed as dist

    model.eval()
    # Bypass the DDP wrapper for the val forward pass. DDP's pre-forward
    # hook (rebuild_buckets logic) was observed to trigger GPU memory
    # access faults during validation even under no_grad. The inner
    # module's weights are identical across ranks (DDP keeps them in
    # sync), so forwarding through it directly produces the same result.
    inner = _core(model)

    keys = ("model_mae", "copy_mae", "pred_delta", "tgt_delta",
            "pred_var", "gt_var")
    M = len(diagnostic_names)
    K = len(keys)
    name_to_kind = {c.name: c.kind for c in inner.diagnostics}
    # fp32 accumulators regardless of autocast — keeps cross-rank
    # all_reduce in fp32 (bf16 all_reduce on RCCL has stability issues)
    # and avoids precision loss across many batches.
    sums_t = torch.zeros(K, M, device=device, dtype=torch.float32)
    n_batches_t = torch.zeros((), device=device, dtype=torch.float32)
    name_to_col = {n: j for j, n in enumerate(diagnostic_names)}

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp else contextlib.nullcontext()
    )
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        # Only the forward pass runs inside autocast; metric math
        # explicitly upcasts to fp32 below.
        with amp_ctx:
            predictions, diag_inputs, targets, masks, _ = forward_batch(
                inner, batch, device
            )
        copy_mod = copy_baseline_mae(batch, inner.diagnostics, device)
        for name in diagnostic_names:
            j = name_to_col[name]
            pred = predictions[name].float()
            inp = diag_inputs[name].float()
            tgt = targets[name].float()
            existing = masks[name].float() if masks[name] is not None else None

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

            sums_t[0, j] += model_mae_v
            sums_t[1, j] += float(copy_mod[name])
            sums_t[2, j] += pred_delta
            sums_t[3, j] += tgt_delta
            # Temporal-variance ratio (TVR) for spectrograms: variance over
            # the time axis per (B,C,F), summed over valid bins. Collapse →
            # tiny pred variance vs GT (ratio ~0.15); recovered modes → ~1.
            if name_to_kind.get(name) == "spectrogram" and cleaned_pred.dim() == 4:
                mvalid = (combined.amax(dim=-1) > 0).float()       # (B,C,F)
                sums_t[4, j] += (cleaned_pred.var(dim=-1) * mvalid).sum()
                sums_t[5, j] += (cleaned_tgt.var(dim=-1) * mvalid).sum()
        n_batches_t += 1.0

    # Single all-reduce across ranks (sums + batch count combined into
    # contiguous fp32 tensors above). Empty-shard ranks contribute
    # zeros and a count of 0, which is the correct behaviour.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_batches_t, op=dist.ReduceOp.SUM)

    denom = float(n_batches_t.item())
    if denom <= 0.0:
        denom = 1.0
    sums = sums_t.detach().cpu().numpy()
    model.train()
    out: Dict[str, Dict[str, float]] = {}
    for name in diagnostic_names:
        j = name_to_col[name]
        model_mae = float(sums[0, j]) / denom
        copy_mae = float(sums[1, j]) / denom
        pred_d = float(sums[2, j]) / denom
        tgt_d = float(sums[3, j]) / denom
        ratio = pred_d / tgt_d if tgt_d > 1e-8 else float("nan")
        pred_var = float(sums[4, j])
        gt_var = float(sums[5, j])
        tvr = pred_var / gt_var if gt_var > 1e-8 else float("nan")
        out[name] = {
            "model_mae": model_mae,
            "copy_mae": copy_mae,
            "pred_delta": pred_d,
            "tgt_delta": tgt_d,
            "delta_ratio": ratio,
            "tvr": tvr,
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
    freeze_slow_ts: bool,
    freeze_fast_ts: bool,
    freeze_video: bool,
    freeze_spectro: bool,
    freeze_backbone: bool,
) -> List[Tuple[str, torch.nn.Parameter]]:
    """Return ``[(label, param), ...]`` for every parameter the caller
    asked to freeze. ``label`` is a short string identifying the source
    (e.g. ``"slow_ts:ts_core_density"``, ``"backbone"``) for log output.

    slow_ts and fast_ts have separate freeze flags (2026-05-19) so the
    auto-injected refine-stack-extension freeze can keep slow_ts pinned
    while letting fast_ts (which got new refine blocks) train.

    No-op categories return no params, so passing ``freeze_video=True``
    on a model without video modules is harmless.
    """
    out: List[Tuple[str, torch.nn.Parameter]] = []
    for cfg in model.diagnostics:
        if cfg.kind == "slow_ts" and freeze_slow_ts:
            label = f"slow_ts:{cfg.name}"
        elif cfg.kind == "fast_ts" and freeze_fast_ts:
            label = f"fast_ts:{cfg.name}"
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
    freeze_slow_ts: bool,
    freeze_fast_ts: bool,
    freeze_video: bool,
    freeze_spectro: bool,
    freeze_backbone: bool,
) -> List[str]:
    """Freeze the per-module parameters indicated by the flags.

    Each flag is independent; pass ``True`` for any subset. Actuator
    tokenizers stay trainable in all cases (they are tiny and
    inseparable from the dynamics the model learns).

    Returns the deduplicated list of frozen labels (for log output).
    """
    pairs = _module_param_iter(
        model,
        freeze_slow_ts=freeze_slow_ts,
        freeze_fast_ts=freeze_fast_ts,
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
    freeze_slow_ts: bool,
    freeze_fast_ts: bool,
    freeze_video: bool,
    freeze_spectro: bool,
    freeze_backbone: bool,
) -> int:
    """Release the freeze applied by :func:`_apply_module_freeze` with
    the same flags; return the number of parameter tensors unfrozen
    (for log output)."""
    pairs = _module_param_iter(
        model,
        freeze_slow_ts=freeze_slow_ts,
        freeze_fast_ts=freeze_fast_ts,
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
    parser.add_argument(
        "--lengths_cache_dir",
        type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model_meta"),
        help="Directory for TokamakMultiFileDataset length-cache sidecar "
        "files (lengths_e2e_stage1_{train,val}.pt). Defaults to the "
        "shared foundation_model_meta dir so all ranks/jobs reuse the "
        "same cache.",
    )
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
    parser.add_argument(
        "--backbone_grad_checkpoint", action="store_true",
        help="Per-block gradient checkpointing on the backbone. Trades "
        "~30%% step-time for ~sqrt(n_layers) reduction in activation "
        "memory. Required when d_model >= ~1024 to fit on 64 GB GCDs.",
    )
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Optim
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--val_batch_size", type=int, default=None,
        help="Per-rank batch size for validation (default: --batch_size). "
             "Set smaller than --batch_size when validation OOMs while "
             "training fits — e.g. the generative spectro head's fp32 "
             "(--no_amp_val) Euler sampling spikes well above the training "
             "footprint at d=1024.",
    )
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
        "--spectro_patch_f", type=int, default=None,
        help="Override the spectrogram freq-patch size for ALL spectro "
             "modalities (default: per-modality registry value). Set to "
             "SPECTRO_FREQ_BINS (512) for a full-frequency patch — one token "
             "spans the whole spectrum. Changes encoder/decoder kernel shape "
             "→ from-scratch (checkpoints with a different patch won't load).",
    )
    parser.add_argument(
        "--spectro_patch_t", type=int, default=None,
        help="Override the spectrogram time-patch size for ALL spectro "
             "modalities (default: registry value 8). Smaller → more "
             "full-spectrum time tokens per window.",
    )
    parser.add_argument(
        "--freeze_ts_steps", type=int, default=0,
        help="DEPRECATED alias: set both --freeze_slow_ts_steps and "
             "--freeze_fast_ts_steps to N. If either of those is also "
             "set explicitly, the explicit one wins for that category.",
    )
    parser.add_argument(
        "--freeze_slow_ts_steps", type=int, default=0,
        help="Warm-start: freeze slow_ts tokenizers + heads for N steps.",
    )
    parser.add_argument(
        "--freeze_fast_ts_steps", type=int, default=0,
        help="Warm-start: freeze fast_ts tokenizers + heads for N steps.",
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
        "--spectro_seam_refine", action="store_true",
        help="Enable the zero-init seam-refine block on spectrogram "
             "output heads (default off — matches historical Stage 1).",
    )
    parser.add_argument(
        "--video_seam_refine", action="store_true",
        help="Enable the zero-init seam-refine block on the video "
             "output head (default off).",
    )
    parser.add_argument(
        "--seam_refine_hidden_ch", type=int, default=16,
        help="Hidden channels of the seam-refine blocks (default 16 = "
             "original architecture; the spec-fix fine-tune uses 64).",
    )
    parser.add_argument(
        "--spectro_refine_kernel", type=int, default=3,
        help="Square kernel size of the spectrogram seam-refine convs.",
    )
    parser.add_argument(
        "--video_refine_kernel", type=int, nargs=3, default=[1, 3, 3],
        help="(T, H, W) kernel of the video seam-refine convs.",
    )
    parser.add_argument(
        "--spec_inv_stem", action="store_true",
        help="Enable the inv_stem feature-space decode branch on "
             "spectrogram heads (fast-TS deconv→inv_stem pattern; "
             "zero-init residual, warm-start safe).",
    )
    parser.add_argument(
        "--spec_inv_stem_ch", type=int, default=64,
        help="Feature channels of the spectrogram inv_stem branch.",
    )
    parser.add_argument(
        "--spec_freq_stem", action="store_true",
        help="Enable the full-frequency encoder stem on spectrogram "
             "tokenizers: a zero-init residual freq->freq Linear mixing "
             "(matmul, MIOpen-free) applied BEFORE patching so each "
             "token encodes whole-spectrum context. Warm-start safe.",
    )
    parser.add_argument(
        "--spec_freq_stem_hidden", type=int, default=128,
        help="Hidden width of the freq stem's low-rank freq mixing.",
    )
    parser.add_argument(
        "--freeze_whole_run", action="store_true",
        help="Apply all --freeze_*_steps freezes BEFORE the DDP wrap and "
             "never release them. Avoids the post-wrap requires_grad flip "
             "that breaks DDP's reducer (see 2026-05-19 emergency patch). "
             "Use for fine-tunes where categories stay frozen for the "
             "entire run; the numeric step values then only act as "
             "on/off switches (any value > 0 = frozen).",
    )
    parser.add_argument(
        "--spec_per_bin_loss", action="store_true",
        help="Spectrogram modalities use per-(channel, freq-bin) weighted MAE "
             "(weight = sigma_channel / sigma_per_bin, clamped). Counters "
             "spec mean-collapse by rebalancing loss across freq bins. "
             "Requires 'log_per_bin' sub-entries in preprocessing_stats. "
             "Default off → identical to historical plain MAE.",
    )
    parser.add_argument(
        "--spec_per_bin_weight_clamp", type=float, default=10.0,
        help="Upper clamp on per-bin weight (lower clamp fixed at 1.0). "
             "Default 10.0. Only used when --spec_per_bin_loss is set.",
    )
    parser.add_argument(
        "--spec_per_bin_weight_power", type=float, default=1.0,
        help="Exponent applied to (sigma_c/sigma_pb) before clamping. "
             "1.0 = linear (mild; real weights peak ~3.6x). 2.0 = "
             "squared (ECE quiet bins ~13x) — stronger mode pressure; "
             "raise --spec_per_bin_weight_clamp to ~20 so squared "
             "values aren't clipped.",
    )
    # ── Generative-head / checkerboard-fix POC flags (2026-06-21) ──
    parser.add_argument(
        "--video_resize_conv", action="store_true",
        help="Use a resize-conv (trilinear upsample → Conv3d block) video "
             "decoder instead of the per-patch ConvTranspose3d. Overlapping "
             "receptive fields across patch seams remove the checkerboard. "
             "Supersedes --video_seam_refine. NOT warm-start safe (changes "
             "the head architecture) — for from-scratch runs.",
    )
    parser.add_argument(
        "--video_resize_conv_hidden", type=int, default=64,
        help="Hidden channels of the resize-conv video decoder block.",
    )
    parser.add_argument(
        "--spec_generative", action="store_true",
        help="Use the generative SpectrogramFlowHead (rectified flow matching "
             "over a deterministic mean) instead of the deterministic "
             "spectrogram head. Samples sharp modes instead of regressing to "
             "the blurry conditional mean. NOT warm-start safe — from scratch.",
    )
    parser.add_argument(
        "--spec_flow_base_ch", type=int, default=64,
        help="Base channel width of the flow head's velocity U-Net.",
    )
    parser.add_argument(
        "--spec_flow_steps", type=int, default=6,
        help="Euler ODE steps used to sample the flow head at eval time.",
    )
    parser.add_argument(
        "--spec_flow_lambda", type=float, default=1.0,
        help="Weight of the flow-matching loss relative to the mean MAE.",
    )
    parser.add_argument(
        "--collapse_aware_best", action="store_true",
        help="Add a temporal-variance-ratio penalty (sum of max(0, 1 - tvr) "
             "over spectro modalities) to the best.pt selection scalar so a "
             "low-MAE mean-collapse cannot win. Default off → plain sum(MAE).",
    )
    parser.add_argument(
        "--collapse_aware_lambda", type=float, default=1.0,
        help="Weight of the TVR penalty in --collapse_aware_best selection.",
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable bf16 mixed precision (default: AMP on when CUDA).",
    )
    parser.add_argument(
        "--no_amp_val", action="store_true",
        help="Disable bf16 autocast during validation only (training still "
        "uses AMP if --no_amp not set). Workaround for the GPU memory-"
        "access faults seen during distributed validation at n_layers=26 "
        "on Frontier ROCm 7.1.1.",
    )
    args = parser.parse_args()

    dm = DistributedManager()

    logging.basicConfig(
        level=logging.INFO if dm.is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{dm.rank}] %(message)s",
    )

    # OOM mitigation. Chained production jobs 4581026/27/28 OOM'd at
    # exactly ~9h45m / ~5850 steps with num_workers=6 (passed by the
    # queued SLURM scripts before this fix landed). Clamp here so
    # already-queued jobs that read this Python source at start-time
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
        args.lengths_cache_dir.mkdir(parents=True, exist_ok=True)
        train_files = filter_video_present_files(
            train_files,
            args.use_video,
            cache_path=args.lengths_cache_dir / "video_present_train.pt",
        )
        val_files = filter_video_present_files(
            val_files,
            args.use_video,
            cache_path=args.lengths_cache_dir / "video_present_val.pt",
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
        spectro_patch_f=args.spectro_patch_f,
        spectro_patch_t=args.spectro_patch_t,
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
        backbone_grad_checkpoint=args.backbone_grad_checkpoint,
        video_seam_refine=args.video_seam_refine,
        spectro_seam_refine=args.spectro_seam_refine,
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
    n_params = sum(p.numel() for p in model.parameters())
    n_total_tokens = model.n_total_tokens

    # --freeze_whole_run: freeze BEFORE the DDP wrap so the reducer is
    # built without the frozen parameters. The post-wrap freeze block
    # (below, near the training loop) is skipped for these categories —
    # flipping requires_grad after wrap either crashes DDP
    # (find_unused_parameters=False expects grads for registered
    # params) or, on release, silently diverges ranks (params train
    # locally but are never all-reduced). Whole-run freezes have no
    # release, so neither failure mode applies.
    if args.freeze_whole_run:
        _wr_slow = max(args.freeze_slow_ts_steps, args.freeze_ts_steps) > 0
        _wr_fast = max(args.freeze_fast_ts_steps, args.freeze_ts_steps) > 0
        labels = _apply_module_freeze(
            model,
            freeze_slow_ts=_wr_slow,
            freeze_fast_ts=_wr_fast,
            freeze_video=args.freeze_video_steps > 0,
            freeze_spectro=args.freeze_spectro_steps > 0,
            freeze_backbone=args.freeze_backbone_steps > 0,
        )
        n_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logger.info(
            f"freeze_whole_run: frozen for the entire run = {labels}; "
            f"trainable params {n_trainable / 1e6:.2f}M / "
            f"{n_params / 1e6:.2f}M"
        )

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
        lengths_cache_dir=args.lengths_cache_dir,
    )
    logger.info(f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}")

    # Per-bin spec loss weights — built once at init from
    # preprocessing_stats. ``None`` here = plain MAE (original behavior).
    # Set via --spec_per_bin_loss flag.
    spec_pb_weights: Optional[Dict[str, torch.Tensor]] = None
    if args.spec_per_bin_loss:
        spec_pb_weights = build_spec_per_bin_weights(
            stats, _core(model).diagnostics, train_ds.signal_configs,
            device=device, clamp_max=args.spec_per_bin_weight_clamp,
            power=args.spec_per_bin_weight_power,
        )
        if not spec_pb_weights:
            logger.warning(
                "--spec_per_bin_loss requested but preprocessing_stats is "
                "missing 'log_per_bin' for one or more spectrogram modalities "
                "— falling back to plain MAE."
            )
            spec_pb_weights = None
        else:
            for n, w in spec_pb_weights.items():
                logger.info(
                    f"Per-bin spec loss [{n}]: weight shape {tuple(w.shape)}, "
                    f"range [{float(w.min()):.2f}, {float(w.max()):.2f}], "
                    f"mean {float(w.mean()):.2f}  "
                    f"(clamp_max={args.spec_per_bin_weight_clamp})"
                )

    # Generative spectrogram heads: set the per-(channel, freq) residual-std
    # buffer once from the per-bin stats so the flow's velocity targets are
    # unit-scale per bin (quiet, mode-carrying bins aren't drowned out).
    # Persisted in the checkpoint → eval reconstructs it. Falls back to ones
    # (no standardisation) if log_per_bin stats are unavailable.
    if args.spec_generative:
        sigma_pb_map = build_spec_per_bin_sigma(
            stats, _core(model).diagnostics, train_ds.signal_configs,
        )
        if not sigma_pb_map:
            logger.warning(
                "--spec_generative: preprocessing_stats missing 'log_per_bin' "
                "— flow heads use unit residual std (no per-bin scaling)."
            )
        core_m = _core(model)
        for cfg in core_m.diagnostics:
            head = core_m.diag_heads[cfg.name]
            if isinstance(head, SpectrogramFlowHead) and cfg.name in sigma_pb_map:
                head.set_sigma_pb(sigma_pb_map[cfg.name].to(device))
                logger.info(
                    f"Flow head [{cfg.name}]: per-bin residual std set, "
                    f"shape {tuple(sigma_pb_map[cfg.name].shape)}"
                )

    # PyTorch's _worker_loop pins each DataLoader worker to a single
    # torch thread regardless of OMP_NUM_THREADS, so we override here to
    # let CPU-side STFT actually use the threads OMP_NUM_THREADS exposes.
    def _worker_init(_worker_id: int) -> None:
        import os as _os
        n = int(_os.environ.get("OMP_NUM_THREADS", "1"))
        torch.set_num_threads(n)

    if dm.distributed:
        # DDP-aware file-level sharding. Preserves TwoLevelSampler's
        # per-worker LRU file-handle cache locality (each rank owns a
        # fixed slice of the file list, iterates its own files
        # sequentially). PyTorch's DistributedSampler, which shards
        # chunk indices instead, was observed to make HDF5 open() the
        # dominant cost (~12 s/step at 2-GPU DDP vs. ~1 s/step
        # single-GPU at the same batch).
        train_sampler = DistributedTwoLevelSampler(
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
        # persistent_workers=False: chained jobs 4581026/27/28 OOM'd at
        # exactly ~9h45m / ~5850 steps with persistent workers — a slow
        # leak (h5py metadata or PyTorch tensor cache) fills 502 GB per
        # node over time. Tearing workers down at end-of-epoch releases
        # the state; spin-up cost (~5-10 s) is negligible vs the ~2 h
        # epoch wall time.
        persistent_workers=False,
        worker_init_fn=_worker_init,
    )
    # Distributed validation: shard the val set across ranks so each
    # rank validates ~1/world_size of it. Matching the train sampler's
    # file-level sharding (preserves LRU file-handle locality and avoids
    # the host-OOM that hit at 64 ranks when every rank held the full
    # val workload independently). Metrics are all-reduced inside
    # validate() so all ranks end up with identical global numbers.
    if dm.distributed:
        val_sampler = DistributedTwoLevelSampler(
            val_ds,
            num_replicas=dm.world_size,
            rank=dm.rank,
            shuffle=False,
            seed=args.seed,
            drop_last=True,
        )
    else:
        val_sampler = TwoLevelSampler(val_ds, shuffle=False)

    # Val loader memory budget. Train workers stay alive during val and
    # hold their prefetched batches (6 workers x 2 prefetch = 12 in flight
    # per rank). With num_workers=6 prefetch=1 the combined peak (18) hits
    # ~97% host RAM on 2-node smokes -> OOM territory. Capping val to
    # 4 workers x 1 prefetch keeps the combined in-flight at 16 batches,
    # within the 502 GB node budget. Workers are torn down at end-of-val.
    val_num_workers = min(4, args.num_workers)
    val_loader = DataLoader(
        val_ds,
        batch_size=(args.val_batch_size or args.batch_size),
        sampler=val_sampler,
        num_workers=val_num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=1,
        pin_memory=False,
        persistent_workers=False,
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
    # Separate flag for validation AMP. Defaults to the training value,
    # but --no_amp_val turns it off independently as a workaround for
    # ROCm-side GPU memory-access faults observed during distributed val.
    use_amp_val = use_amp and not args.no_amp_val

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
    # Auto-surgery flags populated by the resume block — empty on cold start.
    # Used downstream to inject auto-freeze into the freeze_specs.
    reinit_spectro_modalities: List[str] = []
    spectro_refine_extra_modalities: List[str] = []
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
        # Detect spectrogram-tokenizer patch-size changes by comparing the
        # checkpoint's `proj.weight` shape against the current model's. If
        # the patch shape was changed in SPECTROGRAM_MODALITIES, the kernel
        # shape no longer matches, so we cannot load those weights. Strip
        # the four patch-shape-dependent tensors per affected modality
        # (proj.{weight,bias} keeps bias; spatial_pe + missing_token are
        # nominally same-shape but semantically tied to the patch raster
        # order, so we reinit them too) and let the model keep its
        # fresh-init parameters. Auto-trigger a 2-epoch (2360-step) freeze
        # on backbone/ts/video so the new spectro Conv2ds adapt to a
        # stationary target first. See memory: project-session-pause-20260519.
        loaded_sd = resume_ckpt["model_state_dict"]
        for d_cfg in _core(model).diagnostics:
            if d_cfg.kind != "spectrogram":
                continue
            ckpt_w_key = f"diag_tokenizers.{d_cfg.name}.proj.weight"
            if ckpt_w_key not in loaded_sd:
                continue
            ckpt_shape = tuple(loaded_sd[ckpt_w_key].shape)
            model_shape = tuple(
                _core(model).diag_tokenizers[d_cfg.name].proj.weight.shape
            )
            if ckpt_shape != model_shape:
                reinit_spectro_modalities.append(d_cfg.name)
        if reinit_spectro_modalities:
            stale_prefixes: List[str] = []
            for name in reinit_spectro_modalities:
                stale_prefixes += [
                    f"diag_tokenizers.{name}.proj.weight",
                    f"diag_tokenizers.{name}.proj.bias",
                    f"diag_tokenizers.{name}.spatial_pe",
                    f"diag_tokenizers.{name}.missing_token",
                    f"diag_heads.{name}.patch_unembed.weight",
                    f"diag_heads.{name}.patch_unembed.bias",
                ]
            for sd_key in list(loaded_sd.keys()):
                if sd_key in stale_prefixes:
                    del loaded_sd[sd_key]
            allowed_missing = allowed_missing + tuple(stale_prefixes)
            logger.info(
                f"Spectrogram patch-size mismatch in modalities "
                f"{reinit_spectro_modalities}: reinitialising "
                f"tokenizer.{{proj,spatial_pe,missing_token}} + "
                f"head.patch_unembed; freezing backbone/slow_ts/video for "
                f"2 epochs (2360 steps) after this resume."
            )

        # Detect refine-stack extension: the model has more
        # `refine.<i>.*` modules than the checkpoint (e.g., we bumped
        # n_refine_blocks 4 → 12 in SpectrogramTokenizer/Head and 2 → 4
        # in FastTimeSeriesTokenizer/Head). Allow the extra blocks to
        # be missing-from-checkpoint so they keep their fresh-init
        # state. Trigger a 1-epoch (1180-step) freeze on backbone/
        # slow_ts/video while the new blocks settle (fast_ts itself is
        # NOT auto-frozen since it owns new refine blocks). Skips
        # opt-state restore (the optimizer's param list grew, so the
        # saved state-dict indices no longer align). Independent of
        # the patch-size reinit above — only one fires for any given
        # resume.
        for d_cfg in _core(model).diagnostics:
            if d_cfg.kind not in ("spectrogram", "fast_ts"):
                continue
            for mod_path in (
                f"diag_tokenizers.{d_cfg.name}",
                f"diag_heads.{d_cfg.name}",
            ):
                try:
                    mod = _core(model).get_submodule(mod_path)
                except AttributeError:
                    continue
                if not hasattr(mod, "refine"):
                    continue
                n_model = len(mod.refine)
                prefix = f"{mod_path}.refine."
                ckpt_indices = set()
                for k in loaded_sd:
                    if k.startswith(prefix):
                        head, _, _ = k[len(prefix):].partition(".")
                        if head.isdigit():
                            ckpt_indices.add(int(head))
                n_ckpt = (max(ckpt_indices) + 1) if ckpt_indices else 0
                if n_model > n_ckpt:
                    spectro_refine_extra_modalities.append(d_cfg.name)
                    for i in range(n_ckpt, n_model):
                        allowed_missing = allowed_missing + (
                            f"{mod_path}.refine.{i}.",
                        )
        spectro_refine_extra_modalities = sorted(
            set(spectro_refine_extra_modalities)
        )
        if spectro_refine_extra_modalities and not reinit_spectro_modalities:
            logger.info(
                f"Refine-stack extended in modalities "
                f"{spectro_refine_extra_modalities}: existing refine blocks "
                f"load from checkpoint, new blocks remain fresh-init; "
                f"freezing backbone/slow_ts/video for 1 epoch (1180 steps) "
                f"after this resume — fast_ts and spectro train alongside."
            )
        load_state_dict_explicit(
            _core(model),
            loaded_sd,
            allowed_missing_prefixes=allowed_missing,
        )
        if "optimizer_state_dict" in resume_ckpt:
            if reinit_spectro_modalities or spectro_refine_extra_modalities:
                # Skip opt-state restore on either spectro surgery path:
                # (a) patch-size reinit — checkpoint's Adam buffers for
                # proj/patch_unembed have the OLD kernel shape; load
                # would raise on shape mismatch;
                # (b) refine-stack extension — the optimizer's param
                # list grew with the new refine.<i> blocks, so the saved
                # state-dict indices no longer align with current params.
                # In either case, AdamW resets for all params — the
                # auto-freeze keeps the non-spectro modules at their
                # checkpoint weights until Adam re-accumulates momentum.
                logger.info(
                    "Skipping optimizer state restore due to spectro "
                    f"model surgery — "
                    f"patch_reinit={bool(reinit_spectro_modalities)}, "
                    f"refine_extra={bool(spectro_refine_extra_modalities)}."
                )
            else:
                # Generic param-count guard: ANY model surgery that
                # adds/removes params (e.g. VideoOutputHead.refine_block
                # was added 2026-06-08) makes the saved optimizer state
                # incompatible. Compare param counts before load —
                # mismatch ⇒ fresh AdamW. Avoids the unrecoverable
                # `ValueError: loaded state dict contains a parameter
                # group that doesn't match the size of optimizer's
                # group` that crashed the chain 2026-06-08.
                saved_n = sum(
                    len(g["params"])
                    for g in resume_ckpt["optimizer_state_dict"]["param_groups"]
                )
                cur_n = sum(len(g["params"]) for g in opt.param_groups)
                if saved_n != cur_n:
                    logger.warning(
                        f"Skipping optimizer state restore: saved had "
                        f"{saved_n} params, model has {cur_n}. "
                        f"AdamW will start fresh — first few hundred "
                        f"steps may be slightly noisy until momentum "
                        f"re-accumulates."
                    )
                else:
                    opt.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            # Re-target the cosine `T_max` from the *current* --max_steps so
            # that changing --max_steps between chained restarts actually
            # retargets the LR schedule. Without this, load_state_dict
            # restores the OLD T_max baked into the checkpoint and edits to
            # --max_steps have no effect on the cosine decay length.
            fresh_cosine_T_max = max(args.max_steps - args.warmup_steps, 1)
            for sub in scheduler._schedulers:
                if isinstance(sub, torch.optim.lr_scheduler.CosineAnnealingLR) \
                        and sub.T_max != fresh_cosine_T_max:
                    logger.info(
                        f"Cosine T_max retargeted: {sub.T_max} → "
                        f"{fresh_cosine_T_max} (from --max_steps={args.max_steps})"
                    )
                    sub.T_max = fresh_cosine_T_max
                    sub.eta_min = args.min_lr

            # CRITICAL (2026-05-19): PyTorch's CosineAnnealingLR uses a
            # recurrence formula that reads opt.param_groups[*]['lr']
            # when computing the next step's lr. After load_state_dict,
            # the scheduler's INTERNAL _last_lr is correctly restored,
            # but the optimizer's lr is whatever SequentialLR.__init__
            # set it to (= LinearLR warmup-iter-0 ≈ 5e-7 = base_lr ×
            # start_factor). When we skip opt-state restore on spectro
            # surgery, this stale lr in opt stays. The next
            # scheduler.step() then computes new lr via the recurrence
            # off that 5e-7, perpetuating the stuck-at-warmup-zero
            # value. Symptom: 4581033/34/35 trained at LR≈5e-7 for
            # hours — useless. Fix: sync opt's lr to scheduler's
            # restored _last_lr immediately after load.
            for pg, lr in zip(opt.param_groups, scheduler.get_last_lr()):
                pg["lr"] = lr
            logger.info(
                f"Synced optimizer lr from scheduler.get_last_lr(): "
                f"{[f'{lr:.2e}' for lr in scheduler.get_last_lr()]}"
            )
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
        old_n_layers, backbone_extra_allowed = warm_start_extend_backbone(
            _core(model), init_ckpt["model_state_dict"], args.n_layers,
        )
        if backbone_extra_allowed:
            logger.info(
                f"Warm-start: extending backbone from {old_n_layers} to "
                f"{args.n_layers} layers; new blocks "
                f"[{old_n_layers}, {args.n_layers}) initialised as "
                "near-identity (zero attn.out_proj + mlp final linear)."
            )
            allowed_missing = allowed_missing + backbone_extra_allowed
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
    # Auto-inject a freeze on backbone/slow_ts/video when the resume path
    # detected spectro model surgery.
    #
    # 2026-05-19 EMERGENCY DISABLE: 4581033 crashed with
    #   RuntimeError: Expected to have finished reduction in the prior
    #   iteration before starting a new one. Parameters that were not
    #   used in producing loss.
    # because DDP's reducer is built at `dm.wrap(model)` time (~line 1048)
    # BEFORE this freeze block runs. When we then flip requires_grad=False
    # on backbone/slow_ts/video, the reducer still expects gradients for
    # them — DDP errors on the first backward.
    #
    # The proper fix is to apply the freeze BEFORE dm.wrap() (requires
    # peeking the checkpoint to detect surgery early). For now we disable
    # the auto-freeze so production keeps running. The new refine blocks
    # train from fresh init alongside the existing trained backbone —
    # loss may spike briefly but should recover.
    #
    # TODO: refactor to detect surgery + apply freeze before DDP wrap,
    # then re-enable this block.
    auto_freeze_release_step = 0
    _ = (reinit_spectro_modalities, spectro_refine_extra_modalities)  # keep refs
    if reinit_spectro_modalities or spectro_refine_extra_modalities:
        logger.info(
            "Auto-freeze DISABLED (2026-05-19 emergency patch): "
            f"detected reinit_spectro={reinit_spectro_modalities}, "
            f"refine_extra={spectro_refine_extra_modalities}, but "
            "applying the freeze post-wrap triggers a DDP "
            "unused-parameters error. New refine blocks train from "
            "fresh init alongside the existing trained backbone."
        )
    # Back-compat: --freeze_ts_steps applies to both slow_ts and fast_ts
    # unless their per-kind flags are set explicitly.
    args_freeze_slow_ts = max(args.freeze_slow_ts_steps, args.freeze_ts_steps)
    args_freeze_fast_ts = max(args.freeze_fast_ts_steps, args.freeze_ts_steps)
    if args.freeze_whole_run:
        # Freezes were applied pre-wrap (see model construction) and are
        # permanent — no step-based application or release here.
        logger.info(
            "freeze_whole_run: skipping step-based freeze/release logic."
        )
        freeze_specs = []
    else:
        freeze_specs = [
            ("slow_ts", max(args_freeze_slow_ts, auto_freeze_release_step)),
            # fast_ts NOT auto-frozen — has its own fresh-init refine blocks 2-3
            ("fast_ts", args_freeze_fast_ts),
            ("video", max(args.freeze_video_steps, auto_freeze_release_step)),
            ("spectro", args.freeze_spectro_steps),
            ("backbone", max(args.freeze_backbone_steps, auto_freeze_release_step)),
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
            loss, per_mod = compute_step_loss(
                model, batch, device, spec_pb_weights=spec_pb_weights,
            )
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
                n_unfrozen = _release_module_freeze(_core(model), **kwargs)
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
            # Host-RAM trajectory: chained jobs OOM'd reproducibly at
            # ~9h45m with no sampler logs. Inline psutil reading gives
            # us per-step RAM% directly in the .err file so we can
            # diagnose without re-submitting (and without losing the
            # priority boost). Cheap: one syscall per log_every steps.
            vm = psutil.virtual_memory()
            logger.info(
                f"  host_ram used={vm.used/1e9:.1f}GB "
                f"available={vm.available/1e9:.1f}GB "
                f"percent={vm.percent:.1f}%"
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
                use_amp=use_amp_val,
            )
            logger.info(
                "Validation (MAE model vs copy; delta-ratio pred/tgt):"
            )
            for n in diagnostic_names:
                m = metrics[n]
                delta = m["model_mae"] - m["copy_mae"]
                marker = "↓" if delta < 0 else "↑"
                tvr = m.get("tvr", float("nan"))
                tvr_str = f"  tvr={tvr:.3f}" if not math.isnan(tvr) else ""
                logger.info(
                    f"  {n:<25s} "
                    f"model={m['model_mae']:.4f}  copy={m['copy_mae']:.4f}  "
                    f"{marker} {abs(delta):.4f}  | "
                    f"pred_d={m['pred_delta']:.4f}  tgt_d={m['tgt_delta']:.4f}  "
                    f"ratio={m['delta_ratio']:.3f}{tvr_str}"
                )
            val_loss = sum(metrics[n]["model_mae"] for n in diagnostic_names)
            logger.info(f"  [sum model MAE] {val_loss:.4f}")
            # Checkpoint-selection scalar. With --collapse_aware_best, penalise
            # spectro modalities whose temporal-variance ratio is below 1
            # (i.e. mean-collapsed) so a low-MAE collapse can't win best.pt.
            sel_loss = val_loss
            if args.collapse_aware_best:
                tvr_penalty = sum(
                    max(0.0, 1.0 - metrics[n]["tvr"])
                    for n in diagnostic_names
                    if not math.isnan(metrics[n].get("tvr", float("nan")))
                )
                sel_loss = val_loss + args.collapse_aware_lambda * tvr_penalty
                logger.info(
                    f"  [collapse-aware] sel_loss={sel_loss:.4f} "
                    f"(tvr_penalty={tvr_penalty:.4f}, "
                    f"lambda={args.collapse_aware_lambda})"
                )
            # Decide best-update first so both `latest` and `best` share the
            # same final best_val_loss / best_step values — otherwise resume
            # from `latest` would see a stale best.
            is_new_best = sel_loss < best_val_loss
            if is_new_best:
                best_val_loss = sel_loss
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
