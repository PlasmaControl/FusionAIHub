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
import gc
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
from tokamak_foundation_model.e2e.output_heads import (
    FastTimeSeriesCodeHead,
    SlowTimeSeriesCodeHead,
    SpectrogramCodeHead,
    SpectrogramMaskGITHead,
    SpectrogramFlowHead,
    VideoCodeHead,
    VideoFlowHead,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout
from tokamak_foundation_model.utils.distributed import DistributedManager

logger = logging.getLogger("e2e_stage1")


def _trim_host_ram() -> None:
    """Return freed host memory to the OS. glibc keeps freed allocations in the
    process arena (RSS / psutil-used stays high) — on RESUME the ~21GB checkpoint
    + 10.7GB optimizer CPU-copy per rank are freed but NOT returned, so resumes
    start ~180GB/node above the cold baseline (72% vs 38%) and host-OOM at ~3h50m
    while the cold job TIMEOUTs clean. malloc_trim(0) hands the arena back to the
    OS. Throughput-neutral (one-time: at resume + after the first opt.step)."""
    import ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


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
    # ech_tor_angle / ech_pol_angle / ech_polarization DROPPED 2026-07-14 (GATE3-FIX): the ECCD
    # aiming angles are identically ZERO corpus-wide (dataset gap) → constant-zero dead-weight inputs.
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
    # tangtv split into the two divertor views, each its OWN tokenizer+head.
    # Only LIVE channels kept (ch1/3/5 dead in all shots): lower={ch0,ch2},
    # upper={ch4,ch6} → 2 channels each.
    ("tangtv_lower", 2, 3, (120, 360), (3, 12, 12)),
    ("tangtv_upper", 2, 3, (120, 360), (3, 12, 12)),
]

# Video modality name -> the HDF5 group it actually reads. The split divertor
# views both read the single "tangtv" group, so the video-presence filter must
# check "tangtv" (not the non-existent per-view group names). Mapping to the base
# group also reuses the existing video_present_*.pt cache (keyed on "tangtv").
_VIDEO_HDF5_GROUP = {"tangtv_lower": "tangtv", "tangtv_upper": "tangtv"}


def _video_hdf5_groups(use_video: List[str]) -> List[str]:
    return sorted({_VIDEO_HDF5_GROUP.get(n, n) for n in use_video})

# Per-modality spectrogram registry. Each entry is
# ``(name, n_channels, (F_p, T_p))``. STFT shape is fixed by the data
# loader (n_fft=1024, hop=256, fs=500 kHz) so freq_bins=512, time_frames=98
# for the canonical 50 ms window. Only included when the user passes
# ``--use_spectro <name> [<name> ...]``; empty default keeps Phase A
# byte-identical (G2/G3).
SPECTRO_FREQ_BINS = 512
SPECTRO_TIME_FRAMES = 98            # canonical 50 ms window (chunk 0.05 s)
# STFT frame rate: data_loader uses torch.stft center=True → n_frames = T//hop+1,
# with the spectro raw fs=500 kHz and hop=256 → ~1953 frames/s. Deriving the
# frame count from chunk_duration lets the backbone take a LONGER input window
# (multi-window temporal history → it can observe mode-amplitude VELOCITY, the
# single-window Markov limitation). round(0.05*500000/256)=98 (matches).
SPECTRO_STFT_FS = 500_000
SPECTRO_STFT_HOP = 256


def spectro_time_frames(chunk_duration_s: float) -> int:
    return round(chunk_duration_s * SPECTRO_STFT_FS / SPECTRO_STFT_HOP)
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
    prediction_horizon_s: Optional[float] = None,
) -> Tuple[List[DiagnosticConfig], List[ActuatorConfig]]:
    slow_samples = round(chunk_duration_s * SLOW_FS)
    fast_samples = round(chunk_duration_s * FAST_FS)
    # Actuator tokens span the PREDICTION HORIZON (the future actions the world model
    # conditions on to forecast), NOT the input chunk. These coincide only when
    # horizon==chunk (the historical 0.05==0.05 default, so this is byte-identical there);
    # at a longer horizon the actuator window MUST scale, else the tokenizer's conv emits
    # (horizon/chunk)x more tokens than patch_pos (the t+4 warm-start crash, 2026-07-14).
    act_samples = round((prediction_horizon_s if prediction_horizon_s else chunk_duration_s) * FAST_FS)
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
                    window_samples=spectro_time_frames(chunk_duration_s),
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
        ActuatorConfig(name, n_channels, act_samples, n_tokens=5)
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
    history_windows: int = 1,
    val_prediction_horizon_s: Optional[float] = None,
) -> Tuple[TokamakMultiFileDataset, TokamakMultiFileDataset]:
    """Construct Stage 1 train + val datasets.

    Diagnostics are in both ``input_signals`` and ``target_signals`` so the
    loader returns input (t) and target (t+50 ms) halves. Actuators are in
    ``target_signals`` only so we receive the actuator commands driving
    the step-1 transition.

    ``val_prediction_horizon_s`` (default ``None`` → same as
    ``prediction_horizon_s``, byte-identical) lets the K-rollout trainer widen
    the TRAIN future span (many rollout windows) while keeping the VAL span at
    the model horizon, so ``validate()`` stays a single-step eval (the actuator
    tokenizer geometry expects the model horizon; a wide val batch would feed it
    too many patches). Rollout quality is measured per-block by gate4_kprobe.
    """
    input_signals = diagnostic_names
    target_signals = diagnostic_names + actuator_names
    val_horizon = (
        prediction_horizon_s if val_prediction_horizon_s is None
        else val_prediction_horizon_s
    )

    lengths_cache_dir.mkdir(parents=True, exist_ok=True)
    shared = dict(
        chunk_duration_s=chunk_duration_s,
        prediction_mode=True,
        step_size_s=step_size_s,
        warmup_s=warmup_s,
        preprocessing_stats=preprocessing_stats,
        input_signals=input_signals,
        target_signals=target_signals,
        max_open_files=1024,
        history_windows=history_windows,
    )
    train_ds = TokamakMultiFileDataset(
        train_files,
        lengths_cache_path=lengths_cache_dir / "lengths_e2e_stage1_train.pt",
        prediction_horizon_s=prediction_horizon_s,
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=lengths_cache_dir / "lengths_e2e_stage1_val.pt",
        prediction_horizon_s=val_horizon,
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


def build_spec_mode_band_weights(
    diagnostics: List[DiagnosticConfig],
    factor: float,
    device,
    lo_khz: float = 5.0,
    hi_khz: float = 40.0,
    fs: float = 500e3,
    nfft: int = 1024,
) -> Dict[str, torch.Tensor]:
    """Per-modality ``(C, F)`` MAE weight that UP-weights the coherent-mode
    band (``lo_khz``–``hi_khz``, default 5–40 kHz) by ``factor``, 1 elsewhere.

    The plain/per-bin MAE is dominated by the DC/broadband envelope, so the
    thin coherent mode (a few % of the spectrogram energy) gets almost no
    gradient → the head ignores it (mean-collapse / amplitude-undershoot). This
    focuses the loss on the mode band so the head is actually penalised for
    missing the ridge. Uniform across channels; broadcast as (1,C,F,1) by
    :func:`weighted_masked_mae`.
    """
    khz_per_bin = fs / nfft / 1e3
    out: Dict[str, torch.Tensor] = {}
    for cfg in diagnostics:
        if cfg.kind != "spectrogram" or cfg.freq_bins is None:
            continue
        F = int(cfg.freq_bins)
        lo = max(0, int(lo_khz / khz_per_bin))
        hi = min(F, int(hi_khz / khz_per_bin))
        w = torch.ones(cfg.n_channels, F, device=device)
        w[:, lo:hi] = float(factor)
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


_BG_RESIDUAL_FN = None


def _bg_residual_fn():
    """Cached, path-robust handle to ``spectro_bg.baseline_residual_torch``.

    ``spectro_bg`` is a sibling script (not the installed package); this resolves
    it whether the trainer runs as ``__main__`` or is imported by an eval script,
    without touching the module-level import block."""
    global _BG_RESIDUAL_FN
    if _BG_RESIDUAL_FN is None:
        import os
        import sys
        d = os.path.dirname(os.path.abspath(__file__))
        if d not in sys.path:
            sys.path.insert(0, d)
        from spectro_bg import baseline_residual_torch
        _BG_RESIDUAL_FN = baseline_residual_torch
    return _BG_RESIDUAL_FN


def _spectro_head_bg(model, name):
    """(bg_subtract, bg_sigma) for a spectro modality's code head, or (False, 8.0).

    Residual behavior is self-declared by the frozen codec (cfg["bg_subtract"]),
    surfaced on :class:`SpectrogramCodeHead` — so pointing the run at a residual
    codec dir is sufficient; nothing else in the launcher changes."""
    heads = getattr(_core(model), "diag_heads", None)
    head = heads[name] if (heads is not None and name in heads) else None
    return bool(getattr(head, "bg_subtract", False)), float(getattr(head, "bg_sigma", 8.0))


def forward_batch(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
    act_perturb: Optional[Dict[str, float]] = None,
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
            # Residual-codec: split off the smooth per-freq baseline so the
            # backbone input tokenizer sees only R = S - B (the modes/transients
            # on a flat background). Same operator the residual codec was
            # trained with; matched on the target below so both encode R.
            _bg, _sig = _spectro_head_bg(model, cfg.name)
            if _bg:
                _, cleaned = _bg_residual_fn()(cleaned, _sig)
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
        if act_perturb and cfg.name in act_perturb:
            # GATE-3 counterfactual: add a sustained +Δ (in dataset-standardized units) to this
            # actuator's trajectory over the forecast window. Default None → byte-identical.
            cleaned = cleaned + float(act_perturb[cfg.name])
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
            # MULTI-HORIZON (Gate 2b): the loader hands the FULL future (K windows). Normally we
            # truncate the target to one window (trunc_t). When a descriptor head forecasts t+h,
            # KEEP up to max(horizons) windows so the descriptor can read its t+h sub-windows;
            # compute_step_loss slices the base target back to one window (=pred width). No
            # descriptor / single-horizon → _kh=1 → keep exactly trunc_t (byte-identical to before).
            _kh = 1
            _dhs = getattr(_core(model), "spec_descriptor_heads", {})
            if cfg.name in _dhs:
                _kh = max(getattr(_dhs[cfg.name], "horizons", (1,)))
            targets[cfg.name] = targets[cfg.name][..., :trunc_t * _kh]
            # Residual-codec: encode_target must see the SAME R-space as the
            # input above, so the CE targets are residual codes (modes), not
            # full-spectrogram codes (broadband-dominated → mode collapse).
            _bg, _sig = _spectro_head_bg(model, cfg.name)
            if _bg:
                _, targets[cfg.name] = _bg_residual_fn()(targets[cfg.name], _sig)
            masks[cfg.name] = _spectro_loss_gate(cfg, batch, device)
        else:
            mask_key = f"{cfg.name}_mask"
            masks[cfg.name] = (
                batch["targets"][mask_key].to(device, non_blocking=True).float()
                if mask_key in batch["targets"]
                else None
            )
    return predictions, diag_inputs, targets, masks, diag_token_slices


@torch.no_grad()
def build_video_pixel_sigma(core, loader, device, n_batches=8):
    """Per-pixel residual-scale σ for VideoFlowHead modalities, estimated from a
    short pass over the train loader (masked target std per pixel, in the SAME
    per-(B,C) standardised frame the loss uses). Returned T-major as
    ``(C·T, H, W)`` to match :meth:`VideoFlowHead._fold`; all-reduced so every
    DDP rank gets the same σ. ``{}`` if no generative video heads.

    The σ is the video analog of the spectrogram per-bin σ: it confines the flow
    noise to where the target actually varies (clean quiet background)."""
    import torch.distributed as dist
    vids = [
        cfg for cfg in core.diagnostics
        if cfg.kind == "video" and isinstance(core.diag_heads[cfg.name], VideoFlowHead)
    ]
    if not vids:
        return {}
    acc: Dict[str, list] = {}
    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        for cfg in vids:
            # Replicate forward_batch's per-(B,C) standardisation: stats from the
            # INPUT window, applied to the TARGET (so σ lives in the loss frame).
            raw_in = batch["inputs"][cfg.name].to(device, non_blocking=True).float()
            cleaned_in, _ = _clean_and_mask(raw_in, None)
            _, mu, sd = _video_standardize_per_bc(cleaned_in)
            tgt = batch["targets"][cfg.name].to(device, non_blocking=True).float()
            tgt = (tgt - mu) / sd                          # (B,C,T,H,W)
            m = _video_loss_gate(cfg, batch, device).expand_as(tgt)
            s = (tgt * m).sum(dim=0)
            ss = (tgt * tgt * m).sum(dim=0)
            cnt = m.sum(dim=0)
            if cfg.name not in acc:
                acc[cfg.name] = [s, ss, cnt]
            else:
                acc[cfg.name][0] += s; acc[cfg.name][1] += ss; acc[cfg.name][2] += cnt
        seen += 1
    out: Dict[str, torch.Tensor] = {}
    for name, (s, ss, cnt) in acc.items():
        if dist.is_available() and dist.is_initialized():
            for t in (s, ss, cnt):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        cntc = cnt.clamp_min(1.0)
        var = (ss / cntc - (s / cntc) ** 2).clamp_min(0.0)
        std = var.sqrt()                                    # (C,T,H,W)
        C, T, H, W = std.shape
        out[name] = std.permute(1, 0, 2, 3).reshape(T * C, H, W).clamp_min(0.05)
    return out


def build_spec_code_class_weights(core, loader, device, cap, n_batches=50):
    """Per-(dim, level) inverse-frequency CE class weights for every
    :class:`SpectrogramCodeHead`, estimated from a short pass over the train
    loader. Encodes the (dataset-normalized, time-truncated) TARGET
    spectrograms through each FROZEN codec, counts per-dim FSQ level
    frequencies, and returns capped inverse-frequency weights normalized so
    ``E_data[w]=1`` — so the rare MODE codes are up-weighted against the
    frequent background code (avoids the categorical majority-class collapse
    the POC observed). Counts are all-reduced so every DDP rank gets IDENTICAL
    weights. Returns ``{}`` when there is no FSQ head or ``cap<=1`` (uniform CE).

    The spectro target needs NO per-(B,C) z-score (unlike video): the dataset
    already log-standardised it — ``forward_batch`` only time-truncates it — so
    this is the exact space the codec (and ``head.encode_target``) expects."""
    import torch.distributed as dist
    fsq = [
        cfg for cfg in core.diagnostics
        if isinstance(core.diag_heads[cfg.name], SpectrogramCodeHead)
    ]
    if not fsq or cap <= 1.0:
        return {}
    counts = {
        cfg.name: torch.zeros(
            core.diag_heads[cfg.name].dim,
            core.diag_heads[cfg.name].levels, device=device,
        )
        for cfg in fsq
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= n_batches:
                break
            for cfg in fsq:
                head = core.diag_heads[cfg.name]
                _, T_p = cfg.spectrogram_patch_size
                trunc_t = (cfg.window_samples // T_p) * T_p
                tgt = batch["targets"][cfg.name].to(
                    device, non_blocking=True
                ).float()[..., :trunc_t]
                codes = head.encode_target(tgt)                 # (B,n_tok,dim)
                oneh = F.one_hot(codes, head.levels).float()    # (B,n_tok,dim,L)
                counts[cfg.name] += oneh.sum(dim=(0, 1))        # (dim,L)
            seen += 1
    out: Dict[str, torch.Tensor] = {}
    for cfg in fsq:
        c = counts[cfg.name]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
        freq = c / (c.sum(dim=1, keepdim=True) + 1e-8)          # (dim,L) per-dim freq
        cw = 1.0 / (freq + 1e-4)
        norm = (freq * cw).sum(dim=1, keepdim=True) + 1e-8      # E_data[w]=1
        cw = (cw / norm).clamp(max=cap)
        out[cfg.name] = cw.detach().cpu()
    return out


def build_video_code_class_weights(core, loader, device, cap, n_batches=50):
    """Per-(dim, level) inverse-frequency CE class weights for every
    :class:`VideoCodeHead` — video analog of :func:`build_spec_code_class_weights`.
    Encodes the TARGET video (per-(B,C) z-scored the SAME way ``forward_batch``
    does — stats from the input window) through each frozen video codec, counts
    per-dim level frequencies, returns capped inverse-freq weights (E_data[w]=1),
    all-reduced. ``{}`` when no video FSQ head or ``cap<=1``."""
    import torch.distributed as dist
    fsq = [
        cfg for cfg in core.diagnostics
        if isinstance(core.diag_heads[cfg.name], VideoCodeHead)
    ]
    if not fsq or cap <= 1.0:
        return {}
    counts = {
        cfg.name: torch.zeros(core.diag_heads[cfg.name].dim,
                              core.diag_heads[cfg.name].levels, device=device)
        for cfg in fsq
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= n_batches:
                break
            for cfg in fsq:
                head = core.diag_heads[cfg.name]
                # replicate forward_batch: input-window per-(B,C) stats → target
                raw_in = batch["inputs"][cfg.name].to(device, non_blocking=True).float()
                cleaned_in, _ = _clean_and_mask(raw_in, None)
                _, mu, sd = _video_standardize_per_bc(cleaned_in)
                tgt = batch["targets"][cfg.name].to(device, non_blocking=True).float()
                tgt = (tgt - mu) / sd                            # (B,C,T,H,W)
                # Truncate to the codec's single-window frame count BEFORE encoding
                # (mirror build_spec_code_class_weights' [..., :trunc_t]). When
                # prediction_horizon_s > the codec's 0.05s design window (e.g. the
                # rollout-native 0.2s horizon → 5× frames), the full target spans
                # multiple codec windows → n_t>1 → spatial_pe (300 tok) shape
                # mismatch. The per-step rollout loss already feeds ONE subwindow;
                # match the class-weight statistics to that same first subwindow.
                _nf = int(getattr(head.codec.enc, "n_frames", tgt.shape[2]))
                if tgt.shape[2] > _nf:
                    tgt = tgt[:, :, :_nf]                         # (B,C,n_frames,H,W)
                codes = head.encode_target(tgt)                  # (B,n_tok,dim)
                oneh = F.one_hot(codes, head.levels).float()
                counts[cfg.name] += oneh.sum(dim=(0, 1))
            seen += 1
    out: Dict[str, torch.Tensor] = {}
    for cfg in fsq:
        c = counts[cfg.name]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
        freq = c / (c.sum(dim=1, keepdim=True) + 1e-8)
        cw = 1.0 / (freq + 1e-4)
        norm = (freq * cw).sum(dim=1, keepdim=True) + 1e-8
        out[cfg.name] = (cw / norm).clamp(max=cap).detach().cpu()
    return out


def build_fastts_code_class_weights(core, loader, device, cap, n_batches=50):
    """Per-(dim, level) inverse-frequency CE class weights for every
    :class:`FastTimeSeriesCodeHead` — fast-TS analog of the spectro/video versions.
    Encodes the TARGET filterscopes (per-(window, channel) z-scored — the codec's
    training space, matching the POC ``load_fastts_windows``) through each frozen
    codec, counts per-dim level frequencies, returns capped inverse-freq weights
    (E_data[w]=1), all-reduced. ``{}`` when no fast-TS FSQ head or ``cap<=1``."""
    import torch.distributed as dist
    fsq = [
        cfg for cfg in core.diagnostics
        if isinstance(core.diag_heads[cfg.name], FastTimeSeriesCodeHead)
    ]
    if not fsq or cap <= 1.0:
        return {}
    counts = {
        cfg.name: torch.zeros(core.diag_heads[cfg.name].dim,
                              core.diag_heads[cfg.name].levels, device=device)
        for cfg in fsq
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= n_batches:
                break
            for cfg in fsq:
                head = core.diag_heads[cfg.name]
                tgt = batch["targets"][cfg.name].to(device, non_blocking=True).float()
                tgt = torch.nan_to_num(tgt)                      # (B,C,WIN)
                mu = tgt.mean(dim=-1, keepdim=True)
                sd = tgt.std(dim=-1, keepdim=True).clamp(min=1e-3)
                codes = head.encode_target((tgt - mu) / sd)      # (B,n_tok,dim)
                oneh = F.one_hot(codes, head.levels).float()
                counts[cfg.name] += oneh.sum(dim=(0, 1))
            seen += 1
    out: Dict[str, torch.Tensor] = {}
    for cfg in fsq:
        c = counts[cfg.name]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
        freq = c / (c.sum(dim=1, keepdim=True) + 1e-8)
        cw = 1.0 / (freq + 1e-4)
        norm = (freq * cw).sum(dim=1, keepdim=True) + 1e-8
        out[cfg.name] = (cw / norm).clamp(max=cap).detach().cpu()
    return out


def build_slowts_code_class_weights(core, loader, device, cap, n_batches=50):
    """Per-(dim, level) inverse-frequency CE class weights for every
    :class:`SlowTimeSeriesCodeHead`. Slow-TS codecs train in the DATASET-standardized
    space, so the target is encoded AS-IS (no re-normalization, unlike fast-TS/video)."""
    import torch.distributed as dist
    fsq = [
        cfg for cfg in core.diagnostics
        if isinstance(core.diag_heads[cfg.name], SlowTimeSeriesCodeHead)
    ]
    if not fsq or cap <= 1.0:
        return {}
    counts = {
        cfg.name: torch.zeros(core.diag_heads[cfg.name].dim,
                              core.diag_heads[cfg.name].levels, device=device)
        for cfg in fsq
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= n_batches:
                break
            for cfg in fsq:
                head = core.diag_heads[cfg.name]
                tgt = torch.nan_to_num(batch["targets"][cfg.name].to(device, non_blocking=True).float())
                codes = head.encode_target(tgt)                  # (B,n_tok,dim)
                oneh = F.one_hot(codes, head.levels).float()
                counts[cfg.name] += oneh.sum(dim=(0, 1))
            seen += 1
    out: Dict[str, torch.Tensor] = {}
    for cfg in fsq:
        c = counts[cfg.name]
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
        freq = c / (c.sum(dim=1, keepdim=True) + 1e-8)
        cw = 1.0 / (freq + 1e-4)
        norm = (freq * cw).sum(dim=1, keepdim=True) + 1e-8
        out[cfg.name] = (cw / norm).clamp(max=cap).detach().cpu()
    return out


# --------------------------------------------------------------------------- #
# (A) Structural mode-coherence loss for generative spectrogram heads.         #
# EXACT mirror of the production GT-fusion binarization (eval_e2e_animation_   #
# tokamak.fuse_spectro_with_gt, the rule that successfully extracts modes on   #
# shot 200729): gaussian-smooth (sigma_f, sigma_t) -> per-FREQUENCY background #
# mu/sd over TIME -> soft = clip((smooth-mu)/(k*sd), 0, 1)^gamma. The rule is  #
# SELF-NORMALIZING (mu/sd from the input), hence invariant to the per-channel  #
# log_standardize of the model space. A soft-Dice between mode_soft(mu_pred)   #
# and mode_hard(target) rewards sharp coherent ridges over a blurry envelope.  #
# Adds NO parameters and runs every step on mu (which already has grads via    #
# mae+flow) -> DDP-safe + warm-start-safe. Default-OFF (lambda 0.0).           #
# --------------------------------------------------------------------------- #
_SPEC_STRUCT_K = {"ece": 2.5, "co2": 2.0, "bes": 2.0}   # per-modality k
_SPEC_STRUCT_GAMMA = 2.0
_SPEC_STRUCT_SMOOTH_F = 1.0
_SPEC_STRUCT_SMOOTH_T = 2.0
_SPEC_STRUCT_CUT = 0.5


def _spec_gauss1d(sigma: float, device, dtype):
    r = max(1, int(round(3 * sigma)))
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()), r


def _spec_gauss_smooth(x: torch.Tensor, sf: float, st: float) -> torch.Tensor:
    """Separable gaussian blur over (F, T). x: (B, C, F, T)."""
    B, C, Fb, T = x.shape
    kf, rf = _spec_gauss1d(sf, x.device, x.dtype)
    kt, rt = _spec_gauss1d(st, x.device, x.dtype)
    xr = x.reshape(B * C, 1, Fb, T)
    xr = F.conv2d(xr, kf.view(1, 1, -1, 1), padding=(rf, 0))
    xr = F.conv2d(xr, kt.view(1, 1, 1, -1), padding=(0, rt))
    return xr.reshape(B, C, Fb, T)


def _spec_mode_arg(x: torch.Tensor, k: float) -> torch.Tensor:
    """soft_mask argument (smooth-mu)/(k*sd); per-freq mu/sd over TIME."""
    sm = _spec_gauss_smooth(x, _SPEC_STRUCT_SMOOTH_F, _SPEC_STRUCT_SMOOTH_T)
    mu = sm.mean(dim=-1, keepdim=True)
    sd = sm.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (sm - mu) / (k * sd)


def spectro_struct_loss(
    mu_pred: torch.Tensor, target: torch.Tensor, k: float,
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Soft-Dice mode-coherence loss (A). 0 = mu's modes match GT's modes."""
    soft = _spec_mode_arg(mu_pred, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
    with torch.no_grad():
        hard = (
            (_spec_mode_arg(target, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA)
            > _SPEC_STRUCT_CUT
        ).float()
    if gate is not None:                                # (B,1,1,1) presence
        # BINARIZE: the gate is a presence/frame-COUNT (>1), not 0/1. Multiplying
        # soft/hard by a raw count breaks the soft-Dice (num∝g², den∝g →
        # num/den∝g≫1 → 1-num/den goes large NEGATIVE; observed ece_struct≈-12).
        # >0 → present(1)/absent(0) keeps the Dice in [0,1].
        g = (gate > 0).to(soft.dtype)
        soft = soft * g
        hard = hard * g
    num = 2.0 * (soft * hard).sum() + 1.0
    den = soft.sum() + hard.sum() + 1.0
    return 1.0 - num / den


def spectro_mask_loss(
    logits: torch.Tensor, target: torch.Tensor, k: float,
    gate: Optional[torch.Tensor] = None, bce_weight: float = 0.0,
    loss_type: str = "dice", tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(Plan B) Segmentation loss for the predicted mode mask.

    **DICE-ONLY by default (bce_weight=0).** With input-conditioning, BCE is
    HARMFUL: its confident-false-positive penalty (input modes that don't persist
    to the output, ~36 %) drives the persistence prior-gain toward 0 → the prior
    is abandoned → the mask collapses to empty (measured: d(loss)/d(gain) at
    persistence = +0.32 with BCE vs −0.04 dice-only; jobs 4922044→4923929 all
    collapsed to maskdice≈0.05 with BCE on). Dice rewards overlap without the
    per-pixel confident-wrong term, so it HOLDS persistence (maskdice ~0.64).

    The overfit-prediction test showed μ (MAE) and the flow sample (velocity
    MSE) both collapse to a smooth envelope — both are L2, whose optimum is the
    conditional mean. This trains a SEPARATE predicted mask (``sigmoid(logits)``,
    ``(B,C,F,T)``) toward the production-binarized GT mode field via **soft-Dice
    + BCE**, neither of which has a mean-seeking optimum, so it does not
    collapse. Target = the SAME rule as ``fuse_spectro_with_gt`` / the struct
    loss (gauss-smooth → per-freq z over time → ``clip(z/k,0,1)^γ``), so the
    predicted mask is a drop-in for the GT mask at render — a genuine forecast.

    Returns ``(loss, maskdice)`` where ``maskdice`` ∈ [0,1] is the hard overlap
    (pred>0.5 vs GT>0.5) — the metric to watch the mask head LEARN the modes.
    Gate (presence/frame-count, broadcastable to (B,C,F,T)) is binarized; an
    absent modality contributes 0 loss but the logits still enter the graph
    (the mask params get a 0 grad → DDP-safe, no unused parameters).
    """
    with torch.no_grad():
        t_soft = (
            _spec_mode_arg(target, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
        )
    logits = logits.float()
    p = torch.sigmoid(logits)
    if gate is not None:
        g = (gate > 0).to(p.dtype)                  # presence (B,1,1,1)
        p = p * g
        t_soft = t_soft * g
        bce_w = g.expand_as(logits)
        denom = bce_w.sum().clamp_min(1.0)
    else:
        bce_w = None
        denom = torch.tensor(float(logits.numel()), device=logits.device)
    if loss_type == "tversky":
        # Tversky: TP/(TP + α·FP + β·FN). β>α penalizes MISSED modes (FN) more
        # than false positives → drives recall of the sparse (~5 %) mode pixels,
        # with a stronger low-overlap gradient than dice (which stalls near-empty
        # → the ~0.1 plateau). α=0.3, β=0.7.
        tp = (p * t_soft).sum()
        fp = (p * (1.0 - t_soft)).sum()
        fn = ((1.0 - p) * t_soft).sum()
        seg = 1.0 - (tp + 1.0) / (tp + tversky_alpha * fp + tversky_beta * fn + 1.0)
    else:
        # soft-Dice (handles the ~5 % mode-pixel imbalance)
        num = 2.0 * (p * t_soft).sum() + 1.0
        den = p.sum() + t_soft.sum() + 1.0
        seg = 1.0 - num / den
    dice = seg
    # gated per-pixel BCE. loss_type="sparse" → POS-WEIGHTED BCE (weight the mode
    # class by ~1/density) giving a ~40× stronger gradient on the sparse missed
    # modes than dice (which stalls near-empty → the ~0.1 plateau). Verified
    # offline. No persistence prior in this regime, so BCE is safe (its earlier
    # collapse was prior-specific). with_logits → stable + keeps `logits` in the
    # graph even when the modality is absent (gate=0) → DDP-safe.
    pw = None
    if loss_type == "sparse":
        with torch.no_grad():
            pos = t_soft.sum().clamp_min(1.0)
            tot = (bce_w.sum() if bce_w is not None
                   else torch.tensor(float(logits.numel()), device=logits.device))
            pw = ((tot - pos) / pos).clamp(1.0, 50.0)
        bce_weight = 1.0
    bce_map = F.binary_cross_entropy_with_logits(
        logits, t_soft, pos_weight=pw, reduction="none"
    )
    if bce_w is not None:
        bce_map = bce_map * bce_w
    bce = bce_map.sum() / denom
    loss = dice + bce_weight * bce
    with torch.no_grad():
        ph = (p > 0.5).float()
        th = (t_soft > 0.5).float()
        maskdice = (2.0 * (ph * th).sum() + 1.0) / (ph.sum() + th.sum() + 1.0)
    return loss, maskdice


def compute_step_loss(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
    spec_pb_weights: Optional[Dict[str, torch.Tensor]] = None,
    spec_struct_lambda: float = 0.0,
    spec_mask_lambda: float = 0.0,
    spec_mae_lambda: float = 1.0,
    spec_mask_loss_type: str = "dice",
    spec_code_class_weights: Optional[Dict[str, torch.Tensor]] = None,
    spec_code_focal_gamma: float = 0.0,
    spec_ordinal_eps: float = 0.0,
    spec_autoencode: bool = False,
    loss_norm_ema: bool = False,
    loss_norm_beta: float = 0.99,
    loss_priority: Optional[Dict[str, float]] = None,
    video_code_class_weights: Optional[Dict[str, torch.Tensor]] = None,
    fastts_code_class_weights: Optional[Dict[str, torch.Tensor]] = None,
    slow_ts_code_class_weights: Optional[Dict[str, torch.Tensor]] = None,
    spec_descriptor_weight: float = 4.0,
    spec_descriptor_loss: str = "mse",
    spec_descriptor_dist_beta: float = 2.0,
    spec_descriptor_anchor: bool = False,
    spec_descriptor_anchor_beta: Optional[float] = None,
    spec_descriptor_transition_weight: float = 1.0,
    drift_penalty_weight: float = 0.0,
    n_subwindows: int = 1,
    precomputed: Optional[
        Tuple[
            Dict[str, torch.Tensor],            # predictions
            Dict[str, torch.Tensor],            # diag_inputs
            Dict[str, torch.Tensor],            # targets
            Dict[str, Optional[torch.Tensor]],  # masks
            Dict[str, torch.Tensor],            # token_slices
        ]
    ] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one forward pass and return ``(total_loss, per-modality MAE dict)``.

    ``spec_pb_weights`` (default ``None``) enables the per-bin weighted
    MAE for spectrogram modalities. When ``None`` (the historical
    default), every modality uses plain ``masked_mae`` — identical to
    pre-2026-06-12 behavior. When a dict ``{name: weight_tensor(C, F)}``,
    each named spectrogram is scored via ``weighted_masked_mae`` to
    counter spec mean-collapse.

    ``precomputed`` (default ``None``) supports the K-step rollout trainer:
    when provided as ``(predictions, diag_inputs, targets, masks,
    token_slices)`` the internal ``forward_batch`` call is skipped and the
    loss body runs on those tensors instead. This lets the rollout driver
    feed each step's own forward outputs (with the fed-back state in
    ``diag_inputs`` so the descriptor anchor reads the rolled-out state,
    matching the Gate-4 inference wiring). ``None`` is byte-identical to
    the single-step path.
    """
    if precomputed is None:
        predictions, diag_inputs, targets, masks, token_slices = forward_batch(
            model, batch, device
        )
    else:
        predictions, diag_inputs, targets, masks, token_slices = precomputed
    per_modality: Dict[str, float] = {}
    total_loss = torch.zeros((), device=device)
    core = _core(model)
    for cfg in core.diagnostics:
        head = core.diag_heads[cfg.name]
        # MULTI-HORIZON (Gate 2b): under prediction_horizon_s>chunk the target is a K-window
        # extended future (spectro kept to max-horizon in forward_batch; TS/fast-TS naturally 4×)
        # while EVERY head predicts a SINGLE window. Align the BASE target (+mask) to the
        # PREDICTION's time-width (= sub-window-0 = t+1) so base loss/code-CE match the head's own
        # single-window output — identical to the single-step run. The descriptor reads
        # _desc_full_tgt (the FULL extended target, captured here BEFORE the align) for its t+h subs.
        _desc_full_tgt = targets.get(cfg.name)
        if (n_subwindows > 1 and torch.is_tensor(_desc_full_tgt)
                and torch.is_tensor(predictions.get(cfg.name))):
            _pred_t = predictions[cfg.name]
            # Trim the multi-horizon target (+mask) to the PREDICTION's shape on
            # EVERY dim where the target is longer, taking the first pred-worth =
            # sub-window-0 (t+1). Spectro/TS carry the horizon on the LAST (time)
            # dim; VIDEO (B,C,T,H,W) carries it on dim 2 (frames), which the old
            # ``[..., :_pw]`` last-dim-only slice missed → the K-rollout t+K crash
            # (masked_mae pred T=3 vs target T=15). Byte-identical to the old
            # last-dim slice whenever only the last dim differs (spectro/TS).
            if (_desc_full_tgt.dim() == _pred_t.dim()
                    and _desc_full_tgt.shape != _pred_t.shape):
                _sl = tuple(
                    slice(0, ps) if ts > ps else slice(None)
                    for ps, ts in zip(_pred_t.shape, _desc_full_tgt.shape)
                )
                targets[cfg.name] = _desc_full_tgt[_sl]
                _m = masks.get(cfg.name)
                if torch.is_tensor(_m) and _m.dim() == _pred_t.dim():
                    _msl = tuple(
                        slice(0, ps) if ms > ps else slice(None)
                        for ps, ms in zip(_pred_t.shape, _m.shape)
                    )
                    masks[cfg.name] = _m[_msl]
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
        if isinstance(head, SlowTimeSeriesCodeHead):
            # Discrete slow-TS (Thomson/CER/MSE) code prediction (Phase 1b). Codec
            # trains in the DATASET-standardized space, so encode the target AS-IS
            # (no re-normalization, unlike fast-TS/video). Class-weighted CE; gradient
            # flows only through code_logits → DDP-safe. Optional presence gate.
            with torch.no_grad():
                tgt_codes = head.encode_target(
                    torch.nan_to_num(targets[cfg.name].float()))          # (B,n_tok,dim)
            logits = head.code_logits(token_slices[cfg.name])             # (B,n_tok,dim,L)
            B_, n_tok_, dim_, L_ = logits.shape
            logits_flat = logits.reshape(-1, L_)
            tgt_flat = tgt_codes.reshape(-1)
            ce_all = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
            g = masks.get(cfg.name)
            if g is not None:
                pres = (g.reshape(B_, -1).abs().sum(1) > 0).float()
            else:
                pres = torch.ones(B_, device=logits.device)
            pres_flat = pres.view(B_, 1, 1).expand(B_, n_tok_, dim_).reshape(-1)
            cw = (slow_ts_code_class_weights or {}).get(cfg.name)
            if cw is not None:
                cw = cw.to(logits.device)
                dim_idx = torch.arange(B_ * n_tok_ * dim_, device=logits.device) % dim_
                w = cw[dim_idx, tgt_flat] * pres_flat
            else:
                w = pres_flat
            ce = (ce_all * w).sum() / (w.sum() + 1e-8)
            loss = ce
            with torch.no_grad():
                acc = ((logits_flat.argmax(-1) == tgt_flat).float() * pres_flat).sum() \
                    / (pres_flat.sum() + 1e-8)
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_ce"] = ce.item()
            per_modality[f"{cfg.name}_codeacc"] = acc.item()
        elif isinstance(head, FastTimeSeriesCodeHead):
            # Discrete fast-TS (ELM) code prediction (Phase 1b), 1-D analog of the
            # SpectrogramCodeHead branch. The codec lives in per-(window, channel)
            # z-scored space (POC load_fastts_windows), so z-score the dataset
            # target the SAME way before encode_target. Class-weighted CE; gradient
            # flows only through code_logits (frozen codec + argmax-decode carry
            # none) → DDP-safe. Optional per-sample presence gate (masks[name]).
            tgt_ft = torch.nan_to_num(targets[cfg.name].float())
            mu_ft = tgt_ft.mean(dim=-1, keepdim=True)
            sd_ft = tgt_ft.std(dim=-1, keepdim=True).clamp(min=1e-3)
            with torch.no_grad():
                tgt_codes = head.encode_target((tgt_ft - mu_ft) / sd_ft)  # (B,n_tok,dim)
            logits = head.code_logits(token_slices[cfg.name])             # (B,n_tok,dim,L)
            B_, n_tok_, dim_, L_ = logits.shape
            logits_flat = logits.reshape(-1, L_)
            tgt_flat = tgt_codes.reshape(-1)
            ce_all = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
            g = masks.get(cfg.name)
            if g is not None:
                pres = (g.reshape(B_, -1).abs().sum(1) > 0).float()
            else:
                pres = torch.ones(B_, device=logits.device)
            pres_flat = pres.view(B_, 1, 1).expand(B_, n_tok_, dim_).reshape(-1)
            cw = (fastts_code_class_weights or {}).get(cfg.name)
            if cw is not None:
                cw = cw.to(logits.device)
                dim_idx = torch.arange(B_ * n_tok_ * dim_, device=logits.device) % dim_
                w = cw[dim_idx, tgt_flat] * pres_flat
            else:
                w = pres_flat
            ce = (ce_all * w).sum() / (w.sum() + 1e-8)
            loss = ce
            with torch.no_grad():
                acc = ((logits_flat.argmax(-1) == tgt_flat).float() * pres_flat).sum() \
                    / (pres_flat.sum() + 1e-8)
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_ce"] = ce.item()
            per_modality[f"{cfg.name}_codeacc"] = acc.item()
        elif isinstance(head, SpectrogramMaskGITHead):
            # JOINT (MaskGIT) code prediction — BERT-style random masking, ONE
            # parallel forward, CE on the MASKED patches only. The bidirectional
            # transformer makes patch-codes coherent (fixes the independent-head
            # blocky/speckle collapse). Grad flows through the head only (frozen
            # codec) and every param is exercised each step -> DDP-safe.
            with torch.no_grad():
                _spec_tgt = (diag_inputs[cfg.name] if spec_autoencode
                             else targets[cfg.name])
                tgt_codes = head.encode_target(_spec_tgt)           # (B,n_tok,dim)
            B_, n_tok_, dim_ = tgt_codes.shape
            mgmask = head.sample_mask(B_, n_tok_, tgt_codes.device)  # (B,n_tok) bool
            logits = head.masked_logits(token_slices[cfg.name], tgt_codes, mgmask)
            L_ = logits.shape[-1]
            m_flat = mgmask.unsqueeze(-1).expand(B_, n_tok_, dim_).reshape(-1).float()
            ce_all = F.cross_entropy(
                logits.reshape(-1, L_), tgt_codes.reshape(-1), reduction="none")
            if spec_code_focal_gamma > 0.0:
                # Focal (1-p_t)^gamma up-weights the hard/RARE codes (thin coherent
                # mode bands, e.g. co2) that the frequent background code otherwise
                # drowns in the masked CE -> lets MaskGIT learn the bands.
                pt = torch.exp(-ce_all).clamp(max=1.0)
                ce_all = ce_all * (1.0 - pt).pow(spec_code_focal_gamma)
            ce = (ce_all * m_flat).sum() / (m_flat.sum() + 1e-8)
            loss = ce
            with torch.no_grad():
                acc = ((logits.argmax(-1).reshape(-1) == tgt_codes.reshape(-1)).float()
                       * m_flat).sum() / (m_flat.sum() + 1e-8)
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_ce"] = ce.item()
            per_modality[f"{cfg.name}_codeacc"] = acc.item()
        elif isinstance(head, SpectrogramCodeHead):
            # Discrete code prediction (Phase 1b). Class-weighted CE over the
            # FROZEN codec's per-dim codes — categorical, cannot mean-collapse.
            # predictions[name] (=argmax-decode, scored as `mae` above for
            # logging only) carries NO gradient (frozen decoder + detached
            # sampling); the gradient flows ONLY through code_logits here, which
            # exercises every prediction-head param each step -> DDP-safe.
            with torch.no_grad():
                # --spec_autoencode: predict the CURRENT input window's OWN codes
                # (diag_inputs), not the NEXT window's (targets) → isolates
                # representation capacity from forecast-irreducibility.
                _spec_tgt = (diag_inputs[cfg.name] if spec_autoencode
                             else targets[cfg.name])
                tgt_codes = head.encode_target(_spec_tgt)           # (B,n_tok,dim) int
            logits = head.code_logits(token_slices[cfg.name])       # (B,n_tok,dim,L)
            B_, n_tok_, dim_, L_ = logits.shape
            logits_flat = logits.reshape(-1, L_)
            tgt_flat = tgt_codes.reshape(-1)
            if spec_ordinal_eps > 0.0:
                from tokamak_foundation_model.e2e.ordinal_loss import soft_ordinal_ce
                ce_all = soft_ordinal_ce(logits_flat, tgt_flat, eps=spec_ordinal_eps, reduction="none")
            else:
                ce_all = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
            if spec_code_focal_gamma > 0.0:
                # Focal down-weighting: (1-p_true)^gamma damps easy/background codes
                # so rare MODE codes drive the gradient (composes with cw below).
                pt = torch.exp(-ce_all).clamp(max=1.0)
                ce_all = ce_all * (1.0 - pt).pow(spec_code_focal_gamma)
            cw = (spec_code_class_weights or {}).get(cfg.name)
            if cw is not None:
                # per-element weight w[dim, target_level]; data-normalized upstream
                # so E_data[w]=1 and rare MODE codes are up-weighted vs background.
                cw = cw.to(logits.device)
                dim_idx = torch.arange(B_ * n_tok_ * dim_, device=logits.device) % dim_
                w = cw[dim_idx, tgt_flat]
                ce = (ce_all * w).sum() / (w.sum() + 1e-8)
            else:
                ce = ce_all.mean()
            loss = ce
            with torch.no_grad():
                acc = (logits_flat.argmax(-1) == tgt_flat).float().mean()
                tol1 = ((logits_flat.argmax(-1) - tgt_flat).abs() <= 1).float().mean()
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_ce"] = ce.item()
            per_modality[f"{cfg.name}_codeacc"] = acc.item()
            per_modality[f"{cfg.name}_tol1acc"] = tol1.item()   # gate metric (non-gating log)
        elif isinstance(head, SpectrogramFlowHead):
            # predictions[name] == μ in train mode (head.forward returns the
            # deterministic mean); add the rectified-flow velocity loss on the
            # residual. The velocity net runs every step here → all its params
            # get grads (DDP-safe, no unused parameters).
            flow = head.flow_loss(
                token_slices[cfg.name], predictions[cfg.name],
                targets[cfg.name], masks[cfg.name],
                band_weight=(spec_pb_weights or {}).get(cfg.name),
            )
            # spec_mae_lambda default 1.0. Set 0 to REMOVE the mean-seeking pixel
            # MAE's grip on the spectro tokens (the 0*mae term keeps the mean_head
            # in the graph → DDP-safe) so the token slice is shaped only by the
            # mode objective → modes survive into the forecast tokens.
            loss = spec_mae_lambda * mae + head.flow_lambda * flow
            if spec_struct_lambda > 0.0:
                # (A) mode-coherence Dice on the deterministic mean mu.
                k_struct = _SPEC_STRUCT_K.get(cfg.name, 2.0)
                struct = spectro_struct_loss(
                    predictions[cfg.name], targets[cfg.name],
                    k_struct, gate=masks[cfg.name],
                )
                loss = loss + spec_struct_lambda * struct
                per_modality[f"{cfg.name}_struct"] = struct.item()
            if getattr(head, "enable_mask", False) and spec_mask_lambda > 0.0:
                # (Plan B) predicted mode-mask, scored vs the production
                # binarization with soft-Dice+BCE (no L2 collapse). mask_logits
                # runs every step → the mask params always get grads (DDP-safe).
                k_mask = _SPEC_STRUCT_K.get(cfg.name, 2.0)
                # Input-conditioning / persistence prior (recurrence-ready): the
                # input-window mode mask, reduced to per-frequency PRESENCE (max
                # over the input's time frames) and broadcast over the output
                # window. Modes persist (τ½ 201 ms ≫ 50 ms) so "modes at freq f in
                # the past" is a strong prior for "modes at freq f next". Stage 2
                # will instead pass the previous predicted mask (the recurrence).
                prior = None
                if (getattr(head, "enable_input_cond", False)
                        or getattr(head, "enable_input_feat", False)):
                    with torch.no_grad():
                        in_soft = (
                            _spec_mode_arg(diag_inputs[cfg.name], k_mask)
                            .clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
                        )
                        # HARD, FRAME-ALIGNED input mode mask (0/1): mode at (f,t)
                        # in the input window → persistence prior for (f,t) in the
                        # adjacent output window. Preserves BOTH freq and time
                        # structure (a per-freq collapse over-predicts in time —
                        # modes are time-localized — and craters the Dice). Hard so
                        # logit(prior)=±9.2 decisively sets the baseline; the head's
                        # decode learns the soft corrections (fades, drift, new modes).
                        prior = (in_soft > _SPEC_STRUCT_CUT).to(in_soft.dtype)
                m_logits = head.mask_logits(token_slices[cfg.name], prior=prior)
                m_loss, m_dice = spectro_mask_loss(
                    m_logits, targets[cfg.name], k_mask, gate=masks[cfg.name],
                    loss_type=spec_mask_loss_type,
                )
                loss = loss + spec_mask_lambda * m_loss
                per_modality[f"{cfg.name}_mask"] = m_loss.item()
                per_modality[f"{cfg.name}_maskdice"] = m_dice.item()
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_flow"] = flow.item()
        elif isinstance(head, VideoCodeHead):
            # Discrete video-code prediction (Phase 1b), video analog of the
            # SpectrogramCodeHead branch. Class-weighted CE over the FROZEN video
            # codec's per-dim codes; gradient flows only through code_logits (the
            # frozen codec + argmax-decode in forward carry none) → DDP-safe.
            # Gated per-SAMPLE by video presence (a shot may lack this divertor).
            with torch.no_grad():
                tgt_codes = head.encode_target(targets[cfg.name])   # (B,n_tok,dim)
            logits = head.code_logits(token_slices[cfg.name])       # (B,n_tok,dim,L)
            B_, n_tok_, dim_, L_ = logits.shape
            logits_flat = logits.reshape(-1, L_)
            tgt_flat = tgt_codes.reshape(-1)
            ce_all = F.cross_entropy(logits_flat, tgt_flat, reduction="none")
            g = masks[cfg.name]                                     # (B,C,1,1,1) or None
            if g is not None:
                pres = (g.reshape(B_, -1).abs().sum(1) > 0).float()  # (B,) present?
            else:
                pres = torch.ones(B_, device=logits.device)
            pres_flat = pres.view(B_, 1, 1).expand(B_, n_tok_, dim_).reshape(-1)
            cw = (video_code_class_weights or {}).get(cfg.name)
            if cw is not None:
                cw = cw.to(logits.device)
                dim_idx = torch.arange(B_ * n_tok_ * dim_, device=logits.device) % dim_
                w = cw[dim_idx, tgt_flat] * pres_flat
            else:
                w = pres_flat
            ce = (ce_all * w).sum() / (w.sum() + 1e-8)
            loss = ce
            with torch.no_grad():
                acc = ((logits_flat.argmax(-1) == tgt_flat).float() * pres_flat).sum() \
                    / (pres_flat.sum() + 1e-8)
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_ce"] = ce.item()
            per_modality[f"{cfg.name}_codeacc"] = acc.item()
        elif isinstance(head, VideoFlowHead):
            # Same generative recipe for video. The trainer holds video as
            # (B,C,T,H,W) (post-permute, line ~601); VideoFlowHead.flow_loss
            # wants (B,T,C,H,W) + a (B,T,C) present-mask. predictions[name] is μ
            # (train-mode forward). The velocity net runs every step → DDP-safe
            # even when video is absent from the batch (masked loss → 0, but the
            # net still participated in the graph).
            mu_v = predictions[cfg.name].permute(0, 2, 1, 3, 4)
            tgt_v = targets[cfg.name].permute(0, 2, 1, 3, 4)
            gate = masks[cfg.name]                          # (B,C,1,1,1) or None
            if gate is not None:
                Bv = gate.shape[0]
                mask_btc = (
                    gate.reshape(Bv, head.n_channels)[:, None, :]
                    .expand(Bv, head.n_frames, head.n_channels)
                )
            else:
                mask_btc = None
            flow = head.flow_loss(
                token_slices[cfg.name], mu_v, tgt_v, mask_btc,
            )
            loss = mae + head.flow_lambda * flow
            per_modality[cfg.name] = mae.item()
            per_modality[f"{cfg.name}_flow"] = flow.item()
        else:
            loss = mae
            per_modality[cfg.name] = loss.item()
        # ---- FACTORIZATION: auxiliary mode-descriptor loss. Forecast the shift-stable
        # band-power profile (the modes the codes can't carry: descriptor persists
        # 0.75-0.95 vs codes 0.10-0.34). Own EMA key + priority so the code CE never
        # starves it. Logged {name}_desc_ftol = mode-freq forecast accuracy (+-1kHz).
        if getattr(core, "spec_descriptor_heads", None) and cfg.name in core.spec_descriptor_heads:
            dh = core.spec_descriptor_heads[cfg.name]
            d_pred_all = dh(token_slices[cfg.name])                       # (B, H, NF, TCOL)
            horizons = getattr(dh, "horizons", (1,))
            # PERSISTENCE ANCHOR + transition reference = the CURRENT-window descriptor (same
            # for every horizon: persistence = "the mode stays where it is"). The head learns
            # only the DRIFT residual off this at each t+h.
            _inp_desc = dh.descriptor_target(diag_inputs[cfg.name])       # (B,NF,TCOL)
            _anc = None
            if spec_descriptor_anchor:
                _anc = _inp_desc / _inp_desc.amax(dim=1, keepdim=True).clamp_min(1e-6)
            # Per-horizon target = sub-window (h-1) of the FULL extended target (t+h window).
            # Autoencode diagnostic ignores horizon (target = input, single window).
            # sub-window size = the PREDICTION width (the single-window frame count = trunc_t);
            # the extended target is n_subwindows of these. off=(h-1) picks the t+h window.
            _sw = diag_inputs[cfg.name] if spec_autoencode else _desc_full_tgt
            _pw_sub = (predictions[cfg.name].shape[-1]
                       if torch.is_tensor(predictions.get(cfg.name)) else _sw.shape[-1])
            _nsw = 1 if spec_autoencode else max(1, _sw.shape[-1] // max(1, _pw_sub))
            _Tw = _pw_sub if _nsw > 1 else _sw.shape[-1]

            # ANCHOR-ANNEAL (Gate-3-fix unmask): the anchor's PREDICTION weight = _abeta (scheduled,
            # 8→3), letting the actuator-sensitive residual reach the output; the TARGET softmax keeps
            # the FIXED dist_beta (task definition unchanged). Defaults to dist_beta → byte-identical.
            _abeta = spec_descriptor_anchor_beta if spec_descriptor_anchor_beta is not None else spec_descriptor_dist_beta

            def _desc_term(hi, hstep):
                off = 0 if _nsw <= 1 else min(hstep - 1, _nsw - 1)
                _tspec = _sw[..., off * _Tw:(off + 1) * _Tw] if _nsw > 1 else _sw
                _dtgt = dh.descriptor_target(_tspec)                      # (B,NF,TCOL)
                d_pred = d_pred_all[:, hi]                                # (B,NF,TCOL)
                if spec_descriptor_loss == "dist":
                    # distribution-CE over FREQ (per time-col): forces predicted mass at the GT
                    # mode peak. Anchor (if on): pred_logit = persistence-logit*anneal + head residual
                    # (head zero-init → starts AT persistence, learns only drift, cannot collapse).
                    pred_logit = (_anc * _abeta + d_pred) if _anc is not None else d_pred
                    _tn = _dtgt / _dtgt.amax(dim=1, keepdim=True).clamp_min(1e-6)
                    q = F.softmax(_tn * spec_descriptor_dist_beta, dim=1)  # soft target over freq (FIXED beta)
                    ce = -(q * F.log_softmax(pred_logit, dim=1)).sum(1)    # (B,TCOL)
                    # ACTIVE-WEIGHT by target mode prominence (quiescent majority can't dominate
                    # into a flat collapse) + optional TRANSITION-OVERWEIGHT on presence flips
                    # (onset/death from the CURRENT window to t+h — the non-copyable events).
                    prom = (_dtgt.amax(dim=1) - _dtgt.mean(dim=1)).clamp_min(0.0)
                    wgt = prom + 0.05
                    if spec_descriptor_transition_weight > 1.0:
                        tp = _dtgt.amax(dim=1) - _dtgt.mean(dim=1)
                        ip = _inp_desc.amax(dim=1) - _inp_desc.mean(dim=1)
                        thp = tp.median()
                        trans = ((tp > thp) != (ip > thp)).float()
                        wgt = wgt * (1.0 + (spec_descriptor_transition_weight - 1.0) * trans)
                    _t_loss = (ce * wgt).sum() / (wgt.sum() + 1e-8)
                    _pe = pred_logit
                else:
                    _t_loss = F.mse_loss(d_pred, _dtgt)
                    _pe = d_pred
                with torch.no_grad():
                    _ft = ((_pe.argmax(1) - _dtgt.argmax(1)).abs() <= 2).float().mean()
                    # COLLAPSE TRIPWIRES (early-warning as the anchor weakens; per-500-step logged):
                    #  hfrac  = H(pred softmax)/log(NF) → 1.0 = flat mean-collapse (descriptor-wars detector)
                    #  ftp    = PERSISTENCE peak-in-tol (the anchor's fidelity job; _ft must not fall below it)
                    #  fdrift = FALSE-DEATH proxy: on STATIC windows (no target presence-flip vs input),
                    #           fraction where the model moves the peak >2 bins off persistence (spurious dynamics)
                    _NF = _pe.shape[1]
                    _p = F.softmax(_pe, dim=1)
                    _hfrac = float((-(_p * (_p + 1e-9).log()).sum(1)).mean() / math.log(_NF))
                    if _anc is not None:
                        _ftp = float(((_anc.argmax(1) - _dtgt.argmax(1)).abs() <= 2).float().mean())
                        _tp = _dtgt.amax(1) - _dtgt.mean(1); _ip = _inp_desc.amax(1) - _inp_desc.mean(1)
                        _thp = _tp.median()
                        # ACTIVE-STATIC = mode present NOW (_ip>thp) AND still present at t+h (_tp>thp):
                        # sustained mode, no presence-flip. Mirrors the eval false-death "sustained window"
                        # definition; EXCLUDES quiescent windows where argmax is meaningless noise (that
                        # noise inflated the smoke's β8 fdrift to 0.068 vs eval false-death 0.000).
                        _astatic = (_tp > _thp) & (_ip > _thp)
                        _dr = ((_pe.argmax(1) - _anc.argmax(1)).abs() > 2) & _astatic
                        _fdrift = float(_dr.float().sum() / _astatic.float().sum().clamp_min(1.0))
                    else:
                        _ftp = float("nan"); _fdrift = float("nan")
                # ── STRIKE-3 LEVER 1: ASYMMETRIC drift penalty (opt-in) ──────────
                # Penalize ONLY over-drift of the predicted ece descriptor ridge vs
                # ground truth — the K=10-gate pathology (drift 2-6.5×GT). Uses the
                # SAME prominence-weighted freq-centroid gate4_kprobe measures as
                # `drift_pred` (centroid of the descriptor over freq bins, per window),
                # and the same persistence-anchor reference as the fdrift tripwire:
                #   pred_drift = |centroid(pred_desc) - centroid(anchor)|   (bins)
                #   gt_drift   = |centroid(gt_desc)   - centroid(anchor)|   (bins)
                #   L_drift    = relu(pred_drift - gt_drift)   [over-drift ONLY;
                #                under-drift / legit corrections are NOT penalized]
                # WITH grad (so it steers the head); logged (asymmetric, >0 only when
                # the model over-drifts). Default weight 0.0 → block skipped entirely
                # → byte-identical to non-strike-3 runs.
                _drift_pen_val = float("nan")
                if drift_penalty_weight > 0.0 and _anc is not None:
                    _NFd = _pe.shape[1]
                    _fb = torch.arange(
                        _NFd, device=_pe.device, dtype=_pe.dtype
                    )[None, :, None]

                    def _centroid(_prof):
                        # (B,NF,TCOL) -> (B,) prominence-weighted freq centroid (bins),
                        # mean over time-cols — identical to gate4_kprobe.centroid.
                        _w = _prof.clamp_min(0.0)
                        return ((_fb * _w).sum(1)
                                / (_w.sum(1) + 1e-8)).mean(1)               # (B,)

                    # centroid the RAW pred logit `_pe` (= anchor*β + residual),
                    # clamped-at-0 — the EXACT functional form gate4_kprobe.centroid
                    # applies to its `outp = anc_n*β + resid` (so the training penalty
                    # measures the same drift_pred quantity the gate reports). Anchor
                    # and target are the positive band-power / anchor profiles.
                    _cp = _centroid(_pe)
                    _ct = _centroid(_dtgt)
                    _ca = _centroid(_anc)
                    _pred_drift = (_cp - _ca).abs()
                    _gt_drift = (_ct - _ca).abs()
                    _over = F.relu(_pred_drift - _gt_drift)                  # (B,)
                    _drift_loss = drift_penalty_weight * _over.mean()
                    _t_loss = _t_loss + _drift_loss
                    _drift_pen_val = float(_drift_loss.detach())
                return _t_loss, _ft, {"hfrac": _hfrac, "ftp": _ftp,
                                      "fdrift": _fdrift, "drift_pen": _drift_pen_val}

            _terms = []; _last_extra = {}
            for _hi, _hstep in enumerate(horizons):
                _tl, _ft, _ex = _desc_term(_hi, _hstep)
                _terms.append(_tl)
                per_modality[f"{cfg.name}_desc_ftol_t{_hstep}"] = _ft.item()
                # per-horizon loss (WATCH the t2:t4 ratio: t+2 is easier → can shadow t+4,
                # the gated horizon, if it dominates the summed gradient — see Gate 2b notes).
                per_modality[f"{cfg.name}_desc_l_t{_hstep}"] = _tl.item()
                _last_extra = _ex                                         # headline (longest) horizon
            d_loss = sum(_terms) / len(_terms)                            # mean over horizons
            per_modality[f"{cfg.name}_desc"] = d_loss.item()
            # headline ftol = the LONGEST horizon (hardest, paper-relevant); keeps the legacy key.
            per_modality[f"{cfg.name}_desc_ftol"] = per_modality[f"{cfg.name}_desc_ftol_t{horizons[-1]}"]
            # collapse tripwires (headline horizon) — auto-surface via the "_desc" log filter + best.pt gate.
            per_modality[f"{cfg.name}_desc_hfrac"] = _last_extra.get("hfrac", float("nan"))
            per_modality[f"{cfg.name}_desc_ftp"] = _last_extra.get("ftp", float("nan"))
            per_modality[f"{cfg.name}_desc_fdrift"] = _last_extra.get("fdrift", float("nan"))
            # STRIKE-3 lever 1: asymmetric over-drift penalty (headline horizon).
            # >0 only when the model over-drifts (pred_drift>gt_drift); 0 when the
            # model drifts <= GT (relu asymmetry). NaN when the lever is off.
            per_modality[f"{cfg.name}_desc_drift_pen"] = _last_extra.get("drift_pen", float("nan"))
            if loss_norm_ema:
                if not hasattr(core, "_loss_ema"):
                    core._loss_ema, core._loss_ema_init = {}, {}
                dk = f"{cfg.name}__desc"
                _dm = float(d_loss.detach())
                _de = core._loss_ema.get(dk)
                _de = _dm if _de is None else loss_norm_beta * _de + (1.0 - loss_norm_beta) * _dm
                core._loss_ema[dk] = _de
                _dw = spec_descriptor_weight / (_de + 1e-8)
                per_modality[f"{cfg.name}_desc_w"] = _dw
                total_loss = total_loss + _dw * d_loss
            else:
                total_loss = total_loss + spec_descriptor_weight * d_loss
        if loss_norm_ema:
            # Per-modality EMA magnitude normalization: divide each modality's loss by a
            # running EMA of its magnitude so every modality contributes O(1) to the total
            # (fixes the 4-OOM spectro-vs-slowTS gradient starvation). Weight is DETACHED.
            _lm = float(loss.detach())
            if not hasattr(core, "_loss_ema"):
                core._loss_ema, core._loss_ema_init = {}, {}
            _e = core._loss_ema.get(cfg.name)
            _e = _lm if _e is None else loss_norm_beta * _e + (1.0 - loss_norm_beta) * _lm
            core._loss_ema[cfg.name] = _e
            _w = float((loss_priority or {}).get(cfg.name, 1.0)) / (_e + 1e-8)
            core._loss_ema_init.setdefault(cfg.name, _w)
            per_modality[f"{cfg.name}_lossnorm_w"] = _w
            total_loss = total_loss + _w * loss
        else:
            total_loss = total_loss + loss
    return total_loss, per_modality


# ── K-step rollout training driver ─────────────────────────────────────────


def current_K_from_list(step: int, Ks: List[int], block_steps: int) -> int:
    """Curriculum K for this ``step``: ``Ks[min(step//block_steps, len(Ks)-1)]``.

    Advances one K per ``block_steps`` training steps, clamped at the last entry.
    ``block_steps<=0`` is guarded to 1 so it never divides by zero."""
    return Ks[min(step // max(1, block_steps), len(Ks) - 1)]


def rollout_forward_loss(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
    K: int,
    chunk_duration_s: float,
    rollout: "TokenSpaceRollout",
    *,
    compute_step_loss_kwargs: Dict,
    p_tf: float = 0.0,
    grad_checkpoint_every: int = 0,
    feedback_normalize: bool = False,
    k_ge1_weight: float = 1.0,
    k_ge1_weight_start: float = 0.1,
    k_ge1_weight_anneal_steps: int = 0,
    global_step: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """OPT-IN K-step rollout loss for Stage 1.

    Builds the per-step actuator / target / mask / gt-target dicts with the SAME
    construction as ``eval_e2e.rollout_forward_one_batch`` (imported split
    helpers, residual-spectro bg split), runs the PROVEN Gate-4 code-space
    ``argmax`` feedback rollout WITH gradients, then scores each step through
    ``compute_step_loss`` with a ``precomputed`` tuple whose ``diag_inputs`` is
    the DECODED FED-BACK STATE at that step — so the spectro descriptor anchor
    reads the rolled-out state (train == the Gate-4 inference wiring), not the
    GT / step-0 input.

    Returns ``(mean_over_K_loss, per_modality)`` where ``per_modality`` is the
    last step's dict (the deepest-rollout diagnostics — what we watch)."""
    # Reuse the eval construction so train and inference feed IDENTICAL tensors
    # (the train/inference-feedback mismatch that burned this project twice must
    # NOT recur). Imported here (not at module top) to keep the trainer's
    # import graph unchanged for the single-step default path.
    from eval_e2e import (
        _clean_and_mask as _eval_clean_and_mask,
        _eval_spectro_bg_split,
        _spectro_loss_gate,
        _spectro_trunc_t,
        _ts_mask,
        _video_loss_gate,
        _video_standardize_per_bc,
        split_spectro_target_by_step,
        split_target_by_step,
        split_video_target_by_step,
    )

    core = _core(model)
    video_diags = [c.name for c in core.diagnostics if c.kind == "video"]
    spectro_diags = [c.name for c in core.diagnostics if c.kind == "spectrogram"]
    cfg_by_name = {c.name: c for c in core.diagnostics}
    act_names = [c.name for c in core.actuators]
    desc_heads = getattr(core, "spec_descriptor_heads", {}) or {}

    # ── Step-0 diagnostic inputs (mirror forward_batch: spectro input is
    # trunc-truncated + residual-bg split; video per-(B,C) z-scored). ──
    video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    diag_initial: Dict[str, torch.Tensor] = {}
    for cfg in core.diagnostics:
        name = cfg.name
        raw = batch["inputs"][name].to(device, non_blocking=True).float()
        cleaned, _ = _eval_clean_and_mask(raw, None)
        if cfg.kind == "video":
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            video_stats[name] = (mu, sd)
        elif cfg.kind == "spectrogram":
            trunc_t = _spectro_trunc_t(cfg)
            cleaned = cleaned[..., :trunc_t]
            cleaned = _eval_spectro_bg_split(model, name, cleaned)
        diag_initial[name] = cleaned
        if cfg.kind in ("video", "spectrogram"):
            valid_key = f"{name}_valid"
            if valid_key in batch["inputs"]:
                diag_initial[valid_key] = batch["inputs"][valid_key].to(
                    device, non_blocking=True
                )

    # ── Full-horizon target + gate tensors (video / spectro). ──
    video_target_full: Dict[str, torch.Tensor] = {}
    video_gate: Dict[str, torch.Tensor] = {}
    spectro_target_full: Dict[str, torch.Tensor] = {}
    spectro_gate: Dict[str, torch.Tensor] = {}
    spectro_trunc: Dict[str, int] = {}
    for name in video_diags:
        raw = batch["targets"][name].to(device, non_blocking=True).float()
        cleaned, _ = _eval_clean_and_mask(raw, None)
        mu, sd = video_stats[name]
        video_target_full[name] = (cleaned - mu) / sd
        video_gate[name] = _video_loss_gate(cfg_by_name[name], batch, device)
    for name in spectro_diags:
        raw = batch["targets"][name].to(device, non_blocking=True).float()
        cleaned, _ = _eval_clean_and_mask(raw, None)
        spectro_target_full[name] = _eval_spectro_bg_split(model, name, cleaned)
        spectro_gate[name] = _spectro_loss_gate(name, batch, device)
        spectro_trunc[name] = _spectro_trunc_t(cfg_by_name[name])

    # Descriptor horizon reach (in chunk-windows). Each rollout step's spectro
    # target must span max_horizon windows so compute_step_loss can slice the
    # descriptor's t+h sub-windows (h in horizons). No descriptor → 1 (base
    # single-window target only). n_subwindows below is pinned to this.
    max_horizon = 1
    for name in spectro_diags:
        if name in desc_heads:
            max_horizon = max(max_horizon, max(getattr(desc_heads[name], "horizons", (1,))))
    n_subwindows = max_horizon

    # ── Per-step act / target / mask / gt-target dicts (length K). ──
    # Non-spectro targets: one chunk-window per step (eval convention). Spectro
    # targets: OVERLAPPING max_horizon windows starting at step k, so the
    # descriptor reads its t+h subs (matches the single-step _desc_full_tgt).
    act_per_step: List[Dict[str, torch.Tensor]] = []
    target_per_step: List[Dict[str, torch.Tensor]] = []
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
    gt_target_per_step: List[Dict[str, torch.Tensor]] = []
    for k in range(K):
        act_k: Dict[str, torch.Tensor] = {}
        for name in act_names:
            raw = batch["targets"][name].to(device, non_blocking=True).float()
            slc = split_target_by_step(raw, name, K, chunk_duration_s)[k]
            cleaned, _ = _eval_clean_and_mask(slc, None)
            act_k[name] = cleaned
        act_per_step.append(act_k)

        tgt_k: Dict[str, torch.Tensor] = {}
        mk_k: Dict[str, Optional[torch.Tensor]] = {}
        gt_k: Dict[str, torch.Tensor] = {}
        for cfg in core.diagnostics:
            name = cfg.name
            if cfg.kind == "video":
                n_per = video_target_full[name].shape[2] // K
                tgt_k[name] = split_video_target_by_step(
                    video_target_full[name], K, n_per
                )[k]
                mk_k[name] = video_gate[name]
                gt_k[name] = tgt_k[name]
            elif cfg.kind == "spectrogram":
                trunc = spectro_trunc[name]
                # Base single-window target for step k (t+1 chunk) — the GT
                # state fed to the head + used for teacher forcing.
                base = spectro_target_full[name][..., k * trunc : (k + 1) * trunc]
                gt_k[name] = base
                # Descriptor-extended target: max_horizon windows starting at k
                # (windows [k .. k+max_horizon)). Clamp/pad the tail so the last
                # rollout steps (near the end of the loaded future) still have a
                # width divisible by trunc; if the future runs out, fall back to
                # the base window (compute_step_loss then treats it single-sub).
                end = (k + max_horizon) * trunc
                if max_horizon > 1 and end <= spectro_target_full[name].shape[-1]:
                    tgt_k[name] = spectro_target_full[name][..., k * trunc : end]
                else:
                    tgt_k[name] = base
                mk_k[name] = spectro_gate[name]
            else:
                # ── ASYMMETRY FIX (2026-07): the step-0 diag path (L1755), the
                # actuator path (L1813), and the spectro path all sanitize their
                # raw targets via _eval_clean_and_mask BEFORE the tensor can reach
                # a tokenizer. This continuous (slow-TS / cer / mse) branch did
                # NOT — it fed the RAW split target straight into gt_k, which the
                # teacher-forcing feedback path re-tokenizes on-manifold
                # (rollout._tokenize_gt_onmanifold, continuous `else` branch:
                # diag_tokenizers[name](x)). Dead channels carry -inf (e.g. mse
                # ch 3-4 of shot 193735) → NaN feedback tokens → k>=1 backbone
                # input NaN → cross-modality spread (mislabeled as ece by the old
                # spectro-only localizer). Mirror L1755/L1813: clean-and-mask so
                # the tokenizer never sees -inf, and CARRY the finite mask into
                # the loss (cleaning without the mask would silently train the
                # loss on the sanitized-to-0 garbage of the dead channels).
                raw = batch["targets"][name].to(device, non_blocking=True).float()
                slc = split_target_by_step(raw, name, K, chunk_duration_s)[k]
                cleaned, finite_mask = _eval_clean_and_mask(slc, None)
                # finite_mask: 1.0 = finite/valid, 0.0 = non-finite — SAME
                # convention as the data loader's {name}_mask (1=valid, from
                # `raw_valid = nan_mask < 0.5`), so the two multiply safely.
                tgt_k[name] = cleaned          # loss target (finite)
                gt_k[name] = cleaned           # TF feedback GT (finite) — the fix
                mask_key = f"{name}_mask"
                if mask_key in batch["targets"]:
                    raw_mask = batch["targets"][mask_key].to(
                        device, non_blocking=True
                    ).float()
                    batch_mask = split_target_by_step(
                        raw_mask, name, K, chunk_duration_s
                    )[k]
                    # Both masks are 1=valid, same (B, C, per) shape from the
                    # identical split → element-wise AND (multiply).
                    mk_k[name] = batch_mask * finite_mask
                else:
                    mk_k[name] = finite_mask
        target_per_step.append(tgt_k)
        mask_per_step.append(mk_k)
        gt_target_per_step.append(gt_k)

    # ── Run the K-step rollout WITH GRADIENTS (Gate-4 argmax code feedback). ──
    result = rollout(
        diag_initial,
        act_per_step,
        collect_history=False,
        collect_token_slices=True,
        collect_decoded_feedback=True,
        feedback_mode="argmax",
        feedback_temperature=1.0,
        gt_target_per_step=gt_target_per_step,
        p_tf=p_tf,
        grad_checkpoint_every=grad_checkpoint_every,
        feedback_normalize=feedback_normalize,
    )

    # Video predictions come out (B, T, C, H, W); compute_step_loss (via the
    # forward_batch contract) expects (B, C, T, H, W). Flip in place.
    for k in range(len(result.predictions)):
        for name in video_diags:
            if name in result.predictions[k]:
                result.predictions[k][name] = (
                    result.predictions[k][name].permute(0, 2, 1, 3, 4)
                )

    # Force n_subwindows to the descriptor reach (overriding any caller value)
    # so compute_step_loss aligns the base target + slices the descriptor subs.
    cs_kwargs = dict(compute_step_loss_kwargs)
    cs_kwargs["n_subwindows"] = n_subwindows

    total_loss = torch.zeros((), device=device)
    per_modality: Dict[str, float] = {}
    import os as _os_mod
    _nandbg = _os_mod.environ.get("ROLLOUT_NAN_DEBUG", "0") == "1"
    all_diag_names = [c.name for c in core.diagnostics]

    # ── STRIKE-3 LEVER 2: k0-PROTECTED per-k loss re-weighting (opt-in) ──────
    # The rollout objective sums per-step loss over K; later-k gradients dilute /
    # conflict with the k=0 term that carries the banked single-step property.
    # Re-weight so k=0 keeps its FULL Stage-1 gradient share (w_0 = 1.0 PINNED)
    # while k>=1 is down-weighted (w_ge1). Optional linear anneal-up of w_ge1 from
    # `k_ge1_weight_start` to 1.0 over `k_ge1_weight_anneal_steps` global steps
    # (uniform once complete). Defaults (w_ge1=1.0, anneal_steps=0) → every w_k=1
    # → identical to the plain `total_loss += step_loss` sum (byte-identical).
    # Only the backward-driving TOTAL is re-weighted; the per-modality LOGGED
    # losses (floats from step_per_mod) are untouched → tripwires stay comparable.
    if k_ge1_weight_anneal_steps > 0:
        _frac = min(1.0, max(0.0, global_step / float(k_ge1_weight_anneal_steps)))
        _w_ge1 = k_ge1_weight_start + _frac * (1.0 - k_ge1_weight_start)
    else:
        _w_ge1 = k_ge1_weight
    _wk_active = (_w_ge1 != 1.0)   # any reweighting engaged this step?

    def _nan_locate(_k: int) -> None:
        """First-non-finite report PER MODALITY across ALL diagnostics (video +
        spectrogram + continuous slow-TS/cer/mse). Robust to missing dict
        entries (.get may return None per kind/stage). Always-on when a step is
        non-finite — the culprit MODALITY (not just 'ece') must self-identify in
        the FIRST log line, so the next cross-modality contamination is caught
        immediately instead of after chasing the wrong modality for a day.
        NOTE: `gt` is the raw teacher-forcing state fed into the tokenizer — the
        actual bug locus in the 2026-07 mse->ece NaN; check it FIRST."""
        for _nm in all_diag_names:
            for _lbl, _t in (
                ("gt", gt_target_per_step[_k].get(_nm)),
                ("feedback", result.decoded_feedback[_k].get(_nm)
                 if result.decoded_feedback and _k < len(result.decoded_feedback) else None),
                ("target", target_per_step[_k].get(_nm)),
                ("token_slice", result.diag_token_slices[_k].get(_nm)
                 if result.diag_token_slices and _k < len(result.diag_token_slices) else None),
                ("pred", result.predictions[_k].get(_nm)
                 if result.predictions and _k < len(result.predictions) else None),
            ):
                if torch.is_tensor(_t):
                    _frac = (~torch.isfinite(_t)).float().mean().item()
                    if _frac > 0.0:
                        logger.warning(
                            f"[nan-loc] k={_k} {_nm} {_lbl}: nonfinite_frac="
                            f"{_frac:.4f} shape={tuple(_t.shape)}"
                        )

    # PER-K LOSS SHARE (pre-registered CONTINGENCY TRIGGER, always-on). The
    # rollout-native-from-scratch bet fails if the summed objective trades away
    # single-step (k=0) skill as the horizon extends. The early signature is the
    # k=0 loss SHARE collapsing as K grows (later-k terms dominate the sum). We
    # record each step's scalar (backward-weighted) loss contribution and emit
    # normalized shares into per_modality → they surface in the log line's
    # aux_str (keys start with "rollout_"). Costs one .item() per rollout step
    # (already synced by the finite-check below) → negligible.
    _k_loss_vals: List[float] = []
    for k in range(K):
        if _nandbg:
            # Verbose (flag-gated) scan every step — pre-fix diagnostic behavior
            # preserved. The always-on culprit report below fires on failure.
            _nan_locate(k)
        precomputed = (
            result.predictions[k],           # predictions
            result.decoded_feedback[k],      # diag_inputs = FED-BACK state at step k
            target_per_step[k],              # targets (spectro carries the t+h subs)
            mask_per_step[k],                # masks
            result.diag_token_slices[k],     # token_slices
        )
        step_loss, step_per_mod = compute_step_loss(
            model, batch, device, precomputed=precomputed, **cs_kwargs
        )
        _step_finite = bool(torch.isfinite(step_loss).item())
        if not _step_finite:
            _nf = {kk: vv for kk, vv in step_per_mod.items()
                   if isinstance(vv, float) and (math.isnan(vv) or math.isinf(vv))}
            logger.warning(
                f"[rollout] NON-FINITE step_loss at rollout step k={k}/{K} "
                f"(p_tf={p_tf:.3f}); non-finite terms: {_nf or 'aggregate only'}"
            )
            # ALWAYS-ON per-modality culprit scan (no ROLLOUT_NAN_DEBUG needed):
            # zero cost on the finite common path, full per-modality diagnosis on
            # failure so the offending MODALITY+STAGE self-identifies immediately.
            if not _nandbg:
                _nan_locate(k)
        # LEVER 2: w_0 = 1.0 (pinned), w_{k>=1} = _w_ge1 (const or annealed-up).
        _wk = 1.0 if k == 0 else _w_ge1
        total_loss = total_loss + _wk * step_loss
        # Record the backward-weighted per-step contribution for the share log.
        _k_loss_vals.append(float(step_loss.detach().item()) * _wk if _step_finite
                            else float("nan"))
        per_modality = step_per_mod          # last step's dict (deepest rollout)
    if _wk_active:
        # Log the APPLIED per-k weight vector (k=0 always 1.0, k>=1 = _w_ge1) so
        # the smoke/monitor can assert k0 protection. Compact: w0 + w_ge1 + K.
        per_modality["rollout_w0"] = 1.0
        per_modality["rollout_w_ge1"] = float(_w_ge1)
        per_modality["rollout_K"] = float(K)
    # Emit per-k loss shares (contingency trigger). Always-on regardless of the
    # k0-protection lever — production uses UNIFORM weighting, so this is the
    # ONLY window into k=0-share collapse. Shares sum to 1 over finite steps;
    # rollout_k0_share is the headline (watch it fall as K grows). For K=1 the
    # share is trivially 1.0 (single-step-equivalent phase).
    _finite_sum = sum(v for v in _k_loss_vals if not math.isnan(v))
    per_modality["rollout_K"] = float(K)
    if _finite_sum > 0.0:
        for _ki, _kv in enumerate(_k_loss_vals):
            per_modality[f"rollout_k{_ki}_share"] = (
                (_kv / _finite_sum) if not math.isnan(_kv) else float("nan")
            )
    return total_loss / K, per_modality


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
        # Multi-window: the input carries a leading history axis (B, K, C, ...);
        # the persistence baseline is the LAST (most recent) input window.
        if pred.dim() == target.dim() + 1:
            pred = pred[:, -1]
        if cfg.kind == "video":
            # MULTI-HORIZON video: under a rollout-native val horizon
            # (prediction_horizon_s > chunk_duration_s) the loader hands the
            # FULL future (K codec windows → K*n_frames frames on the frame
            # axis, dim 2), while the persistence baseline / model head are
            # ONE codec window (n_frames). Align the target's FRAME axis to the
            # input's before z-scoring so the copy baseline lives in the same
            # single-window shape as the prediction. The generic dim=-1 guard
            # below can't do this (video's last dim is W, not time). No-op for
            # single-step val (target frames == pred frames), so non-rollout /
            # d512 stay byte-identical.
            if pred.dim() == 5 and target.dim() == 5 and (
                target.shape[2] > pred.shape[2]
                and target.shape[2] % pred.shape[2] == 0
            ):
                target = target[:, :, : pred.shape[2]]
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
        # MULTI-HORIZON: the copy baseline (pred=input) is ONE window; under
        # prediction_horizon_s>chunk the TS/fast-TS target arrives as the K-window extended
        # future. Align target (+mask) to the copy's single-window width (spectro already
        # matched via trunc_t above; clean K-multiple guard = no-op for single-step).
        if target.shape[-1] > pred.shape[-1] and target.shape[-1] % pred.shape[-1] == 0:
            target = target[..., :pred.shape[-1]]
            if mask is not None:
                mask = mask[..., :pred.shape[-1]]
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
            # Multi-window: diag_inputs carries the (B, K, ...) history axis;
            # the persistence reference is the LAST input window.
            if inp.dim() == pred.dim() + 1:
                inp = inp[:, -1]
            tgt = targets[name].float()
            existing = masks[name].float() if masks[name] is not None else None
            # MULTI-HORIZON video: the video head predicts ONE codec window
            # (n_frames on the frame axis, dim 2) while a rollout-native val
            # horizon (prediction_horizon_s > chunk_duration_s) makes the loader
            # target span K codec windows (K*n_frames). Align the target's FRAME
            # axis (+ any per-frame mask) to the prediction's before the metric
            # math — the dim=-1 guard below is width (W) for video and can't fix
            # this. No-op when target frames == pred frames (single-step val), so
            # non-rollout / d512 stay byte-identical.
            if pred.dim() == 5 and tgt.dim() == 5 and (
                tgt.shape[2] > pred.shape[2]
                and tgt.shape[2] % pred.shape[2] == 0
            ):
                _npf = pred.shape[2]
                tgt = tgt[:, :, :_npf]
                # The video gate is (B, C, 1, 1, 1) — frame axis is broadcast (1)
                # and needs no slicing. Only slice a mask that actually carries a
                # per-frame axis longer than the prediction's.
                if existing is not None and existing.dim() == 5 \
                        and existing.shape[2] > _npf:
                    existing = existing[:, :, :_npf]
            # MULTI-HORIZON: under prediction_horizon_s>chunk the target is a K-window
            # extended future while the base head predicts ONE window. Align target (+mask)
            # to sub-window-0 (t+1) so the base val metric matches the single-step run; the
            # descriptor's t+h forecast is scored by the separate eval harness, not here.
            _pw = pred.shape[-1]
            if tgt.shape[-1] > _pw and tgt.shape[-1] % _pw == 0:
                tgt = tgt[..., :_pw]
                if existing is not None:
                    existing = existing[..., :_pw]

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
            elif name_to_kind.get(name) == "video" and cleaned_pred.dim() == 5:
                # Spatial-variance ratio: a flat mean-collapse has ~0 spatial
                # variance per frame; recovered structure → ~GT. (H,W are the
                # last two dims regardless of (B,C,T,H,W)/(B,T,C,H,W) order.)
                mvalid = (combined.amax(dim=(-1, -2)) > 0).float()  # (B,·,·)
                sums_t[4, j] += (cleaned_pred.var(dim=(-1, -2)) * mvalid).sum()
                sums_t[5, j] += (cleaned_tgt.var(dim=(-1, -2)) * mvalid).sum()
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
    parser.add_argument(
        "--stop_at_step", type=int, default=None,
        help="If set, break the train loop when step reaches this value while "
        "KEEPING --max_steps for the LR cosine T_max. Lets the K-anneal "
        "curriculum run block-segmented (each block a resume with its own "
        "--rollout_dataset_horizon_s) WITHOUT compressing the one-cosine-over-"
        "max_steps LR recipe. Default None → byte-identical (loop bounded only "
        "by --max_steps). Should be a multiple of --val_every so latest.pt is "
        "saved at the stop (block boundaries 5000/10000/15000 satisfy this).",
    )
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
        "--spec_freq_stem_from_codec", action="store_true",
        help="Warm-init each spectro tokenizer's freq_stem from the FROZEN "
             "codec's already-trained freq_stem (identical shape). The codec "
             "stem already surfaces whole-spectrum mode position, so copying it "
             "jump-starts the backbone stem instead of slow zero-init. Requires "
             "--spec_freq_stem + --spec_fsq. Applied before checkpoint load "
             "(INIT keeps it via allowed-missing; RESUME overwrites with the "
             "trained stem).",
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
        "--video_generative", action="store_true",
        help="Use the generative VideoFlowHead (resize-conv mean + rectified-"
             "flow residual + spatial PE) instead of the deterministic video "
             "head. Robust to imperfect backbone tokens: no checkerboard, no "
             "mean-collapse (see eval_runs/video_test/SUMMARY.md). WARM-START "
             "SAFE via --init_checkpoint (allowed_missing covers diag_heads."
             "<video>) — the new head inits fresh, backbone/encoder warm-load.",
    )
    parser.add_argument(
        "--video_flow_base_ch", type=int, default=64,
        help="Base channel width of the video flow head's velocity U-Net.",
    )
    parser.add_argument(
        "--video_flow_steps", type=int, default=16,
        help="Euler ODE steps used to sample the video flow head at eval.",
    )
    parser.add_argument(
        "--video_flow_lambda", type=float, default=1.0,
        help="Weight of the video flow-matching loss relative to the mean MAE.",
    )
    parser.add_argument(
        "--video_flow_pe_ch", type=int, default=16,
        help="Sinusoidal spatial (H and W) positional-embedding channels at the "
             "video flow velocity net's stem. 0 = off; 16 recommended.",
    )
    parser.add_argument(
        "--video_sigma_spatial", action="store_true",
        help="Per-pixel spatial residual σ for the video flow (the flow_ssig "
             "config): inject flow noise only where the residual varies → clean "
             "background. Off = per-channel scalar σ. Estimated from a short "
             "data pass at train start; recommended on.",
    )
    parser.add_argument(
        "--spec_generative", action="store_true",
        help="Use the generative SpectrogramFlowHead (rectified flow matching "
             "over a deterministic mean) instead of the deterministic "
             "spectrogram head. Samples sharp modes instead of regressing to "
             "the blurry conditional mean. NOT warm-start safe — from scratch.",
    )
    # ---- MaskGIT joint code head (fixes the independent-head collapse) ----
    parser.add_argument(
        "--spec_maskgit", action="store_true",
        help="Use the JOINT MaskGIT SpectrogramMaskGITHead instead of the "
             "independent per-patch SpectrogramCodeHead: a bidirectional "
             "transformer over the patch grid predicts codes jointly (BERT-style "
             "masked training, MaskGIT iterative parallel decode). Fixes the "
             "independent-head collapse (argmax->blocky, sample->speckle). "
             "Requires --spec_fsq + --spec_fsq_codec_dir.",
    )
    parser.add_argument("--backbone_input_skip", action="store_true",
                        help="Gated input->output residual skip around the backbone "
                             "(out = tokens + gate*backbone(tokens); LayerScale gate "
                             "init 0.2). Carries the localized mode past attention "
                             "dilution and is rollout-stable (bounded added term).")
    parser.add_argument("--spec_persistence_anchor", action="store_true",
                        help="Output-level persistence anchor: spectro prediction = "
                             "input window + head delta. Amplitude comes from the real "
                             "input mode (visible, not dampened); head learns the delta. "
                             "Stage-2 needs decode/re-encode rollout to feed the "
                             "prediction back as the next input.")
    parser.add_argument("--spec_warp_anchor", action="store_true",
                        help="Frequency-WARP anchor (drift fix): spectro prediction = "
                             "warp(input window, predicted shift field) + head delta. A "
                             "SpectroFreqWarpHead slides the real input ridge to its "
                             "forecast frequency, so a DRIFTING mode is tracked (not "
                             "stuck at the input frequency like the plain persistence "
                             "anchor). Visible (amplitude from the input), zero-init "
                             "warp → identity → persistence at start. Implies the anchor.")
    parser.add_argument("--spec_warp_max_bins", type=float, default=8.0,
                        help="Max |frequency shift| (in freq bins) the warp anchor can "
                             "apply, tanh-bounded. Default 8.")
    parser.add_argument("--spec_descriptor", action="store_true",
                        help="FACTORIZATION: auxiliary mode-descriptor head per spectro "
                             "modality forecasting the shift-stable band-power profile (the "
                             "modes the FSQ codes can't carry; descriptor persists 0.75-0.95 "
                             "t->t+1 vs codes 0.10-0.34). Main spectro head still makes the "
                             "(B,C,F,T) texture; the descriptor is rendered + gated for modes.")
    parser.add_argument("--use_actuator_film", action="store_true",
                        help="GATE-4 FiLM: inject actuators as per-block FiLM modulation of the backbone "
                             "(conditioning-by-construction) INSTEAD of the actuator-token pathway (removed "
                             "from the sequence). Anchor/descriptor head unchanged. Zero-init => identity at "
                             "warm-start. Tests whether construction-level conditioning survives the rollout.")
    parser.add_argument("--spec_descriptor_weight", type=float, default=4.0,
                        help="Priority for the descriptor loss (EMA-normalized under "
                             "--loss_norm_ema so the code CE can't starve it).")
    parser.add_argument("--spec_descriptor_tcol", type=int, default=6,
                        help="Descriptor time columns per window.")
    parser.add_argument("--spec_descriptor_hidden", type=int, default=512,
                        help="Descriptor head MLP width.")
    parser.add_argument("--spec_descriptor_horizons", type=str, default="1",
                        help="Comma-list of forecast horizons in WINDOWS ahead (e.g. '1' = t+1 "
                             "legacy; '2,4' = predict t+2 AND t+4 = Gate 2b multi-target). The head "
                             "emits one readout per horizon; each is supervised on the matching "
                             "sub-window of the extended target. Requires --prediction_horizon_s = "
                             "max(horizons)*chunk_duration_s so the loader carries that far ahead; "
                             "base (non-descriptor) losses then use sub-window-0 (t+1) only.")
    parser.add_argument("--reinit_act_tokenizers", action="store_true",
                        help="GATE3-FIX: on --init_checkpoint warm-start, DROP the actuator tokenizer "
                             "weights and re-init them fresh (backbone/heads still warm-start). Use when "
                             "the actuator input distribution changed (standardization / channel drop) so "
                             "the checkpoint's raw-scale actuator convs would be mis-calibrated.")
    parser.add_argument("--spec_descriptor_loss", choices=["mse", "dist"], default="mse",
                        help="Descriptor loss: 'dist' = distribution-CE over freq (optimizes "
                             "mode peak LOCATION = what desc_ftol measures; cannot mean-collapse). "
                             "'mse' = profile MSE (mean-collapses to the near-zero residual → "
                             "ftol stuck at chance). Use 'dist'.")
    parser.add_argument("--spec_descriptor_dist_beta", type=float, default=4.0,
                        help="Sharpness of the soft-target freq distribution: "
                             "softmax((profile/peak)*beta) over freq.")
    parser.add_argument("--spec_descriptor_anchor", action="store_true",
                        help="PERSISTENCE ANCHOR: descriptor pred = current-window "
                             "descriptor (logits) + head residual; head final layer "
                             "zero-init -> starts at persistence, learns only the drift. "
                             "Cannot collapse to flat. Requires --spec_descriptor_loss dist.")
    parser.add_argument("--spec_descriptor_anchor_beta_holds", type=str, default="",
                        help="GATE-3-FIX anchor-β ANNEAL (unmask latent actuator conditioning): "
                             "comma β sequence held STEPWISE (e.g. '8,6,5,4,3'). The anchor's "
                             "PREDICTION weight = holds[step//hold_steps]; the TARGET softmax keeps "
                             "the FIXED --spec_descriptor_dist_beta (task definition unchanged). "
                             "Empty = no anneal (byte-identical to fixed-β training).")
    parser.add_argument("--spec_descriptor_anchor_beta_hold_steps", type=int, default=1500,
                        help="Steps held at each β in --spec_descriptor_anchor_beta_holds; a milestone "
                             "checkpoint (equilibrated head at that β) is saved at each hold boundary.")
    parser.add_argument("--desc_false_death_abort", type=float, default=0.01,
                        help="False-drift TRIPWIRE: best.pt promotion is BLOCKED at any val whose "
                             "per-window mean desc_fdrift exceeds this (collapse guard). Log-and-continue: "
                             "the job keeps running + milestones the whole β trajectory; only best.pt is gated.")
    parser.add_argument("--spec_descriptor_transition_weight", type=float, default=1.0,
                        help="Upweight onset/death (input->target presence-change) windows in "
                             "the descriptor loss by this factor — class-imbalance fix on the "
                             "rare, non-copyable, LEARNABLE transition events. 1.0 = off.")
    # ---- OPT-IN K-step rollout training (train == Gate-4 inference feedback) ----
    # All defaults are the OFF/identity value → the single-step path is
    # byte-identical when --k_rollout is absent.
    parser.add_argument(
        "--k_rollout", action="store_true",
        help="Train with the K-step token-space rollout (argmax code feedback, "
             "the Gate-4 wiring) instead of the single-step forward. The spectro "
             "descriptor anchor reads the ROLLED-OUT state at each step so train "
             "matches inference. OFF (default) = byte-identical single-step path.")
    parser.add_argument(
        "--curriculum_Ks", type=str, default="10,20,40,80",
        help="Comma-separated rollout depths; K advances one entry per "
             "--block_steps training steps (clamped at the last). Only used "
             "with --k_rollout.")
    parser.add_argument(
        "--block_steps", type=int, default=5000,
        help="Training steps per curriculum-K block (see --curriculum_Ks).")
    parser.add_argument(
        "--tf_anneal_steps", type=int, default=0,
        help="Teacher-forcing anneal horizon: p_tf = max(0, 1 - step/tf_anneal_steps) "
             "when >0, else 0.0 (pure free rollout). Only used with --k_rollout.")
    parser.add_argument(
        "--rollout_grad_checkpoint_every", type=int, default=0,
        help="Wrap each contiguous group of this many rollout steps in "
             "torch.utils.checkpoint (use_reentrant=False) so activation memory "
             "scales with the group size, not K. 0 (default) = no checkpointing "
             "(results identical either way; memory only). Only used with --k_rollout.")
    parser.add_argument(
        "--feedback_normalize", action="store_true",
        help="With --k_rollout, apply option-2 POST-tokenizer feedback-token "
             "renorm (both the argmax free path and the teacher-forcing path): "
             "tokenize the feedback the normal way (decode(codes) -> tokenizer), "
             "then scale each code-path modality's (B,n_tok,d_model) slice "
             "per-sample by clamp(step0_input_absmax / feedback_absmax, max=1.0) "
             "so its token absmax never exceeds the step-0 input-window band. "
             "Cures the localized ece proj-conv NaN: the patch conv's ONE fixed "
             "near-DC filter saturates on the codec-decoded broadband floor "
             "(feedback proj ~2216 > the ~2101 tolerated input band -> bf16 NaN, "
             "hottest in teacher forcing). The uniform per-sample scalar preserves "
             "relative token structure (mode-safe); it only scales DOWN so in-band "
             "feedback is untouched. OFF (default) = byte-identical feedback (does "
             "not change the running production chain).")
    parser.add_argument(
        "--rollout_dataset_horizon_s", type=float, default=None,
        help="With --k_rollout, the DATASET's prediction_horizon_s (the future "
             "SPAN the loader emits) — decoupled from the MODEL's "
             "--prediction_horizon_s (which stays fixed at 0.2 for actuator/"
             "geometry warm-start compatibility). Default = "
             "max(curriculum_Ks)*chunk_duration_s + prediction_horizon_s so the "
             "loader supplies enough future windows for the deepest rollout + "
             "the descriptor's t+h reach.")
    # ---- STRIKE-3 (K=10-gate failure fix) — two OPT-IN loss levers ----
    # Both default to the IDENTITY value → byte-identical when unset.
    parser.add_argument(
        "--drift_penalty_weight", type=float, default=0.0,
        help="STRIKE-3 LEVER 1 (ASYMMETRIC drift penalty). Adds "
             "drift_penalty_weight * relu(pred_drift - gt_drift) to the ece "
             "descriptor loss, where drift = |centroid(desc) - centroid(anchor)| "
             "(the prominence-weighted freq centroid gate4_kprobe reports as "
             "drift_pred, measured off the persistence anchor). Penalizes ONLY "
             "OVER-drift (relu) — the K=10-gate pathology — never legit "
             "under-drift corrections. 0.0 (default) = OFF, byte-identical.")
    parser.add_argument(
        "--k_ge1_weight", type=float, default=1.0,
        help="STRIKE-3 LEVER 2 (k0-protected per-k re-weighting). Constant "
             "multiplier on the k>=1 rollout-step loss terms; k=0 is PINNED at "
             "1.0 (keeps its Stage-1 gradient share). 1.0 (default) = uniform "
             "summing, byte-identical. Only used with --k_rollout.")
    parser.add_argument(
        "--k_ge1_weight_anneal_steps", type=int, default=0,
        help="STRIKE-3 LEVER 2 anneal: if >0, ramp the k>=1 weight linearly from "
             "--k_ge1_weight_start up to 1.0 over this many GLOBAL steps (k=0 stays "
             "1.0 throughout). 0 (default) = no anneal (constant --k_ge1_weight). "
             "Only used with --k_rollout.")
    parser.add_argument(
        "--k_ge1_weight_start", type=float, default=0.1,
        help="STRIKE-3 LEVER 2 anneal start value for the k>=1 weight (used only "
             "when --k_ge1_weight_anneal_steps>0). Default 0.1. k=0 always 1.0.")
    parser.add_argument("--spec_mode_band_weight", type=float, default=1.0,
                        help="Up-weight the coherent-mode band (--spec_mode_band_lo/hi_khz, "
                             "default 5-40 kHz) in the spectrogram MAE by this factor "
                             "(1.0 = off). Counters the broadband/DC-dominated loss that "
                             "starves the thin mode of gradient (amplitude-undershoot / "
                             "mean-collapse). Applied via weighted_masked_mae.")
    parser.add_argument("--spec_mode_band_lo_khz", type=float, default=5.0)
    parser.add_argument("--spec_mode_band_hi_khz", type=float, default=40.0)
    parser.add_argument("--history_windows", type=int, default=1,
                        help="K>1: feed K consecutive past 50ms windows to a "
                             "spatiotemporal backbone (causal temporal attention "
                             "across windows) and predict the NEXT window. Lets "
                             "the model see mode VELOCITY (drift/growth) that a "
                             "single-window Markov backbone cannot. K=1 = "
                             "single-window (production default, unchanged).")
    parser.add_argument("--spec_maskgit_dim", type=int, default=512,
                        help="MaskGIT transformer width.")
    parser.add_argument("--spec_maskgit_layers", type=int, default=4,
                        help="MaskGIT transformer depth.")
    parser.add_argument("--spec_maskgit_heads", type=int, default=8,
                        help="MaskGIT transformer attention heads.")
    parser.add_argument("--spec_maskgit_decode_steps", type=int, default=10,
                        help="MaskGIT iterative-unmask passes at inference.")
    parser.add_argument("--spec_maskgit_decode_temp", type=float, default=0.5,
                        help="MaskGIT decode temperature (<=1e-3 => argmax; low = "
                             "precise/confident, ~1.0 = diverse sampling).")
    # ---- Phase 1b: discrete FSQ code head (frozen adversarial codec) ----
    parser.add_argument(
        "--spec_fsq", action="store_true",
        help="Use the discrete SpectrogramCodeHead: predict a FROZEN adversarial-"
             "FSQ codec's per-dim codes via class-weighted CE (categorical, cannot "
             "mean-collapse). The frozen decoder renders sharp modes from any codes. "
             "Requires --spec_fsq_codec_dir with spectro_codec_<modality>.pt (Phase "
             "1a). WARM-START SAFE (input tokenization unchanged; only the output "
             "path becomes code-based). Takes precedence over --spec_generative.",
    )
    parser.add_argument(
        "--spec_fsq_codec_dir", type=str, default="",
        help="Directory holding the frozen Phase-1a codecs "
             "(spectro_codec_<modality>.pt), loaded per spectro modality.",
    )
    parser.add_argument(
        "--spec_code_pred_hidden", type=int, default=512,
        help="Hidden width of the per-token code-prediction MLP trunk.",
    )
    parser.add_argument(
        "--spec_code_pred_layers", type=int, default=2,
        help="Number of MLP layers in the code-prediction trunk.",
    )
    parser.add_argument(
        "--spec_code_temperature", type=float, default=1.0,
        help="Softmax temperature for eval-time code sampling (viz diversity).",
    )
    parser.add_argument(
        "--spec_code_class_weight", type=float, default=10.0,
        help="Cap on the per-(dim,level) inverse-frequency CE class weight "
             "(data-normalized so E_data[w]=1). Keeps rare MODE codes in the loss "
             "against the frequent background code (majority-class collapse). "
             "1.0 = uniform CE (off).",
    )
    parser.add_argument(
        "--spec_code_focal_gamma", type=float, default=0.0,
        help="Focal-loss gamma on the spectro code CE: multiply per-token CE by "
             "(1-p_true)^gamma so easy/background codes contribute less gradient and "
             "hard/rare MODE codes dominate. 0.0 = off (plain CE); ~2.0 = focal. "
             "Composes with --spec_code_class_weight.",
    )
    parser.add_argument(
        "--spec_ordinal_eps", type=float, default=0.0,
        help="Soft/ordinal CE for spectro codes: target [eps,1-2eps,eps] over "
             "[k-1,k,k+1] (edge-clamped), rewarding +-1-level predictions (FSQ codes "
             "dither +-1 per the mode audit). 0.0 = hard CE (off). e2e/ordinal_loss.py.",
    )
    parser.add_argument(
        "--spec_code_weight_batches", type=int, default=50,
        help="Number of training batches used to estimate the code-class "
             "frequencies for --spec_code_class_weight (one-time pre-pass).",
    )
    parser.add_argument(
        "--loss_norm_ema", action="store_true",
        help="Per-modality EMA loss normalization: divide each modality's loss by a "
             "running EMA of its magnitude so every modality contributes O(1) to the "
             "total (fixes spectro-vs-slowTS gradient starvation). Off = unchanged.",
    )
    parser.add_argument("--loss_norm_beta", type=float, default=0.99,
                        help="EMA decay for --loss_norm_ema.")
    parser.add_argument("--loss_priority_spectro", type=float, default=1.0,
                        help="Loss-norm priority multiplier for spectro modalities "
                             "(>1 over-drives modes). Requires --loss_norm_ema.")
    parser.add_argument(
        "--spec_autoencode", action="store_true",
        help="DIAGNOSTIC: target the CURRENT input window's own codes "
             "(encode_target(diag_inputs)) instead of the NEXT window's "
             "(forecasting). Isolates representation capacity from "
             "forecast-irreducibility: if the model can autoencode-to-recon "
             "but not forecast-to-recon, the future simply isn't determined "
             "by the past. Spectro code head only; default off.",
    )
    # ---- Phase 1b: discrete FSQ VIDEO code head (frozen video codec) ----
    parser.add_argument(
        "--video_fsq", action="store_true",
        help="Use the discrete VideoCodeHead: predict a FROZEN adversarial-FSQ "
             "video codec's codes via class-weighted CE (frozen resize-conv decoder "
             "→ sharp, checkerboard-free frames). Requires --video_fsq_codec_dir with "
             "video_codec_<modality>.pt. Takes precedence over --video_generative/"
             "--video_resize_conv for the video head.",
    )
    parser.add_argument(
        "--video_fsq_codec_dir", type=str, default="",
        help="Directory holding the frozen video codecs (video_codec_<mod>.pt), "
             "loaded per video modality (tangtv_lower / tangtv_upper).",
    )
    parser.add_argument("--video_code_pred_hidden", type=int, default=512,
                        help="Hidden width of the per-token video code-prediction MLP.")
    parser.add_argument("--video_code_pred_layers", type=int, default=2,
                        help="MLP layers in the video code-prediction trunk.")
    parser.add_argument("--video_code_temperature", type=float, default=1.0,
                        help="Softmax temperature for eval-time video code sampling.")
    parser.add_argument(
        "--video_code_class_weight", type=float, default=4.0,
        help="Cap on per-(dim,level) inverse-freq CE class weight for video codes "
             "(data-normalized, E_data[w]=1). 1.0 = uniform CE (off).",
    )
    parser.add_argument(
        "--video_code_weight_batches", type=int, default=50,
        help="Batches used to estimate video code-class frequencies (one-time pre-pass).",
    )
    parser.add_argument(
        "--fastts_fsq", action="store_true",
        help="Use the discrete FastTimeSeriesCodeHead for filterscopes: predict a "
             "FROZEN fast-TS FSQ codec's per-dim codes via class-weighted CE "
             "(categorical → keeps sharp ELM spikes vs MAE-smoothing). Requires "
             "--fastts_fsq_codec_dir with fastts_codec.pt (or fastts_codec_<mod>.pt). "
             "Input tokenization unchanged (warm-start safe); swaps only the head.",
    )
    parser.add_argument(
        "--fastts_fsq_codec_dir", type=str, default="",
        help="Directory holding the frozen fast-TS codec (fastts_codec.pt or "
             "fastts_codec_<modality>.pt).",
    )
    parser.add_argument("--fastts_code_pred_hidden", type=int, default=512,
                        help="Hidden width of the per-token fast-TS code-prediction MLP.")
    parser.add_argument("--fastts_code_pred_layers", type=int, default=2,
                        help="MLP layers in the fast-TS code-prediction trunk.")
    parser.add_argument("--fastts_code_temperature", type=float, default=1.0,
                        help="Softmax temperature for eval-time fast-TS code sampling.")
    parser.add_argument(
        "--fastts_code_class_weight", type=float, default=4.0,
        help="Cap on per-(dim,level) inverse-freq CE class weight for fast-TS codes "
             "(data-normalized, E_data[w]=1). Up-weights rare ELM-spike codes. "
             "1.0 = uniform CE (off).",
    )
    parser.add_argument(
        "--fastts_code_weight_batches", type=int, default=50,
        help="Batches used to estimate fast-TS code-class frequencies (one-time pre-pass).",
    )
    parser.add_argument(
        "--slow_ts_fsq", action="store_true",
        help="Use the discrete SlowTimeSeriesCodeHead for slow-TS (Thomson/CER/MSE): "
             "predict a FROZEN per-modality slow-TS FSQ codec's codes via class-weighted "
             "CE (unified discrete world-model). Requires --slow_ts_fsq_codec_dir with "
             "slowts_codec_<modality>.pt. Input tokenization unchanged (warm-start safe).",
    )
    parser.add_argument(
        "--slow_ts_fsq_codec_dir", type=str, default="",
        help="Directory holding the frozen slow-TS codecs (slowts_codec_<modality>.pt).",
    )
    parser.add_argument(
        "--no_video_presence_filter", action="store_true",
        help="Train on ALL shots — skip filtering the file list to video-present "
             "shots. Absent video is zero-filled + loss-masked by the per-sample "
             "gates. Default off (keeps the existing video-present chain identical).",
    )
    parser.add_argument("--slow_ts_code_pred_hidden", type=int, default=512,
                        help="Hidden width of the per-token slow-TS code-prediction MLP.")
    parser.add_argument("--slow_ts_code_pred_layers", type=int, default=2,
                        help="MLP layers in the slow-TS code-prediction trunk.")
    parser.add_argument("--slow_ts_code_temperature", type=float, default=1.0,
                        help="Softmax temperature for eval-time slow-TS code sampling.")
    parser.add_argument(
        "--slow_ts_code_class_weight", type=float, default=4.0,
        help="Cap on per-(dim,level) inverse-freq CE class weight for slow-TS codes "
             "(data-normalized, E_data[w]=1). 1.0 = uniform CE (off).",
    )
    parser.add_argument(
        "--slow_ts_code_weight_batches", type=int, default=50,
        help="Batches used to estimate slow-TS code-class frequencies (one-time pre-pass).",
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
        "--spec_struct_lambda", type=float, default=0.0,
        help="(A) Weight of the structural mode-coherence soft-Dice loss on the "
             "generative spectro head's deterministic mean mu. 0.0 = OFF "
             "(default; existing chains unaffected). Mirrors the production "
             "GT-fusion binarization (gaussian smooth -> per-freq z over time -> "
             "clip(z/k,0,1)^2; per-modality k ECE 2.5 / CO2 2.0). Pushes mu "
             "toward sharp coherent ridges. Use a CONSERVATIVE weight (e.g. "
             "0.1-0.3): lambda=1.0 craters reconstruction SSIM in the benchmark.",
    )
    parser.add_argument(
        "--spec_mask", action="store_true",
        help="(Plan B, 2026-06-30) Add a mode-MASK prediction branch to the "
             "generative spectro head: a SEPARATE decode predicting the "
             "production-binarized mode field, trained with soft-Dice+BCE. A "
             "segmentation loss has no L2 mean-seeking optimum, so it does NOT "
             "collapse the way the mu MAE and the flow velocity MSE do (proven "
             "by the overfit-prediction test — both mu AND sample stayed flat). "
             "The mask supplies sharp mode LOCATIONS (presence is predictable, "
             "survival half-life ~201 ms); render fuses mu with the PREDICTED "
             "mask (no GT → genuine forecast). Adds params → warm-start inits "
             "the branch fresh. Needs --spec_generative.",
    )
    parser.add_argument(
        "--spec_mask_lambda", type=float, default=0.0,
        help="Weight of the mode-mask segmentation loss (soft-Dice+BCE). "
             "0.0 = OFF (default; existing chains unaffected).",
    )
    parser.add_argument(
        "--spec_mask_hidden", type=int, default=64,
        help="Hidden channel width of the mode-mask decode head.",
    )
    parser.add_argument(
        "--spec_input_feat", action="store_true",
        help="Give the mode-mask decode the INPUT-window mode mask as LEARNABLE "
             "feature channels (concat), so the convs learn to combine observed "
             "modes with forecast-token dynamics — copy persistence AND predict "
             "changes. Bypasses the forecast-token mode-collapse (the tokens may "
             "not carry modes; the input does). Distinct from --spec_input_cond "
             "(a fixed logit-add that collapsed). AR: rollout feeds the previous "
             "predicted mask. Needs --spec_mask.",
    )
    parser.add_argument(
        "--spec_flow_residual_anchor", action="store_true",
        help="Generative flow head models the residual over the REAL input "
             "window (persistence anchor) instead of a learned mean: "
             "pred = input + sampled_residual. The flow SAMPLES the mode's "
             "stochastic amplitude growth (target-input) → a single draw is "
             "SHARP and full-amplitude; the input anchor floors tvr~1 so it "
             "cannot dampen (fixes the prior flow's tvr 0.23). Needs "
             "--spec_generative + --spec_persistence_anchor; pair with "
             "--spec_mode_band_weight to make the flow model the sharp mode.",
    )
    parser.add_argument(
        "--spec_mae_lambda", type=float, default=1.0,
        help="Weight of the spectrogram pixel MAE. 1.0 = default. Set 0.0 to "
             "REMOVE the mean-seeking MAE's grip on the spectro tokens so they "
             "are shaped only by the (non-collapsing) mode-mask objective — the "
             "architecture fix for mode-collapse. (0*mae keeps the mean_head in "
             "the graph → DDP-safe.)",
    )
    parser.add_argument(
        "--spec_mask_loss", type=str, default="dice",
        choices=["dice", "tversky", "sparse"],
        help="Mode-mask segmentation loss. 'sparse' = dice + POS-WEIGHTED BCE "
             "(mode class weighted ~1/density → ~40x stronger gradient on the "
             "sparse missed modes; fixes the dice near-empty plateau). 'tversky' "
             "(α=0.3,β=0.7) penalizes missed modes more. 'dice' = default.",
    )
    parser.add_argument(
        "--spec_input_cond", action="store_true",
        help="(recurrence-ready input-conditioning, 2026-06-30) Add a PERSISTENCE "
             "prior to the mode-mask head: mask_logits += gain·logit(prior), where "
             "the Stage-1 prior is the input-window mode mask (per-freq presence). "
             "The overfit tests showed the backbone forecast tokens don't carry "
             "specific mode locations, but ECE modes persist (τ½ 201 ms ≫ 50 ms), "
             "so the head learns only the residual change on top of copy-forward. "
             "Head-level skip from observed data — NO extra backbone tokens. Stage 2 "
             "swaps in the previous predicted mask (a decaying mask recurrence). "
             "Needs --spec_mask. Learnable prior gain inits fresh (warm-start safe).",
    )
    parser.add_argument(
        "--spec_flow_freq_pe_ch", type=int, default=0,
        help="Sinusoidal frequency-positional-embedding channels concatenated "
             "at the flow velocity net's stem (gives the convs absolute-freq "
             "awareness to place modes). 0 = off (legacy). Must be even; 16 "
             "recommended. Pair with finer-freq patches (e.g. --spectro_patch_f "
             "64 --spectro_patch_t 32) for 8 freq tokens at the same count.",
    )
    parser.add_argument(
        "--spec_flow_time_pe_ch", type=int, default=0,
        help="Sinusoidal TIME-positional-embedding channels at the flow "
             "velocity net's stem (absolute-time awareness for bursts/chirps). "
             "0 = off. Matters most when finer-freq patches leave few time "
             "tokens (e.g. (64,32) -> 3 time tokens). Must be even; 8 typical.",
    )
    parser.add_argument(
        "--lazy_optimizer_load", action="store_true",
        help="On resume, skip the eager optimizer-state load and let the first "
             "opt.step() allocate the momentum buffers fresh (cold-start's clean "
             "lazy-allocation order), then fill the saved momentum in-place. "
             "Avoids the eager-load fragmentation OOM at the VRAM ceiling. "
             "Default off → eager load.",
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

    # ── K-rollout dataset-horizon decoupling ──
    # The MODEL geometry (actuator tokenizer conv, backbone token layout) is
    # governed by args.prediction_horizon_s (must stay at the warm-start value,
    # 0.2). The DATASET future SPAN is decoupled: build_datasets uses this
    # dataset_horizon so the loader emits enough future windows for the deepest
    # rollout K + the descriptor's t+h reach. Only active with --k_rollout;
    # otherwise the dataset uses prediction_horizon_s (byte-identical).
    _curriculum_Ks: List[int] = [
        int(x) for x in str(args.curriculum_Ks).split(",") if x.strip()
    ]
    if args.k_rollout:
        if not _curriculum_Ks:
            raise SystemExit("--k_rollout requires a non-empty --curriculum_Ks")
        if args.rollout_dataset_horizon_s is not None:
            dataset_horizon_s = float(args.rollout_dataset_horizon_s)
        else:
            dataset_horizon_s = (
                max(_curriculum_Ks) * args.chunk_duration_s
                + args.prediction_horizon_s
            )
    else:
        dataset_horizon_s = args.prediction_horizon_s

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
    # --no_video_presence_filter → train on ALL shots (absent video is zero-filled
    # + loss-masked by the per-sample gates); log it so the choice is visible.
    if args.no_video_presence_filter:
        logger.info(
            f"ALL-shots mode: video-presence filter SKIPPED — "
            f"train {len(train_files)} / val {len(val_files)} files."
        )
    # No-op when args.use_video is empty (G2/G3 stay byte-identical).
    if args.use_video and not args.no_video_presence_filter:
        n_train_before = len(train_files)
        n_val_before = len(val_files)
        args.lengths_cache_dir.mkdir(parents=True, exist_ok=True)
        video_groups = _video_hdf5_groups(args.use_video)
        train_files = filter_video_present_files(
            train_files,
            video_groups,
            cache_path=args.lengths_cache_dir / "video_present_train.pt",
        )
        val_files = filter_video_present_files(
            val_files,
            video_groups,
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
        prediction_horizon_s=args.prediction_horizon_s,
    )
    diagnostic_names = [c.name for c in diagnostics]
    actuator_names = [c.name for c in actuators]
    logger.info(
        f"Diagnostics ({len(diagnostics)}): " + ", ".join(diagnostic_names)
    )
    logger.info(
        f"Actuators ({len(actuators)}): " + ", ".join(actuator_names)
    )

    _desc_horizons = tuple(int(x) for x in str(args.spec_descriptor_horizons).split(",") if x.strip())
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
        video_generative=args.video_generative,
        video_flow_base_ch=args.video_flow_base_ch,
        video_flow_sample_steps=args.video_flow_steps,
        video_flow_lambda=args.video_flow_lambda,
        video_flow_pe_ch=args.video_flow_pe_ch,
        video_sigma_spatial=args.video_sigma_spatial,
        spectro_generative=args.spec_generative,
        spectro_flow_base_ch=args.spec_flow_base_ch,
        spectro_flow_sample_steps=args.spec_flow_steps,
        spectro_flow_lambda=args.spec_flow_lambda,
        spectro_flow_freq_pe_ch=args.spec_flow_freq_pe_ch,
        spectro_flow_time_pe_ch=args.spec_flow_time_pe_ch,
        spectro_mask=args.spec_mask,
        spectro_mask_hidden=args.spec_mask_hidden,
        spectro_input_cond=args.spec_input_cond,
        spectro_input_feat=args.spec_input_feat,
        spectro_flow_residual_anchor=args.spec_flow_residual_anchor,
        spectro_fsq=args.spec_fsq,
        spectro_fsq_codec_dir=args.spec_fsq_codec_dir,
        spectro_code_pred_hidden=args.spec_code_pred_hidden,
        spectro_code_pred_layers=args.spec_code_pred_layers,
        spectro_code_temperature=args.spec_code_temperature,
        backbone_input_skip=args.backbone_input_skip,
        spec_persistence_anchor=args.spec_persistence_anchor,
        spec_warp_anchor=args.spec_warp_anchor,
        spec_warp_max_bins=args.spec_warp_max_bins,
        spec_descriptor=args.spec_descriptor,
        spec_descriptor_tcol=args.spec_descriptor_tcol,
        spec_descriptor_hidden=args.spec_descriptor_hidden,
        spec_descriptor_horizons=_desc_horizons,
        history_windows=args.history_windows,
        use_actuator_film=args.use_actuator_film,
        spectro_maskgit=args.spec_maskgit,
        spectro_maskgit_dim=args.spec_maskgit_dim,
        spectro_maskgit_layers=args.spec_maskgit_layers,
        spectro_maskgit_heads=args.spec_maskgit_heads,
        spectro_maskgit_decode_steps=args.spec_maskgit_decode_steps,
        spectro_maskgit_decode_temp=args.spec_maskgit_decode_temp,
        video_fsq=args.video_fsq,
        video_fsq_codec_dir=args.video_fsq_codec_dir,
        video_code_pred_hidden=args.video_code_pred_hidden,
        video_code_pred_layers=args.video_code_pred_layers,
        video_code_temperature=args.video_code_temperature,
        fastts_fsq=args.fastts_fsq,
        fastts_fsq_codec_dir=args.fastts_fsq_codec_dir,
        fastts_code_pred_hidden=args.fastts_code_pred_hidden,
        fastts_code_pred_layers=args.fastts_code_pred_layers,
        fastts_code_temperature=args.fastts_code_temperature,
        slow_ts_fsq=args.slow_ts_fsq,
        slow_ts_fsq_codec_dir=args.slow_ts_fsq_codec_dir,
        slow_ts_code_pred_hidden=args.slow_ts_code_pred_hidden,
        slow_ts_code_pred_layers=args.slow_ts_code_pred_layers,
        slow_ts_code_temperature=args.slow_ts_code_temperature,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_total_tokens = model.n_total_tokens

    # --spec_freq_stem_from_codec: warm-init each spectro tokenizer's freq_stem
    # from the FROZEN codec's trained freq_stem (identical freq->hidden->freq
    # shape). The codec stem already knows how to surface whole-spectrum mode
    # position; copying it jump-starts the backbone stem vs slow zero-init.
    # BEFORE any checkpoint load: INIT keeps it (freq_stem is allowed-missing),
    # RESUME overwrites it with the trained stem (correct).
    if getattr(args, "spec_freq_stem_from_codec", False):
        _fs_core = _core(model)
        _fs_copied = []
        for _name, _head in _fs_core.diag_heads.items():
            _tok = (_fs_core.diag_tokenizers[_name]
                    if _name in _fs_core.diag_tokenizers else None)
            if (isinstance(_head, SpectrogramCodeHead)
                    and _tok is not None
                    and getattr(_tok, "enable_freq_stem", False)
                    and getattr(_head.codec.enc, "enable_freq_stem", False)):
                _tok.fs_lin1.load_state_dict(_head.codec.enc.fs_lin1.state_dict())
                _tok.fs_lin2.load_state_dict(_head.codec.enc.fs_lin2.state_dict())
                _fs_copied.append(_name)
        logger.info(f"Warm-init freq_stem from codec for: {_fs_copied}")

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
    if getattr(args, "spec_fsq", False):
        logger.info(
            f"Spectro FSQ head — patch_f={args.spectro_patch_f} "
            f"patch_t={args.spectro_patch_t}  "
            f"class_weight_cap={args.spec_code_class_weight} "
            f"focal_gamma={args.spec_code_focal_gamma} "
            f"autoencode={getattr(args, 'spec_autoencode', False)}"
        )

    # ── Datasets ────────────────────────────────────────────────────────
    # NOTE: build_configs (above) used args.prediction_horizon_s for the MODEL
    # geometry; the DATASET future span uses the (possibly decoupled)
    # dataset_horizon_s so --k_rollout gets enough future windows without
    # perturbing the actuator-tokenizer conv shape (warm-start compatibility).
    if args.k_rollout and dataset_horizon_s != args.prediction_horizon_s:
        logger.info(
            f"K-rollout: dataset prediction_horizon_s={dataset_horizon_s:.4f}s "
            f"(decoupled) while MODEL prediction_horizon_s="
            f"{args.prediction_horizon_s:.4f}s (actuator/geometry fixed)."
        )
    train_ds, val_ds = build_datasets(
        args.data_dir,
        train_files,
        val_files,
        preprocessing_stats=stats,
        chunk_duration_s=args.chunk_duration_s,
        prediction_horizon_s=dataset_horizon_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
        lengths_cache_dir=args.lengths_cache_dir,
        history_windows=args.history_windows,
        # K-rollout: VAL stays single-step at the model horizon so validate()'s
        # single-step forward_batch feeds the actuator tokenizer the right patch
        # count. Non-k_rollout: equals dataset_horizon_s → byte-identical.
        val_prediction_horizon_s=args.prediction_horizon_s,
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

    # Mode-band up-weight: multiply (or set) the spectro MAE weight so the
    # coherent-mode band gets FACTOR× the gradient. Counters the DC/broadband
    # domination that starves the thin mode (amplitude-undershoot). Combines
    # with per-bin weights when both are on.
    if args.spec_mode_band_weight > 1.0:
        mb = build_spec_mode_band_weights(
            _core(model).diagnostics, args.spec_mode_band_weight, device,
            lo_khz=args.spec_mode_band_lo_khz, hi_khz=args.spec_mode_band_hi_khz,
        )
        if spec_pb_weights is None:
            spec_pb_weights = mb
        else:
            for n, w in mb.items():
                if n in spec_pb_weights and spec_pb_weights[n].shape == w.shape:
                    spec_pb_weights[n] = spec_pb_weights[n] * w
                else:
                    spec_pb_weights[n] = w
        for n, w in mb.items():
            logger.info(
                f"Mode-band spec loss [{n}]: {args.spec_mode_band_weight:.1f}x in "
                f"[{args.spec_mode_band_lo_khz:.0f},{args.spec_mode_band_hi_khz:.0f}] kHz"
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

    # ── Per-pixel residual σ for the generative video head ──────────────
    # The video analog of the spectro per-bin σ (set above). Estimated from a
    # short pass over the train loader; confines flow noise to where the target
    # varies (clean quiet background). All-reduced inside the helper → DDP-safe.
    if args.video_generative:
        core_v = _core(model)
        vid_sigma = build_video_pixel_sigma(core_v, train_loader, device, n_batches=8)
        for cfg in core_v.diagnostics:
            head = core_v.diag_heads[cfg.name]
            if isinstance(head, VideoFlowHead) and cfg.name in vid_sigma:
                sig = vid_sigma[cfg.name]
                if head.sigma_pb.shape[-1] == 1:           # scalar-σ buffer
                    sig = sig.mean(dim=(1, 2), keepdim=True)
                head.set_sigma_pb(sig.to(device))
                logger.info(
                    f"Video flow head [{cfg.name}]: residual σ set from 8 "
                    f"batches, buffer {tuple(head.sigma_pb.shape)} "
                    f"(mean σ={float(head.sigma_pb.mean()):.3f})"
                )

    # ── FSQ code-class weights (Phase 1b) ──────────────────────────────
    # One-time inverse-frequency weights so the class-weighted CE keeps the rare
    # MODE codes in the loss vs the frequent background code. All-reduced inside
    # the helper → identical across ranks (DDP-correct). {} when --spec_fsq off.
    spec_code_class_weights: Dict[str, torch.Tensor] = {}
    if args.spec_fsq and args.spec_code_class_weight > 0:
        spec_code_class_weights = build_spec_code_class_weights(
            _core(model), train_loader, device,
            cap=args.spec_code_class_weight,
            n_batches=args.spec_code_weight_batches,
        )
        for name, w in spec_code_class_weights.items():
            logger.info(
                f"FSQ code-class weights [{name}]: shape {tuple(w.shape)}, "
                f"range [{float(w.min()):.3f}, {float(w.max()):.3f}], "
                f"mean {float(w.mean()):.3f} (cap={args.spec_code_class_weight}, "
                f"{args.spec_code_weight_batches} batches)"
            )

    video_code_class_weights: Dict[str, torch.Tensor] = {}
    if args.video_fsq:
        video_code_class_weights = build_video_code_class_weights(
            _core(model), train_loader, device,
            cap=args.video_code_class_weight,
            n_batches=args.video_code_weight_batches,
        )
        for name, w in video_code_class_weights.items():
            logger.info(
                f"FSQ VIDEO code-class weights [{name}]: shape {tuple(w.shape)}, "
                f"range [{float(w.min()):.3f}, {float(w.max()):.3f}], "
                f"mean {float(w.mean()):.3f} (cap={args.video_code_class_weight}, "
                f"{args.video_code_weight_batches} batches)"
            )

    fastts_code_class_weights: Dict[str, torch.Tensor] = {}
    if args.fastts_fsq:
        fastts_code_class_weights = build_fastts_code_class_weights(
            _core(model), train_loader, device,
            cap=args.fastts_code_class_weight,
            n_batches=args.fastts_code_weight_batches,
        )
        for name, w in fastts_code_class_weights.items():
            logger.info(
                f"FSQ FASTTS code-class weights [{name}]: shape {tuple(w.shape)}, "
                f"range [{float(w.min()):.3f}, {float(w.max()):.3f}], "
                f"mean {float(w.mean()):.3f} (cap={args.fastts_code_class_weight}, "
                f"{args.fastts_code_weight_batches} batches)"
            )

    slow_ts_code_class_weights: Dict[str, torch.Tensor] = {}
    if args.slow_ts_fsq:
        slow_ts_code_class_weights = build_slowts_code_class_weights(
            _core(model), train_loader, device,
            cap=args.slow_ts_code_class_weight,
            n_batches=args.slow_ts_code_weight_batches,
        )
        for name, w in slow_ts_code_class_weights.items():
            logger.info(
                f"FSQ SLOWTS code-class weights [{name}]: shape {tuple(w.shape)}, "
                f"range [{float(w.min()):.3f}, {float(w.max()):.3f}], "
                f"mean {float(w.mean()):.3f} (cap={args.slow_ts_code_class_weight}, "
                f"{args.slow_ts_code_weight_batches} batches)"
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
    # Holds the saved optimizer state (on CPU) ONLY on a --lazy_optimizer_load
    # resume: the eager load is skipped, the first opt.step() allocates the
    # buffers fresh (cold-start's clean order), then this is filled in in-place
    # after that step. None on cold start / eager resume.
    _pending_opt_state = None

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
            args.resume_checkpoint, weights_only=False,
            # --lazy_optimizer_load: load to CPU so the checkpoint NEVER occupies
            # the GPU. model.load_state_dict then copies weights into the (already
            # GPU-resident) model in-place, so the GPU heap entering the loop is
            # ONLY the freshly-built model — byte-identical to cold-start. The
            # device-load blob (even after del+empty_cache) left the fragmentation
            # that OOM'd the mimic. Default device → eager chain unchanged.
            map_location=("cpu" if args.lazy_optimizer_load else device),
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
                elif args.lazy_optimizer_load:
                    # Cleaner cold-start mimic: do NOT load the optimizer state
                    # now. Let the first opt.step() allocate the momentum buffers
                    # fresh — AdamW's lazy alloc lays them down in one clean pass
                    # post-backward (no fragmentation), exactly as cold-start. Stash
                    # the saved state on CPU (off the GPU heap; del resume_ckpt then
                    # frees the GPU copy) and fill it in-place after that step.
                    # param_groups/lr come from args + the scheduler, so no restore.
                    _pending_opt_state = {
                        i: {k: (v.cpu() if torch.is_tensor(v) else v)
                            for k, v in st.items()}
                        for i, st in
                        resume_ckpt["optimizer_state_dict"]["state"].items()
                    }
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
        # RESUME-OOM FIX (2026-06-24): the checkpoint is loaded with
        # map_location=device (~line 1616) → resume_ckpt holds a full GPU copy
        # of model+optimizer state that is dead weight after load_state_dict.
        # Combined with the resume's allocation-order fragmentation, this
        # strands ~352 MiB reserved-but-unallocated that the first training
        # step can't reuse → HIP OOM (the cold start fit at ~99%; the first
        # resume tipped over and the whole afterany chain 4888244..52
        # cascade-crashed). Drop the copy + return the fragmented pool to the
        # driver before the loop. Build-independent — expandable_segments is
        # silently ignored on this ROCm build (the OOM msg still suggested it
        # even with the env set).
        del resume_ckpt
        gc.collect()
        torch.cuda.empty_cache()
        _trim_host_ram()  # return the freed ~21GB checkpoint HOST RAM to the OS
                          # (glibc retains it → resume baseline ~180GB/node > cold
                          # → host-OOM ~3h50m; cold TIMEOUTs clean). See _trim_host_ram.
    elif args.init_checkpoint is not None:
        # Cold start with weights warm-loaded from another checkpoint. Load to
        # CPU (the file carries a ~14 GB optimizer state we never use here);
        # load_state_dict copies CPU→GPU, then we free the whole blob so it does
        # not waste GPU/host memory during training.
        init_ckpt = torch.load(
            args.init_checkpoint, weights_only=False, map_location="cpu"
        )
        allowed_missing = tuple(
            f"{prefix}{name}." for prefix in (
                "diag_tokenizers.", "diag_heads."
            )
            for name in (*args.use_video, *args.use_spectro)
        )
        # The frequency-warp anchor heads and the backbone-skip gate are NEW
        # modules absent from a pre-warp checkpoint — they init fresh (warp is
        # zero-init → identity → the warm model is unchanged at load). Allow them
        # to be missing so a continuous-head base (e.g. p64pe) warm-starts cleanly.
        allowed_missing = allowed_missing + ("spec_warp_heads.", "spec_descriptor_heads.", "backbone_skip_gate", "actuator_film.")
        # The actuator tokenizer conv geometry scales with --prediction_horizon_s
        # (patch_size = act_samples // n_tokens; act_samples = horizon * FAST_FS).
        # A warm-start at a LONGER horizon (t+4/t+8) than the checkpoint's (t+1)
        # makes act_tokenizers.*.conv.weight shape-mismatch (e.g. [512,C,100] ckpt
        # vs [512,C,400] model). Allow those so the shape-mismatch drop below fires
        # and the fresh longer-window actuator tokenizer inits from scratch on the
        # warm backbone. At the SAME horizon shapes match → nothing dropped → loaded
        # normally (no-op for the common case).
        allowed_missing = allowed_missing + ("act_tokenizers.",)
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
        # Architecture swaps under allowed_missing prefixes (e.g. VideoOutputHead
        # → VideoFlowHead) leave STALE keys in the checkpoint that the new model
        # lacks. load_state_dict_explicit rejects unexpected keys, so drop any
        # checkpoint key the model doesn't have whose name is under an
        # allowed_missing prefix. Backbone/other mismatches are NOT covered →
        # still raise (correct).
        model_keys = set(_core(model).state_dict().keys())
        stale = [
            k for k in list(init_ckpt["model_state_dict"])
            if k not in model_keys and any(k.startswith(p) for p in allowed_missing)
        ]
        for k in stale:
            del init_ckpt["model_state_dict"][k]
        if stale:
            logger.info(
                f"Warm-start: dropped {len(stale)} stale keys from an "
                f"architecture swap (fresh init for those): e.g. {stale[:3]}"
            )
        # Same-name keys whose SHAPE differs from the model (e.g. a resized code
        # head — --spec_code_pred_hidden/layers bigger than the checkpoint's)
        # also break load_state_dict (size mismatch, not suppressed by
        # strict=False). Drop them too, but ONLY under allowed_missing prefixes
        # (diag_heads/tokenizers of the used modalities) so the resized module
        # inits fresh (cold head) on a warm backbone; any other shape mismatch
        # still surfaces as an error (correct).
        model_sd = _core(model).state_dict()
        mismatched = [
            k for k in list(init_ckpt["model_state_dict"])
            if k in model_keys
            and init_ckpt["model_state_dict"][k].shape != model_sd[k].shape
            and any(k.startswith(p) for p in allowed_missing)
        ]
        for k in mismatched:
            del init_ckpt["model_state_dict"][k]
        if mismatched:
            logger.info(
                f"Warm-start: dropped {len(mismatched)} shape-mismatched keys "
                f"(resized module → fresh init): e.g. {mismatched[:3]}"
            )
        if getattr(args, "reinit_act_tokenizers", False):
            # GATE3-FIX: force FRESH actuator tokenizers even when shapes match the checkpoint —
            # the actuator input DISTRIBUTION changed (ech_power now log_standardized, rmp
            # standardized, 3 angle channels dropped), so the checkpoint's raw-scale-trained
            # actuator conv/patch_pos are mis-calibrated. Fresh init + unfrozen backbone lets the
            # model re-learn actuator conditioning. (act_tokenizers. is already in allowed_missing.)
            _act_drop = [k for k in list(init_ckpt["model_state_dict"]) if k.startswith("act_tokenizers.")]
            for k in _act_drop:
                del init_ckpt["model_state_dict"][k]
            if _act_drop:
                logger.info(f"Warm-start: --reinit_act_tokenizers dropped {len(_act_drop)} "
                            "act_tokenizer keys → fresh init (actuator input distribution changed).")
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
        del init_ckpt
        gc.collect()
        torch.cuda.empty_cache()
        _trim_host_ram()  # same host-RAM return as the resume path: warm-start also
                          # loads the ~21GB blob to CPU → glibc retention → host-OOM.
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
    # GATE-3-FIX anchor-β anneal schedule + collapse-tripwire accumulators (mean per val window).
    _abeta_holds = [float(x) for x in args.spec_descriptor_anchor_beta_holds.split(",") if x.strip()]
    _abeta_hold_steps = max(1, args.spec_descriptor_anchor_beta_hold_steps)
    if _abeta_holds:
        logger.info(f"Anchor-β anneal ON: holds={_abeta_holds} × {_abeta_hold_steps} steps each "
                    f"(target softmax β held at {args.spec_descriptor_dist_beta}).")
    _trip_fdrift = 0.0; _trip_hfrac = 0.0; _trip_n = 0
    # ── K-step rollout setup (opt-in). Built ONCE and reused; the wrapper
    # holds no state, just references model_core's tokenizers/backbone/heads. ──
    _krollout = None
    _last_logged_K = None
    if args.k_rollout:
        _krollout = TokenSpaceRollout(_core(model), dt_s=args.chunk_duration_s)
        logger.info(
            f"K-rollout ON: curriculum_Ks={_curriculum_Ks} × block_steps="
            f"{args.block_steps}; tf_anneal_steps={args.tf_anneal_steps}; "
            f"grad_checkpoint_every={args.rollout_grad_checkpoint_every}; "
            f"dataset_horizon_s={dataset_horizon_s:.4f}s / model_horizon_s="
            f"{args.prediction_horizon_s:.4f}s."
        )
    if dm.distributed and hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(epoch_counter)
    train_iter = iter(train_loader)
    while step < args.max_steps and (
        args.stop_at_step is None or step < args.stop_at_step
    ):
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch_counter += 1
            if dm.distributed and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch_counter)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Stepwise anchor-β for this step: holds[step // hold_steps], clamped to the last β.
        # Empty holds → None → compute_step_loss falls back to the fixed dist_beta (no change).
        _abeta_now = (_abeta_holds[min(step // _abeta_hold_steps, len(_abeta_holds) - 1)]
                      if _abeta_holds else None)
        opt.zero_grad()
        # Shared loss-fn kwargs (identical for single-step and K-rollout except
        # n_subwindows, which the rollout driver pins to the descriptor reach).
        _csl_kwargs = dict(
            spec_pb_weights=spec_pb_weights,
            spec_struct_lambda=args.spec_struct_lambda,
            spec_mask_lambda=args.spec_mask_lambda,
            spec_mae_lambda=args.spec_mae_lambda,
            spec_mask_loss_type=args.spec_mask_loss,
            spec_code_class_weights=spec_code_class_weights,
            spec_code_focal_gamma=args.spec_code_focal_gamma,
            spec_ordinal_eps=args.spec_ordinal_eps,
            spec_autoencode=args.spec_autoencode,
            loss_norm_ema=args.loss_norm_ema,
            loss_norm_beta=args.loss_norm_beta,
            loss_priority={m: args.loss_priority_spectro for m in (args.use_spectro or [])},
            video_code_class_weights=video_code_class_weights,
            fastts_code_class_weights=fastts_code_class_weights,
            slow_ts_code_class_weights=slow_ts_code_class_weights,
            spec_descriptor_weight=args.spec_descriptor_weight,
            spec_descriptor_loss=args.spec_descriptor_loss,
            spec_descriptor_dist_beta=args.spec_descriptor_dist_beta,
            spec_descriptor_anchor=args.spec_descriptor_anchor,
            # β pinned at the launcher's value (via single-entry anchor-β holds);
            # do NOT anneal inside the rollout. Single-step keeps the schedule.
            spec_descriptor_anchor_beta=_abeta_now,
            spec_descriptor_transition_weight=args.spec_descriptor_transition_weight,
            # STRIKE-3 lever 1: asymmetric drift penalty (default 0.0 → off).
            drift_penalty_weight=args.drift_penalty_weight,
        )
        with amp_ctx_factory():
            if args.k_rollout:
                K = current_K_from_list(step, _curriculum_Ks, args.block_steps)
                p_tf = (max(0.0, 1.0 - step / args.tf_anneal_steps)
                        if args.tf_anneal_steps > 0 else 0.0)
                if K != _last_logged_K:
                    logger.info(f"K-rollout: step {step} → K={K}  p_tf={p_tf:.3f}")
                    _last_logged_K = K
                loss, per_mod = rollout_forward_loss(
                    model, batch, device, K, args.chunk_duration_s, _krollout,
                    compute_step_loss_kwargs=_csl_kwargs,
                    p_tf=p_tf,
                    grad_checkpoint_every=args.rollout_grad_checkpoint_every,
                    feedback_normalize=args.feedback_normalize,
                    # STRIKE-3 lever 2: k0-protected per-k re-weighting.
                    k_ge1_weight=args.k_ge1_weight,
                    k_ge1_weight_start=args.k_ge1_weight_start,
                    k_ge1_weight_anneal_steps=args.k_ge1_weight_anneal_steps,
                    global_step=step,
                )
            else:
                loss, per_mod = compute_step_loss(
                    model, batch, device,
                    n_subwindows=max(1, round(args.prediction_horizon_s / args.chunk_duration_s)),
                    **_csl_kwargs,
                )
        # NaN/Inf guard: one bad batch or a bf16 rollout overflow must not poison
        # a multi-day run. On non-finite loss, skip backward + opt.step (Adam would
        # still apply momentum on zero grads, so opt.step must be skipped, not just
        # zeroed) while keeping the step/scheduler/val flow intact. The culprit
        # rollout step + per-term breakdown is logged inside rollout_forward_loss.
        # Finite loss (the g3fix single-step norm) → byte-identical to before.
        _loss_finite = bool(torch.isfinite(loss).item())
        if _loss_finite:
            loss.backward()
        else:
            _bad = {k: v for k, v in per_mod.items()
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v))}
            logger.warning(
                f"NON-FINITE loss at step {step}; SKIPPING update. "
                f"non-finite terms: {_bad or 'loss-level only'}"
            )
            opt.zero_grad(set_to_none=True)
        # GATE3-FIX monitor: actuator-tokenizer gradient norm = proxy for whether the (unfrozen)
        # backbone ATTENDS to actuator tokens. If it ignores them, ∂loss/∂(act tokens)≈0 → the act
        # tokenizer params get ~zero gradient; the norm should GROW as the backbone unlearns its
        # actuator-blindness. Discriminates a weak ACT_CF result: FLAT here = never re-attended (needs
        # longer fine-tune); GREW = attended-but-no-causal-signal (physics/data limit). Gated to log
        # steps (post-increment step % log_every) to avoid a per-step GPU→CPU sync; pre-clip magnitude.
        _act_gradnorm = None
        if (step + 1) % args.log_every == 0:
            _act_gradnorm = sum(
                float(p.grad.detach().pow(2).sum())
                for n, p in _core(model).named_parameters()
                if n.startswith("act_tokenizers.") and p.grad is not None
            ) ** 0.5
        if _loss_finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
        if _loss_finite and _pending_opt_state is not None:
            # The first opt.step() just allocated the momentum buffers fresh (the
            # clean cold-start layout). Fill the saved momentum in-place — CPU→GPU
            # copy_ into the existing buffers, NO new GPU allocation — matched by
            # param order (the same mapping opt.load_state_dict uses), shape-guarded
            # against mis-map. That one step ran with zero momentum (negligible over
            # the run); step 2+ uses the restored state.
            _idx = 0
            for _grp in opt.param_groups:
                for _p in _grp["params"]:
                    _sv = _pending_opt_state.get(_idx)
                    _cur = opt.state.get(_p)
                    if _sv is not None and _cur is not None:
                        for _k, _v in _sv.items():
                            if (_k in _cur and torch.is_tensor(_cur[_k])
                                    and torch.is_tensor(_v)
                                    and _cur[_k].shape == _v.shape):
                                _cur[_k].copy_(_v)
                            elif _k in _cur and not torch.is_tensor(_v):
                                _cur[_k] = _v
                    _idx += 1
            _pending_opt_state = None
            _trim_host_ram()  # return the freed 10.7GB optimizer CPU-copy to the OS
        scheduler.step()
        if _loss_finite:
            running_total += loss.item()
            running_count += 1
        step += 1
        # Accumulate collapse tripwires (descriptor modality) for the per-val-window mean + best.pt gate.
        for _tk in per_mod:
            if _tk.endswith("_desc_fdrift") and not math.isnan(per_mod[_tk]):
                _trip_fdrift += per_mod[_tk]; _trip_n += 1
            elif _tk.endswith("_desc_hfrac") and not math.isnan(per_mod[_tk]):
                _trip_hfrac += per_mod[_tk]

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
            # running_count can be 0 if every step in this window was NaN-skipped
            # (the guard) — avoid ZeroDivision; report nan so the skip is visible.
            avg = (running_total / running_count) if running_count > 0 else float("nan")
            lr_now = opt.param_groups[0]["lr"]
            per_mod_str = ", ".join(
                f"{n}={per_mod[n]:.4f}" for n in diagnostic_names
            )
            # also surface generative aux losses (flow / struct) so their
            # magnitude + SIGN are visible (a negative total loss points to one
            # of these going negative). Empty for non-generative heads -> no change.
            aux_str = ", ".join(
                f"{k}={per_mod[k]:.4f}" for k in per_mod
                if k.endswith("_flow") or k.endswith("_struct")
                or k.endswith("_mask") or k.endswith("_maskdice")
                or k.endswith("_codeacc") or k.endswith("_ce")
                or "_desc" in k
                or k.startswith("rollout_")   # STRIKE-3 lever-2 applied w_k vector
            )
            if aux_str:
                per_mod_str += "  ||  " + aux_str
            _agn = f"  act_tok_gradnorm={_act_gradnorm:.4e}" if _act_gradnorm is not None else ""
            _abn = f"  anchor_beta={_abeta_now:.2f}" if _abeta_now is not None else ""
            logger.info(
                f"step {step}/{args.max_steps}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}{_abn}{_agn}  | {per_mod_str}"
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
            # OPT-IN GPU peak-memory probe (default OFF → byte-identical). Set
            # KANNEAL_MEM_PROBE=1 for the Lever-#1 memory-fit acceptance gate:
            # reports per-rank HIP max_memory_allocated so the 0.7s/batch16 peak
            # can be compared against the 48.84 GiB that OOM'd at the 4.2s horizon.
            import os as _os_mem  # os is not module-level in this file
            if _os_mem.environ.get("KANNEAL_MEM_PROBE") == "1" and torch.cuda.is_available():
                _peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                _resv = torch.cuda.max_memory_reserved() / (1024 ** 3)
                logger.warning(
                    f"[mem-probe] step {step} rank{dm.rank}: "
                    f"max_memory_allocated={_peak:.2f} GiB  "
                    f"max_memory_reserved={_resv:.2f} GiB  (64 GiB/GCD)"
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
            # COLLAPSE TRIPWIRES (mean over this val window) — the anchor-anneal guard. false-drift
            # (spurious dynamics on static windows) moves first when the anchor weakens too far; hfrac→1
            # is flat mean-collapse. Log-and-continue: best.pt promotion is BLOCKED while collapsed, but
            # the job keeps running so the whole β trajectory is milestoned for the post-run β pick.
            _fd = _trip_fdrift / max(_trip_n, 1); _hf = _trip_hfrac / max(_trip_n, 1)
            _collapsed = bool(_trip_n > 0 and (_fd > args.desc_false_death_abort or _hf > 0.98))
            if _trip_n > 0:
                logger.info(
                    f"  [tripwire] anchor_beta={(_abeta_now if _abeta_now is not None else args.spec_descriptor_dist_beta):.2f} "
                    f"false_drift={_fd:.4f} (abort>{args.desc_false_death_abort}) entropy_frac={_hf:.4f} "
                    f"(collapse>0.98) → {'COLLAPSED (best.pt gated)' if _collapsed else 'ok'}"
                )
            _trip_fdrift = 0.0; _trip_hfrac = 0.0; _trip_n = 0
            # Decide best-update first so both `latest` and `best` share the
            # same final best_val_loss / best_step values — otherwise resume
            # from `latest` would see a stale best. Gate on the collapse tripwire.
            is_new_best = (sel_loss < best_val_loss) and not _collapsed
            if sel_loss < best_val_loss and _collapsed:
                logger.info("  ⚠ would-be best BLOCKED by collapse tripwire — best.pt not updated.")
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
                # MILESTONE: at each anchor-β hold boundary, save the EQUILIBRATED head at the β just
                # completed (holds[(step//hold_steps)-1]) so ACT_CF can be run per-β and the landing β
                # picked from the curve. Requires hold_steps | val_every so a val lands on the boundary.
                if _abeta_holds and step % _abeta_hold_steps == 0:
                    _mi = min(max(step // _abeta_hold_steps - 1, 0), len(_abeta_holds) - 1)
                    _bsv = _abeta_holds[_mi]
                    _mpath = args.checkpoint_dir / f"e2e_stage1_beta{_bsv:.1f}_step{step}.pt"
                    torch.save(ckpt_state, _mpath)
                    logger.info(f"  ✓ milestone anchor_β={_bsv:.1f} (equilibrated) saved {_mpath.name}")
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
