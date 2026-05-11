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
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

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
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout
from tokamak_foundation_model.utils.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler

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
    cfg_by_name = {c.name: c for c in rollout.model.diagnostics}
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

    result = rollout(diag_initial, act_per_step)
    # Video heads emit (B, T, C, H, W); permute per step to (B, C, T, H, W)
    # so loss / metric paths see a single shape contract.
    for k in range(k_steps):
        for name in video_diag_names:
            result.predictions[k][name] = (
                result.predictions[k][name].permute(0, 2, 1, 3, 4)
            )

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
            pred = result.predictions[k][name]
            target = target_per_step[k][name]
            mask = mask_per_step[k][name]
            if name in video_diag_names or name in spectro_diag_names:
                # Video and spectrogram: MAE only.
                # - Video: cosine in ~900k pixels is meaningless
                #   (project_phase_c_video_design memory).
                # - Spectrogram: displacement loss deferred per Open
                #   Decision #3 in the spectrogram plan; revisit after
                #   reconstruction quality (Step 6) is validated.
                # dir_cos and mag_ratio reported as NaN / 0 for the
                # metric grid in both cases.
                mae = masked_mae(pred, target, mask)
                total_loss = total_loss + mae_weight * mae
                mae_row.append(mae.detach())
                zero = torch.zeros((), device=pred.device)
                dcos_row.append(zero)
                mr_row.append(zero)
                nv_row.append(zero)
                continue
            # Context: teacher-forced — ground-truth state at step k-1
            # (= window index k in the pool). At k=0, ctx is the rollout
            # input (diag_initial).
            ctx = diag_initial[name] if k == 0 else target_per_step[k - 1][name]

            mae = masked_mae(pred, target, mask)
            cos_loss, mag_loss, dcos_t, mr_t, nv_t = displacement_losses(
                pred, target, ctx, mask, min_disp_norm
            )
            step_loss = (
                mae_weight * mae + cos_weight * cos_loss + mag_weight * mag_loss
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
    keys = ("model_mae", "copy_mae", "dir_cos", "mag_ratio")
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
                if name in video_diag_names or name in spectro_diag_names:
                    mae = masked_mae(pred, target, mask).item()
                    copy_mae = masked_mae(
                        diag_initial[name], target, mask
                    ).item()
                    sums[k][name]["model_mae"] += mae
                    sums[k][name]["copy_mae"] += copy_mae
                    counts[k][name]["mae"] += 1
                    # No displacement metrics for video / spectrogram.
                    continue
                ctx = (
                    diag_initial[name] if k == 0 else target_per_step[k - 1][name]
                )
                mae = masked_mae(pred, target, mask).item()
                copy_mae = masked_mae(diag_initial[name], target, mask).item()
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

    rollout.model.train()
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in range(K_max):
        out[k] = {}
        for name in diagnostic_names:
            mae_n = max(counts[k][name]["mae"], 1)
            disp_n = max(counts[k][name]["disp"], 1)
            out[k][name] = {
                "model_mae": sums[k][name]["model_mae"] / mae_n,
                "copy_mae": sums[k][name]["copy_mae"] / mae_n,
                "dir_cos": sums[k][name]["dir_cos"] / disp_n
                if counts[k][name]["disp"] else float("nan"),
                "mag_ratio": sums[k][name]["mag_ratio"] / disp_n
                if counts[k][name]["disp"] else float("nan"),
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
    parser.add_argument("--K_max", type=int, default=10)
    parser.add_argument("--curriculum_steps", type=int, default=25_000)

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
            cache_path=args.checkpoint_dir / "video_present_train.pt",
        )
        val_files = filter_video_present_files(
            val_files, args.use_video,
            cache_path=args.checkpoint_dir / "video_present_val.pt",
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
    )
    diagnostic_names = [c.name for c in diagnostics]
    actuator_names = [c.name for c in actuators]
    video_diag_names = [c.name for c in diagnostics if c.kind == "video"]
    video_n_frames = {c.name: c.window_samples for c in diagnostics if c.kind == "video"}
    spectro_diag_names = [c.name for c in diagnostics if c.kind == "spectrogram"]
    logger.info(f"Diagnostics ({len(diagnostics)}): " + ", ".join(diagnostic_names))
    logger.info(f"Actuators ({len(actuators)}): " + ", ".join(actuator_names))

    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)

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
        load_state_dict_explicit(
            model, ckpt["model_state_dict"], allowed_missing_prefixes=allowed
        )
        logger.info(
            f"Initialised from {args.init_checkpoint.name} "
            f"(val_loss={ckpt.get('val_loss', 'n/a')} "
            f"step={ckpt.get('step', 'n/a')})"
        )
    else:
        logger.warning(
            "No --init_checkpoint; random weights. Smoke-test only — real "
            "Stage 2b must warm-start from Stage 1 best, not Stage 2 best."
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
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage2_delta_train.pt",
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage2_delta_val.pt",
        **shared,
    )
    logger.info(
        f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}  "
        f"prediction_horizon_s={prediction_horizon_s:.3f} (K_max={args.K_max})"
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        # TwoLevelSampler: shuffle file order per epoch, sequential
        # within each file. Keeps the per-worker LRU file-handle
        # cache (max_open_files=100) nearly always hitting.
        # RandomSampler across 7878 files gave ~1% hit rate and
        # spent ~10% of worker time on HDF5 file opens (observed
        # via py-spy on Stage 1 job 2719669).
        sampler=(
            DistributedSampler(
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
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
        # pin_memory=False for val: each iter() call re-creates the main
        # process's pin_memory thread + internal queues, and those pinned
        # allocations ratchet host RSS upward across validations (observed
        # +127 GB on val 1, +27 GB on val 2 with persistent_workers=True,
        # OOM on val 2 at batch=256). Val is 1–20 batches per call so the
        # synchronous H2D cost is negligible.
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
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
        load_state_dict_explicit(
            model, resume_ckpt["model_state_dict"], allowed_missing_prefixes=()
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
            is_new_best = val_loss < best_val_loss
            if is_new_best:
                best_val_loss = val_loss
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