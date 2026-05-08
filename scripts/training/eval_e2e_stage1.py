"""Evaluation script for Stage 1 (Phase A or Phase C) E2E checkpoints.

Loads a frozen Stage 1 checkpoint and runs single-step (K=1) prediction over
the **full** val set. Produces:

  * per-modality MAE / copy-MAE / direction_cos / magnitude_ratio
  * per-channel MAE breakdown (CSV)
  * per-modality pred-vs-target plots (PNG)
  * ``metrics.json`` (machine-readable)
  * ``summary.md`` (human-readable PASS/FAIL on milestone A2 —
    single-step MAE below copy baseline for all modalities, per
    ``ResearchPlan.MD`` §6.1)

Run::

    pixi run python scripts/training/eval_e2e_stage1.py \
        --checkpoint runs/e2e_stage1/e2e_stage1_best.pt \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path scripts/slurm/preprocessing_stats.pt \
        --output_dir runs/e2e_stage1/eval_best

Add ``--use_video tangtv`` for Phase C Stage 1 checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
)
from tokamak_foundation_model.e2e.lora import apply_lora_to_backbone
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

logger = logging.getLogger("eval_stage1")


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


@torch.no_grad()
def forward_one_batch(
    model: E2EFoundationModel,
    batch: Dict,
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],  # predictions (post permute for video)
    Dict[str, torch.Tensor],  # diag_inputs (cleaned, video standardised)
    Dict[str, torch.Tensor],  # targets (raw or standardised for video)
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


# ── File split (mirror of train_e2e_stage1.resolve_shot_files) ───────


def resolve_val_files(
    data_dir: Path, val_fraction: float, seed: int
) -> List[Path]:
    """Reproduce the trainer's deterministic train/val split and return
    just the val files (when no shot YAML is provided)."""
    rng = random.Random(seed)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    rng.shuffle(all_files)
    n_val = max(1, int(val_fraction * len(all_files)))
    return all_files[:n_val]


# ── Metric aggregators ───────────────────────────────────────────────


class GlobalAccumulator:
    """Per-modality accumulator for global K=1 MAE / cos / ratio."""

    def __init__(self, names: List[str]) -> None:
        self.names = names
        self.model_mae_sum = {n: 0.0 for n in names}
        self.copy_mae_sum = {n: 0.0 for n in names}
        self.pred_delta_sum = {n: 0.0 for n in names}
        self.tgt_delta_sum = {n: 0.0 for n in names}
        self.dir_cos_sum = {n: 0.0 for n in names}
        self.mag_ratio_sum = {n: 0.0 for n in names}
        self.n_valid_dir = {n: 0 for n in names}
        self.n_batches = 0

    def update_modality(
        self,
        name: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: torch.Tensor,
        mask: Optional[torch.Tensor],
        copy_pred: torch.Tensor,
        min_disp_norm: float = 0.01,
    ) -> None:
        cleaned_pred, mask_p = _clean_and_mask(pred, None)
        cleaned_tgt, mask_t = _clean_and_mask(target, mask)
        cleaned_ctx, mask_c = _clean_and_mask(ctx, None)
        cleaned_copy, mask_cp = _clean_and_mask(copy_pred, mask)
        joint = mask_p * mask_t * mask_c
        denom = joint.sum().clamp_min(1.0)

        model_mae = (
            (cleaned_pred - cleaned_tgt).abs() * joint
        ).sum() / denom
        copy_joint = mask_cp * mask_t
        copy_denom = copy_joint.sum().clamp_min(1.0)
        copy_mae = (
            (cleaned_copy - cleaned_tgt).abs() * copy_joint
        ).sum() / copy_denom
        pred_delta = ((cleaned_pred - cleaned_ctx).abs() * joint).sum() / denom
        tgt_delta = ((cleaned_tgt - cleaned_ctx).abs() * joint).sum() / denom

        # direction_cos / magnitude_ratio are per-sample; mask zeros out
        # contributions from missing positions so the dot-product is over
        # valid entries only.
        disp_pred = (cleaned_pred - cleaned_ctx) * joint
        disp_tgt = (cleaned_tgt - cleaned_ctx) * joint
        batch = pred.shape[0]
        dp = disp_pred.reshape(batch, -1)
        dt = disp_tgt.reshape(batch, -1)
        tgt_norm = dt.norm(dim=1)
        pred_norm = dp.norm(dim=1)
        valid = tgt_norm > min_disp_norm
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            dir_cos = F.cosine_similarity(dp[valid], dt[valid], dim=1).mean()
            mag_ratio = (
                pred_norm[valid] / tgt_norm[valid].clamp_min(1e-6)
            ).mean()
            self.dir_cos_sum[name] += float(dir_cos.item()) * n_valid
            self.mag_ratio_sum[name] += float(mag_ratio.item()) * n_valid
            self.n_valid_dir[name] += n_valid

        self.model_mae_sum[name] += model_mae.item()
        self.copy_mae_sum[name] += copy_mae.item()
        self.pred_delta_sum[name] += pred_delta.item()
        self.tgt_delta_sum[name] += tgt_delta.item()

    def step(self) -> None:
        self.n_batches += 1

    def finalize(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        denom = max(self.n_batches, 1)
        for n in self.names:
            model_mae = self.model_mae_sum[n] / denom
            copy_mae = self.copy_mae_sum[n] / denom
            pred_d = self.pred_delta_sum[n] / denom
            tgt_d = self.tgt_delta_sum[n] / denom
            ratio = pred_d / tgt_d if tgt_d > 1e-8 else float("nan")
            n_v = self.n_valid_dir[n]
            dir_cos = self.dir_cos_sum[n] / n_v if n_v > 0 else float("nan")
            mag_ratio = self.mag_ratio_sum[n] / n_v if n_v > 0 else float("nan")
            out[n] = {
                "model_mae": model_mae,
                "copy_mae": copy_mae,
                "delta": copy_mae - model_mae,
                "pred_delta": pred_d,
                "tgt_delta": tgt_d,
                "delta_ratio": ratio,
                "direction_cos": dir_cos,
                "magnitude_ratio": mag_ratio,
                "n_valid_dir_samples": n_v,
            }
        return out


class PerChannelAccumulator:
    """Per-channel MAE for both model and copy baseline."""

    def __init__(self, names: List[str]) -> None:
        self.names = names
        self.model_sum: Dict[str, torch.Tensor] = {}
        self.copy_sum: Dict[str, torch.Tensor] = {}
        self.mask_sum: Dict[str, torch.Tensor] = {}
        self._initialised = {n: False for n in names}

    def _init_for(self, name: str, n_channels: int, device: torch.device) -> None:
        self.model_sum[name] = torch.zeros(n_channels, device=device)
        self.copy_sum[name] = torch.zeros(n_channels, device=device)
        self.mask_sum[name] = torch.zeros(n_channels, device=device)
        self._initialised[name] = True

    def update_modality(
        self,
        name: str,
        pred: torch.Tensor,
        copy_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        n_channels = pred.shape[1]
        if not self._initialised[name]:
            self._init_for(name, n_channels, pred.device)

        cleaned_pred, mask_p = _clean_and_mask(pred, None)
        cleaned_copy, _ = _clean_and_mask(copy_pred, None)
        cleaned_tgt, mask_t = _clean_and_mask(target, mask)
        joint = mask_p * mask_t

        # Reduce across all dims except channel.
        reduce_dims = [d for d in range(pred.ndim) if d != 1]
        model_err = (cleaned_pred - cleaned_tgt).abs() * joint
        copy_err = (cleaned_copy - cleaned_tgt).abs() * joint
        self.model_sum[name] += model_err.sum(dim=reduce_dims)
        self.copy_sum[name] += copy_err.sum(dim=reduce_dims)
        self.mask_sum[name] += joint.sum(dim=reduce_dims)

    def finalize(self) -> Dict[str, List[Dict[str, float]]]:
        out: Dict[str, List[Dict[str, float]]] = {}
        for n in self.names:
            if not self._initialised[n]:
                out[n] = []
                continue
            denom = self.mask_sum[n].clamp_min(1.0)
            mae = (self.model_sum[n] / denom).cpu().tolist()
            copy_mae = (self.copy_sum[n] / denom).cpu().tolist()
            valid = (self.mask_sum[n] > 0).cpu().tolist()
            rows = []
            for c, (m, cb, v) in enumerate(zip(mae, copy_mae, valid)):
                rows.append({
                    "channel": c,
                    "model_mae": m if v else float("nan"),
                    "copy_mae": cb if v else float("nan"),
                    "delta": (cb - m) if v else float("nan"),
                    "n_valid": int(self.mask_sum[n][c].item()),
                })
            out[n] = rows
        return out


# ── Sample-level caches for richer plots ─────────────────────────────


class HexbinAccumulator:
    """Reservoir-sampled (pred, target) pairs per modality for Panel C.

    Pools every (sample × channel × timestep) value where the mask is 1, up to
    ``cap`` points per modality. After ``cap``, swaps in new points with
    decreasing probability so the final sample is uniform over the stream.
    """

    def __init__(self, names: List[str], cap: int = 50_000) -> None:
        self.cap = cap
        self.preds: Dict[str, List[float]] = {n: [] for n in names}
        self.tgts: Dict[str, List[float]] = {n: [] for n in names}
        self.seen: Dict[str, int] = {n: 0 for n in names}

    def update(
        self,
        name: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        cleaned_pred, mp = _clean_and_mask(pred, None)
        cleaned_tgt, mt = _clean_and_mask(target, mask)
        joint = (mp * mt).bool()
        if joint.sum() == 0:
            return
        p_flat = cleaned_pred[joint].detach().cpu().numpy().reshape(-1)
        t_flat = cleaned_tgt[joint].detach().cpu().numpy().reshape(-1)
        n_new = p_flat.shape[0]

        # Reservoir-sample to keep memory bounded.
        cur_p = self.preds[name]
        cur_t = self.tgts[name]
        seen = self.seen[name]
        cap = self.cap
        if len(cur_p) + n_new <= cap:
            cur_p.extend(p_flat.tolist())
            cur_t.extend(t_flat.tolist())
        else:
            for i in range(n_new):
                if len(cur_p) < cap:
                    cur_p.append(float(p_flat[i]))
                    cur_t.append(float(t_flat[i]))
                else:
                    j = random.randint(0, seen + i)
                    if j < cap:
                        cur_p[j] = float(p_flat[i])
                        cur_t[j] = float(t_flat[i])
        self.seen[name] = seen + n_new

    def get(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.preds[name]), np.asarray(self.tgts[name])


class PercentileSampleCache:
    """Cache the first ``M`` batches' tensors (CPU) so we can pull
    best / median / worst-MAE samples for Panel D after the eval loop.

    Stores per-modality (pred, target, ctx) and per-sample MAE so the
    final plotter can sort samples by MAE and plot the percentiles."""

    def __init__(self, names: List[str], n_batches: int = 8) -> None:
        self.names = names
        self.n_batches = n_batches
        self.preds: Dict[str, List[torch.Tensor]] = {n: [] for n in names}
        self.tgts: Dict[str, List[torch.Tensor]] = {n: [] for n in names}
        self.ctxs: Dict[str, List[torch.Tensor]] = {n: [] for n in names}
        self.maes: Dict[str, List[torch.Tensor]] = {n: [] for n in names}

    def maybe_update(
        self,
        batch_idx: int,
        name: str,
        pred: torch.Tensor,
        target: torch.Tensor,
        ctx: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        if batch_idx >= self.n_batches:
            return
        cleaned_pred, mp = _clean_and_mask(pred, None)
        cleaned_tgt, mt = _clean_and_mask(target, mask)
        joint = mp * mt
        denom = joint.flatten(1).sum(dim=1).clamp_min(1.0)
        per_sample_mae = (
            ((cleaned_pred - cleaned_tgt).abs() * joint)
            .flatten(1)
            .sum(dim=1)
        ) / denom
        self.preds[name].append(cleaned_pred.detach().cpu())
        self.tgts[name].append(cleaned_tgt.detach().cpu())
        self.ctxs[name].append(ctx.detach().cpu())
        self.maes[name].append(per_sample_mae.detach().cpu())

    def gather(self, name: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.preds[name]:
            return None
        preds = torch.cat(self.preds[name], dim=0)
        tgts = torch.cat(self.tgts[name], dim=0)
        ctxs = torch.cat(self.ctxs[name], dim=0)
        maes = torch.cat(self.maes[name], dim=0)
        return preds, tgts, ctxs, maes


# ── Demo-shot trajectory (Panel A) ────────────────────────────────────


@torch.no_grad()
def collect_demo_shot_trajectory(
    model: E2EFoundationModel,
    file_path: Path,
    chunk_duration_s: float,
    warmup_s: float,
    stats: dict,
    diag_names: List[str],
    act_names: List[str],
    device: torch.device,
    max_chunks: int = 200,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Run the model on every non-overlapping 50 ms window of a single shot
    and stitch the predictions / targets per modality.

    Returns a dict ``{modality_name: {'pred': (C, T_total), 'target': (C, T_total),
    'ctx': (C, T_first), 't_s_pred': (T_total,)}}`` or ``None`` if the file
    has too few chunks.
    """
    try:
        ds = TokamakMultiFileDataset(
            [file_path],
            chunk_duration_s=chunk_duration_s,
            prediction_mode=True,
            prediction_horizon_s=chunk_duration_s,
            step_size_s=chunk_duration_s,           # non-overlapping
            warmup_s=warmup_s,
            preprocessing_stats=stats,
            input_signals=diag_names,
            target_signals=diag_names + act_names,
            lengths_cache_path=None,
        )
    except Exception as exc:
        logger.warning(f"Demo-shot dataset for {file_path.name} failed: {exc}")
        return None
    if len(ds) < 4:
        return None
    n_chunks = min(len(ds), max_chunks)
    loader = DataLoader(
        ds, batch_size=32, shuffle=False, collate_fn=collate_fn,
        num_workers=0, drop_last=False, pin_memory=False,
    )

    pred_chunks: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    tgt_chunks: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    ctx_first: Dict[str, Optional[torch.Tensor]] = {n: None for n in diag_names}
    seen = 0

    for batch in loader:
        if seen >= n_chunks:
            break
        # Forward (mirrors forward_one_batch but only for TS — assumes no video
        # in demo-shot caller). If video diagnostics are present, they'll be
        # tokenised and used as conditioning input but plot path skips them.
        diag_inputs: Dict[str, torch.Tensor] = {}
        for cfg in model.diagnostics:
            raw = batch["inputs"][cfg.name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            if cfg.kind == "video":
                cleaned, _, _ = _video_standardize_per_bc(cleaned)
            diag_inputs[cfg.name] = cleaned
            if cfg.kind == "video":
                vk = f"{cfg.name}_valid"
                if vk in batch["inputs"]:
                    diag_inputs[vk] = batch["inputs"][vk].to(device)
        act_inputs: Dict[str, torch.Tensor] = {}
        for cfg in model.actuators:
            raw = batch["targets"][cfg.name].to(device).float()
            act_inputs[cfg.name], _ = _clean_and_mask(raw, None)
        b = next(iter(diag_inputs.values())).shape[0]
        step_idx = torch.zeros(b, dtype=torch.long, device=device)
        time_off = torch.zeros(b, device=device)
        preds = model(diag_inputs, act_inputs, step_idx, time_off)
        for cfg in model.diagnostics:
            if cfg.kind == "video":
                continue
            pred = preds[cfg.name]
            tgt = batch["targets"][cfg.name].to(device).float()
            tgt, _ = _clean_and_mask(tgt, None)
            if ctx_first[cfg.name] is None:
                ctx_first[cfg.name] = diag_inputs[cfg.name][0].detach().cpu()
            # Take sample 0 from each chunk → effectively iterate the shot.
            pred_chunks[cfg.name].append(pred[0].detach().cpu())
            tgt_chunks[cfg.name].append(tgt[0].detach().cpu())
        seen += b

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for cfg in model.diagnostics:
        if cfg.kind == "video":
            continue
        if ctx_first[cfg.name] is None or not pred_chunks[cfg.name]:
            continue
        pred_full = torch.cat(pred_chunks[cfg.name], dim=-1).numpy()
        tgt_full = torch.cat(tgt_chunks[cfg.name], dim=-1).numpy()
        ctx_full = ctx_first[cfg.name].numpy()
        T_per_chunk = tgt_chunks[cfg.name][0].shape[-1]
        n_chunks_actual = len(pred_chunks[cfg.name])
        # Time axis in seconds: input is at t ∈ [0, chunk_duration_s);
        # pred chunk k spans t ∈ [(k+1)*chunk, (k+2)*chunk).
        t_s_pred = np.arange(n_chunks_actual * T_per_chunk) / (
            T_per_chunk / chunk_duration_s
        ) + chunk_duration_s
        t_s_ctx = np.arange(T_per_chunk) / (T_per_chunk / chunk_duration_s)
        out[cfg.name] = {
            "pred": pred_full,
            "target": tgt_full,
            "ctx": ctx_full,
            "t_s_pred": t_s_pred,
            "t_s_ctx": t_s_ctx,
        }
    return out


# ── Plotting ─────────────────────────────────────────────────────────


def _pick_plot_channels(
    target_np: np.ndarray, n_pick: int, rng: random.Random
) -> List[int]:
    """Pick channels that have non-trivial signal (avoid all-zero / NaN)."""
    n_channels = target_np.shape[1]
    candidates: List[int] = []
    for c in range(n_channels):
        col = target_np[:, c]
        col_finite = col[np.isfinite(col)]
        if col_finite.size == 0:
            continue
        if np.allclose(col_finite, 0.0):
            continue
        candidates.append(c)
    if not candidates:
        candidates = list(range(min(n_channels, 4)))
    rng.shuffle(candidates)
    return candidates[: min(n_pick, len(candidates))]


def _best_improvement_channel(
    per_channel_rows: List[Dict[str, float]]
) -> Optional[int]:
    """Return the channel index with the largest copy − model improvement
    (positive Δ means model beats copy). None if no valid channels."""
    best_c, best_delta = None, -float("inf")
    for r in per_channel_rows:
        d = r.get("delta", float("nan"))
        if np.isfinite(d) and d > best_delta:
            best_delta = d
            best_c = int(r["channel"])
    return best_c


def plot_ts_4panel(
    name: str,
    cfg: DiagnosticConfig,
    per_channel_rows: List[Dict[str, float]],
    hexbin_xy: Tuple[np.ndarray, np.ndarray],
    cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    demo_shot: Optional[Dict[str, np.ndarray]],
    chunk_duration_s: float,
    out_path: Path,
    rng: random.Random,
) -> None:
    """Four-panel evaluation figure for a single TS modality.

    A (top-left): full-shot stitched trajectory of one channel, pred vs target
                  in standardised space, with the model's input window
                  emphasised.
    B (top-right): per-channel MAE bar chart (model + copy), sorted by
                   improvement.
    C (bottom-left): pred-vs-target hexbin density across all val samples
                     (pooled over channels and timesteps), with identity line.
    D (bottom-right): best / median / worst MAE samples, one channel each,
                      stacked with vertical offsets.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    # ── Panel A: demo-shot trajectory ────────────────────────────────
    if demo_shot is not None:
        plot_ch = _best_improvement_channel(per_channel_rows)
        if plot_ch is None:
            plot_ch = 0
        plot_ch = min(plot_ch, demo_shot["pred"].shape[0] - 1)
        t_ctx = demo_shot["t_s_ctx"]
        t_pred = demo_shot["t_s_pred"]
        ax_A.plot(
            t_ctx, demo_shot["ctx"][plot_ch], color="0.5",
            lw=1.0, label="input window",
        )
        ax_A.plot(
            t_pred, demo_shot["target"][plot_ch], color="C0",
            lw=1.0, label="ground truth",
        )
        ax_A.plot(
            t_pred, demo_shot["pred"][plot_ch], color="C3",
            lw=1.0, linestyle="--", alpha=0.85, label="model pred",
        )
        ax_A.axvspan(t_ctx[0], t_ctx[-1], color="0.5", alpha=0.07)
        ax_A.set_xlabel("time (s)", fontsize=9)
        ax_A.set_ylabel("standardised signal", fontsize=9)
        ax_A.set_title(
            f"A) demo shot — channel {plot_ch} (best-improvement)",
            fontsize=10,
        )
        ax_A.legend(fontsize=8, loc="best")
        ax_A.tick_params(labelsize=8)
    else:
        ax_A.text(
            0.5, 0.5, "demo-shot trajectory unavailable",
            transform=ax_A.transAxes, ha="center", va="center", fontsize=10,
        )
        ax_A.set_title("A) demo shot — unavailable", fontsize=10)

    # ── Panel B: per-channel MAE bars ────────────────────────────────
    if per_channel_rows:
        # Sort by Δ = copy_mae − model_mae so the most-improved channels are
        # leftmost. Channels with no valid samples (NaN) go to the right.
        sorted_rows = sorted(
            per_channel_rows,
            key=lambda r: (
                -r["delta"] if np.isfinite(r.get("delta", float("nan")))
                else float("inf")
            ),
        )
        labels = [str(r["channel"]) for r in sorted_rows]
        model_v = [r["model_mae"] if np.isfinite(r["model_mae"]) else 0.0
                   for r in sorted_rows]
        copy_v = [r["copy_mae"] if np.isfinite(r["copy_mae"]) else 0.0
                  for r in sorted_rows]
        x = np.arange(len(labels))
        w = 0.4
        ax_B.bar(x - w / 2, copy_v, width=w, color="C7", label="copy")
        ax_B.bar(x + w / 2, model_v, width=w, color="C3", label="model")
        ax_B.set_xticks(x)
        ax_B.set_xticklabels(labels, fontsize=7, rotation=90)
        ax_B.set_xlabel("channel (sorted by Δ desc)", fontsize=9)
        ax_B.set_ylabel("MAE (standardised)", fontsize=9)
        ax_B.set_title("B) per-channel MAE — model vs copy", fontsize=10)
        ax_B.legend(fontsize=8)
        ax_B.tick_params(axis="y", labelsize=8)
    else:
        ax_B.set_title("B) per-channel MAE — no data", fontsize=10)

    # ── Panel C: pred-vs-target hexbin ───────────────────────────────
    p_arr, t_arr = hexbin_xy
    if p_arr.size > 0:
        finite = np.isfinite(p_arr) & np.isfinite(t_arr)
        p_arr = p_arr[finite]
        t_arr = t_arr[finite]
    if p_arr.size > 0:
        lim_lo = float(min(p_arr.min(), t_arr.min()))
        lim_hi = float(max(p_arr.max(), t_arr.max()))
        pad = (lim_hi - lim_lo) * 0.05 + 1e-6
        lim = (lim_lo - pad, lim_hi + pad)
        hb = ax_C.hexbin(
            t_arr, p_arr, gridsize=60, cmap="viridis",
            mincnt=1, bins="log",
        )
        ax_C.plot(lim, lim, color="white", lw=1.0, linestyle="--", alpha=0.7,
                  label="identity")
        # Slope-1 reference + best-fit slope to visualise mag_ratio < 1.
        slope, intercept = np.polyfit(t_arr, p_arr, 1)
        xs = np.array(lim)
        ax_C.plot(
            xs, slope * xs + intercept, color="red", lw=1.0,
            label=f"fit: slope={slope:.2f}",
        )
        ax_C.set_xlim(lim)
        ax_C.set_ylim(lim)
        ax_C.set_xlabel("ground truth (standardised)", fontsize=9)
        ax_C.set_ylabel("model prediction", fontsize=9)
        ax_C.set_title(
            f"C) pred vs target hexbin (n={p_arr.size:,})", fontsize=10,
        )
        ax_C.legend(fontsize=8, loc="best")
        ax_C.tick_params(labelsize=8)
        cbar = fig.colorbar(hb, ax=ax_C, fraction=0.046, pad=0.02)
        cbar.set_label("count (log)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax_C.set_title("C) pred vs target — no data", fontsize=10)

    # ── Panel D: best / median / worst-MAE samples ───────────────────
    if cache is not None:
        preds, tgts, ctxs, maes = cache
        order = torch.argsort(maes)
        n = order.shape[0]
        if n >= 3:
            idx_best = int(order[max(0, int(0.10 * n))].item())
            idx_med = int(order[int(0.50 * n)].item())
            idx_worst = int(order[min(n - 1, int(0.90 * n))].item())
            picks = [
                ("worst-10% (P90 MAE)", idx_worst, "C3"),
                ("median (P50)", idx_med, "C0"),
                ("best-10% (P10 MAE)", idx_best, "C2"),
            ]
            # Pick a single channel — best-improvement, mirror of Panel A.
            plot_ch = _best_improvement_channel(per_channel_rows)
            if plot_ch is None:
                plot_ch = 0
            plot_ch = min(plot_ch, preds.shape[1] - 1)

            T_per = preds.shape[-1]
            t_ctx = np.arange(T_per)
            t_tgt = np.arange(T_per) + T_per

            # Stack with vertical offsets so all three are visible on one axis.
            offset = 0.0
            ymin, ymax = float("inf"), -float("inf")
            for label, idx, color in picks:
                ctx_v = ctxs[idx, plot_ch].numpy()
                tgt_v = tgts[idx, plot_ch].numpy()
                pred_v = preds[idx, plot_ch].numpy()
                # Shift this trio so its mean lands at `offset`.
                local_mean = float(np.nanmean(np.concatenate([ctx_v, tgt_v])))
                shift = offset - local_mean
                ax_D.plot(t_ctx, ctx_v + shift, color="0.5", lw=1.0, alpha=0.7)
                ax_D.plot(t_tgt, tgt_v + shift, color=color, lw=1.4, label=f"{label} — gt")
                ax_D.plot(
                    t_tgt, pred_v + shift, color=color, lw=1.2,
                    linestyle="--", alpha=0.85, label=f"{label} — pred",
                )
                yvals = np.concatenate([ctx_v + shift, tgt_v + shift, pred_v + shift])
                ymin = min(ymin, float(np.nanmin(yvals)))
                ymax = max(ymax, float(np.nanmax(yvals)))
                offset += 4.0
            ax_D.axvline(T_per, color="k", alpha=0.2, lw=0.7)
            ax_D.set_xlabel("samples (input | prediction)", fontsize=9)
            ax_D.set_ylabel("standardised signal (offset for clarity)", fontsize=9)
            ax_D.set_title(
                f"D) best / median / worst MAE samples — ch {plot_ch}",
                fontsize=10,
            )
            ax_D.legend(fontsize=7, loc="upper right", ncol=1)
            ax_D.tick_params(labelsize=8)
        else:
            ax_D.set_title("D) too few cached samples", fontsize=10)
    else:
        ax_D.set_title("D) no cached samples", fontsize=10)

    fig.suptitle(
        f"{name} — Stage 1 evaluation (K=1; standardised space)",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_video_modality(
    name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    out_path: Path,
) -> None:
    """One sample × all-channels frame-0 thumbnails: ctx / target / pred / |pred-target|."""
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    ctx_np = ctx.detach().cpu().numpy()
    # shape (B, C, T, H, W) — pick sample 0, frame 0
    b, t = 0, 0
    n_channels = pred_np.shape[1]
    fig, axes = plt.subplots(
        n_channels,
        4,
        figsize=(11, 2.0 * n_channels),
        squeeze=False,
    )
    for c in range(n_channels):
        col_imgs = [
            ("input", ctx_np[b, c, t]),
            ("target", tgt_np[b, c, t]),
            ("pred", pred_np[b, c, t]),
            ("|pred-tgt|", np.abs(pred_np[b, c, t] - tgt_np[b, c, t])),
        ]
        for col, (title, im) in enumerate(col_imgs):
            ax = axes[c][col]
            ax.imshow(im, cmap="gray" if col != 3 else "magma", aspect="auto")
            if c == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"ch {c}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{name} — sample 0, frame 0 (standardised)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Output helpers ───────────────────────────────────────────────────


def write_metrics_json(
    out_path: Path,
    checkpoint_path: Path,
    ckpt_step: Optional[int],
    args_used: Dict[str, Any],
    global_metrics: Dict[str, Dict[str, float]],
    per_channel: Dict[str, List[Dict[str, float]]],
    a2_pass: bool,
    a2_failing: List[str],
    sum_mae: float,
    n_batches: int,
) -> None:
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": ckpt_step,
        "args": args_used,
        "n_batches": n_batches,
        "sum_mae": sum_mae,
        "a2_pass": a2_pass,
        "a2_failing_modalities": a2_failing,
        "per_modality": global_metrics,
        "per_channel": per_channel,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def write_per_channel_csv(
    out_path: Path, per_channel: Dict[str, List[Dict[str, float]]]
) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["modality", "channel", "model_mae", "copy_mae", "delta", "n_valid"]
        )
        for name, rows in per_channel.items():
            for r in rows:
                w.writerow(
                    [
                        name,
                        r["channel"],
                        f"{r['model_mae']:.6f}",
                        f"{r['copy_mae']:.6f}",
                        f"{r['delta']:.6f}",
                        r["n_valid"],
                    ]
                )


def write_summary_md(
    out_path: Path,
    checkpoint_path: Path,
    ckpt_step: Optional[int],
    global_metrics: Dict[str, Dict[str, float]],
    a2_pass: bool,
    a2_failing: List[str],
    sum_mae: float,
    n_batches: int,
    n_modalities: int,
) -> None:
    lines: List[str] = []
    lines.append("# Stage 1 evaluation summary\n")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- Step: {ckpt_step if ckpt_step is not None else 'unknown'}")
    lines.append(f"- Val batches: {n_batches}")
    lines.append(f"- Modalities: {n_modalities}")
    lines.append(f"- Sum model MAE: {sum_mae:.4f}")
    gate = "PASS" if a2_pass else "FAIL"
    lines.append(f"- **A2 milestone (model < copy on every modality): {gate}**")
    if not a2_pass:
        lines.append(
            f"  - Failing modalities (model_mae ≥ copy_mae): {', '.join(a2_failing)}"
        )
    lines.append("")
    lines.append("## Per-modality metrics\n")
    lines.append(
        "| modality | model_mae | copy_mae | Δ | dir_cos | mag_ratio | gate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for n, m in global_metrics.items():
        marker = "✓" if m["model_mae"] < m["copy_mae"] else "✗"
        lines.append(
            f"| {n} | {m['model_mae']:.4f} | {m['copy_mae']:.4f} | "
            f"{m['delta']:+.4f} | {m['direction_cos']:.3f} | "
            f"{m['magnitude_ratio']:.3f} | {marker} |"
        )
    lines.append("")
    lines.append("## Notes\n")
    lines.append(
        "- `delta = copy_mae − model_mae` (positive ⇒ model beats copy)."
    )
    lines.append(
        "- `dir_cos` and `mag_ratio` are computed over samples with "
        "`||target − input||₂ > min_disp_norm`."
    )
    out_path.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--stats_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Cap on batches (default: full val set).",
    )
    p.add_argument(
        "--use_video",
        type=str,
        nargs="*",
        default=None,
        help="Camera names to enable (e.g. 'tangtv'). Required for C-Stage 1.",
    )
    p.add_argument("--n_plot_samples", type=int, default=4)
    p.add_argument("--min_disp_norm", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--hexbin_cap", type=int, default=50_000,
        help="Max (pred, target) pairs per modality reservoir-sampled "
             "for the Panel C scatter.",
    )
    p.add_argument(
        "--pct_cache_batches", type=int, default=8,
        help="Number of leading batches whose tensors are cached on CPU "
             "for Panel D best/median/worst-MAE percentile selection.",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    ck_args = ckpt["args"]
    model = E2EFoundationModel(
        diagnostics=diagnostics,
        actuators=actuators,
        d_model=ck_args["d_model"],
        n_heads=ck_args["n_heads"],
        n_layers=ck_args["n_layers"],
        dropout=0.0,
    )
    state_dict = ckpt["model_state_dict"]
    if any(".lora_" in k for k in state_dict):
        rank = int(ck_args.get("lora_rank", 16))
        alpha = float(ck_args.get("lora_alpha", 16.0))
        apply_lora_to_backbone(model.backbone, rank=rank, alpha=alpha)
        logger.info(f"LoRA detected: rank={rank} alpha={alpha}")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    ckpt_step = ckpt.get("step")
    logger.info(
        f"Loaded {args.checkpoint.name}: step={ckpt_step} "
        f"diagnostics={[c.name for c in diagnostics]}"
    )

    # Sanity check: --use_video must match the checkpoint's video diagnostics.
    ckpt_video_names = [c.name for c in diagnostics if c.kind == "video"]
    cli_video = args.use_video or []
    if set(ckpt_video_names) != set(cli_video):
        logger.warning(
            f"--use_video={cli_video} but checkpoint has video diagnostics "
            f"{ckpt_video_names}. Eval will use the checkpoint's set."
        )

    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]

    # ── Build val dataset ────────────────────────────────────────────
    stats = torch.load(args.stats_path, weights_only=False)
    val_files = resolve_val_files(args.data_dir, args.val_fraction, args.seed)
    logger.info(f"Val files: {len(val_files)}")
    if not val_files:
        raise SystemExit(f"No HDF5 files matched {args.data_dir}/*_processed.h5")

    # Lengths cache lives next to the checkpoint, mirroring trainer convention
    # but with an eval-specific suffix so it cannot collide with a running job.
    lengths_cache = (
        args.checkpoint.parent / f"lengths_eval_stage1_val.pt"
    )
    if lengths_cache.exists():
        # Stale caches are the chunk-cache footgun (memory:
        # project_chunk_cache_bug) — safer to recompute on every eval call.
        lengths_cache.unlink()

    ds = TokamakMultiFileDataset(
        val_files,
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=args.chunk_duration_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=lengths_cache,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
    )

    # ── Eval loop ────────────────────────────────────────────────────
    accum = GlobalAccumulator(diag_names)
    per_chan = PerChannelAccumulator(diag_names)
    hexbin = HexbinAccumulator(diag_names, cap=args.hexbin_cap)
    pct_cache = PercentileSampleCache(
        diag_names, n_batches=args.pct_cache_batches
    )
    # Video modalities still use the old single-batch image plot path.
    video_first_batch_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    rng = random.Random(args.seed)
    n_processed = 0
    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        predictions, diag_inputs, targets, masks = forward_one_batch(
            model, batch, device
        )
        for cfg in model.diagnostics:
            n = cfg.name
            copy_pred, copy_target, copy_mask = copy_baseline_for_modality(
                cfg, batch, device
            )
            ctx = diag_inputs[n]
            accum.update_modality(
                n,
                pred=predictions[n],
                target=targets[n],
                ctx=ctx,
                mask=masks[n],
                copy_pred=copy_pred,
                min_disp_norm=args.min_disp_norm,
            )
            per_chan.update_modality(
                n,
                pred=predictions[n],
                copy_pred=copy_pred,
                target=targets[n],
                mask=masks[n],
            )
            if cfg.kind != "video":
                hexbin.update(n, predictions[n], targets[n], masks[n])
                pct_cache.maybe_update(
                    i, n, predictions[n], targets[n], ctx, masks[n]
                )
        accum.step()
        n_processed += 1

        if i == 0:
            for cfg in model.diagnostics:
                if cfg.kind == "video":
                    video_first_batch_cache[cfg.name] = {
                        "pred": predictions[cfg.name].detach().cpu(),
                        "target": targets[cfg.name].detach().cpu(),
                        "ctx": diag_inputs[cfg.name].detach().cpu(),
                    }

        if (i + 1) % 10 == 0:
            logger.info(f"  batch {i + 1} processed")

    logger.info(f"Eval complete: {n_processed} batches.")

    # ── Finalise metrics ─────────────────────────────────────────────
    global_metrics = accum.finalize()
    per_channel_results = per_chan.finalize()
    sum_mae = sum(m["model_mae"] for m in global_metrics.values())
    a2_failing = [
        n for n, m in global_metrics.items() if m["model_mae"] >= m["copy_mae"]
    ]
    a2_pass = not a2_failing

    # ── Print stdout table (trainer-compatible format) ───────────────
    print()
    print("Validation (full val set, K=1; MAE model vs copy):")
    for n, m in global_metrics.items():
        gap = m["copy_mae"] - m["model_mae"]
        arrow = "↓" if gap > 0 else "↑"
        print(
            f"  {n:<24} model={m['model_mae']:.4f}  copy={m['copy_mae']:.4f}  "
            f"{arrow} {abs(gap):.4f}  | dir_cos={m['direction_cos']:+.3f}  "
            f"mag_ratio={m['magnitude_ratio']:.3f}  | "
            f"pred_d={m['pred_delta']:.4f}  tgt_d={m['tgt_delta']:.4f}  "
            f"ratio={m['delta_ratio']:.3f}"
        )
    print(f"  [sum model MAE] {sum_mae:.4f}")
    print(f"  [A2 milestone]  {'PASS' if a2_pass else 'FAIL'}")
    if not a2_pass:
        print(f"  [A2 failing]    {', '.join(a2_failing)}")
    print()

    # ── Persist outputs ──────────────────────────────────────────────
    args_serialisable = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    write_metrics_json(
        args.output_dir / "metrics.json",
        args.checkpoint,
        ckpt_step,
        args_serialisable,
        global_metrics,
        per_channel_results,
        a2_pass,
        a2_failing,
        sum_mae,
        n_processed,
    )
    write_per_channel_csv(
        args.output_dir / "per_channel.csv", per_channel_results
    )
    write_summary_md(
        args.output_dir / "summary.md",
        args.checkpoint,
        ckpt_step,
        global_metrics,
        a2_pass,
        a2_failing,
        sum_mae,
        n_processed,
        len(global_metrics),
    )

    # ── Demo-shot trajectory pass (Panel A) ─────────────────────────
    demo_shot: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    if val_files:
        logger.info(f"Demo-shot trajectory: {val_files[0].name}")
        demo_shot = collect_demo_shot_trajectory(
            model=model,
            file_path=val_files[0],
            chunk_duration_s=args.chunk_duration_s,
            warmup_s=args.warmup_s,
            stats=stats,
            diag_names=diag_names,
            act_names=act_names,
            device=device,
            max_chunks=200,
        )

    # ── Plots ────────────────────────────────────────────────────────
    for cfg in diagnostics:
        out_path = plots_dir / f"{cfg.name}.png"
        try:
            if cfg.kind == "video":
                vcache = video_first_batch_cache.get(cfg.name)
                if vcache is None:
                    continue
                plot_video_modality(
                    cfg.name,
                    pred=vcache["pred"],
                    target=vcache["target"],
                    ctx=vcache["ctx"],
                    out_path=out_path,
                )
            else:
                rows = per_channel_results.get(cfg.name, [])
                hex_xy = hexbin.get(cfg.name)
                cache = pct_cache.gather(cfg.name)
                shot_data = (
                    demo_shot.get(cfg.name) if demo_shot is not None else None
                )
                plot_ts_4panel(
                    name=cfg.name,
                    cfg=cfg,
                    per_channel_rows=rows,
                    hexbin_xy=hex_xy,
                    cache=cache,
                    demo_shot=shot_data,
                    chunk_duration_s=args.chunk_duration_s,
                    out_path=out_path,
                    rng=rng,
                )
        except Exception as exc:
            logger.warning(f"Plot for {cfg.name} failed: {exc}")

    logger.info(f"Wrote: {args.output_dir / 'metrics.json'}")
    logger.info(f"Wrote: {args.output_dir / 'per_channel.csv'}")
    logger.info(f"Wrote: {args.output_dir / 'summary.md'}")
    logger.info(f"Wrote: {plots_dir}/<modality>.png")


if __name__ == "__main__":
    main()