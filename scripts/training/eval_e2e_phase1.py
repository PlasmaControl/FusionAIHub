"""Stage-1 evaluation — Phase 1: metrics only.

Implements the metric-collection half of the pipeline in
``docs/eval_stage1_plan.md`` (§§2-4). Produces three CSV.gz tables:

    per_window_metrics.csv.gz   one row per (shot, window, modality, split)
    per_shot_metrics.csv.gz     aggregated per (shot, modality, split)
    top_bottom_shots.csv.gz     top-N + bottom-N per modality per split,
                                ranked by mae_ratio_mean (worst-by-ratio
                                first — see plan §2-Q2)

No plotting in Phase 1 — plots are Phase 2/3 work.

Modes:
  * SLURM 1-node 8-GPU DDP — each rank handles a shot-shard, writes
    its own per-window CSV.gz, rank 0 aggregates after a barrier.
  * Single-GPU interactive — same code path, world_size=1, one rank
    handles all shots.

Reuses helpers from the shared ``eval_e2e.py``:
``rollout_forward_one_batch``, ``copy_baseline_for_modality``, video
standardisation, mask helpers, checkpoint+LoRA loader.

Run::

    pixi run python scripts/training/eval_e2e_phase1.py \\
        --checkpoint /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt \\
        --data_dir   /lustre/orion/fus187/proj-shared/foundation_model \\
        --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \\
        --output_dir eval_runs/stage1_phase1_smoke \\
        --splits val \\
        --max_shots 10        # smoke; remove for full split
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import random
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
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

# Re-use Phase-0 audit-approved helpers from the sibling legacy eval script.
# scripts/ is NOT a Python package (no __init__.py), so the bare
# `from scripts.training...` form fails when the script is run as
# `python scripts/training/eval_e2e_stage1_phase1.py` — Python only puts
# the script's directory on sys.path, not the repo root. Adding the
# sibling directory explicitly lets us import the legacy module by file
# name. (Future Phase-2 work may extract these helpers into a proper
# package; for Phase 1 this keeps the diff minimal.)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e import (  # type: ignore[import]  # noqa: E402
    _clean_and_mask,
    _ts_mask,
    _video_loss_gate,
    _video_standardize_per_bc,
    copy_baseline_for_modality,
    detect_stage_K,
    forward_one_batch,
    load_checkpoint_with_refine_tolerance,
    make_rollout_if_needed,
    rollout_forward_one_batch,
)

logger = logging.getLogger("eval_stage1_phase1")


# ─────────────────────────────────────────────────────────────────────
# Per-sample metric computation
# ─────────────────────────────────────────────────────────────────────


def _align_shapes(*tensors: torch.Tensor) -> List[torch.Tensor]:
    """Truncate every tensor to the per-dimension minimum across the set.

    Required for spectrogram modalities: the tokenizer's patch size T_p=8
    forces ``trunc_t = (window_samples // T_p) * T_p`` = 96 frames, but
    the raw target tensor still carries the full 98 STFT frames. Without
    alignment, ``pred - target`` raises a shape mismatch. The same
    correction is applied in the stage-2 trainer's ``validate()`` via
    ``[..., :spectro_trunc_t[name]]``; we do it shape-generically here so
    any modality with a similar trunc behavior works without per-kind
    hardcoding.
    """
    # All inputs share ndim and broadcast-compatible non-truncated dims.
    min_shape = tuple(min(t.shape[i] for t in tensors) for i in range(tensors[0].ndim))
    slicer = tuple(slice(0, n) for n in min_shape)
    return [t[slicer] for t in tensors]


@torch.no_grad()
def per_sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    mask: Optional[torch.Tensor],
    copy_pred: torch.Tensor,
    min_disp_norm: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Return per-sample (B,) tensors for model MAE, copy MAE, dcos, mag_ratio.

    Direction cosine and magnitude ratio are NaN where the target's
    displacement norm is below ``min_disp_norm`` (matches trainer semantics).
    Aggregation across the batch is the caller's responsibility — Phase 1
    keeps everything at per-sample resolution and writes to disk.
    """
    # Align all tensors to a common shape — spectrograms come out of the
    # head at trunc_t=96 while the target/mask still carry 98 STFT frames.
    if mask is None:
        # Build a dummy all-ones mask so the align step has something to
        # truncate (cheaper than special-casing the alignment).
        mask = torch.ones_like(target)
    pred, target, ctx, copy_pred, mask = _align_shapes(
        pred, target, ctx, copy_pred, mask
    )

    cleaned_pred, mask_p = _clean_and_mask(pred, None)
    cleaned_tgt, mask_t = _clean_and_mask(target, mask)
    cleaned_ctx, mask_c = _clean_and_mask(ctx, None)
    cleaned_copy, mask_cp = _clean_and_mask(copy_pred, None)

    joint = mask_p * mask_t * mask_c
    copy_joint = mask_cp * mask_t

    B = pred.shape[0]
    flat_axes = list(range(1, pred.ndim))
    denom = joint.sum(dim=flat_axes).clamp_min(1.0)
    copy_denom = copy_joint.sum(dim=flat_axes).clamp_min(1.0)

    model_mae = ((cleaned_pred - cleaned_tgt).abs() * joint).sum(dim=flat_axes) / denom
    copy_mae = ((cleaned_copy - cleaned_tgt).abs() * copy_joint).sum(dim=flat_axes) / copy_denom

    # Direction cosine / magnitude ratio on the per-sample displacement.
    disp_pred = ((cleaned_pred - cleaned_ctx) * joint).reshape(B, -1)
    disp_tgt = ((cleaned_tgt - cleaned_ctx) * joint).reshape(B, -1)
    tgt_norm = disp_tgt.norm(dim=1)
    pred_norm = disp_pred.norm(dim=1)
    dcos = torch.full((B,), float("nan"), device=pred.device)
    mag_ratio = torch.full((B,), float("nan"), device=pred.device)
    valid = tgt_norm > min_disp_norm
    if valid.any():
        dcos[valid] = F.cosine_similarity(
            disp_pred[valid], disp_tgt[valid], dim=1
        )
        mag_ratio[valid] = pred_norm[valid] / tgt_norm[valid].clamp_min(1e-6)

    return {
        "mae": model_mae.detach().cpu(),
        "copy_mae": copy_mae.detach().cpu(),
        "dcos": dcos.detach().cpu(),
        "mag_ratio": mag_ratio.detach().cpu(),
    }


# ─────────────────────────────────────────────────────────────────────
# Split + shot-id helpers
# ─────────────────────────────────────────────────────────────────────


_SHOT_ID_RE = re.compile(r"(\d+)_processed\.h5$")


def parse_shot_id(path: Path) -> int:
    m = _SHOT_ID_RE.search(path.name)
    if m is None:
        raise ValueError(f"Cannot parse shot id from {path.name!r}")
    return int(m.group(1))


def resolve_split_files(
    data_dir: Path, val_fraction: float, seed: int, split: str
) -> List[Path]:
    """Reproduce the trainer's deterministic train/val split.

    ``split='val'`` returns the val files, ``'train'`` returns the train files.
    Identical RNG state and ordering to the trainer's resolve_shot_files.
    """
    rng = random.Random(seed)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    rng.shuffle(all_files)
    n_val = max(1, int(val_fraction * len(all_files)))
    if split == "val":
        return all_files[:n_val]
    if split == "train":
        return all_files[n_val:]
    raise ValueError(f"split must be 'train' or 'val', got {split!r}")


def build_chunk_meta(ds: TokamakMultiFileDataset) -> np.ndarray:
    """Return an (N, 2) int64 array mapping global chunk index to
    ``(file_index_in_dataset, chunk_index_within_file)``.

    The dataset already maintains ``_cumulative_lengths`` and ``_valid_indices``;
    this just materialises the lookup as a flat array so the eval loop can
    fetch per-sample shot-id / window-idx in O(1) by global index.
    """
    n = len(ds)
    cum = np.asarray(ds._cumulative_lengths, dtype=np.int64)
    valid = np.asarray(ds._valid_indices, dtype=np.int64)
    out = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        pos = int(np.searchsorted(cum, i + 1) - 1)
        out[i, 0] = valid[pos]
        out[i, 1] = i - int(cum[pos])
    return out


# ─────────────────────────────────────────────────────────────────────
# DDP setup (compatible with single-GPU mode)
# ─────────────────────────────────────────────────────────────────────


def ddp_init() -> Tuple[int, int, int, torch.device]:
    """Initialise DDP from SLURM env vars; fall back to single-process.

    Returns (rank, world_size, local_rank, device).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if torch.cuda.is_available():
        # SLURM's --gpu-bind=closest makes only the locally-bound GPU
        # visible to each rank, so cuda.device_count() == 1 and the
        # correct index is always 0. Without this fallback, ranks ≥1
        # call torch.cuda.set_device(local_rank) on a non-existent
        # device → HIP error: invalid device ordinal. Matches the
        # pattern in src/.../utils/distributed.py:DistributedManager.
        visible = torch.cuda.device_count()
        device_index = local_rank if visible > 1 else 0
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        # Long timeout: shot-shard imbalance can leave fast ranks
        # idling for hours at the final barrier while slow ranks
        # finish their tail of long shots. The default 10-min NCCL
        # watchdog tripped jobs 4743239 / 4743243; 4 h gives ample
        # headroom for the slowest 8-rank shard.
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(hours=4),
        )
    return rank, world_size, local_rank, device


def ddp_finalise() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────
# Inference + per-window metric collection (per rank)
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_split(
    model: E2EFoundationModel,
    split: str,
    files: List[Path],
    stats: dict,
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
    K: int,
) -> Path:
    """Run K-step rollout inference on this rank's shot-shard. ``K=1`` is
    Stage 1's single-step path; ``K>1`` is Stage 2's autoregressive
    rollout. Writes a per-window CSV.gz with one row per (sample,
    modality, k). Returns the path.
    """
    # Shot-sharding: rank N owns files[N::world_size]. World size 1 ⇒ all files.
    my_files = files[rank::world_size] if world_size > 1 else files
    if args.max_shots and args.max_shots > 0:
        my_files = my_files[: args.max_shots]
    if not my_files:
        # Empty shard; write an empty file so rank 0 can still concatenate.
        out_path = args.output_dir / f"per_window_metrics.{split}.rank{rank}.csv.gz"
        pd.DataFrame(columns=_per_window_columns()).to_csv(out_path, index=False, compression="gzip")
        return out_path

    logger.info(
        f"[rank{rank}] split={split} shard={len(my_files)} files "
        f"(of {len(files)} total across world={world_size}); K={K}"
    )

    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]
    rollout = make_rollout_if_needed(model, K, args.chunk_duration_s)

    lengths_cache = (
        args.checkpoint.parent / f"lengths_eval_stage1_{split}_rank{rank}_K{K}.pt"
    )
    if lengths_cache.exists():
        lengths_cache.unlink()

    ds = TokamakMultiFileDataset(
        my_files,
        chunk_duration_s=args.chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=K * args.chunk_duration_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=lengths_cache,
    )
    if len(ds) == 0:
        logger.warning(f"[rank{rank}] {split}: empty dataset on this shard")
        out_path = args.output_dir / f"per_window_metrics.{split}.rank{rank}.csv.gz"
        pd.DataFrame(columns=_per_window_columns()).to_csv(out_path, index=False, compression="gzip")
        return out_path

    chunk_meta = build_chunk_meta(ds)  # (N, 2): (file_idx_in_shard, window_idx_within_file)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)

    # Stream rows into a list; concat to a DataFrame at the end of the split.
    rows: List[Dict[str, object]] = []
    n_processed = 0

    for batch_idx, batch in enumerate(loader):
        predictions_per_k, diag_initial, targets_per_k, masks_per_k = (
            rollout_forward_one_batch(
                model, rollout, batch, device, K, args.chunk_duration_s
            )
        )
        bs = next(iter(diag_initial.values())).shape[0]
        global_start = batch_idx * args.batch_size
        global_end = global_start + bs
        if global_end > len(chunk_meta):
            global_end = len(chunk_meta)
            bs = global_end - global_start  # last batch may be short
        meta_slice = chunk_meta[global_start:global_end]

        for cfg in model.diagnostics:
            n = cfg.name
            copy_pred = diag_initial[n]  # persistence baseline: echo step 0
            for k in range(K):
                ctx = diag_initial[n] if k == 0 else targets_per_k[k - 1][n]
                stats_b = per_sample_metrics(
                    pred=predictions_per_k[k][n],
                    target=targets_per_k[k][n],
                    ctx=ctx,
                    mask=masks_per_k[k][n],
                    copy_pred=copy_pred,
                    min_disp_norm=args.min_disp_norm,
                )
                mae = stats_b["mae"].numpy()
                copy_mae = stats_b["copy_mae"].numpy()
                dcos = stats_b["dcos"].numpy()
                mrat = stats_b["mag_ratio"].numpy()
                for j in range(bs):
                    file_idx_in_shard, window_idx = meta_slice[j]
                    shot_id = parse_shot_id(my_files[file_idx_in_shard])
                    m = float(mae[j])
                    cm = float(copy_mae[j])
                    ratio = m / cm if cm > 0 else float("nan")
                    rows.append({
                        "split": split,
                        "modality": n,
                        "kind": cfg.kind,
                        "shot_id": int(shot_id),
                        "window_idx": int(window_idx),
                        "window_t_s": float(window_idx) * args.chunk_duration_s,
                        "k": k + 1,
                        "mae": m,
                        "copy_mae": cm,
                        "mae_ratio": ratio,
                        "dcos": float(dcos[j]),
                        "mag_ratio": float(mrat[j]),
                    })
        n_processed += bs
        if (batch_idx + 1) % args.log_every == 0:
            logger.info(
                f"[rank{rank}] {split}: batch {batch_idx + 1}, "
                f"chunks {n_processed}/{len(ds)}"
            )

    df = pd.DataFrame(rows, columns=_per_window_columns())
    out_path = args.output_dir / f"per_window_metrics.{split}.rank{rank}.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")
    logger.info(
        f"[rank{rank}] {split}: wrote {len(df):,} rows → {out_path.name}"
    )
    return out_path


def _per_window_columns() -> List[str]:
    return [
        "split", "modality", "kind",
        "shot_id", "window_idx", "window_t_s",
        "k",
        "mae", "copy_mae", "mae_ratio",
        "dcos", "mag_ratio",
    ]


# ─────────────────────────────────────────────────────────────────────
# Rank-0 aggregation
# ─────────────────────────────────────────────────────────────────────


def aggregate_per_shot(
    per_window_files: Sequence[Path], output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate all per-rank per-window CSV.gz files, compute per-shot
    aggregates, and write both:

      output_dir / per_window_metrics.csv.gz
      output_dir / per_shot_metrics.csv.gz

    Returns ``(per_window_df, per_shot_df)`` for downstream use.
    """
    parts = []
    for f in per_window_files:
        if not f.exists():
            continue
        try:
            parts.append(pd.read_csv(f, compression="gzip"))
        except (pd.errors.EmptyDataError, EOFError):
            continue
    if not parts:
        raise RuntimeError("No per-window CSV.gz files found to aggregate")
    pw = pd.concat(parts, ignore_index=True)

    out_pw = output_dir / "per_window_metrics.csv.gz"
    pw.to_csv(out_pw, index=False, compression="gzip")
    logger.info(f"Wrote {len(pw):,} per-window rows → {out_pw.name}")

    # Per-shot aggregation grouped by (split, modality, shot_id, k).
    grouped = pw.groupby(["split", "modality", "kind", "shot_id", "k"], sort=False)
    agg_rows = []
    for (split, modality, kind, shot_id, k), g in grouped:
        n_win = len(g)
        mae_arr = g["mae"].to_numpy(dtype=np.float64)
        copy_arr = g["copy_mae"].to_numpy(dtype=np.float64)
        ratio_arr = g["mae_ratio"].to_numpy(dtype=np.float64)
        dcos_arr = g["dcos"].to_numpy(dtype=np.float64)
        mrat_arr = g["mag_ratio"].to_numpy(dtype=np.float64)

        # frac_windows_below_diag: fraction of windows where model beats copy.
        frac_below = float(np.nanmean((mae_arr < copy_arr).astype(np.float64)))

        agg_rows.append({
            "split": split,
            "modality": modality,
            "kind": kind,
            "shot_id": int(shot_id),
            "k": int(k),
            "n_windows": n_win,
            "mae_mean": float(np.nanmean(mae_arr)),
            "mae_median": float(np.nanmedian(mae_arr)),
            "mae_p95": float(np.nanpercentile(mae_arr, 95)) if n_win else float("nan"),
            "mae_max": float(np.nanmax(mae_arr)) if n_win else float("nan"),
            "copy_mae_mean": float(np.nanmean(copy_arr)),
            "copy_mae_median": float(np.nanmedian(copy_arr)),
            "mae_ratio_mean": float(np.nanmean(ratio_arr)),
            "mae_ratio_median": float(np.nanmedian(ratio_arr)),
            "frac_windows_below_diag": frac_below,
            "dcos_mean": float(np.nanmean(dcos_arr)),
            "mag_ratio_mean": float(np.nanmean(mrat_arr)),
        })
    ps = pd.DataFrame(agg_rows)
    out_ps = output_dir / "per_shot_metrics.csv.gz"
    ps.to_csv(out_ps, index=False, compression="gzip")
    logger.info(f"Wrote {len(ps):,} per-shot rows → {out_ps.name}")
    return pw, ps


def compute_gates_and_summary(
    per_window_df: pd.DataFrame,
    K: int,
    output_dir: Path,
    checkpoint_path: Path,
    ckpt_step: Optional[int],
    mag_ratio_lo: float = 0.3,
    mag_ratio_hi: float = 3.0,
) -> Dict[str, object]:
    """Aggregate per-window metrics across the val set and emit a
    PASS/FAIL summary.md plus a structured gates dict.

    Gates (ported from the retired eval_e2e_stage2.py):
      G1: model_mae < copy_mae at k=1                   (Stage 1 carry-forward)
      G2: model_mae < copy_mae at k=K                   (rollout-end gate)
      G3: direction_cos > 0 at every k                  (no anti-aligned preds)
      G4: magnitude_ratio in [lo, hi] at every k        (loose under/overshoot)

    For Stage 1 (K=1), G1 and G2 are the same metric — only G1 is reported.
    All gates are evaluated against per-modality means across the val
    split; pass/fail is per-modality and rolled up to a global gate
    (PASS iff every modality passes).
    """
    val_df = per_window_df[per_window_df["split"] == "val"].copy()
    if val_df.empty:
        # No val split in this run — gates can't be computed.
        return {"per_modality": {}, "global": {"g1": None, "g2": None,
                                                "g3": None, "g4": None}}

    modalities = sorted(val_df["modality"].unique())
    per_mod: Dict[str, Dict[str, object]] = {}
    g1_global = g2_global = g3_global = g4_global = True
    for name in modalities:
        m = val_df[val_df["modality"] == name]
        kind = m["kind"].iloc[0]
        k1 = m[m["k"] == 1]
        kK = m[m["k"] == K]
        # Per-k means used by G3/G4.
        per_k = m.groupby("k").agg(
            mae=("mae", "mean"),
            copy_mae=("copy_mae", "mean"),
            dcos=("dcos", "mean"),
            mag_ratio=("mag_ratio", "mean"),
        )
        mae_k1 = float(k1["mae"].mean()) if not k1.empty else float("nan")
        copy_k1 = float(k1["copy_mae"].mean()) if not k1.empty else float("nan")
        mae_kK = float(kK["mae"].mean()) if not kK.empty else float("nan")
        copy_kK = float(kK["copy_mae"].mean()) if not kK.empty else float("nan")
        g1 = np.isfinite(mae_k1) and np.isfinite(copy_k1) and mae_k1 < copy_k1
        if K == 1:
            g2 = g1
        else:
            g2 = (
                np.isfinite(mae_kK) and np.isfinite(copy_kK)
                and mae_kK < copy_kK
            )
        # G3: dir_cos > 0 at every k (NaN values are skipped — they mean
        # the per-window displacement norm was below min_disp_norm, so
        # direction is undefined; treat them as non-failures).
        dcos_min = float(per_k["dcos"].min(skipna=True))
        g3 = bool(np.isnan(dcos_min) or dcos_min > 0)
        # G4: mag_ratio in [lo, hi] at every k (NaN → skip).
        mr_min = float(per_k["mag_ratio"].min(skipna=True))
        mr_max = float(per_k["mag_ratio"].max(skipna=True))
        g4 = bool(
            (np.isnan(mr_min) or mr_min >= mag_ratio_lo)
            and (np.isnan(mr_max) or mr_max <= mag_ratio_hi)
        )
        per_mod[name] = {
            "kind": kind,
            "mae_k1": mae_k1, "copy_mae_k1": copy_k1,
            "mae_kK": mae_kK, "copy_mae_kK": copy_kK,
            "dcos_min_over_k": dcos_min,
            "mag_ratio_min_over_k": mr_min,
            "mag_ratio_max_over_k": mr_max,
            "g1": g1, "g2": g2, "g3": g3, "g4": g4,
        }
        g1_global = g1_global and g1
        g2_global = g2_global and g2
        g3_global = g3_global and g3
        g4_global = g4_global and g4

    # ── Render summary.md ───────────────────────────────────────────
    lines: List[str] = []
    lines.append(f"# E2E evaluation summary (K={K})\n")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- Step: {ckpt_step if ckpt_step is not None else 'unknown'}")
    lines.append(f"- Val modalities: {len(modalities)}")
    lines.append("")
    lines.append("## Gates\n")
    if K == 1:
        lines.append(
            f"- **G1 (model_mae < copy_mae @ k=1): "
            f"{'PASS' if g1_global else 'FAIL'}**"
        )
        lines.append("- G2 collapses into G1 for K=1.")
    else:
        lines.append(
            f"- **G1 (model_mae < copy_mae @ k=1): "
            f"{'PASS' if g1_global else 'FAIL'}**"
        )
        lines.append(
            f"- **G2 (model_mae < copy_mae @ k=K={K}): "
            f"{'PASS' if g2_global else 'FAIL'}**"
        )
    lines.append(
        f"- **G3 (dir_cos > 0 ∀ k): "
        f"{'PASS' if g3_global else 'FAIL'}**"
    )
    lines.append(
        f"- **G4 (mag_ratio ∈ [{mag_ratio_lo}, {mag_ratio_hi}] ∀ k): "
        f"{'PASS' if g4_global else 'FAIL'}**"
    )
    lines.append("")
    lines.append("## Per-modality breakdown\n")
    if K == 1:
        hdr = "| modality | kind | mae | copy_mae | Δ | dir_cos | mag_ratio | G1 | G3 | G4 |"
        sep = "|---|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|"
        lines.append(hdr); lines.append(sep)
        for name, m in per_mod.items():
            delta = m["copy_mae_k1"] - m["mae_k1"]
            lines.append(
                f"| {name} | {m['kind']} | {m['mae_k1']:.4f} | "
                f"{m['copy_mae_k1']:.4f} | {delta:+.4f} | "
                f"{m['dcos_min_over_k']:.3f} | {m['mag_ratio_min_over_k']:.3f}–{m['mag_ratio_max_over_k']:.3f} | "
                f"{'✓' if m['g1'] else '✗'} | "
                f"{'✓' if m['g3'] else '✗'} | "
                f"{'✓' if m['g4'] else '✗'} |"
            )
    else:
        hdr = (
            "| modality | kind | mae@1 | copy@1 | mae@K | copy@K | "
            "dcos_min | mag_min–max | G1 | G2 | G3 | G4 |"
        )
        sep = "|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|"
        lines.append(hdr); lines.append(sep)
        for name, m in per_mod.items():
            lines.append(
                f"| {name} | {m['kind']} | {m['mae_k1']:.4f} | "
                f"{m['copy_mae_k1']:.4f} | {m['mae_kK']:.4f} | "
                f"{m['copy_mae_kK']:.4f} | {m['dcos_min_over_k']:.3f} | "
                f"{m['mag_ratio_min_over_k']:.3f}–{m['mag_ratio_max_over_k']:.3f} | "
                f"{'✓' if m['g1'] else '✗'} | "
                f"{'✓' if m['g2'] else '✗'} | "
                f"{'✓' if m['g3'] else '✗'} | "
                f"{'✓' if m['g4'] else '✗'} |"
            )
    lines.append("")
    lines.append("## Notes\n")
    lines.append(
        "- Gates are evaluated on the val split, averaged across all "
        "windows per (modality, k)."
    )
    lines.append(
        "- `dir_cos` and `mag_ratio` are NaN where the per-window "
        "displacement norm is below `min_disp_norm`; NaN bins are "
        "skipped (treated as non-failures) by G3/G4."
    )
    out_md = output_dir / "summary.md"
    out_md.write_text("\n".join(lines))
    logger.info(f"Wrote {out_md.name}")

    return {
        "per_modality": per_mod,
        "global": {
            "g1": g1_global, "g2": g2_global,
            "g3": g3_global, "g4": g4_global,
        },
    }


def select_top_bottom(
    per_shot_df: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    output_dir: Path,
) -> pd.DataFrame:
    """For each (split, modality), rank shots by mean ``mae_ratio_mean``
    averaged across k, and pick the top-N (best) and bottom-N (worst).
    Plan §2-Q2: worst-by-ratio is the primary failure-mode pool. With K>1
    the average across k surfaces shots that are bad at any horizon (not
    only k=1 or only k=K), giving Phase 2/3 a single visualisation list
    that exercises the full trajectory.
    """
    rows = []
    grouped = per_shot_df.groupby(["split", "modality", "kind", "shot_id"], sort=False)
    # Collapse the k axis: one ranking row per (split, modality, shot).
    shot_rank: List[Dict[str, object]] = []
    for (split, modality, kind, shot_id), g in grouped:
        ratio_arr = g["mae_ratio_mean"].replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=np.float64)
        if np.all(np.isnan(ratio_arr)):
            continue
        shot_rank.append({
            "split": split,
            "modality": modality,
            "kind": kind,
            "shot_id": int(shot_id),
            "mae_ratio_mean": float(np.nanmean(ratio_arr)),
            "mae_mean": float(np.nanmean(g["mae_mean"].to_numpy(dtype=np.float64))),
            "copy_mae_mean": float(np.nanmean(g["copy_mae_mean"].to_numpy(dtype=np.float64))),
            "n_windows": int(g["n_windows"].iloc[0]),
            "frac_windows_below_diag": float(
                np.nanmean(g["frac_windows_below_diag"].to_numpy(dtype=np.float64))
            ),
        })
    ranked = pd.DataFrame(shot_rank)
    for (split, modality, kind), g in ranked.groupby(["split", "modality", "kind"], sort=False):
        sorted_g = g.sort_values("mae_ratio_mean", kind="stable")
        top = sorted_g.head(top_n).assign(rank_kind="top")
        bottom = sorted_g.tail(bottom_n).assign(rank_kind="bottom")
        for tbl in (top, bottom):
            for _, r in tbl.iterrows():
                rows.append({
                    "split": split,
                    "modality": modality,
                    "kind": kind,
                    "rank_kind": r["rank_kind"],
                    "shot_id": int(r["shot_id"]),
                    "mae_ratio_mean": float(r["mae_ratio_mean"]),
                    "mae_mean": float(r["mae_mean"]),
                    "copy_mae_mean": float(r["copy_mae_mean"]),
                    "n_windows": int(r["n_windows"]),
                    "frac_windows_below_diag": float(r["frac_windows_below_diag"]),
                })
    tb = pd.DataFrame(rows)
    out = output_dir / "top_bottom_shots.csv.gz"
    tb.to_csv(out, index=False, compression="gzip")
    logger.info(f"Wrote {len(tb):,} top/bottom rows → {out.name}")
    return tb


# ─────────────────────────────────────────────────────────────────────
# Config / entrypoint
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--stats_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--splits", type=str, nargs="+", default=["val"],
        choices=["train", "val"],
        help="Which splits to evaluate. Any subset of {train, val}.",
    )
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="DataLoader prefetch_factor (batches per worker queue). "
        "Ignored when --num_workers=0. Default 2 (PyTorch default).",
    )
    p.add_argument("--min_disp_norm", type=float, default=0.01)
    p.add_argument(
        "--max_shots", type=int, default=0,
        help="Cap per-rank shot count. 0 = all (production). Small int for smokes.",
    )
    p.add_argument(
        "--top_n", type=int, default=5,
        help="Top-N shots (best fit per modality) for plotting pool.",
    )
    p.add_argument(
        "--bottom_n", type=int, default=5,
        help="Bottom-N shots (worst fit by mae_ratio_mean) — the more "
             "informative pool per plan §2-Q2.",
    )
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument(
        "--K", type=int, default=0,
        help="Rollout horizon. 0 (default) autodetects from checkpoint: "
             "K=1 for Stage 1 checkpoints, K=K_max for Stage 2. Any "
             "positive value overrides — useful for evaluating a "
             "mid-curriculum Stage 2 checkpoint at the K it has actually "
             "been trained to.",
    )
    p.add_argument(
        "--mag_ratio_lo", type=float, default=0.3,
        help="Lower bound for G4 magnitude_ratio gate. Default 0.3 "
             "(loose under-shoot tolerance; tighter §5.9 target is 0.8).",
    )
    p.add_argument(
        "--mag_ratio_hi", type=float, default=3.0,
        help="Upper bound for G4 magnitude_ratio gate. Default 3.0 "
             "(loose over-shoot tolerance; tighter §5.9 target is 1.2).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rank, world_size, local_rank, device = ddp_init()
    if rank == 0:
        logger.info(
            f"Phase 1 eval — world_size={world_size} local_rank={local_rank} "
            f"device={device}"
        )

    # ── Load checkpoint (same on every rank) ─────────────────────────
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
        rank_l = int(ck_args.get("lora_rank", 16))
        alpha_l = float(ck_args.get("lora_alpha", 16.0))
        apply_lora_to_backbone(model.backbone, rank=rank_l, alpha=alpha_l)
        if rank == 0:
            logger.info(f"LoRA detected: rank={rank_l} alpha={alpha_l}")
    load_checkpoint_with_refine_tolerance(model, state_dict)
    model.eval().to(device)
    ckpt_step = ckpt.get("step")

    # Rollout horizon: 0 (default) autodetects from the checkpoint; any
    # positive value overrides. Stage 1 checkpoints have no ``K_max`` in
    # ``ckpt['args']`` and resolve to K=1.
    K = args.K if args.K > 0 else detect_stage_K(ckpt)
    if rank == 0:
        logger.info(
            f"Eval horizon K={K} ({'autodetected' if args.K == 0 else 'override'})"
        )

    stats = torch.load(args.stats_path, weights_only=False)

    # ── Per-split per-rank inference ─────────────────────────────────
    all_per_window_files: Dict[str, List[Path]] = {s: [] for s in args.splits}
    for split in args.splits:
        files = resolve_split_files(args.data_dir, args.val_fraction, args.seed, split)
        if rank == 0:
            logger.info(f"{split}: {len(files)} files in this split")
        out_path = run_split(
            model=model,
            split=split,
            files=files,
            stats=stats,
            args=args,
            device=device,
            rank=rank,
            world_size=world_size,
            K=K,
        )
        all_per_window_files[split].append(out_path)

    # ── Wait for all ranks to finish all splits before aggregating ───
    # Single post-loop barrier (replacing the per-split barrier that
    # tripped jobs 4743239/4743243): rank-0 aggregation reads each
    # rank's CSV.gz from disk and silently drops any file that doesn't
    # yet exist, so stragglers must finish before aggregation starts.
    # The 4 h NCCL timeout configured in ddp_init() makes this safe
    # against shot-shard imbalance.
    if dist.is_initialized():
        dist.barrier()

    # ── Rank-0 aggregation ───────────────────────────────────────────
    if rank == 0:
        # Gather all per-rank files for every split.
        all_files: List[Path] = []
        for split in args.splits:
            for r in range(world_size):
                p = args.output_dir / f"per_window_metrics.{split}.rank{r}.csv.gz"
                if p.exists():
                    all_files.append(p)
        per_window_df, per_shot_df = aggregate_per_shot(
            all_files, args.output_dir
        )
        top_bottom_df = select_top_bottom(
            per_shot_df, top_n=args.top_n, bottom_n=args.bottom_n,
            output_dir=args.output_dir,
        )

        gates = compute_gates_and_summary(
            per_window_df=per_window_df,
            K=K,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            ckpt_step=ckpt_step,
            mag_ratio_lo=args.mag_ratio_lo,
            mag_ratio_hi=args.mag_ratio_hi,
        )

        # Save config snapshot for reproducibility.
        config_path = args.output_dir / "config.json"
        config_path.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "checkpoint_step": ckpt_step,
            "K": K,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "world_size": world_size,
            "n_per_window_rows": int(len(per_window_df)),
            "n_per_shot_rows": int(len(per_shot_df)),
            "n_top_bottom_rows": int(len(top_bottom_df)),
            "gates": gates["global"],
        }, indent=2))
        logger.info(f"Wrote {config_path.name}")

        # Cleanup per-rank intermediate files now that aggregates are written.
        for f in all_files:
            f.unlink()
        logger.info("Phase 1 eval complete.")

    ddp_finalise()


if __name__ == "__main__":
    main()
