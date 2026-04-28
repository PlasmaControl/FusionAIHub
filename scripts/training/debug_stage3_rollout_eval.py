"""Stage 3 rollout evaluation — direction_cos per step and pred-vs-GT plot.

Load a trained Stage 3 checkpoint (with LoRA), run a K-step rollout on a
single validation batch, and emit:

  (1) Per-modality per-step ``(mae, dir_cos, mag_ratio, n_valid)`` table —
      CSV + highlight-step log. Direction-cos is the metric that tells you
      whether k80 MAE improvements reflect real dynamics tracking or
      scale-shrunk-into-copy. Every step is reported (not just the
      ``{k1, k10, k40, k80}`` highlights from the training log).

  (2) One pred-vs-ground-truth trajectory plot: one sample × one channel of
      one modality × ``K × chunk_duration_s`` stitched continuously. The
      step boundaries are drawn as faint verticals so rollout drift is
      visible.

Handles LoRA-in-checkpoint automatically: detects ``lora_*`` keys in the
state_dict and applies ``apply_lora_to_backbone`` before loading.

Run::

    pixi run python scripts/training/debug_stage3_rollout_eval.py \\
        --checkpoint scripts/slurm/runs/e2e_stage3/e2e_stage3_best.pt \\
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \\
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \\
        --output_dir scripts/slurm/runs/e2e_stage3/eval \\
        --K 80 --plot_modality ts_core_temp --plot_channel 15
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.lora import apply_lora_to_backbone
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout

logger = logging.getLogger("stage3_eval")

SAMPLE_RATES_HZ = {
    "ts_core_density": 100.0, "ts_core_temp": 100.0,
    "ts_tangential_density": 100.0, "ts_tangential_temp": 100.0,
    "cer_ti": 100.0, "cer_rot": 100.0, "mse": 100.0,
    "filterscopes": 10_000.0,
    "pin": 10_000.0, "beam_voltage": 10_000.0,
    "ech_power": 10_000.0, "ech_tor_angle": 10_000.0,
    "ech_pol_angle": 10_000.0, "ech_polarization": 10_000.0,
    "gas_flow": 10_000.0, "gas_raw": 10_000.0, "rmp": 10_000.0,
}


def _nanclean(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _split(
    tensor: torch.Tensor, name: str, K: int, chunk_s: float
) -> List[torch.Tensor]:
    per = round(chunk_s * SAMPLE_RATES_HZ[name])
    return [tensor[..., k * per : (k + 1) * per].contiguous() for k in range(K)]


def _step_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    mask: Optional[torch.Tensor],
    min_disp_norm: float,
) -> Tuple[float, float, float, int]:
    """Return ``(mae, dir_cos, mag_ratio, n_valid)`` — all floats."""
    finite_pred = torch.isfinite(pred).float()
    finite_tgt = torch.isfinite(target).float()
    finite_ctx = torch.isfinite(ctx).float()
    cleaned_pred = torch.where(finite_pred.bool(), pred, torch.zeros_like(pred))
    cleaned_tgt = torch.where(finite_tgt.bool(), target, torch.zeros_like(target))
    cleaned_ctx = torch.where(finite_ctx.bool(), ctx, torch.zeros_like(ctx))
    joint = finite_pred * finite_tgt * finite_ctx
    if mask is not None:
        joint = joint * mask

    mae = (
        ((cleaned_pred - cleaned_tgt).abs() * joint).sum()
        / joint.sum().clamp_min(1.0)
    ).item()

    disp_pred = (cleaned_pred - cleaned_ctx) * joint
    disp_tgt = (cleaned_tgt - cleaned_ctx) * joint
    batch = pred.shape[0]
    dp = disp_pred.reshape(batch, -1)
    dt = disp_tgt.reshape(batch, -1)
    tgt_norm = dt.norm(dim=1)
    pred_norm = dp.norm(dim=1)
    valid = tgt_norm > min_disp_norm
    n_valid = int(valid.sum().item())
    if n_valid < 1:
        return mae, float("nan"), float("nan"), 0
    dir_cos = F.cosine_similarity(dp[valid], dt[valid], dim=1).mean().item()
    mag_ratio = (
        pred_norm[valid] / tgt_norm[valid].clamp_min(1e-6)
    ).mean().item()
    return mae, dir_cos, mag_ratio, n_valid


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--K", type=int, default=80)
    parser.add_argument("--min_disp_norm", type=float, default=0.01)
    parser.add_argument("--plot_modality", type=str, default="ts_core_temp")
    parser.add_argument("--plot_channel", type=int, default=15)
    parser.add_argument("--plot_sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint, apply LoRA if present ──────────────────────
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
    has_lora = any(".lora_" in k for k in state_dict)
    if has_lora:
        rank = int(ck_args.get("lora_rank", 16))
        alpha = float(ck_args.get("lora_alpha", 16.0))
        apply_lora_to_backbone(model.backbone, rank=rank, alpha=alpha)
        logger.info(f"LoRA detected in checkpoint: rank={rank} alpha={alpha}")

    model.load_state_dict(state_dict)
    model.eval()
    logger.info(
        f"Loaded {args.checkpoint.name}: step={ckpt.get('step')} "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f}"
    )

    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]

    # ── Build one val batch ─────────────────────────────────────────
    stats = torch.load(args.stats_path, weights_only=False)
    rng = random.Random(args.seed)
    shot_files = sorted(args.data_dir.glob("*_processed.h5"))
    rng.shuffle(shot_files)
    files = shot_files[: args.max_files]

    ds = TokamakMultiFileDataset(
        files,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=args.K * 0.05,
        step_size_s=(args.K + 1) * 0.05,  # non-overlapping chunks
        warmup_s=1.0,
        lengths_cache_path=args.output_dir / f"lengths_eval_K{args.K}.pt",
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )
    batch = next(iter(loader))

    diag_initial: Dict[str, torch.Tensor] = {
        n: _nanclean(batch["inputs"][n].float()) for n in diag_names
    }
    act_per_step: List[Dict[str, torch.Tensor]] = []
    target_per_step: List[Dict[str, torch.Tensor]] = []
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
    for k in range(args.K):
        act_per_step.append({
            n: _nanclean(_split(batch["targets"][n].float(), n, args.K, 0.05)[k])
            for n in act_names
        })
        target_per_step.append({
            n: _split(batch["targets"][n].float(), n, args.K, 0.05)[k]
            for n in diag_names
        })
        mask_per_step.append({
            n: (
                _split(batch["targets"][f"{n}_mask"].float(), n, args.K, 0.05)[k]
                if f"{n}_mask" in batch["targets"] else None
            )
            for n in diag_names
        })

    # ── Rollout ─────────────────────────────────────────────────────
    rollout = TokenSpaceRollout(model, dt_s=0.05)
    result = rollout(diag_initial, act_per_step)
    logger.info(f"Ran K={args.K} rollout on batch size {args.batch_size}.")

    # ── Per-step per-modality metrics ───────────────────────────────
    records: List[Tuple[int, str, float, float, float, int]] = []
    for k in range(args.K):
        for name in diag_names:
            pred = result.predictions[k][name]
            target = target_per_step[k][name]
            mask = mask_per_step[k][name]
            ctx = diag_initial[name] if k == 0 else target_per_step[k - 1][name]
            mae, dcos, mr, n_valid = _step_metrics(
                pred, target, ctx, mask, args.min_disp_norm
            )
            records.append((k + 1, name, mae, dcos, mr, n_valid))

    # CSV
    csv_path = args.output_dir / "rollout_metrics.csv"
    with csv_path.open("w") as f:
        f.write("step,modality,mae,dir_cos,mag_ratio,n_valid\n")
        for k, name, mae, dcos, mr, n_valid in records:
            f.write(
                f"{k},{name},{mae:.6f},{dcos:.6f},{mr:.6f},{n_valid}\n"
            )
    logger.info(f"CSV: {csv_path}")

    # Highlight-step log
    highlight = [k for k in (1, 10, 40, args.K) if k <= args.K]
    for k_report in highlight:
        logger.info(f"--- step {k_report} ---")
        for name in diag_names:
            rec = next(r for r in records if r[0] == k_report and r[1] == name)
            _, _, mae, dcos, mr, n_valid = rec
            logger.info(
                f"  {name:<25}  mae={mae:.4f}  dcos={dcos:+.4f}  "
                f"mr={mr:.3f}  n={n_valid}"
            )

    # Per-modality mean direction_cos across all K steps
    logger.info("")
    logger.info("Per-modality stats across all K steps:")
    logger.info(
        f"  {'modality':<25}  {'mean_dcos':>10}  {'mean_mr':>8}  {'mean_mae':>8}"
    )
    for name in diag_names:
        dcos_vals = [
            r[3] for r in records if r[1] == name and r[3] == r[3]  # nan filter
        ]
        mr_vals = [
            r[4] for r in records if r[1] == name and r[4] == r[4]
        ]
        mae_vals = [r[2] for r in records if r[1] == name]
        logger.info(
            f"  {name:<25}  "
            f"{sum(dcos_vals) / max(1, len(dcos_vals)):>+10.4f}  "
            f"{sum(mr_vals) / max(1, len(mr_vals)):>8.3f}  "
            f"{sum(mae_vals) / max(1, len(mae_vals)):>8.4f}"
        )

    # ── Rollout plot: one sample × one channel × K+1 windows ─────────
    m_name = args.plot_modality
    ch = args.plot_channel
    samp = args.plot_sample
    fs = SAMPLE_RATES_HZ[m_name]

    def _frame(t: torch.Tensor) -> np.ndarray:
        return _nanclean(t[samp, ch]).cpu().numpy()

    gt_segments = [_frame(diag_initial[m_name])]
    pred_segments = [_frame(diag_initial[m_name])]
    for k in range(args.K):
        gt_segments.append(_frame(target_per_step[k][m_name]))
        pred_segments.append(_frame(result.predictions[k][m_name]))
    gt_flat = np.concatenate(gt_segments)
    pred_flat = np.concatenate(pred_segments)
    t_axis = np.arange(len(gt_flat)) / fs

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_axis, gt_flat, label="Ground truth", color="black",
            linewidth=1.5, alpha=0.9)
    ax.plot(t_axis, pred_flat, label="Stage 3 prediction", color="C1",
            linewidth=1.0, alpha=0.85)
    # Step boundaries (excluding t=0)
    for k in range(1, args.K + 1):
        ax.axvline(k * 0.05, color="gray", alpha=0.15, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{m_name} ch {ch} (standardized)")
    ax.set_title(
        f"Stage 3 rollout — {m_name} ch {ch}, sample {samp}, "
        f"{args.K}-step ({args.K * 0.05:.2f}s)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = (
        args.output_dir / f"rollout_plot_{m_name}_ch{ch}_sample{samp}.png"
    )
    fig.savefig(plot_path, dpi=140, bbox_inches="tight")
    logger.info(f"Plot: {plot_path}")

    # Save raw arrays for offline replotting.
    np.savez(
        args.output_dir / f"rollout_traces_{m_name}_ch{ch}_sample{samp}.npz",
        gt=gt_flat, pred=pred_flat, t=t_axis,
    )


if __name__ == "__main__":
    main()