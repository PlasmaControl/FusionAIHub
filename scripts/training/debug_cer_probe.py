"""CER sign/normalisation/collapse probe for a trained E2E checkpoint.

Motivation: §5.9 test 5 (displacement direction) against the Stage 2 best
checkpoint returned ``direction_cos = -0.417`` for ``cer_ti`` and ``-0.192``
for ``cer_rot`` — the predictions move *away* from the target on those
modalities. This probe distinguishes four failure hypotheses:

  (1) **Mode collapse**  — model predicts ~0 regardless of input. Shows up
      as ``std(pred) << std(target)`` and ``||pred - ctx|| ≪ ||tgt - ctx||``.
      The negative direction_cos would then be an artifact of `pred - ctx ≈
      -ctx` being systematically anti-aligned with small target moves.
  (2) **Sign flip**      — preprocessing or head bias inverted. Shows up as
      direction_cos tightly clustered around ``-1``.
  (3) **Normalisation bug** — preprocessing_stats mean/std disagree with
      empirical per-channel moments. Model trained on a shifted manifold;
      predictions look wrong relative to the ground-truth half of the
      batch.
  (4) **Training failure** — neither of the above; direction_cos is
      near-zero-to-negative because the model has not learned CER dynamics.
      Stage 2b (displacement loss) should address this, not CER-specific
      plumbing.

Run::

    pixi run python scripts/training/debug_cer_probe.py \
        --checkpoint scripts/slurm/runs/e2e_stage2/e2e_stage2_best.pt \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --output_dir runs/e2e_stage2/cer_probe \
        --batch_size 64
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

logger = logging.getLogger("cer_probe")

CER_MODALITIES = ("cer_ti", "cer_rot")


def _nanclean(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _per_channel_stats(
    tensor: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel (active_fraction, mean, std) using the provided mask.

    ``tensor`` shape ``(B, C, T)``; mask same shape (float 0/1).
    Returns three tensors of shape ``(C,)``.
    """
    total = mask.sum(dim=(0, 2))
    active_frac = total / (tensor.shape[0] * tensor.shape[2])
    denom = total.clamp_min(1.0)
    mean = (tensor * mask).sum(dim=(0, 2)) / denom
    sq = ((tensor - mean.view(1, -1, 1)) ** 2) * mask
    var = sq.sum(dim=(0, 2)) / denom
    return active_frac, mean, var.clamp_min(0).sqrt()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    mod_args = ckpt["args"]
    device = torch.device("cpu")
    model = E2EFoundationModel(
        diagnostics=diagnostics,
        actuators=actuators,
        d_model=mod_args["d_model"],
        n_heads=mod_args["n_heads"],
        n_layers=mod_args["n_layers"],
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        f"Loaded {args.checkpoint.name}: step={ckpt.get('step')} "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f}"
    )

    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]

    # ── Preprocessing stats for CER ──────────────────────────────────
    stats = torch.load(args.stats_path, weights_only=False)
    logger.info("")
    logger.info("== Preprocessing stats for CER modalities ==")
    for m in CER_MODALITIES:
        if m not in stats:
            logger.warning(f"  {m}: NOT IN preprocessing_stats")
            continue
        entry = stats[m]
        # Structure varies; report whatever keys we find plus key summary.
        keys = list(entry.keys()) if isinstance(entry, dict) else type(entry)
        logger.info(f"  {m}: keys={keys}")
        if isinstance(entry, dict):
            for k, v in entry.items():
                if isinstance(v, torch.Tensor):
                    logger.info(
                        f"    {k}: shape={tuple(v.shape)}  "
                        f"mean={v.mean().item():.4f}  std={v.std().item():.4f}  "
                        f"min={v.min().item():.4f}  max={v.max().item():.4f}"
                    )

    # ── Pull one val batch with K=1 horizon (we compare step-0 input and step-1 target) ──
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
        prediction_horizon_s=0.05,
        step_size_s=0.05,
        warmup_s=1.0,
        lengths_cache_path=args.output_dir / "lengths_cer_probe.pt",
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )
    batch = next(iter(loader))

    diag_inputs = {n: _nanclean(batch["inputs"][n].float()) for n in diag_names}
    act_inputs = {n: _nanclean(batch["targets"][n].float()) for n in act_names}

    # Forward
    step_idx = torch.zeros(next(iter(diag_inputs.values())).shape[0], dtype=torch.long)
    time_offset = torch.zeros_like(step_idx, dtype=torch.float)
    predictions = model(diag_inputs, act_inputs, step_idx, time_offset)

    # ── Per-CER-modality analysis ───────────────────────────────────
    for m in CER_MODALITIES:
        logger.info("")
        logger.info(f"================ {m} ================")
        inp = _nanclean(batch["inputs"][m].float())
        tgt = _nanclean(batch["targets"][m].float())
        mask_key = f"{m}_mask"
        inp_mask = (
            batch["inputs"][mask_key].float() if mask_key in batch["inputs"]
            else torch.ones_like(inp)
        )
        tgt_mask = (
            batch["targets"][mask_key].float() if mask_key in batch["targets"]
            else torch.ones_like(tgt)
        )
        pred = _nanclean(predictions[m].float())

        # Empirical per-channel stats
        inp_frac, inp_mean, inp_std = _per_channel_stats(inp, inp_mask)
        tgt_frac, tgt_mean, tgt_std = _per_channel_stats(tgt, tgt_mask)
        pred_frac, pred_mean, pred_std = _per_channel_stats(pred, torch.ones_like(pred))

        n_active_channels = int((tgt_frac > 0.5).sum().item())
        logger.info(
            f"  Channels: {len(tgt_frac)} total, "
            f"{n_active_channels} with >50% valid (active)"
        )
        logger.info(
            f"  Input  (target-window): frac-active mean={inp_frac.mean().item():.3f}  "
            f"signal mean={inp_mean[tgt_frac > 0.5].mean().item():+.4f}  "
            f"signal std={inp_std[tgt_frac > 0.5].mean().item():.4f}"
        )
        logger.info(
            f"  Target                 : frac-active mean={tgt_frac.mean().item():.3f}  "
            f"signal mean={tgt_mean[tgt_frac > 0.5].mean().item():+.4f}  "
            f"signal std={tgt_std[tgt_frac > 0.5].mean().item():.4f}"
        )
        logger.info(
            f"  Prediction             : signal mean="
            f"{pred_mean[tgt_frac > 0.5].mean().item():+.4f}  "
            f"signal std={pred_std[tgt_frac > 0.5].mean().item():.4f}"
        )

        # Displacement distribution (per-sample)
        disp_pred = (pred - inp).reshape(pred.shape[0], -1)
        disp_tgt = (tgt - inp).reshape(tgt.shape[0], -1)
        # Mask out positions invalid in either pred or tgt (pred has no mask)
        joint = (inp_mask * tgt_mask).reshape(pred.shape[0], -1)
        disp_pred_m = disp_pred * joint
        disp_tgt_m = disp_tgt * joint

        tgt_norm = disp_tgt_m.norm(dim=1)
        pred_norm = disp_pred_m.norm(dim=1)
        valid = tgt_norm > 1e-6
        if valid.sum() < 2:
            logger.warning("  Not enough valid samples to assess displacement.")
            continue

        dir_cos = F.cosine_similarity(disp_pred_m[valid], disp_tgt_m[valid], dim=1)
        mag_ratio = pred_norm[valid] / tgt_norm[valid].clamp_min(1e-8)

        logger.info(
            f"  Direction cos (target moves > 1e-6): "
            f"n={int(valid.sum().item())}  "
            f"mean={dir_cos.mean().item():+.4f}  "
            f"median={dir_cos.median().item():+.4f}  "
            f"p05={dir_cos.kthvalue(max(1, int(0.05 * valid.sum().item()))).values.item():+.4f}  "
            f"p95={dir_cos.kthvalue(max(1, int(0.95 * valid.sum().item()))).values.item():+.4f}"
        )
        logger.info(
            f"  Magnitude ratio (pred/tgt): "
            f"mean={mag_ratio.mean().item():.4f}  "
            f"median={mag_ratio.median().item():.4f}  "
            f"p05={mag_ratio.kthvalue(max(1, int(0.05 * valid.sum().item()))).values.item():.4f}  "
            f"p95={mag_ratio.kthvalue(max(1, int(0.95 * valid.sum().item()))).values.item():.4f}"
        )

        # ── Hypothesis checks ────────────────────────────────────────
        verdict: List[str] = []
        sig_std_ratio = (
            pred_std[tgt_frac > 0.5].mean() / tgt_std[tgt_frac > 0.5].mean().clamp_min(1e-8)
        ).item()
        if sig_std_ratio < 0.1:
            verdict.append(
                f"MODE COLLAPSE: pred std is {sig_std_ratio:.1%} of target std"
            )
        elif sig_std_ratio < 0.5:
            verdict.append(
                f"undershoot: pred std is {sig_std_ratio:.1%} of target std"
            )

        if dir_cos.median().item() < -0.3:
            verdict.append(
                f"SIGN-FLIP suspect: median direction_cos = "
                f"{dir_cos.median().item():+.3f}"
            )

        if mag_ratio.median().item() < 0.1:
            verdict.append(
                f"SCALE BUG suspect: median pred displacement is "
                f"{mag_ratio.median().item():.1%} of target"
            )

        if not verdict:
            verdict.append(
                "No collapse/flip/scale artefacts detected — looks like a "
                "training-landscape issue (hypothesis 4)."
            )
        for v in verdict:
            logger.info(f"  → {v}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = args.output_dir / "cer_probe_log.txt"
    logger.info(f"(Log written to terminal; save path reserved: {out_path})")


if __name__ == "__main__":
    main()