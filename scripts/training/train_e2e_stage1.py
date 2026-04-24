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

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

logger = logging.getLogger("e2e_stage1")


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


def build_configs(
    chunk_duration_s: float,
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
    for cfg in model.diagnostics:
        raw = batch["inputs"][cfg.name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        diag_inputs[cfg.name] = cleaned
    act_inputs: Dict[str, torch.Tensor] = {}
    for cfg in model.actuators:
        raw = batch["targets"][cfg.name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        act_inputs[cfg.name] = cleaned

    batch_size = next(iter(diag_inputs.values())).shape[0]
    step_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
    time_offset = torch.zeros(batch_size, device=device)

    predictions = model(diag_inputs, act_inputs, step_idx, time_offset)

    targets: Dict[str, torch.Tensor] = {}
    masks: Dict[str, Optional[torch.Tensor]] = {}
    for cfg in model.diagnostics:
        targets[cfg.name] = batch["targets"][cfg.name].to(device).float()
        mask_key = f"{cfg.name}_mask"
        masks[cfg.name] = (
            batch["targets"][mask_key].to(device).float()
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
    for cfg in model.diagnostics:
        loss = masked_mae(predictions[cfg.name], targets[cfg.name], masks[cfg.name])
        per_modality[cfg.name] = loss.item()
        total_loss = total_loss + loss
    return total_loss, per_modality


@torch.no_grad()
def copy_baseline_mae(
    batch: Dict,
    diagnostic_names: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """MAE of the trivial ``prediction = input`` baseline (target-sized)."""
    out: Dict[str, float] = {}
    for name in diagnostic_names:
        pred = batch["inputs"][name].to(device).float()
        target = batch["targets"][name].to(device).float()
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

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        predictions, diag_inputs, targets, masks = forward_batch(model, batch, device)
        copy_mod = copy_baseline_mae(batch, diagnostic_names, device)
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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Device: {device}")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    stats = torch.load(args.stats_path, weights_only=False)

    # ── Model + configs ─────────────────────────────────────────────────
    diagnostics, actuators = build_configs(args.chunk_duration_s)
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
    logger.info(
        f"Model — d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads}  tokens={model.n_total_tokens}  "
        f"params={n_params / 1e6:.2f}M"
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
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

    # ── Train ──────────────────────────────────────────────────────────
    logger.info(
        f"Starting training — lr schedule: linear warmup "
        f"{args.warmup_steps} steps → cosine → min_lr {args.min_lr}."
    )
    best_val_loss = float("inf")
    best_step = 0
    step = 0
    running_total = 0.0
    running_count = 0
    train_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        opt.zero_grad()
        loss, per_mod = compute_step_loss(model, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        scheduler.step()
        running_total += loss.item()
        running_count += 1
        step += 1

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
            metrics = validate(
                model,
                val_loader,
                device,
                diagnostic_names,
                max_batches=args.val_max_batches,
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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_path = args.checkpoint_dir / "e2e_stage1_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "metrics": metrics,
                        "diagnostics": [asdict(c) for c in diagnostics],
                        "actuators": [asdict(c) for c in actuators],
                        "args": vars(args),
                    },
                    best_path,
                )
                logger.info(
                    f"  ✓ new best val_loss={val_loss:.4f}  saved {best_path.name}"
                )

    ckpt_path = args.checkpoint_dir / "e2e_stage1_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
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


if __name__ == "__main__":
    main()