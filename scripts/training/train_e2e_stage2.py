"""Stage 2 short-rollout fine-tuning for the end-to-end foundation model.

Implements ``ResearchPlan.MD`` §4.2: wrap the Stage-1-pretrained model in
:class:`TokenSpaceRollout` and train on full-backprop rollouts with a
stepwise ``K = 1 → K_max`` curriculum. The model's own diagnostic-token
predictions flow into the next step (no re-tokenization); actuator tokens
are re-tokenized from fresh per-step commands. Loss = per-modality masked
MAE summed over all ``K`` steps (equal per-step weights).

Smoke test::

    pixi run python scripts/training/train_e2e_stage2.py \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --checkpoint_dir /tmp/e2e_stage2_smoke \
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
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout

logger = logging.getLogger("e2e_stage2")


# ── Modality inventory (duplicated from stage 1 by design — keeps the two ──
#    scripts independent so a Stage 2 iteration can't break a running Stage 1).

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

# Per-modality sampling rates in Hz (match ``TokamakH5Dataset.SIGNAL_CONFIGS``).
# Used to split a ``prediction_horizon_s`` target into K *time-equal* slices —
# each 50 ms slice carries a modality-dependent sample count.
SLOW_FS = 100.0
FAST_FS = 10_000.0
SAMPLE_RATES_HZ: Dict[str, float] = {
    **{name: SLOW_FS for name, _ in SLOW_TS_MODALITIES},
    **{name: FAST_FS for name, _, _ in FAST_TS_MODALITIES},
    **{name: FAST_FS for name, _ in ACTUATOR_MODALITIES},
}


def build_configs(
    chunk_duration_s: float,
) -> Tuple[List[DiagnosticConfig], List[ActuatorConfig]]:
    slow_samples = round(chunk_duration_s * SLOW_FS)
    fast_samples = round(chunk_duration_s * FAST_FS)
    diagnostics: List[DiagnosticConfig] = [
        DiagnosticConfig(name, "slow_ts", n_channels, slow_samples)
        for name, n_channels in SLOW_TS_MODALITIES
    ] + [
        DiagnosticConfig(name, "fast_ts", n_channels, fast_samples, patch)
        for name, n_channels, patch in FAST_TS_MODALITIES
    ]
    actuators: List[ActuatorConfig] = [
        ActuatorConfig(name, n_channels, fast_samples, n_tokens=5)
        for name, n_channels in ACTUATOR_MODALITIES
    ]
    return diagnostics, actuators


# ── Shot-file resolution ─────────────────────────────────────────────────


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


# ── Target splitting (time-based, per-modality) ──────────────────────────


def samples_per_step(name: str, chunk_duration_s: float) -> int:
    """Number of raw samples one 50 ms step contributes for this modality."""
    return round(chunk_duration_s * SAMPLE_RATES_HZ[name])


def split_target_by_step(
    target_tensor: torch.Tensor,
    name: str,
    k_steps: int,
    chunk_duration_s: float,
) -> List[torch.Tensor]:
    """Split a ``(B, C, T_total)`` target into ``k_steps`` per-step slices.

    Splits by *time*, not by sample count: each slice carries
    ``samples_per_step(name, chunk_duration_s)`` samples, derived from the
    modality's native sample rate. Prevents a latent bug if a modality's
    sample rate changes or a new modality with an unusual rate is added.
    """
    per_step = samples_per_step(name, chunk_duration_s)
    expected = per_step * k_steps
    actual = target_tensor.shape[-1]
    if actual < expected:
        raise ValueError(
            f"{name}: target length {actual} < expected {expected} "
            f"(= {per_step} × {k_steps})"
        )
    return [
        target_tensor[..., k * per_step : (k + 1) * per_step].contiguous()
        for k in range(k_steps)
    ]


# ── NaN handling + masked MAE (same semantics as Stage 1) ────────────────


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
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    cleaned_pred, pred_mask = _clean_and_mask(pred, None)
    cleaned_target, target_mask = _clean_and_mask(target, mask)
    combined = pred_mask * target_mask
    diff = (cleaned_pred - cleaned_target).abs() * combined
    return diff.sum() / combined.sum().clamp_min(1.0)


# ── Curriculum ───────────────────────────────────────────────────────────


def current_K(step: int, curriculum_steps: int, K_max: int) -> int:
    """Stepwise curriculum: hold each K for ``curriculum_steps // K_max`` steps.

    - Steps ``[0, B)``: K = 1
    - Steps ``[B, 2B)``: K = 2
    - ...
    - Steps ``[(K_max - 1) * B, curriculum_steps)``: K = K_max
    - Steps ``[curriculum_steps, max_steps)``: K = K_max

    where ``B = max(1, curriculum_steps // K_max)``.
    """
    block = max(1, curriculum_steps // K_max)
    k = min(K_max, step // block + 1)
    return k


# ── Rollout forward + per-step loss ──────────────────────────────────────


def rollout_forward_loss(
    rollout: TokenSpaceRollout,
    batch: Dict,
    diagnostic_names: List[str],
    actuator_names: List[str],
    k_steps: int,
    chunk_duration_s: float,
    device: torch.device,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """Tokenise the step-0 diagnostics, split targets/actuators per-step,
    run the K-step rollout and return (summed loss, per-step per-modality MAE).

    Inputs are NaN-cleaned before the forward pass; loss terms use masks
    combining the dataset's upstream ``_mask`` keys with per-tensor finite masks.
    """
    # Diagnostic initial state (step 0) from the dataset's ``inputs`` half.
    diag_initial: Dict[str, torch.Tensor] = {}
    for name in diagnostic_names:
        raw = batch["inputs"][name].to(device).float()
        cleaned, _ = _clean_and_mask(raw, None)
        diag_initial[name] = cleaned

    # Per-step actuator commands and diagnostic targets from the ``targets`` half.
    act_per_step: List[Dict[str, torch.Tensor]] = []
    target_per_step: List[Dict[str, torch.Tensor]] = []
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []

    for k in range(k_steps):
        act_k: Dict[str, torch.Tensor] = {}
        for name in actuator_names:
            raw = batch["targets"][name].to(device).float()
            slice_k = split_target_by_step(raw, name, k_steps, chunk_duration_s)[k]
            cleaned, _ = _clean_and_mask(slice_k, None)
            act_k[name] = cleaned
        act_per_step.append(act_k)

        tgt_k: Dict[str, torch.Tensor] = {}
        mk_k: Dict[str, Optional[torch.Tensor]] = {}
        for name in diagnostic_names:
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

    # Forward rollout (executes inside the caller's autocast context).
    result = rollout(diag_initial, act_per_step)

    total_loss = torch.zeros((), device=device)
    per_step: List[Dict[str, float]] = []
    for k in range(k_steps):
        per_mod: Dict[str, float] = {}
        for name in diagnostic_names:
            mae = masked_mae(
                result.predictions[k][name],
                target_per_step[k][name],
                mask_per_step[k][name],
            )
            per_mod[name] = mae.item()
            total_loss = total_loss + mae
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
    amp_ctx_factory,
    max_batches: Optional[int] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Run the full K=K_max rollout on val batches; return per-step per-modality
    averaged metrics.

    Returns a nested dict: ``out[k][name]`` has ``model_mae``, ``copy_mae``,
    ``pred_delta``, ``tgt_delta``, ``delta_ratio``. Copy baseline at step k is
    the step-0 diagnostic input — "predict yesterday's state forever".
    """
    rollout.model.eval()
    keys = ("model_mae", "copy_mae", "pred_delta", "tgt_delta")
    sums = {
        k: {name: {m: 0.0 for m in keys} for name in diagnostic_names}
        for k in range(K_max)
    }
    n_batches = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with amp_ctx_factory():
            _, _ = rollout_forward_loss(  # warm-up to reuse infrastructure;
                # keep explicit below for metrics
                rollout, batch, diagnostic_names, actuator_names,
                k_steps=K_max, chunk_duration_s=chunk_duration_s, device=device,
            )
        # Re-run with persistent intermediates for metrics.
        diag_initial: Dict[str, torch.Tensor] = {}
        for name in diagnostic_names:
            raw = batch["inputs"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            diag_initial[name] = cleaned
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

        with amp_ctx_factory():
            result = rollout(diag_initial, act_per_step)

        for k in range(K_max):
            for name in diagnostic_names:
                pred = result.predictions[k][name].float()
                tgt = target_per_step[k][name]
                existing = mask_per_step[k][name]
                inp = diag_initial[name]

                cleaned_pred, mp = _clean_and_mask(pred, None)
                cleaned_tgt, mt = _clean_and_mask(tgt, existing)
                combined = mp * mt
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
                copy_mae_v = (
                    (inp - cleaned_tgt).abs() * combined
                ).sum() / denom

                sums[k][name]["model_mae"] += model_mae_v.item()
                sums[k][name]["copy_mae"] += copy_mae_v.item()
                sums[k][name]["pred_delta"] += pred_delta.item()
                sums[k][name]["tgt_delta"] += tgt_delta.item()
        n_batches += 1

    rollout.model.train()
    denom = max(n_batches, 1)
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in range(K_max):
        out[k] = {}
        for name in diagnostic_names:
            s = sums[k][name]
            model_mae = s["model_mae"] / denom
            tgt_d = s["tgt_delta"] / denom
            pred_d = s["pred_delta"] / denom
            out[k][name] = {
                "model_mae": model_mae,
                "copy_mae": s["copy_mae"] / denom,
                "pred_delta": pred_d,
                "tgt_delta": tgt_d,
                "delta_ratio": pred_d / tgt_d if tgt_d > 1e-8 else float("nan"),
            }
    return out


# ── LR schedule ──────────────────────────────────────────────────────────


def build_scheduler(
    opt: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr: float,
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
        help="Stage 1 best checkpoint to initialize from. Random init if omitted "
        "(smoke-testing only — real Stage 2 should warm-start).",
    )
    parser.add_argument("--train_shots_yaml", type=Path, default=None)
    parser.add_argument("--val_shots_yaml", type=Path, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Data windowing
    parser.add_argument("--chunk_duration_s", type=float, default=0.05)
    parser.add_argument("--step_size_s", type=float, default=0.01)
    parser.add_argument("--warmup_s", type=float, default=1.0)

    # Model (must match the init checkpoint's architecture if loading)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Curriculum
    parser.add_argument("--K_max", type=int, default=10)
    parser.add_argument(
        "--curriculum_steps",
        type=int,
        default=25_000,
        help="Step budget spread over K_max stepwise blocks. After this, hold K_max.",
    )

    # Optim
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
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable bf16 autocast (forces fp32; useful for CPU or debug).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
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

    # ── Model + rollout wrapper ──────────────────────────────────────────
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

    if args.init_checkpoint is not None:
        ckpt = torch.load(
            args.init_checkpoint, weights_only=False, map_location=device
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Initialized from {args.init_checkpoint} "
            f"(val_loss={ckpt.get('val_loss', 'n/a')} at step "
            f"{ckpt.get('step', 'n/a')})"
        )
    else:
        logger.warning(
            "No --init_checkpoint; starting from random weights. "
            "Smoke-test only; real Stage 2 should warm-start from Stage 1 best."
        )

    rollout = TokenSpaceRollout(model, dt_s=args.chunk_duration_s)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model — d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads}  tokens={model.n_total_tokens}  "
        f"params={n_params / 1e6:.2f}M"
    )

    # ── Datasets ────────────────────────────────────────────────────────
    prediction_horizon_s = args.K_max * args.chunk_duration_s
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
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage2_train.pt",
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage2_val.pt",
        **shared,
    )
    logger.info(
        f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}  "
        f"prediction_horizon_s={prediction_horizon_s} (K_max={args.K_max})"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
        pin_memory=device.type == "cuda",
    )

    # ── Optim + schedule + autocast ─────────────────────────────────────
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        opt, args.max_steps, args.warmup_steps, args.min_lr
    )

    use_amp = (not args.no_amp) and device.type == "cuda"
    # bf16 has fp32-range exponents → no GradScaler needed.
    def amp_ctx_factory():
        if use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    logger.info(
        f"Starting Stage 2 — K_max={args.K_max} curriculum_steps="
        f"{args.curriculum_steps}  lr={args.lr}→{args.min_lr} "
        f"warmup={args.warmup_steps}  amp={'bf16' if use_amp else 'off'}"
    )

    # ── Train ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_step = 0
    step = 0
    running = 0.0
    running_count = 0
    prev_K = -1
    train_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        K = current_K(step, args.curriculum_steps, args.K_max)
        if K != prev_K:
            logger.info(f"Curriculum: step {step} → K = {K}")
            prev_K = K

        opt.zero_grad()
        with amp_ctx_factory():
            loss, per_step_per_mod = rollout_forward_loss(
                rollout, batch, diagnostic_names, actuator_names,
                k_steps=K, chunk_duration_s=args.chunk_duration_s, device=device,
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
            # Average across steps, per modality (compact form)
            per_mod_avg = {
                n: sum(psm[n] for psm in per_step_per_mod) / len(per_step_per_mod)
                for n in diagnostic_names
            }
            per_mod_str = ", ".join(f"{n}={v:.4f}" for n, v in per_mod_avg.items())
            logger.info(
                f"step {step}/{args.max_steps}  K={K}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}  | avg-across-steps: {per_mod_str}"
            )
            running = 0.0
            running_count = 0

        if step % args.val_every == 0 or step == args.max_steps:
            metrics = validate(
                rollout, val_loader, device,
                diagnostic_names, actuator_names,
                chunk_duration_s=args.chunk_duration_s,
                K_max=args.K_max,
                amp_ctx_factory=amp_ctx_factory,
                max_batches=args.val_max_batches,
            )
            highlight_steps = sorted({0, min(4, args.K_max - 1), args.K_max - 1})
            # → steps 1, 5, 10 (or equivalents at smaller K_max)
            logger.info(
                f"Validation @ step {step} — per-step MAE at steps "
                + ", ".join(f"{k + 1}" for k in highlight_steps)
                + "; + full K_max sum:"
            )
            for name in diagnostic_names:
                parts = []
                for k in highlight_steps:
                    m = metrics[k][name]
                    parts.append(
                        f"k{k + 1}: model={m['model_mae']:.4f} "
                        f"copy={m['copy_mae']:.4f} ratio={m['delta_ratio']:.3f}"
                    )
                logger.info(f"  {name:<25s} " + " | ".join(parts))
            val_loss = sum(
                metrics[k][name]["model_mae"]
                for k in range(args.K_max)
                for name in diagnostic_names
            )
            logger.info(f"  [sum model MAE over all K × modalities] {val_loss:.4f}")

            # Flag potential Stage-1 forgetting at step 1.
            step1_ratio = {
                name: metrics[0][name]["model_mae"] / max(metrics[0][name]["copy_mae"], 1e-8)
                for name in diagnostic_names
            }
            worst = max(step1_ratio.items(), key=lambda kv: kv[1])
            if worst[1] > 1.5:
                logger.warning(
                    f"  Step-1 MAE for {worst[0]} is {worst[1]:.2f}× copy baseline "
                    "— Stage 1 single-step skill may be eroding. Consider lower LR."
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_path = args.checkpoint_dir / "e2e_stage2_best.pt"
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

    final_path = args.checkpoint_dir / "e2e_stage2_final.pt"
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
        final_path,
    )
    logger.info(
        f"Saved final checkpoint: {final_path}. "
        f"Best val_loss={best_val_loss:.4f} at step {best_step}."
    )


if __name__ == "__main__":
    main()