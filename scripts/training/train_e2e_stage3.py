"""Stage 3 long-rollout LoRA fine-tuning for the end-to-end foundation model.

Implements ``ResearchPlan.MD`` §4.3 with the design decisions recorded for
this project:

  - **LoRA** (``e2e/lora.py``): every attention module in the backbone is
    wrapped with a rank-16 low-rank adapter. Base Stage 2 weights are
    frozen; only LoRA params + (optional) LayerNorms train.
  - **Lightweight replay buffer** (``e2e/replay.py``): 10k entries pointing
    into a ~200-trajectory pool. Buffer state tokens are advanced by the
    model's own predictions; ground-truth and actuator context is looked up
    lazily. ``K_max`` = 80 steps.
  - **Pushforward with per-step logging**: each training step runs
    ``K_current`` pushforward steps. Intermediate predictions are detached
    (zero grad through K−1 steps) so memory equals single-step training.
    Per-step losses are logged for free.
  - **Stepwise curriculum K ∈ {10, 20, 30, 40, 50, 60, 70, 80}**: each block
    held for ``curriculum_steps / 8`` steps.
  - **bf16 autocast** wrapping forward + loss only.

Smoke test::

    pixi run python scripts/training/train_e2e_stage3.py \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --checkpoint_dir /tmp/e2e_stage3_smoke \
        --max_files 4 --max_steps 20 --batch_size 2 --num_workers 0 \
        --K_max 5 --curriculum_steps 16 --pool_size 4 --buffer_size 8 \
        --val_every 1000 --device cpu
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
from tokamak_foundation_model.utils.distributed import DistributedManager
from torch.nn.parallel import DistributedDataParallel as _DDP


def _core(module):
    return module.module if hasattr(module, "module") else module
from tokamak_foundation_model.e2e.lora import (
    apply_lora_to_backbone,
    freeze_non_lora_parameters,
)
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.replay import (
    BufferBatch,
    ReplayBuffer,
    build_pool_from_dataset,
)

logger = logging.getLogger("e2e_stage3")


# ── Modality inventory + sample rates (duplicated from Stage 1/2) ────────

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
    **{n: SLOW_FS for n, _ in SLOW_TS_MODALITIES},
    **{n: FAST_FS for n, _, _ in FAST_TS_MODALITIES},
    **{n: FAST_FS for n, _ in ACTUATOR_MODALITIES},
}


def build_configs(
    chunk_duration_s: float,
) -> Tuple[List[DiagnosticConfig], List[ActuatorConfig]]:
    slow_samples = round(chunk_duration_s * SLOW_FS)
    fast_samples = round(chunk_duration_s * FAST_FS)
    diag: List[DiagnosticConfig] = [
        DiagnosticConfig(n, "slow_ts", c, slow_samples)
        for n, c in SLOW_TS_MODALITIES
    ] + [
        DiagnosticConfig(n, "fast_ts", c, fast_samples, p)
        for n, c, p in FAST_TS_MODALITIES
    ]
    act: List[ActuatorConfig] = [
        ActuatorConfig(n, c, fast_samples, n_tokens=5)
        for n, c in ACTUATOR_MODALITIES
    ]
    return diag, act


# ── Shot-file resolution (same convention as Stages 1/2) ─────────────────


def _load_shot_yaml(path: Path) -> List[int]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    shots = data.get("shots", []) if isinstance(data, dict) else (data or [])
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
    if train_shots_yaml is not None:
        train_files = [
            _shot_to_h5(data_dir, s) for s in _load_shot_yaml(train_shots_yaml)
        ]
        train_files = [p for p in train_files if p.exists()]
        if val_shots_yaml is not None:
            val_files = [
                _shot_to_h5(data_dir, s) for s in _load_shot_yaml(val_shots_yaml)
            ]
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


# ── NaN handling + masked MAE ────────────────────────────────────────────


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
    cleaned_pred, pred_mask = _clean_and_mask(pred, None)
    cleaned_target, target_mask = _clean_and_mask(target, mask)
    combined = pred_mask * target_mask
    diff = (cleaned_pred - cleaned_target).abs() * combined
    return diff.sum() / combined.sum().clamp_min(1.0)


# ── Curriculum ───────────────────────────────────────────────────────────


def current_K(
    step: int,
    curriculum_steps: int,
    K_min: int = 10,
    K_max: int = 80,
    n_blocks: int = 8,
) -> int:
    """Stepwise curriculum: 8 equal-width blocks from K_min to K_max."""
    block_size = max(1, curriculum_steps // n_blocks)
    block_idx = min(step // block_size, n_blocks - 1)
    K_step = (K_max - K_min) // max(1, n_blocks - 1)
    return K_min + block_idx * K_step


# ── One training step (pushforward with per-step logging) ────────────────


def pushforward_step(
    model: E2EFoundationModel,
    batch: BufferBatch,
    K: int,
    chunk_duration_s: float,
    amp_ctx_factory=None,
    *,
    use_displacement_loss: bool = False,
    cos_weight: float = 0.3,
    mag_weight: float = 0.1,
    min_disp_norm: float = 0.01,
    initial_truth: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Dict[str, float]]], torch.Tensor]:
    """Run ``K`` pushforward rollout steps starting from ``batch.state_tokens``.

    ``amp_ctx_factory`` is applied *per iteration* (not wrapping the whole
    loop). Wrapping the outer loop with ``torch.amp.autocast`` and then
    nesting ``torch.no_grad`` inside it corrupts grad tracking on the
    grad-enabled iteration (PyTorch interaction between autocast and
    re-enabling grad after a nested no_grad); the per-iteration pattern
    sidesteps that.

    Displacement loss (optional, ``use_displacement_loss=True``): only
    applied on the final (grad-carrying) step. Adds
    ``cos_weight · (1 − cos_sim(pred−ctx, target−ctx))
     + mag_weight · |log‖pred−ctx‖ − log‖target−ctx‖|``
    to the step's MAE. With only LoRA parameters trainable and heads
    frozen, these gradients flow *only* into the attention LoRA adapters
    — pushing them to route tokens so that the frozen head's decoded
    output has the correct displacement direction and magnitude, rather
    than the copy-like prediction Stage 2's pure-MAE training settled on.

    Per-step context for displacement (teacher-forced):
      - ``k == 0``: ``initial_truth[name]`` — ground-truth state at the
        buffer's ``rollout_step`` window, looked up from the pool.
      - ``k >= 1``: ``batch.gt_per_step[k-1][name]``.

    Returns
    -------
    final_loss
        Scalar loss at rollout step ``K`` — the only term that carries grad.
    per_step_metrics
        Length-``K`` list of ``{modality: {"mae": float, "dir_cos": float,
        "mag_ratio": float}}``. No grad (summary floats).
    last_state_tokens
        ``(B, n_diag_tokens, d_model)`` — diagnostic-token state after the
        final (grad-carrying) step. Detached before returning so the caller
        can write it back into the buffer without pinning the graph.
    """
    if amp_ctx_factory is None:
        amp_ctx_factory = lambda: contextlib.nullcontext()
    batch_size = batch.state_tokens.shape[0]
    n_diag_tokens = batch.state_tokens.shape[1]
    device = batch.state_tokens.device

    # actuator tokenisation helper
    def _tokenize_actuators(
        act_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        for cfg in model.actuators:
            raw = act_inputs[cfg.name]
            cleaned, _ = _clean_and_mask(raw, None)
            pieces.append(model.act_tokenizers[cfg.name](cleaned))
        return torch.cat(pieces, dim=1)

    def _decode(tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        offset = 0
        for cfg in model.diagnostics:
            n = cfg.n_tokens()
            out[cfg.name] = model.diag_heads[cfg.name](
                tokens[:, offset : offset + n]
            )
            offset += n
        return out

    diag_tokens = batch.state_tokens  # already on device
    # Tuples of (mae_stack, dcos_stack, mr_stack) — scalar tensors on-device
    # per modality. Batched to CPU once after the rollout loop so we don't
    # pay hundreds of CUDA syncs per training step.
    per_step_metrics: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    final_loss = torch.zeros((), device=device)
    # ``dt_s`` per rollout step (50 ms in our windowing).
    dt_s = chunk_duration_s
    for k in range(K):
        act_tokens = _tokenize_actuators(batch.act_per_step[k])
        all_tokens = torch.cat([diag_tokens, act_tokens], dim=1)
        step_idx = batch.rollout_step + (k + 1)
        time_s = batch.rollout_step.float() * dt_s + (k + 1) * dt_s
        is_last = k == K - 1

        # Autocast must wrap the compute *inside* each iteration; nesting
        # torch.no_grad inside an outer autocast breaks grad on re-enable.
        grad_ctx = contextlib.nullcontext() if is_last else torch.no_grad()
        with amp_ctx_factory(), grad_ctx:
            out_tokens = model.backbone(all_tokens, step_idx, time_s)
            pred_diag_tokens = out_tokens[:, :n_diag_tokens]
            predictions = _decode(pred_diag_tokens)
            mae_tensors_step: List[torch.Tensor] = []
            dcos_tensors_step: List[torch.Tensor] = []
            mr_tensors_step: List[torch.Tensor] = []
            nv_tensors_step: List[torch.Tensor] = []
            step_loss = torch.zeros((), device=device)
            for cfg in model.diagnostics:
                target = batch.gt_per_step[k][cfg.name]
                mask = batch.mask_per_step[k][cfg.name]
                mae = masked_mae(predictions[cfg.name], target, mask)
                step_loss = step_loss + mae

                # Context for this step's displacement (teacher-forced).
                if k == 0:
                    if initial_truth is None:
                        # Should not happen in training (caller must provide
                        # initial_truth when use_displacement_loss=True), but
                        # fall back to the tokens' own decode for robustness.
                        ctx = _decode(batch.state_tokens)[cfg.name]
                    else:
                        ctx = initial_truth[cfg.name]
                else:
                    ctx = batch.gt_per_step[k - 1][cfg.name]

                cos_loss, mag_loss, dir_cos_t, mag_ratio_t, nv_t = _displacement_terms(
                    predictions[cfg.name], target, ctx, mask, min_disp_norm
                )
                if is_last and use_displacement_loss:
                    step_loss = step_loss + cos_weight * cos_loss + mag_weight * mag_loss
                # Collect scalar tensors; batched .cpu() below avoids
                # per-modality CUDA syncs inside the hot loop.
                mae_tensors_step.append(mae.detach())
                dcos_tensors_step.append(dir_cos_t)
                mr_tensors_step.append(mag_ratio_t)
                nv_tensors_step.append(nv_t)
            per_step_metrics.append(
                (
                    torch.stack(mae_tensors_step),
                    torch.stack(dcos_tensors_step),
                    torch.stack(mr_tensors_step),
                    torch.stack(nv_tensors_step),
                )
            )
            if is_last:
                final_loss = step_loss
            # Advance: the token state for the next step is the diag slice
            # of backbone output. Detach on non-final steps (redundant
            # inside torch.no_grad but explicit).
            diag_tokens = pred_diag_tokens if is_last else pred_diag_tokens.detach()

    # Single cross-device transfer for all (K × n_modalities) scalars.
    mae_mat = torch.stack([t[0] for t in per_step_metrics]).detach().cpu()
    dcos_mat = torch.stack([t[1] for t in per_step_metrics]).detach().cpu()
    mr_mat = torch.stack([t[2] for t in per_step_metrics]).detach().cpu()
    nv_mat = torch.stack([t[3] for t in per_step_metrics]).detach().cpu()
    diagnostic_name_list = [c.name for c in model.diagnostics]
    per_step_metrics_out: List[Dict[str, Dict[str, float]]] = []
    for k in range(len(per_step_metrics)):
        per_mod: Dict[str, Dict[str, float]] = {}
        for j, name in enumerate(diagnostic_name_list):
            nv = float(nv_mat[k, j].item())
            per_mod[name] = {
                "mae": float(mae_mat[k, j].item()),
                "dir_cos": float(dcos_mat[k, j].item()) if nv > 0 else float("nan"),
                "mag_ratio": float(mr_mat[k, j].item()) if nv > 0 else float("nan"),
            }
        per_step_metrics_out.append(per_mod)
    return final_loss, per_step_metrics_out, diag_tokens.detach()


# ── Validation ───────────────────────────────────────────────────────────


def _displacement_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    existing_mask: Optional[torch.Tensor],
    min_disp_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """Displacement loss terms + logging summaries.

    Returns ``(cos_loss, mag_loss, dir_cos, mag_ratio, n_valid)``:
      - ``cos_loss``  — ``(1 − cos_sim(pred − ctx, target − ctx)).mean()``
        over samples where ``‖target − ctx‖ > min_disp_norm``. Carries grad
        through ``pred`` when called outside of ``torch.no_grad``.
      - ``mag_loss``  — ``|log‖pred − ctx‖ − log‖target − ctx‖|.mean()``
        over the same valid subset. Log form so undershoot and overshoot
        are penalised symmetrically.
      - ``dir_cos``   — detached float, for logging.
      - ``mag_ratio`` — detached ``‖pred − ctx‖ / ‖target − ctx‖`` mean.
      - ``n_valid``   — samples that passed the threshold.

    If fewer than one sample passes, both loss tensors are returned as
    ``torch.zeros((), device=pred.device)`` (no gradient contribution), and
    ``dir_cos`` / ``mag_ratio`` are ``NaN``.
    """
    # Mask-weighted reductions on static shapes. Avoids boolean-indexed
    # gathers and ``.item()`` calls in the hot loop; every CUDA sync here
    # fires twice inside a ``torch.utils.checkpoint`` region (forward +
    # backward recompute).
    cleaned_pred, pm = _clean_and_mask(pred, None)
    cleaned_tgt, tm = _clean_and_mask(target, existing_mask)
    cleaned_ctx, cm = _clean_and_mask(ctx, None)
    joint = pm * tm * cm
    disp_pred = (cleaned_pred - cleaned_ctx) * joint
    disp_tgt = (cleaned_tgt - cleaned_ctx) * joint

    batch = pred.shape[0]
    dp_flat = disp_pred.reshape(batch, -1)
    dt_flat = disp_tgt.reshape(batch, -1)
    tgt_norm = dt_flat.norm(dim=1)
    pred_norm = dp_flat.norm(dim=1)
    valid_f = (tgt_norm > min_disp_norm).float()
    denom = valid_f.sum().clamp_min(1.0)

    cos_per = F.cosine_similarity(dp_flat, dt_flat, dim=1, eps=1e-8)
    cos_loss = ((1.0 - cos_per) * valid_f).sum() / denom

    eps = 1e-6
    log_pred = torch.log(pred_norm.clamp_min(eps))
    log_tgt = torch.log(tgt_norm.clamp_min(eps))
    mag_loss = ((log_pred - log_tgt).abs() * valid_f).sum() / denom

    dir_cos = (cos_per.detach() * valid_f).sum() / denom
    mag_ratio = (
        (pred_norm.detach() / tgt_norm.detach().clamp_min(eps)) * valid_f
    ).sum() / denom
    n_valid = valid_f.sum().detach()
    return cos_loss, mag_loss, dir_cos, mag_ratio, n_valid


def validate_rollout(
    model: E2EFoundationModel,
    val_batch: BufferBatch,
    K: int,
    chunk_duration_s: float,
    diagnostic_names: List[str],
    amp_ctx_factory=None,
    initial_truth: Optional[Dict[str, torch.Tensor]] = None,
    min_disp_norm: float = 0.01,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Run a full K-step rollout on a val batch, per-step per-modality metrics.

    Returns ``metrics[k][name] = {model_mae, copy_mae, dir_cos, mag_ratio}``.

    - ``model_mae``: masked L1 between prediction and ground truth at step k+1.
    - ``copy_mae``: masked L1 between the step-0 decoded input (no-change
      prediction) and ground truth at step k+1.
    - ``dir_cos``: cosine similarity of ``pred - ctx`` and ``target - ctx``.
      ``ctx = initial_truth[name]`` at k=0 (teacher-forced true initial
      state); ``ctx = gt_per_step[k-1]`` for k≥1. Gated on
      ``‖target - ctx‖ > min_disp_norm`` — returns NaN if fewer than one
      sample in the batch clears that threshold.
    - ``mag_ratio``: ``‖pred - ctx‖ / ‖target - ctx‖`` over the same valid
      subset; <1 means undershoot, >1 overshoot.

    ``initial_truth`` should hold ground-truth raw signals at the buffer's
    ``rollout_step`` per sample (shape ``(B, C, T)``). If not supplied, we
    fall back to decoding ``val_batch.state_tokens`` — an approximation
    that's OK when tokenizer+head is near-identity but noisier when it
    isn't, so pass the real thing when you can.
    """
    model.eval()
    batch_size = val_batch.state_tokens.shape[0]
    n_diag_tokens = val_batch.state_tokens.shape[1]
    device = val_batch.state_tokens.device
    if amp_ctx_factory is None:
        amp_ctx_factory = lambda: contextlib.nullcontext()

    def _decode(tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        offset = 0
        for cfg in model.diagnostics:
            n = cfg.n_tokens()
            out[cfg.name] = model.diag_heads[cfg.name](
                tokens[:, offset : offset + n]
            )
            offset += n
        return out

    # Copy baseline (step-0 input echoed every step). Also the fallback
    # for ``initial_truth`` when not provided.
    initial_pred = _decode(val_batch.state_tokens)
    if initial_truth is None:
        initial_truth = initial_pred

    diag_tokens = val_batch.state_tokens
    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in range(K):
        with amp_ctx_factory():
            act_pieces = []
            for cfg in model.actuators:
                raw = val_batch.act_per_step[k][cfg.name]
                cleaned, _ = _clean_and_mask(raw, None)
                act_pieces.append(model.act_tokenizers[cfg.name](cleaned))
            act_tokens = torch.cat(act_pieces, dim=1)
            all_tokens = torch.cat([diag_tokens, act_tokens], dim=1)
            step_idx = val_batch.rollout_step + (k + 1)
            time_s = val_batch.rollout_step.float() * chunk_duration_s + (k + 1) * chunk_duration_s
            out_tokens = model.backbone(all_tokens, step_idx, time_s)
            diag_tokens = out_tokens[:, :n_diag_tokens]
            preds = _decode(diag_tokens)

        out[k] = {}
        for name in diagnostic_names:
            target = val_batch.gt_per_step[k][name]
            mask = val_batch.mask_per_step[k][name]
            ctx = initial_truth[name] if k == 0 else val_batch.gt_per_step[k - 1][name]

            model_mae_v = masked_mae(preds[name], target, mask)
            copy_mae_v = masked_mae(initial_pred[name], target, mask)
            _, _, dir_cos_t, mag_ratio_t, nv_t = _displacement_terms(
                preds[name], target, ctx, mask, min_disp_norm
            )
            nv = float(nv_t.item())
            out[k][name] = {
                "model_mae": model_mae_v.item(),
                "copy_mae": copy_mae_v.item(),
                "dir_cos": float(dir_cos_t.item()) if nv > 0 else float("nan"),
                "mag_ratio": float(mag_ratio_t.item()) if nv > 0 else float("nan"),
            }
    model.train()
    return out


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
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="Stage 2 best checkpoint to initialise from.")
    parser.add_argument("--train_shots_yaml", type=Path, default=None)
    parser.add_argument("--val_shots_yaml", type=Path, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # Data windowing
    parser.add_argument("--chunk_duration_s", type=float, default=0.05)
    parser.add_argument("--step_size_s", type=float, default=0.01)
    parser.add_argument("--warmup_s", type=float, default=1.0)

    # Model (must match init checkpoint's architecture)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)

    # Curriculum
    parser.add_argument("--K_min", type=int, default=10)
    parser.add_argument("--K_max", type=int, default=80)
    parser.add_argument("--n_curriculum_blocks", type=int, default=8)
    parser.add_argument("--curriculum_steps", type=int, default=40_000)

    # Dynamics-diagnostics logging. These three metrics go next to MAE in
    # every validation log and produce the signal that ambiguous MAE
    # improvements leave out:
    #   - dir_cos: does the model move in the direction of the target?
    #   - mag_ratio: does the displacement magnitude match?
    #   - k1_regression: is single-step quality degrading vs the init base?
    parser.add_argument("--min_disp_norm", type=float, default=0.01,
                        help="Minimum ‖target − ctx‖ per-sample below which a "
                        "sample is excluded from direction_cos / "
                        "magnitude_ratio stats and from the displacement-loss "
                        "terms.")
    parser.add_argument(
        "--use_displacement_loss",
        action="store_true",
        help="Add cos+log-mag displacement terms to the final-step training "
        "loss (see pushforward_step docstring). With only LoRA adapters "
        "trainable and heads frozen, these gradients shape attention "
        "routing so the frozen head's decode yields the correct "
        "displacement direction and magnitude. Off by default; set for "
        "Stage 3b on the Stage 2b base.",
    )
    parser.add_argument(
        "--cos_weight", type=float, default=0.3,
        help="Weight on the cosine-direction displacement loss term.",
    )
    parser.add_argument(
        "--mag_weight", type=float, default=0.1,
        help="Weight on the log-magnitude displacement loss term.",
    )
    parser.add_argument(
        "--k1_reference_path", type=Path, default=None,
        help="Checkpoint to read the reference k1-MAE-per-modality from "
        "for the Stage 3 single-step regression check. Defaults to "
        "--init_checkpoint; pass explicitly to compare against a "
        "different baseline.",
    )
    parser.add_argument(
        "--k1_regression_warn_ratio", type=float, default=1.10,
        help="Warn when current k1 model-MAE exceeds the reference by more "
        "than this factor (default: >10%% regression).",
    )

    # Replay
    parser.add_argument("--pool_size", type=int, default=200)
    parser.add_argument("--buffer_size", type=int, default=10_000)
    parser.add_argument("--buffer_refresh_period", type=int, default=50)
    parser.add_argument("--buffer_refresh_fraction", type=float, default=0.1)

    # Optim
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=40_000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_batch_size", type=int, default=8)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--resume_checkpoint", type=Path, default=None,
        help="Resume from a *_latest.pt or *_final.pt checkpoint, restoring "
        "model + optimizer + scheduler + step + best_val_loss. LoRA keys in "
        "the state_dict are expected; we apply LoRA before loading. "
        "Overrides --init_checkpoint.",
    )
    args = parser.parse_args()

    dm = DistributedManager()

    logging.basicConfig(
        level=logging.INFO if dm.is_main else logging.WARNING,
        format=f"%(asctime)s %(levelname)s [rank{dm.rank}] %(message)s",
    )
    # Per-rank seed: replay buffers are independent per rank, so each
    # should sample different data — otherwise DDP gradient averaging
    # is pointless.
    torch.manual_seed(args.seed + dm.rank)
    random.seed(args.seed + dm.rank)

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
        args.data_dir, args.train_shots_yaml, args.val_shots_yaml,
        args.max_files, args.val_fraction, args.seed,
    )
    logger.info(f"Files — train: {len(train_files)}  val: {len(val_files)}")
    if not train_files or not val_files:
        raise SystemExit("No train or val files resolved; aborting.")
    stats = torch.load(args.stats_path, weights_only=False)

    # ── Model: build → load Stage 2 weights → apply LoRA → freeze base ──
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
        diagnostics=diagnostics, actuators=actuators,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)

    if args.init_checkpoint is not None:
        ckpt = torch.load(
            args.init_checkpoint, weights_only=False, map_location=device
        )
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            f"Initialized from {args.init_checkpoint.name} "
            f"(val_loss={ckpt.get('val_loss', 'n/a')} step={ckpt.get('step', 'n/a')})"
        )
    else:
        logger.warning(
            "No --init_checkpoint; random weights. Smoke-test only — real "
            "Stage 3 should warm-start from Stage 2 best."
        )

    apply_lora_to_backbone(
        model.backbone, rank=args.lora_rank, alpha=args.lora_alpha
    )
    freeze_non_lora_parameters(model)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"LoRA applied: rank={args.lora_rank}  trainable={n_train / 1e6:.3f}M  "
        f"total={n_total / 1e6:.2f}M  (trainable ratio {n_train / n_total:.1%})"
    )

    # ── k1-MAE reference (Stage 2/2b base, for regression monitoring) ──
    # Extract k1 model-MAE per modality from the init checkpoint's saved
    # validation metrics. If --k1_reference_path is set, use that file
    # instead. Silently skip if neither path yields usable metrics.
    k1_reference: Dict[str, float] = {}
    ref_path = args.k1_reference_path or args.init_checkpoint
    if ref_path is not None and ref_path.exists():
        try:
            ref_ckpt = torch.load(ref_path, weights_only=False, map_location="cpu")
            ref_metrics = ref_ckpt.get("metrics")
            if ref_metrics and 0 in ref_metrics:
                for cfg in diagnostics:
                    entry = ref_metrics[0].get(cfg.name)
                    if entry and "model_mae" in entry:
                        k1_reference[cfg.name] = float(entry["model_mae"])
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            logger.warning(f"Could not read k1 reference from {ref_path}: {exc}")
    if k1_reference:
        logger.info(
            "k1 reference (from "
            f"{ref_path.name if ref_path is not None else 'n/a'}"
            "): "
            + ", ".join(f"{n}={v:.4f}" for n, v in k1_reference.items())
        )
    else:
        logger.info(
            "k1 reference not available — regression check will be skipped."
        )

    # ── Dataset (shared by pool + val) ────────────────────────────────────
    prediction_horizon_s = args.K_max * args.chunk_duration_s
    shared_ds = dict(
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
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage3_train.pt",
        **shared_ds,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage3_val.pt",
        **shared_ds,
    )
    logger.info(f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}")

    # ── Trajectory pool + replay buffer ──────────────────────────────────
    logger.info(
        f"Building trajectory pool ({args.pool_size} trajectories, K_max={args.K_max})"
    )
    pool = build_pool_from_dataset(
        train_ds,
        size=args.pool_size,
        K_max=args.K_max,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
        sample_rates_hz=SAMPLE_RATES_HZ,
        chunk_duration_s=args.chunk_duration_s,
        collate_fn=collate_fn,
        seed=args.seed,
    )

    def tokenize_initial(diag_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Diagnostic-only tokenisation: the tokenizer modules for the diag
        modalities, concatenated, on the model's device. Used by the buffer
        when initialising fresh entries."""
        pieces: List[torch.Tensor] = []
        with torch.no_grad():
            for cfg in model.diagnostics:
                raw = diag_inputs[cfg.name].to(device).float()
                cleaned, _ = _clean_and_mask(raw, None)
                pieces.append(model.diag_tokenizers[cfg.name](cleaned))
        return torch.cat(pieces, dim=1)

    buffer = ReplayBuffer(
        pool=pool,
        size=args.buffer_size,
        K_max=args.K_max,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
        sample_rates_hz=SAMPLE_RATES_HZ,
        chunk_duration_s=args.chunk_duration_s,
        tokenize_initial_fn=tokenize_initial,
        device=device,
        seed=args.seed,
    )
    logger.info("Initialising replay buffer…")
    buffer.initialize()
    logger.info(f"Replay buffer size: {len(buffer.entries)}")

    # Val pool + buffer: small, used purely for periodic evaluation.
    val_pool = build_pool_from_dataset(
        val_ds,
        size=max(args.val_batch_size * 4, 16),
        K_max=args.K_max,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
        sample_rates_hz=SAMPLE_RATES_HZ,
        chunk_duration_s=args.chunk_duration_s,
        collate_fn=collate_fn,
        seed=args.seed + 1,
    )
    val_buffer = ReplayBuffer(
        pool=val_pool, size=args.val_batch_size * 4, K_max=args.K_max,
        diagnostic_names=diagnostic_names, actuator_names=actuator_names,
        sample_rates_hz=SAMPLE_RATES_HZ, chunk_duration_s=args.chunk_duration_s,
        tokenize_initial_fn=tokenize_initial, device=device, seed=args.seed + 1,
    )
    val_buffer.initialize()

    def _initial_truth_from_pool(
        sample_batch: BufferBatch, source_pool,
    ) -> Dict[str, torch.Tensor]:
        """Fetch the ground-truth raw signal at each sample's ``rollout_step``
        window from ``source_pool``, per diagnostic modality. Used as the
        step-0 context for direction_cos / mag_ratio metrics and the
        displacement-loss terms so the displacement basepoint is the actual
        true state, not the model's decoded approximation of it.
        """
        out: Dict[str, torch.Tensor] = {}
        for cfg in model.diagnostics:
            per_sample = []
            per = round(args.chunk_duration_s * SAMPLE_RATES_HZ[cfg.name])
            for e in sample_batch.entries:
                traj = source_pool[e.pool_idx]
                start = e.rollout_step * per
                per_sample.append(traj.diag[cfg.name][..., start : start + per])
            out[cfg.name] = torch.stack(per_sample).to(device)
        return out

    def _initial_truth_for(val_batch: BufferBatch) -> Dict[str, torch.Tensor]:
        return _initial_truth_from_pool(val_batch, val_pool)

    # ── DDP wrapper for training forward ────────────────────────────────
    # Stage 3 training uses pushforward_step(model, ...) which calls
    # model.diag_heads/.diag_tokenizers/.backbone directly. Wrap the
    # per-step compute so DDP forward is invoked once per step.
    class _TrainStepModule(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.model = base
        def forward(
            self, batch, K, chunk_duration_s, use_displacement_loss,
            cos_weight, mag_weight, min_disp_norm, initial_truth,
        ):
            return pushforward_step(
                self.model, batch, K=K,
                chunk_duration_s=chunk_duration_s,
                amp_ctx_factory=amp_ctx_factory,
                use_displacement_loss=use_displacement_loss,
                cos_weight=cos_weight, mag_weight=mag_weight,
                min_disp_norm=min_disp_norm, initial_truth=initial_truth,
            )

    train_step_module: torch.nn.Module = _TrainStepModule(model)
    if dm.distributed:
        train_step_module = _DDP(
            train_step_module,
            device_ids=[dm.device_index],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # ── Optim + schedule + autocast ─────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
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
        f"Starting Stage 3 — curriculum K∈[{args.K_min},{args.K_max}] in "
        f"{args.n_curriculum_blocks} blocks over {args.curriculum_steps} steps; "
        f"lr={args.lr}→{args.min_lr} warmup={args.warmup_steps} "
        f"amp={'bf16' if use_amp else 'off'}"
    )

    best_val_loss = float("inf")
    best_step = 0
    resume_start_step = 0
    # Stage 3's model ALREADY has LoRA applied above (via apply_lora_to_backbone),
    # so resume checkpoints containing lora_* keys load cleanly.
    if args.resume_checkpoint is not None and args.resume_checkpoint.exists():
        resume_ckpt = torch.load(
            args.resume_checkpoint, weights_only=False, map_location=device
        )
        model.load_state_dict(resume_ckpt["model_state_dict"])
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
    step = resume_start_step
    running = 0.0
    running_count = 0
    prev_K = -1
    while step < args.max_steps:
        K = current_K(
            step, args.curriculum_steps, args.K_min, args.K_max,
            args.n_curriculum_blocks,
        )
        if K != prev_K:
            logger.info(f"Curriculum: step {step} → K = {K}")
            prev_K = K

        batch = buffer.sample(args.batch_size, k_steps=K)
        # Only fetch initial_truth when the displacement loss needs it; this
        # is a pool lookup per sample and can be skipped in MAE-only runs.
        train_initial_truth = (
            _initial_truth_from_pool(batch, pool)
            if args.use_displacement_loss
            else None
        )
        opt.zero_grad()
        # autocast is applied per-iteration *inside* pushforward_step; wrapping
        # it at the outer scope corrupts grad propagation through the
        # nested torch.no_grad() of the push-forward prefix.
        final_loss, per_step_metrics, new_state = train_step_module(
            batch, K, args.chunk_duration_s,
            args.use_displacement_loss,
            args.cos_weight, args.mag_weight, args.min_disp_norm,
            train_initial_truth,
        )
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
        opt.step()
        scheduler.step()
        buffer.update(batch.entries, new_state, advance_by=K)
        running += final_loss.item()
        running_count += 1
        step += 1

        if step % args.log_every == 0:
            avg = running / running_count
            lr_now = opt.param_groups[0]["lr"]
            # Per-step MAE sum, and a mean_dir_cos across K × modalities —
            # the same signal Stage 2b logs at every step.
            step_sums = [
                sum(mod["mae"] for mod in per_step_metrics[k].values())
                for k in range(len(per_step_metrics))
            ]
            worst_k = int(max(range(len(step_sums)), key=step_sums.__getitem__))
            all_dc = [
                mod["dir_cos"]
                for step_dict in per_step_metrics
                for mod in step_dict.values()
                if mod["dir_cos"] == mod["dir_cos"]  # not nan
            ]
            mean_dir_cos = sum(all_dc) / max(1, len(all_dc))
            logger.info(
                f"step {step}/{args.max_steps}  K={K}  final_loss={avg:.4f}  "
                f"lr={lr_now:.2e}  dcos={mean_dir_cos:+.3f}  "
                f"| per-step MAE: "
                f"k1={step_sums[0]:.3f} "
                f"kmid={step_sums[len(step_sums) // 2]:.3f} "
                f"kend={step_sums[-1]:.3f}  worst=k{worst_k + 1}"
            )
            running = 0.0
            running_count = 0

        if step % args.buffer_refresh_period == 0:
            buffer.periodic_refresh(fraction=args.buffer_refresh_fraction)

        if step % args.val_every == 0 or step == args.max_steps:
            val_batch = val_buffer.sample(args.val_batch_size, k_steps=args.K_max)
            initial_truth = _initial_truth_for(val_batch)
            # validate_rollout is @torch.no_grad-decorated so backprop
            # corruption doesn't matter here, but keep autocast inside it
            # for consistency; per-iteration autocast reduces coupling to
            # the outer grad-mode state.
            val_metrics = validate_rollout(
                model, val_batch, K=args.K_max,
                chunk_duration_s=args.chunk_duration_s,
                diagnostic_names=diagnostic_names,
                amp_ctx_factory=amp_ctx_factory,
                initial_truth=initial_truth,
                min_disp_norm=args.min_disp_norm,
            )
            highlight_k = sorted({
                0,
                min(9, args.K_max - 1),
                min(39, args.K_max - 1),
                args.K_max - 1,
            })
            logger.info(
                f"Validation @ step {step} — per-modality m(ae) / cos / mratio "
                f"at k ∈ {{{', '.join(str(k + 1) for k in highlight_k)}}}:"
            )
            for name in diagnostic_names:
                parts = []
                for k in highlight_k:
                    m = val_metrics[k][name]
                    parts.append(
                        f"k{k + 1}: m={m['model_mae']:.3f} "
                        f"c={m['copy_mae']:.3f} "
                        f"dcos={m['dir_cos']:+.3f} "
                        f"mr={m['mag_ratio']:.2f}"
                    )
                logger.info(f"  {name:<25s} " + " | ".join(parts))
            val_loss = sum(
                val_metrics[k][name]["model_mae"]
                for k in range(args.K_max)
                for name in diagnostic_names
            )
            logger.info(f"  [sum model MAE over all K × modalities] {val_loss:.4f}")

            # k1 regression check: compare current k1 MAE to the reference
            # extracted from the init (or --k1_reference_path) checkpoint.
            if k1_reference:
                regressions: List[str] = []
                for name in diagnostic_names:
                    if name not in k1_reference:
                        continue
                    cur = val_metrics[0][name]["model_mae"]
                    ref = k1_reference[name]
                    if ref < 1e-8:
                        continue
                    ratio = cur / ref
                    if ratio > args.k1_regression_warn_ratio:
                        regressions.append(
                            f"{name}: {cur:.4f} / {ref:.4f} = {ratio:.2f}×"
                        )
                if regressions:
                    logger.warning(
                        "  k1 REGRESSION (current / reference > "
                        f"{args.k1_regression_warn_ratio:.2f}×): "
                        + "; ".join(regressions)
                    )
                else:
                    max_ratio = max(
                        val_metrics[0][n]["model_mae"] / k1_reference[n]
                        for n in diagnostic_names
                        if n in k1_reference and k1_reference[n] > 1e-8
                    )
                    logger.info(
                        f"  k1 regression OK (max current/reference ratio = "
                        f"{max_ratio:.2f}×)"
                    )

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
                    "metrics": val_metrics,
                    "diagnostics": [asdict(c) for c in diagnostics],
                    "actuators": [asdict(c) for c in actuators],
                    "args": vars(args),
                }
                latest_path = args.checkpoint_dir / "e2e_stage3_latest.pt"
                torch.save(ckpt_state, latest_path)
                if is_new_best:
                    best_path = args.checkpoint_dir / "e2e_stage3_best.pt"
                    torch.save(ckpt_state, best_path)
                    logger.info(
                        f"  ✓ new best val_loss={val_loss:.4f}  saved {best_path.name}"
                    )
            dm.barrier()

    if dm.is_main:
        final_path = args.checkpoint_dir / "e2e_stage3_final.pt"
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