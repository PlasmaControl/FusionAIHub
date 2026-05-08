"""Extended Stage 2 — full-backprop K={10,20,40,80} displacement-loss fine-tuning.

Motivated by Stage 3's k1 regression (LoRA with frozen heads degraded
single-step quality by ~2×). Extended Stage 2 keeps the displacement-loss
formulation from Stage 2b but drops LoRA entirely: every weight (tokenizers,
backbone, step-conditioning MLP, heads) trains. Gradient checkpointing on
the rollout makes K=80 full backprop memory-tractable.

Differences from Stage 2b / Stage 3b:

  - **Init from Stage 2b best** (not Stage 1, not Stage 2 base). Stage 2b
    has already escaped the copy minimum at K≤10; this stage extends that
    to K=80.
  - **Stepwise curriculum K ∈ {10, 20, 40, 80}**, 5k steps per block →
    20k total.
  - **Displacement-loss context = model's own predictions** (detached) at
    k≥1; diag_initial at k=0. Stage 2b used teacher-forced ground-truth
    context; extended Stage 2 matches inference-time rollout geometry.
  - **Full weight updates** — no LoRA, nothing frozen. All ~9.3M params
    receive gradients.
  - **Gradient checkpointing every ``--grad_checkpoint_every`` rollout
    steps** (default 10) via ``torch.utils.checkpoint``. Activation memory
    scales with group size rather than K.
  - **lr 1e-5 → 1e-7 cosine** — an order of magnitude lower than Stage 2b
    since we're fine-tuning a well-trained base, not re-training from
    a Stage-1 copy-like minimum.
  - Validation logs: per-modality dir_cos, mag_ratio, MAE at k ∈
    {1, 10, 40, 80}; k1 regression vs Stage 2b init; head-weight L2
    deltas since init (all params are trainable, so all weights should
    move — head deltas in particular are the signal LoRA suppressed).

Smoke test::

    pixi run python scripts/training/train_e2e_stage2_extended.py \\
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \\
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \\
        --checkpoint_dir /tmp/e2e_stage2_ext_smoke \\
        --max_files 4 --max_steps 15 --batch_size 2 --num_workers 0 \\
        --curriculum_Ks 2,3,4 --block_steps 5 --grad_checkpoint_every 2 \\
        --val_every 15 --log_every 3 --warmup_steps 2 \\
        --d_model 64 --n_layers 4 --n_heads 4 --device cpu
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
import torch.utils.checkpoint as torch_ckpt
import yaml
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
    TwoLevelSampler,
)
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout
from tokamak_foundation_model.utils.distributed import DistributedManager
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as _DDP


def _core(module):
    return module.module if hasattr(module, "module") else module

logger = logging.getLogger("e2e_stage2_ext")


# ── Modality inventory ───────────────────────────────────────────────────

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
    actuators: List[ActuatorConfig] = [
        ActuatorConfig(n, c, fast_samples, n_tokens=5)
        for n, c in ACTUATOR_MODALITIES
    ]
    return diagnostics, actuators


# ── Shot-file resolution (same convention as earlier scripts) ──────────


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
        train_files = [
            _shot_to_h5(data_dir, s) for s in _load_shot_yaml(train_yaml)
        ]
        train_files = [p for p in train_files if p.exists()]
        if val_yaml is not None:
            val_files = [
                _shot_to_h5(data_dir, s) for s in _load_shot_yaml(val_yaml)
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


# ── Utilities ────────────────────────────────────────────────────────────


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


def displacement_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    existing_mask: Optional[torch.Tensor],
    min_disp_norm: float,
) -> Tuple[torch.Tensor, torch.Tensor, float, float, int]:
    """Same signature and semantics as the Stage 3 ``_displacement_terms`` —
    returns ``(cos_loss, mag_loss, dir_cos, mag_ratio, n_valid)``. Tensors
    carry grad; scalars are detached summaries for logging.
    """
    # Mask-weighted reductions on static shapes — no boolean indexing and
    # no ``.item()`` in the hot loop. Critical for Extended Stage 2 because
    # this helper is called inside ``torch.utils.checkpoint`` regions;
    # every CUDA sync fires twice (forward + backward recompute).
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


# ── Curriculum: stepwise through an explicit K list ─────────────────────


def current_K_from_list(step: int, Ks: List[int], block_steps: int) -> int:
    """Block-stepwise K: hold each Ks[i] for ``block_steps`` steps.

    After ``len(Ks) * block_steps`` total steps, the last K in the list is
    held for the remainder of training.
    """
    block_idx = min(step // max(1, block_steps), len(Ks) - 1)
    return int(Ks[block_idx])


# ── Rollout with full-backprop + gradient checkpointing ─────────────────


def _decode_diag(model: E2EFoundationModel, diag_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    offset = 0
    for cfg in model.diagnostics:
        n = cfg.n_tokens()
        out[cfg.name] = model.diag_heads[cfg.name](
            diag_tokens[:, offset : offset + n]
        )
        offset += n
    return out


def _tokenize_act(
    model: E2EFoundationModel, act_inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for cfg in model.actuators:
        raw = act_inputs[cfg.name]
        cleaned, _ = _clean_and_mask(raw, None)
        pieces.append(model.act_tokenizers[cfg.name](cleaned))
    return torch.cat(pieces, dim=1)


def _tokenize_diag(
    model: E2EFoundationModel, diag_inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for cfg in model.diagnostics:
        raw = diag_inputs[cfg.name]
        cleaned, _ = _clean_and_mask(raw, None)
        pieces.append(model.diag_tokenizers[cfg.name](cleaned))
    return torch.cat(pieces, dim=1)


def _make_chunk_fn(
    model: E2EFoundationModel,
    diagnostic_names: List[str],
    group_start: int,
    group_end: int,
    act_tokens_in_group: List[torch.Tensor],
    target_in_group: List[Dict[str, torch.Tensor]],
    mask_in_group: List[Dict[str, Optional[torch.Tensor]]],
    n_diag_tokens: int,
    batch_rollout_step: torch.Tensor,
    dt_s: float,
    mae_weight: float,
    cos_weight: float,
    mag_weight: float,
    min_disp_norm: float,
    use_displacement_loss: bool,
):
    """Returns a function ``chunk_fn(diag_tokens, *prev_pred_list)`` suitable
    for ``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=False``.

    The function runs rollout steps ``[group_start, group_end)`` and returns
    ``(final_diag_tokens, chunk_loss, *last_predictions_flat)``. The
    ``prev_pred_list`` tensors are expected in the order of
    ``diagnostic_names`` and carry the (ctx-role) predictions entering the
    chunk (diag_initial for group 0, last chunk's predictions otherwise).
    """

    def chunk_fn(diag_tokens: torch.Tensor, *prev_pred_tensors: torch.Tensor):
        prev_pred = dict(zip(diagnostic_names, prev_pred_tensors))
        chunk_loss = torch.zeros((), device=diag_tokens.device)
        for i in range(group_end - group_start):
            k = group_start + i
            all_tokens = torch.cat([diag_tokens, act_tokens_in_group[i]], dim=1)
            step_idx = batch_rollout_step + (k + 1)
            time_s = batch_rollout_step.float() * dt_s + (k + 1) * dt_s

            out_tokens = model.backbone(all_tokens, step_idx, time_s)
            diag_tokens = out_tokens[:, :n_diag_tokens]
            predictions = _decode_diag(model, diag_tokens)

            for cfg in model.diagnostics:
                pred = predictions[cfg.name]
                target = target_in_group[i][cfg.name]
                mask = mask_in_group[i][cfg.name]
                # ctx = model's own previous prediction (detached) at k ≥ 1;
                # diag_initial at k = 0 is passed in via prev_pred at the
                # group boundary.
                ctx = prev_pred[cfg.name].detach()

                mae = masked_mae(pred, target, mask)
                cos_loss, mag_loss, _, _, _ = displacement_terms(
                    pred, target, ctx, mask, min_disp_norm
                )
                step_contrib = mae_weight * mae
                if use_displacement_loss:
                    step_contrib = (
                        step_contrib
                        + cos_weight * cos_loss
                        + mag_weight * mag_loss
                    )
                chunk_loss = chunk_loss + step_contrib
            prev_pred = predictions

        last_tensors = tuple(prev_pred[n] for n in diagnostic_names)
        return (diag_tokens, chunk_loss) + last_tensors

    return chunk_fn


def rollout_forward_loss_extended(
    model: E2EFoundationModel,
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
    use_displacement_loss: bool,
    grad_checkpoint_every: int,
) -> torch.Tensor:
    """Full-backprop rollout with gradient checkpointing.

    ctx semantics match Stage 2b for k=0 (ground-truth diag_initial) but
    differ at k≥1: here ctx is the *model's* previous prediction, detached.
    """
    diag_initial: Dict[str, torch.Tensor] = {}
    for name in diagnostic_names:
        raw = batch["inputs"][name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        diag_initial[name] = cleaned

    # Transfer each modality's full batch tensor to GPU ONCE, async. The
    # DataLoader returns pinned float32 CPU tensors, so ``.to(device,
    # non_blocking=True)`` truly overlaps H2D with compute. The earlier
    # lazy per-chunk pattern defeated pinning: ``split_target_by_step``
    # calls ``.contiguous()`` after a last-dim slice, which copies into
    # fresh unpinned storage — making the subsequent ``.to(non_blocking)``
    # silently blocking. Transferring the whole per-modality tensor up
    # front, then slicing on GPU, restores true async transfer. The K
    # per-step shards tile the original so resident memory is ~equal to
    # the batch tensor (no multiplier). Actuator *tokenisation* stays
    # lazy per-group below to bound activation-token residency.
    target_full: Dict[str, torch.Tensor] = {
        name: batch["targets"][name].to(device, non_blocking=True).float()
        for name in diagnostic_names
    }
    mask_full: Dict[str, Optional[torch.Tensor]] = {}
    for name in diagnostic_names:
        mask_key = f"{name}_mask"
        mask_full[name] = (
            batch["targets"][mask_key].to(device, non_blocking=True).float()
            if mask_key in batch["targets"] else None
        )
    act_full: Dict[str, torch.Tensor] = {
        name: batch["targets"][name].to(device, non_blocking=True).float()
        for name in actuator_names
    }

    # Split once per modality on GPU (cheap, no further H2D work).
    target_splits = {
        n: split_target_by_step(target_full[n], n, k_steps, chunk_duration_s)
        for n in diagnostic_names
    }
    mask_splits: Dict[str, Optional[List[torch.Tensor]]] = {
        n: (split_target_by_step(mask_full[n], n, k_steps, chunk_duration_s)
            if mask_full[n] is not None else None)
        for n in diagnostic_names
    }
    act_splits = {
        n: split_target_by_step(act_full[n], n, k_steps, chunk_duration_s)
        for n in actuator_names
    }
    target_per_step: List[Dict[str, torch.Tensor]] = [
        {n: target_splits[n][k] for n in diagnostic_names} for k in range(k_steps)
    ]
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = [
        {
            n: (mask_splits[n][k] if mask_splits[n] is not None else None)
            for n in diagnostic_names
        }
        for k in range(k_steps)
    ]
    act_input_per_step: List[Dict[str, torch.Tensor]] = [
        {n: act_splits[n][k] for n in actuator_names} for k in range(k_steps)
    ]

    # Tokenise the step-0 diag outside the checkpointed region.
    diag_tokens = _tokenize_diag(model, diag_initial)
    n_diag_tokens = diag_tokens.shape[1]

    batch_size = diag_tokens.shape[0]
    batch_rollout_step = torch.zeros(batch_size, dtype=torch.long, device=device)

    # ctx for step 0: true diag_initial tensors.
    prev_pred_tensors: Tuple[torch.Tensor, ...] = tuple(
        diag_initial[n] for n in diagnostic_names
    )

    total_loss = torch.zeros((), device=device)
    group_size = max(1, grad_checkpoint_every)
    for group_start in range(0, k_steps, group_size):
        group_end = min(group_start + group_size, k_steps)
        # Tokenise actuators for this group only — act tokens are a ~10x
        # size expansion over raw, and keeping them lazy per-group bounds
        # the peak residency. Target/mask/raw-actuator slices are already
        # on GPU from the upfront transfer.
        act_tokens_in_group: List[torch.Tensor] = []
        for k in range(group_start, group_end):
            act_inputs_k: Dict[str, torch.Tensor] = {}
            for name in actuator_names:
                cleaned, _ = _clean_and_mask(act_input_per_step[k][name], None)
                act_inputs_k[name] = cleaned
            act_tokens_in_group.append(_tokenize_act(model, act_inputs_k))

        chunk_fn = _make_chunk_fn(
            model=model,
            diagnostic_names=diagnostic_names,
            group_start=group_start,
            group_end=group_end,
            act_tokens_in_group=act_tokens_in_group,
            target_in_group=target_per_step[group_start:group_end],
            mask_in_group=mask_per_step[group_start:group_end],
            n_diag_tokens=n_diag_tokens,
            batch_rollout_step=batch_rollout_step,
            dt_s=chunk_duration_s,
            mae_weight=mae_weight,
            cos_weight=cos_weight,
            mag_weight=mag_weight,
            min_disp_norm=min_disp_norm,
            use_displacement_loss=use_displacement_loss,
        )
        outputs = torch_ckpt.checkpoint(
            chunk_fn, diag_tokens, *prev_pred_tensors, use_reentrant=False,
        )
        diag_tokens = outputs[0]
        chunk_loss = outputs[1]
        prev_pred_tensors = tuple(outputs[2:])
        total_loss = total_loss + chunk_loss

    return total_loss


# ── Validation ───────────────────────────────────────────────────────────


@torch.no_grad()
def validate(
    model: E2EFoundationModel,
    loader: DataLoader,
    device: torch.device,
    diagnostic_names: List[str],
    actuator_names: List[str],
    chunk_duration_s: float,
    K_max: int,
    min_disp_norm: float,
    max_batches: Optional[int] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Full K_max rollout, no checkpointing; return per-step per-modality
    ``{model_mae, copy_mae, dir_cos, mag_ratio}``. Context at k=0 is
    ``diag_initial``; at k≥1 it's the model's own prediction from step k-1
    (matching training-time semantics).
    """
    model.eval()
    keys = ("model_mae", "copy_mae", "dir_cos", "mag_ratio")
    sums = {
        k: {n: {m: 0.0 for m in keys} for n in diagnostic_names}
        for k in range(K_max)
    }
    counts = {
        k: {n: {"mae": 0, "disp": 0} for n in diagnostic_names}
        for k in range(K_max)
    }
    rollout = TokenSpaceRollout(model, dt_s=chunk_duration_s)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
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

        result = rollout(diag_initial, act_per_step, collect_history=False)

        for k in range(K_max):
            for name in diagnostic_names:
                pred = result.predictions[k][name].float()
                target = target_per_step[k][name]
                mask = mask_per_step[k][name]
                # Teacher-forced ctx for metrics (consistency with Stage 2b
                # val and the §5.9 gate tests, which also use GT context).
                ctx = (
                    diag_initial[name] if k == 0 else target_per_step[k - 1][name]
                )
                mae = masked_mae(pred, target, mask).item()
                copy_mae = masked_mae(diag_initial[name], target, mask).item()
                _, _, dir_cos_t, mag_ratio_t, n_valid_t = displacement_terms(
                    pred, target, ctx, mask, min_disp_norm
                )
                # displacement_terms now returns scalar tensors; .item() here
                # is fine — validate runs off the hot training path.
                n_valid_f = float(n_valid_t.item())
                sums[k][name]["model_mae"] += mae
                sums[k][name]["copy_mae"] += copy_mae
                counts[k][name]["mae"] += 1
                if n_valid_f > 0:
                    sums[k][name]["dir_cos"] += float(dir_cos_t.item())
                    sums[k][name]["mag_ratio"] += float(mag_ratio_t.item())
                    counts[k][name]["disp"] += 1
            # Free this step's resident GPU tensors before moving on. The
            # ctx at step k+1 is target_per_step[k], so we keep the current
            # step's target; the previous step's target is safe to drop.
            result.predictions[k] = None  # type: ignore[index]
            act_per_step[k] = None  # type: ignore[index]
            mask_per_step[k] = None  # type: ignore[index]
            if k > 0:
                target_per_step[k - 1] = None  # type: ignore[index]
    model.train()
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


def head_and_tokenizer_weight_l2(
    model: E2EFoundationModel,
) -> Dict[str, float]:
    """L2 norms of each diagnostic head's projection weight AND its sibling
    tokenizer's projection weight — monitored for movement over training.

    LoRA runs showed heads "stuck". With all params trainable here, both
    heads and tokenizers should move; stagnation would be evidence of a
    deeper architectural bottleneck.
    """
    out: Dict[str, float] = {}
    for cfg in model.diagnostics:
        head = model.diag_heads[cfg.name]
        if hasattr(head, "proj"):
            out[f"{cfg.name}/head"] = head.proj.weight.detach().float().norm().item()
        elif hasattr(head, "deconv"):
            out[f"{cfg.name}/head"] = head.deconv.weight.detach().float().norm().item()
        tok = model.diag_tokenizers[cfg.name]
        if hasattr(tok, "proj"):
            out[f"{cfg.name}/tok"] = tok.proj.weight.detach().float().norm().item()
        elif hasattr(tok, "conv"):
            out[f"{cfg.name}/tok"] = tok.conv.weight.detach().float().norm().item()
    return out


# ── Driver ───────────────────────────────────────────────────────────────


def _parse_int_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument(
        "--init_checkpoint", type=Path, default=None,
        help="Stage 2b best checkpoint. Random init if omitted (smoke test).",
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

    # Curriculum
    parser.add_argument(
        "--curriculum_Ks", type=str, default="10,20,40,80",
        help="Comma-separated list of K values for the stepwise curriculum.",
    )
    parser.add_argument(
        "--block_steps", type=int, default=5000,
        help="Training steps held at each K in the curriculum.",
    )

    # Loss
    parser.add_argument("--mae_weight", type=float, default=1.0)
    parser.add_argument("--cos_weight", type=float, default=0.3)
    parser.add_argument("--mag_weight", type=float, default=0.1)
    parser.add_argument("--min_disp_norm", type=float, default=0.01)
    parser.add_argument(
        "--no_displacement_loss", action="store_true",
        help="Disable the cos+log-mag displacement terms (MAE only).",
    )

    # Memory
    parser.add_argument(
        "--grad_checkpoint_every", type=int, default=10,
        help="Group size for torch.utils.checkpoint on the rollout. 0 "
        "disables checkpointing (full activations saved).",
    )

    # Optim
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_max_batches", type=int, default=20)

    # k1 regression monitoring
    parser.add_argument(
        "--k1_reference_path", type=Path, default=None,
        help="Checkpoint whose metrics[0] provides the k1 MAE reference "
        "(defaults to --init_checkpoint).",
    )
    parser.add_argument("--k1_regression_warn_ratio", type=float, default=1.10)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument(
        "--resume_checkpoint", type=Path, default=None,
        help="Resume from *_latest.pt or *_final.pt, restoring model + "
        "optimizer + scheduler + step + best_val_loss. Intended for 24 h-wall "
        "SLURM resubmission. Overrides --init_checkpoint.",
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
    stats = torch.load(args.stats_path, weights_only=False)

    diagnostics, actuators = build_configs(args.chunk_duration_s)
    diagnostic_names = [c.name for c in diagnostics]
    actuator_names = [c.name for c in actuators]
    logger.info(
        f"Diagnostics ({len(diagnostics)}): " + ", ".join(diagnostic_names)
    )
    logger.info(f"Actuators ({len(actuators)}): " + ", ".join(actuator_names))

    curriculum_Ks = _parse_int_list(args.curriculum_Ks)
    K_max = max(curriculum_Ks)
    logger.info(
        f"Curriculum: K ∈ {curriculum_Ks}, {args.block_steps} steps/block; "
        f"K_max = {K_max}"
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
        state_dict = ckpt["model_state_dict"]
        # If the init checkpoint has LoRA keys (unlikely for Stage 2b but
        # possible), drop them — we're training without LoRA and don't
        # want stale adapter weights.
        state_dict = {k: v for k, v in state_dict.items() if ".lora_" not in k}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys (ignored): {unexpected[:5]}…")
        if missing:
            logger.warning(f"Missing keys (left at init): {missing[:5]}…")
        logger.info(
            f"Initialized from {args.init_checkpoint.name} "
            f"(val_loss={ckpt.get('val_loss', 'n/a')} "
            f"step={ckpt.get('step', 'n/a')})"
        )
    else:
        logger.warning(
            "No --init_checkpoint; random weights. Smoke-test only — real "
            "extended Stage 2 must warm-start from Stage 2b best."
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total_tokens = model.n_total_tokens
    logger.info(
        f"Model — d_model={args.d_model} n_layers={args.n_layers} "
        f"n_heads={args.n_heads}  tokens={n_total_tokens}  "
        f"params={n_params / 1e6:.2f}M  trainable={n_train / 1e6:.2f}M"
    )

    # ── DDP wrapper for training forward ────────────────────────────────
    # Stage 2_extended's training forward calls model.backbone(...) directly,
    # bypassing the high-level model.__call__. To make DDP all_reduce fire
    # cleanly, wrap the per-step compute in a tiny Module and DDP that.
    class _TrainStepModule(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.model = base
        def forward(
            self, batch, k_steps, chunk_duration_s, mae_weight, cos_weight,
            mag_weight, min_disp_norm, use_displacement_loss, grad_checkpoint_every,
        ):
            return rollout_forward_loss_extended(
                self.model, batch, diagnostic_names, actuator_names,
                k_steps=k_steps, chunk_duration_s=chunk_duration_s,
                device=device, mae_weight=mae_weight, cos_weight=cos_weight,
                mag_weight=mag_weight, min_disp_norm=min_disp_norm,
                use_displacement_loss=use_displacement_loss,
                grad_checkpoint_every=grad_checkpoint_every,
            )

    train_step_module: torch.nn.Module = _TrainStepModule(model)
    if dm.distributed:
        train_step_module = _DDP(
            train_step_module,
            device_ids=[dm.device_index],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    use_disp = not args.no_displacement_loss
    logger.info(
        f"Loss: mae_w={args.mae_weight} cos_w={args.cos_weight} "
        f"mag_w={args.mag_weight} min_disp={args.min_disp_norm} "
        f"displacement={'on' if use_disp else 'off'}  "
        f"grad_checkpoint_every={args.grad_checkpoint_every}"
    )

    # ── k1 reference ───────────────────────────────────────────────────
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not read k1 reference from {ref_path}: {exc}")
    if k1_reference:
        logger.info(
            "k1 reference: "
            + ", ".join(f"{n}={v:.4f}" for n, v in k1_reference.items())
        )
    else:
        logger.info("k1 reference unavailable — regression check disabled.")

    # ── Dataset ───────────────────────────────────────────────────────
    prediction_horizon_s = K_max * args.chunk_duration_s
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
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage2_ext_train.pt",
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=args.checkpoint_dir / "lengths_e2e_stage2_ext_val.pt",
        **shared,
    )
    logger.info(
        f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}  "
        f"prediction_horizon_s={prediction_horizon_s:.3f}"
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

    # Initial weight snapshot (head + tokenizer norms) for drift monitoring.
    initial_weight_norms = head_and_tokenizer_weight_l2(model)
    logger.info("Initial head/tokenizer L2 (for drift monitoring):")
    for key, val in initial_weight_norms.items():
        logger.info(f"  {key:<30s} {val:.4f}")

    logger.info(
        f"Starting extended Stage 2 — lr={args.lr}→{args.min_lr} "
        f"warmup={args.warmup_steps} amp={'bf16' if use_amp else 'off'}"
    )

    best_val_loss = float("inf")
    best_step = 0
    resume_start_step = 0
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

        K = current_K_from_list(step, curriculum_Ks, args.block_steps)
        if K != prev_K:
            logger.info(f"Curriculum: step {step} → K = {K}")
            prev_K = K

        opt.zero_grad()
        with amp_ctx_factory():
            loss = train_step_module(
                batch, K, args.chunk_duration_s,
                args.mae_weight, args.cos_weight, args.mag_weight,
                args.min_disp_norm, use_disp, args.grad_checkpoint_every,
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
            logger.info(
                f"step {step}/{args.max_steps}  K={K}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}"
            )
            running = 0.0
            running_count = 0

        if step % args.val_every == 0 or step == args.max_steps:
            # Pass bare model — validate constructs its own rollout from it.
            metrics = validate(
                model, val_loader, device,
                diagnostic_names, actuator_names,
                chunk_duration_s=args.chunk_duration_s,
                K_max=K_max,
                min_disp_norm=args.min_disp_norm,
                max_batches=args.val_max_batches,
            )
            highlight = sorted({0, min(9, K_max - 1), min(39, K_max - 1), K_max - 1})
            logger.info(
                f"Validation @ step {step} — per-modality m(ae) / cos / mratio "
                f"at k ∈ {{{', '.join(str(k + 1) for k in highlight)}}}:"
            )
            for name in diagnostic_names:
                parts = []
                for k in highlight:
                    m = metrics[k][name]
                    parts.append(
                        f"k{k + 1}: m={m['model_mae']:.3f} "
                        f"c={m['copy_mae']:.3f} "
                        f"dcos={m['dir_cos']:+.3f} "
                        f"mr={m['mag_ratio']:.2f}"
                    )
                logger.info(f"  {name:<25s} " + " | ".join(parts))
            val_loss = sum(
                metrics[k][name]["model_mae"]
                for k in range(K_max)
                for name in diagnostic_names
            )
            all_dc = [
                metrics[k][name]["dir_cos"]
                for k in range(K_max)
                for name in diagnostic_names
                if metrics[k][name]["dir_cos"] == metrics[k][name]["dir_cos"]
            ]
            mean_dc = sum(all_dc) / max(1, len(all_dc))
            logger.info(
                f"  [sum model MAE] {val_loss:.4f}   "
                f"[mean direction_cos across K×modalities] {mean_dc:+.4f}"
            )

            # k1 regression
            if k1_reference:
                regressions: List[str] = []
                for name in diagnostic_names:
                    if name not in k1_reference:
                        continue
                    cur = metrics[0][name]["model_mae"]
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
                        metrics[0][n]["model_mae"] / k1_reference[n]
                        for n in diagnostic_names
                        if n in k1_reference and k1_reference[n] > 1e-8
                    )
                    logger.info(
                        f"  k1 regression OK (max current/reference ratio = "
                        f"{max_ratio:.2f}×)"
                    )

            # Head + tokenizer drift
            cur_norms = head_and_tokenizer_weight_l2(model)
            deltas = {
                k: abs(cur_norms[k] - initial_weight_norms[k])
                for k in cur_norms
                if k in initial_weight_norms
            }
            head_deltas = {k: v for k, v in deltas.items() if k.endswith("/head")}
            tok_deltas = {k: v for k, v in deltas.items() if k.endswith("/tok")}
            max_head = max(head_deltas.values()) if head_deltas else 0.0
            max_tok = max(tok_deltas.values()) if tok_deltas else 0.0
            logger.info(
                f"  [weight L2 |Δ| from init] max_head={max_head:.5f} "
                f"max_tokenizer={max_tok:.5f}"
            )
            if step >= 5000 and max_head < 1e-4:
                logger.warning(
                    "  Head weights have not moved in 5k+ steps — flat region?"
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
                    "mean_dir_cos": mean_dc,
                    "metrics": metrics,
                    "diagnostics": [asdict(c) for c in diagnostics],
                    "actuators": [asdict(c) for c in actuators],
                    "args": vars(args),
                }
                latest_path = args.checkpoint_dir / "e2e_stage2_ext_latest.pt"
                torch.save(ckpt_state, latest_path)
                if is_new_best:
                    best_path = args.checkpoint_dir / "e2e_stage2_ext_best.pt"
                    torch.save(ckpt_state, best_path)
                    logger.info(
                        f"  ✓ new best val_loss={val_loss:.4f}  saved {best_path.name}"
                    )
            dm.barrier()

    if dm.is_main:
        final_path = args.checkpoint_dir / "e2e_stage2_ext_final.pt"
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