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
import math
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
    DistributedTwoLevelSampler,
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
from tokamak_foundation_model.e2e.output_heads import SpectrogramFlowHead
from tokamak_foundation_model.utils.distributed import DistributedManager
from torch.nn.parallel import DistributedDataParallel as _DDP

# Sibling-module import (scripts/training is on sys.path at launch) — share
# the per-bin sigma builder with the Stage 1 trainer rather than duplicate it.
from train_e2e_stage1 import build_spec_per_bin_sigma  # noqa: E402

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
    ("tin", 8),
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
    spectro_patch_f: Optional[int] = None,
    spectro_patch_t: Optional[int] = None,
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
        spectro_patch_f=spectro_patch_f, spectro_patch_t=spectro_patch_t,
    )
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


def _decode_diag(
    model: E2EFoundationModel, diag_tokens: torch.Tensor,
    return_slices: bool = False,
):
    """Decode per-modality predictions. With ``return_slices`` also return the
    per-modality backbone token slice (a generative head's flow loss needs it
    as conditioning). Default off → unchanged for the metric/eval callers."""
    out: Dict[str, torch.Tensor] = {}
    slices: Dict[str, torch.Tensor] = {}
    offset = 0
    for cfg in model.diagnostics:
        n = cfg.n_tokens()
        sl = diag_tokens[:, offset : offset + n]
        out[cfg.name] = model.diag_heads[cfg.name](sl)
        if return_slices:
            slices[cfg.name] = sl
        offset += n
    if return_slices:
        return out, slices
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
    """Mirrors ``E2EFoundationModel.tokenize`` for the diagnostic side:
    for ``kind in ("video", "spectrogram")`` look up
    ``f"{name}_valid"`` in ``diag_inputs`` and forward as the
    tokenizer's ``mask`` kwarg so missing rows route to the learned
    ``missing_token``. TS path is unchanged.
    """
    pieces: List[torch.Tensor] = []
    for cfg in model.diagnostics:
        raw = diag_inputs[cfg.name]
        cleaned, _ = _clean_and_mask(raw, None)
        if cfg.kind in ("video", "spectrogram"):
            valid = diag_inputs.get(f"{cfg.name}_valid")
            mask = valid.bool() if valid is not None else None
            pieces.append(model.diag_tokenizers[cfg.name](cleaned, mask=mask))
        else:
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
    gt_input_in_group: Optional[List[Dict[str, torch.Tensor]]] = None,
    tf_in_group: Optional[List[bool]] = None,
    video_diag_names: Optional[List[str]] = None,
    spectro_diag_names: Optional[List[str]] = None,
    keep_displacement_for_flow: bool = False,
):
    """Returns a function ``chunk_fn(diag_tokens, *prev_pred_list)`` suitable
    for ``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=False``.

    The function runs rollout steps ``[group_start, group_end)`` and returns
    ``(final_diag_tokens, chunk_loss, *last_predictions_flat)``. The
    ``prev_pred_list`` tensors are expected in the order of
    ``diagnostic_names`` and carry the (ctx-role) predictions entering the
    chunk (diag_initial for group 0, last chunk's predictions otherwise).

    Teacher-forcing scheduled sampling
    ----------------------------------
    When ``tf_in_group[i]`` is True for a step ``k = group_start + i``
    with ``k >= 1``, the input ``diag_tokens`` for that step are
    replaced by re-tokenized ground-truth from
    ``gt_input_in_group[i]`` (the GT diagnostic state at step ``k``,
    which is the rollout target of step ``k-1``). The model still
    *predicts* via ``model.backbone`` and the predictions are still
    scored against the same target — TF only affects what flows IN to
    the backbone, not what's scored. The displacement-loss ``ctx``
    follows the actual input: GT under TF, previous-prediction under
    free-rollout. ``gt_input_in_group`` and ``tf_in_group`` are
    optional; default ``None`` reproduces the prior pure free-rollout
    behaviour byte-for-byte.
    """
    use_tf = tf_in_group is not None and gt_input_in_group is not None
    video_set = set(video_diag_names or [])
    spectro_set = set(spectro_diag_names or [])

    def chunk_fn(diag_tokens: torch.Tensor, *prev_pred_tensors: torch.Tensor):
        prev_pred = dict(zip(diagnostic_names, prev_pred_tensors))
        chunk_loss = torch.zeros((), device=diag_tokens.device)
        for i in range(group_end - group_start):
            k = group_start + i

            # Teacher-forcing substitution at the start of step k (k>=1):
            # replace the rollout's input with re-tokenized GT, and use
            # that GT as the displacement-loss ctx (the actual input
            # state that's flowing into the backbone).
            if use_tf and k > 0 and tf_in_group[i]:
                tf_input = gt_input_in_group[i]
                diag_tokens = _tokenize_diag(model, tf_input)
                ctx_dict = tf_input
            else:
                # Free-rollout: ctx is the model's previous prediction
                # (or diag_initial for k=0 of group 0, passed in via
                # ``prev_pred_tensors``).
                ctx_dict = prev_pred

            all_tokens = torch.cat([diag_tokens, act_tokens_in_group[i]], dim=1)
            step_idx = batch_rollout_step + (k + 1)
            time_s = batch_rollout_step.float() * dt_s + (k + 1) * dt_s

            out_tokens = model.backbone(all_tokens, step_idx, time_s)
            diag_tokens = out_tokens[:, :n_diag_tokens]
            predictions, tok_slices = _decode_diag(
                model, diag_tokens, return_slices=True
            )

            # Video heads emit (B, T, C, H, W); permute to (B, C, T, H, W)
            # so loss / metric / rollout-context paths all see the same
            # shape contract that targets and inputs use.
            for name in video_set:
                if name in predictions:
                    predictions[name] = predictions[name].permute(0, 2, 1, 3, 4)

            for cfg in model.diagnostics:
                pred = predictions[cfg.name]
                target = target_in_group[i][cfg.name]
                mask = mask_in_group[i][cfg.name]

                if cfg.name in video_set:
                    # Video: MAE only. Displacement loss is meaningless
                    # for ~900k pixel dims. Patch-grid smoothness was
                    # dropped 2026-06-10 — the zero-init refine_block
                    # in VideoOutputHead is the anti-checkerboard
                    # mechanism. (The pre-existing reference to
                    # video_smoothness_weight here was a latent closure
                    # bug; it lives in rollout_forward_loss_extended's
                    # scope, not _make_chunk_fn's.)
                    mae = masked_mae(pred, target, mask)
                    chunk_loss = chunk_loss + mae_weight * mae
                    continue

                head = model.diag_heads[cfg.name]
                mae = masked_mae(pred, target, mask)
                if isinstance(head, SpectrogramFlowHead):
                    # Generative spectro: pred == μ (train mode); the flow
                    # loss on the residual owns the mode structure (computed
                    # INSIDE this checkpointed chunk so the velocity net is
                    # recomputed in backward → grad-checkpoint correct, and
                    # runs every step → DDP-safe). Replaces the cos+mag
                    # displacement (the failed deterministic mode-fix) unless
                    # explicitly kept for ablation.
                    flow = head.flow_loss(
                        tok_slices[cfg.name], pred, target, mask
                    )
                    step_contrib = mae_weight * mae + head.flow_lambda * flow
                    if keep_displacement_for_flow:
                        ctx = ctx_dict[cfg.name].detach()[
                            ..., : _spectro_trunc_t(cfg)
                        ]
                        cos_loss, mag_loss, _, _, _ = displacement_terms(
                            pred, target, ctx, mask, min_disp_norm
                        )
                        step_contrib = (
                            step_contrib
                            + cos_weight * cos_loss + mag_weight * mag_loss
                        )
                    chunk_loss = chunk_loss + step_contrib
                    continue
                ctx = ctx_dict[cfg.name].detach()
                if cfg.name in spectro_set:
                    # Spec ctx at k=0 of group 0 comes from
                    # diag_initial (full STFT, e.g. 98 frames) while
                    # pred/target are already truncated to trunc_t
                    # (e.g. 96). Mirror Stage 2b's slice so shapes
                    # line up for displacement_terms. At all other
                    # steps the ctx is already trunc_t-sized; the
                    # slice is a no-op.
                    ctx = ctx[..., : _spectro_trunc_t(cfg)]
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


def _patch_grid_smoothness_loss(
    pred: torch.Tensor,
    patch_size: Tuple[int, int, int],
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalise per-pixel discontinuities at video patch-grid boundaries.

    See ``train_e2e_stage2_delta.py`` for the rationale — duplicated here
    to keep the trainer self-contained. Keep in sync.
    """
    T_p, H_p, W_p = patch_size
    loss = pred.new_zeros(())
    n_terms = 0
    if mask is not None:
        pred = pred * mask
    H = pred.shape[-2]
    if H_p > 0 and H > H_p:
        b = torch.arange(H_p, H, H_p, device=pred.device)
        if b.numel() > 0:
            diff = pred[..., b, :] - pred[..., b - 1, :]
            loss = loss + (diff ** 2).mean()
            n_terms += 1
    W = pred.shape[-1]
    if W_p > 0 and W > W_p:
        b = torch.arange(W_p, W, W_p, device=pred.device)
        if b.numel() > 0:
            diff = pred[..., :, b] - pred[..., :, b - 1]
            loss = loss + (diff ** 2).mean()
            n_terms += 1
    T_dim = pred.shape[1]
    if T_p > 0 and T_dim > T_p:
        b = torch.arange(T_p, T_dim, T_p, device=pred.device)
        if b.numel() > 0:
            diff = pred[:, b, ...] - pred[:, b - 1, ...]
            loss = loss + (diff ** 2).mean()
            n_terms += 1
    return loss / max(n_terms, 1)


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
    p_tf: float = 0.0,
    video_diag_names: Optional[List[str]] = None,
    video_n_frames: Optional[Dict[str, int]] = None,
    spectro_diag_names: Optional[List[str]] = None,
    video_smoothness_weight: float = 0.0,
    keep_displacement_for_flow: bool = False,
) -> torch.Tensor:
    """Full-backprop rollout with gradient checkpointing.

    ctx semantics match Stage 2b for k=0 (ground-truth diag_initial) but
    differ at k≥1: here ctx is the *model's* previous prediction, detached.

    Scheduled sampling (teacher-forcing) is enabled when ``p_tf > 0``.
    For each step ``k >= 1``, with probability ``p_tf`` the input
    ``diag_tokens`` is replaced by re-tokenized ground-truth (the
    rollout target of step ``k-1``); displacement-loss ``ctx`` follows
    the actual input. ``p_tf == 0`` (default) reproduces pure
    free-rollout byte-for-byte.

    Multimodal support
    ------------------
    Video and spectrogram diagnostics are listed in ``video_diag_names``
    and ``spectro_diag_names`` respectively. They follow Stage 2b's
    contract: video targets are standardised per-(B, C) using the step-0
    input statistics, video predictions are permuted from
    ``(B, T, C, H, W)`` to ``(B, C, T, H, W)`` after decode, and both
    modalities use plain MAE with a per-batch presence gate (no
    displacement loss). ``video_n_frames`` maps each camera name to its
    per-step frame count (matched to the tokenizer's expected window).
    Empty defaults reproduce TS-only behaviour byte-for-byte.
    """
    video_diag_names = video_diag_names or []
    video_n_frames = video_n_frames or {}
    spectro_diag_names = spectro_diag_names or []
    video_set = set(video_diag_names)
    spectro_set = set(spectro_diag_names)
    video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # Step-0 inputs. Video gets per-(B, C) z-score; per-modality presence
    # scalars are routed through ``f"{name}_valid"`` so the model's
    # tokenize() can substitute the learned ``missing_token`` for absent
    # samples (matches Stage 2b's diag_initial construction).
    diag_initial: Dict[str, torch.Tensor] = {}
    for name in diagnostic_names:
        raw = batch["inputs"][name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if name in video_set:
            cleaned, mu, sd = _video_standardize_per_bc(cleaned)
            video_stats[name] = (mu, sd)
        diag_initial[name] = cleaned
        if name in video_set or name in spectro_set:
            valid_key = f"{name}_valid"
            if valid_key in batch["inputs"]:
                diag_initial[valid_key] = batch["inputs"][valid_key].to(
                    device, non_blocking=True
                )

    # Transfer each modality's full batch target to GPU ONCE, async. The
    # DataLoader returns pinned float32 CPU tensors, so ``.to(device,
    # non_blocking=True)`` truly overlaps H2D with compute. The earlier
    # lazy per-chunk pattern defeated pinning: ``split_target_by_step``
    # calls ``.contiguous()`` after a last-dim slice, which copies into
    # fresh unpinned storage — making the subsequent ``.to(non_blocking)``
    # silently blocking. Transferring the whole per-modality tensor up
    # front, then slicing on GPU, restores true async transfer. Video and
    # spectro targets follow the same upfront-transfer pattern; their
    # per-step splits are 5-D (video) / 4-D (spectro) but the locality is
    # the same.
    target_full: Dict[str, torch.Tensor] = {}
    mask_full: Dict[str, Optional[torch.Tensor]] = {}
    for name in diagnostic_names:
        raw = batch["targets"][name].to(device, non_blocking=True).float()
        cleaned, _ = _clean_and_mask(raw, None)
        if name in video_set:
            mu, sd = video_stats[name]
            target_full[name] = (cleaned - mu) / sd
            mask_full[name] = None      # uses static per-batch gate, not per-step mask
        elif name in spectro_set:
            target_full[name] = cleaned
            mask_full[name] = None
        else:
            target_full[name] = batch["targets"][name].to(
                device, non_blocking=True
            ).float()
            mask_key = f"{name}_mask"
            mask_full[name] = (
                batch["targets"][mask_key].to(device, non_blocking=True).float()
                if mask_key in batch["targets"] else None
            )

    # Per-modality static gates (per-batch, broadcast over all K steps).
    video_gate: Dict[str, torch.Tensor] = {
        n: _video_loss_gate(n, batch, device) for n in video_diag_names
    }
    spectro_gate: Dict[str, torch.Tensor] = {
        n: _spectro_loss_gate(n, batch, device) for n in spectro_diag_names
    }
    cfg_by_name = {c.name: c for c in model.diagnostics}
    spectro_trunc_t_map: Dict[str, int] = {
        n: _spectro_trunc_t(cfg_by_name[n]) for n in spectro_diag_names
    }

    act_full: Dict[str, torch.Tensor] = {
        name: batch["targets"][name].to(device, non_blocking=True).float()
        for name in actuator_names
    }

    # Per-step splits — branching on cfg.kind for video / spectro.
    target_splits: Dict[str, List[torch.Tensor]] = {}
    mask_splits: Dict[str, Optional[List[torch.Tensor]]] = {}
    for name in diagnostic_names:
        if name in video_set:
            target_splits[name] = split_video_target_by_step(
                target_full[name], k_steps, video_n_frames[name]
            )
            mask_splits[name] = None
        elif name in spectro_set:
            target_splits[name] = split_spectro_target_by_step(
                target_full[name], k_steps, spectro_trunc_t_map[name]
            )
            mask_splits[name] = None
        else:
            target_splits[name] = split_target_by_step(
                target_full[name], name, k_steps, chunk_duration_s
            )
            mask_splits[name] = (
                split_target_by_step(
                    mask_full[name], name, k_steps, chunk_duration_s
                )
                if mask_full[name] is not None else None
            )
    act_splits = {
        n: split_target_by_step(act_full[n], n, k_steps, chunk_duration_s)
        for n in actuator_names
    }
    target_per_step: List[Dict[str, torch.Tensor]] = [
        {n: target_splits[n][k] for n in diagnostic_names} for k in range(k_steps)
    ]
    mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
    for k in range(k_steps):
        mk: Dict[str, Optional[torch.Tensor]] = {}
        for n in diagnostic_names:
            if n in video_set:
                mk[n] = video_gate[n]
            elif n in spectro_set:
                mk[n] = spectro_gate[n]
            else:
                mk[n] = (
                    mask_splits[n][k] if mask_splits[n] is not None else None
                )
        mask_per_step.append(mk)
    act_input_per_step: List[Dict[str, torch.Tensor]] = [
        {n: act_splits[n][k] for n in actuator_names} for k in range(k_steps)
    ]

    # Teacher-forcing scheduled sampling. Pre-build the per-step GT
    # diagnostic INPUTS and pre-draw the TF decisions so the gradient-
    # checkpoint backward pass replays the same coin flips.
    #   gt_input_per_step[k] = GT diagnostic state at step k
    #     k = 0:                diag_initial (already NaN-cleaned)
    #     k >= 1:               target_per_step[k - 1] (NaN-cleaned here)
    #   tf_decisions[k] = whether to TF-substitute at step k (ignored at k=0)
    # For video / spectro, ``f"{name}_valid"`` is per-shot and constant
    # across rollout steps, so we replicate it from diag_initial at every
    # k≥1 entry; the model's tokenize() reads it the same way as at k=0.
    gt_input_per_step: Optional[List[Dict[str, torch.Tensor]]]
    tf_decisions: Optional[List[bool]]
    if p_tf > 0.0:
        gt_input_per_step = [diag_initial]
        valid_keys_to_carry = [
            f"{n}_valid"
            for n in (video_diag_names + spectro_diag_names)
            if f"{n}_valid" in diag_initial
        ]
        for k in range(1, k_steps):
            cleaned_at_k: Dict[str, torch.Tensor] = {}
            for name in diagnostic_names:
                cleaned_t, _ = _clean_and_mask(target_per_step[k - 1][name], None)
                cleaned_at_k[name] = cleaned_t
            for vk in valid_keys_to_carry:
                cleaned_at_k[vk] = diag_initial[vk]
            gt_input_per_step.append(cleaned_at_k)
        tf_decisions = [False]  # k=0 placeholder; never read
        for _ in range(1, k_steps):
            tf_decisions.append(
                bool(torch.rand((), device=device).item() < p_tf)
            )
    else:
        gt_input_per_step = None
        tf_decisions = None

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
            gt_input_in_group=(
                gt_input_per_step[group_start:group_end]
                if gt_input_per_step is not None
                else None
            ),
            tf_in_group=(
                tf_decisions[group_start:group_end]
                if tf_decisions is not None
                else None
            ),
            video_diag_names=video_diag_names,
            spectro_diag_names=spectro_diag_names,
            keep_displacement_for_flow=keep_displacement_for_flow,
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
    video_diag_names: Optional[List[str]] = None,
    video_n_frames: Optional[Dict[str, int]] = None,
    spectro_diag_names: Optional[List[str]] = None,
    step: int = 0,
    ckpt_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Full K_max rollout, no checkpointing; return per-step per-modality
    ``{model_mae, copy_mae, dir_cos, mag_ratio}``. Context at k=0 is
    ``diag_initial``; at k≥1 it's the model's own prediction from step k-1
    (matching training-time semantics).

    For video and spectrogram diagnostics, ``dir_cos`` and ``mag_ratio``
    are reported as ``NaN`` — only ``model_mae`` and ``copy_mae`` are
    meaningful (matches Stage 2b's validate convention).
    """
    video_diag_names = video_diag_names or []
    video_n_frames = video_n_frames or {}
    spectro_diag_names = spectro_diag_names or []
    video_set = set(video_diag_names)
    spectro_set = set(spectro_diag_names)
    cfg_by_name = {c.name: c for c in model.diagnostics}
    spectro_trunc_t_map: Dict[str, int] = {
        n: _spectro_trunc_t(cfg_by_name[n]) for n in spectro_diag_names
    }
    model.eval()
    keys = ("model_mae", "copy_mae", "dir_cos", "mag_ratio",
            "pred_var", "gt_var")
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
        # Step-0 inputs (with video standardisation + per-modality validity)
        diag_initial: Dict[str, torch.Tensor] = {}
        video_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in diagnostic_names:
            raw = batch["inputs"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            if name in video_set:
                cleaned, mu, sd = _video_standardize_per_bc(cleaned)
                video_stats[name] = (mu, sd)
            diag_initial[name] = cleaned
            if name in video_set or name in spectro_set:
                valid_key = f"{name}_valid"
                if valid_key in batch["inputs"]:
                    diag_initial[valid_key] = batch["inputs"][valid_key].to(device)

        # Per-modality static gates for video / spectrogram.
        video_gate: Dict[str, torch.Tensor] = {
            n: _video_loss_gate(n, batch, device) for n in video_diag_names
        }
        spectro_gate: Dict[str, torch.Tensor] = {
            n: _spectro_loss_gate(n, batch, device) for n in spectro_diag_names
        }

        # Per-step targets / masks / actuators (branch on cfg.kind)
        act_per_step: List[Dict[str, torch.Tensor]] = []
        target_per_step: List[Dict[str, torch.Tensor]] = []
        mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
        # Pre-split video / spectro full targets once.
        video_target_full: Dict[str, torch.Tensor] = {}
        for name in video_diag_names:
            raw = batch["targets"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            mu, sd = video_stats[name]
            video_target_full[name] = (cleaned - mu) / sd
        spectro_target_full: Dict[str, torch.Tensor] = {}
        for name in spectro_diag_names:
            raw = batch["targets"][name].to(device).float()
            cleaned, _ = _clean_and_mask(raw, None)
            spectro_target_full[name] = cleaned
        video_splits: Dict[str, List[torch.Tensor]] = {
            n: split_video_target_by_step(
                video_target_full[n], K_max, video_n_frames[n]
            )
            for n in video_diag_names
        }
        spectro_splits: Dict[str, List[torch.Tensor]] = {
            n: split_spectro_target_by_step(
                spectro_target_full[n], K_max, spectro_trunc_t_map[n]
            )
            for n in spectro_diag_names
        }
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
                if name in video_set:
                    tk[name] = video_splits[name][k]
                    mk[name] = video_gate[name]
                    continue
                if name in spectro_set:
                    tk[name] = spectro_splits[name][k]
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

        result = rollout(diag_initial, act_per_step, collect_history=False)
        # Permute video predictions to (B, C, T, H, W) so the loss path
        # matches the target shape contract.
        for k in range(K_max):
            for name in video_set:
                if name in result.predictions[k]:
                    result.predictions[k][name] = (
                        result.predictions[k][name].permute(0, 2, 1, 3, 4)
                    )

        for k in range(K_max):
            for name in diagnostic_names:
                pred = result.predictions[k][name].float()
                target = target_per_step[k][name]
                mask = mask_per_step[k][name]
                if name in video_set:
                    # Video: MAE only — no displacement metrics.
                    mae = masked_mae(pred, target, mask).item()
                    copy_mae = masked_mae(
                        diag_initial[name], target, mask
                    ).item()
                    sums[k][name]["model_mae"] += mae
                    sums[k][name]["copy_mae"] += copy_mae
                    counts[k][name]["mae"] += 1
                    continue
                # Spectrogram diag_initial holds the full STFT output
                # (e.g. 98 frames) while pred/target are trunc_t
                # (e.g. 96). Slice diag_initial for both the copy
                # baseline and the k=0 displacement ctx so shapes
                # match. Slow_ts has no truncation.
                if name in spectro_set:
                    baseline_input = diag_initial[name][
                        ..., : spectro_trunc_t_map[name]
                    ]
                else:
                    baseline_input = diag_initial[name]
                # Teacher-forced ctx for metrics (consistency with
                # Stage 2b val and the §5.9 gate tests).
                ctx = (
                    baseline_input if k == 0 else target_per_step[k - 1][name]
                )
                mae = masked_mae(pred, target, mask).item()
                copy_mae = masked_mae(baseline_input, target, mask).item()
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
                # Temporal-variance ratio (collapse diagnostic) for spectro:
                # var over time, summed over valid bins; pred/gt share the
                # mask so the count cancels in the ratio.
                if name in spectro_set:
                    pf = pred.float()
                    mv = (
                        (mask[..., 0] > 0).float() if mask is not None
                        else torch.ones(pf.shape[:3], device=pf.device)
                    )
                    sums[k][name]["pred_var"] += float(
                        (pf.var(dim=-1) * mv).sum()
                    )
                    sums[k][name]["gt_var"] += float(
                        (target.float().var(dim=-1) * mv).sum()
                    )
            # Free this step's resident GPU tensors before moving on. The
            # ctx at step k+1 is target_per_step[k], so we keep the current
            # step's target; the previous step's target is safe to drop.
            result.predictions[k] = None  # type: ignore[index]
            act_per_step[k] = None  # type: ignore[index]
            mask_per_step[k] = None  # type: ignore[index]
            if k > 0:
                target_per_step[k - 1] = None  # type: ignore[index]
    model.train()

    # Aggregate metrics across DDP ranks. Tier 4 approach (ported from
    # train_e2e_stage2_delta.py 2026-06-03): NO DDP collectives inside
    # validate(). Each rank writes its per-rank sums/counts to a small
    # .pt file in ckpt_dir; rank 0 polls for those files (with a 5-min
    # deadline for stragglers), merges available ones into its own
    # sums/counts, and cleans up. Non-rank-0 ranks return with their
    # rank-LOCAL metrics — fine because only rank 0 logs val_loss and
    # saves best.pt.
    #
    # Earlier collective-based attempts (all_reduce, monitored_barrier
    # + gloo sync) hung because val pipeline rank skew can exceed NCCL's
    # 10-min watchdog (cold val NFS reads). File-based merge + polling
    # tolerates arbitrary skew up to the deadline.
    import torch.distributed as dist
    if (dist.is_available() and dist.is_initialized()
            and ckpt_dir is not None):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        ckpt_dir_p = Path(ckpt_dir)
        my_file = ckpt_dir_p / f"_val_metrics_step{step}_rank{rank:03d}.pt"
        # Atomic write: write to .tmp then rename so rank 0 never sees a
        # half-written file.
        tmp_file = my_file.with_suffix(".pt.tmp")
        try:
            torch.save({"sums": sums, "counts": counts}, tmp_file)
            tmp_file.rename(my_file)
        except Exception as e:
            logger.warning(
                f"[rank{rank}] Tier 4 val metrics write failed: {e!r}"
            )
        if rank == 0:
            import time as _time
            deadline = _time.time() + 300.0  # 5 min for stragglers
            collected = {0}  # already have own sums/counts in memory
            while _time.time() < deadline and len(collected) < world_size:
                for r in range(world_size):
                    if r in collected:
                        continue
                    target = ckpt_dir_p / f"_val_metrics_step{step}_rank{r:03d}.pt"
                    if not target.exists():
                        continue
                    try:
                        other = torch.load(target, weights_only=False)
                    except Exception as e:
                        # Partial file? Will retry on next loop iteration.
                        logger.warning(
                            f"[rank0] failed to load rank {r} metrics "
                            f"(may be partial): {e!r}"
                        )
                        continue
                    for k in range(K_max):
                        for n in diagnostic_names:
                            for m in keys:
                                sums[k][n][m] += other["sums"][k][n][m]
                            for m in ("mae", "disp"):
                                counts[k][n][m] += other["counts"][k][n][m]
                    collected.add(r)
                if len(collected) < world_size:
                    _time.sleep(1.0)
            if len(collected) == world_size:
                logger.info(
                    f"[rank0] val merge: collected all {world_size} "
                    f"rank files at step {step}"
                )
            else:
                missing = [r for r in range(world_size) if r not in collected]
                logger.warning(
                    f"[rank0] val merge: collected {len(collected)}/"
                    f"{world_size} rank files at step {step} (5-min "
                    f"deadline hit). Missing ranks: {missing}. "
                    "val_loss reflects partial sample."
                )
            # Cleanup: remove per-rank files for this step.
            for r in range(world_size):
                target = ckpt_dir_p / f"_val_metrics_step{step}_rank{r:03d}.pt"
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(
                        f"[rank0] failed to unlink {target}: {e!r}"
                    )

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in range(K_max):
        out[k] = {}
        for name in diagnostic_names:
            mae_n = max(counts[k][name]["mae"], 1)
            disp_n = max(counts[k][name]["disp"], 1)
            gt_var = sums[k][name]["gt_var"]
            out[k][name] = {
                "model_mae": sums[k][name]["model_mae"] / mae_n,
                "copy_mae": sums[k][name]["copy_mae"] / mae_n,
                "dir_cos": sums[k][name]["dir_cos"] / disp_n
                if counts[k][name]["disp"] else float("nan"),
                "mag_ratio": sums[k][name]["mag_ratio"] / disp_n
                if counts[k][name]["disp"] else float("nan"),
                "tvr": (sums[k][name]["pred_var"] / gt_var
                        if gt_var > 1e-8 else float("nan")),
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
    parser.add_argument(
        "--backbone_grad_checkpoint", action="store_true",
        help="Per-block gradient checkpointing in the shared backbone. "
        "Required at d_model=1024+ where activations don't fit per-GCD VRAM.",
    )
    parser.add_argument(
        "--lengths_cache_dir",
        type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model_meta"),
        help="Directory for TokamakMultiFileDataset length-cache sidecar "
        "files (lengths_e2e_stage2_ext_{train,val}.pt) and the "
        "video-presence cache (video_present_{train,val}.pt). Defaults "
        "to the shared meta dir where the pre-built caches live — "
        "critical at d=1024 + N>=8: a cold-start scan of 7878 train files "
        "takes ~30 min, longer than the NCCL collective timeout (10 min). "
        "Mirrors --lengths_cache_dir in train_e2e_stage2_delta.py.",
    )

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
        "--video_smoothness_weight",
        type=float, default=0.0,
        help="Weight on the patch-grid smoothness loss for video predictions. "
        "Suppresses the 12×12 checkerboard baked into Stage 2's "
        "representation by the autoregressive round-trip through "
        "ConvTranspose3d(kernel=stride=patch_size). 0.0 disables; "
        "0.1 is a reasonable starting point. Applied per video modality "
        "with the same loss gate as the per-step MAE.",
    )
    parser.add_argument(
        "--no_displacement_loss", action="store_true",
        help="Disable the cos+log-mag displacement terms (MAE only).",
    )
    # ── Generative-head / checkerboard-fix flags (2026-06-21) ──
    parser.add_argument("--video_resize_conv", action="store_true",
                        help="Resize-conv video decoder (kills the AR "
                             "checkerboard); supersedes seam-refine. "
                             "From-scratch / fresh-head only.")
    parser.add_argument("--video_resize_conv_hidden", type=int, default=64)
    parser.add_argument("--spec_generative", action="store_true",
                        help="Generative SpectrogramFlowHead (rectified flow "
                             "over a deterministic mean).")
    parser.add_argument("--spec_flow_base_ch", type=int, default=64)
    parser.add_argument("--spec_flow_steps", type=int, default=6)
    parser.add_argument("--spec_flow_lambda", type=float, default=1.0)
    parser.add_argument("--spec_gen_keep_displacement", action="store_true",
                        help="Keep cos+mag displacement on a generative spectro "
                             "head's mean (default off — the flow loss owns "
                             "mode structure). Ablation knob.")
    parser.add_argument("--collapse_aware_best", action="store_true",
                        help="Penalise low TVR (collapse) in best.pt selection.")
    parser.add_argument("--collapse_aware_lambda", type=float, default=1.0)

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
    parser.add_argument(
        "--val_batch_size", type=int, default=None,
        help="Batch size for the val_loader. Defaults to --batch_size; "
        "set lower (typically 1) when running with large K_max where "
        "the val targets carry K_max × chunk_duration_s seconds of data "
        "per sample. At d=1024 + K=80 every sample's targets ≈ 500 MB, "
        "so val_batch_size=1 keeps the val-transition host RAM budget "
        "well under the 502 GB node ceiling.",
    )
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
    parser.add_argument(
        "--tf_anneal_steps", type=int, default=0,
        help="Scheduled-sampling teacher-forcing schedule. "
        "If > 0: at training step ``step``, "
        "p_tf = max(0, 1 - step / tf_anneal_steps); at each rollout "
        "step k>=1 we replace the input with re-tokenized GT with "
        "probability p_tf. Default 0 disables TF entirely (pure "
        "free-rollout, byte-identical to the un-augmented trainer). "
        "Validation always uses pure free-rollout regardless of this "
        "flag.",
    )

    # Multimodal additions — empty defaults reproduce TS-only Extended
    # Stage 2 behaviour byte-for-byte (G2/G3 fixtures cover this).
    parser.add_argument(
        "--use_video", nargs="*", default=[],
        choices=[entry[0] for entry in VIDEO_MODALITIES],
        help="Camera names to include as video diagnostics. Empty (default) "
        "skips all video paths. Mirrors Stage 2b / Stage 1.",
    )
    parser.add_argument(
        "--use_spectro", nargs="*", default=[],
        choices=[entry[0] for entry in SPECTROGRAM_MODALITIES],
        help="Spectrogram modality names. Empty (default) skips all "
        "spectro paths. Mirrors Stage 2b / Stage 1.",
    )
    parser.add_argument(
        "--spectro_patch_f", type=int, default=None,
        help="Override spectro freq-patch size for all spectro modalities "
        "(default: registry). 512 = full-frequency patch. Must match the "
        "delta init checkpoint's patch shape.",
    )
    parser.add_argument(
        "--spectro_patch_t", type=int, default=None,
        help="Override spectro time-patch size (default: registry). Must "
        "match the delta init checkpoint's patch shape.",
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

    # Video-presence filter: when --use_video is set, retain only shot
    # files where every requested camera's HDF5 group exists. Mirrors
    # Stage 2b's filter call. Cache lives in --lengths_cache_dir (the
    # shared meta dir), NOT the per-run checkpoint dir, so a cold-start
    # smoke / production launch finds the pre-built file and skips the
    # ~30 min full-dataset scan (longer than NCCL's 10 min timeout).
    if args.use_video:
        if dm.is_main:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            args.lengths_cache_dir.mkdir(parents=True, exist_ok=True)
        dm.barrier()
        train_before, val_before = len(train_files), len(val_files)
        train_files = filter_video_present_files(
            train_files, args.use_video,
            cache_path=args.lengths_cache_dir / "video_present_train.pt",
        )
        val_files = filter_video_present_files(
            val_files, args.use_video,
            cache_path=args.lengths_cache_dir / "video_present_val.pt",
        )
        logger.info(
            f"Video-presence filter ({args.use_video}): "
            f"train {train_before} → {len(train_files)}, "
            f"val {val_before} → {len(val_files)}"
        )
        if not train_files or not val_files:
            raise SystemExit(
                f"No files remaining after --use_video filter for "
                f"{args.use_video}; check that the requested cameras' "
                f"HDF5 groups exist in the data dir."
            )

    stats = torch.load(args.stats_path, weights_only=False)

    diagnostics, actuators = build_configs(
        args.chunk_duration_s,
        use_video=args.use_video,
        use_spectro=args.use_spectro,
        spectro_patch_f=args.spectro_patch_f,
        spectro_patch_t=args.spectro_patch_t,
    )
    diagnostic_names = [c.name for c in diagnostics]
    actuator_names = [c.name for c in actuators]
    video_diag_names: List[str] = list(args.use_video)
    spectro_diag_names: List[str] = list(args.use_spectro)
    video_n_frames: Dict[str, int] = {
        c.name: int(c.window_samples)
        for c in diagnostics
        if c.kind == "video"
    }
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

    # Stage 2 enables the VideoOutputHead + SpectrogramOutputHead
    # seam-refine blocks — fights the patch-grid checkerboard that
    # the autoregressive K-step rollout amplifies.
    video_seam_refine_flag = True
    spectro_seam_refine_flag = True
    logger.info(
        f"Seam-refine: video_seam_refine={video_seam_refine_flag} "
        f"spectro_seam_refine={spectro_seam_refine_flag}"
    )
    model = E2EFoundationModel(
        diagnostics=diagnostics, actuators=actuators,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
        backbone_grad_checkpoint=args.backbone_grad_checkpoint,
        video_seam_refine=video_seam_refine_flag,
        spectro_seam_refine=spectro_seam_refine_flag,
        video_resize_conv=args.video_resize_conv,
        video_resize_conv_hidden=args.video_resize_conv_hidden,
        spectro_generative=args.spec_generative,
        spectro_flow_base_ch=args.spec_flow_base_ch,
        spectro_flow_sample_steps=args.spec_flow_steps,
        spectro_flow_lambda=args.spec_flow_lambda,
    ).to(device)
    # Confirm the head modules actually built the refine_block.
    for name in spectro_diag_names:
        head = model.diag_heads[name]
        logger.info(
            f"  spectro head '{name}': enable_seam_refine="
            f"{getattr(head, 'enable_seam_refine', 'MISSING')}, "
            f"has refine_block={hasattr(head, 'refine_block')}"
        )

    if args.init_checkpoint is not None:
        ckpt = torch.load(
            args.init_checkpoint, weights_only=False, map_location=device
        )
        # Allowed-missing prefixes cover the freshly-initialised
        # spectrogram and video modules so that warm-starting from a
        # TS-only Phase A / Stage 2b checkpoint succeeds. Unknown extra
        # keys still raise. When --use_video / --use_spectro are empty
        # (TS-only Extended), the prefix tuple is empty and the load is
        # strict — byte-identical to the pre-multimodal contract.
        allowed_init_prefixes: Tuple[str, ...] = tuple(
            f"diag_{kind}.{n}."
            for kind in ("tokenizers", "heads")
            for n in (*args.use_video, *args.use_spectro)
        )
        load_state_dict_explicit(
            model,
            ckpt["model_state_dict"],
            allowed_missing_prefixes=allowed_init_prefixes,
        )
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
        def __init__(self, base, keep_displacement_for_flow=False):
            super().__init__()
            self.model = base
            self.keep_displacement_for_flow = keep_displacement_for_flow
        def forward(
            self, batch, k_steps, chunk_duration_s, mae_weight, cos_weight,
            mag_weight, min_disp_norm, use_displacement_loss, grad_checkpoint_every,
            p_tf, video_smoothness_weight,
        ):
            return rollout_forward_loss_extended(
                self.model, batch, diagnostic_names, actuator_names,
                k_steps=k_steps, chunk_duration_s=chunk_duration_s,
                device=device, mae_weight=mae_weight, cos_weight=cos_weight,
                mag_weight=mag_weight, min_disp_norm=min_disp_norm,
                use_displacement_loss=use_displacement_loss,
                grad_checkpoint_every=grad_checkpoint_every,
                p_tf=p_tf,
                video_diag_names=video_diag_names,
                video_n_frames=video_n_frames,
                spectro_diag_names=spectro_diag_names,
                video_smoothness_weight=video_smoothness_weight,
                keep_displacement_for_flow=self.keep_displacement_for_flow,
            )

    train_step_module: torch.nn.Module = _TrainStepModule(
        model, keep_displacement_for_flow=args.spec_gen_keep_displacement
    )
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
        lengths_cache_path=args.lengths_cache_dir / "lengths_e2e_stage2_ext_train.pt",
        **shared,
    )
    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path=args.lengths_cache_dir / "lengths_e2e_stage2_ext_val.pt",
        **shared,
    )
    logger.info(
        f"Chunks — train: {len(train_ds)}  val: {len(val_ds)}  "
        f"prediction_horizon_s={prediction_horizon_s:.3f}"
    )
    # Generative spectro heads: set per-(channel, freq) residual std from the
    # per-bin stats (persisted in the checkpoint → eval reconstructs it).
    # Falls back to ones (no standardisation) if log_per_bin is unavailable.
    if args.spec_generative:
        _sigma_map = build_spec_per_bin_sigma(
            stats, model.diagnostics, train_ds.signal_configs,
        )
        if not _sigma_map:
            logger.warning(
                "--spec_generative: stats lack 'log_per_bin' — flow heads "
                "use unit residual std (no per-bin scaling)."
            )
        for _n in spectro_diag_names:
            _h = model.diag_heads[_n]
            if isinstance(_h, SpectrogramFlowHead) and _n in _sigma_map:
                _h.set_sigma_pb(_sigma_map[_n].to(device))
                logger.info(
                    f"Flow head [{_n}]: per-bin residual std set, "
                    f"shape {tuple(_sigma_map[_n].shape)}"
                )
    # num_workers cap — mirrors train_e2e_stage2_delta.py. Past Stage 1
    # chain (4581026/27/28) OOM'd at ~9h45m / ~5850 steps with >4 workers
    # due to slow per-worker leak (h5py metadata + PyTorch caches).
    if args.num_workers > 4:
        logger.warning(
            f"Capping --num_workers {args.num_workers} → 4 (OOM mitigation; "
            "see persistent_workers comment in train_e2e_stage2_delta.py)."
        )
        args.num_workers = 4
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        # TwoLevelSampler: shuffle file order per epoch, sequential
        # within each file. Keeps the per-worker LRU file-handle
        # cache (max_open_files=100) nearly always hitting.
        # RandomSampler across 7878 files gave ~1% hit rate and
        # spent ~10% of worker time on HDF5 file opens (observed
        # via py-spy on Stage 1 job 2719669).
        # DistributedTwoLevelSampler is the DDP-aware sibling: each
        # rank owns a fixed slice of the file list and iterates its
        # own files front-to-back, so the per-worker LRU stays warm
        # across epochs. PyTorch's DistributedSampler shards chunk
        # indices instead and was observed to push step time from
        # ~1 s to ~12 s under 2-GPU DDP on Stage 1.
        sampler=(
            DistributedTwoLevelSampler(
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
        # prefetch_factor=1 at K=80 — delta runs 3 at K=10, but at d=1024
        # + K=80 each batch's targets carry 4 s of all-modality data
        # (~500 MB / batch). With 4 workers × prefetch_factor × 8 ranks
        # = 8 × prefetch buffered batches per node, prefetch=2 → 32 GB
        # / node of persistent host-RAM occupancy that eats the headroom
        # val needs during its worker spawn. Job 4757844 confirmed the
        # 14 GB shortfall. prefetch=1 → 16 GB / node, freeing room.
        prefetch_factor=1 if args.num_workers > 0 else None,
        pin_memory=device.type == "cuda",
        # persistent_workers=False — workers torn down per epoch to
        # release the slow leak (h5py metadata + PyTorch caches) that
        # OOM'd Stage 1 chained jobs at ~9h45m. Mirrors delta.
        persistent_workers=False,
    )
    # Val loader is smaller and isolated from train workers to keep
    # the combined in-flight footprint under the 502 GB node budget
    # during val transitions. Without this delta hit OOM at 97 % host
    # RAM on smokes when val workers spun up alongside the train pool
    # — same failure mode the extended smoke (job 4757314) just hit.
    val_num_workers = min(4, args.num_workers)
    val_batch_size = args.val_batch_size if args.val_batch_size is not None \
        else args.batch_size
    logger.info(
        f"DataLoader — train: batch={args.batch_size} workers={args.num_workers} "
        f"prefetch=2 persistent=False | val: batch={val_batch_size} "
        f"workers={val_num_workers} prefetch=1 persistent=False"
    )
    # Val sampler: under DDP, shard windows across ranks with a
    # shuffled order so each rank sees different shots. Without this,
    # every rank iterates val_ds in the same order and the first
    # val_max_batches windows on every rank come from the same
    # ~1-2 files. With ~62% of shots carrying stub (C, 1) placeholders
    # for BES (~45% for CO2), those clusters can leave val_loss for
    # those modalities reading exactly 0 across the entire 64-rank
    # aggregate (observed ext val 1: co2/bes = 0.0 at every k).
    # DistributedSampler-style strided sharding + shuffle=True is OK
    # here because val runs ~60 batches per call, not the hot training
    # loop where the per-step overhead was measured to be costly.
    from torch.utils.data import DistributedSampler
    val_sampler = (
        DistributedSampler(
            val_ds, num_replicas=dm.world_size, rank=dm.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        if dm.distributed else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch_size,
        shuffle=False,  # sampler handles ordering when distributed
        sampler=val_sampler,
        num_workers=val_num_workers, collate_fn=collate_fn, drop_last=True,
        prefetch_factor=1 if val_num_workers > 0 else None,
        # pin_memory=False for val: each iter() call re-creates the main
        # process's pin_memory thread + internal queues, and those pinned
        # allocations ratchet host RSS upward across validations (observed
        # +127 GB on val 1, +27 GB on val 2 with persistent_workers=True,
        # OOM on val 2 at batch=256). Val is 1–20 batches per call so the
        # synchronous H2D cost is negligible.
        pin_memory=False,
        persistent_workers=False,
    )

    # Pre-warm val NFS cache. The Stage 2 val 1 has been the recurring
    # failure point: val files are mostly disjoint from train, so when
    # the first val event fires after ~10h of training, a subset of
    # ranks hit cold NFS reads and lag the rest. monitored_barrier's
    # 2-min window can't tolerate this, and downstream collectives hang
    # on rank skew. By having each rank's main process briefly open
    # every val file at startup, the per-node NFS metadata+page cache
    # is warm before val 1 fires. ~1-2 min added to startup; eliminates
    # the cold-val-cache hazard. Mirrors delta.
    import time as _time
    import h5py as _h5py
    if dm.distributed:
        n_val_files = len(val_ds.hdf5_paths)
        if dm.is_main:
            logger.info(f"Pre-warming val NFS cache ({n_val_files} files)...")
        t0 = _time.time()
        for path in val_ds.hdf5_paths:
            try:
                with _h5py.File(path, "r"):
                    pass  # open + close warms the per-node NFS cache
            except Exception:
                pass  # skip unreadable files; dataset handles them
        dm.barrier()  # all ranks finish warming before training starts
        if dm.is_main:
            logger.info(
                f"  val NFS cache pre-warm: {n_val_files} files in "
                f"{_time.time()-t0:.1f}s"
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
        # Strict resume: a *_latest.pt was written by THIS run with the
        # same multimodal config; spectro/video keys must already be
        # present. allowed_missing_prefixes catches accidental TS-key
        # renames the same way as in the pre-multimodal contract — and
        # also permits VideoOutputHead + SpectrogramOutputHead
        # refine_block keys (zero-initialised, so missing = bit-identical
        # output to the pre-patch model).
        resume_allowed_missing = tuple(
            f"diag_heads.{n}.refine_block."
            for n in (list(args.use_video) + list(args.use_spectro))
        )
        load_state_dict_explicit(
            model,
            resume_ckpt["model_state_dict"],
            allowed_missing_prefixes=resume_allowed_missing,
        )
        if "optimizer_state_dict" in resume_ckpt:
            # Param-count guard — mirrors train_e2e_stage2_delta.py.
            # Falls back to fresh AdamW state when the model has more
            # params than the saved optimizer (e.g. crossing the
            # 2026-06-08 VideoOutputHead refine_block addition).
            saved_n = sum(
                len(g["params"])
                for g in resume_ckpt["optimizer_state_dict"]["param_groups"]
            )
            cur_n = sum(len(g["params"]) for g in opt.param_groups)
            if saved_n != cur_n:
                logger.warning(
                    f"Skipping optimizer state restore: saved had "
                    f"{saved_n} params, model has {cur_n}. AdamW "
                    f"starts fresh — momentum re-accumulates over the "
                    f"first few hundred steps. Loss may wobble briefly."
                )
            else:
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

        # Scheduled-sampling teacher-forcing probability. Linear ramp
        # from 1.0 (full TF) at step 0 to 0.0 (pure free-rollout) at
        # step ``args.tf_anneal_steps``. After anneal, p_tf stays at 0.
        # ``args.tf_anneal_steps == 0`` disables TF entirely (default
        # behaviour, byte-identical to the un-augmented trainer).
        if args.tf_anneal_steps > 0:
            p_tf = max(0.0, 1.0 - step / args.tf_anneal_steps)
        else:
            p_tf = 0.0

        opt.zero_grad()
        with amp_ctx_factory():
            loss = train_step_module(
                batch, K, args.chunk_duration_s,
                args.mae_weight, args.cos_weight, args.mag_weight,
                args.min_disp_norm, use_disp, args.grad_checkpoint_every,
                p_tf, args.video_smoothness_weight,
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
            tf_str = (
                f"  p_tf={p_tf:.3f}" if args.tf_anneal_steps > 0 else ""
            )
            logger.info(
                f"step {step}/{args.max_steps}  K={K}  loss={avg:.4f}  "
                f"lr={lr_now:.2e}{tf_str}"
            )
            running = 0.0
            running_count = 0

        if step % args.val_every == 0 or step == args.max_steps:
            # Reclaim memory before val transition. At K=80 each batch's
            # targets carry 4 s of all-modality data (~500 MB / batch),
            # so the train DataLoader's worker prefetch queues plus
            # current batch hold ~16-50 GB of host RAM per node. Val
            # workers spawning on top of that breaches the 502 GB ceiling
            # (job 4757844 hit 97 %). Tearing down the train iter with
            # `persistent_workers=False` sends SIGTERM to the workers,
            # which release their queues; gc.collect() + empty_cache()
            # finish the cleanup. Cost: workers respawn after val, so
            # the per-worker HDF5 LRU cache is lost (~10-20 s cold-cache
            # penalty + ~5-10 s respawn). At val_every=2500 over a
            # 20k-step chain that's <0.03 % overhead — negligible.
            import gc as _gc
            del train_iter
            del batch
            _gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if dm.distributed:
                dm.barrier()

            # Pass bare model — validate constructs its own rollout from it.
            metrics = validate(
                model, val_loader, device,
                diagnostic_names, actuator_names,
                chunk_duration_s=args.chunk_duration_s,
                K_max=K_max,
                min_disp_norm=args.min_disp_norm,
                max_batches=args.val_max_batches,
                video_diag_names=video_diag_names,
                video_n_frames=video_n_frames,
                spectro_diag_names=spectro_diag_names,
                step=step,
                ckpt_dir=args.checkpoint_dir,
            )

            # Respawn train workers from scratch. Next `next(train_iter)`
            # call in the outer loop will block briefly while file
            # handles re-open + first prefetch lands.
            train_iter = iter(train_loader)
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

            # Collapse-aware selection: penalise spectro modalities whose TVR
            # is below 1 (mean-collapsed) so a low-MAE collapse can't win
            # best.pt. Default off → plain sum(MAE).
            sel_loss = val_loss
            if args.collapse_aware_best:
                tvr_pen = sum(
                    max(0.0, 1.0 - metrics[k][name]["tvr"])
                    for k in range(K_max)
                    for name in diagnostic_names
                    if not math.isnan(metrics[k][name].get("tvr", float("nan")))
                )
                sel_loss = val_loss + args.collapse_aware_lambda * tvr_pen
                logger.info(
                    f"  [collapse-aware] sel_loss={sel_loss:.4f} "
                    f"(tvr_penalty={tvr_pen:.4f}, "
                    f"lambda={args.collapse_aware_lambda})"
                )
            is_new_best = sel_loss < best_val_loss
            if is_new_best:
                best_val_loss = sel_loss
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