"""§5.9 trained-rollout tests — cluster-submission gate for Stages 2 and 3.

Run offline against a trained E2E checkpoint via env vars::

    E2E_STAGE_CHECKPOINT=/path/to/best.pt \
    E2E_DATA_DIR=/scratch/gpfs/EKOLEMEN/foundation_model \
    E2E_STATS_PATH=/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
    pixi run pytest tests/e2e/test_rollout_trained.py -v

All tests skip when ``E2E_STAGE_CHECKPOINT`` is unset, so the main per-commit
suite is unaffected. Tests 1 and 3 additionally require ``E2E_DATA_DIR`` and
``E2E_STATS_PATH`` for ground-truth trajectories; tests 2 and 4 work on
synthetic in-distribution inputs.

Runtime budget per ``ResearchPlan.MD`` §5.10: < 10 min total.

References:
  - Test 1 (copy baseline win rate):     ResearchPlan.MD §5.9 bullet 4
  - Test 2 (no fixed-point):             ResearchPlan.MD §5.9 bullet 5
  - Test 3 (model vs gt cos_sim gap):    ResearchPlan.MD §5.9 bullet 6
                                         (also Phase A milestone A3, §6.1)
  - Test 4 (actuator sensitivity):       ResearchPlan.MD §5.9 bullet 7
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.e2e.lora import apply_lora_to_backbone
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)
from tokamak_foundation_model.e2e.rollout import RolloutResult, TokenSpaceRollout

CHECKPOINT_ENV = "E2E_STAGE_CHECKPOINT"
DATA_DIR_ENV = "E2E_DATA_DIR"
STATS_PATH_ENV = "E2E_STATS_PATH"
K_ROLLOUT_ENV = "E2E_K_ROLLOUT"

# Parameterised rollout horizon. Default 10 matches Stage 2's K_max; set
# ``E2E_K_ROLLOUT=80`` for the Stage 3 gate. Thresholds in the tests below
# scale with K_ROLLOUT: stricter for shorter rollouts.
K_ROLLOUT = int(os.environ.get(K_ROLLOUT_ENV, "10"))
VAL_BATCH = 32


def _cos_sim_gap_threshold() -> float:
    """Tolerance for ``|model_cos_sim − gt_cos_sim|`` (§5.9 test 3).

    0.05 matches the Phase A A3 milestone for short (K=10) rollouts; the
    plan relaxes to 0.10 at the A4 milestone (K=80).
    """
    return 0.05 if K_ROLLOUT <= 10 else 0.10


def _copy_win_last_step_threshold() -> float:
    """Copy-baseline win-rate threshold at the last rollout step (§5.9 test 1).

    60 % at K=10 (plan §5.9 copy-baseline test); relaxed to 50 % for K>10
    since late-step prediction is intrinsically harder.
    """
    return 0.60 if K_ROLLOUT <= 10 else 0.50


def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name)
    return Path(v) if v else None


# Gate the whole module on a checkpoint being available. Lets the main suite
# pass when run without one, so this file is safe to leave in `tests/e2e/`.
pytestmark = pytest.mark.skipif(
    _env_path(CHECKPOINT_ENV) is None
    or not _env_path(CHECKPOINT_ENV).exists(),  # type: ignore[union-attr]
    reason=(
        f"Set ${CHECKPOINT_ENV}=/path/to/best.pt to run trained-rollout tests. "
        "These tests are the cluster-submission gate for Stages 2/3 and run "
        "offline against a trained checkpoint only."
    ),
)


# ── Helpers ──────────────────────────────────────────────────────────


def _nanclean(t: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries with 0; otherwise a no-op."""
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _flat(t: torch.Tensor) -> torch.Tensor:
    """Flatten everything after the batch dim."""
    return t.reshape(t.shape[0], -1)


def _split_per_step(
    target_tensor: torch.Tensor, k_steps: int
) -> List[torch.Tensor]:
    """Split a ``(B, C, T)`` target into ``k_steps`` equal-length slices along T."""
    n_per = target_tensor.shape[-1] // k_steps
    return [
        target_tensor[..., i * n_per : (i + 1) * n_per].contiguous()
        for i in range(k_steps)
    ]


def _synthetic_diag_inputs(
    model: E2EFoundationModel, batch: int = 2
) -> Dict[str, torch.Tensor]:
    return {
        cfg.name: torch.randn(batch, cfg.n_channels, cfg.window_samples)
        for cfg in model.diagnostics
    }


def _synthetic_act_per_step(
    model: E2EFoundationModel, n_steps: int, batch: int = 2
) -> List[Dict[str, torch.Tensor]]:
    return [
        {
            cfg.name: torch.randn(batch, cfg.n_channels, cfg.window_samples)
            for cfg in model.actuators
        }
        for _ in range(n_steps)
    ]


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rollout_model() -> TokenSpaceRollout:
    """Load E2E model + rollout wrapper from a trained checkpoint.

    The checkpoint is produced by the Stage 1 / Stage 2 training scripts and
    carries its own ``diagnostics`` / ``actuators`` / ``args`` entries so the
    architecture is reconstructed from the checkpoint alone — no reliance on
    CLI defaults which may drift.
    """
    ckpt_path = _env_path(CHECKPOINT_ENV)
    assert ckpt_path is not None  # guarded by pytestmark
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    args = ckpt["args"]
    model = E2EFoundationModel(
        diagnostics=diagnostics,
        actuators=actuators,
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        dropout=0.0,
    )

    # Stage 3 checkpoints carry LoRA adapter parameters. Detect them in
    # the state_dict and wrap the backbone's attention layers before
    # loading, otherwise load_state_dict errors on unexpected keys.
    state_dict = ckpt["model_state_dict"]
    if any(".lora_" in k for k in state_dict):
        apply_lora_to_backbone(
            model.backbone,
            rank=int(args.get("lora_rank", 16)),
            alpha=float(args.get("lora_alpha", 16.0)),
        )

    model.load_state_dict(state_dict)
    model.eval()
    return TokenSpaceRollout(model, dt_s=0.05)


@pytest.fixture(scope="module")
def real_val_rollout(
    rollout_model: TokenSpaceRollout,
) -> Dict[str, Any]:
    """Fetch one real val batch and run a 10-step rollout.

    Skips when ``E2E_DATA_DIR`` and ``E2E_STATS_PATH`` aren't provided —
    tests 1 and 3 need ground-truth trajectories; tests 2 and 4 don't use
    this fixture.
    """
    data_dir = _env_path(DATA_DIR_ENV)
    stats_path = _env_path(STATS_PATH_ENV)
    if data_dir is None or not data_dir.exists():
        pytest.skip(f"Set ${DATA_DIR_ENV} to a directory of *_processed.h5 shots")
    if stats_path is None or not stats_path.exists():
        pytest.skip(f"Set ${STATS_PATH_ENV} to a preprocessing_stats.pt file")

    from torch.utils.data import DataLoader

    from tokamak_foundation_model.data.data_loader import collate_fn
    from tokamak_foundation_model.data.multi_file_dataset import (
        TokamakMultiFileDataset,
    )

    model = rollout_model.model
    diag_names = [c.name for c in model.diagnostics]
    act_names = [c.name for c in model.actuators]

    shot_files = sorted(data_dir.glob("*_processed.h5"))[:5]
    stats = torch.load(stats_path, weights_only=False)

    ds = TokamakMultiFileDataset(
        shot_files,
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=K_ROLLOUT * 0.05,
        step_size_s=0.05,  # non-overlapping — cleaner for eval geometry
        warmup_s=1.0,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
    )
    loader = DataLoader(
        ds,
        batch_size=VAL_BATCH,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
    batch = next(iter(loader))

    diag_initial: Dict[str, torch.Tensor] = {
        n: _nanclean(batch["inputs"][n].float()) for n in diag_names
    }
    diag_target_per_step: List[Dict[str, torch.Tensor]] = []
    act_per_step: List[Dict[str, torch.Tensor]] = []
    for k in range(K_ROLLOUT):
        diag_target_per_step.append(
            {
                n: _nanclean(
                    _split_per_step(batch["targets"][n].float(), K_ROLLOUT)[k]
                )
                for n in diag_names
            }
        )
        act_per_step.append(
            {
                n: _nanclean(
                    _split_per_step(batch["targets"][n].float(), K_ROLLOUT)[k]
                )
                for n in act_names
            }
        )

    with torch.no_grad():
        result = rollout_model(diag_initial, act_per_step)

    return {
        "diag_initial": diag_initial,
        "diag_target_per_step": diag_target_per_step,
        "act_per_step": act_per_step,
        "result": result,
        "names": diag_names,
    }


# ── Test 1: copy baseline win rate ───────────────────────────────────


def test_copy_baseline_win_rate_step_1_and_10(
    real_val_rollout: Dict[str, Any],
) -> None:
    """Model beats deterministic copy baseline > 80 % at step 1, > 60 % at step 10.

    Per-sample comparison: aggregate MAE across modalities (mean of per-modality
    MAEs to avoid letting big-channel modalities dominate). The copy baseline
    is ``diag_initial`` — the input state echoed as the prediction for every
    step. Deterministic targets per the §5 hard-won rule.
    """
    result: RolloutResult = real_val_rollout["result"]
    targets: List[Dict[str, torch.Tensor]] = real_val_rollout["diag_target_per_step"]
    diag_initial: Dict[str, torch.Tensor] = real_val_rollout["diag_initial"]
    names: List[str] = real_val_rollout["names"]

    def aggregate_mae(
        pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Per-sample MAE averaged across modalities (shape ``(B,)``)."""
        batch = next(iter(pred.values())).shape[0]
        acc = torch.zeros(batch)
        for n in names:
            diff = (_nanclean(pred[n]) - _nanclean(target[n])).abs()
            acc = acc + diff.mean(dim=tuple(range(1, diff.dim())))
        return acc / len(names)

    # Step 1 threshold stays at 80 % regardless of K_ROLLOUT (predicting one
    # step is the easy case). Last-step threshold relaxes with K_ROLLOUT.
    last_step_idx = K_ROLLOUT - 1
    for step_index, threshold in [
        (0, 0.80),
        (last_step_idx, _copy_win_last_step_threshold()),
    ]:
        model_mae = aggregate_mae(result.predictions[step_index], targets[step_index])
        copy_mae = aggregate_mae(diag_initial, targets[step_index])
        wins = (model_mae < copy_mae).float().mean().item()
        assert wins > threshold, (
            f"Step {step_index + 1}: model wins only {wins:.1%}, "
            f"need > {threshold:.0%}. "
            f"Mean model MAE = {model_mae.mean().item():.4f}, "
            f"mean copy MAE = {copy_mae.mean().item():.4f}."
        )


# ── Test 2: no fixed-point ───────────────────────────────────────────


def test_no_fixed_point(rollout_model: TokenSpaceRollout) -> None:
    """``K_ROLLOUT``-step rollout: cos_sim(diag_tokens_{k-1}, diag_tokens_k) < 0.99 for all k.

    Uses synthetic in-distribution inputs (standardized ~N(0, 1), matching the
    signal space the model saw during Stage 1). A trained model should produce
    a *moving* trajectory — persistent cos_sim ≥ 0.99 across many steps means
    the rollout has collapsed to a fixed point and the model is effectively
    predicting zero change.
    """
    torch.manual_seed(0)
    model = rollout_model.model

    diag_initial = _synthetic_diag_inputs(model, batch=2)
    act_per_step = _synthetic_act_per_step(model, n_steps=K_ROLLOUT, batch=2)
    with torch.no_grad():
        result = rollout_model(diag_initial, act_per_step)

    tokens = result.diagnostic_tokens  # length K + 1
    for k in range(len(tokens) - 1):
        cs = F.cosine_similarity(
            tokens[k].flatten(), tokens[k + 1].flatten(), dim=0
        ).item()
        assert cs < 0.99, (
            f"Rollout step {k} → {k + 1}: diag-token cos_sim = {cs:.4f} ≥ 0.99. "
            "Trajectory has collapsed to a fixed point."
        )


# ── Test 3: model vs gt cos_sim gap (Phase A milestone A3) ───────────


def test_model_vs_gt_cos_sim_gap_steps_1_to_10(
    real_val_rollout: Dict[str, Any],
) -> None:
    """|model_cos_sim − gt_cos_sim| < threshold per step, averaged across modalities.

    Threshold scales with ``K_ROLLOUT`` via :func:`_cos_sim_gap_threshold` —
    0.05 for K≤10 (Phase A A3), 0.10 for K>10 (A4).

    For each step ``k``:
      - model_cs = cos_sim(model_prediction[k-1], model_prediction[k])
                   (with model_prediction[-1] = diag_initial)
      - gt_cs    = cos_sim(ground_truth[k-1], ground_truth[k])
                   (with ground_truth[-1] = diag_initial)

    Per-modality cos_sim is computed, then averaged across modalities. This
    sidesteps the dimension-weighting issue that would arise from flattening
    all modalities together (filterscopes at 8×500 would drown Thomson at
    44×5).
    """
    result: RolloutResult = real_val_rollout["result"]
    targets: List[Dict[str, torch.Tensor]] = real_val_rollout["diag_target_per_step"]
    diag_initial: Dict[str, torch.Tensor] = real_val_rollout["diag_initial"]
    names: List[str] = real_val_rollout["names"]

    for k in range(K_ROLLOUT):
        gaps: List[float] = []
        for n in names:
            model_prev = (
                diag_initial[n] if k == 0 else result.predictions[k - 1][n]
            )
            model_curr = result.predictions[k][n]
            gt_prev = diag_initial[n] if k == 0 else targets[k - 1][n]
            gt_curr = targets[k][n]

            model_cs = (
                F.cosine_similarity(
                    _flat(_nanclean(model_prev)),
                    _flat(_nanclean(model_curr)),
                    dim=1,
                )
                .mean()
                .item()
            )
            gt_cs = (
                F.cosine_similarity(
                    _flat(_nanclean(gt_prev)),
                    _flat(_nanclean(gt_curr)),
                    dim=1,
                )
                .mean()
                .item()
            )
            gaps.append(abs(model_cs - gt_cs))

        mean_gap = sum(gaps) / len(gaps)
        threshold = _cos_sim_gap_threshold()
        assert mean_gap < threshold, (
            f"Step {k + 1}: mean |model_cos_sim − gt_cos_sim| across "
            f"{len(names)} modalities = {mean_gap:.4f} ≥ {threshold:.2f}. "
            f"Per-modality gaps: "
            + ", ".join(f"{n}={g:.3f}" for n, g in zip(names, gaps))
        )


# ── Test 4: actuator sensitivity in rollout ──────────────────────────


def test_actuator_sensitivity_in_rollout(
    rollout_model: TokenSpaceRollout,
) -> None:
    """Same initial state, two distinct actuator trajectories → cos_sim < 0.9 at step 10.

    If actuators have no learned effect inside the rollout, two radically
    different actuator sequences from the same plasma state will produce
    near-identical predictions at step 10. This guards against the actuator
    branch being implicitly pruned during training.
    """
    torch.manual_seed(0)
    model = rollout_model.model

    diag_initial = _synthetic_diag_inputs(model, batch=2)
    torch.manual_seed(1)
    act_A = _synthetic_act_per_step(model, n_steps=K_ROLLOUT, batch=2)
    torch.manual_seed(2)
    act_B = _synthetic_act_per_step(model, n_steps=K_ROLLOUT, batch=2)

    with torch.no_grad():
        result_A = rollout_model(diag_initial, act_A)
        result_B = rollout_model(diag_initial, act_B)

    names = [c.name for c in model.diagnostics]
    pred_A_flat = torch.cat(
        [_flat(_nanclean(result_A.predictions[-1][n])) for n in names], dim=1
    )
    pred_B_flat = torch.cat(
        [_flat(_nanclean(result_B.predictions[-1][n])) for n in names], dim=1
    )
    cs = F.cosine_similarity(pred_A_flat, pred_B_flat, dim=1).mean().item()
    assert cs < 0.9, (
        f"Step {K_ROLLOUT}: cos_sim(trajectory_A, trajectory_B) = {cs:.4f} ≥ 0.9. "
        "Actuator conditioning has negligible effect inside the rollout."
    )


# ── Test 5: displacement direction ──────────────────────────────────


def test_displacement_direction(
    real_val_rollout: Dict[str, Any],
) -> None:
    """Displacement direction: cos_sim(pred - context, target - context) > 0.5.

    Verifies the model moves toward the target, not just away from context.
    A scaled copy or random displacement would score near 0.0. A model
    producing genuine dynamics scores near 1.0. Threshold 0.5 is the
    minimum for "directionally correct on average."
    """
    result: RolloutResult = real_val_rollout["result"]
    targets = real_val_rollout["diag_target_per_step"]
    diag_initial = real_val_rollout["diag_initial"]
    names = real_val_rollout["names"]

    for k in [0, K_ROLLOUT // 2, K_ROLLOUT - 1]:
        dir_cos_per_mod: List[float] = []
        for n in names:
            context = diag_initial[n] if k == 0 else result.predictions[k - 1][n]
            pred = result.predictions[k][n]
            target = targets[k][n]

            disp_pred = _flat(_nanclean(pred - context))
            disp_tgt = _flat(_nanclean(target - context))

            # Skip samples where target doesn't move (copy is optimal)
            tgt_norm = disp_tgt.norm(dim=1)
            valid = tgt_norm > 1e-6
            if valid.sum() < 2:
                continue

            dc = F.cosine_similarity(
                disp_pred[valid], disp_tgt[valid], dim=1
            ).mean().item()
            dir_cos_per_mod.append(dc)

        if not dir_cos_per_mod:
            continue
        mean_dc = sum(dir_cos_per_mod) / len(dir_cos_per_mod)
        assert mean_dc > 0.5, (
            f"Step {k + 1}: mean direction_cos = {mean_dc:.3f} ≤ 0.5. "
            "Model displacement is not toward the target. "
            "Per-modality: "
            + ", ".join(f"{n}={d:.3f}" for n, d in zip(names, dir_cos_per_mod))
        )