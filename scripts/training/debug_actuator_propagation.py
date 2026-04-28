"""Actuator-propagation audit for a trained E2E foundation-model checkpoint.

Motivated by §5.9 test 4 failing on the Stage 2 best checkpoint
(cos_sim(trajectory_A, trajectory_B) = 0.999 when two different actuator
trajectories are run from the same initial state). That gate says
"actuator conditioning has negligible effect inside the rollout", but
doesn't localise the failure. This script does: for one real val batch,
zero one actuator modality at a time and measure

  (a) per-backbone-layer L2 distance in the *diagnostic* token slice,
  (b) per-diagnostic-modality head-output relative L2 distance,

relative to the baseline (all actuators present). Reveals which actuator
modalities reach which diag outputs, and at which layer the signal
attenuates (if it does).

Run::

    pixi run python scripts/training/debug_actuator_propagation.py \
        --checkpoint scripts/slurm/runs/e2e_stage2/e2e_stage2_best.pt \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --output_dir runs/e2e_stage2/actuator_audit \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

logger = logging.getLogger("act_audit")


def _nanclean(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


@torch.no_grad()
def _forward_with_intermediates(
    model: E2EFoundationModel,
    diag_inputs: Dict[str, torch.Tensor],
    act_inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """Run the full pipeline and return (backbone intermediates, head outputs).

    ``intermediates`` is the list returned by
    :meth:`SharedBackbone.forward(return_intermediates=True)`:
    index 0 = post-step-conditioning, 1..N = per-block outputs, -1 = post
    final_norm.
    """
    batch_size = next(iter(diag_inputs.values())).shape[0]
    step = torch.zeros(batch_size, dtype=torch.long, device=device)
    time = torch.zeros(batch_size, device=device)

    tokens = model.tokenize(diag_inputs, act_inputs)
    intermediates = model.backbone(tokens, step, time, return_intermediates=True)
    # Final-norm output drives the heads.
    head_outputs = model.decode(intermediates[-1])
    return intermediates, head_outputs


def _diag_slice_end(model: E2EFoundationModel) -> int:
    """Where the diagnostic-token slice ends in the backbone's flat layout."""
    return max(
        layout.slice_.stop for layout in model.token_layout if layout.is_diagnostic
    )


def _measure_diag_layer_diff(
    intermediates_a: List[torch.Tensor],
    intermediates_b: List[torch.Tensor],
    diag_end: int,
) -> List[float]:
    """Per-layer mean L2 over diag tokens: ``mean over (B, diag, dim) of |a - b|``."""
    diffs: List[float] = []
    for a, b in zip(intermediates_a, intermediates_b):
        d = (a[:, :diag_end] - b[:, :diag_end]).norm(dim=-1)  # (B, n_diag)
        diffs.append(d.mean().item())
    return diffs


def _measure_head_rel_diff(
    head_a: Dict[str, torch.Tensor],
    head_b: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Per-diagnostic-modality ``||A - B|| / ||A||``."""
    out: Dict[str, float] = {}
    for name, a in head_a.items():
        b = head_b[name]
        num = (a - b).norm().item()
        den = a.norm().item()
        out[name] = num / den if den > 1e-12 else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
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

    # ── Pull one real val batch ──────────────────────────────────────
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
        prediction_horizon_s=0.05,
        step_size_s=0.05,
        warmup_s=1.0,
        lengths_cache_path=args.output_dir / "lengths_act_audit.pt",
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    batch = next(iter(loader))
    diag_inputs = {
        n: _nanclean(batch["inputs"][n].to(device).float()) for n in diag_names
    }
    act_inputs = {
        n: _nanclean(batch["targets"][n].to(device).float()) for n in act_names
    }
    batch_size = next(iter(diag_inputs.values())).shape[0]
    logger.info(f"Val batch: B={batch_size}")

    # ── Baseline forward ─────────────────────────────────────────────
    intermediates_baseline, head_baseline = _forward_with_intermediates(
        model, diag_inputs, act_inputs, device
    )
    diag_end = _diag_slice_end(model)
    n_layers_total = len(intermediates_baseline)
    logger.info(
        f"Diag slice: tokens [0, {diag_end}); backbone layers reported: "
        f"{n_layers_total} (= n_layers + 2 intermediates)."
    )

    # ── Zero-all-actuators perturbation (total actuator contribution) ─
    act_zero = {n: torch.zeros_like(act_inputs[n]) for n in act_names}
    inter_zero, head_zero = _forward_with_intermediates(
        model, diag_inputs, act_zero, device
    )
    layer_diff_all = _measure_diag_layer_diff(
        intermediates_baseline, inter_zero, diag_end
    )
    head_diff_all = _measure_head_rel_diff(head_baseline, head_zero)

    logger.info("")
    logger.info("BASELINE vs ALL-ACTUATORS-ZERO (the total actuator contribution):")
    logger.info(
        "  Per-layer diag-token L2 diff: "
        + ", ".join(f"L{i}={d:.4f}" for i, d in enumerate(layer_diff_all))
    )
    logger.info("  Per-diag head relative diff:")
    for name in diag_names:
        logger.info(f"    {name:<25s} {head_diff_all[name]:.4%}")

    # ── One-actuator-at-a-time perturbation ──────────────────────────
    logger.info("")
    logger.info("PER-ACTUATOR ZEROING — head-output relative diff per diag modality:")
    # Header: diag-modality columns
    header = f"{'actuator':<25}  " + "  ".join(f"{n[:12]:>12}" for n in diag_names)
    logger.info(header)
    logger.info("-" * len(header))

    # Also track last-layer diag-token diff for each actuator for a
    # compact summary.
    per_act_last_layer_diff: Dict[str, float] = {}
    per_act_head_diff: Dict[str, Dict[str, float]] = {}
    for a_name in act_names:
        act_perturbed = {
            n: (torch.zeros_like(act_inputs[n]) if n == a_name else act_inputs[n])
            for n in act_names
        }
        inter_p, head_p = _forward_with_intermediates(
            model, diag_inputs, act_perturbed, device
        )
        layer_diff = _measure_diag_layer_diff(
            intermediates_baseline, inter_p, diag_end
        )
        head_diff = _measure_head_rel_diff(head_baseline, head_p)
        per_act_last_layer_diff[a_name] = layer_diff[-1]
        per_act_head_diff[a_name] = head_diff
        logger.info(
            f"{a_name:<25}  "
            + "  ".join(f"{head_diff[d]:>11.2%}" for d in diag_names)
        )

    # ── Summary: which actuators connect to which diag outputs? ──────
    logger.info("")
    logger.info("Summary — actuator last-layer diag-token L2 (vs baseline):")
    for a_name, d in sorted(
        per_act_last_layer_diff.items(), key=lambda kv: kv[1], reverse=True
    ):
        logger.info(f"  {a_name:<25s} {d:.5f}")

    # Overall diagnostic: is ANY actuator having meaningful effect?
    max_head_diff = max(
        per_act_head_diff[a][d]
        for a in act_names
        for d in diag_names
    )
    logger.info("")
    logger.info(
        f"Max single-actuator head-output relative diff across all "
        f"(act, diag) pairs: {max_head_diff:.2%}"
    )
    sum_all_head_diff = sum(head_diff_all.values()) / len(head_diff_all)
    logger.info(
        f"Mean head-output relative diff when ALL actuators are zeroed: "
        f"{sum_all_head_diff:.2%}"
    )

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "batch_size": batch_size,
        "layer_diff_all_zero": layer_diff_all,
        "head_diff_all_zero": head_diff_all,
        "per_actuator_last_layer_diff": per_act_last_layer_diff,
        "per_actuator_head_diff": per_act_head_diff,
    }
    path = args.output_dir / "actuator_propagation_results.pt"
    torch.save(results, path)
    logger.info(f"Saved: {path}")


if __name__ == "__main__":
    main()
