"""Evaluation script for Stage 2 (delta-loss) E2E checkpoints.

Loads a frozen Stage 2 checkpoint, runs a full K-step autoregressive rollout
over the val set, and produces:

  * per-step per-modality MAE / copy-MAE / direction_cos / magnitude_ratio
  * per-channel MAE breakdown averaged across K rollout steps (CSV)
  * per-modality K-step trajectory plots (PNG)
  * ``metrics.json`` (full per-step nested dump)
  * ``summary.md`` with PASS / FAIL on the Stage 2 gates:
       1. model_mae < copy_mae at k=1 (Stage 1 carry-forward)
       2. model_mae < copy_mae at k=K (rollout-end gate)
       3. direction_cos > 0 at every k (no anti-aligned predictions —
          the §5.9 test 5 motivation for the displacement loss)
       4. magnitude_ratio ∈ [0.3, 3.0] at every k (loose under/overshoot
          guard; the tighter §5.9 target is 0.8–1.2 at k=K)

Run::

    pixi run python scripts/training/eval_e2e_stage2.py \
        --checkpoint runs/e2e_stage2_delta/e2e_stage2_delta_best.pt \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path scripts/slurm/preprocessing_stats.pt \
        --output_dir runs/e2e_stage2_delta/eval_best

Add ``--use_video tangtv`` for any C-Stage 2 checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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
from tokamak_foundation_model.e2e.rollout import TokenSpaceRollout

logger = logging.getLogger("eval_stage2")


# ── Sample-rate registry (per-modality target splitting) ─────────────

SLOW_FS = 100.0
FAST_FS = 10_000.0

_SLOW_TS_NAMES = {
    "ts_core_density",
    "ts_core_temp",
    "ts_tangential_density",
    "ts_tangential_temp",
    "cer_ti",
    "cer_rot",
    "mse",
}
_FAST_TS_NAMES = {"filterscopes"}
_ACTUATOR_NAMES = {
    "pin", "beam_voltage", "ech_power", "ech_tor_angle", "ech_pol_angle",
    "ech_polarization", "gas_flow", "gas_raw", "rmp",
}

SAMPLE_RATES_HZ: Dict[str, float] = {
    **{n: SLOW_FS for n in _SLOW_TS_NAMES},
    **{n: FAST_FS for n in _FAST_TS_NAMES},
    **{n: FAST_FS for n in _ACTUATOR_NAMES},
}


# ── Helpers ──────────────────────────────────────────────────────────


def _clean_and_mask(
    tensor: torch.Tensor, existing_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(tensor)
    cleaned = torch.where(finite, tensor, torch.zeros_like(tensor))
    mask = finite.float()
    if existing_mask is not None:
        mask = mask * existing_mask
    return cleaned, mask


def samples_per_step(name: str, chunk_duration_s: float) -> int:
    return round(chunk_duration_s * SAMPLE_RATES_HZ[name])


def split_target_by_step(
    tensor: torch.Tensor, name: str, k_steps: int, chunk_duration_s: float
) -> List[torch.Tensor]:
    per = samples_per_step(name, chunk_duration_s)
    return [
        tensor[..., k * per : (k + 1) * per].contiguous() for k in range(k_steps)
    ]


def _step_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: torch.Tensor,
    mask: Optional[torch.Tensor],
    min_disp_norm: float,
) -> Tuple[float, float, float, int]:
    """Return ``(mae, dir_cos, mag_ratio, n_valid)`` — all floats / int."""
    cleaned_pred, mp = _clean_and_mask(pred, None)
    cleaned_tgt, mt = _clean_and_mask(target, mask)
    cleaned_ctx, mc = _clean_and_mask(ctx, None)
    joint = mp * mt * mc
    denom = joint.sum().clamp_min(1.0)
    mae = ((cleaned_pred - cleaned_tgt).abs() * joint).sum() / denom

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
        return mae.item(), float("nan"), float("nan"), 0
    dir_cos = F.cosine_similarity(dp[valid], dt[valid], dim=1).mean()
    mag_ratio = (
        pred_norm[valid] / tgt_norm[valid].clamp_min(1e-6)
    ).mean()
    return mae.item(), dir_cos.item(), mag_ratio.item(), n_valid


def _copy_mae(
    diag_initial: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> float:
    """MAE of the trivial ``prediction = diag_initial`` baseline at any step k."""
    cleaned_pred, mp = _clean_and_mask(diag_initial, None)
    cleaned_tgt, mt = _clean_and_mask(target, mask)
    joint = mp * mt
    denom = joint.sum().clamp_min(1.0)
    return (
        ((cleaned_pred - cleaned_tgt).abs() * joint).sum() / denom
    ).item()


def resolve_val_files(
    data_dir: Path, val_fraction: float, seed: int
) -> List[Path]:
    rng = random.Random(seed)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    rng.shuffle(all_files)
    n_val = max(1, int(val_fraction * len(all_files)))
    return all_files[:n_val]


# ── Accumulators ─────────────────────────────────────────────────────


class PerStepAccumulator:
    """Per-(k, modality) sums of MAE / copy_mae / dir_cos / mag_ratio."""

    def __init__(self, names: List[str], K: int) -> None:
        self.names = names
        self.K = K
        self.mae_sum = {k: {n: 0.0 for n in names} for k in range(K)}
        self.copy_sum = {k: {n: 0.0 for n in names} for k in range(K)}
        self.dir_cos_sum = {k: {n: 0.0 for n in names} for k in range(K)}
        self.mag_ratio_sum = {k: {n: 0.0 for n in names} for k in range(K)}
        self.n_valid_disp = {k: {n: 0 for n in names} for k in range(K)}
        self.n_batches = 0

    def update(
        self, k: int, name: str,
        mae: float, copy_mae: float,
        dir_cos: float, mag_ratio: float, n_valid: int,
    ) -> None:
        self.mae_sum[k][name] += mae
        self.copy_sum[k][name] += copy_mae
        if n_valid > 0:
            self.dir_cos_sum[k][name] += dir_cos * n_valid
            self.mag_ratio_sum[k][name] += mag_ratio * n_valid
            self.n_valid_disp[k][name] += n_valid

    def step(self) -> None:
        self.n_batches += 1

    def finalize(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        out: Dict[int, Dict[str, Dict[str, float]]] = {}
        denom = max(self.n_batches, 1)
        for k in range(self.K):
            out[k] = {}
            for n in self.names:
                model_mae = self.mae_sum[k][n] / denom
                copy_mae = self.copy_sum[k][n] / denom
                nv = self.n_valid_disp[k][n]
                dir_cos = (
                    self.dir_cos_sum[k][n] / nv if nv > 0 else float("nan")
                )
                mag_ratio = (
                    self.mag_ratio_sum[k][n] / nv if nv > 0 else float("nan")
                )
                out[k][n] = {
                    "model_mae": model_mae,
                    "copy_mae": copy_mae,
                    "delta": copy_mae - model_mae,
                    "direction_cos": dir_cos,
                    "magnitude_ratio": mag_ratio,
                    "n_valid_dir_samples": nv,
                }
        return out


class PerChannelAccumulator:
    """Per-modality, per-channel MAE summed over batch + time + (for video)
    spatial dims, and across all K rollout steps. Reduced at finalize()."""

    def __init__(self, names: List[str]) -> None:
        self.names = names
        self.model_sum: Dict[str, torch.Tensor] = {}
        self.copy_sum: Dict[str, torch.Tensor] = {}
        self.mask_sum: Dict[str, torch.Tensor] = {}
        self._init = {n: False for n in names}

    def _ensure(self, n: str, n_channels: int, device: torch.device) -> None:
        if not self._init[n]:
            self.model_sum[n] = torch.zeros(n_channels, device=device)
            self.copy_sum[n] = torch.zeros(n_channels, device=device)
            self.mask_sum[n] = torch.zeros(n_channels, device=device)
            self._init[n] = True

    def update(
        self,
        name: str,
        pred: torch.Tensor,
        copy_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        self._ensure(name, pred.shape[1], pred.device)
        cleaned_pred, mp = _clean_and_mask(pred, None)
        cleaned_copy, _ = _clean_and_mask(copy_pred, None)
        cleaned_tgt, mt = _clean_and_mask(target, mask)
        joint = mp * mt
        reduce_dims = [d for d in range(pred.ndim) if d != 1]
        self.model_sum[name] += (
            (cleaned_pred - cleaned_tgt).abs() * joint
        ).sum(dim=reduce_dims)
        self.copy_sum[name] += (
            (cleaned_copy - cleaned_tgt).abs() * joint
        ).sum(dim=reduce_dims)
        self.mask_sum[name] += joint.sum(dim=reduce_dims)

    def finalize(self) -> Dict[str, List[Dict[str, float]]]:
        out: Dict[str, List[Dict[str, float]]] = {}
        for n in self.names:
            if not self._init[n]:
                out[n] = []
                continue
            denom = self.mask_sum[n].clamp_min(1.0)
            mae = (self.model_sum[n] / denom).cpu().tolist()
            cmae = (self.copy_sum[n] / denom).cpu().tolist()
            valid = (self.mask_sum[n] > 0).cpu().tolist()
            rows = []
            for c, (m, cb, v) in enumerate(zip(mae, cmae, valid)):
                rows.append({
                    "channel": c,
                    "model_mae_avg_K": m if v else float("nan"),
                    "copy_mae_avg_K": cb if v else float("nan"),
                    "delta_avg_K": (cb - m) if v else float("nan"),
                    "n_valid": int(self.mask_sum[n][c].item()),
                })
            out[n] = rows
        return out


# ── Plotting ─────────────────────────────────────────────────────────


def _pick_plot_channels(
    target_np: np.ndarray, n_pick: int, rng: random.Random
) -> List[int]:
    n_channels = target_np.shape[1]
    candidates: List[int] = []
    for c in range(n_channels):
        col = target_np[:, c].reshape(-1)
        col_finite = col[np.isfinite(col)]
        if col_finite.size == 0 or np.allclose(col_finite, 0.0):
            continue
        candidates.append(c)
    if not candidates:
        candidates = list(range(min(n_channels, 4)))
    rng.shuffle(candidates)
    return candidates[: min(n_pick, len(candidates))]


def plot_ts_trajectory(
    name: str,
    pred_per_step: List[torch.Tensor],   # length K, each (B, C, T_per)
    target_per_step: List[torch.Tensor],
    diag_initial: torch.Tensor,          # (B, C, T_per) — input window
    n_samples: int,
    out_path: Path,
    rng: random.Random,
) -> None:
    """K-step rollout trajectory plot, rows=samples, cols=channels."""
    K = len(pred_per_step)
    pred_stack = torch.stack(pred_per_step, dim=2)   # (B, C, K, T_per)
    tgt_stack = torch.stack(target_per_step, dim=2)
    pred_np = pred_stack.detach().cpu().numpy()
    tgt_np = tgt_stack.detach().cpu().numpy()
    ctx_np = diag_initial.detach().cpu().numpy()
    B, C, _, T_per = pred_np.shape

    n_samples = min(n_samples, B)
    n_chan_plot = 4
    fig, axes = plt.subplots(
        n_samples,
        n_chan_plot,
        figsize=(3.6 * n_chan_plot, 2.4 * n_samples),
        squeeze=False,
    )

    # Stitch K windows along the time axis for plotting.
    pred_stitched = pred_np.reshape(B, C, K * T_per)
    tgt_stitched = tgt_np.reshape(B, C, K * T_per)

    sample_idx = list(range(B))
    rng.shuffle(sample_idx)
    sample_idx = sample_idx[:n_samples]

    for r, b in enumerate(sample_idx):
        chans = _pick_plot_channels(tgt_np[b : b + 1, :, 0, :], n_chan_plot, rng)
        chans = chans + [chans[-1]] * (n_chan_plot - len(chans))
        for cc, ch in enumerate(chans):
            ax = axes[r][cc]
            t_ctx = np.arange(T_per)
            t_roll = np.arange(K * T_per) + T_per
            ax.plot(t_ctx, ctx_np[b, ch], color="0.6", lw=1.0, label="input")
            ax.plot(t_roll, tgt_stitched[b, ch], color="C0", lw=1.0, label="target")
            ax.plot(
                t_roll, pred_stitched[b, ch], color="C3", lw=1.0,
                linestyle="--", label="pred",
            )
            for k_b in range(1, K + 1):
                ax.axvline(T_per + k_b * T_per, color="k", alpha=0.08, lw=0.5)
            ax.set_title(f"sample {b}, ch {ch}", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and cc == 0:
                ax.legend(fontsize=6, loc="best")
    fig.suptitle(f"{name} — K={K} rollout trajectory", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_video_modality(
    name: str,
    pred_step_0: torch.Tensor,           # (B, C, T_p, H, W) at step 0
    target_step_0: torch.Tensor,
    diag_initial: torch.Tensor,
    out_path: Path,
) -> None:
    """Per-channel ctx / target / pred / |diff| at step 0, frame 0."""
    pred_np = pred_step_0.detach().cpu().numpy()
    tgt_np = target_step_0.detach().cpu().numpy()
    ctx_np = diag_initial.detach().cpu().numpy()
    b, t = 0, 0
    n_channels = pred_np.shape[1]
    fig, axes = plt.subplots(
        n_channels, 4, figsize=(11, 2.0 * n_channels), squeeze=False,
    )
    for c in range(n_channels):
        col_imgs = [
            ("input", ctx_np[b, c, t]),
            ("target", tgt_np[b, c, t]),
            ("pred", pred_np[b, c, t]),
            ("|pred-tgt|", np.abs(pred_np[b, c, t] - tgt_np[b, c, t])),
        ]
        for col, (title, im) in enumerate(col_imgs):
            ax = axes[c][col]
            ax.imshow(im, cmap="gray" if col != 3 else "magma", aspect="auto")
            if c == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"ch {c}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{name} — sample 0, step 0, frame 0", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Output writers ───────────────────────────────────────────────────


def _gates(
    per_step: Dict[int, Dict[str, Dict[str, float]]],
    K: int,
    mag_lo: float,
    mag_hi: float,
) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, List[str]]]:
    """Compute four per-modality boolean gates, plus a list of failing modality
    names per gate."""
    names = list(per_step[0].keys())
    gate_results = {n: {} for n in names}
    failing: Dict[str, List[str]] = {
        "k1_beats_copy": [], "kK_beats_copy": [],
        "dir_cos_positive": [], "mag_ratio_in_range": [],
    }
    for n in names:
        m1 = per_step[0][n]
        mK = per_step[K - 1][n]
        g1 = m1["model_mae"] < m1["copy_mae"]
        g2 = mK["model_mae"] < mK["copy_mae"]
        g3 = all(
            (per_step[k][n]["direction_cos"] > 0)
            or (per_step[k][n]["n_valid_dir_samples"] == 0)
            for k in range(K)
        )
        g4 = all(
            (mag_lo <= per_step[k][n]["magnitude_ratio"] <= mag_hi)
            or (per_step[k][n]["n_valid_dir_samples"] == 0)
            for k in range(K)
        )
        gate_results[n] = {
            "k1_beats_copy": bool(g1),
            "kK_beats_copy": bool(g2),
            "dir_cos_positive": bool(g3),
            "mag_ratio_in_range": bool(g4),
        }
        if not g1: failing["k1_beats_copy"].append(n)
        if not g2: failing["kK_beats_copy"].append(n)
        if not g3: failing["dir_cos_positive"].append(n)
        if not g4: failing["mag_ratio_in_range"].append(n)
    return gate_results, failing


def write_metrics_json(
    out_path: Path,
    checkpoint_path: Path,
    ckpt_step: Optional[int],
    args_used: Dict[str, Any],
    per_step: Dict[int, Dict[str, Dict[str, float]]],
    per_channel: Dict[str, List[Dict[str, float]]],
    gate_results: Dict[str, Dict[str, bool]],
    failing: Dict[str, List[str]],
    sum_mae_at_K: Dict[int, float],
    n_batches: int,
    K: int,
) -> None:
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": ckpt_step,
        "K": K,
        "args": args_used,
        "n_batches": n_batches,
        "sum_mae_per_step": sum_mae_at_K,
        "per_step": {str(k): per_step[k] for k in per_step},
        "per_channel": per_channel,
        "gates_per_modality": gate_results,
        "gates_failing_modalities": failing,
        "all_gates_pass": all(not v for v in failing.values()),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def write_per_channel_csv(
    out_path: Path, per_channel: Dict[str, List[Dict[str, float]]]
) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "modality", "channel",
            "model_mae_avg_K", "copy_mae_avg_K", "delta_avg_K", "n_valid",
        ])
        for name, rows in per_channel.items():
            for r in rows:
                w.writerow([
                    name, r["channel"],
                    f"{r['model_mae_avg_K']:.6f}",
                    f"{r['copy_mae_avg_K']:.6f}",
                    f"{r['delta_avg_K']:.6f}",
                    r["n_valid"],
                ])


def write_summary_md(
    out_path: Path,
    checkpoint_path: Path,
    ckpt_step: Optional[int],
    per_step: Dict[int, Dict[str, Dict[str, float]]],
    K: int,
    gate_results: Dict[str, Dict[str, bool]],
    failing: Dict[str, List[str]],
    sum_mae_at_K: Dict[int, float],
    n_batches: int,
    mag_lo: float,
    mag_hi: float,
) -> None:
    names = list(per_step[0].keys())
    lines: List[str] = []
    lines.append("# Stage 2 evaluation summary\n")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- Step: {ckpt_step if ckpt_step is not None else 'unknown'}")
    lines.append(f"- K (rollout horizon): {K}")
    lines.append(f"- Val batches: {n_batches}")
    lines.append(f"- Sum-of-per-step MAE at k=1: {sum_mae_at_K[0]:.4f}")
    lines.append(f"- Sum-of-per-step MAE at k={K}: {sum_mae_at_K[K - 1]:.4f}")

    all_pass = all(not v for v in failing.values())
    gate = "PASS" if all_pass else "FAIL"
    lines.append(f"- **Stage 2 gates ({gate}):**")
    lines.append(
        f"  - G1 model<copy at k=1     : "
        f"{'PASS' if not failing['k1_beats_copy'] else 'FAIL — ' + ', '.join(failing['k1_beats_copy'])}"
    )
    lines.append(
        f"  - G2 model<copy at k={K}    : "
        f"{'PASS' if not failing['kK_beats_copy'] else 'FAIL — ' + ', '.join(failing['kK_beats_copy'])}"
    )
    lines.append(
        f"  - G3 dir_cos > 0 at all k  : "
        f"{'PASS' if not failing['dir_cos_positive'] else 'FAIL — ' + ', '.join(failing['dir_cos_positive'])}"
    )
    lines.append(
        f"  - G4 mag_ratio ∈ [{mag_lo}, {mag_hi}]: "
        f"{'PASS' if not failing['mag_ratio_in_range'] else 'FAIL — ' + ', '.join(failing['mag_ratio_in_range'])}"
    )
    lines.append("")
    lines.append("## k=1 (single-step) per-modality\n")
    lines.append(
        "| modality | model_mae | copy_mae | Δ | dir_cos | mag_ratio | "
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for n in names:
        m = per_step[0][n]
        lines.append(
            f"| {n} | {m['model_mae']:.4f} | {m['copy_mae']:.4f} | "
            f"{m['delta']:+.4f} | {m['direction_cos']:.3f} | "
            f"{m['magnitude_ratio']:.3f} |"
        )
    lines.append("")
    lines.append(f"## k={K} (rollout end) per-modality\n")
    lines.append(
        "| modality | model_mae | copy_mae | Δ | dir_cos | mag_ratio | "
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for n in names:
        m = per_step[K - 1][n]
        lines.append(
            f"| {n} | {m['model_mae']:.4f} | {m['copy_mae']:.4f} | "
            f"{m['delta']:+.4f} | {m['direction_cos']:.3f} | "
            f"{m['magnitude_ratio']:.3f} |"
        )
    out_path.write_text("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--stats_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--K", type=int, default=10, help="Rollout horizon")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument(
        "--step_size_s", type=float, default=0.5,
        help="Stride between val chunks. Default 0.5s = K*chunk for K=10 "
             "(non-overlapping target horizons).",
    )
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument(
        "--use_video", type=str, nargs="*", default=None,
        help="Camera names (e.g. 'tangtv'); needed for C-Stage 2 checkpoints.",
    )
    p.add_argument("--n_plot_samples", type=int, default=4)
    p.add_argument("--min_disp_norm", type=float, default=0.01)
    p.add_argument("--mag_ratio_lo", type=float, default=0.3)
    p.add_argument("--mag_ratio_hi", type=float, default=3.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    K = int(args.K)

    # ── Load checkpoint ──────────────────────────────────────────────
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
        rank = int(ck_args.get("lora_rank", 16))
        alpha = float(ck_args.get("lora_alpha", 16.0))
        apply_lora_to_backbone(model.backbone, rank=rank, alpha=alpha)
        logger.info(f"LoRA detected: rank={rank} alpha={alpha}")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    rollout = TokenSpaceRollout(model, dt_s=args.chunk_duration_s).to(device)
    rollout.eval()
    ckpt_step = ckpt.get("step")
    logger.info(
        f"Loaded {args.checkpoint.name}: step={ckpt_step} "
        f"diagnostics={[c.name for c in diagnostics]}"
    )

    ckpt_video = [c.name for c in diagnostics if c.kind == "video"]
    cli_video = args.use_video or []
    if set(ckpt_video) != set(cli_video):
        logger.warning(
            f"--use_video={cli_video} but checkpoint has video={ckpt_video}; "
            "using checkpoint's video set."
        )

    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]

    # ── Build val dataset ────────────────────────────────────────────
    stats = torch.load(args.stats_path, weights_only=False)
    val_files = resolve_val_files(args.data_dir, args.val_fraction, args.seed)
    logger.info(f"Val files: {len(val_files)}")
    if not val_files:
        raise SystemExit(f"No HDF5 files matched {args.data_dir}/*_processed.h5")

    lengths_cache = (
        args.checkpoint.parent / "lengths_eval_stage2_val.pt"
    )
    if lengths_cache.exists():
        lengths_cache.unlink()

    ds = TokamakMultiFileDataset(
        val_files,
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
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=False, pin_memory=False,
    )

    # ── Eval loop ────────────────────────────────────────────────────
    accum = PerStepAccumulator(diag_names, K)
    per_chan = PerChannelAccumulator(diag_names)
    plot_cache: Dict[str, Dict[str, Any]] = {}
    rng = random.Random(args.seed)
    n_processed = 0

    for i, batch in enumerate(loader):
        if args.max_batches is not None and i >= args.max_batches:
            break

        diag_initial: Dict[str, torch.Tensor] = {}
        for name in diag_names:
            raw = batch["inputs"][name].to(device, non_blocking=True).float()
            cleaned, _ = _clean_and_mask(raw, None)
            diag_initial[name] = cleaned

        act_per_step: List[Dict[str, torch.Tensor]] = []
        target_per_step: List[Dict[str, torch.Tensor]] = []
        mask_per_step: List[Dict[str, Optional[torch.Tensor]]] = []
        for k in range(K):
            ak: Dict[str, torch.Tensor] = {}
            for name in act_names:
                raw = batch["targets"][name].to(device, non_blocking=True).float()
                slc = split_target_by_step(raw, name, K, args.chunk_duration_s)[k]
                ak[name], _ = _clean_and_mask(slc, None)
            act_per_step.append(ak)

            tk: Dict[str, torch.Tensor] = {}
            mk: Dict[str, Optional[torch.Tensor]] = {}
            for name in diag_names:
                raw = batch["targets"][name].to(device, non_blocking=True).float()
                tk[name] = split_target_by_step(raw, name, K, args.chunk_duration_s)[k]
                mk_key = f"{name}_mask"
                if mk_key in batch["targets"]:
                    raw_mask = batch["targets"][mk_key].to(
                        device, non_blocking=True
                    ).float()
                    mk[name] = split_target_by_step(
                        raw_mask, name, K, args.chunk_duration_s
                    )[k]
                else:
                    mk[name] = None
            target_per_step.append(tk)
            mask_per_step.append(mk)

        result = rollout(diag_initial, act_per_step)

        for k in range(K):
            for name in diag_names:
                pred = result.predictions[k][name].float()
                target = target_per_step[k][name]
                mask = mask_per_step[k][name]
                ctx = diag_initial[name] if k == 0 else target_per_step[k - 1][name]

                mae, dir_cos, mag_ratio, n_valid = _step_metrics(
                    pred, target, ctx, mask, args.min_disp_norm
                )
                copy_mae = _copy_mae(diag_initial[name], target, mask)

                accum.update(k, name, mae, copy_mae, dir_cos, mag_ratio, n_valid)
                per_chan.update(
                    name, pred, diag_initial[name], target, mask
                )
        accum.step()
        n_processed += 1

        if i == 0:
            for name in diag_names:
                preds_K = [result.predictions[k][name].detach().cpu() for k in range(K)]
                tgts_K = [target_per_step[k][name].detach().cpu() for k in range(K)]
                kind = next(c.kind for c in diagnostics if c.name == name)
                plot_cache[name] = {
                    "kind": kind,
                    "preds": preds_K,
                    "targets": tgts_K,
                    "ctx": diag_initial[name].detach().cpu(),
                }

        if (i + 1) % 10 == 0:
            logger.info(f"  batch {i + 1} processed")

    logger.info(f"Eval complete: {n_processed} batches.")

    # ── Finalise ─────────────────────────────────────────────────────
    per_step = accum.finalize()
    per_channel_results = per_chan.finalize()
    sum_mae_at_K = {k: sum(per_step[k][n]["model_mae"] for n in diag_names) for k in range(K)}
    gate_results, failing = _gates(per_step, K, args.mag_ratio_lo, args.mag_ratio_hi)

    # ── Stdout table ─────────────────────────────────────────────────
    print()
    print(f"Stage 2 K={K} evaluation:")
    print(
        f"  {'modality':<24} | "
        f"{'k=1: model / copy / Δ':<28} | "
        f"{'k='+str(K)+': model / copy / Δ':<28} | "
        f"min_dir_cos  mag@K"
    )
    for n in diag_names:
        m1 = per_step[0][n]
        mK = per_step[K - 1][n]
        min_dc = min(per_step[k][n]["direction_cos"]
                     for k in range(K)
                     if per_step[k][n]["n_valid_dir_samples"] > 0)
        print(
            f"  {n:<24} | "
            f"{m1['model_mae']:.4f} / {m1['copy_mae']:.4f} / {m1['delta']:+.4f} | "
            f"{mK['model_mae']:.4f} / {mK['copy_mae']:.4f} / {mK['delta']:+.4f} | "
            f"{min_dc:+.3f}      {mK['magnitude_ratio']:.3f}"
        )
    print(f"  [sum-K MAE @ k=1]   {sum_mae_at_K[0]:.4f}")
    print(f"  [sum-K MAE @ k={K}]  {sum_mae_at_K[K - 1]:.4f}")
    all_pass = all(not v for v in failing.values())
    print(f"  [Stage 2 gates]     {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        for gate_name, mods in failing.items():
            if mods:
                print(f"    {gate_name}: {', '.join(mods)}")
    print()

    # ── Persist ──────────────────────────────────────────────────────
    args_serialisable = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    write_metrics_json(
        args.output_dir / "metrics.json",
        args.checkpoint, ckpt_step, args_serialisable,
        per_step, per_channel_results,
        gate_results, failing,
        sum_mae_at_K, n_processed, K,
    )
    write_per_channel_csv(args.output_dir / "per_channel.csv", per_channel_results)
    write_summary_md(
        args.output_dir / "summary.md",
        args.checkpoint, ckpt_step,
        per_step, K, gate_results, failing,
        sum_mae_at_K, n_processed,
        args.mag_ratio_lo, args.mag_ratio_hi,
    )

    # ── Plots ────────────────────────────────────────────────────────
    for cfg in diagnostics:
        cache = plot_cache.get(cfg.name)
        if cache is None:
            continue
        out_path = plots_dir / f"{cfg.name}.png"
        try:
            if cache["kind"] == "video":
                plot_video_modality(
                    cfg.name,
                    pred_step_0=cache["preds"][0],
                    target_step_0=cache["targets"][0],
                    diag_initial=cache["ctx"],
                    out_path=out_path,
                )
            else:
                plot_ts_trajectory(
                    cfg.name,
                    pred_per_step=cache["preds"],
                    target_per_step=cache["targets"],
                    diag_initial=cache["ctx"],
                    n_samples=args.n_plot_samples,
                    out_path=out_path,
                    rng=rng,
                )
        except Exception as exc:
            logger.warning(f"Plot for {cfg.name} failed: {exc}")

    logger.info(f"Wrote: {args.output_dir / 'metrics.json'}")
    logger.info(f"Wrote: {args.output_dir / 'per_channel.csv'}")
    logger.info(f"Wrote: {args.output_dir / 'summary.md'}")
    logger.info(f"Wrote: {plots_dir}/<modality>.png")


if __name__ == "__main__":
    main()