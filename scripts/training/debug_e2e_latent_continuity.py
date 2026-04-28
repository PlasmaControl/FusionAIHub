"""C3 latent-continuity diagnostic for E2E tokenizers.

Answers the core research-plan question (``ResearchPlan.MD`` §1.1, C3):
does end-to-end training, trained under the prediction objective, produce
per-modality tokenizers whose latent geometry is *monotonic* with the
raw-signal geometry between consecutive 50 ms windows?

Protocol (mirror of ``archive/ae_baseline/scripts/training/debug_latent_continuity.py``,
the AE-baseline diagnostic that produced Spearman ≤ −0.1 across all 8
modalities):

    1. Non-overlapping ``chunk_duration_s = 0.1`` windows with
       ``step_size_s = 0.1`` → each dataset sample carries two consecutive
       50 ms windows stacked along the time axis.
    2. For each sample and each modality ``m``:
         sig_cos = cos_sim(flatten(window_t), flatten(window_{t+1}))
         tok_cos = cos_sim(flatten(tokenizer_m(window_t)),
                           flatten(tokenizer_m(window_{t+1})))
    3. Accumulate ``(sig_cos, tok_cos)`` pairs across many batches; compute
       Spearman rank correlation per modality.
    4. Save scatter plot + per-modality Spearman / Pearson / mean-std table.

Run on CPU (login node is fine)::

    pixi run python scripts/training/debug_e2e_latent_continuity.py \
        --checkpoint scripts/slurm/runs/e2e_stage1/e2e_stage1_best_stage2init.2715505.pt \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \
        --max_files 100 --batch_size 32 --max_batches 500 \
        --output_dir /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/runs/e2e_stage1/c3
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)

logger = logging.getLogger("c3_e2e")

# Match the windowing used during training.
WINDOW_S = 0.05

# Per-modality sample rates (Hz) — same as scripts/training/train_e2e_stage1.py.
SAMPLE_RATES_HZ: Dict[str, float] = {
    "ts_core_density": 100.0,
    "ts_core_temp": 100.0,
    "ts_tangential_density": 100.0,
    "ts_tangential_temp": 100.0,
    "cer_ti": 100.0,
    "cer_rot": 100.0,
    "mse": 100.0,
    "filterscopes": 10_000.0,
}


def _slice_window(
    signal: torch.Tensor, target_fs: float, k: int, dt_s: float = WINDOW_S
) -> torch.Tensor:
    """Return the k-th 50 ms window of ``signal``, with stride ``dt_s`` seconds."""
    n_win = round(WINDOW_S * target_fs)
    n_dt = round(dt_s * target_fs)
    start = k * n_dt
    return signal[..., start : start + n_win]


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity over flattened feature dims → shape ``(B,)``."""
    return F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1)


def _masked_cos(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-sample cosine similarity computed only over positions where
    ``mask`` is 1. Zeroing both vectors at invalid positions is equivalent to
    excluding those positions from both the dot product and the L2 norms.
    """
    a_m = a * mask
    b_m = b * mask
    return _cos(a_m, b_m)


def _valid_fraction(mask: torch.Tensor) -> torch.Tensor:
    """Per-sample fraction of positions that are valid → shape ``(B,)``."""
    flat = mask.reshape(mask.shape[0], -1).float()
    return flat.mean(dim=1)


def _nanclean(t: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _joint_valid_mask(
    x_t: torch.Tensor,
    x_t1: torch.Tensor,
    upstream_mask_t: Optional[torch.Tensor],
    upstream_mask_t1: Optional[torch.Tensor],
) -> torch.Tensor:
    """Build a joint valid mask = valid in BOTH windows, excluding any NaN/Inf.

    Same shape as ``x_t``. Returns a float tensor of 0/1 values.
    """
    m_t = torch.isfinite(x_t)
    m_t1 = torch.isfinite(x_t1)
    joint = m_t & m_t1
    if upstream_mask_t is not None:
        joint = joint & upstream_mask_t.bool()
    if upstream_mask_t1 is not None:
        joint = joint & upstream_mask_t1.bool()
    return joint.float()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--stats_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=500)
    parser.add_argument("--warmup_s", type=float, default=1.0)
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1,
        help="Number of ``dt_s``-offset window pairs per chunk (default 1 → "
        "2 consecutive 50 ms windows).",
    )
    parser.add_argument(
        "--min_valid_fraction",
        type=float,
        default=0.5,
        help="When computing the masked Spearman, drop pairs where the joint "
        "valid-mask fraction is below this (default 0.5). Prevents "
        "heavily-missing inputs from dominating the correlation via learned "
        "embeddings collapsing the token output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + extract tokenizers ────────────────────────────────
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators = [ActuatorConfig(**a) for a in ckpt["actuators"]]
    mod_args = ckpt["args"]
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
        f"Loaded {args.checkpoint.name}: "
        f"step={ckpt.get('step')}  val_loss={ckpt.get('val_loss'):.4f}  "
        f"d_model={mod_args['d_model']}  n_layers={mod_args['n_layers']}"
    )

    diag_names = [c.name for c in diagnostics]
    logger.info(f"Measuring {len(diag_names)} diagnostic tokenizers.")

    # ── Dataset (non-overlapping chunks of ``n_steps + 1`` windows) ───
    chunk_s = WINDOW_S * (args.n_steps + 1)
    stats = torch.load(args.stats_path, weights_only=False)
    rng = random.Random(args.seed)
    all_files = sorted(args.data_dir.glob("*_processed.h5"))
    rng.shuffle(all_files)
    files = all_files[: args.max_files]
    logger.info(f"Files: {len(files)}  chunk_s={chunk_s:.3f}")

    ds = TokamakMultiFileDataset(
        files,
        preprocessing_stats=stats,
        input_signals=diag_names,
        chunk_duration_s=chunk_s,
        step_size_s=chunk_s,
        warmup_s=args.warmup_s,
        prediction_mode=False,
        lengths_cache_path=args.output_dir
        / f"lengths_c3_{args.n_steps}steps.pt",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Chunks: {len(ds)}  batches: {len(loader)}  "
                f"scanning up to {args.max_batches} batches")

    # ── Accumulate per-sample cos pairs ───────────────────────────────
    # For each (sample, modality) we accumulate four per-sample scalars:
    #   - sig_cos_raw  : unmasked, for backwards-comparison with the first run
    #   - sig_cos_mask : mask-aware cos_sim on signal
    #   - tok_cos      : standard cos_sim on tokenizer output
    #   - valid_frac   : fraction of positions valid in BOTH windows
    sig_raw_acc: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    sig_masked_acc: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    tok_acc: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}
    valid_frac_acc: Dict[str, List[torch.Tensor]] = {n: [] for n in diag_names}

    n_batches_done = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        for k in range(args.n_steps):
            for name in diag_names:
                if name not in batch:
                    continue
                fs = SAMPLE_RATES_HZ[name]
                raw_t = _slice_window(batch[name].float(), fs, k)
                raw_t1 = _slice_window(batch[name].float(), fs, k + 1)

                mask_key = f"{name}_mask"
                upstream_t = (
                    _slice_window(batch[mask_key], fs, k)
                    if mask_key in batch
                    else None
                )
                upstream_t1 = (
                    _slice_window(batch[mask_key], fs, k + 1)
                    if mask_key in batch
                    else None
                )
                joint_mask = _joint_valid_mask(
                    raw_t, raw_t1, upstream_t, upstream_t1
                )

                # NaN-clean for downstream numerics (dataset already zeros
                # masked positions, but defensive NaN scrub is cheap).
                win_t = _nanclean(raw_t)
                win_t1 = _nanclean(raw_t1)

                tok_t = model.diag_tokenizers[name](win_t)
                tok_t1 = model.diag_tokenizers[name](win_t1)

                sig_raw_acc[name].append(_cos(win_t, win_t1).cpu())
                sig_masked_acc[name].append(
                    _masked_cos(win_t, win_t1, joint_mask).cpu()
                )
                tok_acc[name].append(_cos(tok_t, tok_t1).cpu())
                valid_frac_acc[name].append(_valid_fraction(joint_mask).cpu())
        n_batches_done += 1
        if n_batches_done % 50 == 0:
            logger.info(
                f"  batch {n_batches_done}/{min(args.max_batches, len(loader))}"
            )

    logger.info(f"Accumulated over {n_batches_done} batches.")

    # ── Per-modality summary + Spearman ───────────────────────────────
    # We report two Spearman values per modality:
    #   - raw      : unmasked sig_cos vs tok_cos, all pairs (matches first run)
    #   - masked   : mask-aware sig_cos vs tok_cos, restricted to pairs with
    #                joint valid-fraction > --min_valid_fraction (default 0.5)
    # Plus the mean missing-fraction so we can see which modalities are
    # dominated by zero-filled positions.
    summary: Dict[str, Dict[str, float]] = {}
    logger.info("")
    logger.info(
        f"{'modality':<23} {'n_raw':>6} {'n_keep':>6}  "
        f"{'valid%':>6}  "
        f"{'sig_raw':>7} {'sig_msk':>7} {'tok':>7}  "
        f"{'sp_raw':>7} {'sp_mask':>8}"
    )
    logger.info("-" * 100)
    for name in diag_names:
        if not sig_raw_acc[name]:
            logger.info(f"{name:<23} -- no data --")
            continue
        sig_raw = torch.cat(sig_raw_acc[name]).numpy()
        sig_msk = torch.cat(sig_masked_acc[name]).numpy()
        tok = torch.cat(tok_acc[name]).numpy()
        vf = torch.cat(valid_frac_acc[name]).numpy()

        finite_all = (
            np.isfinite(sig_raw) & np.isfinite(sig_msk)
            & np.isfinite(tok) & np.isfinite(vf)
        )
        sig_raw = sig_raw[finite_all]
        sig_msk = sig_msk[finite_all]
        tok = tok[finite_all]
        vf = vf[finite_all]

        n_raw = int(len(sig_raw))
        keep = vf >= args.min_valid_fraction
        n_keep = int(keep.sum())
        if n_raw < 3:
            logger.info(f"{name:<23} -- too few finite pairs --")
            continue

        # Raw Spearman across ALL finite pairs (backwards-comparable).
        sp_raw, _ = spearmanr(sig_raw, tok)

        # Masked Spearman across pairs with enough valid content.
        if n_keep >= 3:
            sp_mask, _ = spearmanr(sig_msk[keep], tok[keep])
            sp_mask_f = float(sp_mask)
        else:
            sp_mask_f = float("nan")

        summary[name] = {
            "n_raw": n_raw,
            "n_keep": n_keep,
            "valid_frac_mean": float(vf.mean()),
            "valid_frac_std": float(vf.std()),
            "sig_raw_mean": float(sig_raw.mean()),
            "sig_msk_mean": float(sig_msk[keep].mean()) if n_keep else float("nan"),
            "tok_mean": float(tok.mean()),
            "spearman_raw": float(sp_raw),
            "spearman_masked": sp_mask_f,
        }
        logger.info(
            f"{name:<23} {n_raw:>6d} {n_keep:>6d}  "
            f"{vf.mean():>5.1%}  "
            f"{sig_raw.mean():>+7.4f} "
            f"{(sig_msk[keep].mean() if n_keep else float('nan')):>+7.4f} "
            f"{tok.mean():>+7.4f}  "
            f"{sp_raw:>+7.4f} "
            f"{sp_mask_f:>+8.4f}"
        )

    # Save summary + raw accumulators early so a later printing or plotting
    # crash doesn't cost us the run. Full rerun is ~17 min on CPU.
    results = {
        "checkpoint": str(args.checkpoint),
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "summary": summary,
        "n_batches": n_batches_done,
        "args": vars(args),
    }
    results_path = args.output_dir / "latent_continuity_results.pt"
    torch.save(results, results_path)
    logger.info(f"Results saved (early): {results_path}")

    # ── Verdict line vs plan threshold ────────────────────────────────
    # Use the MASKED Spearman — that's the C3 question without the
    # missing-data confound.
    sp_values = [
        v["spearman_masked"]
        for v in summary.values()
        if np.isfinite(v["spearman_masked"])
    ]
    if sp_values:
        lo, hi = min(sp_values), max(sp_values)
        logger.info("")
        logger.info(
            f"Masked Spearman range: [{lo:+.3f}, {hi:+.3f}] across "
            f"{len(sp_values)} modalities "
            f"(pairs filtered to valid_frac ≥ {args.min_valid_fraction})."
        )
        thr_success = 0.5
        thr_failure = 0.0
        if lo > thr_success:
            logger.info(
                f"  ✓ VERDICT: all masked Spearman > {thr_success}. End-to-end "
                "training produced temporally smooth tokenizers on valid data. "
                "C3 claim supported."
            )
        elif hi <= thr_failure:
            logger.info(
                f"  ✗ VERDICT: no modality exceeds masked Spearman {thr_failure}. "
                "End-to-end tokenizers are as geometrically unordered as the "
                "AE baselines on valid data. C3 claim fails for this checkpoint."
            )
        else:
            logger.info(
                f"  ? VERDICT: mixed — some modalities below the {thr_success} "
                "threshold, some above. Stage 2 may improve the lagging ones."
            )

    # ── Scatter plot (masked-sig_cos vs tok_cos, valid-filtered pairs) ─
    n_mod = len(summary)
    if n_mod > 0:
        n_cols = min(3, n_mod)
        n_rows = (n_mod + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
        )
        for idx, name in enumerate(summary.keys()):
            ax = axes[idx // n_cols][idx % n_cols]
            sig_msk = torch.cat(sig_masked_acc[name]).numpy()
            tok = torch.cat(tok_acc[name]).numpy()
            vf = torch.cat(valid_frac_acc[name]).numpy()
            finite = np.isfinite(sig_msk) & np.isfinite(tok) & np.isfinite(vf)
            sig_msk = sig_msk[finite]
            tok = tok[finite]
            vf = vf[finite]
            keep = vf >= args.min_valid_fraction
            ax.scatter(
                sig_msk[keep], tok[keep], s=6, alpha=0.35,
                edgecolors="none", c="C0", label="kept",
            )
            if (~keep).any():
                ax.scatter(
                    sig_msk[~keep], tok[~keep], s=6, alpha=0.15,
                    edgecolors="none", c="C3",
                    label=f"valid<{args.min_valid_fraction:.0%}",
                )
            lo = -1.0
            hi = 1.0
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            s_mask = summary[name]["spearman_masked"]
            s_raw = summary[name]["spearman_raw"]
            vf_mean = summary[name]["valid_frac_mean"]
            ax.set_title(
                f"{name}\n"
                f"spearman_masked={s_mask:+.3f}  "
                f"raw={s_raw:+.3f}  valid={vf_mean:.0%}",
                fontsize=9,
            )
            ax.set_xlabel("signal_cos (masked)")
            ax.set_ylabel("token_cos")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7, loc="lower right")
        for idx in range(n_mod, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")
        fig.suptitle(
            "E2E tokenizer latent continuity — mask-aware signal_cos vs "
            "token_cos between consecutive 50 ms windows",
            y=1.02,
        )
        fig.tight_layout()
        plot_path = args.output_dir / "latent_continuity_scatter.png"
        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
        logger.info(f"Scatter plot: {plot_path}")



if __name__ == "__main__":
    main()