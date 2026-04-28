#!/usr/bin/env python
"""
Debug: signal-space vs AE-latent-space cosine similarity between
consecutive 500ms windows, per modality.

Motivation
----------
If latent states z_t and z_{t+1} are very close (cos ~ 1), then a
`latent_skip` rollout (run backbone in latent space, decode only for
loss) is plausible: the backbone is asked to make small updates in a
continuous manifold.  If latent states jump around between consecutive
windows, the backbone cannot reasonably operate without re-encoding.

The signal-space cosine is included as a sanity anchor — it reports
the underlying slow/fast nature of the raw signal itself.
"""

from pathlib import Path
import argparse
import logging
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader,
)
from train_foundation_model import (
    DIAGNOSTIC_CONFIGS,
    ACTUATOR_CONFIGS,
    load_ae,
    encode_batch,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WINDOW_S: float = 0.05
DT_S: float = 0.05


def _slice_window(
    signal: torch.Tensor, target_fs: float, k: int,
) -> torch.Tensor:
    """Return the k-th 500ms window of *signal*, stride DT_S."""
    n_win = round(WINDOW_S * target_fs)
    n_dt = round(DT_S * target_fs)
    start = k * n_dt
    return signal[..., start:start + n_win]


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch cosine similarity over flattened feature dims → [B]."""
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="AE latent continuity between consecutive windows")
    parser.add_argument("--data_dir",
                        default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument("--stats_path",
                        default="/projects/EKOLEMEN/foundation_model/"
                                "preprocessing_stats.pt")
    parser.add_argument("--ae_checkpoint_dir",
                        default="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/runs/")
    parser.add_argument("--ae_token_stats_path",
                        default="/projects/EKOLEMEN/foundation_model/"
                                "ae_token_stats.pt")
    parser.add_argument("--max_files", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=1,
                        help="Number of DT_S steps → n_steps cos pairs")
    parser.add_argument("--max_batches", type=int, default=2000)
    parser.add_argument("--warmup_s", type=float, default=1.0)
    parser.add_argument("--plot_path", type=str,
                        default="latent_continuity.png")
    args = parser.parse_args()

    chunk_s = WINDOW_S + args.n_steps * DT_S

    # --- Load AEs ---
    ae_models = {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        ae_dir = Path(args.ae_checkpoint_dir)
        if "ae_checkpoint_path" in cfg:
            ckpt_path = Path(cfg["ae_checkpoint_path"])
        else:
            ckpt_path = ae_dir / f"{name}_{cfg['model_type']}" \
                / "checkpoint_best.pth"
        if not ckpt_path.exists():
            logger.warning(f"AE not found for '{name}': {ckpt_path}")
            continue
        ae_models[name] = load_ae(name, cfg, ckpt_path)
    if not ae_models:
        raise RuntimeError("No AE checkpoints found.")

    active = {k: v for k, v in DIAGNOSTIC_CONFIGS.items() if k in ae_models}
    logger.info(f"Active modalities: {list(active.keys())}")

    ae_token_stats = None
    if args.ae_token_stats_path is not None:
        p = Path(args.ae_token_stats_path)
        if p.exists():
            ae_token_stats = torch.load(p, weights_only=False)

    # --- Dataset ---
    stats = torch.load(args.stats_path, weights_only=False)
    all_signals = list(active.keys()) + list(ACTUATOR_CONFIGS.keys())

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    random.seed(42)
    random.shuffle(all_files)
    if args.max_files is not None:
        all_files = all_files[:args.max_files]
    ds = TokamakMultiFileDataset(
        all_files,
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=chunk_s,
        step_size_s=chunk_s,
        warmup_s=args.warmup_s,
        prediction_mode=False,
        lengths_cache_path="lengths_debug_latent_continuity.pt",
    )
    loader = make_dataloader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False)
    logger.info(f"Chunks: {len(ds)}  batches/epoch: {len(loader)}")

    # accum[name][k] = list of cos values over batches
    sig_accum = {m: [[] for _ in range(args.n_steps)] for m in active}
    lat_accum = {m: [[] for _ in range(args.n_steps)] for m in active}

    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        for k in range(args.n_steps):
            win_t, win_t1 = {}, {}
            for m, cfg in active.items():
                if m not in batch:
                    continue
                fs = cfg["target_fs"]
                win_t[m] = _slice_window(batch[m], fs, k)
                win_t1[m] = _slice_window(batch[m], fs, k + 1)

            z_t = encode_batch(ae_models, win_t, ae_token_stats=ae_token_stats)
            z_t1 = encode_batch(ae_models, win_t1, ae_token_stats=ae_token_stats)

            for m in active:
                if m not in win_t or m not in z_t:
                    continue
                sig_cos = _cos(win_t[m], win_t1[m])
                lat_cos = _cos(z_t[m], z_t1[m])
                sig_accum[m][k].append(sig_cos.cpu())
                lat_accum[m][k].append(lat_cos.cpu())

        n_batches += 1
        if n_batches >= args.max_batches:
            break

    # --- Report ---
    logger.info("\n"
                f"Results over {n_batches} batches "
                f"(batch_size={args.batch_size}, n_steps={args.n_steps})")
    logger.info("=" * 72)
    header = f"{'modality':<28} {'step':>4}  " \
             f"{'signal_cos':>20}  {'latent_cos':>20}"
    logger.info(header)
    logger.info("-" * 72)
    for m in active:
        for k in range(args.n_steps):
            if not sig_accum[m][k]:
                continue
            sig = torch.cat(sig_accum[m][k])
            lat = torch.cat(lat_accum[m][k])
            logger.info(
                f"{m:<28} {k:>4}  "
                f"{sig.mean().item():>7.4f} ± {sig.std().item():>5.4f}   "
                f"{lat.mean().item():>7.4f} ± {lat.std().item():>5.4f}"
            )
        logger.info("-" * 72)

    logger.info("\nAggregate (across all steps and batches):")
    logger.info("=" * 72)
    flat_sig, flat_lat = {}, {}
    for m in active:
        sig_all = torch.cat([c for step in sig_accum[m] for c in step])
        lat_all = torch.cat([c for step in lat_accum[m] for c in step])
        flat_sig[m] = sig_all.numpy()
        flat_lat[m] = lat_all.numpy()
        logger.info(
            f"{m:<28}        "
            f"sig={sig_all.mean().item():.4f} ± {sig_all.std().item():.4f}   "
            f"lat={lat_all.mean().item():.4f} ± {lat_all.std().item():.4f}"
        )

    # --- Correlation: does latent_cos drop when signal_cos drops? ---
    logger.info("\nCorrelation signal_cos vs latent_cos "
                "(Pearson = linear; Spearman = rank/monotonic):")
    logger.info("=" * 72)
    corrs = {}
    for m in active:
        s, z = flat_sig[m], flat_lat[m]
        if len(s) < 3:
            continue
        # Pearson
        s_t = torch.tensor(s, dtype=torch.float32)
        z_t = torch.tensor(z, dtype=torch.float32)
        pearson = torch.corrcoef(torch.stack([s_t, z_t]))[0, 1].item()
        # Spearman (monotonic)
        sp_r, _ = spearmanr(s, z)
        corrs[m] = (pearson, float(sp_r))
        logger.info(
            f"{m:<28}   pearson={pearson:+.4f}   spearman={sp_r:+.4f}"
        )

    # --- Scatter plots ---
    n_mod = len(active)
    n_cols = min(3, n_mod)
    n_rows = (n_mod + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)
    for idx, m in enumerate(active):
        ax = axes[idx // n_cols][idx % n_cols]
        s, z = flat_sig[m], flat_lat[m]
        ax.scatter(s, z, s=6, alpha=0.35, edgecolors="none")
        lo = min(s.min(), z.min())
        hi = max(s.max(), z.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y=x")
        p, sp = corrs.get(m, (float("nan"), float("nan")))
        ax.set_title(f"{m}\n pearson={p:+.3f}  spearman={sp:+.3f}",
                     fontsize=9)
        ax.set_xlabel("signal_cos")
        ax.set_ylabel("latent_cos")
        ax.grid(alpha=0.3)
    for idx in range(n_mod, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle("Signal vs latent cosine similarity "
                 "between consecutive 50ms windows", y=1.02)
    fig.tight_layout()
    out = Path(args.plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info(f"\nWrote scatter plot → {out}")


if __name__ == "__main__":
    main()
