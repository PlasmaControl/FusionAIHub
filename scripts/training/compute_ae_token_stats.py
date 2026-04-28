#!/usr/bin/env python
"""
Precompute per-modality AE token normalization statistics.

Runs all frozen AE encoders over the training set and saves per-element
mean and std for each modality.  These are used to standardize AE tokens
to zero mean, unit variance before they enter the foundation model.

Usage:
    pixi run python scripts/training/compute_ae_token_stats.py \
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model/ \
        --stats_path /projects/EKOLEMEN/foundation_model/preprocessing_stats.pt \
        --ae_checkpoint_dir /projects/EKOLEMEN/foundation_model/ \
        --output_path /projects/EKOLEMEN/foundation_model/ae_token_stats.pt
"""

from pathlib import Path
import argparse
import logging

import torch

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader,
)
from train_foundation_model import (
    DIAGNOSTIC_CONFIGS, ACTUATOR_CONFIGS, load_ae, split_window,
    WINDOW_S, DT_S,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-modality AE token normalization stats")
    parser.add_argument("--data_dir",
                        default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument("--stats_path",
                        default="/projects/EKOLEMEN/foundation_model/"
                                "preprocessing_stats.pt")
    parser.add_argument("--ae_checkpoint_dir",
                        default="/projects/EKOLEMEN/foundation_model/")
    parser.add_argument("--output_path",
                        default="/projects/EKOLEMEN/foundation_model/"
                                "ae_token_stats.pt")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Limit number of HDF5 files. 0 = all files.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # Load AEs
    ae_models = {}
    ae_dir = Path(args.ae_checkpoint_dir)
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        if "ae_checkpoint_path" in cfg:
            ckpt = Path(cfg["ae_checkpoint_path"])
        else:
            ckpt = ae_dir / f"{name}_{cfg['model_type']}" / "checkpoint_best.pth"
        if not ckpt.exists():
            logger.warning(f"AE not found for '{name}': {ckpt} — skipping")
            continue
        ae_models[name] = load_ae(name, cfg, ckpt)

    if not ae_models:
        raise RuntimeError("No AE checkpoints found.")

    # Dataset — single-step chunks (context window only)
    stats = torch.load(args.stats_path, weights_only=False)
    all_signals = list(ae_models.keys()) + list(ACTUATOR_CONFIGS.keys())

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    if args.max_files > 0:
        all_files = all_files[:args.max_files]
    logger.info(f"Using {len(all_files)} files")

    CHUNK_S = WINDOW_S + DT_S  # minimal chunk: context + 1 target
    ds = TokamakMultiFileDataset(
        all_files,
        lengths_cache_path="lengths_ae_stats.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        prediction_mode=False,
    )
    loader = make_dataloader(
        ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True,
    )
    logger.info(f"Chunks: {len(ds)}")

    # Accumulate running statistics (Welford's online algorithm)
    count = {}
    mean_acc = {}
    m2_acc = {}

    for batch_idx, batch in enumerate(loader):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Extract context signals
        ctx_signals = {}
        for name, cfg in DIAGNOSTIC_CONFIGS.items():
            if name not in batch or name not in ae_models:
                continue
            ctx, _ = split_window(batch[name], cfg["target_fs"], n_rollout=1)
            ctx_signals[name] = ctx

        # Encode
        with torch.no_grad():
            for name, ae in ae_models.items():
                if name not in ctx_signals:
                    continue
                z = ae.encoder(ctx_signals[name])  # [B, n_tokens, d_lat]
                z = z.clamp(-50, 50)

                B = z.shape[0]
                # Flatten batch: treat each sample independently
                for i in range(B):
                    sample = z[i]  # [n_tokens, d_lat]

                    # Skip samples with any NaN/Inf — a single bad
                    # sample poisons Welford's running statistics.
                    if not torch.isfinite(sample).all():
                        continue

                    if name not in count:
                        count[name] = 0
                        mean_acc[name] = torch.zeros_like(sample)
                        m2_acc[name] = torch.zeros_like(sample)

                    count[name] += 1
                    delta = sample - mean_acc[name]
                    mean_acc[name] += delta / count[name]
                    delta2 = sample - mean_acc[name]
                    m2_acc[name] += delta * delta2

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Processed {batch_idx + 1} batches "
                        f"({count.get(next(iter(ae_models)), 0)} samples)")

    # Finalize statistics
    result = {}
    for name in count:
        mean = mean_acc[name].cpu()
        std = (m2_acc[name] / max(count[name] - 1, 1)).sqrt().cpu()
        std = std.clamp(min=1e-6)  # prevent division by zero

        result[name] = {"mean": mean, "std": std}

        logger.info(f"{name}: n={count[name]}, "
                    f"mean_norm={mean.norm():.3f}, "
                    f"std_mean={std.mean():.4f}, "
                    f"std_min={std.min():.4f}, "
                    f"std_max={std.max():.4f}")

    torch.save(result, args.output_path)
    logger.info(f"Saved AE token stats to {args.output_path}")


if __name__ == "__main__":
    main()
