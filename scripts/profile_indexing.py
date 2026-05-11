#!/usr/bin/env python3
"""
CPU-only profiler for the file-length indexing pass that train_e2e jobs do
in build_datasets().

Replicates train_e2e_stage1.py's resolve_shot_files() and dataset construction,
times only the indexing step, and reports total wall time and files/sec
throughput. Use this to:

  - Predict how long indexing will take on N files before launching training.
  - Pre-populate the lengths cache so subsequent training jobs skip the wall.

Usage:
    # Quick smoke (10 files):
    python scripts/profile_indexing.py --max_files 10

    # Full pass, write cache to a known location:
    python scripts/profile_indexing.py \
        --cache_dir runs/lengths_cache_e2e_stage1

    # Don't write the cache (pure measurement):
    python scripts/profile_indexing.py --no_cache

CPU-only: imports torch but never touches CUDA. Pure h5py + numpy I/O on Lustre.
"""
import argparse
import logging
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Make sure we can import the project package without installing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# These imports must come after the path tweak. Note: TokamakMultiFileDataset
# pulls in torch but only uses CPU paths during indexing.
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("profile_indexing")


# Defaults match train_e2e_stage1.py's build_configs() for stage1.
DEFAULT_DIAGNOSTICS = [
    "ts_core_density", "ts_core_temp", "ts_tangential_density",
    "ts_tangential_temp", "cer_ti", "cer_rot", "mse", "filterscopes",
]
DEFAULT_ACTUATORS = [
    "pin", "beam_voltage", "ech_power", "ech_tor_angle", "ech_pol_angle",
    "ech_polarization", "gas_flow", "gas_raw", "rmp",
]


def resolve_shot_files(
    data_dir: Path,
    max_files: Optional[int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """Mirror train_e2e_stage1.resolve_shot_files for the no-YAML branch.

    Identical seeding and split logic so the returned file lists are byte-for-
    byte the same as what training would index.
    """
    rng = random.Random(seed)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    rng.shuffle(all_files)
    n = len(all_files)
    if n == 0:
        return [], []
    n_val = max(1, int(val_fraction * n))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]
    if max_files is not None:
        train_files = train_files[:max_files]
        val_files = val_files[: max(1, max_files // 4)]
    return train_files, val_files


def time_indexing(
    label: str,
    files: List[Path],
    cache_path: Optional[Path],
    chunk_duration_s: float,
    prediction_horizon_s: float,
    step_size_s: float,
    warmup_s: float,
    diagnostic_names: List[str],
    actuator_names: List[str],
) -> dict:
    """Build a TokamakMultiFileDataset and time only the indexing pass."""
    logger.info(f"[{label}] indexing {len(files)} files…")
    t0 = time.perf_counter()
    ds = TokamakMultiFileDataset(
        files,
        chunk_duration_s=chunk_duration_s,
        prediction_mode=True,
        prediction_horizon_s=prediction_horizon_s,
        step_size_s=step_size_s,
        warmup_s=warmup_s,
        preprocessing_stats={},
        input_signals=diagnostic_names,
        target_signals=diagnostic_names + actuator_names,
        lengths_cache_path=cache_path,
    )
    dt = time.perf_counter() - t0

    n_total = len(files)
    n_valid = len(ds._valid_indices)
    n_skipped = n_total - n_valid
    n_chunks = int(ds._cumulative_lengths[-1]) if n_valid > 0 else 0
    rate = (n_total / dt) if dt > 0 else float("inf")

    logger.info(
        f"[{label}] {n_total} files in {dt:.2f}s  "
        f"({rate:.2f} files/s)  "
        f"valid={n_valid} skipped={n_skipped} total_chunks={n_chunks}"
    )
    if cache_path is not None:
        logger.info(f"[{label}] cache written: {cache_path}")
    return dict(
        label=label,
        n_total=n_total,
        n_valid=n_valid,
        n_skipped=n_skipped,
        n_chunks=n_chunks,
        wall_s=dt,
        files_per_s=rate,
        cache_path=str(cache_path) if cache_path else None,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Profile build_datasets indexing throughput (CPU-only)."
    )
    ap.add_argument(
        "--data_dir", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model"),
    )
    ap.add_argument("--max_files", type=int, default=None,
                    help="Cap on training files (default: all). val_files is "
                    "max_files // 4 to mirror train_e2e_stage1.")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk_duration_s", type=float, default=0.05)
    ap.add_argument("--prediction_horizon_s", type=float, default=0.05)
    ap.add_argument("--step_size_s", type=float, default=0.01)
    ap.add_argument("--warmup_s", type=float, default=1.0)
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="Where to save the lengths cache. Default: a unique "
                    "tempdir, so every run is a cold cache miss (the point of "
                    "this profiler). Set to a stable path to persist the cache "
                    "for training jobs.")
    ap.add_argument("--no_cache", action="store_true",
                    help="Skip writing the cache entirely.")
    ap.add_argument("--diagnostic_names", type=str, default=None,
                    help="Comma-separated list. Default: stage1 diagnostics.")
    ap.add_argument("--actuator_names", type=str, default=None,
                    help="Comma-separated list. Default: stage1 actuators.")
    ap.add_argument("--skip_val", action="store_true",
                    help="Profile train indexing only.")
    args = ap.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"data_dir not found: {args.data_dir}")

    diagnostic_names = (
        args.diagnostic_names.split(",") if args.diagnostic_names
        else DEFAULT_DIAGNOSTICS
    )
    actuator_names = (
        args.actuator_names.split(",") if args.actuator_names
        else DEFAULT_ACTUATORS
    )

    logger.info(f"data_dir = {args.data_dir}")
    logger.info(f"diagnostics = {diagnostic_names}")
    logger.info(f"actuators   = {actuator_names}")
    logger.info(
        f"chunk_duration_s={args.chunk_duration_s} "
        f"prediction_horizon_s={args.prediction_horizon_s} "
        f"step_size_s={args.step_size_s} warmup_s={args.warmup_s}"
    )

    train_files, val_files = resolve_shot_files(
        args.data_dir, args.max_files, args.val_fraction, args.seed,
    )
    logger.info(f"Resolved files — train: {len(train_files)}  val: {len(val_files)}")
    if not train_files:
        raise SystemExit(f"No *_processed.h5 files matched {args.data_dir}")

    # Cache directory selection.
    if args.no_cache:
        cache_dir = None
        logger.info("Cache: disabled (--no_cache)")
    elif args.cache_dir is not None:
        cache_dir = args.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache dir: {cache_dir}")
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="profile_indexing_"))
        logger.info(f"Cache dir (tempdir, cold-miss every run): {cache_dir}")

    train_cache = (cache_dir / "lengths_e2e_stage1_train.pt") if cache_dir else None
    val_cache = (cache_dir / "lengths_e2e_stage1_val.pt") if cache_dir else None

    results = []
    results.append(time_indexing(
        label="train",
        files=train_files,
        cache_path=train_cache,
        chunk_duration_s=args.chunk_duration_s,
        prediction_horizon_s=args.prediction_horizon_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
    ))

    if val_files and not args.skip_val:
        results.append(time_indexing(
            label="val",
            files=val_files,
            cache_path=val_cache,
            chunk_duration_s=args.chunk_duration_s,
            prediction_horizon_s=args.prediction_horizon_s,
            step_size_s=args.step_size_s,
            warmup_s=args.warmup_s,
            diagnostic_names=diagnostic_names,
            actuator_names=actuator_names,
        ))

    # ─── Aggregate summary ───────────────────────────────────────────────
    total_files = sum(r["n_total"] for r in results)
    total_skipped = sum(r["n_skipped"] for r in results)
    total_chunks = sum(r["n_chunks"] for r in results)
    total_wall = sum(r["wall_s"] for r in results)
    overall_rate = (total_files / total_wall) if total_wall > 0 else float("inf")

    print()
    print("=" * 68)
    print(" INDEXING PROFILE SUMMARY")
    print("=" * 68)
    for r in results:
        print(
            f"  {r['label']:<6}  files={r['n_total']:<6} "
            f"valid={r['n_valid']:<6} skipped={r['n_skipped']:<4} "
            f"chunks={r['n_chunks']:<8} "
            f"time={r['wall_s']:>7.2f}s  rate={r['files_per_s']:>6.2f} files/s"
        )
    print("-" * 68)
    print(
        f"  {'TOTAL':<6}  files={total_files:<6} "
        f"valid={total_files - total_skipped:<6} "
        f"skipped={total_skipped:<4} "
        f"chunks={total_chunks:<8} "
        f"time={total_wall:>7.2f}s  rate={overall_rate:>6.2f} files/s"
    )
    print("=" * 68)

    # Predicted full-dataset cost.
    if args.max_files is not None:
        # Estimate total dataset size by re-globbing without the cap.
        full_count = len(sorted(args.data_dir.glob("*_processed.h5")))
        if full_count > total_files and overall_rate > 0:
            predicted = full_count / overall_rate
            print(
                f"  Predicted full-dataset indexing ({full_count} files): "
                f"{predicted:.0f}s = {predicted / 60:.1f} min"
            )
            print()


if __name__ == "__main__":
    main()
