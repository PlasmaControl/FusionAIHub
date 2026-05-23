#!/usr/bin/env python3
"""
CPU-only builder for the dataset indexing caches that ``train_e2e`` jobs
expect on disk.

Runs the per-file HDF5 scans (video-presence + chunk-count) **in parallel**
via a process pool, then writes cache files in the exact format the
training runtime expects (``filter_video_present_files`` and
``_load_or_compute_lengths`` in ``multi_file_dataset.py``). Training itself
never spawns a process pool — the parallelism lives here on purpose, where
CUDA / NCCL are not initialised, so the ``fork`` foot-gun cannot bite.

Usage:
    # Quick smoke (10 files):
    python scripts/build_dataset_cache.py --max_files 10

    # Full pass, write cache to a known location:
    python scripts/build_dataset_cache.py \
        --cache_dir /lustre/orion/fus187/proj-shared/foundation_model_meta

    # Don't write the cache (pure timing measurement):
    python scripts/build_dataset_cache.py --no_cache

CPU-only: imports torch only for cache I/O, never touches CUDA. Pure h5py +
numpy + multiprocessing for the scans.
"""
import argparse
import logging
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Make sure we can import the project package without installing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Pulled in for SIGNAL_CONFIGS / MOVIE_CONFIGS only (these are class-level
# @dataclass lists, picklable, replicated into each worker process via
# ProcessPoolExecutor's pickle bridge).
from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    TokamakH5Dataset,
)


# ── Worker functions ────────────────────────────────────────────────────
# Must be top-level (picklable) for ProcessPoolExecutor. They re-import
# h5py inside the function so each worker process owns its HDF5 library
# state, matching the runtime behaviour of one shot-file open per call.


def _video_present_worker(args: tuple) -> Optional[str]:
    """Return ``str(path)`` if any requested camera has non-empty data."""
    path, camera_names = args
    try:
        with h5py.File(path, "r") as f:
            for cam in camera_names:
                if cam not in f or "ydata" not in f[cam]:
                    continue
                yd = f[cam]["ydata"]
                xd = f[cam].get("xdata")
                if (
                    yd.size > 0
                    and yd.ndim == 4
                    and xd is not None
                    and xd.size >= 2
                ):
                    return str(path)
    except Exception:
        return None
    return None


def _compute_length_worker(args: tuple) -> int:
    """Return per-file chunk count.

    Inlines the duration arithmetic from
    ``TokamakH5Dataset._compute_duration`` so the worker is self-contained
    and does not need a dataset instance.
    """
    (
        path,
        signal_configs,
        movie_configs,
        max_duration_s,
        warmup_s,
        chunk_duration_s,
        prediction_horizon_s,
        step_size_s,
        prediction_mode,
    ) = args
    try:
        with h5py.File(path, "r") as f:
            duration = 0.0
            for cfg in signal_configs:
                for key_path in cfg.hdf5_keys:
                    try:
                        curr = f
                        for part in key_path.split("/"):
                            curr = curr[part]
                        xdata_s = curr["xdata"][:]
                        if len(xdata_s) < 2:
                            continue
                        duration = max(duration, float(xdata_s[-1]))
                        break
                    except (KeyError, ValueError):
                        continue
            for mcfg in movie_configs:
                for key_path in mcfg.hdf5_keys:
                    try:
                        curr = f
                        for part in key_path.split("/"):
                            curr = curr[part]
                        xdata_ms = curr["xdata"][:]
                        if len(xdata_ms) < 2:
                            continue
                        duration = max(duration, float(xdata_ms[-1]))
                        break
                    except (KeyError, ValueError):
                        continue
        duration = min(duration, max_duration_s) - warmup_s
        if duration <= 0.0:
            return 0
        if prediction_mode:
            total_window = chunk_duration_s + prediction_horizon_s
            return max(
                0, int(np.floor((duration - total_window) / step_size_s)) + 1
            )
        if duration < chunk_duration_s:
            return 0
        return int(np.floor((duration - chunk_duration_s) / step_size_s)) + 1
    except OSError:
        return 0


# ── Parallel scan + cache-write helpers ─────────────────────────────────


def _atomic_torch_save(payload: dict, cache_path: Path) -> None:
    """Write ``payload`` to ``cache_path`` via ``.tmp`` + ``replace`` so a
    crashed write never leaves a half-written zip that the next
    ``torch.load`` would barf on."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(cache_path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(cache_path)


def parallel_video_presence_scan(
    paths: List[Path],
    camera_names: List[str],
    cache_path: Optional[Path],
    num_workers: int,
) -> List[Path]:
    """Return the subset of ``paths`` whose HDF5 has non-empty video data.

    Writes a cache file in the same format as
    ``multi_file_dataset.filter_video_present_files`` so training jobs
    hit it transparently.
    """
    paths_key = tuple(str(p) for p in paths)
    cameras_key = tuple(sorted(camera_names))
    ctx = mp.get_context("forkserver")
    tasks = [(p, camera_names) for p in paths]
    video_present: List[str] = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as exc:
        for result in tqdm(
            exc.map(_video_present_worker, tasks, chunksize=8),
            total=len(tasks),
            desc=f"Video presence ({num_workers} workers)",
        ):
            if result is not None:
                video_present.append(result)
    if cache_path is not None:
        _atomic_torch_save(
            {
                "paths_key": paths_key,
                "cameras_key": cameras_key,
                "video_present": video_present,
            },
            cache_path,
        )
    present = set(video_present)
    return [p for p in paths if str(p) in present]


def parallel_lengths_scan(
    paths: List[Path],
    signal_configs: list,
    movie_configs: list,
    max_duration_s: float,
    warmup_s: float,
    chunk_duration_s: float,
    prediction_horizon_s: float,
    step_size_s: float,
    prediction_mode: bool,
    cache_path: Optional[Path],
    num_workers: int,
) -> List[int]:
    """Return per-file chunk counts in input order. Writes cache in the
    same format as ``multi_file_dataset._load_or_compute_lengths`` so
    training jobs hit it transparently."""
    paths_as_str = [str(p) for p in paths]
    ctx = mp.get_context("forkserver")
    tasks = [
        (
            p,
            signal_configs,
            movie_configs,
            max_duration_s,
            warmup_s,
            chunk_duration_s,
            prediction_horizon_s,
            step_size_s,
            prediction_mode,
        )
        for p in paths
    ]
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as exc:
        lengths = list(
            tqdm(
                exc.map(_compute_length_worker, tasks, chunksize=8),
                total=len(tasks),
                desc=f"Computing lengths ({num_workers} workers)",
            )
        )
    if cache_path is not None:
        _atomic_torch_save(
            {"paths": paths_as_str, "lengths": lengths}, cache_path,
        )
    return lengths

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("build_dataset_cache")


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
    max_duration_s: float,
    num_workers: int,
) -> dict:
    """Run the parallel lengths scan and time it. Writes the cache in the
    on-disk format that the training-runtime dataset expects."""
    logger.info(f"[{label}] indexing {len(files)} files (workers={num_workers})…")
    t0 = time.perf_counter()
    lengths = parallel_lengths_scan(
        paths=files,
        signal_configs=TokamakH5Dataset.SIGNAL_CONFIGS,
        movie_configs=TokamakH5Dataset.MOVIE_CONFIGS,
        max_duration_s=max_duration_s,
        warmup_s=warmup_s,
        chunk_duration_s=chunk_duration_s,
        prediction_horizon_s=prediction_horizon_s,
        step_size_s=step_size_s,
        prediction_mode=True,
        cache_path=cache_path,
        num_workers=num_workers,
    )
    dt = time.perf_counter() - t0

    n_total = len(files)
    n_valid = sum(1 for n in lengths if n > 0)
    n_skipped = n_total - n_valid
    n_chunks = int(sum(lengths))
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
    ap.add_argument(
        "--use_video", nargs="*", default=[],
        help="Camera names to require present (e.g. 'tangtv'). Must match the "
        "training run's --use_video so the resulting lengths cache is keyed "
        "on the same path list. Empty (default) skips the video filter.",
    )
    ap.add_argument(
        "--video_cache_dir", type=Path, default=None,
        help="Where to write/read the video-presence cache. Defaults to "
        "--cache_dir so the training run can reuse it.",
    )
    ap.add_argument(
        "--num_workers", type=int,
        default=int(os.environ.get("INDEXING_WORKERS", "8")),
        help="Process-pool size for the parallel HDF5 scans (default 8, "
        "env override INDEXING_WORKERS). One worker per concurrent open; "
        "bumping this raises Lustre MDS pressure linearly.",
    )
    ap.add_argument(
        "--max_duration_s", type=float, default=12.0,
        help="Cap on shot duration used by the lengths arithmetic. Must "
        "match TokamakMultiFileDataset's default for the cache to be a "
        "drop-in for training.",
    )
    ap.add_argument(
        "--cache_name_prefix", type=str, default="lengths_e2e_stage1",
        help="Filename prefix for the lengths cache. Defaults to "
        "'lengths_e2e_stage1' (matches train_e2e_stage1.py's expected "
        "cache name). Override for other stages, e.g. "
        "'lengths_e2e_stage2_delta'. The lengths cache contents depend "
        "on (paths, prediction_horizon_s, chunk_duration_s, step_size_s, "
        "warmup_s) — stages with different windowing MUST use distinct "
        "prefixes to avoid overwriting each other's cache.",
    )
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
        cache_dir = Path(tempfile.mkdtemp(prefix="build_dataset_cache_"))
        logger.info(f"Cache dir (tempdir, cold-miss every run): {cache_dir}")

    # Apply video-presence filter BEFORE building the lengths cache so the
    # stored `paths` key matches what training will see at run time. Without
    # this, training (with --use_video) builds a smaller filtered list, the
    # cache's `paths` check fails, and the pre-warm is wasted.
    if args.use_video:
        video_cache_dir = args.video_cache_dir or cache_dir
        n_train_before = len(train_files)
        n_val_before = len(val_files)
        train_files = parallel_video_presence_scan(
            paths=train_files,
            camera_names=args.use_video,
            cache_path=(
                video_cache_dir / "video_present_train.pt"
                if video_cache_dir else None
            ),
            num_workers=args.num_workers,
        )
        val_files = parallel_video_presence_scan(
            paths=val_files,
            camera_names=args.use_video,
            cache_path=(
                video_cache_dir / "video_present_val.pt"
                if video_cache_dir else None
            ),
            num_workers=args.num_workers,
        )
        logger.info(
            f"Video-presence filter ({args.use_video}): "
            f"train {n_train_before} -> {len(train_files)}; "
            f"val {n_val_before} -> {len(val_files)}"
        )

    train_cache = (cache_dir / f"{args.cache_name_prefix}_train.pt") if cache_dir else None
    val_cache = (cache_dir / f"{args.cache_name_prefix}_val.pt") if cache_dir else None

    results = []
    results.append(time_indexing(
        label="train",
        files=train_files,
        cache_path=train_cache,
        chunk_duration_s=args.chunk_duration_s,
        prediction_horizon_s=args.prediction_horizon_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        max_duration_s=args.max_duration_s,
        num_workers=args.num_workers,
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
            max_duration_s=args.max_duration_s,
            num_workers=args.num_workers,
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
