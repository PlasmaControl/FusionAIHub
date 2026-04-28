"""One-off probe: where does `TokamakMultiFileDataset.__getitem__` spend time?

Builds the exact Stage 1 dataset config (same signals, same step_size_s,
chunk_duration_s, warmup_s, preprocessing_stats) against a handful of real
files, times N=200 random `__getitem__` calls in the main process (no
workers, no DataLoader), and reports:

  - total wall time and per-call median / p90 / max
  - a cProfile top-20 by cumulative time so we can see whether the cost is
    HDF5 reads, `F.interpolate` resampling, per-element preprocessing,
    NaN handling, or something structural

Run: ``pixi run python scripts/training/probe_stage1_loading.py``
"""

from __future__ import annotations

import cProfile
import pstats
import random
import statistics
import time
from pathlib import Path
from typing import List

import torch

# Stage 1 uses these — import constants so the probe can't drift from prod.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_e2e_stage1 import (  # type: ignore
    SLOW_TS_MODALITIES,
    FAST_TS_MODALITIES,
    ACTUATOR_MODALITIES,
    resolve_shot_files,
)
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset


def main() -> None:
    data_dir = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
    stats_path = Path(
        "/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt"
    )
    n_samples = 200        # calls to `__getitem__`
    rng = random.Random(42)

    diag_names = [n for n, _ in SLOW_TS_MODALITIES] + [
        n for n, _, _ in FAST_TS_MODALITIES
    ]
    act_names = [n for n, _ in ACTUATOR_MODALITIES]
    input_signals = diag_names
    target_signals = diag_names + act_names

    # Use exactly the same file split the Stage 1 job used (seed=42,
    # val_fraction=0.1). Reuses the existing lengths cache so dataset
    # construction is ~1 s, not ~10 min.
    files, _ = resolve_shot_files(
        data_dir=data_dir,
        train_shots_yaml=None,
        val_shots_yaml=None,
        max_files=None,
        val_fraction=0.1,
        seed=42,
    )
    print(f"Using {len(files)} train files from {data_dir}")

    print("Loading preprocessing_stats…")
    stats = torch.load(stats_path, weights_only=False)

    print("Building dataset…")
    t0 = time.time()
    ds = TokamakMultiFileDataset(
        files,
        lengths_cache_path=Path(
            "/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/"
            "runs/e2e_stage1/lengths_e2e_stage1_train.pt"
        ),
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=0.05,
        step_size_s=0.01,
        warmup_s=1.0,
        preprocessing_stats=stats,
        input_signals=input_signals,
        target_signals=target_signals,
    )
    print(
        f"Dataset built in {time.time() - t0:.2f} s; "
        f"len={len(ds)} chunks across {len(files)} files."
    )

    idxs = [rng.randrange(len(ds)) for _ in range(n_samples)]

    # Warm-up: a few calls to open file handles + prime caches.
    print("Warm-up (10 calls)…")
    for i in idxs[:10]:
        _ = ds[i]

    # Wall-time pass.
    print(f"Timing {n_samples} __getitem__ calls…")
    per_call_s: List[float] = []
    t0 = time.time()
    for i in idxs:
        s = time.perf_counter()
        _ = ds[i]
        per_call_s.append(time.perf_counter() - s)
    total = time.time() - t0

    per_call_s.sort()
    print()
    print("=" * 60)
    print("WALL TIME RESULTS")
    print("=" * 60)
    print(f"Total   : {total:.2f} s for {n_samples} calls")
    print(f"Mean    : {1000 * total / n_samples:.1f} ms/call")
    print(f"Median  : {1000 * per_call_s[n_samples // 2]:.1f} ms/call")
    print(f"p90     : {1000 * per_call_s[int(0.9 * n_samples)]:.1f} ms/call")
    print(f"p99     : {1000 * per_call_s[int(0.99 * n_samples)]:.1f} ms/call")
    print(f"Max     : {1000 * per_call_s[-1]:.1f} ms/call")
    print()
    per_batch_256 = 256 * (total / n_samples)
    per_sample_throughput = n_samples / total
    print(f"Extrapolated: 1 batch of 256 samples = {per_batch_256:.1f} s")
    print(f"Samples/sec (single-threaded): {per_sample_throughput:.1f}")
    print(f"With 16 workers: {16 * per_sample_throughput:.1f} samples/sec "
          f"(=> {256 / (16 * per_sample_throughput):.2f} s per b=256 batch)")
    print()

    # cProfile pass on a smaller sample — cProfile adds overhead.
    print("=" * 60)
    print("cProfile on 50 calls — top 20 by cumulative time")
    print("=" * 60)
    profiler = cProfile.Profile()
    profiler.enable()
    for i in idxs[:50]:
        _ = ds[i]
    profiler.disable()
    stats_obj = pstats.Stats(profiler).sort_stats("cumulative")
    stats_obj.print_stats(20)


if __name__ == "__main__":
    main()
