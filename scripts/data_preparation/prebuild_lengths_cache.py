"""Offline pre-build of the HORIZON-SPECIFIC lengths cache for the K-anneal B run.

WHY THIS EXISTS
---------------
Lever #1 (per-block dataset horizon) sets the K-anneal B block-0 dataset future
span to 0.7s (= K*chunk + pred = 10*0.05 + 0.2) instead of the max-K 4.2s. The
per-file window COUNT that TokamakMultiFileDataset caches
(``multi_file_dataset.py::_scan_lengths_local``) is a function of
``prediction_horizon_s`` — so it is HORIZON-SPECIFIC. A cold scan of the full
~7878-shot production set takes ~87 min; if that scan runs on rank 0 INSIDE a
multi-rank training job it blows past NCCL's 10-minute collective watchdog and
crashes all 64 ranks. So the horizon-specific cache MUST be built OFFLINE, in a
single process with NO torch.distributed / NCCL init.

This script constructs the TRAIN and VAL ``TokamakMultiFileDataset`` over the
FULL production shot set using the SAME code paths the trainer uses
(``resolve_shot_files`` + ``build_datasets``), so the resolved file lists and the
cache sidecar filenames (``lengths_e2e_stage1_{train,val}.pt``) are BYTE-IDENTICAL
to what the trainer will look up at runtime. Constructing each dataset triggers
the length scan and the atomic sidecar write.

HORIZON CONVENTION (matches EXPERIMENTS.md "LEVER #1 CHOSEN"):
    rollout_dataset_horizon_s(K) = K*chunk + pred_horizon = K*0.05 + 0.2
    block 0 / K=10 -> 0.7s   (this is what --train_horizon defaults to)

TRAIN vs VAL horizon — IMPORTANT
--------------------------------
The trainer builds the TRAIN dataset at ``dataset_horizon_s`` (= the value passed
via ``--rollout_dataset_horizon_s``, i.e. 0.7 for block 0) but builds the VAL
dataset at ``val_prediction_horizon_s = args.prediction_horizon_s`` = the MODEL
horizon 0.2 (train_e2e_stage1.py:3374; validate() stays single-step). The lengths
cache is keyed ONLY on the file-path list, NOT on the horizon — so a val cache
written at the wrong horizon would be silently loaded (paths match) and give the
wrong window count. We therefore build:
    * TRAIN cache at ``--train_horizon``  (default 0.7)
    * VAL   cache at ``--val_horizon``    (default 0.2 = the trainer's actual val horizon)
so BOTH sidecars the trainer looks up are correct. Override ``--val_horizon`` if a
future block changes validate()'s span. (This uses build_datasets' own
``val_prediction_horizon_s`` arg, mirroring the trainer exactly.)

USAGE (single process, no srun/NCCL):
    source scripts/slurm_frontier/_frontier_common.sh
    python scripts/data_preparation/prebuild_lengths_cache.py
Launched via a 1-node SLURM job (see the sbatch wrapper this script ships with).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "scripts", "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

# Reuse the trainer's OWN file resolver + dataset builder so the resolved file
# lists and the cache sidecar filenames are byte-identical to runtime.
from train_e2e_stage1 import resolve_shot_files, build_datasets  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_dir", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model"),
        help="Production shot dir (globbed for *_processed.h5). MUST match the "
             "trainer's --data_dir so the resolved file list is identical.")
    p.add_argument(
        "--stats_path", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model_meta/"
                     "preprocessing_stats.pt"),
        help="preprocessing_stats.pt (needed to construct the dataset; the "
             "length scan itself does not depend on the stats).")
    p.add_argument(
        "--cache_dir", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/models/"
                     "e2e_g3fix_kanneal_v2/lengths_h0.7"),
        help="B-specific horizon-specific lengths cache dir. NOT the shared "
             "foundation_model_meta cache. Sidecars written here: "
             "lengths_e2e_stage1_{train,val}.pt")
    p.add_argument("--train_horizon", type=float, default=0.7,
                   help="TRAIN dataset prediction_horizon_s (Lever #1 block-0 = 0.7).")
    p.add_argument("--val_horizon", type=float, default=0.2,
                   help="VAL dataset prediction_horizon_s. Default 0.2 = the "
                        "trainer's val_prediction_horizon_s (the MODEL horizon; "
                        "validate() stays single-step).")
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_files", type=int, default=None,
                   help="Leave UNSET for production (full shot set). Set only "
                        "for a small-file dry-run into a throwaway cache dir.")
    args = p.parse_args()

    # HARD GUARD: never write into the shared production cache.
    _shared = Path("/lustre/orion/fus187/proj-shared/foundation_model_meta")
    assert _shared not in args.cache_dir.parents and args.cache_dir != _shared, (
        f"REFUSING to write into the shared cache {args.cache_dir}. Point "
        f"--cache_dir at a B-specific dir.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prebuild] host={os.uname().nodename} pid={os.getpid()}", flush=True)
    print(f"[prebuild] NO torch.distributed init (single process) — "
          f"dist.is_initialized()={torch.distributed.is_initialized() if torch.distributed.is_available() else 'n/a'}",
          flush=True)
    print(f"[prebuild] data_dir={args.data_dir}", flush=True)
    print(f"[prebuild] cache_dir={args.cache_dir}", flush=True)
    print(f"[prebuild] train_horizon={args.train_horizon}s  val_horizon={args.val_horizon}s  "
          f"chunk={args.chunk_duration_s} step={args.step_size_s} warmup={args.warmup_s}  "
          f"val_fraction={args.val_fraction} seed={args.seed} max_files={args.max_files}",
          flush=True)

    # ── Resolve the production file lists EXACTLY as the trainer does ─────────
    # (no yaml → glob + shuffle(seed) + val_fraction split; matches
    # train_e2e_stage1_kanneal.sh which passes neither --train_shots_yaml nor
    # --val_shots_yaml, seed 42, val_fraction 0.1).
    t0 = time.time()
    train_files, val_files = resolve_shot_files(
        args.data_dir,
        None,                # train_shots_yaml
        None,                # val_shots_yaml
        args.max_files,
        args.val_fraction,
        args.seed,
    )
    print(f"[prebuild] resolved files — train={len(train_files)} val={len(val_files)} "
          f"({time.time() - t0:.1f}s)", flush=True)
    if not train_files or not val_files:
        raise SystemExit("No train/val files resolved — check data_dir.")

    stats = torch.load(args.stats_path, weights_only=False)

    # Diagnostic/actuator names do NOT affect the length scan (it reads only
    # shot duration + horizon/chunk/step/warmup). Pass minimal placeholders so
    # build_datasets can construct signal_configs; the scan result is identical.
    diagnostic_names = ["ece"]
    actuator_names: list[str] = []

    # ── Build the datasets → triggers the horizon-specific length scan + save ─
    # build_datasets writes:
    #   TRAIN cache at prediction_horizon_s (= --train_horizon)
    #   VAL   cache at val_prediction_horizon_s (= --val_horizon)
    # to <cache_dir>/lengths_e2e_stage1_{train,val}.pt — the exact filenames the
    # trainer looks up.
    print(f"[prebuild] scanning TRAIN ({len(train_files)} files) @ "
          f"horizon={args.train_horizon}s + VAL ({len(val_files)} files) @ "
          f"horizon={args.val_horizon}s ...  (~87 min cold for the full set)",
          flush=True)
    t1 = time.time()
    train_ds, val_ds = build_datasets(
        args.data_dir,
        train_files,
        val_files,
        preprocessing_stats=stats,
        chunk_duration_s=args.chunk_duration_s,
        prediction_horizon_s=args.train_horizon,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        diagnostic_names=diagnostic_names,
        actuator_names=actuator_names,
        lengths_cache_dir=args.cache_dir,
        history_windows=1,
        val_prediction_horizon_s=args.val_horizon,
    )
    dt = time.time() - t1
    print(f"[prebuild] scan complete in {dt / 60:.1f} min — "
          f"train chunks={len(train_ds)}  val chunks={len(val_ds)}", flush=True)

    for split, ds in (("train", train_ds), ("val", val_ds)):
        side = args.cache_dir / f"lengths_e2e_stage1_{split}.pt"
        ok = side.exists()
        sz = side.stat().st_size if ok else 0
        print(f"[prebuild] {split} sidecar: {side}  exists={ok}  bytes={sz}", flush=True)
        assert ok, f"{split} lengths sidecar was NOT written: {side}"

    print("[prebuild] DONE — horizon-specific lengths cache built. "
          "The B production chain can now point LENGTHS_CACHE_DIR at "
          f"{args.cache_dir}.", flush=True)


if __name__ == "__main__":
    main()
