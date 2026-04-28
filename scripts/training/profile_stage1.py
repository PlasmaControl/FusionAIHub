"""Profile a handful of Stage 1 training steps under ``torch.profiler``.

This is a standalone script — it imports the dataset/model/loss helpers from
``train_e2e_stage1`` so the profile reflects the real pipeline (same signals,
same DataLoader settings, same forward + backward + optimizer step). Nothing
about ``train_e2e_stage1.py`` itself is changed.

What the output gives you:
  - a chrome://tracing JSON trace for the ``active`` steps (visualised
    timeline of data-loader wait / forward / backward / optimizer / all CUDA
    kernels, grouped per step)
  - a text summary (``key_averages`` sorted by CUDA time) printed to stdout
  - per-step wall-clock times, so you can sanity-check against the training
    job's observed s/step

Typical usage — inside a short SLURM job on a GPU node:

    pixi run python scripts/training/profile_stage1.py \\
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \\
        --stats_path /scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt \\
        --output_dir runs/profile_stage1 \\
        --batch_size 256 --num_workers 8

Open the resulting ``trace_step<N>.json`` in ``chrome://tracing`` (or Perfetto).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader

# Let imports resolve train_e2e_stage1 without installing it as a package.
sys.path.insert(0, str(Path(__file__).parent))

from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.e2e.model import E2EFoundationModel
from train_e2e_stage1 import (  # type: ignore
    build_configs,
    build_datasets,
    compute_step_loss,
    resolve_shot_files,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--stats_path", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--lengths_cache_dir", type=Path,
        default=Path("runs/e2e_stage1"),
        help="Directory holding lengths_e2e_stage1_{train,val}.pt. Defaults "
             "to the real Stage 1 run's directory so we don't recompute the "
             "~15-min file-length scan on every profile submission.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--prediction_horizon_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # Profiler schedule: (wait, warmup, active). ``wait`` skips the dataloader
    # spin-up transient; ``warmup`` primes caches so the active window is
    # steady-state; ``active`` is what gets recorded.
    p.add_argument("--profile_wait", type=int, default=5)
    p.add_argument("--profile_warmup", type=int, default=5)
    p.add_argument("--profile_active", type=int, default=20)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"num_workers={args.num_workers}  batch_size={args.batch_size}")

    diagnostics, actuators = build_configs(args.chunk_duration_s)
    diag_names = [c.name for c in diagnostics]
    act_names = [c.name for c in actuators]
    print(f"Diagnostics ({len(diag_names)}): {diag_names}")
    print(f"Actuators ({len(act_names)}): {act_names}")

    train_files, val_files = resolve_shot_files(
        data_dir=args.data_dir,
        train_shots_yaml=None, val_shots_yaml=None,
        max_files=None, val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"Train files: {len(train_files)}  val: {len(val_files)}")

    print("Loading preprocessing_stats…")
    stats = torch.load(args.stats_path, weights_only=False)

    train_ds, _ = build_datasets(
        data_dir=args.data_dir,
        train_files=train_files, val_files=val_files,
        preprocessing_stats=stats,
        chunk_duration_s=args.chunk_duration_s,
        prediction_horizon_s=args.prediction_horizon_s,
        step_size_s=args.step_size_s,
        warmup_s=args.warmup_s,
        diagnostic_names=diag_names,
        actuator_names=act_names,
        lengths_cache_dir=args.lengths_cache_dir,
    )
    print(f"Train chunks: {len(train_ds)}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = E2EFoundationModel(
        diagnostics=diagnostics,
        actuators=actuators,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    total_steps = args.profile_wait + args.profile_warmup + args.profile_active
    print(
        f"Profile schedule: wait={args.profile_wait} "
        f"warmup={args.profile_warmup} active={args.profile_active} "
        f"(total {total_steps} steps)"
    )

    trace_path = args.output_dir / "trace.json"
    summary_path = args.output_dir / "top_ops.txt"

    def on_ready(prof_obj: profile) -> None:
        prof_obj.export_chrome_trace(str(trace_path))
        with summary_path.open("w") as f:
            f.write(
                prof_obj.key_averages().table(
                    sort_by="cuda_time_total", row_limit=25
                )
            )
        print(f"Trace written: {trace_path}")
        print(f"Top ops summary: {summary_path}")

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=1,
        ),
        on_trace_ready=on_ready,
        record_shapes=True,
        with_stack=False,
    )

    model.train()
    step_times: list[float] = []
    t_start = time.time()

    prof.start()
    for step, batch in enumerate(loader):
        if step >= total_steps:
            break
        s = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, _ = compute_step_loss(model, batch, device)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - s)
        prof.step()
    prof.stop()

    print()
    print("=" * 60)
    print(f"Total wall time: {time.time() - t_start:.1f} s")
    print(f"Per-step wall times (s): "
          + " ".join(f"{t:.2f}" for t in step_times))
    active_slice = step_times[args.profile_wait + args.profile_warmup:]
    if active_slice:
        print(
            f"Active-window mean: "
            f"{sum(active_slice) / len(active_slice):.2f} s/step  "
            f"(over {len(active_slice)} steps)"
        )
    print(f"Trace : {trace_path}")
    print(f"Summary: {summary_path}")
    print("Open the trace in chrome://tracing or Perfetto.")


if __name__ == "__main__":
    main()