"""One-shot warm-start for Phase 1 aggregation.

Used when Phase 1 timed out *during* the rank-0 aggregation step (after
all 64 per-rank shards landed on disk). Reads the per-rank shard CSVs,
re-runs aggregate_per_shot + select_top_bottom + compute_gates_and_summary,
and writes config.json. Imports the existing Phase 1 helpers so the
output schema stays identical to a clean run.

Usage:
    pixi run python scripts/training/_finish_phase1_aggregation.py \\
        --output_dir eval_runs/stage2_phase1_e2e_stage2_delta_best_4745298 \\
        --checkpoint /lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_d1024_48L/e2e_stage2_delta_best.pt

K is autodetected from the checkpoint's ``args['K_max']`` (matches Phase 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e import detect_stage_K  # type: ignore[import]  # noqa: E402
from eval_e2e_phase1 import (  # type: ignore[import]  # noqa: E402
    aggregate_per_shot,
    compute_gates_and_summary,
    select_top_bottom,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--top_n", type=int, default=5)
    p.add_argument("--bottom_n", type=int, default=5)
    p.add_argument("--mag_ratio_lo", type=float, default=0.3)
    p.add_argument("--mag_ratio_hi", type=float, default=3.0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("warm_start")

    shards = sorted(args.output_dir.glob("per_window_metrics.val.rank*.csv.gz"))
    shards += sorted(args.output_dir.glob("per_window_metrics.train.rank*.csv.gz"))
    if not shards:
        raise SystemExit(f"No per-rank shard CSVs found in {args.output_dir}")
    log.info(f"Found {len(shards)} per-rank shard files")

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    ckpt_step = ckpt.get("step")
    K = detect_stage_K(ckpt)
    log.info(f"K={K} (autodetected from checkpoint)")

    per_window_df, per_shot_df = aggregate_per_shot(shards, args.output_dir)
    top_bottom_df = select_top_bottom(
        per_shot_df,
        top_n=args.top_n, bottom_n=args.bottom_n,
        output_dir=args.output_dir,
    )
    gates = compute_gates_and_summary(
        per_window_df=per_window_df, K=K,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        ckpt_step=ckpt_step,
        mag_ratio_lo=args.mag_ratio_lo,
        mag_ratio_hi=args.mag_ratio_hi,
    )

    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": ckpt_step,
        "K": K,
        "warm_started_from": "per-rank shards (Phase 1 SLURM timeout)",
        "n_per_window_rows": int(len(per_window_df)),
        "n_per_shot_rows": int(len(per_shot_df)),
        "n_top_bottom_rows": int(len(top_bottom_df)),
        "gates": gates["global"],
    }, indent=2))
    log.info(f"Wrote {config_path.name}")

    for f in shards:
        f.unlink()
    log.info(f"Cleaned up {len(shards)} per-rank shard files")
    log.info("Warm-start aggregation complete.")


if __name__ == "__main__":
    main()
