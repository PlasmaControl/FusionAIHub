"""Fit the centred log10-power Mirnov spectrogram codec (LogBandDCT) on MAST training shots.

Usage::

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= scripts/mast/py.sh scripts/mast/port_fit_logspec.py \
        --out /models/intermediate/mast/stage2r/logspec_fit_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tokamak_foundation_model.ignite.mast.port import logspec

DEFAULT_CACHE_ROOT = "/models/intermediate/mast/cache/v1"
DEFAULT_SPLITS_DIR = "/models/intermediate/mast/stage2r/frame_codes_v1b_plasma"
DEFAULT_ASSIGNMENT = "/models/intermediate/mast/stage1/assignments/v1b.json"
DEFAULT_OUT = str(logspec.DEFAULT_LOGSPEC_OUT)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fit LogBandDCT per-bin floors, log means, and 64-bin grids")
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT, help="MAST cache v1 directory")
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_SPLITS_DIR,
        help="Directory containing splits.json (or path to splits.json)",
    )
    p.add_argument("--assignment", default=DEFAULT_ASSIGNMENT, help="Stage-1 assignment JSON")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output directory for logspec_fit.{npz,json}")
    p.add_argument(
        "--target-shots",
        type=int,
        default=logspec.LOGSPEC_TARGET_SHOTS,
        help="Target training shots with the signal to sample per array",
    )
    p.add_argument("--workers", type=int, default=32, help="Spawn worker count for reading shots")
    return p


def main(argv=None) -> int:
    t0 = time.monotonic()
    args = build_arg_parser().parse_args(argv)
    meta = logspec.fit_logspec(
        cache_root=Path(args.cache_root),
        splits_path=Path(args.cache_dir),
        assignment_path=Path(args.assignment),
        out_dir=Path(args.out),
        target_shots=args.target_shots,
        workers=args.workers,
    )
    summary = {
        "out": str(Path(args.out).resolve()),
        "logspec_fit_sha256": meta["logspec_fit_sha256"],
        "created_utc": meta["created_utc"],
        "n_bins": meta["n_bins"],
        "wall_seconds": round(time.monotonic() - t0, 1),
        "signals": {
            name: {k: v for k, v in info.items() if k != "shots"}
            for name, info in meta["signals"].items()
        },
    }
    print(f"[port_fit_logspec] {json.dumps(summary, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
