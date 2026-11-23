"""Fit MAST ``rgb`` timing convention, ``s_floor``, and ``camera-rgb-stats`` 64-bin grids.

Usage::

    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= scripts/mast/py.sh scripts/mast/port_camera_fit.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tokamak_foundation_model.ignite.mast import paths, transform
from tokamak_foundation_model.ignite.mast.port import camera

DEFAULT_CACHE_V1 = Path("/models/intermediate/mast/cache/v1")
DEFAULT_FRAME_CACHE = camera.DEFAULT_SPLITS_PATH.parent


def run_camera_fit(
    *,
    frames_dir: Path = camera.DEFAULT_FRAMES_DIR,
    cache_v1_root: Path = DEFAULT_CACHE_V1,
    frame_cache_dir: Path = DEFAULT_FRAME_CACHE,
    out_dir: Path = camera.DEFAULT_CAMERA_FIT_DIR,
    timing_shots: int = 30,
) -> dict[str, Any]:
    """Measure timing convention on 30 train shots and fit ``s_floor`` + stats grids on train plasma frames."""
    t_start = time.monotonic()
    frames_path = Path(frames_dir)
    v1_root = Path(cache_v1_root)
    fc_dir = Path(frame_cache_dir)
    out_path = paths.prepare_write_root(Path(out_dir))

    index_path = frames_path / "index.json"
    frames_index_sha = camera.file_sha256(index_path)

    timing = camera.measure_timing_convention(
        frames_path, v1_root, fc_dir, n_shots=int(timing_shots)
    )
    chosen_conv = str(timing["chosen"])

    splits = json.loads((fc_dir / "splits.json").read_text())
    train_shots = sorted((str(s) for s in splits["train"]), key=lambda s: int(s))
    plasma_ranges, dt_frame = camera.load_plasma_ranges(fc_dir, train_shots)
    plasma_exps = camera.compute_plasma_exposures(
        frames_path, plasma_ranges, dt_frame, chosen_conv
    )

    m_chunks: list[np.ndarray] = []
    s_chunks: list[np.ndarray] = []
    used_train_shots: list[str] = []
    for shot in train_shots:
        exps = plasma_exps.get(shot)
        if not exps:
            continue
        fr = camera.load_frames_npz(frames_path / f"{shot}.npz", load_u=False)
        idx = np.asarray(exps, dtype=np.int64)
        m_chunks.append(fr["m"][idx].astype(np.float64))
        s_chunks.append(fr["s"][idx].astype(np.float64))
        used_train_shots.append(shot)

    if not m_chunks:
        raise ValueError(f"no training plasma exposures found in {frames_path}")

    m_all = np.concatenate(m_chunks)
    s_all = np.concatenate(s_chunks)
    s_floor, stats_codec, u_std = camera.fit_camera_stats(
        m_all, s_all, n_bins=camera.CAMERA_STATS_N_BINS, margin=camera.CAMERA_STATS_MARGIN
    )

    stats_sha = transform.stats_sha256(v1_root / "stats.npz")
    meta_doc = {
        "time_convention": chosen_conv,
        "timing": {
            "medians": timing["medians"],
            "best": timing["best"],
            "chosen": chosen_conv,
            "margin_over_centre": timing["margin_over_centre"],
            "n_shots_used": timing["n_shots_used"],
            "shots_used": timing["shots_used"],
        },
        "s_floor": float(s_floor),
        "u_std": float(u_std),
        "n_bins": int(stats_codec.n_bins),
        "margin": float(camera.CAMERA_STATS_MARGIN),
        "grid_ranges": {
            "m": {
                "lo": float(stats_codec.grids.lo[0, 0]),
                "hi": float(stats_codec.grids.hi[0, 0]),
                "width": float(stats_codec.grids.width[0, 0]),
            },
            "log10_s": {
                "lo": float(stats_codec.grids.lo[1, 0]),
                "hi": float(stats_codec.grids.hi[1, 0]),
                "width": float(stats_codec.grids.width[1, 0]),
            },
        },
        "n_train_shots": len(used_train_shots),
        "n_train_frames": int(m_all.size),
        "train_shots": used_train_shots,
        "frames_dir": str(frames_path.resolve()),
        "frames_index_sha256": frames_index_sha,
        "frame_cache_dir": str(fc_dir.resolve()),
        "cache_v1_root": str(v1_root.resolve()),
        "stats_sha256": stats_sha,
        "created_utc": camera._utc_now_from_date(),
    }
    fit_sha = camera.save_camera_fit(out_path, s_floor, stats_codec, meta_doc)
    summary = {
        "out_dir": str(out_path),
        "camera_fit_sha256": fit_sha,
        "time_convention": chosen_conv,
        "timing_medians": timing["medians"],
        "timing_n_shots": timing["n_shots_used"],
        "s_floor": float(s_floor),
        "u_std": float(u_std),
        "grid_ranges": meta_doc["grid_ranges"],
        "n_train_shots": len(used_train_shots),
        "n_train_frames": int(m_all.size),
        "frames_index_sha256": frames_index_sha,
        "wall_seconds": round(time.monotonic() - t_start, 1),
    }
    print(f"[port_camera_fit] {json.dumps(summary, indent=2)}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Fit MAST rgb camera timing, s_floor, and stats grids")
    p.add_argument("--frames-dir", default=str(camera.DEFAULT_FRAMES_DIR))
    p.add_argument("--cache-v1-root", default=str(DEFAULT_CACHE_V1))
    p.add_argument("--cache-dir", default=str(DEFAULT_FRAME_CACHE))
    p.add_argument("--out", default=str(camera.DEFAULT_CAMERA_FIT_DIR))
    p.add_argument("--timing-shots", type=int, default=30)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_camera_fit(
        frames_dir=Path(args.frames_dir),
        cache_v1_root=Path(args.cache_v1_root),
        frame_cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out),
        timing_shots=args.timing_shots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
