"""Build the resumable preprocessed MAST ``rgb`` camera frames cache.

Usage::

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= scripts/mast/py.sh scripts/mast/port_camera_frames.py \
        --workers 64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tokamak_foundation_model.ignite.mast import paths
from tokamak_foundation_model.ignite.mast.port import camera, framecache


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _process_shot(job: tuple[str, str, str, str]) -> tuple[str, str, dict[str, Any]]:
    shot, split, level1_root_str, out_dir_str = job
    shot_zarr = Path(level1_root_str) / f"{shot}.zarr"
    out_npz = Path(out_dir_str) / f"{shot}.npz"
    ok, status = camera.check_rgb_zarr_complete(shot_zarr)
    if not ok:
        return shot, split, {"status": status, "split": split}
    try:
        data, times, exposure_s, _attrs = camera.read_rgb_zarr(shot_zarr)
        prep = camera.preprocess_frames(data)
        camera.save_frames_npz_atomic(
            out_npz,
            u=prep["u"],
            time=times,
            m=prep["m"],
            s=prep["s"],
            raw_mean_counts=prep["raw_mean_counts"],
            exposure_s=exposure_s,
        )
        return (
            shot,
            split,
            {
                "status": "ok",
                "split": split,
                "n_frames": int(data.shape[0]),
                "time_range": [float(times[0]), float(times[-1])] if times.size > 0 else [],
                "exposure_s": float(exposure_s),
                "saturation_frac": float(prep["saturation_frac"]),
            },
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"[:200]
        return shot, split, {"status": f"error: {msg}", "split": split}


def build_camera_frames_cache(
    *,
    level1_root: Path = camera.DEFAULT_LEVEL1_ROOT,
    splits_path: Path = camera.DEFAULT_SPLITS_PATH,
    out_dir: Path = camera.DEFAULT_FRAMES_DIR,
    workers: int = 64,
    max_shots: int = 0,
) -> dict[str, Any]:
    """Build or resume ``frames_rgb_v1`` across all shots in ``splits_path``."""
    t_start = time.monotonic()
    out_path = paths.prepare_write_root(Path(out_dir))
    splits_file = Path(splits_path)
    if splits_file.is_dir():
        splits_file = splits_file / "splits.json"
    splits = json.loads(splits_file.read_text())

    shot_split_list: list[tuple[str, str]] = []
    for split_name in ("train", "val", "test"):
        for s in splits.get(split_name, []):
            shot_split_list.append((str(s), split_name))
    if max_shots > 0:
        shot_split_list = shot_split_list[: int(max_shots)]

    index_path = out_path / "index.json"
    existing_shots: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        try:
            existing_shots = dict(json.loads(index_path.read_text()).get("shots", {}))
        except (OSError, ValueError):
            existing_shots = {}

    shots_rec: dict[str, dict[str, Any]] = {}
    todo: list[tuple[str, str, str, str]] = []
    n_reused = 0
    for shot, split_name in shot_split_list:
        prev = existing_shots.get(shot)
        if prev is not None and prev.get("status") == "ok" and (out_path / f"{shot}.npz").exists():
            rec = dict(prev)
            rec["split"] = split_name
            shots_rec[shot] = rec
            n_reused += 1
        else:
            todo.append((shot, split_name, str(level1_root), str(out_path)))

    print(
        f"[port_camera_frames] total={len(shot_split_list)} reused_ok={n_reused} "
        f"todo={len(todo)} workers={workers}",
        flush=True,
    )

    if todo:
        if workers <= 1 or len(todo) <= 1:
            results = [_process_shot(j) for j in todo]
        else:
            ctx = multiprocessing.get_context("spawn")
            results = []
            with concurrent.futures.ProcessPoolExecutor(
                min(int(workers), len(todo)),
                mp_context=ctx,
                initializer=_limit_worker_threads,
            ) as ex:
                for idx, res in enumerate(
                    ex.map(_process_shot, todo, chunksize=max(1, len(todo) // (workers * 4))),
                    1,
                ):
                    results.append(res)
                    if idx % 1000 == 0 or idx == len(todo):
                        print(
                            f"[port_camera_frames] {idx}/{len(todo)} checked "
                            f"({time.monotonic() - t_start:.1f} s)",
                            flush=True,
                        )
        for shot, _split, rec in results:
            shots_rec[shot] = rec

    per_split_counts: dict[str, dict[str, int]] = {}
    overall_counts: Counter = Counter()
    for shot, split_name in shot_split_list:
        st = str(shots_rec[shot]["status"])
        bucket = "error" if st.startswith("error:") else st
        overall_counts[bucket] += 1
        per_split_counts.setdefault(split_name, {})
        per_split_counts[split_name][bucket] = per_split_counts[split_name].get(bucket, 0) + 1

    index_doc = {
        "level1_root": str(Path(level1_root).resolve()),
        "splits_path": str(splits_file.resolve()),
        "pool": camera.CAMERA_POOL,
        "height": camera.CAMERA_HEIGHT,
        "width": camera.CAMERA_WIDTH,
        "saturation_count": camera.CAMERA_SATURATION_COUNT,
        "updated_utc": camera._utc_now_from_date(),
        "counts": dict(overall_counts),
        "counts_by_split": per_split_counts,
        "shots": shots_rec,
    }
    framecache.save_json_atomic(index_doc, index_path)
    summary = {
        "out_dir": str(out_path),
        "counts": dict(overall_counts),
        "counts_by_split": per_split_counts,
        "reused_ok": n_reused,
        "wall_seconds": round(time.monotonic() - t_start, 1),
    }
    print(f"[port_camera_frames] summary: {json.dumps(summary, indent=2)}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Build preprocessed MAST rgb camera frames cache")
    p.add_argument("--level1-root", default=str(camera.DEFAULT_LEVEL1_ROOT))
    p.add_argument("--splits", default=str(camera.DEFAULT_SPLITS_PATH))
    p.add_argument("--out", default=str(camera.DEFAULT_FRAMES_DIR))
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--max-shots", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    build_camera_frames_cache(
        level1_root=Path(args.level1_root),
        splits_path=Path(args.splits),
        out_dir=Path(args.out),
        workers=args.workers,
        max_shots=args.max_shots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
