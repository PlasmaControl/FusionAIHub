"""Build the resumable preprocessed MAST ``rgb`` camera frames cache.

Usage::

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= scripts/mast/py.sh scripts/mast/port_camera_frames.py \
        --workers 64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import zarr

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


def _revalidate_shot(
    job: tuple[str, str, str, str, float | None],
) -> tuple[str, str, str | None, int]:
    """Re-check one previously-``ok`` shot's cached exposure and timestamps against its zarr."""
    shot, split, level1_root_str, out_dir_str, index_exp = job
    shot_zarr = Path(level1_root_str) / f"{shot}.zarr"
    out_npz = Path(out_dir_str) / f"{shot}.npz"
    ok, status = camera.check_rgb_zarr_complete(shot_zarr)
    if not ok:
        return shot, split, f"zarr_{status}", 0
    try:
        grp = zarr.open_group(str(shot_zarr / "rgb"), mode="r")
        zarr_exp = camera.extract_exposure_s(dict(grp.attrs))
        zarr_time = np.asarray(grp["time"][:], dtype=np.float64)
        n_invalid_ts = int(zarr_time.size - camera.valid_exposure_indices(zarr_time).size)
        fr = camera.load_frames_npz(out_npz, load_u=False)
        cached_exp = float(fr["exposure_s"])
        if not (
            math.isfinite(cached_exp)
            and math.isclose(cached_exp, zarr_exp, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return (
                shot,
                split,
                f"exposure_mismatch(cached={cached_exp},zarr={zarr_exp})",
                n_invalid_ts,
            )
        if index_exp is None or not (
            math.isfinite(float(index_exp))
            and math.isclose(float(index_exp), zarr_exp, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return (
                shot,
                split,
                f"index_exposure_mismatch(index={index_exp},zarr={zarr_exp})",
                n_invalid_ts,
            )
        cached_time = fr["time"]
        if cached_time.shape != zarr_time.shape or not np.array_equal(
            cached_time, zarr_time, equal_nan=True
        ):
            return shot, split, "time_mismatch", n_invalid_ts
        return shot, split, None, n_invalid_ts
    except Exception as e:
        return shot, split, f"read_error:{type(e).__name__}", 0


def build_camera_frames_cache(
    *,
    level1_root: Path = camera.DEFAULT_LEVEL1_ROOT,
    splits_path: Path = camera.DEFAULT_SPLITS_PATH,
    out_dir: Path = camera.DEFAULT_FRAMES_DIR,
    workers: int = 64,
    max_shots: int = 0,
    revalidate: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build or resume ``frames_rgb_v1`` across all shots in ``splits_path``."""
    t_start = time.monotonic()
    out_path = Path(out_dir) if dry_run else paths.prepare_write_root(Path(out_dir))
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
    ok_candidates: list[tuple[str, str, dict[str, Any]]] = []
    todo_non_ok: list[tuple[str, str, str, str]] = []
    for shot, split_name in shot_split_list:
        prev = existing_shots.get(shot)
        if prev is not None and prev.get("status") == "ok" and (out_path / f"{shot}.npz").exists():
            ok_candidates.append((shot, split_name, prev))
        else:
            todo_non_ok.append((shot, split_name, str(level1_root), str(out_path)))

    revalidated_disagreed: dict[str, str] = {}
    revalidated_invalid_timestamps: dict[str, int] = {}
    todo_reprocess: list[tuple[str, str, str, str]] = []
    n_reused = 0

    if revalidate and ok_candidates:
        reval_jobs = [
            (
                shot,
                split_name,
                str(level1_root),
                str(out_path),
                prev.get("exposure_s"),
            )
            for shot, split_name, prev in ok_candidates
        ]
        if workers <= 1 or len(reval_jobs) <= 1:
            reval_results = [_revalidate_shot(j) for j in reval_jobs]
        else:
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                min(int(workers), len(reval_jobs)),
                mp_context=ctx,
                initializer=_limit_worker_threads,
            ) as ex:
                reval_results = list(
                    ex.map(
                        _revalidate_shot,
                        reval_jobs,
                        chunksize=max(1, len(reval_jobs) // (workers * 4)),
                    )
                )
        prev_by_shot = {shot: prev for shot, _split, prev in ok_candidates}
        for shot, split_name, reason, n_invalid_ts in reval_results:
            if n_invalid_ts > 0:
                revalidated_invalid_timestamps[shot] = n_invalid_ts
            if reason is not None:
                revalidated_disagreed[shot] = reason
                todo_reprocess.append((shot, split_name, str(level1_root), str(out_path)))
            else:
                rec = dict(prev_by_shot[shot])
                rec["split"] = split_name
                shots_rec[shot] = rec
                n_reused += 1
    else:
        for shot, split_name, prev in ok_candidates:
            rec = dict(prev)
            rec["split"] = split_name
            shots_rec[shot] = rec
            n_reused += 1

    todo = todo_reprocess + todo_non_ok

    print(
        f"[port_camera_frames] total={len(shot_split_list)} reused_ok={n_reused} "
        f"reprocess={len(todo_reprocess)} todo_non_ok={len(todo_non_ok)} "
        f"revalidate={revalidate} dry_run={dry_run} workers={workers}",
        flush=True,
    )

    if dry_run:
        summary: dict[str, Any] = {
            "out_dir": str(out_path),
            "dry_run": True,
            "revalidate": bool(revalidate),
            "revalidated_ok": len(ok_candidates) if revalidate else 0,
            "reused_ok": n_reused,
            "revalidated_disagreed": revalidated_disagreed,
            "revalidated_invalid_timestamps": revalidated_invalid_timestamps,
            "todo_reprocess": len(todo_reprocess),
            "todo_non_ok": len(todo_non_ok),
            "todo_total": len(todo),
            "wall_seconds": round(time.monotonic() - t_start, 1),
        }
        print(f"[port_camera_frames] dry-run summary: {json.dumps(summary, indent=2)}", flush=True)
        return summary

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
        "revalidated_ok": len(ok_candidates) if revalidate else 0,
        "revalidated_disagreed": revalidated_disagreed,
        "revalidated_invalid_timestamps": revalidated_invalid_timestamps,
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
    p.add_argument(
        "--revalidate",
        action="store_true",
        help="Re-check every ok shot's cached exposure and timestamps against zarr; re-process mismatches",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be processed/re-processed without writing any files",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    build_camera_frames_cache(
        level1_root=Path(args.level1_root),
        splits_path=Path(args.splits),
        out_dir=Path(args.out),
        workers=args.workers,
        max_shots=args.max_shots,
        revalidate=args.revalidate,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
