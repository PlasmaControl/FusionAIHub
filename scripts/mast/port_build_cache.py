"""Build a dev-peter-compatible IGNITE Phase-B frame-code cache from MAST cache v1.

Usage::

    OMP_NUM_THREADS=8 scripts/mast/py.sh scripts/mast/port_build_cache.py \
        --assignment /models/intermediate/mast/stage1/assignments/v1b.json \
        --fit /models/intermediate/mast/stage1/fit/v1 \
        --cache-root /models/intermediate/mast/cache/v1 \
        --out /models/intermediate/mast/stage2r/frame_codes_v1b \
        --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from tokamak_foundation_model.ignite.mast import paths, registry, transform
from tokamak_foundation_model.ignite.mast.port import framecache
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit

DEFAULT_ASSIGNMENT = "/models/intermediate/mast/stage1/assignments/v1b.json"
DEFAULT_FIT = "/models/intermediate/mast/stage1/fit/v1"
DEFAULT_CACHE_ROOT = "/models/intermediate/mast/cache/v1"

# Per-worker state initialized once per spawned worker process.
_STATS_STATE: tuple | None = None
_ENCODE_STATE: tuple | None = None


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _init_stats_worker(state: tuple) -> None:
    global _STATS_STATE
    _limit_worker_threads()
    _STATS_STATE = state


def _stats_job(shot_path_str: str):
    act_specs, plasma_names, stats_sha = _STATS_STATE
    return framecache.shot_actuator_moments(
        Path(shot_path_str), act_specs, plasma_names, stats_sha256=stats_sha
    )


def _init_encode_worker(state: tuple) -> None:
    global _ENCODE_STATE
    _limit_worker_threads()
    signals, cache_root, fit_dir, assignment_path, recoders_dir, act_specs, act_stats, plasma_names, stats_sha, out_dir, force = state
    codecs, _stats, _assignment = framecache.build_mast_codecs(
        signals, Path(cache_root), Path(fit_dir), Path(assignment_path), Path(recoders_dir)
    )
    modality_order = tuple(s.name for s in signals if s.name in codecs)
    _ENCODE_STATE = (
        codecs, modality_order, act_specs, act_stats, plasma_names, stats_sha, Path(out_dir), bool(force)
    )


def _encode_job(item: tuple[str, str, str]):
    shot, split, shot_path_str = item
    codecs, modality_order, act_specs, act_stats, plasma_names, stats_sha, out_dir, force = _ENCODE_STATE
    out_file = out_dir / f"{shot}.pt"
    if out_file.exists() and not force:
        try:
            existing = torch.load(out_file, map_location="cpu", weights_only=False)
            return shot, split, "existing", int(existing["n_frames"]), existing.get("present")
        except Exception:
            pass
    try:
        entry, reason = framecache.encode_shot(
            Path(shot_path_str),
            codecs,
            modality_order,
            act_specs,
            act_stats,
            plasma_names,
            stats_sha256=stats_sha,
        )
        if entry is None:
            return shot, split, reason, 0, None
        framecache.save_pt_atomic(entry, out_file)
        return shot, split, "written", int(entry["n_frames"]), entry["present"]
    except Exception as e:
        return shot, split, f"error:{type(e).__name__}", 0, None


def _init_split_worker() -> None:
    _limit_worker_threads()


def _split_job(shot_path_str: str) -> tuple[str, str, str]:
    p = Path(shot_path_str)
    return p.stem, framecache.read_shot_split(p), shot_path_str


def _discover_shots(cache_root: Path, workers: int, max_shots: int) -> list[tuple[str, str, str]]:
    h5_files = sorted((cache_root / "shots").glob("*.h5"))
    if max_shots > 0:
        h5_files = h5_files[:max_shots]
    paths_str = [str(p) for p in h5_files]
    if workers <= 1 or len(paths_str) <= 1:
        return [_split_job(p) for p in paths_str]
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        min(workers, len(paths_str)), mp_context=ctx, initializer=_init_split_worker
    ) as ex:
        return list(ex.map(_split_job, paths_str, chunksize=max(1, len(paths_str) // (workers * 4))))


def _compute_actuator_stats(
    train_paths: list[str],
    act_specs: tuple,
    plasma_names: tuple[str, ...],
    stats_sha: str,
    workers: int,
) -> framecache.ActuatorStats:
    state = (act_specs, plasma_names, stats_sha)
    if workers <= 1 or len(train_paths) <= 1:
        moments = [
            m for p in train_paths
            if (m := framecache.shot_actuator_moments(Path(p), act_specs, plasma_names, stats_sha256=stats_sha)) is not None
        ]
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            min(workers, len(train_paths)),
            mp_context=ctx,
            initializer=_init_stats_worker,
            initargs=(state,),
        ) as ex:
            moments = [
                m for m in ex.map(_stats_job, train_paths, chunksize=max(1, len(train_paths) // (workers * 4)))
                if m is not None
            ]
    return framecache.finalize_actuator_stats(act_specs, moments)


def build_cache(
    *,
    assignment_path: Path,
    fit_dir: Path,
    cache_root: Path,
    out_dir: Path,
    workers: int = 8,
    max_shots: int = 0,
    force: bool = False,
) -> dict:
    """Build (or resume) the MAST Phase-B frame-code cache under ``out_dir``."""
    t_start = time.monotonic()
    out_dir = paths.prepare_write_root(Path(out_dir))
    cache_root = Path(cache_root)
    fit_dir = Path(fit_dir)
    assignment_path = Path(assignment_path)

    manifest_v1 = json.loads((cache_root / "manifest.json").read_text())
    stats_sha = str(manifest_v1["stats_sha256"])
    if transform.stats_sha256(cache_root / "stats.npz") != stats_sha:
        raise ValueError(f"{cache_root}/stats.npz sha256 disagrees with manifest.json")

    signals = registry.all_signals()
    act_specs = framecache.actuator_specs()
    act_names_set = {s.name for s in act_specs}
    plasma_names = tuple(s.name for s in signals if s.name not in act_names_set)

    codecs, _stats, assignment = framecache.build_mast_codecs(
        signals, cache_root, fit_dir, assignment_path
    )
    assign_sha = assign.assignment_sha256(assignment)
    fit_sha = fit.fit_sha256(fit_dir)

    shot_items = _discover_shots(cache_root, workers=workers, max_shots=max_shots)
    train_paths = [p for _s, sp, p in shot_items if sp == "train"]
    if not train_paths:
        raise ValueError("no training shots found to fit global actuator statistics")

    print(
        f"[port_build_cache] discovered {len(shot_items)} shots "
        f"(train={len(train_paths)}), fitting actuator stats...",
        flush=True,
    )
    act_stats = _compute_actuator_stats(
        train_paths, act_specs, plasma_names, stats_sha, workers=workers
    )

    manifest = framecache.build_manifest(
        signals,
        codecs,
        act_stats,
        assignment_path=assignment_path,
        assignment_sha256=assign_sha,
        fit_dir=fit_dir,
        fit_sha256=fit_sha,
        cache_root=cache_root,
        stats_sha256=stats_sha,
    )
    framecache.save_json_atomic(manifest, out_dir / "_codec_manifest.json")

    counts: Counter = Counter()
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    presence: dict[str, dict[str, bool]] = {}
    frame_lengths: list[int] = []

    enc_state = (
        tuple(signals),
        str(cache_root),
        str(fit_dir),
        str(assignment_path),
        str(framecache.recoders_root()),
        act_specs,
        act_stats,
        plasma_names,
        stats_sha,
        str(out_dir),
        force,
    )

    if workers <= 1 or len(shot_items) <= 1:
        _init_encode_worker(enc_state)
        results = [_encode_job(item) for item in shot_items]
    else:
        ctx = multiprocessing.get_context("spawn")
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            min(workers, len(shot_items)),
            mp_context=ctx,
            initializer=_init_encode_worker,
            initargs=(enc_state,),
        ) as ex:
            for idx, res in enumerate(
                ex.map(_encode_job, shot_items, chunksize=max(1, len(shot_items) // (workers * 8))),
                1,
            ):
                results.append(res)
                if idx % 500 == 0 or idx == len(shot_items):
                    print(
                        f"[port_build_cache] {idx}/{len(shot_items)} shots processed "
                        f"({time.monotonic() - t_start:.1f} s)",
                        flush=True,
                    )

    for shot, split, status, n_frames, pres in results:
        counts[status] += 1
        if status in ("written", "existing"):
            if split in splits:
                splits[split].append(shot)
            frame_lengths.append(n_frames)
            if pres is not None:
                presence[shot] = {str(k): bool(v) for k, v in pres.items()}

    for part in splits:
        splits[part].sort()
    framecache.save_json_atomic(splits, out_dir / "splits.json")
    framecache.save_json_atomic(
        {"cache_dir": str(out_dir), "presence": presence}, out_dir / "_presence.json"
    )

    summary = {
        "counts": dict(counts),
        "splits": {k: len(v) for k, v in splits.items()},
        "n_frames": (
            {
                "min": int(np.min(frame_lengths)),
                "p10": int(np.percentile(frame_lengths, 10)),
                "p50": int(np.percentile(frame_lengths, 50)),
                "p90": int(np.percentile(frame_lengths, 90)),
                "max": int(np.max(frame_lengths)),
            }
            if frame_lengths
            else {}
        ),
        "wall_seconds": round(time.monotonic() - t_start, 1),
    }
    print(f"[port_build_cache] summary: {json.dumps(summary)}", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Build MAST IGNITE Phase-B frame-code cache")
    p.add_argument("--assignment", default=DEFAULT_ASSIGNMENT, help="Stage-1 assignment JSON")
    p.add_argument("--fit", default=DEFAULT_FIT, help="Stage-1 fit directory")
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT, help="MAST cache v1 directory")
    p.add_argument("--out", required=True, help="Output frame-code cache directory")
    p.add_argument("--workers", type=int, default=8, help="Spawn worker count")
    p.add_argument("--max-shots", type=int, default=0, help="Cap total shots (0 = all)")
    p.add_argument("--force", action="store_true", help="Overwrite existing {shot}.pt files")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    build_cache(
        assignment_path=Path(args.assignment),
        fit_dir=Path(args.fit),
        cache_root=Path(args.cache_root),
        out_dir=Path(args.out),
        workers=args.workers,
        max_shots=args.max_shots,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
