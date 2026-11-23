"""Build a dev-peter-compatible IGNITE Phase-B frame-code cache from MAST cache v1.

Usage::

    OMP_NUM_THREADS=8 scripts/mast/py.sh scripts/mast/port_build_cache.py \
        --assignment /models/intermediate/mast/stage1/assignments/v1b.json \
        --fit /models/intermediate/mast/stage1/fit/v1 \
        --cache-root /models/intermediate/mast/cache/v1 \
        --out /models/intermediate/mast/stage2r/frame_codes_v1b_plasma \
        --workers 96
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

import numpy as np
import torch

from tokamak_foundation_model.ignite.mast import paths, registry, transform
from tokamak_foundation_model.ignite.mast.port import framecache, logspec
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
    cache_root, act_specs, modality_order, stats_sha = state
    stats = transform.load_stats(Path(cache_root) / "stats.npz")
    _STATS_STATE = (act_specs, modality_order, stats, stats_sha)


def _stats_job(shot_path_str: str):
    act_specs, modality_order, stats, stats_sha = _STATS_STATE
    return framecache.shot_training_moments(
        Path(shot_path_str),
        act_specs,
        modality_order,
        stats,
        stats_sha256=stats_sha,
    )


def _init_encode_worker(state: tuple) -> None:
    global _ENCODE_STATE
    _limit_worker_threads()
    (
        signals,
        cache_root,
        fit_dir,
        assignment_path,
        recoders_dir,
        logspec_fit,
        act_specs,
        act_stats,
        stats_sha,
        out_dir,
        force,
    ) = state
    codecs, stats, _assignment = framecache.build_mast_codecs(
        signals,
        Path(cache_root),
        Path(fit_dir),
        Path(assignment_path),
        Path(recoders_dir),
        logspec_fit=Path(logspec_fit) if logspec_fit is not None else None,
    )
    modality_order = tuple(s.name for s in signals if s.name in codecs)
    _ENCODE_STATE = (
        codecs,
        modality_order,
        act_specs,
        act_stats,
        stats,
        stats_sha,
        Path(out_dir),
        bool(force),
    )


def _encode_job(item: tuple[str, str, str]):
    shot, split, shot_path_str = item
    codecs, modality_order, act_specs, act_stats, stats, stats_sha, out_dir, force = _ENCODE_STATE
    out_file = out_dir / f"{shot}.pt"
    if out_file.exists() and not force:
        try:
            existing = torch.load(out_file, map_location="cpu", weights_only=False)
            return shot, split, "existing", int(existing["n_frames"]), existing.get("present"), None
        except Exception:
            pass
    try:
        entry, reason = framecache.encode_shot(
            Path(shot_path_str),
            codecs,
            modality_order,
            act_specs,
            act_stats,
            stats,
            stats_sha256=stats_sha,
        )
        if entry is None:
            return shot, split, reason, 0, None, None
        framecache.save_pt_atomic(entry, out_file)
        return shot, split, "written", int(entry["n_frames"]), entry["present"], None
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)}"[:200]
        return shot, split, f"error:{type(e).__name__}", 0, None, msg


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


def _compute_train_stats(
    train_paths: list[str],
    cache_root: Path,
    act_specs: tuple,
    modality_order: tuple[str, ...],
    stats: dict[str, transform.ChannelStats],
    stats_sha: str,
    workers: int,
) -> tuple[framecache.ActuatorStats, dict[str, float]]:
    if workers <= 1 or len(train_paths) <= 1:
        pairs = [
            m
            for p in train_paths
            if (
                m := framecache.shot_training_moments(
                    Path(p), act_specs, modality_order, stats, stats_sha256=stats_sha
                )
            )
            is not None
        ]
    else:
        state = (str(cache_root), act_specs, modality_order, stats_sha)
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            min(workers, len(train_paths)),
            mp_context=ctx,
            initializer=_init_stats_worker,
            initargs=(state,),
        ) as ex:
            pairs = [
                m
                for m in ex.map(
                    _stats_job,
                    train_paths,
                    chunksize=max(1, len(train_paths) // (workers * 4)),
                )
                if m is not None
            ]
    act_moments = [a for a, _s in pairs]
    sig_moments = [s for _a, s in pairs]
    act_stats = framecache.finalize_actuator_stats(act_specs, act_moments, stats=stats)
    sig_stds = framecache.finalize_signal_physical_std(modality_order, sig_moments)
    return act_stats, sig_stds


def _check_existing_manifest(
    manifest_path: Path,
    *,
    assignment_sha256: str,
    fit_sha256: str,
    stats_sha256: str,
    plasma_ip_threshold_a: float,
    logspec_fit_sha256: str | None = None,
    n_train_shots: int | None = None,
) -> None:
    """Refuse to resume into an existing cache dir if any provenance or stats key disagrees."""
    existing = json.loads(manifest_path.read_text())
    diffs: list[str] = []
    if existing.get("assignment_sha256") != assignment_sha256:
        diffs.append(
            f"assignment_sha256 ({existing.get('assignment_sha256')} != {assignment_sha256})"
        )
    if existing.get("fit_sha256") != fit_sha256:
        diffs.append(f"fit_sha256 ({existing.get('fit_sha256')} != {fit_sha256})")
    if existing.get("stats_sha256") != stats_sha256:
        diffs.append(f"stats_sha256 ({existing.get('stats_sha256')} != {stats_sha256})")
    if existing.get("logspec_fit_sha256") != logspec_fit_sha256:
        diffs.append(
            f"logspec_fit_sha256 ({existing.get('logspec_fit_sha256')!r} != {logspec_fit_sha256!r})"
        )
    existing_thresh = existing.get("plasma_ip_threshold_A")
    if existing_thresh is None or not math.isclose(
        float(existing_thresh), float(plasma_ip_threshold_a), rel_tol=0.0, abs_tol=1e-9
    ):
        diffs.append(
            f"plasma_ip_threshold_A ({existing_thresh!r} != {plasma_ip_threshold_a!r})"
        )
    if n_train_shots is not None:
        existing_n = (existing.get("actuators") or {}).get("n_train_shots")
        if existing_n != int(n_train_shots):
            diffs.append(
                f"actuators.n_train_shots ({existing_n!r} != {int(n_train_shots)!r})"
            )
    if diffs:
        raise ValueError(
            f"refusing to resume into {manifest_path.parent} without --force; "
            f"existing _codec_manifest.json disagrees on: {'; '.join(diffs)}"
        )


def build_cache(
    *,
    assignment_path: Path,
    fit_dir: Path,
    cache_root: Path,
    out_dir: Path,
    logspec_fit: Path | None = None,
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
    logspec_fit_path = Path(logspec_fit) if logspec_fit is not None else None
    logspec_sha = (
        logspec.logspec_fit_sha256(logspec_fit_path) if logspec_fit_path is not None else None
    )

    manifest_v1 = json.loads((cache_root / "manifest.json").read_text())
    stats_sha = str(manifest_v1["stats_sha256"])
    if transform.stats_sha256(cache_root / "stats.npz") != stats_sha:
        raise ValueError(f"{cache_root}/stats.npz sha256 disagrees with manifest.json")

    signals = registry.all_signals()
    act_specs = framecache.actuator_specs()

    codecs, stats, assignment = framecache.build_mast_codecs(
        signals, cache_root, fit_dir, assignment_path, logspec_fit=logspec_fit_path
    )
    modality_order = tuple(s.name for s in signals if s.name in codecs)
    assign_sha = assign.assignment_sha256(assignment)
    fit_sha = fit.fit_sha256(fit_dir)

    manifest_path = out_dir / "_codec_manifest.json"
    if manifest_path.exists() and not force:
        _check_existing_manifest(
            manifest_path,
            assignment_sha256=assign_sha,
            fit_sha256=fit_sha,
            stats_sha256=stats_sha,
            plasma_ip_threshold_a=framecache.PLASMA_IP_THRESHOLD_A,
            logspec_fit_sha256=logspec_sha,
        )

    shot_items = _discover_shots(cache_root, workers=workers, max_shots=max_shots)
    train_paths = [p for _s, sp, p in shot_items if sp == "train"]
    if not train_paths:
        raise ValueError("no training shots found to fit global actuator statistics")

    print(
        f"[port_build_cache] discovered {len(shot_items)} shots "
        f"(train={len(train_paths)}), fitting actuator & physical-std stats...",
        flush=True,
    )
    act_stats, sig_stds = _compute_train_stats(
        train_paths,
        cache_root,
        act_specs,
        modality_order,
        stats,
        stats_sha,
        workers=workers,
    )

    if manifest_path.exists() and not force:
        _check_existing_manifest(
            manifest_path,
            assignment_sha256=assign_sha,
            fit_sha256=fit_sha,
            stats_sha256=stats_sha,
            plasma_ip_threshold_a=framecache.PLASMA_IP_THRESHOLD_A,
            logspec_fit_sha256=logspec_sha,
            n_train_shots=act_stats.n_train_shots,
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
        stats=stats,
        signal_physical_std=sig_stds,
        logspec_fit_path=logspec_fit_path,
        logspec_fit_sha256=logspec_sha,
    )
    framecache.save_json_atomic(manifest, manifest_path)

    counts: Counter = Counter()
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    presence: dict[str, dict[str, bool]] = {}
    frame_lengths: list[int] = []
    errors: list[tuple[str, str]] = []

    enc_state = (
        tuple(signals),
        str(cache_root),
        str(fit_dir),
        str(assignment_path),
        str(framecache.recoders_root()),
        str(logspec_fit_path) if logspec_fit_path is not None else None,
        act_specs,
        act_stats,
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
                ex.map(
                    _encode_job,
                    shot_items,
                    chunksize=max(1, len(shot_items) // (workers * 8)),
                ),
                1,
            ):
                results.append(res)
                if idx % 500 == 0 or idx == len(shot_items):
                    print(
                        f"[port_build_cache] {idx}/{len(shot_items)} shots processed "
                        f"({time.monotonic() - t_start:.1f} s)",
                        flush=True,
                    )

    for shot, split, status, n_frames, pres, err_msg in results:
        counts[status] += 1
        if err_msg is not None:
            errors.append((shot, err_msg))
            print(f"[port_build_cache] ERROR shot {shot}: {err_msg[:200]}", flush=True)
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
        "n_errors": len(errors),
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
    p.add_argument(
        "--logspec-fit",
        default=None,
        help="Optional LogBandDCT fit directory replacing the two Mirnov BandDCT codecs",
    )
    p.add_argument("--out", required=True, help="Output frame-code cache directory")
    p.add_argument("--workers", type=int, default=8, help="Spawn worker count")
    p.add_argument("--max-shots", type=int, default=0, help="Cap total shots (0 = all)")
    p.add_argument("--force", action="store_true", help="Overwrite existing {shot}.pt files")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_cache(
        assignment_path=Path(args.assignment),
        fit_dir=Path(args.fit),
        cache_root=Path(args.cache_root),
        out_dir=Path(args.out),
        logspec_fit=Path(args.logspec_fit) if args.logspec_fit else None,
        workers=args.workers,
        max_shots=args.max_shots,
        force=args.force,
    )
    return 1 if summary.get("n_errors", 0) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
