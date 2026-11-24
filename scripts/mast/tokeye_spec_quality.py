"""Score every shot's Mirnov spectrograms with TokEye (``port.specquality``), on CPU.

    scripts/mast/py.sh scripts/mast/tokeye_spec_quality.py --out DIR [--workers 24 --threads 6]

Writes ``DIR/shots/<shot>.npz`` (one per shot, skipped when present) and ``DIR/meta.json``.
The plasma window of each shot is the frame cache's (``t0``, ``n_frames``): the frames where
|Ip| is above the cache's threshold. Weights: TokEye's ``big_tf_unet_251210.pt``, verified
against the sha256 pinned in ``labeler.events.unet``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import numpy as np

DEFAULT_CACHE_V1 = Path("/models/intermediate/mast/cache/v1")
DEFAULT_FRAMES = Path("/models/intermediate/mast/stage2r/frame_codes_v1b_logspec_cam_live_da")
DEFAULT_CKPT = Path("/models/tokeye_dl/big_tf_unet/big_tf_unet_251210.pt")

_MODEL = None
_STATS = None
_SHA = None


def _init(ckpt: str, cache_v1: str, threads: int) -> None:
    global _MODEL, _STATS, _SHA
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch

    from labeler.events.unet import load_unet
    from tokamak_foundation_model.ignite.mast import transform

    torch.set_num_threads(threads)
    _MODEL = load_unet(ckpt)
    root = Path(cache_v1)
    _STATS = transform.load_stats(root / "stats.npz")
    _SHA = str(json.loads((root / "manifest.json").read_text())["stats_sha256"])


def _one(job: tuple[str, str, str, str]) -> tuple[str, str, float]:
    shot, cache_v1, frames, out = job
    import torch

    from tokamak_foundation_model.ignite.mast import cache, transform
    from tokamak_foundation_model.ignite.mast.port import framecache, specquality

    t_start = time.time()
    dst = Path(out) / "shots" / f"{shot}.npz"
    if dst.exists():
        return shot, "exists", 0.0
    entry = torch.load(Path(frames) / f"{shot}.pt", weights_only=False, map_location="cpu")
    sd = cache.read_shot(Path(cache_v1) / "shots" / f"{shot}.h5", list(specquality.SIGNALS),
                         stats_sha256=_SHA)
    sigs = {}
    for name in specquality.SIGNALS:
        cs = sd.get(name)
        if cs is None or cs.values.shape[-1] == 0:
            continue
        sigs[name] = (transform.to_physical(cs.values, _STATS[name]), float(cs.t0), float(cs.dt))
    res = specquality.score_shot(_MODEL, sigs, float(entry["t0"]), int(entry["n_frames"]),
                                 framecache.DT_FRAME)
    res["signals_present"] = np.asarray([specquality.SHORT[s] for s in sigs], dtype="U8")
    tmp = dst.with_name(f"{dst.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **res)
    os.replace(tmp, dst)
    return shot, "ok", time.time() - t_start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache_v1", type=Path, default=DEFAULT_CACHE_V1)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--shots", nargs="*", default=None, help="default: every shot in splits.json")
    args = ap.parse_args()

    from labeler.events.unet import CHECKPOINT_SHA256
    from tokamak_foundation_model.ignite.mast import paths
    from tokamak_foundation_model.ignite.mast.port import framecache, specquality

    out = paths.prepare_write_root(args.out)
    (out / "shots").mkdir(exist_ok=True)
    splits = json.loads((args.frames / "splits.json").read_text())
    shots = args.shots or sorted({str(s) for k in ("train", "val", "test") for s in splits[k]}, key=int)
    meta = {
        "created_utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
        "ckpt": str(args.ckpt), "ckpt_sha256": CHECKPOINT_SHA256,
        "cache_v1": str(args.cache_v1), "frames": str(args.frames),
        "params": {k: getattr(specquality, k) for k in (
            "F_UP", "CLIP_LOW", "CLIP_HIGH", "LIT", "VALID_BIN_FRAC", "PICKUP_OFF_FRAC",
            "MIN_OFF_COLUMNS", "PICKUP_STATIONARY_FRAC", "PICKUP_DILATE", "SMOOTH")},
        "frame_dt": framecache.DT_FRAME, "n_shots": len(shots),
    }
    framecache.save_json_atomic(meta, out / "meta.json")
    jobs = [(s, str(args.cache_v1), str(args.frames), str(out)) for s in shots]
    ctx = multiprocessing.get_context("spawn")
    t0 = time.time()
    n_ok = n_err = 0
    with concurrent.futures.ProcessPoolExecutor(
        args.workers, mp_context=ctx, initializer=_init,
        initargs=(str(args.ckpt), str(args.cache_v1), args.threads),
    ) as ex:
        futs = {ex.submit(_one, j): j[0] for j in jobs}
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            shot = futs[f]
            try:
                _, status, dt = f.result()
                n_ok += 1
            except Exception as e:  # one bad shot must not stop the sweep; it is logged
                n_err += 1
                print(f"[tokeye] {shot} ERROR {type(e).__name__}: {e}", flush=True)
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"[tokeye] {i}/{len(jobs)} ok={n_ok} err={n_err} {el / 60:.1f} min "
                      f"eta {el / i * (len(jobs) - i) / 60:.1f} min", flush=True)
    print(f"[tokeye] done ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
