"""TokEye over whole shots: `ae/masks-full` and `ae/dataset-full` (round three).

The GPU job, in the tokeye repo's own venv, as `ae_dataset.sbatch` runs v1:

    cd /scratch/gpfs/nc1514/FusionAIHub
    LABELER_NO_FETCH=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \\
        /scratch/gpfs/nc1514/tokeye/.venv/bin/python scripts/labeler/ae_masks_full.py \\
        [--shots 170720,176053 | --pilot] [--device auto] [--batch 2] \\
        [--tile-ms 0] [--workers 4] [--overwrite]

and then the check against v1, in the labelmaker env:
`python -m labeler.ae.full --check`.

v1 (`ae_dataset.py --stage probs`) ran TokEye over the aemodes arrow cache,
which holds 0-2 s only. This runs the same transform
(`ae_dataset.spectrogram_tokeye`), the same model (`ae_dataset.load_model`,
`ae_dataset.mask_probs`) and the same cleaning (`clean_mask` at `PROB_CODE`,
then `persist_open` over `PERSIST_FRAMES`), and writes the same files under
v1's names and keys. What differs:

- **the input** is the raw cache's CO2 over the whole shot, 0 to min(record
  end, `full.CROP_END_MS`), resampled to ~500 kHz (`full.input_signals`), so the
  frames are that record's (`full.frame_times`), not v1's 7820;
- **the annotation** (`ann`, `lfm`, `frame_labels`) is UCI's from v1's clean
  file, at the nearest v1 frame and False after v1's last (`full.annotation`);
- **the extras** (`full.EXTRA_KEYS`) in each clean file: `ann_until_ms`, the
  input's first sample `t0_input_ms` and exact rate `fs_hz`, and `tile_ms`;
- **the tiles**: `--tile-ms` > 0 runs the model over runs of frames with a margin
  either side (`full.tiles`, `full.tiled_probs`), the fallback for a shot that
  does not fit on the GPU; 0, the default, is one pass over the whole shot.

`dataset-full/<stem>.npz` is the spectrogram half v1's `probs` stage writes;
nothing here writes a label. A pool of `--workers` CPU processes reads,
resamples and transforms the shots ahead of the GPU. A stem whose three files
all exist is skipped unless `--overwrite`, so a cut job resumes where it
stopped. Nothing is fetched: `LABELER_NO_FETCH` is set to 1 unless given, so a
shot the raw cache lacks is an error.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ae_dataset  # the sys.path inserts above make these work

from labeler.ae.full import (
    annotation,
    dataset_full_dir,
    frame_times,
    input_signals,
    masks_full_dir,
    tiled_probs,
    tiles,
)
from labeler.ae.labels import (
    BAND_HI_BIN,
    BAND_LO_BIN,
    HOP,
    N_BINS,
    PERSIST_FRAMES,
    bin_active_fraction,
    bin_freqs_khz,
    clean_mask,
    persist_open,
)
from labeler.ae.xpower import tokeye_masks
from labeler.ae.xpower.data import seldnet_split
from labeler.config import Paths, atomic_path
from labeler.events.raw import NO_FETCH_ENV

#: The owner's three shots (spec section 0), first in the pilot.
OWNER_SHOTS = (170720, 176053, 176041)
#: Shots in the pilot, the owner's three included.
PILOT_SIZE = 16


def _shot(stem: str) -> int:
    """The shot of a stem `<shot>_<split>`."""
    return int(stem.split("_")[0])


def stems(paths: Paths, shots: list[int] | None) -> list[str]:
    """f"{shot}_{split}" of seldnet_split(tokeye_masks(paths)), in shot order,
    or only `shots` (SystemExit naming any shot not among the 180)."""
    masks = tokeye_masks(paths)
    split = seldnet_split(masks)
    if not split:
        raise SystemExit(f"{masks}: no v1 TokEye clean files")
    wanted = sorted(split) if shots is None else sorted({int(s) for s in shots})
    unknown = [s for s in wanted if s not in split]
    if unknown:
        raise SystemExit(f"not among the {len(split)} AE shots of {masks}: {unknown}")
    return [f"{s}_{split[s]}" for s in wanted]


def pilot(all_stems: list[str]) -> list[str]:
    """OWNER_SHOTS' stems, then the others in order, PILOT_SIZE in all."""
    by_shot = {_shot(s): s for s in all_stems}
    first = [by_shot[s] for s in OWNER_SHOTS if s in by_shot]
    rest = [s for s in all_stems if s not in first]
    return (first + rest)[:PILOT_SIZE]


def prepare(shot: int, root: str, raw_cache: str | None) -> dict:
    """CPU, in a worker: input_signals, then ae_dataset.spectrogram_tokeye;
    {"raw", "norm", "mean", "std", "n_samples", "t0_ms", "fs_hz"}."""
    paths = replace(
        Paths.from_env(),
        root=Path(root),
        raw_cache=Path(raw_cache) if raw_cache else None,
    )
    signals, t0_ms, fs_hz = input_signals(shot, paths=paths)
    raw, norm, mean, std = ae_dataset.spectrogram_tokeye(signals)
    return {
        "raw": raw,
        "norm": norm,
        "mean": mean,
        "std": std,
        "n_samples": int(signals.shape[1]),
        "t0_ms": float(t0_ms),
        "fs_hz": float(fs_hz),
    }


def _outputs(paths: Paths, stem: str) -> tuple[Path, Path, Path]:
    """A stem's probs, clean and dataset files."""
    out = masks_full_dir(paths)
    return (
        out / f"{stem}_probs.npz",
        out / f"{stem}_clean.npz",
        dataset_full_dir(paths) / f"{stem}.npz",
    )


def _savez_compressed(path: Path, **arrays) -> None:
    """`np.savez_compressed` through a temporary sibling `<name>.npz.tmp`, renamed
    into place as `ae_dataset.save_npz` does."""
    with atomic_path(path) as tmp, open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def write_shot(
    stem: str,
    prepared: dict,
    *,
    paths: Paths,
    model,
    device: str,
    batch: int,
    tile_ms: float,
) -> dict:
    """TokEye's probabilities, the cleaned masks and the spectrogram of one shot,
    written as v1's `run_probs` writes them, plus EXTRA_KEYS in the clean file.
    Returns {"stem", "frames", "clean_pixels", "seconds"}."""
    started = time.time()
    raw, norm = prepared["raw"], prepared["norm"]
    mean, std = prepared["mean"], prepared["std"]
    fs_hz, t0_ms = prepared["fs_hz"], prepared["t0_ms"]
    t_ms = frame_times(prepared["n_samples"], t0_ms, fs_hz)
    n_frames = len(t_ms)
    if raw.shape != (len(ae_dataset.CHANNELS), N_BINS, n_frames):
        raise ValueError(
            f"{stem}: spectrogram {raw.shape}, not "
            f"{(len(ae_dataset.CHANNELS), N_BINS, n_frames)}"
        )
    tile_frames = round(tile_ms / (HOP / fs_hz * 1000)) if tile_ms > 0 else 0
    prob = tiled_probs(
        ae_dataset.mask_probs,
        model,
        norm,
        device,
        batch,
        tiles(n_frames, tile_frames),
    )

    code = ae_dataset.PROB_CODE
    coherent = prob[:, 0] >= code
    mode = clean_mask(prob[:, 0], prob[:, 1], threshold=code)
    clean = persist_open(mode, PERSIST_FRAMES)
    labels = annotation(tokeye_masks(paths) / f"{stem}_clean.npz", t_ms)
    ann = labels["ann"]
    freqs = bin_freqs_khz(fs_hz / 1000)
    centroids = ae_dataset.window_centroids(raw, clean, freqs, ann)

    prob_path, clean_path, spec_path = _outputs(paths, stem)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    _savez_compressed(prob_path, prob=prob, ann=ann)
    _savez_compressed(
        clean_path,
        mask_clean=np.packbits(clean, axis=-1),
        mask_raw=np.packbits(coherent, axis=-1),
        active_frac_clean=bin_active_fraction(clean).astype(np.float32),
        active_frac_raw=bin_active_fraction(coherent).astype(np.float32),
        n_pixels_raw=coherent.sum(axis=(1, 2)).astype(np.int64),
        n_pixels_no_transient=mode.sum(axis=(1, 2)).astype(np.int64),
        n_pixels_clean=clean.sum(axis=(1, 2)).astype(np.int64),
        protected=ae_dataset.protected_bins(centroids, freqs),
        window_centroids_khz=np.asarray(centroids, dtype=np.float32),
        ann=ann,
        lfm=labels["lfm"],
        frame_labels=labels["frame_labels"],
        t_ms=t_ms.astype(np.float32),
        freqs=freqs.astype(np.float32),
        spec_mean=mean,
        spec_std=std,
        ann_until_ms=np.float64(labels["ann_until_ms"]),
        t0_input_ms=np.float64(t0_ms),
        fs_hz=np.float64(fs_hz),
        tile_ms=np.float64(tile_ms),
    )
    ae_dataset.save_npz(
        spec_path,
        spec=norm[:, BAND_LO_BIN:BAND_HI_BIN, :].astype(np.float16),
        annotated=ann.astype(np.uint8),
        split=np.asarray(stem.rsplit("_", 1)[1]),
        spec_mean=mean,
        spec_std=std,
        freq_khz_bins=freqs[BAND_LO_BIN:BAND_HI_BIN].astype(np.float32),
    )
    return {
        "stem": stem,
        "frames": n_frames,
        "clean_pixels": int(clean.sum()),
        "seconds": time.time() - started,
    }


def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = p.add_mutually_exclusive_group()
    which.add_argument(
        "--shots", help="comma-separated shot numbers; default every AE180 shot"
    )
    which.add_argument(
        "--pilot",
        action="store_true",
        help=f"the owner's three shots, then the next, {PILOT_SIZE} in all",
    )
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--batch", type=int, default=2, help="CO2 chords per forward pass")
    p.add_argument(
        "--tile-ms",
        type=float,
        default=0.0,
        help="run the model over tiles of this many ms; 0 is one pass a shot",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="CPU processes preparing shots ahead of the GPU; 0 prepares in line",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.batch < 1:
        p.error("--batch must be at least 1")
    if args.workers < 0:
        p.error("--workers must be 0 or more")
    shots = None
    if args.shots:
        try:
            shots = [int(s) for s in args.shots.split(",") if s.strip()]
        except ValueError:
            p.error(f"--shots {args.shots!r}: comma-separated shot numbers")
        if not shots:
            p.error(f"--shots {args.shots!r}: no shot number")

    # Before the pool, so its workers inherit it: a shot the raw cache lacks is
    # an error, never a live fetch from the GPU node.
    os.environ.setdefault(NO_FETCH_ENV, "1")
    paths = Paths.from_env()
    todo = stems(paths, shots)
    if args.pilot:
        todo = pilot(todo)
    if not args.overwrite:
        todo = [s for s in todo if not all(f.is_file() for f in _outputs(paths, s))]
    device = _device(args.device)
    print(
        f"device={device} shots={len(todo)} batch={args.batch} "
        f"tile_ms={args.tile_ms:g} workers={args.workers}",
        flush=True,
    )
    if not todo:
        return 0

    root, raw_cache = str(paths.root), str(paths.raw_cache)
    queue, ahead = deque(todo), deque()
    pool = None
    if args.workers > 0:
        # Before the model is loaded, so the workers start on the first shots
        # while it loads, and spawned rather than forked from a CUDA process.
        pool = ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        )

    def submit() -> None:
        """Keep at most workers + 1 shots in preparation."""
        while queue and len(ahead) < args.workers + 1:
            stem = queue.popleft()
            ahead.append((stem, pool.submit(prepare, _shot(stem), root, raw_cache)))

    try:
        if pool is not None:
            submit()
        model = ae_dataset.load_model(device)
        started = time.time()
        for n in range(1, len(todo) + 1):
            if pool is None:
                stem, future = queue.popleft(), None
            else:
                stem, future = ahead.popleft()
                submit()
            try:
                if future is None:
                    prepared = prepare(_shot(stem), root, raw_cache)
                else:
                    prepared = future.result()
                done = write_shot(
                    stem,
                    prepared,
                    paths=paths,
                    model=model,
                    device=device,
                    batch=args.batch,
                    tile_ms=args.tile_ms,
                )
            except Exception as exc:
                exc.add_note(f"while running TokEye over {stem}")
                raise
            # The shot's two spectrograms go before the wait on the next one.
            del prepared
            print(
                f"[{n}/{len(todo)}] {stem} frames={done['frames']} "
                f"clean pixels={done['clean_pixels']} "
                f"model and write {done['seconds']:.1f}s, "
                f"{(time.time() - started) / n:.1f}s/shot",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if device == "cuda":
        import torch

        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak GPU memory: {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
