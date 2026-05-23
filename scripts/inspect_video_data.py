"""Read-only inspection of tangtv / irtv video data.

Step 0 of the Phase C video tokenizer plan
(``docs/video_tokenizer_plan.md``).

Goals
-----
* Confirm native frame rate (~100 fps) and frame count per 50 ms window.
* Measure raw pixel value range (min/max/mean/std) — informs preprocessing
  and stem initialization.
* Report camera availability across a sample of shots — informs the
  validity-mask design and the missing-camera token's training signal.
* Verify HDF5 layout (``ydata`` shape, ``xdata`` length, channel count).

Usage
-----
    pixi run python scripts/inspect_video_data.py \\
        --data_dir /scratch/gpfs/EKOLEMEN/foundation_model \\
        --n_shots 20

Read-only: opens HDF5 files with ``mode='r'`` and never writes anything.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ("tangtv", "irtv")


def inspect_one(
    h5_path: Path, camera: str, sample_window_s: float
) -> dict | None:
    """Inspect one camera in one shot. Return None if camera is missing."""
    with h5py.File(h5_path, "r") as f:
        if camera not in f:
            return None
        grp = f[camera]
        if "ydata" not in grp or "xdata" not in grp:
            return None
        ydata = grp["ydata"]
        xdata = grp["xdata"]
        if ydata.size == 0 or xdata.size < 2:
            return {"present": True, "empty": True}

        x = xdata[:]
        n_frames = x.shape[0]
        t_start, t_end = float(x[0]), float(x[-1])
        duration = t_end - t_start
        actual_fps = (n_frames - 1) / duration if duration > 0 else float("nan")

        shape = tuple(int(s) for s in ydata.shape)
        dtype = str(ydata.dtype)

        # Sample one mid-shot frame for pixel statistics. Avoids loading the
        # full multi-GB array. Layout per the loader is (C, T, H, W).
        mid = n_frames // 2
        frame = ydata[:, mid, :, :]  # (C, H, W)
        frame = np.asarray(frame, dtype=np.float32)
        finite = frame[np.isfinite(frame)]
        nan_frac = float(1.0 - finite.size / frame.size) if frame.size else 0.0

        stats = {
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "p01": float(np.percentile(finite, 1)) if finite.size else float("nan"),
            "p99": float(np.percentile(finite, 99)) if finite.size else float("nan"),
        }

        # Frames inside a representative 50 ms window centered on mid-shot.
        t_mid = (t_start + t_end) / 2.0
        win_lo = t_mid - sample_window_s / 2.0
        win_hi = t_mid + sample_window_s / 2.0
        in_window = int(((x >= win_lo) & (x < win_hi)).sum())

        return {
            "present": True,
            "empty": False,
            "shape": shape,
            "dtype": dtype,
            "n_frames": n_frames,
            "t_start": t_start,
            "t_end": t_end,
            "duration": duration,
            "actual_fps": actual_fps,
            "frames_in_50ms_window": in_window,
            "nan_frac_mid_frame": nan_frac,
            **stats,
        }


def summarise(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"  ({label}: no data)")
        return
    arr = lambda key: np.array([r[key] for r in rows if key in r], dtype=float)

    fps = arr("actual_fps")
    fr50 = arr("frames_in_50ms_window")
    mn = arr("min")
    mx = arr("max")
    mu = arr("mean")
    sd = arr("std")
    nanf = arr("nan_frac_mid_frame")
    p01 = arr("p01")
    p99 = arr("p99")
    nfr = arr("n_frames")

    def line(name, values):
        if values.size == 0:
            print(f"    {name}: (no values)")
            return
        finite = values[np.isfinite(values)]
        n_nan = int(values.size - finite.size)
        nan_note = f"  [{n_nan} NaN]" if n_nan else ""
        if finite.size == 0:
            print(f"    {name}: all NaN ({values.size} shots)")
            return
        print(
            f"    {name}: "
            f"min={finite.min():.3g}  "
            f"med={np.median(finite):.3g}  "
            f"max={finite.max():.3g}  "
            f"(mean={finite.mean():.3g}){nan_note}"
        )

    print(f"  {label} ({len(rows)} shots):")
    line("actual_fps", fps)
    line("frames_in_50ms_window", fr50)
    line("n_frames_total", nfr)
    line("pixel min", mn)
    line("pixel max", mx)
    line("pixel mean", mu)
    line("pixel std", sd)
    line("p01", p01)
    line("p99", p99)
    line("nan_frac (mid frame)", nanf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/scratch/gpfs/EKOLEMEN/foundation_model"),
    )
    ap.add_argument("--n_shots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_window_s", type=float, default=0.05)
    args = ap.parse_args()

    files = sorted(args.data_dir.glob("*_processed.h5"))
    if not files:
        raise SystemExit(f"No *_processed.h5 in {args.data_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[: args.n_shots]

    print(f"Inspecting {len(files)} shots from {args.data_dir}\n")

    by_camera: dict[str, list[dict]] = {c: [] for c in CAMERAS}
    presence: dict[str, int] = {c: 0 for c in CAMERAS}
    empties: dict[str, int] = {c: 0 for c in CAMERAS}
    sample_shape_by_camera: dict[str, tuple] = {}

    for f in files:
        for cam in CAMERAS:
            try:
                row = inspect_one(f, cam, args.sample_window_s)
            except Exception as e:
                print(f"  ! error reading {f.name}::{cam}: {e}")
                continue
            if row is None:
                continue
            presence[cam] += 1
            if row.get("empty"):
                empties[cam] += 1
                continue
            by_camera[cam].append(row)
            if cam not in sample_shape_by_camera:
                sample_shape_by_camera[cam] = row["shape"]

    print("Camera availability across sampled shots:")
    for cam in CAMERAS:
        present = presence[cam]
        empty = empties[cam]
        usable = present - empty
        frac_present = present / len(files)
        frac_usable = usable / len(files)
        print(
            f"  {cam:7s}: group present in {present}/{len(files)} "
            f"({100 * frac_present:.0f}%); "
            f"non-empty {usable}/{len(files)} ({100 * frac_usable:.0f}%); "
            f"empty {empty}"
        )
    print()

    print("Sample HDF5 ydata shape (first usable shot per camera):")
    for cam, shape in sample_shape_by_camera.items():
        print(f"  {cam}: shape={shape}")
    print()

    print("Aggregate stats:")
    for cam in CAMERAS:
        summarise(by_camera[cam], cam)
        print()


if __name__ == "__main__":
    main()