"""Export raw tangtv cam frames (target + prediction) at a chosen
shot time. No transformations, no overlays, no tokamak layout —
just two greyscale PNGs side by side.

Usage:
    python scripts/training/export_tangtv_cam_frames.py \\
        --checkpoint /path/to/best.pt --shot_id 200729 [--t_s 2.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_e2e_animation_tokamak import (  # noqa: E402
    collect_shot_predictions_limited, load_model,
)
from eval_e2e import detect_stage_K  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--data_dir", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model"),
    )
    p.add_argument(
        "--stats_path", type=Path,
        default=Path("/lustre/orion/fus187/proj-shared/foundation_model_meta/"
                     "preprocessing_stats.pt"),
    )
    p.add_argument("--shot_id", type=int, default=200729)
    p.add_argument(
        "--output_dir", type=Path,
        default=Path("eval_runs/animations"),
    )
    p.add_argument("--t_s", type=float, default=2.5,
                   help="Shot time in seconds to export.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--chunk_duration_s", type=float, default=0.05)
    p.add_argument("--step_size_s", type=float, default=0.01)
    p.add_argument("--warmup_s", type=float, default=1.0)
    p.add_argument("--K", type=int, default=0)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--max_chunks", type=int, default=64,
        help="Cap inference at the first N windows. Default keeps "
             "inference to one batch since we only need one frame.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    shot_file = args.data_dir / f"{args.shot_id}_processed.h5"
    if not shot_file.exists():
        raise SystemExit(f"shot file not found: {shot_file}")

    # ── GT cam frame at t = args.t_s from raw H5 (channel [4] = PERP) ─
    with h5py.File(shot_file, "r") as f:
        x = f["tangtv/xdata"][:]
        gt_idx = int(np.argmin(np.abs(x - args.t_s)))
        gt_frame = f["tangtv/ydata"][4, gt_idx]      # (H, W)
        gt_t_s = float(x[gt_idx])
    print(f"GT frame: index {gt_idx}, t = {gt_t_s:.3f} s, "
          f"shape = {gt_frame.shape}, "
          f"range = [{np.nanmin(gt_frame):.1f}, {np.nanmax(gt_frame):.1f}]")

    # ── Prediction cam frame via model inference ──────────────────
    print(f"loading model from {args.checkpoint}")
    model, ckpt = load_model(args.checkpoint, device)
    K = args.K if args.K > 0 else detect_stage_K(ckpt)
    print(f"K = {K}; running inference (cap {args.max_chunks} windows)…")
    stats = torch.load(args.stats_path, weights_only=False)
    blobs = collect_shot_predictions_limited(
        model=model, file_path=shot_file, device=device,
        args=args, stats=stats, K=K, max_windows=args.max_chunks,
    )
    if "tangtv" not in blobs:
        raise SystemExit("model did not return tangtv predictions")
    pred_video = blobs["tangtv"]["pred"].numpy()
    # Window w predicts t = warmup + (w+1) * chunk_duration_s ..
    #                       warmup + (w+2) * chunk_duration_s
    # We use the LAST of n_output_frames=3 → t at end of window.
    n_w = pred_video.shape[0]
    win_end_t = (args.warmup_s
                 + (np.arange(n_w) + 2) * args.chunk_duration_s)
    pred_idx = int(np.argmin(np.abs(win_end_t - args.t_s)))
    pred_frame = pred_video[pred_idx, 0, -1]       # (H, W) — PERP, last frame
    print(f"pred frame: window {pred_idx}, t = {win_end_t[pred_idx]:.3f} s, "
          f"shape = {pred_frame.shape}, "
          f"range = [{np.nanmin(pred_frame):.3f}, {np.nanmax(pred_frame):.3f}]")

    # ── Save both as plain greyscale PNGs + raw .npy ───────────────
    # PNG: no title, no axes, no padding; figure background transparent.
    # NPY: raw float values, preserving the original dynamic range
    # (PNG quantises to 8-bit grey; .npy keeps the model's float
    # output / raw H5 intensities exactly).
    for name, frame in [("target", gt_frame), ("prediction", pred_frame)]:
        png_out = args.output_dir / f"{args.shot_id}_cam_{name}.png"
        fig, ax = plt.subplots(figsize=(7.2, 2.4))
        ax.imshow(frame, cmap="gray", aspect="equal")
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            png_out, dpi=140, transparent=True,
            bbox_inches="tight", pad_inches=0,
        )
        plt.close(fig)
        print(f"saved: {png_out}")
        npy_out = args.output_dir / f"{args.shot_id}_cam_{name}.npy"
        np.save(npy_out, frame.astype(np.float32))
        print(f"saved: {npy_out}  (shape {frame.shape}, "
              f"dtype float32)")


if __name__ == "__main__":
    main()
