"""Rollout movie from a demo data file: measured against predicted, revealed in time.

Reads the ``extract`` output of ``demo_controllability_fig.py`` (physical units, no model
or codecs needed) and writes an mp4 in the layout of ``scripts/training/eval_e2e_animation.py``:
the measured and predicted camera on top, growing T_e / n_e / T_i / v_tor traces below left,
the measured and predicted spectrogram below right, all sharing one time cursor. Before the prediction start the model sees the
measurement; from there on it runs on the actuators alone.

Usage::

    python analysis/demo_rollout_video.py --data demo_201240_both_data.npz --out 201240_rollout.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_controllability_fig as dm  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spectro", default="", help="spectrogram to show (default: first in the data file)")
    p.add_argument("--channels", default="", help='override chords, e.g. "cer_ti=24,2,1;cer_rot=19,18,20"')
    p.add_argument("--camera", default="tangtv_upper", help="camera row (tangtv_upper, tangtv_lower, none)")
    p.add_argument("--camera_sigma", type=float, default=1.5, help="Gaussian smoothing of camera frames (px)")
    p.add_argument("--ylim", type=float, default=0, help="max frequency shown (kHz); 0 = full range")
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--n_core", type=int, default=3)
    p.add_argument("--substeps", type=int, default=4, help="movie frames per 50 ms data frame")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hold_s", type=float, default=2.0, help="seconds to hold the final frame")
    p.add_argument("--font", type=float, default=13)
    p.add_argument("--dpi", type=int, default=150)
    a = p.parse_args()
    camera, a.camera = a.camera, ""
    ctx = dm._setup(a)
    d, t, k0, F = ctx["d"], ctx["t"], ctx["k0"], ctx["F"]
    name = a.spectro or ctx["spectro"]
    gt_key = f"spectro_gt__{name}" if f"spectro_gt__{name}" in d.files else "spectro_gt"
    pred_key = f"spectro_pred__real__{name}" if f"spectro_pred__real__{name}" in d.files else "spectro_pred__real"
    fmax_key = f"spectro_fmax_khz__{name}" if f"spectro_fmax_khz__{name}" in d.files else "spectro_fmax_khz"
    shown = dm.spectro_label(d, name)
    finite = d[gt_key][np.isfinite(d[gt_key])]
    lim = (float(np.percentile(finite, 1.0)), float(np.percentile(finite, 99.5)))
    t_end = t[-1] + dm.FRAME_S
    extent = (t[0], t_end, 0, float(d[fmax_key]))
    t_pred = dm.T0 + dm.FRAME_S * k0

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    cams = None
    if f"camera_gt__{camera}" in d.files:
        from scipy.ndimage import gaussian_filter

        gt_cam = np.asarray(d[f"camera_gt__{camera}"], dtype=np.float32)
        pred_cam = np.asarray(d[f"camera_pred__real__{camera}"], dtype=np.float32)
        mean = gt_cam[k0:].mean(0)
        floor = np.median(mean) - 1.0
        rows, cols = np.where(mean.mean(1) > floor)[0], np.where(mean.mean(0) > floor)[0]
        crop = (slice(None), slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        sig = (0, a.camera_sigma, a.camera_sigma)
        cams = [gaussian_filter(gt_cam[crop], sig), gaussian_filter(pred_cam[crop], sig)]
        cam_lim = np.percentile(np.concatenate([c.ravel() for c in cams]), [1.0, 99.7])
        top = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.93, bottom=0.66, wspace=0.04)
        cam_ims = []
        for j, (seq, label) in enumerate(zip(cams, ("measured", "prediction"))):
            ax = fig.add_subplot(top[0, j])
            cmap = dm.CMAPS["truth" if j == 0 else "real"]
            cam_ims.append(ax.imshow(seq[0], cmap=cmap, vmin=cam_lim[0], vmax=cam_lim[1], aspect="auto"))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{dm.CAMERA_NAMES[camera]}, {label}", fontsize=a.font)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], left=0.06, right=0.98, bottom=0.07,
                          top=0.59 if cams else 0.93, wspace=0.32, hspace=0.35 if cams else 0.3)
    lines, cursors = [], []
    for i, n in enumerate(dm.SLOW):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        gt, pred = dm._tr(ctx, a, "gt", "real", n), dm._tr(ctx, a, "pred", "real", n)
        pred[:k0] = np.nan
        both = np.concatenate([gt[np.isfinite(gt)], pred[np.isfinite(pred)]])
        pad = 0.08 * (both.max() - both.min() + 1e-9)
        ax.set(xlim=(t[0], t_end), ylim=(both.min() - pad, both.max() + pad), ylabel=dm.LABELS[n][0])
        if i >= 2:
            ax.set_xlabel("time (s)")
        ax.axvspan(t[0], t_pred, color="0.93", lw=0)
        (l_gt,) = ax.plot([], [], color=dm.COLORS["truth"], lw=2, label="measured")
        (l_pr,) = ax.plot([], [], color=dm.COLORS["real"], lw=2, ls="--", label="prediction")
        lines.append((l_gt, gt, l_pr, pred))
        cursors.append(ax.axvline(t[0], color="0.4", lw=1))
        dm._label_trace(ax, ctx, n, a.font)
        if i == 0:
            ax.legend(loc="lower right", frameon=False)
            ax.text(0.5 * (t[0] + t_pred), ax.get_ylim()[1], "context", ha="center", va="bottom",
                    fontsize=a.font - 2, color="0.35")
    masks = []
    for row, (img, label) in enumerate(((d[gt_key], f"{shown} measured"), (d[pred_key], f"{shown} prediction"))):
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(img, aspect="auto", origin="lower", cmap=dm.CMAPS["truth" if row == 0 else "real"],
                  vmin=lim[0], vmax=lim[1],
                  extent=extent, interpolation="none")
        ax.axvline(t_pred, color="w", lw=1, ls=":")
        ax.set(title=label, ylabel="freq (kHz)", xlim=(t[0], t_end))
        if a.ylim:
            ax.set_ylim(0, a.ylim)
        if row == 1:
            ax.set_xlabel("time (s)")
        masks.append(ax.axvspan(t[0], t_end, color="white", lw=0, zorder=3))
        cursors.append(ax.axvline(t[0], color="0.4", lw=1, zorder=4))
    clock = fig.text(0.06, 0.965, "", fontsize=a.font + 2, ha="left", va="center")
    fig.text(0.98, 0.965, f"DIII-D #{ctx['shot']}  (held-out)", fontsize=a.font, ha="right", va="center",
             color="0.35")

    n_move = (F - 1) * a.substeps + 1
    n_total = n_move + int(a.hold_s * a.fps)

    def update(i):
        tc = t[0] + (t_end - t[0]) * min(i, n_move - 1) / (n_move - 1)
        idx = int(np.searchsorted(t, tc, side="right"))
        for l_gt, gt, l_pr, pred in lines:
            l_gt.set_data(t[:idx], gt[:idx])
            l_pr.set_data(t[:idx], pred[:idx])
        for c in cursors:
            c.set_xdata([tc, tc])
        for m in masks:
            m.set_x(tc)
            m.set_width(t_end - tc)
        if cams:
            for im, seq in zip(cam_ims, cams):
                im.set_data(seq[max(idx - 1, 0)])
        clock.set_text(f"t = {tc:.2f} s" + ("   prediction" if tc >= t_pred else "   measured context"))
        return []

    matplotlib.rcParams["animation.ffmpeg_path"] = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    anim = animation.FuncAnimation(fig, update, frames=n_total, blit=False)
    anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, codec="libx264", bitrate=6000,
                                                   extra_args=["-pix_fmt", "yuv420p"]), dpi=a.dpi)
    update(n_total - 1)
    fig.savefig(str(Path(a.out).with_suffix(".png")), dpi=a.dpi)
    print(f"wrote {a.out} ({n_total} frames, {n_total / a.fps:.1f} s)")


if __name__ == "__main__":
    main()
