"""Export the camera movies shown in the paper figures as IDL save files.

Exactly what the figures plot, taken from the figure data files (the ``camera_gt__`` and
``camera_pred__`` / ``camera_member__`` arrays that ``make_figures.py`` and
``overview_main.py`` draw):

| Figure | Shot | Series | Prediction shown |
|---|---|---|---|
| Fig. 1 replay | 204011 | measured actuators | mean of 8 rollouts |
| Overview shot | 204011 | measured actuators, torque -> 0 | one rollout (``camera_member``) |
| Fig. 3 detachment | 201081, 201085 | measured actuators | mean of 8 rollouts |
| Fig. 4 gas paths | 201081 | gas of 201079 / 201083 / 201084 | mean of 8 rollouts |

The figures' "measured" row is the measured frames after the camera codec; it is exported
as ``..._measured_codec.sav``. Fig. 5's upper camera of 204011 is not exported: the camera
did not record, so there is no measured brightness to restore counts with.

File layout: that of the colleague's raw camera files
(``emission_structure_cam240paria_206879_raw.sav``): structure ``EMISSION_STRUCTURE``
with ``VID_FRAMES`` (n,) float32, ``VID_TIMES`` (n,) float64 in ms and ``VID``
(n, 240, 720) uint8 counts, written by ``idl_sav.write_struct``.

Frames. One frame per 50 ms model frame over the prediction (2-6 s, 80 frames), as in the
figures: the mean of the codec window's 5 sub-frames, i.e. the mean of two 20 ms grid
samples of the camera, stamped at their mid time. The model's 120 x 360 image is
upsampled to 240 x 720 (bilinear). Display-only steps of the figures (crop, 1.5 px
Gaussian smoothing, colour limits) are not applied.

Brightness. The codec z-scores each window (mean and std over its 5 sub-frames and all
pixels, std clamped at 1) and the model predicts the z-scored image only. Counts are
restored with the measured window's own mean and std (decided 2026-11-17), so brightness
level and contrast are the measurement's and the spatial structure is the model's.

Checks (``manifest.csv``): the figure arrays are recomputed from the rollout archives
(decode, view, sub-frame mean, seed mean) and must match to float16 precision; the
measured windows re-encoded with the frozen codec must give the archive's measured codes.

Usage (CPU node): ``python analysis/paper_figures/export_camera_sav.py --out DIR``
"""
import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import idl_sav  # noqa: E402

from tokamak_foundation_model.ignite import eval_dynamics as ev  # noqa: E402
from tokamak_foundation_model.ignite import train_dynamics as td  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUN = Path("/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4/runs/soup_maskfree")
TMPL = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/{m}/codec_best.pt"
ENS = Path("/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures_live_unroll/ens8_soup_maskfree")
WORK = Path("/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data8_maskfree")
# Processed tangtv groups rebuilt from the raw tree (claude_scratch/camera_sav/
# rebuild_tangtv.py; bitwise equal to the processed files where those exist).
PROCESSED = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/camera_sav/processed_tangtv"
CAM = "tangtv_lower"
CAM_NAME = {0: "cam240paria", 1: "cam240perp"}  # view index within tangtv_lower
T0, FRAME_S, SEEDS = 1.0, 0.05, range(1, 9)
# figure, data file, shot, data key, archive case, shown times (s), prediction kind
CASES = [
    ("fig1_replay", ENS / "demo_204011_val_data.npz", "204011", "real", "real",
     "2.5,3.5,4.5,5.5", "mean"),
    ("overview_shot", WORK / "fs_204011_data.npz", "204011", "real", "real", "5.0", "member"),
    ("overview_shot", WORK / "fs_204011_data.npz", "204011", "tq_lin_zero", "tq_lin_zero",
     "5.0", "member"),
    ("fig3_detachment", ENS / "demo_201081_cm_data.npz", "201081", "real", "real",
     "2.5,4.0,5.5", "mean"),
    ("fig3_detachment", ENS / "demo_201085_cm_data.npz", "201085", "real", "real",
     "2.5,4.0,5.5", "mean"),
] + [("fig4_gas_paths", ENS / "demo_201081_cm_data.npz", "201081", f"donor_{s}", f"donor{s}",
      "2.5,4.0,5.5", "mean") for s in (201079, 201083, 201084)]


def archive(shot, case, seed):
    d = RUN / f"_demo_{shot}_{case}_s30000_g3_seed{seed}"
    return np.load(next(d.glob(f"codes_{shot}_step*.npz")), allow_pickle=True)


class Shot:
    """Measured windows of one shot: codec stats, re-encoded codes, frame times."""

    def __init__(self, shot, codec, cfg, fam):
        ds = td._single_shot_dataset(CAM, fam, cfg, shot, PROCESSED, t0_start=T0)
        w = torch.stack([td._window(ds[t]) for t in range(100)]).float()
        self.mu = w.mean(dim=(2, 3, 4)).numpy()
        self.sd = w.std(dim=(2, 3, 4)).clamp(min=1.0).numpy()
        with torch.no_grad():
            self.codes = td._encode_frames(ds, codec, 100).numpy()
        x = h5py.File(Path(PROCESSED) / f"{shot}_processed.h5", "r")["tangtv"]["xdata"][:]
        fps = (len(x) - 1) / (x[-1] - x[0])
        assert round(FRAME_S * fps) == 2, "expected two 20 ms grid samples per window"
        start = np.array([round((T0 + FRAME_S * f - x[0]) * fps) for f in range(100)])
        # Mid time of the window's two grid samples (ms).
        self.times = (x[start] + 0.5 / fps) * 1e3


def view_frames(decoded, view):
    """(F, C, 5, H, W) decoded windows -> (F, H, W): one view, sub-frames averaged."""
    return np.nanmean(np.asarray(decoded)[:, view], axis=1)


def counts(z, mu, sd):
    """z-scored (n, 120, 360) -> counts (n, 240, 720) uint8."""
    c = torch.from_numpy((z * sd[:, None, None] + mu[:, None, None]).astype(np.float32))
    up = F.interpolate(c[:, None], size=(240, 720), mode="bilinear", align_corners=False)
    return np.clip(np.rint(up[:, 0].numpy()), 0, 255).astype(np.uint8)


def write(path, vid, times):
    path.parent.mkdir(parents=True, exist_ok=True)
    idl_sav.write_struct(path, "emission_structure", {
        "VID_FRAMES": np.arange(len(vid), dtype=np.float32),
        "VID_TIMES": np.asarray(times, dtype=np.float64),
        "VID": vid,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    out = Path(ap.parse_args().out)
    torch.set_num_threads(16)
    codec, cfg, fam = td.load_frozen_codecs([CAM], repo=REPO, tmpl=TMPL)[CAM]
    codec.eval()
    dev = torch.device("cpu")
    shots, rows, done_gt = {}, [], set()
    for fig, data, shot, key, case, shown, kind in CASES:
        d = np.load(data, allow_pickle=True)
        k0 = int(d["k0"])
        if shot not in shots:
            shots[shot] = Shot(shot, codec, cfg, fam)
        s = shots[shot]
        arcs = [archive(shot, case, k) for k in SEEDS]
        gt_codes = np.asarray(arcs[0][f"gt__{CAM}"])
        agree = float((s.codes[k0:] == gt_codes[k0:100]).mean())
        dec_gt = ev.decode_flat_chunked(codec, ev.archived(arcs[0], f"gt__{CAM}")[:100], dev)
        # The figure's view: the measured image with structure inside a 10 px margin.
        gt_mean = np.nanmean(np.asarray(dec_gt)[k0:100], axis=(0, 2))
        view = int(np.argmax([np.nanstd(v[10:-10, 10:-10]) for v in gt_mean]))
        gt_fig = np.asarray(d[f"camera_gt__{CAM}"], np.float32)
        err_gt = float(np.abs(view_frames(dec_gt, view) - gt_fig).max())
        each = [view_frames(ev.decode_flat_chunked(codec, ev.archived(z, f"pred__{CAM}")[:100], dev),
                            view) for z in arcs]
        if kind == "mean":
            pred_fig = np.asarray(d[f"camera_pred__{key}__{CAM}"], np.float32)
            err_pred = float(np.abs(np.mean(each, axis=0) - pred_fig).max())
            label, seeds = f"{case}_mean8", "1-8"
        else:
            member = int(d["camera_member"])
            pred_fig = np.asarray(d[f"camera_member__{key}__{CAM}"], np.float32)
            err_pred = float(np.abs(each[member] - pred_fig).max())
            label, seeds = f"{case}_seed{member + 1}", str(member + 1)
        if max(err_gt, err_pred) > 0.02:
            raise ValueError(f"{fig} {shot} {key}: figure arrays not reproduced "
                             f"(measured {err_gt:.4f}, prediction {err_pred:.4f})")
        sl = slice(k0, 100)
        mu, sd, times = s.mu[sl, view], s.sd[sl, view], s.times[sl]
        cam = CAM_NAME[view]
        if (shot, fig) not in done_gt:
            done_gt.add((shot, fig))
            gpath = out / fig / f"emission_structure_{cam}_{shot}_measured_codec.sav"
            write(gpath, counts(gt_fig[sl], mu, sd), times)
            rows.append(dict(file=str(gpath.relative_to(out)), figure=fig, shot=shot,
                             series="measured (after codec)", seeds="", camera=cam,
                             shown_s=shown, n_frames=len(times), t0_ms=f"{times[0]:.2f}",
                             t1_ms=f"{times[-1]:.2f}", figure_max_abs_err=f"{err_gt:.4f}",
                             token_agreement=f"{agree:.4f}"))
        ppath = out / fig / f"emission_structure_{cam}_{shot}_pred_{label}.sav"
        write(ppath, counts(pred_fig[sl], mu, sd), times)
        rows.append(dict(file=str(ppath.relative_to(out)), figure=fig, shot=shot,
                         series=f"prediction, {case}", seeds=seeds, camera=cam, shown_s=shown,
                         n_frames=len(times), t0_ms=f"{times[0]:.2f}", t1_ms=f"{times[-1]:.2f}",
                         figure_max_abs_err=f"{err_pred:.4f}", token_agreement=f"{agree:.4f}"))
        print(f"{ppath.relative_to(out)}: view {view}, figure arrays reproduced to "
              f"{err_gt:.4f} / {err_pred:.4f}, measured codes re-encoded {agree:.3f}", flush=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> {out}")


if __name__ == "__main__":
    main()
