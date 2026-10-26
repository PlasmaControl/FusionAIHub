"""Rank spectrogram channels by visible coherent modes and by how well the rollout reproduces them.

For each spectrogram diagnostic and channel of one rollout (``codes_<shot>_step<N>.npz``),
the measured and predicted images are stitched in time and reduced to their mode content:
the per-frequency temporal median is removed (stationary lines such as pickup are not modes),
each time column is robust-normalised over frequency, and only the positive excess is kept.

  * ``mode``  -- mean of the 99th-percentile excess per column over the predicted window
                 (how clearly lines stand out in the measurement)
  * ``match`` -- correlation of the time-smoothed measured and predicted excess images
  * ``score`` -- mode * max(match, 0)

Usage::

    python analysis/demo_spectro_channels.py --run <run dir> --shot 200729 \
        --codec_tmpl '<codecs>/{m}/codec_best.pt' [--modalities ece,mhr,bes,co2,mirnov] [--top 15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter


def _excess(img):
    """Positive excess of narrow, non-stationary features, in robust sigmas."""
    x = np.nan_to_num(img.astype(np.float32))
    x = x - np.median(x, axis=1, keepdims=True)
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True) + 1e-6
    return np.clip((x - med) / (1.4826 * mad), 0, None)


def main():
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev
    from tokamak_foundation_model.ignite import train_dynamics as td

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--shot", required=True)
    p.add_argument("--step", type=int, default=20600)
    p.add_argument("--codec_tmpl", required=True)
    p.add_argument("--modalities", default="ece,mhr,bes,co2,mirnov")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    names = [m for m in a.modalities.split(",") if m]
    codecs = td.load_frozen_codecs(names, repo=Path.cwd(), tmpl=a.codec_tmpl)
    device = torch.device("cpu")
    codes = np.load(Path(a.run) / f"codes_{a.shot}_step{a.step}.npz", allow_pickle=True)
    K0, F = int(codes["K0"]), int(codes["F"])
    rows = []
    for name in names:
        codec = codecs[name][0]
        dec = {w: np.asarray(ev.decode_flat_chunked(codec, torch.from_numpy(codes[f"{w}__{name}"]).long()[:F], device))
               for w in ("gt", "pred")}
        for ch in range(dec["gt"].shape[1]):
            gt, pr = ev._stitch(dec["gt"], ch), ev._stitch(dec["pred"], ch)
            if not np.isfinite(gt).any() or np.nanstd(gt) < 1e-6:
                continue
            start = int(gt.shape[1] * K0 / F)
            eg, ep = _excess(gt)[:, start:], _excess(pr)[:, start:]
            mode = float(np.mean(np.percentile(eg, 99, axis=0)))
            sg, sp = uniform_filter(eg, (3, 15)), uniform_filter(ep, (3, 15))
            match = float(np.corrcoef(sg.ravel(), sp.ravel())[0, 1])
            rows.append((name, ch, mode, match, mode * max(match, 0.0)))
    rows.sort(key=lambda r: -r[4])
    print(f"shot {a.shot}  run {a.run}")
    print("modality ch   mode   match  score")
    for r in rows[: a.top]:
        print(f"{r[0]:8s} {r[1]:3d} {r[2]:6.2f} {r[3]:+6.2f} {r[4]:6.2f}")
    for name in names:
        best = [r for r in rows if r[0] == name][:3]
        print(f"best {name}: " + ", ".join(f"ch{r[1]} (score {r[4]:.2f}, match {r[3]:+.2f})" for r in best))


if __name__ == "__main__":
    main()
