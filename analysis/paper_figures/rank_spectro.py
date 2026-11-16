"""Rank every spectrogram channel of the paper rollouts, for replay and for scenario response.

For one shot, decodes the measurement and every seed of each named run on all channels of
the five spectrogram modalities, reduces each stitched image to its mode content (the
excess of ``analysis/demo_spectro_channels.py``) and scores each channel:

  * ``mode``   -- how clearly lines stand out in the measurement (mean 99th-percentile
                  excess per column over the predicted window)
  * ``match``  -- correlation of the time-smoothed measured and predicted excess (seed mean)
  * ``replay`` -- mode * max(match, 0)
  * ``resp:<run>`` -- 1 - correlation of the seed-mean excess of <run> and of the
                  measured-actuator run after the switch frame, times the run's own mode
                  strength (how much the actuator change alters the lines)

Writes ``<out>.json`` (all scores) and ``<out>.npz`` with the stitched measurement, and the
medoid seed of every run, for the top channels of each score (keys
``gt__<m>__<ch>``, ``pred__<run>__<m>__<ch>``, ``fmax__<m>``).

    python rank_spectro.py --shot 204011 --runs real,tq_lin_zero,tq_lin_half,ech_lin_off,ech_lin_half \
        --switch 50 --out /scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data/rank_204011
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
import torch

from tokamak_foundation_model.ignite import eval_dynamics as ev
from tokamak_foundation_model.ignite import train_dynamics as td

RUNS = Path(os.environ.get("PAPER_FIG_RUN", "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_prod_v4/runs/soup_lu18_h16_30"))
TMPL = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/models/ignite_codecs_v4/{m}/codec_best.pt"
STEP = int(os.environ.get("PAPER_FIG_STEP", 30000))


def excess(img):
    x = np.nan_to_num(img.astype(np.float32))
    x = x - np.median(x, axis=1, keepdims=True)
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True) + 1e-6
    return np.clip((x - med) / (1.4826 * mad), 0, None)


def corr(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", required=True)
    p.add_argument("--runs", default="real")
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--switch", type=int, default=0, help="first frame of a scenario's change")
    p.add_argument("--modalities", default="ece,mhr,bes,co2,mirnov")
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    runs = a.runs.split(",")
    seeds = a.seeds.split(",")
    names = a.modalities.split(",")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codecs = td.load_frozen_codecs(names, tmpl=TMPL)
    arch = {r: [np.load(RUNS / f"_demo_{a.shot}_{r}_s{STEP}_g3_seed{s}" / f"codes_{a.shot}_step{STEP}.npz",
                        allow_pickle=True) for s in seeds] for r in runs}
    first = arch[runs[0]][0]
    K0, F = int(first["K0"]), int(first["F"])
    scores, images = {}, {}
    for name in names:
        codec = codecs[name][0].to(dev).eval()
        with torch.no_grad():
            gt = np.asarray(ev.decode_flat_chunked(codec, ev.archived(first, f"gt__{name}")[:F], dev))
            pred = {r: [np.asarray(ev.decode_flat_chunked(codec, ev.archived(z, f"pred__{name}")[:F], dev))
                        for z in arch[r]] for r in runs}
        images[f"fmax__{name}"] = gt.shape[2] * (500_000.0 / 1024) / 1e3
        rows = []
        for ch in range(gt.shape[1]):
            g = ev._stitch(gt, ch)
            if not np.isfinite(g).any() or np.nanstd(g) < 1e-6:
                continue
            cols = g.shape[1]
            k0, ks = int(cols * K0 / F), int(cols * max(a.switch, K0) / F)
            eg = excess(g)
            row = {"ch": ch, "mode": float(np.mean(np.percentile(eg[:, k0:], 99, axis=0)))}
            sm = {}
            for r in runs:
                ex = [uniform_filter(excess(ev._stitch(x, ch)), (3, 15)) for x in pred[r]]
                sm[r] = np.mean(ex, axis=0)
                mean = sm[r][:, k0:]
                row[f"medoid:{r}"] = int(np.argmin([np.mean((e[:, k0:] - mean) ** 2) for e in ex]))
                row[f"mode:{r}"] = float(np.mean(np.percentile(sm[r][:, k0:], 99, axis=0)))
            sg = uniform_filter(eg, (3, 15))
            row["match"] = float(np.mean([corr(sg[:, k0:], uniform_filter(excess(ev._stitch(x, ch)), (3, 15))[:, k0:])
                                          for x in pred[runs[0]]]))
            row["replay"] = row["mode"] * max(row["match"], 0.0)
            for r in runs[1:]:
                row[f"resp:{r}"] = (1.0 - corr(sm[r][:, ks:], sm[runs[0]][:, ks:])) * row["replay"]
            rows.append(row)
        scores[name] = rows
        keep = set()
        for key in ["replay"] + [f"resp:{r}" for r in runs[1:]]:
            keep |= {r["ch"] for r in sorted(rows, key=lambda r: -r[key])[: a.top]}
        by_ch = {r["ch"]: r for r in rows}
        for ch in sorted(keep):
            images[f"gt__{name}__{ch}"] = ev._stitch(gt, ch).astype(np.float16)
            m = by_ch[ch][f"medoid:{runs[0]}"]  # the same seed for every run: they match before the switch
            for r in runs:
                images[f"pred__{r}__{name}__{ch}"] = ev._stitch(pred[r][m], ch).astype(np.float16)
                images[f"mean__{r}__{name}__{ch}"] = ev._stitch(np.mean(pred[r], axis=0), ch).astype(np.float16)
        for key in ["replay"] + [f"resp:{r}" for r in runs[1:]]:
            top = sorted(rows, key=lambda r: -r[key])[: a.top]
            print(f"{a.shot} {name:6s} {key:18s} " + "  ".join(f"ch{r['ch']}:{r[key]:.2f}(m{r['match']:+.2f})" for r in top),
                  flush=True)
        del gt, pred
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"shot": a.shot, "runs": runs, "k0": K0, "F": F,
                                                    "switch": a.switch, "scores": scores}, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), **images)
    print("wrote", out.with_suffix(".npz"))


if __name__ == "__main__":
    main()
