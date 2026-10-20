"""Hallucinated versus missed activity of a rollout, per spectrogram modality.

The code-space scores of ``rollout_npz_score.py`` say how often a token is right; they
do not say whether a wrong prediction *invents* activity (a mode, a broadband burst)
that the shot never had, or *misses* activity it did have. Both are measured here on
the decoded spectrograms of a finished evaluation (its ``codes_<shot>_step<N>.npz``):

    base  = per-bin median of the ground truth over the whole window (its quiet level)
    E_x   = relu(x - base) over the predicted frames [K0, F)
    hallucinated = sum relu(E_pred - E_gt) / sum E_pred
    missed       = sum relu(E_gt - E_pred) / sum E_gt

so 0 is perfect and 1 means no overlap at all. Persistence (the last seed frame held)
is scored the same way as the reference a forecast has to beat.

Usage::

    python analysis/rollout_hallucination.py --split 190729,190735,190736 \
        --cache_dir <frame_codes> --codec_tmpl '<codecs>/{m}/codec_best.pt' <eval dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from tokamak_foundation_model.ignite import eval_dynamics as ev
from tokamak_foundation_model.ignite import train_dynamics as td


SPECTRO = ("ece", "bes", "mhr", "co2", "mirnov")


def _excess(x, base):
    return np.clip(np.nan_to_num(x - base, nan=0.0), 0.0, None)


def _rates(gt, pred, K0):
    """Hallucinated and missed fractions of one prediction against its truth."""
    base = np.nanmedian(gt, axis=0, keepdims=True)
    e_gt, e_pred = _excess(gt[K0:], base), _excess(pred[K0:], base)
    eps = 1e-12
    hallucinated = float(np.clip(e_pred - e_gt, 0, None).sum() / (e_pred.sum() + eps))
    missed = float(np.clip(e_gt - e_pred, 0, None).sum() / (e_gt.sum() + eps))
    return hallucinated, missed


def score_dir(eval_dir, codecs, device, pinned):
    """Per-shot, per-modality rates of every archive in ``eval_dir``."""
    rows = []
    for archive in sorted(Path(eval_dir).glob("codes_*_step*.npz")):
        codes = np.load(archive, allow_pickle=True)
        shot, K0, F = str(codes["shot"]), int(codes["K0"]), int(codes["F"])
        gt = {k[4:]: torch.from_numpy(codes[k]).long() for k in codes.files if k.startswith("gt__")}
        pred = {k[6:]: torch.from_numpy(codes[k]).long() for k in codes.files if k.startswith("pred__")}
        decoded = ev.decode_all(codecs, gt, pred, K0, F, device, raw_gt=None)
        for name, entry in decoded.items():
            if entry.get("family") != "spectro" or ev._is_static(entry["gt"]):
                continue
            persistence = np.repeat(entry["gt"][K0 - 1 : K0], F, axis=0)
            h, m = _rates(entry["gt"], entry["pred"], K0)
            hp, mp = _rates(entry["gt"], persistence, K0)
            rows.append(dict(shot=shot, modality=name, pinned=shot in pinned,
                             hallucinated=h, missed=m, hallucinated_pers=hp, missed_pers=mp))
    return rows


def _summary(rows, label):
    if not rows:
        return f"  {label:10s} (no scored pairs)"
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    return (f"  {label:10s} hallucinated {mean('hallucinated'):.3f} (pers "
            f"{mean('hallucinated_pers'):.3f})   missed {mean('missed'):.3f} (pers "
            f"{mean('missed_pers'):.3f})   [{len(rows)} pairs]")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dirs", nargs="+", help="finished evaluation folders")
    parser.add_argument("--split", default="", help="comma-separated pinned (train) shots")
    parser.add_argument("--cache_dir", required=True, help="frame-code cache of the evaluation")
    parser.add_argument("--codec_tmpl", default=None, help="codec checkpoint template")
    args = parser.parse_args()
    pinned = {s for s in args.split.split(",") if s}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, manifest_error = ev._read_manifest(args.cache_dir)
    tmpl = ev._codec_template(args.codec_tmpl, manifest, manifest_error, print)
    # Only the spectrogram codecs: decoding the video and profile modalities is most of
    # the cost and none of them is scored here.
    names = [m for m in ev.EVAL_MODALITIES if m in SPECTRO]
    codecs = td.load_frozen_codecs(names, repo=Path.cwd(), tmpl=tmpl)
    for codec, _cfg, _family in codecs.values():
        codec.to(device)
    for d in args.dirs:
        rows = score_dir(d, codecs, device, pinned)
        Path(d, "hallucination.json").write_text(json.dumps(rows, indent=1))
        print(f"=== {Path(d).parent.name}/{Path(d).name}   (lower is better; 0 = perfect)")
        for name in sorted({r["modality"] for r in rows}):
            print(_summary([r for r in rows if r["modality"] == name], name))
        print(_summary([r for r in rows if r["pinned"]], "PINNED"))
        print(_summary([r for r in rows if not r["pinned"]], "HELD-OUT"))


if __name__ == "__main__":
    main()
