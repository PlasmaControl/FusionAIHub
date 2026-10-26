"""Rollout scorecard of dynamics checkpoints: what the masked validation loss does not measure.

Per checkpoint, from finished evaluation folders (no new rollouts):

  * profile skill  -- median over held-out shots of the mean RMSE skill against
                      persistence of T_e, n_e, T_i, v_tor, from the archived codes
  * hallucinated / missed -- mean over ece, mhr, mirnov of ``hallucination.json``
  * NBI response   -- beams off at the switch frame: change of core T_i against the same
                      rollout under the real actuators (archived ``real__*`` codes), in keV,
                      averaged over the scored window; median over shots and the fraction of
                      shots where T_i falls by more than 5 %
  * ECH response   -- the same for core T_e with the ECH off

Usage::

    python analysis/checkpoint_scorecard.py --codec_tmpl '<codecs>/{m}/codec_best.pt' \
        --entry 'btf@16800=<val dir>,<nbi dir>,<ech dir>' --entry ... [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_controllability_fig as dm  # noqa: E402

PROFILES = ("ts_core_temp", "ts_core_density", "cer_ti", "cer_rot")
SPECTRO = ("ece", "mhr", "mirnov")


def _profile_skill(folder, codecs, conv, device):
    """Median over shots of the mean RMSE skill against persistence of the core profiles.

    Scored from the archived codes in physical units (core average of the 3 strongest
    channels), so every checkpoint is scored by the same code regardless of the
    eval_dynamics version that produced the rollout. Unrecorded (flat) signals are skipped.
    """
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev

    skills = []
    for path in sorted(glob.glob(f"{folder}/codes_*_step*.npz")):
        codes = np.load(path, allow_pickle=True)
        K0, F = int(codes["K0"]), int(codes["F"])
        per_signal = []
        for name in PROFILES:
            gt, pred = (np.nanmean(conv[name](ev.decode_flat_chunked(
                codecs[name][0], torch.from_numpy(codes[f"{w}__{name}"]).long()[:F], device)), axis=-1)
                for w in ("gt", "pred"))
            channels = dm._core(gt, K0, 3)
            with np.errstate(invalid="ignore"):
                g, p_ = np.nanmean(gt[:, channels], 1), np.nanmean(pred[:, channels], 1)
            truth = g[K0:F]
            if np.isfinite(truth).sum() < 10 or np.nanstd(truth) < 1e-6 * max(1.0, abs(np.nanmean(truth))):
                continue
            rmse = np.sqrt(np.nanmean((p_[K0:F] - truth) ** 2))
            rmse_p = np.sqrt(np.nanmean((g[K0 - 1] - truth) ** 2))
            if np.isfinite(rmse) and rmse_p > 0:
                per_signal.append(1.0 - rmse / rmse_p)
        if len(per_signal) >= 2:
            skills.append(np.mean(per_signal))
    return (float(np.median(skills)) if skills else np.nan), len(skills)


def _hallucination(folder):
    path = Path(folder) / "hallucination.json"
    if not path.exists():
        return np.nan, np.nan
    rows = [r for r in json.load(open(path)) if r["modality"] in SPECTRO]
    return (float(np.nanmean([r["hallucinated"] for r in rows])),
            float(np.nanmean([r["missed"] for r in rows])))


def _response(folder, name, codecs, conv, switch, window, device):
    import torch

    from tokamak_foundation_model.ignite import eval_dynamics as ev

    deltas, relative = [], []
    for path in sorted(glob.glob(f"{folder}/codes_*_step*.npz")):
        codes = np.load(path, allow_pickle=True)
        if f"real__{name}" not in codes.files:
            continue
        K0, F = int(codes["K0"]), int(codes["F"])
        decoded = {
            w: np.nanmean(conv[name](ev.decode_flat_chunked(
                codecs[name][0], torch.from_numpy(codes[f"{w}__{name}"]).long()[:F], device)), axis=-1)
            for w in ("gt", "pred", "real")
        }
        channels = dm._core(decoded["gt"], K0, 3)
        scored = slice(switch + 3, min(F, switch + window))
        with np.errstate(invalid="ignore"):
            scen = np.nanmean(decoded["pred"][scored][:, channels])
            real = np.nanmean(decoded["real"][scored][:, channels])
        if np.isfinite(scen) and np.isfinite(real) and real > 0:
            deltas.append((scen - real) * 1e-3)
            relative.append((scen - real) / real)
    if not deltas:
        return np.nan, np.nan, 0
    return float(np.median(deltas)), float(np.mean(np.asarray(relative) < -0.05)), len(deltas)


def main():
    import torch

    from tokamak_foundation_model.ignite import train_dynamics as td

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--codec_tmpl", required=True)
    p.add_argument("--entry", action="append", required=True, help="label=val_dir,nbi_dir,ech_dir")
    p.add_argument("--switch_frame", type=int, default=50)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--json", default="")
    a = p.parse_args()
    codecs = td.load_frozen_codecs(list(PROFILES), repo=Path.cwd(), tmpl=a.codec_tmpl)
    conv = dm._physical(codecs)
    device = torch.device("cpu")
    rows = []
    for item in a.entry:
        label, _, dirs = item.partition("=")
        val_dir, nbi_dir, ech_dir = (dirs.split(",") + ["", "", ""])[:3]
        skill, n_val = _profile_skill(val_dir, codecs, conv, device)
        hal, miss = _hallucination(val_dir)
        nbi = _response(nbi_dir, "cer_ti", codecs, conv, a.switch_frame, a.window, device) if nbi_dir else (np.nan, np.nan, 0)
        ech = _response(ech_dir, "ts_core_temp", codecs, conv, a.switch_frame, a.window, device) if ech_dir else (np.nan, np.nan, 0)
        rows.append({"label": label, "profile_skill": skill, "n_val": n_val, "hallucinated": hal, "missed": miss,
                     "nbi_dTi_keV": nbi[0], "nbi_frac_down": nbi[1], "n_nbi": nbi[2],
                     "ech_dTe_keV": ech[0], "ech_frac_down": ech[1], "n_ech": ech[2]})
    print(f"{'checkpoint':18s} {'skill':>7s} {'halluc':>7s} {'missed':>7s} | {'NBI off: dTi':>13s} {'Ti down':>8s} | "
          f"{'ECH off: dTe':>13s} {'Te down':>8s}")
    for r in rows:
        print(f"{r['label']:18s} {r['profile_skill']:+7.3f} {r['hallucinated']:7.3f} {r['missed']:7.3f} | "
              f"{r['nbi_dTi_keV']:+9.2f} keV {r['nbi_frac_down']:7.0%} ({r['n_nbi']:2d}) | "
              f"{r['ech_dTe_keV']:+9.2f} keV {r['ech_frac_down']:7.0%} ({r['n_ech']:2d})")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
