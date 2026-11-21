"""Held-out shots for the actuator-response test of a checkpoint.

A shot qualifies for the NBI test when the injected beam power stays high from the switch
frame over the scored window (so switching it off is a large, clean step) and CER ion
temperature is recorded; for the ECH test the same with ECH power and Thomson T_e. Shots
come from the trainer's validation split only, pinned shots excluded.

Usage::

    python analysis/select_response_shots.py --split <run>/split_shots.json \
        --act_global <act_global> --cache_dir <frame_codes> --switch_frame 50 --n 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Columns of the actuator cache (train_dynamics._ACT_SPEC order).
_ECH, _PINJ = slice(0, 12), slice(12, 20)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", required=True)
    p.add_argument("--act_global", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--switch_frame", type=int, default=50)
    p.add_argument("--window", type=int, default=30, help="scored frames after the switch")
    p.add_argument("--nbi_min_mw", type=float, default=2.5)
    p.add_argument("--ech_min_mw", type=float, default=1.0)
    p.add_argument("--n", type=int, default=16)
    a = p.parse_args()
    split = json.load(open(a.split))
    pinned = {str(s) for k in ("pin_val", "pin_train", "pin_both") for s in split.get(k, [])}
    shots = sorted(str(s) for s in split["val"] if str(s) not in pinned)
    lo, hi = a.switch_frame, a.switch_frame + a.window
    found = {"nbi": [], "ech": []}
    for shot in shots:
        act_path, code_path = Path(a.act_global) / f"{shot}.pt", Path(a.cache_dir) / f"{shot}.pt"
        if not act_path.exists() or not code_path.exists():
            continue
        raw = np.asarray(torch.load(act_path, map_location="cpu", weights_only=False)["actuators_raw"])
        codes = torch.load(code_path, map_location="cpu", weights_only=False)["codes"]
        if raw.shape[0] < hi + 2:
            continue
        nbi, ech = raw[lo:hi, _PINJ].sum(1) / 1e6, raw[lo:hi, _ECH].sum(1) / 1e6

        def recorded(name):
            c = np.asarray(codes[name])[lo - 20:hi]
            return len(np.unique(c.reshape(len(c), -1), axis=0)) > 0.5 * len(c)

        if np.nanmin(nbi) >= a.nbi_min_mw and recorded("cer_ti") and recorded("cer_rot"):
            found["nbi"].append((float(np.nanmean(nbi)), shot))
        if np.nanmin(ech) >= a.ech_min_mw and recorded("ts_core_temp"):
            found["ech"].append((float(np.nanmean(ech)), shot))
    for test, rows in found.items():
        rows.sort(reverse=True)
        pick = [s for _, s in rows[: a.n]]
        print(f"{test}: {len(rows)} qualify; picked {len(pick)} (mean power "
              f"{np.mean([p for p, _ in rows[: a.n]]):.1f} MW)")
        print(f"{test.upper()}_SHOTS={','.join(pick)}")


if __name__ == "__main__":
    main()
