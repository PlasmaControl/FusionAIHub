"""How predictable are a frame cache's Mirnov codes from the previous frame? Truth only, no model.

    scripts/mast/py.sh scripts/mast/mirnov_code_persistence.py \
        /models/intermediate/mast/stage2r/frame_codes_v1b_binspec_cam_live_da \
        --tokeye /models/intermediate/mast/stage2r/tokeye/scores_v2/frames.parquet

For each Mirnov array over ``--n-shots`` val shots (seed 0): the mean |code_t - code_{t-1}| over positions
recorded in both frames ("step") against the same with a random frame of the shot ("shuffle"). A
step/shuffle ratio near 1 means the previous frame says nothing about the codes; near 0, it predicts them.
With ``--tokeye``, "active" pairs are those whose TokEye frame mode share is >= 0.05 in both frames. Also
prints the mean per-position entropy of the 64 levels. The missing code (64) is excluded.

Measured 2026-11-25 on 200 val shots: band-DCT (frame_codes_v1b_logspec_cam_live_da) OMV 0.81 active /
0.89 all, CC 0.81 / 0.91; per-bin BinSpec (frame_codes_v1b_binspec_cam_live_da) OMV 0.34 / 0.50,
CC 0.36 / 0.36 (docs/models/ignite-mast-spectrogram-quality.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SIGS = {"magnetics-b_field_pol_probe_omv_voltage": "omv", "magnetics-b_field_tor_probe_cc_field": "cc"}
MISSING = 64


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framecache", type=Path)
    ap.add_argument("--n-shots", type=int, default=200)
    ap.add_argument("--tokeye", default="", help="tokeye_spec_quality.py frames.parquet")
    a = ap.parse_args()
    shots = json.loads((a.framecache / "splits.json").read_text())["val"]
    rng = np.random.default_rng(0)
    shots = sorted(rng.choice(shots, size=min(a.n_shots, len(shots)), replace=False).tolist(), key=int)
    mode = {}
    if a.tokeye:
        fr = pd.read_parquet(a.tokeye)
        fr = fr[fr["shot"].isin([str(x) for x in shots])]
        mode = {(sh, sg): dict(zip(g["frame"].astype(int), g["mode"].astype(float)))
                for (sh, sg), g in fr.groupby(["shot", "sig"])}
    acc: dict = {}
    codes: dict = {s: [] for s in SIGS}
    for shot in shots:
        d = torch.load(a.framecache / f"{shot}.pt", map_location="cpu", weights_only=False)
        for s, short in SIGS.items():
            c = d["codes"].get(s)
            if c is None:
                continue
            c = c.numpy().astype(np.float64)
            c[c >= MISSING] = np.nan
            n = c.shape[0]
            if n < 3 or np.isnan(c).all():
                continue
            codes[s].append(c)
            md = mode.get((str(shot), short), {})
            m = np.array([md.get(f, np.nan) for f in range(n)])
            perm = rng.permutation(n)
            step = np.abs(c[1:] - c[:-1])
            shuf = np.abs(c[1:] - c[perm][1:])
            with np.errstate(invalid="ignore"):
                active = (m[1:] >= 0.05) & (m[:-1] >= 0.05)
            for name, sel in (("all", np.ones(n - 1, bool)), ("active", active)):
                r = acc.setdefault((s, name), [0.0, 0, 0.0, 0])
                r[0] += float(np.nansum(step[sel]))
                r[1] += int(np.isfinite(step[sel]).sum())
                r[2] += float(np.nansum(shuf[sel]))
                r[3] += int(np.isfinite(shuf[sel]).sum())
    for (s, name), r in sorted(acc.items()):
        st, sh = r[0] / max(r[1], 1), r[2] / max(r[3], 1)
        print(f"{SIGS[s]:4s} {name:6s} step {st:5.2f} shuffle {sh:5.2f} "
              f"step/shuffle {st / max(sh, 1e-9):4.2f} n {r[1]}")
    for s, arrays in codes.items():
        if not arrays:
            continue
        allc = np.concatenate(arrays, 0)
        ent = []
        for j in range(allc.shape[1]):
            v = allc[:, j]
            v = v[np.isfinite(v)].astype(int)
            if v.size:
                p = np.bincount(v, minlength=MISSING) / v.size
                p = p[p > 0]
                ent.append(float(-(p * np.log2(p)).sum()))
        print(f"{SIGS[s]:4s} positions {allc.shape[1]} mean entropy {np.mean(ent):.2f} bits "
              f"(max 6), frames {allc.shape[0]}")


if __name__ == "__main__":
    main()
