"""Rank finished demo scenario runs by baseline match and physical response.

For every ``<prefix>_<shot>_<scenario>_g<w>`` folder set under the run directory (from
``logs/_demoG_*.sh`` / ``_demoE_*.sh``), prints per shot and guidance: the baseline
match to the measurement (mean |error| after the prediction start, physical units),
and the response after the switch -- T_i and |v_tor| under NBI off, T_e under ECH off,
|v_tor| under no torque -- relative to the real-actuator prediction.

Usage::

    python analysis/demo_response_table.py --runs <run dir> --prefix _demoG --switch_frame 50 \
        --codec_tmpl '<codecs>/{m}/codec_best.pt'
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_controllability_fig as dm  # noqa: E402

from tokamak_foundation_model.ignite import train_dynamics as td  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", required=True)
    p.add_argument("--prefix", default="_demoG")
    p.add_argument("--switch_frame", type=int, default=50)
    p.add_argument("--codec_tmpl", required=True)
    p.add_argument("--step", type=int, default=20600)
    a = p.parse_args()
    codecs = td.load_frozen_codecs(list(dm.SLOW), repo=Path.cwd(), tmpl=a.codec_tmpl)
    conv = dm._physical(codecs)
    device = torch.device("cpu")
    pattern = re.compile(rf"{a.prefix}_(\d+)_(real|nbi_off|ech_off|torque0)_g([\d.]+)$")
    found = {}
    for d in glob.glob(f"{a.runs}/{a.prefix}_*"):
        m = pattern.search(d)
        if m and glob.glob(f"{d}/codes_*_step{a.step}.npz"):
            found.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = d
    print("shot   g  | baseline |err|: Te keV  Ti keV  vtor km/s | NBI off: dTi keV  d|vtor| | ECH off: dTe keV | torque0: d|vtor|")
    for (shot, g), runs in sorted(found.items()):
        if "real" not in runs:
            continue
        data = {}
        for key, d in runs.items():
            dec, k0, F = dm._decode(codecs, Path(d) / f"codes_{shot}_step{a.step}.npz", list(dm.SLOW), device)
            data[key] = {n: (conv[n](dec[n][0]), conv[n](dec[n][1])) for n in dm.SLOW}
        gt = {n: data["real"][n][0] for n in dm.SLOW}
        ch = {n: dm._core_channel(gt[n], k0) for n in dm.SLOW}
        tr = lambda key, n, i=1: dm._trace(data[key][n][i], ch[n])  # noqa: E731
        after = slice(a.switch_frame + 2, min(F, a.switch_frame + 30))
        err = lambda n, s: np.nanmean(np.abs(tr("real", n) - tr("real", n, 0))[k0:]) * s  # noqa: E731
        diff = lambda key, n, s=1.0, absval=False: (  # noqa: E731
            np.nanmean((np.abs(tr(key, n)) if absval else tr(key, n))[after])
            - np.nanmean((np.abs(tr("real", n)) if absval else tr("real", n))[after])
        ) * s if key in data else np.nan
        print(f"{shot} {g:>3} | {err('ts_core_temp', 1e-3):6.2f} {err('cer_ti', 1e-3):7.2f} {err('cer_rot', 1):9.1f} | "
              f"{diff('nbi_off', 'cer_ti', 1e-3):+8.2f} {diff('nbi_off', 'cer_rot', 1, True):+8.1f} | "
              f"{diff('ech_off', 'ts_core_temp', 1e-3):+8.2f} | {diff('torque0', 'cer_rot', 1, True):+8.1f}")


if __name__ == "__main__":
    main()
