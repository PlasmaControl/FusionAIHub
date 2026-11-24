"""Fit the peak + baseline Mirnov spectrogram codec (`port.peakspec`) on TokEye-kept training shots.

    scripts/mast/py.sh scripts/mast/port_fit_peakspec.py \
        --frames-dir /models/intermediate/mast/stage2r/frame_codes_v1b_logspec_cam_live_da \
        --keep-dir /models/intermediate/mast/stage2r/tokeye/scores_v2 \
        --out /models/intermediate/mast/stage2r/peakspec_fit_v1

Writes ``logspec_fit.{npz,json}`` (``"codec": "peakspec"``) into ``--out``; pass that directory to
``port_build_cache.py --logspec-fit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokamak_foundation_model.ignite.mast import paths
from tokamak_foundation_model.ignite.mast.port import logspec, peakspec


def main() -> None:
    p = argparse.ArgumentParser("Fit the PeakSpec Mirnov codec")
    p.add_argument("--cache-root", default=str(paths.cache_dir("v1")))
    p.add_argument("--frames-dir", required=True, help="any port frame-code cache (plasma frames)")
    p.add_argument("--logspec-fit", default=str(logspec.DEFAULT_LOGSPEC_OUT),
                   help="band-DCT logspec fit whose floor and log mean are reused")
    p.add_argument("--keep-dir", required=True, help="TokEye keep map (signals/probes.parquet)")
    p.add_argument("--out", required=True)
    p.add_argument("--n-shots", type=int, default=300)
    p.add_argument("--rich-share", type=float, default=0.5)
    p.add_argument("--m", type=int, default=peakspec.PEAKSPEC_M)
    p.add_argument("--n-bands", type=int, default=peakspec.PEAKSPEC_N_BANDS)
    p.add_argument("--threshold", type=float, default=peakspec.PEAKSPEC_THRESHOLD)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    meta = peakspec.fit_peakspec(
        Path(a.cache_root), Path(a.frames_dir), Path(a.logspec_fit), Path(a.keep_dir), Path(a.out),
        n_shots=a.n_shots, rich_share=a.rich_share, m=a.m, n_bands=a.n_bands,
        threshold=a.threshold, workers=a.workers, seed=a.seed,
    )
    short = {k: v for k, v in meta.items() if k != "signals"}
    short["signals"] = {
        n: {k: v for k, v in s.items() if k != "shots"} for n, s in meta["signals"].items()
    }
    print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
