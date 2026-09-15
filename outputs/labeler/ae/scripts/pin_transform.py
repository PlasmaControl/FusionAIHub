"""Pin `labeler.ae.transform` against tokeye's own transform, on real data.

`labeler.ae.transform.compute_stft` is a port of
`tokeye.transforms.compute_stft`, and labelmaker must not import tokeye at
runtime (it lives in a read-only venv with its own torch). This script is the
evidence that the port agrees: it reads one real corpus CO2 record, runs both
implementations over the same samples, and prints the max absolute difference
per stage.

Run it from the tokeye venv, which is the only interpreter that can import
tokeye - the same recipe task 7a used:

    cd /scratch/gpfs/nc1514/FusionAIHub && \
    PYTHONPATH=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/pylibs:$PWD/src \
    /scratch/gpfs/nc1514/tokeye/.venv/bin/python \
    outputs/labelmaker/ae/scripts/pin_transform.py --shot 198279 \
    --out outputs/labelmaker/ae/transform_pin.json

Note the two interpreters do NOT share a scipy: the pixi env that runs
inference has one version and the tokeye venv another, so this measures the
port AND that version gap together, which is the number that actually matters
for a label.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import scipy
from tokeye.transforms import compute_stft as tokeye_compute_stft

from labeler.ae import transform as tr

CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", type=int, default=198279)
    ap.add_argument("--t-lo", type=float, default=0.0)
    ap.add_argument("--t-hi", type=float, default=6.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    path = CORPUS / f"{args.shot}_processed.h5"
    with h5py.File(path, "r") as f:
        x = np.asarray(f["co2"]["xdata"], dtype=np.float64)
        keep = np.flatnonzero((x >= args.t_lo) & (x < args.t_hi))
        lo, hi = int(keep[0]), int(keep[-1]) + 1
        y = np.asarray(f["co2"]["ydata"][:, lo:hi], dtype=np.float64)
        x = x[lo:hi]

    ours = np.stack([tr.compute_stft(row) for row in y], axis=0)
    theirs = np.stack(
        [tokeye_compute_stft(row[None, :], n_fft=tr.N_FFT, hop=tr.HOP).astype(np.float32)
         for row in y],
        axis=0,
    )
    stft_max = float(np.abs(ours - theirs).max())
    norm_max = float(np.abs(tr.standardise(ours) - tr.standardise(theirs)).max())
    band_max = float(
        np.abs(tr.restrict_to_band(tr.standardise(ours))
               - tr.restrict_to_band(tr.standardise(theirs))).max()
    )

    report = {
        "shot": args.shot,
        "corpus_file": str(path),
        "window_s": [args.t_lo, args.t_hi],
        "n_samples": int(y.shape[1]),
        # From the span, not a median diff: xdata is float32, and at t ~ 3 s
        # its spacing (2.4e-7) quantises a 2 us step into 1.9e-6, which
        # reads as 524,288 Hz instead of 500,000.
        "fs_hz": float((y.shape[1] - 1) / (x[-1] - x[0])),
        "spectrogram_shape": list(ours.shape),
        "value_range_stft": [float(theirs.min()), float(theirs.max())],
        "max_abs_diff_stft": stft_max,
        "max_abs_diff_standardised": norm_max,
        "max_abs_diff_band": band_max,
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
