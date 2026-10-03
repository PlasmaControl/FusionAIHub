#!/usr/bin/env python
"""Calibrate n=2 harmonic rejection using development shots alone.

The reference consists of strong n=1 runs with a continuous 50 ms crossing
of 12 G and a rotating line at 0 < f1 <= 30 kHz. A bin has n=1-only evidence
when the measured n=2 line agrees with the temporal second harmonic,
|f2 - 2*f1| <= max(1 kHz, 0.1*2*f1). Independent n=2 lines are excluded.
The upper 99th percentile of N2RMS/N1RMS defines the rejection ratio.
Quiet-time percentiles use development bins labelled absent by the original
rule, excluding any 12/6 G seed. These are a preliminary activity floor,
not independent physical validation of absence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from labeler.tearing.magfeatures import FEATURE_NAMES

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
OUT = ROOT / "round4/tm"


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)


def continuous(mask, t_ms, minimum_ms):
    """Keep qualifying runs without joining gaps, including acquisition gaps."""
    result = np.zeros(len(mask), dtype=bool)
    dt = float(np.median(np.diff(t_ms)))
    barriers = np.r_[False, np.diff(t_ms) > dt * 1.5]
    for a, b in runs(mask):
        starts = np.r_[a, np.flatnonzero(barriers[a:b]) + a]
        starts = np.unique(starts)
        for lo, hi in zip(starts, np.r_[starts[1:], b], strict=True):
            if t_ms[hi - 1] - t_ms[lo] + dt >= minimum_ms:
                result[lo:hi] = True
    return result


def quantiles(values):
    return (
        {
            str(q): float(np.quantile(values, q))
            for q in (0.5, 0.9, 0.95, 0.975, 0.99, 0.995)
        }
        if len(values)
        else {}
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--cohort", type=Path, default=REPO / "data/events/catalog/cohort.csv"
    )
    parser.add_argument("--signals-dir", type=Path, default=OUT / "signals")
    parser.add_argument("--freq-dir", type=Path, default=OUT / "signals_freq")
    parser.add_argument("--mag-dir", type=Path, default=OUT / "magfeatures")
    parser.add_argument("--old-labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--frozen-calibration",
        type=Path,
        help="replay only the reference and quiet shot lists from this JSON",
    )
    args = parser.parse_args(argv)
    cohort = pd.read_csv(args.cohort)
    development = cohort[cohort.split != "test"]
    labels = pd.read_csv(args.old_labels)
    frozen = (
        json.loads(args.frozen_calibration.read_text())
        if args.frozen_calibration is not None
        else None
    )
    reference_allowed = (
        {int(r["shot"]) for r in frozen["reference_shots"]} if frozen else None
    )
    quiet_allowed = (
        set(
            frozen.get(
                "quiet_shots", set(development.shot) - set(frozen["missing_rms_shots"])
            )
        )
        if frozen
        else None
    )
    magnetic_quiet_allowed = (
        set(frozen.get("magnetic_quiet_shots", quiet_allowed)) if frozen else None
    )
    if frozen and (
        (reference_allowed | quiet_allowed | magnetic_quiet_allowed)
        - set(development.shot)
    ):
        raise ValueError("frozen calibration contains a non-development shot")
    ratios, quiet = [], {1: [], 2: []}
    magnetic_quiet = {1: [], 2: []}
    reviewed_weak = {}
    sources, missing = [], []
    quiet_shots, magnetic_quiet_shots = [], []
    for row in development.itertuples(index=False):
        shot = int(row.shot)
        rms_path = args.signals_dir / f"{shot}.npz"
        freq_path = args.freq_dir / f"{shot}.npz"
        if not rms_path.is_file():
            missing.append(shot)
            continue
        with np.load(rms_path) as data:
            t = data["t_ms"]
            dt = float(np.median(np.diff(t)))
            width = max(1, round(5 / dt)) | 1
            n1 = median_filter(data["n1rms"], width)
            n2 = median_filter(data["n2rms"], width)
        window = (t >= row.window_start_ms) & (t < row.window_end_ms)
        finite = np.isfinite(n1) & np.isfinite(n2)
        absent = np.zeros(len(t), dtype=bool)
        for span in labels[(labels.shot == shot) & (labels.category == 0)].itertuples():
            absent |= (t >= span.t_start) & (t < span.t_end)
        low = window & finite & absent & (n1 < 12) & (n2 < 6)
        use_quiet = quiet_allowed is None or shot in quiet_allowed
        if use_quiet:
            quiet_shots.append(shot)
            quiet[1].append(n1[low])
            quiet[2].append(n2[low])
        mag_path = args.mag_dir / f"{shot}.npz"
        use_magnetic_quiet = (
            magnetic_quiet_allowed is None or shot in magnetic_quiet_allowed
        )
        if mag_path.is_file() and use_quiet and use_magnetic_quiet:
            magnetic_quiet_shots.append(shot)
            with np.load(mag_path) as data:
                mt, features = data["centres_ms"], data["features"]
            baseline = np.interp(mt, t, low.astype(float)) >= 0.999
            # Quiet measurement background is stricter than the old absent class,
            # which contained weak coherent modes. Both RMS traces must be <=0.2 G.
            baseline &= np.interp(mt, t, n1) <= 0.2
            baseline &= np.interp(mt, t, n2) <= 0.2
            summary = {}
            for n in (1, 2):
                indices = [
                    i
                    for i, name in enumerate(FEATURE_NAMES)
                    if name.startswith(f"a{n}_")
                ]
                amp = np.max(features[:, indices], axis=1)
                magnetic_quiet[n].append(amp[baseline & np.isfinite(amp)])
                fit = features[:, FEATURE_NAMES.index(f"fit{n}")]
                prominence = features[:, FEATURE_NAMES.index("line_prominence_db")]
                weak = (mt >= 2000) & (mt < 5000)
                coherent = weak & (fit >= 0.9) & (prominence >= 10)
                summary[str(n)] = {
                    "log10_coherent_amplitude_quantiles_2_to_5s": quantiles(amp[weak]),
                    "phase_fit_quantiles_2_to_5s": quantiles(fit[weak]),
                    "n_bins_2_to_5s": int(weak.sum()),
                    "n_coherent_bins_fit09_prominence10db": int(coherent.sum()),
                    "max_continuous_coherent_ms": max(
                        (float(mt[b - 1] - mt[a] + 10) for a, b in runs(coherent)),
                        default=0,
                    ),
                }
            if shot in (196494, 187072):
                reviewed_weak[str(shot)] = summary
        if not freq_path.is_file() or (
            reference_allowed is not None and shot not in reference_allowed
        ):
            continue
        with np.load(freq_path) as data:
            f1 = np.interp(t, data["t_ms"], data["n1freq"], left=np.nan, right=np.nan)
            f2 = np.interp(t, data["t_ms"], data["n2freq"], left=np.nan, right=np.nan)
        strong = continuous(window & finite & (n1 >= 12), t, 50)
        coherent = (f1 > 0) & (f1 <= 30)
        harmonic = np.isfinite(f2) & (f2 > 0)
        harmonic &= np.abs(f2 - 2 * f1) <= np.maximum(1.0, 0.2 * f1)
        selected = strong & coherent & harmonic
        if not selected.any():
            continue
        values = n2[selected] / n1[selected]
        ratios.append(values)
        sources.append(
            {
                "shot": shot,
                "split": row.split,
                "n_bins": len(values),
                "seconds": float(len(values) * dt / 1000),
                "ratio_quantiles": quantiles(values),
            }
        )
    values = np.concatenate(ratios) if ratios else np.array([])
    record = {
        "made_by": "scripts/labeler/tm_calibrate_rule.py",
        "cohort": str(args.cohort),
        "old_labels": str(args.old_labels),
        "signals_dir": str(args.signals_dir),
        "freq_dir": str(args.freq_dir),
        "magfeatures_dir": str(args.mag_dir),
        "frozen_reference_calibration": str(args.frozen_calibration)
        if args.frozen_calibration is not None
        else None,
        "quiet_shots": quiet_shots,
        "magnetic_quiet_shots": magnetic_quiet_shots,
        "split": "train + validation only; every blind test shot excluded",
        "n_development_shots": len(development),
        "n_reference_shots": len(sources),
        "n_reference_bins": len(values),
        "reference_shots": sources,
        "definition": __doc__,
        "ratio_quantiles": quantiles(values),
        "selected_quantile": 0.99,
        "harmonic_ratio": float(np.quantile(values, 0.99)) if len(values) else None,
        "quiet_rms": {
            str(n): {
                "n_bins": sum(len(a) for a in arrays),
                "quantiles_g": quantiles(np.concatenate(arrays)),
            }
            for n, arrays in quiet.items()
            if arrays
        },
        "magnetic_background": {
            str(n): {
                "definition": "original absent bins with n1rms,n2rms <=0.2 G; "
                "maximum log10 phase-fit>=0.9 mode amplitude over 0.5-30 kHz",
                "n_bins": sum(len(a) for a in arrays),
                "quantiles_log10_amplitude": quantiles(np.concatenate(arrays)),
            }
            for n, arrays in magnetic_quiet.items()
            if arrays
        },
        "reviewed_weak_shots": reviewed_weak,
        "missing_rms_shots": missing,
        "disclosure": "The previous 0.4 ratio referred to blind test shot 187043; "
        "that reference is removed. This calibration never opens its signals.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "n_reference_shots",
                    "n_reference_bins",
                    "harmonic_ratio",
                    "quiet_rms",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
