#!/usr/bin/env python
"""Calibrate the n = 2 harmonic veto on the bins whose line at 2 f1 fits toroidal n = 2.

`tm_calibrate_rule.py` took every strong (>= 12 G for 50 ms) n = 1 bin whose n = 2 line
sits at twice the n = 1 line's frequency and set the veto at the 99th percentile of
N2RMS / N1RMS there. The second harmonic of a rotating, non-sinusoidal n = 1 waveform
has toroidal number 2 at 2 f1, so the bins that are really the harmonic are the ones
whose six midplane-probe phases at 2 f1 fit n = 2 (best-fit |n| = 2 with a fit of at
least 0.9, `COHERENT_FIT`). Those bins are the harmonic reference: the 99th percentile
of N2RMS / N1RMS over them, rounded up to two decimals, is the veto.

The bins that fit n = 1 at 2 f1 are NOT the harmonic reference (a line of toroidal
number 1 at 2 f1 is a different n = 1 line), and earlier rounds calibrated on them in
error. The record keeps every set for comparison.

Limit, stated plainly: toroidal phase cannot separate a harmonic of the n = 1 mode from
a co-rotating, frequency-coupled n = 2 mode (a 3/2 mode locked to the 2/1), because both
have toroidal number 2 at 2 f1. The veto is therefore a heuristic, and it may also
remove real 3/2 modes; `tm_rule_diagnostics.py` reports how many seeds it removes.

Only development shots from the frozen reference list of `calibration_dev_fix1.json`
are read; no blind test shot is opened.

    PYTHONPATH=$PWD/src LABELER_NO_FETCH=1 pixi run --frozen --no-install -e labelmaker \\
        python scripts/labeler/tm_harmonic_calibration.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "src", Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tm_calibrate_rule import continuous, quantiles

from labeler.config import git_sha
from labeler.events.panels._shared import finite, stft
from labeler.events.verify import NoDataError, corpus_signal
from labeler.tearing import magfeatures

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
OUT = ROOT / "round4/tm"
CATALOG = REPO / "data/events/catalog"
SOURCES = REPO / "data/events/neoclassical_tearing_mode/benchmark/sources"
#: The set of phase-coherent harmonic bins must hold at least this many bins and shots.
MIN_BINS = 1000
MIN_SHOTS = 10
#: Margin (kHz) around 2 f1 inside which the strongest cell is taken as the harmonic.
HARMONIC_BAND_KHZ = 1.0
MARGIN_MS = 300.0


def harmonic_cells(spec, f_khz, t_cols, t_ms, f1_khz):
    """`(n, fit)` of the line at 2 f1 for each sample time, from `(P, F, T)` spectra."""
    which, fit = magfeatures.best_n(spec)
    power = (np.abs(spec) ** 2).mean(axis=0)
    column = np.clip(np.searchsorted(t_cols, t_ms), 0, len(t_cols) - 1)
    near = np.clip(np.searchsorted(t_cols, t_ms) - 1, 0, len(t_cols) - 1)
    column = np.where(
        np.abs(t_cols[near] - t_ms) < np.abs(t_cols[column] - t_ms), near, column
    )
    n_out = np.zeros(len(t_ms), dtype=int)
    fit_out = np.full(len(t_ms), np.nan)
    for i, (c, f1) in enumerate(zip(column, f1_khz, strict=True)):
        band = np.flatnonzero(np.abs(f_khz - 2.0 * f1) <= HARMONIC_BAND_KHZ)
        if not band.size:
            continue
        k = band[np.argmax(power[band, c])]
        n_out[i], fit_out[i] = which[k, c], fit[k, c]
    return n_out, fit_out


def reference_bins(shot, row):
    """`(ratio, best_n, fit)` per strong, frequency-matched harmonic bin of one shot."""
    with np.load(OUT / "signals" / f"{shot}.npz") as data:
        t = data["t_ms"]
        dt = float(np.median(np.diff(t)))
        width = max(1, round(5 / dt)) | 1
        n1 = median_filter(data["n1rms"], width)
        n2 = median_filter(data["n2rms"], width)
    with np.load(OUT / "signals_freq" / f"{shot}.npz") as data:
        f1 = np.interp(t, data["t_ms"], data["n1freq"], left=np.nan, right=np.nan)
        f2 = np.interp(t, data["t_ms"], data["n2freq"], left=np.nan, right=np.nan)
    window = (t >= row.window_start_ms) & (t < row.window_end_ms)
    finite_rms = np.isfinite(n1) & np.isfinite(n2)
    strong = continuous(window & finite_rms & (n1 >= 12), t, 50)
    coherent = (f1 > 0) & (f1 <= 30)
    harmonic = np.isfinite(f2) & (f2 > 0)
    harmonic &= np.abs(f2 - 2 * f1) <= np.maximum(1.0, 0.2 * f1)
    chosen = np.flatnonzero(strong & coherent & harmonic)
    if not chosen.size:
        return None
    array = corpus_signal(
        shot,
        "mirnov",
        channels=list(magfeatures.PROBE_ROWS),
        t_range=(row.window_start_ms - MARGIN_MS, row.window_end_ms + MARGIN_MS),
    )
    values = np.stack([finite(y) for y in array.y])
    t_cols, f_hz, spec = stft(
        array.x,
        values,
        rate_hz=magfeatures.RATE_HZ,
        nperseg=magfeatures.NPERSEG,
        hop=magfeatures.HOP,
    )
    keep = f_hz <= 70_000.0
    spec = spec[:, keep].astype(np.complex64)
    n_best, fit = harmonic_cells(
        spec, f_hz[keep] / 1000.0, t_cols, t[chosen], f1[chosen]
    )
    return n2[chosen] / n1[chosen], n_best, fit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--frozen", type=Path, default=SOURCES / "calibration_dev_fix1.json"
    )
    ap.add_argument("--out", type=Path, default=SOURCES / "calibration_dev_fix4.json")
    args = ap.parse_args(argv)
    cohort = pd.read_csv(CATALOG / "cohort.csv").set_index("shot")
    frozen = json.loads(args.frozen.read_text())
    shots = sorted(int(r["shot"]) for r in frozen["reference_shots"])
    blind = [s for s in shots if cohort.loc[s, "split"] == "test"]
    if blind:
        raise ValueError(f"blind test shots in the reference list: {blind}")
    ratios, bests, fits, sources, missing = [], [], [], [], []
    for shot in shots:
        try:
            got = reference_bins(shot, cohort.loc[shot])
        except (NoDataError, KeyError, OSError, ValueError) as exc:
            missing.append(
                {"shot": shot, "reason": f"{type(exc).__name__}: {exc}"[:200]}
            )
            continue
        if got is None:
            continue
        ratio, best, fit = got
        ratios.append(ratio)
        bests.append(best)
        fits.append(fit)
        coherent_one = (np.abs(best) == 1) & (fit >= magfeatures.COHERENT_FIT)
        sources.append(
            {
                "shot": shot,
                "n_bins": len(ratio),
                "n_phase_coherent_n1": int(coherent_one.sum()),
                "n_best_fit_n2": int(((np.abs(best) == 2) & (fit >= 0.9)).sum()),
            }
        )
    ratio = np.concatenate(ratios) if ratios else np.array([])
    best = np.concatenate(bests) if bests else np.array([], dtype=int)
    fit = np.concatenate(fits) if fits else np.array([])
    coherent_fit = np.isfinite(fit) & (fit >= magfeatures.COHERENT_FIT)
    sets = {
        "frequency_matched_all": np.ones(len(ratio), bool),
        "phase_coherent_best_fit_n1": coherent_fit & (np.abs(best) == 1),
        "phase_coherent_best_fit_n2": coherent_fit & (np.abs(best) == 2),
        "no_phase_coherent_line": ~coherent_fit,
    }
    owner = (
        np.concatenate(
            [np.full(len(r), s["shot"]) for r, s in zip(ratios, sources, strict=True)]
        )
        if ratios
        else np.array([], dtype=int)
    )
    summary = {
        name: {
            "n_bins": int(mask.sum()),
            "n_shots": len(set(owner[mask].tolist())),
            "ratio_quantiles": quantiles(ratio[mask]),
        }
        for name, mask in sets.items()
    }
    main_set = summary["phase_coherent_best_fit_n2"]
    enough = main_set["n_bins"] >= MIN_BINS and main_set["n_shots"] >= MIN_SHOTS
    if enough:
        quantile, source = 0.99, "phase_coherent_best_fit_n2"
        value = float(np.quantile(ratio[sets[source]], quantile))
    else:
        quantile, source = 0.95, "frequency_matched_all"
        value = float(np.quantile(ratio, quantile))
    record = {
        "made_by": "scripts/labeler/tm_harmonic_calibration.py",
        "git_sha": git_sha(),
        "definition": __doc__,
        "split": "development reference shots of calibration_dev_fix1.json only",
        "n_reference_shots": len(sources),
        "n_reference_bins": len(ratio),
        "sets": summary,
        "minimum_bins": MIN_BINS,
        "minimum_shots": MIN_SHOTS,
        "harmonic_set_large_enough": bool(enough),
        "selected_set": source,
        "selected_quantile": quantile,
        "harmonic_ratio_unrounded": value,
        "harmonic_ratio": math.ceil(value * 100) / 100,
        "frozen_reference_harmonic_ratio": frozen["harmonic_ratio"],
        "limit": (
            "toroidal phase cannot separate a harmonic of n = 1 from a co-rotating "
            "coupled n = 2 mode; the veto is a heuristic that may also remove real "
            "3/2 modes"
        ),
        "reference_shots": sources,
        "shots_without_mirnov": missing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "n_reference_shots",
                    "n_reference_bins",
                    "selected_set",
                    "selected_quantile",
                    "harmonic_ratio_unrounded",
                    "harmonic_ratio",
                )
            }
            | {
                name: {k: v[k] for k in ("n_bins", "n_shots")}
                for name, v in summary.items()
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
