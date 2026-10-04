#!/usr/bin/env python
"""Score the detachment states against divertor Thomson Te near the outer target.

The processed divertor Thomson Te (`\\ELECTRONS::TSTE_DIV`, 14 or 16 chords at R = 1.485 m,
20 ms, fetched by `detach_fetch_round4.py --stage dts`) is a temperature
measurement that none of the three indicators uses, so it checks them where the
label cannot: the cold-target criterion that the literature itself uses.

* Chords: those 1.5 to 5 cm above the outer shelf (Z = -1.25 m). The lowest chord
  (0.7 cm) is left out for stray light (Eldon 2017) and Chen 2026's own chords on
  shot 201081 are Z = -1.222 and -1.205 m. A chord counts at a time only while it
  lies on the SOL side of the separatrix and in the near SOL, 1.000 < psiN <= 1.05,
  by the multi-slice EFIT map on disk (the strike point moves off the chord on
  other shots); samples outside 0.1-50 eV are rejected (Eldon 2017).
* Bands: attached is Te >= 10 eV, detached Te <= 5 eV (Eldon 2017's detachment
  threshold; the Te cliff jumps between about 1-3 eV and 8-20 eV and 3-8 eV values
  are rare). Chen 2026 reports 20 eV to 3 eV across the cliffs of shot 201081.
  The bands were fixed from those digests before scoring.
* A bin's Te is the median of the selected chords' valid samples inside it.

Reported per primary state and per indicator vote: how many bins and shots carry a
Te, the Te quantiles, the share inside the band, and a threshold-free AUROC of
-Te for the detached against the attached bins (pooled and within shots, with
shot-bootstrap intervals). Shot 201081's two cliff times are read from the data and
set beside the published ones. No threshold is fitted here.

    pixi run --frozen -e labelmaker python scripts/labeler/detach_te_check.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import detach_benchmark as bench
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, signals
from labeler.events.detachment import thresholds as th

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
OUT = REPO / "docs/labeler/results/detachment_te_check.json"
CHORD_MIN_ABOVE_SHELF_M = 0.015
CHORD_MAX_ABOVE_SHELF_M = 0.05
PSI_N_MIN, PSI_N_MAX = 1.0, 1.05
TE_VALID_EV = (0.1, 50.0)
TE_ATTACHED_MIN_EV = 10.0
TE_DETACHED_MAX_EV = 5.0
PUBLISHED_CLIFFS_MS = {201081: (2650.0, 4450.0)}
MIN_SHOT_BINS = 5


def load_dts(shot: int):
    path = ROOT / "dts" / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as f:
        if "te" not in f.files:
            return None
        return {k: f[k] for k in f.files if k != "status"}


def target_te(shot: int, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin Te (eV, NaN if none) and the number of chord samples behind it."""
    dts = load_dts(shot)
    maps = signals.load_flux_map(shot)
    nan = np.full(len(starts), np.nan)
    if dts is None or maps is None:
        return nan, np.zeros(len(starts), int)
    te, t, r, z = dts["te"], dts["t_ms"], dts["r"], dts["z"]
    above = z - th.SHELF_Z
    chords = np.flatnonzero(
        (above >= CHORD_MIN_ABOVE_SHELF_M) & (above <= CHORD_MAX_ABOVE_SHELF_M)
    )
    if not len(chords):
        return nan, np.zeros(len(starts), int)
    psi = signals.flux_at_positions(maps, t, np.stack([r[chords], z[chords]], axis=1))
    chord_te = te[chords].astype(float)
    ok = (
        np.isfinite(chord_te)
        & (chord_te >= TE_VALID_EV[0])
        & (chord_te <= TE_VALID_EV[1])
        & (psi > PSI_N_MIN)
        & (psi <= PSI_N_MAX)
    )
    chord_te = np.where(ok, chord_te, np.nan)
    width = core.BIN_MS
    index = np.floor((t - starts[0]) / width).astype(int)
    keep = (index >= 0) & (index < len(starts))
    value = np.full(len(starts), np.nan)
    count = np.zeros(len(starts), int)
    for b in np.unique(index[keep]):
        samples = chord_te[:, keep & (index == b)].ravel()
        samples = samples[np.isfinite(samples)]
        if len(samples):
            value[b], count[b] = np.median(samples), len(samples)
    return value, count


def attach_te(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["te_ev"] = np.nan
    frame["te_n"] = 0
    for shot, idx in frame.groupby("shot").indices.items():
        starts = frame.start_ms.to_numpy(float)[idx]
        grid = np.arange(starts.min(), starts.max() + core.BIN_MS, core.BIN_MS)
        value, count = target_te(int(shot), grid)
        place = np.rint((starts - grid[0]) / core.BIN_MS).astype(int)
        frame.iloc[idx, frame.columns.get_loc("te_ev")] = value[place]
        frame.iloc[idx, frame.columns.get_loc("te_n")] = count[place]
    return frame


def summarise(te: np.ndarray, band) -> dict:
    te = te[np.isfinite(te)]
    if not len(te):
        return {"n_bins": 0}
    return {
        "n_bins": len(te),
        "te_ev_quantiles_10_50_90": [float(v) for v in np.percentile(te, [10, 50, 90])],
        "share_in_band": float(np.mean(band(te))),
        "share_between_bands": float(
            np.mean((te > TE_DETACHED_MAX_EV) & (te < TE_ATTACHED_MIN_EV))
        ),
    }


def score_states(frame, state_column, rng, mask=None) -> dict:
    """Te summaries of the attached/detached/MARFE bins and the AUROC of -Te."""
    sel = frame[np.isfinite(frame.te_ev)]
    if mask is not None:
        sel = sel[mask[np.isfinite(frame.te_ev).to_numpy()]]
    out = {}
    for code, name, band in (
        (core.ATTACHED, "attached", lambda x: x >= TE_ATTACHED_MIN_EV),
        (core.DETACHED, "detached", lambda x: x <= TE_DETACHED_MAX_EV),
        (core.MARFE, "marfe", lambda x: x <= TE_DETACHED_MAX_EV),
    ):
        rows = sel[sel[state_column] == code]
        out[name] = {
            **summarise(rows.te_ev.to_numpy(float), band),
            "n_shots": int(rows.shot.nunique()),
            "shot_ids": sorted(int(s) for s in rows.shot.unique()),
        }
    two = sel[sel[state_column].isin((core.ATTACHED, core.DETACHED))]
    pooled = bench.auroc_boot(
        -two.te_ev.to_numpy(float),
        (two[state_column] == core.DETACHED).to_numpy(),
        two.shot.to_numpy(),
        rng,
    )
    per_shot = []
    for _, rows in two.groupby("shot"):
        positive = (rows[state_column] == core.DETACHED).to_numpy()
        if min(positive.sum(), (~positive).sum()) >= MIN_SHOT_BINS:
            per_shot.append(bench.auroc(-rows.te_ev.to_numpy(float), positive))
    finite = np.asarray([v for v in per_shot if np.isfinite(v)])
    draws = [
        float(np.mean(finite[rng.integers(0, len(finite), len(finite))]))
        for _ in range(bench.REPLICATES if len(finite) else 0)
    ]
    out["auroc_neg_te_detached_vs_attached"] = {
        "pooled": pooled,
        "within_shot": {
            "min_class_bins": MIN_SHOT_BINS,
            "n_shots": len(finite),
            "mean": float(finite.mean()) if len(finite) else None,
            "mean_ci95": bench.interval(draws),
        },
    }
    return out


def cliffs(frame: pd.DataFrame, shot: int) -> dict | None:
    """Te-cliff times of one shot read from the data: the first and last bin in
    its flat top where the Te median of 3 bins crosses the 5-10 eV gap."""
    rows = frame[(frame.shot == shot) & np.isfinite(frame.te_ev)]
    if rows.empty:
        return None
    t = rows.start_ms.to_numpy(float) + core.BIN_MS / 2
    te = rows.te_ev.rolling(3, center=True, min_periods=1).median().to_numpy(float)
    cold = te <= TE_DETACHED_MAX_EV
    warm = te >= TE_ATTACHED_MIN_EV
    first_cold = float(t[cold][0]) if cold.any() else None
    last_cold = float(t[cold][-1]) if cold.any() else None
    return {
        "first_cold_ms": first_cold,
        "last_cold_ms": last_cold,
        "warm_before_first_cold_share": float(np.mean(warm[t < first_cold]))
        if first_cold
        else None,
        "published_cliffs_ms": list(PUBLISHED_CLIFFS_MS.get(shot, ())),
        "te_ev_at_2000_ms_and_3500_ms_and_5000_ms": [
            float(
                np.nanmedian(
                    rows.te_ev[(rows.start_ms >= x - 100) & (rows.start_ms <= x + 100)]
                )
            )
            for x in (2000.0, 3500.0, 5000.0)
        ],
    }


def main() -> int:
    frame = pd.read_csv(ROOT / "labels_bins.csv.gz")
    frame = attach_te(frame)
    rng = np.random.default_rng(0)
    fetched = sorted(int(p.stem) for p in (ROOT / "dts").glob("*.npz"))
    usable = sorted(
        int(s) for s in frame.loc[np.isfinite(frame.te_ev), "shot"].unique()
    )
    record = {
        "bin_ms": core.BIN_MS,
        "method": __doc__.split("\n\n")[1:5],
        "bands_ev": {
            "attached_min": TE_ATTACHED_MIN_EV,
            "detached_max": TE_DETACHED_MAX_EV,
            "valid": list(TE_VALID_EV),
        },
        "chord_selection": {
            "above_shelf_m": [CHORD_MIN_ABOVE_SHELF_M, CHORD_MAX_ABOVE_SHELF_M],
            "shelf_z_m": th.SHELF_Z,
            "psi_n_window": [PSI_N_MIN, PSI_N_MAX],
        },
        "shots_with_dts_fetched": fetched,
        "shots_with_te_in_assessed_bins": usable,
        "bins_with_te": int(np.isfinite(frame.te_ev).sum()),
        "bins_assessed": len(frame),
        "primary_state": score_states(frame, "state_rule", rng),
        "indicator_votes": {
            name: score_states(
                frame.assign(
                    vote=frame[f"{name}_vote"].where(frame[f"{name}_valid"], 0)
                ),
                "vote",
                rng,
            )
            for name in ("afrac", "prad", "tangtv")
        },
        "cliff_check": {str(s): cliffs(frame, s) for s in PUBLISHED_CLIFFS_MS},
    }
    OUT.write_text(dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("bins_with_te", "bins_assessed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
