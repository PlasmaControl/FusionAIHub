#!/usr/bin/env python
"""Audit weak coherent tracks using n-resolved bands instead of the dominant line.

The aN features already require a best-fitting toroidal n and phase fit >=0.9
in each Fourier cell. Requiring fitN at the strongest overall line can suppress
another coherent n line. Report both masks; weak amplitude floors are frozen
development quiet-time percentiles, not tuned on these review examples.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.tearing import rule
from labeler.tearing.magfeatures import FEATURE_NAMES

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
TM = ROOT / "round4/tm"


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )


def tracks(mask, available, t, *, core_ms=100.0, gap_ms=50.0):
    """Joined tracks must contain a pre-joining core; unavailable bins are barriers."""
    dt = float(np.median(np.diff(t)))
    raw = runs(mask & available)
    joined = []
    for a, b in raw:
        core = (b - a) * dt >= core_ms
        if (
            joined
            and (a - joined[-1][1]) * dt <= gap_ms
            and available[joined[-1][1] : a].all()
        ):
            joined[-1][1] = b
            joined[-1][2] |= core
        else:
            joined.append([a, b, core])
    return [
        [float(t[a] - dt / 2), float(t[b - 1] + dt / 2)]
        for a, b, core in joined
        if core
    ]


def anchored_tracks(seed, release, available, t, *, core_ms=100.0, gap_ms=50.0):
    """Lower-level tracks containing a continuous high-level core before joining."""
    dt = float(np.median(np.diff(t)))
    cores = [(a, b) for a, b in runs(seed & available) if (b - a) * dt >= core_ms]
    joined = []
    for a, b in runs(release & available):
        if (
            joined
            and (a - joined[-1][1]) * dt <= gap_ms
            and available[joined[-1][1] : a].all()
        ):
            joined[-1][1] = b
        else:
            joined.append([a, b])
    return [
        [float(t[a] - dt / 2), float(t[b - 1] + dt / 2)]
        for a, b in joined
        if any(a <= lo and hi <= b for lo, hi in cores)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, nargs="+", default=[196494, 187072])
    parser.add_argument(
        "--calibration", type=Path, default=TM / "labels/calibration_dev_fix1.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--island-inventory-shot", type=int)
    parser.add_argument("--label-meta", type=Path)
    args = parser.parse_args(argv)
    cohort = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data/events/catalog/cohort.csv"
    )
    blind = set(cohort.loc[cohort.split.eq("test"), "shot"])
    args.shots = [shot for shot in args.shots if shot not in blind]
    if args.island_inventory_shot is not None:
        shot = args.island_inventory_shot
        if shot in blind or shot not in set(cohort.shot):
            raise ValueError("island inventory requires a nonblind development shot")
        paths = Paths.from_env()
        candidates = [paths.corpus / f"{shot}_processed.h5"]
        candidates += [
            replace(paths, root=root).features_file(shot)
            for root in (TM / "lroot", ROOT)
        ]
        inventory = []
        for path in candidates:
            item = {"path": str(path), "exists": path.is_file(), "groups": {}}
            if path.is_file():
                with h5py.File(path, "r") as f:
                    for name in f:
                        if "ece" not in name.lower() and "qpsi" not in name.lower():
                            continue
                        group = f[name]
                        item["groups"][name] = {
                            "attrs": {
                                key: str(value) for key, value in group.attrs.items()
                            },
                            "datasets": {
                                key: {
                                    "shape": list(value.shape),
                                    "attrs": {
                                        k: str(v) for k, v in value.attrs.items()
                                    },
                                }
                                for key, value in group.items()
                                if isinstance(value, h5py.Dataset)
                            },
                        }
            inventory.append(item)
        record = {
            "made_by": "scripts/labeler/tm_weak_diagnostics.py",
            "shot": shot,
            "split": str(cohort.loc[cohort.shot.eq(shot), "split"].iloc[0]),
            "inventory": inventory,
            "m_supported": False,
            "assessment": "Inventory-only island-radius attempt: ECE temperature "
            "samples or channel coordinates alone do not establish a localized "
            "tearing-related temperature flattening or calibrated island radius. "
            "No validated island/flattening reconstruction is provided in these "
            "local groups. EFIT q(rho) alone cannot assign m. Retain m empty; "
            "no ECE fetch attempted.",
        }
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        if args.label_meta is not None:
            meta = json.loads(args.label_meta.read_text())
            meta["m"]["inventory"] = {
                "record": str(args.out),
                "made_by": record["made_by"],
                "shot": shot,
                "assessment": record["assessment"],
            }
            args.label_meta.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return 0
    calibration = json.loads(args.calibration.read_text())
    record = {
        "made_by": "scripts/labeler/tm_weak_diagnostics.py",
        "policy": __doc__,
        "calibration": str(args.calibration),
        "shots": {},
    }
    for shot in args.shots:
        with np.load(TM / "magfeatures" / f"{shot}.npz") as data:
            t, features = data["centres_ms"], data["features"]
        available = np.isfinite(features).all(axis=1)
        with np.load(TM / "signals" / f"{shot}.npz") as data:
            rt, rms, _ = rule.uniform(data["t_ms"], data["n1rms"])
        # Any acquisition loss in a 10 ms bin makes it a hard track barrier.
        missing_bins = np.searchsorted(t + 5.0, rt[~np.isfinite(rms)])
        missing_bins = missing_bins[(missing_bins >= 0) & (missing_bins < len(t))]
        available[missing_bins] = False
        available &= (t >= rt[0]) & (t <= rt[-1])
        prominence = features[:, FEATURE_NAMES.index("line_prominence_db")]
        summary = {}
        for n in (1, 2):
            indices = [
                i for i, name in enumerate(FEATURE_NAMES) if name.startswith(f"a{n}_")
            ]
            amplitude = features[:, indices].max(axis=1)
            fit = features[:, FEATURE_NAMES.index(f"fit{n}")]
            floor = calibration["magnetic_background"][str(n)][
                "quantiles_log10_amplitude"
            ]["0.95"]
            band_mask = (amplitude > floor) & (prominence >= 10) & available
            dominant_mask = band_mask & (fit >= 0.9)
            inside = (t >= 2000) & (t < 5000) & available
            summary[str(n)] = {
                "frozen_amplitude_floor_log10": floor,
                "n_available_bins_2_to_5s": int(inside.sum()),
                "n_band_only_bins_2_to_5s": int((inside & band_mask).sum()),
                "n_dominant_fit_bins_2_to_5s": int((inside & dominant_mask).sum()),
                "band_only_tracks_ms": tracks(band_mask, available, t),
                "dominant_fit_tracks_ms": tracks(dominant_mask, available, t),
                "raw_band_only_core_runs_ms": tracks(band_mask, available, t, gap_ms=0),
                "raw_dominant_fit_core_runs_ms": tracks(
                    dominant_mask, available, t, gap_ms=0
                ),
                "p95_seed_release_at_floor": {
                    "release_floor_log10": floor,
                    "tracks_ms": anchored_tracks(
                        band_mask,
                        (amplitude > floor) & (prominence >= 10),
                        available,
                        t,
                    ),
                },
                "development_amplitude_percentile_sensitivity": {
                    quantile: {
                        "floor_log10": value,
                        "n_band_bins_2_to_5s": int(
                            (inside & (amplitude > value) & (prominence >= 10)).sum()
                        ),
                        "band_tracks_ms": tracks(
                            (amplitude > value) & (prominence >= 10), available, t
                        ),
                    }
                    for quantile, value in calibration["magnetic_background"][str(n)][
                        "quantiles_log10_amplitude"
                    ].items()
                    if quantile in ("0.5", "0.9", "0.95", "0.99")
                },
                "per_100ms_2_to_5s": [
                    {
                        "start_ms": start,
                        "median_amplitude_log10": float(
                            np.median(amplitude[(t >= start) & (t < start + 100)])
                        ),
                        "n_band_only": int(
                            (band_mask & (t >= start) & (t < start + 100)).sum()
                        ),
                        "n_dominant_fit": int(
                            (dominant_mask & (t >= start) & (t < start + 100)).sum()
                        ),
                    }
                    for start in range(2000, 5000, 100)
                ],
            }
        record["shots"][str(shot)] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                shot: {
                    n: {k: v for k, v in data.items() if k != "per_100ms_2_to_5s"}
                    for n, data in modes.items()
                }
                for shot, modes in record["shots"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
