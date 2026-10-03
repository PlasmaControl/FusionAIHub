"""Freeze detector guards on fixed TRAIN shots and measure native-filter impact."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_physics import (
    OUTPUT,
    REPO,
    REVIEW,
    WORK,
    process_shot,
    records_at,
    save_json,
)

from labeler.sawtooth.metrics import event_cells
from labeler.sawtooth.physics import Rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "impact"])
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    experts = set(pd.read_csv(REVIEW).shot)
    eligible = sorted(set(cohort[cohort.split == "train"].shot) - experts)
    shots = sorted(np.random.default_rng(20261004).choice(eligible, 16, False).tolist())
    windows = {
        int(r.shot): (r.window_start_ms / 1000, r.window_end_ms / 1000)
        for r in cohort.itertuples()
    }
    if args.stage == "freeze":
        pilot = args.work / "train_calibration"
        permissive = replace(Rule(), central_relative_drop=0.02)
        save_json(pilot / "freeze.json", {"rule": asdict(permissive)})
        with ProcessPoolExecutor(max_workers=8) as pool:
            records = list(
                pool.map(
                    process_shot, [(s, str(pilot), False, windows[s]) for s in shots]
                )
            )
        amplitudes = [
            r["attrs"]["central_relative_drop"]
            for rec in records
            for r in rec["crashes"]
        ]
        if not amplitudes:
            raise ValueError("no physical train candidates on development shots")
        # Lower-tail development amplitude, rounded down to avoid false precision.
        amplitude = float(
            np.floor(np.clip(np.quantile(amplitudes, 0.1), 0.05, 0.15) * 100) / 100
        )
        rule = replace(Rule(), central_relative_drop=amplitude)
        prior_shots = sorted(
            int(p.stem) for p in (WORK.parent / "pilot_shots").glob("*.json")
        )
        prior_splits = (
            cohort[cohort.shot.isin(prior_shots)].split.value_counts().to_dict()
        )
        frozen = {
            "seed": 20261004,
            "development_shots": shots,
            "excluded_experts": sorted(experts),
            "split": "train",
            "rule": asdict(rule),
            "amplitude_selection": {
                "formula": "floor(100 * clip(train candidate q10, 0.05, 0.15))/100",
                "candidate_count": len(amplitudes),
                "quantiles": np.quantile(amplitudes, [0.05, 0.1, 0.5, 0.9]).tolist(),
                "selected": amplitude,
            },
            "guards": {
                k: {
                    "value": v,
                    "shots": shots,
                    "origin": (
                        "Gude filter/profile prior"
                        if k
                        in {
                            "sigma_ms",
                            "frame_ms",
                            "posr_threshold",
                            "coincidence_ms",
                            "minimum_channels",
                            "significance",
                            "maximum_net",
                            "minimum_block",
                        }
                        else "TRAIN lower-tail amplitude with 5% prior floor"
                        if k == "central_relative_drop"
                        else "explicit review/implementation prior; not data optimized"
                    ),
                    "evidence": "per-shot train_audit below; no expert validation",
                }
                for k, v in asdict(rule).items()
            },
            "reader_guards": {
                "core_proxy_policy": (
                    "actual RF frequency/Bt/axis mapping when supported, otherwise "
                    "hottest physically screened channel and adjacent neighbors"
                ),
                "ece_valid_range_kev": [0, 100],
                "density_quantile": 0.9,
                "density_cutoff_margin": 0.9,
                "density_no_bt_guard_m3": 8e19,
                "fir_half_length_decimation_factors": 10,
                "sample_rate_hz": 10000,
                "shots": shots,
                "cutoff_note": (
                    "density/B proxy; channel-specific calibration only if actual "
                    "frequency metadata are present"
                ),
            },
            "profile_and_support_guards": {
                "shots": shots,
                "channel_fractional_sign": 0.005,
                "uncalibrated_hot_slack_channels": 2,
                "uncalibrated_hot_block_fraction": 0.7,
                "pre_post_windows_ms": [[-1.5, -0.3], [0.3, 1.5]],
                "minimum_valid_profile_channels": 4,
                "minimum_analysis_samples": 1000,
                "minimum_finite_ece_fraction": 0.1,
                "population_ip_extent_peak_fraction": 0.1,
                "auxiliary_near_ms": 2,
                "auxiliary_background_ms": 25,
                "origin": "fixed priors audited solely on development TRAIN shots",
            },
            "train_audit": [
                {
                    "shot": r["shot"],
                    "crashes": len(r["crashes"]),
                    "intervals": len(r["intervals"]),
                    "error": r.get("error"),
                    "density_guard": r.get("density_guard"),
                    "rejected": r.get("rejected", {}),
                    "state_seconds": r.get("state_seconds", {}),
                    "absence_diagnostics": r.get("absence_diagnostics", {}),
                    "core_geometry": r.get("core_geometry", {}),
                }
                for r in records
            ],
            "prior_pilot_correction": {
                "shot_count": len(prior_shots),
                "shots": prior_shots,
                "split_counts": prior_splits,
                "expert_shots_in_pilot": sorted(set(prior_shots) & experts),
                "statement": (
                    "Prior train-only pilot claim withdrawn; mixed pilot cannot "
                    "establish leakage-free calibration. All fix-round guards "
                    "refrozen using only listed TRAIN shots."
                ),
            },
        }
        save_json(args.work / "freeze.json", frozen)
        save_json(OUTPUT / "freeze.json", frozen)
        print(
            f"Frozen relative drop {amplitude}; {len(amplitudes)} train candidates",
            flush=True,
        )
    else:
        import json

        frozen = json.loads((args.work / "freeze.json").read_text())
        shots = frozen["development_shots"][:4]
        aliased = args.work / "alias_comparison"
        save_json(aliased / "freeze.json", frozen)
        with ProcessPoolExecutor(max_workers=4) as pool:
            old = list(
                pool.map(
                    process_shot,
                    [(s, str(aliased), False, windows[s], "aliased") for s in shots],
                )
            )
        fresh = {r["shot"]: r for r in records_at(args.work, shots)}
        comparisons = []
        for prior in old:
            rec = fresh[prior["shot"]]
            a = np.array([r["time_s"] for r in prior["crashes"]])
            b = np.array([r["time_s"] for r in rec["crashes"]])
            from labeler.scoring.events import match

            paired = match(a * 1000, b * 1000, 2)
            residuals = [(b[j] - a[i]) * 1000 for i, j in paired.pairs]
            comparisons.append(
                {
                    "shot": rec["shot"],
                    "old_candidates": prior.get("candidates"),
                    "native_filtered_candidates": rec.get("candidates"),
                    "old_crashes": len(a),
                    "native_filtered_crashes": len(b),
                    "cells_2ms": event_cells(a, b).astype(int).tolist(),
                    "matched_timing_residual_ms": residuals,
                    "median_absolute_timing_change_ms": float(
                        np.median(np.abs(residuals))
                    )
                    if residuals
                    else None,
                    "old_error": prior.get("error"),
                    "new_error": rec.get("error"),
                }
            )
        save_json(
            OUTPUT / "antialias_impact.json",
            {
                "shots": shots,
                "split": "train",
                "rule": frozen["rule"],
                "by_shot": comparisons,
                "comparison": "same frozen detector; only ECE reader changes",
            },
        )
        print(comparisons, flush=True)


if __name__ == "__main__":
    main()
