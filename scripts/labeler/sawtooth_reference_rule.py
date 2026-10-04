"""Run the frozen sawtooth rule on the Muscatello reference shots 141182 and 141195.

The independent derivative picker of ``sawtooth_geometry_fix3.py reference`` is
the control. This script reads the same 10 kHz central-window arrays it saved
(40 channels, 2.0-5.1 s) and the same EFIT/ECE geometry bundles, then reports,
for the frozen rule and for the picker, crashes found against the published
expectation, picks the other method does not confirm, and the period and
amplitude against the published bands (85 +/- 5 ms, 0.35 +/- 0.02).

A second rule row switches only the ECE-validity test off. It is a diagnostic
of why the frozen rule abstains, not a rule: nothing here tunes the frozen rule.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import h5py
import numpy as np
from sawtooth_physics import ECE_GEOMETRY_ARCHIVE, OUTPUT, WORK, frozen_rule, save_json

from labeler.config import Paths
from labeler.sawtooth.geometry import load_radius_geometry, select_core
from labeler.sawtooth.physics import detect

REFERENCE = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix3/reference"
)  # read-only inputs of the earlier reference stage
WINDOWS = ((2.7, 3.0), (4.55, 4.8))
PERIOD_BAND_MS = (85.0, 5.0)
AMPLITUDE_BAND = (0.35, 0.02)
KNOWN_CRASH_S = {141182: 2.8371}
MATCH_S = 0.002


def central_amplitudes(trace, clock, times):
    """Published definition: pre/post medians over the cycle's first/last 10%."""
    stamps = np.asarray(times, dtype=float)
    out = []
    for index, stamp in enumerate(stamps):
        before_period = stamp - stamps[index - 1] if index else 0.085
        after_period = stamps[index + 1] - stamp if index + 1 < len(stamps) else 0.085
        before = trace[(clock >= stamp - 0.1 * before_period) & (clock < stamp - 3e-4)]
        after = trace[(clock >= stamp + 3e-4) & (clock < stamp + 0.1 * after_period)]
        if len(before) and len(after) and np.median(before) > 0:
            out.append(
                float((np.median(before) - np.median(after)) / np.median(before))
            )
    return out


def window_report(times, trace, clock, others):
    """Per published window: counts, period, amplitude, bands and agreement."""
    times = np.asarray(sorted(times), dtype=float)
    others = np.asarray(sorted(others), dtype=float)
    rows = []
    for left, right in WINDOWS:
        inside = times[(times >= left) & (times <= right)]
        gaps = np.diff(inside) * 1000
        amplitudes = central_amplitudes(trace, clock, inside)
        low = int(
            np.floor((right - left) / (PERIOD_BAND_MS[0] + PERIOD_BAND_MS[1]) * 1000)
        )
        high = int(
            np.ceil((right - left) / (PERIOD_BAND_MS[0] - PERIOD_BAND_MS[1]) * 1000)
        )
        period = float(np.median(gaps)) if len(gaps) else None
        amplitude = float(np.median(amplitudes)) if len(amplitudes) else None
        rows.append(
            {
                "window_s": [left, right],
                "crash_times_s": inside.tolist(),
                "crashes_found": len(inside),
                "crashes_expected_range": [low, high],
                "count_within_expected": bool(low <= len(inside) <= high),
                "median_period_ms": period,
                "period_pass": (
                    None
                    if period is None
                    else bool(abs(period - PERIOD_BAND_MS[0]) <= PERIOD_BAND_MS[1])
                ),
                "median_amplitude": amplitude,
                "amplitude_pass": (
                    None
                    if amplitude is None
                    else bool(abs(amplitude - AMPLITUDE_BAND[0]) <= AMPLITUDE_BAND[1])
                ),
            }
        )
    return rows


def agreement(times, others, span):
    """Picks without a counterpart within 2 ms, and counterparts left unpicked."""
    times, others = np.asarray(times, float), np.asarray(others, float)
    inside = lambda x: x[(x >= span[0]) & (x <= span[1])]
    mine, theirs = inside(times), inside(others)
    unconfirmed = [
        float(x) for x in mine if not len(theirs) or np.min(abs(theirs - x)) > MATCH_S
    ]
    missed = [
        float(x) for x in theirs if not len(mine) or np.min(abs(mine - x)) > MATCH_S
    ]
    return {"unconfirmed_picks_s": unconfirmed, "unpicked_reference_s": missed}


def run_rule(shot, clock, values, rule, work):
    padded = np.full((48, len(clock)), np.nan, dtype=np.float32)
    padded[:40] = values
    radius, geometry = load_radius_geometry(
        shot,
        clock,
        48,
        Paths.from_env(),
        archive_root=ECE_GEOMETRY_ARCHIVE,
        metadata_root=work / "geometry",
    )
    if radius is not None and radius.lcfs_outer_R_m is not None:
        padded[radius.R_m < 2 / 3 * radius.lcfs_outer_R_m[None]] = np.nan
    core = select_core(
        padded,
        radius=radius,
        minimum_channels=rule.minimum_channels,
        maximum_to_core=rule.maximum_channel_to_core,
    )
    padded[~core.physical_channels] = np.nan
    with h5py.File(work / "geometry" / f"{shot}.h5", "r", locking=False) as file:
        profiles = np.asarray(file["eq/qpsi"], dtype=float)
        valid = np.isfinite(profiles) & (profiles > 0)
        minimum = np.min(np.where(valid, profiles, np.inf), axis=1)
        minimum[~np.isfinite(minimum)] = np.nan
        qmin = np.asarray(file["eq/gtime"]) / 1000, minimum
    found = detect(
        clock,
        padded,
        shot=shot,
        rule=rule,
        qmin=qmin,
        q_source="EFIT01",
        radius_geometry=radius,
        spatially_verified=radius is not None,
        core_channels=core.core_channels,
    )
    return found, core, geometry


def states_summary(found, clock):
    dt = float(np.median(np.diff(clock)))
    return {
        "observable_s": float(found.observable.sum() * dt),
        "absent_s": float(found.absent_mask.sum() * dt),
        "q_prior_only_s": float((found.q_prior_mask & ~found.absent_mask).sum() * dt),
        "present_train_spans_s": [
            [float(e.t0_s), float(e.t1_s)] for e in found.intervals
        ],
        "uncertain_spans": len(found.uncertain_intervals),
        "record_s": float(len(clock) * dt),
    }


def calibration_step(values, clock, core_channel):
    """The ECEVS15 -> ECEVS17 level step and the central temperature."""
    row = {}
    for left, right in WINDOWS:
        inside = (clock >= left) & (clock <= right)
        levels = {
            c + 1: float(np.median(values[c, inside])) for c in (13, 14, 15, 16, 17)
        }
        row[f"{left}-{right}"] = {
            "median_te_kev_by_ecevs": levels,
            "ecevs17_over_ecevs15": levels[17] / levels[15] if levels[15] > 0 else None,
            "central_channel_median_te_kev": float(
                np.median(values[core_channel, inside])
            ),
        }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    rule = frozen_rule(args.work)
    off = replace(rule, ece_step_ratio=1e9, ece_axis_to_max=0.0)
    report = {
        "purpose": ("frozen rule on the published DIII-D crash references; no tuning"),
        "published": {
            "period_ms": PERIOD_BAND_MS,
            "amplitude": AMPLITUDE_BAND,
            "windows_s": WINDOWS,
            "known_crash_s": {str(k): v for k, v in KNOWN_CRASH_S.items()},
            "source": ".tmp/label_papers/Muscatello_ST.md",
        },
        "rule": asdict(rule),
        "match_tolerance_s": MATCH_S,
        "by_shot": [],
    }
    for shot in (141182, 141195):
        archive = np.load(args.reference / f"reference_{shot}.npz")
        clock, values = archive["t"], archive["y"]
        central = int(archive["central_channel"])
        picker = archive["crash_s"]
        record = {
            "shot": shot,
            "central_channel": central,
            "calibration": calibration_step(values, clock, central),
            "derivative_picker": {
                "windows": window_report(picker, values[central], clock, []),
                "crash_times_s": picker.tolist(),
            },
        }
        if shot in KNOWN_CRASH_S:
            nearest = float(np.min(abs(picker - KNOWN_CRASH_S[shot])) * 1000)
            record["derivative_picker"]["known_crash_offset_ms"] = nearest
        for name, variant in (("frozen_rule", rule), ("rule_validity_test_off", off)):
            found, core, _ = run_rule(shot, clock, values, variant, args.reference)
            times = [e.t0_s for e in found.crashes]
            row = {
                "crash_times_s": times,
                "windows": window_report(times, values[central], clock, picker),
                "states": states_summary(found, clock),
                "ece_validity": found.absence_diagnostics["ece_validity"],
                "guard_accounting": found.absence_diagnostics["guard_accounting"],
                "central_channel": core.info["central_channel"],
                "crash_states": {
                    state: sum(e.attrs["state"] == state for e in found.crashes)
                    for state in ("present", "uncertain")
                },
            }
            for window in row["windows"]:
                window["agreement_with_picker"] = agreement(
                    times, picker, window["window_s"]
                )
            if shot in KNOWN_CRASH_S and times:
                row["known_crash_offset_ms"] = float(
                    np.min(abs(np.asarray(times) - KNOWN_CRASH_S[shot])) * 1000
                )
            record[name] = row
        report["by_shot"].append(record)
    save_json(OUTPUT / "muscatello_rule_check.json", report)
    print(json.dumps(report["by_shot"][0]["frozen_rule"]["states"]))


if __name__ == "__main__":
    main()
