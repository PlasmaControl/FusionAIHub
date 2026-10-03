#!/usr/bin/env python
"""Record the published MARFE witness and probe/ELM physics provenance.

With --compute, regenerate only the named diagnostic shots into a separate
directory. Never overwrites the full bin set. Produces a small JSON record
for source-backed reporting of 199166 at 3705 ms and 206879 at 4150 ms.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import detach_bins
import numpy as np

from labeler.events.detachment import core, signals
from labeler.events.detachment import thresholds as th

ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
FIELDS = (
    "start_ms",
    "afrac_value",
    "afrac_valid",
    "afrac_reason",
    "afrac_vote",
    "prad_value",
    "prad_valid",
    "prad_reason",
    "prad_vote",
    "tangtv_value",
    "tangtv_valid",
    "tangtv_reason",
    "tangtv_vote",
    "tangtv_tier",
    "tangtv_marfe_candidate",
    "tangtv_marfe_spatial",
    "tangtv_marfe_second_cue",
    "tangtv_marfe_efit_source",
    "aux_greenwald_fraction",
    "greenwald_source",
    "aux_elm_share",
    "aux_elm_known",
    "aux_jsat_selected_probe",
    "aux_jsat_selected_r_m",
    "aux_jsat_selected_z_m",
    "aux_jsat_selected_psin",
    "aux_jsat_strike_r_m",
    "aux_jsat_radial_margin_m",
    "aux_jsat_selected_distance_m",
    "afrac_efit_source",
)


def scalar(value):
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def frame_witness(shot, target_ms):
    inv, maps = detach_bins.load_inversion(shot), signals.load_flux_map(shot)
    if inv is None or maps is None:
        return []
    rows = []
    for j in np.flatnonzero(np.abs(inv["times_ms"] - target_ms) <= 150):
        frame = inv["frames"][j]
        if not np.isfinite(frame).any():
            continue
        iz, ir = np.unravel_index(np.nanargmax(frame), frame.shape)
        t = float(inv["times_ms"][j])
        r, z = float(inv["radii"][ir]), float(inv["elevation"][iz])
        k = int(np.argmin(np.abs(maps["gtime_ms"] - t)))
        psi = signals.flux_at_positions(maps, [t], [[r, z]])[0, 0]
        rows.append(
            {
                "time_ms": t,
                "peak_r_m": r,
                "peak_z_m": z,
                "peak_psi_n": scalar(psi),
                "map_time_ms": float(maps["gtime_ms"][k]),
                "map_source": str(maps["source"]),
                "x_r_m": scalar(maps["rxpt1"][k]),
                "x_z_m": scalar(maps["zxpt1"][k]),
            }
        )
    return rows


def audit_shot(shot, directory):
    with np.load(directory / f"{shot}.npz") as f:
        bins = {k: f[k] for k in f.files}
    row = {
        "shot": shot,
        "bins": len(bins["start_ms"]),
        "valid": {k: int(bins[k + "_valid"].sum()) for k in detach_bins.INDICATORS},
        "marfe_votes": int(np.sum(bins["tangtv_vote"] == core.MARFE)),
        "elm_unknown_bins": int(np.sum(~bins["aux_elm_known"])),
        "nan_bins": {
            key: int(np.sum(~np.isfinite(bins[key])))
            for key in FIELDS
            if key in bins and bins[key].dtype.kind == "f"
        },
        "tier_counts": {
            str(tier): int(np.sum(bins["tangtv_tier"] == tier))
            for tier in np.unique(bins["tangtv_tier"])
        },
    }
    with np.load(signals.cache_file(shot)) as f:
        row["density"] = {
            "node": str(f.get("density_v2__node", "unknown")),
            "units": str(f.get("density_v2__units", "unknown")),
            "unit_confirmed_si": "density_v2_si__y" in f,
        }
    maps = signals.load_flux_map(shot)
    row["flux_map"] = (
        None
        if maps is None
        else {
            "source": str(maps["source"]),
            "slices": len(maps["gtime_ms"]),
        }
    )
    targets = {199166: [3705.0], 206879: [4150.0], 180257: [2400.0, 4800.0]}
    if shot in targets:
        requested_times = targets[shot]
        target = requested_times[0]
        centres = core.bin_centres(
            np.r_[bins["start_ms"], bins["start_ms"][-1] + core.BIN_MS]
        )
        selected = np.any(
            np.abs(centres[:, None] - np.asarray(requested_times)[None, :]) <= 150,
            axis=1,
        )
        row["witness_times_ms"] = requested_times
        row["witness_bins"] = [
            {key: scalar(bins[key][j]) for key in FIELDS if key in bins}
            for j in np.flatnonzero(selected)
        ]
        for witness in row["witness_bins"]:
            witness["failed_marfe_gates"] = [
                name
                for name, passed in (
                    ("camera_valid", witness["tangtv_valid"]),
                    ("height_candidate", witness["tangtv_marfe_candidate"]),
                    ("inside_separatrix_near_x", witness["tangtv_marfe_spatial"]),
                    (
                        "density_limit_or_back_transition",
                        witness["tangtv_marfe_second_cue"],
                    ),
                )
                if not passed
            ]
        if shot == 199166:
            row["reference"] = "Chen 2026 NF66 036014: MARFE observed at3705ms"
            row["reference_used_for_threshold_tuning"] = False
            row["peak_frames"] = frame_witness(shot, target)
        if shot == 180257:
            row["reference"] = "Eldon2021 workedexample: attached2400ms, detached4800ms"
            row["reference_used_for_threshold_tuning"] = False
    return row


def scan_inputs():
    """Availability and raw units of the full inversion population, no fetching."""
    shots = sorted(int(p.stem) for p in (ROOT / "inversions").glob("*.npz"))
    rows = []
    for shot in shots:
        with np.load(signals.cache_file(shot)) as f:
            row = {
                "shot": shot,
                "raw_density_unit": str(f.get("density_v2__units", "missing")),
                "density_node": str(f.get("density_v2__node", "missing")),
                "density_si_available": "density_v2_si__y" in f,
                "geometric_r0_available": "r0__y" in f,
                "cached_filterscope_channels": [
                    name
                    for name in ("fs01", "fs02", "fs03", "fs04")
                    if name + "__y" in f
                ],
            }
        maps = signals.load_flux_map(shot)
        row["multislice_map_source"] = (
            "unavailable" if maps is None else str(maps["source"])
        )
        rows.append(row)
    units = sorted({row["raw_density_unit"] for row in rows})
    sources = sorted({row["multislice_map_source"] for row in rows})
    return {
        "inversion_shots": shots,
        "inversion_shot_count": len(shots),
        "raw_unit_counts": {
            unit: sum(r["raw_density_unit"] == unit for r in rows) for unit in units
        },
        "density_si_shot_count": sum(r["density_si_available"] for r in rows),
        "geometric_r0_shot_count": sum(r["geometric_r0_available"] for r in rows),
        "multislice_map_counts": {
            source: sum(r["multislice_map_source"] == source for r in rows)
            for source in sources
        },
        "cached_filterscope_shot_count": sum(
            bool(r["cached_filterscope_channels"]) for r in rows
        ),
        "shots": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[199166, 206879, 189057]
    )
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--bins-dir", type=Path, default=ROOT / "physics_diagnostics")
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    if args.compute:
        for shot in args.shots:
            print(detach_bins.process(shot, out_dir=args.bins_dir), flush=True)
    record = {
        "bin_width_ms": core.BIN_MS,
        "bin_directory": str(args.bins_dir),
        "thresholds": {
            "elm_mask_half_width_ms": th.ELM_MASK_HALF_WIDTH_MS,
            "probe_strike_margin_m": th.PROBE_STRIKE_MARGIN_M,
            "probe_max_distance_m": th.PROBE_MAX_DISTANCE_M,
            "probe_sol_psi_n_min": th.PROBE_SOL_PSI_N_MIN,
            "marfe_dz_min": th.DZ_MARFE_MIN,
            "marfe_greenwald_min": th.GREENWALD_CUE_MIN,
            "marfe_min_adjacent_bins": 2,
        },
        "fetch_coverage": scan_inputs(),
        "shots": [audit_shot(s, args.bins_dir) for s in args.shots],
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(args.record, flush=True)


if __name__ == "__main__":
    main()
