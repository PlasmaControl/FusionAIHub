#!/usr/bin/env python
"""Audit the isolated DSM detector's repaired inputs and their native scales."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import dsm, labels, prepare
from labeler.timebase import window_mean

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/dsm/detection_input_audit.json"


def source_units(out, pace):
    """Probe permitted BCI diagnostics only; stop at any authentication error."""
    from labeler.features.resolve_fdp import _fetch_mds

    record = {
        "shot": 190643,
        "units": {},
        "errors": {},
        "git": git_sha(full=True),
        "script_sha256": sha256_of(__file__),
    }
    for name in ("DENV2F", "DENV3F", "DENV2UF", "DENV3UF"):
        try:
            result = _fetch_mds(r"\BCI::" + name, "bci", 190643, dims=["dim0"])
        except Exception as exc:  # noqa: BLE001 - preserve source/auth failures
            cause = str(exc)
            record["errors"][name] = cause
            if any(
                word in cause.lower()
                for word in (
                    "auth",
                    "login",
                    "credential",
                    "kerberos",
                    "expired",
                    "permission denied",
                    "unauthor",
                )
            ):
                record["stopped_on_auth_error"] = True
                break
        else:
            record["units"][name] = result["units"]
        time.sleep(pace)
    out.write_text(json.dumps(record, indent=1))
    print(json.dumps(record), flush=True)


def audit(paths):
    review = labels.review_table(prepare.review_csv(paths))
    shots = sorted(map(int, review.shot.unique()))
    work = paths.root / "round4/elm/dsm"
    unit_path = OUT.with_name("density_source_units.json")
    unit_probe = json.loads(unit_path.read_text()) if unit_path.exists() else None
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(),
        "script_sha256": sha256_of(__file__),
        "shots": shots,
        "n_shots": len(shots),
        "raw_detector_store": str(work / "repaired_raw_rows"),
        "source_units": unit_probe,
        "training_co2_units": "cm^-2; canonical co2 feature contract and source H5",
        "unit_decision": "The live DENV2F/3F metadata says V, whereas slow CO2 "
        "training columns are cm^-2. Compare paired 50 ms means numerically; "
        "do not infer a physical conversion from magnitude. Use native ordinates "
        "with training-only detector normalization; no physical rescaling applied.",
        "source_photodiode_pickle": "/projects/EKOLEMEN/wpqh_elm_hiro/data/dalpha_wpqh.pkl",
        "column_sources": {},
        "paired_density_scale": {},
    }
    columns = {n: i for i, n in enumerate(dsm.spec.COLUMNS)}
    for name in ("pcphd02", "pcphd03", "co2_v2", "co2_v3"):
        grouped = {}
        for shot in shots:
            row = dsm.load_rows(shot, work / "repaired_raw_rows" / f"{shot}.npz")
            if row is None:
                raise ValueError(f"repaired rows missing: {shot}")
            src = row.resolvers[name]
            grouped.setdefault(src, []).append(shot)
        record["column_sources"][name] = {
            source: {"n_shots": len(ids), "shots": ids}
            for source, ids in grouped.items()
        }
    for j, chord in enumerate(("v2", "v3")):
        ratios, relative, paired_shots = [], [], []
        for shot in shots:
            row = dsm.load_rows(shot, work / "raw_rows" / f"{shot}.npz")
            name = f"co2_density_slow_{chord}_downsampled"
            if row is None or name in row.filled:
                continue
            with np.load(prepare.signals_dir(paths) / f"{shot}.npz") as z:
                fast = window_mean(
                    z["t_int_ms"], z["interferometer"][j], dsm.ROW_T_MS - 50, 50
                )
            slow = row.x[:, columns[name]]
            keep = (
                row.usable
                & (dsm.ROW_T_MS >= 1000)
                & (dsm.ROW_T_MS <= 4500)
                & np.isfinite(fast)
                & (slow > 1e12)
                & (np.abs(fast) < 1e16)
            )
            if keep.any():
                ratios.extend((fast[keep] / slow[keep]).tolist())
                relative.extend((np.abs(fast[keep] - slow[keep]) / slow[keep]).tolist())
                paired_shots.append(shot)
        record["paired_density_scale"][chord] = {
            "shots": paired_shots,
            "n_shots": len(paired_shots),
            "rows": len(ratios),
            "ratio_fast_over_slow_p05_p50_p95": np.quantile(
                ratios, [0.05, 0.5, 0.95]
            ).tolist(),
            "relative_difference_p05_p50_p95": np.quantile(
                relative, [0.05, 0.5, 0.95]
            ).tolist(),
            "row_selection": "usable 50 ms means ending 1000-4500 ms; slow >1e12; "
            "finite fast with magnitude <1e16; descriptive scale audit, not calibration",
        }
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-units", action="store_true")
    ap.add_argument("--pace", type=float, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.pace < 1:
        ap.error("--pace must be at least 1 second")
    if args.probe_units:
        source_units(args.out, args.pace)
    else:
        args.out.write_text(json.dumps(audit(Paths.from_env()), indent=1))


if __name__ == "__main__":
    main()
