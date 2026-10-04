#!/usr/bin/env python
"""Audit onset provenance and probe existing n=1 amplitude candidate nodes.

The node inventory establishes locators, not their physical meaning. These
OPERATIONS channels are therefore candidates only, never registered RWM sensors
or model inputs. The probe checks units, cadence and coverage on three Hanson
shots. New arrays go only to $LABELER_ROOT/round4/rwm/sensor_probe. Run on the
login node under pixi's labelmaker environment and ``fdp run python`` with
TMPDIR=$LABELER_ROOT/scratch/tmp-fetch. Fetching is serial, paced by one second,
and stops at the first authentication error without retrying.

``--source-only`` records the disk search without any network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.events.raw import FetchSpec
from labeler.features.resolve_fdp import _fetch_mds

MAIN = Path("/scratch/gpfs/nc1514/FusionAIHub")
SCRATCH = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4")
INVENTORY = SCRATCH / "tmp/detach/d3d_nodes_199166.txt"
REPORT = REPO / "outputs/labeler/rwm/sensor_probe.json"
SHOTS = (156785, 158021, 176068)
# These exact locators are in the on-disk archive inventory, not guessed nodes.
# Their sensor/coil meaning and corrections are unresolved; BAMP is not taken
# to mean an RWM magnetic amplitude merely because its spelling suggests it.
CANDIDATES = {
    name.lower(): FetchSpec(
        exprs=(rf"\OPERATIONS::{name}",), via="mds", tree="OPERATIONS"
    )
    for name in ("CN1BAMP", "ILN1BAMP", "IUN1BAMP")
}


def source_audit() -> dict:
    """Hash the supplied sources and record what they can establish."""
    sources = [
        MAIN / "data/events/resistive_wall_mode/raw/rwm_onsets_2017.csv",
        MAIN / "data/events/resistive_wall_mode/raw/rwm_onsets_2024.csv",
        MAIN / "data/events/resistive_wall_mode/README.md",
        MAIN / ".tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md",
        MAIN / ".tmp/label_papers/outside/Piccione_tsdw2021_RWM_poster.md",
        REPO / "src/labeler/features/namespace.py",
        REPO / "src/labeler/events/raw.py",
        MAIN / ".tmp/magnetic_mapping.csv",
        INVENTORY,
    ]
    entries = []
    pattern = re.compile(
        r"ONSET_TIME|RWM sensor|saddle|CN1BAMP|ILN1BAMP|IUN1BAMP", re.IGNORECASE
    )
    for path in sources:
        if not path.is_file():
            entries.append({"path": str(path), "exists": False})
            continue
        content = path.read_bytes()
        lines = content.decode(errors="replace").splitlines()
        entries.append(
            {
                "path": str(path),
                "exists": True,
                "sha256": hashlib.sha256(content).hexdigest(),
                "relevant_lines": [
                    {"line": i, "text": line}
                    for i, line in enumerate(lines, 1)
                    if pattern.search(line)
                ],
            }
        )
    return {
        "sources": entries,
        "onset_time_detection_meaning_verified": False,
        "onset_finding": (
            "Hanson CSV headers and supplied category README provide onset points "
            "without defining growth-start versus detection/threshold timing. "
            "The Piccione digests' sensor threshold concerns NSTX t_RWM, not "
            "Hanson's DIII-D ONSET_TIME."
        ),
        "sensor_finding": (
            "No source identifies a corrected low-frequency n=1 RWM sensor "
            "locator. The inventory lists the three OPERATIONS n=1 amplitude "
            "candidate locators exactly, but establishes neither diagnostic "
            "meaning nor RWM specificity. MHD N1RMS/N2RMS are generic RMS; "
            "magnetic_mapping.csv documents probes, not an RWM-sensor product."
        ),
        "physical_annotation": "20 ms pre-onset windows are category 2 uncertain",
        "forecast_target": "separate point-time forecast target remains unchanged",
    }


def authentication_error(error: Exception) -> bool:
    """Recognize authentication in the full exception chain, without retries."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        text = f"{type(error).__name__}: {error}".lower()
        if any(
            s in text
            for s in (
                "auth",
                "login",
                "log in",
                "unauthorized",
                "permission denied",
                "expired token",
                "invalid token",
                "access denied",
                "401",
                "403",
            )
        ):
            return True
        error = error.__cause__ or error.__context__
    return False


def write_report(report: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--workers", type=int, choices=[1], default=1)
    parser.add_argument("--record-launch-auth-error", action="store_true")
    args = parser.parse_args()
    if args.pace < 1.0:
        parser.error("pace must be at least one second")
    report = {
        "script": "scripts/labeler/rwm_sensor_probe.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_search": source_audit(),
        "candidate_specs": [
            {"name": name, **asdict(spec)} for name, spec in CANDIDATES.items()
        ],
        "validated_rwm_sensor_specs": [],
        "requested_shots": list(SHOTS),
        "workers": args.workers,
        "pace_seconds": args.pace,
        "fetch_attempted": False,
        "stopped_on_auth_error": False,
        "production_corpus_or_cache_written": False,
        "probes": [],
        "status": "no corrected low-frequency RWM sensor identified on disk",
    }
    if args.record_launch_auth_error:
        report.update(
            fetch_attempted=True,
            stopped_on_auth_error=True,
            status="fdp launch authentication failed; no sensor fetch; stopped",
        )
        write_report(report)
        return 3
    write_report(report)
    if args.source_only:
        return 0
    output = Paths.from_env().root / "round4/rwm/sensor_probe"
    # Confirm each candidate is present verbatim in the archived node inventory.
    inventory = INVENTORY.read_text()
    if any(spec.exprs[0] + " SIGNAL" not in inventory for spec in CANDIDATES.values()):
        raise ValueError("a candidate lacks its exact archived locator")
    report["fetch_attempted"] = True
    write_report(report)
    for shot in SHOTS:
        for name, spec in CANDIDATES.items():
            row = {"shot": shot, "name": name, "expr": spec.exprs[0]}
            started = time.monotonic()
            try:
                record = _fetch_mds(spec.exprs[0], spec.tree, shot, dims=["dim0"])
                values = np.asarray(record["data"], dtype=float)
                clock = np.asarray(record["dim0"], dtype=float)
                if values.ndim != 1 or clock.ndim != 1 or len(clock) != len(values):
                    raise ValueError("candidate is not a scalar signal with a clock")
                if len(clock) < 2 or not np.isfinite(clock).all():
                    raise ValueError("candidate has no valid sampled clock")
                steps = np.diff(clock)
                if np.any(steps <= 0):
                    raise ValueError("candidate clock does not increase strictly")
                output.mkdir(parents=True, exist_ok=True)
                target = output / f"{shot}_{name}.npz"
                np.savez_compressed(target, x=clock, y=values, expr=spec.exprs[0])
                row.update(
                    fetched=True,
                    units={str(k): str(v) for k, v in record.get("units", {}).items()},
                    samples=len(clock),
                    clock_start=float(clock[0]),
                    clock_end=float(clock[-1]),
                    median_step=float(np.median(steps)),
                    finite_fraction=float(np.isfinite(values).mean()),
                    nonzero_samples=int(
                        np.count_nonzero(np.isfinite(values) & (values != 0))
                    ),
                    array=str(target),
                    sensor_semantics_verified=False,
                )
            except Exception as error:  # noqa: BLE001 - record probe failures
                row.update(
                    fetched=False,
                    error_type=type(error).__name__,
                    error=str(error)[:500],
                )
                if authentication_error(error):
                    report.update(
                        stopped_on_auth_error=True,
                        status="candidate probe authentication failed; stopped",
                    )
                    report["probes"].append(row)
                    write_report(report)
                    print(
                        json.dumps(
                            {
                                "shot": shot,
                                "name": name,
                                "stopped": "authentication failure",
                            }
                        ),
                        flush=True,
                    )
                    return 3
            row["seconds"] = round(time.monotonic() - started, 2)
            report["probes"].append(row)
            write_report(report)
            print(json.dumps(row), flush=True)
            time.sleep(args.pace)
    report["status"] = (
        "candidate probe completed; corrected low-frequency RWM sensor semantics "
        "remain unverified; no new sensor or model input registered"
    )
    write_report(report)
    return 0


if __name__ == "__main__":
    # Match rwm_fetch.py: MDSplus may fault during Python interpreter teardown.
    # Reports are synchronously written and progress is flushed before exit.
    os._exit(main())
