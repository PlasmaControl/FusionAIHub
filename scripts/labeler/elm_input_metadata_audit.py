#!/usr/bin/env python
"""Audit retained ELM ordinate metadata offline, without inferring physical units.

Optionally write provenance sidecars beside existing prepared NPY inputs without
rewriting their numerical contents. No fetching or source-store mutation occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from labeler.config import Paths, sha256_of
from labeler.elm import inputs, labels, prepare


def _attribute(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, h5py.Empty):
        return None
    return value.tolist() if hasattr(value, "tolist") else value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write-sidecars", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/labeler/elm/density_units.json"),
    )
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    shots = sorted(
        map(int, labels.review_table(prepare.review_csv(paths)).shot.unique())
    )
    records = {}
    for shot in shots:
        source = prepare.signals_dir(paths) / f"{shot}.npz"
        metadata = inputs.read_metadata(source)
        target = prepare.inputs_dir(paths) / f"{shot}.npy"
        before = sha256_of(target)
        if args.write_sidecars:
            target.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        after = sha256_of(target)
        assert before == after, f"numerical inputs changed for shot {shot}"
        records[str(shot)] = {
            **metadata,
            "prepared_record": {
                "path": str(target),
                "sha256_before": before,
                "sha256_after": after,
                "numerical_bytes_unchanged": before == after,
            },
        }
    inspected = []
    for shot in (185808, 190643):
        for suffix in ("fast", "slow"):
            path = Path("/scratch/gpfs/EKOLEMEN/hackathon/raw_h5_files") / (
                f"{shot}_{suffix}.h5"
            )
            record = {"path": str(path), "exists": path.exists()}
            if path.exists():
                with h5py.File(path) as f:
                    record["fast_chord_groups"] = [
                        n for n in f if "denv2f" in n.lower() or "denv3f" in n.lower()
                    ]
                    if "co2_density_slow" in f:
                        group = f["co2_density_slow"]
                        record["separate_slow_diagnostic_metadata"] = {
                            "group": {k: _attribute(v) for k, v in group.attrs.items()},
                            "datasets": {
                                n: {k: _attribute(v) for k, v in obj.attrs.items()}
                                for n, obj in group.items()
                            },
                            "fast_ordinate_unit_evidence": False,
                        }
            inspected.append(record)
    missing = [
        int(s)
        for s, r in records.items()
        if any(unit is None for unit in r["interferometer"]["ordinate_units"])
    ]
    result = {
        "producer": "scripts/labeler/elm_input_metadata_audit.py",
        "source_sha256": sha256_of(Path(__file__)),
        "scope": "offline source metadata only; no network calls",
        "reviewed_shots": shots,
        "n_shots": len(shots),
        "shots_without_fast_ordinate_units": missing,
        "decision": (
            "Fast physical ordinate units remain unverified. Remove the unsupported "
            "m^-2 declaration; retain the input divisor 1e14 in native ordinate "
            "units, with no numerical rescaling. Slow CO2 is a separate record "
            "and cannot establish the fast-channel units."
        ),
        "fetch_metadata_loss": {
            "fetch_script": "scripts/labeler/elmo_fetch.py",
            "legacy_cache_fields": [
                "t_int_ms",
                "interferometer",
                "t_fs_ms",
                "filterscopes",
            ],
            "shared_fetch_wrapper": "src/labeler/events/verify.py:fdp_signal",
            "wrapper_metadata": "time units retained; source ordinate units absent",
        },
        "inspected_staged_sources": inspected,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shots": len(shots),
                "unverified": len(missing),
                "numerical_bytes_unchanged": True,
                "record": str(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
