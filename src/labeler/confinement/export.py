"""Export complementary H/L products from the one reconciled target table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..events.interval_tables import (
    grid_intervals,
    write_interval_table,
    write_label_grid,
)
from .labels import digest


def export(run_dir: Path, out: Path | None = None):
    run_dir = Path(run_dir).resolve()
    out = (run_dir / "format" if out is None else Path(out)).resolve()
    paths = Paths.from_env()
    for protected in (
        paths.corpus.resolve(),
        (paths.label_tables / "confinement/raw").resolve(),
    ):
        if out == protected or protected in out.parents:
            raise ValueError("Exports must be outside production corpus and raw inputs")
    targets_path = run_dir / "targets.csv"
    targets = pd.read_csv(targets_path)
    if not np.isclose(targets.t_end - targets.t_start, 50).all():
        raise ValueError("Export requires the workflow's 50 ms bins")
    if targets.duplicated(["shot", "t_start"]).any():
        raise ValueError("Target bins must be unique")
    result = {}
    for category, high in (
        ("high_confinement_mode", True),
        ("low_confinement_mode", False),
    ):
        folder = out / category
        (folder / "shots").mkdir(parents=True, exist_ok=True)
        tables = []
        names = {
            "0": "Absent within another explicitly annotated regime",
            "1": "Present",
        }
        for shot, rows in targets.groupby("shot", sort=True):
            stop = max(6000.0, float(rows.t_end.max()))
            times = np.arange(0.0, np.ceil(stop / 50) * 50, 50.0)
            labels = np.full(len(times), np.nan)
            known = rows.loc[rows.label >= 0]
            labels[(known.t_start / 50).astype(int)] = (
                known.label if high else 1 - known.label
            )
            write_label_grid(
                folder / "shots" / f"{int(shot)}.npz", times, labels, categories=names
            )
            tables.append(
                grid_intervals(int(shot), {"time_ms": times, "label": labels})
            )
        table = pd.concat(tables, ignore_index=True)
        target = folder / f"{category}_merged_v1.csv"
        metadata = {
            "category": category,
            "categories": names,
            "made_from": {"targets": str(targets_path), "sha256": digest(targets_path)},
            "source_provenance": str(run_dir / "labels.json"),
            "bin_ms": 50,
            "aggregation": "Full-bin coverage with one unambiguous binary regime",
            "unknown": "No source assessment, incomplete coverage, transition or H/L conflict",
            "H_family": ["H", "QH", "WP"],
            "confidence": "unknown",
            "relationship": "H and L are complementary only in known bins",
            "balancing": "CSV preserves all targets; class/shot training weights reside in targets.csv",
        }
        write_interval_table(table, target, metadata)
        result[category] = {
            "path": str(target),
            "sha256": digest(target),
            "rows": len(table),
        }
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Paths.from_env().root / "confinement/v1"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.run_dir, args.out), indent=2))


if __name__ == "__main__":
    main()
