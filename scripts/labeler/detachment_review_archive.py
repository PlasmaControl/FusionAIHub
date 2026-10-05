"""Archive superseded detachment JSON and browser folders without rebuilding."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from labeler.config import sha256_of

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    archive = args.out / "archive"
    record = {"archive": str(archive), "files": [], "browser_folders": []}
    old = REPO / "docs/labeler/results/archive"
    for source in sorted(old.rglob("*.json")):
        destination = archive / "results" / source.relative_to(old)
        digest = sha256_of(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_of(destination) != digest:
            raise ValueError(f"archive collision: {destination}")
        shutil.copy2(source, destination)
        if sha256_of(destination) != digest:
            raise ValueError(f"archive copy differs: {destination}")
        record["files"].append(
            {
                "source": str(source.relative_to(REPO)),
                "destination": str(destination),
                "sha256": digest,
                "bytes": source.stat().st_size,
            }
        )
        source.unlink()
    for source in sorted(args.out.glob("browser-*")):
        if source.name.startswith("browser-fix6-") or not source.is_dir():
            continue
        destination = archive / source.name
        if destination.exists():
            raise ValueError(f"browser archive already exists: {destination}")
        shutil.move(str(source), destination)
        record["browser_folders"].append(str(destination))
    record["file_count"] = len(record["files"])
    record["bytes_moved"] = sum(item["bytes"] for item in record["files"])
    record["store_rebuilds"] = 0
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "file_count",
                    "bytes_moved",
                    "store_rebuilds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
