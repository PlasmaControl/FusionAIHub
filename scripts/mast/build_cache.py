"""Build the standardized float32 cache of the 39 benchmark signals (one HDF5 file per shot).

Run through `scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/build_cache.py --workers 96            # all three splits
    IGNITE_MAST_ROOT=/models/intermediate/mast_trial \
        scripts/mast/py.sh scripts/mast/build_cache.py --limit 50 --workers 32    # a trial

The cache goes to `paths.cache_dir()` (`<root>/cache/v1`). The statistics are always drawn from
TokaMark's full training split (2,000 shots), also with `--limit`, so a trial's constants are the
real ones; `--limit N` only restricts the shots that are written to N per split, evenly spaced
over the sorted split so that a trial spans the campaigns. A rerun skips shots already written
from the same statistics. Exit status 1 if any shot failed (the manifest lists them).

The `__main__` guard is required: the shot-writing pool starts its workers with *spawn*, which
imports this file in every worker.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    from tokamak_foundation_model.ignite.mast import cache, paths, tasks

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="shots per split, evenly spaced (a trial)")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--force", action="store_true",
                    help="build over a manifest made from other statistics")
    args = ap.parse_args()

    root = paths.prepare_write_root(paths.cache_dir())
    train = tasks.split_shots("train")
    shots = {}
    for split in args.splits:
        lst = train if split == "train" else tasks.split_shots(split)
        shots[split] = cache.pick_evenly(lst, args.limit) if args.limit else lst
    manifest = cache.build(root, shots, workers=args.workers, force=args.force, stats_pool=train)

    n_files = sum(c["files"] for c in manifest["counts"].values())
    print(f"cache at {root}: {n_files} shot files, {manifest['size_bytes'] / 1e9:.2f} GB, "
          f"{manifest['wall_seconds']:.0f} s")
    if manifest["errors"]:
        print(f"{len(manifest['errors'])} shots failed (manifest.json, 'errors'); first:", file=sys.stderr)
        print(json.dumps(dict(list(manifest["errors"].items())[:5]), indent=1), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
