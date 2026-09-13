#!/usr/bin/env python
"""Write the `frame_codes/<shot>.json` provenance sidecars for caches encoded before they existed.

    python scripts/ideate/frame_codes_provenance.py --backfill --dry-run
    python scripts/ideate/frame_codes_provenance.py --backfill

The 500 production caches under `$IDEATE_DATA_ROOT/frame_codes` were written by `ideate encode`
before it wrote a sidecar. This reconstructs what the run manifests under `runs/encode/` still
say -- which task encoded which shots, on which device -- and writes one sidecar per cache,
marked `backfilled: true`. It writes NOTHING inside the `.pt` files and never overwrites a
sidecar a real encode wrote; see `ideate.design.provenance` for what is recoverable and what is
recorded as null rather than guessed (the manifests carry no commit and no thread count).

The behaviour lives in `ideate.design.provenance.backfill`, which is what the tests exercise;
this script is the shell entry point and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ideate import config
from ideate.design import provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--backfill",
        action="store_true",
        required=True,
        help="required: this script does one thing, and it is a production write",
    )
    ap.add_argument("--codes-dir", type=Path, default=None, help="default: <data_root>/frame_codes")
    ap.add_argument("--runs-dir", type=Path, default=None, help="default: <data_root>/runs/encode")
    ap.add_argument(
        "--default-device",
        default="cpu",
        help="what a cache no run manifest names is recorded as (marked 'assumed')",
    )
    ap.add_argument("--dry-run", action="store_true", help="report what would be written")
    ap.add_argument(
        "--force",
        action="store_true",
        help="rewrite sidecars that are themselves backfilled (never one an encode wrote)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    paths = config.load_paths()
    codes_dir = args.codes_dir or Path(paths.data_root) / "frame_codes"
    runs_dir = args.runs_dir or Path(paths.data_root) / "runs" / "encode"
    report = provenance.backfill(
        codes_dir=codes_dir,
        runs_dir=runs_dir,
        default_device=args.default_device,
        dry_run=args.dry_run,
        force=args.force,
        log=(lambda *_: None) if args.quiet else (lambda line: print(f"  {line}")),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["n_caches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
