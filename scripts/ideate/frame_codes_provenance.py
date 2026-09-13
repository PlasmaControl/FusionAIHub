#!/usr/bin/env python
"""Write the `frame_codes/<shot>.json` provenance sidecars for caches encoded before they existed.

    python scripts/ideate/frame_codes_provenance.py --backfill --dry-run
    python scripts/ideate/frame_codes_provenance.py --backfill
    python scripts/ideate/frame_codes_provenance.py --audit        # read-only census

The 500 production caches under `$IDEATE_DATA_ROOT/frame_codes` were written by `ideate encode`
before it wrote a sidecar. `--backfill` reconstructs what the run manifests under `runs/encode/`
still say -- which task encoded which shots, on which device -- and writes one sidecar per cache,
marked `backfilled: true`. It writes NOTHING inside the `.pt` files and never overwrites a
sidecar a real encode wrote.

WHAT A BACKFILLED SIDECAR CONTAINS, EXACTLY. `device` and `device_source` (the run manifest that
names the shot, or "assumed cpu" for the caches whose OOM-killed task never wrote one),
`torch_threads` and `torch_threads_source` (the OMP_NUM_THREADS the sbatch exports -- the script,
not the run), `encoded_at` and `encoded_at_source` (the cache file's own mtime), `run_manifest`
where one exists, `schema`, `shot` and `backfilled: true`.

WHAT IT DOES NOT CONTAIN, AND WHY. No input fingerprint: `input_file` is null and
`input_fingerprint.kind` is `unknown` with `size_bytes`, `mtime_ns` and `sha256` all null,
because stat-ing the corpus file during the backfill would describe it as it is today, not as it
was at encode time. Null likewise: `torch_version`, `git_sha`, `ignite_bundle`,
`ignite_bundle_sha`, `ignite_revision`, `n_frames`, `modalities`, `include_video`. The run
manifests record none of them, and a provenance field holding a plausible wrong answer is worse
than one holding null. So "every cache has a provenance sidecar" is TRUE and
"every cache's encode is reproducible from its sidecar" is NOT: run `--audit` and read the
counts rather than taking either sentence on trust.

`--audit` is read-only -- it opens the JSON siblings, never the `.pt`, and writes nothing -- and
prints the census: caches, sidecars, missing sidecars, `input_fingerprint.kind`, device,
`backfilled`, and how many sidecars hold null in each field a reconstruction cannot fill.

The behaviour lives in `ideate.design.provenance.backfill` and `.audit`, which is what the tests
exercise; this script is the shell entry point and nothing else.
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
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--backfill",
        action="store_true",
        help="write the missing sidecars: this is a production write",
    )
    what.add_argument(
        "--audit",
        action="store_true",
        help="read-only: print the census of what the existing sidecars say",
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
    if args.audit:
        census = provenance.audit(codes_dir=codes_dir)
        print(json.dumps(census, indent=2))
        return 0 if census["n_caches"] else 1
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
