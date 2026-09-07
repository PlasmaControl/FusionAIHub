#!/usr/bin/env python
"""Fetch `ip` through fdp for shots whose labelmaker feature file has none, and merge it in.

WHY THIS EXISTS. `ideate corpus select`'s rule (d) needs a measured Ip flat-top for every shot of
`recommender_v1`, and it reads it out of `$LABELMAKER_ROOT/features/<shot>_features.h5`, group
`ip`. `python -m labelmaker.run features --models ...` writes only the features the named models
consume, and none of the four models in that run consumes `ip`: the 352-shot features job left
`ip` in 19 of its 352 files (those came from the archive resolver, which writes what the archive
carries). So the flat-top verification could not finalize the list.

This script closes that gap and nothing else: one feature name, merged into whatever the file
already holds.

WHERE IT RUNS. The LOGIN node, not SLURM. The compute nodes have no route to DIII-D's PTDATA
server, so this is not a batch job and there are no jobstats for it. And it must run under the
`fdp run` wrapper, which supplies the server configuration:

    pixi run -e labelmaker fdp run python scripts/labelmaker/fetch_ip_features.py \
        --shot-file $LABELMAKER_ROOT/recommender_v1_pending_features.txt --workers 4 --retries 5

Without the wrapper every PTDATA fetch fails with `PtDataError` and the process reports
`getservbyname failed for task 'PTSERVER'` -- see `labelmaker.features.store.TRANSIENT_CAUSES`,
where that trap is recorded, and `resolve_fdp`'s module docstring.

HOW IT IS SAFE. `store.write_features(..., merge=True)` keeps every group already in the file and
drops `ip` from the recorded misses when it resolves; it writes through a temporary file and
renames, so a file is never left half-written. A shot whose fetch fails keeps everything it had,
and the cause is recorded in the file's `missing` attribute -- but only if the file already
exists: this script never creates a feature file that would hold nothing but a miss, because
`preferred_shots` and the census both read the mere existence of one as "this shot has features".

The worker pool is forked before toksearch is imported anywhere (its ptserver reader is not
fork-safe), which is why every labelmaker import in `_fetch` is inside the function -- the same
rule `labelmaker.run` follows.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

FEATURE = "ip"

# Set once in each worker by `_init`, because `Pool.imap_unordered` passes one argument and the
# alternative -- a tuple per shot -- would repeat the same two values 352 times.
_FEATURES_DIR: Path | None = None
_RETRIES = 1


def read_shots(path: Path) -> list[int]:
    """One shot per line, `#` starting a comment. The pending file's own format."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        head = line.split("#", 1)[0].strip()
        if head:
            out.append(int(head))
    return out


def has_ip(path: Path) -> bool:
    """True when the file holds an `ip` group a consumer could measure a flat-top from.

    The test is the CONSUMER's, not the store's: `ideate.shotdb.select.measured_flattop` wants
    `ip/xdata` and `ip/ydata` with something in them, so a group that exists but carries one
    sample -- the corpus' "signal absent" sentinel -- is not an `ip` for this purpose.
    """
    if not path.exists():
        return False
    import h5py

    try:
        with h5py.File(path, "r", locking=False) as f:
            if FEATURE not in f:
                return False
            g = f[FEATURE]
            return "xdata" in g and "ydata" in g and g["ydata"].shape[-1] >= 2
    except OSError:
        return False


def _init(features_dir: str, retries: int) -> None:
    global _FEATURES_DIR, _RETRIES
    _FEATURES_DIR = Path(features_dir)
    _RETRIES = retries


def _fetch(shot: int) -> dict:
    """One shot: fetch `ip` and merge it in. Returns a row for the caller to print and count.

    Every labelmaker import is deferred into this function so the parent process never imports
    toksearch (its ptserver reader is not fork-safe) and the pool is forked clean.
    """
    from labelmaker.features import resolve_fdp, store

    path = _FEATURES_DIR / f"{shot}_features.h5"
    t0 = time.monotonic()
    if has_ip(path):
        return {"shot": shot, "status": "present", "cause": "", "secs": 0.0, "n": 0}
    try:
        arrays, missing = resolve_fdp.resolve(shot, [FEATURE], retries=_RETRIES)
    except Exception as exc:  # noqa: BLE001 - one shot's failure is not the run's
        return {
            "shot": shot,
            "status": "failed",
            "cause": f"{type(exc).__name__}: {exc}"[:120],
            "secs": time.monotonic() - t0,
            "n": 0,
        }
    secs = time.monotonic() - t0
    if FEATURE not in arrays:
        cause = missing.get(FEATURE, "NoArrayNoCause")
        if path.exists():
            # The miss is recorded in the file it belongs to, so a later run can see what was
            # tried. A file that does not exist is left alone: see the module docstring.
            try:
                store.write_features(path, shot, {}, {FEATURE: cause}, merge=True)
            except Exception as exc:  # noqa: BLE001
                cause = f"{cause} (+ {type(exc).__name__} recording it)"
        return {"shot": shot, "status": "failed", "cause": cause, "secs": secs, "n": 0}
    try:
        store.write_features(path, shot, {FEATURE: arrays[FEATURE]}, {}, merge=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "shot": shot,
            "status": "failed",
            "cause": f"write: {type(exc).__name__}: {exc}"[:120],
            "secs": secs,
            "n": 0,
        }
    return {
        "shot": shot,
        "status": "fetched",
        "cause": "",
        "secs": secs,
        "n": int(arrays[FEATURE].y.shape[-1]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shot-file", required=True, help="one shot per line")
    ap.add_argument("--features-dir", help="default: $LABELMAKER_ROOT/features")
    ap.add_argument("--workers", type=int, default=4, help="network-bound; 4 is polite")
    ap.add_argument("--retries", type=int, default=5, help="per-signal retries inside resolve()")
    ap.add_argument("--limit", type=int, help="only the first N shots (a smoke test)")
    args = ap.parse_args(argv)

    root = os.environ.get("LABELMAKER_ROOT")
    features_dir = Path(args.features_dir) if args.features_dir else (
        Path(root) / "features" if root else None
    )
    if features_dir is None:
        print("no --features-dir and no $LABELMAKER_ROOT", file=sys.stderr)
        return 2
    if not features_dir.is_dir():
        print(f"no feature store at {features_dir}", file=sys.stderr)
        return 2

    shots = read_shots(Path(args.shot_file))
    if args.limit:
        shots = shots[: args.limit]
    # Checked in the parent, before the fork: a shot that already has `ip` costs no worker and no
    # network call, and the printed count then separates "was already there" from "fetched now".
    todo = [s for s in shots if not has_ip(features_dir / f"{s}_features.h5")]
    already = len(shots) - len(todo)
    print(
        f"{len(shots)} shot(s) from {args.shot_file}: {already} already have {FEATURE}, "
        f"{len(todo)} to fetch, {args.workers} worker(s), {args.retries} retries",
        flush=True,
    )

    rows: list[dict] = []
    t0 = time.monotonic()
    if todo:
        with Pool(
            args.workers, initializer=_init, initargs=(str(features_dir), args.retries)
        ) as pool:
            for i, row in enumerate(pool.imap_unordered(_fetch, todo), start=1):
                rows.append(row)
                extra = f"{row['n']:,} samples" if row["status"] == "fetched" else row["cause"]
                print(
                    f"[{i:>4}/{len(todo)}] {row['shot']} {row['status']:<8}"
                    f" {row['secs']:5.1f}s  {extra}",
                    flush=True,
                )

    counts = Counter(r["status"] for r in rows)
    causes = Counter(r["cause"] for r in rows if r["status"] == "failed")
    print(
        f"\n{len(shots)} shot(s) in {time.monotonic() - t0:.0f}s: "
        f"{counts.get('fetched', 0)} fetched, {already + counts.get('present', 0)} already present,"
        f" {counts.get('failed', 0)} failed"
    )
    for cause, n in causes.most_common():
        print(f"  {n:>4}  {cause}")
    failed = [r["shot"] for r in rows if r["status"] == "failed"]
    if failed:
        print("failed shots: " + " ".join(str(s) for s in sorted(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
