"""Refresh the sawtooth rows of existing events files with the v3 detector.

    python -m labeler.events.refresh_sawtooth [--shots ...] [--out DIR]

The pipeline's sawtooth step moved from v2 (`heuristics.sawtooth_events`) to
v3 (`pipeline.sawtooth_block`) on 2026-09-30, and re-running the whole
`events` stage to pick that up would re-run TokEye on every shot. This runs
that one step instead, over `$LABELER_ROOT/events/<shot>_events.parquet` and
`<shot>_sources.parquet`:

- the `ece_sawtooth` events and the `ece_sawtooth` sources rows are
  replaced by v3's, through `schema.write_events` / `schema.write_sources`
  with `sources=["ece_sawtooth"]`, so v2's ECE row goes even where v3
  writes an SXR row beside it or cannot run at all;
- every other row is carried forward as it was, in the same order: both
  writers keep the rows they do not own and sort stably;
- each file is written atomically (`config.atomic_path`).

In place, EVERY file it will rewrite is first copied to `--backup` (default
`<root>/events-backup-sawtooth-v2-<today>`), all of them before the first
write, and a file already in the backup is never overwritten, so a second
run cannot replace v2's copy with v3's. With `--out DIR` the originals are
copied into DIR and refreshed there, and nothing under the root is written.

A shot with no corpus file is not refreshed (its v2 rows stay), because the
pipeline's own answer to a missing corpus file is `error`, not a skip. A
shot with an events file and no sources file is left alone and listed: the
pipeline writes the two together, so one alone is not a file this knows
how to patch.

`<root>/events_index.parquet` is not rewritten; `driver --rebuild-index`
derives it from the per-shot files.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import logging
import os
import shutil
import statistics
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import Paths
from . import coverage, heuristics, pipeline, schema

log = logging.getLogger(__name__)

#: What the backup directory is called, before the date.
BACKUP_PREFIX = "events-backup-sawtooth-v2-"


def listed(events_dir) -> tuple[list[int], dict[int, str]]:
    """The shots with both files, and the others with why they are left."""
    events_dir = Path(events_dir)
    have: dict[int, set[str]] = {}
    for path in events_dir.glob("*_*.parquet"):
        head, _, kind = path.stem.partition("_")
        if head.isdigit() and kind in ("events", "sources"):
            have.setdefault(int(head), set()).add(kind)
    both = sorted(s for s, kinds in have.items() if len(kinds) == 2)
    alone = {
        s: f"no {({'events', 'sources'} - kinds).pop()} file"
        for s, kinds in sorted(have.items())
        if len(kinds) == 1
    }
    return both, alone


def _files(events_dir: Path, shot: int) -> list[Path]:
    return [
        events_dir / f"{shot}_events.parquet",
        events_dir / f"{shot}_sources.parquet",
    ]


def copy_files(events_dir, shots: Sequence[int], dest, *, keep: bool) -> dict[str, int]:
    """Copy each shot's two files into `dest` and check the copies.

    `keep` leaves a file already in `dest` as it is - the backup's rule, so
    that the copy there is always the first one taken. Without it, `dest`
    must not hold them yet (the `--out` copy).
    """
    events_dir, dest = Path(events_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    counts = {"copied": 0, "kept": 0}
    for shot in shots:
        for src in _files(events_dir, shot):
            to = dest / src.name
            if to.exists():
                if not keep:
                    raise FileExistsError(f"{to} is already there")
                counts["kept"] += 1
                continue
            shutil.copy2(src, to)
            if not filecmp.cmp(src, to, shallow=False):
                raise OSError(f"{to} differs from {src} after the copy")
            counts["copied"] += 1
    return counts


def refresh_shot(shot: int, *, corpus_file, events_dir, run_id: str) -> dict[str, Any]:
    """Run v3 on one shot and replace its sawtooth rows; return the counts."""
    shot = int(shot)
    events_file, sources_file = _files(Path(events_dir), shot)
    before = schema.read_events(events_file, source=heuristics.SAWTOOTH_SOURCE)
    out: dict[str, Any] = {"shot": shot, "v2": len(before)}
    if not Path(corpus_file).exists():
        return out | {"error": f"no corpus file {corpus_file}"}
    started = time.monotonic()
    found, ran, skipped = pipeline.sawtooth_block(shot, corpus_file)
    found = coverage.attach_intervals(found, ran)
    owned = [heuristics.SAWTOOTH_SOURCE]
    schema.write_events(
        events_file, shot, found, run_id=run_id, merge=True, sources=owned
    )
    schema.write_sources(
        sources_file,
        shot,
        coverage.source_records(shot, ran=ran, skipped=skipped, events=found),
        run_id=run_id,
        merge=True,
        sources=owned,
    )
    by_diag: dict[str, int] = {}
    for event in found:
        by_diag[event.diag] = by_diag.get(event.diag, 0) + 1
    return out | {
        "v3": len(found),
        "diags": sorted(diag for _, diag, _, _ in ran),
        "by_diag": by_diag,
        "skipped": skipped,
        "elapsed_s": round(time.monotonic() - started, 2),
    }


def _one(job: tuple[int, str, str, str]) -> dict[str, Any]:
    shot, corpus_file, events_dir, run_id = job
    try:
        return refresh_shot(
            shot, corpus_file=corpus_file, events_dir=events_dir, run_id=run_id
        )
    except Exception as exc:  # noqa: BLE001 - one shot's failure is reported
        return {"shot": int(shot), "error": f"{type(exc).__name__}: {exc}"}


def summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The v2 -> v3 totals over the shots refreshed."""
    done = [r for r in rows if "v3" in r]
    v2 = [int(r["v2"]) for r in done]
    v3 = [int(r["v3"]) for r in done]
    diags: dict[str, int] = {}
    for r in done:
        for diag, n in r["by_diag"].items():
            diags[diag] = diags.get(diag, 0) + int(n)
    return {
        "shots": len(rows),
        "refreshed": len(done),
        "errors": {int(r["shot"]): r["error"] for r in rows if "error" in r},
        "v2_events": sum(v2),
        "v3_events": sum(v3),
        "v3_by_diag": diags,
        "v2_shots_with_any": sum(n > 0 for n in v2),
        "v3_shots_with_any": sum(n > 0 for n in v3),
        "v2_median": statistics.median(v2) if v2 else None,
        "v3_median": statistics.median(v3) if v3 else None,
        "sxr_skipped": sum("sawtooth sxr" in r["skipped"] for r in done),
        "sawtooth_skipped": sum("sawtooth" in r["skipped"] for r in done),
    }


def run(
    shots: Sequence[int],
    *,
    paths: Paths,
    events_dir,
    run_id: str,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Refresh `shots` in `events_dir`, printing one line per shot."""
    jobs = [
        (int(s), str(paths.corpus_file(int(s))), str(events_dir), run_id) for s in shots
    ]
    rows: list[dict[str, Any]] = []

    def report(row: dict[str, Any]) -> None:
        rows.append(row)
        if "error" in row:
            print(f"{row['shot']}: ERROR {row['error']}", flush=True)
            return
        skipped = f" skipped {row['skipped']}" if row["skipped"] else ""
        print(
            f"{row['shot']}: v2 {row['v2']} -> v3 {row['v3']} "
            f"{row['by_diag']} ran {row['diags']} {row['elapsed_s']} s{skipped}",
            flush=True,
        )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_one, jobs):
                report(row)
    else:
        for job in jobs:
            report(_one(job))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, nargs="+", help="only these shots")
    parser.add_argument(
        "--out",
        type=Path,
        help="copy the files here and refresh the copies, leaving the root's",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help=f"in place: where the originals go (default <root>/{BACKUP_PREFIX}"
        "<today>)",
    )
    parser.add_argument("--run-id", help="the rows' run_id (default: stamped)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    started = time.monotonic()
    paths = Paths.from_env()
    both, alone = listed(paths.events)
    if args.shots:
        wanted = set(args.shots)
        unknown = sorted(wanted - set(both))
        if unknown:
            parser.error(f"no events and sources files for {unknown}")
        both = [s for s in both if s in wanted]
        alone = {}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"sawtooth-v3-{stamp}-{os.getpid()}"
    print(
        f"refresh_sawtooth: {len(both)} shots under {paths.events}, "
        f"corpus {paths.corpus}, run_id {run_id}, {args.workers} workers",
        flush=True,
    )
    for shot, why in alone.items():
        print(f"{shot}: left alone, {why}", flush=True)

    if args.out is not None:
        events_dir = args.out
        copied = copy_files(paths.events, both, events_dir, keep=False)
        print(f"copied {copied['copied']} files to {events_dir}", flush=True)
    else:
        events_dir = paths.events
        today = datetime.now(UTC).astimezone().date()  # the local day
        backup = args.backup or paths.root / f"{BACKUP_PREFIX}{today}"
        copied = copy_files(paths.events, both, backup, keep=True)
        print(
            f"backed up {copied['copied']} files to {backup} "
            f"({copied['kept']} already there, kept)",
            flush=True,
        )

    rows = run(
        both,
        paths=paths,
        events_dir=events_dir,
        run_id=run_id,
        workers=args.workers,
    )
    totals = summary(rows) | {
        "left_alone": alone,
        "events_dir": str(events_dir),
        "run_id": run_id,
        "elapsed_s": round(time.monotonic() - started, 1),
    }
    print(json.dumps(totals))
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
