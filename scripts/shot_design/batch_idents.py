#!/usr/bin/env python
"""Theme-stratified, append-only design-id list for simulate_batch.sbatch.

Reads the design batch ledger (designs.jsonl) and the prompts it came from
(prompts.jsonl, for each shot's theme), and appends every saved design id not yet
listed to --out, round-robin over themes so that however far a GPU job gets through
the file, the simulated set stays diverse. Existing lines are never reordered or
removed (the sbatch splits work by line index). With --follow it repeats every
--interval seconds until the design batch process is gone, then writes "<out>.final",
which tells FOLLOW=1 simulate jobs to stop re-reading.

    python scripts/shot_design/batch_idents.py --designs $ROOT/designs/designs.jsonl \
        --prompts $ROOT/prompts/prompts.jsonl --out $ROOT/runs/idents.txt --follow
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_new(designs: Path, prompts: Path, out: Path) -> int:
    theme = {r["shot"]: r.get("theme") or "none" for r in _rows(prompts)}
    listed = set(out.read_text().split()) if out.exists() else set()
    by_theme: dict[str, list[str]] = defaultdict(list)
    for r in _rows(designs):
        ident = r.get("design_id")
        if ident and ident not in listed:
            by_theme[theme.get(r["shot"], "none")].append(ident)
            listed.add(ident)
    order: list[str] = []
    queues = [by_theme[t] for t in sorted(by_theme)]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    if order:
        with out.open("a") as f:
            f.write("".join(f"{i}\n" for i in order))
    return len(order)


def _design_batch_running() -> bool:
    return (
        subprocess.run(
            ["pgrep", "-f", "python scripts/shot_design/batch_design"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--designs", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        added = append_new(args.designs, args.prompts, args.out)
        total = len(args.out.read_text().split()) if args.out.exists() else 0
        print(f"{time.strftime('%H:%M:%S')} +{added} -> {total} ids", flush=True)
        if not args.follow:
            return 0
        if not _design_batch_running():
            append_new(args.designs, args.prompts, args.out)
            Path(f"{args.out}.final").touch()
            print("design batch finished; wrote .final", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
