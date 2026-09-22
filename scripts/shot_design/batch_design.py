#!/usr/bin/env python
"""Run `shot_design assistant` over a table of prompts, several at a time, and record the ids.

The design harness is prompt-driven and one design per invocation, so a batch is a loop
over prompts; this keeps the loop resumable (a prompt whose row already carries a design id
is skipped), concurrent (the LLM round trips dominate and the retrieval is light), and
honest about failures (the assistant's stderr tail is kept beside the row, never a design
id it did not print).

    SHOT_DESIGN_DATA_ROOT=<batch root> python scripts/shot_design/batch_design.py \
        --prompts <root>/prompts/prompts.jsonl --out <root>/designs/designs.jsonl \
        --provider agy --workers 6 [--limit N]

prompts.jsonl rows: {"shot": int, "theme": str, "prompt": str, ...}; the output rows copy
the input row and add design_id / error / elapsed_s / trace path. Runs the interpreter it
was started with (never `pixi run`, whose activation would re-point the data root).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _one(row: dict, args, trace_dir: Path) -> dict:
    trace = trace_dir / f"{row['shot']}.trace.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "shot_design",
        "assistant",
        "--prompt",
        row["prompt"],
        "--provider",
        args.provider,
        "--trace",
        str(trace),
    ]
    if not args.no_anchor:
        cmd += ["--ref-shot", str(row["shot"])]
    if args.llm_model:
        cmd += ["--llm-model", args.llm_model]
    t0 = time.time()
    out = dict(row)
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, check=False
        )
        last = (proc.stdout.strip().splitlines() or [""])[-1].strip()
        if (
            proc.returncode == 0
            and len(last) == 32
            and all(c in "0123456789abcdef" for c in last)
        ):
            out["design_id"] = last
            out["error"] = None
        else:
            out["design_id"] = None
            tail = (
                proc.stderr.strip().splitlines()
                or proc.stdout.strip().splitlines()
                or [""]
            )
            out["error"] = f"rc={proc.returncode}: " + " | ".join(tail[-3:])[:600]
    except subprocess.TimeoutExpired:
        out["design_id"] = None
        out["error"] = f"timeout after {args.timeout}s"
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["trace"] = str(trace)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--provider", default="agy")
    ap.add_argument(
        "--llm-model",
        default=None,
        help="passed through as `assistant --llm-model` (one tag for every alias)",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N pending prompts only"
    )
    ap.add_argument("--timeout", type=int, default=900, help="seconds per design")
    ap.add_argument(
        "--retry-failed", action="store_true", help="re-run rows that errored"
    )
    ap.add_argument(
        "--no-anchor",
        action="store_true",
        help="do not pass the row's shot as `assistant --ref-shot` (text retrieval only)",
    )
    args = ap.parse_args()

    root = os.environ.get("SHOT_DESIGN_DATA_ROOT", "")
    if not root or root.rstrip("/") == "/scratch/gpfs/EKOLEMEN/nc1514/ideate":
        print(
            "SHOT_DESIGN_DATA_ROOT must name a batch root, not production",
            file=sys.stderr,
        )
        return 2
    prompts = _load_jsonl(args.prompts)
    done = {r["shot"]: r for r in _load_jsonl(args.out)}
    pending = [
        r
        for r in prompts
        if r["shot"] not in done
        or (args.retry_failed and done[r["shot"]].get("design_id") is None)
    ]
    if args.limit:
        pending = pending[: args.limit]
    print(
        f"{len(prompts)} prompts, {len(done)} already recorded, {len(pending)} to run "
        f"with {args.workers} workers via provider {args.provider}",
        flush=True,
    )
    trace_dir = args.out.parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = [
        r
        for r in done.values()
        if not (args.retry_failed and r.get("design_id") is None)
    ]
    ok = fail = 0
    t0 = time.time()
    with args.out.open("w") as f, ThreadPoolExecutor(args.workers) as pool:
        for r in kept:
            f.write(json.dumps(r) + "\n")
        f.flush()
        futs = {pool.submit(_one, r, args, trace_dir): r for r in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            f.write(json.dumps(res) + "\n")
            f.flush()
            if res["design_id"]:
                ok += 1
            else:
                fail += 1
            if i % 10 == 0 or res["design_id"] is None:
                print(
                    f"[{i}/{len(pending)}] shot {res['shot']} -> "
                    f"{res['design_id'] or 'FAILED: ' + str(res['error'])[:160]} "
                    f"({res['elapsed_s']} s; {ok} ok / {fail} failed; "
                    f"{(time.time() - t0) / 60:.1f} min)",
                    flush=True,
                )
    print(f"done: {ok} designs, {fail} failures in {(time.time() - t0) / 60:.1f} min")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
