"""Score the mean and persistence baselines and compare them with the TokaMark paper's numbers.

If this harness reproduces the published columns, its scores can be believed. Run through
`scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/check_harness.py --num-workers 64 \
        2>&1 | tee /models/intermediate/mast/stage0/check_harness.log

Each score goes to `<stage0>/scores/<task>/<predictor>/<split>/` (TokaMark's evaluator files in
a `<task>` sub-directory, our `score.json` beside it). A score whose `score.json` exists is not
repeated unless `--force`; it is still listed in the table.

The published numbers are test-split numbers. With `--split test` the command exits 1 if any
row differs from them by more than 0.002 (the paper gives three decimals, so rounding alone
allows 0.0005; 0.002 leaves room for a small benchmark-version difference and no more). With
`--split val_report` there is no published value for those shots: the table shows our score
beside the published test value, the diff column reads "n/a" and the exit code is 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks

TOLERANCE = 0.002
PREDICTORS = ("mean", "persistence")


def _predictor(name: str, task: tasks.TaskSpec):
    return {"mean": baselines.mean_predictor, "persistence": baselines.persistence_predictor}[name](task)


def _rows_to_score(task_names: list[str]) -> list[tuple[str, str, float]]:
    """(task, predictor, published value) for every score that has a published value to compare with."""
    rows = []
    for task in task_names:
        for predictor in PREDICTORS:
            published = tasks.PUBLISHED[task][predictor]
            if published is not None:
                rows.append((task, predictor, published))
    return rows


def _format_table(rows: list[dict], split: str) -> str:
    header = ("task", "predictor", "ours", "published", "diff")
    body = []
    for r in rows:
        diff = "n/a" if split != "test" else f"{r['ours'] - r['published']:+.4f}"
        body.append((r["task"], r["predictor"], f"{r['ours']:.4f}", f"{r['published']:.3f}", diff))
    widths = [max(len(str(line[i])) for line in [header, *body]) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(line, widths)).rstrip() for line in [header, *body]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", nargs="+", default=list(tasks.ALL_TASKS), choices=list(tasks.ALL_TASKS),
                    metavar="TASK", help="default: all 14 tasks")
    ap.add_argument("--split", default="test", choices=("test", "val_report"))
    ap.add_argument("--num-workers", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="score again even if score.json exists")
    args = ap.parse_args(argv)

    root = paths.prepare_write_root(paths.stage0_dir())
    results = []
    for task_name, predictor, published in _rows_to_score(args.tasks):
        out_dir = root / "scores" / task_name / predictor / args.split
        score_json = out_dir / "score.json"
        if score_json.exists() and not args.force:
            ours = float(json.loads(score_json.read_text())["nrmse"])
            print(f"{task_name} {predictor} {args.split}: kept {score_json}", flush=True)
        else:
            task = tasks.load_task(task_name)
            started = time.monotonic()
            result = harness.score(task, _predictor(predictor, task), out_dir, split=args.split,
                                   batch_size=args.batch_size, num_workers=args.num_workers)
            baselines.write_score_json(result, predictor, args.split)
            ours = result.task_nrmse
            print(f"{task_name} {predictor} {args.split}: {result.n_windows} windows, "
                  f"{time.monotonic() - started:.0f} s", flush=True)
        results.append({"task": task_name, "predictor": predictor, "ours": ours, "published": published})

    print()
    print(_format_table(results, args.split))
    if args.split != "test":
        print("\nThe published values are test-split values; there is none for these shots, so no diff.")
        return 0
    bad = [r for r in results if abs(r["ours"] - r["published"]) > TOLERANCE]
    for r in bad:
        print(f"OUTSIDE {TOLERANCE}: {r['task']} {r['predictor']} ours {r['ours']:.4f} "
              f"published {r['published']:.3f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
