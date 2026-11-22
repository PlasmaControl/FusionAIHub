"""Score the mean and persistence baselines and compare them with the TokaMark paper's numbers.

If this harness reproduces the published columns, its scores can be believed. Run through
`scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/check_harness.py --num-workers 64 \
        2>&1 | tee /models/intermediate/mast/stage0/check_harness.log

Each score goes to `<stage0>/scores/<task>/<predictor>/<split>/` (TokaMark's evaluator files in
a `<task>` sub-directory, our `score.json` beside it). A kept score (see below) is not
repeated unless `--force`; it is still listed in the table.

Three predictors are scored. `mean` is scored on the task's own windows. `persistence` is
scored on the windows of TokaMark's persistence script (`tasks.persistence_reference`: markovian,
only the signals that are both input and output, no actuators), which is how the published column
is made: in sparse test mode a window needs a finite input once there are no actuators, so these
are the windows that have a finite input. `persistence_all` is the same predictor on the task's own
windows (on those without a finite input it can only give the mean) and is never judged against a
published value. Measured 2026-11-22 for task_3-1: 0.5364 on 285,885 windows (`persistence`)
against 0.6868 on all 332,362 (`persistence_all`); published 0.536.

The published numbers are test-split numbers. With `--split test` the command exits 1 if any
row differs from them by more than 0.002 (the paper gives three decimals, so rounding alone
allows 0.0005; 0.002 leaves room for a small benchmark-version difference and no more). With
`--split val_report` there is no published value for those shots: the table shows our score
beside the published test value, the diff column reads "n/a" and the exit code is 0, unless a
score is not finite (that is outside on every split: exit 1).

A kept score must be the same score: `baselines.read_kept_score` accepts a stored `score.json`
only if it names this task, predictor and split and was written with the current scoring code
(sha256 of harness.py, tasks.py and baselines.py); otherwise the score is made again.

The limit of that rule: the hash covers our three files only. A score's windows, statistics,
splits, outlier file and metric come from TokaMark's source (`/models/tokamind/tokamark/src`) and
its data from `IGNITE_MAST_TOKAMARK`, and neither the TokaMark revision nor the data root is
recorded in `score.json` or compared. After a TokaMark update, or a move to another Zarr root, this
command still prints "kept" for every row and exits 0 on the old scores. Run it with `--force` after
either change. (Today the two agree: TokaMark's head is 146b0a0f8738, the revision in the cache
manifest of v1. Recording both in `score.json` is left for the next change of `baselines.py`.)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks

TOLERANCE = 0.002
PREDICTORS = ("mean", "persistence", "persistence_all")


def _predictor(name: str, task: tasks.TaskSpec):
    """(predict_fn, the task spec whose windows it is scored on)."""
    if name == "mean":
        return baselines.mean_predictor(task), task
    if name == "persistence":
        reference = tasks.persistence_reference(task)
        return baselines.persistence_predictor(reference), reference
    return baselines.persistence_predictor(task), task


def _rows_to_score(task_names: list[str]) -> list[tuple[str, str, float | None]]:
    """(task, predictor, published value or None) of every score to make.

    `persistence` and `persistence_all` exist for the tasks that have a published persistence
    value; `persistence_all` is never judged against it.
    """
    rows = []
    for task in task_names:
        persistence = tasks.PUBLISHED[task]["persistence"]
        rows.append((task, "mean", tasks.PUBLISHED[task]["mean"]))
        if persistence is not None:
            rows.append((task, "persistence", persistence))
            rows.append((task, "persistence_all", None))
    return rows


def _format_table(rows: list[dict], split: str) -> str:
    header = ("task", "predictor", "ours", "published", "diff")
    body = []
    for r in rows:
        judged = split == "test" and r["published"] is not None
        published = "n/a" if r["published"] is None else f"{r['published']:.3f}"
        diff = f"{r['ours'] - r['published']:+.4f}" if judged else "n/a"
        body.append((r["task"], r["predictor"], f"{r['ours']:.4f}", published, diff))
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
        stored, reason = (None, "--force") if args.force else baselines.read_kept_score(
            out_dir, task_name, predictor, args.split)
        if stored is not None:
            ours = float(stored["nrmse"])
            print(f"kept {task_name} {predictor} {args.split} "
                  f"(scored at {stored['git_revision']}, {stored['date']})", flush=True)
        else:
            if reason != "no score.json":
                print(f"re-scoring {task_name} {predictor} {args.split}: {reason}", flush=True)
            predict_fn, scored_task = _predictor(predictor, tasks.load_task(task_name))
            started = time.monotonic()
            result = harness.score(scored_task, predict_fn, out_dir, split=args.split,
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
    # No difference is judged off the test split, but a non-finite score is outside on every split.
    judged = args.split == "test"
    bad = [r for r in results
           if baselines.outside_tolerance(r["ours"], r["published"] if judged else None, TOLERANCE)]
    for r in bad:
        published = "n/a" if r["published"] is None else f"{r['published']:.3f}"
        print(f"OUTSIDE {TOLERANCE}: {r['task']} {r['predictor']} ours {r['ours']} published {published}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
