"""Score a headroom predictor on one task and split.

Run through `scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/run_headroom.py --task task_1-2 --predictor floor \
        --split val_report --num-workers 64

The score goes to `<stage0>/scores/<task>/<predictor>/<split>/` (TokaMark's evaluator files in a
`<task>` sub-directory, our `score.json` beside it), through `harness.score` and
`baselines.write_score_json`. `mean` and `persistence` are not predictors of this command:
`scripts/mast/check_harness.py` scores them, and a second command writing the same directories
would be a second source of truth for their definition.

Predictors: `floor` (TokaMind's output-representation floor, `headroom/floor.py`). `shrunk`,
`ridge` and `direct` are the fitted ones (`headroom.FITTED`); they are accepted by name, are
refused on `--split test` (exit status 2, before any data is opened: the test shots are read once,
for the final table, and only predictors with nothing fitted are scored there) and exit with
status 2 while they have no implementation. A later task adds one entry to `PREDICTORS`.

Provenance. `score.json`'s `code_sha256` covers the scoring code (harness.py, tasks.py,
baselines.py), not the predictor. This command therefore adds `predictor_sha256`, the sha256 of
the files that define the predictor as loaded, and keeps a stored score only if
`baselines.read_kept_score` keeps it and the two hashes agree; otherwise it says why and scores
again. `--force` always scores again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Callable

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED, floor


def _floor(task_name: str, task: tasks.TaskSpec) -> harness.PredictFn:
    return floor.floor_predictor(task, floor.tokamind_output_indices(task_name))


# name -> (builds the predict function from the task name and spec,
#          sha256 of the predictor's code as loaded). A later task adds one entry.
PREDICTORS: dict[str, tuple[Callable[[str, tasks.TaskSpec], harness.PredictFn], Callable[[], str]]] = {
    "floor": (_floor, lambda: floor.PREDICTOR_SHA256),
}


def _kept_score(out_dir, task: str, predictor: str, split: str, predictor_sha256: str, force: bool):
    """(stored record or None, reason it is not kept)."""
    if force:
        return None, "--force"
    stored, reason = baselines.read_kept_score(out_dir, task, predictor, split)
    if stored is None:
        return None, reason
    if stored.get("predictor_sha256") != predictor_sha256:
        return None, f"predictor code changed since {stored.get('git_revision', 'unknown')}"
    return stored, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", required=True, choices=list(tasks.ALL_TASKS))
    ap.add_argument("--predictor", required=True, choices=sorted({*PREDICTORS, *FITTED}))
    ap.add_argument("--split", required=True, choices=("val_report", "test"))
    ap.add_argument("--num-workers", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=None, help="accepted for the predictors that use a GPU; the floor does not")
    ap.add_argument("--force", action="store_true", help="score again even if a score is kept")
    args = ap.parse_args(argv)

    # Both refusals come before anything is opened.
    if args.split == "test" and args.predictor in FITTED:
        print(f"refused: {args.predictor} is fitted on data, and fitted predictors are scored on "
              "validation shots only (the test shots are read once, for the final table); "
              "use --split val_report", file=sys.stderr)
        return 2
    if args.predictor not in PREDICTORS:
        print(f"{args.predictor} is not implemented yet", file=sys.stderr)
        return 2

    build, code_sha256 = PREDICTORS[args.predictor]
    root = paths.prepare_write_root(paths.stage0_dir())
    out_dir = root / "scores" / args.task / args.predictor / args.split
    stored, reason = _kept_score(out_dir, args.task, args.predictor, args.split, code_sha256(), args.force)
    if stored is not None:
        print(f"kept {args.task} {args.predictor} {args.split} "
              f"(scored at {stored['git_revision']}, {stored['date']}): nrmse {float(stored['nrmse']):.4f}",
              flush=True)
        return 0 if math.isfinite(float(stored["nrmse"])) else 1
    if reason != "no score.json":
        print(f"re-scoring {args.task} {args.predictor} {args.split}: {reason}", flush=True)

    started = time.monotonic()
    task = tasks.load_task(args.task)
    result = harness.score(task, build(args.task, task), out_dir, split=args.split,
                           batch_size=args.batch_size, num_workers=args.num_workers)
    path = baselines.write_score_json(result, args.predictor, args.split)
    record = json.loads(path.read_text())
    record["predictor_sha256"] = code_sha256()
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.task} {args.predictor} {args.split}: nrmse {result.task_nrmse:.4f}, "
          f"{result.n_windows} windows, {time.monotonic() - started:.0f} s", flush=True)
    return 0 if math.isfinite(result.task_nrmse) else 1


if __name__ == "__main__":
    sys.exit(main())
