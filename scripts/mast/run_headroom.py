"""Score a headroom predictor on one task and split.

Run through `scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/run_headroom.py --task task_1-2 --predictor floor \
        --split val_report --num-workers 64

The score goes to `<stage0>/scores/<task>/<predictor>/<split>/` (TokaMark's evaluator files in a
`<task>` sub-directory, our `score.json` beside it), through `harness.score` and
`baselines.write_score_json`. `mean` and `persistence` are not predictors of this command:
`scripts/mast/check_harness.py` scores them, and a second command writing the same directories
would be a second source of truth for their definition.

Predictors: `floor` (TokaMind's output-representation floor, `headroom/floor.py`); `shrunk` and
`ridge` (`headroom/cheap.py`: shrunk persistence and ridge regression on frame means). `shrunk`,
`ridge` and `direct` are the fitted ones (`headroom.FITTED`); they are accepted by name, are
refused on `--split test` (exit status 2, before any data is opened: the test shots are read once,
for the final table, and only predictors with nothing fitted are scored there) and exit with
status 2 while they have no implementation (`direct`). A later task adds one entry to `PREDICTORS`.

Fits. `shrunk` and `ridge` are fitted by this command, on 3,000 training shots, and the fit is saved
as `<stage0>/cheap/<task>/<predictor>.npz`. A fit file is reused when its metadata (code hash,
training shot list, stride) matches the present code, otherwise the command fits again; `--force`
fits again and scores again. The score is tied to the file it was computed with by the extra
provenance key `fit_sha256` (the sha256 of the file's bytes).

Provenance. `score.json`'s `code_sha256` covers the scoring code (harness.py, tasks.py,
baselines.py), not the predictor. This command therefore adds `predictor_sha256`, the sha256 of
the files that define the predictor as loaded, and keeps a stored score only if
`baselines.read_kept_score` keeps it and the two hashes agree; otherwise it says why and scores
again. A predictor may declare more provenance keys, a function of the task that returns
`{name: hex digest}` (what else the score depends on, such as TokaMind's coefficient indices or a
checkpoint): they are stored beside `predictor_sha256`, and a stored score is kept only if every
one matches. `--force` always scores again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED, cheap, floor


@dataclass(frozen=True)
class Context:
    """What a predictor's builder and provenance function may need: the task and how to run."""
    task_name: str
    task: tasks.TaskSpec
    root: Path             # the stage-0 write root
    num_workers: int
    batch_size: int
    force: bool


def _floor(ctx: Context) -> harness.PredictFn:
    return floor.floor_predictor(ctx.task, floor.tokamind_output_indices(ctx.task_name),
                                 floor.tokamind_nan_imputation(ctx.task_name))


def _floor_provenance(ctx: Context) -> dict[str, str]:
    return {"tokamind_representation_sha256": floor.representation_sha256(
        floor.tokamind_output_indices(ctx.task_name), floor.tokamind_nan_imputation(ctx.task_name))}


def _cheap(name: str) -> Callable[[Context], harness.PredictFn]:
    def build(ctx: Context) -> harness.PredictFn:
        return cheap.obtain(name, ctx.task, ctx.root, num_workers=ctx.num_workers,
                            batch_size=ctx.batch_size, force=ctx.force).predict
    return build


def _cheap_provenance(name: str) -> Callable[[Context], dict[str, str]]:
    def provenance(ctx: Context) -> dict[str, str]:
        # The digest of the fit file that is on disk and reusable now; `cheap.NO_CURRENT_FIT` (which no
        # stored score carries) if there is none, so that a score is not kept without its fit.
        return {"fit_sha256": cheap.current_fit_digest(cheap.fit_path(ctx.root, ctx.task_name, name),
                                                       name, ctx.task_name)}
    return provenance


# name -> (builds the predict function from the context,
#          sha256 of the predictor's code as loaded,
#          extra provenance keys {name: digest} of the task). A later task adds one entry.
PREDICTORS: dict[str, tuple[Callable[[Context], harness.PredictFn],
                            Callable[[], str],
                            Callable[[Context], dict[str, str]]]] = {
    "floor": (_floor, lambda: floor.PREDICTOR_SHA256, _floor_provenance),
    "shrunk": (_cheap("shrunk"), lambda: cheap.PREDICTOR_SHA256, _cheap_provenance("shrunk")),
    "ridge": (_cheap("ridge"), lambda: cheap.PREDICTOR_SHA256, _cheap_provenance("ridge")),
}


def _kept_score(out_dir, task: str, predictor: str, split: str, provenance: dict[str, str], force: bool):
    """(stored record or None, reason it is not kept). `provenance` maps every key the stored score
    must carry (`predictor_sha256` first) to its current value."""
    if force:
        return None, "--force"
    stored, reason = baselines.read_kept_score(out_dir, task, predictor, split)
    if stored is None:
        return None, reason
    for key, value in provenance.items():
        if stored.get(key) != value:
            what = "predictor code" if key == "predictor_sha256" else key
            return None, f"{what} changed since {stored.get('git_revision', 'unknown')}"
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

    build, code_sha256, extra_provenance = PREDICTORS[args.predictor]
    task = tasks.load_task(args.task)
    root = paths.prepare_write_root(paths.stage0_dir())
    ctx = Context(args.task, task, root, args.num_workers, args.batch_size, args.force)
    provenance = {"predictor_sha256": code_sha256(), **extra_provenance(ctx)}
    out_dir = root / "scores" / args.task / args.predictor / args.split
    stored, reason = _kept_score(out_dir, args.task, args.predictor, args.split, provenance, args.force)
    if stored is not None:
        print(f"kept {args.task} {args.predictor} {args.split} "
              f"(scored at {stored['git_revision']}, {stored['date']}): nrmse {float(stored['nrmse']):.4f}",
              flush=True)
        return 0 if math.isfinite(float(stored["nrmse"])) else 1
    if reason != "no score.json":
        print(f"re-scoring {args.task} {args.predictor} {args.split}: {reason}", flush=True)

    started = time.monotonic()
    predict_fn = build(ctx)  # a fitted predictor is fitted here if there is no current fit file
    provenance = {"predictor_sha256": code_sha256(), **extra_provenance(ctx)}  # now with the fit it uses
    result = harness.score(task, predict_fn, out_dir, split=args.split,
                           batch_size=args.batch_size, num_workers=args.num_workers)
    path = baselines.write_score_json(result, args.predictor, args.split)
    record = json.loads(path.read_text())
    record.update(provenance)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.task} {args.predictor} {args.split}: nrmse {result.task_nrmse:.4f}, "
          f"{result.n_windows} windows, {time.monotonic() - started:.0f} s", flush=True)
    return 0 if math.isfinite(result.task_nrmse) else 1


if __name__ == "__main__":
    sys.exit(main())
