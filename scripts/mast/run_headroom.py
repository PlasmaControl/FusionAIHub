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
`ridge` (`headroom/cheap.py`: shrunk persistence and ridge regression on frame means); `direct` (`headroom/direct.py`:
a transformer trained per task). `shrunk`, `ridge` and `direct` are the fitted ones (`headroom.FITTED`); they
are refused on `--split test` (exit status 2, before any data is opened: the test shots are read once, for the
final table, and only predictors with nothing fitted are scored there). A predictor name without an
entry in `PREDICTORS` exits with status 2 ("not implemented yet").

Fits. `shrunk` and `ridge` are fitted by this command, on 3,000 training shots, and the fit is saved
as `<stage0>/cheap/<task>/<predictor>.npz`. A fit file is reused when its metadata (code hash,
training shot list, stride) matches the present code, otherwise the command fits again; `--force`
fits again and scores again. The score is tied to the file it was computed with by the extra
provenance key `fit_sha256` (the sha256 of the file's bytes).

`direct`. Needs `--gpu N`: the command sets `CUDA_VISIBLE_DEVICES=N` before `torch` is imported (it is
imported lazily, after the arguments are parsed) and the process then uses `cuda:0`. It refuses (exit
status 2, naming the pids) a GPU on which `nvidia-smi` lists a compute process, and a GPU it cannot read
`nvidia-smi` for. `--train` trains into `<stage0>/direct/<task>/<variant>/` (`best.pt`, `last.pt`,
`train_log.csv`, `config.json`; it refuses to overwrite an existing `best.pt` unless `--force`) and then
scores it on `--split val_report`; without `--train` it scores the existing `best.pt` ("train first" if
there is none). `--variant NAME` (default `base`) selects another configuration: the base variant is
scored as predictor `direct` into `scores/<task>/direct/<split>/`, another as `direct.<variant>` into
`scores/<task>/direct.<variant>/<split>/`; both are fitted, so never scored on the test split. The
configuration options (`--max-steps --d-model --depth --heads --history-s --train-batch-size --lr
--val-every`) apply to `--train` only: scoring rebuilds the model from the `config.json` beside the
checkpoint, never from the command line. Its provenance key is `checkpoint_sha256`, the sha256 of the
`best.pt` the predictions come from.

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
import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    gpu: int | None = None         # `direct`: the GPU index as nvidia-smi numbers it
    variant: str = "base"          # `direct`: which configuration of the model
    train: bool = False            # `direct`: train before scoring
    overrides: dict = field(default_factory=dict)   # `direct`: DirectConfig fields set on the command line


# After CUDA_VISIBLE_DEVICES is set to the one GPU asked for, that GPU is device 0 of the process.
DEVICE = "cuda:0"
BASE_VARIANT = "base"


def _direct_module():
    """`headroom.direct`, imported here and not at the top: it imports torch, which must not be imported
    before `main` has set CUDA_VISIBLE_DEVICES."""
    from tokamak_foundation_model.ignite.mast.headroom import direct
    return direct


def direct_run_dir(root: Path, task_name: str, variant: str) -> Path:
    return root / "direct" / task_name / variant


def predictor_name(predictor: str, variant: str) -> str:
    """`direct` for the base variant, `direct.<variant>` for another; the other predictors have no variants."""
    return predictor if variant == BASE_VARIANT else f"{predictor}.{variant}"


def _checkpoint(ctx: Context) -> Path:
    return direct_run_dir(ctx.root, ctx.task_name, ctx.variant) / "best.pt"


def _direct(ctx: Context) -> harness.PredictFn:
    return _direct_module().direct_predictor(ctx.task, _checkpoint(ctx), DEVICE)


def _direct_provenance(ctx: Context) -> dict[str, str]:
    return {"checkpoint_sha256": hashlib.sha256(_checkpoint(ctx).read_bytes()).hexdigest()}


def _train_direct(ctx: Context) -> None:
    direct = _direct_module()
    cfg = dataclasses.replace(direct.default_config(ctx.task), **ctx.overrides)
    run_dir = direct_run_dir(ctx.root, ctx.task_name, ctx.variant)
    if ctx.force:  # a refused overwrite was checked before; --force starts the run again
        for name in ("best.pt", "last.pt"):
            (run_dir / name).unlink(missing_ok=True)
    started = time.monotonic()
    direct.train_direct(ctx.task, cfg, run_dir, DEVICE, num_workers=ctx.num_workers)
    print(f"trained {ctx.task_name} {ctx.variant}: {cfg.max_steps} steps, {time.monotonic() - started:.0f} s",
          flush=True)


def _nvidia_smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True,
                          timeout=60).stdout


def busy_pids(index: int, gpus_csv: str, apps_csv: str) -> list[int]:
    """Pids of the compute processes on the GPU with this index.

    `gpus_csv` is `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`, `apps_csv` is
    `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`. Raises ValueError for
    output that does not parse or does not list the GPU, so that nothing is started on a guess.
    """
    def rows(text: str, what: str) -> list[list[str]]:
        out = []
        for line in text.splitlines():
            if line.strip():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2 or not parts[1]:
                    raise ValueError(f"cannot parse nvidia-smi {what} line {line!r}")
                out.append(parts)
        return out

    uuid = None
    for i, u in rows(gpus_csv, "GPU"):
        if not i.isdigit():
            raise ValueError(f"cannot parse nvidia-smi GPU line {i!r}, {u!r}")
        if int(i) == index:
            uuid = u
    if uuid is None:
        raise ValueError(f"nvidia-smi lists no GPU with index {index}")
    pids = []
    for u, pid in rows(apps_csv, "compute-app"):
        if not pid.isdigit():
            raise ValueError(f"cannot parse nvidia-smi compute-app pid {pid!r}")
        if u == uuid:
            pids.append(int(pid))
    return pids


def _refuse_busy_gpu(gpu: int) -> bool:
    """Print why and return True if the GPU has a compute process or its state cannot be read."""
    try:
        pids = busy_pids(gpu, _nvidia_smi("--query-gpu=index,uuid", "--format=csv,noheader"),
                         _nvidia_smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"))
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"refused: cannot tell whether GPU {gpu} is free (nvidia-smi: {e})", file=sys.stderr)
        return True
    if pids:
        print(f"refused: GPU {gpu} has compute processes (pids {', '.join(map(str, pids))}); "
              "use a GPU without one", file=sys.stderr)
        return True
    return False


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
    "direct": (_direct, lambda: _direct_module().PREDICTOR_SHA256, _direct_provenance),
}

# Command-line option -> DirectConfig field, for `--train`.
DIRECT_OPTIONS = {"max_steps": int, "d_model": int, "depth": int, "heads": int, "history_s": float,
                  "train_batch_size": int, "lr": float, "val_every": int}
_CONFIG_FIELD = {"train_batch_size": "batch_size"}


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
    ap.add_argument("--gpu", type=int, default=None, help="`direct`: the GPU index (required); the others use none")
    ap.add_argument("--force", action="store_true", help="score again even if a score is kept (`direct --train`: "
                                                         "train again over an existing best.pt)")
    ap.add_argument("--variant", default=BASE_VARIANT, help="`direct`: the configuration, scored as direct.<variant>")
    ap.add_argument("--train", action="store_true", help="`direct`: train, then score on --split val_report")
    for option, kind in DIRECT_OPTIONS.items():
        ap.add_argument("--" + option.replace("_", "-"), type=kind, default=None, dest=option,
                        help="`direct --train`: override the DirectConfig field "
                             f"{_CONFIG_FIELD.get(option, option)}")
    args = ap.parse_args(argv)
    is_direct = args.predictor == "direct"
    overrides = {_CONFIG_FIELD.get(k, k): getattr(args, k) for k in DIRECT_OPTIONS if getattr(args, k) is not None}

    # Both refusals come before anything is opened.
    if args.split == "test" and args.predictor in FITTED:
        print(f"refused: {args.predictor} is fitted on data, and fitted predictors are scored on "
              "validation shots only (the test shots are read once, for the final table); "
              "use --split val_report", file=sys.stderr)
        return 2
    if args.predictor not in PREDICTORS:
        print(f"{args.predictor} is not implemented yet", file=sys.stderr)
        return 2
    if not is_direct and (args.variant != BASE_VARIANT or args.train or overrides):
        print("refused: --variant, --train and the configuration options belong to --predictor direct",
              file=sys.stderr)
        return 2
    if is_direct:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.variant):
            print(f"refused: --variant {args.variant!r} is not a name (letters, digits, - and _)", file=sys.stderr)
            return 2
        if args.gpu is None:
            print("refused: --predictor direct needs --gpu N (a GPU without a compute process)", file=sys.stderr)
            return 2
        if overrides and not args.train:
            print("refused: the configuration options apply to --train; scoring rebuilds the model from "
                  "the config.json beside its checkpoint", file=sys.stderr)
            return 2
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            print("refused: CUDA is already initialized in this process, so --gpu cannot select the GPU",
                  file=sys.stderr)
            return 2
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # before torch is imported (by `_direct_module`)

    build, code_sha256, extra_provenance = PREDICTORS[args.predictor]
    name = predictor_name(args.predictor, args.variant)
    task = tasks.load_task(args.task)
    root = paths.prepare_write_root(paths.stage0_dir())
    ctx = Context(args.task, task, root, args.num_workers, args.batch_size, args.force,
                  args.gpu, args.variant, args.train, overrides)
    gpu_checked = False
    if is_direct:
        checkpoint = _checkpoint(ctx)
        if args.train and checkpoint.exists() and not args.force:
            print(f"refused: {checkpoint} exists; --force trains again over it (or choose another --variant)",
                  file=sys.stderr)
            return 2
        if not args.train and not checkpoint.exists():
            print(f"refused: there is no checkpoint at {checkpoint}: train first (--train)", file=sys.stderr)
            return 2
        if args.train:
            if _refuse_busy_gpu(args.gpu):
                return 2
            gpu_checked = True
            _train_direct(ctx)
    provenance = {"predictor_sha256": code_sha256(), **extra_provenance(ctx)}
    out_dir = root / "scores" / args.task / name / args.split
    stored, reason = _kept_score(out_dir, args.task, name, args.split, provenance, args.force)
    if stored is not None:
        print(f"kept {args.task} {name} {args.split} "
              f"(scored at {stored['git_revision']}, {stored['date']}): nrmse {float(stored['nrmse']):.4f}",
              flush=True)
        return 0 if math.isfinite(float(stored["nrmse"])) else 1
    if reason != "no score.json":
        print(f"re-scoring {args.task} {name} {args.split}: {reason}", flush=True)

    if is_direct and not gpu_checked and _refuse_busy_gpu(args.gpu):
        return 2
    started = time.monotonic()
    predict_fn = build(ctx)  # a fitted predictor is fitted here if there is no current fit file
    provenance = {"predictor_sha256": code_sha256(), **extra_provenance(ctx)}  # now with the fit it uses
    result = harness.score(task, predict_fn, out_dir, split=args.split, history_s=getattr(predict_fn, "history_s", None),
                           batch_size=args.batch_size, num_workers=args.num_workers)
    path = baselines.write_score_json(result, name, args.split)
    record = json.loads(path.read_text())
    record.update(provenance)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.task} {name} {args.split}: nrmse {result.task_nrmse:.4f}, "
          f"{result.n_windows} windows, {time.monotonic() - started:.0f} s", flush=True)
    return 0 if math.isfinite(result.task_nrmse) else 1


if __name__ == "__main__":
    sys.exit(main())
