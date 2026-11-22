"""Score a headroom predictor on one task and split.

Run through `scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/run_headroom.py --task task_1-2 --predictor floor \
        --split val_report --num-workers 64

The score goes to `<stage0>/scores/<task>/<predictor>/<split>/` (TokaMark's evaluator files in a
`<task>` sub-directory, our `score.json` beside it), through `harness.score` and
`baselines.write_score_json`. `mean` and `persistence` are not predictors of this command:
`scripts/mast/check_harness.py` scores them, and a second command writing the same directories
would be a second source of truth for their definition.

`--table` (no other option) builds the headroom table (`headroom/table.py`) from the scores that are stored under
`<stage0>/scores`: it scores nothing and fits nothing, writes `<stage0>/headroom_table.csv` and prints the table as
Markdown, then the projected mean (not computed, with a count, while a stage-0 task has no direct score). Whether
a stored score is current is decided by `stale_reason`, which applies the rule of `_kept_score` that the command
uses before it scores again; a score that is not current is shown and named, not dropped.

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

`direct`. Needs `--gpu N`, the GPU's index as `nvidia-smi` numbers it. One `nvidia-smi` lookup gives the
GPU's uuid and its compute processes. The command sets `CUDA_VISIBLE_DEVICES` to that uuid in the form
`nvidia-smi` prints (`GPU-<uuid>`), not to the index, so that nvidia-smi's index and CUDA's enumeration order
need not agree, and the process then uses `cuda:0`. `torch` may already be imported (`harness.py`, which every
predictor imports, imports it at module level): CUDA reads the variable when it is initialized, and the command
refuses (exit status 2) if CUDA already is initialized in the process, so that `--gpu` could not take effect.
By default it refuses a GPU on which the lookup lists a compute process (naming the pids), before training
and before scoring; `--share-gpu` (direct only) uses such a GPU anyway and prints one line naming the GPU
and the pids it is shared with (nothing extra on a GPU without a compute process). Sharing changes how fast a
run goes, not what it computes, and nothing in `score.json` or the checkpoint records tells that a GPU was
shared. It always refuses an index or output the lookup cannot resolve. `--train` trains into `<stage0>/direct/<task>/<variant>/` (`best.pt`, `last.pt`,
`train_log.csv`, `config.json`; it refuses to overwrite an existing `best.pt` unless `--force`) and then
scores it on `--split val_report` (it first removes the stored score of that predictor name, so no number
outlives the checkpoint it was computed from while the retraining runs); without `--train` it scores the existing `best.pt` ("train first" if
there is none). `--variant NAME` (default `base`) selects another configuration: the base variant is
scored as predictor `direct` into `scores/<task>/direct/<split>/`, another as `direct.<variant>` into
`scores/<task>/direct.<variant>/<split>/`; both are fitted, so never scored on the test split. The
configuration options (`--max-steps --d-model --depth --heads --history-s --train-batch-size --lr
--val-every`) apply to `--train` only: scoring rebuilds the model from the `config.json` beside the
checkpoint, never from the command line. Its provenance key is `checkpoint_sha256`, the sha256 of the
`best.pt` the predictions come from. Its `score.json` also records, without comparing them when a score is
kept: `checkpoint_step` (the step of the `best.pt` scored), `trained_steps` (the step of `last.pt`, null if it
cannot be read) and `max_steps` (from the run's `config.json`); a run whose `trained_steps` is below
`max_steps`, or null, is scored with one warning line, since its number is for a run that did not finish.

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
import pickle
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED, cheap, floor
from tokamak_foundation_model.ignite.mast.headroom import table as headroom_table


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


# After CUDA_VISIBLE_DEVICES is set to the uuid of the one GPU asked for, that GPU is device 0 of the process.
DEVICE = "cuda:0"
BASE_VARIANT = "base"


def _direct_module():
    """`headroom.direct`, imported when a direct predictor is used and not at the top, so that the floor and
    the cheap predictors do not load it. This does not keep torch out of the process (`harness.py` imports
    it); what matters is that CUDA is not initialized before `main` has set CUDA_VISIBLE_DEVICES."""
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


def _train_direct(ctx: Context, score_dir: Path) -> None:
    direct = _direct_module()
    cfg = dataclasses.replace(direct.default_config(ctx.task), **ctx.overrides)
    run_dir = direct_run_dir(ctx.root, ctx.task_name, ctx.variant)
    # Whatever is stored for the predictor goes before the training starts: a score of a checkpoint that is
    # gone, or of the run this one replaces, must not be readable for the hours the training takes (or for
    # good if it dies).
    shutil.rmtree(score_dir, ignore_errors=True)
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


def gpu_state(index: int, gpus_csv: str, apps_csv: str) -> tuple[str, list[int]]:
    """(uuid, pids of the compute processes) of the GPU with this index, from one nvidia-smi lookup.

    `gpus_csv` is `nvidia-smi --query-gpu=index,uuid --format=csv,noheader`, `apps_csv` is
    `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`. The uuid is in the form nvidia-smi
    prints (`GPU-<uuid>`), which CUDA_VISIBLE_DEVICES accepts. Raises ValueError for output that does not
    parse or does not list the GPU, so that nothing is started on a guess.
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
    return uuid, pids


def busy_pids(index: int, gpus_csv: str, apps_csv: str) -> list[int]:
    """Pids of the compute processes on the GPU with this index (see `gpu_state`)."""
    return gpu_state(index, gpus_csv, apps_csv)[1]


def _cuda_initialized() -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and torch.cuda.is_initialized()


def _look_up_gpu(gpu: int) -> tuple[str, list[int]] | None:
    """(uuid, compute pids) of GPU `gpu`, or None after printing why the lookup failed."""
    try:
        return gpu_state(gpu, _nvidia_smi("--query-gpu=index,uuid", "--format=csv,noheader"),
                         _nvidia_smi("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"))
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        print(f"refused: cannot select GPU {gpu} or tell whether it is free (nvidia-smi: {e})", file=sys.stderr)
        return None


def _refuse_busy_gpu(gpu: int, pids: list[int], share: bool = False) -> bool:
    """Print why and return True if the GPU has a compute process and `share` (`--share-gpu`) is not given.

    With `share` the pids are named on stdout and the GPU is used (False); a GPU without a compute process
    prints nothing either way.
    """
    if not pids:
        return False
    listed = ", ".join(map(str, pids))
    if share:
        print(f"sharing GPU {gpu} with compute processes (pids {listed})", flush=True)
        return False
    print(f"refused: GPU {gpu} has compute processes (pids {listed}); use a GPU without one",
          file=sys.stderr)
    return True


def _direct_record(ctx: Context) -> dict:
    """What the checkpoint files say about the run behind a direct score (records, not provenance)."""
    import torch  # only reached for `direct`, after the GPU is selected
    run = direct_run_dir(ctx.root, ctx.task_name, ctx.variant)
    checkpoint_step = int(torch.load(run / "best.pt", map_location="cpu", weights_only=True)["step"])
    try:
        trained_steps = int(torch.load(run / "last.pt", map_location="cpu", weights_only=True)["step"])
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError):
        trained_steps = None
    max_steps = int(json.loads((run / "config.json").read_text())["cfg"]["max_steps"])
    return {"checkpoint_step": checkpoint_step, "trained_steps": trained_steps, "max_steps": max_steps}


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


def provenance_of(predictor: str, ctx: Context) -> dict[str, str]:
    """What a stored score of `predictor` must carry to be the score of the code and files in use now:
    `predictor_sha256` first, then the predictor's own keys. One definition for the command (before it keeps
    a stored score, and for the provenance it stores) and for `stale_reason` (the headroom table)."""
    _, code_sha256, extra_provenance = PREDICTORS[predictor]
    return {"predictor_sha256": code_sha256(), **extra_provenance(ctx)}


def stale_reason(root: Path, task_name: str, name: str, split: str) -> str:
    """"" if the stored score of predictor `name` (`direct.<variant>` included) is current, else why not.

    Scores nothing and fits nothing; reads the scoring code as loaded, the predictor's code, and the files its
    provenance names (a fit file, a checkpoint, TokaMind's run). It is the rule the command applies before it
    scores again (`_kept_score`), exposed for the headroom table, which shows a stale score and names it.
    `mean`, `persistence` and `persistence_all` are written by `check_harness.py` and have no predictor: for
    them the scoring-code rule `baselines.read_kept_score` is the whole rule. A provenance file that is not
    there (no checkpoint, no TokaMind run) is a reason, not an error.
    """
    out_dir = root / "scores" / task_name / name / split
    predictor, _, variant = name.partition(".")
    if predictor not in PREDICTORS:
        return baselines.read_kept_score(out_dir, task_name, name, split)[1]
    # No task spec: the provenance functions need the names and the root only, and loading the spec
    # needs TokaMark.
    ctx = Context(task_name, None, root, 0, 0, False, None, variant or BASE_VARIANT, False, {})
    try:
        provenance = provenance_of(predictor, ctx)
    except (OSError, ValueError, KeyError) as e:
        return f"provenance cannot be computed now ({type(e).__name__}: {e})"
    return _kept_score(out_dir, task_name, name, split, provenance, False)[1]


def print_table() -> int:
    """`--table`: build the headroom table from the stored scores, write `<stage0>/headroom_table.csv` and print
    it as Markdown with the projected mean. Scores nothing and writes nothing else."""
    root = paths.prepare_write_root(paths.stage0_dir())
    t = headroom_table.build_table(root / "scores", headroom_table.TOKAMIND_REPLICATION,
                                   stale_reason=lambda task, name, split: stale_reason(root, task, name, split))
    csv_path = root / "headroom_table.csv"
    t.to_csv(csv_path)
    print(f"Headroom table of {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, from the scores under {root / 'scores'}\n")
    print(headroom_table.to_markdown(t))
    variants = headroom_table.variants_markdown(root / "scores")
    if variants:
        print("Direct variants (the table takes the one with the lowest best val_nrmse among those with a stored "
              "score):\n")
        print(variants)
    n_direct = int(t["direct_val"].notna().sum())
    projected = headroom_table.projected_mean(t)
    if math.isnan(projected):
        print(f"projected mean: not computed, {len(t) - n_direct} of {len(t)} stage-0 tasks have no direct score")
    else:
        marked = int((t["stale"] != "").sum() + (t["unfinished"] != "").sum())
        print(f"projected mean over the 14 tasks (direct model on the {len(t)} stage-0 tasks, best published score on "
              f"the other {len(tasks.ALL_TASKS) - len(t)}): {projected:.4f}; the bar is 0.183, matching the best "
              f"published score everywhere would give 0.194"
              + (f"; {marked} row(s) carry stale or unfinished marks" if marked else ""))
    print(f"wrote {csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", action="store_true",
                    help="build the headroom table from the stored scores (nothing is scored), write "
                         "<stage0>/headroom_table.csv and print it as Markdown; takes no other option")
    ap.add_argument("--task", choices=list(tasks.ALL_TASKS))
    ap.add_argument("--predictor", choices=sorted({*PREDICTORS, *FITTED}))
    ap.add_argument("--split", choices=("val_report", "test"))
    ap.add_argument("--num-workers", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=None, help="`direct`: the GPU index (required); the others use none")
    ap.add_argument("--share-gpu", action="store_true",
                    help="`direct`: use the GPU although other compute processes are on it (their pids are "
                         "printed); without it a busy GPU is refused")
    ap.add_argument("--force", action="store_true", help="score again even if a score is kept (`direct --train`: "
                                                         "train again over an existing best.pt)")
    ap.add_argument("--variant", default=BASE_VARIANT, help="`direct`: the configuration, scored as direct.<variant>")
    ap.add_argument("--train", action="store_true", help="`direct`: train, then score on --split val_report")
    for option, kind in DIRECT_OPTIONS.items():
        ap.add_argument("--" + option.replace("_", "-"), type=kind, default=None, dest=option,
                        help="`direct --train`: override the DirectConfig field "
                             f"{_CONFIG_FIELD.get(option, option)}")
    args = ap.parse_args(argv)
    if args.table:
        if any(getattr(args, k) is not None for k in ("task", "predictor", "split")):
            ap.error("--table builds the table of all stage-0 tasks from the stored scores; "
                     "it takes no --task, --predictor or --split")
        return print_table()
    if args.task is None or args.predictor is None or args.split is None:
        ap.error("the following arguments are required: --task, --predictor, --split")
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
    if not is_direct and (args.variant != BASE_VARIANT or args.train or overrides or args.share_gpu):
        print("refused: --variant, --train, --share-gpu and the configuration options belong to "
              "--predictor direct",
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
        if _cuda_initialized():
            print("refused: CUDA is already initialized in this process, so --gpu cannot select the GPU",
                  file=sys.stderr)
            return 2
        looked_up = _look_up_gpu(args.gpu)  # one lookup serves the selection and the busy checks below
        if looked_up is None:
            return 2
        gpu_uuid, gpu_pids = looked_up
        # CUDA reads the variable when it initializes, which has not happened (checked above); torch itself
        # may be imported already. The uuid in nvidia-smi's own form, not the index: the two orders may differ.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    build = PREDICTORS[args.predictor][0]
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
            if _refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
                return 2
            gpu_checked = True
            _train_direct(ctx, root / "scores" / args.task / name / args.split)
    provenance = provenance_of(args.predictor, ctx)
    out_dir = root / "scores" / args.task / name / args.split
    stored, reason = _kept_score(out_dir, args.task, name, args.split, provenance, args.force)
    if stored is not None:
        print(f"kept {args.task} {name} {args.split} "
              f"(scored at {stored['git_revision']}, {stored['date']}): nrmse {float(stored['nrmse']):.4f}",
              flush=True)
        return 0 if math.isfinite(float(stored["nrmse"])) else 1
    if reason != "no score.json":
        print(f"re-scoring {args.task} {name} {args.split}: {reason}", flush=True)

    if is_direct and not gpu_checked and _refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
        return 2
    started = time.monotonic()
    predict_fn = build(ctx)  # a fitted predictor is fitted here if there is no current fit file
    run_record = _direct_record(ctx) if is_direct else {}
    if is_direct and (run_record["trained_steps"] is None or run_record["trained_steps"] < run_record["max_steps"]):
        trained = ("an unknown number of steps (no readable last.pt)" if run_record["trained_steps"] is None
                   else f"{run_record['trained_steps']} of {run_record['max_steps']} steps")
        print(f"warning: {name} was trained for {trained}, short of max_steps {run_record['max_steps']}; "
              f"the run did not finish, so this score is not final (checkpoint scored: step "
              f"{run_record['checkpoint_step']})", file=sys.stderr, flush=True)
    provenance = provenance_of(args.predictor, ctx)  # now with the fit it uses
    result = harness.score(task, predict_fn, out_dir, split=args.split, history_s=getattr(predict_fn, "history_s", None),
                           batch_size=args.batch_size, num_workers=args.num_workers)
    path = baselines.write_score_json(result, name, args.split)
    record = json.loads(path.read_text())
    record.update(provenance)
    record.update(run_record)  # records, not provenance: `_kept_score` does not compare them
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.task} {name} {args.split}: nrmse {result.task_nrmse:.4f}, "
          f"{result.n_windows} windows, {time.monotonic() - started:.0f} s", flush=True)
    return 0 if math.isfinite(result.task_nrmse) else 1


if __name__ == "__main__":
    sys.exit(main())
