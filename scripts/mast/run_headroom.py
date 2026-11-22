"""Score a headroom predictor on one task and split.

Run through `scripts/mast/py.sh`:

    scripts/mast/py.sh scripts/mast/run_headroom.py --task task_1-2 --predictor floor \
        --split val_report --num-workers 64

The score goes to `<stage0>/scores/<task>/<predictor>/<split>/` (TokaMark's evaluator files in a
`<task>` sub-directory, our `score.json` beside it), through `harness.score` and
`baselines.write_score_json`. `mean` and `persistence` are not predictors of this command:
`scripts/mast/check_harness.py` scores them, and a second command writing the same directories
would be a second source of truth for their definition.

`--table` (no other option: every other option is refused) builds the headroom table (`headroom/table.py`) from
the scores that are stored under `<stage0>/scores`: it scores nothing and fits nothing, writes
`<stage0>/headroom_table.csv` and prints the table as Markdown, the direct variants listing with the same marks, then
the projected mean (not computed, with a count, while a stage-0 task has no direct score; else with the number of
rows whose direct score is stale, unfinished or not competitive). Whether a stored score is current is decided by
`stale_reason`, which applies the rule of `_kept_score` that the command uses before it scores again, plus, for a
direct score, a mark when the checkpoint was trained by other code than the code now loaded (see `direct`); a score
that is not current is shown and named, not dropped.

Predictors: `floor` (TokaMind's output-representation floor, `headroom/floor.py`); `shrunk` and
`ridge` (`headroom/cheap.py`: shrunk persistence and ridge regression on frame means); `direct` (`headroom/direct.py`:
a transformer trained per task). `shrunk`, `ridge` and `direct` are the fitted ones (`headroom.FITTED`); they
are refused on `--split test` (exit status 2, before any data is opened: the test shots are read once, for the
final table, and only predictors with nothing fitted are scored there). A predictor name without an
entry in `PREDICTORS` exits with status 2 ("not implemented yet").

Fits. `shrunk` and `ridge` are fitted by this command, on 3,000 training shots, and the fit is saved
as `<stage0>/cheap/<task>/<predictor>.npz`. A fit file is reused when its metadata (code hash,
training shot list, stride) matches the present code, otherwise the command fits again (a file this code
cannot read counts as no fit); `--force` fits again and scores again. The score is tied to the file it was
computed with by the extra provenance key `fit_sha256`: the sha256 of the bytes `cheap.obtain` loaded or
wrote, not of a second read of the path; the command refuses to store the "no current fit" sentinel as a
digest. `score.json` also records, without comparing them when a score is kept: `fit_n_windows` and
`fit_n_channels_without_target` (both fits), and for ridge `fit_lambda` and `fit_lambda_at_edge` ("lower" or
"upper" when the chosen penalty is the smallest or the largest candidate, null otherwise).

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
shared. It always refuses an index or output the lookup cannot resolve.

`direct --train`. Trains into `<stage0>/direct/<task>/<variant>/` (`best.pt`, `last.pt`, `train_log.csv`,
`config.json`, `trained_by.json`; it refuses to overwrite an existing `best.pt` unless `--force`) and then scores
it on `--split val_report`. `--force` means "train again" with `--train` and "score again" without it. Before
anything is deleted the configuration is validated (the model is built on the CPU; sizes, steps and the learning
rate must be usable) and a run that is being trained is refused: `train.lock` in the run directory holds the pid
of the training, a second launch is refused naming the pid, and a lock whose pid is not running is taken over
with one printed line (a lock with no readable pid is not; a reused pid needs the lock removed by hand). The lock
is removed when the training ends or fails, and is not a checkpoint (`best.pt` is what the checks look at). Then
the stored score of that predictor name is removed, so that no number outlives the checkpoint it was computed from
while the retraining runs; a removal that fails stops the command, because the old number would stay readable.
`trained_by.json` (exactly the keys `predictor_sha256` and `code_sha256`: `direct.py` as loaded and the scoring
code) is written before the first step, so a stopped training carries it, and replaced by a forced retraining.
`--history-s` is refused for a markovian task (it changes nothing there) and for a value with
`round(history_s / dt) < 1` for any input or actuator signal, naming the signal (the harness would deliver an
all-NaN last sample), both before anything is written.

`direct` scoring. Without `--train` it scores the existing `best.pt` ("train first" if there is none).
`--variant NAME` (default `base`) selects another configuration: the base variant is scored as predictor `direct`
into `scores/<task>/direct/<split>/`, another as `direct.<variant>` into `scores/<task>/direct.<variant>/<split>/`;
both are fitted, so never scored on the test split. The configuration options (`--max-steps --d-model --depth
--heads --history-s --train-batch-size --lr --val-every`) apply to `--train` only: scoring rebuilds the model
from the `config.json` beside the checkpoint, never from the command line. Its provenance key is
`checkpoint_sha256`, the sha256 of the `best.pt` the predictions come from; the command refuses (exit status 1) if
that hash differs between the check that decides whether to keep a score and the one after the predictor was
built (a concurrent writer). Its `score.json` also records, without comparing them when a score is kept:
`checkpoint_step` (the step of the `best.pt` scored), `trained_steps` (the step of `last.pt`, null if it cannot be
read), `max_steps` (from the run's `config.json`), and `trained_predictor_sha256` and `trained_code_sha256` (from
`trained_by.json`, null if there is no readable one). A run whose `trained_steps` is below `max_steps`, or null,
is scored with one warning line, since its number is for a run that did not finish. These five keys are
refreshed from the run's files whenever a stored direct score is kept, so that a score taken in the middle of a
training does not keep its step counts after the training finished. The two trained-by hashes are a mark and not
part of the rule that keeps a score (scoring again cannot cure a training by other code): the table marks the
cell "trained by other code than the code now loaded", or "training code not recorded" (which is also what a
run trained before `trained_by.json` existed shows until a sidecar is written for it).

Limits of what "current" means. The direct predictor's hash covers `direct.py` only. It imports the 5 ms frame
rule (`FRAME_S`, `frame_len`) from `cheap.py`, which the hash does not cover; a test pins both values. A score is
kept on the hash of our scoring code (harness.py, tasks.py, baselines.py), the predictor's hash and its own
provenance, so it is not marked when TokaMark's code (windows, statistics, splits, the metric) or the Zarr data
root changed: the record holds neither.

Blind scoring. The fitted predictors (`shrunk`, `ridge`, `direct`) are called with a copy of the batch whose
targets are NaN arrays of the same shapes and dtypes (`_blind_to_targets`; the wrapper keeps the predict
function's attributes such as `history_s`), so that "does not read its targets" is structural. The harness still
scores against the real targets, and the floor, which reads its targets by definition, is not wrapped.

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

import numpy as np

from tokamak_foundation_model.ignite.mast import baselines, harness, paths, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED, cheap, floor
from tokamak_foundation_model.ignite.mast.headroom import table as headroom_table
# One definition of these three, in the table (which has to find the same directories and names).
from tokamak_foundation_model.ignite.mast.headroom.table import BASE_VARIANT, direct_run_dir, predictor_name


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
    # What a builder learned that the provenance and the record must use: `fit_sha256` and `fit_record` of a
    # cheap predictor (the digest of the bytes the fit was made from, read once by `cheap.obtain`).
    built: dict = field(default_factory=dict, compare=False)


# After CUDA_VISIBLE_DEVICES is set to the uuid of the one GPU asked for, that GPU is device 0 of the process.
DEVICE = "cuda:0"


def _direct_module():
    """`headroom.direct`, imported when a direct predictor is used and not at the top, so that the floor and
    the cheap predictors do not load it. This does not keep torch out of the process (`harness.py` imports
    it); what matters is that CUDA is not initialized before `main` has set CUDA_VISIBLE_DEVICES."""
    from tokamak_foundation_model.ignite.mast.headroom import direct
    return direct


def _checkpoint(ctx: Context) -> Path:
    return direct_run_dir(ctx.root, ctx.task_name, ctx.variant) / "best.pt"


def _direct(ctx: Context) -> harness.PredictFn:
    return _direct_module().direct_predictor(ctx.task, _checkpoint(ctx), DEVICE)


def _direct_provenance(ctx: Context) -> dict[str, str]:
    return {"checkpoint_sha256": hashlib.sha256(_checkpoint(ctx).read_bytes()).hexdigest()}


class Refused(Exception):
    """A refusal that is not an error of the code: `main` prints it and exits with status 2."""


TRAINED_BY = "trained_by.json"      # in a run directory: the code that trained it (written before the first step)
LOCK = "train.lock"                 # in a run directory: the pid of the training that holds it


def _write_json_atomic(path: Path, record: dict) -> None:
    """Write `record` as JSON to `path` through a temporary name in the same directory, then replace: a reader
    sees the old file or the new one, never half of one."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_trained_by(run_dir: Path) -> dict | None:
    """The sidecar of a run (exactly the two hashes, both strings), or None if there is no readable one."""
    try:
        record = json.loads((run_dir / TRAINED_BY).read_text())
    except (OSError, ValueError):
        return None
    if (isinstance(record, dict) and set(record) == {"predictor_sha256", "code_sha256"}
            and all(isinstance(v, str) for v in record.values())):
        return record
    return None


def _validate_direct_config(direct, task: tasks.TaskSpec, cfg) -> None:
    """Raise `Refused` for a configuration the training would fail on, before anything is deleted. The model
    is built (on the CPU, then dropped), which is the check `train_direct` itself makes first; the rest are
    values for which a run could not produce a `best.pt`."""
    for name in ("max_steps", "val_every", "batch_size", "d_model", "heads"):
        if getattr(cfg, name) < 1:
            raise Refused(f"invalid configuration: {name} is {getattr(cfg, name)}, it must be at least 1")
    if not (math.isfinite(cfg.lr) and cfg.lr > 0):
        raise Refused(f"invalid configuration: lr is {cfg.lr}, it must be finite and positive")
    try:
        direct.DirectModel(task, cfg)
    except ValueError as e:
        raise Refused(f"invalid configuration: {e}") from None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:      # the process exists and belongs to somebody else
        return True
    return True


def _take_lock(run_dir: Path) -> Path:
    """Create `<run dir>/train.lock` holding this pid, or raise `Refused`. A lock whose pid is not alive is
    taken over with one printed line; one with no readable pid is not (nothing says it is dead). The lock is
    a file of its own: it is not a checkpoint, and `best.pt` is what the command's checks look at. A pid can
    be reused by another process after a crash; then the lock must be removed by hand."""
    lock = run_dir / LOCK
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                owner = int(lock.read_text().strip())
            except (OSError, ValueError):
                raise Refused(f"{lock} exists and holds no readable pid; if no training of this run is going, "
                              "remove it by hand") from None
            if _pid_alive(owner):
                raise Refused(f"another training of this run is going (pid {owner}, {lock}); wait for it, or "
                              "train another --variant")
            print(f"taking over {lock}: its pid {owner} is not running", flush=True)
            lock.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
        return lock
    raise Refused(f"could not take {lock}: another process took it first")


def _release_lock(lock: Path) -> None:
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def _train_direct(ctx: Context, score_dir: Path) -> None:
    """Train the variant: validate the configuration, take the run's lock, remove what is stored for the
    predictor (and with `--force` the checkpoints), write `trained_by.json`, train. Raises `Refused` before
    anything is deleted if the configuration is invalid, the run is locked, or a removal fails."""
    direct = _direct_module()
    cfg = dataclasses.replace(direct.default_config(ctx.task), **ctx.overrides)
    _validate_direct_config(direct, ctx.task, cfg)
    run_dir = direct_run_dir(ctx.root, ctx.task_name, ctx.variant)
    created = not run_dir.exists()
    paths.prepare_write_root(run_dir)
    lock = _take_lock(run_dir)
    try:
        # Whatever is stored for the predictor goes before the training starts: a score of a checkpoint that is
        # gone, or of the run this one replaces, must not be readable for the hours the training takes (or for
        # good if it dies). A removal that fails stops the command: the old score would stay readable.
        try:
            shutil.rmtree(score_dir)
        except FileNotFoundError:
            pass
        except OSError as e:
            raise Refused(f"cannot remove the stored score {score_dir} ({e}); the old number would stay readable "
                          "during the training") from None
        if ctx.force:  # a refused overwrite was checked before; --force starts the run again
            for name in ("best.pt", "last.pt"):
                try:
                    (run_dir / name).unlink(missing_ok=True)
                except OSError as e:
                    raise Refused(f"cannot remove {run_dir / name} ({e})") from None
        # The code that trains, recorded before the first step so that a stopped training carries it too; a
        # forced retraining replaces it. `_direct_record` copies it into score.json.
        _write_json_atomic(run_dir / TRAINED_BY, {"predictor_sha256": direct.PREDICTOR_SHA256,
                                                  "code_sha256": baselines.scoring_code_sha256()})
        started = time.monotonic()
        direct.train_direct(ctx.task, cfg, run_dir, DEVICE, num_workers=ctx.num_workers)
        print(f"trained {ctx.task_name} {ctx.variant}: {cfg.max_steps} steps, {time.monotonic() - started:.0f} s",
              flush=True)
    finally:
        _release_lock(lock)
        if created:
            try:
                run_dir.rmdir()  # nothing came of this run: leave no empty variant behind
            except OSError:
                pass


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
    trained_by = _read_trained_by(run) or {}
    return {"checkpoint_step": checkpoint_step, "trained_steps": trained_steps, "max_steps": max_steps,
            "trained_predictor_sha256": trained_by.get("predictor_sha256"),
            "trained_code_sha256": trained_by.get("code_sha256")}


def _floor(ctx: Context) -> harness.PredictFn:
    return floor.floor_predictor(ctx.task, floor.tokamind_output_indices(ctx.task_name),
                                 floor.tokamind_nan_imputation(ctx.task_name))


def _blind_to_targets(predict: harness.PredictFn) -> harness.PredictFn:
    """`predict` as a fitted predictor must be scored: it gets a copy of the batch whose targets are NaN arrays of
    the same shapes and dtypes. "Does not read its targets" is then structural, not a property each predictor has
    to keep. The harness still scores against the real targets (the wrapper changes only what the predictor sees),
    and the attributes the command reads from a predict function (`history_s`, ...) are kept. Not for the floor,
    which reads its targets by definition."""
    def blind(batch: harness.WindowBatch) -> dict:
        nan = {name: np.full(a.shape, np.nan, dtype=a.dtype if a.dtype.kind == "f" else np.float32)
               for name, a in batch.targets.items()}
        return predict(dataclasses.replace(batch, targets=nan))

    blind.__dict__.update(getattr(predict, "__dict__", {}))
    return blind


def _floor_provenance(ctx: Context) -> dict[str, str]:
    return {"tokamind_representation_sha256": floor.representation_sha256(
        floor.tokamind_output_indices(ctx.task_name), floor.tokamind_nan_imputation(ctx.task_name))}


def _cheap(name: str) -> Callable[[Context], harness.PredictFn]:
    def build(ctx: Context) -> harness.PredictFn:
        predictor, digest = cheap.obtain(name, ctx.task, ctx.root, num_workers=ctx.num_workers,
                                         batch_size=ctx.batch_size, force=ctx.force)
        ctx.built.update(fit_sha256=digest, fit_record=predictor.fit_record())
        return predictor.predict
    return build


def _cheap_provenance(name: str) -> Callable[[Context], dict[str, str]]:
    def provenance(ctx: Context) -> dict[str, str]:
        # After the predictor was built: the digest of the bytes it was made from (`cheap.obtain` reads them
        # once). Before: the digest of the fit file on disk if it is reusable now, `cheap.NO_CURRENT_FIT` (which no
        # stored score carries) if there is none, so that a score is not kept without its fit.
        if "fit_sha256" in ctx.built:
            return {"fit_sha256": ctx.built["fit_sha256"]}
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


def _trained_by_reason(record: dict, loaded_predictor_sha256: str) -> str:
    """The mark for a direct score whose checkpoint was trained by other code than the code now loaded (the
    predictor's file or the scoring code), or by code that was not recorded. A mark only: it is not part of
    `_kept_score`, because scoring again cannot cure a training by other code."""
    trained = (record.get("trained_predictor_sha256"), record.get("trained_code_sha256"))
    if None in trained:
        return "training code not recorded"
    if trained != (loaded_predictor_sha256, baselines.scoring_code_sha256()):
        return "trained by other code than the code now loaded"
    return ""


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
    # No task spec: the provenance functions read the names, the root and the files they name (the floor's
    # provenance loads the task spec, the cheap one reads the split file), and loading a spec needs TokaMark.
    # Any error of one cell marks that cell; it does not stop the table.
    ctx = Context(task_name=task_name, task=None, root=root, num_workers=0, batch_size=0, force=False,
                  variant=variant or BASE_VARIANT)
    try:
        provenance = provenance_of(predictor, ctx)
    except Exception as e:  # noqa: BLE001
        return f"provenance cannot be computed now ({type(e).__name__}: {e})"
    stored, reason = _kept_score(out_dir, task_name, name, split, provenance, False)
    if stored is not None and predictor == "direct":
        return _trained_by_reason(stored, provenance["predictor_sha256"])
    return reason


def _direct_marked_rows(t) -> int:
    """Rows whose direct score carries a stale or unfinished mark, or is not competitive: a row is counted once,
    and a mark on another cell (the mean, the fits) is not the direct score's."""
    def on_direct(column):
        return t[column].map(lambda v: any(e.startswith("direct_val:") for e in str(v).split(" | ")))

    return int((on_direct("stale") | on_direct("unfinished") | (t["not_competitive"] == "yes")).sum())


def print_table() -> int:
    """`--table`: build the headroom table from the stored scores, write `<stage0>/headroom_table.csv` and print
    it as Markdown with the projected mean. Scores nothing and writes nothing else."""
    root = paths.prepare_write_root(paths.stage0_dir())

    def stale(task: str, name: str, split: str) -> str:
        return stale_reason(root, task, name, split)

    t = headroom_table.build_table(root / "scores", headroom_table.TOKAMIND_REPLICATION, stale_reason=stale)
    csv_path = root / "headroom_table.csv"
    t.to_csv(csv_path)
    print(f"Headroom table of {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, from the scores under {root / 'scores'}\n")
    print(headroom_table.to_markdown(t))
    variants = headroom_table.variants_markdown(root / "scores", stale_reason=stale)
    if variants:
        print("Direct variants (the table takes the one with the lowest best val_nrmse among those with a stored "
              "score):\n")
        print(variants)
    n_direct = int(t["direct_val"].notna().sum())
    projected = headroom_table.projected_mean(t)
    if math.isnan(projected):
        print(f"projected mean: not computed, {len(t) - n_direct} of {len(t)} stage-0 tasks have no direct score")
    else:
        marked = _direct_marked_rows(t)
        print(f"projected mean over the 14 tasks (direct model on the {len(t)} stage-0 tasks, best published score on "
              f"the other {len(tasks.ALL_TASKS) - len(t)}): {projected:.4f}; the bar is 0.183, matching the best "
              f"published score everywhere would give 0.194"
              + (f"; {marked} row(s) have a direct score that is stale, unfinished or not competitive (see the "
                 "marks above)" if marked else ""))
    print(f"wrote {csv_path}")
    return 0


def _history_problem(task: tasks.TaskSpec, history_s: float) -> str | None:
    """Why `--history-s` cannot apply to `task`, or None. A markovian task has TokaMark's own windows
    (`tasks.TaskSpec.n_in`, `n_act`), so the option would change nothing but the name of the run. Otherwise
    `round(history_s / dt)` must be at least 1 for every input and actuator signal: with 0 samples the harness's
    `_last_finite` returns all NaN even when the history has finite samples (`harness.py`)."""
    if task.markovian:
        return (f"--history-s has no effect on the markovian task {task.name} (its windows are TokaMark's own), so "
                "a run with it would be identical to the base run")
    short = [f"{s.name} (dt {s.dt:g} s)" for s in (*task.inputs, *task.actuators)
             if not history_s > 0 or round(history_s / s.dt) < 1]       # not > 0 also catches nan
    if short:
        return (f"--history-s {history_s:g} is not at least half a sample (round(history_s / dt) < 1, or not a "
                f"positive time) for: {', '.join(short)}")
    return None


def _refresh_kept_record(ctx: Context, path: Path, stored: dict) -> None:
    """A kept direct score keeps its number, but what the files say about the run has moved on since it was
    written (a score taken in the middle of the training, or before `trained_by.json`): rewrite the record keys
    of `_direct_record` from the files, atomically, and say so. The run may be unreadable now; the score is
    kept anyway."""
    try:
        fresh = _direct_record(ctx)
    except Exception as e:  # noqa: BLE001
        print(f"warning: cannot refresh the record keys of the kept score ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)
        return
    changed = [k for k, v in fresh.items() if stored.get(k) != v]
    if changed:
        _write_json_atomic(path, {**stored, **fresh})
        print(f"refreshed {', '.join(changed)} of the kept score", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", action="store_true",
                    help="build the headroom table from the stored scores (nothing is scored), write "
                         "<stage0>/headroom_table.csv and print it as Markdown; takes no other option")
    ap.add_argument("--task", choices=list(tasks.ALL_TASKS))
    ap.add_argument("--predictor", choices=sorted({*PREDICTORS, *FITTED}))
    ap.add_argument("--split", choices=("val_report", "test"))
    ap.add_argument("--num-workers", type=int, default=None, help="loader workers (default 64)")
    ap.add_argument("--batch-size", type=int, default=None, help="scoring batch size (default 64)")
    ap.add_argument("--gpu", type=int, default=None, help="`direct`: the GPU index (required); the others use none")
    ap.add_argument("--share-gpu", action="store_true",
                    help="`direct`: use the GPU although other compute processes are on it (their pids are "
                         "printed); without it a busy GPU is refused")
    ap.add_argument("--force", action="store_true",
                    help="without --train: score again even if a score is kept. With `direct --train`: train "
                         "again over an existing best.pt (and score the new run)")
    ap.add_argument("--variant", default=None,
                    help=f"`direct`: the configuration, scored as direct.<variant> (default {BASE_VARIANT})")
    ap.add_argument("--train", action="store_true", help="`direct`: train, then score on --split val_report")
    for option, kind in DIRECT_OPTIONS.items():
        ap.add_argument("--" + option.replace("_", "-"), type=kind, default=None, dest=option,
                        help="`direct --train`: override the DirectConfig field "
                             f"{_CONFIG_FIELD.get(option, option)}")
    args = ap.parse_args(argv)
    if args.table:
        given = [f"--{k.replace('_', '-')}" for k in ("task", "predictor", "split", "num_workers", "batch_size", "gpu",
                                                      "variant", *DIRECT_OPTIONS) if getattr(args, k) is not None]
        given += [f"--{k.replace('_', '-')}" for k in ("share_gpu", "force", "train") if getattr(args, k)]
        if given:
            ap.error("--table builds the table of all stage-0 tasks from the stored scores and takes no other "
                     f"option (got {', '.join(given)})")
        return print_table()
    args.num_workers = 64 if args.num_workers is None else args.num_workers
    args.batch_size = 64 if args.batch_size is None else args.batch_size
    args.variant = BASE_VARIANT if args.variant is None else args.variant
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
    if args.history_s is not None and (problem := _history_problem(task, args.history_s)):
        print(f"refused: {problem}", file=sys.stderr)  # before anything is written or deleted
        return 2
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
            try:
                _train_direct(ctx, root / "scores" / args.task / name / args.split)
            except Refused as e:
                print(f"refused: {e}", file=sys.stderr)
                return 2
    provenance = provenance_of(args.predictor, ctx)
    out_dir = root / "scores" / args.task / name / args.split
    stored, reason = _kept_score(out_dir, args.task, name, args.split, provenance, args.force)
    if stored is not None:
        print(f"kept {args.task} {name} {args.split} "
              f"(scored at {stored['git_revision']}, {stored['date']}): nrmse {float(stored['nrmse']):.4f}",
              flush=True)
        if is_direct:
            _refresh_kept_record(ctx, out_dir / "score.json", stored)
        return 0 if math.isfinite(float(stored["nrmse"])) else 1
    if reason != "no score.json":
        print(f"re-scoring {args.task} {name} {args.split}: {reason}", flush=True)

    if is_direct and not gpu_checked and _refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
        return 2
    started = time.monotonic()
    predict_fn = build(ctx)  # a fitted predictor is fitted here if there is no current fit file
    if args.predictor in FITTED:
        predict_fn = _blind_to_targets(predict_fn)  # the floor is the one predictor that reads its targets
    run_record = _direct_record(ctx) if is_direct else dict(ctx.built.get("fit_record", {}))
    if is_direct and (run_record["trained_steps"] is None or run_record["trained_steps"] < run_record["max_steps"]):
        trained = ("an unknown number of steps (no readable last.pt)" if run_record["trained_steps"] is None
                   else f"{run_record['trained_steps']} of {run_record['max_steps']} steps")
        print(f"warning: {name} was trained for {trained}, short of max_steps {run_record['max_steps']}; "
              f"the run did not finish, so this score is not final (checkpoint scored: step "
              f"{run_record['checkpoint_step']})", file=sys.stderr, flush=True)
    first_provenance, provenance = provenance, provenance_of(args.predictor, ctx)  # now with the fit it uses
    if is_direct and provenance["checkpoint_sha256"] != first_provenance["checkpoint_sha256"]:
        print(f"refused: {_checkpoint(ctx)} changed while the predictor was being built (its hash was "
              f"{first_provenance['checkpoint_sha256'][:12]}, now {provenance['checkpoint_sha256'][:12]}): a score "
              "must come from one checkpoint; a training of the same run may be writing it", file=sys.stderr)
        return 1
    if no_digest := [k for k, v in provenance.items() if v == cheap.NO_CURRENT_FIT]:
        print(f"refused: {args.predictor} has no current fit ({', '.join(no_digest)}), and the sentinel is not a "
              "digest to store in a score", file=sys.stderr)
        return 1
    result = harness.score(task, predict_fn, out_dir, split=args.split, history_s=getattr(predict_fn, "history_s", None),
                           batch_size=args.batch_size, num_workers=args.num_workers)
    path = baselines.write_score_json(result, name, args.split)
    record = json.loads(path.read_text())
    record.update(provenance)
    record.update(run_record)  # records, not provenance: `_kept_score` does not compare them
    _write_json_atomic(path, record)
    print(f"{args.task} {name} {args.split}: nrmse {result.task_nrmse:.4f}, "
          f"{result.n_windows} windows, {time.monotonic() - started:.0f} s", flush=True)
    return 0 if math.isfinite(result.task_nrmse) else 1


if __name__ == "__main__":
    sys.exit(main())
