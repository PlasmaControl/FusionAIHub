"""Fit the fixed tokenizers of stage 1 and score what they cost, on TokaMark's evaluator.

`fit` fits the ranges and the bases of every signal on training shots spread evenly over the train split,
and writes them to `<stage1>/fit/v1/` (`fit.npz`, `fit.json`). It refuses to overwrite a fit without
`--force`. `score` scores one task on `val_select` or `val_report` (never on the test split) and writes
`score.json` and `signals.csv` to `<stage1>/scores/<task>/<predictor>/<label>/`. The instrument decides
what is scored: `floor` (the default, no GPU) sends the true targets through the codecs of an assignment;
`passthrough` sends the prediction of a trained direct model through them; `inputs` sends the model's
inputs and actuators through them; `raw` scores the direct model unchanged. The last three need a GPU
(`--gpu N`) and a checkpoint in the stage-0 directory.

A stored score carries the hash of the tokenizer code, of the assignment, of the fit and of the statistics,
the TokaMark revision and data root, and the hash of the checkpoint when there is one. It is kept (printed
as `kept`) while all of them are what is loaded now, and scored again otherwise or with `--force`.
Everything is written under `IGNITE_MAST_ROOT` (default `paths.mast_root()`), after `paths.prepare_write_root`
has checked the directory.

Later sub-commands (`ladder`, `assign`, `gate`) are added as one more sub-parser and one more function each.
This file does nothing at import time except define: `fit` starts spawned workers, and they import it again.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from tokamak_foundation_model.ignite.mast import baselines, cache, harness, paths, tasks, transform
from tokamak_foundation_model.ignite.mast.headroom import table
from tokamak_foundation_model.ignite.mast.tokenizers import assign, code_sha256, grid
from tokamak_foundation_model.ignite.mast.tokenizers import fit, floor

STAGE1_SPLITS = ("val_select", "val_report")
INSTRUMENTS = ("floor", "passthrough", "inputs", "raw")
FIT_VERSION = "v1"
HEADROOM_SCRIPT = Path(__file__).resolve().with_name("run_headroom.py")
_VARIANT = re.compile(r"[A-Za-z0-9_-]+")


class Refused(Exception):
    """A refusal that is not an error of the code: `main` prints it and exits with status 2."""


# --------------------------------------------------------------------------------------- names

def predictor_name(instrument: str, assignment_name: str | None, mode: str | None,
                   variant: str | None = None) -> str:
    """`tok.<assignment>.<mode>` (floor), `pass.<assignment>.<mode>.<variant>`, `inputs.<assignment>.<mode>.
    <variant>` or `raw.<variant>`. `assignment_name` is the file name of the assignment without `.json`."""
    if instrument == "floor":
        return f"tok.{assignment_name}.{mode}"
    if instrument == "passthrough":
        return f"pass.{assignment_name}.{mode}.{variant}"
    if instrument == "inputs":
        return f"inputs.{assignment_name}.{mode}.{variant}"
    if instrument == "raw":
        return f"raw.{variant}"
    raise ValueError(f"unknown instrument {instrument!r}; one of {INSTRUMENTS}")


def split_label(split: str, n_shots: int | None) -> str:
    """`split`, or `<split>-n<N>` for the subset of `N` shots."""
    return split if n_shots is None else f"{split}-n{n_shots}"


def _prepare(path: Path) -> Path:
    """`paths.prepare_write_root(path)`; `Refused` if it refuses the directory (repository, TokaMind, TokaMark)."""
    try:
        return paths.prepare_write_root(path)
    except ValueError as e:
        raise Refused(str(e)) from e


def _write_json_atomic(path: Path, record: dict) -> None:
    """Write `record` as JSON to `path` through a temporary name in the same directory, then replace: a reader
    sees the old file or the new one, never half of one. Same as `run_headroom.py:_write_json_atomic`."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- provenance, keeping

def kept_score(out_dir: Path, task: str, predictor: str, label: str, provenance: Mapping[str, str],
               force: bool) -> tuple[dict | None, str]:
    """(stored record or None, reason it is not kept). An accepted copy of
    `scripts/mast/run_headroom.py:_kept_score`, with `label` (the split with its subset) in the place of the
    split: the record of `out_dir` is kept if `baselines.read_kept_score` keeps it and every key of
    `provenance` equals the stored value."""
    if force:
        return None, "--force"
    stored, reason = baselines.read_kept_score(out_dir, task, predictor, label)
    if stored is None:
        return None, reason
    for key, value in provenance.items():
        if stored.get(key) != value:
            return None, f"{key} changed since {stored.get('git_revision', 'unknown')}"
    return stored, ""


def provenance_of(assignment_path: Path | None, fit_dir: Path | None, cache_root: Path,
                  checkpoint: Path | None = None) -> dict[str, str]:
    """What a stored score must carry to be the score of the code and files in use now.

    In this order: `tokenizer_code_sha256`, `assignment_sha256`, `fit_sha256`, `stats_sha256` (these four only
    with an assignment), `tokamark_revision` (from ``manifest.json`` of `cache_root`), `data_root`, and
    `checkpoint_sha256` (the sha256 of the file `checkpoint`). `ValueError` naming "statistics" if
    ``stats.npz``, the manifest and the fit's meta do not carry one hash. Arrays are not read.
    """
    manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
    out: dict[str, str] = {}
    if assignment_path is not None:
        stats_sha = transform.stats_sha256(Path(cache_root) / "stats.npz")
        fit_meta = json.loads((Path(fit_dir) / "fit.json").read_text())["meta"]
        shas = {"stats.npz": stats_sha, "manifest.json": manifest["stats_sha256"],
                "fit.json": fit_meta["stats_sha256"]}
        if len(set(shas.values())) != 1:
            raise ValueError("the statistics are not one set: " + ", ".join(f"{k} {v[:12]}" for k, v in shas.items()))
        out = {"tokenizer_code_sha256": code_sha256(),
               "assignment_sha256": assign.assignment_sha256(assign.load_assignment(assignment_path)),
               "fit_sha256": fit.fit_sha256(Path(fit_dir)), "stats_sha256": stats_sha}
    out["tokamark_revision"] = manifest["tokamark_revision"]
    out["data_root"] = str(paths.tokamark_zarr_root())
    if checkpoint is not None:
        out["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return out


def score_one(task: tasks.TaskSpec, predict_fn: harness.PredictFn, *, predictor: str, out_dir: Path, split: str,
              label: str, shots: list[int] | None, provenance: Mapping[str, str], records: Mapping[str, object],
              force: bool = False, num_workers: int = 32, batch_size: int = 64) -> dict:
    """The kept record of `out_dir`, or the record of a new score.

    A new score is `harness.score` of `predict_fn` (arrays in the harness's physical units) on `split`
    restricted to `shots`; `score.json` is stored under `label`, with `provenance` and `records` added
    (written atomically), and `signals.csv` holds the per-signal NRMSE (TokaMark units).
    """
    stored, _ = kept_score(out_dir, task.name, predictor, label, provenance, force)
    if stored is not None:
        return stored
    out_dir = _prepare(Path(out_dir))
    result = harness.score(task, predict_fn, out_dir, split=split, shots=shots,
                           history_s=getattr(predict_fn, "history_s", None),
                           batch_size=batch_size, num_workers=num_workers)
    path = baselines.write_score_json(result, predictor, label)
    record = json.loads(path.read_text())
    record.update(provenance)
    record.update(records)
    _write_json_atomic(path, record)
    result.per_signal.to_csv(out_dir / "signals.csv", index=False)
    return record


# -------------------------------------------------------------------------------------------- fit

def _cache_root(args: argparse.Namespace) -> Path:
    return Path(args.cache_dir) if args.cache_dir else paths.cache_dir()


def run_fit(args: argparse.Namespace) -> int:
    """Fit on `--n-shots` training shots spread evenly (`cache.pick_evenly`) and write `<stage1>/fit/v1/`.
    Prints one line per signal; the arrays it fits are in model units (`transform.to_model`)."""
    fit_dir = paths.stage1_dir() / "fit" / FIT_VERSION
    if (fit_dir / "fit.npz").exists() and not args.force:
        print(f"refused: {fit_dir / 'fit.npz'} exists; --force fits again over it", file=sys.stderr)
        return 2
    if args.n_shots < 1 or args.workers < 1:
        print("refused: --n-shots and --workers are at least 1", file=sys.stderr)
        return 2
    _prepare(fit_dir)
    root = _cache_root(args)
    manifest = json.loads((root / "manifest.json").read_text())
    shots = cache.pick_evenly(tasks.split_shots("train"), args.n_shots)
    started = time.monotonic()
    fits = fit.fit_all(root, shots, workers=args.workers)
    for name, f in fits.items():
        if f.n_examples == 0:
            print(f"{name}: no data (n_shots {f.n_shots}, n_skipped {f.n_skipped})", flush=True)
        else:
            print(f"{name}: n_shots {f.n_shots}, n_skipped {f.n_skipped}, n_examples {f.n_examples}", flush=True)
    meta = {"date": datetime.now(timezone.utc).isoformat(), "n_shots": len(shots), "shots": [int(s) for s in shots],
            "stats_sha256": manifest["stats_sha256"], "tokamark_revision": manifest["tokamark_revision"],
            "tokenizer_code_sha256": code_sha256(), "cache_dir": str(root),
            "seconds": round(time.monotonic() - started, 1)}
    fit.save_fit(fits, fit_dir, meta)
    return 0


# ------------------------------------------------------------------------------------------ score

def _run_headroom():
    """`scripts/mast/run_headroom.py` loaded by path (its GPU helpers; `py.sh` runs Python with `-I`)."""
    name = "run_headroom_for_tokenizer"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HEADROOM_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look their module up here
        spec.loader.exec_module(module)
    return sys.modules[name]


def _refuse_wrong_options(args: argparse.Namespace) -> str | None:
    """The reason the options do not fit the instrument (nothing is read), or None."""
    gpu_instrument = args.instrument != "floor"
    if args.instrument == "raw":
        if args.assignment or args.mode:
            return "--instrument raw takes no --assignment and no --mode: it scores the model unchanged"
    elif not args.assignment or not args.mode:
        return f"--instrument {args.instrument} needs --assignment FILE and --mode {{{','.join(grid.READ_MODES)}}}"
    if gpu_instrument:
        if args.gpu is None:
            return f"--instrument {args.instrument} needs --gpu N (a GPU without a compute process)"
        if args.variant is not None and not _VARIANT.fullmatch(args.variant):
            return f"--variant {args.variant!r} is not a name (letters, digits, - and _)"
    elif args.gpu is not None or args.share_gpu or args.variant is not None:
        return "--instrument floor uses no GPU and no model: --gpu, --share-gpu and --variant belong to the others"
    if args.n_shots is not None and args.n_shots < 1:
        return "--n-shots is at least 1"
    return None


def _needed_signals(task: tasks.TaskSpec, instrument: str) -> tuple[tasks.SignalSpec, ...]:
    """The signals the instrument sends through a codec: the task's outputs, or its inputs and actuators."""
    if instrument == "inputs":
        return (*task.inputs, *task.actuators)
    return task.outputs if instrument in ("floor", "passthrough") else ()


def _build_predictor(args: argparse.Namespace, task: tasks.TaskSpec, cache_root: Path, assignment_path: Path | None,
                     fit_dir: Path, checkpoint: Path | None) -> harness.PredictFn:
    """The harness predictor of the instrument (physical units in and out). `Refused` if a signal it needs
    has no fit, no data or no setting."""
    inner = None
    if checkpoint is not None:
        rh = _run_headroom()
        inner = floor.blind_to_targets(rh._direct_module().direct_predictor(task, checkpoint, rh.DEVICE))
        if args.instrument == "raw":
            return inner
    signals = _needed_signals(task, args.instrument)
    fits, _ = fit.load_fit(fit_dir)
    assignment = assign.load_assignment(assignment_path)
    stats = transform.load_stats(cache_root / "stats.npz")
    for s in signals:
        if s.name not in fits or fits[s.name].n_examples == 0:
            raise Refused(f"signal {s.name} has no data in the fit, and {task.name} needs it")
        if s.name not in assignment:
            raise Refused(f"the assignment has no setting for signal {s.name}, and {task.name} needs it")
    codecs = fit.build_codecs(signals, stats, fits, assignment)
    if args.instrument == "floor":
        return floor.floor_predictor(task, codecs, args.mode)
    if args.instrument == "passthrough":
        return floor.passthrough_predictor(inner, task, codecs, args.mode)
    return floor.input_roundtrip_predictor(inner, task, codecs, args.mode)


def run_score(args: argparse.Namespace) -> int:
    """Score `--task` on `--split` with the instrument; print `<task> <predictor> <label>: nrmse <v>, <n>
    windows, <s> s`, or `kept ...`. Exit 2 for a refusal (before any file is read for the options), 1 for a
    score that is not finite."""
    if (problem := _refuse_wrong_options(args)) is not None:
        print(f"refused: {problem}", file=sys.stderr)
        return 2
    gpu_instrument = args.instrument != "floor"
    stage0 = Path(args.stage0_dir) if args.stage0_dir else paths.stage0_dir()
    variant = (args.variant or table.BASE_VARIANT) if gpu_instrument else None
    assignment_path = Path(args.assignment) if args.assignment else None
    checkpoint = None
    gpu_pids: list[int] = []
    if gpu_instrument:
        rh = _run_headroom()
        if rh._cuda_initialized():
            print("refused: CUDA is already initialized in this process, so --gpu cannot select the GPU",
                  file=sys.stderr)
            return 2
        looked_up = rh._look_up_gpu(args.gpu)
        if looked_up is None:
            return 2
        gpu_uuid, gpu_pids = looked_up
        # CUDA reads this when it initializes, which has not happened (checked above): the GPU is then cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        checkpoint = table.direct_run_dir(stage0, args.task, variant) / "best.pt"
        if not checkpoint.exists():
            print(f"refused: there is no checkpoint at {checkpoint}: train it with run_headroom.py first",
                  file=sys.stderr)
            return 2
    if assignment_path is not None and not assignment_path.exists():
        print(f"refused: there is no assignment at {assignment_path}", file=sys.stderr)
        return 2

    cache_root = _cache_root(args)
    stage1 = paths.stage1_dir()
    fit_dir = stage1 / "fit" / FIT_VERSION
    if assignment_path is not None and not (fit_dir / "fit.npz").exists():
        print(f"refused: there is no fit at {fit_dir}: run `fit` first", file=sys.stderr)
        return 2
    name = predictor_name(args.instrument, assignment_path.name.removesuffix(".json") if assignment_path else None,
                          args.mode, variant)
    label = split_label(args.split, args.n_shots)
    out_dir = stage1 / "scores" / args.task / name / label
    try:
        provenance = provenance_of(assignment_path, fit_dir if assignment_path else None, cache_root, checkpoint)
    except (OSError, KeyError, ValueError) as e:
        print(f"refused: the provenance cannot be made ({type(e).__name__}: {e})", file=sys.stderr)
        return 2

    stored, reason = kept_score(out_dir, args.task, name, label, provenance, args.force)
    if stored is not None:
        print(f"kept {args.task} {name} {label} (scored at {stored['git_revision']}, {stored['date']}): "
              f"nrmse {float(stored['nrmse']):.4f}", flush=True)
        return 0 if math.isfinite(float(stored["nrmse"])) else 1
    if reason != "no score.json":
        print(f"re-scoring {args.task} {name} {label}: {reason}", flush=True)
    _prepare(out_dir)
    if gpu_instrument and rh._refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
        return 2

    started = time.monotonic()
    task = tasks.load_task(args.task)
    records: dict[str, object] = {"instrument": args.instrument, "mode": args.mode,
                                  "assignment": assignment_path.name if assignment_path else None,
                                  "variant": variant, "n_shots_asked": args.n_shots}
    if gpu_instrument:
        stale = rh.stale_reason(stage0, args.task, table.predictor_name("direct", variant), "val_report")
        records["direct_stale_reason"] = stale
        records["scored_blind"] = True
        if stale:
            print(f"warning: the direct score of {variant} is stale ({stale}); the model is the checkpoint "
                  "scored here, whatever that score says", file=sys.stderr, flush=True)
    predict_fn = _build_predictor(args, task, cache_root, assignment_path, fit_dir, checkpoint)
    if gpu_instrument and provenance["checkpoint_sha256"] != provenance_of(
            assignment_path, fit_dir if assignment_path else None, cache_root, checkpoint)["checkpoint_sha256"]:
        print(f"refused: {checkpoint} changed while the predictor was being built", file=sys.stderr)
        return 1
    shots = (cache.pick_evenly(tasks.split_shots(args.split), args.n_shots)
             if args.n_shots is not None else None)
    record = score_one(task, predict_fn, predictor=name, out_dir=out_dir, split=args.split, label=label,
                       shots=shots, provenance=provenance, records=records, force=args.force,
                       num_workers=args.num_workers, batch_size=args.batch_size)
    print(f"{args.task} {name} {label}: nrmse {float(record['nrmse']):.4f}, {record['n_windows']} windows, "
          f"{time.monotonic() - started:.0f} s", flush=True)
    return 0 if math.isfinite(float(record["nrmse"])) else 1


# -------------------------------------------------------------------------------------------- main

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="fit the ranges and bases on training shots")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.add_argument("--n-shots", type=int, default=2000, help="training shots, spread evenly (default 2000)")
    p.add_argument("--workers", type=int, default=64, help="reader processes (default 64)")
    p.add_argument("--force", action="store_true", help="fit again over an existing fit")
    p.set_defaults(run=run_fit)

    p = sub.add_parser("score", help="score a task with an instrument")
    p.add_argument("--task", required=True, choices=list(tasks.ALL_TASKS))
    p.add_argument("--split", required=True, choices=STAGE1_SPLITS)
    p.add_argument("--assignment", default=None, help="an assignment file (not for --instrument raw)")
    p.add_argument("--mode", default=None, choices=grid.READ_MODES, help="the read mode (not for raw)")
    p.add_argument("--n-shots", type=int, default=None, help="a subset of the split, spread evenly")
    p.add_argument("--instrument", default="floor", choices=INSTRUMENTS)
    p.add_argument("--variant", default=None, help="the direct model's variant (default base; not for floor)")
    p.add_argument("--gpu", type=int, default=None, help="the GPU index (required but for floor, which refuses it)")
    p.add_argument("--share-gpu", action="store_true", help="use a GPU that has compute processes")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.add_argument("--stage0-dir", default=None, help="stage 0's directory, for the checkpoints")
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force", action="store_true", help="score again even if a score is kept")
    p.set_defaults(run=run_score)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
