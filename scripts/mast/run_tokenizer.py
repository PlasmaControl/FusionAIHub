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

`ladder --task T` scores every rung of the tokenizers of task `T`'s outputs on the true targets (floor) from
one list of `val_select` windows kept in memory, with the mean baseline on the same windows, and, with
`--variant V --gpu N`, the prediction of the direct model through each rung and the model unchanged. It writes
`<stage1>/ladders/<task>.csv` and `.json`, and every pass as a score under `scores/<task>/tok.ladder-*`.
`ladder --inputs` reads the reconstruction error of the signals no task predicts from the cache (rule 4;
no GPU, no windows). `assign` applies the rules to the seven ladder files and writes
`<stage1>/assignments/<name>.json`; it refuses a name that exists, and ladders made with another fit.
Every refusal comes before the first write, and no refusal leaves a directory behind.

Later sub-commands (`gate`) are added as one more sub-parser and one more function each. This file does
nothing at import time except define: `fit` and `ladder --inputs` start spawned workers, and they import it
again.
"""

from __future__ import annotations

import argparse
import dataclasses
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

import numpy as np
import pandas as pd

from tokamak_foundation_model.ignite.mast import baselines, cache, harness, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.headroom import table
from tokamak_foundation_model.ignite.mast.tokenizers import assign, code_sha256, grid
from tokamak_foundation_model.ignite.mast.tokenizers import fit, floor, ladder

STAGE1_SPLITS = ("val_select", "val_report")
INSTRUMENTS = ("floor", "passthrough", "inputs", "raw")
FIT_VERSION = "v1"
LADDER_MODE = "unquantized"
LADDER_SPLIT = "val_select"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
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
    root = _cache_root(args)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["stats_sha256"], manifest["tokamark_revision"]
    except (OSError, ValueError, KeyError) as e:
        print(f"refused: the manifest of {root} cannot be read ({type(e).__name__}: {e})", file=sys.stderr)
        return 2
    shots = cache.pick_evenly(tasks.split_shots("train"), args.n_shots)
    _prepare(fit_dir)  # after every refusal and read that can fail: nothing is left behind by one
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
    if gpu_instrument and rh._refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
        return 2

    started = time.monotonic()
    task = tasks.load_task(args.task)
    records: dict[str, object] = {"instrument": args.instrument, "mode": args.mode,
                                  "assignment": assignment_path.name if assignment_path else None,
                                  "variant": variant, "n_shots_asked": args.n_shots}
    predict_fn = _build_predictor(args, task, cache_root, assignment_path, fit_dir, checkpoint)
    if gpu_instrument:
        stale = rh.stale_reason(stage0, args.task, table.predictor_name("direct", variant), "val_report")
        records["direct_stale_reason"] = stale
        records["scored_blind"] = True
        if stale:
            print(f"warning: the direct score of {variant} is stale ({stale}); the model is the checkpoint "
                  "scored here, whatever that score says", file=sys.stderr, flush=True)
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


# ---------------------------------------------------------------------------------------- ladder

def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Write `frame` as CSV to `path` through a temporary name in the same directory, then replace."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _fit_provenance(fit_dir: Path, cache_root: Path) -> dict[str, str]:
    """`tokenizer_code_sha256`, `fit_sha256` and `stats_sha256` of what is loaded now (the three that mark a
    ladder as made from this fit). `ValueError` naming "statistics" if ``stats.npz``, the manifest and the
    fit's meta do not carry one hash. Arrays are not read."""
    manifest = json.loads((Path(cache_root) / "manifest.json").read_text())
    stats_sha = transform.stats_sha256(Path(cache_root) / "stats.npz")
    meta = json.loads((Path(fit_dir) / "fit.json").read_text())["meta"]
    shas = {"stats.npz": stats_sha, "manifest.json": manifest["stats_sha256"], "fit.json": meta["stats_sha256"]}
    if len(set(shas.values())) != 1:
        raise ValueError("the statistics are not one set: " + ", ".join(f"{k} {v[:12]}" for k, v in shas.items()))
    return {"tokenizer_code_sha256": code_sha256(), "fit_sha256": fit.fit_sha256(Path(fit_dir)),
            "stats_sha256": stats_sha}


def _refuse_wrong_ladder_options(args: argparse.Namespace) -> str | None:
    """The reason the options of `ladder` do not fit together (nothing is read), or None."""
    if bool(args.task) == bool(args.inputs):
        return "ladder needs exactly one of --task T and --inputs"
    for name in ("n_shots", "workers", "num_workers", "batch_size"):
        if getattr(args, name) < 1:
            return f"--{name.replace('_', '-')} is at least 1"
    if args.inputs:
        if args.variant is not None or args.gpu is not None or args.share_gpu:
            return "--inputs reads the cache and uses no model: --variant, --gpu and --share-gpu belong to --task"
        return None
    if (args.variant is None) != (args.gpu is None):
        return "--variant V and --gpu N go together (the direct model needs a GPU), or neither"
    if args.share_gpu and args.gpu is None:
        return "--share-gpu belongs to --gpu"
    if args.variant is not None and not _VARIANT.fullmatch(args.variant):
        return f"--variant {args.variant!r} is not a name (letters, digits, - and _)"
    return None


def _ladder_files(stage1: Path, name: str) -> tuple[Path, Path]:
    return stage1 / "ladders" / f"{name}.csv", stage1 / "ladders" / f"{name}.json"


def _ladder_record(shots: list[int], args: argparse.Namespace, prov: Mapping[str, str], **more) -> dict:
    return {"shots": [int(s) for s in shots], "n_shots_asked": args.n_shots, "split": LADDER_SPLIT,
            "fit_sha256": prov["fit_sha256"], "tokenizer_code_sha256": prov["tokenizer_code_sha256"],
            "stats_sha256": prov["stats_sha256"], "date": datetime.now(timezone.utc).isoformat(), **more}


def _score_windows(task: tasks.TaskSpec, batches: list[harness.WindowBatch], predict_fn: harness.PredictFn, *,
                   predictor: str, label: str, provenance: Mapping[str, str],
                   records: Mapping[str, object]) -> dict[str, float]:
    """Score `predict_fn` (physical units) on the kept `batches` and store it as `score.json` and `signals.csv` of
    ``<stage1>/scores/<task>/<predictor>/<label>/``, with `provenance` and `records` added. Returns the NRMSE
    per signal (TokaMark units). A stored score of that directory is replaced (a ladder does not keep scores)."""
    out_dir = _prepare(paths.stage1_dir() / "scores" / task.name / predictor / label)
    result = harness.score_batches(task.name, batches, predict_fn, out_dir)
    path = baselines.write_score_json(result, predictor, label)
    record = json.loads(path.read_text())
    record.update(provenance)
    record.update(records)
    _write_json_atomic(path, record)
    result.per_signal.to_csv(out_dir / "signals.csv", index=False)
    return {str(n): float(v) for n, v in zip(result.per_signal["feature_name"], result.per_signal["NRMSE"])}


def _band_tables(task: tasks.TaskSpec, a: pd.DataFrame, b: pd.DataFrame, k0: int,
                 contrast: Mapping[str, list[float]]) -> pd.DataFrame:
    """The ladder table of a spectrogram task from the two `ladder.run_ladder` tables of its steps: columns
    `signal`, `step`, `k0`, `k`, `values_per_frame`, `floor`, `raw`, `passed`, `contrast`, `mean_baseline`.
    Step a has one row per `k0` (the others at 32), step b one row per common K of `assign.BAND_RUNGS` at `k0`;
    `contrast` maps a signal to the pooled contrast of each step-b rung (dimensionless)."""
    a = a.assign(step="a", k0=a["k"], k=assign.BAND_WHOLE, contrast=np.nan)
    rung = b.groupby("signal", sort=False).cumcount().to_numpy()
    ks = [setting.k[1] for setting in assign.band_rungs(k0)]
    b = b.assign(step="b", k0=k0, k=[ks[i] for i in rung],
                 contrast=[contrast[sig][i] for sig, i in zip(b["signal"], rung)])
    cols = ["signal", "step", "k0", "k", "values_per_frame", "floor", "raw", "passed", "contrast", "mean_baseline"]
    return pd.concat([a, b], ignore_index=True)[cols]


def _ladder_task(args: argparse.Namespace, stage1: Path, cache_root: Path, fit_dir: Path,
                 prov: Mapping[str, str]) -> int:
    started = time.monotonic()
    task = tasks.load_task(args.task)
    spectro = [s.name in tasks.SPECTROGRAM_SIGNALS for s in task.outputs]
    if any(spectro) and not all(spectro):
        raise Refused(f"{task.name} has spectrogram and other outputs; a ladder of one kind at a time")
    fits, _ = fit.load_fit(fit_dir)
    stats = transform.load_stats(cache_root / "stats.npz")
    for s in task.outputs:
        if s.name not in fits or fits[s.name].n_examples == 0:
            raise Refused(f"signal {s.name} has no data in the fit, and {task.name} needs it")
    shots = cache.pick_evenly(tasks.split_shots(LADDER_SPLIT), args.n_shots)
    if all(spectro):
        gone = [s for s in shots if not (cache_root / "shots" / f"{s}.h5").exists()]
        if gone:
            raise Refused(f"{len(gone)} of the {len(shots)} shots have no cache file in {cache_root} (the contrast "
                          f"reads it); the first is {gone[0]}")
    variant, checkpoint, gpu_pids, rh = args.variant, None, [], None
    stage0 = Path(args.stage0_dir) if args.stage0_dir else paths.stage0_dir()
    if variant is not None:
        rh = _run_headroom()
        if rh._cuda_initialized():
            raise Refused("CUDA is already initialized in this process, so --gpu cannot select the GPU")
        looked_up = rh._look_up_gpu(args.gpu)
        if looked_up is None:
            return 2
        gpu_uuid, gpu_pids = looked_up
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid       # CUDA reads it when it initializes: the GPU is cuda:0
        checkpoint = table.direct_run_dir(stage0, args.task, variant) / "best.pt"
        if not checkpoint.exists():
            raise Refused(f"there is no checkpoint at {checkpoint}: train it with run_headroom.py first")
        prov = {**prov, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        if rh._refuse_busy_gpu(args.gpu, gpu_pids, args.share_gpu):
            return 2
    base = {**provenance_of(None, None, cache_root, checkpoint), **prov}
    _prepare(stage1 / "ladders")                            # after every refusal above

    label = split_label(LADDER_SPLIT, args.n_shots)
    inner, history_s, stale = None, None, ""
    if variant is not None:
        inner = floor.blind_to_targets(rh._direct_module().direct_predictor(task, checkpoint, rh.DEVICE))
        history_s = getattr(inner, "history_s", None)
        stale = rh.stale_reason(stage0, args.task, table.predictor_name("direct", variant), "val_report")
        if stale:
            print(f"warning: the direct score of {variant} is stale ({stale}); the model is the checkpoint "
                  "scored here, whatever that score says", file=sys.stderr, flush=True)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != base["checkpoint_sha256"]:
            print(f"refused: {checkpoint} changed while the predictor was being built", file=sys.stderr)
            return 1
    batches = list(harness.iter_windows(task, shots, test_mode=True, history_s=history_s,
                                        batch_size=args.batch_size, num_workers=args.num_workers))
    print(f"{task.name}: {len(batches)} batches of windows of {len(shots)} {LADDER_SPLIT} shots, "
          f"{time.monotonic() - started:.0f} s", flush=True)

    def key(batch: harness.WindowBatch) -> tuple[int, int]:
        return int(batch.shot_ids[0]), int(batch.window_indices[0])

    raw_predictions = {key(b): inner(b) for b in batches} if inner is not None else None

    def lookup(batch: harness.WindowBatch) -> dict[str, np.ndarray]:
        return raw_predictions[key(batch)]

    memo: dict[str, dict[str, float]] = {}
    current: dict[str, assign.Setting] = {}                 # the settings of the pass being built, for its record
    records = {"mode": LADDER_MODE, "variant": variant, "n_shots_asked": args.n_shots}

    def build(settings: dict[str, assign.Setting]):
        current.clear()
        current.update(settings)
        return fit.build_codecs(task.outputs, stats, fits, settings)

    def score(kind: str, codecs, name: str):
        extra = {**records, "pass": name, "settings": {n: dataclasses.asdict(st) for n, st in current.items()}}
        if kind == "floor":
            return _score_windows(task, batches, floor.floor_predictor(task, codecs, LADDER_MODE),
                                  predictor=predictor_name("floor", name, LADDER_MODE), label=label,
                                  provenance=base, records={**extra, "instrument": "floor"})
        if raw_predictions is None:
            return None
        if kind == "pass":
            return _score_windows(task, batches, floor.passthrough_predictor(lookup, task, codecs, LADDER_MODE),
                                  predictor=predictor_name("passthrough", name, LADDER_MODE, variant), label=label,
                                  provenance=base, records={**extra, "instrument": "passthrough"})
        if "raw" not in memo:                               # asked once per step; scored once
            memo["raw"] = _score_windows(task, batches, lookup, predictor=predictor_name("raw", None, None, variant),
                                         label=label, provenance=base, records={**records, "instrument": "raw"})
        return memo["raw"]

    compress = {s.name: stats[s.name].compress for s in task.outputs}
    step_a = ladder.passes(task, compress)
    table_a = ladder.run_ladder(task, step_a, build, score)
    k0 = None
    if all(spectro):
        k0 = ladder.choose_low_band(table_a.assign(step="a", k0=table_a["k"])[["signal", "step", "k0", "floor"]])
        table_b = ladder.run_ladder(task, ladder.band_passes(task, k0), build, score)
        contrast = {}
        for s in task.outputs:
            per_shot = list(transform.map_shots(ladder.band_contrast_job, shots,
                                                (str(cache_root), base["stats_sha256"], str(fit_dir), s, k0),
                                                args.workers))
            contrast[s.name] = [ladder.pooled_contrast([r[i] for r in per_shot if r is not None])
                                for i in range(len(assign.BAND_RUNGS))]
    mean = _score_windows(task, batches, baselines.mean_predictor(task), predictor="mean.ladder", label=label,
                          provenance=base, records={**records, "instrument": "mean"})
    table_a["mean_baseline"] = table_a["signal"].map(mean)
    if k0 is not None:
        table_b["mean_baseline"] = table_b["signal"].map(mean)
        result = _band_tables(task, table_a, table_b, k0, contrast)
    else:
        result = table_a
    csv, js = _ladder_files(stage1, task.name)
    _write_csv_atomic(csv, result)
    _write_json_atomic(js, _ladder_record(shots, args, base, task=task.name, variant=variant, history_s=history_s,
                                          checkpoint_sha256=base.get("checkpoint_sha256"), band_k0=k0,
                                          seconds=round(time.monotonic() - started, 1)))
    print(result.head(3).to_string(index=False))
    print(f"{task.name}: {len(result)} rows -> {csv} in {time.monotonic() - started:.0f} s", flush=True)
    return 0


def _ladder_inputs(args: argparse.Namespace, stage1: Path, cache_root: Path, fit_dir: Path,
                   prov: Mapping[str, str]) -> int:
    started = time.monotonic()
    fits, _ = fit.load_fit(fit_dir)
    stats = transform.load_stats(cache_root / "stats.npz")
    specs = {s.name: s for s in registry.all_signals()}
    names = sorted(n for n, by in assign.predicted_by().items() if not by)
    shots = cache.pick_evenly(tasks.split_shots(LADDER_SPLIT), args.n_shots)
    gone = [s for s in shots if not (cache_root / "shots" / f"{s}.h5").exists()]
    if gone:
        raise Refused(f"{len(gone)} of the {len(shots)} shots have no cache file in {cache_root}; the first is {gone[0]}")
    _prepare(stage1 / "ladders")                            # after every refusal above
    results = list(transform.map_shots(ladder.input_errors_job, shots,
                                       (str(cache_root), prov["stats_sha256"], str(fit_dir),
                                        tuple(specs[n] for n in names)), args.workers))
    rows = []
    for name in names:
        spec = specs[name]
        settings = ladder.signal_ladder(spec, stats[name].compress)
        errors = np.array([r[name] for r in results if r[name] is not None]).reshape(-1, len(settings))
        fitted = name in fits and fits[name].n_examples > 0
        finite = np.isfinite(errors)
        n_used = finite.sum(axis=0)                                       # shots with a value, per setting
        mean = np.divide(np.where(finite, errors, 0.0).sum(axis=0), n_used, out=np.full(len(settings), np.nan),
                         where=n_used > 0)
        for j, st in enumerate(settings):
            note = "" if n_used[j] else ("no data" if fitted else "not fitted: no training shot has it")
            rows.append({"signal": name, "domain": st.domain, "k": int(st.k[0]),
                         "values_per_frame": ladder.values_per_frame(spec, st), "floor": mean[j], "raw": math.nan,
                         "passed": math.nan, "n_shots": int(n_used[j]), "n_skipped": len(shots) - int(n_used[j]),
                         "note": note})
        used = int(n_used.max()) if len(settings) else 0
        print(f"{name}: " + (f"{used} of {len(shots)} shots; error {mean[0]:.4f} at the first rung, "
                             f"{mean[len(settings) - 1]:.4f} at the last" if used else "no data"), flush=True)
    result = pd.DataFrame(rows)
    csv, js = _ladder_files(stage1, "inputs")
    _write_csv_atomic(csv, result)
    _write_json_atomic(js, _ladder_record(shots, args, prov, task=None, variant=None, checkpoint_sha256=None,
                                          signals=names, seconds=round(time.monotonic() - started, 1)))
    print(result.head(3).to_string(index=False))
    print(f"inputs: {len(names)} signals, {len(result)} rows -> {csv} in {time.monotonic() - started:.0f} s",
          flush=True)
    return 0


def run_ladder_command(args: argparse.Namespace) -> int:
    """`ladder --task T` or `ladder --inputs`: see the module docstring. Exit 2 for a refusal (before any write)."""
    if (problem := _refuse_wrong_ladder_options(args)) is not None:
        raise Refused(problem)
    cache_root = _cache_root(args)
    stage1 = paths.stage1_dir()
    fit_dir = stage1 / "fit" / FIT_VERSION
    if not (fit_dir / "fit.npz").exists():
        raise Refused(f"there is no fit at {fit_dir}: run `fit` first")
    try:
        prov = _fit_provenance(fit_dir, cache_root)
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the provenance cannot be made ({type(e).__name__}: {e})") from e
    return (_ladder_inputs if args.inputs else _ladder_task)(args, stage1, cache_root, fit_dir, prov)


# ---------------------------------------------------------------------------------------- assign

def _k_text(setting: assign.Setting) -> str:
    return "/".join(str(v) for v in setting.k)


def _print_assignment(a: Mapping[str, assign.Setting], notes: Mapping[str, str],
                      specs: Mapping[str, tasks.SignalSpec]) -> None:
    total = 0
    for name in sorted(a):
        st = a[name]
        n = ladder.values_per_frame(specs[name], st)
        total += n
        print(f"{name}: {st.codec} K {_k_text(st)} domain {st.domain} values/frame {n}; {notes.get(name, '')}")
    print(f"total values per frame: {total}", flush=True)


def _raise_inputs(old: Mapping[str, assign.Setting], old_notes: Mapping[str, str], old_name: str,
                  specs: Mapping[str, tasks.SignalSpec]) -> tuple[dict[str, assign.Setting], dict[str, str]]:
    """Rule 4: `old` with each signal no task predicts one rung higher in its domain's ladder (the last rung
    stays). `Refused` for such a signal whose setting is not a rung of its ladder."""
    a, notes = dict(old), dict(old_notes)
    for name in sorted(n for n, by in assign.predicted_by().items() if not by):
        rungs = assign.rungs(specs[name], old[name].domain)
        if old[name] not in rungs:
            raise Refused(f"{old_name}: the setting of {name} ({old[name]}) is not a rung of its ladder")
        i = rungs.index(old[name])
        a[name] = rungs[min(i + 1, len(rungs) - 1)]
        notes[name] = (f"rule 4: raised from K {_k_text(old[name])} to {_k_text(a[name])}" if a[name] != old[name]
                       else f"rule 4: already the largest rung, K {_k_text(old[name])}") + f" (first assignment {old_name})"
    return a, notes


def run_assign(args: argparse.Namespace) -> int:
    """Write ``<stage1>/assignments/<name>.json`` from the ladders, from `--from` with the inputs raised, or the
    reference; print one line per signal and the total of values per frame. Exit 2 for a refusal."""
    if not _NAME.fullmatch(args.name):
        raise Refused(f"--name {args.name!r} is not a file name (letters, digits, . _ -)")
    if args.raise_inputs != (args.from_ is not None):
        raise Refused("--from OLD and --raise-inputs go together")
    if args.reference and args.from_ is not None:
        raise Refused("--reference writes the reference assignment: it takes no --from")
    stage1 = paths.stage1_dir()
    out = stage1 / "assignments" / f"{args.name}.json"
    if out.exists():
        raise Refused(f"{out} exists: an assignment is not overwritten; choose another --name")
    specs = {s.name: s for s in registry.all_signals()}
    if args.reference:
        a, notes = assign.reference_assignment(list(specs.values())), {n: "reference: the signal's own mean" for n in specs}
    elif args.from_ is not None:
        old_name = args.from_.removesuffix(".json")
        old_path = stage1 / "assignments" / f"{old_name}.json"
        if not _NAME.fullmatch(old_name) or not old_path.exists():
            raise Refused(f"there is no assignment {old_path}")
        old = assign.load_assignment(old_path)
        a, notes = _raise_inputs(old, json.loads(old_path.read_text()).get("notes", {}), old_name, specs)
    else:
        cache_root = _cache_root(args)
        fit_dir = stage1 / "fit" / FIT_VERSION
        if not (fit_dir / "fit.npz").exists():
            raise Refused(f"there is no fit at {fit_dir}: run `fit` first")
        try:
            prov = _fit_provenance(fit_dir, cache_root)
            stats = transform.load_stats(cache_root / "stats.npz")
        except (OSError, KeyError, ValueError) as e:
            raise Refused(f"the provenance cannot be made ({type(e).__name__}: {e})") from e
        problems, frames_ = [], []
        for name in (*ladder.LADDER_TASKS, "inputs"):
            csv, js = _ladder_files(stage1, name)
            gone = [p.name for p in (csv, js) if not p.exists()]
            if gone:
                problems.append(f"{', '.join(gone)} missing")
                continue
            made = json.loads(js.read_text()).get("fit_sha256")
            if made != prov["fit_sha256"]:
                problems.append(f"{js.name} was made with another fit ({str(made)[:12]}, this fit is "
                                f"{prov['fit_sha256'][:12]})")
                continue
            frames_.append(pd.read_csv(csv))
        if problems:
            raise Refused("the ladders are not usable: " + "; ".join(problems) + ". `ladder --task T` makes "
                          "<T>.csv, `ladder --inputs` makes inputs.csv")
        ladders = pd.concat(frames_, ignore_index=True)
        key = [c for c in ("signal", "domain", "step", "k0", "k") if c in ladders]
        twice = ladders[ladders.duplicated(key, keep=False)]
        if len(twice):
            raise Refused(f"the ladder files score {twice['signal'].iloc[0]} more than once")
        try:
            a, notes = ladder.build_assignment(list(specs.values()), {n: stats[n].compress for n in specs}, ladders,
                                               assign.targets())
        except ValueError as e:
            raise Refused(str(e)) from e
    _print_assignment(a, notes, specs)
    _prepare(stage1 / "assignments")                        # after every refusal above
    assign.save_assignment(a, out, notes)
    print(f"wrote {out}", flush=True)
    return 0


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

    p = sub.add_parser("ladder", help="score the rungs of a task's tokenizers, or the error of the input-only signals")
    p.add_argument("--task", default=None, choices=list(tasks.ALL_TASKS), help="the task whose outputs are laddered")
    p.add_argument("--inputs", action="store_true", help="rule 4: the signals no task predicts, from the cache")
    p.add_argument("--n-shots", type=int, default=100, help="val_select shots, spread evenly (default 100)")
    p.add_argument("--variant", default=None, help="the direct model's variant: also score the pass-through")
    p.add_argument("--gpu", type=int, default=None, help="the GPU index (with --variant)")
    p.add_argument("--share-gpu", action="store_true", help="use a GPU that has compute processes")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.add_argument("--stage0-dir", default=None, help="stage 0's directory, for the checkpoints")
    p.add_argument("--num-workers", type=int, default=32, help="window loader processes (default 32)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=64, help="cache reader processes for --inputs and the contrast")
    p.set_defaults(run=run_ladder_command)

    p = sub.add_parser("assign", help="choose K for every signal from the ladders, by the rules of the plan")
    p.add_argument("--name", required=True, help="the assignment's file name, without .json")
    p.add_argument("--from", dest="from_", default=None, help="an assignment to copy (with --raise-inputs)")
    p.add_argument("--raise-inputs", action="store_true", help="rule 4: raise each input-only signal one rung")
    p.add_argument("--reference", action="store_true", help="write the own-mean reference assignment")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.set_defaults(run=run_assign)
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
