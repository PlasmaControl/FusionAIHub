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
`<stage1>/assignments/<name>.json`; it refuses a name that exists, and ladders made with another fit,
other tokenizer code or other statistics (and with other scoring code, if the ladder file records it).
Every refusal comes before the first write, and no refusal leaves a directory behind.

`gate --assignment FILE` judges every (task, output signal) pair of the 14 tasks from stored scores: the
floors of `tok.<name>.<mode>` in the three read-outs and the own-mean floor of `tok.ref.unquantized` (the
score of ``<stage1>/assignments/ref.json``, written by `assign --reference --name ref`), each only if it is
kept with the current provenance, and the cache-side columns measured over `val_report` shots. It prints the
table and writes `<stage1>/gate/<name>.csv` and `.md` after every refusal; it exits 1 if a row blocks.

An assignment may name a learned recoder (plan 1c): a setting whose codec is `recoded` or `recoded_concat` and
whose `source` is a directory of ``<stage1>/recoders/`` (made by `train_tokenizer.py linear` or `train`). Every
sub-command builds its codecs through `learned.build`, and the provenance of such an assignment adds
`learned_code_sha256` and `sources_sha256` (the bytes of each recoder it names). `recode --name NEW --from OLD
--arm R[,R...]` writes `<stage1>/assignments/NEW.json` from OLD with the signals of each recoder behind it, the
values per frame kept (`--scale` sizes a concatenated recoder); it refuses a name that exists, a network that
collapsed or has no `result.json`, and a recoder made on another fit or statistics than the ones in use (so do
`score`, `gate` and `compare` for an assignment that names one). `score --instrument
floor` runs on the CPU for a network recoder too (one torch thread), so it still refuses `--gpu`. The gate reads
a recoded assignment through a copy of `gate.cache_side_job` that builds its codecs with `learned.build`, and
refuses a concatenated band codec, whose kept positions it cannot tell.

`compare --a NAME --b NAME` sets two assignments (`<stage1>/assignments/NAME.json`) side by side, per signal whose
setting differs (or `--signals`), as the paired gap of plan 1c's rule 1 with its 95% interval from 2,000 paired
resamples of the shots (`learned.compare`). A signal that a task predicts is read from the stored floors
`tok.<NAME>.unquantized` under `--label`, per shot, of the task with the smallest best published score, and only if
`kept_score` keeps them (otherwise the row says `not scored` and the `score` command that is missing is printed);
a signal no task predicts is read from the cache over the label's shots. It writes
`<stage1>/compare/<a>__<b>.<label>.csv`; `--drop-channel` and `--reach TARGET` add the cost of a missing channel and
the values per frame needed to reach TARGET, for a series signal with several channels read from the cache.
`seeds --pairs a0:b,a1:b,... --label L` reads the stored files of the pairs and prints, per signal, the gaps of the
seeds, their spread and the verdict of the first pair with that spread (rule 2).

Later sub-commands are added as one more sub-parser and one more function each. This file does
nothing at import time except define: `fit`, `ladder --inputs` and `gate` start spawned workers, and they
import it again.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
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
from tokamak_foundation_model.ignite.mast.tokenizers import fit, floor, gate, ladder
from tokamak_foundation_model.ignite.mast.tokenizers import learned
from tokamak_foundation_model.ignite.mast.tokenizers.learned import build as learned_build
from tokamak_foundation_model.ignite.mast.tokenizers.learned import compare as learned_compare
from tokamak_foundation_model.ignite.mast.tokenizers.learned import recode as learned_recode

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


def recoders_root() -> Path:
    """Where the recoders of an assignment are read from: ``<stage1>/recoders``."""
    return paths.stage1_dir() / "recoders"


_default_recoders_root = recoders_root                  # one definition; `provenance_of` names it as its default


def _names_a_recoder(assignment: Mapping[str, assign.Setting]) -> bool:
    return bool(learned_build.sources(assignment)) or any(
        s.codec in (learned_build.RECODED, learned_build.RECODED_CONCAT) for s in assignment.values())


def recoder_fit_mismatch(name: str, meta: Mapping[str, object], fit_sha256: str, stats_sha256: str) -> str | None:
    """Why a recoder does not belong to the fit and statistics in use, or None if it does.

    `meta` is the recoder's ``recoder.json`` meta. `train_tokenizer.py` records, under ``meta["store"]``, the
    `fit_sha256` and `stats_sha256` of the stores the recoder was fitted on; a network over a fit's PCA basis
    (net.profile, net.map) and every recoded codec (`learned.build` rebuilds the front from the current fit) read
    in the wrong basis after a `fit --force`. A hash that is not recorded is a reason too: the recoder cannot be
    tied to the fit (no data is not a match). The words name the recoder, the hash and both 8-character values."""
    recorded = meta.get("store") if isinstance(meta.get("store"), dict) else {}
    for key, now in (("fit_sha256", fit_sha256), ("stats_sha256", stats_sha256)):
        got = recorded.get(key)
        if not isinstance(got, str) or not got:
            return f"recoder {name} records no {key}, so it cannot be tied to the fit and statistics in use ({now[:8]})"
        if got != now:
            return (f"recoder {name}: its {key} is {got[:8]}, the {key} in use is {now[:8]}: it was made on another "
                    f"fit or statistics than these; make it again (`train_tokenizer.py linear` or `train`)")
    return None


def provenance_of(assignment_path: Path | None, fit_dir: Path | None, cache_root: Path,
                  checkpoint: Path | None = None, recoders_root: Path | None = None) -> dict[str, str]:
    """What a stored score must carry to be the score of the code and files in use now.

    In this order: `tokenizer_code_sha256`, `assignment_sha256`, `fit_sha256`, `stats_sha256` (these four only
    with an assignment), `learned_code_sha256` and `sources_sha256` (only for an assignment that names a
    recoder: the hash of `learned`'s modules and of the bytes of every recoder it names, read from
    `recoders_root`, default ``<stage1>/recoders`` (which `score` relies on); `FileNotFoundError` for a recoder
    that is not saved),
    `tokamark_revision` (from ``manifest.json`` of `cache_root`), `data_root`, and `checkpoint_sha256` (the
    sha256 of the file `checkpoint`). `ValueError` naming "statistics" if ``stats.npz``, the manifest and the
    fit's meta do not carry one hash, and `ValueError` (`recoder_fit_mismatch`) if a recoder it names was made on
    another fit or statistics than these: every command that scores or gates an assignment reads its provenance
    first, so none writes with such a recoder. Arrays are not read. An assignment that names no recoder gets exactly
    plan 1b's keys.
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
        assignment = assign.load_assignment(assignment_path)
        out = {"tokenizer_code_sha256": code_sha256(), "assignment_sha256": assign.assignment_sha256(assignment),
               "fit_sha256": fit.fit_sha256(Path(fit_dir)), "stats_sha256": stats_sha}
        if _names_a_recoder(assignment):
            root = Path(recoders_root) if recoders_root is not None else _default_recoders_root()
            out["learned_code_sha256"] = learned.code_sha256()
            out["sources_sha256"] = learned_build.sources_sha256(assignment, root)
            for source in learned_build.sources(assignment):    # a recoder must belong to the fit and statistics in use
                meta = json.loads((Path(root) / source / learned_recode.RECODER_JSON).read_text())["meta"]
                if (why := recoder_fit_mismatch(source, meta, out["fit_sha256"], out["stats_sha256"])) is not None:
                    raise ValueError(why)
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
    (written atomically), and `signals.csv` holds the per-signal NRMSE (TokaMark units). A stored `score.json`
    of `out_dir` is removed first, then `signals.csv` and `score.json` are written atomically in that order.
    """
    stored, _ = kept_score(out_dir, task.name, predictor, label, provenance, force)
    if stored is not None:
        return stored
    out_dir = _prepare(Path(out_dir))
    result = harness.score(task, predict_fn, out_dir, split=split, shots=shots,
                           history_s=getattr(predict_fn, "history_s", None),
                           batch_size=batch_size, num_workers=num_workers)
    (out_dir / "score.json").unlink(missing_ok=True)                  # the old pair goes first: a crash below
    _write_csv_atomic(out_dir / "signals.csv", result.per_signal)     # leaves no score.json (re-scored) or a
    path = baselines.write_score_json(result, predictor, label)       # matching pair; the gate trusts signals.csv
    record = json.loads(path.read_text())
    record.update(provenance)
    record.update(records)
    _write_json_atomic(path, record)
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


def _one_thread_for_networks(codecs: Mapping[str, object]) -> None:
    """Limit torch to one thread when a codec runs a network recoder (ruling C7: network floors run on the CPU,
    and a floor pass is not made faster by many threads over small tokens). torch is imported only then, here
    and never in the library."""
    if any(getattr(getattr(c, "recoder", None), "kind", None) == "net" for c in codecs.values()):
        import torch
        torch.set_num_threads(1)


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
    codecs = learned_build.build_codecs(signals, stats, fits, assignment, recoders_root())
    _one_thread_for_networks(codecs)
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
        # No `recoders_root` here: 1b's test_tok_command.py mocks `provenance_of` with a 3-4 positional-argument lambda
        # (the default root is `<stage1>/recoders`, the one this command reads).
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
            "stats_sha256": prov["stats_sha256"], "scoring_code_sha256": baselines.scoring_code_sha256(),
            "date": datetime.now(timezone.utc).isoformat(), **more}


def _score_windows(task: tasks.TaskSpec, batches: list[harness.WindowBatch], predict_fn: harness.PredictFn, *,
                   predictor: str, label: str, provenance: Mapping[str, str],
                   records: Mapping[str, object]) -> dict[str, float]:
    """Score `predict_fn` (physical units) on the kept `batches` and store it as `score.json` and `signals.csv` of
    ``<stage1>/scores/<task>/<predictor>/<label>/``, with `provenance` and `records` added. Returns the NRMSE
    per signal (TokaMark units). A stored score of that directory is replaced (a ladder does not keep scores)."""
    out_dir = _prepare(paths.stage1_dir() / "scores" / task.name / predictor / label)
    result = harness.score_batches(task.name, batches, predict_fn, out_dir)
    (out_dir / "score.json").unlink(missing_ok=True)                  # the old pair goes first, as in `score_one`
    _write_csv_atomic(out_dir / "signals.csv", result.per_signal)
    path = baselines.write_score_json(result, predictor, label)
    record = json.loads(path.read_text())
    record.update(provenance)
    record.update(records)
    _write_json_atomic(path, record)
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
        return learned_build.build_codecs(task.outputs, stats, fits, settings, recoders_root())

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


def _stale_ladder_keys(file_name: str, made: Mapping[str, object], prov: Mapping[str, str]) -> list[str]:
    """One message per key by which the ladder record `made` is not from the code and statistics in use now.

    `tokenizer_code_sha256` and `stats_sha256` (of `prov`) must equal the stored value. `scoring_code_sha256`
    is checked against `baselines.scoring_code_sha256()` only if the record has it: the seven ladders of the
    first run predate the key (`ladder` records it from now on)."""
    current = {"tokenizer_code_sha256": prov["tokenizer_code_sha256"], "stats_sha256": prov["stats_sha256"]}
    if "scoring_code_sha256" in made:
        current["scoring_code_sha256"] = baselines.scoring_code_sha256()
    return [f"{file_name} was made with other {key} ({str(made.get(key))[:12]}, now {value[:12]})"
            for key, value in current.items() if made.get(key) != value]


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
            made = json.loads(js.read_text())
            if made.get("fit_sha256") != prov["fit_sha256"]:
                problems.append(f"{js.name} was made with another fit ({str(made.get('fit_sha256'))[:12]}, this "
                                f"fit is {prov['fit_sha256'][:12]})")
                continue
            if stale := _stale_ladder_keys(js.name, made, prov):
                problems.extend(stale)
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


# --------------------------------------------------------------------------------------- recode

def run_recode(args: argparse.Namespace) -> int:
    """Write ``<stage1>/assignments/<name>.json`` from ``<from>.json`` with the signals of each recoder of
    `--arm` (a comma list of directories of ``<stage1>/recoders``) behind that recoder
    (`learned.build.recode_assignment`: the values per frame kept, `--scale` for a concatenated recoder,
    `--signals` to recode only some). The recoders' signals must not overlap after `--signals`. Prints, per
    changed signal, the values per frame before and after. Exit 2 for a refusal (before any write): among them a
    network recoder whose ``result.json`` is missing (its training did not finish) or does not say ``collapsed``
    false (rule 7, ruling C33), and a recoder whose recorded fit or statistics are not the ones in use
    (`recoder_fit_mismatch`). A linear recoder has no verdict. An existing name is never overwritten."""
    for label, value in (("--name", args.name), ("--from", args.from_.removesuffix(".json"))):
        if not _NAME.fullmatch(value):
            raise Refused(f"{label} {value!r} is not a file name (letters, digits, . _ -)")
    stage1 = paths.stage1_dir()
    out = stage1 / "assignments" / f"{args.name}.json"
    if out.exists():
        raise Refused(f"{out} exists: an assignment is not overwritten; choose another --name")
    old_path = stage1 / "assignments" / f"{args.from_.removesuffix('.json')}.json"
    if not old_path.is_file():
        raise Refused(f"there is no assignment {old_path}")
    sources = [a for a in args.arm.split(",") if a]
    if not sources or any(not _NAME.fullmatch(a) for a in sources):
        raise Refused(f"--arm {args.arm!r} must be recoder directory names, separated by commas")
    wanted = [n for n in args.signals.split(",") if n] if args.signals else None
    if not args.scale > 0:
        raise Refused(f"--scale must be positive, got {args.scale}")
    for a in sources:
        if not (recoders_root() / a / learned_recode.RECODER_JSON).is_file():
            raise Refused(f"there is no recoder {recoders_root() / a}: `train_tokenizer.py linear --arm {a}` "
                          "(or `train`) makes it")
    for a in sources:                                       # rule 7: a network that collapsed, or never got its verdict, is out
        run = recoders_root() / a
        try:
            kind = json.loads((run / learned_recode.RECODER_JSON).read_text())["kind"]
        except (OSError, KeyError, ValueError) as e:
            raise Refused(f"recoder {a} cannot be read ({type(e).__name__}: {e})") from e
        if kind != "net":
            continue
        if not (run / "result.json").is_file():
            raise Refused(f"recoder {a} has no result.json: its training did not finish, so it has no rule 7 verdict")
        try:
            doc = json.loads((run / "result.json").read_text())
            collapsed = doc["collapsed"]
        except (OSError, ValueError, TypeError) as e:
            raise Refused(f"recoder {a}: result.json cannot be read ({type(e).__name__}: {e})") from e
        except KeyError:
            raise Refused(f"recoder {a}: result.json has no `collapsed`, so the network has no rule 7 verdict") from None
        if collapsed is True:
            raise Refused(f"recoder {a} collapsed under rule 7 (C33): a collapsed tokenizer takes part in no comparison")
        if collapsed is not False:
            raise Refused(f"recoder {a}: result.json says collapsed {collapsed!r}, not false: no verdict, no comparison")
    try:
        in_use = _fit_provenance(stage1 / "fit" / FIT_VERSION, _cache_root(args))
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the fit and statistics in use cannot be read ({type(e).__name__}: {e})") from e
    try:
        base = assign.load_assignment(old_path)
        notes = dict(json.loads(old_path.read_text()).get("notes", {}))
        specs = {s.name: s for s in registry.all_signals()}
        loaded = [learned_recode.load_recoder(recoders_root() / a) for a in sources]
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"the inputs cannot be read ({type(e).__name__}: {e})") from e
    for a, got in zip(sources, loaded):                     # a recoder must belong to the fit and statistics in use
        if (why := recoder_fit_mismatch(a, got.meta, in_use["fit_sha256"], in_use["stats_sha256"])) is not None:
            raise Refused(why)
    owner: dict[str, str] = {}
    for a, got in zip(sources, loaded):
        for n in got.meta["signals"]:
            if wanted is not None and n not in wanted:
                continue
            if n in owner:
                raise Refused(f"signal {n} is in both recoder {owner[n]} and recoder {a}: one recoder per signal")
            owner[n] = a
    if wanted is not None and (missing := [n for n in wanted if n not in owner]):
        raise Refused(f"--signals names {missing}, which none of the recoders {sources} lists")
    if not owner:
        raise Refused(f"the recoders {sources} list no signal to recode")
    new = dict(base)
    try:
        for a, got in zip(sources, loaded):
            new = learned_build.recode_assignment(new, got, a, scale=args.scale, signals=wanted)
        counts = {n: (ladder.values_per_frame(specs[n], base[n]), ladder.values_per_frame(specs[n], new[n]))
                  for n in sorted(owner)}
    except (KeyError, ValueError) as e:
        raise Refused(str(e)) from e
    for n, (before, after) in counts.items():
        notes[n] = (f"recoded by {owner[n]}: {before} values per frame kept as {after} "
                    f"({base[n].codec} K {_k_text(base[n])} to {new[n].codec} K {_k_text(new[n])}; from {old_path.stem})")
        print(f"{n}: values per frame {before} -> {after}; {notes[n]}")
    _prepare(stage1 / "assignments")                        # after every refusal above
    assign.save_assignment(new, out, notes)
    print(f"wrote {out}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------ gate

GATE_SPLIT = "val_report"
NARROW_RANGE = 1.01                  # rule 6: `ranged` above this times `unquantized` means a range too narrow
REF_NAME = "ref"


def _stored_floors(stage1: Path, task: str, name: str, mode: str, label: str,
                   prov: Mapping[str, str]) -> tuple[dict[str, float | None] | None, float | None, str]:
    """(per-signal NRMSE, task NRMSE, reason it is missing) of the score of `tok.<name>.<mode>` stored under
    `label`, all in TokaMark units. Kept only as `kept_score` keeps it (with `prov`) and with a readable
    ``signals.csv``; a NaN is a missing number (None)."""
    predictor = predictor_name("floor", name, mode)
    out_dir = stage1 / "scores" / task / predictor / label
    stored, reason = kept_score(out_dir, task, predictor, label, prov, False)
    if stored is None:
        return None, None, reason
    try:
        per = pd.read_csv(out_dir / "signals.csv")
        got = {str(n): gate._num(v) for n, v in zip(per["feature_name"], per["NRMSE"], strict=True)}
        return got, gate._num(stored["nrmse"]), ""
    except (OSError, KeyError, ValueError) as e:
        return None, None, f"signals.csv cannot be read ({type(e).__name__}: {e})"


def _nrmse_text(record: Mapping[str, object] | None) -> str:
    """The ``nrmse`` of a stored score record with four decimals; ``no score`` for no record, or a value that
    is missing, not a number or not finite."""
    try:
        value = float(record["nrmse"])
    except (TypeError, ValueError, KeyError):
        return "no score"
    return f"{value:.4f}" if math.isfinite(value) else "no score"


def _narrow_range_lines(table_: pd.DataFrame) -> list[str]:
    """Plan 1b's rule 6: one line per (task, signal) whose `ranged` floor is more than 1% above its
    `unquantized` floor, which means that the range of the fit is too narrow. Reported, not fixed here: it
    changes no verdict. A row with either floor missing (NaN) has nothing to compare."""
    lines = []
    for row in table_.itertuples():
        floor_, ranged = row.unquantized, row.ranged
        if math.isfinite(floor_) and math.isfinite(ranged) and ranged > NARROW_RANGE * floor_:
            above = f"{100 * (ranged / floor_ - 1):.1f}% above" if floor_ > 0 else "above"
            lines.append(f"range too narrow (rule 6; reported, not fixed here): {row.task} {row.signal}: "
                         f"ranged {ranged:.4f} is {above} unquantized {floor_:.4f}")
    return lines


@functools.lru_cache(maxsize=64)
def _gate_codec_of(fit_dir: str, cache_root: str, recoders_dir: str, spec: tasks.SignalSpec,
                   setting: assign.Setting):
    """The codec of `setting` for `spec` from the fit, or None if the signal has no fitted data; built once in a
    process and kept. A copy of `gate._codec_of` (hashed, so it cannot change) that builds the codec with
    `learned_build.build_codec`, so that a setting naming a recoder is built too. A recoder is read once per
    process for all the signals that name it."""
    fits, stats = gate._fit_and_stats(fit_dir, cache_root)
    f = fits.get(spec.name)
    if f is None or f.n_examples == 0:
        return None
    codec = learned_build.build_codec(spec, stats[spec.name], f, setting, Path(recoders_dir),
                                      _GATE_LOADED.setdefault(recoders_dir, {}))
    _one_thread_for_networks({spec.name: codec})
    return codec


_GATE_LOADED: dict[str, dict] = {}       # per process: recoders_dir -> {source: learned_recode.Loaded}


def gate_cache_side_job(job: tuple) -> dict[str, dict | None]:
    """One shot of the cache-side pass, for `transform.map_shots`: ``job = (shot, cache_root, stats_sha256,
    fit_dir, specs, settings, recoders_dir)``. A copy of `gate.cache_side_job` (hashed, so it cannot change)
    that builds its codecs with `learned_build.build_codec` (through `_gate_codec_of`) and then calls the
    hashed `gate.shot_side`; the extra last item names the directory of the recoders. Returns, per signal
    name, `gate.shot_side` of the shot's cached values (model units), or None if the shot does not have the
    signal or the signal was not fitted."""
    shot, cache_root, sha, fit_dir, specs, settings, recoders_dir = job
    got = cache.read_shot(Path(cache_root) / "shots" / f"{shot}.h5", [s.name for s in specs], stats_sha256=sha)
    out: dict[str, dict | None] = {}
    for spec, setting in zip(specs, settings, strict=True):
        c = got[spec.name]
        codec = None if c is None else _gate_codec_of(str(fit_dir), str(cache_root), str(recoders_dir), spec, setting)
        out[spec.name] = None if codec is None else gate.shot_side(codec, spec, c.values)
    return out


def run_gate(args: argparse.Namespace) -> int:
    """Judge every (task, output signal) pair; print the table, a line per task, the scores that are not
    kept and the count line (``<n> rows, <m> not scored, <b> block``); write `<stage1>/gate/<name>.csv` and
    `.md`. Exit 2 for a refusal (before any write), 1 if a row blocks."""
    if args.n_shots < 1 or args.workers < 1:
        raise Refused("--n-shots and --workers are at least 1")
    if not _NAME.fullmatch(args.label):
        raise Refused(f"--label {args.label!r} is not a label (letters, digits, . _ -)")
    assignment_path = Path(args.assignment)
    name = assignment_path.name.removesuffix(".json")
    if not _NAME.fullmatch(name) or not assignment_path.is_file():
        raise Refused(f"there is no assignment {assignment_path} (a file named by letters, digits, . _ -)")
    cache_root = _cache_root(args)
    stage1 = paths.stage1_dir()
    stage0 = Path(args.stage0_dir) if args.stage0_dir else paths.stage0_dir()
    fit_dir = stage1 / "fit" / FIT_VERSION
    if not (fit_dir / "fit.npz").exists():
        raise Refused(f"there is no fit at {fit_dir}: run `fit` first")
    try:
        assignment = assign.load_assignment(assignment_path)
        task_specs = {t: tasks.load_task(t) for t in tasks.ALL_TASKS}
    except (OSError, KeyError, ValueError, ImportError) as e:
        raise Refused(f"the inputs cannot be read ({type(e).__name__}: {e})") from e
    outputs = {t: spec.outputs for t, spec in task_specs.items()}
    unique = {s.name: s for ss in outputs.values() for s in ss}
    unassigned = sorted(n for n in unique if n not in assignment)
    if unassigned:
        raise Refused(f"the assignment has no setting for {len(unassigned)} output signals; the first is {unassigned[0]}")
    for n, spec in sorted(unique.items()):         # ruling C2: the gate's kept-position mask reads one K per band
        if assignment[n].codec == learned_build.RECODED_CONCAT and assign.codec_kind(spec) == "band_dct":
            raise Refused(f"{n} is a spectrogram with a concatenated recoder ({assignment[n].source!r}); the gate "
                          "cannot tell which positions of a concatenated band codec are kept, so it judges only "
                          "per-stream band codecs")
    try:
        prov = provenance_of(assignment_path, fit_dir, cache_root, recoders_root=recoders_root())
        ref_path = stage1 / "assignments" / f"{REF_NAME}.json"
        ref_prov = (provenance_of(ref_path, fit_dir, cache_root, recoders_root=recoders_root())
                    if ref_path.is_file() else None)
    except (OSError, KeyError, ValueError, ImportError) as e:
        raise Refused(f"the inputs cannot be read ({type(e).__name__}: {e})") from e
    shots = cache.pick_evenly(tasks.split_shots(GATE_SPLIT), args.n_shots)
    gone = [s for s in shots if not (cache_root / "shots" / f"{s}.h5").exists()]
    if gone:
        raise Refused(f"{len(gone)} of the {len(shots)} shots have no cache file in {cache_root}; the first is {gone[0]}")

    if ref_prov is None:
        print(f"note: there is no {ref_path}, so the own-mean floor is missing: "
              f"`assign --reference --name {REF_NAME}`, then `score --assignment {ref_path} --mode unquantized`")
    started = time.monotonic()
    floors: dict[tuple[str, str], dict[str, float | None] | None] = {}
    task_nrmse: dict[tuple[str, str], float | None] = {}
    not_kept: list[str] = []
    absent = {read: 0 for read in gate.READS}
    for t in tasks.ALL_TASKS:
        for read in gate.READS:
            mode, who, p = (("unquantized", REF_NAME, ref_prov) if read == "ref" else (read, name, prov))
            got, whole, why = (None, None, "no score.json") if p is None else _stored_floors(stage1, t, who, mode,
                                                                                         args.label, p)
            floors[(t, read)], task_nrmse[(t, read)] = got, whole
            if got is None and why == "no score.json":
                absent[read] += 1
            elif got is None:
                not_kept.append(f"{t} {read}: {why}")

    fits, _ = fit.load_fit(fit_dir)
    specs = tuple(unique[n] for n in sorted(unique))
    settings = tuple(assignment[s.name] for s in specs)
    per_shot = list(transform.map_shots(gate_cache_side_job, shots, (str(cache_root), prov["stats_sha256"],
                                                                      str(fit_dir), specs, settings,
                                                                      str(recoders_root())), args.workers))
    stats = transform.load_stats(cache_root / "stats.npz")
    side: dict[str, dict] = {}
    loaded: dict[str, learned_recode.Loaded] = {}
    for s in specs:
        parts = [r[s.name] for r in per_shot if r[s.name] is not None]
        if s.name in fits and fits[s.name].n_examples > 0:
            codec = learned_build.build_codec(s, stats[s.name], fits[s.name], assignment[s.name], recoders_root(),
                                              loaded)
            side[s.name] = gate.combine_sides(codec, parts, len(per_shot) - len(parts))
        else:
            side[s.name] = gate.empty_side(len(per_shot))
    table_ = gate.gate_rows(outputs, assignment, floors, {t: tasks.best_published(t) for t in tasks.ALL_TASKS}, side)
    text = gate.to_markdown(table_)
    print(text)
    for t in tasks.ALL_TASKS:
        reads = ", ".join(f"{r} " + ("not scored" if task_nrmse[(t, r)] is None else f"{task_nrmse[(t, r)]:.4f}")
                          for r in gate.READS)
        s0 = []
        for pred in ("mean", "floor"):
            rec, _ = baselines.read_kept_score(stage0 / "scores" / t / pred / GATE_SPLIT, t, pred, GATE_SPLIT)
            s0.append(f"{pred} {_nrmse_text(rec)}")
        print(f"{t}: task NRMSE {reads}; a fifth of the best published {gate.FIDELITY * tasks.best_published(t):.4f}; "
              f"stage 0 on {GATE_SPLIT}: {', '.join(s0)}")
    for read, n in absent.items():
        if n:
            print(f"no stored score for {n} of the {len(tasks.ALL_TASKS)} tasks in read-out {read}")
    for line in _narrow_range_lines(table_):
        print(line)
    for line in not_kept:
        print(f"not kept, so not scored: {line}")
    n_rows = len(table_)
    n_unscored = int((table_["verdict"] == "not scored").sum())
    n_block = int(table_["verdict"].str.startswith("blocks").sum())
    out_dir = _prepare(stage1 / "gate")                      # after every refusal above
    _write_csv_atomic(out_dir / f"{name}.csv", table_)
    md = out_dir / f"{name}.md"
    tmp = md.with_name(f"{md.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text + "\n")
        os.replace(tmp, md)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"cache side: {len(shots)} {GATE_SPLIT} shots, {time.monotonic() - started:.0f} s in all")
    print(f"{n_rows} rows, {n_unscored} not scored, {n_block} block", flush=True)
    return 1 if n_block else 0


# ---------------------------------------------------------------------------------------- compare

COMPARE_LABEL = "val_select-n100"
REACH_FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0)          # the sizes of the reach curve, times the setting's values
FRONT_VALUES = 16                                     # the top rung of a series: values per frame and channel
_LABEL = re.compile(r"(val_select|val_report)(?:-n([1-9][0-9]*))?")
COMPARE_COLUMNS = ["signal", "source", "values_a", "values_b", "n_shots", "mean_a", "mean_b", "gap", "lo", "hi",
                   "gap_ba", "lo_ba", "hi_ba", "one_seed", "drop_a", "drop_b", "reach_a", "reach_b"]
_COMPARE_INTS = ("values_a", "values_b", "n_shots", "reach_a", "reach_b")


def _is_series_array(spec: tasks.SignalSpec) -> bool:
    """A series signal of several channels (a probe array): the signals whose missing channel and reach are read."""
    return assign.codec_kind(spec) == "frame_dct" and len(spec.values_shape) == 1 and spec.values_shape[0] > 1


def _reach_settings(spec: tasks.SignalSpec, setting: assign.Setting,
                    loaded: dict[str, learned_recode.Loaded]) -> dict[int, assign.Setting]:
    """``{values per frame: setting}`` at 1/8, 1/4, 1/2, 1 and 2 times the values of `setting` (the sizes that exist).

    A per-channel setting is sized by its K per channel (`learned.compare.scaled_k` with step 1 and limit 16, the
    top rung); a concatenated one by its values per frame, in whole tokens of the recoder, up to all its values
    (`loaded` keeps the recoders read)."""
    step, limit = 1, FRONT_VALUES
    if setting.codec == learned_build.RECODED_CONCAT:
        if setting.source not in loaded:
            loaded[setting.source] = learned_recode.load_recoder(recoders_root() / setting.source)
        step, limit = loaded[setting.source].recoder.n_val, loaded[setting.source].recoder.n_values
    out: dict[int, assign.Setting] = {}
    for factor in REACH_FACTORS:
        k = learned_compare.scaled_k(int(setting.k[0]), factor, step, limit)
        if k is not None:
            sized = dataclasses.replace(setting, k=(k,))
            out[ladder.values_per_frame(spec, sized)] = sized
    return out


def compare_cache_side_job(job: tuple) -> dict[str, dict[str, object]]:
    """One shot of the cache-side pass, for `transform.map_shots`: ``job = (shot, cache_root, stats_sha256, fit_dir,
    recoders_dir, plan)`` with ``plan`` a tuple of ``(spec, variants, channel)`` and ``variants`` a tuple of
    ``(key, setting, with_drop)``. Returns, per signal name and variant key, the round-trip RMSE of the shot in
    TokaMark units (`learned.compare.per_shot_cache`; None for a shot without the signal or a valid sample), and
    for a variant with `with_drop` also ``key + "!"``: `learned.compare.missing_stream_cost` of `channel`."""
    shot, cache_root, sha, fit_dir, recoders_dir, plan = job
    got = cache.read_shot(Path(cache_root) / "shots" / f"{shot}.h5", [spec.name for spec, _, _ in plan],
                          stats_sha256=sha)
    out: dict[str, dict[str, object]] = {}
    for spec, variants, channel in plan:
        c = got[spec.name]
        res: dict[str, object] = {}
        for key, setting, with_drop in variants:
            codec = None if c is None else _gate_codec_of(str(fit_dir), str(cache_root), str(recoders_dir), spec, setting)
            res[key] = (None if codec is None
                        else learned_compare.per_shot_cache(codec, spec, [(shot, c.values)]).get(shot))
            if with_drop:
                res[f"{key}!"] = (None if codec is None
                                  else learned_compare.missing_stream_cost(codec, spec, c.values, channel))
        out[spec.name] = res
    return out


def _compare_inputs(args: argparse.Namespace) -> tuple[Path, dict[str, Path], dict[str, dict[str, assign.Setting]],
                                                         dict[str, tasks.SignalSpec], list[str]]:
    """The checked inputs of `compare`: ``(stage1, {side: assignment path}, {side: assignment}, specs, signals)``.
    `Refused` for a bad option, an assignment that is not there, or a signal the assignments do not both have."""
    for flag, value in (("--a", args.a), ("--b", args.b), ("--label", args.label)):
        if not _NAME.fullmatch(value):
            raise Refused(f"{flag} {value!r} is not a name (letters, digits, . _ -)")
    if args.a == args.b:
        raise Refused(f"--a and --b are the same assignment, {args.a}")
    if not _LABEL.fullmatch(args.label):
        raise Refused(f"--label {args.label!r} is not <split> or <split>-n<N>, with a split of {STAGE1_SPLITS}")
    if args.workers < 1:
        raise Refused("--workers is at least 1")
    if args.reach is not None and not (math.isfinite(args.reach) and args.reach > 0):
        raise Refused(f"--reach is a positive error in TokaMark units, got {args.reach}")
    stage1 = paths.stage1_dir()
    files = {"a": stage1 / "assignments" / f"{args.a}.json", "b": stage1 / "assignments" / f"{args.b}.json"}
    for path in files.values():
        if not path.is_file():
            raise Refused(f"there is no assignment {path}")
    try:
        assignments = {side: assign.load_assignment(path) for side, path in files.items()}
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"an assignment cannot be read ({type(e).__name__}: {e})") from e
    specs = {s.name: s for s in registry.all_signals()}
    if args.signals:
        wanted = [n for n in args.signals.split(",") if n]
        unknown = [n for n in wanted if n not in specs]
        if unknown or not wanted:
            raise Refused(f"--signals names {unknown or args.signals!r}, which are not signals of the registry")
        lacking = [n for n in wanted if n not in assignments["a"] or n not in assignments["b"]]
        if lacking:
            raise Refused(f"--signals names {lacking}, which an assignment has no setting for")
        signals = list(dict.fromkeys(wanted))
    else:
        signals = [n for n in sorted(specs) if n in assignments["a"] and n in assignments["b"]
                   and assignments["a"][n] != assignments["b"][n]]
        if not signals:
            raise Refused(f"{args.a} and {args.b} have the same setting for every signal; --signals names signals to "
                          "compare anyway")
    return stage1, files, assignments, specs, signals


def run_compare(args: argparse.Namespace) -> int:
    """Compare assignments `--a` and `--b` signal by signal; print the table and write
    ``<stage1>/compare/<a>__<b>.<label>.csv``. Exit 2 for a refusal (before a score or the cache is read), 0
    otherwise, also when some rows are not scored."""
    stage1, files, assignments, specs, signals = _compare_inputs(args)
    m = _LABEL.fullmatch(args.label)
    split, n_asked = m.group(1), (int(m.group(2)) if m.group(2) else None)
    fit_dir = stage1 / "fit" / FIT_VERSION
    if not (fit_dir / "fit.npz").exists():
        raise Refused(f"there is no fit at {fit_dir}: run `fit` first")
    cache_root = _cache_root(args)
    try:
        predicted = assign.predicted_by()
        prov = {side: provenance_of(path, fit_dir, cache_root, recoders_root=recoders_root())
                for side, path in files.items()}
        values = {n: tuple(ladder.values_per_frame(specs[n], assignments[side][n]) for side in ("a", "b"))
                  for n in signals}
    except (OSError, KeyError, ValueError, ImportError) as e:
        raise Refused(f"the inputs cannot be read ({type(e).__name__}: {e})") from e
    source = {n: (min(predicted[n], key=tasks.best_published) if predicted.get(n) else "cache") for n in signals}
    from_cache = [n for n in signals if source[n] == "cache"]
    shots: list[int] = []
    if from_cache:
        every = tasks.split_shots(split)
        shots = cache.pick_evenly(every, n_asked) if n_asked is not None else sorted(int(s) for s in every)
        gone = [s for s in shots if not (cache_root / "shots" / f"{s}.h5").exists()]
        if gone:
            raise Refused(f"{len(gone)} of the {len(shots)} shots have no cache file in {cache_root}; the first is {gone[0]}")

    # the signals of a task: the stored floors, per shot
    started = time.monotonic()
    errors: dict[str, dict[str, dict[int, float] | None]] = {}
    missing: dict[tuple[str, str], str] = {}
    for n in signals:
        if source[n] == "cache":
            continue
        errors[n] = {}
        for side, name in (("a", args.a), ("b", args.b)):
            predictor = predictor_name("floor", name, "unquantized")
            out_dir = stage1 / "scores" / source[n] / predictor / args.label
            stored, why = kept_score(out_dir, source[n], predictor, args.label, prov[side], False)
            if stored is None:
                errors[n][side] = None
                missing[(source[n], name)] = why
            else:
                errors[n][side] = learned_compare.per_shot_harness(out_dir, source[n], n)

    # the signals no task predicts: the round trip of the cache, per shot, with the variants of the extras
    extras = [n for n in from_cache if _is_series_array(specs[n]) and (args.drop_channel or args.reach is not None)]
    curves: dict[str, dict[str, dict[int, str]]] = {}
    plan = []
    loaded: dict[str, learned_recode.Loaded] = {}
    try:
        for n in from_cache:
            spec = specs[n]
            with_extras = n in extras
            channel = int(np.prod(spec.values_shape, dtype=np.int64)) // 2
            variants = []
            curves[n] = {}
            for side in ("a", "b"):
                setting = assignments[side][n]
                variants.append((side, setting, bool(args.drop_channel and with_extras)))
                if with_extras and args.reach is not None:
                    curves[n][side] = {}
                    for vpf, sized in _reach_settings(spec, setting, loaded).items():
                        key = side if sized == setting else f"{side}:{vpf}"
                        curves[n][side][vpf] = key
                        if sized != setting:
                            variants.append((key, sized, False))
            plan.append((spec, tuple(variants), channel))
    except (OSError, KeyError, ValueError, ImportError) as e:
        raise Refused(f"a recoder cannot be read ({type(e).__name__}: {e})") from e
    per_shot: list[dict] = []
    if plan:
        per_shot = list(transform.map_shots(compare_cache_side_job, shots, (str(cache_root), prov["a"]["stats_sha256"],
                                                                            str(fit_dir), str(recoders_root()),
                                                                            tuple(plan)), args.workers))
    for spec, variants, _ in plan:
        n = spec.name
        errors[n] = {side: {s: r[n][side] for s, r in zip(shots, per_shot) if r[n][side] is not None}
                     for side in ("a", "b")}

    rows = []
    for n in signals:
        row = dict.fromkeys(COMPARE_COLUMNS)
        row.update(signal=n, source=source[n], values_a=values[n][0], values_b=values[n][1])
        ea, eb = errors[n]["a"], errors[n]["b"]
        if ea is None or eb is None:
            row["one_seed"] = "not scored"
        else:
            ab, ba = learned_compare.paired_gap(ea, eb), learned_compare.paired_gap(eb, ea)
            row.update(n_shots=ab["n_shots"], mean_a=ab["mean_a"], mean_b=ab["mean_b"], gap=ab["gap"], lo=ab["lo"],
                       hi=ab["hi"], gap_ba=ba["gap"], lo_ba=ba["lo"], hi_ba=ba["hi"],
                       one_seed=learned_compare.word(ab, ba))
            if n in extras and args.drop_channel:
                for side in ("a", "b"):
                    pairs = [r[n][f"{side}!"] for r in per_shot if r[n].get(f"{side}!") is not None]
                    if pairs:
                        before, after = (float(np.mean([p[i] for p in pairs])) for i in (0, 1))
                        row[f"drop_{side}"] = after / before if before > 0 else math.nan
            if n in extras and args.reach is not None:
                for side in ("a", "b"):
                    curve = {}
                    for vpf, key in curves[n][side].items():
                        vals = [r[n][key] for r in per_shot if r[n][key] is not None]
                        if vals:
                            curve[vpf] = float(np.mean(vals))
                    row[f"reach_{side}"] = learned_compare.values_to_reach(curve, args.reach)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=COMPARE_COLUMNS)
    for col in COMPARE_COLUMNS:
        if col in _COMPARE_INTS:
            frame[col] = frame[col].astype("Int64")                 # a number of values or shots, blank when missing
        elif col not in ("signal", "source", "one_seed"):
            frame[col] = frame[col].astype(float)
    shown = frame.copy()
    for col in _COMPARE_INTS:
        shown[col] = [("" if pd.isna(v) else str(int(v))) for v in frame[col]]
    for col in COMPARE_COLUMNS:
        if col not in _COMPARE_INTS and col not in ("signal", "source", "one_seed"):
            shown[col] = [("" if pd.isna(v) else f"{v:.4f}") for v in frame[col]]
    print(shown.to_string(index=False))
    if (args.drop_channel or args.reach is not None) and not extras:
        print("note: --drop-channel and --reach apply to a series signal of several channels that is read from the "
              "cache; none of the compared signals is one")
    for (task, name), why in sorted(missing.items()):
        shots_flag = f" --n-shots {n_asked}" if n_asked is not None else ""
        print(f"not scored: {task} {predictor_name('floor', name, 'unquantized')} {args.label} ({why}); "
              f"scripts/mast/py.sh scripts/mast/run_tokenizer.py score --task {task} --split {split}{shots_flag} "
              f"--assignment {stage1 / 'assignments' / (name + '.json')} --mode unquantized")
    out_dir = _prepare(stage1 / "compare")                  # after every refusal and every read above
    out = out_dir / f"{args.a}__{args.b}.{args.label}.csv"
    _write_csv_atomic(out, frame)
    print(f"wrote {out}; {len(frame)} signals, {time.monotonic() - started:.0f} s", flush=True)
    return 0


def run_seeds(args: argparse.Namespace) -> int:
    """From the stored comparison files of `--pairs` (``a:b`` names, comma-separated, at least two, one per seed
    of the network, `b` the same in each and every `a` different) print, per signal, the `n_shots` of each pair,
    the a-over-b gaps of the seeds and their spread, the b-over-a gaps and their spread, and the verdict of the
    first pair (rule 2: a gap above the spread over the seeds is above 0 in every seed). "a beats b" is read
    against the spread of the `gap`s, "b beats a" against that of the `gap_ba`s (largest minus smallest; the two
    differ, because the b-over-a gap is over the mean of a and the a-over-b gap over the mean of b):
    `learned_compare.word` takes one spread for both directions, so it is called once with each and each call is
    read for its own direction only. A signal whose pairs were measured over different numbers of shots gets no
    verdict ("shots differ: n1, n2, ...": a gap over other shots is not the same comparison), and one with a gap
    not measured gets "no data". Exit 2 for a refusal (a differing `b`, a repeated `a`, a missing file); nothing
    is written."""
    if not _NAME.fullmatch(args.label):
        raise Refused(f"--label {args.label!r} is not a label (letters, digits, . _ -)")
    pairs = [p for p in args.pairs.split(",") if p]
    names = [p.split(":") for p in pairs]
    if len(pairs) < 2 or any(len(n) != 2 or not all(_NAME.fullmatch(x) for x in n) for n in names):
        raise Refused(f"--pairs {args.pairs!r} must be at least two pairs a:b of assignment names, comma-separated")
    if len({b for _, b in names}) != 1:
        raise Refused(f"the pairs must all have the same b (one baseline); --pairs has {sorted({b for _, b in names})}")
    if len({a for a, _ in names}) != len(names):
        repeated = sorted({a for a, _ in names if [x for x, _ in names].count(a) > 1})
        raise Refused(f"the a of each pair is one seed of the network, so the a's must differ; {repeated} repeat")
    stage1 = paths.stage1_dir()
    files = [stage1 / "compare" / f"{a}__{b}.{args.label}.csv" for a, b in names]
    gone = [str(f) for f in files if not f.is_file()]
    if gone:
        raise Refused(f"no comparison file {', '.join(gone)}: `compare --a A --b B --label {args.label}` writes it")
    try:
        frames_ = [pd.read_csv(f).set_index("signal") for f in files]
    except (OSError, KeyError, ValueError) as e:
        raise Refused(f"a comparison file cannot be read ({type(e).__name__}: {e})") from e

    def show(values: list[float]) -> str:
        return ", ".join("nan" if not math.isfinite(g) else f"{g:.4f}" for g in values)

    for signal in frames_[0].index:
        rows = [f.loc[signal] if signal in f.index else None for f in frames_]
        gaps = [math.nan if r is None else float(r["gap"]) for r in rows]
        gaps_ba = [math.nan if r is None else float(r["gap_ba"]) for r in rows]
        shots = [math.nan if r is None else float(r["n_shots"]) for r in rows]
        spread, spread_ba = learned_compare.seed_spread(gaps), learned_compare.seed_spread(gaps_ba)
        shots_text = ", ".join("nan" if not math.isfinite(n) else str(int(n)) for n in shots)
        if rows[0] is None or not all(math.isfinite(g) for g in (*gaps, *gaps_ba)):
            verdict = "no data"
        elif len(set(shots)) != 1:
            verdict = f"shots differ: {shots_text}"
        else:
            ab = {k: float(rows[0][k]) for k in ("gap", "lo", "hi")}
            ba = {k: float(rows[0][f"{k}_ba"]) for k in ("gap", "lo", "hi")}
            word_a = learned_compare.word(ab, ba, spread)               # read for "a beats b" only
            word_b = learned_compare.word(ab, ba, spread_ba)            # read for "b beats a" only
            verdict = word_a if word_a.startswith("a ") else word_b if word_b.startswith("b ") else (
                "no data" if word_a == "no data" else "tie")
        print(f"{signal}: n_shots {shots_text}; gaps {show(gaps)}; spread {spread:.4f}; "
              f"gaps b over a {show(gaps_ba)}; spread {spread_ba:.4f}; {verdict}")
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

    p = sub.add_parser("recode", help="an assignment with the signals of learned recoders behind their recoders")
    p.add_argument("--name", required=True, help="the new assignment's file name, without .json")
    p.add_argument("--from", dest="from_", required=True, help="the assignment to copy, in <stage1>/assignments")
    p.add_argument("--arm", required=True, help="recoder directories of <stage1>/recoders, comma-separated")
    p.add_argument("--scale", type=float, default=1.0, help="a concatenated recoder keeps this part of the values")
    p.add_argument("--signals", default=None, help="recode only these signals (comma-separated)")
    p.add_argument("--cache-dir", default=None, help="the cache whose statistics are in use (default <root>/cache/v1)")
    p.set_defaults(run=run_recode)

    p = sub.add_parser("gate", help="judge every (task, signal) pair from the stored floors and the cache")
    p.add_argument("--assignment", required=True, help="the assignment file to judge")
    p.add_argument("--label", default=GATE_SPLIT, help="the label the scores are stored under (default val_report)")
    p.add_argument("--n-shots", type=int, default=200, help="val_report shots for the cache side (default 200)")
    p.add_argument("--workers", type=int, default=64, help="cache reader processes (default 64)")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.add_argument("--stage0-dir", default=None, help="stage 0's directory, for the mean and floor scores")
    p.set_defaults(run=run_gate)

    p = sub.add_parser("compare", help="two assignments side by side per signal: the paired gap of rule 1")
    p.add_argument("--a", required=True, help="the first assignment's name, in <stage1>/assignments")
    p.add_argument("--b", required=True, help="the second assignment's name; the gap is (mean b - mean a) / mean b")
    p.add_argument("--label", default=COMPARE_LABEL, help="the split and subset of the scores and shots (default val_select-n100)")
    p.add_argument("--signals", default=None, help="compare these signals (comma-separated) instead of those that differ")
    p.add_argument("--drop-channel", action="store_true", help="the cost of a missing channel (series arrays, from the cache)")
    p.add_argument("--reach", type=float, default=None, help="values per frame needed to reach this error (TokaMark units)")
    p.add_argument("--workers", type=int, default=64, help="cache reader processes (default 64)")
    p.add_argument("--cache-dir", default=None, help="the cache (default <root>/cache/v1)")
    p.set_defaults(run=run_compare)

    p = sub.add_parser("seeds", help="the gaps of the seeds of one network against one baseline, and their spread")
    p.add_argument("--pairs", required=True, help="a0:b,a1:b,a2:b: the compared assignments, one pair per seed")
    p.add_argument("--label", default=COMPARE_LABEL, help="the label of the comparison files (default val_select-n100)")
    p.set_defaults(run=run_seeds)
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
