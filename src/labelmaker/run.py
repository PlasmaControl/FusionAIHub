"""The labelmaker CLI: `python -m labelmaker.run <stage>`.

Stages, each independently rerunnable, with a per-shot HDF5 file
between the first two:

    features  resolve the union of canonical features the requested models
              need, per shot, into <root>/features/<shot>_features.h5
    infer     build each model's inputs from that file, predict, and write
              <root>/labels/<shot>_labels.h5
    validate  score adapter fidelity, reconstruction fidelity and label
              quality (see `validate.py`'s module docstring), write their
              JSON reports under <root>/validation/<slug>/, and fold the
              headline numbers into the model card's `model-index`
    all       features, then infer, then validate
    events    a different pipeline over the same shots and no model at all:
              run the pinned TokEye U-Net over the planned spectrogram
              channels, and write <root>/masks/<shot>_masks.npz and the
              discrete things every detector, heuristic and logbook entry
              claims into <root>/events/<shot>_events.parquet
              (see `events/pipeline.py`)

Every shot is isolated: one try/except and one SIGALRM timeout per shot, so
a corrupt HDF5 or a hung read costs one shot and not the run (IGNITE
measured ~56 of 3,000 corpus shots hanging on reads). Workers return summary
rows; the parent writes the parquet index, because many processes appending
to one parquet file would race.

The worker pool is forked before the first fetch and before toksearch is
imported anywhere: toksearch_d3d's ptserver reader is not fork-safe. This is
why `_resolve_one_source` defers its `resolve_fdp` import into the function
body, and why `resolve_fdp` in turn defers its own toksearch imports. Both
halves are checked by `test_run.py`.

Two things this stage reports that are easy to miss:

* A `partial` shot is the normal case off the archive, not a failure. Only
  the archive carries `ech_rho`, only the corpus carries `pinj_total` and
  `tinj_total`, so a shot outside the archive's 5,000 reaches at best 15 of
  the 16 canonical features the tearing model wants, and fewer when the
  corpus' actuator groups are the absent-signal sentinel. The summary
  counts `ok` and `partial` separately for exactly this reason.

* A shot whose features came from more than one source is flagged `mixed`.
  Features are resolved cheapest-source-first *per feature*, so one shot's
  file can legitimately hold archive rows beside corpus or fdp rows - and
  the archive's row k is stamped 25 ms later than the interval it actually
  averages (measured to 3.9e-08; see `resolve_archive`'s docstring). Task 15
  resolved the resulting one-`dt` misalignment at the point model inputs are
  assembled (`InputSpec.build` samples each field per the resolver that
  produced it, `ns.SAMPLING_BY_SOURCE`), so it is no longer a live
  correctness concern for anything that reads features through `build()`.
  It is still recorded per shot in the run log and aggregated into
  `runs/<run_id>/summary.json`, as provenance: which sources served which
  features is useful on its own, and a consumer reading the raw feature
  file directly (bypassing `build()`) still meets the offset.
"""
from __future__ import annotations

import json
import math
import os
import platform
import signal
import socket
import sys
import time
from argparse import ArgumentParser
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import __version__, analyze
from .catalog import corpus_shots, overlap_shots, read_shot_file, sample_shots
from .config import Paths, git_sha, sha256_of
from .features import namespace as ns
from .features import resolve_archive, resolve_corpus
from .features.store import (
    FeatureArray,
    is_complete,
    missing_names,
    permanent_names,
    present,
    read_feature,
    resolvers,
    write_features,
)
from .labels.schema import artifact_digest, specs_for
from .labels.store import (
    append_index,
    index_rows,
    labelled,
    read_label,
    write_labels,
)
from .models import registry

STAGES = ("features", "infer", "validate", "all", "analyze", "events")

#: Stages that need no model at all. `events` reads the corpus and one
#: pinned U-Net checkpoint and writes masks and event rows; `--models` is a
#: flag it could do nothing with, so it is not required and not accepted.
MODEL_FREE_STAGES = ("events",)

#: Exit codes. Anything non-zero means no labels should be trusted from this
#: run; 3 and 4 mean nothing ran at all.
EXIT_OK = 0
EXIT_NO_SHOTS = 1
#: 2 is argparse's own usage error.
EXIT_UNVERIFIED_WEIGHTS = 3
EXIT_BAD_MODEL = 4
#: Not 2: argparse already owns that code, and a caller testing `$? == 2`
#: would then confuse "you passed bad flags" with
#: "the evaluator disagrees with the framework it is supposed to match",
#: which is the single most important failure this package can report.
EXIT_FIDELITY_FAILED = 5
#: Distinct from EXIT_FIDELITY_FAILED: set when one or more validation
#: reports raised (recorded as `{"error": ...}`) but none failed fidelity
#: outright, which keeps the more severe verdict. Without it a validate run
#: whose every report errored printed the same lines a success prints and
#: exited 0 - `fidelity.get("passed") is False` is false for an error dict
#: just as it is for a real pass.
EXIT_VALIDATE_ERRORED = 6
#: `--databases-only`: the `tables.yaml` manifest, or a CSV it names, cannot
#: be believed. Run-level like a bad model and for the same reason - it is
#: one file, it is wrong for every shot, and discovering it per shot would
#: bury the cause under N identical errors.
EXIT_BAD_LABEL_TABLE = 7

#: Faults that make a requested model unusable before any shot is touched:
#: a scaffold spec (NotImplementedError), a spec with no ADAPTER
#: (AttributeError), a card that is absent (OSError) or malformed
#: (ValueError, KeyError). All of them are run-level, not per-shot.
_MODEL_FAULTS = (
    AttributeError,
    ImportError,
    KeyError,
    NotImplementedError,
    OSError,
    ValueError,
)


class StageTimeout(Exception):
    """A single shot exceeded its time budget."""


@contextmanager
def time_limit(seconds: int):
    """SIGALRM guard. Works in each worker, which is its process' main thread."""

    def handler(signum, frame):
        raise StageTimeout(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class RunContext:
    """Everything a worker needs, picklable and read-only.

    Deliberately holds model *slugs* and not loaded adapters: an adapter
    carries a `load` callable, which in general is a closure and would make
    this unpicklable. Each worker loads what it needs once, in
    `_predictor`.
    """

    paths: Paths
    archive_files: tuple[Path, ...]
    models: tuple[str, ...]
    run_id: str
    timeout_s: int
    force: bool


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m labelmaker.run",
        description="Run trained models over the FAITH shot corpus.",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--models", nargs="+", metavar="SLUG",
                        help="required by every stage but analyze, which takes "
                             "its models from --config")
    parser.add_argument("--config", type=Path, default=analyze.DEFAULT_CONFIG,
                        help="analyze: the labels and context to produce")
    parser.add_argument("--out", type=Path, default=None,
                        help="analyze: where <shot>/<shot>_analysis.json and "
                             "<shot>_labels.png go (default <root>/analysis)")
    picker = parser.add_mutually_exclusive_group(required=True)
    picker.add_argument("--shots", nargs="+", type=int, metavar="SHOT")
    picker.add_argument("--shot-file", type=Path)
    picker.add_argument("--corpus", action="store_true",
                        help="every shot in the corpus")
    picker.add_argument("--overlap", action="store_true",
                        help="shots present in both the corpus and the "
                             "tearing-mode training archive")
    parser.add_argument("--sample", type=int, default=0,
                        help="take a seeded sample of the selected shots")
    parser.add_argument("--limit", type=int, default=0,
                        help="keep only the first N of the selected shots, "
                             "after --sample; 0 means all")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds per shot per stage")
    parser.add_argument("--force", action="store_true",
                        help="redo shots that are already complete")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--archive", nargs="+", type=Path, default=None)
    events = parser.add_argument_group(
        "events", "the TokEye mask run: masks/<shot>_masks.npz and "
                  "events/<shot>_events.parquet"
    )
    events.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    events.add_argument("--tile-batch", type=int, default=32,
                        help="512-column tiles per forward pass; about "
                             "memory, not speed (measured flat 8-32)")
    events.add_argument("--amp", action="store_true",
                        help="fp16 autocast; CUDA only, ignored on cpu")
    events.add_argument("--norm", default="record", choices=list(pipeline_norms()),
                        help="record: z-score over the whole record, as the "
                             "AE path; plasma: over the fast groups' "
                             "coverage intersection")
    events.add_argument("--passes", nargs="+", default=list(masks_pass_names()),
                        choices=list(masks_pass_names()))
    events.add_argument("--unet", type=Path, default=None,
                        help="checkpoint; default "
                             "<root>/models/tokeye/big_tf_unet_251210.pt")
    events.add_argument("--run-id", default=None,
                        help="name this run instead of "
                             "<stage>-<timestamp>-<pid>")
    events.add_argument("--databases-only", action="store_true",
                        help="ingest the curated label tables "
                             "(data/labels/tables.yaml) over the shot list "
                             "and write nothing else: no corpus read, no "
                             "U-Net, no masks. Seconds over 10,000 shots, "
                             "and the only way a table reaches a shot whose "
                             "corpus file we do not have")
    events.add_argument("--rules-only", action="store_true",
                        help="run only the steps that read the features "
                             "store and the curated tables - the Ip "
                             "flat-top, the q-min regime rule, the label "
                             "tables - and write nothing else: no corpus "
                             "read, no U-Net, no masks. Minutes over the "
                             "500-shot recommender_v1 list on a login node")
    events.add_argument("--refresh-text", action="store_true",
                        help="re-ask the logbook about the shots recorded in "
                             "text/logs_subset.missing; a miss is a fact "
                             "about the source at the time, not forever")
    return parser


def masks_pass_names() -> tuple[str, ...]:
    """`("wide", "zoom")` without importing `events.masks` (and torch).

    `build_parser` is called by `--help` and by every test that parses
    argv, and `events.masks` imports torch at module scope. The values are
    pinned against their owners by `test_events_pipeline.py`.
    """
    return ("wide", "zoom")


def pipeline_norms() -> tuple[str, ...]:
    """`("record", "plasma")`, for the same reason as `masks_pass_names`."""
    return ("record", "plasma")


def shot_list(args, paths: Paths) -> list[int]:
    if args.shots:
        shots = sorted(set(args.shots))
    elif args.shot_file:
        shots = read_shot_file(args.shot_file)
    elif args.corpus:
        shots = corpus_shots(paths)
    else:
        shots = overlap_shots(paths)
    if args.sample:
        shots = sample_shots(shots, args.sample, args.seed)
    # After the sample, so `--sample 200 --limit 5` is five of the same two
    # hundred a full run would do and not five of a different draw.
    if getattr(args, "limit", 0):
        shots = shots[:int(args.limit)]
    return shots


def _feature_names(adapters) -> list[str]:
    """The union of canonical features the requested models consume."""
    names: list[str] = []
    for adapter in adapters:
        for name in adapter.input_spec.canonical_names:
            if name not in names:
                names.append(name)
    return names


def _resolve_one_source(source, shot, want, ctx):
    if source == "archive":
        return resolve_archive.resolve(shot, want, files=tuple(ctx.archive_files))
    if source == "corpus":
        return resolve_corpus.resolve(shot, want, corpus=ctx.paths.corpus)
    if source == "events":
        # Deferred like `resolve_fdp` below, and for a second reason: it
        # imports `events.windows`, which imports torch through
        # `events.masks`, and a features run that wants no window feature
        # must not pay for that.
        from .features import resolve_events

        return resolve_events.resolve(shot, want, paths=ctx.paths)
    from .features import resolve_fdp  # deferred: keeps the parent fork-safe

    return resolve_fdp.resolve(shot, want)


def _features_row(shot: int, path, names, status: str) -> dict:
    """One shot's features-stage summary, read back from the written file.

    Read back rather than assembled from whatever this call resolved, for
    two reasons: `write_features` can demote a feature into `missing` after
    the fact, and a merge carries earlier runs' results forward - so a
    locally computed row would understate a rerun and misreport a demotion.
    Every status shares this shape, `skipped` included, so a consumer
    reading `log.txt` never has to special-case one.
    """
    wanted = set(names)
    served = {n: r for n, r in resolvers(path).items() if n in wanted}
    used = sorted(set(served.values()))
    return {
        "shot": shot,
        "status": status,
        "resolved": len(served),
        "missing": {n: c for n, c in missing_names(path).items() if n in wanted},
        "resolvers": served,
        "sources": used,
        # See this module's docstring: one whole dt of misalignment between
        # features in the same row, invisible in the numbers themselves.
        "mixed": len(used) > 1,
    }


def features_for_shot(shot: int, names, ctx: RunContext) -> dict:
    """Resolve every requested feature for one shot, cheapest source first."""
    path = ctx.paths.features_file(shot)
    if not ctx.force and is_complete(path, names):
        return _features_row(shot, path, names, "skipped")
    # A permanently-missed name is as settled as a stored one: an absent
    # archive column and an absent MDSplus node will be absent again. Only
    # names that are neither stored nor permanently missed are retried, so
    # one transient timeout does not re-fetch the whole shot.
    known: set[str] = set()
    if not ctx.force:
        known = present(path) | permanent_names(path)
    todo = [n for n in names if n not in known]
    arrays: dict[str, FeatureArray] = {}
    causes: dict[str, list[str]] = {}
    timed_out = False
    for source in ns.SOURCES:
        want = [
            n for n in todo
            if source in ns.by_name(n).sources and n not in arrays
        ]
        if not want:
            continue
        try:
            got, missed = _resolve_one_source(source, shot, want, ctx)
        except StageTimeout:
            # MUST be handled before the generic clause below, which would
            # otherwise swallow it. SIGALRM is one-shot: an alarm absorbed
            # into a per-source miss leaves every remaining source running
            # with no budget at all, so a shot that hangs on the second
            # source hangs forever - the exact failure the per-shot timeout
            # exists to prevent. Measured: with the alarm swallowed, a
            # resolver sleeping past the budget produced a `partial` shot
            # and slept again for each remaining source.
            #
            # Stop trying sources, but keep what the earlier ones produced
            # and record a transient miss for the rest, so the next run
            # picks up exactly the remainder instead of refetching a shot
            # that will hang on the same source again.
            got, missed, timed_out = {}, dict.fromkeys(want, "StageTimeout"), True
        except Exception as exc:  # noqa: BLE001 - per-source isolation
            # A resolver raising is one source failing, not the shot failing:
            # the remaining sources must still be tried, and the cause is
            # recorded so the next run knows whether to retry. Nothing here
            # can enumerate what h5py, MDSplus or ptserver raise.
            got, missed = {}, {n: type(exc).__name__ for n in want}
        arrays.update(got)
        for name, cause in missed.items():
            causes.setdefault(name, []).append(f"{source}:{cause}")
        if timed_out:
            break
    missing = {n: ",".join(c) for n, c in causes.items() if n not in arrays}
    write_features(path, shot, arrays, missing, merge=not ctx.force)
    row = _features_row(shot, path, names, "ok")
    if timed_out:
        # A timeout gets its own status rather than hiding inside `partial`:
        # it is the one incomplete outcome that says nothing about the data.
        row["status"] = "timeout"
    elif row["missing"]:
        row["status"] = "partial"
    return row


_PREDICTORS: dict[str, tuple] = {}


def _predictor(slug: str, ctx: RunContext):
    """Load a model once per worker process, not once per shot.

    The card is parsed here too: `specs_for` needs the artifact digest, and
    re-reading a YAML front matter for each of 16,909 shots is pure waste.
    """
    if slug not in _PREDICTORS:
        model_dir = ctx.paths.models / slug
        # Checked again here, inside the worker, even though `main` verified
        # every model before forking: this catches an artifact swapped out
        # mid-run.
        registry.verify_artifacts(slug, model_dir)
        adapter = registry.load_adapter(slug)
        sha_map = (registry.read_card(slug)["labelmaker"].get("upstream") or {}).get(
            "sha256"
        ) or {}
        predict = adapter.load(model_dir)
        adapter = replace(adapter, output_spec=getattr(predict, "output_spec", adapter.output_spec))
        specs = specs_for(adapter, artifact_digest(sha_map))
        _PREDICTORS[slug] = (adapter, predict, specs)
    return _PREDICTORS[slug]


def _build_inputs(features_path, adapter):
    """This model's input arrays for one shot, from its stored features."""
    stored = present(features_path)
    features = {
        name: read_feature(features_path, name)
        for name in adapter.input_spec.canonical_names
        if name in stored
    }
    return adapter.input_spec.build(features, ns.GRID_S)


def infer_for_shot(shot: int, slug: str, ctx: RunContext) -> dict:
    """Build inputs, predict, and write one model's labels for one shot."""
    features_path = ctx.paths.features_file(shot)
    labels_path = ctx.paths.labels_file(shot)
    if not features_path.exists():
        return {"shot": shot, "status": "no-features"}
    adapter, predict, specs = _predictor(slug, ctx)
    wanted = {f"{slug}/{f.name}" for f in adapter.output_spec.fields}
    if not ctx.force and wanted <= labelled(labels_path):
        return {"shot": shot, "status": "skipped"}
    built = _build_inputs(features_path, adapter)
    members = predict(built)
    decoded = adapter.output_spec.decode(members)
    write_labels(
        labels_path,
        shot,
        built.t,
        decoded,
        specs,
        built.valid,
        run_id=ctx.run_id,
        features_sha256=sha256_of(features_path),
    )
    used = sorted(set(built.resolvers.values()))
    return {
        "shot": shot,
        "status": "ok",
        "n_valid": int(np.asarray(built.valid).sum()),
        "n_total": int(built.t.size),
        "missing_inputs": list(built.missing),
        # Rows each rule alone rejected - the answer to "why is this shot 95%
        # invalid" (on the 2024 tearing shots: ECH power with no location).
        "invalid_reasons": dict(built.invalid_reasons),
        # Per-input provenance, as the model actually saw it. `mixed` is the
        # one that matters: see this module's docstring.
        "resolvers": dict(built.resolvers),
        "sources": used,
        "mixed": len(used) > 1,
        # This model's rows only. `index_rows` reports every label in the
        # file, so an unfiltered list would re-append another model's rows
        # once per model in the run.
        "rows": [r for r in index_rows(labels_path) if r["slug"] == slug],
    }


def _guarded(fn, shot, ctx, *args):
    started = time.monotonic()
    try:
        with time_limit(ctx.timeout_s):
            row = fn(shot, *args, ctx)
    except Exception as exc:  # noqa: BLE001 - per-shot isolation
        # The whole point of the stage: one shot's corrupt file, hung read or
        # exhausted budget costs that shot and nothing else.
        row = {"shot": shot, "status": "error", "error": type(exc).__name__,
               "detail": str(exc)[:200]}
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def _features_worker(payload):
    shot, names, ctx = payload
    return _guarded(features_for_shot, shot, ctx, names)


def _infer_worker(payload):
    shot, slug, ctx = payload
    return _guarded(infer_for_shot, shot, ctx, slug)


def analyze_for_shot(shot: int, cfg, out_dir, ctx: RunContext) -> dict:
    """Features, then inference, then the JSON summary and figure for one shot.

    Both earlier stages skip work that is complete, so on a shot that has
    already been run this is a read of the label file and a plot.
    """
    adapters = {slug: _predictor(slug, ctx)[0] for slug in cfg.slugs}
    names = _feature_names(adapters.values())
    names += [c for c in cfg.context if c not in names]
    frow = features_for_shot(shot, names, ctx)
    for slug in cfg.slugs:
        infer_for_shot(shot, slug, ctx)
    features_path = ctx.paths.features_file(shot)
    labels_path = ctx.paths.labels_file(shot)
    # Rebuilt here rather than taken from `infer_for_shot`'s row, which is
    # empty when inference was skipped as already complete.
    provenance = {}
    for slug, adapter in adapters.items():
        built = _build_inputs(features_path, adapter)
        provenance[slug] = {
            "invalid_reasons": dict(built.invalid_reasons),
            "resolvers": dict(built.resolvers),
            "missing_inputs": list(built.missing),
        }
    # Deferred like every other framework-specific import in this module:
    # `validate` loads torch at module scope, and this runs inside a worker.
    from . import validate as validation

    truth = validation.archived_truth(shot, ctx.paths)
    have = labelled(labels_path)
    summaries = {}
    for label in cfg.labels:
        slug, name = label.split("/", 1)
        if label in have:
            summaries[label] = analyze.summarize_label(
                labels_path, slug, name, threshold=cfg.threshold_for(label),
                infer_row=provenance[slug], truth=truth,
            )
            series = read_label(labels_path, slug, name)
            series_valid = read_label(labels_path, slug, f"{name}_valid")
            summaries[label]["truth"] = validation.score_against_truth(
                label, series.y[0], series_valid.y[0].astype(bool), truth,
                threshold=summaries[label]["threshold"],
            )
        else:
            summaries[label] = analyze.empty_summary(
                provenance[slug], note=f"{slug} wrote no labels for this shot"
            )
            summaries[label]["truth"] = {"scored": False, "reason": "no labels"}
    shot_dir = Path(out_dir) / str(shot)
    shot_dir.mkdir(parents=True, exist_ok=True)
    png = analyze.plot_shot(
        shot, analyze.panels_for(features_path, labels_path, cfg, truth=truth),
        shot_dir / f"{shot}_labels.png",
        title_ids=[adapters[s].card_id for s in cfg.slugs],
    )
    summary = {
        "shot": shot,
        "config": cfg.as_dict(),
        "truth": {k: v for k, v in truth.items()
                  if k in ("available", "reason", "onset_s", "n_rows")},
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": ctx.run_id,
        "labels": summaries,
        "labels_file": str(labels_path),
        "features_file": str(features_path),
        "features": {k: frow.get(k) for k in ("status", "missing", "resolvers")},
        "figure": str(png),
    }
    out_json = shot_dir / f"{shot}_analysis.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    n_rows = sum(v["n_rows"] for v in summaries.values())
    return {
        "shot": shot,
        "status": "ok" if n_rows else "no-labels",
        "n_labels": sum(1 for v in summaries.values() if v["n_rows"]),
        "valid_fraction": {k: v["valid_fraction"] for k, v in summaries.items()},
        "summary": str(out_json),
        "figure": str(png),
    }


def _analyze_worker(payload):
    shot, cfg, out_dir, ctx = payload
    return _guarded(analyze_for_shot, shot, ctx, cfg, out_dir)


def events_stage(shots, ctx: RunContext, args) -> tuple[list[dict], dict]:
    """The `events` stage: one U-Net, one logbook pass, then shot by shot.

    Sequential on purpose. The other stages fan out over a `Pool`, and this
    one must not: the network is a 7.8M-parameter torch module that would be
    pickled to every worker (or, on CUDA, inherited across a fork into a
    context that cannot use it), and the work per shot is already a batched
    GPU job whose parallelism lives inside `masks.infer`. `--workers` is
    ignored here and the SLURM script parallelises over shot CHUNKS instead
    (spec A5).

    The logbook subset is built ONCE, here, before the loop: `logs.jsonl` is
    616 MB and `text_events` would otherwise stream it per shot. The count of
    records added and the subset's path go into the run's JSON, so a run that
    found no text can be told from one that never looked.

    Every shot gets the same per-shot SIGALRM and try/except as every other
    stage: a hung corpus read costs one shot.
    """
    # Deferred, like every other framework-specific import in this module:
    # `events.masks` loads torch at module scope, and `features`/`infer`
    # runs must not pay for it.
    from .events import lexicon as lx
    from .events import pipeline, text_weak, unet

    model = unet.load_unet(args.unet, device=args.device)
    n_added = 0
    text_note = ""
    try:
        n_added = text_weak.build_logs_subset(
            shots, paths=ctx.paths, refresh_missing=args.refresh_text
        )
    except OSError as exc:
        # A misconfigured or unmounted logbook is not a reason to lose the
        # masks: the text is one of eight sources and the only one that is
        # not in the corpus.
        text_note = f"{type(exc).__name__}: {exc}"
        print(f"events: no shot-scope text this run - {text_note}",
              file=sys.stderr)
    lexicon = lx.load_lexicon()

    rows: list[dict] = []
    for shot in shots:
        started = time.monotonic()
        try:
            with time_limit(ctx.timeout_s):
                result = pipeline.process_shot(
                    shot, ctx.paths, model=model, device=args.device,
                    passes=tuple(args.passes), tile_batch=args.tile_batch,
                    amp=args.amp, norm=args.norm, lexicon=lexicon,
                    run_id=ctx.run_id,
                )
            row = result.as_row()
            print(result.line())
        except Exception as exc:  # noqa: BLE001 - per-shot isolation
            row = {"shot": shot, "status": "error", "error": type(exc).__name__,
                   "detail": str(exc)[:200]}
            print(f"{shot}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
        row["seconds"] = round(time.monotonic() - started, 2)
        rows.append(row)

    totals = pipeline.summarise(rows)
    totals.update(
        n_log_records_added=int(n_added),
        logs_subset=str(ctx.paths.logs_subset),
        text_note=text_note,
    )
    return rows, totals


def _jsonable(record: Mapping[str, object]) -> dict:
    """One source record with its NaNs as `None`, for the run JSON.

    `json.dumps` writes a bare `NaN` literal, which RFC 8259 has no word
    for: Python and `jq` read it, a strict parser does not, and a curated
    table's record is two NaNs (`t_cov0_s`, `t_cov1_s`) out of ten fields.
    The parquet keeps the NaN - it is a float column and `null` there would
    be a different dtype - so this is the JSON boundary only, and `null`
    means exactly what the NaN means: nobody recorded a coverage.
    """
    return {
        key: (None
              if isinstance(value, float) and not math.isfinite(value)
              else value)
        for key, value in record.items()
    }


def databases_stage(shots, ctx: RunContext) -> tuple[list[dict], dict]:
    """`events --databases-only`: the curated tables over a shot list.

    No corpus, no network, no masks - a table is a list somebody made, and
    reading it is a CSV lookup per shot. That is what makes it a separate
    mode rather than a step of the GPU path: a new table has to be
    ingestible over a whole shot list without a GPU, and it has to reach
    the shots whose corpus files we do not hold (all 33 of the RWM tables'
    shots, as it happens).

    The cost is per NAMED shot, not per shot in the list. An unnamed shot
    is a lookup; a named one is ~0.11 s (write the events, read them back,
    append the index, write the sources row), flat, measured over 2000
    synthetic shots. So the 33 RWM shots take about 4 s, `recommender_v1`'s
    500 unnamed shots about 1.7 s, and a table naming all 16,909 corpus
    shots would take roughly half an hour - serially, since this ignores
    `--workers`, which at this cost is a choice.

    The summary line is the point of the mode. "0 of 500 shots are named by
    any table" is an ANSWER - the RWM tables stop at 176092 and the corpus
    starts at 185601 - and a run that says it exits 0. Failing there would
    teach the next person to expect a table to overlap, which is exactly
    the assumption a curated list is not allowed to make.
    """
    from .events import databases, schema

    paths = ctx.paths
    try:
        specs = databases.load_manifest(paths.label_tables)
    except databases.DatabaseError as exc:
        print(f"events --databases-only: refusing to run - {exc}",
              file=sys.stderr)
        raise
    rows: list[dict] = []
    n_named = 0
    by_source: dict[str, int] = {}
    for shot in shots:
        started = time.monotonic()
        try:
            events, records = databases.events_for_shot(shot, specs,
                                                        paths.label_tables)
        except Exception as exc:  # noqa: BLE001 - per-shot isolation
            rows.append({"shot": int(shot), "status": "error",
                         "error": type(exc).__name__, "detail": str(exc)[:200],
                         "seconds": round(time.monotonic() - started, 2)})
            print(f"{shot}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if records:
            # No rows and no file for a shot no table names: an events file
            # written here would say a run looked at this shot, and nothing
            # did.
            n_named += 1
            path = paths.events_file(shot)
            schema.write_events(path, shot, events, run_id=ctx.run_id,
                                merge=True)
            # Beside the events, and on the same contract as the GPU path:
            # one `ok` row per table that named the shot, `reason=""`, and
            # coverage NaN - the table says nothing about which interval
            # of the shot anybody examined.
            schema.write_sources(paths.sources_file(shot), shot, records,
                                 run_id=ctx.run_id, merge=True)
            append_index(paths.events_index, schema.index_rows(path),
                         keys=["shot", "source", "phenomenon"])
            print(f"{shot}: {len(events)} events from "
                  f"{len(records)} table(s)")
        for event in events:
            by_source[event.source] = by_source.get(event.source, 0) + 1
        rows.append({
            "shot": int(shot),
            "status": "ok",
            "n_events": len(events),
            "n_tables": len(records),
            # Two spellings on purpose: `sources` is the list of names
            # `_stage_summary` groups shots by, `source_records` is the
            # sources-file contract's own rows, kept whole.
            "sources": [str(r["source"]) for r in records],
            "source_records": [_jsonable(r) for r in records],
            "seconds": round(time.monotonic() - started, 2),
        })
    totals = {
        "n_shots": len(rows),
        "n_shots_named": n_named,
        "n_events": sum(int(r.get("n_events", 0)) for r in rows),
        "n_source_records": sum(int(r.get("n_tables", 0)) for r in rows),
        "events_by_source": dict(sorted(by_source.items())),
        "tables": [spec.source for spec in specs],
        "summary_line": (
            f"{n_named} of {len(rows)} shots are named by any table"
        ),
    }
    return rows, totals


def rules_stage(shots, ctx: RunContext) -> tuple[list[dict], dict]:
    """`events --rules-only`: the features store and the tables, no corpus.

    A loop around `pipeline.rules_shot`, and it is a separate mode for the
    same reason `--databases-only` is: everything it runs is cheap and
    none of it needs the GPU path, so the answer to "which shots ran at
    elevated q-min" should not cost a mask run. MEASURED over the 500
    shots of `recommender_v1` on a login node: about a minute, serially.

    The summary is the product. `shots_with_band` counts SHOTS, not rows -
    a shot with three separate hybrid bands is one hybrid shot - because
    that is the number the rule was calibrated against (hybrid 271,
    elevated 60, high 50 of 500) and the number a later run has to
    reproduce to show the rule has not moved.
    """
    from .events import pipeline

    paths = ctx.paths
    rows: list[dict] = []
    by_source: dict[str, int] = {}
    with_band: dict[str, int] = {}
    for shot in shots:
        try:
            res = pipeline.rules_shot(shot, paths, run_id=ctx.run_id)
        except Exception as exc:  # noqa: BLE001 - per-shot isolation
            rows.append({"shot": int(shot), "status": "error",
                         "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            print(f"{shot}: ERROR {type(exc).__name__}: {str(exc)[:120]}")
            continue
        row = res.as_row()
        rows.append(row)
        for source, n in res.by_source.items():
            by_source[source] = by_source.get(source, 0) + n
        for band in sorted(res.by_phenomenon):
            with_band[band] = with_band.get(band, 0) + 1
        print(res.line())

    totals = pipeline.summarise(rows)
    totals.update(
        n_shots=len(rows),
        shots_with_band=dict(sorted(with_band.items())),
        summary_line=(
            "shots with a q-min band: " + (", ".join(
                f"{band}={n}" for band, n in sorted(with_band.items())
            ) or "none")
        ),
    )
    totals["events_by_source"] = dict(sorted(by_source.items()))
    return rows, totals


def write_events_run(paths: Paths, run_id: str, payload: dict) -> Path:
    """`runs/events/<run_id>.json`: the settings, the shots and the totals.

    Beside the stage-agnostic `runs/<run_id>/{manifest,summary}.json` every
    run writes, and not instead of it: this one is per-shot and per-source,
    which is what the pilot and the production gate read, and it is in one
    directory so that a hundred mask runs can be compared without walking a
    hundred run directories.
    """
    out = paths.runs / "events"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def write_manifest(paths: Paths, run_id: str, payload: dict) -> Path:
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _log(paths: Paths, run_id: str, rows) -> None:
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "log.txt", "a") as fh:
        fh.writelines(
            json.dumps({k: v for k, v in row.items() if k != "rows"}, default=str)
            + "\n"
            for row in rows
        )


def _stage_summary(stage: str, rows) -> dict:
    """Counts, errors and the source mix - the shape written to summary.json."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    combos: dict[str, list[int]] = {}
    for row in rows:
        if row.get("sources"):
            combos.setdefault("+".join(row["sources"]), []).append(row["shot"])
    return {
        "stage": stage,
        "counts": dict(sorted(counts.items())),
        "source_combinations": {k: len(v) for k, v in sorted(combos.items())},
        "mixed_source_shots": sorted(r["shot"] for r in rows if r.get("mixed")),
        "errors": [
            {"shot": r["shot"], "error": r.get("error"), "detail": r.get("detail")}
            for r in rows
            if r["status"] == "error"
        ],
    }


#: Printed and stored beside the mixed-source count, so the number arrives
#: with the reason it matters instead of needing a reader to go and find it.
#:
#: The archive's row k is stamped 25 ms later than the 50 ms interval it
#: averages, while corpus and fdp carry true time axes.
#: `models.base.InputSpec.build` reconciles that at the point model inputs
#: are assembled - sampling is keyed on which resolver produced each stored
#: array (`ns.SAMPLING_BY_SOURCE`), so an archive-served field (read
#: nearest-sample, since it is already the boxcar) and a corpus- or
#: fdp-served field (windowed into that same boxcar) refer to the same
#: physical 50 ms interval. A mixed-source shot is therefore provenance
#: information, not a correctness warning, for any consumer that reads
#: features through `build()` - which every model adapter does. It stays
#: worth recording per shot: which sources served which features is useful
#: on its own, and a consumer reading the raw feature file directly still
#: meets the offset.
MIXED_SOURCE_NOTE = (
    "features from more than one source in the same row: the archive's row k "
    "is stamped 25 ms - one whole dt - later than the 50 ms interval it "
    "averages (measured to 3.9e-08), while corpus and fdp carry true time "
    "axes. `InputSpec.build` (models/base.py) reconciles this per-resolver "
    "when assembling model inputs, so a mixed shot's labels are not "
    "misaligned; the raw feature file still carries the offset described "
    "here for a consumer reading it directly. See features/resolve_archive.py."
)


def _summarise(summary: dict) -> None:
    counts = summary["counts"]
    print(f"{summary['stage']}: "
          + ", ".join(f"{n} {status}" for status, n in counts.items()))
    for err in summary["errors"]:
        print(f"  {err['shot']}: {err['error']} {err.get('detail') or ''}")
    mixed = summary["mixed_source_shots"]
    if mixed:
        combos = ", ".join(
            f"{k} x{n}" for k, n in summary["source_combinations"].items() if "+" in k
        )
        print(f"  {len(mixed)} shot(s) mixed sources ({combos}) - "
              f"{MIXED_SOURCE_NOTE}")


def _run_pool(worker, payloads, workers: int):
    if workers <= 1:
        return [worker(p) for p in payloads]
    # Forked before the first fetch: the ptserver reader is not fork-safe.
    with Pool(workers) as pool:
        return pool.map(worker, payloads)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = None
    if args.stage == "analyze":
        if args.models:
            parser.error("analyze takes its models from --config, not --models")
        try:
            cfg = analyze.load_config(args.config)
        except analyze.ConfigError as exc:
            parser.error(str(exc))
        args.models = list(cfg.slugs)
    elif args.stage in MODEL_FREE_STAGES:
        if args.models:
            parser.error(f"{args.stage} takes no --models")
        args.models = []
    elif not args.models:
        parser.error("--models is required")
    if args.databases_only and args.stage != "events":
        parser.error("--databases-only belongs to the events stage")
    if args.rules_only and args.stage != "events":
        parser.error("--rules-only belongs to the events stage")
    if args.rules_only and args.databases_only:
        # `--rules-only` already runs the curated tables; asking for both
        # would run them twice and write the second copy over the first.
        parser.error("--rules-only already ingests the curated tables; "
                     "--databases-only is the tables ALONE")
    base = Paths.from_env()
    # `replace` and not a fresh `Paths`: `text_root` and `logs_jsonl` have no
    # flag of their own and are read from the environment, and building a
    # bare `Paths(root=..., corpus=...)` here silently put them back to their
    # defaults - which the `events` stage's shot-scope text comes out of.
    paths = replace(
        base,
        root=args.root or base.root,
        corpus=args.corpus_dir or base.corpus,
    )
    paths.mkdirs()
    # Keyed by slug alone, so a second `main` in one process - a test, a
    # notebook, a driver script - would otherwise reuse a predictor loaded
    # from a different `--root`, and skip the weight guard while doing it.
    _PREDICTORS.clear()
    archive_files = tuple(args.archive) if args.archive else resolve_archive.ARCHIVE_FILES
    shots = shot_list(args, paths)
    if not shots:
        print("no shots selected", file=sys.stderr)
        return EXIT_NO_SHOTS

    # Everything model-level happens here, in the parent, before any pool is
    # forked and before any shot is touched.
    try:
        adapters = [registry.load_adapter(slug) for slug in args.models]
        cards = {slug: registry.read_card(slug) for slug in args.models}
    except _MODEL_FAULTS as exc:
        print(f"cannot load the requested models: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return EXIT_BAD_MODEL
    if args.stage in ("infer", "validate", "all", "analyze"):
        # Verify the weights once, here, before either pool forks and before
        # the features stage spends hours on a run whose labels could not be
        # trusted anyway. A digest mismatch is a run-level fault, not a
        # per-shot one: leaving it to `_predictor` inside each worker would
        # turn one bad artifact into N identical per-shot errors and bury the
        # cause. The worker still checks, which catches an artifact changed
        # mid-run. `validate` shares this guard too: `reconstruction_fidelity`
        # and `label_quality` both call `adapter.load(paths.models / slug)`
        # directly, with no per-shot re-check of their own the way
        # `_predictor` has, so this up-front guard is the only thing standing
        # between a validate run and unverified weights.
        for slug in args.models:
            try:
                registry.verify_artifacts(slug, paths.models / slug)
            except _MODEL_FAULTS as exc:
                print(f"{args.stage} {slug}: refusing to run - {exc}", file=sys.stderr)
                return EXIT_UNVERIFIED_WEIGHTS

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # The pid disambiguates two runs of the same stage started in the same
    # second, which would otherwise share a run directory and interleave
    # their log lines under one manifest.
    run_id = args.run_id or f"{args.stage}-{stamp}-{os.getpid()}"
    ctx = RunContext(
        paths=paths,
        archive_files=archive_files,
        models=tuple(args.models),
        run_id=run_id,
        timeout_s=args.timeout,
        force=args.force,
    )
    names = _feature_names(adapters)
    write_manifest(
        paths,
        run_id,
        {
            "stage": args.stage,
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "labelmaker_version": __version__,
            "git_sha": git_sha(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "workers": args.workers,
            "timeout_s": args.timeout,
            "force": args.force,
            "models": list(args.models),
            "analyze_config": cfg.as_dict() if cfg is not None else None,
            "cards": {
                slug: card["labelmaker"].get("upstream", {})
                for slug, card in cards.items()
            },
            "features": names,
            "root": str(paths.root),
            "corpus": str(paths.corpus),
            "archive_files": [str(p) for p in archive_files],
            "shots": shots,
        },
    )

    summaries = []
    if args.stage == "events" and args.databases_only:
        from .events.databases import DatabaseError

        try:
            rows, totals = databases_stage(shots, ctx)
        except DatabaseError:
            return EXIT_BAD_LABEL_TABLE
        _log(paths, run_id, rows)
        summary = _stage_summary("events --databases-only", rows)
        _summarise(summary)
        summaries.append(summary)
        print(totals["summary_line"])
        print("events by source: " + (", ".join(
            f"{s}={n}" for s, n in totals["events_by_source"].items()
        ) or "none"))
        out = write_events_run(paths, run_id, {
            "run_id": run_id,
            "stage": "events",
            "git_sha": git_sha(),
            "labelmaker_version": __version__,
            "hostname": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "databases_only": True,
                "label_tables": str(paths.label_tables),
                "limit": args.limit,
                "root": str(paths.root),
            },
            "totals": totals,
            "shots": rows,
        })
        print(f"events: {out}")
    elif args.stage == "events" and args.rules_only:
        rows, totals = rules_stage(shots, ctx)
        _log(paths, run_id, rows)
        summary = _stage_summary("events --rules-only", rows)
        _summarise(summary)
        summaries.append(summary)
        print(totals["summary_line"])
        print("events by source: " + (", ".join(
            f"{s}={n}" for s, n in totals["events_by_source"].items()
        ) or "none"))
        out = write_events_run(paths, run_id, {
            "run_id": run_id,
            "stage": "events",
            "git_sha": git_sha(),
            "labelmaker_version": __version__,
            "hostname": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "rules_only": True,
                "features": str(paths.features),
                "label_tables": str(paths.label_tables),
                "limit": args.limit,
                "root": str(paths.root),
            },
            "totals": totals,
            "shots": rows,
        })
        print(f"events: {out}")
    elif args.stage == "events":
        rows, totals = events_stage(shots, ctx, args)
        _log(paths, run_id, rows)
        summary = _stage_summary("events", rows)
        _summarise(summary)
        summaries.append(summary)
        print("events by source: " + (", ".join(
            f"{s}={n}" for s, n in totals["events_by_source"].items()
        ) or "none"))
        out = write_events_run(paths, run_id, {
            "run_id": run_id,
            "stage": "events",
            "git_sha": git_sha(),
            "labelmaker_version": __version__,
            "hostname": socket.gethostname(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": {
                "device": args.device,
                "passes": list(args.passes),
                "tile_batch": args.tile_batch,
                "amp": bool(args.amp),
                "norm": args.norm,
                "timeout_s": args.timeout,
                "limit": args.limit,
                "refresh_text": bool(args.refresh_text),
                "unet": str(args.unet) if args.unet else "",
                "root": str(paths.root),
                "corpus": str(paths.corpus),
            },
            "n_log_records_added": totals["n_log_records_added"],
            "logs_subset": totals["logs_subset"],
            "text_note": totals["text_note"],
            "totals": totals,
            "shots": rows,
        })
        print(f"events: {out}")

    if args.stage == "analyze":
        out_dir = args.out or paths.root / "analysis"
        rows = _run_pool(
            _analyze_worker, [(s, cfg, out_dir, ctx) for s in shots], args.workers
        )
        _log(paths, run_id, rows)
        summary = _stage_summary("analyze", rows)
        _summarise(summary)
        summaries.append(summary)
        for row in rows:
            if row["status"] != "error":
                print(f"  {row['shot']}: {row['figure']}")

    if args.stage in ("features", "all"):
        rows = _run_pool(
            _features_worker, [(s, names, ctx) for s in shots], args.workers
        )
        _log(paths, run_id, rows)
        summary = _stage_summary("features", rows)
        _summarise(summary)
        summaries.append(summary)

    if args.stage in ("infer", "all"):
        all_rows = []
        for slug in args.models:
            rows = _run_pool(
                _infer_worker, [(s, slug, ctx) for s in shots], args.workers
            )
            _log(paths, run_id, rows)
            summary = _stage_summary(f"infer {slug}", rows)
            _summarise(summary)
            summaries.append(summary)
            all_rows.extend(rows)
        index = [r for row in all_rows for r in row.get("rows", [])]
        if index:
            append_index(paths.labels_index, index)
            print(f"index: {len(index)} rows -> {paths.labels_index}")

    # Collected across the whole loop rather than
    # returned from inside it, so (a) one model's fidelity failure does not
    # stop later models in a multi-`--models` run from being validated at
    # all, and (b) the exit code and the summary.json write both happen
    # once, at the bottom, after every model has had its turn - a
    # `--stage all` run that fails fidelity still gets a manifest for the
    # features/infer work it completed, instead of none.
    fidelity_failed: list[str] = []
    errored_reports: list[tuple[str, str]] = []

    if args.stage in ("validate", "all"):
        # Deferred import, like every other framework-specific import in this
        # module: `validate` loads torch at module scope.
        from . import validate as validation

        for slug in args.models:
            reports: dict = {}
            try:
                reports["adapter_fidelity"] = validation.adapter_fidelity(slug)
            except OSError as exc:
                # Addendum item 4: the golden file is committed, but the
                # golden file's own `meta["models"]` names a path under
                # `/projects/EKOLEMEN` - a different directory from
                # `paths.models`, and not covered by the `verify_artifacts`
                # guard above - which may not be mounted here. That is an
                # environment limitation, not a run-level fault the way a
                # bad digest is, so it is recorded as a skip rather than
                # aborting the whole validate stage.
                reports["adapter_fidelity"] = {
                    "skipped": f"{type(exc).__name__}: {exc}"
                }
            except Exception as exc:  # noqa: BLE001 - see comment below
                # `adapter_fidelity` is meaningful only for a model with a
                # committed golden file (today, only
                # `d3d_tearing_onset_cnn1d`); calling it for any other slug's
                # adapter (e.g. an empty `artifacts` tuple) fails inside
                # numpy/torch with something other than OSError. One model
                # not being validatable this way is that model's fault, not
                # a reason to crash a multi-model `--stage all` run.
                reports["adapter_fidelity"] = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                reports["reconstruction"] = validation.reconstruction_fidelity(
                    slug, shots, paths, timeout_s=args.timeout
                )
            except Exception as exc:  # noqa: BLE001 - see comment above
                # `reconstruction_fidelity` and `label_quality` are likewise
                # specific to `d3d_tearing_onset_cnn1d`'s scalar column order
                # - MATCH_COLUMNS indexing a different slug's shorter
                # or differently-ordered scalar_fields raises before any
                # per-shot work, which is by design (see validate.py), but
                # it must not take down the rest of a multi-model run.
                reports["reconstruction"] = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                reports["label_quality"] = validation.label_quality(
                    slug, shots, paths, timeout_s=args.timeout
                )
            except Exception as exc:  # noqa: BLE001 - see comment above
                reports["label_quality"] = {"error": f"{type(exc).__name__}: {exc}"}
            if any(key.startswith(f"{slug}/") for key in validation.ARCHIVE_TRUTH):
                try:
                    reports["alarm_quality"] = validation.alarm_quality(
                        slug, shots, paths, timeout_s=args.timeout
                    )
                except Exception as exc:  # noqa: BLE001 - isolate a model's report
                    reports["alarm_quality"] = {"error": f"{type(exc).__name__}: {exc}"}
            # Every slug whose archive truth includes a horizon can be
            # calibrated, not merely the checkpoint the study was written for:
            # naming one slug here left the retrained variant publishing NaN
            # isotonic columns for no reason. Note the side effect - the study
            # writes `calibration.json` into the model directory, so the next
            # `infer` for this slug publishes its `*_isotonic` labels from the
            # map fitted here.
            if any(rule["kind"] == "onset_within"
                   for key, rule in validation.ARCHIVE_TRUTH.items()
                   if key.startswith(f"{slug}/")):
                try:
                    reports["calibration_study"] = validation.calibration_study(
                        slug, shots, paths, timeout_s=args.timeout
                    )
                except Exception as exc:  # noqa: BLE001 - isolate a model's report
                    reports["calibration_study"] = {"error": f"{type(exc).__name__}: {exc}"}
            for name, payload in reports.items():
                out = validation.write_report(paths, slug, name, payload)
                print(f"validate {slug}: {name} -> {out}")
                if isinstance(payload, dict) and "error" in payload:
                    errored_reports.append((slug, name))
                    print(
                        f"validate {slug}: {name} ERRORED: {payload['error']}",
                        file=sys.stderr,
                    )
            results = validation.model_index_results(reports)
            if results:
                registry.update_model_index(slug, results)
                print(f"validate {slug}: {len(results)} results written to the card")
            fidelity = reports["adapter_fidelity"]
            if fidelity.get("passed") is False:
                print(
                    f"validate {slug}: ADAPTER FIDELITY FAILED "
                    f"(max_abs_diff={fidelity.get('max_abs_diff')})",
                    file=sys.stderr,
                )
                fidelity_failed.append(slug)

    exit_code = EXIT_OK
    if fidelity_failed:
        exit_code = EXIT_FIDELITY_FAILED
    elif errored_reports:
        # At least one report is `{"error": ...}` and none failed
        # fidelity outright - still not a clean run, and `$?` must say so.
        print(
            f"validate: {len(errored_reports)} report(s) errored: "
            + ", ".join(f"{s}/{n}" for s, n in errored_reports),
            file=sys.stderr,
        )
        exit_code = EXIT_VALIDATE_ERRORED

    (paths.runs / run_id / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "mixed_source_note": MIXED_SOURCE_NOTE,
                "stages": summaries,
                # An error count in the run summary, not only on stderr -
                # a caller inspecting summary.json after the fact (rather
                # than capturing stderr at run time) must be able to see the
                # same verdict $? carried.
                "validate_fidelity_failed": fidelity_failed,
                "validate_errors": [
                    {"slug": s, "report": n} for s, n in errored_reports
                ],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
