"""The labelmaker CLI: `python -m labelmaker.run <stage>`.

Three stages, each independently rerunnable, with a per-shot HDF5 file
between them:

    features  resolve the union of canonical features the requested models
              need, per shot, into <root>/features/<shot>_features.h5
    infer     build each model's inputs from that file, predict, and write
              <root>/labels/<shot>_labels.h5
    all       features, then infer

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
  averages (measured to 3.9e-08; see `resolve_archive`'s docstring). One
  whole `dt` of misalignment between features in the same row is not
  visible in the numbers, so it is recorded per shot in the run log and
  aggregated into `runs/<run_id>/summary.json` for Task 15 to act on.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import socket
import sys
import time
from argparse import ArgumentParser
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import __version__
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
from .labels.store import append_index, index_rows, labelled, write_labels
from .models import registry

STAGES = ("features", "infer", "all")

#: Exit codes. Anything non-zero means no labels should be trusted from this
#: run; 3 and 4 mean nothing ran at all.
EXIT_OK = 0
EXIT_NO_SHOTS = 1
#: 2 is argparse's own usage error.
EXIT_UNVERIFIED_WEIGHTS = 3
EXIT_BAD_MODEL = 4

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
    parser.add_argument("--models", nargs="+", required=True, metavar="SLUG")
    picker = parser.add_mutually_exclusive_group(required=True)
    picker.add_argument("--shots", nargs="+", type=int, metavar="SHOT")
    picker.add_argument("--shot-file", type=Path)
    picker.add_argument("--corpus", action="store_true",
                        help="every shot in the corpus")
    picker.add_argument("--overlap", action="store_true",
                        help="shots present in both the corpus and the "
                             "tearing-mode training archive")
    parser.add_argument("--sample", type=int, default=0,
                        help="with --corpus/--overlap: take a seeded sample")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds per shot per stage")
    parser.add_argument("--force", action="store_true",
                        help="redo shots that are already complete")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--archive", nargs="+", type=Path, default=None)
    return parser


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
        specs = specs_for(adapter, artifact_digest(sha_map))
        _PREDICTORS[slug] = (adapter, adapter.load(model_dir), specs)
    return _PREDICTORS[slug]


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
    stored = present(features_path)
    features = {
        name: read_feature(features_path, name)
        for name in adapter.input_spec.canonical_names
        if name in stored
    }
    built = adapter.input_spec.build(features, ns.GRID_S)
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
MIXED_SOURCE_NOTE = (
    "features from more than one source in the same row: the archive's row k "
    "is stamped 25 ms - one whole dt - later than the 50 ms interval it "
    "averages (measured to 3.9e-08), while corpus and fdp carry true time "
    "axes. Pure-archive and pure-fdp shots are unaffected. See "
    "features/resolve_archive.py."
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
    args = build_parser().parse_args(argv)
    base = Paths.from_env()
    paths = Paths(
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
    if args.stage in ("infer", "all"):
        # Verify the weights once, here, before either pool forks and before
        # the features stage spends hours on a run whose labels could not be
        # trusted anyway. A digest mismatch is a run-level fault, not a
        # per-shot one: leaving it to `_predictor` inside each worker would
        # turn one bad artifact into N identical per-shot errors and bury the
        # cause. The worker still checks, which catches an artifact changed
        # mid-run.
        for slug in args.models:
            try:
                registry.verify_artifacts(slug, paths.models / slug)
            except _MODEL_FAULTS as exc:
                print(f"infer {slug}: refusing to run - {exc}", file=sys.stderr)
                return EXIT_UNVERIFIED_WEIGHTS

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # The pid disambiguates two runs of the same stage started in the same
    # second, which would otherwise share a run directory and interleave
    # their log lines under one manifest.
    run_id = f"{args.stage}-{stamp}-{os.getpid()}"
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

    (paths.runs / run_id / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "mixed_source_note": MIXED_SOURCE_NOTE,
                "stages": summaries,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
