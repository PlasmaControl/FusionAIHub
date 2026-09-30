"""Run a method's model over the roster and the population, as suggestions
(round three, Part B; spec §3.5; D47-D50).

    python -m labeler.frames.apply --method M --roster [--index I --count N]
        [--limit N] [--shots ...] [--workers W]
    python -m labeler.frames.apply --method M --population [--index I --count N]
        [--limit N] [--shots ...] [--workers W]
    python -m labeler.frames.apply --method M --merge

**Shots.** Two sets, each sharded as `prepare.shard` shards (I and N default to
SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT, D67):
- `--roster`: the store event's review roster less the cohort's blind shots,
  read from their review stores (`frames.store_path`);
- `--population`: the v1 population's non-blind shots outside the roster
  (`spans.population`). Each shot's required groups must be on disk
  (`raw.record_tier`, a metadata check), and its rows are built in memory, as a
  store would be (`review.build.BUILDERS`, `panel_rows.build` for these events),
  with no store written (D48).
A shot's window is its catalog window (`targets.catalog_windows`, the cohort's
then the population's); nothing is fetched.

**Frames.** Every whole 10 ms frame of the window. The model's frame logits are
pooled to the spec's bins as it was trained (`model.bin_logits`), and each frame
takes its bin's P (`frame_probs`). A bin with a frame not observed is not
observable, and so is a partial bin at the window's edge, one with fewer than
its `bin_ms / 10` frames inside the window (F7): training and the test score
whole bins only, and v1's pooling of such a bin over the frames it had put a
lone short H bin at the start of every H-mode shot. The others are present
where P reaches the model's threshold (`train.fit`), else absent.
The confidence is P for a present frame, 1 - P for an absent one, and blank for
one not observable, as the AE extension writes them.

**Shards.** Each writes `<set>-<I>-of-<N>.npz` (each shot's P per frame, NaN
where not observable), `.failed.jsonl` and, last, `.json` (the shots it was
given, done, skipped with their reasons, and failed; the model's sha256) under
`suggestions/<method>/<frames.VERSION>/shards/`. A shot that raises, whatever the error, is
failed with its error's type and message, and the shard goes on: the failures
are counted, not fatal, and the merge's tallies say how many there were. A
limited run (`--limit`, `--shots`) is a pilot and writes under `shards/pilot/`,
which `--merge` never reads.

**Merge.** Whatever the bar (D50), once the test is scored: every shard of both
sets must be there for the model that `evaluation.json` scored, each given its
shard's shots. It writes `suggestions.table_path(paths, event, method,
frames.VERSION)` (`suggestions.write_table`) with a meta holding the bar's
verdict, whether the model is effectively always (`effectively_always`, from
`evaluation.json`, F6), the tier `suggestions` and the model's sha256 (D47,
D49), and `frames.summary_file`, one
row a shot in `ae.xpower.extend.SUMMARY_COLUMNS`, the year its campaign's
(`evaluate.shot_years`), blank where none is known; and `failed.jsonl`, the
failed shots, beside the table, the meta counting them (`sets.<set>.failed`)
and naming the file (`failed_file`). `hmode_frames`' merge also writes
`lmode_frames`' table, summary and `failed.jsonl` (D38): L where the model says
not H, on the same frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import parent_process
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..ae.xpower.data import window_frames
from ..ae.xpower.extend import SUMMARY_COLUMNS
from ..config import Paths, atomic_path, git_sha, sha256_of
from ..events import raw, spans, suggestions
from ..events.catalog.cohort import read_cohort
from ..events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT
from ..events.review import build as review_build
from ..events.review import panel_rows
from . import (
    DERIVED,
    SPECS,
    VERSION,
    EventSpec,
    model_dir,
    roster_shots,
    store_path,
    summary_file,
)
from . import evaluate as frames_evaluate
from . import train as frames_train
from .features import features
from .prepare import _run, shard
from .targets import catalog_windows

log = logging.getLogger(__name__)

SETS = ("roster", "population")
#: The methods whose merge also writes a derived one's table (D38).
ALSO = {source: derived for derived, (source, _) in DERIVED.items()}
SHARD = re.compile(r"^(roster|population)-(\d+)-of-(\d+)\.json$")
_MODELS: dict = {}


class Skipped(Exception):
    """A shot the set does not label: no window, no store, or no groups."""


@dataclass(frozen=True)
class Labelled:
    """One shot's P per frame (NaN where not observable) over `window`'s frames
    from `first`, and the model's inputs, for the gallery."""

    shot: int
    window: tuple[int, int]
    first: int
    prob: np.ndarray
    x: np.ndarray
    observed: np.ndarray


def blind_shots(paths: Paths) -> set[int]:
    return {int(s) for s in read_cohort(spans.cohort_path(paths)).query("blind").shot}


def set_shots(paths: Paths, spec: EventSpec, which: str) -> list[int]:
    """A set's shots, before sharding (module docstring)."""
    roster = roster_shots(paths, spec.store_event)
    if which == "roster":
        return sorted(roster - blind_shots(paths))
    if which == "population":
        return sorted(set(spans.population(paths).shot.astype(int)) - roster)
    raise ValueError(f"no set {which!r}: one of {SETS}")


def windows(paths: Paths) -> dict[int, tuple[int, int]]:
    """Each catalog shot's window in whole ms, rounded inward."""
    out = {}
    for shot, (lo, hi) in catalog_windows(paths).items():
        lo, hi = math.ceil(lo), math.floor(hi)
        if lo < hi:
            out[shot] = (lo, hi)
    return out


def frame_probs(model, spec: EventSpec, x, observed, first: int) -> np.ndarray:
    """Each frame's bin's P, the frame logits pooled per bin as in training; NaN
    in a bin with a frame not observed, and in a partial bin at the window's
    edge, one with fewer than a bin's frames (F7)."""
    per = frames_train.frames_per_bin(spec)
    observed = np.asarray(observed, bool)
    n = len(observed)
    if not n:
        return np.zeros(0)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None])
    logits = logits[0].numpy().astype(np.float64)
    _, inverse = np.unique((first + np.arange(n)) // per, return_inverse=True)
    frames = np.bincount(inverse)
    if spec.pool == "max":
        pooled = np.full(len(frames), -np.inf)
        np.maximum.at(pooled, inverse, logits)
    else:
        pooled = np.bincount(inverse, logits) / frames
    prob = 1.0 / (1.0 + np.exp(-pooled))
    seen = (np.bincount(inverse, observed) == frames) & (frames == per)
    return np.where(seen[inverse], prob[inverse], np.nan)


def store_of(paths: Paths, spec: EventSpec, shot: int, roster: bool):
    """A roster shot's store path; any other shot's rows, built in memory (D48)."""
    if roster:
        path = store_path(paths, spec, shot)
        if not path.is_file():
            raise Skipped("no store")
        return path
    for group in spec.required_groups:
        if raw.record_tier(shot, group, paths=paths) is None:
            raise Skipped(f"no {group} on disk")
    builder = review_build.BUILDERS.get(spec.store_event, panel_rows.build)
    return builder(spec.store_event, int(shot), paths)


def label(paths, spec, model, shot: int, window, *, roster: bool) -> Labelled:
    """One shot's P per frame (module docstring)."""
    store = store_of(paths, spec, shot, roster)
    x, observed = features(store, spec, window)
    first, _ = window_frames(window)
    prob = frame_probs(model, spec, x, observed, first)
    return Labelled(int(shot), tuple(window), first, prob, x, observed)


def _model(path):
    """The model, loaded once in each process."""
    key = str(path)
    if key not in _MODELS:
        _MODELS[key] = frames_train.load(path)[0]
    return _MODELS[key]


def _label_job(job):
    paths, method, model_path, shot, window, roster = job
    if parent_process() is not None and not _MODELS:
        torch.set_num_threads(1)  # a spawned worker: one thread each
    try:
        found = label(
            paths, SPECS[method], _model(model_path), shot, window, roster=roster
        )
    except Skipped as reason:
        return shot, "skipped", str(reason)
    except Exception as error:  # noqa: BLE001 - one bad shot must not stop the shard
        return shot, "failed", f"{type(error).__name__}: {error}"
    return shot, "done", found


def frame_states(prob, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """`(states, confidence)` per frame; the confidence NaN where not observable."""
    prob = np.asarray(prob, dtype=np.float64)
    seen = np.isfinite(prob)
    present = seen & (np.nan_to_num(prob) >= threshold)
    states = np.where(present, PRESENT, np.where(seen, ABSENT, NOT_OBSERVABLE))
    return states, np.where(present, prob, 1.0 - prob)


def complement(states, confidence) -> tuple[np.ndarray, np.ndarray]:
    """L's frames from H's (D38): present and absent swapped, the same confidence."""
    states = np.asarray(states)
    swapped = np.select(
        [states == PRESENT, states == ABSENT], [ABSENT, PRESENT], states
    )
    return swapped, confidence


def shot_rows(shot: int, first: int, states, confidence) -> list[list]:
    """The suggestion rows, blank confidence where not observable."""
    table = suggestions.frame_rows(shot, first, states, np.nan_to_num(confidence))
    return [r if r[1] != NOT_OBSERVABLE else [*r[:4], ""] for r in table]


def summary_row(shot, window, states, confidence, rows, year) -> dict:
    """One `SUMMARY_COLUMNS` row; `max_prob` is the largest P of the event's
    state over the observable frames."""
    states = np.asarray(states)
    seen = states != NOT_OBSERVABLE
    said = np.where(states == PRESENT, confidence, 1.0 - np.asarray(confidence))
    return {
        "shot": int(shot),
        "year": "" if year is None else int(year),
        "window_start_ms": int(window[0]),
        "window_end_ms": int(window[1]),
        "frames": len(states),
        "present_frames": int((states == PRESENT).sum()),
        "not_observable_frames": int((~seen).sum()),
        "present_runs": sum(1 for r in rows if r[1] == PRESENT),
        "max_prob": round(float(np.max(said[seen])), 4) if seen.any() else "",
    }


def _setup(paths: Paths, method: str):
    path = model_dir(paths, method, VERSION) / "model.pt"
    model, blob = frames_train.load(path)
    return path, model, blob


def label_all(paths: Paths, method: str, shots, *, workers: int = 1):
    """Each of `shots` labelled: `(shot, "done" | "skipped" | "failed", Labelled or
    the reason)`, in shot order; a roster shot from its store, any other built
    in memory."""
    spec = SPECS[method]
    path, _, _ = _setup(paths, method)
    roster = roster_shots(paths, spec.store_event)
    blind, found = blind_shots(paths), windows(paths)
    jobs, early = [], []
    for shot in sorted({int(s) for s in shots}):
        if shot in blind:
            early.append((shot, "skipped", "blind"))
        elif shot not in found:
            early.append((shot, "skipped", "no window"))
        else:
            jobs.append((paths, method, path, shot, found[shot], shot in roster))
    results = {shot: (shot, kind, why) for shot, kind, why in early}
    for result in _run(_label_job, jobs, workers):
        results[result[0]] = result
    return [results[shot] for shot in sorted(results)]


def apply(paths: Paths, method: str, shots, *, index: int = 0, count: int = 1):
    """Shard `index` of `count` of `shots` labelled: its suggestion rows."""
    _, _, blob = _setup(paths, method)
    rows = []
    for shot, kind, found in label_all(paths, method, shard(shots, index, count)):
        if kind == "done":
            states, confidence = frame_states(found.prob, blob["threshold"])
            rows += shot_rows(shot, found.first, states, confidence)
        else:
            log.info("shot %d %s: %s", shot, kind, found)
    return rows


def shards_dir(paths: Paths, spec: EventSpec, *, pilot: bool = False) -> Path:
    folder = suggestions.table_path(paths, spec.event, spec.method, VERSION).parent
    return folder / "shards" / ("pilot" if pilot else "")


def run_shard(
    paths: Paths,
    method: str,
    which: str,
    index: int,
    count: int,
    *,
    limit: int = 0,
    shots=None,
    workers: int = 1,
) -> dict:
    """Label one shard of a set and write its files (module docstring)."""
    spec = SPECS[method]
    started = time.monotonic()
    given = shard(
        set_shots(paths, spec, which) if shots is None else shots, index, count
    )
    if limit:
        given = given[:limit]
    pilot = bool(limit or shots is not None)
    model_path, _, blob = _setup(paths, method)
    done, skipped, failed = [], {}, []
    probs, meta = [], []
    for shot, kind, found in label_all(paths, method, given, workers=workers):
        if kind == "done":
            done.append(shot)
            probs.append(found.prob.astype(np.float32))
            meta.append((shot, found.first, *found.window))
        elif kind == "skipped":
            skipped[str(shot)] = found
        else:
            failed.append({"shot": shot, "error": found})
    folder = shards_dir(paths, spec, pilot=pilot)
    stem = f"{which}-{index}-of-{count}"
    table = np.asarray(meta, dtype=np.int64).reshape(-1, 4)
    with atomic_path(folder / f"{stem}.npz") as tmp, open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            shots=table[:, 0],
            first=table[:, 1],
            window=table[:, 2:],
            n=np.asarray([len(p) for p in probs], dtype=np.int64),
            prob=np.concatenate(probs) if probs else np.zeros(0, np.float32),
        )
    with atomic_path(folder / f"{stem}.failed.jsonl") as tmp:
        tmp.write_text("".join(json.dumps(row) + "\n" for row in failed))
    manifest = {
        "method": method,
        "set": which,
        "index": index,
        "count": count,
        "limit": limit,
        "pilot": pilot,
        "shots": given,
        "done": done,
        "skipped": skipped,
        "failed": [row["shot"] for row in failed],
        "model": str(model_path),
        "model_sha256": sha256_of(model_path),
        "threshold": blob["threshold"],
        "workers": workers,
        "wall_seconds": round(time.monotonic() - started, 1),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(folder / f"{stem}.json") as tmp:
        tmp.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _read_shards(paths: Paths, spec: EventSpec, which: str, model_sha: str):
    """A set's shards, checked complete: `(manifests, {shot: (first, window, P)})`."""
    folder = shards_dir(paths, spec)
    found = {}
    for path in folder.glob(f"{which}-*-of-*.json"):
        m = SHARD.match(path.name)
        if m:
            found[(int(m[2]), int(m[3]))] = path
    counts = {n for _, n in found}
    if len(counts) != 1:
        raise ValueError(f"{folder}: {which} shards of {sorted(counts) or 'no'} counts")
    (count,) = counts
    missing = [i for i in range(count) if (i, count) not in found]
    if missing:
        raise FileNotFoundError(
            f"{folder}: {which} shards {missing} of {count} missing"
        )
    expected = set_shots(paths, spec, which)
    manifests, labelled = [], {}
    for index in range(count):
        manifest = json.loads(found[(index, count)].read_text())
        where = found[(index, count)]
        if manifest.get("model_sha256") != model_sha:
            raise ValueError(f"{where}: made with another model")
        if manifest["shots"] != shard(expected, index, count):
            raise ValueError(f"{where}: its shots are not its shard's")
        seen = manifest["done"] + [int(s) for s in manifest["skipped"]]
        if Counter(seen + manifest["failed"]) != Counter(manifest["shots"]):
            raise ValueError(f"{where}: done, skipped and failed are not its shots")
        with np.load(where.with_suffix(".npz")) as z:
            ends = np.cumsum(z["n"])
            for i, shot in enumerate(z["shots"]):
                prob = z["prob"][ends[i] - z["n"][i] : ends[i]].astype(np.float64)
                window = tuple(int(v) for v in z["window"][i])
                labelled[int(shot)] = (int(z["first"][i]), window, prob)
        if sorted(labelled.keys() & set(manifest["done"])) != sorted(manifest["done"]):
            raise ValueError(f"{where}: its npz lacks shots it did")
        manifests.append(manifest)
    return manifests, labelled


def _failures(paths: Paths, spec: EventSpec, which: str, manifests) -> list[dict]:
    folder, rows = shards_dir(paths, spec), []
    for m in manifests:
        stem = f"{which}-{m['index']}-of-{m['count']}.failed.jsonl"
        text = (folder / stem).read_text()
        rows += [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows


def _folder(paths: Paths, event: str, method: str) -> Path:
    """Where a method's table, summary and `failed.jsonl` go."""
    return suggestions.table_path(paths, event, method, VERSION).parent


def _write(paths, method, event, labelled, threshold, meta, years, derive=False):
    rows, summary = [], []
    for shot in sorted(labelled):
        first, window, prob = labelled[shot]
        states, confidence = frame_states(prob, threshold)
        if derive:
            states, confidence = complement(states, confidence)
        table = shot_rows(shot, first, states, confidence)
        rows += table
        summary.append(
            summary_row(shot, window, states, confidence, table, years.of(shot))
        )
    path = suggestions.table_path(paths, event, method, VERSION)
    suggestions.write_table(path, rows, meta)
    frame = pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS))
    with atomic_path(summary_file(paths, method, VERSION)) as tmp:
        frame.to_csv(tmp, index=False)
    return path


def merge(paths: Paths, method: str) -> dict:
    """Write the method's table, meta and summary from its shards (module
    docstring), and `lmode_frames`' for `hmode_frames`."""
    spec = SPECS[method]
    model_path = model_dir(paths, method, VERSION) / "model.pt"
    evaluation_path = model_path.parent / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"{evaluation_path}: the test is scored before a merge")
    evaluation = json.loads(evaluation_path.read_text())
    model_sha = sha256_of(model_path)
    if evaluation["model"]["sha256"] != model_sha:
        raise ValueError(f"{evaluation_path}: it scored another model")
    _, blob = frames_train.load(model_path)
    labelled, sets, failed = {}, {}, []
    for which in SETS:
        manifests, found = _read_shards(paths, spec, which, model_sha)
        labelled |= found
        failures = _failures(paths, spec, which, manifests)
        failed += failures
        reasons = Counter(r for m in manifests for r in m["skipped"].values())
        sets[which] = {
            "shards": len(manifests),
            "shots": sum(len(m["shots"]) for m in manifests),
            "done": len(found),
            "skipped": dict(sorted(reasons.items())),
            "failed": len(failures),
        }
    years = frames_evaluate.shot_years(paths)
    population = paths.catalog / "population.csv"
    meta = {
        "method": method,
        "version": VERSION,
        "event": spec.event,
        "tier": "suggestions",
        "bar": evaluation["bar"],
        "effectively_always": evaluation["effectively_always"],
        "bar_criteria": evaluation.get("bar_criteria"),
        "model": str(model_path),
        "model_sha256": model_sha,
        "threshold": blob["threshold"],
        "split_sha256": blob["split_sha256"],
        "evaluation": str(evaluation_path),
        "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        "sets": sets,
        "failed_file": str(_folder(paths, spec.event, method) / "failed.jsonl"),
        "population": str(population),
        "population_sha256": sha256_of(population),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    threshold = float(blob["threshold"])
    written = {
        method: _write(paths, method, spec.event, labelled, threshold, meta, years)
    }
    if method in ALSO:
        derived = ALSO[method]
        event = DERIVED[derived][1]
        also = meta | {
            "method": derived,
            "event": event,
            "derived_from": method,
            "note": "L where the H-mode model says not H, on the same frames (D38)",
            "failed_file": str(_folder(paths, event, derived) / "failed.jsonl"),
        }
        written[derived] = _write(
            paths, derived, event, labelled, threshold, also, years, derive=True
        )
    for path in written.values():
        with atomic_path(path.parent / "failed.jsonl") as tmp:
            tmp.write_text("".join(json.dumps(row) + "\n" for row in failed))
    return {
        "tables": {k: str(v) for k, v in written.items()},
        "shots": len(labelled),
        "failed": len(failed),
        "bar": evaluation["bar"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=list(SPECS))
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--roster", action="store_true", help="the roster's stores")
    which.add_argument(
        "--population", action="store_true", help="the population, in memory"
    )
    which.add_argument("--merge", action="store_true", help="write the tables")
    p.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument(
        "--count", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    )
    p.add_argument("--limit", type=int, default=0, help="the shard's first N, a pilot")
    p.add_argument("--shots", type=int, nargs="+", help="these shots, a pilot")
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    args = p.parse_args(argv)
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = Paths.from_env()
    try:
        if args.merge:
            print(json.dumps(merge(paths, args.method)), flush=True)
            return 0
        manifest = run_shard(
            paths,
            args.method,
            "roster" if args.roster else "population",
            args.index,
            args.count,
            limit=args.limit,
            shots=args.shots,
            workers=args.workers,
        )
    except (OSError, ValueError) as error:
        p.error(str(error))
    line = {
        k: manifest[k] if isinstance(manifest[k], (int, bool)) else len(manifest[k])
        for k in ("index", "count", "pilot", "shots", "done", "skipped", "failed")
    }
    print(json.dumps({"method": args.method, "set": manifest["set"], **line}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
