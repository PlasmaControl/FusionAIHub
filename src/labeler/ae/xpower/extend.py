"""Run the chosen AE model over the population's CO2 shots, as suggestions.

    python -m labeler.ae.xpower.extend --shard K --of N [--workers W] [--limit M]
    python -m labeler.ae.xpower.extend --merge --of N

**Shots.** The v1 population (`$LABELER_ROOT/catalog/population.csv`) rows whose
corpus CO2 spans at least 2 s (`span_co2_s`), excluding the frozen cohort's blind
test shots (`$LABELER_ROOT/catalog/cohort.csv`) before sharding. A missing or
unreadable cohort is an error. The corpus carries CO2 from about shot 197,545
on, so these are nearly all 2024-2025 shots. Nothing is fetched: a shot whose
corpus CO2 cannot be read is logged in the shard's `failed` file and skipped.

**Frames.** Every whole 10 ms frame of the population window (v1 rule 4's Ip
window). A frame the CO2 rows do not cover is not observable.

**Output.** Shard K writes `shards/K.csv` (suggestion rows, `events.suggestions`),
`shards/K.summary.csv` (one line per shot), `shards/K.npz` (P(AE) per frame)
and `shards/K.failed.jsonl` under
`$LABELER_ROOT/suggestions/ae_xpower/<version>/`, and one JPEG per shot into
the gallery's `extension/`; the shard's manifest lists the shots it drew. A
pilot (`--limit M`) writes all of these under `shards/pilot/`, its pictures in
`shards/pilot/extension/`, never into the gallery. `--merge` checks every shard
is there and writes `alfven_eigenmode_suggest_ae_xpower_<version>.csv`, its
meta, `summary.csv`, and gallery index rows for the pictures the merged shards'
manifests list, and no others.

**Gate.** The chosen model's full test evaluation must pass A1 and A2 (D47) and
name the model and choice by sha256; a models directory under `runs/` is never
the gate's, and `--version` must be the directory's name and the checkpoint's.
Failures are JSON lines, each with an integer `shot` and string `error`.
Merge refuses when failures exceed 2 % of the eligible non-blind population
(`MAX_FAILED_FRACTION = 0.02`).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events import suggestions
from ...events.catalog.check import CatalogError
from ...events.catalog.cohort import read_cohort
from ...events.catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT
from ...events.review.rows import Grid, pool
from ...events.verify import corpus_signal
from ...scoring.frames import FRAME_MS
from . import (
    EVENT,
    METHOD,
    VERSION,
    check_bound,
    gallery_dir,
    model_dir,
    pilot_area,
    suggestions_dir,
)
from .data import raw_rows, window_frames
from .evaluate import chosen_model
from .gallery import draw, run_all, write_index
from .train import load, probabilities

MIN_CO2_S = 2.0
MAX_FAILED_FRACTION = 0.02
PICTURE_LEVEL = 8
SUMMARY_COLUMNS = (
    "shot",
    "year",
    "window_start_ms",
    "window_end_ms",
    "frames",
    "present_frames",
    "not_observable_frames",
    "present_runs",
    "max_prob",
)


def population_shots(path, min_co2_s: float = MIN_CO2_S) -> pd.DataFrame:
    """Population shots with at least `min_co2_s` of corpus CO2 and a window."""
    frame = pd.read_csv(path)
    keep = (
        (pd.to_numeric(frame["span_co2_s"], errors="coerce") >= min_co2_s)
        & frame["window_start_ms"].notna()
        & frame["window_end_ms"].notna()
    )
    out = frame.loc[keep, ["shot", "year", "window_start_ms", "window_end_ms"]]
    return out.astype(int).sort_values("shot", ignore_index=True)


def shard_shots(paths: Paths, k: int, of: int) -> pd.DataFrame:
    """Eligible non-blind shots, in the same order for execution and merge."""
    if of <= 0 or not 0 <= k < of:
        raise ValueError("shard must be in 0 .. of - 1, with of positive")
    cohort_path = paths.catalog / "cohort.csv"
    try:
        cohort = read_cohort(cohort_path)
    except CatalogError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise CatalogError(f"{cohort_path}: {error}") from error
    blind = cohort.loc[cohort["blind"], "shot"]
    population = paths.catalog / "population.csv"
    try:
        jobs = population_shots(population)
        jobs = jobs.loc[~jobs["shot"].isin(blind)]
        if jobs.shot.duplicated().any():
            raise ValueError("duplicate eligible shot IDs")
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{population}: {error}") from error
    return jobs.iloc[k::of]


def _shard_inputs(paths: Paths, models: Path) -> dict:
    return {
        "model_sha256": sha256_of(chosen_model(models)),
        "evaluation_sha256": sha256_of(models / "evaluation.json"),
        "population_sha256": sha256_of(paths.catalog / "population.csv"),
        "cohort_sha256": sha256_of(paths.catalog / "cohort.csv"),
        "min_co2_s": MIN_CO2_S,
    }


def frame_states(prob, observed, threshold: float) -> np.ndarray:
    states = np.where(np.asarray(prob) >= threshold, PRESENT, ABSENT)
    return np.where(np.asarray(observed, bool), states, NOT_OBSERVABLE)


_WORKER: dict = {}


def _init(model_file: str, root: str, corpus: str) -> None:
    import torch

    torch.set_num_threads(1)
    model, blob = load(model_file)
    _WORKER.update(
        model=model, blob=blob, paths=Paths(root=Path(root), corpus=Path(corpus))
    )


def label_shot(job: tuple[int, int, int, int], pictures: bool = True) -> dict:
    """One shot's suggestion rows, P(AE) and summary; its picture unless told not."""
    shot, year, lo, hi = job
    w = _WORKER
    blob, paths = w["blob"], w["paths"]
    version = w["version"]
    pictures_dir = w.get("pictures_dir") or gallery_dir(paths, version) / "extension"
    co2 = corpus_signal(shot, "co2", corpus=paths.corpus)
    rows = raw_rows(co2.x, co2.y)
    first, n = window_frames((lo, hi))
    prob, observed = probabilities(w["model"], rows, first, n, band=blob["band_khz"])
    states = frame_states(prob, observed, blob["threshold"])
    confidence = np.where(states == PRESENT, prob, 1.0 - prob)
    table = suggestions.frame_rows(shot, first, states, confidence)
    table = [r if r[1] != NOT_OBSERVABLE else [*r[:4], ""] for r in table]
    if pictures:
        grid, values, y0, dy = rows
        coarse = Grid(
            grid.t0_ms, grid.dt_ms * PICTURE_LEVEL, -(-grid.n // PICTURE_LEVEL)
        )
        title = (
            f"{shot} ({year})   AE suggestions, {METHOD} {version} "
            f"({blob['candidate']}), not reviewed"
        )
        draw(
            Path(pictures_dir) / f"{shot}.jpg",
            title=title,
            grid=coarse,
            values=pool(values, PICTURE_LEVEL, "image"),
            y0=y0,
            dy=dy,
            first=first,
            prob=prob,
            threshold=blob["threshold"],
        )
    runs = sum(1 for r in table if r[1] == PRESENT)
    summary = {
        "shot": shot,
        "year": year,
        "window_start_ms": lo,
        "window_end_ms": hi,
        "frames": n,
        "present_frames": int((states == PRESENT).sum()),
        "not_observable_frames": int((states == NOT_OBSERVABLE).sum()),
        "present_runs": runs,
        "max_prob": round(float(prob.max()), 4) if n else "",
    }
    return {
        "rows": table,
        "prob": prob.astype(np.float16),
        "first": first,
        "summary": summary,
        "drawn": pictures,
    }


def _work(job):
    return label_shot(job, pictures=_WORKER.get("pictures", True))


def _init_shard(
    model_file: str,
    root: str,
    corpus: str,
    pictures: bool,
    version: str = VERSION,
    pictures_dir: str | None = None,
) -> None:
    _init(model_file, root, corpus)
    _WORKER.update(pictures=pictures, version=version, pictures_dir=pictures_dir)


def _passing_bar(models: Path, runs: Path | None = None) -> dict:
    """D47: extension requires an evaluation with both A1 and A2 passed, of a
    models directory outside `runs` (where a scoring can be repeated)."""
    path = models / "evaluation.json"
    if runs is not None and pilot_area(models, runs):
        raise ValueError(
            f"{models}: the extension's gate is never read under {runs}; "
            "a pilot's evaluation is not a test"
        )
    message = f"{path}: extension requires bar A1 and A2 both true"
    try:
        evaluation = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{message}: {error}") from error
    bar = evaluation.get("bar") if isinstance(evaluation, dict) else None
    if not isinstance(bar, dict) or any(bar.get(k) is not True for k in ("A1", "A2")):
        raise ValueError(message)
    meta = evaluation.get("meta")
    if not isinstance(meta, dict) or meta.get("limit") != 0:
        raise ValueError(f"{path}: extension requires a full evaluation (limit 0)")
    choice = models / "chosen.json"
    try:
        candidate = json.loads(choice.read_bytes())["candidate"]
        if meta.get("candidate") != candidate:
            raise ValueError(f"{choice}: candidate differs from the evaluation")
        for key, file in (
            ("chosen_sha256", choice),
            ("model_sha256", models / candidate / "model.pt"),
        ):
            if not meta.get(key) or meta[key] != sha256_of(file):
                raise ValueError(f"{file}: missing or different {key}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error
    return bar


def _bound_model(models: Path, version: str) -> Path:
    """The chosen model, its directory and checkpoint bound to `version`."""
    check_bound(version, models)
    file = chosen_model(models)
    check_bound(version, models, load(file)[1], file)
    return file


def run_shard(
    paths: Paths,
    *,
    models: Path,
    k: int,
    of: int,
    workers: int = 1,
    limit: int = 0,
    pictures: bool = True,
    version: str = VERSION,
) -> dict:
    _passing_bar(models, paths.runs)
    _bound_model(models, version)
    jobs = shard_shots(paths, k, of)
    inputs = _shard_inputs(paths, models)
    if limit:
        jobs = jobs.iloc[:limit]
    jobs = [tuple(int(v) for v in row) for row in jobs.itertuples(index=False)]
    out = suggestions_dir(paths, version) / "shards"
    if limit > 0:
        out /= "pilot"
    init = (
        str(chosen_model(models)),
        str(paths.root),
        str(paths.corpus),
        pictures,
        version,
        str(out / "extension") if limit > 0 else None,
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"{k}.json"
    # A partial replacement must never inherit the previous completion marker.
    manifest.unlink(missing_ok=True)
    rows, summaries, probs, failed, drawn = [], [], {}, [], []
    for job, outcome in run_all(_work, jobs, workers, _init_shard, init):
        if isinstance(outcome, Exception):
            failed.append(
                {"shot": job[0], "error": f"{type(outcome).__name__}: {outcome}"}
            )
            continue
        rows += outcome["rows"]
        summaries.append(outcome["summary"])
        if outcome.get("drawn"):
            drawn.append(job[0])
        probs[f"p{job[0]}"] = outcome["prob"]
        probs[f"f{job[0]}"] = np.int64(outcome["first"])
    with atomic_path(out / f"{k}.csv") as tmp:
        pd.DataFrame(rows, columns=list(suggestions.COLUMNS)).to_csv(tmp, index=False)
    with atomic_path(out / f"{k}.summary.csv") as tmp:
        pd.DataFrame(summaries, columns=list(SUMMARY_COLUMNS)).to_csv(tmp, index=False)
    with atomic_path(out / f"{k}.npz") as tmp, open(tmp, "wb") as f:
        np.savez_compressed(f, **probs)
    with atomic_path(out / f"{k}.failed.jsonl") as tmp:
        tmp.write_text("".join(json.dumps(row) + "\n" for row in failed))
    if _shard_inputs(paths, models) != inputs:
        raise ValueError(f"{manifest}: shard inputs changed during execution")
    record = {
        "k": k,
        "of": of,
        "shots": [job[0] for job in jobs],
        "pictures": sorted(drawn),
        **inputs,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(manifest) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    return {
        "shard": k,
        "shots": len(jobs),
        "done": len(summaries),
        "failed": len(failed),
    }


def _check_payload(shards: Path, k: int, summary: pd.DataFrame) -> pd.DataFrame:
    """Reconcile shot coverage, tiled frame windows/counts and probability arrays."""
    csv = shards / f"{k}.csv"
    table = pd.read_csv(csv, keep_default_na=False)
    done = set(summary.shot)
    difference = set(table.shot) ^ done
    if difference:
        raise ValueError(
            f"{csv}: shot {sorted(difference)} differs from successful shots"
        )
    file = shards / f"{k}.npz"
    expected_keys = {f"{prefix}{shot}" for shot in done for prefix in ("p", "f")}
    with np.load(file, allow_pickle=False) as arrays:
        difference = set(arrays.files) ^ expected_keys
        if difference or len(arrays.files) != len(expected_keys):
            raise ValueError(
                f"{file}: shot keys {sorted(difference)} differ from successful shots"
            )
        for row in summary.itertuples(index=False):
            shot = row.shot
            rows = table[table.shot == shot].sort_values("t_start")
            first, n = window_frames((row.window_start_ms, row.window_end_ms))
            starts = rows.t_start.to_numpy(dtype=float)
            ends = rows.t_end.to_numpy(dtype=float)
            if (
                not len(rows)
                or starts[0] != first * FRAME_MS
                or ends[-1] != (first + n) * FRAME_MS
                or not np.array_equal(starts[1:], ends[:-1])
                or np.any(ends <= starts)
                or np.any(starts % FRAME_MS)
                or np.any(ends % FRAME_MS)
                or not rows.category.isin([ABSENT, PRESENT, NOT_OBSERVABLE]).all()
            ):
                raise ValueError(
                    f"{csv}: shot {shot} rows do not tile its frame window"
                )
            lengths = (ends - starts) / FRAME_MS
            compared = {
                "frames": lengths.sum(),
                "present_frames": lengths[rows.category == PRESENT].sum(),
                "not_observable_frames": lengths[rows.category == NOT_OBSERVABLE].sum(),
                "present_runs": int((rows.category == PRESENT).sum()),
            }
            for column, value in compared.items():
                if getattr(row, column) != value:
                    raise ValueError(
                        f"{csv}: shot {shot} {column} differs from summary"
                    )
            prob, offset = arrays[f"p{shot}"], arrays[f"f{shot}"]
            if prob.shape != (n,) or offset.shape != () or offset.item() != first:
                raise ValueError(f"{file}: shot {shot} array grid differs from summary")
            if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
                raise ValueError(f"{file}: shot {shot} has invalid probabilities")
    return table


def _read_failures(path: Path) -> list[dict]:
    failures = [json.loads(line) for line in path.read_text().splitlines()]
    for row in failures:
        if (
            not isinstance(row, dict)
            or type(row.get("shot")) is not int
            or not isinstance(row.get("error"), str)
        ):
            raise ValueError(f"{path}: expected an integer shot and string error")
    return failures


def merge(paths: Paths, *, models: Path, of: int, version: str = VERSION) -> dict:
    bar = _passing_bar(models, paths.runs)
    _bound_model(models, version)
    shards = suggestions_dir(paths, version) / "shards"
    if of <= 0:
        raise ValueError(f"{shards}: of must be positive")
    missing = [
        k
        for k in range(of)
        if not all(
            (shards / f"{k}{suffix}").is_file()
            for suffix in (".json", ".summary.csv", ".csv", ".failed.jsonl", ".npz")
        )
    ]
    if missing:
        raise FileNotFoundError(
            f"{shards}: shards {missing} of {of} have not been written completely"
        )
    inputs = _shard_inputs(paths, models)
    manifests, summaries, tables, failed, given = [], [], [], [], []
    drawn = set()
    for k in range(of):
        path = shards / f"{k}.json"
        try:
            manifest = json.loads(path.read_text())
            for key, expected in {"k": k, "of": of, **inputs}.items():
                if manifest.get(key) != expected:
                    raise ValueError(f"{key} differs from the current inputs")
            assigned = manifest["shots"]
            expected = shard_shots(paths, k, of).shot.tolist()
            if assigned != expected:
                raise ValueError("given shots differ from the eligible shard")
            summary = pd.read_csv(shards / f"{k}.summary.csv")
            failures = _read_failures(shards / f"{k}.failed.jsonl")
            done = summary.shot.tolist()
            failed_shots = [row["shot"] for row in failures]
            if Counter(done + failed_shots) != Counter(dict.fromkeys(assigned, 1)):
                raise ValueError("done plus failed must equal given shots exactly once")
            pictures = manifest.get("pictures", [])
            if not isinstance(pictures, list) or not set(pictures) <= set(done):
                raise ValueError("pictures must be shots the shard labelled")
            tables.append(_check_payload(shards, k, summary))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}: {error}") from error
        manifests.append(manifest)
        summaries.append(summary)
        failed.extend(failures)
        given.extend(assigned)
        drawn |= set(pictures)
    eligible = shard_shots(paths, 0, 1).shot.tolist()
    if Counter(given) != Counter(dict.fromkeys(eligible, 1)):
        raise ValueError(f"{shards}: given shots do not cover the eligible population")
    if len(failed) > MAX_FAILED_FRACTION * len(eligible):
        raise ValueError(
            f"{shards}: {len(failed)} of {len(eligible)} eligible shots failed; "
            "exceeds 2 % (MAX_FAILED_FRACTION = 0.02)"
        )
    rows = pd.concat(tables, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True).sort_values(
        "shot", ignore_index=True
    )
    file = chosen_model(models)
    _, blob = load(file)
    meta = {
        "method": METHOD,
        "version": version,
        "event": EVENT,
        "tier": "suggestions",
        "model": str(file),
        "model_sha256": manifests[0]["model_sha256"],
        "candidate": blob["candidate"],
        "threshold": blob["threshold"],
        "band_khz": blob["band_khz"],
        "bar": bar,
        "population": str(paths.catalog / "population.csv"),
        "population_sha256": sha256_of(paths.catalog / "population.csv"),
        "min_co2_s": MIN_CO2_S,
        "shards": of,
        "failed": len(failed),
        "max_failed_fraction": MAX_FAILED_FRACTION,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    table = suggestions.table_path(paths, EVENT, METHOD, version)
    suggestions.write_table(table, rows.itertuples(index=False), meta)
    with atomic_path(table.parent / "summary.csv") as tmp:
        summary.to_csv(tmp, index=False)
    with atomic_path(table.parent / "failed.jsonl") as tmp:
        tmp.write_text("".join(json.dumps(row) + "\n" for row in failed))
    index = [
        {
            "shot": int(r.shot),
            "group": "extension",
            "split": "extension",
            "file": f"extension/{int(r.shot)}.jpg",
            "window_start_ms": int(r.window_start_ms),
            "window_end_ms": int(r.window_end_ms),
            "model_present_frames": int(r.present_frames),
            "reference_present_frames": "",
            "f1_vs_owner": "",
            "threshold": blob["threshold"],
            "candidate": blob["candidate"],
            "version": version,
        }
        for r in summary.itertuples(index=False)
        # Only the pictures these shards drew: never a pilot's or an earlier run's.
        if int(r.shot) in drawn
        and (gallery_dir(paths, version) / "extension" / f"{int(r.shot)}.jpg").is_file()
    ]
    write_index(gallery_dir(paths, version) / "index.csv", index)
    return {
        "table": str(table),
        "shots": len(summary),
        "failed": len(failed),
        "pictures": len(index),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    )
    p.add_argument(
        "--limit", type=int, default=0, help="the first N shots of the shard"
    )
    p.add_argument("--no-pictures", action="store_true")
    p.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_xpower/<version>"
    )
    p.add_argument("--version", default=VERSION)
    args = p.parse_args(argv)
    paths = Paths.from_env()
    models = args.models or model_dir(paths, args.version)
    if not args.merge and not 0 <= args.shard < args.of:
        p.error("--shard must be in 0 .. --of - 1")
    try:
        if args.merge:
            result = merge(paths, models=models, of=args.of, version=args.version)
        else:
            result = run_shard(
                paths,
                models=models,
                k=args.shard,
                of=args.of,
                workers=args.workers,
                limit=args.limit,
                pictures=not args.no_pictures,
                version=args.version,
            )
    except (CatalogError, OSError, ValueError) as error:
        p.error(str(error))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
