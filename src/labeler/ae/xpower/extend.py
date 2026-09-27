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
and `shards/K.failed.txt` under `$LABELER_ROOT/suggestions/ae_xpower/v1/`, and
one JPEG per shot into the gallery's `extension/`. `--merge` checks every shard
is there and writes `alfven_eigenmode_suggest_ae_xpower_v1.csv`, its meta,
`summary.csv`, and the gallery index rows.
"""

from __future__ import annotations

import argparse
import json
import os
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
from . import EVENT, METHOD, VERSION, gallery_dir, model_dir, suggestions_dir
from .data import raw_rows, window_frames
from .evaluate import chosen_model
from .gallery import draw, run_all, write_index
from .train import load, probabilities

MIN_CO2_S = 2.0
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
            f"{shot} ({year})   AE suggestions, {METHOD} {VERSION} "
            f"({blob['candidate']}), not reviewed"
        )
        draw(
            gallery_dir(paths) / "extension" / f"{shot}.jpg",
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
    }


def _work(job):
    return label_shot(job, pictures=_WORKER.get("pictures", True))


def _init_shard(model_file: str, root: str, corpus: str, pictures: bool) -> None:
    _init(model_file, root, corpus)
    _WORKER["pictures"] = pictures


def _passing_bar(models: Path) -> dict:
    """D47: extension requires an evaluation with both A1 and A2 passed."""
    path = models / "evaluation.json"
    message = f"{path}: extension requires bar A1 and A2 both true"
    try:
        evaluation = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{message}: {error}") from error
    bar = evaluation.get("bar") if isinstance(evaluation, dict) else None
    if not isinstance(bar, dict) or any(bar.get(k) is not True for k in ("A1", "A2")):
        raise ValueError(message)
    return bar


def run_shard(
    paths: Paths,
    *,
    models: Path,
    k: int,
    of: int,
    workers: int = 1,
    limit: int = 0,
    pictures: bool = True,
) -> dict:
    _passing_bar(models)
    cohort_path = paths.catalog / "cohort.csv"
    try:
        cohort = read_cohort(cohort_path)
    except CatalogError:
        # Preserve the catalog's diagnostic before the broader ValueError handler.
        raise
    except (OSError, ValueError, TypeError) as error:
        raise CatalogError(f"{cohort_path}: {error}") from error
    blind = cohort.loc[cohort["blind"], "shot"]
    jobs = population_shots(paths.catalog / "population.csv")
    jobs = jobs.loc[~jobs["shot"].isin(blind)].iloc[k::of]
    if limit:
        jobs = jobs.iloc[:limit]
    jobs = [tuple(int(v) for v in row) for row in jobs.itertuples(index=False)]
    init = (str(chosen_model(models)), str(paths.root), str(paths.corpus), pictures)
    rows, summaries, probs, failed = [], [], {}, []
    for job, outcome in run_all(_work, jobs, workers, _init_shard, init):
        if isinstance(outcome, Exception):
            failed.append(f"{job[0]}\t{type(outcome).__name__}: {outcome}")
            continue
        rows += outcome["rows"]
        summaries.append(outcome["summary"])
        probs[f"p{job[0]}"] = outcome["prob"]
        probs[f"f{job[0]}"] = np.int64(outcome["first"])
    out = suggestions_dir(paths) / "shards"
    out.mkdir(parents=True, exist_ok=True)
    with atomic_path(out / f"{k}.csv") as tmp:
        pd.DataFrame(rows, columns=list(suggestions.COLUMNS)).to_csv(tmp, index=False)
    with atomic_path(out / f"{k}.summary.csv") as tmp:
        pd.DataFrame(summaries, columns=list(SUMMARY_COLUMNS)).to_csv(tmp, index=False)
    with atomic_path(out / f"{k}.npz") as tmp, open(tmp, "wb") as f:
        np.savez_compressed(f, **probs)
    with atomic_path(out / f"{k}.failed.txt") as tmp:
        tmp.write_text("".join(line + "\n" for line in failed))
    return {
        "shard": k,
        "shots": len(jobs),
        "done": len(summaries),
        "failed": len(failed),
    }


def merge(paths: Paths, *, models: Path, of: int) -> dict:
    bar = _passing_bar(models)
    shards = suggestions_dir(paths) / "shards"
    missing = [k for k in range(of) if not (shards / f"{k}.summary.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"shards {missing} of {of} have not been written")
    rows = pd.concat(
        [pd.read_csv(shards / f"{k}.csv", keep_default_na=False) for k in range(of)],
        ignore_index=True,
    )
    summary = pd.concat(
        [pd.read_csv(shards / f"{k}.summary.csv") for k in range(of)], ignore_index=True
    ).sort_values("shot", ignore_index=True)
    failed = [
        line
        for k in range(of)
        for line in (shards / f"{k}.failed.txt").read_text().splitlines()
    ]
    file = chosen_model(models)
    _, blob = load(file)
    meta = {
        "method": METHOD,
        "version": VERSION,
        "event": EVENT,
        "tier": "suggestions",
        "model": str(file),
        "model_sha256": sha256_of(file),
        "candidate": blob["candidate"],
        "threshold": blob["threshold"],
        "band_khz": blob["band_khz"],
        "bar": bar,
        "population": str(paths.catalog / "population.csv"),
        "population_sha256": sha256_of(paths.catalog / "population.csv"),
        "min_co2_s": MIN_CO2_S,
        "shards": of,
        "failed": len(failed),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    table = suggestions.table_path(paths, EVENT, METHOD, VERSION)
    suggestions.write_table(table, rows.itertuples(index=False), meta)
    with atomic_path(table.parent / "summary.csv") as tmp:
        summary.to_csv(tmp, index=False)
    with atomic_path(table.parent / "failed.txt") as tmp:
        tmp.write_text("".join(line + "\n" for line in failed))
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
        }
        for r in summary.itertuples(index=False)
        if (gallery_dir(paths) / "extension" / f"{int(r.shot)}.jpg").is_file()
    ]
    write_index(gallery_dir(paths) / "index.csv", index)
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
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_xpower/v1"
    )
    args = p.parse_args(argv)
    paths = Paths.from_env()
    models = args.models or model_dir(paths)
    if not args.merge and not 0 <= args.shard < args.of:
        p.error("--shard must be in 0 .. --of - 1")
    try:
        if args.merge:
            result = merge(paths, models=models, of=args.of)
        else:
            result = run_shard(
                paths,
                models=models,
                k=args.shard,
                of=args.of,
                workers=args.workers,
                limit=args.limit,
                pictures=not args.no_pictures,
            )
    except (CatalogError, FileNotFoundError, ValueError) as error:
        p.error(str(error))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
