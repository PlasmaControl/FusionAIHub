"""The headroom table: one row per stage-0 task, built from the scores that are already stored.

This module trains and scores nothing. It reads `<stage0>/scores/<task>/<predictor>/<split>/score.json`
(and the evaluator's `shots_metrics.csv` beside it, for the direct model's interval) that
`scripts/mast/check_harness.py` and `scripts/mast/run_headroom.py` wrote, and `<stage0>/direct/<task>/
<variant>/train_log.csv` to choose among direct variants.

Columns (`COLUMNS`). `mean`, `persistence`, `persistence_all`, `floor` on `test` and on `val_report`
(`_test`, `_val`); `shrunk`, `ridge` and `direct` on `val_report` only, because fitted predictors never
see the test shots (`headroom.FITTED`, enforced by `run_headroom.py`). `direct_lo` and `direct_hi` are the
95% interval of the direct model's mean over shots (`bootstrap_ci`); `tokamind_repl` is our replication
of TokaMind's fine-tuned score (`TOKAMIND_REPLICATION`); `best_published` is `tasks.best_published` and
`best_published_by` names whose score it is; `direct_over_best` and `verdict` compare the direct model with
it. The published scores are on TokaMark's test split and the direct model's is on `val_report`: the same
predictor differs between the two splits by up to 0.04 (measured 2026-11-22, task 4-5 persistence 0.910
against 0.870), which is why the baselines and the floor stay beside it on both splits. The verdict is a
comparison across splits and says so on the page.

Missing, stale, unfinished.
- A cell with no stored score is NaN, never 0.
- A stored score that is not current is shown, and named in the `stale` column as `<column>: <reason>`.
  Whether a score is current is decided by one function. `build_table` takes it as `stale_reason(task,
  predictor, split) -> str` (empty: current), which `scripts/mast/run_headroom.py` supplies as
  `stale_reason`: the same `_kept_score` that makes the command score again, with the provenance of the
  predictor as loaded (predictor hash, fit or checkpoint hash, TokaMind's coefficient indices) and, for a
  direct score, the code that trained the checkpoint (`trained_by.json`, copied into the record as
  `trained_predictor_sha256` and `trained_code_sha256`: "trained by other code than the code now loaded", or
  "training code not recorded"; a mark, not part of the rule that keeps a score). Without it only the scoring
  code is checked, by `baselines.read_kept_score`, which the command's rule starts with. The rule covers our
  code, the fit or checkpoint and TokaMind's files only: a score is NOT marked when TokaMark's code (windows,
  statistics, splits, the metric) or the Zarr data root changed, because the record holds neither.
- A direct score whose `trained_steps` is below `max_steps`, or that has no `trained_steps`, comes from a
  training that did not finish. It counts and is named in the `unfinished` column.
- The variants listing (`direct_variants`, `variants_markdown`) carries the same two marks for every variant,
  not only for the one the table takes: it is the evidence for whether the base run is under-trained.
- The direct cell of a task is the variant whose `train_log.csv` has the lowest best `val_nrmse` (selection
  on `val_select`; the score is on `val_report`), `direct_variants` lists them all. A variant whose log
  cannot be read ranks last.
- `not_competitive` is "yes" where the direct score is worse than our TokaMind replication on the task (the
  rule `headroom/direct.py` states: a weak model is not a headroom finding). Both numbers are on different
  splits (`val_report` against test), as for the verdict.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from tokamak_foundation_model.ignite.mast import baselines, tasks
from tokamak_foundation_model.ignite.mast.headroom import FITTED  # noqa: F401  (the table's own name for it)

# Our replication of TokaMind's fine-tuned column with today's benchmark code, on TokaMark's test split.
# Measured 2026-11-22 (runs finished 04:36 UTC), source /models/tokamind/compare_to_paper.py. Every task is
# within 0.0022 of the paper's fine-tuned column (largest difference task 1-3); the mean is 0.2124. A constant
# here and not in `tasks.PUBLISHED` because `tasks.py` is hashed into every stored score.
TOKAMIND_REPLICATION: dict[str, float | None] = {
    "task_1-1": 0.1503, "task_1-2": 0.0433, "task_1-3": 0.1198,
    "task_2-1": 0.1809, "task_2-2": 0.0439, "task_2-3": 0.0917,
    "task_3-1": 0.3457, "task_3-2": 0.2671, "task_3-3": 0.2197,
    "task_4-1": 0.2147, "task_4-2": 0.2155, "task_4-3": 0.0810, "task_4-4": 0.3502, "task_4-5": 0.6499,
}

BOOTSTRAP_SEED = 0
N_BOOT = 2000
SPLIT_SUFFIX = {"test": "test", "val_report": "val"}

# (column, predictor, split). `direct` is the variant chosen per task, not a fixed directory name.
CELLS: tuple[tuple[str, str, str], ...] = (
    ("mean_test", "mean", "test"), ("mean_val", "mean", "val_report"),
    ("persistence_test", "persistence", "test"), ("persistence_val", "persistence", "val_report"),
    ("persistence_all_test", "persistence_all", "test"), ("persistence_all_val", "persistence_all", "val_report"),
    ("floor_test", "floor", "test"), ("floor_val", "floor", "val_report"),
    ("shrunk_val", "shrunk", "val_report"), ("ridge_val", "ridge", "val_report"),
    ("direct_val", "direct", "val_report"),
)
COLUMNS = (*(c for c, _, _ in CELLS), "direct_lo", "direct_hi", "direct_variant", "tokamind_repl",
           "best_published", "best_published_by", "direct_over_best", "verdict", "not_competitive", "stale",
           "unfinished")
BASE_VARIANT = "base"
# What the † mark says (the table's legend and the variants listing use it). Every reason the command's rule
# can give is named; the last sentence is the limit of the rule (the record holds no TokaMark revision or data
# root, so a change of either is not seen).
STALE_LEGEND = (
    "† stale: the stored score is not the score of what is loaded now. The scoring code (harness, tasks, baselines), "
    "the predictor's code, its fit or checkpoint, or TokaMind's coefficient files differ from those it was made "
    "with; or the record or its provenance is unreadable or cannot be computed now; or the per-shot file does not "
    "match the score; for a direct score also 'trained by other code than the code now loaded' (the checkpoint "
    "was trained by other code than the predictor and scoring code now loaded) and 'training code not "
    "recorded' (no `trained_by.json`). A score is not marked when TokaMark's code (windows, statistics, splits, "
    "the metric) or the Zarr data root changed: the record holds neither.")
_CONSISTENT = 1e-6   # the per-shot file's mean must equal the stored score to this


def bootstrap_ci(per_shot_nrmse: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """The 95% interval of the mean over shots: percentiles 2.5 and 97.5 of `n_boot` resampled means."""
    x = np.asarray(per_shot_nrmse, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"bootstrap_ci needs a non-empty 1-D array of per-shot scores, got shape {x.shape}")
    idx = np.random.default_rng(seed).integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def verdict(score: float, lo: float, hi: float, reference: float) -> str:
    """"ahead" if the whole interval is below the reference, "behind" if it is above, else "tie"."""
    if not all(math.isfinite(v) for v in (score, lo, hi, reference)):
        return "missing"
    if hi < reference:
        return "ahead"
    if lo > reference:
        return "behind"
    return "tie"


def _read_record(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _nrmse(record: dict) -> float:
    try:
        return float(record["nrmse"])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _score_dir(scores_dir: Path, task: str, predictor: str, split: str) -> Path:
    return Path(scores_dir) / task / predictor / split


def _per_shot(out_dir: Path, task: str) -> np.ndarray | None:
    """A shot's value is the mean of its rows in `shots_metrics.csv` (TokaMark's task score is the mean over
    shots of the shot's mean over signals; `harness.score_batches` checks that identity)."""
    try:
        rows = pd.read_csv(out_dir / task / "shots_metrics.csv", usecols=["shot_id", "NRMSE_mean"])
    except (OSError, ValueError):
        return None
    values = rows.groupby("shot_id")["NRMSE_mean"].mean().dropna().to_numpy()
    return values if values.size else None


def _best_val_nrmse(run_dir: Path) -> float:
    try:
        return float(pd.read_csv(run_dir / "train_log.csv")["val_nrmse"].min())
    except (OSError, ValueError, KeyError):
        return math.nan


def predictor_name(predictor: str, variant: str) -> str:
    """`direct` for the base variant, `direct.<variant>` for another; the other predictors have no variants.
    `scripts/mast/run_headroom.py` imports this, so the two cannot disagree on a name."""
    return predictor if variant == BASE_VARIANT else f"{predictor}.{variant}"


def direct_run_dir(root: Path, task_name: str, variant: str) -> Path:
    """`<stage0>/direct/<task>/<variant>`: a direct model's checkpoints, config and training log."""
    return Path(root) / "direct" / task_name / variant


def _variant_of(predictor: str) -> str:
    return BASE_VARIANT if predictor == "direct" else predictor.partition(".")[2]


def _predictor_of(variant: str) -> str:
    return predictor_name("direct", variant)


def _reason(scores_dir: Path, task: str, name: str, split: str,
            stale_reason: Callable[[str, str, str], str] | None) -> str:
    """Why the stored score of `name` is not current: the command's rule if given, else the scoring code's."""
    if stale_reason is not None:
        return stale_reason(task, name, split)
    return baselines.read_kept_score(_score_dir(scores_dir, task, name, split), task, name, split)[1]


def _config(run_dir: Path) -> dict:
    try:
        cfg = json.loads((run_dir / "config.json").read_text())["cfg"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


def direct_variants(scores_dir: Path, task: str,
                    stale_reason: Callable[[str, str, str], str] | None = None) -> pd.DataFrame:
    """Every direct variant of the task that has a training log or a `val_report` score.

    Columns: `variant`, `best_val_nrmse` (lowest of its log, on `val_select`), `val_report` (its stored
    score, NaN if none), `differs` (the `config.json` fields that differ from the base variant's, as
    `name=value`; empty for the base, and for a variant when there is no base config to compare with), `stale`
    (why its stored score is not current, as `build_table` decides it with `stale_reason`; empty if current or
    there is no score) and `unfinished` (as the table's column). The base variant comes first, the others by
    name.
    """
    scores_dir = Path(scores_dir)
    stage0 = scores_dir.parent
    runs = direct_run_dir(stage0, task, BASE_VARIANT).parent
    names = {p.name for p in runs.glob("*") if p.is_dir()} if runs.is_dir() else set()
    for p in (scores_dir / task).glob("direct*") if (scores_dir / task).is_dir() else ():
        if p.name == "direct" or p.name.startswith("direct."):
            if (p / "val_report" / "score.json").exists():
                names.add(_variant_of(p.name))
    base_cfg = _config(direct_run_dir(stage0, task, BASE_VARIANT))
    rows = []
    for variant in sorted(names, key=lambda v: (v != BASE_VARIANT, v)):
        out_dir = _score_dir(scores_dir, task, _predictor_of(variant), "val_report")
        record = _read_record(out_dir / "score.json")
        cfg = _config(direct_run_dir(stage0, task, variant))
        differs = ""
        if variant != BASE_VARIANT and base_cfg and cfg:
            differs = ", ".join(f"{k}={cfg.get(k)}" for k in sorted({*base_cfg, *cfg})
                                if cfg.get(k) != base_cfg.get(k))
        stale = unfinished = ""
        if record is not None:
            stale = _reason(scores_dir, task, _predictor_of(variant), "val_report", stale_reason)
            unfinished = _unfinished(record)
        elif (out_dir / "score.json").exists():
            stale = "unreadable score.json"
        rows.append({"variant": variant, "best_val_nrmse": _best_val_nrmse(direct_run_dir(stage0, task, variant)),
                     "val_report": _nrmse(record) if record else math.nan, "differs": differs,
                     "stale": stale, "unfinished": unfinished})
    return pd.DataFrame(rows, columns=["variant", "best_val_nrmse", "val_report", "differs", "stale", "unfinished"])


def _chosen_direct(scores_dir: Path, task: str) -> str | None:
    """The variant with a stored `val_report` score and the lowest best `val_nrmse`; None if there is none."""
    candidates = [r for r in direct_variants(scores_dir, task).itertuples()
                  if (_score_dir(scores_dir, task, _predictor_of(r.variant), "val_report") / "score.json").exists()]
    if not candidates:
        return None
    best = min(candidates, key=lambda r: (r.best_val_nrmse if math.isfinite(r.best_val_nrmse) else math.inf,
                                          r.variant != BASE_VARIANT, r.variant))
    return best.variant


def _unfinished(record: dict) -> str:
    steps, max_steps = record.get("trained_steps"), record.get("max_steps")
    if steps is None:
        return "trained steps not recorded"
    if max_steps is None:
        return "max_steps not recorded"
    return f"trained {steps} of {max_steps} steps" if steps < max_steps else ""


def build_table(scores_dir: Path, tokamind_repl: dict[str, float | None],
                stale_reason: Callable[[str, str, str], str] | None = None,
                task_names: Sequence[str] = tasks.STAGE0_TASKS) -> pd.DataFrame:
    """The headroom table of the tasks of `task_names` (default: the seven stage-0 tasks), one row each in
    that order (see the module docstring); index `task`.

    `scores_dir` is `<stage0>/scores`. `stale_reason(task, predictor, split)` is asked once for each stored
    record and returns "" or why the record is not current; without it the scoring code is checked.
    """
    scores_dir = Path(scores_dir)
    rows = []
    for task in task_names:
        row: dict = {c: math.nan for c in COLUMNS}
        row.update(direct_variant="", verdict="missing", not_competitive="")
        stale, unfinished = [], []
        chosen = _chosen_direct(scores_dir, task)
        direct_record = None
        for column, predictor, split in CELLS:
            name = _predictor_of(chosen) if predictor == "direct" and chosen else predictor
            out_dir = _score_dir(scores_dir, task, name, split)
            if not (out_dir / "score.json").exists():
                continue
            record = _read_record(out_dir / "score.json")
            if record is None:
                stale.append(f"{column}: unreadable score.json")
                continue
            row[column] = _nrmse(record)
            reason = _reason(scores_dir, task, name, split, stale_reason)
            if reason:
                stale.append(f"{column}: {reason}")
            if predictor == "direct":
                direct_record, row["direct_variant"] = record, chosen
                if why := _unfinished(record):
                    unfinished.append(f"{column}: {why}")
                shots = _per_shot(out_dir, task)
                if shots is not None and abs(float(shots.mean()) - row[column]) <= _CONSISTENT:
                    row["direct_lo"], row["direct_hi"] = bootstrap_ci(shots)
                elif shots is not None:
                    stale.append(f"{column}: per-shot file does not match score.json "
                                 f"(mean {float(shots.mean()):.6f}, score {row[column]:.6f})")
        row["tokamind_repl"] = tokamind_repl.get(task) if tokamind_repl.get(task) is not None else math.nan
        row["best_published"] = tasks.best_published(task)
        published = tasks.PUBLISHED[task]
        row["best_published_by"] = next(k for k in ("tokamind_paper", "tokamind_rerun", "gentoka")
                                        if published[k] == row["best_published"])
        if direct_record is not None and math.isfinite(row["direct_val"]):
            row["direct_over_best"] = row["direct_val"] / row["best_published"]
            row["verdict"] = ("no interval" if math.isnan(row["direct_lo"]) else
                              verdict(row["direct_val"], row["direct_lo"], row["direct_hi"],
                                      row["best_published"]))
        if math.isfinite(row["direct_val"]) and math.isfinite(row["tokamind_repl"]) \
                and row["direct_val"] > row["tokamind_repl"]:
            row["not_competitive"] = "yes"  # the promise of `direct.py`: worse than our replication on the task
        row["stale"], row["unfinished"] = " | ".join(stale), " | ".join(unfinished)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(list(task_names), name="task"), columns=list(COLUMNS))


def projected_mean(table: pd.DataFrame) -> float:
    """The 14-task mean with the direct model's score on the tasks that are rows of `table` and the best
    published score on the tasks that are not. NaN while any row has no direct score: a mean over a partial
    set would read as a result. A stale or unfinished direct score counts (it is named in the table)."""
    direct = table["direct_val"]
    if direct.isna().any():
        return math.nan
    values = [float(direct[t]) if t in direct.index else tasks.best_published(t) for t in tasks.ALL_TASKS]
    return float(np.mean(values))


# ---------------------------------------------------------------- Markdown for the results page

_PERSISTENCE_COLUMNS = ("persistence_test", "persistence_val", "persistence_all_test", "persistence_all_val")


def _marks(table: pd.DataFrame, task: str) -> dict[str, str]:
    marks: dict[str, str] = {}
    for column, mark in (("stale", "†"), ("unfinished", "‡")):
        for entry in str(table.loc[task, column]).split(" | "):
            if entry:
                cell = entry.partition(": ")[0]
                marks[cell] = marks.get(cell, "") + mark
    return marks


def _cell(table: pd.DataFrame, task: str, column: str, marks: dict[str, str], digits: int = 4) -> str:
    value = table.loc[task, column]
    if not isinstance(value, str) and pd.isna(value):
        undefined = column in _PERSISTENCE_COLUMNS and tasks.PUBLISHED[task]["persistence"] is None
        return "undefined" if undefined else "missing"
    text = value if isinstance(value, str) else f"{value:.{digits}f}"
    return f"{text} {marks[column]}" if column in marks else text


def _interval(table: pd.DataFrame, task: str, marks: dict[str, str]) -> str:
    text = _cell(table, task, "direct_val", marks)
    lo, hi = table.loc[task, "direct_lo"], table.loc[task, "direct_hi"]
    return text if pd.isna(lo) or pd.isna(hi) else f"{text} [{lo:.4f}, {hi:.4f}]"


def _md(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|",
            *("| " + " | ".join(r) + " |" for r in rows)]


def to_markdown(table: pd.DataFrame) -> str:
    """The table as two Markdown tables (references on both splits; fitted predictors against the published
    scores), a legend, and the stale and unfinished entries by name. Pasteable into the results page."""
    refs, head, notes = [], [], []
    for task in table.index:
        marks, label = _marks(table, task), task.removeprefix("task_")
        refs.append([label, *(_cell(table, task, c, marks) for c in (
            "mean_test", "mean_val", "persistence_test", "persistence_val", "persistence_all_test",
            "persistence_all_val", "floor_test", "floor_val"))])
        head.append([label, _cell(table, task, "shrunk_val", marks), _cell(table, task, "ridge_val", marks),
                     _interval(table, task, marks), str(table.loc[task, "direct_variant"]) or "none",
                     _cell(table, task, "tokamind_repl", marks), _cell(table, task, "best_published", marks),
                     str(table.loc[task, "best_published_by"]), _cell(table, task, "direct_over_best", marks, 2),
                     str(table.loc[task, "verdict"])
                     + ("; not competitive" if table.loc[task, "not_competitive"] == "yes" else "")])
        for column, kind in (("stale", "stale"), ("unfinished", "unfinished")):
            notes += [f"- {label} {entry} ({kind})" for entry in str(table.loc[task, column]).split(" | ") if entry]
    lines = ["Scores on the benchmark's windows, NRMSE (lower is better). `test` is TokaMark's test split, `val` is "
             "`val_report`.", "",
             *_md(["task", "mean test", "mean val", "persistence test", "persistence val",
                   "persistence (all windows) test", "persistence (all windows) val", "floor test", "floor val"],
                  refs), "",
             "Fitted predictors and the direct model on `val_report` (with the 95% interval of the direct model's "
             "mean over shots), against the published scores on the test split.", "",
             *_md(["task", "shrunk val", "ridge val", "direct val [95% interval]", "direct variant",
                   "TokaMind replication test", "best published test", "by", "direct / best", "verdict"], head), "",
             "missing: no score is stored. undefined: the benchmark defines no persistence for the task. "
             + STALE_LEGEND + " ‡ unfinished: the direct model's training "
             "stopped before `max_steps`. not competitive: the direct model's `val_report` score is worse than our "
             "TokaMind replication's test score on the task."]
    if notes:
        lines += ["", *notes]
    return "\n".join(lines) + "\n"


def variants_markdown(scores_dir: Path, stale_reason: Callable[[str, str, str], str] | None = None,
                      task_names: Sequence[str] = tasks.STAGE0_TASKS) -> str:
    """Every trained direct variant of each task of `task_names` (default: the seven stage-0 tasks), as
    Markdown: what differs in its `config.json` from the base variant, its best `val_nrmse` (on `val_select`,
    from `train_log.csv`), its `val_report` score ("no score" when none is stored: the table cannot take it,
    which matters when its selection error is the lowest) with the table's marks (`stale_reason` as for
    `build_table`: † stale, ‡ unfinished) and the reasons below the block, and whether the table's direct cell
    is this variant. Empty if no task has a variant. A legend follows when any mark is shown."""
    blocks, marked = [], False
    for task in task_names:
        variants = direct_variants(scores_dir, task, stale_reason)
        if variants.empty:
            continue
        chosen = _chosen_direct(scores_dir, task)
        rows, notes = [], []
        for r in variants.itertuples():
            mark = ("†" if r.stale else "") + ("‡" if r.unfinished else "")
            score = "no score" if math.isnan(r.val_report) else f"{r.val_report:.4f}" + (f" {mark}" if mark else "")
            rows.append([r.variant, r.differs or "-",
                         "unreadable" if math.isnan(r.best_val_nrmse) else f"{r.best_val_nrmse:.4f}", score,
                         "yes" if r.variant == chosen else "no"])
            for text, kind in ((r.stale, "stale"), (r.unfinished, "unfinished")):
                if text:
                    notes.append(f"- {task.removeprefix('task_')} {r.variant} val_report: {text} ({kind})")
        marked = marked or bool(notes)
        blocks += [f"Task {task.removeprefix('task_')}:", "",
                   *_md(["variant", "differs from base", "best val_nrmse (val_select)", "val_report", "in the table"],
                        rows), "", *([*notes, ""] if notes else [])]
    if marked:
        blocks += [STALE_LEGEND + " ‡ unfinished: the training stopped before `max_steps`.", ""]
    return "\n".join(blocks)
