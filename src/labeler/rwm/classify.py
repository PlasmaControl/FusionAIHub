"""RWM presence per 10 ms bin (owner, 2026-10-14: every event's model is a
classifier at t, none a forecaster).

A slice at t (`features.slice_table`, every 10 ms, trailing inputs only) is the
bin [t - 10, t), the frame its inputs read; its target is that frame's category
in the `rwm_onsets` table (`grid.frames`). Present (1) is positive. On Hanson's
shots the rest of the measured time is unassessed (4) in the table, as no rule
tests absence there; it is scored negative here because Hanson's lists are
taken as complete for those shots, the assumption round four's forecaster made.
Uncertain (2: screen candidates, the stretch before each onset, the time after an
onset with no delta-Bp response) and not observable (3) leave both classes, as
the other events' scores leave them; `UNCERTAIN_NEGATIVE` is the sensitivity
that scores uncertain time negative with the same predictions.

The folds hold out one calendar run day of Hanson's shots at a time (the run
records 20180314 and 20180314A share a day), and the F1 cut is chosen on the
training days only, by an inner leave-one-day-out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import grid
from . import metrics
from .evaluate import Imputer
from .forest import BalancedForest

PRESENT, UNCERTAIN, NOT_OBSERVABLE, UNASSESSED = 1, 2, 3, 4
#: Scored negative on Hanson's shots: unassessed time, and absent if ever written.
NEGATIVE = (0, UNASSESSED)
#: The primary score leaves these out of both classes ...
EXCLUDED = (UNCERTAIN, NOT_OBSERVABLE)
#: ... and the sensitivity scores uncertain time negative.
UNCERTAIN_NEGATIVE = "uncertain_negative"
FOREST = {"n_estimators": 300, "max_depth": 8, "min_leaf": 5}


def calendar_day(run_record: str) -> str:
    """The calendar day of a run record: its date, less a same-day letter."""
    return str(run_record).rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def bin_categories(rows: pd.DataFrame, t_ms) -> np.ndarray:
    """The table category of the frame [t - 10, t) for each slice time t, from
    one shot's table `rows` (`category`, `t_start`, `t_end`, painted in order);
    `grid.UNASSESSED` outside the table's window or before t = 10 ms."""
    t = np.asarray(t_ms, dtype=float)
    out = np.full(len(t), grid.UNASSESSED, dtype=np.int8)
    if rows.empty or not len(t):
        return out
    spans = list(
        zip(
            rows.t_start.astype(float),
            rows.t_end.astype(float),
            rows.category.astype(int),
            strict=True,
        )
    )
    window = (min(a for a, _, _ in spans), max(b for _, b, _ in spans))
    n = int(np.ceil(max(window[1], np.nanmax(t)) / grid.GRID_MS)) + 1
    painted = grid.frames(window, spans, 0, n)
    k = np.round(t / grid.GRID_MS).astype(int) - 1
    ok = (k >= 0) & (k < n) & np.isclose(t, (k + 1) * grid.GRID_MS)
    out[ok] = painted[k[ok]]
    return out


def targets(categories, view: str | None = None) -> np.ndarray:
    """1 present, 0 negative, -1 left out; `view=UNCERTAIN_NEGATIVE` scores
    uncertain bins negative."""
    c = np.asarray(categories)
    y = np.full(len(c), -1, dtype=np.int8)
    y[c == PRESENT] = 1
    negative = np.isin(c, NEGATIVE)
    if view == UNCERTAIN_NEGATIVE:
        negative |= c == UNCERTAIN
    elif view is not None:
        raise ValueError(f"unknown view {view!r}")
    y[negative] = 0
    return y


def fit(train: pd.DataFrame, columns, seed=0, forest=None):
    """A balanced forest on `train`'s scored bins (`y` 0 or 1)."""
    rows = train[train.y >= 0]
    imputer = Imputer().fit(rows, columns)
    model = BalancedForest(seed=seed, **(forest or FOREST)).fit(
        imputer.transform(rows), rows.y.to_numpy()
    )
    return lambda frame: model.predict_proba(imputer.transform(frame))


def f1_cut(score, label) -> float:
    """The cut `score >= cut` of highest F1 (ties: the higher cut)."""
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    order = np.argsort(-score, kind="stable")
    s, y = score[order], label[order]
    last = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.cumsum(y)[last]
    called = last + 1
    f1 = 2 * tp / (called + y.sum())
    return float(s[last][int(np.argmax(f1))])


def cross_validate(table: pd.DataFrame, columns, seed=0, forest=None) -> pd.DataFrame:
    """Held-out `score` and `call` for every row of `table` (`shot`, `day`, `y`
    and `columns`): each day predicted by a forest fitted on the other days, at
    the F1 cut of an inner leave-one-day-out over those days (`cut`)."""
    out = table[["shot", "t_ms", "day", "y"]].copy()
    out["score"], out["cut"] = np.nan, np.nan
    for day in sorted(table.day.unique()):
        held = (table.day == day).to_numpy()
        train = table[~held]
        inner = []
        for other in sorted(train.day.unique()):
            part = (train.day == other).to_numpy()
            scored = train[part & (train.y >= 0).to_numpy()]
            score = fit(train[~part], columns, seed, forest)(scored)
            inner.append((score, scored.y.to_numpy()))
        cut = f1_cut(*(np.concatenate(v) for v in zip(*inner, strict=True)))
        out.loc[held, "score"] = fit(train, columns, seed, forest)(table[held])
        out.loc[held, "cut"] = cut
    out["call"] = out.score >= out.cut
    return out


def shot_records(oof: pd.DataFrame, y_column: str = "y") -> list[dict]:
    """One record per shot with scored bins: its scores, calls and targets."""
    records = []
    for shot, part in oof[oof[y_column] >= 0].groupby("shot", sort=True):
        records.append(
            {
                "shot": int(shot),
                "day": str(part.day.iloc[0]),
                "score": part.score.to_numpy(dtype=float),
                "call": part.call.to_numpy(dtype=bool),
                "y": part[y_column].to_numpy(dtype=bool),
            }
        )
    return records


def statistic(groups: dict) -> dict:
    """Bin-level F1 (at each fold's cut), precision, recall, AUROC and AUPRC, and
    the prevalence, pooled over the shots of every group."""
    records = [r for v in groups.values() for r in v]
    score = np.concatenate([r["score"] for r in records])
    call = np.concatenate([r["call"] for r in records])
    y = np.concatenate([r["y"] for r in records])
    tp, fp = int((call & y).sum()), int((call & ~y).sum())
    fn = int((~call & y).sum())
    nan = float("nan")
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else nan,
        "precision": tp / (tp + fp) if tp + fp else nan,
        "recall": tp / (tp + fn) if tp + fn else nan,
        "auroc": metrics.auroc(score, y),
        "auprc": metrics.auprc(score, y),
        "prevalence": float(y.mean()) if len(y) else nan,
    }


def counts(records: list[dict]) -> dict:
    """Scored bins and shots, and the prevalence the trivial baselines use."""
    positive = int(sum(r["y"].sum() for r in records))
    negative = int(sum((~r["y"]).sum() for r in records))
    return {
        "shots": len(records),
        "positive_shots": sum(bool(r["y"].any()) for r in records),
        "positive_bins": positive,
        "negative_bins": negative,
        "prevalence": positive / (positive + negative),
    }


def baseline(prevalence: float) -> dict:
    """A constant predictor on the same bins: F1 of calling every bin present,
    chance AUROC, and AUPRC equal to the prevalence."""
    return {"f1": 2 * prevalence / (1 + prevalence), "auroc": 0.5, "auprc": prevalence}
