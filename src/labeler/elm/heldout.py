"""Score the frozen ELM models on shots a later review labelled (Ricardo Shousha, 2026-10-23).

`scripts/labeler/elm_heldout_evaluate.py` runs it; this module holds what needs a test:
the pins, the provenance of his individual labels, the onset agreement and the scoring
of each method on his panels. No model is refit here.

**Panels.** `A` is the old reviewed set (pin `2026-10-17`, 119 scored shots); `B1` is
the shots of his that are not on a `protect: holdout` list and `B2` the ones that are
(the ELM challenge key's shots, evaluation only). A panel's shots are scored as the
benchmark scores them: the 50 ms bins of `labels.scored_bins`, so a bin counts only when
it lies wholly inside one crowd or absent row. His individual 10 ms rows hold no bin;
they are scored by onset agreement. Uncertain rows (class 2) are in no score.

**What his individual rows are.** 17,109 of his 17,128 individual rows are the `elm_jump`
v2 onsets the page prefilled, floored to the 10 ms grid and accepted (`prefill_accepted`);
the rest are `edited`: added, or moved off the grid cell. An onset method that finds
what the jump-and-hold rule finds therefore scores high, and `elm_jump` itself scores
high by construction. What he changed independently is the spans (`span_agreement`) and
the shot-level calls.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, sha256_of
from . import compare, inputs, labels, methods, score

REVIEWER = "Ricardo Shousha"
GRID_MS = 10.0
ONSET_TOLERANCES_MS = (5.0, 10.0, 20.0)
#: tp, fp, fn, the false alarms that lie in absent time, then the individual rows that
#: were the accepted prefill (and how many a method found), then the edited ones.
ONSET_COLUMNS = (
    "tp",
    "fp",
    "fn",
    "fp_in_absent",
    "prefill_accepted",
    "prefill_accepted_found",
    "edited",
    "edited_found",
)


# ---------------------------------------------------------------- pins and saves


def verify_pin(directory: Path) -> dict[str, str]:
    """The `name -> sha256` of a pin's `SHA256SUMS`, each file re-hashed; a mismatch or
    a missing file is an error (a pin is written once and never edited)."""
    directory = Path(directory)
    out = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.strip().lstrip("*")
        actual = sha256_of(directory / name)
        if actual != digest:
            raise ValueError(f"{directory / name} is not what {directory}/SHA256SUMS says")
        out[name] = digest
    return out


def last_saves(history: Path, name: str = REVIEWER) -> dict[int, dict]:
    """Each shot's last save by `name` in a review `history.jsonl`, and how many he made."""
    out: dict[int, dict] = {}
    for line in Path(history).read_text().splitlines():
        record = json.loads(line)
        if record.get("name") != name:
            continue
        shot = int(record["shot"])
        out[shot] = {**record, "saves": out.get(shot, {}).get("saves", 0) + 1}
    return out


# ------------------------------------------------------------------- provenance


def prefill_cells(onsets_ms) -> np.ndarray:
    """The 10 ms cells (left edges) the page's prefilled onsets fall in."""
    t = np.asarray(onsets_ms, dtype=float)
    return np.unique(np.floor(t / GRID_MS).astype(np.int64) * int(GRID_MS))


def individual_provenance(rows: pd.DataFrame, onsets_ms, window) -> tuple[pd.DataFrame, int]:
    """One shot's individual rows with a `provenance` column, and the prefilled onsets
    he removed.

    `rows` are his non-crowd present rows (`t_start`, `t_end`), `onsets_ms` the shot's
    onsets of the suggestion table the page prefilled from, `window` the window of his
    save. A row is `prefill_accepted` when it is the 10 ms cell of a prefilled onset and
    `edited` otherwise. A removed onset is the cell of one inside the window that has
    no row.
    """
    cells = prefill_cells(onsets_ms)
    starts = np.round(rows.t_start.to_numpy(float)).astype(np.int64)
    accepted = np.isin(starts, cells)
    out = rows.copy()
    out["provenance"] = np.where(accepted, "prefill_accepted", "edited")
    inside = cells[(cells >= window[0]) & (cells < window[1])]
    return out, int((~np.isin(inside, starts)).sum())


def span_agreement(source: pd.DataFrame, his: pd.DataFrame) -> dict:
    """Milliseconds of the source's present spans and of his present spans, and what
    they share: how much of the draft he kept (`precision`) and how much of his
    spans the draft had (`recall`). Both frames have `t_start`, `t_end`."""
    s0, s1 = labels.merge_intervals(source.t_start, source.t_end)
    c0, c1 = labels.merge_intervals(his.t_start, his.t_end)
    both0, both1 = methods.intersect(s0, s1, c0, c1)
    total = float((s1 - s0).sum()) if len(s0) else 0.0
    his_ms = float((c1 - c0).sum()) if len(c0) else 0.0
    shared = float((both1 - both0).sum()) if len(both0) else 0.0
    return {
        "source_ms": total,
        "his_present_ms": his_ms,
        "shared_ms": shared,
        "precision": shared / total if total else float("nan"),
        "recall": shared / his_ms if his_ms else float("nan"),
    }


# -------------------------------------------------------------- onset agreement


@dataclass(frozen=True)
class OnsetTruth:
    """One shot's individual rows as onset truth, and where a detection is judged."""

    mid: np.ndarray  # (n,) row centres, ms
    accepted: np.ndarray  # (n,) bool: the row is the accepted prefill
    defined: np.ndarray  # (n_ms,) bool, 1 ms cells from `inputs.GRID0_MS`
    absent: np.ndarray  # (n_ms,) bool, the defined cells inside absent rows


def label_masks(spans: pd.DataFrame, n_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """The 1 ms cells where a detected onset is judged, and the absent ones among them.

    Judged is the shot's labelled window less its uncertain and not-observable rows: an
    onset there is right when a row is near it and false otherwise.
    """
    cells = inputs.GRID0_MS + np.arange(n_ms) + 0.5
    defined = (cells >= spans.t_start.min()) & (cells < spans.t_end.max())
    absent = np.zeros(n_ms, bool)
    for r in spans.itertuples():
        a = int(np.searchsorted(cells, r.t_start, side="left"))
        b = int(np.searchsorted(cells, r.t_end, side="left"))
        if r.kind in ("uncertain", "not_observable"):
            defined[a:b] = False
        elif r.kind == "absent":
            absent[a:b] = True
    return defined, absent & defined


def match_onsets(found_ms, truth_ms, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Which truth onsets, and which detections, `labeler.elm.onset.match` pairs.

    The same greedy rule (truth in time order, each takes the nearest unused detection
    within `tol`), with the flags in the input order.
    """
    found = np.asarray(found_ms, dtype=float)
    truth = np.asarray(truth_ms, dtype=float)
    order = np.argsort(found, kind="stable")
    ordered = found[order]
    used = np.zeros(found.size, dtype=bool)
    hit = np.zeros(truth.size, dtype=bool)
    for i in np.argsort(truth, kind="stable"):
        d = np.abs(ordered - truth[i])
        d[used] = np.inf
        if d.size and d.min() <= tol:
            used[int(np.argmin(d))] = True
            hit[i] = True
    matched = np.zeros(found.size, dtype=bool)
    matched[order] = used
    return hit, matched


def to_grid(found_ms) -> np.ndarray:
    """Detections as the centres of the distinct 10 ms cells they fall in.

    His individual rows hold one row a 10 ms cell, so a method that fires several times
    inside one cell has only its first agreement to find: the extra detections are
    below the labels' resolution, and counted one to one they would all be false.
    """
    cells = np.unique(np.floor(np.asarray(found_ms, dtype=float) / GRID_MS))
    return (cells + 0.5) * GRID_MS


def onset_counts(found_ms, truth: OnsetTruth, cover: pd.DataFrame, tol: float) -> np.ndarray:
    """`ONSET_COLUMNS` of one shot's detections against its individual rows, inside the
    analysed time `cover` (`t_start_ms`, `t_end_ms`, merged)."""
    cov0, cov1 = (cover[c].to_numpy(float) for c in ("t_start_ms", "t_end_ms"))
    keep = methods.inside(truth.mid, cov0, cov1)
    mid, accepted = truth.mid[keep], truth.accepted[keep]
    found = np.asarray(found_ms, dtype=float)
    found = found[methods.inside(found, cov0, cov1)]
    hit, matched = match_onsets(found, mid, tol)
    cell = np.floor(found - inputs.GRID0_MS).astype(np.int64)
    ok = (cell >= 0) & (cell < len(truth.defined))
    judged = np.zeros(found.size, dtype=bool)
    in_absent = np.zeros(found.size, dtype=bool)
    judged[ok] = truth.defined[cell[ok]]
    in_absent[ok] = truth.absent[cell[ok]]
    false = ~matched & judged
    return np.array(
        [
            hit.sum(),
            false.sum(),
            (~hit).sum(),
            (false & in_absent).sum(),
            accepted.sum(),
            (hit & accepted).sum(),
            (~accepted).sum(),
            (hit & ~accepted).sum(),
        ],
        dtype=np.int64,
    )


def ratio_summary(numerator: np.ndarray, denominator: np.ndarray, boot: np.ndarray) -> dict:
    """`sum(numerator) / sum(denominator)` over shots, with a shot-bootstrap interval."""
    num, den = np.asarray(numerator), np.asarray(denominator)
    sampled_n, sampled_d = num[boot].sum(axis=1), den[boot].sum(axis=1)
    values = np.full(len(boot), np.nan)
    np.divide(sampled_n, sampled_d, out=values, where=sampled_d > 0)
    audit = score.bootstrap_summary(
        {"value": values}, n_shots=len(num), positive_shots=int((den > 0).sum())
    )
    return {
        "point": float(num.sum() / den.sum()) if den.sum() else float("nan"),
        "numerator": int(num.sum()),
        "denominator": int(den.sum()),
        "ci95": audit["ci95"]["value"],
        "interval_eligible": audit["interval_eligible"]["value"],
    }


def onset_summary(per_shot: np.ndarray, boot: np.ndarray) -> dict:
    """Pooled onset precision, recall and F1 of `(shots, len(ONSET_COLUMNS))` rows, with
    the recall of the accepted-prefill rows and of the edited rows apart."""
    out = methods.onset_summary(per_shot[:, :3], boot)
    col = {name: per_shot[:, i] for i, name in enumerate(ONSET_COLUMNS)}
    out["false_alarms_in_absent"] = int(col["fp_in_absent"].sum())
    out["recall_prefill_accepted"] = ratio_summary(
        col["prefill_accepted_found"], col["prefill_accepted"], boot
    )
    out["recall_edited"] = ratio_summary(col["edited_found"], col["edited"], boot)
    return out


def onset_set(
    shots: list[int],
    covers: dict[int, pd.DataFrame],
    truth: dict[int, OnsetTruth],
    found: dict[str, dict[int, np.ndarray]],
    tolerances=ONSET_TOLERANCES_MS,
) -> dict:
    """Each method's onset agreement with the individual rows on `shots`, by tolerance.

    A method is scored on the shots it has an output for (`shots_scored`); a detection
    in time no row judges (uncertain, outside the window or analysed time) counts for
    nothing.
    """
    out = {}
    for name, by_shot in found.items():
        scored = [s for s in shots if s in by_shot]
        if not scored:
            continue
        boot = score.draws(len(scored))
        block = {"shots_scored": scored, "tolerances_ms": {}}
        for tol in tolerances:
            rows = np.stack(
                [onset_counts(by_shot[s], truth[s], covers[s], tol) for s in scored]
            )
            block["tolerances_ms"][str(tol)] = onset_summary(rows, boot)
        out[name] = block
    return out


# ------------------------------------------------------------------- occupancy


@dataclass(frozen=True)
class Panel:
    """One shot's review rows with its analysed time and scored bins."""

    shot: int
    spans: pd.DataFrame
    cover: pd.DataFrame  # `methods.cover_frame` of the merged analysed intervals
    bins: labels.Bins


@dataclass(frozen=True)
class Called:
    """A method's output on one shot: a 1 ms trace at a threshold, or detected spans."""

    trace: np.ndarray | None = None
    threshold: float | None = None
    spans: pd.DataFrame | None = None


def panel_of(shot: int, spans: pd.DataFrame, valid: np.ndarray) -> Panel:
    """A panel from the shot's `inputs.VALID` row: bins on the analysed intervals the
    way `train.load` makes them and a cover frame the way `compare.own_coverage` does."""
    c0, c1 = inputs.valid_intervals(valid)
    cov0, cov1 = labels.merge_intervals(c0, c1)
    cover = methods.cover_frame(
        *labels.merge_intervals(c0, c1, tol=compare.COVER_BOUNDARY_TOL_MS)
    )
    return Panel(shot, spans, cover, labels.scored_bins(spans, cov0, cov1))


def bes_panel(panel: Panel, chunks_ms) -> Panel:
    """The panel on ELM-O's analysed chunks `[(start, stop), ...]` within its own: the
    benchmark's `bes73` bins (`compare.load_sets`)."""
    chunks = np.asarray(chunks_ms, dtype=float).reshape(-1, 2)
    order = np.argsort(chunks[:, 0])
    raw0, raw1 = labels.merge_intervals(
        chunks[order, 0], chunks[order, 1], tol=compare.COVER_BOUNDARY_TOL_MS
    )
    cover = methods.cover_frame(
        *methods.intersect(
            raw0, raw1, panel.cover.t_start_ms, panel.cover.t_end_ms
        )
    )
    bins = labels.scored_bins(
        panel.spans, cover.t_start_ms.to_numpy(float), cover.t_end_ms.to_numpy(float)
    )
    return Panel(panel.shot, panel.spans, cover, bins)


def part_of(panel: Panel, called: Called) -> score.ShotScore:
    if called.trace is not None:
        return methods.trace_part(
            panel.spans,
            panel.shot,
            panel.bins,
            panel.cover,
            np.nan_to_num(called.trace, nan=0.0, posinf=0.0, neginf=0.0),
            called.threshold,
        )
    spans = called.spans if called.spans is not None else methods.span_frame([], [])
    return methods.span_part(panel.spans, panel.shot, panel.bins, panel.cover, spans)


def always_present(panels: dict[int, Panel]) -> list[score.ShotScore]:
    """The rule that calls every bin present, on the panels' bins."""
    return [
        score.ShotScore(
            s,
            p.bins.truth,
            p.bins.kind,
            np.ones(len(p.bins.truth), bool),
            np.zeros(len(p.bins.truth)),
            {},
        )
        for s, p in sorted(panels.items())
    ]


def occupancy_set(panels: dict[int, Panel], called: dict[str, dict[int, Called]]) -> dict:
    """The bin metrics of every method on the panels it has an output for.

    `crowd_bin_recall` is the share of crowd bins called present and
    `false_alarm_bin_rate` the share of absent bins called present; F1, AUROC and AUPRC
    pool the crowd (positive) and absent (negative) bins. Intervals resample shots, as
    `score.summarise` does, with the draws for a given number of shots shared.
    """
    shots = sorted(panels)
    kinds = pd.Series(np.concatenate([panels[s].bins.kind for s in shots]), dtype=object)
    out = {
        "n_shots": len(shots),
        "shots": shots,
        "bins": len(kinds),
        "bins_by_kind": {k: int(v) for k, v in kinds.value_counts().items()},
        "methods": {},
    }
    for name, by_shot in called.items():
        scored = [s for s in shots if s in by_shot]
        if not scored:
            continue
        parts = [part_of(panels[s], by_shot[s]) for s in scored]
        block = score.summarise(parts, score.draws(len(parts)))
        block["shots_scored"] = scored
        out["methods"][name] = block
    rule = always_present(panels)
    block = score.summarise(rule, score.draws(len(rule)))
    block["shots_scored"] = shots
    out["methods"]["always present"] = block
    return out


# -------------------------------------------------------------------- run days


def run_days(paths: Paths, shots) -> dict[int, str]:
    """The calendar run day of each shot: the operator logs' `run` date, else the
    cohort's `run_id`; a shot with neither is absent from the result.

    `logs.jsonl` holds the operator text; only the shot and the run date are read.
    """
    wanted = {int(s) for s in shots}
    found: dict[int, str] = {}
    pattern = re.compile(r'^\{"shot": (\d+), "run": "(\d{8})[^"]*"')
    with paths.logs_jsonl.open() as handle:
        for line in handle:
            match = pattern.match(line)
            if match and int(match[1]) in wanted:
                found[int(match[1])] = match[2]
    cohort = pd.read_csv(paths.catalog / "cohort.csv")
    for row in cohort.itertuples():
        if int(row.shot) in wanted and pd.notna(row.run_id):
            found.setdefault(int(row.shot), str(int(row.run_id)))
    return found


def finite(value):
    """`value` with every NaN as null and numpy scalars and arrays as plain Python."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value
