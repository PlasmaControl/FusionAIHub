"""The ELM reference swap: fixed predictions under onset and occupancy references.

The review's dense ELM spans are the reference this package scores against. The lab's
older ELM annotation is the compiled legacy onset table (`edge_localized_mode/format`).
The compiler counts original 1 ms onset labels in 50 ms bins, assigns category 1
to nonzero counts, deduplicates matching WPQH subsets, and compresses consecutive
equal bins into half-open intervals (data/events/README.md). It covers 8 of 119 reviewed
shots. The coverage audit uses per-bin majority occupancy over all legacy-covered
bins in the review window. Detector comparisons use the stricter benchmark bins;
that restriction is a deviation from the AE audit.

**Conversion.** A scored bin is the cell `[50k, 50k + 50)` of the shot's clock; the
table's bins are the same cells, so a bin takes the category of the table row that
contains its midpoint. A bin the table does not cover has no legacy value and is
dropped from the comparison. The legacy reference marks a bin present exactly when an
onset falls in it; the review marks interval occupancy. Interior-bin scoring
usually excludes span starts, so agreement does not measure missed physical ELMs.
Occupancy sensitivities merge positive intervals separated by covered gaps of at
most 100, 200 or 300 ms. Merging never crosses missing legacy coverage, and never
extends beyond the first or last positive interval. No tolerance is selected from
the scores. The onset reference remains available alongside every sensitivity.
`start_agreement` separately asks whether a legacy onset is near each span start;
these starts are not independently verified millisecond ELM onset labels.

**Finding 1** (`agreement_counts`): the legacy reference read as a detector of the
review's present bins; `|M|` is the number of review-present bins it marks absent,
`|P|` the number of review-absent bins it marks present.

**Finding 2** (`retruth`, `rankings`): every method's calls and scores, unchanged,
scored against the legacy truth over the bins both references cover.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import labels, score

LEGACY_TABLE = (
    Path("edge_localized_mode") / "format" / "edge_localized_mode_format_2026_v1.csv"
)
BIN_MS = labels.BIN_MS
OCCUPANCY_GAPS_MS = (100, 200, 300)


def legacy_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def positive_intervals(table, shot: int, gap_ms: float | None = None) -> pd.DataFrame:
    """Legacy positive intervals, merging short gaps only inside continuous coverage."""
    rows = table[table.shot == shot]
    positive = rows[rows.category == 1]
    if gap_ms is None:
        return positive
    if gap_ms < 0:
        raise ValueError("occupancy gap must be nonnegative")
    cov0, cov1 = labels.merge_intervals(rows.t_start, rows.t_end, tol=0)
    starts, stops = [], []
    for a, b in zip(cov0, cov1, strict=True):
        inside = positive[(positive.t_start < b) & (positive.t_end > a)]
        s, e = labels.merge_intervals(
            np.maximum(inside.t_start, a), np.minimum(inside.t_end, b), tol=gap_ms
        )
        starts.extend(s)
        stops.extend(e)
    return pd.DataFrame({"t_start": starts, "t_end": stops})


def table_truth(
    table: pd.DataFrame, shot: int, bins: labels.Bins, gap_ms: float | None = None
) -> np.ndarray:
    """Legacy onset/merged-occupancy truth: 0, 1, or -1 where not covered."""
    out = np.full(len(bins.t0), -1, dtype=np.int8)
    mid = bins.t0 + 0.5 * BIN_MS
    for r in table[table.shot == shot].itertuples():
        inside = (mid >= r.t_start) & (mid < r.t_end)
        out[inside] = int(r.category)
    if gap_ms is not None:
        for r in positive_intervals(table, shot, gap_ms).itertuples():
            inside = (out >= 0) & (mid >= r.t_start) & (mid < r.t_end)
            out[inside] = 1
    return out


def table_alignment(table: pd.DataFrame, shots) -> dict:
    """How well the table's rows sit on the 50 ms grid: rows with an off-grid edge."""
    t = table[table.shot.isin(shots)]
    edges = np.concatenate([t.t_start.to_numpy(float), t.t_end.to_numpy(float)])
    off = ~np.isclose(edges % BIN_MS, 0.0) & ~np.isclose(edges % BIN_MS, BIN_MS)
    return {"rows": len(t), "off_grid_edges": int(off.sum())}


def _occupancy(t0, spans) -> np.ndarray:
    """Milliseconds of interval union in each bin (adjacent rows may share it)."""
    a, b = labels.merge_intervals(spans.t_start, spans.t_end, tol=0)
    return np.clip(
        np.minimum(t0[:, None] + BIN_MS, b) - np.maximum(t0[:, None], a),
        0,
        None,
    ).sum(axis=1)


def coverage_bins(
    table, shot: int, spans, gap_ms: float | None = None
) -> tuple[labels.Bins, np.ndarray, np.ndarray]:
    """All legacy-covered grid cells intersecting the review window.

    At least half a bin must be legacy-covered. Legacy presence and each review
    state require >=25 ms occupancy, like the AE audit's >=0.5 rule. Ties prefer
    present then absent; uncertain, not observable and mixed/unlabelled cells
    stay outside M/P, rather than becoming review-absent. No diagnostic/DSM or
    wholly-inside-one-span restriction is imposed here.
    """
    first = int(np.floor(spans.t_start.min() / BIN_MS))
    last = int(np.ceil(spans.t_end.max() / BIN_MS))
    t0 = np.arange(first, last, dtype=float) * BIN_MS
    legacy_spans = table[table.shot == shot]
    keep = _occupancy(t0, legacy_spans) >= BIN_MS / 2
    t0 = t0[keep]
    legacy = _occupancy(t0, positive_intervals(table, shot, gap_ms)) >= BIN_MS / 2
    status = np.full(len(t0), "mixed_or_unlabelled", dtype=object)
    truth = np.full(len(t0), -1, dtype=np.int8)
    kind = np.full(len(t0), "unknown", dtype=object)
    for state, kinds, value in (
        ("not_observable", ["not_observable"], -1),
        ("uncertain", ["uncertain"], -1),
        ("absent", ["absent"], 0),
        ("present", ["non_crowd", "crowd"], 1),
    ):
        hit = _occupancy(t0, spans[spans.kind.isin(kinds)]) >= BIN_MS / 2
        status[hit], truth[hit] = state, value
        kind[hit] = kinds[0] if state != "present" else "present"
    # Kind counts describe occupancy only; publication uses event-start agreement.
    for name in ("non_crowd", "crowd"):
        hit = (truth == 1) & (_occupancy(t0, spans[spans.kind == name]) >= BIN_MS / 2)
        kind[hit] = name
    return (
        labels.Bins(t0, truth, kind, np.full(len(t0), -1)),
        legacy.astype(np.int8),
        status,
    )


def start_agreement(spans, onsets_ms, tolerance_ms: float) -> dict:
    """Per-span-start agreement, allowing a legacy onset within inclusive +/-k ms.

    This is a proximity query, not one-to-one ELM counting; a long non-crowd span
    is one review annotation, and a crowd start is not every ELM in that crowd.
    """
    onsets = np.sort(np.asarray(onsets_ms, dtype=float))
    out = {}
    for kind in ("non_crowd", "crowd"):
        starts = spans[spans.kind == kind].t_start.to_numpy(float)
        distances = (
            np.min(np.abs(starts[:, None] - onsets), axis=1)
            if len(onsets)
            else np.full(len(starts), np.inf)
        )
        hit = distances <= tolerance_ms
        out[kind] = {
            "spans": len(starts),
            "matched": int(hit.sum()),
            "share": float(hit.mean()) if len(hit) else float("nan"),
            "starts_ms": starts.tolist(),
            "matched_starts_ms": starts[hit].tolist(),
            "nearest_onset_distance_ms": [
                float(d) if np.isfinite(d) else None for d in distances
            ],
        }
    return out


def sweep_bin_scores(sweep: pd.DataFrame, bins: labels.Bins) -> np.ndarray:
    """Largest eta whose ELM-O detections touch a bin; unhit bins tie at -1.

    Ranking these scores exactly reproduces the finite nested eta sweep with
    the closing all-positive point. It can be restricted to any identical bin
    set and evaluated under either reference without changing detections.
    """
    out = np.full(len(bins.t0), -1.0)
    for eta, spans in sweep.groupby("eta"):
        spans = spans.sort_values("t_start_ms")
        hit = labels.hard_hits(
            spans.t_start_ms.to_numpy(float), spans.t_end_ms.to_numpy(float), bins
        )
        out[hit] = np.maximum(out[hit], float(eta))
    return out


def onset_truth(onsets_ms, bins: labels.Bins) -> np.ndarray:
    """1 for a bin holding at least one onset, else 0 (the legacy convention)."""
    on = np.sort(np.asarray(onsets_ms, dtype=float))
    lo = np.searchsorted(on, bins.t0, side="left")
    hi = np.searchsorted(on, bins.t0 + BIN_MS, side="left")
    return (hi > lo).astype(np.int8)


def retruth(part: score.ShotScore, truth: np.ndarray) -> score.ShotScore:
    """`part` scored against another reference: bins with `truth` -1 dropped."""
    keep = truth >= 0
    t = truth[keep].astype(np.int8)
    return replace(
        part,
        truth=t,
        kind=np.where(t == 1, "present", "absent").astype(object),
        call=part.call[keep],
        score=None if part.score is None else part.score[keep],
        spans={},
    )


def as_method(
    shot: int, call: np.ndarray, truth: np.ndarray, kind: np.ndarray
) -> score.ShotScore:
    """A reference read as a detector: its present bins are its calls."""
    return score.ShotScore(shot, truth, kind, call.astype(bool), None, {})


def agreement_counts(
    review_truth: np.ndarray, kind: np.ndarray, legacy: np.ndarray
) -> np.ndarray:
    """`[tp, fp, fn, tn, crowd_bins, crowd_legacy, non_crowd_bins, non_crowd_legacy]`
    of the legacy marks against the review, over the bins the legacy covers."""
    keep = legacy >= 0
    t, c = review_truth[keep].astype(bool), legacy[keep].astype(bool)
    k = kind[keep]
    crowd, ind = k == "crowd", k == "non_crowd"
    return np.array(
        [
            (t & c).sum(),
            (~t & c).sum(),
            (t & ~c).sum(),
            (~t & ~c).sum(),
            crowd.sum(),
            (crowd & c).sum(),
            ind.sum(),
            (ind & c).sum(),
        ],
        dtype=float,
    )


AGREEMENT_NAMES = (
    "tp",
    "fp",
    "fn",
    "tn",
    "crowd_bins",
    "crowd_legacy_present",
    "non_crowd_bins",
    "non_crowd_legacy_present",
)


def agreement_rates(total: np.ndarray) -> dict[str, float]:
    c = dict(zip(AGREEMENT_NAMES, total, strict=True))

    def ratio(a, b):
        return float(a / b) if b > 0 else float("nan")

    return {
        "precision": ratio(c["tp"], c["tp"] + c["fp"]),
        "recall": ratio(c["tp"], c["tp"] + c["fn"]),
        "f1": ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]),
        "missed_share_of_present": ratio(c["fn"], c["tp"] + c["fn"]),
        "crowd_bin_recall": ratio(c["crowd_legacy_present"], c["crowd_bins"]),
        "non_crowd_bin_recall": ratio(
            c["non_crowd_legacy_present"], c["non_crowd_bins"]
        ),
        "review_prevalence": ratio(
            c["tp"] + c["fn"], c["tp"] + c["fn"] + c["fp"] + c["tn"]
        ),
        "legacy_prevalence": ratio(
            c["tp"] + c["fp"], c["tp"] + c["fn"] + c["fp"] + c["tn"]
        ),
    }


def agreement_summary(per_shot: np.ndarray, boot: np.ndarray) -> dict:
    """Finding 1: pooled counts, `|M|`, `|P|` and rates with bootstrap intervals."""
    total = per_shot.sum(axis=0)
    point = agreement_rates(total)
    reps = [agreement_rates(per_shot[d].sum(axis=0)) for d in boot]
    ci = {k: score._ci(np.array([r[k] for r in reps])) for k in point}
    counts = dict(zip(AGREEMENT_NAMES, (int(v) for v in total), strict=True))
    return {
        "counts": counts,
        "M": counts["fn"],
        "P": counts["fp"],
        "bins": int(counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]),
        "point": point,
        "ci95": ci,
    }


def rankings(points: dict[str, dict[str, float]], metrics=("auroc", "f1")) -> dict:
    """Method names ordered best first by each metric (methods without it skipped)."""
    out = {}
    for m in metrics:
        have = {n: p[m] for n, p in points.items() if m in p and np.isfinite(p[m])}
        out[m] = sorted(have, key=lambda n: -have[n])
    return out
