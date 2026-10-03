"""The reference swap for ELMs: the same bins scored against two references.

The review's dense ELM spans are the reference this package scores against. The lab's
older ELM annotation is Hiro's table of 50 ms bins (`edge_localized_mode/format`, one
row per run of bins, `category` 1 where at least one ELM *onset* falls in the bin and 0
where none does). It marks onsets, not ELMy time, and covers 8 of the 119 reviewed
shots. This module converts it to the scored bins, in the manner of the AE audit, and
measures the two references against each other.

**Conversion.** A scored bin is the cell `[50k, 50k + 50)` of the shot's clock; the
table's bins are the same cells, so a bin takes the category of the table row that
contains its midpoint. A bin the table does not cover has no legacy value and is
dropped from the comparison. The legacy reference marks a bin present exactly when an
onset falls in it; the review marks it present when it lies wholly inside an
individual-ELM span or a crowd (an ELMing period). The two agree on a bin holding one
isolated ELM's onset and disagree on the rest of a crowd or an ELM longer than one bin
(`onset_truth` converts any list of onsets the same way).

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


def legacy_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def table_truth(table: pd.DataFrame, shot: int, bins: labels.Bins) -> np.ndarray:
    """The legacy table's category of each scored bin: 0, 1, or -1 where not covered."""
    out = np.full(len(bins.t0), -1, dtype=np.int8)
    mid = bins.t0 + 0.5 * BIN_MS
    for r in table[table.shot == shot].itertuples():
        inside = (mid >= r.t_start) & (mid < r.t_end)
        out[inside] = int(r.category)
    return out


def table_alignment(table: pd.DataFrame, shots) -> dict:
    """How well the table's rows sit on the 50 ms grid: rows with an off-grid edge."""
    t = table[table.shot.isin(shots)]
    edges = np.concatenate([t.t_start.to_numpy(float), t.t_end.to_numpy(float)])
    off = ~np.isclose(edges % BIN_MS, 0.0) & ~np.isclose(edges % BIN_MS, BIN_MS)
    return {"rows": len(t), "off_grid_edges": int(off.sum())}


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
    """`[tp, fp, fn, tn, crowd_bins, crowd_legacy, individual_bins, individual_legacy]`
    of the legacy marks against the review, over the bins the legacy covers."""
    keep = legacy >= 0
    t, c = review_truth[keep].astype(bool), legacy[keep].astype(bool)
    k = kind[keep]
    crowd, ind = k == "crowd", k == "individual"
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
    "individual_bins",
    "individual_legacy_present",
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
        "individual_bin_recall": ratio(
            c["individual_legacy_present"], c["individual_bins"]
        ),
        "review_prevalence": ratio(
            c["tp"] + c["fn"], c["tp"] + c["fn"] + c["fp"] + c["tn"]
        ),
        "legacy_prevalence": ratio(
            c["tp"] + c["fp"], c["tp"] + c["fn"] + c["fp"] + c["tn"]
        ),
    }


def agreement_summary(per_shot: np.ndarray, boot: np.ndarray) -> dict:
    """Finding 1: pooled counts, `|M|`, `|P|` and rates with shot-bootstrap intervals."""
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
