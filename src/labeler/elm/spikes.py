"""`elm-inst2`: the ELMs of a train as separate spikes, from reviewed onsets.

`elm-inst` (`inst.py`) finds 22 of the challenge key's 194 ELMs that have another
within 10 ms (`docs/labeler/elm_challenge.md`). Inside crowds its only instance
targets were the clock's onsets at half weight, and its focal loss rewards a high
heatmap at an onset cell without asking it to fall again before the next ELM, so a
train comes out as a low plateau. Two changes here:

* **Onsets inside trains.** Ricardo Shousha's review of 2026-10-23 marks each ELM as a
  10 ms cell (a `non_crowd` row); 17,109 of his 17,128 cells are the review page's
  `elm_jump` v2 onsets, prefilled and accepted (`docs/labeler/elm_heldout.md`). The
  onsets inside his cells come from the `elm_jump` table (`accepted_onsets`). Half the
  gaps between those onsets are under 3 ms, a second jump on one ELM's rise (2.8 % of the
  clock's crowd intervals are), so an onset less than `MERGE_MS` after the last kept one
  is dropped (`merge`), and peaks are decoded `MIN_SEP_MS` apart.
* **A spike loss** (`spike_loss`), beside the focal loss: at each anchor onset the
  heatmap's maximum within `PEAK_HALF_MS` should be 1 and its minimum on either side,
  `DIP_NEAR_MS` to `DIP_FAR_MS` away, should be 0. The maximum makes it a
  multiple-instance term (a peak a few cells off still counts); the two minima ask for
  a dip between neighbours, so a train is a row of peaks rather than a plateau.

The served threshold is the one that does best on development data by the challenge
merit function's published costs (`merit_gain`), fixed before the key is scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import inst

DT_MS = inst.DT_MS
MERGE_MS = 3.0
MIN_SEP_MS = MERGE_MS
PEAK_HALF_MS = 0.5
DIP_NEAR_MS = 0.8
DIP_FAR_MS = 2.5
SPIKE_WEIGHT = 1.0
#: Rows of the merit function's `miscounting_penalties` (commit b71f991) for the types a
#: detection may be reported as, against a key with no event, a large ELM (type 1) and a
#: small ELM (type 2); and the row of no detection. The merit function counts an event
#: under three readings of the key's edges, weighted 1, 0.1 and 0.05, and scales by
#: 10 / 1.15, so a whole event costs 10x its entry. Each detection also changes the
#: flag's mismatched time by its own length.
PENALTY = {1: (1.00, 0.00, 0.30), 2: (0.30, 0.50, 0.00), 5: (0.95, 0.70, 0.35)}
MISSED = (0.00, 1.20, 0.40)
UNIT_MS = 10.0  # the review's grid: one individual row is one 10 ms cell


def prices(reported: int) -> tuple[float, float, float]:
    """What one detection reported as type `reported` saves on a large ELM and on a
    small one, against no detection, and what it costs where the key has no event."""
    none, large, small = PENALTY[reported]
    return 10 * (MISSED[1] - large), 10 * (MISSED[2] - small), 10 * none


#: The detection-only type 5 (`elm-inst`, `elm-inst2`): a found large ELM saves 5.0, a
#: found small one 0.5, and a false detection costs 9.5. The ELMs of a crowd are small
#: (they rise 0.11 to 0.34 decades against 0.94 for Smith's isolated ELMs,
#: `docs/labeler/elm_train_methods.md`), so a found one is priced at `GAIN_TP_SMALL`.
GAIN_TP, GAIN_TP_SMALL, COST_FP = prices(5)


def merge(onsets_ms, merge_ms: float = MERGE_MS) -> np.ndarray:
    """Sorted onsets, each dropped when less than `merge_ms` after the last kept one."""
    keep: list[float] = []
    for t in np.sort(np.asarray(onsets_ms, dtype=float)):
        if not keep or t - keep[-1] >= merge_ms:
            keep.append(float(t))
    return np.array(keep, dtype=float)


def accepted_onsets(spans: pd.DataFrame, jump_ms) -> tuple[np.ndarray, dict]:
    """One shot's onsets: the `elm_jump` onsets `jump_ms` inside its individual rows
    (`kind == "non_crowd"`), merged; and counts of rows, onsets and rows with none."""
    rows = spans[spans.kind == "non_crowd"]
    t = np.sort(np.asarray(jump_ms, dtype=float))
    inside = np.zeros(len(t), dtype=bool)
    empty = 0
    for a, b in zip(rows.t_start.to_numpy(float), rows.t_end.to_numpy(float)):
        hit = (t >= a) & (t < b)
        inside |= hit
        empty += not hit.any()
    kept = merge(t[inside])
    return kept, {
        "rows": len(rows),
        "rows_without_onset": int(empty),
        "onsets_in_rows": int(inside.sum()),
        "kept": len(kept),
    }


def review_onset_target(spans: pd.DataFrame, onsets_ms, n_cells: int) -> inst.Target:
    """Instance targets of a review whose ELMs are marked: guarded absent time known
    empty, as `inst.review_target`; each onset a full-weight peak known within five
    sigma, as Smith's; crowd, individual and uncertain time otherwise ignored."""
    heat = np.zeros(n_cells, np.float32)
    known = np.zeros(n_cells, bool)
    weight = np.zeros(n_cells, np.float32)
    for r in spans.itertuples():
        if r.kind != "absent":
            continue
        a, b = r.t_start + inst.ABSENT_GUARD_MS, r.t_end - inst.ABSENT_GUARD_MS
        if b > a:
            lo, hi = max(0, int(inst.cell_of(a))), min(n_cells, int(inst.cell_of(b)))
            known[lo:hi] = True
            weight[lo:hi] = 1.0
    for t in np.asarray(onsets_ms, dtype=float):
        inst._bump(heat, known, weight, t, 5 * inst.SIGMA_MS, 1.0, n_cells)
    return inst.Target(heat, known, weight)


def anchors(onsets_ms, n_cells: int) -> np.ndarray:
    """Boolean `(n_cells,)`: the cells of the onsets the spike loss is asked at."""
    out = np.zeros(n_cells, dtype=bool)
    c = inst.cell_of(onsets_ms)
    out[c[(c >= 0) & (c < n_cells)]] = True
    return out


def spike_loss(logits: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Mean over the anchor cells of `-log(peak) - log(1 - dip)`: `peak` the heatmap's
    maximum within `PEAK_HALF_MS` of the anchor, `dip` the larger of its minima on the
    two sides, `DIP_NEAR_MS` to `DIP_FAR_MS` away. A side beyond the crop counts as a
    dip. `logits` and `anchor` are `(B, cells)`."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    if not bool(anchor.any()):
        return p.sum() * 0.0
    h = round(PEAK_HALF_MS / DT_MS)
    peak = F.max_pool1d(p[:, None], 2 * h + 1, stride=1, padding=h)[:, 0]
    near, far = round(DIP_NEAR_MS / DT_MS), round(DIP_FAR_MS / DT_MS)
    # low[:, j] is the minimum of the zero-padded heatmap over [j, j + far - near], so a
    # cell c's left side [c - far, c - near] is low[c] and its right side
    # [c + near, c + far] is low[c + near + far]
    low = -F.max_pool1d(-F.pad(p, (far, far))[:, None], far - near + 1, stride=1)[:, 0]
    c = torch.arange(p.shape[1], device=p.device)
    dip = torch.maximum(low[:, c], low[:, c + near + far])
    terms = -torch.log(peak[anchor]) - torch.log(1 - dip[anchor])
    return terms.mean()


def merit_gain(
    tp: int, fp: int, duration_ms: float, tp_small: int = 0, reported: int = 5
) -> float:
    """What detections reported as type `reported` save against none on the merit
    function's published costs (`prices`): each found large ELM saves its price plus
    its flag's length, each found small one (`tp_small`, not counted in `tp`) its price
    plus it, and each false detection costs its price plus its flag's length."""
    large, small, false = prices(reported)
    return (
        (large + duration_ms) * tp
        + (small + duration_ms) * tp_small
        - (false + duration_ms) * fp
    )


def report_type(p_none, p_large, p_small, types=(2, 1)) -> np.ndarray:
    """Per detection, the type of `types` with the lowest expected cost under `PENALTY`,
    given the probabilities that it is no event, a large ELM or a small one (the
    minimum-expected-cost rule; a tie goes to the first of `types`)."""
    p = np.column_stack([p_none, p_large, p_small]).astype(float)
    cost = np.stack([p @ np.asarray(PENALTY[t]) for t in types], axis=1)
    return np.asarray(types)[np.argmin(cost, axis=1)]


def in_crowd(onsets_ms, spans: pd.DataFrame) -> np.ndarray:
    """Boolean per onset: inside one of the shot's crowd rows."""
    t = np.asarray(onsets_ms, dtype=float)
    out = np.zeros(len(t), dtype=bool)
    for r in spans[spans.kind == "crowd"].itertuples():
        out |= (t >= r.t_start) & (t < r.t_end)
    return out


def match_pairs(found_ms, truth_ms, tol_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """`(found matched, truth matched)` boolean masks, one to one, nearest first: the
    pairing of `inst.match`, which returns counts only."""
    found = np.asarray(found_ms, dtype=float)
    truth = np.asarray(truth_ms, dtype=float)
    used_f = np.zeros(len(found), bool)
    used_t = np.zeros(len(truth), bool)
    if not len(found) or not len(truth):
        return used_f, used_t
    d = np.abs(found[:, None] - truth[None, :])
    pairs = np.argwhere(d <= tol_ms)
    order = np.argsort(d[pairs[:, 0], pairs[:, 1]], kind="stable")
    for i, j in pairs[order]:
        if not used_f[i] and not used_t[j]:
            used_f[i] = used_t[j] = True
    return used_f, used_t


def judge(found_ms, onsets_ms, spans: pd.DataFrame, tol_ms: float = 2.0) -> dict:
    """Detections against one shot's kept onsets from a review of individual rows.

    `tp`: onsets matched within `tol_ms`; `fn`: onsets unmatched. Of the unmatched
    detections, `fp` are judged false: in guarded absent time, or within `MERGE_MS`
    of a kept onset (a second detection on one ELM). The rest are unjudged: `empty`
    in a present 10 ms cell with no individual row (the reviewer kept no onset there),
    `other` elsewhere (inside an individual row, or in unlabelled time). `found` is
    the detections sorted and `kind` each one's `tp`, `fp`, `empty` or `other`.
    """
    found = np.sort(np.asarray(found_ms, dtype=float))
    onsets = np.asarray(onsets_ms, dtype=float)
    used_f, used_t = match_pairs(found, onsets, tol_ms)
    free = found[~used_f]
    g = inst.ABSENT_GUARD_MS
    absent = np.zeros(len(free), bool)
    present = np.zeros(len(free), bool)
    for r in spans.itertuples():
        inside = (free >= r.t_start) & (free < r.t_end)
        if r.kind == "absent":
            absent |= (free >= r.t_start + g) & (free < r.t_end - g)
        elif r.kind in ("crowd", "non_crowd"):
            present |= inside
    near = np.zeros(len(free), bool)
    if len(onsets) and len(free):
        near = np.abs(free[:, None] - onsets[None, :]).min(axis=1) < MERGE_MS
    rows = spans[spans.kind == "non_crowd"]
    cells = set(np.floor(rows.t_start.to_numpy(float) / UNIT_MS).astype(int).tolist())
    in_row = np.array(
        [int(np.floor(t / UNIT_MS)) in cells for t in free], dtype=bool
    ).reshape(-1)
    false = absent | near
    empty = ~false & present & ~in_row
    kind = np.full(len(found), "tp", dtype=object)
    kind[~used_f] = np.where(false, "fp", np.where(empty, "empty", "other"))
    return {
        "tp": int(used_t.sum()),
        "fn": int((~used_t).sum()),
        "fp": int(false.sum()),
        "fp_absent": int(absent.sum()),
        "fp_near": int((near & ~absent).sum()),
        "empty": int(empty.sum()),
        "other": int((~false & ~empty).sum()),
        "matched_onsets": used_t,
        "found": found,
        "kind": kind,
    }
