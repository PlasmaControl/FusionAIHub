#!/usr/bin/env python
"""Record exploratory indicator agreement; no independent benchmark is available.

    python scripts/labeler/detach_benchmark.py

Reads `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` (written by
`detach_label.py`: every assessed bin with the votes, the values and both labelers'
states) and writes `docs/labeler/results/detachment_benchmark.json`.

Pairwise cast votes are compared before consensus selection, separately for each
``tangtv_tier``. The paper population is the upper-shelf Prad--TangTV pair. The
f_div scored threshold-free is the per-shot relative value (the exported vote's
value); the absolute value is the `prad_abs` sensitivity pair.
Cohen's kappa and binary attached/not-attached kappa use 1000 shot-bootstrap
replicates. Lower-shelf measurements remain provisional. An indicator scored
against a consensus containing its own vote cannot establish detector accuracy,
so that comparison is intentionally absent from this record.

The historical filename is retained for consumers of the reproducible records.
Shared metric/model helpers below also support the exploratory CNN scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from detach_json import dumps
from scipy.stats import rankdata

from labeler.events.detachment import core, label_model
from labeler.events.detachment.label_model import COMPATIBLE, LF_NAMES

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "labeler" / "results" / "detachment_benchmark.json"
MODEL_RECORD = (
    REPO
    / "data"
    / "events"
    / "detachment"
    / "extend_detach_vote"
    / "records"
    / "label_model.json"
)
REPLICATES = 1000
#: Every shot bootstrap of a statistic starts a fresh generator at this seed, so the
#: same population gives the same interval in every script and whatever else was
#: drawn before it (a shared stream made one statistic print two intervals).
BOOT_SEED = 0
STATES = (1, 2, 3)
#: The sign that makes a larger score mean "not attached".
DIRECTION = {"afrac": -1.0, "prad": 1.0, "prad_abs": 1.0, "tangtv": 1.0}
#: Indicator name -> (value column, validity column, vote column). The exported
#: f_div vote is the per-shot relative one (`prad_rel_value`); `prad_abs` is the
#: absolute-cutoff sensitivity (`prad_value`, `prad_abs_valid`, `prad_abs_vote`).
COLUMNS = {
    "afrac": ("afrac_value", "afrac_valid", "afrac_vote"),
    "prad": ("prad_rel_value", "prad_valid", "prad_vote"),
    "prad_abs": ("prad_value", "prad_abs_valid", "prad_abs_vote"),
    "tangtv": ("tangtv_value", "tangtv_valid", "tangtv_vote"),
}


def boot_rng() -> np.random.Generator:
    return np.random.default_rng(BOOT_SEED)


def mean_boot(values) -> list:
    """95% shot-bootstrap interval of the mean of per-shot values (one per shot)."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    rng = boot_rng()
    draws = [
        float(np.mean(finite[rng.integers(0, len(finite), len(finite))]))
        for _ in range(REPLICATES if len(finite) else 0)
    ]
    return interval(draws)


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def kappa_from(table: np.ndarray) -> float:
    """Cohen's kappa of a confusion table; NaN (undefined) on a degenerate table.

    Degenerate means either rater used a single class: chance agreement is then
    1 (both constant) or the table carries no information about agreement beyond
    one rater's constancy, and 0 or 1 would read as a measurement. NaN becomes
    JSON null and is left out of bootstrap intervals (counted as invalid draws).
    """
    n = table.sum()
    if n == 0:
        return float("nan")
    if (table.sum(axis=1) > 0).sum() < 2 or (table.sum(axis=0) > 0).sum() < 2:
        return float("nan")
    observed = np.trace(table) / n
    expected = float(table.sum(axis=1) @ table.sum(axis=0)) / n**2
    return float((observed - expected) / (1 - expected))


def f1_from(table: np.ndarray, k: int) -> float:
    tp = table[k, k]
    pred, true = table[:, k].sum(), table[k, :].sum()
    if true == 0:
        return float("nan")
    return float(2 * tp / (pred + true))


def metrics(
    table: np.ndarray,
    name: str,
    n_valid: float,
    n_ref: float,
    reference_classes: tuple[int, ...] = STATES,
) -> dict:
    """All scalar metrics from a (ref, vote) confusion table of 1..3 states."""
    total = table.sum()
    compat = sum(
        table[r - 1, v - 1]
        for v in STATES
        for r in COMPATIBLE[name].get(v, ())
        if r in STATES
    )
    merged = np.array(
        [
            [table[0, 0], table[0, 1:].sum()],
            [table[1:, 0].sum(), table[1:, 1:].sum()],
        ]
    )
    # A bootstrap draw cannot redefine the estimand by dropping its absent class.
    supported = all(table[k - 1].sum() > 0 for k in reference_classes)
    return {
        "n_bins": float(total),
        "vote_rate": float(total / n_ref) if n_ref else float("nan"),
        "valid_rate": float(n_valid),
        "agreement": float(np.trace(table) / total) if total else float("nan"),
        "compatible_agreement": float(compat / total) if total else float("nan"),
        "kappa": kappa_from(table),
        "f1_attached": f1_from(table, 0),
        "f1_detached": f1_from(table, 1),
        "f1_marfe": f1_from(table, 2),
        "macro_f1": float(np.mean([f1_from(table, k - 1) for k in reference_classes]))
        if reference_classes and supported
        else float("nan"),
        "binary_f1_attached": f1_from(merged, 0),
        "binary_f1_not_attached": f1_from(merged, 1),
        "binary_kappa": kappa_from(merged),
        "binary_agreement": float(np.trace(merged) / total) if total else float("nan"),
    }


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties half (Mann-Whitney)."""
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_model() -> label_model.LabelModel:
    """The label model `detach_label.py` fitted (parameters from its record)."""
    record = json.loads(MODEL_RECORD.read_text())["model"]
    model = label_model.LabelModel(
        names=tuple(record["names"]), corr=tuple(tuple(p) for p in record["corr"])
    )
    model.theta = np.asarray(record["theta"], dtype=float)
    return model


def loo_state(frame: pd.DataFrame, drop: str, model, threshold: float):
    """Reference on bins where the other two indicators agree.

    The scored vote is withheld. Compatibility and posterior gates select a
    subset; this is diagnostic agreement, not an independent accuracy reference.
    """
    keep = [n for n in LF_NAMES if n != drop]
    votes = np.stack([frame[f"{n}_vote"].to_numpy() for n in LF_NAMES], axis=1)
    votes[:, LF_NAMES.index(drop)] = core.ABSTAIN
    both = np.stack([frame[f"{n}_valid"].to_numpy() for n in keep], axis=1).all(axis=1)
    post = model.posterior(votes)
    resolves = votes[:, LF_NAMES.index("tangtv")] > 0
    post = label_model.pool_marfe(post, resolves)
    state = label_model.decide(post, both, (votes > 0).sum(axis=1) >= 2, threshold)
    valid = np.stack([frame[f"{n}_valid"].to_numpy() for n in LF_NAMES], axis=1)
    valid[:, LF_NAMES.index(drop)] = False
    state[label_model.rule(votes, valid) == core.UNCERTAIN] = core.UNCERTAIN
    return state


def per_shot_tables(frame, name, reference) -> tuple[np.ndarray, list, np.ndarray]:
    """Per-shot (3, 3) confusion counts (ref x vote) and (valid, ref-certain) counts."""
    shots = sorted(frame.shot.unique())
    tables = np.zeros((len(shots), 3, 3))
    counts = np.zeros((len(shots), 2))
    vote = frame[f"{name}_vote"].to_numpy()
    valid = frame[f"{name}_valid"].to_numpy()
    certain = np.isin(reference, STATES)
    for i, (_, idx) in enumerate(frame.groupby("shot").indices.items()):
        ok = certain[idx] & valid[idx]
        counts[i] = (ok.sum(), certain[idx].sum())
        cast = ok & (vote[idx] > 0)
        np.add.at(tables[i], (reference[idx][cast] - 1, vote[idx][cast] - 1), 1)
    return tables, shots, counts


def bootstrap(tables, counts, name, rng):
    """Point estimate and 95% shot-bootstrap interval of every metric."""
    reference_classes = tuple(
        int(k + 1) for k in np.flatnonzero(tables.sum(axis=0).sum(axis=1) > 0)
    )
    point = metrics(
        tables.sum(axis=0),
        name,
        counts[:, 0].sum() / max(counts[:, 1].sum(), 1),
        counts[:, 0].sum(),
        reference_classes,
    )
    draws = []
    n = len(tables)
    for _ in range(REPLICATES if n else 0):
        pick = rng.integers(0, n, n)
        c = counts[pick].sum(axis=0)
        draws.append(
            metrics(
                tables[pick].sum(axis=0),
                name,
                c[0] / max(c[1], 1),
                c[0],
                reference_classes,
            )
        )
    out = {}
    for key, value in point.items():
        sample = np.array([d[key] for d in draws], dtype=float)
        finite = sample[np.isfinite(sample)]
        lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        out[key] = {
            "value": value,
            "ci95": [float(lo), float(hi)],
            "valid_replicates": len(finite),
            "replicates": REPLICATES,
        }
    return out


def auroc_boot(score, positive, shots) -> dict:
    """AUROC with a shot-bootstrap 95% interval (fresh generator, `BOOT_SEED`)."""
    rng = boot_rng()
    if not len(score):
        nan = float("nan")
        return {
            "value": nan,
            "ci95": [nan, nan],
            "n_bins": 0,
            "n_not_attached": 0,
            "n_shots": 0,
            "valid_replicates": 0,
            "replicates": REPLICATES,
        }
    by_shot = {s: np.flatnonzero(shots == s) for s in np.unique(shots)}
    keys = list(by_shot)
    draws = []
    for _ in range(REPLICATES):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_shot[keys[j]] for j in pick])
        draws.append(auroc(score[idx], positive[idx]))
    finite = np.asarray(draws)[np.isfinite(draws)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
    return {
        "value": auroc(score, positive),
        "ci95": [float(lo), float(hi)],
        "n_bins": len(score),
        "n_not_attached": int(positive.sum()),
        "n_shots": len(keys),
        "valid_replicates": len(finite),
        "replicates": REPLICATES,
    }


def pairwise_ci(frame: pd.DataFrame, rng) -> dict:
    """Unselected pairwise votes, with 1000-replicate shot-bootstrap intervals."""
    out = {}
    for a, b in itertools.combinations(LF_NAMES, 2):
        both = frame[f"{a}_valid"].to_numpy(bool) & frame[f"{b}_valid"].to_numpy(bool)
        cast = both & (frame[f"{a}_vote"] > 0) & (frame[f"{b}_vote"] > 0)
        tables = []
        for _, rows in frame.loc[cast].groupby("shot"):
            keep = cast[rows.index]
            table = np.zeros((3, 3))
            np.add.at(
                table,
                (
                    rows[f"{a}_vote"].to_numpy(int)[keep] - 1,
                    rows[f"{b}_vote"].to_numpy(int)[keep] - 1,
                ),
                1,
            )
            tables.append(table)
        tables = np.asarray(tables).reshape(-1, 3, 3)
        counts = np.stack([tables.sum(axis=(1, 2))] * 2, axis=1)
        scores = bootstrap(tables, counts, a, rng)
        out[f"{a}__{b}"] = {
            "population_bins": len(frame),
            "population_shots": int(frame.shot.nunique()),
            "population_shot_ids": sorted(int(s) for s in frame.shot.unique()),
            "both_valid_bins": int(both.sum()),
            "both_vote_bins": int(cast.sum()),
            "both_vote_shots": int(frame.loc[cast, "shot"].nunique()),
            "both_vote_shot_ids": sorted(
                int(s) for s in frame.loc[cast, "shot"].unique()
            ),
            "agreement": scores["agreement"],
            "kappa": scores["kappa"],
            "binary_kappa": scores["binary_kappa"],
            "binary_agreement": scores["binary_agreement"],
            "count_labels": ["attached", "detached", "marfe"],
            "count_rows": a,
            "count_columns": b,
            "counts": tables.sum(axis=0).astype(int).tolist(),
        }
    return out


def auroc_ci(frame, name, reference) -> dict:
    ok = np.isin(reference, STATES) & frame[f"{name}_valid"].to_numpy()
    sub = frame[ok]
    score = DIRECTION[name] * sub[f"{name}_value"].to_numpy(dtype=float)
    positive = reference[ok] != 1
    finite = np.isfinite(score)
    return auroc_boot(score[finite], positive[finite], sub.shot.to_numpy()[finite])


def conflicts_resolved(frame: pd.DataFrame) -> dict:
    """Bins where the rule says uncertain (conflict) and the model is certain.

    How many carry a TangTV vote and how often the model's state is that vote.
    """
    rule = frame.state_rule.to_numpy()
    lm = frame.get("state_model_diagnostic", frame.state_lm).to_numpy()
    sel = (rule == core.UNCERTAIN) & np.isin(lm, STATES)
    tv = frame["tangtv_vote"].to_numpy()
    voted = sel & (tv > 0)
    return {
        "bins": int(sel.sum()),
        "with_tangtv_vote": int(voted.sum()),
        "model_state_equals_tangtv_vote": int(np.sum(lm[voted] == tv[voted])),
    }


def failure_analysis(frame, references) -> dict:
    """Where each indicator goes wrong; every number is a count or a fraction."""
    out = {}
    afrac_ref = references["afrac"]
    certain = np.isin(afrac_ref, STATES)
    # Prad: radiation is not detachment
    tv = frame["tangtv_valid"].to_numpy() & (frame["tangtv_vote"].to_numpy() == 1)
    pr = frame["prad_valid"].to_numpy()
    sel = tv & pr
    out["prad_votes_detached_where_tangtv_attached"] = {
        "tangtv_attached_bins_with_valid_prad": int(sel.sum()),
        "fraction_prad_detached": float(
            np.mean(frame["prad_vote"].to_numpy()[sel] == 2)
        )
        if sel.any()
        else None,
        "median_f_div": float(np.nanmedian(frame["prad_value"].to_numpy()[sel]))
        if sel.any()
        else None,
    }
    # Afrac: a shot detached throughout is mis-called attached in its upper tail
    per_shot = {}
    for shot, idx in frame.groupby("shot").indices.items():
        ref = afrac_ref[idx]
        known = np.isin(ref, STATES)
        if known.sum() < 20:
            continue
        per_shot[int(shot)] = {
            "detached_or_marfe_share": float(np.mean(ref[known] != 1)),
            "afrac_attached_votes": int(
                np.sum(
                    (frame["afrac_vote"].to_numpy()[idx] == 1)
                    & frame["afrac_valid"].to_numpy()[idx]
                )
            ),
            "afrac_valid_bins": int(frame["afrac_valid"].to_numpy()[idx].sum()),
        }
    mostly = {s: v for s, v in per_shot.items() if v["detached_or_marfe_share"] >= 0.8}
    out["afrac_in_shots_detached_most_of_the_time"] = {
        "n_shots": len(mostly),
        "afrac_attached_vote_share": float(
            sum(v["afrac_attached_votes"] for v in mostly.values())
            / max(sum(v["afrac_valid_bins"] for v in mostly.values()), 1)
        ),
    }
    # by ELM share and by heating power: agreement of Afrac and Prad with the label
    for column, label, bounds in (
        ("aux_elm_share", "by_elm_share", (0.0, 0.2, 0.5, 1.01)),
        ("aux_p_in_w", "by_input_power_mw", (0.0, 2e6, 5e6, 1e9)),
    ):
        if column not in frame:
            continue
        x = frame[column].to_numpy(dtype=float)
        rows = {}
        for name in ("afrac", "prad"):
            ok = (
                np.isin(references[name], STATES)
                & frame[f"{name}_valid"].to_numpy()
                & (frame[f"{name}_vote"].to_numpy() > 0)
            )
            per_range = {}
            for lo, hi in itertools.pairwise(bounds):
                bin_ok = ok & (x >= lo) & (x < hi)
                per_range[
                    f"{lo / (1e6 if column == 'aux_p_in_w' else 1):g}-"
                    f"{hi / (1e6 if column == 'aux_p_in_w' else 1):g}"
                ] = {
                    "n_bins": int(bin_ok.sum()),
                    "binary_agreement": float(
                        np.mean(
                            (frame[f"{name}_vote"].to_numpy()[bin_ok] == 1)
                            == (references[name][bin_ok] == 1)
                        )
                    )
                    if bin_ok.any()
                    else None,
                }
            rows[name] = per_range
        out[label] = rows
    # invalid reasons: what silenced each indicator
    out["invalid_reasons"] = {
        name: {
            str(k): int(v)
            for k, v in frame.loc[~frame[f"{name}_valid"], f"{name}_reason"]
            .value_counts()
            .items()
        }
        for name in LF_NAMES
    }
    worst = sorted(
        (
            (
                s,
                float(
                    np.mean(
                        (
                            frame["afrac_vote"].to_numpy()[idx][
                                certain[idx]
                                & frame["afrac_valid"].to_numpy()[idx]
                                & (frame["afrac_vote"].to_numpy()[idx] > 0)
                            ]
                            == 1
                        )
                        == (
                            afrac_ref[idx][
                                certain[idx]
                                & frame["afrac_valid"].to_numpy()[idx]
                                & (frame["afrac_vote"].to_numpy()[idx] > 0)
                            ]
                            == 1
                        )
                    )
                ),
            )
            for s, idx in frame.groupby("shot").indices.items()
            if (
                certain[idx]
                & frame["afrac_valid"].to_numpy()[idx]
                & (frame["afrac_vote"].to_numpy()[idx] > 0)
            ).sum()
            >= 20
        ),
        key=lambda item: item[1],
    )[:5]
    out["afrac_worst_shots_binary_agreement"] = [
        {"shot": int(s), "agreement": a} for s, a in worst
    ]
    return out


#: Pairs scored threshold-free: the indicator whose continuous value is the score,
#: and the indicator whose vote (attached versus detached or MARFE) is the reference.
THRESHOLD_FREE_PAIRS = (
    ("prad", "tangtv"),
    ("afrac", "tangtv"),
    ("prad", "afrac"),
    ("prad_abs", "tangtv"),
)
#: A shot enters a within-shot statistic with this many bins where both indicators
#: are valid, and (AUROC) this many bins of each reference class.
MIN_SHOT_BINS = 20
MIN_CLASS_BINS = 5


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def interval(draws) -> list:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(v) for v in np.percentile(finite, [2.5, 97.5])]


def pooled_rho(frame, a, b) -> dict:
    """Spearman of the two oriented values on bins where both are valid, with a
    shot-bootstrap interval. Oriented: both increase with detachment."""
    rng = boot_rng()
    (va, ka, _), (vb, kb, _) = COLUMNS[a], COLUMNS[b]
    ok = (
        frame[ka].to_numpy(bool)
        & frame[kb].to_numpy(bool)
        & np.isfinite(frame[va].to_numpy(float))
        & np.isfinite(frame[vb].to_numpy(float))
    )
    sub = frame[ok]
    x = DIRECTION[a] * sub[va].to_numpy(float)
    y = DIRECTION[b] * sub[vb].to_numpy(float)
    shots = sub.shot.to_numpy()
    out = {"n_bins": int(ok.sum()), "n_shots": len(np.unique(shots))}
    if not len(x):
        return {**out, "value": float("nan"), "ci95": [float("nan")] * 2}
    by_shot = {s: np.flatnonzero(shots == s) for s in np.unique(shots)}
    keys = list(by_shot)
    draws = []
    for _ in range(REPLICATES):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_shot[keys[j]] for j in pick])
        draws.append(spearman(x[idx], y[idx]))
    return {**out, "value": spearman(x, y), "ci95": interval(draws)}


def within_shot(frame, a, b) -> dict:
    """Per-shot AUROC and Spearman, summarised by their mean and a shot bootstrap.

    A shot counts when `MIN_SHOT_BINS` bins have both indicators valid (Spearman)
    and `MIN_CLASS_BINS` bins of each reference class (AUROC). This removes the
    between-shot offset that one global cutoff cannot absorb.
    """
    (va, ka, _), (vb, kb, qb) = COLUMNS[a], COLUMNS[b]
    both = (
        frame[ka].to_numpy(bool)
        & frame[kb].to_numpy(bool)
        & np.isfinite(frame[va].to_numpy(float))
        & np.isfinite(frame[vb].to_numpy(float))
    )
    vote_b = frame[qb].to_numpy()
    rows = []
    for shot, idx in frame.groupby("shot").indices.items():
        keep = idx[both[idx]]
        if len(keep) < MIN_SHOT_BINS:
            continue
        x = DIRECTION[a] * frame[va].to_numpy(float)[keep]
        y = DIRECTION[b] * frame[vb].to_numpy(float)[keep]
        cast = vote_b[keep] > 0
        positive = vote_b[keep][cast] != core.ATTACHED
        auc = (
            auroc(x[cast], positive)
            if min(positive.sum(), (~positive).sum()) >= MIN_CLASS_BINS
            else float("nan")
        )
        rows.append((int(shot), len(keep), spearman(x, y), auc))
    table = np.array(rows, dtype=float).reshape(-1, 4)
    out = {"min_shot_bins": MIN_SHOT_BINS, "min_class_bins": MIN_CLASS_BINS}
    for column, name in ((2, "spearman"), (3, "auroc")):
        values = table[:, column]
        finite = values[np.isfinite(values)]
        out[name] = {
            "n_shots": len(finite),
            "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "mean_ci95": mean_boot(finite),
            "median": float(np.median(finite)) if len(finite) else float("nan"),
            "share_at_least_0_8": float(np.mean(finite >= 0.8))
            if len(finite)
            else float("nan"),
            "per_shot": {
                str(int(s)): {"n_bins": int(n), "value": float(v)}
                for s, n, v in zip(
                    table[:, 0],
                    table[:, 1],
                    values,
                    strict=True,
                )
                if np.isfinite(v)
            },
        }
    return out


def threshold_free(frame: pd.DataFrame) -> dict:
    """Agreement that needs no cutoff, per TangTV geometry tier and indicator pair.

    AUROC: how well indicator A's continuous value (oriented so larger means more
    detached) ranks the bins that indicator B votes detached or MARFE above the
    ones it votes attached. Spearman: rank correlation of the two oriented values
    where both are valid. Both pooled over shots and within shots, with 95%
    shot-bootstrap intervals; `kappa` of the cast votes is in `pairwise_agreement`.
    """
    out = {}
    for tier, rows in frame.groupby("tangtv_tier", sort=True):
        rows = rows.reset_index(drop=True)
        pairs = {}
        for a, b in THRESHOLD_FREE_PAIRS:
            (va, ka, _), (_, kb, qb) = COLUMNS[a], COLUMNS[b]
            ok = (
                rows[ka].to_numpy(bool)
                & rows[kb].to_numpy(bool)
                & (rows[qb].to_numpy() > 0)
                & np.isfinite(rows[va].to_numpy(float))
            )
            sub = rows[ok]
            score = DIRECTION[a] * sub[va].to_numpy(float)
            positive = sub[qb].to_numpy() != core.ATTACHED
            pairs[f"{a}__{b}"] = {
                "score": f"{va}, larger = more detached"
                if a != "afrac"
                else "afrac_value, larger Afrac = more attached (sign flipped)",
                "reference": f"{b} vote: detached or marfe versus attached",
                "auroc_pooled": auroc_boot(score, positive, sub.shot.to_numpy()),
                "spearman_pooled": pooled_rho(rows, a, b),
                "within_shot": within_shot(rows, a, b),
            }
        out[str(tier)] = pairs
    return out


def tier_agreement(frame: pd.DataFrame, rng) -> dict:
    """Every pairwise comparison retains its accepted TangTV geometry tier."""
    return {
        str(tier): pairwise_ci(rows.reset_index(drop=True), rng)
        for tier, rows in frame.groupby("tangtv_tier", sort=True)
    }


def attached_conflict(frame: pd.DataFrame) -> dict:
    """Radiation can remain high while an accepted imaging front is attached."""
    selected = (
        frame.tangtv_valid.to_numpy(bool)
        & frame.prad_valid.to_numpy(bool)
        & frame.tangtv_vote.eq(core.ATTACHED).to_numpy()
    )
    detached = selected & frame.prad_vote.eq(core.DETACHED).to_numpy()
    cast = selected & frame.prad_vote.gt(core.ABSTAIN).to_numpy()
    absolute = (
        frame.tangtv_valid.to_numpy(bool)
        & frame.prad_abs_valid.to_numpy(bool)
        & frame.tangtv_vote.eq(core.ATTACHED).to_numpy()
    )
    absolute_detached = absolute & frame.prad_abs_vote.eq(core.DETACHED).to_numpy()
    return {
        "tangtv_attached_bins_with_valid_prad": int(selected.sum()),
        "tangtv_attached_bins_with_cast_prad": int(cast.sum()),
        "prad_detached_bins": int(detached.sum()),
        "prad_abstain_bins": int((selected & ~cast).sum()),
        "shot_ids": sorted(int(s) for s in frame.loc[selected, "shot"].unique()),
        "fraction_prad_detached": float(detached.sum() / selected.sum())
        if selected.any()
        else None,
        "absolute_cutoffs_sensitivity": {
            "tangtv_attached_bins_with_valid_prad": int(absolute.sum()),
            "prad_detached_bins": int(absolute_detached.sum()),
            "fraction_prad_detached": float(absolute_detached.sum() / absolute.sum())
            if absolute.any()
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bins", default=str(root() / "labels_bins.csv.gz"))
    parser.add_argument("--bins-dir", type=Path, default=root() / "bins")
    parser.add_argument("--out", default=str(RESULT))
    args = parser.parse_args()

    frame = pd.read_csv(args.bins)
    for name in LF_NAMES:
        frame[f"{name}_valid"] = frame[f"{name}_valid"].astype(bool)
    rng = np.random.default_rng(0)
    fit_mask = (frame.split != "test").to_numpy()
    result = {
        "bin_ms": core.BIN_MS,
        "replicates": REPLICATES,
        "n_shots": int(frame.shot.nunique()),
        "n_shots_by_split": {
            s: int(frame[frame.split == s].shot.nunique())
            for s in ("train", "val", "test", "outside")
        },
        "n_assessed_bins": len(frame),
        "scope": "exploratory coverage and indicator agreement",
        "sources": {
            "labels": str(args.bins),
            "labels_sha256": hashlib.sha256(Path(args.bins).read_bytes()).hexdigest(),
            "bins_dir": str(args.bins_dir),
            "script": "scripts/labeler/detach_benchmark.py",
        },
        "independent_benchmark": False,
        "reference_note": (
            "No independent benchmark. Pairwise cast votes are compared before "
            "consensus selection and stratified by tangtv_tier. The paper "
            "population is upper-shelf Prad/TangTV only; lower-shelf votes are "
            "provisional. Agreement does not establish physical state accuracy."
        ),
        "bootstrap": {
            "unit": "shot with at least one compared vote pair",
            "seed": 0,
            "generator": "pairwise kappa: one stream; every other statistic: a "
            "fresh generator at the seed (`boot_rng`), so one population has one "
            "interval in every record",
            "replicates": REPLICATES,
            "ci": "percentile 95%; undefined draws excluded and counted",
        },
    }
    result["rule_conflicts_resolved_by_model"] = conflicts_resolved(frame)
    # Match agreement.json's eligible-shot population, including unassessed bins.
    from detach_label import load_all

    eligible = load_all(args.bins_dir)
    eligible = eligible[eligible.shot.isin(frame.shot.unique())].reset_index(drop=True)
    fit_shots = frame.loc[fit_mask, "shot"].unique()
    valid = eligible[[f"{n}_valid" for n in LF_NAMES]].to_numpy(bool)
    fitting = eligible.shot.isin(fit_shots).to_numpy() & (valid.sum(axis=1) >= 2)
    all_tiers = tier_agreement(eligible, rng)
    fit_tiers = tier_agreement(eligible[fitting].reset_index(drop=True), rng)
    result["pairwise_agreement"] = {
        "schema_note": (
            "all_eligible_bins and fit_bins_train_val_outside now nest "
            "tangtv_tier then indicator pair. No cross-tier agreement rollup "
            "is emitted. by_tangtv_tier and fit_by_tangtv_tier are named aliases."
        ),
        "all_eligible_bins": all_tiers,
        "fit_bins_train_val_outside": fit_tiers,
        "by_tangtv_tier": all_tiers,
        "fit_by_tangtv_tier": fit_tiers,
    }
    result["threshold_free_agreement"] = threshold_free(eligible)
    result["paper_agreement"] = (
        result["pairwise_agreement"]["by_tangtv_tier"]
        .get("upper_shelf", {})
        .get("prad__tangtv")
    )
    result["failure_analysis"] = {
        "prad_detached_where_tangtv_attached_by_tier": {
            str(tier): attached_conflict(rows)
            for tier, rows in eligible.groupby("tangtv_tier")
        },
        "invalid_reasons_by_tier": {
            str(tier): {
                name: {
                    str(k): int(v)
                    for k, v in rows.loc[~rows[f"{name}_valid"], f"{name}_reason"]
                    .value_counts()
                    .items()
                }
                for name in LF_NAMES
            }
            for tier, rows in eligible.groupby("tangtv_tier")
        },
    }
    result["divertor_te_check"] = {
        "record": "docs/labeler/results/detachment_te_check.json",
        "script": "scripts/labeler/detach_te_check.py",
    }
    result["indicator_names"] = {
        "afrac": "Jsat over each probe's own attached reference, read at the probe "
        "nearest the separatrix (local proxy, L-mode abstains)",
        "prad": "Prad,div,L / P_in over the shot's baseline (cutoffs from the 201081 "
        "anchor; the absolute cutoffs are the prad_abs sensitivity)",
        "tangtv": "C-III front height DZ (shelf geometry, MARFE evidence)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(dumps(result, indent=1))
    for tier, pairs in result["pairwise_agreement"]["by_tangtv_tier"].items():
        for pair, entry in pairs.items():
            if entry["both_vote_bins"]:
                print(
                    f"{tier} {pair}: {entry['both_vote_bins']} bins/"
                    f"{entry['both_vote_shots']} shots; "
                    f"kappa={entry['kappa']['value']:.3f}"
                )


if __name__ == "__main__":
    main()
