#!/usr/bin/env python
"""Benchmark each single detachment indicator against the combined label.

    python scripts/labeler/detach_benchmark.py

Reads `$LABELER_ROOT/round4/detach/labels_bins.csv.gz` (written by
`detach_label.py`: every assessed bin with the votes, the values and both labelers'
states) and writes `docs/labeler/results/detachment_benchmark.json`.

For each indicator (Afrac, Prad,div, TangTV front height), on the bins where it
is valid and the reference is a certain state (attached, detached or marfe):

* strict agreement and Cohen's kappa of its vote against the reference (bins where
  it abstains are counted in `vote_rate`, not in the agreement);
* `compatible_agreement`: the vote is consistent with the reference state (a
  "detached" vote from an indicator that cannot see the X-point is right when the
  reference is detached or marfe);
* per-state F1 (attached / detached / marfe), and the binary attached versus
  not-attached F1 and kappa, the scale Afrac and Prad can be judged on;
* AUROC of the continuous value for attached versus not attached (no threshold);
* 1000-replicate shot-bootstrap 95% intervals on all of them.

Two references. `combined` is the consensus state, which includes the indicator
being scored (circular diagnostic agreement). `loo` means "bins where the other
two indicators agree", with the scored vote withheld and compatibility gates
applied. This selects easier bins and is not independent physical truth. The
label model was fitted on non-test shots only; metrics are
given on every shot, on the fitting shots and on the cohort's test split alone, and
for TangTV also by where its front height came from (`source_inversion`,
`source_surrogate`: the owner's inversion or the regression from the raw frame).

The label is also checked against divertor Thomson Te (`divertor_te_check`), which
no indicator reads. Failure analysis follows: where the indicators go wrong, by ELM share and heating
power, and the known limits of each (see `docs/labeler/detachment.md`).
"""

from __future__ import annotations

import argparse
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
STATES = (1, 2, 3)
#: The sign that makes a larger score mean "not attached".
DIRECTION = {"afrac": -1.0, "prad": 1.0, "tangtv": 1.0}


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def kappa_from(table: np.ndarray) -> float:
    n = table.sum()
    if n == 0:
        return float("nan")
    observed = np.trace(table) / n
    expected = float(table.sum(axis=1) @ table.sum(axis=0)) / n**2
    return (
        float((observed - expected) / (1 - expected))
        if 1 - expected > np.finfo(float).eps
        else float("nan")
    )


def f1_from(table: np.ndarray, k: int) -> float:
    tp = table[k, k]
    pred, true = table[:, k].sum(), table[k, :].sum()
    if pred + true == 0:
        return float("nan")
    return float(2 * tp / (pred + true))


def metrics(table: np.ndarray, name: str, n_valid: float, n_ref: float) -> dict:
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
    present = np.flatnonzero(table.sum(axis=1) > 0)
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
        "macro_f1": float(np.mean([f1_from(table, k) for k in present]))
        if len(present)
        else float("nan"),
        "binary_f1_attached": f1_from(merged, 0),
        "binary_f1_not_attached": f1_from(merged, 1),
        "binary_kappa": kappa_from(merged),
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
    point = metrics(
        tables.sum(axis=0),
        name,
        counts[:, 0].sum() / max(counts[:, 1].sum(), 1),
        counts[:, 0].sum(),
    )
    draws = []
    n = len(tables)
    for _ in range(REPLICATES if n else 0):
        pick = rng.integers(0, n, n)
        c = counts[pick].sum(axis=0)
        draws.append(metrics(tables[pick].sum(axis=0), name, c[0] / max(c[1], 1), c[0]))
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


def auroc_boot(score, positive, shots, rng) -> dict:
    """AUROC with a shot-bootstrap 95% interval."""
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
        for _, rows in frame.groupby("shot"):
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
            "counts": tables.sum(axis=0).astype(int).tolist(),
        }
    return out


def auroc_ci(frame, name, reference, rng) -> dict:
    ok = np.isin(reference, STATES) & frame[f"{name}_valid"].to_numpy()
    sub = frame[ok]
    score = DIRECTION[name] * sub[f"{name}_value"].to_numpy(dtype=float)
    positive = reference[ok] != 1
    finite = np.isfinite(score)
    return auroc_boot(score[finite], positive[finite], sub.shot.to_numpy()[finite], rng)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bins", default=str(root() / "labels_bins.csv.gz"))
    parser.add_argument("--bins-dir", type=Path, default=root() / "bins")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--out", default=str(RESULT))
    args = parser.parse_args()

    frame = pd.read_csv(args.bins)
    for name in LF_NAMES:
        frame[f"{name}_valid"] = frame[f"{name}_valid"].astype(bool)
    rng = np.random.default_rng(0)
    fit_mask = (frame.split != "test").to_numpy()
    lm_state = frame.state_lm.to_numpy()
    subsets = {
        "all": np.ones(len(frame), bool),
        "fit_shots": fit_mask,
        "test_shots": ~fit_mask,
    }
    result = {
        "bin_ms": core.BIN_MS,
        "replicates": REPLICATES,
        "n_shots": int(frame.shot.nunique()),
        "n_shots_by_split": {
            s: int(frame[frame.split == s].shot.nunique())
            for s in ("train", "val", "test", "outside")
        },
        "n_assessed_bins": len(frame),
        "reference_note": (
            "combined = diagnostic consensus including the indicator scored "
            "(circular); loo = bins where the other two indicators agree, with "
            "the scored vote withheld and compatibility/posterior gates. "
            "Selection favors agreement. TangTV's reference is the low-confidence "
            "Afrac/Prad proxy pair; neither reference is independently reviewed truth."
        ),
        "reference_labels": {
            "combined": "compatible diagnostic consensus (circular)",
            "loo": "bins where the other two indicators agree",
        },
    }
    model = load_model()
    loo = {name: loo_state(frame, name, model, args.threshold) for name in LF_NAMES}
    result["indicators"] = {name: {} for name in LF_NAMES}
    result["model_accuracies"] = model.accuracies()
    for name in LF_NAMES:
        for ref_name, reference in (("combined", lm_state), ("loo", loo[name])):
            masks = dict(subsets)
            if name == "tangtv" and "tangtv_source" in frame:
                for source in ("inversion", "surrogate"):
                    masks[f"source_{source}"] = (
                        frame.tangtv_source == source
                    ).to_numpy()
            for subset, mask in masks.items():
                sub = frame[mask].reset_index(drop=True)
                ref = reference[mask]
                if not np.isin(ref, STATES).any():
                    continue
                tables, shot_ids, counts = per_shot_tables(sub, name, ref)
                entry = bootstrap(tables, counts, name, rng)
                entry["auroc_attached_vs_not"] = auroc_ci(sub, name, ref, rng)
                entry["population_shots"] = int(sub.shot.nunique())
                entry["n_shots"] = int(np.sum(tables.sum(axis=(1, 2)) > 0))
                entry["population_shot_ids"] = [int(s) for s in shot_ids]
                entry["scored_shot_ids"] = [
                    int(s)
                    for s, n in zip(shot_ids, tables.sum(axis=(1, 2)), strict=True)
                    if n > 0
                ]
                entry["reference_certain_bins"] = int(np.isin(ref, STATES).sum())
                entry["valid_reference_bins"] = int(counts[:, 0].sum())
                result["indicators"][name].setdefault(ref_name, {})[subset] = entry
    result["failure_analysis"] = failure_analysis(frame, loo)
    result["rule_conflicts_resolved_by_model"] = conflicts_resolved(frame)
    # Match agreement.json's eligible-shot population, including unassessed bins.
    from detach_label import load_all

    eligible = load_all(args.bins_dir)
    eligible = eligible[eligible.shot.isin(frame.shot.unique())].reset_index(drop=True)
    fit_shots = frame.loc[frame.split != "test", "shot"].unique()
    valid = eligible[[f"{n}_valid" for n in LF_NAMES]].to_numpy(bool)
    fitting = eligible.shot.isin(fit_shots).to_numpy() & (valid.sum(axis=1) >= 2)
    result["pairwise_agreement"] = {
        "all_eligible_bins": pairwise_ci(eligible, rng),
        "fit_bins_train_val_outside": pairwise_ci(
            eligible[fitting].reset_index(drop=True), rng
        ),
    }
    result["divertor_te_check"] = {
        "status": "withdrawn",
        "reason": "Unlocalised real-time sample-and-hold points; no processed DTS/psiN chord check. No temperature accuracy claims.",
    }
    result["indicator_names"] = {
        "afrac": "uncalibrated Jsat ratio (local proxy)",
        "prad": "Prad,div/P_in (local development thresholds)",
        "tangtv": "C-III front (local geometry, quality and MARFE gates)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(dumps(result, indent=1))
    for name in LF_NAMES:
        for ref_name in ("combined", "loo"):
            e = result["indicators"][name].get(ref_name, {}).get("all")
            if e:
                print(
                    f"{name:7s} vs {ref_name:8s} agree {e['agreement']['value']:.2f} "
                    f"kappa {e['kappa']['value']:.2f} "
                    f"binF1(att) {e['binary_f1_attached']['value']:.2f} "
                    f"auroc {e['auroc_attached_vs_not']['value']:.2f} "
                    f"bins {e['n_bins']['value']:.0f}"
                )


if __name__ == "__main__":
    main()
