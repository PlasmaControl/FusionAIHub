#!/usr/bin/env python
"""Fit the label model, label every eligible shot and write the records.

    python scripts/labeler/detach_label.py

Inputs: the per-shot bins of `detach_bins.py` (`$LABELER_ROOT/round4/detach/bins`)
and the cohort split (`data/events/catalog/cohort.csv`). A bin is ASSESSED when at
least two indicators are valid on it; a shot is ELIGIBLE when at least two
indicators are each valid on at least `MIN_VALID_BINS` bins and it has that many
assessed bins. The label model is fitted on eligible shots of the cohort's `train`
and `val` splits and on shots outside the cohort; the `test` split is never used to
fit or select anything (it is labelled with the fitted model like any other shot).

Outputs:

* `data/events/detachment/extend_detach_vote/<list>.csv` (+ `.meta.json` and one
  50 ms grid per shot): the chosen labeler's states as intervals, schema of
  `data/events/README.md`; coding 0 absent, 1 attached, 2 detached, 3 marfe,
  4 uncertain; time that was not assessed has no row (unknown in the grids);
* `data/events/detachment/extend_detach_vote/records/*.json`: coverage, indicator
  agreement, learned parameters, structure selection, rule-versus-model matrix;
* `$LABELER_ROOT/round4/detach/labels_bins.csv.gz`: every assessed bin with both
  labelers' states, the posterior, the votes and the values;
* `$LABELER_ROOT/round4/detach/labels_rule.csv`: the other labeler's states as
  intervals;
* `$LABELER_ROOT/round4/detach/indicators/<shot>.csv`: the indicator traces in the
  interchange schema of the detachment review page
  (`docs/labeler/detachment_review.md`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.events.detachment import core, label_model
from labeler.events.detachment.label_model import LF_NAMES
from labeler.events.interval_tables import (
    SAMPLE_MS,
    category_labels,
    write_interval_table,
    write_label_grid,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "events" / "detachment" / "extend_detach_vote"
LIST_NAME = "detach_shots"
MIN_VALID_BINS = 20  # one second of 50 ms bins
POSTERIOR_THRESHOLD = 0.7
CV_FOLDS = 5
MIN_GRID_END_MS = 6000.0  # the catalog grid covers at least 0 to 5950 ms
STRUCTURES = {
    "independent": (),
    "prad_tangtv": (("prad", "tangtv"),),
    "afrac_tangtv": (("afrac", "tangtv"),),
    "afrac_prad": (("afrac", "prad"),),
    "all_pairs": (("afrac", "prad"), ("afrac", "tangtv"), ("prad", "tangtv")),
}


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def git_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def load_one(path: Path) -> pd.DataFrame:
    """One shot's bins (`detach_bins.py`) as a frame, a row per bin."""
    with np.load(path) as npz:
        data = {k: npz[k] for k in npz.files}
    n = len(data["start_ms"])
    row = {"shot": np.full(n, int(path.stem)), "start_ms": data["start_ms"]}
    for key, values in data.items():
        if key != "start_ms":
            row[key] = values
    return pd.DataFrame(row)


def load_all(bins_dir: Path) -> pd.DataFrame:
    """Every shot's bins as one frame (a row per bin)."""
    paths = [p for p in sorted(bins_dir.glob("*.npz")) if not p.name.startswith(".")]
    return pd.concat([load_one(p) for p in paths], ignore_index=True)


def cohort_split(shots) -> dict[int, str]:
    table = pd.read_csv(REPO / "data" / "events" / "catalog" / "cohort.csv")
    split = dict(zip(table.shot.astype(int), table.split, strict=True))
    return {int(s): split.get(int(s), "outside") for s in shots}


def matrices(frame: pd.DataFrame):
    votes = np.stack([frame[f"{n}_vote"].to_numpy() for n in LF_NAMES], axis=1)
    valid = np.stack([frame[f"{n}_valid"].to_numpy() for n in LF_NAMES], axis=1)
    return votes.astype(int), valid.astype(bool)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    labels = np.union1d(a, b)
    table = np.array([[np.sum((a == x) & (b == y)) for y in labels] for x in labels])
    n = table.sum()
    observed = np.trace(table) / n
    expected = float(table.sum(axis=1) @ table.sum(axis=0)) / n**2
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def pairwise_agreement(votes, valid) -> dict:
    """For each pair of indicators: bins where both are valid, where both vote, the
    agreement of the two votes, Cohen's kappa, and the vote-by-vote counts."""
    out = {}
    for i, a in enumerate(LF_NAMES):
        for j in range(i + 1, len(LF_NAMES)):
            b = LF_NAMES[j]
            both_valid = valid[:, i] & valid[:, j]
            both_vote = both_valid & (votes[:, i] > 0) & (votes[:, j] > 0)
            va, vb = votes[both_vote, i], votes[both_vote, j]
            counts = {
                f"{a}_{core.STATE_NAMES[x]}__{b}_{core.STATE_NAMES[y]}": int(
                    np.sum((va == x) & (vb == y))
                )
                for x in core.VOTE_STATES
                for y in core.VOTE_STATES
            }
            out[f"{a}__{b}"] = {
                "both_valid_bins": int(both_valid.sum()),
                "both_vote_bins": int(both_vote.sum()),
                "agreement": float(np.mean(va == vb)) if len(va) else None,
                "kappa": cohen_kappa(va, vb) if len(va) else None,
                "counts": counts,
            }
    return out


def coverage(frame: pd.DataFrame, eligible: set[int]) -> dict:
    out = {"n_shots_with_bins": int(frame.shot.nunique()), "n_bins": len(frame)}
    per_shot = frame.groupby("shot")
    for name in LF_NAMES:
        valid = per_shot[f"{name}_valid"].sum()
        reasons = frame.loc[~frame[f"{name}_valid"], f"{name}_reason"]
        out[name] = {
            "valid_bins": int(frame[f"{name}_valid"].sum()),
            "shots_with_valid_bins": int((valid > 0).sum()),
            f"shots_with_{MIN_VALID_BINS}_valid_bins": int(
                (valid >= MIN_VALID_BINS).sum()
            ),
            "invalid_reason_bins": {
                str(k): int(v) for k, v in reasons.value_counts().items()
            },
            "votes": {
                core.STATE_NAMES[s]: int((frame[f"{name}_vote"] == s).sum())
                for s in core.VOTE_STATES
            },
        }
    out["n_eligible_shots"] = len(eligible)
    return out


def cv_structure(votes, valid, shots, seed=0) -> dict:
    """Held-out marginal log-likelihood per bin of each correlation structure."""
    unique = np.unique(shots)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(unique, rng.permutation(len(unique)) % CV_FOLDS, strict=True))
    fold = np.array([fold_of[s] for s in shots])
    result = {}
    for name, corr in STRUCTURES.items():
        scores = []
        for k in range(CV_FOLDS):
            train, test = fold != k, fold == k
            model = label_model.LabelModel(corr=corr).fit(votes[train], valid[train])
            scores.append(model.loglik_per_obs(votes[test], valid[test]))
        result[name] = {
            "mean_heldout_loglik_per_bin": float(np.mean(scores)),
            "folds": [float(s) for s in scores],
        }
    return result


def choose_structure(structure: dict) -> str:
    """One-standard-error rule: the simplest structure (fewest correlated pairs)
    whose held-out log-likelihood is within one standard error of the best one."""
    top = max(structure, key=lambda k: structure[k]["mean_heldout_loglik_per_bin"])
    folds = np.asarray(structure[top]["folds"])
    margin = folds.std(ddof=1) / np.sqrt(len(folds))
    floor = structure[top]["mean_heldout_loglik_per_bin"] - margin
    ok = [k for k in structure if structure[k]["mean_heldout_loglik_per_bin"] >= floor]
    return min(ok, key=lambda k: (len(STRUCTURES[k]), k))


def smooth_segments(
    state: np.ndarray, start_ms: np.ndarray, width_ms: float = core.BIN_MS
) -> np.ndarray:
    """`despeckle` inside each stretch of consecutive assessed bins."""
    state = state.copy()
    assessed = state != core.ABSENT
    breaks = np.flatnonzero(
        np.r_[
            True,
            (np.diff(start_ms) != width_ms) | (assessed[1:] != assessed[:-1]),
        ]
    )
    for s, e in zip(breaks, np.r_[breaks[1:], len(state)], strict=True):
        if assessed[s]:
            state[s:e] = label_model.despeckle(state[s:e])
    return state


def label_frame(frame, model, threshold, width_ms=core.BIN_MS):
    """Both labelers' states on every bin of `frame`."""
    votes, valid = matrices(frame)
    assessed = valid.sum(axis=1) >= 2
    has_vote = (votes > 0).any(axis=1)
    resolves = votes[:, LF_NAMES.index("tangtv")] > 0
    posterior = label_model.pool_marfe(model.posterior(votes), resolves)
    state_rule = label_model.rule(votes, valid)
    state_rule[~assessed] = core.ABSENT
    out = frame[["shot", "start_ms"]].copy()
    out["assessed"] = assessed
    out["post_attached"], out["post_detached"], out["post_marfe"] = posterior.T
    out["state_lm"] = label_model.decide(posterior, assessed, has_vote, threshold)
    out["state_rule"] = state_rule
    # a flicker of one bin between two bins of the same state is not an event
    for idx in frame.groupby("shot").indices.values():
        for key in ("state_lm", "state_rule"):
            column = out.columns.get_loc(key)
            out.iloc[idx, column] = smooth_segments(
                out[key].to_numpy()[idx], frame.start_ms.to_numpy()[idx], width_ms
            )
    return out, votes, valid


def intervals(frame: pd.DataFrame, column: str, confidence=None) -> pd.DataFrame:
    """Runs of one non-absent state over consecutive bins as interval rows.

    The confidence of a run is the mean of `confidence` over its bins; blank for an
    uncertain run (the posterior of the best state is below the threshold there).
    """
    rows = []
    for shot, group in frame.groupby("shot"):
        start = group.start_ms.to_numpy()
        state = group[column].to_numpy()
        conf = None if confidence is None else group[confidence].to_numpy()
        i, n = 0, len(group)
        while i < n:
            j = i + 1
            while (
                j < n
                and state[j] == state[i]
                and start[j] == start[j - 1] + core.BIN_MS
            ):
                j += 1
            if state[i] != core.ABSENT:
                known = conf is not None and state[i] != core.UNCERTAIN
                rows.append(
                    [
                        int(shot),
                        int(state[i]),
                        float(start[i]),
                        float(start[j - 1] + core.BIN_MS),
                        round(float(np.mean(conf[i:j])), 3) if known else None,
                    ]
                )
            i = j
    return pd.DataFrame(
        rows, columns=["shot", "category", "t_start", "t_end", "confidence"]
    )


def write_grids(frame: pd.DataFrame, column: str, out_dir: Path) -> int:
    """One 50 ms grid per shot, 0 ms to its last assessed bin (at least 5950 ms);
    bins that were not assessed are unknown (NaN), never 0."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = category_labels("detachment")
    count = 0
    for shot, group in frame.groupby("shot"):
        group = group[group[column] != core.ABSENT]
        if group.empty:
            continue
        start = group.start_ms.to_numpy()
        end = max(MIN_GRID_END_MS, float(start[-1]) + SAMPLE_MS)
        axis = np.arange(0.0, end, SAMPLE_MS)
        labels = np.full(len(axis), np.nan)
        labels[np.rint(start / SAMPLE_MS).astype(int)] = group[column].to_numpy()
        write_label_grid(out_dir / f"{int(shot)}.npz", axis, labels, categories=names)
        count += 1
    shots = set(frame.shot)
    for stale in out_dir.glob("*.npz"):
        if stale.stem.isdecimal() and int(stale.stem) not in shots:
            stale.unlink()
    return count


def write_indicator_csvs(frame: pd.DataFrame, out_dir: Path) -> int:
    """Per-shot indicator traces for the review page (`detachment_review.md`):
    t_ms, afrac, prad_div (MW), tangtv_front_height (m) and their validity flags."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nan = np.full(len(frame), np.nan)
    for shot, group in frame.groupby("shot"):
        n = len(group)
        zx = group["aux_zxpt1"].to_numpy(float) if "aux_zxpt1" in group else nan[:n]
        zs = group["aux_zvsod"].to_numpy(float) if "aux_zvsod" in group else nan[:n]
        p_in = group["aux_p_in_w"].to_numpy(float) if "aux_p_in_w" in group else nan[:n]
        dz = group["tangtv_value"].to_numpy(dtype=float)
        table = pd.DataFrame(
            {
                "t_ms": group.start_ms.to_numpy() + core.BIN_MS / 2,
                "afrac": group["afrac_value"].to_numpy(),
                "afrac_valid": group["afrac_valid"].to_numpy().astype(int),
                "prad_div": group["prad_value"].to_numpy(dtype=float) * p_in / 1e6,
                "prad_div_valid": group["prad_valid"].to_numpy().astype(int),
                "tangtv_front_height": zx - (1.0 - dz) * (zx - zs),
                "tangtv_front_height_valid": group["tangtv_valid"]
                .to_numpy()
                .astype(int),
            }
        )
        table.to_csv(out_dir / f"{int(shot)}.csv", index=False, float_format="%.5g")
    return int(frame.shot.nunique())


def confusion(a: np.ndarray, b: np.ndarray) -> dict:
    keys = (1, 2, 3, 4)
    return {
        core.STATE_NAMES[x]: {
            core.STATE_NAMES[y]: int(np.sum((a == x) & (b == y))) for y in keys
        }
        for x in keys
    }


def table_meta(args, best, eligible, labeler, producer) -> dict:
    return {
        "category": "detachment",
        "categories": category_labels("detachment"),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "scripts/labeler/detach_label.py",
        "made_from": [{"producer": producer, "git_sha": git_sha()}],
        "producer": producer,
        "shot_list": f"{args.list_name}.csv",
        "shot_list_sha256": hashlib.sha256(
            ",".join(str(s) for s in sorted(eligible)).encode()
        ).hexdigest(),
        "n_requested_shots": len(eligible),
        "table_kind": "intervals",
        "coverage": (
            "Bins where at least two of the three indicators (Afrac, Prad,div, "
            "TangTV front) are valid. Time with no row was not assessed; it is not "
            "attached. 4 (uncertain) is an assessed bin where the labeler is "
            "below its confidence threshold or every valid indicator abstains."
        ),
        "labeler": labeler,
        "posterior_threshold": args.threshold,
        "label_model_structure": best,
        "fit_excluded_split": "test",
        "per_shot_files": {
            "path": f"{args.list_name}/<shot>.npz",
            "encoding": "sparse_sampled_integer_grid",
            "axis_order": ["time", "rho"],
            "time_grid": (
                "50 ms bin starts from 0 ms to the shot's last assessed bin, at "
                "least 0..5950 ms"
            ),
            "sample_interval_ms": SAMPLE_MS,
            "rho_edges": np.linspace(0, 1, 21).tolist(),
            "radial_mapping": "broadcast each scalar label across all 20 rho bins",
            "classes": category_labels("detachment"),
            "unknown": "unknown_indices: bins not assessed; not 0",
            "label_origin": f"{labeler} on the bin's indicator votes",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bins-dir", default=str(root() / "bins"))
    parser.add_argument("--list-name", default=LIST_NAME)
    parser.add_argument("--no-grids", action="store_true")
    parser.add_argument("--threshold", type=float, default=POSTERIOR_THRESHOLD)
    parser.add_argument(
        "--primary", choices=("label_model", "rule"), default="label_model"
    )
    args = parser.parse_args()

    frame = load_all(Path(args.bins_dir))
    split = cohort_split(frame.shot.unique())
    frame["split"] = frame.shot.map(split)
    votes, valid = matrices(frame)
    assessed = valid.sum(axis=1) >= 2
    per_shot_valid = pd.DataFrame(valid, columns=LF_NAMES).groupby(frame.shot).sum()
    per_shot_assessed = pd.Series(assessed).groupby(frame.shot).sum()
    eligible = set(
        per_shot_valid.index[
            ((per_shot_valid >= MIN_VALID_BINS).sum(axis=1) >= 2)
            & (per_shot_assessed >= MIN_VALID_BINS)
        ].astype(int)
    )
    work = frame[frame.shot.isin(eligible)].reset_index(drop=True)
    votes, valid = matrices(work)
    fit_mask = (work.split != "test").to_numpy() & (valid.sum(axis=1) >= 2)

    records = OUT / "records"
    records.mkdir(parents=True, exist_ok=True)
    cov = coverage(frame, eligible)
    cov["eligible_by_split"] = {
        s: sum(1 for e in eligible if split[e] == s)
        for s in ("train", "val", "test", "outside")
    }
    (records / "coverage.json").write_text(json.dumps(cov, indent=1))

    shots_fit = work.shot.to_numpy()[fit_mask]
    structure = cv_structure(votes[fit_mask], valid[fit_mask], shots_fit)
    best = choose_structure(structure)
    model = label_model.LabelModel(corr=STRUCTURES[best]).fit(
        votes[fit_mask], valid[fit_mask]
    )
    labelled, votes, valid = label_frame(work, model, args.threshold)
    for key in work.columns:
        if key.startswith(("aux_", "afrac_", "prad_", "tangtv_")) or key == "split":
            labelled[key] = work[key].to_numpy()
    # the posterior of the state the bin ended up with (a bin the despeckle pass
    # moved into its neighbours' state keeps its own, low, posterior of that state)
    post = labelled[["post_attached", "post_detached", "post_marfe"]].to_numpy()
    final = labelled.state_lm.to_numpy()
    certain = np.isin(final, core.VOTE_STATES)
    labelled["confidence"] = np.where(
        certain, post[np.arange(len(post)), np.clip(final - 1, 0, 2)], np.nan
    )
    labelled[labelled.assessed].to_csv(root() / "labels_bins.csv.gz", index=False)

    lm_state = labelled.state_lm.to_numpy()
    rule_state = labelled.state_rule.to_numpy()
    both = (lm_state != core.ABSENT) & (rule_state != core.ABSENT)
    posterior = labelled[["post_attached", "post_detached", "post_marfe"]].to_numpy()
    has_vote = (votes > 0).any(axis=1)
    by_threshold = {
        str(t): {
            core.STATE_NAMES[k]: int(
                np.sum(
                    label_model.decide(
                        posterior, labelled.assessed.to_numpy(), has_vote, t
                    )
                    == k
                )
            )
            for k in (1, 2, 3, 4)
        }
        for t in (0.5, 0.6, 0.7, 0.8, 0.9)
    }
    report = {
        "bin_ms": core.BIN_MS,
        "posterior_threshold": args.threshold,
        "fit": {
            "n_shots": len(np.unique(shots_fit)),
            "n_bins": int(fit_mask.sum()),
            "excluded_split": "test",
        },
        "structure_selection": {
            "cv_folds": CV_FOLDS,
            "rule": "simplest structure within one standard error of the best",
            "candidates": structure,
        },
        "chosen_structure": best,
        "model": {
            "names": list(model.names),
            "corr": [list(pair) for pair in model.corr],
            "theta": [float(x) for x in model.theta],
            "theta_layout": "prior(3), propensity(J), accuracy(J), correlation(P)",
        },
        "prior_logit": [float(x) for x in model.theta[:3]],
        "labelling_functions": model.accuracies(),
        "state_counts_bins": {
            "label_model": {
                core.STATE_NAMES[k]: int(np.sum(lm_state == k)) for k in (1, 2, 3, 4)
            },
            "rule": {
                core.STATE_NAMES[k]: int(np.sum(rule_state == k)) for k in (1, 2, 3, 4)
            },
        },
        "rule_vs_label_model": {
            "bins": int(both.sum()),
            "agreement": float(np.mean(lm_state[both] == rule_state[both])),
            "kappa": cohen_kappa(lm_state[both], rule_state[both]),
            "rows_rule_columns_label_model": confusion(
                rule_state[both], lm_state[both]
            ),
        },
        "uncertain_by_threshold": by_threshold,
        "primary_labeler": args.primary,
    }
    (records / "label_model.json").write_text(json.dumps(report, indent=1))
    agreement = {
        "all_eligible_bins": pairwise_agreement(votes, valid),
        "fit_bins_train_val_outside": pairwise_agreement(
            votes[fit_mask], valid[fit_mask]
        ),
    }
    (records / "agreement.json").write_text(json.dumps(agreement, indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    lm_primary = args.primary == "label_model"
    primary_column = "state_lm" if lm_primary else "state_rule"
    table = intervals(labelled, primary_column, "confidence" if lm_primary else None)
    write_interval_table(
        table,
        OUT / f"{args.list_name}.csv",
        table_meta(args, best, eligible, args.primary, "detach_vote"),
    )
    other_name = "rule" if lm_primary else "label_model"
    write_interval_table(
        intervals(labelled, "state_rule" if lm_primary else "state_lm"),
        root() / f"labels_{other_name}.csv",
        table_meta(args, best, eligible, other_name, "detach_vote"),
    )
    n_grid = (
        0
        if args.no_grids
        else write_grids(labelled, primary_column, OUT / args.list_name)
    )
    write_indicator_csvs(labelled[labelled.assessed], root() / "indicators")
    print(
        f"{len(eligible)} eligible shots, {len(table)} intervals, {n_grid} grids; "
        f"structure {best}; implied accuracies "
        + ", ".join(
            f"{k} {v['implied_accuracy']:.2f}" for k, v in model.accuracies().items()
        )
    )


if __name__ == "__main__":
    main()
