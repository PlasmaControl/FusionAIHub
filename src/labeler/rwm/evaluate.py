"""Shot-grouped, nested cross-validated scoring of the RWM baselines.

The unit of everything here is the shot: folds, thresholds and the bootstrap never
split one. A slice table (`data.shot_table`, all shots stacked) carries a `role`
(`hanson`: examined, labelled; `comparison`: unlabelled, scored as negative only as an
upper bound on false alarms) and a `label` for the chosen horizon. For each outer fold
the model, the slice cutoff (the ROC point nearest (0, 1), as the paper does) and the
hysteresis alarm rule are all chosen on out-of-fold scores of the *training* shots, an
inner cross-validation, so a held-out shot never sets a threshold that scores it.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import alarm, features, labels, metrics
from .forest import BalancedForest
from .nnpu import NnPU

OUTER_FOLDS = 5
INNER_FOLDS = 3
#: Alarm-rule grid. Levels are quantiles of the training negatives' own scores, so
#: the same grid suits a forest, a PU network and a raw feature; the hold is in ms.
HIGH_QUANTILES = (0.90, 0.95, 0.98, 0.99)
LOW_FRACTIONS = (0.5, 0.75, 1.0)
HOLD_MS = (0, 20, 50)
#: A step above this is a hole in the record, across which an alarm cannot run.
MAX_GAP_MS = 60.0


# ---- labels and features -------------------------------------------------------


def relabel(table, onsets, other_onsets, horizon_ms, post_ms=labels.POST_MS):
    """The table with `label` recomputed for `horizon_ms`.

    `onsets` and `other_onsets` map a shot to its target and other-kind onset times
    (ms). A `comparison` shot is `UNLABELLED` whatever its onsets; a `hanson` shot is
    labelled from its target onsets.
    """
    table = table.copy()
    label = np.full(len(table), labels.UNLABELLED, dtype=np.int8)
    for shot, index in table.groupby("shot").indices.items():
        if table.role.iloc[index[0]] != "hanson":
            continue
        label[index] = labels.slice_labels(
            table.t_ms.to_numpy()[index],
            onsets.get(int(shot), []),
            horizon_ms=horizon_ms,
            post_ms=post_ms,
            other_onsets_ms=other_onsets.get(int(shot), []),
        )
    table["label"] = label
    return table


class Imputer:
    """Training-fold medians for missing features; an all-missing column becomes 0."""

    def fit(self, frame, columns):
        self.columns = list(columns)
        medians = frame[self.columns].median()
        self.fill_ = medians.fillna(0.0).to_numpy(dtype=float)
        return self

    def transform(self, frame):
        x = frame[self.columns].to_numpy(dtype=float)
        return np.where(np.isfinite(x), x, self.fill_)


# ---- the models ------------------------------------------------------------------


class Brf:
    """Piccione et al.'s balanced random forest on the slice features.

    `use_comparison` adds the unlabelled comparison slices as assumed negatives.
    """

    def __init__(self, columns, use_comparison=True, seed=0, **forest):
        self.columns, self.use_comparison = tuple(columns), use_comparison
        self.seed, self.forest = seed, forest

    def fit(self, train):
        kept = train.label.isin([labels.POSITIVE, labels.NEGATIVE])
        if self.use_comparison:
            kept |= train.label == labels.UNLABELLED
        rows = train[kept]
        self.imputer = Imputer().fit(rows, self.columns)
        y = (rows.label == labels.POSITIVE).to_numpy().astype(float)
        self.model = BalancedForest(seed=self.seed, **self.forest).fit(
            self.imputer.transform(rows), y
        )
        return self

    def score(self, frame):
        return self.model.predict_proba(self.imputer.transform(frame))


class Nnpu:
    """The same features under the non-negative PU risk; only positives are labelled.

    `prior_scale` multiplies the class prior estimated from the training slices (the
    share of positive slices among all training slices), for the sensitivity runs.
    """

    def __init__(self, columns, prior_scale=1.0, seed=0, **network):
        self.columns, self.prior_scale = tuple(columns), prior_scale
        self.seed, self.network = seed, network

    def fit(self, train):
        rows = train[train.label != labels.EXCLUDED]
        labelled = (rows.label == labels.POSITIVE).to_numpy().astype(int)
        prior = float(np.clip(labelled.mean() * self.prior_scale, 1e-4, 0.5))
        self.prior = prior
        self.imputer = Imputer().fit(rows, self.columns)
        self.model = NnPU(prior=prior, seed=self.seed, **self.network).fit(
            self.imputer.transform(rows), labelled
        )
        return self

    def score(self, frame):
        return self.model.predict_proba(self.imputer.transform(frame))


class Rule:
    """A fixed score with nothing to fit: a table column read straight off it.

    A `binary` rule (the candidate screen) is already a call: its cutoff and alarm rule
    are fixed at 0.5 with no hold, and nothing is chosen on training shots.
    """

    def __init__(self, column, sign=1.0, binary=False):
        self.column, self.sign, self.binary = column, sign, binary

    def fit(self, train):
        return self

    def score(self, frame):
        values = self.sign * frame[self.column].to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, -np.inf)


# ---- folds, thresholds and alarms ----------------------------------------------


def make_folds(shots, strata, k, seed=0):
    """`k` disjoint shot sets, each stratum dealt round-robin so every fold has a share.

    `shots` and `strata` are parallel; the result is a list of index arrays into them.
    """
    rng = np.random.default_rng(seed)
    shots, strata = np.asarray(shots), np.asarray(strata)
    folds = [[] for _ in range(k)]
    offset = 0
    for stratum in sorted(set(strata.tolist())):
        members = np.flatnonzero(strata == stratum)
        members = members[rng.permutation(len(members))]
        for i, member in enumerate(members):
            folds[(i + offset) % k].append(member)
        offset += len(members)
    return [np.array(sorted(f), dtype=int) for f in folds]


def shot_traces(table, scores):
    """`{shot: (t_ms, score)}` of one scored table."""
    out = {}
    for shot, index in table.groupby("shot").indices.items():
        out[int(shot)] = (table.t_ms.to_numpy()[index], scores[index])
    return out


def score_alarms(traces, onsets, rule):
    """Alarm outcomes per shot for a `(k_low, k_high, hold_ms)` rule.

    Returns `{shot: {"alarms": [...], "warning_ms": [per onset or None],
    "false": [unmatched alarm times]}}`.
    """
    k_low, k_high, hold = rule
    out = {}
    for shot, (t, s) in traces.items():
        times = alarm.hysteresis_alarms(
            t, s, k_low, k_high, hold, max_gap_ms=MAX_GAP_MS
        )
        per_onset, unmatched = alarm.match_alarms(times, onsets.get(shot, []))
        out[shot] = {"alarms": times, "warning_ms": per_onset, "false": unmatched}
    return out


def choose_rule(traces, onsets, negative_scores):
    """The `(k_low, k_high, hold_ms)` maximising shot-level detection minus false alarms.

    The objective is the share of onsets warned in time minus the share of shots with
    any unexplained alarm, over the shots of `traces` (inner out-of-fold scores of the
    training shots). Levels are quantiles of `negative_scores`; ties go to the first
    rule tried, which has the higher level.
    """
    finite = negative_scores[np.isfinite(negative_scores)]
    if not len(finite):
        return (np.inf, np.inf, 0.0)
    best, best_value = None, -np.inf
    highs = sorted(
        {float(np.quantile(finite, q)) for q in HIGH_QUANTILES}, reverse=True
    )
    for k_high, fraction, hold in itertools.product(highs, LOW_FRACTIONS, HOLD_MS):
        k_low = float(finite.min() + fraction * (k_high - finite.min()))
        outcome = score_alarms(traces, onsets, (k_low, k_high, float(hold)))
        warned = [w is not None for o in outcome.values() for w in o["warning_ms"]]
        false = [bool(o["false"]) for o in outcome.values()]
        value = (np.mean(warned) if warned else 0.0) - (
            np.mean(false) if false else 0.0
        )
        if value > best_value + 1e-12:
            best, best_value = (k_low, k_high, float(hold)), value
    return best


# ---- the nested cross-validation ---------------------------------------------------


def _cut(table, shot_ids):
    return table[table.shot.isin({int(s) for s in shot_ids})]


def _fit_score(make_model, train, test, seed):
    model = make_model(seed).fit(train)
    return model.score(test)


def cross_validate(
    table,
    make_model,
    all_onsets,
    *,
    outer=OUTER_FOLDS,
    inner=INNER_FOLDS,
    seed=0,
):
    """Out-of-fold scores, slice calls and alarms for every shot of `table`.

    `make_model(seed)` returns a fresh model with `fit(table)` and `score(table)`.
    `all_onsets` maps every shot to all its onset times (both mode numbers); it explains
    alarms and sets the per-onset warning list. Returns `(oof, alarms, rules)` where
    `oof` is the table's rows with `score`, `fold` and `called` columns, `alarms` the
    `{shot: outcome}` map and `rules` the per-fold `(cutoff, rule)` choices.
    """
    if getattr(make_model(0), "binary", False):
        return _fixed_call(table, make_model(0), all_onsets)
    shots = np.array(sorted(table.shot.unique()))
    info = table.drop_duplicates("shot").set_index("shot").loc[shots]
    strata = (info.role + "_" + info.campaign.astype(str)).to_numpy()
    folds = make_folds(shots, strata, outer, seed)
    pieces, alarms, rules = [], {}, []
    for fold_number, held in enumerate(folds):
        test_shots = shots[held]
        train_shots = np.setdiff1d(shots, test_shots)
        train, test = _cut(table, train_shots), _cut(table, test_shots)
        # Inner cross-validation over the training shots, for the cutoff and the rule.
        inner_info = info.loc[train_shots]
        inner_strata = (
            inner_info.role + "_" + inner_info.campaign.astype(str)
        ).to_numpy()
        inner_scores = np.full(len(train), np.nan)
        for inner_number, inner_held in enumerate(
            make_folds(train_shots, inner_strata, inner, seed + 101)
        ):
            inner_test_shots = train_shots[inner_held]
            mask = train.shot.isin({int(s) for s in inner_test_shots}).to_numpy()
            inner_scores[mask] = _fit_score(
                make_model,
                train[~mask],
                train[mask],
                seed + 1000 * (fold_number + 1) + inner_number,
            )
        labelled = (train.role == "hanson") & train.label.isin(
            [labels.POSITIVE, labels.NEGATIVE]
        )
        cutoff = metrics.roc_cutoff(
            inner_scores[labelled.to_numpy()],
            (train.label[labelled] == labels.POSITIVE).to_numpy(),
        )
        negatives = inner_scores[
            (train.label != labels.POSITIVE).to_numpy()
            & (train.label != labels.EXCLUDED).to_numpy()
        ]
        rule = choose_rule(
            shot_traces(train, inner_scores),
            {s: all_onsets.get(s, []) for s in train_shots},
            negatives,
        )
        scores = _fit_score(
            make_model, train, test, seed + 1000 * (fold_number + 1) + 999
        )
        piece = test.assign(score=scores, fold=fold_number)
        piece["called"] = piece.score >= cutoff
        pieces.append(piece)
        alarms.update(
            score_alarms(
                shot_traces(piece, piece.score.to_numpy()),
                {int(s): all_onsets.get(int(s), []) for s in test_shots},
                rule,
            )
        )
        rules.append({"fold": fold_number, "cutoff": float(cutoff), "rule": list(rule)})
    return pd.concat(pieces, ignore_index=True), alarms, rules


def _fixed_call(table, model, all_onsets):
    """A binary rule needs no folds: its call is the score, its alarm the first call."""
    scores = model.score(table)
    piece = table.assign(score=scores, fold=0)
    piece["called"] = piece.score >= 0.5
    shots = sorted(piece.shot.unique())
    alarms = score_alarms(
        shot_traces(piece, scores),
        {int(s): all_onsets.get(int(s), []) for s in shots},
        (0.5, 0.5, 0.0),
    )
    return piece, alarms, [{"fold": 0, "cutoff": 0.5, "rule": [0.5, 0.5, 0.0]}]


# ---- scoring the pooled out-of-fold predictions ------------------------------------


def shot_records(oof, alarms, target_onsets, all_onsets):
    """Per-shot records for the bootstrap: `{"hanson": [...], "comparison": [...]}`.

    A record holds the shot's labelled slices (`score`, `label`, `called`) and its
    alarm outcome, with warning times only for `target_onsets`.
    """
    groups = {"hanson": [], "comparison": []}
    for shot, index in oof.groupby("shot").indices.items():
        rows = oof.iloc[index]
        outcome = alarms[int(shot)]
        every = all_onsets.get(int(shot), [])
        target = set(target_onsets.get(int(shot), []))
        warnings = [
            w for onset, w in zip(every, outcome["warning_ms"]) if onset in target
        ]
        groups[rows.role.iloc[0]].append(
            {
                "shot": int(shot),
                "campaign": int(rows.campaign.iloc[0]),
                "score": rows.score.to_numpy(),
                "label": rows.label.to_numpy(),
                "called": rows.called.to_numpy(),
                "warning_ms": warnings,
                "false_alarms": len(outcome["false"]),
            }
        )
    return groups


def _stack(records, key):
    return np.concatenate([r[key] for r in records]) if records else np.array([])


def statistic(groups):
    """Every reported number of one configuration, from per-shot records."""
    hanson, comparison = groups.get("hanson", []), groups.get("comparison", [])
    score, label, called = (_stack(hanson, k) for k in ("score", "label", "called"))
    keep = np.isin(label, [labels.POSITIVE, labels.NEGATIVE])
    y = label[keep] == labels.POSITIVE
    out = {
        "slice_auroc": metrics.auroc(score[keep], y),
        "slice_auprc": metrics.auprc(score[keep], y),
    }
    called_k = called[keep].astype(bool)
    tp, fp = int((called_k & y).sum()), int((called_k & ~y).sum())
    fn, tn = int((~called_k & y).sum()), int((~called_k & ~y).sum())
    out["slice_tpr"] = tp / (tp + fn) if tp + fn else np.nan
    out["slice_fpr"] = fp / (fp + tn) if fp + tn else np.nan
    out["slice_precision"] = tp / (tp + fp) if tp + fp else np.nan
    out["slice_f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    # Comparison slices scored as negatives: a lower bound, they may hold unlisted RWMs.
    score_c = _stack(comparison, "score")
    s_all = np.concatenate([score[keep], score_c])
    y_all = np.concatenate([y, np.zeros(len(score_c), dtype=bool)])
    out["mixed_auroc"] = metrics.auroc(s_all, y_all)
    out["mixed_auprc"] = metrics.auprc(s_all, y_all)
    called_c = _stack(comparison, "called").astype(bool)
    out["mixed_fpr"] = (
        (fp + int(called_c.sum())) / (fp + tn + len(called_c))
        if fp + tn + len(called_c)
        else np.nan
    )
    # The paper's per-shot scoring.
    warnings = [w for r in hanson for w in r["warning_ms"]]
    detected = [w for w in warnings if w is not None]
    out["onset_detection_rate"] = len(detected) / len(warnings) if warnings else np.nan
    out["warning_ms_mean"] = float(np.mean(detected)) if detected else np.nan
    out["warning_ms_median"] = float(np.median(detected)) if detected else np.nan
    out["hanson_false_alarm_shot_rate"] = (
        float(np.mean([r["false_alarms"] > 0 for r in hanson])) if hanson else np.nan
    )
    out["hanson_false_alarms_per_shot"] = (
        float(np.mean([r["false_alarms"] for r in hanson])) if hanson else np.nan
    )
    out["comparison_false_alarm_shot_rate"] = (
        float(np.mean([r["false_alarms"] > 0 for r in comparison]))
        if comparison
        else np.nan
    )
    out["comparison_false_alarms_per_shot"] = (
        float(np.mean([r["false_alarms"] for r in comparison]))
        if comparison
        else np.nan
    )
    return out


def counts(groups):
    """The sizes behind a configuration's numbers, so the intervals can be read."""
    hanson, comparison = groups["hanson"], groups["comparison"]
    label = _stack(hanson, "label")
    warnings = [w for r in hanson for w in r["warning_ms"]]
    return {
        "hanson_shots": len(hanson),
        "comparison_shots": len(comparison),
        "positive_slices": int((label == labels.POSITIVE).sum()),
        "negative_slices": int((label == labels.NEGATIVE).sum()),
        "excluded_slices": int((label == labels.EXCLUDED).sum()),
        "comparison_slices": len(_stack(comparison, "score")),
        "target_onsets": len(warnings),
        "onsets_warned": sum(w is not None for w in warnings),
        "hanson_shots_with_a_false_alarm": sum(r["false_alarms"] > 0 for r in hanson),
        "comparison_shots_with_an_alarm": sum(
            r["false_alarms"] > 0 for r in comparison
        ),
    }


def single_feature_auroc(table, columns=features.FEATURES):
    """AUROC of each feature alone on the labelled Hanson slices (no fitting).

    The direction is the better of the two, so each is `max(auc, 1 - auc)` with its
    sign recorded; this is read off the data it scores and is optimistic by that
    choice. A missing value ranks below every present one.
    """
    rows = table[(table.role == "hanson") & table.label.isin([0, 1])]
    y = (rows.label == labels.POSITIVE).to_numpy()
    out = {}
    for column in columns:
        values = rows[column].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, -np.inf)
        auc = metrics.auroc(values, y)
        out[column] = {
            "auroc": float(max(auc, 1 - auc)) if np.isfinite(auc) else None,
            "higher_means_unstable": bool(auc >= 0.5) if np.isfinite(auc) else None,
            "present": float(np.isfinite(rows[column].to_numpy(dtype=float)).mean()),
        }
    return out
