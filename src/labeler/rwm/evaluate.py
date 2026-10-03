"""Shot-grouped, nested cross-validated scoring of the RWM baselines.

The unit of everything here is the shot: folds, thresholds and the bootstrap never
split one. A slice table (`data.shot_table`, all shots stacked) carries a `role`
(`hanson`: onset-derived weak positives and assumed negatives; `comparison`:
unlabelled) and a `label` for the chosen horizon. For each outer fold
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
    labelled from its target onsets under the assumption of complete onset listing.
    `label_broad` retains post-last-onset negatives for sensitivity scoring only.
    High-beta p95 uses the whole analysis window to define an evaluation stratum,
    never a model feature or an alarm threshold.
    """
    table = table.copy()
    label = np.full(len(table), labels.UNLABELLED, dtype=np.int8)
    broad = label.copy()
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
        broad[index] = labels.slice_labels(
            table.t_ms.to_numpy()[index],
            onsets.get(int(shot), []),
            horizon_ms=horizon_ms,
            post_ms=post_ms,
            other_onsets_ms=other_onsets.get(int(shot), []),
            negative_scope="broad",
        )
    table["label"] = label
    table["label_broad"] = broad
    p95 = table.groupby("shot").betan.transform(lambda v: v.quantile(0.95))
    table["high_beta"] = table.betan >= 0.8 * p95
    table["above_proxy"] = table.betan_over_li > features.NO_WALL_FACTOR
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

    def __init__(self, columns, use_comparison=False, seed=0, **forest):
        self.columns, self.use_comparison = tuple(columns), use_comparison
        self.seed, self.forest = seed, forest

    def fit(self, train):
        kept = (train.role == "hanson") & train.label.isin(
            [labels.POSITIVE, labels.NEGATIVE]
        )
        if self.use_comparison:
            kept |= (train.role == "comparison") & (train.label == labels.UNLABELLED)
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
    """Development-only PU experiment; excluded from formal comparisons.

    `prior_scale` multiplies the class prior estimated from the training slices (the
    share of positive slices among all training slices). That does not identify
    the positive prior among unlabelled slices; this is not a validated PNU setup.
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


def score_alarms(
    traces,
    target_onsets,
    rule,
    *,
    explanation_onsets=None,
    shot_roles=None,
    alarm_scope="primary",
):
    """Alarm outcomes per shot for a `(k_low, k_high, hold_ms)` rule.

    Primary Hanson scoring ignores alarms after the last explanation onset +100 ms
    (n=1 and n=2), a small tolerance beyond the pre-last-target slice mask. Comparison
    traces retain their full span. `alarm_scope="full"` (or "full_trace") preserves
    the previous full-Hanson-trace definition for sensitivity. `shot_roles` maps
    shot IDs to roles; absent it, onset-map members are assumed Hanson.

    Outcomes contain considered `alarms`, per-target `warning_ms`, unexplained
    `false`, `early`, `ignored`, `category` and considered `span_ms`. The category
    follows `alarm.shot_outcome`; comparison categories are Alarm/No alarm and
    describe incidence on unlabelled shots, never verified false positives.
    """
    if alarm_scope not in {"primary", "full", "full_trace"}:
        raise ValueError("alarm_scope must be 'primary' or 'full'")
    k_low, k_high, hold = rule
    explanation_onsets = (
        target_onsets if explanation_onsets is None else explanation_onsets
    )
    out = {}
    for shot, (t, s) in traces.items():
        role = (
            shot_roles[shot]
            if shot_roles is not None
            else "hanson"
            if shot in target_onsets or shot in explanation_onsets
            else "comparison"
        )
        explanations = explanation_onsets.get(shot, [])
        end = (
            float(max(explanations) + labels.POST_MS)
            if alarm_scope == "primary" and role == "hanson" and len(explanations)
            else None
        )
        times = alarm.hysteresis_alarms(
            t, s, k_low, k_high, hold, max_gap_ms=MAX_GAP_MS
        )
        outcome = alarm.shot_outcome(
            times, target_onsets.get(shot, []), explanations, ignore_after_ms=end
        )
        t = np.asarray(t, dtype=float)
        start = float(t.min()) if len(t) else 0.0
        stop = float(t.max()) if len(t) else start
        if end is not None:
            stop = min(stop, end)
        outcome["span_ms"] = max(0.0, stop - start)
        if role == "comparison":
            outcome["category"] = "Alarm" if outcome["alarms"] else "No alarm"
        out[shot] = outcome
    return out


def choose_rule(
    traces,
    target_onsets,
    negative_scores,
    *,
    explanation_onsets=None,
    shot_roles=None,
    alarm_scope="primary",
):
    """The `(k_low, k_high, hold_ms)` maximising shot-level detection minus false alarms.

    The objective is the share of onsets warned in time minus the share of shots with
    any unexplained alarm, over the shots of `traces` (inner out-of-fold scores of the
    training shots). Levels are quantiles of `negative_scores`; ties go to the first
    rule tried, which has the higher level. The chosen `alarm_scope` applies during
    tuning as well as evaluation; ignored aftermath never incurs a primary penalty.
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
        outcome = score_alarms(
            traces,
            target_onsets,
            (k_low, k_high, float(hold)),
            explanation_onsets=explanation_onsets,
            shot_roles=shot_roles,
            alarm_scope=alarm_scope,
        )
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


def _outer_folds(table, shots, info, outer, seed, outer_groups, outer_shot_folds):
    """Outer shot indices, optionally leaving out a whole supplied group at a time."""
    if outer_groups is not None and outer_shot_folds is not None:
        raise ValueError("supply outer_groups or outer_shot_folds, not both")
    if outer_shot_folds is not None:
        supplied = [list(f) for f in outer_shot_folds]
        flat = [int(s) for fold in supplied for s in fold]
        if (
            len(supplied) < 2
            or any(not f for f in supplied)
            or len(flat) != len(shots)
            or len(set(flat)) != len(shots)
            or set(flat) != set(shots)
        ):
            raise ValueError("outer_shot_folds must contain every shot exactly once")
        lookup = {int(shot): i for i, shot in enumerate(shots)}
        return [np.array([lookup[int(s)] for s in f], dtype=int) for f in supplied]
    if outer_groups is not None:
        if isinstance(outer_groups, str):
            if table.groupby("shot")[outer_groups].nunique(dropna=False).ne(1).any():
                raise ValueError("outer_groups must have one group per shot")
            groups = info[outer_groups]
        else:
            groups = pd.Series(outer_groups).reindex(shots)
        if groups.isna().any():
            raise ValueError(
                "outer_groups must cover every shot without missing groups"
            )
        distinct = sorted(groups.unique(), key=str)
        if len(distinct) < 2:
            raise ValueError("outer_groups must contain at least two groups")
        return [np.flatnonzero(groups.to_numpy() == group) for group in distinct]
    strata = (info.role + "_" + info.campaign.astype(str)).to_numpy()
    return make_folds(shots, strata, outer, seed)


def cross_validate(
    table,
    make_model,
    target_onsets,
    *,
    explanation_onsets=None,
    outer=OUTER_FOLDS,
    inner=INNER_FOLDS,
    seed=0,
    alarm_scope="primary",
    outer_groups=None,
    outer_shot_folds=None,
):
    """Out-of-fold scores, slice calls and alarms for every shot of `table`.

    `make_model(seed)` returns a fresh model with `fit(table)` and `score(table)`.
    `target_onsets` is the headline n=1 set, used for tuning and detection.
    `explanation_onsets` may also contain n=2 events: they can explain alarms but
    never reward detection. Comparison shots never tune primary thresholds or rules.
    `alarm_scope` controls both tuning and scoring, as in `score_alarms`.
    `outer_groups` is a table column name or a shot-to-group map: each group is held
    out once, overriding `outer` (e.g. four run records yield four folds). Alternatively,
    `outer_shot_folds` lists explicit disjoint held-out shot sets covering every shot.
    Inner threshold/rule selection remains nested and shot-grouped in all cases.
    Returns `(oof, alarms, rules)` where
    `oof` is the table's rows with `score`, `fold` and `called` columns, `alarms` the
    `{shot: outcome}` map and `rules` the per-fold `(cutoff, rule)` choices.
    """
    if getattr(make_model(0), "binary", False):
        return _fixed_call(
            table, make_model(0), target_onsets, explanation_onsets, alarm_scope
        )
    shots = np.array(sorted(table.shot.unique()))
    info = table.drop_duplicates("shot").set_index("shot").loc[shots]
    folds = _outer_folds(
        table, shots, info, outer, seed, outer_groups, outer_shot_folds
    )
    shot_roles = info.role.to_dict()
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
            ((train.role == "hanson") & (train.label == labels.NEGATIVE)).to_numpy()
        ]
        known = (train.role == "hanson").to_numpy()
        rule = choose_rule(
            shot_traces(train[known], inner_scores[known]),
            target_onsets,
            negatives,
            explanation_onsets=explanation_onsets,
            shot_roles=shot_roles,
            alarm_scope=alarm_scope,
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
                target_onsets,
                rule,
                explanation_onsets=explanation_onsets,
                shot_roles=shot_roles,
                alarm_scope=alarm_scope,
            )
        )
        rules.append({"fold": fold_number, "cutoff": float(cutoff), "rule": list(rule)})
    return pd.concat(pieces, ignore_index=True), alarms, rules


def _fixed_call(table, model, target_onsets, explanation_onsets, alarm_scope):
    """A binary rule needs no folds: its call is the score, its alarm the first call."""
    scores = model.score(table)
    piece = table.assign(score=scores, fold=0)
    piece["called"] = piece.score >= 0.5
    alarms = score_alarms(
        shot_traces(piece, scores),
        target_onsets,
        (0.5, 0.5, 0.0),
        explanation_onsets=explanation_onsets,
        shot_roles=table.drop_duplicates("shot").set_index("shot").role.to_dict(),
        alarm_scope=alarm_scope,
    )
    return piece, alarms, [{"fold": 0, "cutoff": 0.5, "rule": [0.5, 0.5, 0.0]}]


def replay_alarms(
    oof,
    rules,
    target_onsets,
    *,
    explanation_onsets=None,
    alarm_scope="primary",
):
    """Replay saved outer-fold alarm rules without fitting or changing slice calls.

    Each shot must occur in one fold, and each saved fold must have exactly one
    rule. The scope and onset maps must match those used to select the rules.
    """
    folds = [r["fold"] for r in rules]
    if len(set(folds)) != len(folds) or set(folds) != set(oof.fold):
        raise ValueError("saved rules must cover every prediction fold exactly once")
    if not oof.groupby("shot").fold.nunique(dropna=False).eq(1).all():
        raise ValueError("saved predictions must have one fold per shot")
    roles = oof.drop_duplicates("shot").set_index("shot").role.to_dict()
    outcomes = {}
    for saved in rules:
        part = oof[oof.fold == saved["fold"]]
        outcomes.update(
            score_alarms(
                shot_traces(part, part.score.to_numpy()),
                target_onsets,
                saved["rule"],
                explanation_onsets=explanation_onsets,
                shot_roles=roles,
                alarm_scope=alarm_scope,
            )
        )
    return outcomes


# ---- scoring the pooled out-of-fold predictions ------------------------------------


def shot_records(oof, alarms, target_onsets, all_onsets=None):
    """Per-shot records for the bootstrap: `{"hanson": [...], "comparison": [...]}`.

    A record holds the shot's labelled slices (`score`, `label`, `called`) and its
    alarm outcome, with warning times only for `target_onsets`. Categories and Early
    counts follow `alarm.shot_outcome`; target-shot categories are mutually exclusive,
    while the onset detection count retains every target (and overlapping warnings).
    """
    groups = {"hanson": [], "comparison": []}
    for shot, index in oof.groupby("shot").indices.items():
        rows = oof.iloc[index]
        outcome = alarms[int(shot)]
        warnings = outcome["warning_ms"]
        if len(warnings) != len(target_onsets.get(int(shot), [])):
            raise ValueError("alarm warning list must match the target onset set")
        groups[rows.role.iloc[0]].append(
            {
                "shot": int(shot),
                "campaign": int(rows.campaign.iloc[0]),
                "score": rows.score.to_numpy(),
                "label": rows.label.to_numpy(),
                "label_broad": rows.get("label_broad", rows.label).to_numpy(),
                "high_beta": rows.high_beta.to_numpy(),
                "above_proxy": rows.above_proxy.to_numpy(),
                "called": rows.called.to_numpy(),
                "warning_ms": warnings,
                "false_alarms": len(outcome["false"]),
                "alarms": len(outcome["alarms"]),
                "early_alarms": len(outcome.get("early", [])),
                "ignored_alarms": len(outcome.get("ignored", [])),
                "category": outcome.get(
                    "category",
                    "Detected"
                    if any(w is not None for w in warnings)
                    else "Missed"
                    if warnings
                    else "No target",
                ),
                "span_ms": outcome.get(
                    "span_ms", float(rows.t_ms.max() - rows.t_ms.min())
                ),
            }
        )
    return groups


def chance_detection(records, window_ms=alarm.MAX_WARNING_MS - alarm.MIN_WARNING_MS):
    """Expected share of onsets warned if each shot's alarms fell at random.

    Each of a shot's `alarms` is placed uniformly and independently over its scored span
    (`span_ms`); an onset is warned when one lands in its warning range of `window_ms`.
    The expectation for an onset is `1 - (1 - window / span) ** alarms`, averaged over
    the target onsets of `records` (the shots' `warning_ms` entries). It is the share a
    scorer with no skill but this alarm rate would get, to read the detection rate against.
    """
    expected, onsets = 0.0, 0
    for record in records:
        n = len(record["warning_ms"])
        if not n:
            continue
        share = (
            min(1.0, window_ms / record["span_ms"]) if record["span_ms"] > 0 else 1.0
        )
        expected += n * (1.0 - (1.0 - share) ** record["alarms"])
        onsets += n
    return expected / onsets if onsets else float("nan")


def _stack(records, key):
    return np.concatenate([r[key] for r in records]) if records else np.array([])


def paired_time_by_campaign(first, second, *, replicates=1000, seed=0):
    """Forest-minus-reference slice intervals with draws restricted by campaign.

    Both prediction sets must contain the same ordered shots in each role. These
    basic paired intervals condition on the fixed fitted predictions and masks.
    """
    for role in first.keys() | second.keys():
        a = [(r["shot"], r["campaign"]) for r in first.get(role, [])]
        b = [(r["shot"], r["campaign"]) for r in second.get(role, [])]
        if a != b:
            raise ValueError("paired campaign records must have the same shot order")
    campaigns = sorted({r["campaign"] for rows in first.values() for r in rows})
    out = {}
    for year in campaigns:
        parts = [
            {k: [r for r in v if r["campaign"] == year] for k, v in groups.items()}
            for groups in (first, second)
        ]
        interval = metrics.paired_bootstrap(
            *parts, statistic, replicates=replicates, seed=seed, method="basic"
        )
        out[str(year)] = {
            k: v
            for k, v in interval.items()
            if k.startswith(("slice_", "high_beta_", "above_proxy_"))
            and k.endswith(("auroc", "auprc"))
        }
    return out


def onset_physics(table, onsets, signals_by_shot):
    """Offline-held inputs at each merged n=1 point and its pre-onset window.

    Onset values hold original EFIT samples to the exact listed time with the same
    age limit as the features. The separate window snapshot uses the first cached
    10 ms slice in [o-20, o); it must not be described as the value at onset.
    High-beta coverage counts slices in the separate 100 ms forecast window.
    """
    rows = []
    for shot in sorted(onsets):
        frame = table[table.shot == shot].sort_values("t_ms")
        signals = signals_by_shot[shot]
        for onset in onsets[shot]:
            window = frame[
                (frame.t_ms >= onset - labels.GROWTH_MS) & (frame.t_ms < onset)
            ]
            snapshot = window.iloc[0] if len(window) else None
            values = {
                key: float(
                    features.hold(*signals[key], [onset], features.EFIT_MAX_AGE_MS)[0]
                )
                for key in ("betan", "li")
            }
            ratio = (
                values["betan"] / values["li"]
                if values["li"] > 0
                else float("nan")
            )
            missing = not np.isfinite(ratio)
            row = {
                "shot": int(shot),
                "campaign": int(frame.campaign.iloc[0]),
                "onset_ms": float(onset),
                "sample_ms": float(snapshot.t_ms) if snapshot is not None else None,
                **{
                    k: float(snapshot[k]) if snapshot is not None else float("nan")
                    for k in ("betan", "li", "betan_over_li")
                },
                "elapsed_time_ms": float(snapshot[features.TIME_COLUMN])
                if snapshot is not None
                else float("nan"),
                "onset_betan": values["betan"],
                "onset_li": values["li"],
                "onset_betan_over_li": ratio,
                "onset_elapsed_time_ms": float(
                    features.time_since_flattop(*signals["ip"], [onset])[0]
                ),
                "onset_efit_missing": missing,
                "onset_below_proxy": bool(not missing and ratio < 4),
                "n_high_beta_pre_onset_slices": int(
                    (
                        (frame.t_ms >= onset - labels.HORIZON_MS)
                        & (frame.t_ms < onset)
                        & frame.high_beta
                    ).sum()
                ),
            }
            row["efit_missing"] = not np.isfinite(row["betan_over_li"])
            row["below_proxy"] = bool(
                not row["efit_missing"] and row["betan_over_li"] < 4
            )
            rows.append(row)
    by_campaign = {}
    for year in sorted({r["campaign"] for r in rows}):
        part = [r for r in rows if r["campaign"] == year]
        no_high_beta = sum(r["n_high_beta_pre_onset_slices"] == 0 for r in part)
        by_campaign[str(year)] = {
            "onsets": len(part),
            "below_proxy": sum(r["below_proxy"] for r in part),
            "efit_missing": sum(r["efit_missing"] for r in part),
            "onset_below_proxy": sum(r["onset_below_proxy"] for r in part),
            "onset_efit_missing": sum(r["onset_efit_missing"] for r in part),
            "no_high_beta_pre_onset_slices": no_high_beta,
            "no_high_beta_pre_onset_fraction": no_high_beta / len(part),
        }
    return {
        "scope": f"all {len(rows)} merged Hanson n=1 onset points, by campaign",
        "snapshot_scope": "first cached 10 ms slice in [o-20 ms, o); separate from onset inputs",
        "onset_scope": "last EFIT sample at or before exact onset; age <= EFIT_MAX_AGE_MS; offline-held inputs",
        "efit_max_age_ms": features.EFIT_MAX_AGE_MS,
        "elapsed_time_scope": "time since first |Ip| >= 0.5 MA sample",
        "high_beta_coverage_scope": "slices in [o-100 ms, o) with beta_N >= 0.8 times whole-window p95",
        "rows": rows,
        "by_campaign": by_campaign,
    }


def statistic(groups):
    """Every reported number of one configuration, from per-shot records.

    Slice metrics use only Hanson shots, conditional on assumed negative coverage.
    Broad scoring changes the mask, not the trained model or its predictions.
    """
    hanson, comparison = groups.get("hanson", []), groups.get("comparison", [])
    score, label, called = (_stack(hanson, k) for k in ("score", "label", "called"))
    keep = np.isin(label, [labels.POSITIVE, labels.NEGATIVE])
    y = label[keep] == labels.POSITIVE
    out = {
        "slice_auroc": metrics.auroc(score[keep], y),
        "slice_auprc": metrics.auprc(score[keep], y),
    }
    for name, mask, target in (
        (
            "broad",
            np.isin(_stack(hanson, "label_broad"), [0, 1]),
            _stack(hanson, "label_broad"),
        ),
        ("high_beta", keep & _stack(hanson, "high_beta").astype(bool), label),
        ("above_proxy", keep & _stack(hanson, "above_proxy").astype(bool), label),
    ):
        out[f"{name}_auroc"] = metrics.auroc(
            score[mask], target[mask] == labels.POSITIVE
        )
        out[f"{name}_auprc"] = metrics.auprc(
            score[mask], target[mask] == labels.POSITIVE
        )
    called_k = called[keep].astype(bool)
    tp, fp = int((called_k & y).sum()), int((called_k & ~y).sum())
    fn, tn = int((~called_k & y).sum()), int((~called_k & ~y).sum())
    out["slice_tpr"] = tp / (tp + fn) if tp + fn else np.nan
    out["slice_fpr"] = fp / (fp + tn) if fp + tn else np.nan
    out["slice_precision"] = tp / (tp + fp) if tp + fp else np.nan
    out["slice_f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    # Onset detection retains all targets; shot categories use any-hit precedence.
    warnings = [w for r in hanson for w in r["warning_ms"]]
    detected = [w for w in warnings if w is not None]
    out["onset_detection_rate"] = len(detected) / len(warnings) if warnings else np.nan
    out["uniform_alarm_reference"] = chance_detection(hanson)
    out["detection_minus_uniform_reference"] = (
        out["onset_detection_rate"] - out["uniform_alarm_reference"]
    )
    out["warning_ms_mean"] = float(np.mean(detected)) if detected else np.nan
    out["warning_ms_median"] = float(np.median(detected)) if detected else np.nan
    target_shots = [r for r in hanson if r["warning_ms"]]
    for category, key in (
        ("Detected", "detection"),
        ("Early", "early"),
        ("Missed", "miss"),
    ):
        out[f"hanson_shot_{key}_rate"] = (
            float(np.mean([r["category"] == category for r in target_shots]))
            if target_shots
            else np.nan
        )
    for key in ("early", "ignored"):
        out[f"hanson_{key}_alarm_incidence"] = (
            float(np.mean([r[f"{key}_alarms"] > 0 for r in hanson]))
            if hanson
            else np.nan
        )
        out[f"hanson_{key}_alarms_per_shot"] = (
            float(np.mean([r[f"{key}_alarms"] for r in hanson])) if hanson else np.nan
        )
    out["hanson_unexplained_alarm_incidence"] = (
        float(np.mean([r["false_alarms"] > 0 for r in hanson])) if hanson else np.nan
    )
    out["hanson_unexplained_alarms_per_shot"] = (
        float(np.mean([r["false_alarms"] for r in hanson])) if hanson else np.nan
    )
    out["comparison_alarm_incidence"] = (
        float(np.mean([r["alarms"] > 0 for r in comparison])) if comparison else np.nan
    )
    out["comparison_alarms_per_shot"] = (
        float(np.mean([r["alarms"] for r in comparison])) if comparison else np.nan
    )
    return out


def counts(groups):
    """The sizes behind a configuration's numbers, so the intervals can be read."""
    hanson, comparison = groups["hanson"], groups["comparison"]
    label = _stack(hanson, "label")
    warnings = [w for r in hanson for w in r["warning_ms"]]
    out = {
        "hanson_shots": len(hanson),
        "comparison_shots": len(comparison),
        "positive_slices": int((label == labels.POSITIVE).sum()),
        "negative_slices": int((label == labels.NEGATIVE).sum()),
        "excluded_slices": int((label == labels.EXCLUDED).sum()),
        "comparison_slices": len(_stack(comparison, "score")),
        "target_onsets": len(warnings),
        "onsets_warned": sum(w is not None for w in warnings),
        "hanson_target_shots": sum(bool(r["warning_ms"]) for r in hanson),
        "hanson_detected_shots": sum(r["category"] == "Detected" for r in hanson),
        "hanson_early_shots": sum(r["category"] == "Early" for r in hanson),
        "hanson_missed_shots": sum(r["category"] == "Missed" for r in hanson),
        "hanson_no_target_shots": sum(not r["warning_ms"] for r in hanson),
        "hanson_shots_with_an_unexplained_alarm": sum(
            r["false_alarms"] > 0 for r in hanson
        ),
        "comparison_shots_with_an_alarm": sum(r["alarms"] > 0 for r in comparison),
    }
    for key in ("early", "ignored"):
        out[f"hanson_{key}_alarms"] = sum(r[f"{key}_alarms"] for r in hanson)
        out[f"hanson_shots_with_an_{key}_alarm"] = sum(
            r[f"{key}_alarms"] > 0 for r in hanson
        )
    primary = np.isin(label, [labels.POSITIVE, labels.NEGATIVE])
    for name, mask, target in (
        (
            "broad",
            np.isin(_stack(hanson, "label_broad"), [0, 1]),
            _stack(hanson, "label_broad"),
        ),
        ("high_beta", primary & _stack(hanson, "high_beta").astype(bool), label),
        ("above_proxy", primary & _stack(hanson, "above_proxy").astype(bool), label),
    ):
        pos = int((target[mask] == labels.POSITIVE).sum())
        neg = int((target[mask] == labels.NEGATIVE).sum())
        out[f"{name}_positive_slices"] = pos
        out[f"{name}_negative_slices"] = neg
        out[f"{name}_prevalence"] = pos / (pos + neg) if pos + neg else np.nan
    total = out["positive_slices"] + out["negative_slices"]
    out["prevalence"] = out["positive_slices"] / total if total else np.nan
    return out


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
