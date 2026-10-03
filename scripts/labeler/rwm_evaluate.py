#!/usr/bin/env python
"""Score a Hanson-only RWM forest and fixed rules with shot-bootstrap intervals.

Primary labels follow Piccione's pre-onset negative definition, extended to the last
n=1 onset for shots with multiple events. Negatives remain assumed, not verified.
Broader Hanson negatives and two conditional regimes are scored on the same OOF
predictions. Comparison shots are unlabelled and contribute only alarm incidence.
nnPU is excluded because its earlier development used an outer evaluation fold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.events import rwm
from labeler.rwm import evaluate as ev
from labeler.rwm import features, labels, metrics

# DUSBRADIAL is zero on most 2014 traces and flagged corrupted on 2018 Hanson shots.
ALL = tuple(f for f in features.FEATURES if f != "lock_v")
FOREST = {"n_estimators": 300, "max_depth": 8, "min_leaf": 5}
SEED = 0
CONFIGS = {
    "rwm-brf": {"kind": "brf", "columns": ALL, "comparison": False},
    "rule-time-since-flattop": {"kind": "rule", "column": features.TIME_COLUMN},
    "rule-betan": {"kind": "rule", "column": "betan"},
    "rule-betan-over-li": {"kind": "rule", "column": "betan_over_li"},
    "rule-rwm-candidates": {
        "kind": "rule",
        "column": "rule_candidate",
        "binary": True,
    },
}
SPLIT_SEEDS = (1, 2, 3, 4)
PAIRS = tuple(
    ("rwm-brf", n) for n in CONFIGS if n not in ("rwm-brf", "rule-rwm-candidates")
)
_STATE: dict = {}


def clean(value):
    """Replace NaN/infinity with JSON null."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def factory(config):
    if config["kind"] == "brf":
        return lambda seed: ev.Brf(
            config["columns"], use_comparison=False, seed=seed, **FOREST
        )
    return lambda seed: ev.Rule(config["column"], binary=config.get("binary", False))


def _init(slices_path, replicates):
    paths = Paths.from_env()
    table = rwm.onset_table(paths)
    onsets, other = {}, {}
    for row in table.itertuples():
        (onsets if row.ntor == 1 else other).setdefault(int(row.shot), []).append(
            float(row.t_ms)
        )
    onsets = {s: labels.merge_close(v) for s, v in onsets.items()}
    other = {s: labels.merge_close(v) for s, v in other.items()}
    slices = pd.read_parquet(slices_path)
    roster = pd.read_csv(paths.root / "round4" / "rwm" / "shots.csv")
    slices["run_day"] = slices.shot.map(roster.set_index("shot").run.astype(str))
    _STATE.update(
        slices=slices,
        onsets=onsets,
        other=other,
        replicates=replicates,
        out_dir=paths.root / "round4" / "rwm",
    )


def summarise(oof, alarms, target, rules):
    """Bootstrap held-out predictions, including campaign-specific resamples."""
    groups = ev.shot_records(oof, alarms, target)
    result = {
        "counts": ev.counts(groups),
        "rules_by_fold": rules,
        "metrics": metrics.shot_bootstrap(
            groups, ev.statistic, replicates=_STATE["replicates"], seed=SEED
        ),
        "per_shot": sorted(
            (
                {
                    "shot": r["shot"],
                    "role": role,
                    "campaign": r["campaign"],
                    "onsets": len(r["warning_ms"]),
                    "warning_ms": r["warning_ms"],
                    "unexplained_alarms": r["false_alarms"],
                    "unexplained_alarm_times_ms": alarms[r["shot"]]["false"],
                    "early_alarms": r["early_alarms"],
                    "ignored_alarms": r["ignored_alarms"],
                    "category": r["category"],
                    "alarms": r["alarms"],
                    "span_ms": r["span_ms"],
                }
                for role, records in groups.items()
                for r in records
            ),
            key=lambda r: r["shot"],
        ),
    }
    for year in sorted({r["campaign"] for g in groups.values() for r in g}):
        part = {k: [r for r in v if r["campaign"] == year] for k, v in groups.items()}
        result.setdefault("by_campaign", {})[str(year)] = {
            "counts": ev.counts(part),
            "metrics": metrics.shot_bootstrap(
                part, ev.statistic, replicates=_STATE["replicates"], seed=SEED
            ),
        }
    return result, groups


def run_config(args):
    name, config, seed = args
    slices, target, other = _STATE["slices"], _STATE["onsets"], _STATE["other"]
    every = {
        s: sorted(target.get(s, []) + other.get(s, []))
        for s in set(target) | set(other)
    }
    table = ev.relabel(slices, target, other, labels.HORIZON_MS)
    oof, alarms, rules = ev.cross_validate(
        table, factory(config), target, explanation_onsets=every, seed=seed
    )
    groups = ev.shot_records(oof, alarms, target)
    oof_path = _STATE["out_dir"] / f"predictions_{name}_seed{seed}.parquet"
    oof.to_parquet(oof_path, index=False)
    result = {
        "model": name,
        "kind": config["kind"],
        "horizon_ms": labels.HORIZON_MS,
        "options": config,
        "fold_seed": seed,
        "counts": ev.counts(groups),
        "rules_by_fold": rules,
        "predictions": str(oof_path),
        "chance_detection": ev.chance_detection(groups["hanson"]),
    }
    if seed == SEED:
        summary, groups = summarise(oof, alarms, target, rules)
        result.update(summary)
        if name == "rule-time-since-flattop":
            primary = oof[(oof.role == "hanson") & oof.label.isin([0, 1])]
            top = primary.nlargest(60, "score")
            result["top_score_concentration"] = {
                "slices": len(top),
                "shots": {str(s): int(n) for s, n in top.shot.value_counts().items()},
            }
        # The former objective is independently re-tuned on the same inner folds.
        full_oof, full_alarms, full_rules = ev.cross_validate(
            table,
            factory(config),
            target,
            explanation_onsets=every,
            seed=seed,
            alarm_scope="full_trace",
        )
        full_path = _STATE["out_dir"] / f"predictions_{name}_full_trace.parquet"
        full_oof.to_parquet(full_path, index=False)
        sensitivity, _ = summarise(full_oof, full_alarms, target, full_rules)
        sensitivity["predictions"] = str(full_path)
        result["full_trace_alarm_sensitivity"] = sensitivity
    else:
        result["metrics"] = ev.statistic(groups)
    print(f"completed {name} seed={seed}", flush=True)
    return name, seed, clean(result), groups if seed == SEED else None


def run_pair(args):
    first, second, groups_first, groups_second = args
    interval = metrics.paired_bootstrap(
        groups_first,
        groups_second,
        ev.statistic,
        replicates=_STATE["replicates"],
        seed=SEED,
        method="basic",
    )
    return f"{first} - {second}", clean(interval)


def run_days(_):
    """Four held-out Hanson run records; no same-day siblings train or tune."""
    target, other = _STATE["onsets"], _STATE["other"]
    every = {
        s: sorted(target.get(s, []) + other.get(s, []))
        for s in set(target) | set(other)
    }
    table = ev.relabel(_STATE["slices"], target, other, labels.HORIZON_MS)
    table = table[table.role == "hanson"].copy()
    days = sorted(table.run_day.unique())
    if len(days) != 4 or table.run_day.isna().any():
        raise ValueError(f"expected exactly four Hanson run records, got {days}")
    oof, alarms, rules = ev.cross_validate(
        table,
        factory(CONFIGS["rwm-brf"]),
        target,
        explanation_onsets=every,
        seed=SEED,
        outer_groups="run_day",
    )
    result, _ = summarise(oof, alarms, target, rules)
    path = _STATE["out_dir"] / "predictions_rwm-brf_leave_run_day_out.parquet"
    oof.to_parquet(path, index=False)
    result["predictions"] = str(path)
    result["protocol"] = {
        "outer_folds": 4,
        "held_out_runs": days,
        "inner_folds": ev.INNER_FOLDS,
        "scope": "Hanson only; all sibling shots of the held-out run excluded from training and tuning",
        "interval_scope": "1000 shot resamples at fixed fitted run-held-out predictions; not a four-run population CI",
    }
    result["by_run_day"] = {}
    for day in days:
        part = oof[oof.run_day == day]
        groups = ev.shot_records(
            part, {int(s): alarms[int(s)] for s in part.shot.unique()}, target
        )
        result["by_run_day"][day] = {
            "counts": ev.counts(groups),
            "metrics": ev.statistic(groups),
        }
    print("completed four-run-day holdout", flush=True)
    return clean(result)


def screen_audit(_):
    table = ev.relabel(
        _STATE["slices"], _STATE["onsets"], _STATE["other"], labels.HORIZON_MS
    )
    called = table[table.rule_candidate > 0.5]
    hanson = called[called.role == "hanson"]
    every = {
        s: sorted(_STATE["onsets"].get(s, []) + _STATE["other"].get(s, []))
        for s in set(_STATE["onsets"]) | set(_STATE["other"])
    }
    before = sum(r.t_ms < min(every[int(r.shot)]) for r in hanson.itertuples())
    return {
        "primary_calls": int(((hanson.label == 0) | (hanson.label == 1)).sum()),
        "hanson_calls": len(hanson),
        "calls_before_first_listed_onset": int(before),
        "calls_after_first_listed_onset": int(len(hanson) - before),
        "scope": "binary screen calls on cached 10 ms Hanson slices, not comparison reference times",
    }


def legacy_record():
    """Published NSTX cells from the local Piccione digest, not fitted DIII-D data."""
    return {
        "model": "NSTX RUS forest",
        "slice_auroc": 0.918,
        "slice_tpr": 0.924,
        "slice_fpr": 0.214,
        "onsets_warned": 10,
        "target_onsets": 11,
        "comparison_shots_with_an_alarm": 2,
        "comparison_shots": 17,
        "slice_f1": None,
        "date": "2022",
        "source": "/scratch/gpfs/nc1514/FusionAIHub/.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md",
        "doi": "10.1088/1741-4326/ac44af",
        "context": "different machine, expert-reviewed stable shots, not comparable",
        "validation": "106 training and 28 test shots; seven NSTX inputs; 5 ms slices, 100 ms forecast; separate slice cutoff 0.450 and hysteresis 0.36/0.71/60 ms",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slices", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--only", nargs="*", choices=list(CONFIGS))
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs/labeler/rwm/evaluation.json"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    paths = Paths.from_env()
    slices_path = args.slices or paths.root / "round4" / "rwm" / "slices.parquet"
    names = args.only or list(CONFIGS)
    jobs = [(n, CONFIGS[n], SEED) for n in names]
    if "rwm-brf" in names:
        jobs += [("rwm-brf", CONFIGS["rwm-brf"], s) for s in SPLIT_SEEDS]
    with Pool(
        args.workers, initializer=_init, initargs=(slices_path, args.replicates)
    ) as pool:
        done = pool.map(run_config, jobs, chunksize=1)
        kept = {n: g for n, s, _, g in done if g is not None}
        pairs = [(a, b, kept[a], kept[b]) for a, b in PAIRS if a in kept and b in kept]
        paired = dict(pool.map(run_pair, pairs, chunksize=1))
        leave_run_day_out = (
            pool.map(run_days, [None])[0] if "rwm-brf" in names else None
        )
        screen = pool.map(screen_audit, [None])[0]
    results = {n: r for n, s, r, _ in done if s == SEED}
    for name, seed, result, _ in done:
        if seed != SEED:
            results[name].setdefault("split_seeds", {})[str(seed)] = result
    record = {
        "script": "scripts/labeler/rwm_evaluate.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": {
            "outer_folds": ev.OUTER_FOLDS,
            "inner_folds": ev.INNER_FOLDS,
            "fold_seed": SEED,
            "bootstrap_replicates": args.replicates,
            "bootstrap_strata": ["hanson", "comparison"],
            "forest": FOREST,
            "step_ms": features.STEP_MS,
            "primary_training": "Hanson positives and assumed negatives only",
            "primary_scoring": "Hanson only, before last n=1 onset; no verified negatives",
            "broad_sensitivity": "same predictions; post-last-onset and n=2-only negatives added",
            "high_beta": "beta_N >= 0.8 times each shot's whole-window p95 (evaluation only)",
            "above_proxy": "beta_N/l_i > 4 (evaluation only)",
            "time_baseline": "elapsed since first |Ip| >= 0.5 MA; fixed causal start proxy",
            "alarm_targets": "merged n=1 onsets only",
            "alarm_explanations": "merged n=1 and n=2 onsets",
            "alarm_tuning": (
                "threshold grid from primary negatives; alarm objective over full "
                "inner-OOF Hanson traces through last n=1/n=2 onset +100 ms; "
                "comparison shots never tune alarms"
            ),
            "primary_alarm_scope": "ignore alarms after last n=1/n=2 onset +100 ms; physical state remains unassessed",
            "full_trace_alarm_sensitivity": "independently retune on full Hanson traces; same forest fits and inner folds",
            "shot_categories": "any Detected warning takes precedence, then Early (>400 ms before a target), then Missed; n=2-only No target; comparison FP means alarm incidence, not verified stable-shot FPR",
            "alarm_grid": {
                "high_quantiles": ev.HIGH_QUANTILES,
                "low_fractions": ev.LOW_FRACTIONS,
                "hold_ms": ev.HOLD_MS,
            },
            "split": "shot-grouped; no frozen cohort overlap (shots.json/cohort_overlap)",
            "nnpu": "excluded: outer-fold-informed development and unidentified U prior",
            "rotation_claim": "removed; no isolated rotation benefit claimed",
            "interval_scope": "shot sampling at fixed fitted OOF predictions; split sensitivity separate",
        },
        "configs": results,
        "paired": paired,
        "paired_method": "basic",
        "leave_one_run_day_out": leave_run_day_out,
        "legacy": legacy_record(),
        "screen_audit": screen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(clean(record), indent=2, default=float, allow_nan=False) + "\n"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
