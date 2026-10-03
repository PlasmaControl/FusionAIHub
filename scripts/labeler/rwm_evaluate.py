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
import re
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
from labeler.rwm import data, features, labels, metrics
from labeler.rwm import evaluate as ev
from labeler.rwm.records import (
    evaluation_details_path,
    load_evaluation,
    write_evaluation,
)

# DUSBRADIAL is zero on most 2014 traces and flagged corrupted on 2018 Hanson shots.
ALL = tuple(f for f in features.FEATURES if f != "lock_v")
FOREST = {"n_estimators": 300, "max_depth": 8, "min_leaf": 5}
SEED = 0
CONFIGS = {
    "rwm-brf": {"kind": "brf", "columns": ALL, "comparison": False},
    "rwm-rule-elapsed-time": {"kind": "rule", "column": features.TIME_COLUMN},
    "rwm-rule-betan": {"kind": "rule", "column": "betan"},
    "rwm-rule-betan-over-li": {"kind": "rule", "column": "betan_over_li"},
    "rwm-rule-rwm-candidates": {
        "kind": "rule",
        "column": "rule_candidate",
        "binary": True,
    },
}
SPLIT_SEEDS = (1, 2, 3, 4)
PAIRS = tuple(
    ("rwm-brf", n) for n in CONFIGS if n not in ("rwm-brf", "rwm-rule-rwm-candidates")
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
            config["columns"],
            use_comparison=config.get("comparison", False),
            seed=seed,
            **FOREST,
        )
    return lambda seed: ev.Rule(config["column"], binary=config.get("binary", False))


def _init(slices_path, replicates, saved_record=None):
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
    slices["run_record"] = slices.shot.map(roster.set_index("shot").run.astype(str))
    _STATE.update(
        slices=slices,
        roster=roster,
        onsets=onsets,
        other=other,
        replicates=replicates,
        out_dir=paths.root / "round4" / "rwm",
        saved_record=saved_record,
    )


def replay(saved, target, every, alarm_scope="primary"):
    oof = pd.read_parquet(saved["predictions"])
    oof = oof.rename(columns={"run_day": "run_record"})
    rules = saved["rules_by_fold"]
    alarms = ev.replay_alarms(
        oof,
        rules,
        target,
        explanation_onsets=every,
        alarm_scope=alarm_scope,
    )
    return oof, alarms, rules


def summarise(oof, alarms, target, rules):
    """Bootstrap held-out predictions, including campaign-specific resamples."""
    groups = ev.shot_records(oof, alarms, target)
    result = {
        "counts": ev.counts(groups),
        "within_shot_auroc": ev.within_shot_auroc(oof),
        "phase_controlled_auroc": ev.phase_controlled_bootstrap(
            groups, replicates=_STATE["replicates"], seed=SEED
        ),
        "phase_bin_width_sensitivity": {
            str(width)
            if width is not None
            else "no_time_control": ev.phase_controlled_auroc(groups, bin_ms=width)
            for width in (50, 100, 200, 400, 800, None)
        },
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
                    "any_alarm_category": r["any_alarm_category"],
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
    roster = _STATE["roster"].set_index("shot")
    comparison = groups["comparison"]
    for column, key in (("run_title", "run_title"), ("role", "pool_role")):
        cells = {}
        for r in comparison:
            value = roster.loc[r["shot"], column]
            cell = cells.setdefault(value, {"shots": [], "alarm_shots": []})
            cell["shots"].append(r["shot"])
            if r["alarms"]:
                cell["alarm_shots"].append(r["shot"])
        result[f"comparison_by_{key}"] = {
            value: {
                **cell,
                "n_shots": len(cell["shots"]),
                "n_alarm_shots": len(cell["alarm_shots"]),
                "alarm_incidence": len(cell["alarm_shots"]) / len(cell["shots"]),
            }
            for value, cell in sorted(cells.items())
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
    saved_record = _STATE["saved_record"]
    saved = saved_record["configs"][name] if saved_record is not None else None
    if saved is not None and seed != SEED:
        saved = saved["split_seeds"][str(seed)]
    if saved_record is not None and saved is None:
        raise ValueError(
            f"saved replay has no prediction record for {name} seed={seed}"
        )
    if saved is not None:
        oof, alarms, rules = replay(saved, target, every)
        oof_path = Path(saved["predictions"])
    else:
        oof, alarms, rules = ev.cross_validate(
            table, factory(config), target, explanation_onsets=every, seed=seed
        )
        oof_path = _STATE["out_dir"] / f"predictions_{name}_seed{seed}.parquet"
        oof.to_parquet(oof_path, index=False)
    summary, groups = summarise(oof, alarms, target, rules)
    result = {
        "model": name,
        "kind": config["kind"],
        "horizon_ms": labels.HORIZON_MS,
        "options": config,
        "fold_seed": seed,
        "predictions": str(oof_path),
        "chance_detection": ev.chance_detection(groups["hanson"]),
        **summary,
    }
    if seed == SEED:
        if name in ("rwm-brf", "rwm-rule-elapsed-time"):
            mask = ev.first_onset_mask(oof, target)
            first_groups = {
                "hanson": [
                    {
                        "score": part.loc[mask.loc[part.index], "score"].to_numpy(),
                        "label": part.loc[mask.loc[part.index], "label"].to_numpy(),
                    }
                    for _, part in oof[oof.role == "hanson"].groupby("shot")
                ]
            }

            def first_statistic(draws):
                rows = draws["hanson"]
                return metrics.auroc(
                    np.concatenate([r["score"] for r in rows]),
                    np.concatenate([r["label"] for r in rows]),
                )

            selected = oof[mask]
            result["first_onset_only"] = {
                "scope": "primary Hanson slices strictly before first merged n=1 onset; unchanged saved scores",
                "n_positive": int(selected.label.eq(labels.POSITIVE).sum()),
                "n_negative": int(selected.label.eq(labels.NEGATIVE).sum()),
                "n_shots": int(selected.shot.nunique()),
                "pooled_auroc": metrics.shot_bootstrap(
                    first_groups,
                    first_statistic,
                    replicates=_STATE["replicates"],
                    seed=SEED,
                ),
            }
        if name == "rwm-rule-elapsed-time":
            primary = oof[(oof.role == "hanson") & oof.label.isin([0, 1])]
            top = primary.nlargest(60, "score")
            result["top_score_concentration"] = {
                "slices": len(top),
                "shots": {str(s): int(n) for s, n in top.shot.value_counts().items()},
            }
        # The former objective is independently re-tuned on the same inner folds.
        if saved is not None:
            full_saved = saved["full_trace_alarm_sensitivity"]
            full_oof, full_alarms, full_rules = replay(
                full_saved, target, every, "full_trace"
            )
            full_path = Path(full_saved["predictions"])
        else:
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
    print(f"completed {name} seed={seed}", flush=True)
    return name, seed, clean(result), groups


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
    interval["within_shot_auroc"] = ev.paired_within_shot_auroc(
        groups_first, groups_second, replicates=_STATE["replicates"], seed=SEED
    )
    interval["phase_controlled_auroc"] = ev.phase_controlled_bootstrap(
        groups_first, groups_second, replicates=_STATE["replicates"], seed=SEED
    )
    return f"{first} - {second}", clean(interval)


def run_records(references):
    """Four held-out Hanson run records; no same-record siblings train or tune."""
    target, other = _STATE["onsets"], _STATE["other"]
    every = {
        s: sorted(target.get(s, []) + other.get(s, []))
        for s in set(target) | set(other)
    }
    table = ev.relabel(_STATE["slices"], target, other, labels.HORIZON_MS)
    table = table[table.role == "hanson"].copy()
    runs = sorted(table.run_record.unique())
    if len(runs) != 4 or table.run_record.isna().any():
        raise ValueError(f"expected exactly four Hanson run records, got {runs}")
    saved_record = _STATE["saved_record"]
    if saved_record is not None:
        saved = (
            saved_record.get("leave_one_run_record_out")
            or saved_record["leave_one_run_day_out"]
        )
        oof, alarms, rules = replay(saved, target, every)
    else:
        oof, alarms, rules = ev.cross_validate(
            table,
            factory(CONFIGS["rwm-brf"]),
            target,
            explanation_onsets=every,
            seed=SEED,
            outer_groups="run_record",
        )
    result, groups = summarise(oof, alarms, target, rules)
    # The holdout excludes comparisons; pair the identical Hanson elapsed ranks.
    time_groups, beta_groups = references
    reference = {k: time_groups[k] if k == "hanson" else [] for k in groups}
    result["paired_time"] = metrics.paired_bootstrap(
        groups,
        reference,
        ev.statistic,
        replicates=_STATE["replicates"],
        seed=SEED,
        method="basic",
    )
    result["paired_time_by_campaign"] = ev.paired_time_by_campaign(
        groups, reference, replicates=_STATE["replicates"], seed=SEED
    )
    result["paired_phase"] = {
        name: ev.phase_controlled_bootstrap(
            groups, baseline, replicates=_STATE["replicates"], seed=SEED
        )
        for name, baseline in (
            ("rwm-rule-elapsed-time", time_groups),
            ("rwm-rule-betan-over-li", beta_groups),
        )
    }
    path = _STATE["out_dir"] / "predictions_rwm-brf_leave_run_record_out.parquet"
    if saved_record is None:
        oof.to_parquet(path, index=False)
    result["predictions"] = str(path)
    result["protocol"] = {
        "outer_folds": 4,
        "held_out_runs": runs,
        "name": "leave-one-run-record-out",
        "calendar_date_note": "20180314 and 20180314A share a date and remain separate; no calendar-day isolation is claimed",
        "inner_folds": ev.INNER_FOLDS,
        "scope": "Hanson only; all sibling shots of the held-out run excluded from training and tuning",
        "interval_scope": "1000 shot resamples at fixed fitted run-held-out predictions; not a four-run population CI",
    }
    result["by_run_record"] = {}
    for run in runs:
        part = oof[oof.run_record == run]
        groups = ev.shot_records(
            part, {int(s): alarms[int(s)] for s in part.shot.unique()}, target
        )
        result["by_run_record"][run] = {
            "counts": ev.counts(groups),
            "metrics": ev.statistic(groups),
        }
    print("completed leave-one-run-record-out", flush=True)
    return clean(result)


def split_summary(config, paired_by_seed, paired_by_campaign, phase_pairs, holdout):
    """Point-estimate ranges across all splits, separate from shot-sampling CIs."""
    runs = {str(SEED): config, **config["split_seeds"]}
    keys = ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
    ranges = {}
    for campaign in ("pooled", *config["by_campaign"]):
        ranges[campaign] = {}
        for key in keys:
            values = {
                seed: (run if campaign == "pooled" else run["by_campaign"][campaign])[
                    "metrics"
                ][key]["estimate"]
                for seed, run in runs.items()
            }
            ranges[campaign][key] = {
                "by_seed": values,
                "min": min(values.values()),
                "max": max(values.values()),
            }
    paired_ranges = {}
    for key in keys:
        values = [row[key] for row in paired_by_seed.values()]
        paired_ranges[key] = {
            "min": min(v["estimate"] for v in values),
            "max": max(v["estimate"] for v in values),
            "all_cis_include_zero": all(v["low"] <= 0 <= v["high"] for v in values),
        }
    return {
        "seeds": [int(s) for s in runs],
        "range_scope": "min/max point AUROC across five fixed-hyperparameter shot-fold splits, not a CI",
        "paired_reference": "fixed elapsed-time ranks from seed 0 (no fitting; scores identical for every split); pooled primary and two conditional masks",
        "auroc_ranges": ranges,
        "paired_time_by_seed": paired_by_seed,
        "paired_time_ranges": paired_ranges,
        "paired_time_by_campaign": paired_by_campaign,
        "phase_controlled_auroc": {
            "by_seed": {s: r["phase_controlled_auroc"] for s, r in runs.items()},
            "min": min(r["phase_controlled_auroc"]["estimate"] for r in runs.values()),
            "max": max(r["phase_controlled_auroc"]["estimate"] for r in runs.values()),
            "lowest_seed": min(
                runs, key=lambda s: runs[s]["phase_controlled_auroc"]["estimate"]
            ),
            "holdout": holdout["phase_controlled_auroc"],
        },
        "paired_phase": {
            name: {
                "by_seed": rows,
                "holdout": holdout["paired_phase"][name],
                "min": min(r["estimate"] for r in rows.values()),
                "max": max(r["estimate"] for r in rows.values()),
                "positive_estimates": sum(r["estimate"] > 0 for r in rows.values()),
                "cis_excluding_zero": sum(
                    r["low"] > 0 or r["high"] < 0 for r in rows.values()
                ),
            }
            for name, rows in phase_pairs.items()
        },
        "alarm_ranges": {
            key: {
                "by_seed": {s: r["metrics"][key] for s, r in runs.items()},
                "min": min(r["metrics"][key]["estimate"] for r in runs.values()),
                "max": max(r["metrics"][key]["estimate"] for r in runs.values()),
                "all_cis_include_zero": all(
                    r["metrics"][key]["low"] <= 0 <= r["metrics"][key]["high"]
                    for r in runs.values()
                ),
            }
            for key in (
                "onset_detection_rate",
                "detection_minus_uniform_reference",
                "warning_ms_median",
            )
        },
    }


def run_campaign_pair(args):
    seed, first, second = args
    return str(seed), clean(
        ev.paired_time_by_campaign(
            first, second, replicates=_STATE["replicates"], seed=SEED
        )
    )


def run_onset_physics(_):
    table = ev.relabel(
        _STATE["slices"], _STATE["onsets"], _STATE["other"], labels.HORIZON_MS
    )
    signals = {
        shot: data.load_signals(shot, Paths.from_env()) for shot in _STATE["onsets"]
    }
    result = ev.onset_physics(table, _STATE["onsets"], signals)
    result["labelled_slice_qmin"] = {
        "scope": (
            "Hanson primary 10 ms slices labelled 0 or 1; held offline qmin; "
            "fraction among all labelled slices, with missing values in "
            "denominator; finite-only fraction separate"
        ),
        "by_campaign": {
            str(year): {
                "n_labelled_slices": len(part),
                "n_finite": int(np.isfinite(part.qmin).sum()),
                "n_above_two": int((part.qmin > 2).sum()),
                "fraction_above_two": float((part.qmin > 2).mean()),
                "finite_fraction_above_two": float(
                    (part.qmin[np.isfinite(part.qmin)] > 2).mean()
                ),
            }
            for year, part in table[
                (table.role == "hanson") & table.label.isin([0, 1])
            ].groupby("campaign")
        },
    }
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


def forecast_audit(_):
    """Slice-level label, input-missingness and raw-rotation audits, no fitting."""
    table = ev.relabel(
        _STATE["slices"], _STATE["onsets"], _STATE["other"], labels.HORIZON_MS
    )
    primary = (table.role == "hanson") & table.label.isin([0, 1])
    before = ev.first_onset_mask(table, _STATE["onsets"])
    positive, negative = primary & table.label.eq(1), primary & table.label.eq(0)
    efit = ["betan", "li", "q95", "qmin", "wmhd_mj"]
    missing = ~np.isfinite(table[efit]).all(axis=1)
    return {
        "scope": "primary Hanson 10 ms slices; after first merged n=1 onset",
        "negative_after_first": int((negative & ~before).sum()),
        "positive_before_repeats": int((positive & ~before).sum()),
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
        "broad_n2_only_negatives": int(
            (
                (table.role == "hanson")
                & table.label.eq(labels.EXCLUDED)
                & table.label_broad.eq(0)
                & ~table.shot.isin(_STATE["onsets"])
            ).sum()
        ),
        "efit_missing": {
            "scope": "at least one missing beta_N/li/q95/qmin/W_MHD input",
            "positive": int((missing & positive).sum()),
            "negative": int((missing & negative).sum()),
            "positive_fraction": float(missing[positive].mean()),
            "negative_fraction": float(missing[negative].mean()),
        },
        "rotation_raw": {
            column: {
                "median": float(table[column].median()),
                "max": float(table[column].max()),
            }
            for column in ("rot_core_khz", "rot_mid_khz")
        },
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
        "stable_shots_with_an_alarm": 2,
        "stable_shots": 17,
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
        "--compact-saved",
        action="store_true",
        help="move saved shot details externally without rescoring or fitting",
    )
    parser.add_argument(
        "--rescore-saved",
        action="store_true",
        help="replay predictions and fold rules from --out without refitting or tuning",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs/labeler/rwm/evaluation.json"
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    paths = Paths.from_env()
    details_path = evaluation_details_path(
        args.out,
        paths.root / "round4/rwm",
        canonical_summary=REPO / "outputs/labeler/rwm/evaluation.json",
    )
    if args.compact_saved:
        if args.rescore_saved or args.only:
            parser.error("--compact-saved cannot be combined with scoring options")
        write_evaluation(
            load_evaluation(args.out, details=True), args.out, details_path
        )
        print(f"compacted {args.out}; complete record at {details_path}")
        return
    slices_path = args.slices or paths.root / "round4" / "rwm" / "slices.parquet"
    names = args.only or list(CONFIGS)
    if "rwm-brf" in names:
        for reference in ("rwm-rule-elapsed-time", "rwm-rule-betan-over-li"):
            if reference not in names:
                names = [*names, reference]
    if args.rescore_saved and args.only:
        parser.error("--rescore-saved requires the complete saved configuration set")
    saved_record = load_evaluation(args.out) if args.rescore_saved else None
    if args.rescore_saved and not isinstance(saved_record, dict):
        parser.error("saved replay requires an evaluation object; fitting is forbidden")
    if saved_record is not None:
        renames = {n.removeprefix("rwm-"): n for n in CONFIGS if "-rule-" in n}

        def rename_saved(value):
            if isinstance(value, str):
                return re.sub(
                    r"(?<!rwm-)"
                    + "|(?<!rwm-)".join(
                        re.escape(n) for n in sorted(renames, key=len, reverse=True)
                    ),
                    lambda match: renames[match.group()],
                    value,
                )
            if isinstance(value, dict):
                return {rename_saved(k): rename_saved(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rename_saved(v) for v in value]
            return value

        saved_record = rename_saved(saved_record)
        if set(saved_record["configs"]) != set(CONFIGS):
            parser.error(
                "saved replay requires all model records; fitting is forbidden"
            )
        for old, new in renames.items():
            for path in (paths.root / "round4" / "rwm").glob(
                f"predictions_{old}_*.parquet"
            ):
                target = path.with_name(path.name.replace(old, new))
                if target.exists():
                    raise FileExistsError(
                        f"refusing to replace saved predictions: {target}"
                    )
                path.rename(target)
    jobs = [(n, CONFIGS[n], SEED) for n in names]
    if "rwm-brf" in names:
        jobs += [("rwm-brf", CONFIGS["rwm-brf"], s) for s in SPLIT_SEEDS]
    with Pool(
        args.workers,
        initializer=_init,
        initargs=(slices_path, args.replicates, saved_record),
    ) as pool:
        done = pool.map(run_config, jobs, chunksize=1)
        kept = {n: g for n, s, _, g in done if s == SEED}
        pairs = [(a, b, kept[a], kept[b]) for a, b in PAIRS if a in kept and b in kept]
        paired = dict(pool.map(run_pair, pairs, chunksize=1))
        leave_run_record_out = (
            pool.map(
                run_records,
                [(kept["rwm-rule-elapsed-time"], kept["rwm-rule-betan-over-li"])],
            )[0]
            if "rwm-brf" in names
            else None
        )
        time_pairs = [
            (str(s), "time", g, kept["rwm-rule-elapsed-time"])
            for n, s, _, g in done
            if n == "rwm-brf" and "rwm-rule-elapsed-time" in kept
        ]
        time_results = pool.map(run_pair, time_pairs, chunksize=1)
        paired_time = {
            name.split(" - ")[0]: {
                k: v
                for k, v in row.items()
                if k.startswith(("slice_", "high_beta_", "above_proxy_"))
                and k.endswith(("auroc", "auprc"))
            }
            for name, row in time_results
        }
        beta_pairs = [
            (str(s), "beta", g, kept["rwm-rule-betan-over-li"])
            for n, s, _, g in done
            if n == "rwm-brf"
        ]
        phase_pairs = {
            "rwm-rule-elapsed-time": {
                n.split(" - ")[0]: r["phase_controlled_auroc"] for n, r in time_results
            },
            "rwm-rule-betan-over-li": {
                n.split(" - ")[0]: r["phase_controlled_auroc"]
                for n, r in pool.map(run_pair, beta_pairs, chunksize=1)
            },
        }
        campaigns_by_seed = dict(
            pool.map(
                run_campaign_pair,
                [(int(s), a, b) for s, _, a, b in time_pairs],
                chunksize=1,
            )
        )
        paired_campaigns = {
            year: {s: row[year] for s, row in campaigns_by_seed.items()}
            for year in sorted(
                {year for row in campaigns_by_seed.values() for year in row}
            )
        }
        screen = pool.map(screen_audit, [None])[0]
        physics = pool.map(run_onset_physics, [None])[0]
        label_audit = pool.map(forecast_audit, [None])[0]
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
            "split_seeds": [SEED, *SPLIT_SEEDS],
            "prediction_mode": "saved replay; no refit or retuning"
            if args.rescore_saved
            else "nested CV fitting",
            "bootstrap_replicates": args.replicates,
            "bootstrap_strata": ["hanson", "comparison"],
            "forest": FOREST,
            "step_ms": features.STEP_MS,
            "primary_training": "Hanson positives and assumed negatives only",
            "primary_scoring": "Hanson only, before last n=1 onset; no verified negatives",
            "within_shot_auroc": (
                "primary and broad Hanson masks, fixed score orientation; "
                "equal-shot mean/median over shots with both classes; "
                "per-shot class counts include excluded one-class shots"
            ),
            "paired_within_shot_auroc": (
                "forest minus each continuous scalar, primary and broad masks; "
                "equal-shot mean of per-shot AUROCs over shared two-class Hanson "
                "shots; 95% basic paired shot-bootstrap intervals, seed 0"
            ),
            "phase_controlled_auroc": (
                "primary Hanson slices; positive-negative pairs restricted to "
                f"campaign x floor(elapsed_ms/{ev.PHASE_BIN_MS:g}), "
                f"bins [{ev.PHASE_BIN_MS:g}*k,{ev.PHASE_BIN_MS:g}*(k+1)); "
                "pair-count-weighted cell AUROC; omit cells with fewer than "
                f"{ev.PHASE_MIN_SLICES} slices or one class and missing "
                "time; 1000 campaign-stratified shot resamples, seed 0; individual "
                "percentile and forest-minus-scalar basic paired intervals; "
                "within-copy or different-shot pairs only, no same-shot cross-copy pairs"
            ),
            "phase_bin_ms": ev.PHASE_BIN_MS,
            "phase_min_slices": ev.PHASE_MIN_SLICES,
            "phase_bin_width_sensitivity": (
                "50/100/200/400/800 ms and no time-bin control; primary Hanson "
                "mask, campaign control retained, same minimum cell size; "
                "point estimates only, no tuning or selection"
            ),
            "primary_mask_mechanism": (
                "primary negatives end at last n=1 onset; final labelled slices "
                "are positive, giving increasing elapsed time nearly perfect "
                "within-shot ranking"
            ),
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
            "primary_alarm_scope": "Hanson traces end at last n=1/n=2 onset +100 ms; comparison traces retain full span; physical state remains unassessed",
            "full_trace_alarm_sensitivity": "independently retune on full Hanson traces; same forest fits and inner folds",
            "shot_categories": (
                "first considered alarm decides: Detected if it warns any n=1 "
                "target by 10-400 ms; Early if unexplained and >400 ms before a "
                "future target; otherwise Missed; n=2-only No target; comparison "
                "FP means alarm incidence, not verified stable-shot FPR"
            ),
            "any_alarm_category_sensitivity": (
                "any target warning wins, otherwise any Early alarm, otherwise "
                "Missed; category sensitivity only, no refit or retuning; "
                "per-onset metrics retain all alarms"
            ),
            "alarm_grid": {
                "high_quantiles": ev.HIGH_QUANTILES,
                "low_fractions": ev.LOW_FRACTIONS,
                "hold_ms": ev.HOLD_MS,
            },
            "split": "shot-grouped; no frozen cohort overlap (shots.json/cohort_overlap)",
            "nnpu": "excluded: outer-fold-informed development and unidentified U prior",
            "rotation_claim": "removed; no isolated rotation benefit claimed",
            "interval_scope": "shot sampling at fixed fitted OOF predictions; split sensitivity separate",
            "multiplicity": (
                "exploratory intervals, unadjusted for multiple comparisons"
            ),
        },
        "configs": results,
        "paired": paired,
        "paired_method": "basic",
        "leave_one_run_record_out": leave_run_record_out,
        "split_sensitivity": split_summary(
            results["rwm-brf"],
            paired_time,
            paired_campaigns,
            phase_pairs,
            leave_run_record_out,
        )
        if paired_time
        else None,
        "legacy": legacy_record(),
        "screen_audit": screen,
        "onset_physics": physics,
        "forecast_label_audit": label_audit,
    }
    write_evaluation(clean(record), args.out, details_path)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
