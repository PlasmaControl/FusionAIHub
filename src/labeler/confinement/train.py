"""Validation-selected confinement models and a single frozen test evaluation."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from ..config import Paths
from .data import DIAGNOSTICS, _counts, feature_columns, physical_eligibility
from .labels import MODES, balanced_weights, digest

SEED = 20261001
# Same eight candidate configurations per eligible model/cohort; no frame split.
GRID = [
    {
        "learning_rate": lr,
        "max_leaf_nodes": leaves,
        "max_iter": 150,
        "l2_regularization": 1.0,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": SEED,
    }
    for lr in (0.05, 0.1)
    for leaves in (7, 15, 23, 31)
]


def validate_split(split, test_only):
    if split.shot.duplicated().any() or not set(split.split) <= {
        "train",
        "val",
        "test",
    }:
        raise ValueError("Split must have exactly one known assignment per shot")
    if set(split.loc[split.split != "test", "shot"]) & set(test_only):
        raise ValueError("Raw test-only shot found outside test")


def training_rows(frame, columns, *, multiclass=False):
    target = "regime_label" if multiclass else "label"
    used = frame.loc[(frame[target] >= 0) & physical_eligibility(frame, columns)]
    if not multiclass:
        weights = balanced_weights(used.label, used.shot)
    else:
        classes = np.unique(used[target])
        if len(classes) != 4:
            raise ValueError("Four-class learner requires every class in training")
        weights = np.zeros(len(used))
        for cls in classes:
            members, counts = np.unique(
                used.loc[used[target] == cls, "shot"], return_counts=True
            )
            for shot, count in zip(members, counts, strict=True):
                weights[(used[target] == cls) & (used.shot == shot)] = len(used) / (
                    4 * len(members) * count
                )
    return used, weights


def select_threshold(y, probability):
    """Maximize unweighted validation macro-F1; ties favor nearest 0.5."""
    y, probability = np.asarray(y), np.asarray(probability)
    unique = np.unique(probability)
    candidates = np.unique(
        np.r_[
            np.linspace(0.05, 0.95, 181),
            0.5,
            0.0,
            1.0,
            unique,
            (unique[:-1] + unique[1:]) / 2,
        ]
    )
    scores = []
    for t in candidates:
        pred = probability >= t
        tp, tn = np.sum(pred & (y == 1)), np.sum(~pred & (y == 0))
        fp, fn = np.sum(pred & (y == 0)), np.sum(~pred & (y == 1))
        denom_h, denom_l = 2 * tp + fp + fn, 2 * tn + fp + fn
        scores.append(
            (
                (2 * tp / denom_h if denom_h else 0.0)
                + (2 * tn / denom_l if denom_l else 0.0)
            )
            / 2
        )
    best = max(scores)
    return float(
        min(
            (
                t
                for t, s in zip(candidates, scores, strict=True)
                if abs(s - best) < 1e-12
            ),
            key=lambda t: (abs(t - 0.5), t),
        )
    )


def binary_metrics(y, probability, threshold):
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    per_class = {}
    for i, name in enumerate(("L", "H")):
        actual, predicted, correct = matrix[i].sum(), matrix[:, i].sum(), matrix[i, i]
        precision = float(correct / predicted) if predicted else None
        recall = float(correct / actual) if actual else None
        denom = actual + predicted
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": float(2 * correct / denom) if denom else None,
            "support": int(actual),
        }
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    both = set(np.unique(y)) == {0, 1}
    return {
        "support": {name: int((y == i).sum()) for i, name in enumerate(("L", "H"))},
        "confusion_matrix_LH": matrix.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()]))
        if both
        else None,
        "balanced_accuracy": float(np.mean(recalls)) if both else None,
        "auroc": float(roc_auc_score(y, probability)) if both else None,
        "auprc_H": float(average_precision_score(y, probability)) if both else None,
        "auprc_L": float(average_precision_score(1 - y, 1 - probability))
        if both
        else None,
    }


def without_bes(frame):
    frame = frame.copy()
    for col in frame:
        if col.startswith("bes__"):
            frame[col] = 0.0 if col.endswith("__present") else np.nan
    return frame


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _multiclass_metrics(y, probability):
    matrix = confusion_matrix(y, np.argmax(probability, axis=1), labels=np.arange(4))
    per_class = {}
    for i, name in enumerate(MODES):
        actual, predicted, correct = matrix[i].sum(), matrix[:, i].sum(), matrix[i, i]
        per_class[name] = {
            "support": int(actual),
            "precision": float(correct / predicted) if predicted else None,
            "recall": float(correct / actual) if actual else None,
            "f1": float(2 * correct / (actual + predicted))
            if actual + predicted
            else 0.0,
        }
    return {
        "confusion_matrix_L_H_QH_WP": matrix.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
        "balanced_accuracy": float(
            np.mean(
                [x["recall"] for x in per_class.values() if x["recall"] is not None]
            )
        ),
    }


def _bootstrap(frame, probabilities, thresholds, repetitions=2000):
    """Paired resampling of whole shots on a fixed common eligible cohort."""
    shots = np.unique(frame.shot)
    indices = [np.flatnonzero(frame.shot.to_numpy() == shot) for shot in shots]
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in probabilities}
    metric_values = {name: {} for name in probabilities}
    differences = {
        name: [] for name in probabilities if name != "full" and "full" in probabilities
    }
    metric_differences = {name: {} for name in differences}
    nondegenerate = 0
    for _ in range(repetitions):
        ix = np.concatenate(
            [indices[i] for i in rng.integers(len(shots), size=len(shots))]
        )
        y = frame.label.to_numpy()[ix]
        if len(np.unique(y)) != 2:
            continue
        nondegenerate += 1
        current = {}
        current_scalars = {}
        for name, probability in probabilities.items():
            metrics = binary_metrics(y, probability[ix], thresholds[name])
            score = metrics["macro_f1"]
            values[name].append(score)
            current[name] = score
            scalars = {
                key: metrics[key]
                for key in (
                    "macro_f1",
                    "balanced_accuracy",
                    "auroc",
                    "auprc_H",
                    "auprc_L",
                )
            }
            scalars.update(
                {
                    f"{metric}_{cls}": metrics["per_class"][cls][metric]
                    for cls in ("L", "H")
                    for metric in ("f1", "precision", "recall")
                }
            )
            current_scalars[name] = scalars
            for key, value in scalars.items():
                if value is not None:
                    metric_values[name].setdefault(key, []).append(value)
        for name, values_for_name in differences.items():
            values_for_name.append(current["full"] - current[name])
            for key, value in current_scalars["full"].items():
                baseline = current_scalars[name][key]
                if value is not None and baseline is not None:
                    metric_differences[name].setdefault(key, []).append(
                        value - baseline
                    )
    quantile = lambda x: (
        [float(z) for z in np.quantile(x, [0.025, 0.975])] if x else None
    )
    return {
        "unit": "shot",
        "requested_repetitions": repetitions,
        "nondegenerate_repetitions": nondegenerate,
        "shots": len(shots),
        "bins": len(frame),
        "macro_f1_95ci": {n: quantile(v) for n, v in values.items()},
        "metric_95ci": {
            n: {k: quantile(v) for k, v in metrics.items()}
            for n, metrics in metric_values.items()
        },
        "metric_valid_repetitions": {
            n: {k: len(v) for k, v in metrics.items()}
            for n, metrics in metric_values.items()
        },
        "full_minus_baseline_metric_95ci": {
            n: {k: quantile(v) for k, v in metrics.items()}
            for n, metrics in metric_differences.items()
        },
        "full_minus_baseline_macro_f1_95ci": {
            n: quantile(v) for n, v in differences.items()
        },
    }


def _multiclass_bootstrap(frame, probability, repetitions=2000):
    shots = np.unique(frame.shot)
    indices = [np.flatnonzero(frame.shot.to_numpy() == shot) for shot in shots]
    rng = np.random.default_rng(SEED)
    values = {}
    complete = 0
    for _ in range(repetitions):
        ix = np.concatenate(
            [indices[i] for i in rng.integers(len(shots), size=len(shots))]
        )
        y = frame.regime_label.to_numpy()[ix]
        if len(np.unique(y)) != 4:
            continue
        complete += 1
        metrics = _multiclass_metrics(y, probability[ix])
        scalars = {key: metrics[key] for key in ("macro_f1", "balanced_accuracy")}
        scalars.update(
            {
                f"{key}_{mode}": metrics["per_class"][mode][key]
                for mode in MODES
                for key in ("f1", "precision", "recall")
            }
        )
        for key, value in scalars.items():
            if value is not None:
                values.setdefault(key, []).append(value)
    return {
        "unit": "shot",
        "requested_repetitions": repetitions,
        "complete_class_repetitions": complete,
        "policy": "CI conditions on bootstrap draws containing all four reference classes; incomplete draws excluded and counted",
        "metric_95ci": {
            k: [float(x) for x in np.quantile(v, [0.025, 0.975])]
            for k, v in values.items()
        },
    }


def fit_models(frame, out, hashes):
    """No test row is consumed by fitting, thresholding or model selection."""
    out = Path(out)
    (out / "model").mkdir(parents=True, exist_ok=True)
    columns = feature_columns()
    definitions = {
        "full": columns,
        "no_bes": [c for c in columns if not c.startswith("bes__")],
        "dalpha_nbi": [c for c in columns if c.split("__")[0] in ("dalpha", "nbi")],
        "bes_only": [c for c in columns if c.startswith("bes__")],
        "four_class": columns,
    }
    train, val = frame[frame.split == "train"], frame[frame.split == "val"]
    records, models = {}, {}
    for name, cols in definitions.items():
        multiclass = name == "four_class"
        try:
            used, weights = training_rows(train, cols, multiclass=multiclass)
        except ValueError as exc:
            records[name] = {"unavailable": str(exc)}
            continue
        # sklearn 1.9.1 weighted binning rejects entirely NaN columns.
        # Prune using only eligible TRAIN rows; partially missing stays NaN.
        pruned = [
            c for c in cols if not np.isfinite(used[c].to_numpy(dtype=float)).any()
        ]
        cols = [c for c in cols if c not in pruned]
        target = "regime_label" if multiclass else "label"
        validation = val.loc[(val[target] >= 0) & physical_eligibility(val, cols)]
        if validation.empty or (not multiclass and validation.label.nunique() < 2):
            records[name] = {"unavailable": "Validation does not support selection"}
            continue
        best_score, chosen, candidates = -1.0, None, []
        for config in GRID:
            model = HistGradientBoostingClassifier(**config)
            model.fit(used[cols], used[target], sample_weight=weights)
            probability = model.predict_proba(validation[cols])
            threshold = (
                None
                if multiclass
                else select_threshold(validation.label.to_numpy(), probability[:, 1])
            )
            metrics = (
                _multiclass_metrics(validation[target], probability)
                if multiclass
                else binary_metrics(validation.label, probability[:, 1], threshold)
            )
            candidates.append(
                {
                    "config": config,
                    "threshold": threshold,
                    "validation_metrics": metrics,
                }
            )
            if metrics["macro_f1"] > best_score + 1e-12:
                best_score = metrics["macro_f1"]
                chosen = (model, config, threshold, metrics)
        model, config, threshold, metrics = chosen
        path = out / "model" / f"{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "features": cols,
                    "threshold": threshold,
                    "task": "four_class" if multiclass else "binary",
                },
                f,
            )
        weights_path = out / "model" / f"{name}_train_weights.csv"
        used[["shot", "t_start", "t_end", "label", "regime_label"]].assign(
            effective_weight=weights
        ).to_csv(weights_path, index=False)
        models[name] = {
            "model": model,
            "features": cols,
            "threshold": threshold,
            "task": "four_class" if multiclass else "binary",
        }
        records[name] = {
            "selected_config": config,
            "threshold": threshold,
            "validation_metrics": metrics,
            "candidates": candidates,
            "train_support": _counts(used),
            "validation_support": _counts(validation),
            "train_class_effective_mass": {
                str(i): float(weights[used[target] == i].sum())
                for i in np.unique(used[target])
            },
            "feature_schema": cols,
            "model_sha256": digest(path),
            "train_weights_sha256": digest(weights_path),
            "train_shots": sorted(int(x) for x in used.shot.unique()),
            "validation_shots": sorted(int(x) for x in validation.shot.unique()),
            "train_feature_finite_counts": {
                c: int(np.isfinite(used[c].to_numpy(dtype=float)).sum())
                for c in definitions[name]
            },
            "validation_feature_finite_counts": {
                c: int(np.isfinite(validation[c].to_numpy(dtype=float)).sum())
                for c in definitions[name]
            },
        }
        records[name]["train_all_missing_columns_pruned"] = pruned
        print(f"selected {name}: validation macro-F1 {best_score:.4f}", flush=True)
    frozen = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "test_opened": False,
        "selection_metric": "natural-prevalence validation macro-F1",
        "candidate_budget_each": len(GRID),
        "seed": SEED,
        "input_hashes": hashes,
        "compiled_pipeline_code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("data.py", "train.py", "apply.py", "labels.py")
        },
        "frozen_binary_bars": {
            "H1": "F1(H)>=0.95 AND F1(L)>=0.70",
            "H2": "lower95% paired-shot bootstrap CI of F1(H)-alwaysH >0",
            "policy": "Legacy manuscript bars unchanged; no automatic promotion; failed bars retained without test retuning",
        },
        "threshold_policy": "Validation only: evaluate fixed 0.05..0.95 grid plus 0/1/0.5 and unique validation probabilities/midpoints; maximize natural macro-F1, tie nearest0.5",
        "weights": "Recomputed after each model's diagnostic exclusions: equal class mass and equal shot mass within class, train only.",
        "feature_constraints": "No shot ID, absolute clock or annotator source features; native NaN; no all-physical-missing samples.",
        "models": records,
    }
    # Freeze is written and hashed BEFORE predict_proba sees test observations.
    _write(out / "training.json", frozen)
    return models, frozen


def evaluate(frame, models, out, repetitions=2000):
    out = Path(out)
    test = frame[frame.split == "test"].copy()
    output = {
        "freeze_sha256": digest(out / "training.json"),
        "test_opened_utc": datetime.now(UTC).isoformat(),
        "scope": "Exploratory imported dependent supervision; annotation-conditioned prevalence, not independent gold. No comparison with historical CNN cohorts.",
        "models": {},
        "paired_bootstrap": {},
    }
    for name, bundle in models.items():
        cols = bundle["features"]
        eligible = physical_eligibility(test, cols)
        if bundle["task"] == "four_class":
            eligible &= test.regime_label >= 0
        cohort = test.loc[eligible].copy()
        if bundle["task"] == "binary":
            test[f"{name}_probability_H"] = np.nan
            test[f"{name}_prediction"] = np.nan
        if cohort.empty:
            output["models"][name] = {
                "support": _counts(cohort),
                "metrics": None,
                "excluded_missing_support": _counts(test.loc[~eligible]),
                "unavailable": "No eligible test observation",
            }
            continue
        probability = bundle["model"].predict_proba(cohort[cols])
        if bundle["task"] == "four_class":
            cohort["four_class_prediction"] = np.argmax(probability, axis=1)
            for i, mode in enumerate(MODES):
                test.loc[eligible, f"four_class_probability_{mode}"] = probability[:, i]
            metrics = _multiclass_metrics(cohort.regime_label, probability)
            output["models"][name] = {
                "support": _counts(cohort),
                "metrics": metrics,
                "shot_bootstrap": _multiclass_bootstrap(
                    cohort, probability, repetitions
                ),
                "subtype_support": {
                    mode: {
                        "bins": int((cohort.regime_label == i).sum()),
                        "shots": int(
                            cohort.loc[cohort.regime_label == i, "shot"].nunique()
                        ),
                    }
                    for i, mode in enumerate(MODES)
                },
            }
            continue
        p = probability[:, 1]
        test.loc[eligible, f"{name}_probability_H"] = p
        test.loc[eligible, f"{name}_prediction"] = (p >= bundle["threshold"]).astype(
            int
        )
        cohort["probability_H"] = p
        metrics = binary_metrics(cohort.label, p, bundle["threshold"])
        subgroups = {
            "raw_test_only": cohort.reason == "raw_test_only",
            "seeded_test": cohort.reason != "raw_test_only",
            "BES_present": cohort.bes__present == 1,
            "BES_absent": cohort.bes__present == 0,
        }
        per_shot = {
            str(shot): {
                "support": _counts(g),
                "metrics": binary_metrics(
                    g.label, g.probability_H, bundle["threshold"]
                ),
            }
            for shot, g in cohort.groupby("shot")
        }
        patterns = (
            cohort[[f"{d}__present" for d in DIAGNOSTICS]]
            .astype(int)
            .astype(str)
            .agg("".join, axis=1)
        )
        missing_patterns = {
            pattern: {
                "support": _counts(cohort.loc[ix]),
                "metrics": binary_metrics(
                    cohort.loc[ix, "label"],
                    cohort.loc[ix, "probability_H"],
                    bundle["threshold"],
                ),
            }
            for pattern, ix in patterns.groupby(patterns).groups.items()
        }
        output["models"][name] = {
            "support": _counts(cohort),
            "excluded_missing_support": _counts(test.loc[~eligible]),
            "metrics": metrics,
            "all_H_baseline": binary_metrics(cohort.label, np.ones(len(cohort)), 0.5),
            "subgroups": {
                n: {
                    "support": _counts(cohort.loc[m]),
                    "metrics": binary_metrics(
                        cohort.loc[m, "label"],
                        cohort.loc[m, "probability_H"],
                        bundle["threshold"],
                    ),
                }
                for n, m in subgroups.items()
                if m.any()
            },
            "per_shot": per_shot,
            "missing_pattern_diagnostic_order": list(DIAGNOSTICS),
            "missing_patterns": missing_patterns,
        }
    if "full" in models:
        full = models["full"]
        forced = without_bes(test)
        mask = physical_eligibility(forced, full["features"])
        p = (
            full["model"].predict_proba(forced.loc[mask, full["features"]])[:, 1]
            if mask.any()
            else np.array([])
        )
        test.loc[mask, "full_forced_no_bes_probability_H"] = p
        output["models"]["full_forced_no_bes"] = {
            "support": _counts(test.loc[mask]),
            "metrics": binary_metrics(test.loc[mask, "label"], p, full["threshold"])
            if mask.any()
            else None,
        }
        all_full = test.full_probability_H.notna()
        cohort = test.loc[all_full]
        if len(cohort):
            output["paired_bootstrap"]["full_cohort"] = _bootstrap(
                cohort,
                {
                    "full": cohort.full_probability_H.to_numpy(),
                    "all_H": np.ones(len(cohort)),
                },
                {"full": full["threshold"], "all_H": 0.5},
                repetitions,
            )
        for name in ("no_bes", "dalpha_nbi", "bes_only", "full_forced_no_bes"):
            col = f"{name}_probability_H"
            if col not in test:
                continue
            cohort = test.loc[all_full & test[col].notna()]
            if cohort.empty:
                continue
            threshold = (
                full["threshold"]
                if name == "full_forced_no_bes"
                else models[name]["threshold"]
            )
            output["paired_bootstrap"][f"full_vs_{name}"] = _bootstrap(
                cohort,
                {
                    "full": cohort.full_probability_H.to_numpy(),
                    name: cohort[col].to_numpy(),
                },
                {"full": full["threshold"], name: threshold},
                repetitions,
            )
    keep = [c for c in test if "__" not in c and c != "train_weight"]
    test[keep].to_csv(out / "predictions.csv", index=False)
    output["predictions_sha256"] = digest(out / "predictions.csv")
    if "full" in output["models"] and output["models"]["full"]["metrics"]:
        cls = output["models"]["full"]["metrics"]["per_class"]
        h2_ci = output["paired_bootstrap"]["full_cohort"][
            "full_minus_baseline_metric_95ci"
        ]["all_H"]["f1_H"]
        output["legacy_binary_bars"] = {
            "H1": {
                "passed": cls["H"]["f1"] >= 0.95 and cls["L"]["f1"] >= 0.70,
                "F1_H": cls["H"]["f1"],
                "F1_L": cls["L"]["f1"],
            },
            "H2": {"passed": h2_ci[0] > 0.0, "F1_H_minus_alwaysH_95ci": h2_ci},
            "promotion": "No automatic promotion of existing default detector",
        }
    _write(out / "evaluation.json", output)
    return output


def run(features_dir, out, *, repetitions=2000):
    features_dir, out = Path(features_dir), Path(out)
    paths = Paths.from_env()
    for root in (
        paths.corpus,
        paths.raw_cache,
        paths.features,
        paths.label_tables / "confinement/raw",
    ):
        if out.resolve() == root.resolve() or root.resolve() in out.resolve().parents:
            raise ValueError("Outputs must be outside raw diagnostic stores")
    # Never silently fit another model after test opening.
    if (out / "evaluation.json").exists() or (out / "training.json").exists():
        raise ValueError(
            "Frozen training/evaluation already exists; do not retune on opened test"
        )
    metadata = json.loads((features_dir / "features.json").read_text())
    labels = json.loads((features_dir / "labels.json").read_text())
    split = pd.read_csv(features_dir / "split.csv")
    validate_split(split, set(labels["test_only_shots"]))
    frame = pd.read_csv(features_dir / "features/all.csv")
    actual = frame[["shot", "split", "reason"]].drop_duplicates()
    expected = split.merge(
        actual, on="shot", suffixes=("_expected", "_actual"), validate="one_to_one"
    )
    if (
        not (expected.split_expected == expected.split_actual).all()
        or not (expected.reason_expected == expected.reason_actual).all()
    ):
        raise ValueError("Features do not preserve frozen split")
    if digest(features_dir / "features/all.csv") != metadata["feature_sha256"]:
        raise ValueError("Feature hash changed since coverage audit")
    hashes = {
        "features.json": digest(features_dir / "features.json"),
        "features/all.csv": metadata["feature_sha256"],
        "train.py": digest(Path(__file__)),
        **metadata["input_hashes"],
    }
    for name, expected_hash in metadata["input_hashes"].items():
        if digest(features_dir / name) != expected_hash:
            raise ValueError(f"Frozen input hash changed: {name}")
    models, _frozen = fit_models(frame, out, hashes)
    return evaluate(frame, models, out, repetitions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features", type=Path, default=Paths.from_env().root / "confinement/v1"
    )
    parser.add_argument(
        "--out", type=Path, default=Paths.from_env().root / "confinement/v1"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    result = run(args.features, args.out, repetitions=args.bootstrap)
    print(
        json.dumps({n: r.get("metrics") for n, r in result["models"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
