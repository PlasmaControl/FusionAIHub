#!/usr/bin/env python
"""Audit saved development predictions and export the tearing benchmark for the paper.

No fitting, fetching or blind-test scoring. Recompute the four out-of-fold rows
from their saved probabilities and current labels, reconstruct the shot splits,
and bundle source JSON records beside a summary. The table and Figure 2 JSON use
those records. Legacy temporal coverage counts observed native label samples;
interval coverage counts observable 10 ms bins, including quiet bins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import tm_ours
import tm_prior_published
import tm_prior_retrain

from labeler.config import Paths, git_sha
from labeler.tearing import scoring
from labeler.validate import archived_truth

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
TM = ROOT / "round4/tm"
LOCAL = REPO / "data/events/neoclassical_tearing_mode/benchmark"
DOCS = REPO / "docs/labeler"
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)


def read(path):
    return json.loads(path.read_text())


def write(path, record):
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")


def threshold_provenance(record):
    """Identify prespecified validation folds without an estimable F1 optimum."""
    unestimable = []
    for info in record.get("fold_info", []):
        positive = info["validation_bins_positive"]
        info["threshold_estimable"] = bool(positive)
        info["threshold_status"] = (
            "inner-validation F1 optimum"
            if positive
            else "no positive validation bins; deterministic highest-edge fallback, "
            "not an estimable F1 optimum"
        )
        if not positive:
            unestimable.append(
                {
                    "fold": info["fold"],
                    "positive_bins": positive,
                    "negative_bins": info["validation_bins_negative"],
                    "threshold": info["threshold"],
                }
            )
    record["threshold_estimation"] = {
        "unestimable_folds": unestimable,
        "zero_positive_convention": (
            "highest candidate histogram lower edge, from the deterministic "
            "all-zero-F1 tie convention; no evidence-supported F1 optimum"
        ),
        "f1_limitation": (
            "headline F1 is sensitive to absent-positive inner folds; prespecified "
            "shared splits and choices retained, without selecting on held scores"
            if unestimable
            else None
        ),
    }
    return unestimable


def rule_audit_bundle():
    """Commit source records; keep interval-level audit details out of summary."""
    records, sources = {}, {}
    for filename in (
        "calibration_dev_fix1.json",
        "audit_fix1_current.json",
        "audit_before_fix1.json",
        "weak_diagnostics_fix1.json",
    ):
        record = read(TM / "labels" / filename)
        destination = LOCAL / "sources" / filename
        write(destination, record)
        if destination.stat().st_size > 2_000_000:
            destination.write_text(json.dumps(record, separators=(",", ":")) + "\n")
        assert destination.stat().st_size <= 2_000_000, destination
        records[filename] = record
        sources[filename] = str(destination.relative_to(REPO))
    calibration = records["calibration_dev_fix1.json"]
    current = records["audit_fix1_current.json"]
    result = {
        "sources": sources,
        "calibration": {
            k: calibration[k]
            for k in (
                "split",
                "n_development_shots",
                "n_reference_shots",
                "n_reference_bins",
                "selected_quantile",
                "harmonic_ratio",
                "disclosure",
            )
        },
        "source_sha256": current.get("source_sha256"),
        "label_sha256": current.get("label_sha256"),
        "reviewed_shots": current.get("reviewed_shots"),
        "weak_diagnostics_source": sources["weak_diagnostics_fix1.json"],
    }
    for scope in ("cohort", "population"):
        result[scope] = {
            "changes": {
                k: v for k, v in current[scope]["changes"].items() if k.startswith("n_")
            },
            **{
                label: {
                    k: v for k, v in current[scope][label].items() if k != "details"
                }
                for label in ("before_seed_audit", "after_seed_audit")
            },
        }
    return result


def historical_cnn_sources():
    """Keep the superseded ranking evidence behind the earlier-round caveat."""
    result = {}
    for label, filename in (
        ("retrained", "tm_prior_retrained_tm-onsetcnn_cv.json"),
        ("published", "tm_prior_published_tm-onsetcnn_tokamak-si_dev.json"),
    ):
        destination = LOCAL / "sources" / f"historical_before_fix1_{filename}"
        if not destination.is_file():
            original = read(LOCAL / "sources" / filename)
            assert "cv_cohort_shots" not in original, filename
            original["historical_scope"] = (
                "Superseded target and protocol before fix round 1; retained only "
                "to document earlier CNN ranking, not a current benchmark row."
            )
            write(destination, original)
        record = read(destination)
        result[label] = {
            "source": str(destination.relative_to(REPO)),
            "auroc": record["metrics"]["auroc"],
            "auprc": record["metrics"]["auprc"],
            "shots": record["metrics"]["n_shots"],
        }
    return result


def rescore_oof(stem, cohort):
    """Current-label metrics and explicit train/validation/held shot lists."""
    record = read(TM / "results" / f"{stem}.json")
    assert record["labels_sha256"] == hashlib.sha256(LABELS.read_bytes()).hexdigest()
    dev = sorted(record["dev_shots"])
    blind = set(cohort.query("split == 'test'").shot)
    assert not set(dev) & blind, stem
    if "prior" in stem:
        kind = "cnn" if "onsetcnn" in stem else "dsm"
        data, _ = tm_prior_retrain.load(kind, dev)
        y = {s: data[s]["y_bins"] for s in dev}
        valid = {s: data[s]["valid_bins"] for s in dev}
        oof_name = stem.removesuffix("_cv").replace(
            "tm_prior_retrained", "oof_tm_prior_retrained"
        )
    else:
        data, _ = tm_ours.load(dev, "magnetics_rms" in stem)
        y, valid = tm_ours.truth_of(data, dev)
        oof_name = "oof_" + stem.removesuffix("_cv")
    with np.load(TM / "results" / f"{oof_name}.npz") as z:
        assert set(z.files) == {f"s{s}" for s in dev}
        scores = {s: z[f"s{s}"] for s in dev}
    cohort_dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    folds = {int(s): int(k) for s, k in record["folds"].items()}
    shared_folds, shared_splits = scoring.shared_cv(cohort_dev)
    assert folds == shared_folds
    thresholds, splits = {}, []
    for info in record["fold_info"]:
        k = info["fold"]
        split = shared_splits[k]
        held = [s for s in split["held"] if s in data]
        train = [s for s in split["train"] if s in data]
        val = [s for s in split["validation"] if s in data]
        assert info["train"] == train and info["validation"] == val
        assert info["held"] == held
        assert not set(held) & (set(train) | set(val))
        assert not set(train) & set(val)
        assert not (set(train) | set(val) | set(held)) & blind
        with np.load(info["validation_scores"]) as z:
            assert set(z.files) == {f"s{s}" for s in val}
            validation_scores = {s: z[f"s{s}"] for s in val}
        validation_stats = [
            scoring.shot_stats(
                s, y[s], valid[s], validation_scores[s], tm_ours.THRESHOLD_EDGES
            )
            for s in val
        ]
        reproduced_threshold = scoring.best_threshold(
            validation_stats, tm_ours.THRESHOLD_EDGES
        )
        assert reproduced_threshold == info["threshold"], (stem, k)
        for key, attribute in (
            ("validation_bins_scored", "n_bins"),
            ("validation_bins_positive", "n_pos"),
            ("validation_bins_negative", "n_neg"),
        ):
            assert info[key] == sum(getattr(s, attribute) for s in validation_stats)
        thresholds.update({s: info["threshold"] for s in held})
        splits.append(
            {
                **split,
                "available_train": train,
                "available_validation": val,
                "available_held": held,
            }
        )
    metrics = scoring.evaluate(dev, y, valid, scores, thresholds, n=1000, seed=0)
    delta = {
        k: metrics[k]["value"] - record["metrics"][k]["value"]
        for k in ("auroc", "auprc", "f1", "segf1_0.5")
    }
    # Saved probabilities are float32; the original inference sometimes used float64.
    assert max(abs(v) for v in delta.values()) < 1e-4, (stem, delta)
    assert metrics["bins_scored"] == record["metrics"]["bins_scored"]
    record.update(
        metrics=metrics,
        rescore_git_sha=git_sha(),
        rescore_from=str(TM / "results" / f"{oof_name}.npz"),
        rescore_delta=delta,
        explicit_splits=splits,
        blind_test_excluded=True,
        validation_thresholds_reproduced=True,
        labels_sha256=hashlib.sha256(LABELS.read_bytes()).hexdigest(),
    )
    threshold_provenance(record)
    write(TM / "results" / f"{stem}.json", record)
    print(stem, "verified", metrics["n_shots"], "shots", delta, flush=True)


def coverage(cohort):
    out = {
        "scope": (
            "fixed cohort, all splits for coverage only; common 10 ms catalog-window "
            "grid and observable plasma domain for both legacy and interval labels"
        )
    }
    interval_masks, observable_masks, grids = {}, {}, {}
    for name, path in (
        ("ours", LABELS),
        ("ours_population", TM / "labels/tm_interval_population.csv"),
    ):
        table = pd.read_csv(path)
        catalog = "population" if "population" in name else "cohort"
        start_path = TM / f"labels/plasma_start_{catalog}.json"
        starts = read(start_path)
        windows = pd.read_csv(REPO / f"data/events/catalog/{catalog}.csv")
        by_shot = {int(s): g for s, g in table.groupby("shot")}
        bins = positive = observable = classifiable = plasma_bins = 0
        for row in windows.itertuples():
            if row.shot not in by_shot:
                continue
            centres = scoring.bin_centres((row.window_start_ms, row.window_end_ms))
            plasma_window = (centres >= starts[str(int(row.shot))]["start_ms"]) & (
                centres < row.window_end_ms
            )
            y, valid = scoring.label_bins(by_shot[row.shot], centres)
            valid &= plasma_window
            # Uncertain mode evidence remains observable plasma, but unlabelled.
            unavailable = by_shot[row.shot].query("category == 3")
            _, plasma = scoring.label_bins(unavailable, centres)
            plasma &= plasma_window
            plasma_bins += int(plasma.sum())
            if name == "ours":
                grids[int(row.shot)] = centres
                interval_masks[int(row.shot)] = valid
                observable_masks[int(row.shot)] = plasma
            bins += int(valid.sum())
            positive += int((y.astype(bool) & valid).sum())
            classifiable += int(valid.any())
            observable += int(plasma.any())
        out[name] = {
            "labelled_shots": len(by_shot),
            "classifiable_shots": classifiable,
            "observable_shots": observable,
            "labelled_seconds": bins * 0.01,
            "present_seconds": positive * 0.01,
            "observable_plasma_seconds": plasma_bins * 0.01,
            "definition": (
                "labelled 10 ms bins from the measured plasma start to catalog "
                "window end; uncertain and unobservable bins excluded"
            ),
            "observable_definition": (
                "10 ms plasma bins from the measured plasma start to catalog "
                "window end; uncertain included, unavailable excluded"
            ),
            "plasma_start_source": str(start_path),
            "source": str(path),
        }
    paths = replace(Paths.from_env(), root=TM / "lroot")
    records, skipped = {}, {}
    for shot in cohort.shot:
        truth = archived_truth(int(shot), paths)
        if truth.get("available"):
            s = int(shot)
            centres = grids[s]
            support = (
                np.isfinite(
                    scoring.align_scores(
                        truth["t"] * 1000,
                        np.ones(len(truth["t"])),
                        centres,
                        max_gap_ms=25.0,
                    )
                )
                & observable_masks[s]
            )
            positive = (
                scoring.align_scores(
                    truth["t"] * 1000,
                    truth["tm_label"].astype(float),
                    centres,
                    max_gap_ms=25.0,
                )
                >= 0.5
            )
            records[s] = {
                "labelled_bins": int(support.sum()),
                "positive_bins": int((positive & support).sum()),
                "observable_plasma_bins": int(observable_masks[s].sum()),
                "interval_labelled_bins": int(interval_masks[s].sum()),
                "common_labelled_bins": int((support & interval_masks[s]).sum()),
            }
        else:
            skipped[int(shot)] = truth.get("reason", "unavailable")
    out["legacy_seo"] = {
        "labelled_shots": len(records),
        "labelled_seconds": sum(v["labelled_bins"] for v in records.values()) * 0.01,
        "positive_seconds": sum(v["positive_bins"] for v in records.values()) * 0.01,
        "observable_plasma_seconds": sum(
            v["observable_plasma_bins"] for v in records.values()
        )
        * 0.01,
        "interval_seconds_on_same_shots": sum(
            v["interval_labelled_bins"] for v in records.values()
        )
        * 0.01,
        "common_labelled_seconds": sum(
            v["common_labelled_bins"] for v in records.values()
        )
        * 0.01,
        "definition": (
            "matched growth-phase label support on observable catalog-window "
            "10 ms bins; temporal holes excluded"
        ),
        "per_shot": records,
        "skipped": skipped,
    }
    path = REPO / "data/events/neoclassical_tearing_mode/raw/tm_labels.h5"
    if not path.is_file():
        path = (
            Path("/scratch/gpfs/nc1514/FusionAIHub/data/events")
            / "neoclassical_tearing_mode/raw/tm_labels.h5"
        )
    records = {}
    with h5py.File(path) as f:
        for shot in cohort.shot:
            if str(shot) not in f:
                continue
            s = int(shot)
            t = f[str(shot)]["time"][:]
            label = f[str(shot)]["label"][:]
            support = (
                np.isfinite(
                    scoring.align_scores(
                        t,
                        np.where(np.isfinite(label), 1.0, np.nan),
                        grids[s],
                        max_gap_ms=20.0,
                    )
                )
                & observable_masks[s]
            )
            records[s] = {
                "labelled_bins": int(support.sum()),
                "observable_plasma_bins": int(observable_masks[s].sum()),
                "interval_labelled_bins": int(interval_masks[s].sum()),
                "common_labelled_bins": int((support & interval_masks[s]).sum()),
            }
    out["legacy_survival"] = {
        "labelled_shots": len(records),
        "labelled_seconds": sum(v["labelled_bins"] for v in records.values()) * 0.01,
        "observable_plasma_seconds": sum(
            v["observable_plasma_bins"] for v in records.values()
        )
        * 0.01,
        "interval_seconds_on_same_shots": sum(
            v["interval_labelled_bins"] for v in records.values()
        )
        * 0.01,
        "common_labelled_seconds": sum(
            v["common_labelled_bins"] for v in records.values()
        )
        * 0.01,
        "definition": (
            "survival-label support on observable catalog-window 10 ms bins; "
            "fixed-grid time outside observable plasma excluded"
        ),
        "per_shot": records,
        "source": str(path),
    }
    return out


def paired_common_shots(cohort):
    """Both models on exactly the same shots and scored 10 ms bins; paired CIs."""
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    ours, _ = tm_ours.load(dev, False)
    cnn, _ = tm_prior_retrain.load("cnn", dev)
    shots = sorted(set(ours) & set(cnn))
    models = {
        "tm-ours": ("tm_ours_magnetics_cv", "oof_tm_ours_magnetics"),
        "tm-onsetcnn-retrained": (
            "tm_prior_retrained_tm-onsetcnn_cv",
            "oof_tm_prior_retrained_tm-onsetcnn",
        ),
    }
    scores, thresholds = {}, {}
    for model, (stem, probability) in models.items():
        record = read(TM / "results" / f"{stem}.json")
        fold_thresholds = {i["fold"]: i["threshold"] for i in record["fold_info"]}
        thresholds[model] = {s: fold_thresholds[record["folds"][str(s)]] for s in shots}
        with np.load(TM / "results" / f"{probability}.npz") as z:
            scores[model] = {s: z[f"s{s}"] for s in shots}
    y, valid = {}, {}
    for s in shots:
        assert np.array_equal(ours[s][0], cnn[s]["centres"]), s
        assert np.array_equal(ours[s][2], cnn[s]["y_bins"]), s
        y[s] = ours[s][2]
        valid[s] = (
            ours[s][3]
            & cnn[s]["valid_bins"]
            & np.isfinite(scores["tm-ours"][s])
            & np.isfinite(scores["tm-onsetcnn-retrained"][s])
        )
    shots = [s for s in shots if valid[s].any()]
    results, stats = {}, {}
    for model in models:
        edges = scoring.edges_for(
            np.concatenate([scores[model][s][valid[s]] for s in shots])
        )
        results[model] = scoring.evaluate(
            shots,
            y,
            valid,
            scores[model],
            thresholds[model],
            edges=edges,
            n=1000,
            seed=0,
        )
        stats[model] = [
            scoring.shot_stats(
                s, y[s], valid[s], scores[model][s], edges, thresholds[model][s]
            )
            for s in shots
        ]
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(1000):
        weights = np.bincount(
            rng.integers(0, len(shots), len(shots)), minlength=len(shots)
        )
        a = scoring.metrics(stats["tm-ours"], weights)
        b = scoring.metrics(stats["tm-onsetcnn-retrained"], weights)
        draws.append({k: a[k] - b[k] for k in ("auroc", "auprc", "f1", "segf1_0.5")})
    differences = {}
    for metric in draws[0]:
        samples = np.array([d[metric] for d in draws])
        samples = samples[np.isfinite(samples)]
        differences[metric] = {
            "value": results["tm-ours"][metric]["value"]
            - results["tm-onsetcnn-retrained"][metric]["value"],
            "lo": float(np.percentile(samples, 2.5)),
            "hi": float(np.percentile(samples, 97.5)),
        }
    return {
        "shots": shots,
        "bins_scored": results["tm-ours"]["bins_scored"],
        "scored_seconds": results["tm-ours"]["scored_seconds"],
        "metrics": results,
        "difference_ours_minus_cnn": differences,
        "policy": (
            "same shots, same observable label and score bins; original fold "
            "thresholds; paired 1000-draw shot bootstrap"
        ),
    }


def cnn_ranking_comparison(cohort):
    """Published and retrained CNN rankings on identical eligible shots and bins."""
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    data, _ = tm_prior_retrain.load("cnn", dev)
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}
    published, y, valid, used, _ = tm_prior_published.si_bins(
        tm_prior_published.CNN, None, 25.0, dev, cohort, by_shot, table
    )
    record = read(TM / "results/tm_prior_retrained_tm-onsetcnn_cv.json")
    original = read(TM / "results/tm_prior_published_tm-onsetcnn_tokamak-si_dev.json")
    with np.load(TM / "results/oof_tm_prior_retrained_tm-onsetcnn.npz") as z:
        retrained = {s: z[f"s{s}"] for s in used if s in data and f"s{s}" in z}
    shots = sorted(retrained)
    for s in shots:
        valid[s] &= (
            data[s]["valid_bins"]
            & np.isfinite(retrained[s])
            & np.isfinite(published[s])
        )
    shots = [s for s in shots if valid[s].any()]
    results = {}
    for model, source, scores in (
        ("published", original, published),
        ("retrained", record, retrained),
    ):
        levels = {i["fold"]: i["threshold"] for i in source["fold_info"]}
        thresholds = {s: levels[source["folds"][str(s)]] for s in shots}
        results[model] = scoring.evaluate(
            shots, y, valid, scores, thresholds, n=1000, seed=0
        )
    return {
        "shots": shots,
        "bins_scored": results["published"]["bins_scored"],
        "metrics": results,
        "retrained_exceeds_published": {
            k: results["retrained"][k]["value"] > results["published"][k]["value"]
            for k in ("auroc", "auprc")
        },
        "note": (
            "Same observable bins; CNN published-training overlap remains unknown. "
            "Earlier rule ranking did not improve after retraining; repaired-target "
            "ranking must be assessed from these regenerated measurements."
        ),
    }


def display(metric):
    if metric is None:
        return "--"
    values = [f"{metric[k]:.3f}".removeprefix("0") for k in ("value", "lo", "hi")]
    return f"{values[0]} [{values[1]},{values[2]}]"


def table_tex(rows, caption, label):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrllll}",
        r"\toprule",
        (
            r"Model & Setting & Shots & AUROC [CI] & AUPRC [CI] & F1 [CI] & "
            r"Segmental F1 [CI] \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        metrics = row["metrics"]
        name = row["model"] + (r"$\dagger$" if not row.get("held_out", True) else "")
        lines.append(
            " & ".join(
                [
                    name,
                    row["setting"],
                    str(metrics["n_shots"]),
                    *[
                        display(metrics.get(k))
                        for k in ("auroc", "auprc", "f1", "segf1_0.5")
                    ],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "\\caption{" + caption + "}",
            "\\label{" + label + "}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--gallery-reviewed", action="store_true")
    args = ap.parse_args(argv)
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    blind = set(cohort.query("split == 'test'").shot)
    historical = historical_cnn_sources()
    if args.rescore:
        for stem in (
            "tm_prior_retrained_tm-onsetcnn_cv",
            "tm_prior_retrained_tm-dsm_cv",
            "tm_ours_magnetics_cv",
            "tm_ours_magnetics_rms_cv",
        ):
            rescore_oof(stem, cohort)
    rows, appendix = [], []

    def add(
        stem,
        model,
        setting,
        *,
        held_out=True,
        metric_key="metrics",
        note="",
        extra=False,
    ):
        source = TM / "results" / f"{stem}.json"
        record = read(source)
        shots = record.get("shots", record.get("dev_shots", []))
        if metric_key.startswith("metrics_outside_training"):
            trained = set(record["shots_in_training"])
            shots = [s for s in shots if s not in trained]
            selection = "fixed_published" if ":fixed" in metric_key else "cv_tuned"
            metrics = record["metrics_outside_training"][selection]
        else:
            metrics = record[metric_key]
        assert not set(shots) & blind, stem
        assert metrics["replicates"] == 1000, stem
        empty = set(metrics["shots_without_a_scored_bin"])
        scored = [s for s in shots if s not in empty]
        assert len(scored) == metrics["n_shots"], stem
        snapshot = LOCAL / "sources" / source.name
        unestimable = threshold_provenance(record)
        write(source, record)
        write(snapshot, record)
        fixed_threshold = "fixed" in metric_key or "Fixed" in setting
        (appendix if extra else rows).append(
            {
                "model": model,
                "setting": setting,
                "source": str(snapshot.relative_to(REPO)),
                "original_source": str(source),
                "metrics": metrics,
                "shots": scored,
                "requested_shots": shots,
                "held_out": held_out,
                "note": note,
                "threshold_unestimable_folds": [] if fixed_threshold else unestimable,
                "threshold": (
                    record.get("fixed_published_threshold", record.get("threshold_g"))
                    if fixed_threshold
                    else record.get("threshold", record.get("threshold_g"))
                ),
                "thresholds_by_fold": [
                    i["threshold"]
                    for i in record.get("fold_info", [])
                    if not fixed_threshold
                ],
            }
        )

    add(
        "tm_prior_published_tm-onsetcnn_legacy_cohort",
        "Onset CNN",
        "Legacy",
        held_out=False,
        note=(
            "Training overlap unknown: archive is the training store; "
            "descriptive score only."
        ),
    )
    for h in ("250ms", "500ms", "1s"):
        label = {"250ms": "250 ms", "500ms": "500 ms", "1s": "1 s"}[h]
        add(
            f"tm_prior_published_tm-dsm-{h}_legacy",
            f"Survival MLP ({label})",
            "Legacy",
            note="Outside published training list; cohort blind test excluded.",
            extra=h != "500ms",
        )
    add(
        "tm_prior_published_tm-onsetcnn_tokamak-si_dev",
        "Onset CNN",
        "Interval, published",
        held_out=False,
        note="Published weights; original training overlap unknown.",
    )
    for h in ("250ms", "500ms", "1s"):
        label = {"250ms": "250 ms", "500ms": "500 ms", "1s": "1 s"}[h]
        add(
            f"tm_prior_published_tm-dsm-{h}_tokamak-si_dev",
            f"Survival MLP ({label})",
            "Interval, published",
            metric_key="metrics_outside_training",
            note="Outside published training list; fold-tuned risk threshold.",
            extra=h != "500ms",
        )
    add("tm_prior_retrained_tm-onsetcnn_cv", "Onset CNN", "Tokamak-SI")
    add("tm_prior_retrained_tm-dsm_cv", "Survival MLP", "Tokamak-SI")
    add("tm_ours_magnetics_cv", "Magnetic detector", "Tokamak-SI")
    add("tm_baseline_n1rms_cv_tuned_dev", "RMS threshold", "Tokamak-SI")
    add(
        "tm_baseline_n1rms_12g_dev",
        "RMS threshold (12 G)",
        "Fixed threshold extra",
        extra=True,
    )
    add(
        "tm_ours_magnetics_rms_cv",
        "Magnetic detector + RMS",
        "Tokamak-SI",
        note="Circular input ablation: RMS defines the label.",
    )
    fixed_rows = [
        ("tm_prior_published_tm-onsetcnn_legacy_cohort", "Onset CNN, legacy"),
        ("tm_prior_published_tm-onsetcnn_tokamak-si_dev", "Onset CNN, interval"),
    ]
    for horizon, label in (("250ms", "250 ms"), ("500ms", "500 ms"), ("1s", "1 s")):
        fixed_rows.extend(
            [
                (
                    f"tm_prior_published_tm-dsm-{horizon}_legacy",
                    f"Survival MLP ({label}), legacy",
                ),
                (
                    f"tm_prior_published_tm-dsm-{horizon}_tokamak-si_dev",
                    f"Survival MLP ({label}), interval",
                ),
            ]
        )
    for stem, model in fixed_rows:
        add(
            stem,
            model,
            "Fixed published threshold extra",
            metric_key=(
                "metrics_outside_training:fixed"
                if "dsm" in stem and "tokamak" in stem
                else "metrics_fixed_published"
            ),
            held_out="Onset" not in model,
            extra=True,
            note="DSM survival threshold 0.7 corresponds to risk threshold 0.3.",
        )
    agreements = {}
    for ref in ("seo", "survival"):
        for split in ("dev", "all"):
            name = f"agreement_{ref}_cohort_{split}.json"
            agreements[f"{ref}_{split}"] = read(TM / "agreement" / name)
            write(LOCAL / "sources" / name, agreements[f"{ref}_{split}"])
    sensitivity = "sensitivity_survival_dev.json"
    agreements["survival_sensitivity_dev"] = read(TM / "agreement" / sensitivity)
    write(LOCAL / "sources" / sensitivity, agreements["survival_sensitivity_dev"])
    # Supplementary population CNN legacy row, never represented as held out.
    source = TM / "results/tm_prior_published_tm-onsetcnn_legacy_population.json"
    write(LOCAL / "sources" / source.name, read(source))
    cov = coverage(cohort)
    paired = paired_common_shots(cohort)
    cnn_ranking = cnn_ranking_comparison(cohort)
    audit = rule_audit_bundle()
    locking = {}
    frequencies = {int(p.stem) for p in (TM / "signals_freq").glob("*.npz")}
    for split in ("cohort", "population"):
        full = TM / f"labels/tm_intervals_full_{split}.csv"
        intervals = pd.read_csv(full)
        mode_shots = set(intervals.shot)
        known = intervals.locked_known.fillna(False).isin((True, "True", "true"))
        locking[split] = {
            "shots_with_intervals": len(mode_shots),
            "with_frequency": len(mode_shots & frequencies),
            "without_frequency": sorted(int(s) for s in mode_shots - frequencies),
            "intervals_locked_known": int(known.sum()),
            "intervals_locked_unknown": int((~known).sum()),
            "shots_with_unknown_locking_intervals": sorted(
                int(s) for s in intervals.loc[~known, "shot"].unique()
            ),
            "confirmed_locked_intervals": int(
                intervals.locked.fillna(False).isin((True, "True", "true")).sum()
            ),
            "source": str(full),
            "meaning": (
                "Confirmed locking requires frequency drop plus locked-mode evidence; "
                "interval ends at that time. Missing frequency means unknown locking "
                "status."
            ),
        }
    summary = {
        "made_by": "scripts/labeler/tm_benchmark.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_sha": git_sha(),
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "evaluation": (
            "development only; all main F1 thresholds tuned inside the same cohort-450 "
            "CV inner validation shot lists; fixed published thresholds explicitly "
            "separated as extras"
        ),
        "bootstrap": {
            "unit": "shot",
            "replicates": 1000,
            "confidence": 0.95,
            "seed": 0,
        },
        "legacy_segmental_f1": (
            "not applicable: legacy targets are growth-phase rows or forecasts, "
            "not present spans"
        ),
        "rows": rows,
        "appendix_rows": appendix,
        "paired_common_shots": paired,
        "cnn_ranking_common_bins": cnn_ranking,
        "historical_cnn_ranking": historical,
        "rule_audit": audit,
        "agreement": agreements,
        "coverage": cov,
        "label_counts": {
            "cohort": read(LABELS.with_suffix(".meta.json")),
            "population": read(TM / "labels/tm_interval_population.meta.json"),
        },
        "locking_coverage": locking,
        "gallery": {
            d: read(TM / f"figures/tm_gallery_{d}.json") for d in ("mhr", "mirnov")
        },
        "gallery_review": {
            "reviewed": args.gallery_reviewed,
            "rule_changed": True,
            "rule_sha256": hashlib.sha256(
                (REPO / "src/labeler/tearing/rule.py").read_bytes()
            ).hexdigest(),
            "finding": (
                "Gallery uses live MHR row 2 and the Mirnov array. Continuous seed "
                "and sub-30 kHz coherent-line evidence define strong rotating n=1/n=2 "
                "modes; unsupported candidates and sustained sub-seed lines are "
                "uncertain."
            ),
        },
        "blind_test": {
            "shots": sorted(int(s) for s in blind),
            "reporting": "excluded from benchmark, model selection and new scoring",
            "historical_exposure": (
                "Previous implementer wrote *_test.json records with --final. "
                "They remain external and are excluded from this paper export."
            ),
        },
    }
    write(LOCAL / "tm_benchmark.json", summary)
    write(TM / "results/tm_benchmark.json", summary)
    caption = (
        "Strong rotating n=1/n=2 mode (tearing-mode proxy) benchmarks. F1 thresholds "
        "follow shared "
        "development CV inner-validation splits. Zero-positive validation folds use "
        "a deterministic highest-edge fallback, making F1 sensitive to calibration "
        "coverage. Brackets give 95\\% shot-bootstrap intervals (1,000 draws). "
        "Segmental F1 uses IoU 0.5 and hard barriers at unavailable bins. "
        "Legacy targets "
        "differ; a dagger marks unknown training overlap. RMS inputs define the label. "
        "DSM uses 500 ms; other horizons and fixed-threshold extras are in the "
        "appendix. Blind tests are excluded."
    )
    assert len(caption.split()) <= 80
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrllll}",
        r"\toprule",
        (
            r"Model & Setting & Shots & AUROC [CI] & AUPRC [CI] & F1 [CI] & "
            r"Segmental F1 [CI] \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        m = row["metrics"]
        name = row["model"] + (r"$\dagger$" if not row["held_out"] else "")
        lines.append(
            " & ".join(
                [
                    name,
                    row["setting"],
                    str(m["n_shots"]),
                    *[display(m.get(k)) for k in ("auroc", "auprc", "f1", "segf1_0.5")],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "\\caption{" + caption + "}",
            r"\label{tab:tm-benchmark}",
            r"\end{table*}",
        ]
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "table_tm_benchmark.tex").write_text("\n".join(lines) + "\n")
    (DOCS / "table_tm_benchmark_appendix.tex").write_text(
        table_tex(
            appendix,
            "Additional DSM horizons and explicitly fixed-threshold results. Fixed DSM "
            "alarms use risk 0.3, equivalent to upstream survival probability 0.7. "
            "Other rows use shared-fold validation thresholds. Brackets give 95\\% "
            "shot-bootstrap intervals; blind cohort tests are excluded.",
            "tab:tm-benchmark-appendix",
        )
    )
    paired_rows = [
        {"model": model, "setting": "Common shots and bins", "metrics": metrics}
        for model, metrics in paired["metrics"].items()
    ]
    (DOCS / "table_tm_paired.tex").write_text(
        table_tex(
            paired_rows,
            "Paired comparison on identical development shots and available 10 ms "
            "bins. Thresholds retain their shared-fold validation values. "
            "Unavailable bins are "
            "hard boundaries for segmental IoU 0.5; brackets give 95\\% shot-bootstrap "
            "intervals (1,000 draws).",
            "tab:tm-paired",
        )
    )
    figure_rows = []
    for model in (
        "Onset CNN",
        "Survival MLP",
        "Magnetic detector",
        "RMS threshold",
    ):
        legacy_name = "Survival MLP (500 ms)" if model == "Survival MLP" else model
        a = next(
            (r for r in rows if r["model"] == legacy_name and r["setting"] == "Legacy"),
            None,
        )
        b = next(
            r for r in rows if r["model"] == model and r["setting"] == "Tokamak-SI"
        )
        figure_rows.append(
            {
                "model": model,
                "legacy": None
                if a is None
                else {
                    "f1": a["metrics"]["f1"],
                    "auprc": a["metrics"]["auprc"],
                    "shots": a["metrics"]["n_shots"],
                    "held_out": a["held_out"],
                    "source": a["source"],
                },
                "tokamak_si": {
                    "f1": b["metrics"]["f1"],
                    "auprc": b["metrics"]["auprc"],
                    "shots": b["metrics"]["n_shots"],
                    "source": b["source"],
                },
            }
        )
    write(
        DOCS / "figure2_tm.json",
        {
            "made_by": "scripts/labeler/tm_benchmark.py",
            "models": figure_rows,
            "coverage": {
                k: {
                    key: v[key]
                    for key in (
                        "labelled_shots",
                        "classifiable_shots",
                        "observable_shots",
                        "labelled_seconds",
                        "observable_plasma_seconds",
                        "definition",
                        "observable_definition",
                    )
                    if key in v
                }
                for k, v in cov.items()
                if isinstance(v, dict)
            },
            "scope": cov["scope"],
            "like_for_like_coverage": {
                k: {
                    key: value
                    for key, value in v.items()
                    if key
                    in (
                        "labelled_seconds",
                        "interval_seconds_on_same_shots",
                        "common_labelled_seconds",
                        "observable_plasma_seconds",
                        "labelled_shots",
                    )
                }
                for k, v in cov.items()
                if k.startswith("legacy")
            },
            "threshold_policy": (
                "F1 at shared-fold inner-validation-tuned thresholds for every row"
            ),
            "paired_common_shots": paired,
            "note": (
                "Legacy and interval targets differ; no paired performance gain is "
                "implied. Null means no legacy model. CNN legacy overlap is unknown; "
                "DSM legacy uses 500 ms. Coverage is clipped to observable plasma "
                "within catalog windows on a common 10 ms grid and includes quiet "
                "labelled time."
            ),
            "source": str((LOCAL / "tm_benchmark.json").relative_to(REPO)),
        },
    )
    print(
        "paper exports:",
        LOCAL / "tm_benchmark.json",
        DOCS / "table_tm_benchmark.tex",
        DOCS / "figure2_tm.json",
    )


if __name__ == "__main__":
    main()
