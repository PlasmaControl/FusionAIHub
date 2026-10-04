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
from tm_cv_plan import load_plan

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
        assert positive > 0, "positive validation support required"
        assert info["threshold"] != 0.999, "forbidden fallback threshold"
        info["threshold_status"] = "positive-bearing inner-validation F1 optimum"
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
        "zero_positive_convention": "error: no fallback is allowed",
        "f1_limitation": None,
    }
    return unestimable


def rule_audit_bundle():
    """Bundle current-only audits; no historical blind-cohort record is consumed."""
    result = {"sources": {}}
    for filename in (
        "audit_fix4_current.json",
        "criterion_support_fix4.json",
        "island_inventory_fix2.json",
    ):
        record = read(TM / "labels" / filename)
        destination = LOCAL / "sources" / filename
        write(destination, record)
        if destination.stat().st_size > 2_000_000:
            destination.write_text(json.dumps(record, separators=(",", ":")) + "\n")
        assert destination.stat().st_size <= 2_000_000
        result["sources"][filename] = str(destination.relative_to(REPO))
        result[filename.removesuffix(".json")] = {
            key: value for key, value in record.items() if key != "details"
        }
    for scope in ("cohort", "population"):
        current = result["audit_fix4_current"][scope]
        current["after_seed_audit"] = {
            k: v for k, v in current["after_seed_audit"].items() if k != "details"
        }
    return result


BASELINE_NAMES = {"n1rms": "tm-rms", "tworms": "tm-rms-2line"}
UNCERTAIN_NEGATIVE_POLICY = (
    "category 2 (uncertain) scored as negative, category 3 (not observable) still "
    "excluded; the same fitted models and the same inner-validation thresholds as the "
    "primary row, so interval boundaries and the weak-line margin are scored too"
)


def uncertain_negative_records(cohort):
    """Every Tokamak-SI row of the main table again, with uncertain time as negatives.

    Writes one record per row to ``results/<stem>_uncertain_negative_dev.json``; the
    scores and thresholds are those of the primary row, only the target mask changes.
    Returns `{row key: stem}`.
    """
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    table = pd.read_csv(LABELS)
    rows = {int(s): g for s, g in table.groupby("shot")}
    nothing = table.iloc[:0]
    stems = {}

    def emit(key, model, shots, centres, scores, thresholds, finite=None, **extra):
        y, valid = {}, {}
        for s in shots:
            y[s], valid[s] = scoring.label_bins(
                rows.get(s, nothing), centres[s], uncertain_negative=True
            )
            if finite is not None:
                valid[s] &= finite[s]
        metrics = scoring.evaluate(shots, y, valid, scores, thresholds, n=1000, seed=0)
        stem = f"{key}_uncertain_negative_dev"
        write(
            TM / "results" / f"{stem}.json",
            {
                "made_by": "scripts/labeler/tm_benchmark.py",
                "git_sha": git_sha(),
                "model": model,
                "shots": shots,
                "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
                "policy": UNCERTAIN_NEGATIVE_POLICY,
                "metrics": metrics,
                **extra,
            },
        )
        stems[key] = stem

    def by_fold(record, shots):
        level = {i["fold"]: i["threshold"] for i in record["fold_info"]}
        return {s: level[record["folds"][str(s)]] for s in shots}

    def saved(path, shots):
        with np.load(path) as z:
            return {s: z[f"s{s}"] for s in shots}

    for tag in ("magnetics", "magnetics_rms"):
        data, _ = tm_ours.load(dev, tag == "magnetics_rms")
        record = read(TM / f"results/tm_ours_{tag}_cv.json")
        shots = sorted(data)
        emit(
            f"tm_ours_{tag}",
            "tm-ours" + ("-rms" if "rms" in tag else ""),
            shots,
            {s: data[s][0] for s in shots},
            saved(TM / f"results/oof_tm_ours_{tag}.npz", shots),
            by_fold(record, shots),
            finite={s: np.isfinite(data[s][1]).all(axis=1) for s in shots},
        )
    for kind, name in (("cnn", "tm-onsetcnn"), ("dsm", "tm-dsm")):
        data, _ = tm_prior_retrain.load(kind, dev)
        record = read(TM / f"results/tm_prior_retrained_{name}_cv.json")
        path = TM / f"results/oof_tm_prior_retrained_{name}.npz"
        with np.load(path) as z:
            shots = sorted(s for s in data if f"s{s}" in z.files)
        emit(
            f"tm_prior_retrained_{name}",
            f"{name}-retrained",
            shots,
            {s: data[s]["centres"] for s in shots},
            saved(path, shots),
            by_fold(record, shots),
        )
    data, _ = tm_ours.load_baseline(dev)
    shots = sorted(data)
    finite = {s: np.isfinite(data[s][1][:, 0]) for s in shots}
    for kind, (_, fixed, _) in tm_ours.BASELINES.items():
        score = tm_ours.run_baseline(data, shots, kind)
        record = read(TM / f"results/tm_baseline_{kind}_cv_tuned_dev.json")
        tuned = {int(s): v for s, v in record["thresholds_by_shot"].items()}
        for label, thr in (
            ("cv_tuned", tuned),
            ("12g" if kind == "n1rms" else "seed", fixed),
        ):
            emit(
                f"tm_baseline_{kind}_{label}",
                BASELINE_NAMES[kind],
                shots,
                {s: data[s][0] for s in shots},
                score,
                thr,
                finite=finite,
            )
    for name, (slug, column, shift, fixed) in tm_prior_published.SI_MODELS.items():
        if name not in ("tm-onsetcnn", "tm-dsm-500ms"):
            continue
        record = read(TM / f"results/tm_prior_published_{name}_tokamak-si_dev.json")
        score, _, _, used, _ = tm_prior_published.si_bins(
            slug, column, shift, dev, cohort, rows, table
        )
        if "dsm" in name:
            trained = set(record["shots_in_training"])
            used = [s for s in used if s not in trained]
        centres = {
            s: scoring.bin_centres(
                tuple(
                    float(v)
                    for v in cohort.set_index("shot").loc[
                        s, ["window_start_ms", "window_end_ms"]
                    ]
                )
            )
            for s in used
        }
        for label, thr in (("tuned", by_fold(record, used)), ("fixed", fixed)):
            emit(
                f"tm_prior_published_{name}_tokamak-si_{label}",
                f"{name}-published",
                used,
                centres,
                score,
                thr,
            )
    return stems


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
    shared_folds, shared_splits = load_plan(cohort_dev)
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
    blind = set(cohort.query("split == 'test'").shot)
    out = {
        "scope": (
            "development cohort and population excluding blind; common 10 ms catalog-window "
            "grid and observable plasma domain for both legacy and interval labels"
        )
    }
    interval_masks, observable_masks, grids = {}, {}, {}
    for name, path in (
        ("ours", LABELS),
        ("ours_population", TM / "labels/tm_interval_population.csv"),
    ):
        table = pd.read_csv(path)
        table = table[~table.shot.isin(blind)]
        catalog = "population" if "population" in name else "cohort"
        start_path = TM / f"labels/plasma_start_{catalog}.json"
        starts = read(start_path)
        windows = pd.read_csv(REPO / f"data/events/catalog/{catalog}.csv")
        windows = windows[~windows.shot.isin(blind)]
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
            "uncertain_seconds": (plasma_bins - bins) * 0.01,
            "uncertain_fraction": (plasma_bins - bins) / plasma_bins,
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
    for shot in cohort.query("split != 'test'").shot:
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
        for shot in cohort.query("split != 'test'").shot:
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


def _paired_block(shots, y, valid, scores, thresholds, pairs):
    """Metrics of every model and the paired differences on one shot and bin set.

    Each model is scored at its own thresholds on the same shots and bins; the shot
    bootstrap draws one set of weights per replicate and applies it to every model, so
    each difference (`pairs`: name to the two models it subtracts) carries a paired
    interval.
    """
    shots = [s for s in shots if valid[s].any()]
    results, stats = {}, {}
    for model, model_scores in scores.items():
        edges = scoring.edges_for(
            np.concatenate([model_scores[s][valid[s]] for s in shots])
        )
        results[model] = scoring.evaluate(
            shots,
            y,
            valid,
            model_scores,
            thresholds[model],
            edges=edges,
            n=1000,
            seed=0,
        )
        stats[model] = [
            scoring.shot_stats(
                s, y[s], valid[s], model_scores[s], edges, thresholds[model][s]
            )
            for s in shots
        ]
    keys = ("auroc", "auprc", "f1", "segf1_0.5")
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(1000):
        weights = np.bincount(
            rng.integers(0, len(shots), len(shots)), minlength=len(shots)
        )
        got = {m: scoring.metrics(stats[m], weights) for m in scores}
        draws.append(
            {
                name: {k: got[a][k] - got[b][k] for k in keys}
                for name, (a, b) in pairs.items()
            }
        )
    differences = {}
    for name, (a, b) in pairs.items():
        differences[name] = {}
        for metric in keys:
            samples = np.array([d[name][metric] for d in draws])
            samples = samples[np.isfinite(samples)]
            lo, hi = (
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            )
            differences[name][metric] = {
                "value": results[a][metric]["value"] - results[b][metric]["value"],
                "lo": lo,
                "hi": hi,
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
    first = next(iter(results.values()))
    return {
        "shots": shots,
        "bins_scored": first["bins_scored"],
        "scored_seconds": first["scored_seconds"],
        "metrics": results,
        "differences": differences,
    }


def paired_common_shots(cohort):
    """tm-ours against the two-line baseline, and against the retrained CNN, paired.

    Each comparison uses the shots and the available 10 ms bins that BOTH of its models
    score, each model at its own inner-validation thresholds, and one shot bootstrap
    whose draws are shared by the two models, so a difference carries a paired
    interval. `full_set` pairs tm-ours with the two-line baseline: neither needs
    anything but the Mirnov features, so it holds every development shot and bin.
    `cnn_subset` pairs tm-ours with the retrained CNN, whose inputs exist on fewer
    shots (the baseline adds no restriction there); it is the only set a CNN
    comparison may use. Each comparison has a primary block (uncertain time excluded)
    and an `uncertain_negative` block (uncertain time scored as negative on the same
    scores and thresholds).
    """
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    ours, _ = tm_ours.load(dev, False)
    cnn, _ = tm_prior_retrain.load("cnn", dev)
    base, _ = tm_ours.load_baseline(dev)
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}

    def fold_scores(stem, probability, shots):
        record = read(TM / "results" / f"{stem}.json")
        level = {i["fold"]: i["threshold"] for i in record["fold_info"]}
        with np.load(TM / "results" / f"{probability}.npz") as z:
            scores = {s: z[f"s{s}"] for s in shots}
        return scores, {s: level[record["folds"][str(s)]] for s in shots}

    def block_for(models, shots, pair, mask):
        """Primary and uncertain-negative blocks of `models` on their common bins."""
        scores = {m: models[m][0] for m in models}
        thresholds = {m: models[m][1] for m in models}
        y, valid, y_un, valid_un = {}, {}, {}, {}
        for s in shots:
            assert np.array_equal(ours[s][0], cnn[s]["centres"]), s
            assert np.array_equal(ours[s][2], cnn[s]["y_bins"]), s
            finite = np.ones(len(ours[s][0]), bool)
            for model in scores:
                finite &= np.isfinite(scores[model][s])
            y[s] = ours[s][2]
            valid[s] = ours[s][3] & mask(s) & finite
            y_un[s], label_valid = scoring.label_bins(
                by_shot.get(s, table.iloc[:0]), ours[s][0], uncertain_negative=True
            )
            valid_un[s] = label_valid & finite & np.isfinite(ours[s][1]).all(axis=1)
        pairs = {f"{pair[0]}_minus_{pair[1]}": pair}
        return {
            "primary": _paired_block(shots, y, valid, scores, thresholds, pairs),
            "uncertain_negative": _paired_block(
                shots, y_un, valid_un, scores, thresholds, pairs
            ),
        }

    shots = sorted(set(ours) & set(base))
    record = read(TM / "results/tm_baseline_tworms_cv_tuned_dev.json")
    tuned = record["thresholds_by_shot"]
    ours_models = fold_scores("tm_ours_magnetics_cv", "oof_tm_ours_magnetics", shots)
    baseline = (
        tm_ours.run_baseline(base, shots, "tworms"),
        {s: tuned[str(s)] for s in shots},
    )
    full = block_for(
        {"tm-ours": ours_models, "tm-rms-2line": baseline},
        shots,
        ("tm-ours", "tm-rms-2line"),
        lambda s: True,
    )
    shots = sorted(set(ours) & set(cnn))
    ours_models = fold_scores("tm_ours_magnetics_cv", "oof_tm_ours_magnetics", shots)
    twin = fold_scores(
        "tm_prior_retrained_tm-onsetcnn_cv",
        "oof_tm_prior_retrained_tm-onsetcnn",
        shots,
    )
    subset = block_for(
        {"tm-ours": ours_models, "tm-onsetcnn-retrained": twin},
        shots,
        ("tm-ours", "tm-onsetcnn-retrained"),
        lambda s: cnn[s]["valid_bins"],
    )
    return {
        "full_set": full,
        "cnn_subset": subset,
        "policy": (
            "each comparison scores the shots and available 10 ms bins both of its "
            "models have, each model at its own fold thresholds, on one paired "
            "1000-draw shot bootstrap; full_set (tm-ours, two-line baseline) holds "
            "every development shot and is the set for any comparison with the "
            "baseline; cnn_subset (tm-ours, retrained CNN) is restricted to the shots "
            "and bins where the CNN has inputs and is used only for comparisons with "
            "the CNN; the uncertain_negative blocks score uncertain time as negative "
            "on the same scores and thresholds"
        ),
    }


def like_for_like(cohort, kind):
    """The published model and its retrained twin on one target, one mask, one shot set.

    Both are scored on the Tokamak-SI labels, on exactly the shots and bins both have
    a score for (for the DSM, outside the published model's training list), with the
    uncertain time excluded (primary) and scored as negative. The published model is
    shown at its own threshold and at the fold-tuned one; the retrained twin at its
    fold-tuned threshold.
    """
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    name, slug, column, shift, fixed = {
        "cnn": ("tm-onsetcnn", tm_prior_published.CNN, None, 25.0, 0.5),
        "dsm": ("tm-dsm", tm_prior_published.DSM, 1, 0.0, 0.3),
    }[kind]
    horizon = "" if kind == "cnn" else "-500ms"
    data, _ = tm_prior_retrain.load(kind, dev)
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}
    published, _, _, used, _ = tm_prior_published.si_bins(
        slug, column, shift, dev, cohort, by_shot, table
    )
    original = read(
        TM / f"results/tm_prior_published_{name}{horizon}_tokamak-si_dev.json"
    )
    retrained_record = read(TM / f"results/tm_prior_retrained_{name}_cv.json")
    with np.load(TM / f"results/oof_tm_prior_retrained_{name}.npz") as z:
        retrained = {s: z[f"s{s}"] for s in used if s in data and f"s{s}" in z}
    shots = sorted(retrained)
    if kind == "dsm":
        trained = set(original["shots_in_training"])
        shots = [s for s in shots if s not in trained]

    def thresholds(record):
        level = {i["fold"]: i["threshold"] for i in record["fold_info"]}
        return {s: level[record["folds"][str(s)]] for s in shots}

    out = {
        "published_model": f"{name}{horizon}-published",
        "retrained_model": f"{name}-retrained",
        "published_threshold": fixed,
    }
    for mode, negative in (("primary", False), ("uncertain_negative", True)):
        y, valid = {}, {}
        for s in shots:
            y[s], label_valid = scoring.label_bins(
                by_shot.get(s, table.iloc[:0]),
                data[s]["centres"],
                uncertain_negative=negative,
            )
            valid[s] = (
                label_valid & np.isfinite(retrained[s]) & np.isfinite(published[s])
            )
        scored = [s for s in shots if valid[s].any()]
        got = {}
        for label, scores, thr in (
            ("published_at_published_threshold", published, fixed),
            ("published_at_tuned_threshold", published, thresholds(original)),
            ("retrained", retrained, thresholds(retrained_record)),
        ):
            got[label] = scoring.evaluate(scored, y, valid, scores, thr, n=1000, seed=0)
        out[mode] = {
            "shots": scored,
            "n_shots": len(scored),
            "bins_scored": got["retrained"]["bins_scored"],
            "prevalence": got["retrained"]["prevalence"],
            "metrics": got,
        }
    out["retrained_exceeds_published"] = {
        k: out["primary"]["metrics"]["retrained"][k]["value"]
        > out["primary"]["metrics"]["published_at_tuned_threshold"][k]["value"]
        for k in ("auroc", "auprc")
    }
    return out


def short_number(value):
    """Three decimals without the leading zero; a rounded zero has no sign."""
    text = f"{value:.3f}"
    if float(text) == 0.0:
        return ".000"
    return text.replace("0.", ".", 1)


def display(metric):
    if metric is None:
        return "--"
    values = [short_number(metric[k]) for k in ("value", "lo", "hi")]
    return f"{values[0]} [{values[1]},{values[2]}]"


#: Row setting -> the heading of its group in the tables.
SETTING_TITLE = {
    "Legacy": "Legacy targets, at the published thresholds",
    "Legacy, tuned": "Legacy targets, inner-validation thresholds",
    "Legacy, fixed": "Legacy targets, at the published thresholds",
    "Tokamak-SI": "Tokamak-SI labels, uncertain time excluded",
    "Tokamak-SI, uncertain = negative": "Tokamak-SI labels, uncertain time scored as negative",
    "Tokamak-SI, fixed": "Tokamak-SI labels, fixed thresholds, uncertain time excluded",
    "Tokamak-SI, uncertain = negative, fixed": "Tokamak-SI labels, fixed thresholds, uncertain time scored as negative",
}


def compact(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def table_tex(rows, caption, label):
    """A booktabs table; consecutive rows with one `setting` share a heading row."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.75in}"
            r"rllll}"
        ),
        r"\toprule",
        (
            r"Model & Shots / bins & AUROC [CI] & AUPRC [CI] & F1 [CI] & "
            r"Segmental F1 [CI] \\"
        ),
    ]
    setting = None
    for row in rows:
        metrics = row["metrics"]
        if row["setting"] != setting:
            setting = row["setting"]
            lines.append(r"\midrule")
            lines.append(
                r"\multicolumn{6}{l}{\emph{"
                + SETTING_TITLE.get(setting, setting).replace("&", r"\&")
                + "}} \\\\"
            )
        name = row["model"] + (f" ({row['variant']})" if row.get("variant") else "")
        name = name + (r"$\dagger$" if not row.get("held_out", True) else "")
        shots = row.get("shots_text") or (
            f"{metrics['n_shots']} / {compact(metrics['bins_scored'])}"
        )
        lines.append(
            " & ".join(
                [
                    name,
                    shots,
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


def shot_sets_tex(rows, caption, label):
    """Shot sets of every row, grouped by target, set, shot count and bin count."""
    groups: dict = {}
    for row in rows:
        m = row["metrics"]
        if row["setting"].startswith("Legacy"):
            target = "Legacy targets"
        elif "uncertain = negative" in row["setting"]:
            target = "Tokamak-SI, uncertain = negative"
        else:
            target = "Tokamak-SI, uncertain excluded"
        key = (target, row["shot_set"], m["n_shots"], m["bins_scored"])
        names = groups.setdefault(key, [])
        if row["model"] not in names:
            names.append(row["model"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.2in}"
            r">{\raggedright\arraybackslash}p{2.0in}rr"
            r">{\raggedright\arraybackslash}p{1.7in}}"
        ),
        r"\toprule",
        r"Target & Shot set & Shots & Bins & Rows \\",
        r"\midrule",
    ]
    for (target, shot_set, n_shots, bins), names in groups.items():
        lines.append(
            " & ".join(
                [
                    target.replace("&", r"\&"),
                    shot_set,
                    str(n_shots),
                    compact(bins),
                    "; ".join(names),
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


SETTING_ORDER = (
    "Legacy",
    "Tokamak-SI",
    "Tokamak-SI, uncertain = negative",
    "Legacy, tuned",
    "Tokamak-SI, fixed",
    "Tokamak-SI, uncertain = negative, fixed",
)
SHOTS = {
    "legacy_cnn": "development shots whose Seo archive rows the row match places",
    "legacy_dsm": "development shots outside the DSM training list that the "
    "survival labels cover",
    "si_published": "development shots with exported detector inputs",
    "si_dsm": "development shots with exported detector inputs, outside the DSM "
    "training list",
    "retrained": "development shots with exported detector inputs; out-of-fold",
    "ours": "development shots with Mirnov features; out-of-fold",
    "rms": "development shots with N1RMS and N2RMS records",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--gallery-reviewed", action="store_true")
    args = ap.parse_args(argv)
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    blind = set(cohort.query("split == 'test'").shot)
    if args.rescore:
        for stem in (
            "tm_prior_retrained_tm-onsetcnn_cv",
            "tm_prior_retrained_tm-dsm_cv",
            "tm_ours_magnetics_cv",
            "tm_ours_magnetics_rms_cv",
        ):
            rescore_oof(stem, cohort)
    uncertain = uncertain_negative_records(cohort)
    rows, appendix = [], []

    def add(
        stem,
        model,
        setting,
        *,
        shot_set,
        held_out=True,
        metric_key="metrics",
        note="",
        extra=False,
        variant="",
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
        fixed_threshold = (
            "fixed" in metric_key
            or "fixed" in setting
            or bool(
                variant
                and ("thr." in variant or "seed" in variant or "12 G" in variant)
            )
        )
        (appendix if extra else rows).append(
            {
                "model": model,
                "setting": setting,
                "variant": variant,
                "source": str(snapshot.relative_to(REPO)),
                "original_source": str(source),
                "metrics": metrics,
                "shots": scored,
                "requested_shots": shots,
                "shot_set": shot_set,
                "held_out": held_out,
                "note": note,
                "threshold_unestimable_folds": [] if fixed_threshold else unestimable,
                "threshold": (
                    record.get(
                        "fixed_published_threshold", record.get("threshold_native")
                    )
                    if fixed_threshold
                    else record.get("threshold", record.get("threshold_native"))
                ),
                "thresholds_by_fold": [
                    i["threshold"]
                    for i in record.get("fold_info", [])
                    if not fixed_threshold
                ],
            }
        )

    UN = "Tokamak-SI, uncertain = negative"
    cnn = "tm_prior_published_tm-onsetcnn"
    dsm500 = "tm_prior_published_tm-dsm-500ms"
    # Legacy targets, published thresholds (the CNN card's 0.5, the DSM card's 0.7
    # survival level = risk 0.3).
    add(
        f"{cnn}_legacy_cohort",
        "tm-onsetcnn-published",
        "Legacy",
        metric_key="metrics_fixed_published",
        variant="thr. 0.5",
        held_out=False,
        shot_set=SHOTS["legacy_cnn"],
        note="Training overlap unknown: the archive is the training store; descriptive.",
    )
    add(
        f"{dsm500}_legacy",
        "tm-dsm-500ms-published",
        "Legacy",
        metric_key="metrics_fixed_published",
        variant="risk 0.3",
        shot_set=SHOTS["legacy_dsm"],
        note="Outside the published training list; cohort blind test excluded.",
    )
    for horizon in ("250ms", "1s"):
        add(
            f"tm_prior_published_tm-dsm-{horizon}_legacy",
            f"tm-dsm-{horizon}-published",
            "Legacy",
            metric_key="metrics_fixed_published",
            variant="risk 0.3",
            shot_set=SHOTS["legacy_dsm"],
            extra=True,
        )
    # Legacy at an inner-validation threshold, secondary.
    add(
        f"{cnn}_legacy_cohort",
        "tm-onsetcnn-published",
        "Legacy, tuned",
        held_out=False,
        shot_set=SHOTS["legacy_cnn"],
        extra=True,
    )
    for horizon in ("250ms", "500ms", "1s"):
        add(
            f"tm_prior_published_tm-dsm-{horizon}_legacy",
            f"tm-dsm-{horizon}-published",
            "Legacy, tuned",
            shot_set=SHOTS["legacy_dsm"],
            extra=True,
        )
    # Tokamak-SI labels, uncertain time excluded.
    add(
        f"{cnn}_tokamak-si_dev",
        "tm-onsetcnn-published",
        "Tokamak-SI",
        held_out=False,
        shot_set=SHOTS["si_published"],
        note="Published weights; original training overlap unknown.",
    )
    add(
        f"{cnn}_tokamak-si_dev",
        "tm-onsetcnn-published",
        "Tokamak-SI",
        metric_key="metrics_fixed_published",
        variant="thr. 0.5",
        held_out=False,
        shot_set=SHOTS["si_published"],
        note="Published weights and published threshold; training overlap unknown.",
    )
    add(
        f"{dsm500}_tokamak-si_dev",
        "tm-dsm-500ms-published",
        "Tokamak-SI",
        metric_key="metrics_outside_training",
        shot_set=SHOTS["si_dsm"],
        note="Outside the published training list; fold-tuned risk threshold.",
    )
    for horizon in ("250ms", "1s"):
        add(
            f"tm_prior_published_tm-dsm-{horizon}_tokamak-si_dev",
            f"tm-dsm-{horizon}-published",
            "Tokamak-SI",
            metric_key="metrics_outside_training",
            shot_set=SHOTS["si_dsm"],
            extra=True,
        )
    add(
        "tm_prior_retrained_tm-onsetcnn_cv",
        "tm-onsetcnn-retrained",
        "Tokamak-SI",
        shot_set=SHOTS["retrained"],
    )
    add(
        "tm_prior_retrained_tm-dsm_cv",
        "tm-dsm-retrained",
        "Tokamak-SI",
        shot_set=SHOTS["retrained"],
    )
    add(
        "tm_baseline_tworms_cv_tuned_dev",
        "tm-rms-2line",
        "Tokamak-SI",
        shot_set=SHOTS["rms"],
        note="No training: max(n=1 RMS / 12 G, n=2 RMS / 6 G), threshold tuned.",
    )
    add(
        "tm_baseline_tworms_seed_dev",
        "tm-rms-2line",
        "Tokamak-SI",
        variant="seed levels",
        shot_set=SHOTS["rms"],
        note="No training, no tuning: either line at its seed amplitude.",
    )
    add(
        "tm_baseline_n1rms_cv_tuned_dev",
        "tm-rms",
        "Tokamak-SI",
        shot_set=SHOTS["rms"],
        note="n=1 RMS only, threshold tuned.",
    )
    add("tm_ours_magnetics_cv", "tm-ours", "Tokamak-SI", shot_set=SHOTS["ours"])
    add(
        "tm_ours_magnetics_rms_cv",
        "tm-ours-rms",
        "Tokamak-SI",
        shot_set=SHOTS["ours"],
        note="Circular input ablation: the RMS defines the label.",
    )
    # Tokamak-SI labels, uncertain time scored as negative (co-primary).
    for key, model, variant, in_main, shot_set in (
        (f"{cnn}_tokamak-si_tuned", "tm-onsetcnn-published", "", True, "si_published"),
        (
            f"{cnn}_tokamak-si_fixed",
            "tm-onsetcnn-published",
            "thr. 0.5",
            True,
            "si_published",
        ),
        (f"{dsm500}_tokamak-si_tuned", "tm-dsm-500ms-published", "", True, "si_dsm"),
        (
            "tm_prior_retrained_tm-onsetcnn",
            "tm-onsetcnn-retrained",
            "",
            True,
            "retrained",
        ),
        ("tm_prior_retrained_tm-dsm", "tm-dsm-retrained", "", True, "retrained"),
        ("tm_baseline_tworms_cv_tuned", "tm-rms-2line", "", True, "rms"),
        ("tm_baseline_tworms_seed", "tm-rms-2line", "seed levels", False, "rms"),
        ("tm_baseline_n1rms_cv_tuned", "tm-rms", "", False, "rms"),
        ("tm_ours_magnetics", "tm-ours", "", True, "ours"),
        ("tm_ours_magnetics_rms", "tm-ours-rms", "", False, "ours"),
    ):
        add(
            uncertain[key],
            model,
            UN,
            variant=variant,
            held_out="published" not in model or "dsm" in model,
            shot_set=SHOTS[shot_set],
            extra=not in_main,
            note="Same scores and thresholds as the primary row; category 3 excluded.",
        )
    # Fixed-threshold extras (appendix).
    add(
        "tm_baseline_n1rms_12g_dev",
        "tm-rms",
        "Tokamak-SI, fixed",
        variant="12 G",
        shot_set=SHOTS["rms"],
        extra=True,
    )
    for horizon in ("250ms", "500ms", "1s"):
        add(
            f"tm_prior_published_tm-dsm-{horizon}_tokamak-si_dev",
            f"tm-dsm-{horizon}-published",
            "Tokamak-SI, fixed",
            variant="risk 0.3",
            metric_key="metrics_outside_training:fixed",
            shot_set=SHOTS["si_dsm"],
            extra=True,
            note="DSM survival threshold 0.7 corresponds to risk threshold 0.3.",
        )
    rows.sort(key=lambda r: SETTING_ORDER.index(r["setting"]))
    appendix.sort(key=lambda r: SETTING_ORDER.index(r["setting"]))

    def find(model, setting, variant=""):
        for row in rows + appendix:
            if (row["model"], row["setting"], row["variant"]) == (
                model,
                setting,
                variant,
            ):
                return row
        raise KeyError((model, setting, variant))

    agreements = {}
    for ref in ("seo", "survival"):
        name = f"agreement_{ref}_cohort_dev.json"
        agreements[f"{ref}_dev"] = read(TM / "agreement" / name)
        write(LOCAL / "sources" / name, agreements[f"{ref}_dev"])
    sensitivity = "sensitivity_survival_dev.json"
    agreements["survival_sensitivity_dev"] = read(TM / "agreement" / sensitivity)
    write(LOCAL / "sources" / sensitivity, agreements["survival_sensitivity_dev"])
    # Supplementary population CNN legacy row, never represented as held out.
    source = TM / "results/tm_prior_published_tm-onsetcnn_legacy_population.json"
    write(LOCAL / "sources" / source.name, read(source))
    cov = coverage(cohort)
    paired = paired_common_shots(cohort)
    like = {kind: like_for_like(cohort, kind) for kind in ("cnn", "dsm")}
    audit = rule_audit_bundle()
    write(
        LOCAL / "sources/inner_splits_fix4.json",
        read(TM / "results/inner_splits_fix4.json"),
    )
    diagnostics = read(LOCAL / "sources/rule_diagnostics_fix4.json")
    calibration = read(LOCAL / "sources/calibration_dev_fix4.json")
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
                "A lock is confirmed by a step of the n=1 radial field (|DUSBRADIAL|, "
                "native ptdata units): the median over 20 to 120 ms after a candidate "
                "time (a frequency drop at least 50 ms after the seed starts, a "
                "collapse, an interval end) exceeds the median over 200 to 20 ms "
                "before it by at least 5. "
                "A field that is already high and drifting is not a step. Unknown "
                "locking status is not a negative; time after a collapse stays "
                "uncertain until a measured release or the discharge end."
            ),
        }
    sizes = {
        "retrained_models_train_shots_per_fold": {
            "mean": float(
                np.mean(
                    [
                        i["n_train"]
                        for i in read(
                            TM / "results/tm_prior_retrained_tm-onsetcnn_cv.json"
                        )["fold_info"]
                    ]
                )
            ),
            "folds": 5,
        },
        "published_dsm_training_shots": len(tm_prior_published.dsm_training_shots()),
        "published_cnn_training_shots": "not listed; the card reports a corpus of "
        "thousands of shots",
        "meaning": (
            "The published models were trained on thousands of shots; the retrained "
            "twins see about 324 per outer fold, so a retrained model that ranks "
            "below its published twin is confounded by training-set size. No "
            "population-scale retrain is part of this benchmark."
        ),
    }
    summary = {
        "made_by": "scripts/labeler/tm_benchmark.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_sha": git_sha(),
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "evaluation": (
            "development only; AUROC and AUPRC primary; F1 and segmental F1 at "
            "thresholds tuned on interval-presence-stratified inner validation with "
            "positive support for every model; outer folds shared. Two co-primary "
            "target definitions: uncertain time excluded, and uncertain time scored "
            "as negative (which scores interval boundaries)."
        ),
        "primary_metrics_do_not_score": (
            "Boundary placement. Uncertain time borders the intervals, is excluded "
            "from bin metrics and acts as a barrier in segmental IoU, so the "
            "excluded-uncertain group does not test where an interval starts or ends; "
            "the uncertain-as-negative group does."
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
        "like_for_like": like,
        "training_size_confound": sizes,
        "rule_audit": audit,
        "rule_diagnostics": diagnostics,
        "harmonic_calibration": {
            k: v
            for k, v in calibration.items()
            if k not in ("definition", "reference_shots")
        },
        "inner_split_plan": (
            "data/events/neoclassical_tearing_mode/benchmark/sources/"
            "inner_splits_fix4.json"
        ),
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
        "column_examples": {
            d: read(TM / f"figures/tm_examples_column_{d}.json")
            for d in ("mhr", "mirnov")
        },
        "gallery_review": {
            "reviewed": args.gallery_reviewed,
            "rule_sha256": hashlib.sha256(
                (REPO / "src/labeler/tearing/rule.py").read_bytes()
            ).hexdigest(),
            "finding": (
                "Gallery uses live MHR row 2 and the Mirnov array. Continuous seed "
                "and coherent-line evidence define strong rotating n=1/n=2 modes; "
                "unsupported candidates, sustained sub-seed lines and locked phases "
                "are uncertain."
            ),
        },
        "blind_test": {
            "shots": sorted(int(s) for s in blind),
            "reporting": "excluded from the benchmark, model selection and scoring",
            "labels": "the blind split carries no tearing-mode labels",
            "exposure": (
                "Earlier development wrote test-split score files; they were moved "
                "unread into results/quarantine_blind_test/ and enter no calculation."
            ),
        },
    }
    write(LOCAL / "tm_benchmark.json", summary)
    write(TM / "results/tm_benchmark.json", summary)
    uncertain_share = diagnostics["uncertain_fraction"]["after"]["window"][
        "pooled_fraction"
    ]
    train_per_fold = round(sizes["retrained_models_train_shots_per_fold"]["mean"])
    caption = (
        "Recovery of a strong rotating n=1/n=2 magnetic rule (a tearing-mode "
        "proxy), not independent identification. AUROC and AUPRC are primary; "
        "F1 uses inner-validation thresholds. Uncertain time "
        f"({100 * uncertain_share:.1f}\\% of observable catalog-window time) is "
        "excluded in the "
        "upper block and scored as negative in the lower; boundary placement is "
        "scored only by the lower. tm-rms-2line is max(n=1/12 G, n=2/6 G), no "
        f"training. Published models saw thousands of shots, retrained ones "
        f"{train_per_fold} per fold. $\\dagger$: training overlap unknown. Brackets: "
        "95\\% shot-bootstrap intervals. Shot sets: appendix."
    )
    assert len(caption.split()) <= 100, len(caption.split())
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "table_tm_benchmark.tex").write_text(
        table_tex(rows, caption, "tab:tm-benchmark")
    )
    (DOCS / "table_tm_benchmark_appendix.tex").write_text(
        table_tex(
            appendix,
            "Additional rows. Legacy rows at inner-validation thresholds, other DSM "
            "horizons, fixed-threshold results (DSM risk 0.3 equals upstream survival "
            "probability 0.7), and the uncertain-as-negative values of the rows not "
            "in the main table. Legacy bins are 25 ms, Tokamak-SI bins 10 ms. "
            "$\\dagger$: published CNN training overlap unknown. Brackets: 95\\% "
            "shot-bootstrap intervals.",
            "tab:tm-benchmark-appendix",
        )
    )
    (DOCS / "table_tm_shot_sets.tex").write_text(
        shot_sets_tex(
            rows + appendix,
            "Shot set of every benchmark row. Every set is a subset of the 450 "
            "development shots (the 50 blind shots are never opened); the shot lists "
            "are in each row's source JSON.",
            "tab:tm-shot-sets",
        )
    )
    keys = ("auroc", "auprc", "f1", "segf1_0.5")
    paired_rows = []
    for target, mode in (
        ("uncertain time excluded", "primary"),
        ("uncertain time scored as negative", "uncertain_negative"),
    ):
        for comparison, scope in (
            ("full_set", "tm-ours and the two-line baseline, all development shots"),
            ("cnn_subset", "tm-ours and the retrained CNN, shots where it has inputs"),
        ):
            setting = f"{target[0].upper()}{target[1:]}: {scope}"
            block = paired[comparison][mode]
            count = f"{len(block['shots'])} / {compact(block['bins_scored'])}"
            paired_rows += [
                {
                    "model": model,
                    "setting": setting,
                    "metrics": metrics,
                    "shots_text": count,
                }
                for model, metrics in block["metrics"].items()
            ]
            for name, diff in block["differences"].items():
                a, b = name.split("_minus_")
                paired_rows.append(
                    {
                        "model": f"Difference, {a} $-$ {b}",
                        "setting": setting,
                        "metrics": {k: diff[k] for k in keys},
                        "shots_text": count,
                    }
                )
    (DOCS / "table_tm_paired.tex").write_text(
        table_tex(
            paired_rows,
            "Paired comparisons on development shots, with uncertain time excluded "
            "(upper two blocks) and scored as negative (lower two). Each comparison "
            "uses the shots and available 10 ms bins both of its models score (shot "
            "and bin counts in the second column). Each model keeps its "
            "inner-validation threshold. Difference rows are paired shot-bootstrap "
            "intervals (1,000 draws); a difference whose interval spans zero is not "
            "resolved. "
            "Unavailable bins are hard barriers for segmental IoU 0.5.",
            "tab:tm-paired",
        )
    )

    def cell(row):
        m = row["metrics"]
        return {
            "f1": m["f1"],
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "prevalence": m["prevalence"],
            "shots": m["n_shots"],
            "bins": m["bins_scored"],
            "bin_ms": m["bin_ms"],
            "source": row["source"],
        }

    figure_rows = []
    for arch, legacy_model, si_model, kind, published in (
        (
            "onsetcnn",
            "tm-onsetcnn-published",
            "tm-onsetcnn-retrained",
            "cnn",
            "tm-onsetcnn-published",
        ),
        (
            "dsm",
            "tm-dsm-500ms-published",
            "tm-dsm-retrained",
            "dsm",
            "tm-dsm-500ms-published",
        ),
        ("magnetic-detector", None, "tm-ours", None, None),
        ("two-line-rms", None, "tm-rms-2line", None, None),
        ("n1-rms", None, "tm-rms", None, None),
    ):
        if legacy_model is None:
            legacy = None
        else:
            fixed = find(
                legacy_model, "Legacy", "thr. 0.5" if kind == "cnn" else "risk 0.3"
            )
            tuned = find(legacy_model, "Legacy, tuned")
            legacy = {
                "model": legacy_model,
                "held_out": fixed["held_out"],
                "f1_published_threshold": fixed["metrics"]["f1"],
                "f1_tuned": tuned["metrics"]["f1"],
                "auroc": fixed["metrics"]["auroc"],
                "auprc": fixed["metrics"]["auprc"],
                "prevalence": fixed["metrics"]["prevalence"],
                "shots": fixed["metrics"]["n_shots"],
                "bins": fixed["metrics"]["bins_scored"],
                "bin_ms": fixed["metrics"]["bin_ms"],
                "source": fixed["source"],
            }
        entry = {
            "architecture": arch,
            "legacy_model": legacy_model,
            "tokamak_si_model": si_model,
            "legacy": legacy,
            "tokamak_si": cell(find(si_model, "Tokamak-SI")),
            "tokamak_si_uncertain_negative": cell(find(si_model, UN)),
            "like_for_like": None if kind is None else like_cell_pair(like[kind]),
        }
        figure_rows.append(entry)
    write(
        DOCS / "figure2_tm.json",
        {
            "made_by": "scripts/labeler/tm_benchmark.py",
            "schema": FIGURE2_SCHEMA,
            "rows": figure_rows,
            "baselines": {
                "two_line_rms_seed_levels": cell(
                    find("tm-rms-2line", "Tokamak-SI", "seed levels")
                ),
            },
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
                "Legacy F1 at the model's published threshold (CNN 0.5, DSM risk 0.3) "
                "is primary and the inner-validation value secondary; Tokamak-SI F1 "
                "at positive-bearing presence-stratified inner-validation thresholds."
            ),
            "primary_comparison": "AUROC and AUPRC, with prevalence beside AUPRC",
            "paired_common_shots": paired,
            "training_size_confound": sizes,
            "note": (
                "Legacy and Tokamak-SI targets differ in target, bin width, shot set, "
                "prevalence and mask, so no cross-setting gain is implied; the "
                "like-for-like pair scores a published model and its retrained twin "
                "on one target, one mask and one shot set. Null means no legacy model."
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


FIGURE2_SCHEMA = {
    "rows": "one per architecture, in drawing order",
    "rows[].architecture": "key shared by the legacy and Tokamak-SI model",
    "rows[].legacy_model": "published model scored on its own legacy target (null: none)",
    "rows[].tokamak_si_model": "model scored on the Tokamak-SI labels (retrained twin "
    "or ours)",
    "rows[].legacy": "f1_published_threshold (primary), f1_tuned (secondary), auroc, "
    "auprc, prevalence (beside AUPRC: it differs from the Tokamak-SI target), shots, "
    "bins, bin_ms (25), held_out",
    "rows[].tokamak_si": "f1 (inner-validation threshold), auroc, auprc, prevalence, "
    "shots, bins, bin_ms (10), uncertain time excluded",
    "rows[].tokamak_si_uncertain_negative": "the same with uncertain time scored as "
    "negative",
    "rows[].like_for_like": "published model and retrained twin on one target, mask and "
    "shot set: primary and uncertain_negative, each with the published model at its "
    "published and tuned thresholds and the retrained model",
    "paired_common_shots": "full_set (tm-ours and the two-line baseline, every "
    "development shot) and cnn_subset (tm-ours and the retrained CNN, the shots and "
    "bins where the CNN has inputs), each with a primary block (uncertain time "
    "excluded) and an uncertain_negative block; every block has the shots, bins, the "
    "models' metrics and the paired differences with their 95% intervals",
    "metric cells": "{value, lo, hi}, 95% shot-bootstrap, 1000 draws",
}


def like_cell_pair(block):
    """Compact like-for-like record of `like_for_like` for the Figure 2 JSON."""
    out = {
        "published_model": block["published_model"],
        "retrained_model": block["retrained_model"],
        "published_threshold": block["published_threshold"],
    }
    for mode in ("primary", "uncertain_negative"):
        part = block[mode]
        out[mode] = {
            "shots": part["n_shots"],
            "bins": part["bins_scored"],
            "prevalence": part["prevalence"],
            **{
                label: {k: m[k] for k in ("auroc", "auprc", "f1", "segf1_0.5")}
                for label, m in part["metrics"].items()
            },
        }
    return out


if __name__ == "__main__":
    main()
