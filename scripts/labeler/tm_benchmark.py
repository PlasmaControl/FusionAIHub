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


def rescore_oof(stem, cohort):
    """Current-label metrics and explicit train/validation/held shot lists."""
    record = read(TM / "results" / f"{stem}.json")
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
    folds = {int(s): int(k) for s, k in record["folds"].items()}
    assert folds == scoring.shot_folds(dev, 5, seed=0)
    thresholds, splits = {}, []
    for info in record["fold_info"]:
        k = info["fold"]
        held = [s for s in dev if folds[s] == k]
        pool = [s for s in dev if folds[s] != k]
        train, val = scoring.inner_split(pool, 100 + k, 0.1)
        assert not set(held) & (set(train) | set(val))
        assert not set(train) & set(val)
        assert not (set(train) | set(val) | set(held)) & blind
        thresholds.update({s: info["threshold"] for s in held})
        splits.append({"fold": k, "train": train, "validation": val, "held": held})
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
        labels_sha256=hashlib.sha256(LABELS.read_bytes()).hexdigest(),
    )
    write(TM / "results" / f"{stem}.json", record)
    print(stem, "verified", metrics["n_shots"], "shots", delta, flush=True)


def coverage(cohort):
    out = {
        "scope": "fixed cohort comparison, plus separately named population supplement; all splits for coverage only, no model test scores"
    }
    for name, path in (
        ("ours", LABELS),
        ("ours_population", TM / "labels/tm_interval_population.csv"),
    ):
        table = pd.read_csv(path)
        windows = pd.read_csv(
            REPO
            / f"data/events/catalog/{'population' if 'population' in name else 'cohort'}.csv"
        )
        by_shot = {int(s): g for s, g in table.groupby("shot")}
        bins = positive = observable = 0
        for row in windows.itertuples():
            if row.shot not in by_shot:
                continue
            centres = scoring.bin_centres((row.window_start_ms, row.window_end_ms))
            y, valid = scoring.label_bins(by_shot[row.shot], centres)
            bins += int(valid.sum())
            positive += int((y.astype(bool) & valid).sum())
            observable += int(valid.any())
        out[name] = {
            "labelled_shots": len(by_shot),
            "observable_shots": observable,
            "labelled_seconds": bins * 0.01,
            "present_seconds": positive * 0.01,
            "definition": "observable 10 ms bins; uncertain and unobservable excluded",
            "source": str(path),
        }
    paths = replace(Paths.from_env(), root=TM / "lroot")
    records, skipped = {}, {}
    for shot in cohort.shot:
        truth = archived_truth(int(shot), paths)
        if truth.get("available"):
            records[int(shot)] = {
                "samples": len(truth["t"]),
                "positive": int(truth["tm_label"].sum()),
            }
        else:
            skipped[int(shot)] = truth.get("reason", "unavailable")
    out["legacy_seo"] = {
        "labelled_shots": len(records),
        "labelled_seconds": sum(v["samples"] for v in records.values()) * 0.025,
        "positive_seconds": sum(v["positive"] for v in records.values()) * 0.025,
        "definition": "matched archived tm_label samples, each 25 ms; growth-phase target",
        "per_shot": records,
        "skipped": skipped,
    }
    path = REPO / "data/events/neoclassical_tearing_mode/raw/tm_labels.h5"
    if not path.is_file():
        path = Path(
            "/scratch/gpfs/nc1514/FusionAIHub/data/events/neoclassical_tearing_mode/raw/tm_labels.h5"
        )
    records = {}
    with h5py.File(path) as f:
        for shot in cohort.shot:
            if str(shot) not in f:
                continue
            t = f[str(shot)]["time"][:]
            dt = float(np.median(np.diff(t))) if len(t) > 1 else 20.0
            records[int(shot)] = {"samples": len(t), "step_ms": dt}
    out["legacy_survival"] = {
        "labelled_shots": len(records),
        "labelled_seconds": sum(
            v["samples"] * v["step_ms"] / 1000 for v in records.values()
        ),
        "definition": "native survival-label samples; an onset target, not present spans",
        "per_shot": records,
        "source": str(path),
    }
    return out


def display(metric):
    if metric is None:
        return "--"
    values = [f"{metric[k]:.3f}".removeprefix("0") for k in ("value", "lo", "hi")]
    return f"{values[0]} [{values[1]},{values[2]}]"


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
    rows = []

    def add(stem, model, setting, *, held_out=True, metric_key="metrics", note=""):
        source = TM / "results" / f"{stem}.json"
        record = read(source)
        shots = record.get("shots", record.get("dev_shots", []))
        if metric_key == "metrics_outside_training":
            trained = set(record["shots_in_training"])
            shots = [s for s in shots if s not in trained]
            metrics = record[metric_key]["published"]
        else:
            metrics = record[metric_key]
        assert not set(shots) & blind, stem
        assert metrics["replicates"] == 1000, stem
        empty = set(metrics["shots_without_a_scored_bin"])
        scored = [s for s in shots if s not in empty]
        assert len(scored) == metrics["n_shots"], stem
        snapshot = LOCAL / "sources" / source.name
        write(snapshot, record)
        rows.append(
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
                "threshold": record.get("threshold", record.get("threshold_g")),
                "thresholds_by_fold": [
                    i["threshold"] for i in record.get("fold_info", [])
                ],
            }
        )

    add(
        "tm_prior_published_tm-onsetcnn_legacy_cohort",
        "Onset CNN",
        "Legacy",
        held_out=False,
        note="Training overlap unknown: archive is the training store; descriptive score only.",
    )
    for h in ("250ms", "500ms", "1s"):
        label = {"250ms": "250 ms", "500ms": "500 ms", "1s": "1 s"}[h]
        add(
            f"tm_prior_published_tm-dsm-{h}_legacy",
            f"Survival MLP ({label})",
            "Legacy",
            note="Outside published training list; cohort blind test excluded.",
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
            note="Outside published training list; published threshold 0.7.",
        )
    add("tm_prior_retrained_tm-onsetcnn_cv", "Onset CNN", "Tokamak-SI")
    add("tm_prior_retrained_tm-dsm_cv", "Survival MLP", "Tokamak-SI")
    add("tm_ours_magnetics_cv", "Magnetic detector", "Tokamak-SI")
    add("tm_baseline_n1rms_12g_dev", "RMS threshold (12 G)", "Tokamak-SI")
    add(
        "tm_ours_magnetics_rms_cv",
        "Magnetic detector + RMS",
        "Tokamak-SI",
        note="Circular input ablation: RMS defines the label.",
    )
    agreements = {}
    for ref in ("seo", "survival"):
        for split in ("dev", "all"):
            name = f"agreement_{ref}_cohort_{split}.json"
            agreements[f"{ref}_{split}"] = read(TM / "agreement" / name)
            write(LOCAL / "sources" / name, agreements[f"{ref}_{split}"])
    # Supplementary population CNN legacy row, never represented as held out.
    source = TM / "results/tm_prior_published_tm-onsetcnn_legacy_population.json"
    write(LOCAL / "sources" / source.name, read(source))
    cov = coverage(cohort)
    locking = {}
    frequencies = {int(p.stem) for p in (TM / "signals_freq").glob("*.npz")}
    for split in ("cohort", "population"):
        full = TM / f"labels/tm_intervals_full_{split}.csv"
        mode_shots = set(pd.read_csv(full).shot)
        locking[split] = {
            "shots_with_intervals": len(mode_shots),
            "with_frequency": len(mode_shots & frequencies),
            "without_frequency": sorted(int(s) for s in mode_shots - frequencies),
            "source": str(full),
            "meaning": "A locked flag requires frequency evidence. Unflagged population intervals without that record have unknown locking status.",
        }
    summary = {
        "made_by": "scripts/labeler/tm_benchmark.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_sha": git_sha(),
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "evaluation": "development only; shot-grouped CV for retrained models; published thresholds for fixed models",
        "bootstrap": {
            "unit": "shot",
            "replicates": 1000,
            "confidence": 0.95,
            "seed": 0,
        },
        "legacy_segmental_f1": "not applicable: legacy targets are growth-phase rows or forecasts, not present spans",
        "rows": rows,
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
            "rule_changed": False,
            "rule_sha256": hashlib.sha256(
                (REPO / "src/labeler/tearing/rule.py").read_bytes()
            ).hexdigest(),
            "finding": "Strong RMS intervals agree with growth/decay on the Mirnov gallery. MHR is partially covered and less informative. Weak coherent lines below the published RMS seed level remain outside the label definition.",
        },
        "blind_test": {
            "shots": sorted(int(s) for s in blind),
            "reporting": "excluded from benchmark, model selection and new scoring",
            "historical_exposure": "Previous implementer wrote *_test.json records with --final. They remain external and are excluded from this paper export.",
        },
    }
    write(LOCAL / "tm_benchmark.json", summary)
    write(TM / "results/tm_benchmark.json", summary)
    caption = (
        "Tearing-mode benchmarks. Legacy rows use published original targets. "
        "Tokamak-SI learned rows use shot-grouped development cross-validation; the RMS rule is fixed. "
        "Published interval rows reuse original weights. Brackets give 95\\% shot-bootstrap "
        "intervals (1,000 draws). Segmental F1 uses temporal IoU 0.5; legacy targets lack "
        "present spans. Shot counts reflect input availability. A dagger marks unknown "
        "published training overlap. The RMS-input detector is a circular ablation. "
        "Blind cohort test shots are excluded."
    )
    assert len(caption.split()) <= 80
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrllll}",
        r"\toprule",
        r"Model & Setting & Shots & AUROC [CI] & AUPRC [CI] & F1 [CI] & Segmental F1 [CI] \\",
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
    figure_rows = []
    for model in (
        "Onset CNN",
        "Survival MLP",
        "Magnetic detector",
        "RMS threshold (12 G)",
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
                    "shots": a["metrics"]["n_shots"],
                    "held_out": a["held_out"],
                    "source": a["source"],
                },
                "tokamak_si": {
                    "f1": b["metrics"]["f1"],
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
                    for key in ("labelled_shots", "labelled_seconds", "definition")
                }
                for k, v in cov.items()
                if isinstance(v, dict)
            },
            "scope": cov["scope"],
            "note": "Legacy and interval targets differ; this is not a paired performance gain. Null means no legacy model. CNN legacy overlap is unknown; DSM legacy uses 500 ms horizon. Coverage includes quiet labelled time, not just positives.",
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
