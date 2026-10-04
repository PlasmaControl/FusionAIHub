#!/usr/bin/env python
"""Verify current TM numerical exports, positive validation and figure provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from labeler.events.interval_tables import parse_attrs
from labeler.tearing import rule, scoring

REPO = Path(__file__).resolve().parents[2]
TM = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)
LOCAL = REPO / "data/events/neoclassical_tearing_mode/benchmark"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, check=True):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=check
    )


def committed(sha_text, what):
    """`sha_text` names a commit that is an ancestor of HEAD (a clean provenance)."""
    assert sha_text and not sha_text.endswith("-dirty"), (what, sha_text)
    assert (
        git("cat-file", "-e", f"{sha_text}^{{commit}}", check=False).returncode == 0
    ), (
        what,
        sha_text,
    )
    assert (
        git("merge-base", "--is-ancestor", sha_text, "HEAD", check=False).returncode
        == 0
    ), (what, sha_text)


def main():
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    blind = set(cohort.query("split == 'test'").shot)
    dev = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    labels = (
        REPO
        / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
    )
    label_sha = sha(labels)
    assert set(pd.read_csv(labels).shot) == set(dev)
    for table_path in (labels, TM / "labels/tm_interval_population.csv"):
        assert not pd.read_csv(table_path).duplicated().any(), table_path
    # provenance: the code that made every output is committed and unmodified
    assert not git("status", "--porcelain", "--", "src", "scripts", "tests").stdout
    assert not set(pd.read_csv(TM / "labels/tm_interval_population.csv").shot) & blind
    plan = read(TM / "results/inner_splits_fix4.json")
    assert plan["labels_sha256"] == label_sha
    assert {int(s): f for s, f in plan["folds"].items()} == scoring.shot_folds(dev)
    for split in plan["splits"]:
        train, val, held = (set(split[k]) for k in ("train", "validation", "held"))
        assert train | val | held == set(dev)
        assert not train & val and not held & (train | val)
        for group in plan["positive_input_support"].values():
            assert val & set(group)
    benchmark = read(LOCAL / "tm_benchmark.json")
    assert benchmark["labels_sha256"] == label_sha
    checked = []
    for row in benchmark["rows"] + benchmark["appendix_rows"]:
        assert not set(row["requested_shots"]) & blind
        assert not row["threshold_unestimable_folds"]
        assert row["metrics"]["replicates"] == 1000
        record = read(REPO / row["source"])
        if "labels_sha256" in record:
            assert record["labels_sha256"] == label_sha
        if "uncertain_negative" in row["source"]:
            committed(record["git_sha"], row["source"])
        for fold in record.get("fold_info", []):
            assert fold["validation_bins_positive"] > 0
            assert fold["threshold"] != 0.999
            assert not set(fold["validation"]) & set(fold["held"])
        checked.append([row["model"], row["setting"]])
    keys = {(r["model"], r["setting"], r["variant"]) for r in benchmark["rows"]}
    for key in (
        ("tm-onsetcnn-published", "Tokamak-SI", "thr. 0.5"),
        ("tm-onsetcnn-published", "Tokamak-SI", ""),
        ("tm-onsetcnn-published", "Tokamak-SI, uncertain = negative", "thr. 0.5"),
        ("tm-onsetcnn-published", "Tokamak-SI, uncertain = negative", ""),
        ("tm-rms-2line", "Tokamak-SI", ""),
        ("tm-rms-2line", "Tokamak-SI", "seed levels"),
        ("tm-ours", "Tokamak-SI, uncertain = negative", ""),
        ("tm-rms-2line", "Tokamak-SI, uncertain = negative", ""),
    ):
        assert key in keys, key
    assert {r["setting"] for r in benchmark["rows"]} <= {
        "Legacy",
        "Tokamak-SI",
        "Tokamak-SI, uncertain = negative",
    }
    # published rows are not interleaved with retrained ones, per setting
    for setting in ("Tokamak-SI", "Tokamak-SI, uncertain = negative"):
        names = [r["model"] for r in benchmark["rows"] if r["setting"] == setting]
        kind = ["published" in n for n in names]
        assert kind == sorted(kind, reverse=True), (setting, names)
    paired = benchmark["paired_common_shots"]
    for comparison in ("full_set", "cnn_subset"):
        for block in paired[comparison].values():
            for difference in block["differences"].values():
                for cell in difference.values():
                    assert cell["excludes_zero"] == (cell["lo"] > 0 or cell["hi"] < 0)
    # the baseline comparison is on every development shot, and each model's value
    # there is the value of its own row in the main table (same bins, same thresholds)
    by_key = {(r["model"], r["setting"], r["variant"]): r for r in benchmark["rows"]}
    for block, setting in (
        (paired["full_set"]["primary"], "Tokamak-SI"),
        (paired["full_set"]["uncertain_negative"], "Tokamak-SI, uncertain = negative"),
    ):
        assert block["shots"] == dev, setting
        assert set(block["metrics"]) == {"tm-ours", "tm-rms-2line"}
        for model, metrics in block["metrics"].items():
            row = by_key[(model, setting, "")]["metrics"]
            assert row["bins_scored"] == block["bins_scored"], (model, setting)
            for key in ("auroc", "auprc", "f1", "segf1_0.5"):
                assert abs(metrics[key]["value"] - row[key]["value"]) < 1e-6, (
                    model,
                    setting,
                    key,
                )
    for block in paired["cnn_subset"].values():
        assert set(block["metrics"]) == {"tm-ours", "tm-onsetcnn-retrained"}
        assert len(block["shots"]) < len(dev)
    for name in ("tm_benchmark.json", "rule_diagnostics_fix4.json"):
        source = read(
            LOCAL / name if name == "tm_benchmark.json" else LOCAL / "sources" / name
        )
        committed(source["git_sha"], name)
    calibration = read(LOCAL / "sources/calibration_dev_fix4.json")
    committed(calibration["git_sha"], "calibration")
    # the rule's hard-coded harmonic level is the calibration's rounded-up percentile
    assert rule.N2_RULE.harmonic_ratio == calibration["harmonic_ratio"]
    assert (
        math.ceil(calibration["harmonic_ratio_unrounded"] * 100) / 100
        == calibration["harmonic_ratio"]
    )
    figure2 = read(REPO / "docs/labeler/figure2_tm.json")
    for row in figure2["rows"]:
        assert {"architecture", "legacy_model", "tokamak_si_model"} <= set(row)
        if row["legacy"] is not None:
            assert (
                row["legacy"]["f1_published_threshold"]
                and "prevalence" in row["legacy"]
            )
            assert row["like_for_like"] is not None
    audit = read(LOCAL / "sources/audit_fix4_current.json")
    for scope in ("cohort", "population"):
        a = audit[scope]
        assert a["locking_audit"]["n_abrupt_incorrect_decay"] == 0
        for key in (
            "raw_failure_count",
            "median_failure_count",
            "supported_failure_count",
        ):
            assert a["after_seed_audit"][key] == 0
    assert audit["source_sha256"]["src/labeler/tearing/rule.py"] == sha(
        REPO / "src/labeler/tearing/rule.py"
    )
    assert audit["source_sha256"]["scripts/labeler/tm_label.py"] == sha(
        REPO / "scripts/labeler/tm_label.py"
    )
    figures = []
    for stem in (
        "tm_gallery_mhr",
        "tm_gallery_mirnov",
        "tm_examples_column_mhr",
        "tm_examples_column_mirnov",
    ):
        base = TM / "figures" / stem
        p = read(base.with_suffix(".json"))
        assert p["labels_sha256"] == label_sha
        committed(p["git_sha"], stem)
        assert p["source_sha256"] == sha(REPO / "scripts/labeler/tm_gallery.py")
        assert not set(p["shots"]) & blind
        assert p["png_sha256"] == sha(base.with_suffix(".png"))
        assert p["pdf_sha256"] == sha(base.with_suffix(".pdf"))
        if "column" in stem:
            assert p["width_inches"] == 3.25 and p["font_pt"] >= 7
            # the locking example is the largest-step cohort lock, picked by a rule
            best = read(LOCAL / "sources/rule_diagnostics_fix4.json")["lock_steps"][
                "cohort"
            ]["largest_step"]
            assert best["shot"] in p["shots"] and p["lock_example_rule"], stem
            assert 189138 not in p["shots"], stem
        figures.append(str(base))
    for stem in (
        "table_tm_benchmark",
        "table_tm_benchmark_appendix",
        "table_tm_paired",
        "table_tm_shot_sets",
    ):
        base = TM / "figures" / f"{stem}_preview"
        p = read(base.with_suffix(".json"))
        assert p["tex_sha256"] == sha(REPO / p["tex_source"])
        assert p["benchmark_sha256"] == sha(LOCAL / "tm_benchmark.json")
        assert p["text_width_inches"] == 6.75 and not p["overfull_boxes"]
        assert p["png_sha256"] == sha(base.with_suffix(".png"))
        assert p["pdf_sha256"] == sha(base.with_suffix(".pdf"))
        figures.append(str(base))
    assert not list((TM / "results").glob("*_test.json"))
    doc = read(TM / "results/document_fix4.json")
    assert doc["source_sha256"] == sha(REPO / "scripts/labeler/tm_write_doc.py")
    assert doc["document_sha256"] == sha(REPO / doc["document"])
    assert doc["benchmark_sha256"] == sha(LOCAL / "tm_benchmark.json")
    readme_path = REPO / "data/events/neoclassical_tearing_mode/README.md"
    readme = readme_path.read_text()
    assert doc["readme_sha256"] == sha(readme_path)
    for span in ("veto", "paired"):
        assert re.search(
            rf"<!-- gen:{span} -->[^x].*<!-- /gen:{span} -->", readme, re.DOTALL
        )
    # onset rows with a degenerate window carry the flag, as the label meta counts
    meta = read(labels.with_suffix(".meta.json"))["counts"]
    frame = pd.read_csv(labels)
    flagged = [
        bool(parse_attrs(a).get("onset_window_degenerate"))
        for a in frame.loc[
            (frame.category == 1) & (frame.t_end == frame.t_start), "attrs"
        ]
    ]
    assert sum(flagged) == meta["n_onset_rows_flagged_degenerate"] > 0
    # hand-typed numbers of the README against the record they come from
    counts = benchmark["label_counts"]["cohort"]["counts"]
    assert (
        counts["n_onset_rows_flagged_degenerate"]
        == meta["n_onset_rows_flagged_degenerate"]
    )
    agreement = {
        ref: benchmark["agreement"][f"{ref}_dev"]["agreement"]["n1"]
        for ref in ("seo", "survival")
    }
    flat = " ".join(readme.split())
    for ref, label in (("seo", "Seo"), ("survival", "survival")):
        a = agreement[ref]
        assert f"{label} **{a['matched']}/{a['reference_onsets']}**" in flat, ref
    assert f"{counts['n_locked']} confirmed locks" in flat
    share = benchmark["rule_diagnostics"]["uncertain_fraction"]["after"]["window"]
    assert f"{100 * share['pooled_fraction']:.1f}% of the observable" in flat
    on_file = benchmark["rule_diagnostics"]["lock_records"]
    assert (
        f"on file for {on_file['dev_shots_with_record']} of the {on_file['dev_shots']}"
    ) in flat
    assert "**latest**: none" in readme or "**latest**: tm-ours" not in readme
    assert "no tearing-mode labels" in readme  # the blind split carries none
    text = (REPO / doc["document"]).read_text()
    for banned in (
        "volts, not gauss",
        "review_baseline",
        "435.7",
        "909.9",
        "earlier 500-shot",
        "near-circular",
        "189879",
        "49.6%",
        "10.4%",
        "earlier level",
        "went from",
    ):
        assert banned not in text, banned
        assert banned not in readme, banned
    # one uncertain-share statistic, the same rounded value in the caption and the doc
    caption = (REPO / "docs/labeler/table_tm_benchmark.tex").read_text()
    in_caption = re.search(r"\((\d+\.\d)\\% of observable catalog-window", caption)
    in_doc = re.search(r"\*\*(\d+\.\d)%\*\* of the observable catalog-window", text)
    assert in_caption and in_doc and in_caption.group(1) == in_doc.group(1)
    assert (REPO / "docs/labeler/tearing_detection_changelog.md").is_file()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("Sources:"):
            assert lines[i - 1] == "", "Sources: needs a blank line before it"
    assert "{'" not in text, "a Python dict repr leaked into the document"
    assert "review_baseline" not in (LOCAL / "tm_benchmark.json").read_text()
    for path in LOCAL.rglob("*.json"):
        assert path.stat().st_size <= 2_000_000, path
    record = {
        "made_by": "scripts/labeler/tm_verify_artifacts.py",
        "labels_sha256": label_sha,
        "benchmark_sha256": sha(LOCAL / "tm_benchmark.json"),
        "development_shots": len(dev),
        "benchmark_rows_checked": checked,
        "figures_verified": figures,
        "no_zero_positive_folds": True,
        "outer_assignments_unchanged": True,
        "no_abrupt_decay": True,
        "no_seed_audit_failures": True,
        "blind_excluded": True,
        "file_sizes_below_2mb": True,
        "status": "passed",
    }
    for path in (
        TM / "results/artifact_verification_fix4.json",
        LOCAL / "sources/artifact_verification_fix4.json",
    ):
        path.write_text(json.dumps(record, indent=2) + "\n")
    print(
        "Verified current labels, unchanged outer folds, positive validation, all score sources, mask sensitivity, seed/locking audits, figures, quarantine, document and file sizes."
    )


if __name__ == "__main__":
    main()
