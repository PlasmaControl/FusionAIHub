#!/usr/bin/env python
"""Render the current ELM protocol and README Models from canonical records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from labeler.config import Paths, sha256_of
from labeler.elm.net import ElmUNet

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs/labeler/elm"
NAMES = {
    "elm-ours": "elm-ours",
    "elm-elmo": "ELM-O",
    "elm-clock": "elm-clock",
    "elm-dsm": "elm-dsm refit",
    "elm-dsm-detect": "elm-dsm detection",
    "elm-dsm-detect-init": "elm-dsm detection init",
    "always present": "Always-present",
}


def metric(res, key):
    if key not in res["point"]:
        return "--"
    point = res["point"][key]
    lo, hi = res["ci95"][key]
    value = f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
    if key == "f1" and res["point"]["recall"] >= 0.99:
        value += "†"
    return value


def benchmark_rows(record):
    all_set, bes_set = (record["sets"][tag] for tag in ("all119", "bes73"))
    lines = [
        (
            f"All reviewed shots: {all_set['n_shots']} shots/{all_set['bins']:,} bins. "
            f"BES subset, ELM-O chunks: {bes_set['n_shots']} shots/{bes_set['bins']:,} bins."
        ),
        "",
        "| Set | Method | AUROC | AUPRC | F1 | Precision / recall |",
        "|---|---|---|---|---|---|",
    ]
    for tag in ("all119", "bes73"):
        subset = record["sets"][tag]
        for name, res in subset["methods"].items():
            cells = [metric(res, key) for key in ("auroc", "auprc", "f1")]
            p, r = res["point"]["precision"], res["point"]["recall"]
            lines.append(
                f"| {tag} | {NAMES[name]} | "
                + " | ".join(cells)
                + f" | {p:.3f} / {r:.3f} |"
            )
    return "\n".join(lines)


def alarm_rows(ours):
    lines = [
        "| Set | Method | Raw | 25 ms guard, same denominator | Nonempty interiors |",
        "|---|---|---|---|---|",
    ]
    for tag in ("all119", "bes73"):
        for name in ("elm-ours", "elm-elmo"):
            res = ours["sets"][tag]["methods"].get(name)
            if res is None:
                continue
            cells = [
                metric(res, key)
                for key in (
                    "absent_span_alarm_rate",
                    "absent_span_alarm_rate_guard25",
                    "absent_span_interior_alarm_rate_guard25",
                )
            ]
            lines.append(f"| {tag} | {NAMES[name]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def audit_rows(swap):
    audits = {"Onset bins": swap["interval_audit"]}
    audits.update(
        {
            f"Occupancy τ={gap} ms": swap["interval_audit_occupancy"][f"gap_{gap}ms"]
            for gap in (100, 200, 300)
        }
    )
    lines = [
        "| Reference | Known bins | M | P | Legacy recall | Legacy F1 |",
        "|---|---|---|---|---|---|",
    ]
    for title, audit in audits.items():
        res = audit["known_review_majority"]
        lines.append(
            f"| {title} | {res['bins']} | {res['M']} | {res['P']} | "
            + metric(res, "recall")
            + " | "
            + metric(res, "f1")
            + " |"
        )
    return "\n".join(lines)


def ranking_rows(swap):
    lines = [
        "| Set | Reference | M / P | Recall | AUROC order | F1 order |",
        "|---|---|---|---|---|---|",
    ]
    for tag in ("overlap", "overlap_bes"):
        subset = swap["swap"][tag]
        refs = {"Onset bins": subset}
        refs.update(
            {
                f"Occupancy τ={gap} ms": subset["occupancy"][f"gap_{gap}ms"]
                for gap in (100, 200, 300)
            }
        )
        for title, record in refs.items():
            finding = record["finding_1"]
            order = record["legacy"]["ranking"]
            orders = [
                " > ".join(NAMES.get(n, n) for n in order[key])
                for key in ("auroc", "f1")
            ]
            lines.append(
                f"| {tag} | {title} | {finding['M']} / {finding['P']} | "
                + metric(finding, "recall")
                + " | "
                + " | ".join(orders)
                + " |"
            )
    return "\n".join(lines)


def repeat_rows(repeats):
    lines = [
        "| Training seed | all119 AUROC | all119 AUPRC | all119 F1 |",
        "|---|---|---|---|",
    ]
    for row in repeats["results"]:
        res = row["sets"]["all119"]
        lines.append(
            f"| {row['training_seed']} | "
            + " | ".join(metric(res, key) for key in ("auroc", "auprc", "f1"))
            + " |"
        )
    lines += [
        "",
        "| Set | Metric | Three new seeds: min–max | Sample SD |",
        "|---|---|---|---|",
    ]
    for tag, metrics in repeats["seed_ranges"].items():
        for key in ("auroc", "auprc", "f1"):
            spread = metrics[key]["three_new_seeds"]
            lines.append(
                f"| {tag} | {key} | {spread['min']:.3f}–{spread['max']:.3f} | "
                f"{spread['sample_sd']:.3f} |"
            )
    return "\n".join(lines)


def main():
    files = {
        "ours": OUTPUTS / "ours/evaluation.json",
        "dsm": OUTPUTS / "dsm/evaluation.json",
        "swap": OUTPUTS / "swap/evaluation.json",
        "repeats": OUTPUTS / "ours/seed_repeats.json",
    }
    records = {key: json.loads(path.read_text()) for key, path in files.items()}
    ours, dsm, swap, repeats = (records[key] for key in files)
    parameter_count = sum(p.numel() for p in ElmUNet().parameters())
    cohort = pd.read_csv(Paths.from_env().catalog / "cohort.csv")
    cohort_counts = {
        str(key): int(value) for key, value in cohort.split.value_counts().items()
    }
    reviewed_counts = {
        str(key): int(value)
        for key, value in cohort[cohort.shot.isin(ours["sets"]["all119"]["shots"])]
        .split.value_counts()
        .items()
    }
    doc = """# ELM occupancy detection: current protocol

## What is measured

The task is to detect reviewed ELM-present time from DIII-D diagnostics. The
reference is interval occupancy, including crowds, rather than independently
verified individual physical ELMs. A separate reference swap compares the same
predictions with legacy onset bins and an occupancy conversion of that source.

## Data and splits

There are 119 reviewed shots, all in the fixed cohort train or validation split.
No cohort blind-test shot enters new detector fitting, checkpoint selection or
threshold selection. Primary all119 bins use fetched filterscope/interferometer
coverage. The BES subset uses ELM-O chunks; common DSM panels additionally require
both offline-risk and detection rows. All methods within a panel use identical
shots and bins. Sources: `outputs/labeler/elm/ours/evaluation.json:sets` and
`outputs/labeler/elm/dsm/evaluation.json:sets`.

Five outer folds group entire physical shots. Each outer training side reserves
14 inner-validation shots for checkpoint and threshold selection. Fixed
partitions and every shot list are in `ours/seed_repeats.json:folds` and the
large run records under `$LABELER_ROOT/round4/elm/cv/`. The original partition
seed is 20261003; three repeats change only training randomness.

## Models and inputs

- **elm-ours:** a 1D U-Net with 401,714 parameters. It uses FS02–FS04 D-alpha,
  DENV2F/DENV3F fast density and a validity mask; no BES. Samples are reduced to
  0.1 ms cells (filterscope maximum, density mean), with log D-alpha level,
  0.5 s median contrast, density and 0.2 s high-pass channels. FS01 was not
  fetched or evaluated. See `src/labeler/elm/inputs.py` for scaling and source
  ordinate provenance; no speculative density rescaling is applied.
- **ELM-O:** the paper reimplementation, requiring BES, with hard-call eta 0.997.
  Its rank scores use the saved nested eta sweep on the same evaluated bins.
- **elm-clock:** the original rule that seeded the review; this reference is
  therefore dependent on the clock. The clock has no continuous rank score.
- **elm-dsm refit:** a 60-column survival refit, compared as an offline forward-risk
  score. **elm-dsm detection** trains one occupancy logit from scratch; **elm-dsm
  detection init** initializes its embedding from the refit. All variants share
  the upstream preprocessing, not just the survival and initialized variants.
- **Always-present:** the trivial baseline, retained in every table.

Every DSM variant has **no D-alpha input (pcphd02/03 mean-filled); 50 ms-mean
serving of a 1 ms-trained model; CO2 missing on 75/119**. The survival refit is an
**offline risk score with 25 ms centered-NBI lookahead (not a causal forecast)**.
The 1 ms training describes the source survival refit; detection heads are
refitted on the reviewed 50 ms-mean rows.
All DSM variants use upstream normalization constants computed **before the
upstream split**, including blind-cohort source shots **190646 and 190532**.
This is feature-statistics exposure, separate from reviewed-label leakage. The
scratch detection variant inherits it too. Sources:
`dsm/evaluation.json:model_context`, `row_diagnostics`; source notebook
`/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/data_processing.ipynb:4406`.

Physical-shot fitting, selection and normalization membership is retained in
`src/labeler/models/d3d_elm_time_to_event_dsm/training_membership.json`. The
adapter's exposed-shot membership prevents general validation from labelling
these shots held out. DSM source-validation scores used for early stopping are
validation evidence. These supplemental models do not certify blind-cohort
isolation. The serving risk scale is poorly calibrated, and out-of-filter rows
are clipped to normalized ±10 instead of being discarded.

## Training and metrics

elm-ours trains 25 epochs per fold, 40 iterations per epoch, batch 16, 4096 ms
crops, learning rate 0.002, weight decay 0.01 and dropout 0.1. Masked event loss
uses present/crowd versus absent milliseconds; uncertain, unobservable and
unlabelled time is ignored. The existing auxiliary span-start loss remains in
the checkpoints, but the onset head is **dropped from paper outputs**, with no
onset retraining: span starts are not adjudicated physical onsets and its
agreement was weak. Only occupancy scores and detections are plotted/reported.
Checkpoint selection maximizes finite inner-validation AUPRC; an all-NaN
history now raises a clear error instead of returning an undefined best row.

A scored 50 ms cell lies wholly inside one known reviewed span and analysed
time; the span must have at least half its duration analysed. Scores are pooled
bin AUROC, average precision (AUPRC), precision, recall and F1. elm-ours bin
scores average the occupancy trace; thresholds maximize inner-validation F1.
Detected runs use a centered 50 ms moving mean. Non-crowd touch recall and absent
touch rates count any interval overlap and do not establish onset accuracy.
Intervals are 95% percentile physical-shot bootstraps with 1000 shared draws;
paired differences use identical draws. † marks recall ≥0.99; numeric F1 and
intervals remain visible, with precision/recall beside them.

## Results

### Primary reviewed bins

"""
    doc += benchmark_rows(ours)
    doc += """

Source: `outputs/labeler/elm/ours/evaluation.json:sets.<set>.methods`.
The BES comparison has similar point estimates and no significant difference
detected; this is not an equivalence result. Paired differences and intervals
are in the same record's `paired` fields.

### Common DSM bins

"""
    doc += benchmark_rows(dsm)
    doc += """

Source: `outputs/labeler/elm/dsm/evaluation.json:sets.<set>.methods`.
Input availability, architecture and objective differ together; their effects
cannot be separated causally. In-sample DSM source exposure is disclosed below.

### Absent-span boundary sensitivity

"""
    doc += alarm_rows(ours)
    counts = ours["sets"]["all119"]["methods"]["elm-ours"]["counts"]
    doc += (
        f"\n\nThe all119 raw denominator is {counts['absent_spans']} absent spans; "
        f"{counts['absent_spans_guard25_empty']} have no interior after removing "
        "25 ms at both edges. The guarded share retains the raw denominator; "
        "the final column uses only nonempty interiors. Centered 50 ms smoothing "
        "can spill detected runs across annotation edges, particularly short "
        "inter-ELM gaps. A boundary touch can trigger the raw metric without "
        "an interior alarm. Sources: `ours/evaluation.json:span_alarm_definition` "
        "and `sets.<set>.methods.<method>.{point,ci95,counts}`.\n"
    )
    doc += """

### Seed stability

Three new full five-fold CV runs retain the original outer and inner shot
partitions and all hyperparameters. Checkpoints and thresholds are selected
independently within each fixed inner-validation fold. All four runs are
reported; no best seed is selected. Shot intervals quantify sampling uncertainty;
the seed range/SD quantify training variability and are not confidence intervals.

"""
    doc += repeat_rows(repeats)
    doc += """

Source: `outputs/labeler/elm/ours/seed_repeats.json:{results,seed_ranges}`;
each entry points to its large evaluation, run record, hashes, selected epochs
and fold thresholds. All predictions remain out of fold by physical shot.

## Reference-swap protocol and results

The **legacy onset table** uses Hiro Farre Josep Kaga's `wpqh_elm_hiro`
annotations, compiled into 50 ms bins by `scripts/labeler/labels_format.py` and
the category formatter (`labeler.events.source_formatters`), as described in
`data/events/README.md`. It overlaps eight reviewed shots: 189885, 190637, 190643,
192721, 192732, 192751, 196541 and 200385; seven have BES/ELM-O chunks.
The all-covered audit includes every legacy-covered cell intersecting the
review window, assigning review state by ≥25 ms occupancy. Unknown review time
is listed separately. The ranking sets require complete known-span interior
bins and DSM rows, a restriction relative to the all-covered audit.

**Finding 1** is agreement of legacy reference with reviewed occupancy:
M = reviewed-present/legacy-absent; P = legacy-present/reviewed-absent.
The onset version is retained. Occupancy references merge legacy-positive
50 ms intervals whose intervening gaps are ≤τ, at τ=100/200/300 ms. Missing
legacy coverage is never bridged. No margin extends beyond first/last positive
intervals, and no τ is chosen on performance. These are target-definition
sensitivities, not counts of independently verified missing or false ELMs.

"""
    doc += audit_rows(swap)
    doc += """

Sources: `swap/evaluation.json:interval_audit.known_review_majority` and
`interval_audit_occupancy.gap_<tau>ms.known_review_majority`. Uncertain positives
on shot 192751 remain unknown rather than being treated as reviewed absent.

**Finding 2** rescored identical saved predictions/calls under each reference;
thresholds were not selected again. The review order and every reference's
pair reversals are in `swap.<set>.{reviewed,legacy}.ranking`, `comparison`, and
`occupancy.gap_<tau>ms.{legacy.ranking,comparison}`. The reference orders are:

"""
    doc += ranking_rows(swap)
    doc += """

The DSM refit trained on **190637, 190643, 192721, 192751 and 196541**,
five of the eight overlap shots. Its refit row and initialized embedding are
marked **in-sample with respect to source fitting**, even though reviewed-label
threshold selection is out of fold. The three-source-unexposed-shot analysis
(189885, 192732, 200385) is separately reported at
`swap.overlap_dsm_heldout` and `swap.overlap_bes_dsm_heldout`, under every
reference. Upstream feature-statistics exposure remains disclosed; these are
not a new blindly isolated DSM fit. The small overlap gives broad intervals.
Numerical AUROC/F1 with intervals and precision/recall for all variants are in
the committed swap tables; no high-recall F1 is hidden.

## Limitations

- The annotations differ between shots: 33 shots mark non-crowd/per-ELM spans,
  while 76 mark whole ELMing periods as crowds. Short absent gaps between
  per-ELM spans and sustained crowd occupancy are different targets. Their
  starts are not verified physical ELM onsets. Source: `ours/evaluation.json:
  annotation_modes`.
- The clock seeded the review; clock scores and detector-derived proxies are
  dependent-reference diagnostics. Raw absent-span touches are annotation
  disagreements, not verified physical false alarms or annotation errors.
- Shot CV does not group run days. Earlier development runs had outer-fold
  predictions, so these results are development estimates. Same-day exposure
  and historical provenance are recorded in `training_history.json`.
- GroupNorm sees crops during training and whole shots during inference;
  no separate effect-size study was performed. Three seed repeats do not
  replace run-day validation or expert event adjudication.
- DSM diagnostics, serving resolution, noncausal NBI preprocessing,
  source normalization exposure, in-sample swap shots and poor calibration
  limit interpretation.
- The offline source-metadata audit cannot verify physical ordinate units for
  DENV2F/DENV3F on any reviewed shot: the original fast-channel cache discarded
  them, and accessible offline source files do not retain them. The unsupported
  m⁻² declaration is removed. The scale is **10¹⁴ native source-ordinate units
  (physical units unverified)**, with no numerical rescaling. Source paths,
  channel names, cache/array hashes and metadata sidecars are preserved in
  `density_units.json`; no network fetch was authorized for this round.

## Artifacts and reproduction

Committed tables: `outputs/labeler/elm/table_elm_benchmark.tex` and
`outputs/labeler/elm/swap/table_elm_*.tex`. Large PDF/150 dpi PNG examples and
table copies live under `$LABELER_ROOT/round4/elm/`; `figures/fig_elm_examples.json`
specifies the selection rule, windows, source units and per-panel fold thresholds.
Non-crowd shading and clock bars have distinct colors. Use the figure at its
recorded two-column width. Source/table hashes are in `tables.json`.

Run CPU Python through the mandated pixi wrapper with this worktree's PYTHONPATH,
LABELER_ROOT, LABELER_LABEL_TABLES, LABELER_NO_FETCH=1 and the ELM TMPDIR. Arguments:

```text
scripts/labeler/elm_ours_evaluate.py --run cv2
scripts/labeler/elm_dsm_evaluate.py --run cv2 --rescore --refresh-report
scripts/labeler/elm_reference_swap.py --run cv2
scripts/labeler/elm_seed_repeats.py --aggregate
scripts/labeler/elm_round2_records.py
scripts/labeler/elm_paper_tables.py
scripts/labeler/elm_example_figure.py --run cv2
scripts/labeler/elm_protocol.py
scripts/labeler/elm_fix_verify.py
```

Seed training uses `elm_seed_repeats.py --train` in the prescribed CUDA venv,
CUDA_VISIBLE_DEVICES=0 with modest resources. Tests use only covering files via
the required pt.sh wrapper; scoped Ruff and format checks are recorded in
`fix_round2_verification.json`. No labels or production stores are modified.

## Appendix: fix history

Round one corrected physical DSM phase IDs, source-shot overlap, onset-proximity
interpretation, diagnostic fetching and fit/rescore provenance. Round two adds
occupancy references, explicit DSM exposures/serving conditions, visible high-recall
F1, guarded alarm rates, three seeds, adapter membership and current-state artifacts.
Superseded DSM snapshots are archived outside git with hashes at
`round2_records.json:archived_records`; one current consolidated DSM record remains.
The complete chronological history is in the stream's round-four report, not this
protocol. Earlier elm-ours cv2 provenance is retrospective; new repeats capture
provenance automatically. No external reviewer score is claimed here.
"""
    doc = doc.replace("401,714", f"{parameter_count:,}")
    doc = doc.replace(
        "There are 119 reviewed shots, all in the fixed cohort train or validation split.",
        "The fixed cohort contains "
        + ", ".join(f"{cohort_counts[tag]} {tag}" for tag in ("train", "val", "test"))
        + ". The 119 reviewed shots contain "
        + ", ".join(f"{reviewed_counts.get(tag, 0)} {tag}" for tag in ("train", "val"))
        + "; these counts are recorded in `outputs/labeler/elm/protocol.json:metadata`.",
    )
    (REPO / "docs/labeler/elm_ours.md").write_text(doc)
    readme = REPO / "data/events/edge_localized_mode/README.md"
    text = readme.read_text()
    model_lines = [
        "## Models",
        "",
        "**stable**: elm-dsm | 2026_09_06 (offline risk score)",
        "",
        "**latest**: elm-ours | 2026_10_03",
        "",
        "**all**:",
        "",
    ]
    for name, source, tag in (
        ("elm-ours", ours, "all119"),
        ("elm-elmo", ours, "bes73"),
        ("elm-clock", ours, "all119"),
        ("elm-dsm", dsm, "all119"),
        ("elm-dsm-detect", dsm, "all119"),
        ("elm-dsm-detect-init", dsm, "all119"),
        ("always present", ours, "all119"),
    ):
        subset = source["sets"][tag]
        res = subset["methods"][name]
        label = "BES subset, ELM-O chunks" if tag == "bes73" else "all reviewed shots"
        cells = [
            f"{key.upper()}: {metric(res, key)}" for key in ("auroc", "auprc", "f1")
        ]
        model_lines.append(
            f"- {NAMES[name]} | "
            + "; ".join(cells)
            + f" ({subset['n_shots']} shots/{subset['bins']:,} 50 ms bins; {label})"
        )
    model_lines += [
        "",
        (
            "Brackets are 95% shot-bootstrap intervals; † marks recall ≥0.99. "
            "Always-present precision equals prevalence and recall is 1; full numeric "
            "precision/recall and intervals are in the tables and protocol."
        ),
        "",
        (
            "All detector fits use five physical-shot-grouped folds and exclude cohort "
            "blind-test shots from new fitting/selection. ELM-O needs BES; elm-clock "
            "seeded the review. Primary and common DSM bin sets differ and are labelled "
            "separately. Sources: `outputs/labeler/elm/{ours,dsm}/evaluation.json:sets`."
        ),
        "",
        "Raw / 25 ms edge-guarded absent-span touch rates (same denominator):",
        "",
    ]
    for tag, name in (
        ("all119", "elm-ours"),
        ("bes73", "elm-ours"),
        ("bes73", "elm-elmo"),
    ):
        res = ours["sets"][tag]["methods"][name]
        model_lines.append(
            f"- {NAMES[name]}, {tag}: "
            + metric(res, "absent_span_alarm_rate")
            + " / "
            + metric(res, "absent_span_alarm_rate_guard25")
        )
    model_lines += [
        "",
        (
            "Centered 50 ms smoothing can spill detected runs across span edges. "
            "Short empty guarded interiors are counted separately; the protocol also "
            "gives the rate restricted to nonempty interiors. These are annotation "
            "disagreements, not independently verified physical false alarms. "
            "The occupancy target differs between 33 per-ELM/non-crowd-annotated shots "
            "and 76 crowd-annotated shots. The unsupported onset head is dropped from "
            "paper outputs; checkpoints retain its auxiliary training loss."
        ),
        "",
        (
            "Every elm-dsm variant has no D-alpha input (pcphd02/03 mean-filled); "
            "50 ms-mean serving of a 1 ms-trained model; CO2 missing on 75/119. "
            "The 1 ms training describes the source survival refit; detection heads "
            "train on reviewed 50 ms-mean rows. "
            "The refit is an offline risk score with 25 ms centered-NBI lookahead "
            "(not a causal forecast). Every variant, including scratch detection, "
            "uses upstream normalization constants computed before the upstream split, "
            "including blind-cohort source shots 190646 and 190532 (feature-statistics "
            "exposure). Refit and initialized detection also inherit source fitting; "
            "five overlap shots (190637,190643,192721,192751,196541) are in-sample. "
            "Physical membership is propagated to the adapter so exposed shots are "
            "not called held out. Source: `dsm/evaluation.json:model_context,own_target`."
        ),
        "",
        (
            "The legacy onset table uses Hiro Farre Josep Kaga annotations, "
            "compiled by labels_format.py/source_formatters, and overlaps "
            "eight reviewed shots. The swap reports onset bins beside occupancy "
            "conversion at 100/200/300 ms gap tolerances, with M/P/recall and rankings "
            "under every reference. A separate three-shot DSM-source-unexposed analysis "
            "retains the normalization disclosure. Three full new-seed CV repeats "
            "report spread without choosing a best seed. Sources: "
            "`swap/evaluation.json` and `ours/seed_repeats.json`."
        ),
        "",
        (
            "Current protocol, all intervals, seed results, limitations and artifacts: "
            "[elm_ours.md](../../../docs/labeler/elm_ours.md). "
            "ELM-O's independent BES-window benchmark is described in "
            "[elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md)."
        ),
        "",
        "",
    ]
    begin, end = text.index("## Models"), text.index("## Inputs")
    readme.write_text(text[:begin] + "\n".join(model_lines) + text[end:])
    manifest = {
        "metadata": {
            "elm_ours_parameters": parameter_count,
            "cohort_split_counts": cohort_counts,
            "reviewed_split_counts": reviewed_counts,
            "cohort_source": str(Paths.from_env().catalog / "cohort.csv"),
            "cohort_sha256": sha256_of(Paths.from_env().catalog / "cohort.csv"),
        },
        "sources": {
            key: {"path": str(path.relative_to(REPO)), "sha256": sha256_of(path)}
            for key, path in files.items()
        },
        "artifacts": {
            str(path.relative_to(REPO)): sha256_of(path)
            for path in (REPO / "docs/labeler/elm_ours.md", readme)
        },
    }
    (OUTPUTS / "protocol.json").write_text(json.dumps(manifest, indent=1))
    print("protocol", "docs/labeler/elm_ours.md")
    print("large artifacts", Paths.from_env().root / "round4/elm")


if __name__ == "__main__":
    main()
