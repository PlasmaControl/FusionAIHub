#!/usr/bin/env python
"""Render the current ELM protocol and README Models from canonical records."""

from __future__ import annotations

import json
import math
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
    "elm-dsm": "elm-dsm refit‡ (supplemental)",
    "elm-dsm-detect": "elm-dsm detection (isolated)",
    "elm-dsm-detect-exposed": "elm-dsm detection‡ (exposed; supplemental)",
    "elm-dsm-detect-init": "elm-dsm detection init‡ (supplemental)",
    "always present": "always present",
}


def metric(res, key):
    if key not in res["point"]:
        return "--"
    point = res["point"][key]
    if not math.isfinite(point):
        return "--"
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


def kind_rows(ours):
    lines = [
        "| Set | Method | Crowd-bin recall | Non-crowd span-touch recall |",
        "|---|---|---|---|",
    ]
    for tag in ("all119", "bes73"):
        for name, res in ours["sets"][tag]["methods"].items():
            if name == "always present":
                continue
            lines.append(
                f"| {tag} | {NAMES[name]} | "
                + metric(res, "crowd_bin_recall")
                + " | "
                + metric(res, "non_crowd_span_touch_recall")
                + " |"
            )
    return "\n".join(lines)


def mode_rows(ours):
    lines = [
        (
            "| Annotation mode | Shots / bins | AUROC | F1 | Precision | Recall | "
            "Absent-bin call fraction |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, group in ours["sets"]["all119"]["annotation_modes"].items():
        res = group["methods"]["elm-ours"]
        lines.append(
            f"| {name.replace('_', ' ')} | {group['n_shots']} / {group['bins']:,} | "
            + " | ".join(
                (
                    "--"
                    if name == "no_present" and key != "false_alarm_bin_rate"
                    else metric(res, key)
                )
                for key in (
                    "auroc",
                    "f1",
                    "precision",
                    "recall",
                    "false_alarm_bin_rate",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def own_target_rows(dsm):
    lines = [
        "| Horizon (ms) | AUROC | Cases / controls |",
        "|---|---|---|",
    ]
    for res in dsm["own_target"]["horizons"].values():
        point = res["auroc"]
        low, high = res["auroc_ci95"]
        lines.append(
            f"| {res['horizon_ms']:g}‡ | {point:.3f} [{low:.3f}, {high:.3f}] | "
            f"{res['cases']:,} / {res['controls']:,} |"
        )
    return "\n".join(lines)


def filterscope_rows(record):
    lines = ["| Input | Tree PMT | View from metadata |", "|---|---|---|"]
    for name in record["used_channels"]:
        item = record["views"][name]
        view = item["view"] or "unverified: no divertor/midplane field"
        lines.append(f"| {name} | {item['pmt']} | {view} |")
    lines += [
        "",
        (
            f"FS01 has finite corpus samples on {record['fs01_finite_shots']}/119 shots. "
            "It is outside the existing three-channel native-rate cache and all retained "
            "fits; exclusion was a data-pipeline choice. Fetching is allowed this round. "
            "Read-only tree queries verified the PMT aliases but returned only signal "
            "and calibration metadata, with no sightline labels. Each channel's divertor "
            "versus midplane view therefore remains unverified. Source: "
            "`outputs/labeler/elm/filterscope_metadata.json`."
        ),
    ]
    return "\n".join(lines)


def native_rows(record):
    lines = [
        "| Native panel | Horizon (ms) | Scored shots / 1 ms rows | AUROC |",
        "|---|---|---|---|",
    ]
    for panel in ("reviewed_exact_export", "reviewed_reconstructed", "own_target"):
        for name, result in record[panel]["horizons"].items():
            low, high = result["auroc_ci95"]
            lines.append(
                f"| {panel}‡ | {name[1:-2]} | {result['n_shots']} / "
                f"{result['rows']:,} | {result['auroc']:.3f} "
                f"[{low:.3f}, {high:.3f}] |"
            )
    lines += [
        "",
        (
            "The native checkpoint uses all **124 original inputs, including both "
            "photodiodes and 64 BES columns, at 1 ms**. Exact original exports exist "
            "on five reviewed shots, but 192751 has no scored reviewed overlap: "
            "four contribute 11,565 rows. All four have source exposure: 196541 "
            "was optimizer-trained; 190637, 190643 and 192721 were used for "
            "checkpoint selection. This is a separate source-exposed panel, not "
            "the 119-shot 50 ms benchmark."
        ),
        "",
        (
            "Original diagnostic records plus paced photodiode fetches make 45 "
            "shots reconstructable; only 33 contribute reviewed rows after the "
            "original native-domain filter (no mean fill or clipping). Reconstructed "
            "inputs are a **sensitivity**, because within-shot NBI smoothing cannot "
            "reproduce the source's smoothing of concatenated filtered phase rows "
            "across boundaries. The exact export is the faithful native-input "
            "evaluation. Native own-target AUROC uses 326 physical early-stopping "
            "validation shots from the original reversed split, not untouched test "
            "evidence. Every panel has 1000 physical-shot bootstrap replicates."
        ),
        "",
        (
            "Sources: `dsm/native_evaluation.json:{checkpoint,coverage,exposure,"
            "reviewed_exact_export,reviewed_reconstructed,own_target}` and "
            "`dsm/native_fetch.json`. Coverage lists every unavailable original "
            "column per shot. Missing raw diagnostic groups, including complete "
            "BES/ECE/actuator records on later shots, prevent a complete native "
            "reconstruction; 50 ms adapter means cannot recover those 1 ms inputs. "
            "78 photodiode records were fetched on 39 otherwise complete-input "
            "shots, one worker, pace 1, with no authentication error."
        ),
    ]
    return "\n".join(lines)


def main():
    files = {
        "ours": OUTPUTS / "ours/evaluation.json",
        "dsm": OUTPUTS / "dsm/evaluation.json",
        "swap": OUTPUTS / "swap/evaluation.json",
        "repeats": OUTPUTS / "ours/seed_repeats.json",
        "strata": OUTPUTS / "ours/annotation_strata.json",
        "filterscopes": OUTPUTS / "filterscope_metadata.json",
        "native": OUTPUTS / "dsm/native_evaluation.json",
        "native_fetch": OUTPUTS / "dsm/native_fetch.json",
    }
    records = {key: json.loads(path.read_text()) for key, path in files.items()}
    ours, dsm, swap, repeats = (
        records[key] for key in ("ours", "dsm", "swap", "repeats")
    )
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
    doc = """# ELMy-phase occupancy: current ELM protocol

## What is measured

The task is **ELMy-phase occupancy** from DIII-D diagnostics. About 97% of
present scored bins are crowd bins (whole ELMing periods); aggregate F1 mainly
measures that target. Non-crowd spans and mixed annotation modes are reported
separately below. **The requested onset deliverable is incomplete:** there are
no independently adjudicated physical onsets, and the onset head is withdrawn.
Span-touch recall cannot establish event precision or millisecond timing accuracy.
A reference swap compares fixed predictions with legacy onset bins and occupancy
conversions of that source.

## Data and splits

There are 119 reviewed shots, all in the fixed cohort train or validation split.
No cohort blind-test shot enters the U-Net or isolated DSM detector's fitting,
normalization, checkpoint selection or threshold selection. Supplemental DSM
source models retain upstream exposure described below. Primary all119 bins use
fetched filterscope/interferometer
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
  0.5 s median contrast, density and 0.2 s high-pass channels. FS01 was excluded
  because the existing native-rate ELM-O cache contains FS02–FS04 only; it was
  not trained or evaluated, rather than being universally unavailable. The
  metadata and corpus-availability audit is `filterscope_metadata.json`.
  See `src/labeler/elm/inputs.py` for scaling and source
  ordinate provenance; no speculative density rescaling is applied.
- **ELM-O:** the paper reimplementation, requiring BES, with hard-call eta 0.997.
  Its rank scores use the saved nested eta sweep on the same evaluated bins.
- **elm-clock:** the original rule that seeded the review; this reference is
  therefore dependent on the clock. The clock has no continuous rank score.
- **elm-dsm detection (isolated):** the fair detection baseline: the same
  60-input, 128-unit embedding and one occupancy logit, trained for 40 epochs
  in the fixed shot folds. It starts from independent random weights and fits
  normalization using measured, usable, labelled rows of the outer training
  partition only. Missing inputs are filled at that partition's mean.
- **elm-dsm refit‡:** a **one-epoch**, 60/124-input survival refit, supplemental
  offline forward-risk score, distinct from a native-checkpoint evaluation.
  Its served checkpoint was selected after the first epoch of a seven-epoch run.
- **elm-dsm detection (exposed), detection init‡:** historical 40-epoch
  detection fits with upstream normalization; the initialized embedding also
  inherits source-fitting and checkpoint-selection exposure. Supplemental only.
- **always present:** the trivial baseline, retained in every table.

FILTERSCOPE_AUDIT

The 60-input deployment adaptations have **no D-alpha input (pcphd02/03
mean-filled); CO2 missing on 75/119**. Survival refit inputs are served as 50 ms
means although it was trained on 1 ms rows; detection fits train and serve on
the reviewed 50 ms-mean rows. The survival refit is an
**offline risk score with 25 ms centered-NBI lookahead (not a causal forecast)**.
The 1 ms training describes the source survival refit; detection heads are
refitted on the reviewed 50 ms-mean rows.
Only the supplemental DSM variants use upstream normalization constants
computed **before the upstream split**, including blind-cohort source shots
**190646 and 190532**.
This is feature-statistics exposure, separate from reviewed-label leakage. The
historical scratch detection variant inherits it too; the new isolated detector
does not load these constants or source weights. Sources:
`dsm/evaluation.json:model_context`, `row_diagnostics`; source notebook
`/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/data_processing.ipynb:4406`.

Physical-shot fitting, selection and normalization membership is retained in
`src/labeler/models/d3d_elm_time_to_event_dsm/training_membership.json`. The
adapter's exposed-shot membership prevents general validation from labelling
these shots held out. DSM source-validation scores used for early stopping are
validation evidence. The supplemental models do not certify blind-cohort
isolation. The serving risk scale is poorly calibrated, and out-of-filter rows
are clipped to normalized ±10 instead of being discarded.

Reviewers labelled from D-alpha after starting from the D-alpha clock. The
U-Net therefore has direct input coupling to the review, while the adapted DSM
inputs are D-alpha-free. Architecture, objective and input choice vary together;
the U-Net–DSM gap cannot be attributed to architecture alone.

## Training and metrics

elm-ours trains 25 epochs per fold, 40 iterations per epoch, batch 16, 4096 ms
crops, learning rate 0.002, weight decay 0.01 and dropout 0.1. Masked event loss
uses present/crowd versus absent milliseconds; uncertain, unobservable and
unlabelled time is ignored. The existing auxiliary span-start loss remains in
the checkpoints, but the onset head is **withdrawn from paper outputs**, with no
onset retraining: span starts are not adjudicated physical onsets and its
agreement was weak. Only occupancy scores and detections are plotted/reported.
Checkpoint selection maximizes finite inner-validation AUPRC; an all-NaN
history now raises a clear error instead of returning an undefined best row.
This single-epoch selection is noisy: original fold 3 selected epoch 2 during
OneCycle warm-up, and fold 4 selected a single validation spike. Across four
seeds thresholds span 0.11–0.91. Full histories and a three-epoch moving-average
selection audit are reported below; rerunning all twenty U-Net fits is deferred
because historical runs retained only the selected checkpoints. Reported scores
retain their original selection rule without selecting a better seed.

A scored 50 ms cell lies wholly inside one known reviewed span and analysed
time; the span must have at least half its duration analysed. Scores are pooled
bin AUROC, average precision (AUPRC), precision, recall and F1. elm-ours bin
scores average the occupancy trace and call a bin present when its mean is
≥ the fold threshold; thresholds maximize inner-validation F1. ELM-O and the
clock use **any-touch** calls: one detected interval touching a bin suffices.
ELM-O detects individual bursts, so an occupied crowd bin containing no detected
burst counts against it under this phase-occupancy benchmark.
Detected runs use a centered 50 ms moving mean. Non-crowd touch recall and absent
touch rates count any interval overlap and do not establish onset accuracy.
Every detected interval is intersected with the panel's analysed coverage before
raw or guarded span counting; detections outside that coverage receive no credit.
Intervals are 95% percentile physical-shot bootstraps with 1000 shared draws;
paired differences use identical draws. † marks recall ≥0.99; numeric F1 and
intervals remain visible, with precision/recall beside them. ‡ marks supplemental
rows containing shots used in source fitting, preprocessing or checkpoint
selection, even when reviewed-label fitting and threshold selection are out of fold.

## Results

### Primary reviewed bins

"""
    doc += benchmark_rows(ours)
    doc += """

Source: `outputs/labeler/elm/ours/evaluation.json:sets.<set>.methods`.
The BES comparison has similar point estimates and no significant difference
detected; this is not an equivalence result. Paired differences and intervals
are in the same record's `paired` fields.

### Crowd and non-crowd targets

"""
    counts = ours["sets"]["all119"]["methods"]["elm-ours"]["counts"]
    positive = counts["tp"] + counts["fn"]
    crowd = counts["crowd_bins"]
    doc += (
        f"Of {positive:,} present bins, {crowd:,} ({100 * crowd / positive:.2f}%) "
        f"are crowd bins; only {positive - crowd} are non-crowd.\n\n"
    )
    doc += kind_rows(ours)
    doc += "\n\n### Annotation-mode shot groups\n\n"
    doc += mode_rows(ours)
    doc += """

Groups use each shot's complete reviewed annotation; metrics use covered bins.
The disjoint counts are 56 crowd-only, 13 non-crowd-only, 20 mixed and 30 with
no present span (119 total). Thus 33 shots contain non-crowd spans, 76 contain
crowds, and 20 are in both. The earlier count of 29 no-present shots omitted
shot 194600, which has absent/uncertain annotations but no scored bins. Thus 29
no-present shots contribute bins while the complete group has 30. No-present
F1/AUROC/recall are undefined; its absent-bin call fraction
is the relevant metric. Every displayed interval resamples physical shots within
its group. Non-crowd-only F1 near 0.42 shows that phase occupancy can fill the
gaps between individually annotated spans. Sources: `ours/evaluation.json:
annotation_modes,sets.<set>.annotation_modes` and `sets.<set>.methods`.

### Common DSM bins

"""
    doc += benchmark_rows(dsm)
    doc += """

Source: `outputs/labeler/elm/dsm/evaluation.json:sets.<set>.methods`.
Input availability, architecture and objective differ together; their effects
cannot be separated causally. In-sample DSM source exposure is disclosed below.

### Survival refit on its own target (supplemental)

"""
    doc += own_target_rows(dsm)
    own = dsm["own_target"]
    doc += (
        f"\n\nThe **one-epoch** 60/124-input refit uses {own['rows']:,} original "
        f"1 ms rows from {own['shots']} physical shots. These are early-stopping "
        "validation rows (called test upstream), not untouched test evidence; "
        "phase records of one shot occur on both source split sides. Intervals "
        "group physical shots. The 40-epoch isolated detection retrain above is "
        "the fair adapted DSM detection baseline. The one-epoch survival adaptation "
        "must not be presented as the native 124-input checkpoint. Source: "
        "`dsm/evaluation.json:own_target`.\n"
    )
    doc += "\n### Native 124-input checkpoint and 1 ms coverage\n\n"
    doc += native_rows(records["native"])
    doc += """

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

The checkpoint audit found that a prespecified trailing three-epoch mean after
warm-up would select a different epoch in 15/20 folds. Fold 3's original epoch-2
AUPRC was 0.961 versus 0.911 in its immediate neighborhood; fold 4's selected
epoch-12 AUPRC was 0.936 versus 0.768 in its neighborhood. The saved artifacts
contain only the selected weights. A smoothed result therefore needs all twenty
folds retrained: observed training alone took 0.945 serial GPU-hours, before
inference and artifact serialization. This is not a cheap checkpoint rescore.
No improved smoothed-model result is claimed. Source: `ours/annotation_strata.json:
checkpoint_selection_audit`.

## Reference-swap protocol and results

Legacy overlap is reported first: the **onset table has 576 shots / 8 reviewed
overlaps**. The shot-level `elm_all_ground_truth` has **365 / 5**, all labelled
yes; shot 192751 has no reviewed present span, so this is not bin truth. The
independent Smith BES windows have **211 / 0** and cannot support this reference swap.
Source: `swap/evaluation.json:overlap`.

The **legacy onset table** uses the `wpqh_elm_hiro`
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
thresholds were not selected again. **Learned-method F1 operating points were
tuned against the review** in inner validation, including the survival refit's
risk threshold. Thus only **AUROC** supports cross-reference method comparisons;
F1 orders describe fixed review-tuned operating points and favor that target.
The review order and every reference's
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
reference. Upstream feature-statistics exposure remains in supplemental models;
the new isolated detector has no source weights or normalization. The small
overlap gives broad intervals.
Numerical AUROC/F1 with intervals and precision/recall for all variants are in
the committed swap tables; no high-recall F1 is hidden.

The legacy onset-bin reference misses **31%** of reviewed-present bins
(M/P = 60/14 on 782 known bins); 200 ms occupancy misses **17%** (34/60).
These are occupancy disagreements, not adjudicated missed physical ELMs.
Across the complete eight-shot overlap, the historical AUROC order and leader
remain unchanged under every conversion: no AUROC reversal was observed.
F1 changes involve review-tuned operating points and the degenerate high-recall
survival refit. With eight shots a reversal can be neither shown nor ruled out;
the three-shot source-unexposed and two-shot BES subsets are still less precise.
The evidence is **inconclusive** and does not establish the AE ranking-reversal
result. Sources: `swap/evaluation.json:interval_audit,interval_audit_occupancy,
swap.<set>.comparison`.

The added isolated DSM arm preserves the complete eight-shot primary AUROC
order as well. On the seven-shot BES companion, however, its AUROC point order
with ELM-O crosses under the 100/200/300 ms occupancy conversions. This new
point crossing does not alter the leader and does not establish a reversal
with this small overlap; it must not be hidden by the historical-family result.
Source: `swap/evaluation.json:swap.overlap_bes.occupancy.<gap>.comparison`.

## Limitations

- The annotations differ between shots: 33 contain non-crowd/per-ELM spans
  and 76 contain whole ELMing-period crowds, with **20 shots in both**.
  Short absent gaps between
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
  `density_units.json`. Round-three filterscope metadata fetching is authorized
  and recorded separately; this does not establish fast-density ordinate units.

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
scripts/labeler/elm_annotation_strata.py --run cv2
scripts/labeler/elm_dsm_evaluate.py --run cv2 --rescore --refresh-report
scripts/labeler/elm_dsm_native.py
scripts/labeler/elm_reference_swap.py --run cv2
scripts/labeler/elm_seed_repeats.py --aggregate
scripts/labeler/elm_paper_tables.py
scripts/labeler/elm_table_proofs.py
scripts/labeler/elm_table_proofs.py --record-viewed
scripts/labeler/elm_example_figure.py --run cv2
scripts/labeler/elm_protocol.py
scripts/labeler/elm_fix3_verify.py
scripts/labeler/elm_fix3_report.py
```

Inspect every rendered table PNG before running `--record-viewed`; the latter
checks unchanged source/render hashes and records that completed inspection.
Seed training uses `elm_seed_repeats.py --train` in the prescribed CUDA venv,
CUDA_VISIBLE_DEVICES=0 with modest resources. Tests use only covering files via
the required pt.sh wrapper; scoped Ruff and format checks are recorded in
`fix_round3_verification.json`. No labels or production stores are modified.

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

Fix round 3 clips detections to panel coverage, adds annotation-mode intervals,
fits a source-isolated 40-epoch DSM detector with fold-local normalization,
audits native 124-input evaluation, and replaces table boilerplate with compact
captions and a shared appendix note. The onset deliverable remains incomplete.
"""
    doc = doc.replace("401,714", f"{parameter_count:,}")
    doc = doc.replace("FILTERSCOPE_AUDIT", filterscope_rows(records["filterscopes"]))
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
        "**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (offline risk score)",
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
        ("elm-dsm-detect-exposed", dsm, "all119"),
        ("elm-dsm-detect-init", dsm, "all119"),
        ("always present", ours, "all119"),
    ):
        subset = source["sets"][tag]
        res = subset["methods"][name]
        label = "BES subset, ELM-O chunks" if tag == "bes73" else "all reviewed shots"
        slugs = {
            "elm-elmo": "elmo",
            "elm-clock": "elm_clock",
            "elm-dsm": "d3d_elm_time_to_event_dsm‡",
            "elm-dsm-detect-exposed": "elm-dsm-detect-exposed‡",
            "elm-dsm-detect-init": "elm-dsm-detect-init‡",
        }
        dates = {
            "elm-elmo": "2026_10_01",
            "elm-clock": "2026_09_13",
            "elm-dsm": "2026_09_06",
        }
        cells = [
            f"{key.upper()}: {metric(res, key)}" for key in ("auroc", "auprc", "f1")
        ]
        role = ""
        if name in ("elm-dsm", "elm-dsm-detect-exposed", "elm-dsm-detect-init"):
            role = "; supplemental"
        elif name == "elm-dsm-detect":
            role = "; 40-epoch isolated detection"
        model_lines.append(
            f"- {slugs.get(name, name)} | {dates.get(name, '2026_10_03')} | "
            + " | ".join(cells)
            + f" ({subset['n_shots']} shots/{subset['bins']:,} 50 ms bins; {label}{role})"
        )
    model_lines += [
        "",
        (
            "Brackets are 95% shot-bootstrap intervals; † marks recall ≥0.99; "
            "‡ marks supplemental source fitting, preprocessing or selection exposure. "
            "always present precision equals prevalence and recall is 1; full numeric "
            "precision/recall and intervals are in the tables and protocol."
        ),
        "",
        (
            "The task is ELMy-phase occupancy: about 97% of present bins are crowds. "
            "The U-Net and isolated 40-epoch DSM use five physical-shot-grouped folds "
            "and exclude cohort blind-test shots from fitting/preprocessing/selection. "
            "ELM-O needs BES; reviewers labelled from D-alpha after starting from "
            "elm-clock. U-Net inputs share D-alpha with that reference; adapted DSM "
            "inputs are D-alpha-free, so their gap also reflects input choice. "
            "Primary and common DSM bin sets differ and are labelled "
            "separately. Sources: `outputs/labeler/elm/{ours,dsm}/evaluation.json:sets`."
        ),
        "",
        "Per-kind recall (95% shot-bootstrap intervals):",
        "",
        kind_rows(ours),
        "",
        "Annotation-mode groups, elm-ours (same group bootstrap):",
        "",
        mode_rows(ours),
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
            "There are 33 shots with non-crowd spans and 76 with crowds; 20 are in "
            "both (56 crowd-only, 13 non-crowd-only, 20 mixed, 30 no-present). "
            "One no-present shot contributes no scored bins; 29 contribute bins. "
            "Onset deliverable incomplete: no independently adjudicated onsets; "
            "onset head withdrawn from paper outputs. Checkpoints retain its "
            "auxiliary loss. Hard calls use bin-mean ≥ fold threshold for elm-ours "
            "and any-touch for ELM-O/clock. The latter burst detector is penalized "
            "when a crowd bin contains no burst."
        ),
        "",
        (
            "The adapted elm-dsm variants have no D-alpha input (pcphd02/03 "
            "mean-filled); CO2 missing on 75/119. The survival adaptation is a "
            "one-epoch 60/124-input refit (selected after the first of seven run "
            "epochs), served on 50 ms means. "
            "The 1 ms training describes the source survival refit; detection heads "
            "train on reviewed 50 ms-mean rows. "
            "The refit is an offline risk score with 25 ms centered-NBI lookahead "
            "(not a causal forecast). Supplemental survival, historical scratch "
            "and source-initialized detection use upstream normalization constants "
            "computed before the upstream split, "
            "including blind-cohort source shots 190646 and 190532 (feature-statistics "
            "exposure). The isolated detector fits preprocessing within each outer "
            "training partition and uses independent random initialization. "
            "Refit and initialized detection also inherit source fitting; "
            "five overlap shots (190637,190643,192721,192751,196541) are in-sample. "
            "Physical membership is propagated to the adapter so exposed shots are "
            "not called held out. Source: `dsm/evaluation.json:model_context,own_target`."
        ),
        "",
        "Survival refit own-target AUROC (early-stopping validation; supplemental):",
        "",
        own_target_rows(dsm),
        "",
        "Native 124-input checkpoint, 1 ms evaluation (separate coverage):",
        "",
        native_rows(records["native"]),
        "",
        filterscope_rows(records["filterscopes"]),
        "",
        (
            "Inner-validation checkpoint selection is noisy: epoch 2 during warm-up "
            "in original fold 3, a single fold-4 spike, thresholds 0.11–0.91 across "
            "seeds. A trailing three-epoch mean chooses different endpoints in 15/20 "
            "folds; only selected weights were saved, so a smoothed result requires "
            "20-fold retraining (observed training 0.945 GPU-hours). Source: "
            "`ours/annotation_strata.json:checkpoint_selection_audit`."
        ),
        "",
        (
            "Legacy overlaps: onset table 576 shots/8 overlap; elm_all_ground_truth "
            "365/5 (all yes; no reviewed present span on 192751); Smith BES windows 211/0. "
            "The swap reports onset bins beside occupancy "
            "conversion at 100/200/300 ms gap tolerances, with M/P/recall and rankings "
            "under every reference. A separate three-shot DSM-source-unexposed analysis "
            "retains supplemental normalization exposure. Legacy onset bins miss 31% "
            "of reviewed-present bins, versus 17% at 200 ms occupancy (M/P 60/14 "
            "and 34/60 on 782 known bins). Historical AUROC order and leader are "
            "unchanged under every conversion; no reversal observed. F1 uses "
            "review-tuned thresholds, so only AUROC compares across references. "
            "With eight shots a reversal can be neither shown nor ruled out: "
            "evidence inconclusive. The added isolated DSM arm also preserves "
            "primary eight-shot AUROC order, but crosses ELM-O's point AUROC on "
            "the seven-shot BES occupancy companions; leadership is unchanged "
            "and a statistically supported reversal is not established. "
            "Three full new-seed CV repeats "
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
    rendered = text[:begin] + "\n".join(model_lines) + text[end:]
    rendered = rendered.replace(
        "Every variant uses upstream\nnormalization constants computed before its split, "
        "including blind-cohort\nsource shots 190646 and 190532.",
        "The survival adapter and supplemental historical detection variants use "
        "upstream normalization including blind-cohort shots 190646 and 190532. "
        "The isolated detection retrain fits preprocessing within each outer "
        "training partition and uses independent random weights.",
    )
    readme.write_text(rendered)
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
