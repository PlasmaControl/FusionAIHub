"""Render sawtooth result tables and data scope from recorded evaluation JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sawtooth_physics import REPO, WORK, records_at, save_json

OUTPUT = REPO / "outputs/labeler/sawtooth"


def read(name):
    return json.loads((OUTPUT / name).read_text())


def number(value):
    return "unavailable" if value is None else f"{value:.3f}"


def scored(value, interval):
    text = number(value)
    if interval is not None:
        text += f" [{interval[0]:.3f}, {interval[1]:.3f}]"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    code_files = sorted((REPO / "src/labeler/sawtooth").glob("*.py"))
    code_files += [
        REPO / "scripts/labeler" / name
        for name in (
            "sawtooth_physics.py",
            "sawtooth_benchmark.py",
            "sawtooth_results.py",
        )
    ]
    fingerprints = {
        str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in code_files
    }
    provenance_path = OUTPUT / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["sha256"] != fingerprints:
            raise ValueError(
                "Source changed after recorded analysis; rerun stages before replacing provenance"
            )
    else:
        save_json(
            provenance_path,
            {
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                ).strip(),
                "sha256": fingerprints,
                "python": platform.python_version(),
                "packages": {
                    name: version(name)
                    for name in (
                        "numpy",
                        "scipy",
                        "pandas",
                        "h5py",
                        "torch",
                        "matplotlib",
                    )
                },
                "population_slurm_array": 2952132,
                "note": "Training/inference functions unchanged since initial implementation; final evaluation uses shared absolute bin grid. CPU execution.",
            },
        )
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    records = {r["shot"]: r for r in records_at(WORK, cohort.shot)}
    splits = {}
    for split in ("train", "val", "test"):
        shots = sorted(int(s) for s in cohort[cohort.split == split].shot)
        kept = [records[s] for s in shots]
        splits[split] = {
            "shots": shots,
            "shot_count": len(shots),
            "crashes": sum(len(r["crashes"]) for r in kept),
            "intervals": sum(len(r["intervals"]) for r in kept),
            "positive_shots": sum(bool(r["intervals"]) for r in kept),
            "mirnov_available_shots": sum(r["mirnov_available"] for r in kept),
            "ip_available_shots": sum(r["ip_available"] for r in kept),
        }
    save_json(OUTPUT / "data_summary.json", {"splits": splits})
    validation = read("validation.json")
    benchmark = read("benchmark.json")
    population = read("population_labels.json")
    text = [
        "# Sawtooth evaluation records",
        "",
        (
            "These are research candidates: weak expert validation and unavailable "
            "radial calibration preclude a stable-model recommendation."
        ),
        "",
        (
            "All model predictions below are out of fold on fixed train-cohort shots. "
            "Fixed validation shots select checkpoints and crash thresholds. "
            "All expert-reviewed shots and all fixed test shots are excluded from "
            "training and tuning. Intervals are 95% shot-bootstrap percentiles "
            "(1000 replicates). Presence bins are 2 ms, threshold 0.5."
        ),
        "",
        "## Data",
        "",
        "| Fixed split | Shots | Crash candidates | Train spans | Positive shots |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in splits.items():
        text.append(
            f"| {split} | {row['shot_count']} | {row['crashes']} | "
            f"{row['intervals']} | {row['positive_shots']} |"
        )
    text += [
        "",
        (
            "Source: `outputs/labeler/sawtooth/data_summary.json` and "
            "`outputs/labeler/sawtooth/cohort_labels.json`."
        ),
        "",
        (
            f"Population: {population['requested_count']} corpus files inspected; "
            f"{len(population['processed_shots'])} usable ECE shots, "
            f"{population['crashes']} crash candidates, {population['intervals']} "
            f"train spans, {len(population['errors'])} absent/unusable/corrupt records. "
            "Source: `outputs/labeler/sawtooth/population_labels.json`."
        ),
        "",
        "## Physics labels versus expert spans",
        "",
        "| Metric | Value [95% CI] |",
        "|---|---:|",
    ]
    expert = validation["expert"]
    for key in ("precision", "recall", "f1"):
        text.append(
            f"| Bin {key} | "
            f"{scored(expert['presence'][key], expert['ci95']['presence_' + key])} |"
        )
    for key in ("precision", "recall", "f1"):
        span = expert["span_matching"]
        text.append(
            f"| Span {key}, IoU >= 0.1 | "
            f"{scored(span['metrics'][key], span['ci95'][key])} |"
        )
    text += [
        "",
        (
            f"Any-overlap span recall: {number(expert['span_overlap_recall'])}. "
            f"Supported picks: {expert['crash']['supported_picks']}/"
            f"{expert['crash']['assessed_picks']} inside positive expert spans. "
            "This is not crash precision. True crash precision/recall/F1 are "
            "unavailable because expert point times were not supplied. "
            "All known spans on shots 186636, 189324 and 190637 are used; "
            "category >=2 abstains and waveform coverage clips the assessment. "
            "Source: `outputs/labeler/sawtooth/validation.json`."
        ),
        "",
        "## Legacy / Tokamak-SI model results",
        "",
        (
            "Legacy HL-3 figures are published three-regime window classification: "
            f"real-time stated accuracy {number(benchmark['legacy']['real_time']['accuracy_stated'])} "
            f"(count-derived {number(benchmark['legacy']['real_time']['accuracy_from_counts'])}); "
            f"offline stated accuracy {number(benchmark['legacy']['offline']['accuracy_stated'])} "
            f"(count-derived {number(benchmark['legacy']['offline']['accuracy_from_counts'])}). "
            "The source is OuYang et al., PPCF 67 (2025) 105004. "
            "No published crash-tolerance score or interval CI is available."
        ),
        "",
        (
            "| Tokamak-SI model | Crash F1, 1 ms | Crash F1, 2 ms | "
            "Train-bin AUROC | Train-bin AUPRC | Train-bin F1 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in benchmark["Tokamak-SI"].items():
        one, two = row["crash_tolerance_1ms"], row["crash_tolerance_2ms"]
        values = [
            scored(one["crash"]["f1"], one["ci95"]["crash_f1"]),
            scored(two["crash"]["f1"], two["ci95"]["crash_f1"]),
        ]
        values += [
            scored(two["presence"][k], two["ci95"]["presence_" + k])
            for k in ("auroc", "auprc", "f1")
        ]
        text.append(f"| {name} | " + " | ".join(values) + " |")
    baseline = benchmark["Tokamak-SI"]["saw-hl3"]
    text += [
        "",
        (
            f"DIII-D baseline three-class window accuracy: "
            f"{scored(baseline['three_class_window_accuracy'], baseline['three_class_accuracy_ci95'])}. "
            "The smaller/longer period boundary is fitted separately in each "
            "training fold. ECE replaces the HL-3 SXR pair. Its crash score uses "
            "a regime-gated derivative picker because the published architecture "
            "does not output point times. These task/input adaptations prevent "
            "direct comparison with the legacy accuracy."
        ),
        "",
        (
            "Source for every model value: `outputs/labeler/sawtooth/benchmark.json`. "
            "Exact shots, boundaries, thresholds and training budgets are in "
            "`split_manifest.json` and each model's `*_fold_*.json` records."
        ),
        "",
        "## Independent expert check of trained models",
        "",
        "| Three-fold ensemble | Train-bin AUROC | Train-bin AUPRC | Train-bin F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in benchmark["Tokamak-SI"].items():
        expert = row["expert"]
        values = [
            scored(expert["presence"][k], expert["ci95"]["presence_" + k])
            for k in ("auroc", "auprc", "f1")
        ]
        text.append(f"| {name} | " + " | ".join(values) + " |")
    text += [
        "",
        (
            "Same reviewed shots/known bins as detector validation. True expert "
            "crash scores remain unavailable. Three shots yield fragile confidence "
            "intervals. Source: `outputs/labeler/sawtooth/benchmark.json`, expert blocks."
        ),
        "",
        (
            "See [the method and limitations](sawtooth_physics.md) for reproduction, "
            "signal coverage and scientific qualifications."
        ),
        "",
    ]
    (REPO / "docs/labeler/sawtooth_results.md").write_text("\n".join(text))
    if args.report is not None:
        full_report(args.report, "\n".join(text))


def full_report(path, results):
    """Dispatch report, including provenance and the limits on scientific claims."""
    commits = subprocess.check_output(
        ["git", "log", "21183f0..HEAD", "--format=%h %s"], cwd=REPO, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()
    validation, _benchmark, cohort = (
        read("validation.json"),
        read("benchmark.json"),
        read("cohort_labels.json"),
    )
    split, gallery = read("split_manifest.json"), read("gallery.json")
    legacy = validation["legacy_agreement"]
    tails = []
    for name in ("tests.log", "lint.log", "format.log", "tmpsweep.log"):
        file = WORK / name
        tails.append(
            f"### {name}\n\n```text\n"
            + "\n".join(file.read_text().splitlines()[-8:])
            + "\n```\n"
        )
    folds = []
    for name in ("saw-hl3", "saw-ours"):
        for fold in range(3):
            row = read(f"{name}_fold_{fold}.json")
            folds.append(
                f"| {name} | {fold} | {len(row['training_shots'])} | {len(row['heldout_shots'])} | "
                f"{row['best_epoch']} | {row['period_boundary_ms']:.3f} | "
                f"{row['selected_crash_threshold']['threshold']} | {row['selected_crash_threshold']['z']} |"
            )
    report = (
        f"""# Stream saw: implementation report

Status: **DONE_WITH_CONCERNS**. Commit range: `21183f0..{head}` on `r4-saw`.
The implementation, cohort/population exports, trained models, confidence
intervals, gallery and documentation are complete. The method does not demonstrate
improved labels against expert spans. Physical radius validation and expert point
crash scoring remain unavailable, and no stable model is recommended.

## Worktree and commits

Initial `git status --short` was empty on branch `r4-saw`. There were no partial
edits from the stopped implementer to keep or revert. The main checkout, manuscript,
production corpus, existing label stores, HF_HOME and dependencies were not modified.
No push, merge, rebase, fetch, stash or destructive repository cleanup was performed.
The two pilot output directories were moved aside under this stream's output root
after the core-proxy refinement; they are not used by any reported evaluation.

```text
{commits}
```

Every commit carries the required `labeler:` prefix and Codex coauthor trailer.

## What was built

- `src/labeler/sawtooth/physics.py`: frozen `Rule`, `posr`, `noise_calibration`,
  `inversion_profile`, `trains` and `detect`; additive point and train-span events,
  missing-data support checks, optional q/current/geometry/D-alpha/SXR inputs.
- `src/labeler/sawtooth/models.py`: paper-based `HL3`, `PhasePicker`, and
  truncated Gaussian point targets, independently reimplemented from digests.
- `src/labeler/sawtooth/metrics.py`: absolute-time bin centers, one-to-one point
  and interval matching, histogram AUROC/AP and shot-bootstrap aggregation.
- `scripts/labeler/sawtooth_physics.py`: bounded-memory local-store reader,
  resumable cohort/population processing, compact cohort CSV exports, validation,
  and gallery. `sawtooth_population.sbatch` runs four disjoint population shards.
- `scripts/labeler/sawtooth_benchmark.py`: training-only normalizers and class
  boundaries, shot-grouped CV, fixed-validation checkpoint/threshold selection,
  held-out predictions, independent expert ensembles, and evaluation JSON.
- `scripts/labeler/sawtooth_results.py`: regenerates the data-scope JSON, results
  document and this report directly from evaluation records.
- Focused regression tests in `tests/labeler/test_sawtooth_physics.py`; category
  README model/input/method lines; `docs/labeler/sawtooth_physics.md` and
  `docs/labeler/sawtooth_results.md`; compact CSV shards under
  `data/events/sawtooth_oscillation/extend_saw_physics/`.

Read all seven required literature digests, all three owner IDL picker files,
the existing detector/geometry code, and the referenced omnimode source. The
IDL routines are derivative-threshold/holdoff pickers, not Gude profile tests;
none was copied. No external repository code or PDFs were needed.

## Data, splits, thresholds and records

The fixed cohort is unchanged. Exact shot lists are in
`outputs/labeler/sawtooth/cohort_labels.json` and `split_manifest.json`.
There are {len(split["training_cohort"])} training-cohort shots, {len(split["validation_shots"])}
validation shots after excluding all {len(split["expert_shots"])} reviewed shots, and
{len(split["blind_test_excluded"])} blind test shots excluded from the benchmark.
The reviewed shots 186636, 189324 and 190637 all belong to the fixed validation
split and enter neither training nor checkpoint/threshold selection. Algorithmic
labels were generated on all cohort shots, including test, as requested; no
model fitting, tuning, gallery selection or model prediction used test shots.

The detector uses the fixed cohort plasma windows; outside-cohort population
windows use local Ip or ECE extent. Noise and gallery seeds, bin sizes, coverage,
period limits, q/current cutoffs and every detector threshold are recorded in
`cohort_labels.json`, `noise_calibration.json`, `validation.json` and the method
document. Global 2 ms bin centers make expert detector/model comparisons use
identical assessed bins. Probabilities are quantized to 512 bins for AUROC/AP;
1000 shot resamples form 95% intervals. Point matching uses +/-1 and +/-2 ms.

Local q-min was usable on {cohort["qmin_shots"]} cohort shots, and SXR on
{cohort["sxr_shots"]}. Calibrated ECE radius/q=1 mapping was unavailable on
every cohort shot: inversion channel is recorded, physical radius is null.
The synthetic tests exercise calibrated-radius acceptance, missing-surface
rejection and radius mismatch, but those are not real-data validations.

| Model | Fold | Train shots | Held-out shots | Best epoch | Period boundary (ms) | Crash gate | Derivative z |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(folds)
        + """

Sources: each `outputs/labeler/sawtooth/saw-*_fold_*.json`. Presence threshold
is fixed at 0.5. Crash thresholds are selected only using the excluded fixed
validation shots. The legacy classifier's timing adaptation is documented.

## Results with source JSON

"""
        + results.replace("# Sawtooth evaluation records\n", "")
        + f"""

### Agreement with the existing detector

Relative to the read-only production `ece_sawtooth` point rows (ECE/SXR union),
at +/-2 ms over the same cohort waveform coverage:

- Precision {scored(legacy["crash"]["precision"], legacy["ci95"]["crash_precision"])}.
- Recall {scored(legacy["crash"]["recall"], legacy["ci95"]["crash_recall"])}.
- F1 {scored(legacy["crash"]["f1"], legacy["ci95"]["crash_f1"])}.

These are detector agreement, not ground-truth accuracy. Exact shots and any
missing production files are recorded in `outputs/labeler/sawtooth/validation.json`.

## Artifacts and figures

Large outputs are under `{WORK}/`: `shots/` (per-shot JSON), `signals/` (cohort
ECE and adapted baseline inputs), `population_labels/` (full population CSVs),
`population_shard_*_labels/` (shard exports), `models/<model>/fold_<n>/checkpoint.pt`,
`predictions/<model>/fold_<n>/<shot>.npz`, and `gallery/`. No write fallback was
needed; `.r4out/` was not created. Temporary files used the assigned scratch
TMPDIR. Full profile-test attributes remain in the external per-shot records;
committed CSVs retain only inversion radius/channel and period attributes to
keep them reviewable and below the per-file size limit.

Gallery selection: uniform random usable nonreview train shots, without
replacement, seed {gallery["seed"]}. All final PNGs were opened and inspected.
They show both visible trains and misses/extra candidates. Inversion radius
versus q=1 is explicitly unavailable; the lower panel displays inversion
channels and available EFIT q-min. No version labels occur in figure text.
Each figure is a vector PDF plus 150-dpi PNG at 3.5-inch width, fonts >=7 pt.
The full list follows (source: `outputs/labeler/sawtooth/gallery.json`):

"""
        + "\n".join(f"- `{p}`" for p in gallery["figures"])
        + """

## Tests, lint and commands

Tests ran only the covering file, through the required wrapper:

```bash
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh /scratch/gpfs/nc1514/FusionAIHub-r4-saw tests/labeler/test_sawtooth_physics.py -q -p no:cacheprovider
```

Ruff check covered the new package, all three Python scripts, and the focused
test file. Ruff format --check covered the same files. Shell syntax was checked
with `bash -n scripts/labeler/sawtooth_population.sbatch`. No full suite ran.
The final output tails follow:

"""
        + "\n".join(tails)
        + """

All Python commands used the required `pixi run --frozen --no-install
--manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker`
prefix, with worktree PYTHONPATH, assigned TMPDIR, read-only label table root,
and LABELER_NO_FETCH=1. Exact stage commands are in the method document.
Population SLURM array: 2952132, four tasks; final unsharded command consolidated
their resumable records. Required temporary sweeps ran after long jobs.

## Deviations, concerns and recommended choices

1. **Label quality is not improved by the available independent evidence.**
   The frozen rule produces false candidates and misses, and weak expert span
   scores. The nonreview train-shot pilot led to the explicit core-proxy guard
   before freezing; expert validation and blind shots did not retune the rule.
   Keep this source additive and investigational; do not promote it to stable.
2. **Physical radius validation could not run.** No verified local calibrated
   ECE psi/channel mapping was available. EFIT q-min is applied conservatively,
   and gallery misses suggest reconstruction bias can veto real sawtooth trains.
   No q=1 radius was guessed from channel number or q-min. After supplying
   verified local `ece_psi` and `qpsi`, run the following exact command with the
   environment exports in the method document, using a new output directory:

   ```bash
   pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/sawtooth_physics.py labels --workers 8 --work /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/calibrated
   ```

3. **Expert crash recall/precision cannot be measured.** Expert spans contain no
   point times; null metrics and pick-in-positive-span support fractions are
   reported separately. Only three reviewed shots yield fragile bootstrap CIs.
4. **HL-3 task/architecture ambiguities require adaptations.** Use one
   bidirectional 32-unit LSTM and table learning rate 0.001, four real-time
   proxy inputs, training-fold median period boundary, and a separately gated
   derivative picker for crash timing. Published count-derived and stated
   accuracies disagree and are both retained; they are not compared as the same
   metric as crash F1. AMPD was read but not used to replace the requested physics
   supervision. Time-shift augmentation was omitted to avoid mislabeled boundary
   windows; amplitude and noise augmentation are implemented.
5. **CPU training completed instead of GPU training.** The required labelmaker
   environment exposes CPU-only PyTorch even though GPUs are present. No package
   installation or network request was made. Both models trained successfully
   with the documented 20-epoch/80-step budget and early stopping, four CPU
   threads, and modest memory. No training step remains deferred.
6. **SXR is corroboration, not a spatially validated independent detector.**
   No verified SXR radial calibration is available. Gude's two-sided asymmetry
   test, inverted core ECE handling, and pellet identification are not claimed.
   First sampling at 50 kHz lacks antialias filtering; population fallback windows
   and ECE-cutoff observability also require further validation.
7. **Population errors are explicit.** The source corpus includes absent,
   insufficient or corrupt/truncated files; they were not modified or repaired.
   Their exact shot/error ledger accompanies the population JSON.

## Next work

Obtain verified ECE/SXR radial calibration and reliable MSE/EFIT uncertainty,
then validate full spatial redistribution and q=1 consistency. Label individual
crash times on more held-out expert shots, especially gallery misses and
startup/shutdown false candidates. Compare native-rate antialiased filtering
against the current preparation. Freeze a better rule on a separate development
set, reserve new experts for independent testing, then retrain both architectures
with a larger budget. Stronger consistency against candidate labels alone does
not establish better physical labels or a better independent model.
"""
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)


if __name__ == "__main__":
    main()
