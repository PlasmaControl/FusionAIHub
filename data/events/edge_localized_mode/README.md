# Edge Localized Mode

## Description
Edge localized modes (ELMs) are quasi-periodic relaxations of the H-mode pedestal where the steep edge pressure gradient and the bootstrap current it drives cross the coupled peeling-ballooning stability boundary causing rapid filamentary bursts to expel stored energy in ~1 ms.

ELMs were first observed on ASDEX with the discovery of the H-mode (1982).

Type-I ELMs have frequencies of tens to a few hundred Hz on DIII-D and their frequency typically rises with heating power. Type-III ELMs are smaller,
faster and appear near the L-H power threshold. Breakthrough ELMs are large ELMs that occur during wide pedestal quiescent high confinement (WPQH) experiments.

These are typically found via filterscope D-alpha bursts, divertor Langmuir probes, magnetics and BES. Each ELM is a burst on the divertor D-alpha signal 2-5 ms wide.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `elm_labels_dict.pkl`, `elm_labels_dict_wpqh.pkl`, `elm_survival_labels.pkl`, `elm_survival_labels_wpqh.pkl`

**Author**: Hiro Farre Josep Kaga

**Description**: The original `wpqh_elm_hiro` label pickles are now in `raw/`. The separate D-alpha clock described above remains a producer.

**Publications**:

### Dataset 2

**Dataset File(s)**: `labeled-elm-events.hdf5`, `labeled_elm_events_long_windows_20220921.hdf5` (read in place from `/projects/EKOLEMEN/dsmith/data/`, not copied)

**Author**: David Smith

**Description**: Hand-labelled ELMs on the 64-channel BES at 1 MHz: 481 events on 164 shots, and 2,008 events on 199 shots in longer windows. Each event is a short window holding one ELM (BES `signals`, `time` in ms, a per-sample `labels` flag for the ELM region, shot number). The ELM-O detector was tuned on a hand-labelled set of this kind (972 ELMs); which events are in it is not recorded. These windows (2,316 after dropping 173 repeats, on 211 shots, none of them in the corpus or the review labels) are the truth the ELM-O benchmark is scored on: the re-implementation reproduces the published precision and recall on them (0.997 and 0.980; paper 0.995 and 0.976). Not yet ingested as a label source here.

**Publications**:

### Dataset 3

**Dataset File(s)**:

**Author**:

**Description**: The elm-dsm survival refit uses the ELM-survival rows of `wpqh_elm_hiro` with 60 non-BES columns. Its served output is an offline risk score with centered-NBI lookahead; see the current protocol for input, normalization and physical-shot exposure.

**Publications**:

## Models

**stable**: elm-dsm | 2026_09_06 (offline risk score)

**latest**: elm-ours | 2026_10_03

**all**:

- elm-ours | AUROC: 0.941 [0.906, 0.967]; AUPRC: 0.877 [0.774, 0.949]; F1: 0.830 [0.778, 0.871] (119 shots/12,409 50 ms bins; all reviewed shots)
- ELM-O | AUROC: 0.918 [0.877, 0.951]; AUPRC: 0.833 [0.753, 0.890]; F1: 0.842 [0.789, 0.885] (73 shots/6,843 50 ms bins; BES subset, ELM-O chunks)
- elm-clock | AUROC: --; AUPRC: --; F1: 0.757 [0.680, 0.827] (119 shots/12,409 50 ms bins; all reviewed shots)
- elm-dsm refit | AUROC: 0.777 [0.730, 0.824]; AUPRC: 0.662 [0.592, 0.736]; F1: 0.627 [0.559, 0.692] (119 shots/11,653 50 ms bins; all reviewed shots)
- elm-dsm detection | AUROC: 0.850 [0.802, 0.895]; AUPRC: 0.761 [0.671, 0.834]; F1: 0.743 [0.680, 0.798] (119 shots/11,653 50 ms bins; all reviewed shots)
- elm-dsm detection init | AUROC: 0.863 [0.817, 0.905]; AUPRC: 0.792 [0.710, 0.857]; F1: 0.738 [0.675, 0.793] (119 shots/11,653 50 ms bins; all reviewed shots)
- Always-present | AUROC: 0.500 [0.500, 0.500]; AUPRC: 0.376 [0.322, 0.430]; F1: 0.546 [0.487, 0.601]† (119 shots/12,409 50 ms bins; all reviewed shots)

Brackets are 95% shot-bootstrap intervals; † marks recall ≥0.99. Always-present precision equals prevalence and recall is 1; full numeric precision/recall and intervals are in the tables and protocol.

All detector fits use five physical-shot-grouped folds and exclude cohort blind-test shots from new fitting/selection. ELM-O needs BES; elm-clock seeded the review. Primary and common DSM bin sets differ and are labelled separately. Sources: `outputs/labeler/elm/{ours,dsm}/evaluation.json:sets`.

Raw / 25 ms edge-guarded absent-span touch rates (same denominator):

- elm-ours, all119: 0.596 [0.478, 0.689] / 0.427 [0.343, 0.495]
- elm-ours, bes73: 0.610 [0.452, 0.742] / 0.436 [0.351, 0.511]
- ELM-O, bes73: 0.440 [0.301, 0.623] / 0.307 [0.200, 0.444]

Centered 50 ms smoothing can spill detected runs across span edges. Short empty guarded interiors are counted separately; the protocol also gives the rate restricted to nonempty interiors. These are annotation disagreements, not independently verified physical false alarms. The occupancy target differs between 33 per-ELM/non-crowd-annotated shots and 76 crowd-annotated shots. The unsupported onset head is dropped from paper outputs; checkpoints retain its auxiliary training loss.

Every elm-dsm variant has no D-alpha input (pcphd02/03 mean-filled); 50 ms-mean serving of a 1 ms-trained model; CO2 missing on 75/119. The 1 ms training describes the source survival refit; detection heads train on reviewed 50 ms-mean rows. The refit is an offline risk score with 25 ms centered-NBI lookahead (not a causal forecast). Every variant, including scratch detection, uses upstream normalization constants computed before the upstream split, including blind-cohort source shots 190646 and 190532 (feature-statistics exposure). Refit and initialized detection also inherit source fitting; five overlap shots (190637,190643,192721,192751,196541) are in-sample. Physical membership is propagated to the adapter so exposed shots are not called held out. Source: `dsm/evaluation.json:model_context,own_target`.

The legacy onset table uses Hiro Farre Josep Kaga annotations, compiled by labels_format.py/source_formatters, and overlaps eight reviewed shots. The swap reports onset bins beside occupancy conversion at 100/200/300 ms gap tolerances, with M/P/recall and rankings under every reference. A separate three-shot DSM-source-unexposed analysis retains the normalization disclosure. Three full new-seed CV repeats report spread without choosing a best seed. Sources: `swap/evaluation.json` and `ours/seed_repeats.json`.

Current protocol, all intervals, seed results, limitations and artifacts: [elm_ours.md](../../../docs/labeler/elm_ours.md). ELM-O's independent BES-window benchmark is described in [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md).

## Inputs
**elm-dsm** (internal adapter slug `d3d_elm_time_to_event_dsm`):
- `ip`, `bt`, `gas`, `pinj`, `tinj`, `ech`
- `co2_density_slow` `r0`, `v1`, `v2`, `v3`
- `ece_slow` (48 ch)
- `pcphd02`, `pcphd03` (always mean-filled)

**elm-clock**:
- `D-alpha FS01..FS08` (first finite channel)
- `CO2 density R0`, `pinj_total` (L-mode gate, `dalpha_lh`)
- `Ip` (plasma start)

**ELM-O**:
- `interferometer` chords `DENV2F`, `DENV3F` (`\BCI::`, 100 kS/s)
- `filterscopes` FS02, FS03, FS04 (`\SPECTROSCOPY::`, 50 kS/s)
- `BES` (64 ch; 500 kS/s in the corpus, 1 MS/s in Smith's windows)

**elm-ours**:
- `filterscopes` FS02, FS03, FS04 (`\SPECTROSCOPY::`, 50 kS/s; log level and contrast to a 0.5 s running median)
- `interferometer` chords `DENV2F`, `DENV3F` (`\BCI::`, 100 kS/s; density and its 0.2 s high-pass)
- a validity mask; 10 kHz grid, no BES

**elm_frames (round three)**:
- `D-alpha FS` (the ELM spans' channel)

## Method
The elm-clock detector reads the eight real D-alpha filterscope channels
(10 kHz): each min-max-normalized filterscope signal is smoothed (0.64 ms) and
peaks are picked with prominence 0.03 and a minimum separation of 3 ms
(`elmcycle.detect_elms`, ported). A candidate is accepted only if its
half-prominence width is <= 5 ms (`DALPHA_MAX_WIDTH_MS`), which removes broad
signal humps without verifying their physical origin. Each accepted burst is a point event
`elm`; the clock also writes `elm_free` intervals (rate < 5 Hz over a 100 ms window
for >= 50 ms) and the ELM rate. Coverage is the D-alpha span actually processed.

elm-dsm writes offline risk scores for an ELM within 5, 10, 20 and 50 ms.
Centered NBI preprocessing includes 25 ms lookahead, so these are not causal
forecasts and do not report observed events. Every variant uses upstream
normalization constants computed before its split, including blind-cohort
source shots 190646 and 190532.

Known gap: a narrow ELM riding on a broad D-alpha hump measures wide and is dropped.

Benchmark: the rule-based ELM-O detector (O'Shea et al. 2023: interferometers, filterscopes and BES, no learning) is the benchmark for these labels. It was re-implemented from the paper (the public code has no licence) and scored on David Smith's labelled windows and on the reviewed ELM spans, non-crowd present spans and crowd spans apart: [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md). It needs BES, which 73 of the 119 review shots have in the corpus.

`elm-ours` (`labeler.elm`) is a 1D U-Net (401,714 parameters) that reads
filterscopes and fast density without BES and writes per-millisecond occupancy
probability. It uses five shot-grouped folds over the 119 reviewed shots,
with three additional training-seed repeats on the same partitions. The
auxiliary span-start head is dropped from paper outputs; the existing
checkpoints retain its loss. Fast-density values retain their numerical scale;
physical ordinate units were not retained in the original source cache and are
explicitly unverified, rather than speculatively rescaled.

## Alias
edge localized mode, edge localised mode, elm, elms, elmy, elming

## Future Implementations
- Ingest David Smith's labelled ELMs, and David Eldon's ELM dataset when it arrives, as further sources beside the breakthrough ELMs of `wpqh_elm_hiro` (the crowd flag distinguishes non-crowd annotations from ELMing periods)

## Reference
- H. Zohm, "Edge localized modes (ELMs)", Plasma Phys. Control. Fusion 38, 105
  (1996).
- A. W. Leonard, "Edge-localized-modes in tokamaks", Phys. Plasmas 21, 090501
  (2014).
- F. H. O'Shea, S. Joung, D. R. Smith and R. Coffee, "Automatic identification of
  edge localized modes in the DIII-D tokamak", APL Mach. Learn. 1, 026102 (2023).

## Contact
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
- **Semin Joung**:
- **Hiro Farre Josep Kaga**:
- **Jalal Butt**:
