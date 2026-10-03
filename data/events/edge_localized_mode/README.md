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

**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (offline risk score)

**latest**: elm-ours | 2026_10_03

**all**:

- elm-ours | 2026_10_03 | AUROC: 0.941 [0.906, 0.967] | AUPRC: 0.877 [0.774, 0.949] | F1: 0.830 [0.778, 0.871] (119 shots/12,409 50 ms bins; all reviewed shots)
- elmo | 2026_10_01 | AUROC: 0.918 [0.877, 0.951] | AUPRC: 0.833 [0.753, 0.890] | F1: 0.842 [0.789, 0.885] (73 shots/6,843 50 ms bins; BES subset, ELM-O chunks)
- elm_clock | 2026_09_13 | AUROC: -- | AUPRC: -- | F1: 0.757 [0.680, 0.827] (119 shots/12,409 50 ms bins; all reviewed shots)
- d3d_elm_time_to_event_dsm‡ | 2026_09_06 | AUROC: 0.777 [0.730, 0.824] | AUPRC: 0.662 [0.592, 0.736] | F1: 0.627 [0.559, 0.692] (119 shots/11,653 50 ms bins; all reviewed shots; supplemental)
- elm-dsm-detect | 2026_10_03 | AUROC: 0.855 [0.813, 0.896] | AUPRC: 0.778 [0.696, 0.844] | F1: 0.754 [0.696, 0.811] (119 shots/11,653 50 ms bins; all reviewed shots; 40-epoch isolated detection)
- elm-dsm-detect-exposed‡ | 2026_10_03 | AUROC: 0.850 [0.802, 0.895] | AUPRC: 0.761 [0.671, 0.834] | F1: 0.743 [0.680, 0.798] (119 shots/11,653 50 ms bins; all reviewed shots; supplemental)
- elm-dsm-detect-init‡ | 2026_10_03 | AUROC: 0.863 [0.817, 0.905] | AUPRC: 0.792 [0.710, 0.857] | F1: 0.738 [0.675, 0.793] (119 shots/11,653 50 ms bins; all reviewed shots; supplemental)
- always present | 2026_10_03 | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.376 [0.322, 0.430] | F1: 0.546 [0.487, 0.601]† (119 shots/12,409 50 ms bins; all reviewed shots)

Brackets are 95% shot-bootstrap intervals; † marks recall ≥0.99; ‡ marks supplemental source fitting, preprocessing or selection exposure. always present precision equals prevalence and recall is 1; full numeric precision/recall and intervals are in the tables and protocol.

The task is ELMy-phase occupancy: about 97% of present bins are crowds. The U-Net and isolated 40-epoch DSM use five physical-shot-grouped folds and exclude cohort blind-test shots from fitting/preprocessing/selection. ELM-O needs BES; reviewers labelled from D-alpha after starting from elm-clock. U-Net inputs share D-alpha with that reference; adapted DSM inputs are D-alpha-free, so their gap also reflects input choice. Primary and common DSM bin sets differ and are labelled separately. Sources: `outputs/labeler/elm/{ours,dsm}/evaluation.json:sets`.

Per-kind recall (95% shot-bootstrap intervals):

| Set | Method | Crowd-bin recall | Non-crowd span-touch recall |
|---|---|---|---|
| all119 | elm-ours | 0.851 [0.789, 0.903] | 0.683 [0.379, 0.834] |
| all119 | elm-clock | 0.680 [0.581, 0.789] | 0.233 [0.122, 0.521] |
| bes73 | elm-ours | 0.877 [0.805, 0.942] | 0.699 [0.277, 0.871] |
| bes73 | elm-clock | 0.628 [0.492, 0.771] | 0.215 [0.090, 0.631] |
| bes73 | ELM-O | 0.852 [0.788, 0.901] | 0.634 [0.267, 0.859] |

Annotation-mode groups, elm-ours (same group bootstrap):

| Annotation mode | Shots / bins | AUROC | F1 | Precision | Recall | Absent-bin call fraction |
|---|---|---|---|---|---|---|
| crowd only | 56 / 6,434 | 0.939 [0.887, 0.979] | 0.895 [0.848, 0.931] | 0.886 [0.819, 0.944] | 0.904 [0.840, 0.951] | 0.152 [0.081, 0.225] |
| non crowd only | 13 / 1,003 | 0.916 [0.841, 0.970] | 0.418 [0.209, 0.627] | 0.298 [0.132, 0.549] | 0.698 [0.396, 0.913] | 0.092 [0.031, 0.174] |
| mixed | 20 / 2,110 | 0.917 [0.868, 0.959] | 0.718 [0.625, 0.819] | 0.857 [0.742, 0.950] | 0.617 [0.495, 0.764] | 0.086 [0.029, 0.168] |
| no present | 30 / 2,862 | -- | -- | -- | -- | 0.092 [0.036, 0.167] |

Raw / 25 ms edge-guarded absent-span touch rates (same denominator):

- elm-ours, all119: 0.596 [0.478, 0.689] / 0.427 [0.343, 0.495]
- elm-ours, bes73: 0.610 [0.452, 0.742] / 0.436 [0.351, 0.511]
- ELM-O, bes73: 0.440 [0.301, 0.623] / 0.307 [0.200, 0.444]

Centered 50 ms smoothing can spill detected runs across span edges. Short empty guarded interiors are counted separately; the protocol also gives the rate restricted to nonempty interiors. These are annotation disagreements, not independently verified physical false alarms. There are 33 shots with non-crowd spans and 76 with crowds; 20 are in both (56 crowd-only, 13 non-crowd-only, 20 mixed, 30 no-present). One no-present shot contributes no scored bins; 29 contribute bins. Onset deliverable incomplete: no independently adjudicated onsets; onset head withdrawn from paper outputs. Checkpoints retain its auxiliary loss. Hard calls use bin-mean ≥ fold threshold for elm-ours and any-touch for ELM-O/clock. The latter burst detector is penalized when a crowd bin contains no burst.

The adapted elm-dsm variants have no D-alpha input (pcphd02/03 mean-filled); CO2 missing on 75/119. The survival adaptation is a one-epoch 60/124-input refit (selected after the first of seven run epochs), served on 50 ms means. The 1 ms training describes the source survival refit; detection heads train on reviewed 50 ms-mean rows. The refit is an offline risk score with 25 ms centered-NBI lookahead (not a causal forecast). Supplemental survival, historical scratch and source-initialized detection use upstream normalization constants computed before the upstream split, including blind-cohort source shots 190646 and 190532 (feature-statistics exposure). The isolated detector fits preprocessing within each outer training partition and uses independent random initialization. Refit and initialized detection also inherit source fitting; five overlap shots (190637,190643,192721,192751,196541) are in-sample. Physical membership is propagated to the adapter so exposed shots are not called held out. Source: `dsm/evaluation.json:model_context,own_target`.

Survival refit own-target AUROC (early-stopping validation; supplemental):

| Horizon (ms) | AUROC | Cases / controls |
|---|---|---|
| 5‡ | 0.758 [0.700, 0.811] | 1,775 / 140,528 |
| 10‡ | 0.764 [0.708, 0.817] | 3,254 / 138,683 |
| 20‡ | 0.770 [0.716, 0.822] | 6,100 / 135,110 |
| 50‡ | 0.777 [0.722, 0.830] | 13,233 / 125,917 |

Native 124-input checkpoint, 1 ms evaluation (separate coverage):

| Native panel | Horizon (ms) | Scored shots / 1 ms rows | AUROC |
|---|---|---|---|
| reviewed_exact_export‡ | 5 | 4 / 11,565 | 0.471 [0.232, 0.662] |
| reviewed_exact_export‡ | 10 | 4 / 11,565 | 0.469 [0.229, 0.660] |
| reviewed_exact_export‡ | 20 | 4 / 11,565 | 0.465 [0.210, 0.658] |
| reviewed_exact_export‡ | 50 | 4 / 11,565 | 0.465 [0.139, 0.653] |
| reviewed_reconstructed‡ | 5 | 33 / 138,100 | 0.608 [0.540, 0.700] |
| reviewed_reconstructed‡ | 10 | 33 / 138,100 | 0.609 [0.540, 0.702] |
| reviewed_reconstructed‡ | 20 | 33 / 138,100 | 0.611 [0.540, 0.704] |
| reviewed_reconstructed‡ | 50 | 33 / 138,100 | 0.689 [0.611, 0.773] |
| own_target‡ | 5 | 326 / 699,512 | 0.926 [0.915, 0.937] |
| own_target‡ | 10 | 326 / 697,855 | 0.931 [0.919, 0.941] |
| own_target‡ | 20 | 326 / 694,596 | 0.939 [0.927, 0.949] |
| own_target‡ | 50 | 326 / 685,337 | 0.955 [0.945, 0.964] |

The native checkpoint uses all **124 original inputs, including both photodiodes and 64 BES columns, at 1 ms**. Exact original exports exist on five reviewed shots, but 192751 has no scored reviewed overlap: four contribute 11,565 rows. All four have source exposure: 196541 was optimizer-trained; 190637, 190643 and 192721 were used for checkpoint selection. This is a separate source-exposed panel, not the 119-shot 50 ms benchmark.

Original diagnostic records plus paced photodiode fetches make 45 shots reconstructable; only 33 contribute reviewed rows after the original native-domain filter (no mean fill or clipping). Reconstructed inputs are a **sensitivity**, because within-shot NBI smoothing cannot reproduce the source's smoothing of concatenated filtered phase rows across boundaries. The exact export is the faithful native-input evaluation. Native own-target AUROC uses 326 physical early-stopping validation shots from the original reversed split, not untouched test evidence. Every panel has 1000 physical-shot bootstrap replicates.

Sources: `dsm/native_evaluation.json:{checkpoint,coverage,exposure,reviewed_exact_export,reviewed_reconstructed,own_target}` and `dsm/native_fetch.json`. Coverage lists every unavailable original column per shot. Missing raw diagnostic groups, including complete BES/ECE/actuator records on later shots, prevent a complete native reconstruction; 50 ms adapter means cannot recover those 1 ms inputs. 78 photodiode records were fetched on 39 otherwise complete-input shots, one worker, pace 1, with no authentication error.

| Input | Tree PMT | View from metadata |
|---|---|---|
| FS02 | PMT11 | unverified: no divertor/midplane field |
| FS03 | PMT12 | unverified: no divertor/midplane field |
| FS04 | PMT13 | unverified: no divertor/midplane field |

FS01 has finite corpus samples on 104/119 shots. It is outside the existing three-channel native-rate cache and all retained fits; exclusion was a data-pipeline choice. Fetching is allowed this round. Read-only tree queries verified the PMT aliases but returned only signal and calibration metadata, with no sightline labels. Each channel's divertor versus midplane view therefore remains unverified. Source: `outputs/labeler/elm/filterscope_metadata.json`.

Inner-validation checkpoint selection is noisy: epoch 2 during warm-up in original fold 3, a single fold-4 spike, thresholds 0.11–0.91 across seeds. A trailing three-epoch mean chooses different endpoints in 15/20 folds; only selected weights were saved, so a smoothed result requires 20-fold retraining (observed training 0.945 GPU-hours). Source: `ours/annotation_strata.json:checkpoint_selection_audit`.

Legacy overlaps: onset table 576 shots/8 overlap; elm_all_ground_truth 365/5 (all yes; no reviewed present span on 192751); Smith BES windows 211/0. The swap reports onset bins beside occupancy conversion at 100/200/300 ms gap tolerances, with M/P/recall and rankings under every reference. A separate three-shot DSM-source-unexposed analysis retains supplemental normalization exposure. Legacy onset bins miss 31% of reviewed-present bins, versus 17% at 200 ms occupancy (M/P 60/14 and 34/60 on 782 known bins). Historical AUROC order and leader are unchanged under every conversion; no reversal observed. F1 uses review-tuned thresholds, so only AUROC compares across references. With eight shots a reversal can be neither shown nor ruled out: evidence inconclusive. The added isolated DSM arm also preserves primary eight-shot AUROC order, but crosses ELM-O's point AUROC on the seven-shot BES occupancy companions; leadership is unchanged and a statistically supported reversal is not established. Three full new-seed CV repeats report spread without choosing a best seed. Sources: `swap/evaluation.json` and `ours/seed_repeats.json`.

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
forecasts and do not report observed events. The survival adapter and supplemental historical detection variants use upstream normalization including blind-cohort shots 190646 and 190532. The isolated detection retrain fits preprocessing within each outer training partition and uses independent random weights.

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
