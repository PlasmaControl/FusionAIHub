# Edge Localized Mode

## Description
Edge localized modes (ELMs) are quasi-periodic relaxations of the H-mode pedestal where the steep edge pressure gradient and the bootstrap current it drives cross the coupled peeling-ballooning stability boundary causing rapid filamentary bursts to expel stored energy in ~1 ms.

ELMs were first observed on ASDEX with the discovery of the H-mode (1982).

Type-I ELMs have frequencies of tens to a few hundred Hz on DIII-D and their frequency typically rises with heating power. Type-III ELMs are smaller,
faster and appear near the L-H power threshold. Breakthrough ELMs are large ELMs that occur during wide pedestal quiescent high confinement (WPQH) experiments.

These are typically found via filterscope D-alpha bursts, divertor Langmuir probes, magnetics and BES. D-alpha pulse morphology and duration depend on diagnostic view, ELM type and detachment; 2--5 ms bursts are a common diagnostic signature, not a universal definition of an ELM.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `elm_labels_dict.pkl`, `elm_labels_dict_wpqh.pkl`, `elm_survival_labels.pkl`, `elm_survival_labels_wpqh.pkl`

**Author**: Hiro Farre Josep Kaga

**Description**: The original `wpqh_elm_hiro` label pickles are now in `raw/`. The separate D-alpha clock described above remains a producer.

**Publications**:

### Dataset 2

**Dataset File(s)**: `labeled-elm-events.hdf5`, `labeled_elm_events_long_windows_20220921.hdf5` (read in place from `/projects/EKOLEMEN/dsmith/data/`, not copied)

**Author**: David Smith

**Description**: Hand-labelled ELMs on the 64-channel BES at 1 MHz: 481 events on 164 shots, and 2,008 events on 199 shots in longer windows. Each event is a short window holding one ELM (BES `signals`, `time` in ms, a per-sample `labels` flag for the ELM region, shot number). The ELM-O detector was tuned on a hand-labelled set of this kind (972 ELMs); which events are in it is not recorded. These windows (2,316 after dropping 173 repeats, on 211 shots, none of them in the corpus or the review labels) are the truth for the selected-window ELM-O benchmark. The current one-to-one onset and region-overlap scores, distinct from the paper's 972-event tuning score, are in [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md). The experimental elm-ours-onset head uses Smith-only shot CV; these windows are not ingested as catalog labels.

**Publications**:

### Dataset 3

**Dataset File(s)**:

**Author**:

**Description**: The elm-dsm survival refit uses the ELM-survival rows of `wpqh_elm_hiro` with 60 non-BES columns. Its served output is an offline risk score with centered-NBI lookahead; see the current protocol for input, normalization and physical-shot exposure.

**Publications**:

## Models
**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (forecast, not a detector)

**latest**: elm-ours | 2026_10_03

**all**:
- elm-ours | 2026_10_03 | AUROC: 0.939 [0.901, 0.966] | AUPRC: 0.878 [0.775, 0.951] | F1: 0.834 [0.782, 0.875] (119 shots / 11,653 common bins)
- d3d_elm_time_to_event_dsm | 2026_09_06 | AUROC: 0.777 [0.730, 0.824] | AUPRC: 0.662 [0.592, 0.736] | F1: 0.627 [0.559, 0.692] (elm-dsm refit; supplemental offline risk score reusing source data; 119 shots / 11,653 common bins)
- elm-dsm (60-input 1×128 refit, detection) | 2026_10_03 | AUROC: 0.845 [0.796, 0.893] | AUPRC: 0.748 [0.651, 0.832] | F1: 0.742 [0.681, 0.798] (119 shots / 11,653 common bins)
- elm-clock | 2026_09_13 | AUROC: -- | AUPRC: -- | F1: 0.759 [0.683, 0.829] (119 shots / 11,653 common bins)
- elm-ours (BES subset) | 2026_10_03 | AUROC: 0.939 [0.893, 0.972] | AUPRC: 0.876 [0.752, 0.962] | F1: 0.848 [0.790, 0.899] (73 shots / 6,527 common bins)
- elm-elmo | 2026_10_01 | AUROC: 0.914 [0.869, 0.948] | AUPRC: 0.833 [0.755, 0.892] | F1: 0.841 [0.786, 0.885] (73 shots / 6,527 common bins)
- always-present | 2026_10_03 | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.391 [0.332, 0.450] | F1: 0.562 [0.499, 0.621] (119 shots / 11,653 common bins)
- elm-feature | 2026_10_03 | AUROC: 0.833 [0.782, 0.879] | AUPRC: 0.704 [0.612, 0.789] | F1: 0.713 [0.649, 0.770] (119 shots / 11,653 common bins)
- elm-ours-onset | 2026_10_03 | Recall ±2/5 ms: 0.930 [0.915, 0.945] | Matched errors ≤0.82 ms; 94% correct 1 ms cell (211 Smith shots; developmental selected-window shot CV)

Brackets are 95% shot-bootstrap intervals. Review results are developmental shot-CV occupancy estimates (97% crowd positives; clock-seeded review). `elm-ours` matches ELM-O without BES; extends coverage to all 119 shots. On the common BES subset, paired elm-ours minus ELM-O is AUROC +0.025 [-0.013, 0.071], AUPRC +0.044 [-0.056, 0.131], F1 +0.007 [-0.044, 0.063]; every interval includes zero. Source: [dsm/evaluation.json](../../../outputs/labeler/elm/dsm/evaluation.json), `sets.bes73.paired`. Catalog physical-onset output is withheld. Every Smith method's precision/F1 is conditional on selected windows; continuous-discharge precision/F1 is unavailable. Inputs, run-day sharing and timing limits: [elm_ours.md](../../../docs/labeler/elm_ours.md).

## Inputs
**elm-dsm** (internal adapter slug `d3d_elm_time_to_event_dsm`):
- `ip`, `bt`, `gas`, `pinj`, `tinj`, `ech`
- `co2_density_slow` `r0`, `v1`, `v2`, `v3`
- `ece_slow` (48 ch)
- `pcphd02`, `pcphd03` (detection: measured on all 119 review shots)

**elm-clock**:
- `D-alpha FS01..FS08` (first finite channel)
- `CO2 density R0`, `pinj_total` (L-mode gate, `dalpha_lh`)
- `Ip` (plasma start)

**elm-elmo**:
- `interferometer` chords `DENV2F`, `DENV3F` (`\BCI::`, 100 kS/s)
- `filterscopes` FS02, FS03, FS04 (`\SPECTROSCOPY::`, 50 kS/s)
- `BES` (64 ch; 500 kS/s in the corpus, 1 MS/s in Smith's windows)

**elm-ours**:
- `filterscopes` FS02, FS03, FS04 (`\SPECTROSCOPY::`, 50 kS/s; log level and contrast to a 0.5 s running median)
- `interferometer` chords `DENV2F`, `DENV3F` (`\BCI::`, 100 kS/s; density and its 0.2 s high-pass)
- a validity mask; 10 kHz grid, no BES

FS01 is omitted from elm-ours because its retained input cache contains FS02–04
only. The available metadata does not establish whether FS01–04 view the
divertor or midplane.

**elm-frames (label view)**:
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
forecasts and do not report observed events. The survival adapter and supplemental
historical detection variants use upstream normalization including blind-cohort
shots 190646 and 190532. The detection baseline is a reduced-input adaptation
trained and evaluated on 50 ms rows with 60 inputs and one 128-unit layer;
preprocessing is fitted within each outer training partition and weights start
randomly. The source DSM trained on native 1 ms rows with 124 inputs and layers
[100, 1000], so this adaptation is not an objective-only retrain of that model.

Known gap: a narrow ELM riding on a broad D-alpha hump measures wide and is dropped.

Benchmark: the rule-based ELM-O detector (O'Shea et al. 2023: interferometers, filterscopes and BES, no learning) is the benchmark for these labels. It was re-implemented from the paper (the public code has no licence) and scored on David Smith's labelled windows and on the reviewed ELM spans, non-crowd present spans and crowd spans apart: [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md). It needs BES, which 73 of the 119 review shots have in the corpus.

`elm-ours` (`labeler.elm`) is a 1D U-Net (401,714 parameters) that reads
filterscopes and fast density without BES and writes per-millisecond occupancy
probability. It uses five shot-grouped folds over the 119 reviewed shots,
with three additional training-seed repeats on the same partitions. The
auxiliary span-start head is dropped from paper outputs; the existing
checkpoints retain its loss. Fast-density values retain their numerical scale;
physical ordinate units were not retained in the original source cache and are
unverified. Fixed numerical preprocessing divides fast density by `1e14`
native ordinate units and clips it to `[-3, 12]`; ten times its 0.2 s high-pass
is clipped to `[-10, 10]`. Chords with median absolute native magnitude above
`1e16` are zeroed by the heuristic failed-digitiser screen. This scaling,
clipping and magnitude screen do not establish physical calibration or validate
diagnostic failure. Offline metadata audits made no new fetches and changed no
saved inputs or weights; see `density_units.json` and
`filterscope_metadata.json` under `outputs/labeler/elm/`.

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
