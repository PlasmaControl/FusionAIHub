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

**Description**: Hand-labelled ELMs on the 64-channel BES at 1 MHz: 481 events on 164 shots, and 2,008 events on 199 shots in longer windows. Each event is a short window holding one ELM (BES `signals`, `time` in ms, a per-sample `labels` flag for the ELM region, shot number). The ELM-O detector was tuned on a hand-labelled set of this kind (972 ELMs); which events are in it is not recorded. These windows (2,316 after dropping 173 repeats, on 211 shots, none of them in the corpus or the review labels) support selected-window ELM-O metrics; they do not establish continuous-discharge precision or physical-onset equivalence with the paper's 972 tuning events. Current onset-matching and region-overlap scores: [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md). Not yet ingested as a catalog label source here.

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
- elm-ours | 2026_10_03 | AUROC: 0.941 [0.906, 0.967] | AUPRC: 0.877 [0.774, 0.949] | F1: 0.830 [0.778, 0.871] (primary all119; 119 shots / 12,409 bins; on the bes73 scope, 73 shots / 6,843 bins: AUROC 0.941 [0.898, 0.972], AUPRC 0.875 [0.751, 0.961], F1 0.845 [0.786, 0.896])
- d3d_elm_time_to_event_dsm | 2026_09_06 | AUROC: 0.777 [0.730, 0.824] | AUPRC: 0.662 [0.592, 0.736] | F1: 0.627 [0.559, 0.692] (elm-dsm-survival; supplemental offline risk score reusing source data; 119 shots / 11,653 common bins)
- elm-dsm (60-input 1×128 refit, detection) | 2026_10_03 | AUROC: 0.845 [0.796, 0.893] | AUPRC: 0.748 [0.651, 0.832] | F1: 0.742 [0.681, 0.798] (secondary control; lower bound on DSM detection skill under our recipe; 3 post-warm-up repeats give mean AUROC 0.865 (0.857–0.870); 119 shots / 11,653 common bins)
- elm-clock | 2026_09_13 | AUROC: -- | AUPRC: -- | F1: 0.757 [0.680, 0.827] (primary all119; 119 shots / 12,409 bins)
- elm-elmo | 2026_10_01 | AUROC: 0.918 [0.877, 0.951] | AUPRC: 0.833 [0.753, 0.890] | F1: 0.842 [0.789, 0.885] (primary bes73; 73 shots / 6,843 bins)
- always-present | 2026_10_03 | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.376 [0.322, 0.430] | F1: 0.546 [0.487, 0.601] (baseline; primary all119; 119 shots / 12,409 bins)
- elm-feature | 2026_10_03 | AUROC: 0.833 [0.783, 0.878] | AUPRC: 0.695 [0.606, 0.779] | F1: 0.703 [0.640, 0.758] (control; primary all119; 119 shots / 12,409 bins)
- elm-dsm-detect (60-input 1×128) | 2026_10_03 | AUROC: 0.845 [0.796, 0.893] | AUPRC: 0.748 [0.651, 0.832] | F1: 0.742 [0.681, 0.798] (secondary control; lower bound on DSM detection skill under our recipe; 3 post-warm-up repeats give mean AUROC 0.865 (0.857–0.870); 119 shots / 11,653 common bins)

Brackets are 95% shot-bootstrap intervals. Review results are developmental shot-CV
estimates of ELMy-period occupancy, not onsets (97% of positive bins lie in crowd
spans); the review was seeded by the D-alpha clock, which favours D-alpha-input models
over elm-elmo. A frozen-model score on the blind split awaits review of those shots.
Primary benchmark: all119 (12,409 bins) and bes73 (6,843 bins). On bes73, paired
elm-ours minus elm-elmo is AUROC +0.023 [-0.014, +0.067], AUPRC +0.043 [-0.058, +0.129],
F1 +0.002 [-0.052, +0.058]; every interval includes zero, so no significant difference
is detected (equivalence untested), while elm-ours extends BES-free coverage to all 119
shots. On the secondary common bins the same difference is AUROC +0.025 [-0.013,
+0.071], AUPRC +0.044 [-0.056, +0.131], F1 +0.007 [-0.044, +0.063]. On non-crowd-only
shots elm-ours precision is 0.298 [0.132, 0.549]; its absent-span alarm rate is 0.596
[0.478, 0.689] (0.427 with 25 ms trimmed from each absent-span edge) against 0.105
[0.063, 0.168] (0.096) for the clock, whose boundaries seeded the review. The elm-dsm
detection rows are lower bounds on DSM detection skill under our recipe. Serving runs a
shot whole through the five fold models with the mean fold threshold (0.417,
unevaluated); 4,096 ms tiled inference and run-day-grouped folds are sensitivities in
the protocol document. Catalog physical-onset output is withheld. Every Smith method's
precision/F1 is conditional on selected windows; continuous-discharge precision/F1 is
unavailable. The experimental Smith onset head is omitted from catalog model claims.
Inputs, run-day sharing and timing limits:
[elm_ours.md](../../../docs/labeler/elm_ours.md).

## Inputs
**elm-dsm** (internal adapter slug `d3d_elm_time_to_event_dsm`):
- `ip`, `bt`, `gas`, `pinj`, `tinj`, `ech`
- `co2_density_slow` `r0`, `v1`, `v2`, `v3`
- `ece_slow` (48 ch)
- `pcphd02`, `pcphd03` (survival adapter: mean-filled; detection adaptation: measured on all 119 reviewed shots)

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
The elm-clock detector reads one D-alpha filterscope channel, the first finite
one of FS01..FS08 (10 kHz): the min-max-normalized signal is smoothed (0.64 ms) and
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
Its detection rows are lower bounds on DSM detection skill under our recipe; refits
with post-warm-up checkpoint selection and further seeds are in the protocol document.

Known gap: a narrow ELM riding on a broad D-alpha hump measures wide and is dropped.

Benchmark: the rule-based ELM-O detector (O'Shea et al. 2023: interferometers, filterscopes and BES, no learning) is the benchmark for these labels. It was re-implemented from the paper (the public code has no licence) and scored on David Smith's labelled windows and on the reviewed ELM spans, non-crowd present spans and crowd spans apart: [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md). It needs BES, which 73 of the 119 review shots have in the corpus.

`elm-ours` (`labeler.elm`) is a 1D U-Net (401,714 parameters) that reads
filterscopes and fast density without BES and writes per-millisecond occupancy
probability. It uses five shot-grouped folds over the 119 reviewed shots,
with three additional training-seed repeats on the same partitions. Serving runs a
shot whole, not tiled, through the five fold models, averages their probabilities and
uses the mean of the five fold thresholds; that ensemble and threshold are unevaluated,
because every reviewed shot lies in some fold's training set. Tiled 4,096 ms inference
and run-day-grouped folds are sensitivities, reported in the protocol document. The
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
