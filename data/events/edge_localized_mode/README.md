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

**Author**:

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

**Description**: The DSM forecast was fitted by labelmaker on the ELM-survival rows of the `wpqh_elm_hiro` project (629,023 rows, 60 non-BES columns), because upstream's graphs need 64 BES channels the corpus fills on 2 of 24 sampled shots.

**Publications**:

## Models
**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (forecast, not a detector)

**latest**: elm-ours | 2026_10_03

**all**:
- elm-ours | 2026_10_03 | AUROC: 0.941 | AUPRC: 0.877 | F1: 0.830 (1D U-Net on filterscope D-alpha and line density, no BES; the 119 review shots, 5-fold shot-grouped CV)
- d3d_elm_time_to_event_dsm | 2026_09_06 | AUROC: 0.780 | AUPRC: 0.666 | F1: 0.638 (`no_bes` fit; horizons 5/10/20/50 ms; as published, `elm-dsm`; a forecast, scored here as a detector on the 11,653 bins it can serve)
- elm-dsm-detect | 2026_10_03 | AUROC: 0.855 | AUPRC: 0.762 | F1: 0.735 (the same 60 inputs and embedding, objective changed to detection, one logit head, trained on the review in the same folds)
- elm_clock | 2026_09_13 (rule-based detector, `labeler.events.transients`)
- elmo | 2026_10_01 | AUROC: 0.918 | AUPRC: 0.833 | F1: 0.842 (rule-based detector, O'Shea et al. 2023, re-implemented)

`elm-ours` scores are pooled out-of-fold, 12,409 bins of 50 ms (38 % present), intervals over shot resamples: AUROC 0.941 [0.906, 0.967], AUPRC 0.877 [0.774, 0.949], F1 0.830 [0.778, 0.871] at a threshold chosen on each fold's inner-validation shots (no cohort test shot is read). On the 73 shots with BES it scores AUROC 0.941, AUPRC 0.875, F1 0.845 against ELM-O's 0.918, 0.833 and 0.842, a difference inside the intervals, without BES. It finds 85 % of the bins inside crowd spans and 68 % of the single-ELM spans; its onset head is weak (F1 0.045 at 5 ms) and is not a result. The survival model (`elm-dsm`) is scored on bins it can serve (the corpus lacks `ip`, `bt` and the photodiodes on most shots, mean-filled) and on its own target the published split gives AUROC 0.758 to 0.777 at 5 to 50 ms. Reference swap: Hiro's 50 ms onset table covers 8 of the 119 reviewed shots (|M| = 40 of 147 present bins, |P| = 4 of 641) and does not change the order of the methods. Protocol, tables and caveats: [elm_ours.md](../../../docs/labeler/elm_ours.md).

Scores for ELM-O are as published (eta 0.997, BES threshold 1 V) against the owner-reviewed ELM spans: 50 ms bins of the 73 review shots that have BES in the corpus (precision 0.840, recall 0.844; the other 46 shots have no BES, which ELM-O needs). ELM-O makes hard calls, so its AUROC and AUPRC come from sweeping its detection threshold eta over the same bins, not from a probability. It finds 85 % of the bins inside crowd spans and 59 of 93 individual spans, and puts ELMs in 96 of 218 spans marked absent, where the D-alpha of the worst shows ELM trains. `elm_clock` scores F1 0.708 on the same bins, but the reviewers started from it. On David Smith's labelled windows the re-implementation reproduces the published scores (precision 0.997, recall 0.980 over 2,316 windows; paper 0.995 and 0.976). Protocol, variants and caveats: [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md).

## Inputs
**d3d_elm_time_to_event_dsm**:
- `ip`, `bt`, `gas`, `pinj`, `tinj`, `ech`
- `co2_density_slow` `r0`, `v1`, `v2`, `v3`
- `ece_slow` (48 ch)
- `pcphd02`, `pcphd03` (always mean-filled)

**elm_clock**:
- `D-alpha FS01..FS08` (first finite channel)
- `CO2 density R0`, `pinj_total` (L-mode gate, `dalpha_lh`)
- `Ip` (plasma start)

**elmo**:
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
The detector is `elm_clock` over the eight real D-alpha filterscope channels
(10 kHz): the column activity of the TokEye transient mask is smoothed (0.64 ms) and
peaks are picked with prominence 0.03 and a minimum separation of 3 ms
(`elmcycle.detect_elms`, ported). A candidate is accepted only if its
half-prominence width is <= 5 ms (`DALPHA_MAX_WIDTH_MS`), which rejects the broad
humps of gas puffs and L-H transitions. Each accepted burst is a point event
`elm`; the clock also writes `elm_free` intervals (rate < 5 Hz over a 100 ms window
for >= 50 ms) and the ELM rate. Coverage is the D-alpha span actually processed.

`d3d_elm_time_to_event_dsm` writes **forecasts** - the probability of an ELM within
5, 10, 20 and 50 ms - and a forecast is never reported as an observed event.

Known gap: a narrow ELM riding on a broad D-alpha hump measures wide and is dropped.

Benchmark: the rule-based ELM-O detector (O'Shea et al. 2023: interferometers, filterscopes and BES, no learning) is the benchmark for these labels. It was re-implemented from the paper (the public code has no licence) and scored on David Smith's labelled windows and on the reviewed ELM spans, individual ELMs and crowd spans apart: [elm_benchmark_elmo.md](../../../docs/labeler/elm_benchmark_elmo.md). It needs BES, which 73 of the 119 review shots have in the corpus.

`elm-ours` (`labeler.elm`) is a 1D U-Net (PhaseNet / U-Time kind, 401,714 parameters) that reads the filterscopes and the line density, not BES, and writes per millisecond the probability of ELMy time and a second head for ELM onsets. It is trained on the reviewed spans (present, crowds included, against absent) with five shot-grouped folds over the 119 review shots, so every shot has one out-of-fold prediction.

## Alias
edge localized mode, edge localised mode, elm, elms, elmy, elming

## Future Implementations
- Ingest David Smith's labelled ELMs, and David Eldon's ELM dataset when it arrives, as further sources beside the breakthrough ELMs of `wpqh_elm_hiro` (the crowd flag tells isolated ELMs from ELMing periods)

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
