# Edge Localized Mode

> The label data is `data/events/edge_localized_mode/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/raw_archive/`.

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

**Description**: Hand-labelled ELMs on the 64-channel BES at 1 MHz: 481 events on 164 shots, and 2,008 events on 199 shots in longer windows. Each event is a short window holding one ELM (BES `signals`, `time` in ms, a per-sample `labels` flag for the ELM region, shot number). The ELM-O detector was tuned on a hand-labelled set of this kind (972 ELMs); which events are in it is not recorded. These windows (2,316 after dropping 173 repeats, on 211 shots, none of them in the corpus or the review labels) support selected-window ELM-O metrics; they do not establish continuous-discharge precision or physical-onset equivalence with the paper's 972 tuning events. Current onset-matching and region-overlap scores: [elm_benchmark_elmo.md](../../elm_benchmark_elmo.md). Not yet ingested as a catalog label source here.

**Publications**:

### Dataset 3

**Dataset File(s)**:

**Author**:

**Description**: The elm-dsm survival refit uses the ELM-survival rows of `wpqh_elm_hiro` with 60 non-BES columns. Its served output is an offline risk score with centered-NBI lookahead; see the current protocol for input, normalization and physical-shot exposure.

**Publications**:

### Dataset 4

**Dataset File(s)**: `elm_challenge_key_b71f991.json` (raw); `edge_localized_mode_challenge_format_2026_v1.csv` (format, grids in `format/challenge_shots/`)

**Author**: David Eldon (General Atomics), the DIII-D ELM detection challenge (`DIII-D/elm_detection_challenge`, commit `b71f991`, 2026-10-07); used with the maintainers' permission

**Description**: The challenge's answer key: hand-timed and classified ELMs in short windows of 24 DIII-D shots (354 events, 46.97 s assessed; the key's 4 NSTX cases are skipped), each edge with an uncertainty. Types: large ELM, small ELM, large and small "other" transients, pellet, pellet + ELM, hidden pellet. The format table keeps the windows only: ELMs (types 1, 2, 6) are present, "other" transients (3, 4) uncertain, pellets without an ELM (5, 7) and the rest of each window absent, and time outside a window has no row. Types, counts and edge uncertainties stay in the raw key, which the challenge's merit function scores against. Six of the 24 windows hold no event, and shot 169908 lists one event twice. These shots are `holdout` rows of `shots.csv`: an evaluation set, never used for fitting or tuning; only 163518 is also in Dataset 1. The key is still growing upstream, so a newer commit is a new dated dataset beside this one. The clock, the review page's jump-and-hold onsets, elm-ours (50 ms bins only), the two onset heads (the review-trained `cv2` head and the Smith-only head, as the first instance-detection baseline) and the challenge's sample submissions are scored on it with the challenge's merit function, our boundary matching and 50 ms bins in `docs/labeler/elm_challenge.md` (`outputs/labeler/elm/challenge/evaluation.json`); these scores are not in the Models list below, which `scripts/labeler/elm_protocol.py` writes.

**Publications**:

## Models
**stable**: d3d_elm_time_to_event_dsm | 2026_09_06 (forecast, not a detector)

**latest**: elm-ours | 2026_10_03

**all**:
- elm-ours | 2026_10_03 | AUROC: 0.941 [0.906, 0.967] | AUPRC: 0.877 [0.774, 0.949] | F1: 0.830 [0.778, 0.871] (primary all119; 119 shots / 12,409 bins; on the bes73 scope, 73 shots / 6,843 bins: AUROC 0.941 [0.898, 0.972], AUPRC 0.875 [0.751, 0.961], F1 0.845 [0.786, 0.896])
- d3d_elm_time_to_event_dsm | 2026_09_06 | AUROC: 0.777 [0.730, 0.824] | AUPRC: 0.662 [0.592, 0.736] | F1: 0.627 [0.559, 0.692] (elm-dsm-survival; supplemental offline risk score reusing source data; 119 shots / 11,653 common bins)
- elm-dsm-detect (60-input 1×128) | 2026_10_03 | AUROC: 0.845 [0.796, 0.893] | AUPRC: 0.748 [0.651, 0.832] | F1: 0.742 [0.681, 0.798] (secondary control; lower bound on DSM detection skill under our recipe; 3 post-warm-up repeats give mean AUROC 0.865 (0.857–0.870); 119 shots / 11,653 common bins)
- elm-clock | 2026_09_13 | AUROC: -- | AUPRC: -- | F1: 0.757 [0.680, 0.827] (primary all119; 119 shots / 12,409 bins)
- elm-elmo | 2026_10_01 | AUROC: 0.918 [0.877, 0.951] | AUPRC: 0.833 [0.753, 0.890] | F1: 0.842 [0.789, 0.885] (primary bes73; 73 shots / 6,843 bins)
- always-present | 2026_10_03 | AUROC: 0.500 [0.500, 0.500] | AUPRC: 0.376 [0.322, 0.430] | F1: 0.546 [0.487, 0.601] (baseline; primary all119; 119 shots / 12,409 bins)
- elm-feature | 2026_10_03 | AUROC: 0.833 [0.783, 0.878] | AUPRC: 0.695 [0.606, 0.779] | F1: 0.703 [0.640, 0.758] (control; primary all119; 119 shots / 12,409 bins)
- elm_jump | 2026_10_14 | AUROC: -- | AUPRC: -- | F1: 0.823 [0.768, 0.869] (review source since 2026-10-14; in-sample: its run rule and gate were chosen on these shots; precision 0.768, recall 0.886; primary all119 on the regridded review labels, 119 shots / 12,481 bins, where the clock scores F1 0.755 and elm_jump minus the clock is +0.068 [-0.016, +0.151]; F1 by divertor side: lower 75 shots 0.831 [0.761, 0.889], upper 44 shots 0.811 [0.718, 0.887]; the clock by side: lower 75 shots 0.696 [0.580, 0.796], upper 44 shots 0.825 [0.723, 0.897])
- elm-clock-v2 | 2026_10_14 | AUROC: -- | AUPRC: -- | F1: 0.755 [0.679, 0.824] (primary all119 on the regridded review labels, 119 shots / 12,481 bins; its present spans call the same bins as v1's there)

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
[0.063, 0.168] (0.096) for the clock, whose boundaries seeded the review. The
elm-dsm-detect rows are lower bounds on DSM detection skill under our recipe. elm_jump
(the page's side-aware jump-and-hold onsets in runs of at least five, gated by the
confinement layers) replaced the clock as the review source on 2026-10-14, since its
all119 F1 beat the clock's; the clock's edges seeded the review, which favours the clock
in that comparison. elm_jump's run rule and gate were chosen among five variants scored
on these 119 reviewed shots (10 ms frames, F1 .733 to .805), so its scores are in-sample
for that choice. Rule and scores: [elm_jump.md](../../elm_jump.md).
Serving runs a shot whole through the five fold models with the mean fold threshold
(0.417, unevaluated); 4,096 ms tiled inference and run-day-grouped folds are
sensitivities in the protocol document. Catalog physical-onset output is withheld. Every
Smith method's precision/F1 is conditional on selected windows; continuous-discharge
precision/F1 is unavailable. The experimental Smith onset head is omitted from catalog
model claims. Inputs, run-day sharing and timing limits:
[elm_ours.md](../../elm_ours.md).

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
only. The available metadata does not establish whether FS02–04 view the
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

The review source is `elm_jump` v2 since 2026-10-14 (v1 earlier that day, `elm_clock`
v1 before it; `python -m labeler.events.elm_jump`). It draws ELMy spans from the page's own
jump-and-hold onsets on D-alpha summed over the strike-point chords (upper or lower
divertor): runs of at least 5 onsets no more than 50 ms apart, padded by 5 ms, less the
time the confinement layers call L-mode or QH-mode (never the D-alpha L-H detector, the
clock's gate, nor the confinement source's rows taken from it). On the 119 reviewed shots
its F1 is 0.823 [0.768, 0.869] against the clock's 0.755 [0.679, 0.824] (difference
+0.068 [-0.016, +0.151]; recall 0.886 against 0.671, precision 0.768 against 0.862); the
gain is on the 75 lower-divertor shots (0.831 against 0.696). The run rule and the gate were chosen among five jump-and-hold variants
scored on these 119 reviewed shots (10 ms frames, F1 .733 to .805), so these scores are
in-sample for that choice. Its table covers 627 roster, silver and gold shots, 626 of them drafted (v2 adds 25
silver shots whose filterscopes were fetched after v1, and leaves out the regimes rows
the detector supplied, which moves 3 reviewed shots by at most 69 ms of present time);
each onset's normalised jump is kept beside it as a size. Rule, coverage and scores:
[elm_jump.md](../../elm_jump.md).

On the review page a shot with no save opens with the D-alpha jump-and-hold onsets (the
yellow ticks) as one-frame Individual annotations beside the source's ELMy-period Span
annotations, so the candidates are edited like the source label itself; see
`docs/labeler/review.md` ("ELM onset candidates are in the draft"). The tables are not
changed by this: the onsets are added when the page opens the shot, and a save writes them.

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

Benchmark: the rule-based ELM-O detector (O'Shea et al. 2023: interferometers, filterscopes and BES, no learning) is the benchmark for these labels. It was re-implemented from the paper (the public code has no licence) and scored on David Smith's labelled windows and on the reviewed ELM spans, non-crowd present spans and crowd spans apart: [elm_benchmark_elmo.md](../../elm_benchmark_elmo.md). It needs BES, which 73 of the 119 review shots have in the corpus.

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

Gold-set extension: `extend_elm_ours/gold_shots.csv` holds `elm-ours` occupancy (present or absent on the 10 ms grid; the five-fold mean at the mean fold threshold 0.417, both unevaluated) for 7 gold shots that have no ELM label (176070, 199597, 199603, 199607 and the I-mode shots 204501, 204503, 204505), over each shot's valid input from 0 ms, from native-rate fetched records. It is unreviewed model output. It first held 17 shots; the 13 from 2017-2019 (before the reviewed shots, 185857-204937) came out under 5 % present and were dropped on 2026-10-11 at the owner's word. When the gold set was rebalanced on 2026-10-12, five new I-mode shots had no ELM table: 204501, 204503 and 204505 are published, and 204497 and 204499 came out 0.0 present and were not, by the same rule (`gold_shots.meta.json`, `scored_not_published`).

Silver and gold extension (2026-10-14): `extend_elm_ours/silver_gold.csv` holds the same `elm-ours` output for the silver and gold shots outside both the run's folds and `gold_shots.csv`: 391 shots requested, inputs fetched for 389 (fdp has no FS02 record for 178872 and no DENV2F record for 203401). 225 shots are published (11 of them gold; 693.6 s present, 933.5 s absent). Not published: 116 shots under 5 % present by the same rule (23 gold; `scored_not_published`), 17 whose DENV2F chord is empty or screened as a failed digitiser, and 31 whose filterscope level lies outside the training range (`shots_skipped`). Seven gold shots (192751, 195016, 195024, 195102, 195103, 195111, 200954) are reviewed shots of the `cv2` folds: four of the five fold models trained on their labels, so `elm-ours` is not served on them (`labeler.elm.serve` now refuses any fold shot). In the silver view and its grid export (`review.silver`) an extend table takes precedence over the review source, so there the ELM draft of these 225 shots is `elm-ours`, filled by `elm_jump` only where `elm-ours` has no row.

## Alias
edge localized mode, edge localised mode, elm, elms, elmy, elming

## Future Implementations
- Ingest David Smith's labelled ELMs as a further source beside the breakthrough ELMs of `wpqh_elm_hiro` (the crowd flag distinguishes non-crowd annotations from ELMing periods)
- Score elm-elmo on Dataset 4 (it needs BES, which the challenge's allowed signals exclude), and classify event types for the merit function's classification term. The onset heads are scored there (`docs/labeler/elm_challenge.md`); a head that separates individual ELMs from crowds and has an end is still to do

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
