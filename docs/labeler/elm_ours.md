# ELMy-phase occupancy: current ELM protocol

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

The fixed cohort contains 400 train, 50 val, 50 test. The 119 reviewed shots contain 69 train, 50 val; these counts are recorded in `outputs/labeler/elm/protocol.json:metadata`.
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

| Input | Tree PMT | View from metadata |
|---|---|---|
| FS02 | PMT11 | unverified: no divertor/midplane field |
| FS03 | PMT12 | unverified: no divertor/midplane field |
| FS04 | PMT13 | unverified: no divertor/midplane field |

FS01 has finite corpus samples on 104/119 shots. It is outside the existing three-channel native-rate cache and all retained fits; exclusion was a data-pipeline choice. Fetching is allowed this round. Read-only tree queries verified the PMT aliases but returned only signal and calibration metadata, with no sightline labels. Each channel's divertor versus midplane view therefore remains unverified. Source: `outputs/labeler/elm/filterscope_metadata.json`.

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

All reviewed shots: 119 shots/12,409 bins. BES subset, ELM-O chunks: 73 shots/6,843 bins.

| Set | Method | AUROC | AUPRC | F1 | Precision / recall |
|---|---|---|---|---|---|
| all119 | elm-ours | 0.941 [0.906, 0.967] | 0.877 [0.774, 0.949] | 0.830 [0.778, 0.871] | 0.818 / 0.842 |
| all119 | elm-clock | -- | -- | 0.757 [0.680, 0.827] | 0.866 / 0.673 |
| all119 | always present | 0.500 [0.500, 0.500] | 0.376 [0.322, 0.430] | 0.546 [0.487, 0.601]† | 0.376 / 1.000 |
| bes73 | elm-ours | 0.941 [0.898, 0.972] | 0.875 [0.751, 0.961] | 0.845 [0.786, 0.896] | 0.826 / 0.864 |
| bes73 | elm-clock | -- | -- | 0.708 [0.602, 0.800] | 0.822 / 0.621 |
| bes73 | ELM-O | 0.918 [0.877, 0.951] | 0.833 [0.753, 0.890] | 0.842 [0.789, 0.885] | 0.840 / 0.844 |
| bes73 | always present | 0.500 [0.500, 0.500] | 0.402 [0.325, 0.479] | 0.573 [0.491, 0.648]† | 0.402 / 1.000 |

Source: `outputs/labeler/elm/ours/evaluation.json:sets.<set>.methods`.
The BES comparison has similar point estimates and no significant difference
detected; this is not an equivalence result. Paired differences and intervals
are in the same record's `paired` fields.

### Crowd and non-crowd targets

Of 4,664 present bins, 4,539 (97.32%) are crowd bins; only 125 are non-crowd.

| Set | Method | Crowd-bin recall | Non-crowd span-touch recall |
|---|---|---|---|
| all119 | elm-ours | 0.851 [0.789, 0.903] | 0.683 [0.379, 0.834] |
| all119 | elm-clock | 0.680 [0.581, 0.789] | 0.233 [0.122, 0.521] |
| bes73 | elm-ours | 0.877 [0.805, 0.942] | 0.699 [0.277, 0.871] |
| bes73 | elm-clock | 0.628 [0.492, 0.771] | 0.215 [0.090, 0.631] |
| bes73 | ELM-O | 0.852 [0.788, 0.901] | 0.634 [0.267, 0.859] |

### Annotation-mode shot groups

| Annotation mode | Shots / bins | AUROC | F1 | Precision | Recall | Absent-bin call fraction |
|---|---|---|---|---|---|---|
| crowd only | 56 / 6,434 | 0.939 [0.887, 0.979] | 0.895 [0.848, 0.931] | 0.886 [0.819, 0.944] | 0.904 [0.840, 0.951] | 0.152 [0.081, 0.225] |
| non crowd only | 13 / 1,003 | 0.916 [0.841, 0.970] | 0.418 [0.209, 0.627] | 0.298 [0.132, 0.549] | 0.698 [0.396, 0.913] | 0.092 [0.031, 0.174] |
| mixed | 20 / 2,110 | 0.917 [0.868, 0.959] | 0.718 [0.625, 0.819] | 0.857 [0.742, 0.950] | 0.617 [0.495, 0.764] | 0.086 [0.029, 0.168] |
| no present | 30 / 2,862 | -- | -- | -- | -- | 0.092 [0.036, 0.167] |

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

All reviewed shots: 119 shots/11,653 bins. BES subset, ELM-O chunks: 73 shots/6,527 bins.

| Set | Method | AUROC | AUPRC | F1 | Precision / recall |
|---|---|---|---|---|---|
| all119 | elm-ours | 0.939 [0.901, 0.966] | 0.878 [0.775, 0.951] | 0.834 [0.782, 0.875] | 0.820 / 0.848 |
| all119 | elm-dsm refit‡ (supplemental) | 0.777 [0.730, 0.824] | 0.662 [0.592, 0.736] | 0.627 [0.559, 0.692] | 0.569 / 0.698 |
| all119 | elm-dsm detection (isolated) | 0.855 [0.813, 0.896] | 0.778 [0.696, 0.844] | 0.754 [0.696, 0.811] | 0.678 / 0.848 |
| all119 | elm-dsm detection‡ (exposed; supplemental) | 0.850 [0.802, 0.895] | 0.761 [0.671, 0.834] | 0.743 [0.680, 0.798] | 0.668 / 0.837 |
| all119 | elm-dsm detection init‡ (supplemental) | 0.863 [0.817, 0.905] | 0.792 [0.710, 0.857] | 0.738 [0.675, 0.793] | 0.673 / 0.818 |
| all119 | elm-clock | -- | -- | 0.759 [0.683, 0.829] | 0.869 / 0.674 |
| all119 | always present | 0.500 [0.500, 0.500] | 0.391 [0.332, 0.450] | 0.562 [0.499, 0.621]† | 0.391 / 1.000 |
| bes73 | elm-ours | 0.939 [0.893, 0.972] | 0.876 [0.752, 0.962] | 0.848 [0.790, 0.899] | 0.827 / 0.871 |
| bes73 | elm-dsm refit‡ (supplemental) | 0.715 [0.633, 0.794] | 0.631 [0.536, 0.730] | 0.591 [0.504, 0.674] | 0.520 / 0.686 |
| bes73 | elm-dsm detection (isolated) | 0.856 [0.806, 0.902] | 0.767 [0.668, 0.861] | 0.780 [0.705, 0.842] | 0.691 / 0.896 |
| bes73 | elm-dsm detection‡ (exposed; supplemental) | 0.843 [0.788, 0.892] | 0.750 [0.648, 0.845] | 0.762 [0.681, 0.826] | 0.668 / 0.888 |
| bes73 | elm-dsm detection init‡ (supplemental) | 0.853 [0.796, 0.900] | 0.779 [0.681, 0.860] | 0.749 [0.671, 0.811] | 0.663 / 0.861 |
| bes73 | ELM-O | 0.914 [0.869, 0.948] | 0.833 [0.755, 0.892] | 0.841 [0.786, 0.885] | 0.839 / 0.843 |
| bes73 | elm-clock | -- | -- | 0.714 [0.610, 0.806] | 0.828 / 0.627 |
| bes73 | always present | 0.500 [0.500, 0.500] | 0.417 [0.336, 0.500] | 0.588 [0.502, 0.666]† | 0.417 / 1.000 |

Source: `outputs/labeler/elm/dsm/evaluation.json:sets.<set>.methods`.
Input availability, architecture and objective differ together; their effects
cannot be separated causally. In-sample DSM source exposure is disclosed below.

### Survival refit on its own target (supplemental)

| Horizon (ms) | AUROC | Cases / controls |
|---|---|---|
| 5‡ | 0.758 [0.700, 0.811] | 1,775 / 140,528 |
| 10‡ | 0.764 [0.708, 0.817] | 3,254 / 138,683 |
| 20‡ | 0.770 [0.716, 0.822] | 6,100 / 135,110 |
| 50‡ | 0.777 [0.722, 0.830] | 13,233 / 125,917 |

The **one-epoch** 60/124-input refit uses 142,745 original 1 ms rows from 80 physical shots. These are early-stopping validation rows (called test upstream), not untouched test evidence; phase records of one shot occur on both source split sides. Intervals group physical shots. The 40-epoch isolated detection retrain above is the fair adapted DSM detection baseline. The one-epoch survival adaptation must not be presented as the native 124-input checkpoint. Source: `dsm/evaluation.json:own_target`.

### Native 124-input checkpoint and 1 ms coverage

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

### Absent-span boundary sensitivity

| Set | Method | Raw | 25 ms guard, same denominator | Nonempty interiors |
|---|---|---|---|---|
| all119 | elm-ours | 0.596 [0.478, 0.689] | 0.427 [0.343, 0.495] | 0.466 [0.366, 0.551] |
| bes73 | elm-ours | 0.610 [0.452, 0.742] | 0.436 [0.351, 0.511] | 0.490 [0.369, 0.610] |
| bes73 | ELM-O | 0.440 [0.301, 0.623] | 0.307 [0.200, 0.444] | 0.345 [0.240, 0.468] |

The all119 raw denominator is 342 absent spans; 29 have no interior after removing 25 ms at both edges. The guarded share retains the raw denominator; the final column uses only nonempty interiors. Centered 50 ms smoothing can spill detected runs across annotation edges, particularly short inter-ELM gaps. A boundary touch can trigger the raw metric without an interior alarm. Sources: `ours/evaluation.json:span_alarm_definition` and `sets.<set>.methods.<method>.{point,ci95,counts}`.


### Seed stability

Three new full five-fold CV runs retain the original outer and inner shot
partitions and all hyperparameters. Checkpoints and thresholds are selected
independently within each fixed inner-validation fold. All four runs are
reported; no best seed is selected. Shot intervals quantify sampling uncertainty;
the seed range/SD quantify training variability and are not confidence intervals.

| Training seed | all119 AUROC | all119 AUPRC | all119 F1 |
|---|---|---|---|
| 20261003 | 0.941 [0.906, 0.967] | 0.877 [0.774, 0.949] | 0.830 [0.778, 0.871] |
| 20261004 | 0.949 [0.923, 0.971] | 0.913 [0.852, 0.955] | 0.841 [0.787, 0.884] |
| 20261005 | 0.941 [0.907, 0.967] | 0.897 [0.825, 0.944] | 0.815 [0.753, 0.864] |
| 20261006 | 0.924 [0.885, 0.957] | 0.858 [0.756, 0.929] | 0.820 [0.764, 0.865] |

| Set | Metric | Three new seeds: min–max | Sample SD |
|---|---|---|---|
| bes73 | auroc | 0.931–0.955 | 0.014 |
| bes73 | auprc | 0.842–0.921 | 0.044 |
| bes73 | f1 | 0.830–0.854 | 0.014 |
| all119 | auroc | 0.924–0.949 | 0.013 |
| all119 | auprc | 0.858–0.913 | 0.028 |
| all119 | f1 | 0.815–0.841 | 0.014 |

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

| Reference | Known bins | M | P | Legacy recall | Legacy F1 |
|---|---|---|---|---|---|
| Onset bins | 782 | 60 | 14 | 0.692 [0.443, 0.813] | 0.785 [0.568, 0.873] |
| Occupancy τ=100 ms | 782 | 37 | 49 | 0.810 [0.508, 0.938] | 0.786 [0.536, 0.926] |
| Occupancy τ=200 ms | 782 | 34 | 60 | 0.826 [0.512, 0.946] | 0.774 [0.527, 0.926] |
| Occupancy τ=300 ms | 782 | 33 | 65 | 0.831 [0.525, 0.950] | 0.768 [0.524, 0.921] |

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

| Set | Reference | M / P | Recall | AUROC order | F1 order |
|---|---|---|---|---|---|
| overlap | Onset bins | 40 / 4 | 0.728 [0.413, 0.837] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) | elm-ours > elm-clock > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap | Occupancy τ=100 ms | 20 / 13 | 0.864 [0.491, 0.960] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) | elm-ours > elm-clock > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap | Occupancy τ=200 ms | 19 / 18 | 0.871 [0.514, 0.966] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) | elm-ours > elm-dsm refit‡ (supplemental) > elm-clock > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap | Occupancy τ=300 ms | 18 / 23 | 0.878 [0.531, 0.970] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) | elm-ours > elm-dsm refit‡ (supplemental) > elm-clock > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap_bes | Onset bins | 32 / 4 | 0.761 [0.452, 0.882] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > ELM-O > elm-dsm detection (isolated) | elm-ours > ELM-O > elm-clock > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap_bes | Occupancy τ=100 ms | 13 / 13 | 0.903 [0.524, 0.974] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) > ELM-O | elm-ours > ELM-O > elm-dsm refit‡ (supplemental) > elm-clock > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap_bes | Occupancy τ=200 ms | 12 / 18 | 0.910 [0.571, 0.981] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) > ELM-O | elm-ours > elm-dsm refit‡ (supplemental) > ELM-O > elm-clock > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |
| overlap_bes | Occupancy τ=300 ms | 11 / 23 | 0.918 [0.595, 0.982] | elm-ours > elm-dsm refit‡ (supplemental) > elm-dsm detection‡ (exposed; supplemental) > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) > ELM-O | elm-ours > elm-dsm refit‡ (supplemental) > ELM-O > elm-dsm detection‡ (exposed; supplemental) > elm-clock > elm-dsm detection init‡ (supplemental) > elm-dsm detection (isolated) |

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
