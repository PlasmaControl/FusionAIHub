# Sawtooth physics-rule labels: current state

These are **physics-rule labels validated only by the checks described here**. There are no blind expert crash times. Neither physical label accuracy nor improvement over the production catalog is established. The reviewed spans are anchored to old suggestions and were consulted in previous rule revisions; they are exploratory, not untouched validation. No model is recommended as latest or stable.

The cohort has 487/500 successful records and 6052 diagnostic crash candidates. OOF model scores assess 123,066/740,299 observable bins (16.6%).

Source: `outputs/labeler/sawtooth/fix4/cohort_labels.json`.

The population run is complete: all 16909 corpus shots were attempted; 13319 have a usable label record and 3590 do not (reader or physical-core failures; the counts by reason are under Label states). 155480 diagnostic points are in the population records.

Source: `outputs/labeler/sawtooth/fix4/population_labels.json`.

**What changed in absence.** The previous round called time absent when EFIT01 q_min stayed at or above 1.5 for 50 ms, and 93.9% of its cohort absent seconds (525 s) were that high-q rule alone. q_min≥1.5 is neither necessary nor sufficient for no sawtooth, so it no longer makes a negative. Absent now means an ECE quiet-core test passed: 96 s on the cohort (5.2% of observable time). Time supported only by q_min is `absent_q_prior` (368 s, 20.0%), exported as `uncertain` with reason `q_prior_only` and excluded from benchmark negatives. The assessed set is therefore smaller and more positive-heavy; the old-negatives sensitivity below keeps the previous scoring for comparison.

Source: `outputs/labeler/sawtooth/fix4/absent_composition.json`.

Conditional OOF crash F1 at ±2 ms: derivative 0.903, HL-3-gated derivative 0.893, saw-ours 0.879. Single-channel derivative / ±125 ms presence has the highest crash-F1 point estimate; the paired OOF differences in crash F1 are: Adapted HL-3 / derivative picker gated by HL-3 minus the derivative picker is -0.010 [-0.019, -0.003] (excludes zero); PhaseNet-style picker minus the derivative picker is -0.024 [-0.062, 0.006] (includes zero). HL-3's expert-shot ranking remains inverted on shot 190637; independent physical validation remains pending.

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*`.

## Method

The detector reads ECE channels 1–40 at 10 kHz (native-rate antialiasing, then decimation), the EFIT01 axis, boundary and q profile, and optional D-alpha, Mirnov, neutron, NBI and Ip traces. Every number below is in `freeze.json` (rule values, the 16 development shots, the excluded reviewed shots), which is hashed beside the labels.

1. **Observability.** A sample is observable when at least 2 core channels (nearest the EFIT axis) read at least 0.5 keV, and unobservable where R<(2/3)R_LCFS,out (third-harmonic overlap), where the Thomson density exceeds 0.9 of the X2 cutoff 2(f_ce/8.98 GHz)² with f_ce from the local field at the axis resonance (B0·R0/R), or where the ECE-validity test holds for at least 20 ms on a 5 ms moving mean: an adjacent-channel step ratio above 2 inside nominal ρ<0.7, or the channel within 0.1 m of the axis below 0.6 of the profile maximum.

2. **Edge filter and POSR.** Each channel is filtered with a Gaussian first derivative (σ=0.25 ms), and a local maximum of its absolute value is a candidate edge when the step is at least 0.5% of the local Te and its POSR reaches 6. POSR is Gude's: the peak's distance from the mean of a 10.3 ms frame, in standard deviations of that frame after dropping its ⌈7σ⌉ largest absolute values (the kernel length). The simulated noise-frame rate is in `noise_calibration.json`.

3. **Multichannel coincidence.** Candidates within 0.5 ms form a cluster. A cluster needs at least 2 channels and an observable core, and the next cluster is dropped inside the same holdoff.

4. **Inversion profile (Gude's A_norm and A_net).** From the per-channel step across the crash, valid channels hotter than 1.5× the local core level are masked. A_norm=Σ|step|/ΣTe must reach 0.02; A_net=|Σstep|/Σ|step| must stay below 0.9, which rejects a profile that falls everywhere. A contiguous loss block of at least 2 channels (steps below −0.5% of local Te) needs a contiguous gain block of at least 2 channels within 6 channels. The block boundary is the inversion channel.

5. **Central drop and edge rejection.** At the channel nearest the nominal EFIT axis the relative drop (mean over 0.3–1.5 ms after against 0.3–1.5 ms before) must be at least 0.05. A crash whose loss is outside the core and that coincides with a D-alpha burst (z≥6) is rejected as an ELM or edge event; Ip below 0.3 MA is rejected. Neutron and Mirnov bursts corroborate when present and never downgrade an ECE crash.

6. **Trains.** At least 3 accepted crashes with gaps of 20–250 ms, successive-gap ratio at most 2.5, and an inversion channel spread within 2 channels form a present train; trains split at every unobservable sample.

7. **Uncertainty reasons.** A train crash is `uncertain` when EFIT01 q_min exceeds 1.4, when the nominal inversion R is more than 0.15 m from the same-side EFIT01 q=1 R, or when the inversion has no nominal R. Geometry is the nominal second-harmonic vacuum resonance, not flux.

8. **States.** `present` is a train span. `absent` is **tested absence**: no POSR-periodic edge (3 or more, 20–250 ms) on any valid channel at nominal ρ<0.5 over a complete ±375 ms observable context, noise-resolved on at least two channels, with no profile candidate or slow relaxation phase nearby and no isolated edge within 5.15 ms. `absent_q_prior` is time supported only by sustained EFIT01 q_min≥1.5 (50 ms); it is exported as `uncertain` with reason `q_prior_only` and is never a benchmark negative. `uncertain` is observable time without definite evidence and `unassessed` is unobservable time. `assessed` means present or absent. High q is neither necessary nor sufficient for absence: a q_min≥1.5 shot can still show a sawtooth-like core relaxation.

Source: `outputs/labeler/sawtooth/fix4/freeze.json` → `rule`.

## Tested absence and the density and cutoff guards

Absent class before (previous round) and after (tested absence), in seconds of 10 kHz samples:

| Set | Absent before s | High-q share before | Tested absent s | Q-prior only s | Tested share of former absent class | Shots with tested absence |
|---|---:|---:|---:|---:|---:|---:|
| train | 422.3 | 92.9% | 80.4 | 288.3 | 21.8% | 166 of 390 |
| val | 49.7 | 98.9% | 5.3 | 40.1 | 11.7% | 16 of 48 |
| test | 53.3 | 97.6% | 10.1 | 39.6 | 20.3% | 18 of 49 |
| population | 9756.9 | 94.5% | 1577.9 | 7104.6 | 18.2% | 3866 of 13319 |

The tested absent seconds split by whether EFIT01 q_min also stayed high: 60 s with high q and 36 s without.

Source: `outputs/labeler/sawtooth/fix4/absent_composition.json` → `before; after`.

The density guard now uses the local field at the axis resonance (B0·R0/R) instead of the field at R0, and a new ECE-side validity test removes time where a static channel-to-channel calibration step or a cold axis channel shows the ECE core is not resolved. Core-observable seconds and what each guard removed:

| Set | Core-observable s | Removed by the earlier B0(R0) guard | Removed by the local-field guard | Removed by ECE validity | Shots with >20% removed by validity |
|---|---:|---:|---:|---:|---:|
| train | 1856.0 | 61.5 (3.3%) | 88.8 (4.8%) | 280.1 (15.1%) | 108 of 388 |
| val | 228.8 | 13.0 (5.7%) | 18.4 (8.0%) | 33.7 (14.7%) | 16 of 47 |
| test | 223.4 | 10.3 (4.6%) | 16.4 (7.4%) | 26.1 (11.7%) | 8 of 47 |
| population | 46515.7 | 1017.5 (2.2%) | 1529.7 (3.3%) | 6208.9 (13.3%) | 2428 of 10857 |

Source: `outputs/labeler/sawtooth/fix4/absent_composition.json` → `guards`.

The guard accounting for each reviewed shot is in the radial-drop section below.

## Geometry and equilibrium

Channels 0–39 use the archived fixed RF grid; same-shot setup takes precedence, including the documented exceptional archived shot. Channels 40–47 are excluded from core, outer, coincidence, redistribution and inversion evidence. Nominal vacuum resonance is R=2×27.992 GHz/T×|F_boundary|/f. The EFIT magnetic axis selects the core. Every time sample with R₂<(2/3)R_LCFS,out is excluded; the shot core screen also rejects channels in that overlap region. Missing radial metadata does not establish definite-positive spatial evidence. Previously terminal-dependent candidates are retained as uncertainty.

The archived grid audit covers 25 shots with 1 documented setup exception. The fetched cohort setup check matches 500/500 first-40 grids. The fixed grid is transferred to other population shots; an unaudited historical setup change cannot be excluded.

Source: `outputs/labeler/sawtooth/fix4/fetched_frequency_audit.json`.

The adapted HL-3 outer input uses low-field-side nominal geometric ρ=0.4–0.65, beyond the typical inversion region, rather than adjacent array rows. Geometric ρ=|R−R_axis|/(R_LCFS,out−R_axis) is **not** normalized flux. Vacuum mapping omits relativistic and optical-depth corrections. Missing ECEZH uses the published first-40 midplane assumption explicitly. Full EFIT profiles are needed for q=1; minimal field/axis/boundary metadata cannot supply that comparison.

Source: `outputs/labeler/sawtooth/fix4/geometry_metadata_audit.json`.

The q=1 audit contains 11118 EFIT-supported shots, 11118 with a profile intersection check, and 4289 with paired diagnostic inversion points. All-point ΔR (inversion R minus same-side q=1 R, metres): n=95609; mean -0.056 m; quantiles at 0/5/25/50/75/95/100%: -0.327, -0.158, -0.102, -0.056, -0.012, 0.053, 0.396 m; 92.3% within 0.15 m. The full per-shot ledger includes checks with no axis-connected surface, checks with no candidate, and missing-profile cases. Paired nominal differences greater than 0.15 m flag uncertainty; a missing EFIT01 surface near q≈1 cannot distinguish reconstruction bias from the observed ECE train. It is recorded as incomparable. The comparison is low-field-side dominated: 94543 of 95609 paired points (98.9%) lie on the low-field side and 1066 on the high-field side, so the high-field side is effectively untested. The inversion lies close to one radius: |R_inversion − R_axis| is n=95609; mean 0.147 m; quantiles at 0/5/25/50/75/95/100%: 0.000, 0.106, 0.132, 0.146, 0.162, 0.191, 0.474 m.

Source: `outputs/labeler/sawtooth/fix4/q1_radius_audit.json` → `paired_point_side`.

TRAIN definite-present nominal inversion ρ: n=4134; mean 0.252; quantiles at 0/5/25/50/75/95/100%: 0.020, 0.149, 0.221, 0.251, 0.285, 0.352, 0.677. The shot-median outer input lies beyond the candidate inversion at 99.9% of comparable TRAIN definite-present points; the full ledger retains the distribution rather than assuming this holds for every event.

EFIT01 conflict is q_min>1.4. Sustained q_min≥1.5 for at least 50 ms is a prior, not an absence test: it marks time `absent_q_prior`, which is exported uncertain and is never a benchmark negative. A magnetics-only reconstruction is not an MSE-constrained central-current measurement: the review-prescribed 1.3–1.5 band replaces the unsupported 1.05 cutoff, rather than estimating a calibrated q correction. Only prior TRAIN candidates enter the bias audit, read from the hashed snapshot of the earlier round's inputs (`labels/prior_inputs/fix2_inputs.json`). MSE-constrained q retains the stricter conflict test. Conflicting inversion-qualified trains remain uncertain over their full context; isolated POSR edges protect finite edge support without vetoing an entire high-q phase.

Prior TRAIN candidate q_min: n=1233; mean 1.278; quantiles at 0/5/25/50/75/95/100%: 1.050, 1.062, 1.121, 1.218, 1.314, 1.741, 5.290. Shot 186532 current state seconds: present 0.00 s, absent 0.00 s, absent_q_prior 0.82 s, uncertain 4.20 s, unassessed 1.11 s.

Source: `outputs/labeler/sawtooth/fix4/qmin_bias_audit.json`.

Source: `outputs/labeler/sawtooth/fix4/freeze.json`.

This TRAIN subset measures sensitivity to the previous cutoff, not the true EFIT bias. The 1.4/1.5 guards are prescribed conservative tolerances, with no calibrated q correction. Muscatello's radial reference uses MSE-constrained EFIT; the present nominal comparison uses EFIT01. See [Muscatello et al. (2012)](https://doi.org/10.1088/0741-3335/54/2/025006) and the locally archived `Muscatello_ST.md` digest.

## Relaxation phases and support

Phase edges require both a ≥2% fractional drop and POSR≥6. The phase period floor is 10 ms, above the old 5.15 ms picker holdoff artifact and conservatively below Muscatello's DIII-D reference periods. Positive trains retain the frozen 20–250 ms bounds. Phase expansion requires at least six edges and shuffled-time p≤0.05. Each null preserves count, span and picker holdoff, repeats the same grouping, and compares the minimum gap CV over all groups. It accounts for search and multiplicity within each observable run. Independent generated noise and regular-train checks test this calibration, not physical label validity.

In 1,000 independent shuffled trials, the full search accepts 52/1000 noise sequences (5.2%; 95% CI 3.99–6.76%). It accepts 1000/1000 jittered regular trains and rejects a regular 6 ms sequence. Qualification by POSR is conditioned on in this timing-null audit, rather than simulated. There is no global familywise guarantee across shots/runs.

Source: `outputs/labeler/sawtooth/fix4/phase_null_audit.json`.

## Label states

State seconds. `Absent` is tested absence; `absent_q_prior` is q-prior only (exported as uncertain, no benchmark negatives).

| Split | Shots | Present s | Absent s | Absent (q prior only) s | Uncertain s | Unassessed s | Assessed / observable |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 400 | 165.696 | 80.413 | 288.279 | 945.731 | 642.438 | 16.6% |
| val | 50 | 19.647 | 5.310 | 40.056 | 110.879 | 88.938 | 14.2% |
| test | 50 | 13.965 | 10.087 | 39.600 | 116.615 | 85.729 | 13.3% |
| population | 16909 | 5131.348 | 1577.906 | 7104.607 | 24795.374 | 43555.745 | 17.4% |

Population: 16909 of 16909 corpus shots have a record (complete); 13319 are usable and 3590 have none and contribute no state seconds, so their absence is not evidence of absence. Exclusion reasons: 2835 × ValueError: absent waveform or incompatible clock, 266 × ValueError: insufficient physically plausible ECE core channels, 262 × OSError: Unable to synchronously open file, 151 × ValueError: insufficient finite ECE in analysis window, 63 × ValueError: geometry must have finite, increasing sample times, 7 × ValueError: no ECE group, 4 × KeyError: "Unable to synchronously open object, 2 × ValueError: insufficient coherent physical ECE core channels. The population row counts the cohort shots as well. Records whose archived equilibrium lacks the field or axis metadata (radius status `bt_or_efit_axis_unavailable`) carry no present candidates and no absent seconds; they are almost entirely unassessed.

Source: `outputs/labeler/sawtooth/fix4/data_summary.json` → `splits; population`.

## Conditional agreement with the physics rule on assessed bins

OOF shots: 400 requested, 390 with usable physical inputs, 285 with assessed outcomes. Unsupported shots remain explicit unknown entries and supply no invented predictions or negatives.

OOF: fixed TRAIN shots, three whole-shot folds; inner shots select weights, LR/regularisation and operating points. Fixed val/test shots are excluded from all selection. Assessment masks affect loss and scoring, never the input definition. Uncertain and unassessed bins supply no negative truth. Excluded picks have unknown outcomes, not established false positives. CIs and paired comparisons use 1,000 whole-shot bootstrap draws.

The trivial picker differentiates only the EFIT-axis-selected single ECE channel. Inner shots independently select z for crash F1 and for the presence rule: any edge ≥zσ within ±125 ms. Thresholds, fitting/inner/scoring shot IDs and support counts are recorded for every fold. Always present supplies no crash time.

| Method | Crash F1 ±2 ms [95% CI] | Presence F1 [95% CI] | AUROC | AUPRC | Assessed / excluded / observable picks |
|---|---:|---:|---:|---:|---:|
| Single-channel derivative / ±125 ms presence | 0.903 [0.883, 0.919] | 0.992 [0.986, 0.996] | 0.984 | 0.985 | 4106 / 17434 / 21540 |
| Always present | — | 0.805 [0.756, 0.846] | 0.500 | 0.673 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.893 [0.869, 0.911] | 0.814 [0.770, 0.849] | 0.744 | 0.846 | 4049 / 17329 / 21378 |
| PhaseNet-style picker | 0.879 [0.843, 0.904] | 0.982 [0.972, 0.989] | 0.989 | 0.994 | 3754 / 11784 / 15538 |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.crash_tolerance_2ms; assessment_totals`.

Second held-out set: 47 nonexpert fixed-validation shots requested; 45 have usable physical inputs and 30 have assessed outcomes. Frozen three-fold ensembles and means of inner-selected thresholds. These still measure conditional teacher agreement.

| Method | Crash F1 ±2 ms [95% CI] | Presence F1 [95% CI] | AUROC | AUPRC | Assessed / excluded / observable picks |
|---|---:|---:|---:|---:|---:|
| Single-channel derivative / ±125 ms presence | 0.880 [0.820, 0.927] | 0.994 [0.974, 1.000] | 0.979 | 0.988 | 407 / 2068 / 2475 |
| Always present | — | 0.872 [0.760, 0.938] | 0.500 | 0.773 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.879 [0.819, 0.925] | 0.896 [0.798, 0.947] | 0.708 | 0.895 | 410 / 2133 / 2543 |
| PhaseNet-style picker | 0.825 [0.697, 0.908] | 0.994 [0.977, 1.000] | 1.000 | 1.000 | 305 / 990 / 1295 |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.fixed_validation`.

| Held-out set | Method minus derivative-only | Δ crash F1 [95% CI] | Δ presence F1 [95% CI] |
|---|---|---:|---:|
| OOF | Single-channel derivative / ±125 ms presence (reference) | 0 | 0 |
| OOF | Adapted HL-3 / derivative picker gated by HL-3 | -0.010 [-0.019, -0.003] | -0.177 [-0.219, -0.144] |
| OOF | PhaseNet-style picker | -0.024 [-0.062, 0.006] | -0.010 [-0.021, -0.001] |
| OOF | Always present | — | -0.187 [-0.234, -0.147] |
| Fixed validation | Single-channel derivative / ±125 ms presence (reference) | 0 | 0 |
| Fixed validation | Adapted HL-3 / derivative picker gated by HL-3 | -0.001 [-0.005, 0.003] | -0.098 [-0.187, -0.048] |
| Fixed validation | PhaseNet-style picker | -0.055 [-0.205, 0.039] | 0.000 [-0.018, 0.019] |
| Fixed validation | Always present | — | -0.122 [-0.230, -0.058] |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `paired_vs_derivative`.

#### Sensitivity to the previous negatives

Benchmark negatives now come only from ECE-tested absence. The previous negatives were q-prior time: this table restores `absent_q_prior` as scoring-only assessed time with no crashes, using the same fitted models and thresholds. Models are not refit. A large gap shows how much of the earlier score came from q-prior time.

| Held-out set | Method | Presence F1 now | Presence F1, previous negatives | AUROC now | AUROC, previous negatives | Crash F1 ±2 ms now | Crash F1, previous negatives |
|---|---|---:|---:|---:|---:|---:|---:|
| OOF | Single-channel derivative / ±125 ms presence | 0.992 | 0.856 | 0.984 | 0.924 | 0.903 | 0.860 |
| OOF | Always present | 0.805 | 0.473 | 0.500 | 0.500 | — | — |
| OOF | Adapted HL-3 / derivative picker gated by HL-3 | 0.814 | 0.491 | 0.744 | 0.606 | 0.893 | 0.852 |
| OOF | PhaseNet-style picker | 0.982 | 0.832 | 0.989 | 0.945 | 0.879 | 0.837 |
| Fixed validation | Single-channel derivative / ±125 ms presence | 0.994 | 0.879 | 0.979 | 0.938 | 0.880 | 0.855 |
| Fixed validation | Always present | 0.872 | 0.475 | 0.500 | 0.500 | — | — |
| Fixed validation | Adapted HL-3 / derivative picker gated by HL-3 | 0.896 | 0.511 | 0.708 | 0.636 | 0.879 | 0.855 |
| Fixed validation | PhaseNet-style picker | 0.994 | 0.757 | 1.000 | 0.981 | 0.825 | 0.801 |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.old_negatives_sensitivity`.

#### Operating points and their spread across folds

Presence and crash thresholds are selected on each fold's inner shots. A wide presence-threshold range for saw-ours means its presence operating point is not stable across folds, so the pooled presence F1 mixes operating points.

| Method | Fold | Presence threshold | Crash threshold | Crash z | Inner shots | Best / completed epochs |
|---|---:|---:|---:|---:|---:|---:|
| Adapted HL-3 / derivative picker gated by HL-3 | 0 | 0.100 | 0.150 | 8.00 | 52 | 4 / 12 |
| Adapted HL-3 / derivative picker gated by HL-3 | 1 | 0.600 | 0.400 | 8.00 | 52 | 2 / 10 |
| Adapted HL-3 / derivative picker gated by HL-3 | 2 | 0.050 | 0.150 | 10.00 | 52 | 3 / 11 |
| Adapted HL-3 / derivative picker gated by HL-3 | spread | 0.050–0.600 (range 0.550) | | | | |
| PhaseNet-style picker | 0 | 0.400 | 0.850 | 0.00 | 52 | 18 / 26 |
| PhaseNet-style picker | 1 | 0.050 | 0.800 | 0.00 | 52 | 22 / 30 |
| PhaseNet-style picker | 2 | 0.100 | 0.975 | 0.00 | 52 | 20 / 28 |
| PhaseNet-style picker | spread | 0.050–0.400 (range 0.350) | | | | |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.operating_points`.

HL-3 is an adapted external architecture, replacing the paper's SXR pair with geometry-selected ECE plus Mirnov and Ip. It has no crash head: its timing score is the **derivative picker gated by HL-3**. LR, weight decay and dropout are selected on inner shots. The U-Net probability grid extends through 0.999; fold records retain selected operating points and boundary flags.

Source: `outputs/labeler/sawtooth/fix4/saw-hl3_fold_0.json`.

Source: `outputs/labeler/sawtooth/fix4/saw-hl3_fold_1.json`.

Source: `outputs/labeler/sawtooth/fix4/saw-hl3_fold_2.json`.

Source: `outputs/labeler/sawtooth/fix4/saw-ours_fold_0.json`.

Source: `outputs/labeler/sawtooth/fix4/saw-ours_fold_1.json`.

Source: `outputs/labeler/sawtooth/fix4/saw-ours_fold_2.json`.

| Three-regime window classifier | Accuracy | Macro-F1 |
|---|---:|---:|
| Adapted HL-3 | 0.625 | 0.612 |
| Single-channel derivative with period | 0.879 | 0.870 |
| Fitting-chosen majority | 0.478 | 0.216 |
| Always present (period class undefined) | — | — |

Derivative period abstentions count as errors on the original class support. Always present has no period class or crash time.

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.three_class`.

Published HL-3 context, different task/population: real-time accuracy stated 0.922, count-derived 0.835; offline accuracy stated 0.956, count-derived 0.907. These three-regime classification scores are not DIII-D crash scores. The stated and count-derived accuracies differ in the source.

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `legacy`.

## Exploratory reviewed spans and old-rule disagreement

| Method | Pooled reviewed-span AUROC | Per-shot reviewed-span F1 | Known / excluded / observable picks |
|---|---:|---|---|
| Single-channel derivative / ±125 ms presence | 0.536 | 186636: 0.481; 189324: 0.755; 190637: 0.202 | 207 / 0 / 207 |
| Always present | 0.500 | 186636: 0.257; 189324: 0.682; 190637: 0.792 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.230 | 186636: 0.257; 189324: 0.682; 190637: 0.012 | 206 / 0 / 206 |
| PhaseNet-style picker | 0.659 | 186636: 0.327; 189324: 0.724; 190637: 0.804 | 89 / 0 / 89 |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.expert`.

Reviewed-score support is the intersection of known reviewed spans and observable inputs. Picks outside it have unknown outcomes. The JSON separately records picks excluded by physics-rule assessment for the conditional agreement diagnostic. Model predictions on reviewed spans are scored independently of the physics rule's assessment mask.

HL-3 ranking check: AUROC below 0.5 persists on listed exploratory expert shots; no expert-based inversion, retuning or threshold selection performed. No expert-based inversion or retuning was performed.

| Reviewed shot | HL-3 AUROC | HL-3 assessed diagnostic AUROC | Saw-ours AUROC |
|---|---:|---:|---:|
| 186636 | 0.127 | — | 0.611 |
| 189324 | 0.426 | 0.642 | 0.758 |
| 190637 | 0.044 | — | 0.651 |

Source: `outputs/labeler/sawtooth/fix4/benchmark.json` → `Tokamak-SI.*.expert.by_shot`.

HL-3's residual inversion is concentrated in shot 190637 and persists on assessed support. The physics rule calls no definite-present phase there while the reviewed spans contain substantial positive support. The radial profile below resolves which side the data support.

Source: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix4/shots/190637.json` → `state_seconds`; `outputs/labeler/sawtooth/fix4/validation.json` → `expert.by_shot`.

## Radial drop profiles of the reviewed spans

Each profile is the median over relaxation events of every ECE channel's relative Te change across the event (medians over 0.3–1.5 ms after against 0.3–1.5 ms before), plotted against signed nominal ρ (negative on the high-field side). Resolution is one ECE channel, about 0.03–0.05 in ρ near the axis, so it cannot separate structure inside about 0.05 of the axis, and ρ is geometric, not flux. Events are the rule's accepted crash points and periodic core edges inside the expert-positive spans (all events of 186532, which has no reviewed span). The verified sawtooth shot 192148 (68 events) is the reference: central -0.231, outer rise 0.131. The verdict rule was written before any reviewed shot was read: central drop ≤ −0.05 inside |ρ|<0.15 and an outer rise ≥ +0.02 at ρ 0.3–0.6 is sawtooth-like; no central drop with a monotone outward decline to ≤ −0.15 at the outermost channel is edge-driven; otherwise indeterminate.

| Shot | Events | Central change | Outer rise (ρ 0.3–0.6) | Outermost change (ρ) | Span s | Observable after guards s | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 186636 | 22 | -0.022 | 0.018 | 0.001 (0.45) | 2.86 | 0.43 | indeterminate |
| 189324 | 40 | -0.252 | 0.165 | 0.014 (0.87) | 2.74 | 2.74 | sawtooth-like |
| 190637 | 43 | -0.002 | -0.028 | -0.132 (0.86) | 3.92 | 3.92 | indeterminate |
| 186532 | 64 | -0.005 | -0.003 | -0.004 (0.45) | 6.13 | 5.02 | indeterminate |

**Shot 190637: the data support an edge or off-axis relaxation, not a central sawtooth.** The span is 3.92 s and fully observable. The median event has no central drop (-0.002, against -0.231 for the verified sawtooth) and no outer rise (-0.028); the drop grows outward to -0.132 at ρ≈0.86. The pre-registered edge-driven bar (−0.15) is missed by 0.02, so the formal verdict is indeterminate, but nothing here supports a central crash. EFIT01 q_min is at least 1.5 for 4.23 s of the shot, consistent with no q=1 surface. The rule is right not to call this span a present sawtooth; the expert-positive label, taken as a central sawtooth, is not supported by ECE (it may mark edge relaxations, as the catalog README already warns).

**Shot 186636: cutoff, so the span is untestable, not wrong.** Only 0.43 s of the 2.86 s span (15%) stays observable. With the local-field density guard the cutoff removes 25,981 of 59,515 core-observable samples (44%; the earlier B0(R0) guard removed 19%) and ECE validity 6% more. The pre-event Te profile (figure) has a steep edge, falling from about 2.6 to 1.2 keV within about 5 cm just outside the axis. On the events that survive the central change is -0.022 with an outer rise of 0.018: the right sign but about a tenth of the reference amplitude, below the pre-registered thresholds. ECE cannot confirm or refute the expert span on most of it; the rule abstains correctly (1.70 s uncertain, 3.17 s unassessed, no present) and the expert span stays untested.

**Shot 189324 agrees.** Central -0.252 and outer rise 0.165 match the reference sawtooth shape, and the rule labels 1.84 s present. **Shot 186532** (not a reviewed shot; it carried the weak gallery crash panel of earlier rounds) shows no relaxation at its 64 rule events (central -0.005, outer -0.003); its three accepted crash points are weak and uncertain.

Figures: `$LABELER_ROOT/round4/saw/fix4/figures/radial_drop_<shot>.pdf` and `.png` for 186636, 189324, 190637 and 186532.

Source: `outputs/labeler/sawtooth/fix4/radial_drop_profiles.json`.

| Reviewed shot | New recall | Legacy recall | New F1 | Uncertain / observable positive bins |
|---|---:|---:|---:|---:|
| 186636 | 0.000 | 0.624 | 0.000 | 218 / 218 |
| 189324 | 0.500 | 0.253 | 0.598 | 683 / 1367 |
| 190637 | 0.000 | 0.250 | 0.000 | 1957 / 1957 |

Span-supported picks are not crash precision/recall.

Source: `outputs/labeler/sawtooth/fix4/validation.json` → `expert; legacy_agreement`.

Native clipped legacy replay reproduces every retained pick array on the seven requested shots; clocks are exact. This excludes a reader timestamp bug. The ledger separates missing raw edges, greedy 10 ms suppression and inversion-window rejection; only training shots enter the diagnostic holdoff ablation. Current catalog v3 and the retained legacy v2 are distinct detectors, so a catalog-wide defect cannot be inferred from the legacy cache alone.

Source: `outputs/labeler/sawtooth/fix4/legacy_reader_audit.json`.

Per-shot timing and gate diagnosis (old minus physics, ms):

| Shot / split | Prior offset ms | Fresh offset ms (pairs) | Fresh present picks | Observable s | Derivative matches: legacy / no holdoff |
|---|---:|---:|---:|---:|---:|
| 190602 / train | -9.30 | — (0) | 0 | 0.000 | 0 / 0 |
| 190604 / test | -9.90 | — (0) | 0 | 0.000 | — / — |
| 192090 / train | -4.85 | -4.55 (10) | 30 | 5.845 | 12 / 18 |
| 201948 / train | 9.90 | — (0) | 0 | 5.848 | 0 / 0 |
| 192154 / test | 9.90 | 10.70 (13) | 59 | 5.551 | — / — |
| 191384 / train | -8.70 | -8.60 (22) | 33 | 5.372 | 3 / 6 |
| 203349 / train | -10.30 | -4.55 (32) | 102 | 5.760 | 0 / 1 |

Prior offsets refer to the superseded rule, whose spatial selection included unverified channels. Fresh offsets retain pairs within 15 ms; nearest neighbours are descriptive and can be reused. One-to-one cells are separate. Shots 190602/190604 have too few uncontaminated channels at their field for a verified redistribution profile and correctly abstain. The train-only ablation changes holdoff solely for diagnosis. Its full ledger gives the inversion rejection reasons and core/outer changes at both rules' picks. Shortening the legacy profile windows alone does not recover the matches: its block interiority, gain adjacency and amplitude conditions reject many central derivative edges.

Source: `outputs/labeler/sawtooth/fix4/legacy_disagreement.json` → `by_shot; figures`.

| Shot 192090 current catalog | Total picks | In physics-present support | TP / FP / FN ±2 ms | Median offset ms |
|---|---:|---:|---|---:|
| all | 53 | 18 | 17 / 1 / 13 | 0.050 |
| ece | 42 | 15 | 15 / 0 / 15 | 0.150 |
| sxr | 11 | 3 | 2 / 1 / 28 | 1.850 |

These TP/FP/FN cells name algorithm agreement against the physics rule. They do not establish physical errors. The current detector does not reproduce the retained legacy's systematic 10 ms offset on this available shot. The other six current event stores are unavailable locally.

Source: `outputs/labeler/sawtooth/fix4/legacy_disagreement.json` → `by_shot.current_production_v3_comparison`.

## Uncertain gaps inside present trains

A present train is split wherever a crash is missing or uncertain, and the gap between present spans is exported `uncertain`. Joining present spans across gaps of at most 0.5 s that are entirely uncertain gives 196 blocks with 791 gaps on the cohort. The gaps are 79.3 s, 31.4% of block time and 4.3% of observable time. 531 gaps (48.7 s) contain no accepted crash: a crash was rejected by the profile, central-drop or coincidence tests, or fell inside the holdoff, and the following gap breaks the period-ratio test; the span is then neither present nor tested absent. 260 gaps (30.6 s) contain an accepted crash left uncertain (282 × unverified_spatial_adjacency, 12 × qmin_conflict, 195 × nominal_q1_radius_mismatch).

Shot 192148 has 8 such gaps totalling 0.96 s: 1.62–1.75 s (uncertain crash); 1.83–1.93 s (no accepted crash); 2.28–2.38 s (no accepted crash); 2.49–2.74 s (no accepted crash); 3.00–3.11 s (no accepted crash); 3.20–3.31 s (no accepted crash); 4.04–4.14 s (no accepted crash); 5.08–5.13 s (no accepted crash). Durations run 43–254 ms against the shot's median accepted-crash period of about 50 ms, so most are one or two missed crashes and the longest is about five.

Source: `outputs/labeler/sawtooth/fix4/uncertain_gaps.json`.

## Muscatello references and blind queue

The frozen rule and the independent derivative picker were run on the two published DIII-D reference shots, 141182 and 141195 (central-channel windows 2.7–3.0 s and 4.55–4.8 s; bands 85 ± 5 ms and 0.35 ± 0.02). Nothing here selects a threshold. Results by method:

| Shot | Window s | Expected crashes | Method | Found | Period ms (85 ± 5) | Amplitude (0.35 ± 0.02) | Unconfirmed / unpicked |
|---|---|---:|---|---:|---|---|---|
| 141182 | 2.7–3.0 | 3–4 | derivative picker | 3 | 92.1 (FAIL) | 0.355 (pass) | — |
| 141182 | 4.55–4.8 | 2–4 | derivative picker | 3 | 87.9 (pass) | 0.380 (FAIL) | — |
| 141182 | 2.7–3.0 | 3–4 | frozen rule | 0 | — (no crash) | — (no crash) | 0 / 3 |
| 141182 | 4.55–4.8 | 2–4 | frozen rule | 0 | — (no crash) | — (no crash) | 0 / 3 |
| 141182 | 2.7–3.0 | 3–4 | frozen rule, validity test off (diagnostic) | 3 | 92.1 (FAIL) | 0.355 (pass) | 0 / 0 |
| 141182 | 4.55–4.8 | 2–4 | frozen rule, validity test off (diagnostic) | 2 | 86.0 (pass) | 0.389 (FAIL) | 0 / 1 |
| 141195 | 2.7–3.0 | 3–4 | derivative picker | 3 | 97.7 (FAIL) | 0.354 (pass) | — |
| 141195 | 4.55–4.8 | 2–4 | derivative picker | 3 | 86.4 (pass) | 0.492 (FAIL) | — |
| 141195 | 2.7–3.0 | 3–4 | frozen rule | 0 | — (no crash) | — (no crash) | 0 / 3 |
| 141195 | 4.55–4.8 | 2–4 | frozen rule | 0 | — (no crash) | — (no crash) | 0 / 3 |
| 141195 | 2.7–3.0 | 3–4 | frozen rule, validity test off (diagnostic) | 3 | 97.7 (FAIL) | 0.354 (pass) | 0 / 0 |
| 141195 | 4.55–4.8 | 2–4 | frozen rule, validity test off (diagnostic) | 0 | — (no crash) | — (no crash) | 0 / 3 |

**The frozen rule fails this check.** With the ECE-validity test it has 0.00 s of 3.1 s observable on 141182 and 0.47 s on 141195, and finds 0 and 3 crashes, none of them inside the published windows (both windows expect 2–4). Picks are not wrong, they are absent: no picks the picker does not confirm. The cause is the validity step test. ECEVS17 reads 2.77–2.86 times ECEVS15 in every window, a static multiplicative calibration step (the TECEF calibration, applied between ECEVS15 and ECEVS17) that is constant through the crashes; the central channel reads 9.1–10.9 keV against the published 2–5 keV. The rule treats such an array as an unresolved Te profile and abstains, which is the intended behaviour for an uncalibrated array and the cost of that behaviour on these shots.

With only the validity test switched off, as a diagnostic and not a rule, the same detector finds 5 crashes in the windows on 141182 and 3 on 141195, with 0 picks the picker does not confirm. Its period and amplitude pass or fail the bands as the table shows. The derivative picker fails some bands too (92.1 ms against 85 ± 5 on 141182, amplitude 0.380 above 0.35 ± 0.02), so the amplitude definition on the stepped calibration, with the central channel at about 9 keV, is not comparable with the published 2–5 keV central Te and these band failures are not evidence about the rule alone.

Source: `outputs/labeler/sawtooth/fix4/muscatello_rule_check.json`.

Blind queue: 200 windows on 44 shots, including 141 primary random windows. Stratum counts: `{'random_observable': 141, 'algorithm_uncertain': 15, 'zero_candidate_shot': 15, 'disagreement': 14, 'model_predicted_negative': 14, 'random_observable_reserve': 1}`. Random observable windows are frozen before prediction access. Candidate-free, model-negative, uncertain and disagreement supplements are separate strata. Only sensor inputs and blank targets enter the annotation pack; private selection and all predictions remain hidden. About 97 independent positive events give a worst-case 95% recall half-width of 0.1; clustered events do not supply that effective sample size automatically. Primary sampling weights and whole-shot CIs are preregistered. Annotation remains pending.

Source: `outputs/labeler/sawtooth/fix4/crash_time_queue.json`.

## Artifacts, reproduction and verification

Complete labels: `$LABELER_ROOT/round4/saw/fix4/labels/`. Manifest `SHA256SUMS` sha256: `39ec0062962bf732f8d00816d8af2a6ee129612f6e43a476b76877f506e87b32`. The population run is complete: all 16909 corpus shots were attempted; 13319 have a usable label record and 3590 do not (reader or physical-core failures; the counts by reason are under Label states). Population shards export current states without duplicate cohort rows; failed-shot records remain in the ledger. The cohort bundle is separate. Production stores were read only. The manifest also hashes `prior_inputs/fix2_inputs.json`, the snapshot of the previous round's inputs the rule reads (terminal-dependent candidate times, failed-shot list, TRAIN q_min conflict candidates), so the labels no longer depend on a deleted directory; `freeze.json` is hashed beside them.

Source: `outputs/labeler/sawtooth/fix4/label_manifest.json`.

Large signals, EFIT metadata, models, predictions, annotation pack, PDFs and 150-dpi PNGs are under the same fix4 directory. Paper example, three old-rule figures, the 12-shot gallery and the four radial-profile figures were inspected; figure ledgers retain paths and hashes.

Source: `outputs/labeler/sawtooth/fix4/paper_example.json`.

Source: `outputs/labeler/sawtooth/fix4/gallery.json`.

Source: `outputs/labeler/sawtooth/fix4/figure_inspection.json`.

Covering tests: 150 passed. Ruff passes on all changed Python files; formatting passes on all new Python files; git diff --check passes. Long jobs used timeouts and logs; temporary storage was swept afterward. GPU training used CUDA_VISIBLE_DEVICES=1 within the assigned memory budget.

Source: `outputs/labeler/sawtooth/fix4/verification.json`.

```text
........................................................................ [ 48%]
........................................................................ [ 96%]
......                                                                   [100%]
150 passed in 13.53s
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/lib/python3.11/site-packages/XRootD/client/finalize.py:46: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if isinstance(obj, File) and obj.is_open():
All checks passed!
5 files already formatted
```

Reproduce with the prescribed pixi environment and TMPDIR. Entrypoints: sawtooth_fix4_setup.py; sawtooth_geometry_fix3.py audit/frequency-audit/reference; sawtooth_physics.py labels (cohort, then population with sawtooth_population.sbatch); sawtooth_reference_rule.py; sawtooth_fix3_records.py records/manifest; sawtooth_absent_composition.py; sawtooth_uncertain_gaps.py; sawtooth_radial_drop.py; sawtooth_fix_validation.py validate; sawtooth_fix3_artifacts.py queue-base/reader-audit/queue/figures; sawtooth_benchmark.py train/predict/evaluate; sawtooth_gallery.py; sawtooth_phase_null_audit.py; sawtooth_fix3_report.py. Run queue-base before model prediction access. Full command logs remain under the large output directory.

## Concerns and next work

Known limitations, all left as they are because a rule change would force a full population rerun:

- Uncertain dominates: 64.2% of observable population time and 29707 of 155480 candidates are uncertain. Present plus tested absent is 17.4% of observable time, so all conditional scores describe that fraction.
- The negative class is thin: tested absence is 5.2% of observable cohort time, so the assessed set is positive-heavy and presence AUROC and precision are less informative than before; the old-negatives sensitivity table is the comparison to the earlier scoring.
- The ECE-validity step test reads a static calibration step as an invalid profile. On the Muscatello reference shots it removes nearly all time (observable 0.00 s and 0.47 s of 3.1 s), so the frozen rule labels almost nothing on them. It is the review's test and it is not tuned on those shots.
- The EFIT01 q=1 radius conflict is conservative. Shot 203563 (EFIT01 q_min 0.77, inversion minus q=1 radius -0.198 m) has 26 of 29 candidates uncertain although it shows regular trains. The paired comparison is 98.9% low-field side.
- Uncertain gaps inside present trains cover 31.4% of block time on the cohort; most are missed crashes that break the period-ratio test.
- Population shots without archived field or axis metadata, and shots without a usable ECE waveform, receive no definite label (see the exclusion counts above).
- The reviewed 186636 span is 85% cutoff or invalid ECE and is untestable here; the 190637 span is not a central sawtooth by ECE. Neither is a calibrated physical truth.
- The legacy-offset diagnosis traces the old detector's offsets to its inversion-profile gate and holdoff, not to the reader. The current catalog detector could be compared only on shot 192090.
- Each gallery crash panel shows the accepted crash with the largest A_norm (present crashes first; an uncertain crash only when the shot has no present one), a fixed rule, so panels are not chosen to be easy or hard.

Blind physical accuracy remains unmeasured. Reviewed-span failures and any inverted HL-3 ranking remain failures, not reasons to retune on those shots. Nominal geometry is not a calibrated flux measurement; missing field/profile data limits population evidence. Algorithm-assessed benchmarks remain conditional and can be solved by derivative rules. Next: obtain the queued blind crash/span/ambiguity annotations, lock them, and evaluate all frozen methods over independently observable support without retuning.

## Appendix: superseded history

Earlier rounds used a hottest-channel proxy, sparse same-shot RF localization, a q_min>1.05 conflict and unbounded fractional edge phases. Their results are superseded. The first-round `outputs/labeler/sawtooth/fix` records and the two earlier plan documents were removed; `outputs/labeler/sawtooth/fix2` and `fix3` remain as predecessor records (fix3 counted q_min≥1.5 time as absent; its absent-class and score numbers are superseded by this round), and the earlier rounds' narrative is in the stream report appendix.
