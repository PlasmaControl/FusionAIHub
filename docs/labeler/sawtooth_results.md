# Sawtooth physics-rule labels: current state

These are **physics-rule labels validated only by the checks described here**. There are no blind expert crash times. Neither physical label accuracy nor improvement over the production catalog is established. The reviewed spans are anchored to old suggestions and were consulted in previous rule revisions; they are exploratory, not untouched validation. No model is recommended as latest or stable.

The cohort has 487/500 successful records and 6167 diagnostic crash candidates. OOF model scores assess 114,309/746,129 observable bins (15.3%).

Source: `outputs/labeler/sawtooth/fix5/cohort_labels.json`.

The population run is complete: all 16909 corpus shots were attempted; 13319 have a usable label record and 3590 do not (reader or physical-core failures; the counts by reason are under Label states). 158230 diagnostic points are in the population records.

Source: `outputs/labeler/sawtooth/fix5/population_labels.json`.

**What changed in absence.** The previous round called time absent when EFIT01 q_min stayed at or above 1.5 for 50 ms, and 93.9% of its cohort absent seconds (525 s) were that high-q rule alone. q_min≥1.5 is neither necessary nor sufficient for no sawtooth, so it no longer makes a negative. Absent now means an ECE quiet-core test passed: 71 s on the cohort (3.9% of observable time). Time with sustained q_min≥1.5 and no tested absence is the q-prior (390 s, 21.1% of observable time), exported as `uncertain` and excluded from benchmark negatives. 59.5% of it (232 s) lies within the absence-test context of a periodic edge or a profile candidate (`q_prior_ece_contradicted`): the ECE shows relaxation evidence there, so this share measures how much of the old q-only negative class the ECE contradicts. The rest (`q_prior_untested`) was not tested by the ECE. The assessed set is therefore smaller and more positive-heavy; the old-negatives sensitivity below keeps the previous scoring for comparison.

Source: `outputs/labeler/sawtooth/fix5/absent_composition.json`.

Conditional OOF crash F1 at ±2 ms: derivative 0.903, HL-3-gated derivative 0.894, saw-ours 0.873. Single-channel derivative / ±125 ms presence has the highest crash-F1 point estimate; the paired OOF differences in crash F1 are: Adapted HL-3 / derivative picker gated by HL-3 minus the derivative picker is -0.009 [-0.018, -0.003] (excludes zero); PhaseNet-style picker minus the derivative picker is -0.030 [-0.055, -0.009] (excludes zero). HL-3's expert-shot ranking remains inverted on shot 190637; independent physical validation remains pending.

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*`.

## Method

The detector reads ECE channels 1–40 at 10 kHz (native-rate antialiasing, then decimation), the EFIT01 axis, boundary and q profile, and optional D-alpha, Mirnov, neutron, NBI and Ip traces. Every number below is in `freeze.json` (rule values, the 16 development shots, the excluded reviewed shots), which is hashed beside the labels.

1. **Observability.** A sample is observable when at least 2 core channels (nearest the EFIT axis) read at least 0.5 keV, and unobservable where R<(2/3)R_LCFS,out (third-harmonic overlap), where the Thomson density exceeds 0.9 of the X2 cutoff 2(f_ce/8.98 GHz)² or where the ECE-validity test holds for at least 20 ms on a 5 ms moving mean: an adjacent-channel step ratio above 2 inside nominal ρ<0.7, or the channel within 0.1 m of the axis below 0.6 of the profile maximum. f_ce in the cutoff comes from the local field at the EFIT axis resonance (F/R_axis) wherever that axis field is mapped, with or without a Bt trace; the field Bt(R0) is used only where the axis field is unmapped but Bt exists, and a fixed 8×10¹⁹ m⁻³ guard only where neither exists (the record's `cutoff_field` names the branch). A channel whose record median is below the 0.5 keV floor, or that shows no fluctuation, is a dead channel and is left out of both validity tests: a cutoff step is one-sided and a dead channel is low on both sides.

2. **Edge filter and POSR.** Each channel is filtered with a Gaussian first derivative (σ=0.25 ms), and a local maximum of its absolute value is a candidate edge when the step is at least 0.5% of the local Te and its POSR reaches 6. POSR is Gude's: the peak's distance from the mean of a 10.3 ms frame, in standard deviations of that frame after dropping its ⌈7σ⌉ largest absolute values (the kernel length). The simulated noise-frame rate is in `noise_calibration.json`.

3. **Multichannel coincidence.** Candidates within 0.5 ms form a cluster. A cluster needs at least 2 channels and an observable core, and the next cluster is dropped inside the same holdoff.

4. **Inversion profile (Gude's A_norm and A_net).** From the per-channel step across the crash, valid channels hotter than 1.5× the local core level are masked. A_norm=Σ|step|/ΣTe must reach 0.02; A_net=|Σstep|/Σ|step| must stay below 0.9, which rejects a profile that falls everywhere. A contiguous loss block of at least 2 channels (steps below −0.5% of local Te) needs a contiguous gain block of at least 2 channels within 6 channels. The block boundary is the inversion channel.

5. **Central drop and edge rejection.** At the channel nearest the nominal EFIT axis the relative drop (mean over 0.3–1.5 ms after against 0.3–1.5 ms before) must be at least 0.05. A crash whose loss is outside the core and that coincides with a D-alpha burst (z≥6) is rejected as an ELM or edge event; Ip below 0.3 MA is rejected. Neutron and Mirnov bursts corroborate when present and never downgrade an ECE crash.

6. **Trains.** At least 3 accepted crashes with gaps of 20–250 ms, successive-gap ratio at most 2.5, and an inversion channel spread within 2 channels form a present train; trains split at every unobservable sample.

7. **Uncertainty reasons.** A train crash is `uncertain` when EFIT01 q_min exceeds 1.4, when the nominal inversion R is more than 0.15 m from the same-side EFIT01 q=1 R, or when the inversion has no nominal R. Geometry is the nominal second-harmonic vacuum resonance, not flux.

8. **States.** `present` is a train span. `absent` is **tested absence**: no POSR-periodic edge (3 or more, 20–250 ms) on any valid channel at nominal ρ<0.5 over a complete ±375 ms observable context, noise-resolved on at least two channels, with no profile candidate or slow relaxation phase nearby and no isolated edge within 50 ms. An edge counts against absence only when its relative change reaches `significance` = 2% of the local Te, so a core relaxation train below 2% is called quiet. The isolated-edge length was derived on TRAIN shots only (390 shots, `edge_context_derivation.json`): it is the longest candidate isolated-edge veto (5.15 ms frame holdoff, 50 ms, 375 ms full absence-test context) that keeps at least half of the TRAIN tested-absent time left by the frame holdoff; TRAIN cohort shots only, no val or test shot. Time supported only by sustained EFIT01 q_min≥1.5 (50 ms) is never absent. It is `q_prior_ece_contradicted` where it lies within the absence-test context of a periodic edge or a profile candidate, which is where the ECE shows relaxation evidence, and `q_prior_untested` elsewhere, where the ECE test did not run or was inconclusive. Both are exported as `uncertain` with the state name as the reason and are never a benchmark negative. `uncertain` is observable time without definite evidence and `unassessed` is unobservable time. `assessed` means present or absent. High q is neither necessary nor sufficient for absence: a q_min≥1.5 shot can still show a sawtooth-like core relaxation.

Source: `outputs/labeler/sawtooth/fix5/freeze.json` → `rule`.

## Tested absence and the density and cutoff guards

Absent class before (previous round) and after (tested absence), in seconds of 10 kHz samples:

| Set | Absent before s | High-q share before | Tested absent s | Q-prior, ECE-contradicted s | Q-prior, untested s | ECE-contradicted share of q-prior | Tested share of former absent class | Shots with tested absence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 422.3 | 92.9% | 60.4 | 180.6 | 126.4 | 58.8% | 16.4% | 161 of 390 |
| val | 49.7 | 98.9% | 3.7 | 24.7 | 16.6 | 59.7% | 8.3% | 15 of 48 |
| test | 53.3 | 97.6% | 7.2 | 26.7 | 15.2 | 63.8% | 14.7% | 17 of 49 |
| population | 9756.9 | 94.5% | 1240.8 | 4420.0 | 3044.6 | 59.2% | 14.3% | 3738 of 13319 |

The tested absent seconds split by whether EFIT01 q_min also stayed high: 43 s with high q and 28 s without.

Source: `outputs/labeler/sawtooth/fix5/absent_composition.json` → `before; after`.

Sensitivity to the isolated-edge veto length (5.15 ms is the frame holdoff; 375 ms is the full absence-test context). The rule's value was derived on TRAIN shots only; the rebuilt negatives on every split, and the scored OOF negatives of the PhaseNet-style picker, follow from the stored frame-holdoff masks without any refit:

| Isolated-edge veto ms | Train tested absent s | Val tested absent s | Test tested absent s | Tested share of observable (train) | OOF benchmark negative bins | OOF positive bins |
|---:|---:|---:|---:|---:|---:|---:|
| 5.15 | 81.2 | 5.3 | 9.9 | 5.4% | 40,580 | 84,109 |
| 50 (rule value) | 60.4 | 3.7 | 7.2 | 4.1% | 30,200 | 84,109 |
| 375 | 15.4 | 0.2 | 1.6 | 1.0% | 7,704 | 84,109 |

Source: `outputs/labeler/sawtooth/fix5/absent_composition.json` → `edge_context_sensitivity; benchmark.json Tokamak-SI.saw-ours.edge_context_sensitivity; edge_context_derivation.json`.

The density guard uses the local field at the axis resonance (F/R_axis) instead of the field at R0 wherever it is mapped, and an ECE-side validity test removes time where a static channel-to-channel calibration step or a cold axis channel shows the ECE core is not resolved; dead channels are left out of that test. Core-observable seconds and what each guard removed:

| Set | Core-observable s | Removed by the earlier guard (Bt at R0, fixed 8e19 without Bt) | Removed by the current guard | Removed by ECE validity | Shots with >20% removed by validity |
|---|---:|---:|---:|---:|---:|
| train | 1856.0 | 61.5 (3.3%) | 192.0 (10.3%) | 166.2 (9.0%) | 65 of 388 |
| val | 228.8 | 13.0 (5.7%) | 28.4 (12.4%) | 20.7 (9.0%) | 8 of 47 |
| test | 223.4 | 10.3 (4.6%) | 24.2 (10.8%) | 18.7 (8.4%) | 5 of 47 |
| population | 46515.7 | 1017.5 (2.2%) | 3730.8 (8.0%) | 3392.1 (7.3%) | 1123 of 10857 |

Source: `outputs/labeler/sawtooth/fix5/absent_composition.json` → `guards`.

Records by the density-guard branch they took. The cutoff uses the local axis field wherever it is mapped; the other branches are the fallbacks:

| Set | Records | Local axis field | Bt(R0), axis field unmapped | Fixed 8e19 guard | No Thomson density | Records with dead channels | Dead channels |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 390 | 384 | 0 | 0 | 6 | 302 | 2603 |
| val | 48 | 46 | 0 | 0 | 2 | 41 | 371 |
| test | 49 | 48 | 0 | 0 | 1 | 39 | 356 |
| population | 13319 | 9926 | 4 | 961 | 2428 | 8948 | 113688 |

Source: `outputs/labeler/sawtooth/fix5/absent_composition.json` → `guards.*.records_by_density_guard_status`.

The guard accounting for each reviewed shot is in the radial-drop section below.

## Geometry and equilibrium

Channels 0–39 use the archived fixed RF grid; same-shot setup takes precedence, including the documented exceptional archived shot. Channels 40–47 are excluded from core, outer, coincidence, redistribution and inversion evidence. Nominal vacuum resonance is R=2×27.992 GHz/T×|F_boundary|/f. The EFIT magnetic axis selects the core. Every time sample with R₂<(2/3)R_LCFS,out is excluded; the shot core screen also rejects channels in that overlap region. Missing radial metadata does not establish definite-positive spatial evidence. Previously terminal-dependent candidates are retained as uncertainty.

The archived grid audit covers 25 shots with 1 documented setup exception. The fetched cohort setup check matches 500/500 first-40 grids. The fixed grid is transferred to other population shots; an unaudited historical setup change cannot be excluded.

Source: `outputs/labeler/sawtooth/fix5/fetched_frequency_audit.json`.

The adapted HL-3 outer input uses low-field-side nominal geometric ρ=0.4–0.65, beyond the typical inversion region, rather than adjacent array rows. Geometric ρ=|R−R_axis|/(R_LCFS,out−R_axis) is **not** normalized flux. Vacuum mapping omits relativistic and optical-depth corrections. Missing ECEZH uses the published first-40 midplane assumption explicitly. Full EFIT profiles are needed for q=1; minimal field/axis/boundary metadata cannot supply that comparison.

Source: `outputs/labeler/sawtooth/fix5/geometry_metadata_audit.json`.

The q=1 audit contains 11118 EFIT-supported shots, 11118 with a profile intersection check, and 4392 with paired diagnostic inversion points. All-point ΔR (inversion R minus same-side q=1 R, metres): n=97156; mean -0.056 m; quantiles at 0/5/25/50/75/95/100%: -0.327, -0.158, -0.102, -0.057, -0.013, 0.053, 0.396 m; 92.3% within 0.15 m. The full per-shot ledger includes checks with no axis-connected surface, checks with no candidate, and missing-profile cases. Paired nominal differences greater than 0.15 m flag uncertainty; a missing EFIT01 surface near q≈1 cannot distinguish reconstruction bias from the observed ECE train. It is recorded as incomparable. The comparison is low-field-side dominated: 96042 of 97156 paired points (98.9%) lie on the low-field side and 1114 on the high-field side, so the high-field side is effectively untested. The inversion lies close to one radius: |R_inversion − R_axis| is n=97156; mean 0.147 m; quantiles at 0/5/25/50/75/95/100%: 0.000, 0.106, 0.132, 0.146, 0.162, 0.191, 0.474 m.

Source: `outputs/labeler/sawtooth/fix5/q1_radius_audit.json` → `paired_point_side`.

TRAIN definite-present nominal inversion ρ: n=4191; mean 0.251; quantiles at 0/5/25/50/75/95/100%: 0.020, 0.148, 0.219, 0.250, 0.284, 0.348, 0.677. The shot-median outer input lies beyond the candidate inversion at 99.9% of comparable TRAIN definite-present points; the full ledger retains the distribution rather than assuming this holds for every event.

EFIT01 conflict is q_min>1.4. Sustained q_min≥1.5 for at least 50 ms is a prior, not an absence test: it marks time `q_prior_ece_contradicted` (inside the absence-test context of a periodic edge or profile candidate) or `q_prior_untested`, exported uncertain and never a benchmark negative. A magnetics-only reconstruction is not an MSE-constrained central-current measurement: the review-prescribed 1.3–1.5 band replaces the unsupported 1.05 cutoff, rather than estimating a calibrated q correction. Only prior TRAIN candidates enter the bias audit, read from the hashed snapshot of the earlier round's inputs (`labels/prior_inputs/fix2_inputs.json`). MSE-constrained q retains the stricter conflict test. Conflicting inversion-qualified trains remain uncertain over their full context; isolated POSR edges protect finite edge support without vetoing an entire high-q phase.

Prior TRAIN candidate q_min: n=1233; mean 1.278; quantiles at 0/5/25/50/75/95/100%: 1.050, 1.062, 1.121, 1.218, 1.314, 1.741, 5.290. Shot 186532 current state seconds: present 0.00 s, absent 0.00 s, q_prior_ece_contradicted 0.63 s, q_prior_untested 0.19 s, uncertain 4.20 s, unassessed 1.11 s.

Source: `outputs/labeler/sawtooth/fix5/qmin_bias_audit.json`.

Source: `outputs/labeler/sawtooth/fix5/freeze.json`.

This TRAIN subset measures sensitivity to the previous cutoff, not the true EFIT bias. The 1.4/1.5 guards are prescribed conservative tolerances, with no calibrated q correction. Muscatello's radial reference uses MSE-constrained EFIT; the present nominal comparison uses EFIT01. See [Muscatello et al. (2012)](https://doi.org/10.1088/0741-3335/54/2/025006) and the locally archived `Muscatello_ST.md` digest.

## Relaxation phases and support

Phase edges require both a ≥2% fractional drop and POSR≥6. The phase period floor is 10 ms, above the old 5.15 ms picker holdoff artifact and conservatively below Muscatello's DIII-D reference periods. Positive trains retain the frozen 20–250 ms bounds. Phase expansion requires at least six edges and shuffled-time p≤0.05. Each null preserves count, span and picker holdoff, repeats the same grouping, and compares the minimum gap CV over all groups. It accounts for search and multiplicity within each observable run. Independent generated noise and regular-train checks test this calibration, not physical label validity.

In 1,000 independent shuffled trials, the full search accepts 52/1000 noise sequences (5.2%; 95% CI 3.99–6.76%). It accepts 1000/1000 jittered regular trains and rejects a regular 6 ms sequence. Qualification by POSR is conditioned on in this timing-null audit, rather than simulated. There is no global familywise guarantee across shots/runs.

Source: `outputs/labeler/sawtooth/fix5/phase_null_audit.json`.

## Label states

State seconds. `Absent` is tested absence; the two q-prior states are q-prior only (exported as uncertain, no benchmark negatives).

| Split | Shots | Present s | Absent s | Q-prior, ECE-contradicted s | Q-prior, untested s | Uncertain s | Unassessed s | Assessed / observable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 400 | 168.195 | 60.420 | 180.591 | 126.384 | 956.240 | 630.728 | 15.3% |
| val | 50 | 22.858 | 3.726 | 24.662 | 16.630 | 111.158 | 85.797 | 14.8% |
| test | 50 | 13.870 | 7.245 | 26.721 | 15.180 | 116.975 | 86.005 | 11.7% |
| population | 16909 | 5229.408 | 1240.767 | 4420.038 | 3044.616 | 25301.251 | 42928.900 | 16.5% |

Population: 16909 of 16909 corpus shots have a record (complete); 13319 are usable and 3590 have none and contribute no state seconds, so their absence is not evidence of absence. Exclusion reasons: 2835 × ValueError: absent waveform or incompatible clock, 266 × ValueError: insufficient physically plausible ECE core channels, 262 × OSError: Unable to synchronously open file, 151 × ValueError: insufficient finite ECE in analysis window, 63 × ValueError: geometry must have finite, increasing sample times, 7 × ValueError: no ECE group, 4 × KeyError: "Unable to synchronously open object, 2 × ValueError: insufficient coherent physical ECE core channels. The population row counts the cohort shots as well. Records whose archived equilibrium lacks the field or axis metadata (radius status `bt_or_efit_axis_unavailable`) carry no present candidates and no absent seconds; they are almost entirely unassessed.

Source: `outputs/labeler/sawtooth/fix5/data_summary.json` → `splits; population`.

## Conditional agreement with the physics rule on assessed bins

OOF shots: 400 requested, 390 with usable physical inputs, 285 with assessed outcomes. Unsupported shots remain explicit unknown entries and supply no invented predictions or negatives.

OOF: fixed TRAIN shots, three whole-shot folds; inner shots select weights, LR/regularisation and operating points. Fixed val/test shots are excluded from all selection. Assessment masks affect loss and scoring, never the input definition. Uncertain and unassessed bins supply no negative truth. Excluded picks have unknown outcomes, not established false positives. CIs and paired comparisons use 1,000 whole-shot bootstrap draws.

The trivial picker differentiates only the EFIT-axis-selected single ECE channel. Inner shots independently select z for crash F1 and for the presence rule: any edge ≥zσ within ±125 ms. Thresholds, fitting/inner/scoring shot IDs and support counts are recorded for every fold. Always present supplies no crash time.

**What this benchmark can and cannot show.** Presence is a sanity check, close to trivial: the single-channel derivative baseline reaches AUROC 0.985 out of fold and 0.992 on the fixed validation shots, so a model has little room to separate itself. The labels are built from the same ECE edges the derivative baseline reads, so the derivative baseline's crash F1 (0.903) is higher than saw-ours (0.873): crash F1 here is agreement with the rule, not physical accuracy. Only 86 of the 285 assessed out-of-fold shots have both present and tested-absent bins, so most per-shot presence scores rest on one class. The blind expert queue is the real test of the labels and the models; nothing here replaces it. Presence is led by the threshold-free AUPRC, with F1 at one fixed threshold (0.5) beside the F1 at the inner-selected one.

| Method | Presence AUPRC [95% CI] | Presence AUROC [95% CI] | Presence F1, fixed 0.5 [95% CI] | Presence F1, inner-selected threshold [95% CI] | Crash F1 ±2 ms [95% CI] | Assessed / excluded / observable picks |
|---|---:|---:|---:|---:|---:|---:|
| Single-channel derivative / ±125 ms presence | 0.990 [0.983, 0.995] | 0.985 [0.977, 0.992] | 0.994 [0.990, 0.997] | 0.994 [0.990, 0.997] | 0.903 [0.884, 0.919] | 4155 / 17616 / 21771 |
| Always present | 0.736 [0.675, 0.790] | 0.500 [0.500, 0.500] | 0.848 [0.806, 0.882] | 0.848 [0.806, 0.882] | — | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.890 [0.853, 0.919] | 0.765 [0.697, 0.823] | 0.852 [0.813, 0.882] | 0.843 [0.796, 0.881] | 0.894 [0.872, 0.912] | 4071 / 16734 / 20805 |
| PhaseNet-style picker | 0.990 [0.982, 0.995] | 0.980 [0.967, 0.988] | 0.935 [0.908, 0.957] | 0.976 [0.965, 0.984] | 0.873 [0.847, 0.893] | 3647 / 11016 / 14663 |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.crash_tolerance_2ms; assessment_totals`.

Second held-out set: 47 nonexpert fixed-validation shots requested; 45 have usable physical inputs and 29 have assessed outcomes. Frozen three-fold ensembles and means of inner-selected thresholds. These still measure conditional teacher agreement.

| Method | Presence AUPRC [95% CI] | Presence AUROC [95% CI] | Presence F1, fixed 0.5 [95% CI] | Presence F1, inner-selected threshold [95% CI] | Crash F1 ±2 ms [95% CI] | Assessed / excluded / observable picks |
|---|---:|---:|---:|---:|---:|---:|
| Single-channel derivative / ±125 ms presence | 0.997 [0.987, 1.000] | 0.992 [0.967, 1.000] | — | 0.999 [0.993, 1.000] | 0.893 [0.830, 0.939] | 466 / 2054 / 2520 |
| Always present | 0.851 [0.713, 0.928] | 0.500 [0.500, 0.500] | — | 0.919 [0.832, 0.962] | — | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.957 [0.897, 0.986] | 0.784 [0.630, 0.905] | — | 0.928 [0.847, 0.968] | 0.892 [0.827, 0.938] | 465 / 2034 / 2499 |
| PhaseNet-style picker | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | — | 0.999 [0.995, 1.000] | 0.870 [0.825, 0.906] | 364 / 814 / 1178 |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.fixed_validation`.

| Held-out set | Method minus derivative-only | Δ crash F1 [95% CI] | Δ presence F1 [95% CI] |
|---|---|---:|---:|
| OOF | Single-channel derivative / ±125 ms presence (reference) | 0 | 0 |
| OOF | Adapted HL-3 / derivative picker gated by HL-3 | -0.009 [-0.018, -0.003] | -0.151 [-0.197, -0.114] |
| OOF | PhaseNet-style picker | -0.030 [-0.055, -0.009] | -0.018 [-0.029, -0.009] |
| OOF | Always present | — | -0.146 [-0.187, -0.113] |
| Fixed validation | Single-channel derivative / ±125 ms presence (reference) | 0 | 0 |
| Fixed validation | Adapted HL-3 / derivative picker gated by HL-3 | -0.001 [-0.004, 0.000] | -0.070 [-0.149, -0.032] |
| Fixed validation | PhaseNet-style picker | -0.023 [-0.065, 0.028] | 0.000 [-0.004, 0.005] |
| Fixed validation | Always present | — | -0.079 [-0.164, -0.037] |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `paired_vs_derivative`.

#### Sensitivity to the previous negatives

Benchmark negatives now come only from ECE-tested absence. The previous negatives were q-prior time: this table restores the q-prior states as scoring-only assessed time with no crashes, using the same fitted models and thresholds. Models are not refit. A large gap shows how much of the earlier score came from q-prior time.

| Held-out set | Method | Presence F1 now | Presence F1, previous negatives | AUROC now | AUROC, previous negatives | Crash F1 ±2 ms now | Crash F1, previous negatives |
|---|---|---:|---:|---:|---:|---:|---:|
| OOF | Single-channel derivative / ±125 ms presence | 0.994 | 0.855 | 0.985 | 0.922 | 0.903 | 0.860 |
| OOF | Always present | 0.848 | 0.478 | 0.500 | 0.500 | — | — |
| OOF | Adapted HL-3 / derivative picker gated by HL-3 | 0.843 | 0.491 | 0.765 | 0.621 | 0.894 | 0.855 |
| OOF | PhaseNet-style picker | 0.976 | 0.802 | 0.980 | 0.929 | 0.873 | 0.835 |
| Fixed validation | Single-channel derivative / ±125 ms presence | 0.999 | 0.897 | 0.992 | 0.938 | 0.893 | 0.872 |
| Fixed validation | Always present | 0.919 | 0.519 | 0.500 | 0.500 | — | — |
| Fixed validation | Adapted HL-3 / derivative picker gated by HL-3 | 0.928 | 0.549 | 0.784 | 0.717 | 0.892 | 0.871 |
| Fixed validation | PhaseNet-style picker | 0.999 | 0.773 | 1.000 | 0.967 | 0.870 | 0.846 |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.old_negatives_sensitivity`.

#### Operating points and their spread across folds

Presence and crash thresholds are selected on each fold's inner shots. A wide presence-threshold range for saw-ours means its presence operating point is not stable across folds, so the pooled presence F1 mixes operating points. Per fold, presence at one fixed threshold of 0.5 beside the inner-selected one:

| Method | Fold | Presence threshold | Crash threshold | Crash z | Inner shots | Best / completed epochs |
|---|---:|---:|---:|---:|---:|---:|
| Adapted HL-3 / derivative picker gated by HL-3 | 0 | 0.050 | 0.050 | 8.00 | 52 | 9 / 17 |
| Adapted HL-3 / derivative picker gated by HL-3 | 1 | 0.650 | 0.300 | 9.00 | 52 | 3 / 11 |
| Adapted HL-3 / derivative picker gated by HL-3 | 2 | 0.050 | 0.350 | 10.00 | 52 | 6 / 14 |
| Adapted HL-3 / derivative picker gated by HL-3 | spread | 0.050–0.650 (range 0.600) | | | | |
| PhaseNet-style picker | 0 | 0.250 | 0.950 | 0.00 | 52 | 21 / 29 |
| PhaseNet-style picker | 1 | 0.100 | 0.950 | 0.00 | 52 | 18 / 26 |
| PhaseNet-style picker | 2 | 0.550 | 0.975 | 0.00 | 52 | 16 / 24 |
| PhaseNet-style picker | spread | 0.100–0.550 (range 0.450) | | | | |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.operating_points`.

| Method | Fold | Held-out shots | Inner-selected threshold | Presence F1 at the selected threshold | Presence F1 at 0.5 | AUPRC | Positive / negative bins |
|---|---:|---:|---:|---:|---:|---:|---|
| Adapted HL-3 / derivative picker gated by HL-3 | 0 | 130 | 0.050 | 0.848 | 0.835 | 0.907 | 28,840 / 9,759 |
| Adapted HL-3 / derivative picker gated by HL-3 | 1 | 130 | 0.650 | 0.822 | 0.862 | 0.876 | 34,425 / 11,784 |
| Adapted HL-3 / derivative picker gated by HL-3 | 2 | 130 | 0.050 | 0.866 | 0.859 | 0.898 | 20,844 / 8,657 |
| PhaseNet-style picker | 0 | 130 | 0.250 | 0.971 | 0.970 | 0.991 | 28,840 / 9,759 |
| PhaseNet-style picker | 1 | 130 | 0.100 | 0.982 | 0.986 | 0.995 | 34,425 / 11,784 |
| PhaseNet-style picker | 2 | 130 | 0.550 | 0.971 | 0.828 | 0.985 | 20,844 / 8,657 |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.presence_fixed_threshold`.

HL-3 is an adapted external architecture, replacing the paper's SXR pair with geometry-selected ECE plus Mirnov and Ip. It has no crash head: its timing score is the **derivative picker gated by HL-3**. LR, weight decay and dropout are selected on inner shots. The U-Net probability grid extends through 0.999; fold records retain selected operating points and boundary flags.

Source: `outputs/labeler/sawtooth/fix5/saw-hl3_fold_0.json`.

Source: `outputs/labeler/sawtooth/fix5/saw-hl3_fold_1.json`.

Source: `outputs/labeler/sawtooth/fix5/saw-hl3_fold_2.json`.

Source: `outputs/labeler/sawtooth/fix5/saw-ours_fold_0.json`.

Source: `outputs/labeler/sawtooth/fix5/saw-ours_fold_1.json`.

Source: `outputs/labeler/sawtooth/fix5/saw-ours_fold_2.json`.

| Three-regime window classifier | Accuracy | Macro-F1 |
|---|---:|---:|
| Adapted HL-3 | 0.644 | 0.624 |
| Single-channel derivative with period | 0.871 | 0.872 |
| Fitting-chosen majority | 0.522 | 0.229 |
| Always present (period class undefined) | — | — |

Derivative period abstentions count as errors on the original class support. Always present has no period class or crash time.

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.three_class`.

Published HL-3 context, different task/population: real-time accuracy stated 0.922, count-derived 0.835; offline accuracy stated 0.956, count-derived 0.907. These three-regime classification scores are not DIII-D crash scores. The stated and count-derived accuracies differ in the source.

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `legacy`.

## Exploratory reviewed spans and old-rule disagreement

| Method | Pooled reviewed-span AUROC | Per-shot reviewed-span F1 | Known / excluded / observable picks |
|---|---:|---|---|
| Single-channel derivative / ±125 ms presence | 0.536 | 186636: 0.481; 189324: 0.755; 190637: 0.202 | 207 / 0 / 207 |
| Always present | 0.500 | 186636: 0.257; 189324: 0.682; 190637: 0.792 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.268 | 186636: 0.257; 189324: 0.682; 190637: 0.016 | 206 / 0 / 206 |
| PhaseNet-style picker | 0.751 | 186636: 0.295; 189324: 0.718; 190637: 0.806 | 94 / 0 / 94 |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.expert`.

Reviewed-score support is the intersection of known reviewed spans and observable inputs. Picks outside it have unknown outcomes. The JSON separately records picks excluded by physics-rule assessment for the conditional agreement diagnostic. Model predictions on reviewed spans are scored independently of the physics rule's assessment mask.

HL-3 ranking check: AUROC below 0.5 persists on listed exploratory expert shots; no expert-based inversion, retuning or threshold selection performed. No expert-based inversion or retuning was performed.

| Reviewed shot | HL-3 AUROC | HL-3 assessed diagnostic AUROC | Saw-ours AUROC |
|---|---:|---:|---:|
| 186636 | 0.205 | — | 0.808 |
| 189324 | 0.468 | 0.659 | 0.868 |
| 190637 | 0.085 | — | 0.741 |

Source: `outputs/labeler/sawtooth/fix5/benchmark.json` → `Tokamak-SI.*.expert.by_shot`.

HL-3's residual inversion is concentrated in shot 190637 and persists on assessed support. The physics rule calls no definite-present phase there while the reviewed spans contain substantial positive support. The radial profile below resolves which side the data support.

Source: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix5/shots/190637.json` → `state_seconds`; `outputs/labeler/sawtooth/fix5/validation.json` → `expert.by_shot`.

## Radial drop profiles of the reviewed spans

Each profile is the median over relaxation events of every ECE channel's relative Te change across the event (medians over 0.3–1.5 ms after against 0.3–1.5 ms before), plotted against signed nominal ρ (negative on the high-field side). Resolution is one ECE channel, about 0.03–0.05 in ρ near the axis, so it cannot separate structure inside about 0.05 of the axis, and ρ is geometric, not flux. Events are the rule's accepted crash points and periodic core edges inside the expert-positive spans (all events of 186532, which has no reviewed span). The verified sawtooth shot 192148 (68 events) is the reference: central -0.231, outer rise 0.131. The verdict rule was written before any reviewed shot was read: central drop ≤ −0.05 inside |ρ|<0.15 and an outer rise ≥ +0.02 at ρ 0.3–0.6 is sawtooth-like; no central drop with a monotone outward decline to ≤ −0.15 at the outermost channel is edge-driven; otherwise indeterminate. A second, independent test counts the fraction of events within ±1 ms of a filterscope D-alpha spike (6 robust standard deviations over the surrounding ±25 ms), against the same fraction at the event times shifted by 35–100 ms; a shot marks edge-localized modes when at least half of its events coincide with a spike and at least twice the shifted fraction does.

| Shot | Events | Central change | Outer rise (ρ 0.3–0.6) | Outermost change (ρ) | Span s | Observable after guards s | D-alpha coincident / shifted | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 192148 (reference) | 68 | -0.231 | 0.131 | — | — | — | 0.00 / 0.00 | sawtooth-like |
| 186636 | 22 | -0.022 | 0.018 | 0.001 (0.45) | 2.86 | 0.43 | 0.09 / 0.07 | indeterminate |
| 189324 | 40 | -0.252 | 0.165 | 0.014 (0.87) | 2.74 | 2.74 | 0.00 / 0.00 | sawtooth-like |
| 190637 | 43 | -0.002 | -0.028 | -0.132 (0.86) | 3.92 | 3.92 | 0.74 / 0.05 | indeterminate |
| 186532 | 64 | -0.005 | -0.003 | -0.004 (0.45) | 6.13 | 5.02 | 0.38 / 0.14 | indeterminate |

**Shot 190637: the expert span marks edge-localized modes, not a central sawtooth. 74% of its 43 events lie within ±1 ms of a filterscope D-alpha spike, against 5% at the same times shifted by 35–100 ms (reference sawtooth shot 192148: 0.00).** The span is 3.92 s and 3.92 s of it is observable after the guards. The median event has no central drop (-0.002, against -0.231 for the verified sawtooth) and no outer rise (-0.028); the drop grows outward to -0.132 at ρ≈0.86. The pre-registered edge-driven bar (−0.15) is missed by 0.02, so the formal ECE verdict is indeterminate. Q-prior time (EFIT01 q_min at least 1.5 for 50 ms) covers 4.29 s of the shot, consistent with no q=1 surface. The rule is right not to call this span a present sawtooth, and the expert-positive label, taken as a central sawtooth, is not supported by ECE.

**Shot 186636: cutoff, so the span is untestable, not wrong.** Only 0.43 s of the 2.86 s span (15%) stays observable. With the local-field density guard the cutoff removes 25,981 of 59,515 core-observable samples (44%; the earlier B0(R0) guard removed 19%) and ECE validity 6% more. The pre-event Te profile (figure) has a steep edge, falling from about 2.6 to 1.2 keV within about 5 cm just outside the axis. On the events that survive the central change is -0.022 with an outer rise of 0.018: the right sign but about a tenth of the reference amplitude, below the pre-registered thresholds. ECE cannot confirm or refute the expert span on most of it; the rule abstains correctly (1.70 s uncertain, 3.17 s unassessed, no present) and the expert span stays untested.

**Shot 189324 agrees.** Central -0.252 and outer rise 0.165 match the reference sawtooth shape, and the rule labels 1.84 s present. **Shot 186532** (not a reviewed shot; it carried the weak gallery crash panel of earlier rounds) shows no relaxation at its 64 rule events (central -0.005, outer -0.003); its three accepted crash points are weak and uncertain. The panel is left as it is: the rule is not changed for one shot and the weak panel is reported, not replaced.

Figures: `$LABELER_ROOT/round4/saw/fix5/figures/radial_drop_<shot>.pdf` and `.png` for 186636, 189324, 190637 and 186532.

Source: `outputs/labeler/sawtooth/fix5/radial_drop_profiles.json`.

| Reviewed shot | New recall | Legacy recall | New F1 | Uncertain / observable positive bins |
|---|---:|---:|---:|---:|
| 186636 | 0.000 | 0.624 | 0.000 | 218 / 218 |
| 189324 | 0.500 | 0.253 | 0.598 | 683 / 1367 |
| 190637 | 0.000 | 0.250 | 0.000 | 1957 / 1957 |

Span-supported picks are not crash precision/recall.

Source: `outputs/labeler/sawtooth/fix5/validation.json` → `expert; legacy_agreement`.

Native clipped legacy replay reproduces every retained pick array on the seven requested shots; clocks are exact. This excludes a reader timestamp bug. The ledger separates missing raw edges, greedy 10 ms suppression and inversion-window rejection; only training shots enter the diagnostic holdoff ablation. The current catalog detector and the retained legacy detector are distinct, so a catalog-wide defect cannot be inferred from the legacy cache alone.

Source: `outputs/labeler/sawtooth/fix5/legacy_reader_audit.json`.

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

Source: `outputs/labeler/sawtooth/fix5/legacy_disagreement.json` → `by_shot; figures`.

| Shot 192090 current catalog | Total picks | In physics-present support | TP / FP / FN ±2 ms | Median offset ms |
|---|---:|---:|---|---:|
| all | 53 | 18 | 17 / 1 / 13 | 0.050 |
| ece | 42 | 15 | 15 / 0 / 15 | 0.150 |
| sxr | 11 | 3 | 2 / 1 / 28 | 1.850 |

These TP/FP/FN cells name algorithm agreement against the physics rule. They do not establish physical errors. The current detector does not reproduce the retained legacy's systematic 10 ms offset on this available shot. The other six current event stores are unavailable locally.

Source: `outputs/labeler/sawtooth/fix5/legacy_disagreement.json` → `by_shot.current_production_v3_comparison`.

## Uncertain gaps inside present trains

A present train is split wherever a crash is missing or uncertain, and the gap between present spans is exported `uncertain`. Joining present spans across gaps of at most 0.5 s that are entirely uncertain gives 199 blocks with 804 gaps on the cohort. The gaps are 80.0 s, 30.6% of block time and 4.3% of observable time. 543 gaps (49.4 s) contain no accepted crash: a crash was rejected by the profile, central-drop or coincidence tests, or fell inside the holdoff, and the following gap breaks the period-ratio test; the span is then neither present nor tested absent. 261 gaps (30.6 s) contain an accepted crash left uncertain (283 × unverified_spatial_adjacency, 7 × qmin_conflict, 202 × nominal_q1_radius_mismatch).

Shot 192148 has 8 such gaps totalling 0.96 s: 1.62–1.75 s (uncertain crash); 1.83–1.93 s (no accepted crash); 2.28–2.38 s (no accepted crash); 2.49–2.74 s (no accepted crash); 3.00–3.11 s (no accepted crash); 3.20–3.31 s (no accepted crash); 4.04–4.14 s (no accepted crash); 5.08–5.13 s (no accepted crash). Durations run 43–254 ms against the shot's median accepted-crash period of about 50 ms, so most are one or two missed crashes and the longest is about five.

Source: `outputs/labeler/sawtooth/fix5/uncertain_gaps.json`.

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

Which gate decides the reference crashes. Each derivative-picker crash inside the published windows is attributed to the first gate that turned the rule's candidate down (`rejected_events` of the detector), or to the train test when every gate accepted it, or to no candidate when no two-channel edge cluster exists there:

- Shot 141182: frozen rule, 6 × core not observable at the crash. With the validity test off: 5 × found by the rule; 1 × passed every gate but is not part of a stable train.
- Shot 141195: frozen rule, 6 × core not observable at the crash. With the validity test off: 3 × found by the rule; 3 × profile falls or rises everywhere (A_net).

Source: `outputs/labeler/sawtooth/fix5/muscatello_rule_check.json`.

Blind queue: 200 windows on 44 shots, including 141 primary random windows. Stratum counts: `{'random_observable': 141, 'zero_candidate_shot': 15, 'random_observable_reserve': 6, 'algorithm_uncertain': 15, 'model_predicted_negative': 9, 'disagreement': 14}`. Random observable windows are frozen before prediction access. Candidate-free, model-negative, uncertain and disagreement supplements are separate strata. Only sensor inputs and blank targets enter the annotation pack; private selection and all predictions remain hidden. About 97 independent positive events give a worst-case 95% recall half-width of 0.1; clustered events do not supply that effective sample size automatically. Primary sampling weights and whole-shot CIs are preregistered. Annotation remains pending.

Source: `outputs/labeler/sawtooth/fix5/crash_time_queue.json`.

## Artifacts, reproduction and verification

Complete labels: `$LABELER_ROOT/round4/saw/fix5/labels/`. Manifest `SHA256SUMS` sha256: `47081aebcad868911bf3e38817b3e0c79e4ba9f02efe5fcee84780ad00a0ef14`. The population run is complete: all 16909 corpus shots were attempted; 13319 have a usable label record and 3590 do not (reader or physical-core failures; the counts by reason are under Label states). Population shards export current states without duplicate cohort rows; failed-shot records remain in the ledger. The cohort bundle is separate. Production stores were read only. The manifest also hashes `prior_inputs/fix2_inputs.json`, the snapshot of the previous round's inputs the rule reads (terminal-dependent candidate times, failed-shot list, TRAIN q_min conflict candidates), so the labels no longer depend on a deleted directory; `freeze.json` is hashed beside them.

Source: `outputs/labeler/sawtooth/fix5/label_manifest.json`.

Large signals, EFIT metadata, models, predictions, annotation pack, PDFs and 150-dpi PNGs are under the same fix5 directory. Paper example, three old-rule figures, the 12-shot gallery and the four radial-profile figures were inspected; figure ledgers retain paths and hashes.

Source: `outputs/labeler/sawtooth/fix5/paper_example.json`.

Source: `outputs/labeler/sawtooth/fix5/gallery.json`.

Source: `outputs/labeler/sawtooth/fix5/figure_inspection.json`.

Covering tests: 168 passed. Ruff passes on all changed Python files; formatting passes on all new Python files; git diff --check passes. Long jobs used timeouts and logs; temporary storage was swept afterward. GPU training used CUDA_VISIBLE_DEVICES=1 within the assigned memory budget.

Source: `outputs/labeler/sawtooth/fix5/verification.json`.

```text
........................................................................ [ 42%]
........................................................................ [ 85%]
........................                                                 [100%]
168 passed in 14.95s
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/lib/python3.11/site-packages/XRootD/client/finalize.py:46: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if isinstance(obj, File) and obj.is_open():
All checks passed!
3 files already formatted
```

Reproduce with the prescribed pixi environment and TMPDIR. Entrypoints: sawtooth_edge_context.py run/derive (TRAIN only, first); sawtooth_fix5_setup.py; sawtooth_geometry_fix3.py audit/frequency-audit/reference; sawtooth_physics.py labels (cohort, then population with sawtooth_population.sbatch); sawtooth_reference_rule.py; sawtooth_fix3_records.py records/manifest; sawtooth_absent_composition.py; sawtooth_uncertain_gaps.py; sawtooth_radial_drop.py; sawtooth_fix_validation.py validate; sawtooth_fix3_artifacts.py queue-base/reader-audit/queue/figures; sawtooth_benchmark.py train/predict/evaluate; sawtooth_gallery.py; sawtooth_phase_null_audit.py; sawtooth_fix3_report.py. Run queue-base before model prediction access. Full command logs remain under the large output directory.

## Concerns and next work

Known limitations, all left as they are because a rule change would force a full population rerun:

- Uncertain dominates: 64.5% of observable population time and 30252 of 158230 candidates are uncertain. Present plus tested absent is 16.5% of observable time, so all conditional scores describe that fraction.
- The negative class is thin: tested absence is 3.9% of observable cohort time, so the assessed set is positive-heavy and presence AUROC and precision are less informative than before; the old-negatives sensitivity table is the comparison to the earlier scoring.
- The ECE-validity step test reads a static calibration step as an invalid profile. On the Muscatello reference shots it removes nearly all time (observable 0.00 s and 0.47 s of 3.1 s), so the frozen rule labels almost nothing on them. It is the review's test and it is not tuned on those shots.
- The EFIT01 q=1 radius conflict is conservative. Shot 203563 (EFIT01 q_min 0.77, inversion minus q=1 radius -0.198 m) has 26 of 29 candidates uncertain although it shows regular trains. The paired comparison is 98.9% low-field side.
- Uncertain gaps inside present trains cover 30.6% of block time on the cohort; most are missed crashes that break the period-ratio test.
- Population shots without archived field or axis metadata, and shots without a usable ECE waveform, receive no definite label (see the exclusion counts above).
- The reviewed 186636 span is 85% cutoff or invalid ECE and is untestable here; the 190637 span is not a central sawtooth by ECE. Neither is a calibrated physical truth.
- The legacy-offset diagnosis traces the old detector's offsets to its inversion-profile gate and holdoff, not to the reader. The current catalog detector could be compared only on shot 192090.
- Each gallery crash panel shows the accepted crash with the largest A_norm (present crashes first; an uncertain crash only when the shot has no present one), a fixed rule, so panels are not chosen to be easy or hard.

Blind physical accuracy remains unmeasured. Reviewed-span failures and any inverted HL-3 ranking remain failures, not reasons to retune on those shots. Nominal geometry is not a calibrated flux measurement; missing field/profile data limits population evidence. Algorithm-assessed benchmarks remain conditional and can be solved by derivative rules. Next: obtain the queued blind crash/span/ambiguity annotations, lock them, and evaluate all frozen methods over independently observable support without retuning.

## Appendix: superseded history

Earlier rounds used a hottest-channel proxy, sparse same-shot RF localization, a q_min>1.05 conflict and unbounded fractional edge phases. Their results are superseded. The first-round `outputs/labeler/sawtooth/fix` records and the two earlier plan documents were removed; `outputs/labeler/sawtooth/fix2` and `fix3` remain as predecessor records (fix3 counted q_min≥1.5 time as absent; its absent-class and score numbers are superseded by this round), and the earlier rounds' narrative is in the stream report appendix.
