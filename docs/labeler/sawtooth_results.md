# Sawtooth physics-rule labels: current state

These are **physics-rule labels validated only by the checks described here**. There are no blind expert crash times. Neither physical label accuracy nor improvement over the production catalog is established. The reviewed spans are anchored to old suggestions and were consulted in previous rule revisions; they are exploratory, not untouched validation. No model is recommended as latest or stable.

The cohort has 487/500 successful records and 6727 diagnostic crash candidates. The population has 13319/16909 successful records and 170856 diagnostic points. OOF model scores assess 299,978/892,635 observable bins (33.6%).

Source: `outputs/labeler/sawtooth/fix3/cohort_labels.json`.

Source: `outputs/labeler/sawtooth/fix3/population_labels.json`.

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.saw-ours.assessment_totals`.

Conditional OOF crash F1 at ±2 ms: derivative 0.843, HL-3-gated derivative 0.838, saw-ours 0.797. The derivative baseline has the highest point estimate; the HL-3 paired crash-F1 interval includes zero. HL-3's expert-shot ranking remains inverted on shot 190637; independent physical validation remains pending.

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.*`.

## Geometry and equilibrium

Channels 0–39 use the archived fixed RF grid; same-shot setup takes precedence, including the documented exceptional archived shot. Channels 40–47 are excluded from core, outer, coincidence, redistribution and inversion evidence. Nominal vacuum resonance is R=2×27.992 GHz/T×|F_boundary|/f. The EFIT magnetic axis selects the core. Every time sample with R₂<(2/3)R_LCFS,out is excluded; the shot core screen also rejects channels in that overlap region. Missing radial metadata does not establish definite-positive spatial evidence. Previously terminal-dependent candidates are retained as uncertainty.

The archived grid audit covers 25 shots with 1 documented setup exception. The fetched cohort setup check matches 500/500 first-40 grids. The fixed grid is transferred to other population shots; an unaudited historical setup change cannot be excluded.

Source: `outputs/labeler/sawtooth/fix3/fetched_frequency_audit.json`.

The adapted HL-3 outer input uses low-field-side nominal geometric ρ=0.4–0.65, beyond the typical inversion region, rather than adjacent array rows. Geometric ρ=|R−R_axis|/(R_LCFS,out−R_axis) is **not** normalized flux. Vacuum mapping omits relativistic and optical-depth corrections. Missing ECEZH uses the published first-40 midplane assumption explicitly. Full EFIT profiles are needed for q=1; minimal field/axis/boundary metadata cannot supply that comparison.

Source: `outputs/labeler/sawtooth/fix3/geometry_metadata_audit.json`.

The q=1 audit contains 11118 EFIT-supported shots, 11118 with a profile intersection check, and 4677 with paired diagnostic inversion points. All-point ΔR (inversion R minus same-side q=1 R, metres): n=101132; mean -0.056 m; quantiles at 0/5/25/50/75/95/100%: -0.327, -0.160, -0.103, -0.057, -0.012, 0.056, 0.411 m; 91.7% within 0.15 m. The full per-shot ledger includes checks with no axis-connected surface, checks with no candidate, and missing-profile cases. Paired nominal differences greater than 0.15 m flag uncertainty; a missing EFIT01 surface near q≈1 cannot distinguish reconstruction bias from the observed ECE train. It is recorded as incomparable.

Source: `outputs/labeler/sawtooth/fix3/q1_radius_audit.json`.

TRAIN definite-present nominal inversion ρ: n=4416; mean 0.248; quantiles at 0/5/25/50/75/95/100%: 0.002, 0.134, 0.215, 0.249, 0.284, 0.356, 0.787. The shot-median outer input lies beyond the candidate inversion at 99.6% of comparable TRAIN definite-present points; the full ledger retains the distribution rather than assuming this holds for every event.

EFIT01 conflict is q_min>1.4. Sustained q_min≥1.5 for at least 50 ms supplies absence evidence. A magnetics-only reconstruction is not an MSE-constrained central-current measurement: the review-prescribed 1.3–1.5 band replaces the unsupported 1.05 cutoff, rather than estimating a calibrated q correction. Only prior TRAIN candidates enter the bias audit. MSE-constrained q retains the stricter conflict test. Conflicting inversion-qualified trains remain uncertain over their full context; isolated POSR edges protect finite edge support without vetoing an entire high-q phase.

Prior TRAIN candidate q_min: n=1233; mean 1.278; quantiles at 0/5/25/50/75/95/100%: 1.050, 1.062, 1.121, 1.218, 1.314, 1.741, 5.289. Shot 186532 current state seconds: present 0.00 s, absent 1.09 s, uncertain 4.77 s, unassessed 0.28 s.

Source: `outputs/labeler/sawtooth/fix3/qmin_bias_audit.json`.

Source: `outputs/labeler/sawtooth/fix3/freeze.json`.

This TRAIN subset measures sensitivity to the previous cutoff, not the true EFIT bias. The 1.4/1.5 guards are prescribed conservative tolerances, with no calibrated q correction. Muscatello's radial reference uses MSE-constrained EFIT; the present nominal comparison uses EFIT01. See [Muscatello et al. (2012)](https://doi.org/10.1088/0741-3335/54/2/025006) and the locally archived `Muscatello_ST.md` digest.

## Relaxation phases and support

Phase edges require both a ≥2% fractional drop and POSR≥6. The phase period floor is 10 ms, above the old 5.15 ms picker holdoff artifact and conservatively below Muscatello's DIII-D reference periods. Positive trains retain the frozen 20–250 ms bounds. Phase expansion requires at least six edges and shuffled-time p≤0.05. Each null preserves count, span and picker holdoff, repeats the same grouping, and compares the minimum gap CV over all groups. It accounts for search and multiplicity within each observable run. Independent generated noise and regular-train checks test this calibration, not physical label validity.

In 1,000 independent shuffled trials, the full search accepts 52/1000 noise sequences (5.2%; 95% CI 3.99–6.76%). It accepts 1000/1000 jittered regular trains and rejects a regular 6 ms sequence. Qualification by POSR is conditioned on in this timing-null audit, rather than simulated. There is no global familywise guarantee across shots/runs.

Source: `outputs/labeler/sawtooth/fix3/phase_null_audit.json`.

| Split | Shots | Present s | Absent s | Uncertain s | Unassessed s | Assessed / observable |
|---|---:|---:|---:|---:|---:|---:|
| train | 400 | 177.529 | 422.319 | 1185.450 | 337.260 | 33.6% |
| val | 50 | 25.720 | 49.705 | 139.739 | 49.666 | 35.1% |
| test | 50 | 15.150 | 53.261 | 144.111 | 53.474 | 32.2% |
| population | 16909 | 5595.259 | 9756.898 | 29952.766 | 36860.057 | 33.9% |

Population: 13319 of 16909 shots have a label record; 3590 have none and contribute no state seconds, so their absence is not evidence of absence. Exclusion reasons: 2835 × ValueError: absent waveform or incompatible clock, 266 × ValueError: insufficient physically plausible ECE core channels, 262 × OSError: Unable to synchronously open file, 151 × ValueError: insufficient finite ECE in analysis window, 63 × ValueError: geometry must have finite, increasing sample times, 7 × ValueError: no ECE group, 4 × KeyError: "Unable to synchronously open object, 2 × ValueError: insufficient coherent physical ECE core channels. The population row counts the cohort shots as well. Records whose archived equilibrium lacks the field or axis metadata (radius status `bt_or_efit_axis_unavailable`) carry no present candidates and no absent seconds; they are almost entirely unassessed.

Source: `outputs/labeler/sawtooth/fix3/data_summary.json` → `splits; population`.

## Conditional agreement with the physics rule on assessed bins

OOF shots: 400 requested, 390 with usable physical inputs, 387 with assessed outcomes. Unsupported shots remain explicit unknown entries and supply no invented predictions or negatives.

OOF: fixed TRAIN shots, three whole-shot folds; inner shots select weights, LR/regularisation and operating points. Fixed val/test shots are excluded from all selection. Assessment masks affect loss and scoring, never the input definition. Uncertain and unassessed bins supply no negative truth. Excluded picks have unknown outcomes, not established false positives. CIs and paired comparisons use 1,000 whole-shot bootstrap draws.

The trivial picker differentiates only the EFIT-axis-selected single ECE channel. Inner shots independently select z for crash F1 and for the presence rule: any edge ≥zσ within ±125 ms. Thresholds, fitting/inner/scoring shot IDs and support counts are recorded for every fold. Always present supplies no crash time.

| Method | Crash F1 ±2 ms [95% CI] | Presence F1 [95% CI] | AUROC | AUPRC | Assessed / excluded / observable picks |
|---|---:|---:|---:|---:|---:|
| Single-channel derivative / ±125 ms presence | 0.843 [0.812, 0.870] | 0.907 [0.881, 0.927] | 0.943 | 0.841 | 4297 / 14985 / 19282 |
| Always present | — | 0.457 [0.403, 0.506] | 0.500 | 0.296 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.838 [0.804, 0.864] | 0.632 [0.574, 0.681] | 0.835 | 0.631 | 4159 / 13297 / 17456 |
| PhaseNet-style picker | 0.797 [0.763, 0.825] | 0.837 [0.804, 0.865] | 0.954 | 0.919 | 3652 / 8892 / 12544 |

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.*.crash_tolerance_2ms; assessment_totals`.

Second held-out set: 47 nonexpert fixed-validation shots requested; 45 have usable physical inputs and 44 have assessed outcomes. Frozen three-fold ensembles and means of inner-selected thresholds. These still measure conditional teacher agreement.

| Method | Crash F1 ±2 ms [95% CI] | Presence F1 [95% CI] | AUROC | AUPRC | Assessed / excluded / observable picks |
|---|---:|---:|---:|---:|---:|
| Single-channel derivative / ±125 ms presence | 0.825 [0.716, 0.898] | 0.928 [0.836, 0.976] | 0.945 | 0.884 | 485 / 1652 / 2137 |
| Always present | — | 0.528 [0.346, 0.666] | 0.500 | 0.359 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.823 [0.716, 0.898] | 0.736 [0.588, 0.844] | 0.864 | 0.729 | 473 / 1525 / 1998 |
| PhaseNet-style picker | 0.752 [0.651, 0.842] | 0.908 [0.843, 0.946] | 0.983 | 0.965 | 357 / 799 / 1156 |

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.*.fixed_validation`.

| Held-out set | Method minus derivative-only | Δ crash F1 [95% CI] | Δ presence F1 [95% CI] |
|---|---|---:|---:|
| OOF | Single-channel derivative / ±125 ms presence (reference) | 0 | 0 |
| OOF | Adapted HL-3 / derivative picker gated by HL-3 | -0.005 [-0.015, 0.002] | -0.276 [-0.319, -0.235] |
| OOF | PhaseNet-style picker | -0.046 [-0.083, -0.011] | -0.071 [-0.101, -0.044] |
| OOF | Always present | — | -0.450 [-0.488, -0.412] |
| Fixed validation | Single-channel derivative / ±125 ms presence (reference) | 0 | 0 |
| Fixed validation | Adapted HL-3 / derivative picker gated by HL-3 | -0.002 [-0.012, 0.006] | -0.192 [-0.327, -0.090] |
| Fixed validation | PhaseNet-style picker | -0.073 [-0.196, 0.042] | -0.020 [-0.080, 0.055] |
| Fixed validation | Always present | — | -0.400 [-0.523, -0.286] |

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `paired_vs_derivative`.

HL-3 is an adapted external architecture, replacing the paper's SXR pair with geometry-selected ECE plus Mirnov and Ip. It has no crash head: its timing score is the **derivative picker gated by HL-3**. LR, weight decay and dropout are selected on inner shots. The U-Net probability grid extends through 0.999; fold records retain selected operating points and boundary flags.

Source: `outputs/labeler/sawtooth/fix3/saw-hl3_fold_0.json`.

Source: `outputs/labeler/sawtooth/fix3/saw-hl3_fold_1.json`.

Source: `outputs/labeler/sawtooth/fix3/saw-hl3_fold_2.json`.

Source: `outputs/labeler/sawtooth/fix3/saw-ours_fold_0.json`.

Source: `outputs/labeler/sawtooth/fix3/saw-ours_fold_1.json`.

Source: `outputs/labeler/sawtooth/fix3/saw-ours_fold_2.json`.

| Three-regime window classifier | Accuracy | Macro-F1 |
|---|---:|---:|
| Adapted HL-3 | 0.653 | 0.562 |
| Single-channel derivative with period | 0.901 | 0.804 |
| Fitting-chosen majority | 0.704 | 0.275 |
| Always present (period class undefined) | — | — |

Derivative period abstentions count as errors on the original class support. Always present has no period class or crash time.

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.*.three_class`.

Published HL-3 context, different task/population: real-time accuracy stated 0.922, count-derived 0.835; offline accuracy stated 0.956, count-derived 0.907. These three-regime classification scores are not DIII-D crash scores. The stated and count-derived accuracies differ in the source.

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `legacy`.

## Exploratory reviewed spans and old-rule disagreement

| Method | Pooled reviewed-span AUROC | Per-shot reviewed-span F1 | Known / excluded / observable picks |
|---|---:|---|---|
| Single-channel derivative / ±125 ms presence | 0.508 | 186636: 0.369; 189324: 0.730; 190637: 0.000 | 226 / 0 / 226 |
| Always present | 0.500 | 186636: 0.546; 189324: 0.658; 190637: 0.792 | 0 / 0 / 0 |
| Adapted HL-3 / derivative picker gated by HL-3 | 0.422 | 186636: 0.505; 189324: 0.753; 190637: 0.006 | 224 / 0 / 224 |
| PhaseNet-style picker | 0.665 | 186636: 0.309; 189324: 0.767; 190637: 0.043 | 69 / 0 / 69 |

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.*.expert`.

Reviewed-score support is the intersection of known reviewed spans and observable inputs. Picks outside it have unknown outcomes. The JSON separately records picks excluded by physics-rule assessment for the conditional agreement diagnostic. Model predictions on reviewed spans are scored independently of the physics rule's assessment mask.

HL-3 ranking check: AUROC below 0.5 persists on listed exploratory expert shots; no expert-based inversion, retuning or threshold selection performed. No expert-based inversion or retuning was performed.

| Reviewed shot | HL-3 AUROC | HL-3 assessed diagnostic AUROC | Saw-ours AUROC |
|---|---:|---:|---:|
| 186636 | 0.725 | 0.920 | 0.706 |
| 189324 | 0.771 | 0.670 | 0.758 |
| 190637 | 0.040 | 0.048 | 0.630 |

Source: `outputs/labeler/sawtooth/fix3/benchmark.json` → `Tokamak-SI.*.expert.by_shot`.

HL-3's residual inversion is concentrated in shot 190637 and persists on assessed support. The physics rule calls no definite-present phase there while the reviewed spans contain substantial positive support. This is a teacher/review disagreement; it does not establish which labels are physically correct. The model is not validated for use.

Source: `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix3/shots/190637.json` → `state_seconds`; `outputs/labeler/sawtooth/fix3/validation.json` → `expert.by_shot`.

| Reviewed shot | New recall | Legacy recall | New F1 | Uncertain / observable positive bins |
|---|---:|---:|---:|---:|
| 186636 | 0.000 | 0.471 | 0.000 | 788 / 904 |
| 189324 | 0.500 | 0.253 | 0.598 | 683 / 1367 |
| 190637 | 0.000 | 0.250 | 0.000 | 54 / 1957 |

Span-supported picks are not crash precision/recall.

Source: `outputs/labeler/sawtooth/fix3/validation.json` → `expert; legacy_agreement`.

Native clipped legacy replay reproduces every retained pick array on the seven requested shots; clocks are exact. This excludes a reader timestamp bug. The ledger separates missing raw edges, greedy 10 ms suppression and inversion-window rejection; only training shots enter the diagnostic holdoff ablation. Current catalog v3 and the retained legacy v2 are distinct detectors, so a catalog-wide defect cannot be inferred from the legacy cache alone.

Source: `outputs/labeler/sawtooth/fix3/legacy_reader_audit.json`.

Per-shot timing and gate diagnosis (old minus physics, ms):

| Shot / split | Prior offset ms | Fresh offset ms (pairs) | Fresh present picks | Observable s | Derivative matches: legacy / no holdoff |
|---|---:|---:|---:|---:|---:|
| 190602 / train | -9.30 | — (0) | 0 | 0.000 | 0 / 0 |
| 190604 / test | -9.90 | — (0) | 0 | 0.000 | — / — |
| 192090 / train | -4.85 | -4.55 (10) | 30 | 6.000 | 12 / 18 |
| 201948 / train | 9.90 | — (0) | 0 | 5.848 | 0 / 0 |
| 192154 / test | 9.90 | 10.70 (13) | 59 | 5.551 | — / — |
| 191384 / train | -8.70 | -8.60 (22) | 33 | 5.372 | 3 / 6 |
| 203349 / train | -10.30 | -4.55 (32) | 102 | 5.760 | 0 / 1 |

Prior offsets refer to the superseded rule, whose spatial selection included unverified channels. Fresh offsets retain pairs within 15 ms; nearest neighbours are descriptive and can be reused. One-to-one cells are separate. Shots 190602/190604 have too few uncontaminated channels at their field for a verified redistribution profile and correctly abstain. The train-only ablation changes holdoff solely for diagnosis. Its full ledger gives the inversion rejection reasons and core/outer changes at both rules' picks. Shortening the legacy profile windows alone does not recover the matches: its block interiority, gain adjacency and amplitude conditions reject many central derivative edges.

Source: `outputs/labeler/sawtooth/fix3/legacy_disagreement.json` → `by_shot; figures`.

| Shot 192090 current catalog | Total picks | In physics-present support | TP / FP / FN ±2 ms | Median offset ms |
|---|---:|---:|---|---:|
| all | 53 | 18 | 17 / 1 / 13 | 0.050 |
| ece | 42 | 15 | 15 / 0 / 15 | 0.150 |
| sxr | 11 | 3 | 2 / 1 / 28 | 1.850 |

These TP/FP/FN cells name algorithm agreement against the physics rule. They do not establish physical errors. The current detector does not reproduce the retained legacy's systematic 10 ms offset on this available shot. The other six current event stores are unavailable locally.

Source: `outputs/labeler/sawtooth/fix3/legacy_disagreement.json` → `by_shot.current_production_v3_comparison`.

## Muscatello references and blind queue

| Shot | Window s | Period ms | Published period ms | Relative amplitude | Published amplitude | Status |
|---|---|---:|---:|---:|---:|---|
| 141182 | 2.7–3.0 | 92.10 | 85 ± 5 | 0.355 | 0.35 ± 0.02 | independent_reference_measurement |
| 141182 | 4.55–4.8 | 87.90 | 85 ± 5 | 0.384 | 0.35 ± 0.02 | independent_reference_measurement |
| 141195 | 2.7–3.0 | 97.75 | 85 ± 5 | 0.350 | 0.35 ± 0.02 | independent_reference_measurement |
| 141195 | 4.55–4.8 | 86.45 | 85 ± 5 | 0.492 | 0.35 ± 0.02 | independent_reference_measurement |

Compare only the published windows and amplitude definition; the reference measurements are physical sanity checks and never model/rule threshold selection.

Source: `outputs/labeler/sawtooth/fix3/muscatello_reference.json`.

Blind queue: 200 windows on 44 shots, including 141 primary random windows. Stratum counts: `{'random_observable': 141, 'disagreement': 14, 'algorithm_uncertain': 15, 'model_predicted_negative': 15, 'zero_candidate_shot': 15}`. Random observable windows are frozen before prediction access. Candidate-free, model-negative, uncertain and disagreement supplements are separate strata. Only sensor inputs and blank targets enter the annotation pack; private selection and all predictions remain hidden. About 97 independent positive events give a worst-case 95% recall half-width of 0.1; clustered events do not supply that effective sample size automatically. Primary sampling weights and whole-shot CIs are preregistered. Annotation remains pending.

Source: `outputs/labeler/sawtooth/fix3/crash_time_queue.json`.

## Artifacts, reproduction and verification

Complete labels: `$LABELER_ROOT/round4/saw/fix3/labels/`. Manifest `SHA256SUMS` sha256: `3c70d44325cf98a0d4e9cb92efd3f6219e2c3fd13bf4f4bd7c57ee52d24168c3`. Population shards export current states without duplicate cohort rows; failed-shot records remain in the ledger. The cohort bundle is separate. Production stores were read only.

Source: `outputs/labeler/sawtooth/fix3/label_manifest.json`.

Large signals, EFIT metadata, models, predictions, annotation pack, PDFs and 150-dpi PNGs are under the same fix3 directory. Paper example, three old-rule figures and the 12-shot gallery were inspected; figure ledgers retain paths and hashes.

Source: `outputs/labeler/sawtooth/fix3/paper_example.json`.

Source: `outputs/labeler/sawtooth/fix3/gallery.json`.

Source: `outputs/labeler/sawtooth/fix3/figure_inspection.json`.

Covering tests: 142 passed. Ruff passes on all changed Python files; formatting passes on all new Python files; git diff --check passes. Long jobs used timeouts and logs; temporary storage was swept afterward. GPU training used CUDA_VISIBLE_DEVICES=1 within the assigned memory budget.

Source: `outputs/labeler/sawtooth/fix3/verification.json`.

```text
........................................................................ [ 50%]
......................................................................   [100%]
142 passed in 11.33s
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/lib/python3.11/site-packages/XRootD/client/finalize.py:46: FutureWarning: `torch.distributed.reduce_op` is deprecated, please use `torch.distributed.ReduceOp` instead
  if isinstance(obj, File) and obj.is_open():
All checks passed!
7 files already formatted
```

Reproduce with the prescribed pixi environment and TMPDIR. Entrypoints: sawtooth_geometry_fix3.py fetch/audit/reference; sawtooth_physics.py labels; sawtooth_benchmark.py train/predict/evaluate; sawtooth_fix_validation.py validate; sawtooth_fix3_artifacts.py queue-base/queue/figures; sawtooth_fix3_records.py records/manifest; sawtooth_phase_null_audit.py; sawtooth_fix3_report.py. Run queue-base before model prediction access. Full command logs remain under the large output directory.

## Concerns and next work

Known limitations, all left as they are because a rule change would force a full population rerun:

- Uncertain dominates: 66.1% of observable population time and 34382 of 170856 candidates are uncertain. Only about a third of observable time is assessed, so all conditional scores describe that third.
- The EFIT01 q=1 radius conflict is conservative. Shot 203563 (EFIT01 q_min 0.77, inversion minus q=1 radius -0.198 m) has 26 of 29 candidates uncertain although it shows regular trains.
- Population shots without archived field or axis metadata, and shots without a usable ECE waveform, receive no definite label (see the exclusion counts above).
- The central-ECE reference windows give periods 86–98 ms and amplitudes 0.35–0.49, against 85 ± 5 ms and 0.35 ± 0.02 published; the largest deviations are 12.7 ms and 0.14. They are sanity checks, not calibration.
- The legacy-offset diagnosis traces the old detector's offsets to its inversion-profile gate and holdoff, not to the reader. The current catalog detector could be compared only on shot 192090.
- Each gallery crash panel prefers the middle uncertain train when a shot has one, to expose unresolved q/ECE conflicts, so the gallery over-represents hard cases.

Blind physical accuracy remains unmeasured. Reviewed-span failures and any inverted HL-3 ranking remain failures, not reasons to retune on those shots. Nominal geometry is not a calibrated flux measurement; missing field/profile data limits population evidence. Algorithm-assessed benchmarks remain conditional and can be solved by derivative rules. Next: obtain the queued blind crash/span/ambiguity annotations, lock them, and evaluate all frozen methods over independently observable support without retuning.

## Appendix: superseded history

Earlier rounds used a hottest-channel proxy, sparse same-shot RF localization, a q_min>1.05 conflict and unbounded fractional edge phases. Their results are superseded. The first-round `outputs/labeler/sawtooth/fix` records and the two earlier plan documents were removed; `outputs/labeler/sawtooth/fix2` remains as the immediate predecessor record, and the earlier rounds' narrative is in the stream report appendix.
