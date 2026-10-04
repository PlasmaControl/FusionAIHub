# Whole-interval tearing-mode labels and magnetic-rule recovery

The target is a **strong rotating n=1/n=2 magnetic mode**, a tearing-mode proxy without independent island identification. Because the label is a threshold rule on the same magnetic RMS the detectors read, `tm-ours` recovers that rule from Mirnov spectrogram features, and a two-line RMS baseline with no training (`max(n1 RMS / 12 G, n2 RMS / 6 G)`) is the reference it has to beat. These results do not establish a better TM detector. AUROC and AUPRC are the primary comparisons; F1 depends on calibration.

All labeling, feature screening, fetching, agreement, fitting, scoring, coverage and galleries exclude the 50 cohort blind shots before opening their signals; the blind split carries no tearing-mode labels. The cohort table holds the 450 development shots. The population table covers every non-blind shot with fetched RMS (4798 shots) and **includes** the development shots (450 of them), so cohort and population counts overlap and must not be added. Earlier blind-split score files were moved unread into `results/quarantine_blind_test/`.

## Rule and uncertainty

Both raw RMS and its 5 ms median must exceed 12 G (n1) or 6 G (n2) continuously for 50 ms, before joining runs. The n1 level follows Farre-Kaga et al.; n2 is a local extension. Each qualified seed extends to max(1 G, 10% of its peak). Release gaps up to 50 ms may join; acquisition gaps cannot. An EFIT rational surface alone does not determine m and no ECE island radius is resolved, so m stays unassigned.

Seed and span frequency evidence uses `coherent_frequency` (a 50 ms window with p90−p10 width ≤ max(2 kHz, 25% of its median) and ≥80% coherent support over a span). A seed needs a coherent line at or above **1.5 kHz** and at or below the cap, whether the line comes from `N1FREQ`/`N2FREQ` or from the Mirnov fallback. Where `N1FREQ`/`N2FREQ` is missing the fallback requires an n-resolved phase fit ≥ 0.9, prominence ≥ 10 dB and coherent amplitude above the frozen development quiet-time p95; its line search covers 1–30 kHz, so inside a span (not for a seed) a fallback line between 1 and 1.5 kHz still counts as support. The frequency cap scales with the toroidal number: 30 kHz for n = 1, 60 kHz for n = 2 (the Mirnov features stop at 30 kHz, so the n = 2 cap acts through `N2FREQ` only). These are magnetic proxies, not proof of an island.

**Harmonic veto.** An n = 2 seed is dropped where n2/n1 is at or below the veto level, so that the second harmonic of a rotating n = 1 mode, whose amplitude is a bounded fraction of the n = 1 amplitude, is not read as a separate n = 2 mode. The level is the 99th percentile of n2/n1 over development bins where the n = 2 line sits at twice the n = 1 frequency **and** the six midplane Mirnov probes' phases at that frequency fit toroidal n = 2 (best-fit |n| = 2, fit ≥ 0.9): 28174 bins on 87 shots, a 99th percentile of 0.561, rounded up to **0.57**. These are the bins where the line at 2 f1 is the harmonic, because the harmonics of a rotating, non-sinusoidal n = 1 waveform carry toroidal number 2. **The limit:** toroidal phase cannot separate such a harmonic from a co-rotating, frequency-coupled n = 2 mode (a 3/2 mode locked to the 2/1), because both have n = 2 at 2 f1, so the veto is a heuristic and may also remove real 3/2 modes. It removes 9 of the 86 n = 2 seeds (50 ms above 6 G on the RMS alone) on 27 development shots; the 60 kHz frequency cap alone removes 9, and the two cuts together 15. The finished rule yields 20 n = 2 intervals on the development shots (19 without the veto): the veto removes seeds, not intervals, and a removed seed can change how its neighbours merge, so its net effect on finished n = 2 intervals is +1.

**Weak tracks and uncertainty.** A weak track needs a 100 ms coherent core and is uncertain; it is released at the weak-line amplitude floor itself, and the same weak screen runs over ramp-up and flat-top. Time above the frozen weak RMS thresholds (n1 2.0282 G, n2 1.8280 G) that the screen cannot assess is uncertain rather than absent. Quiet time is absent, except that in otherwise-absent flat-top time a radial field that steps up (the test below) and stays at least 5 above the median of that time for 100 ms is uncertain with reason `locked_unseeded`: a field event with no mode seen, not a mode. The label is a strong-mode label, not exhaustive TM truth: weak modes that fail the screen stay absent.

Uncertain time is **34.0%** of the observable catalog-window time of the development shots (absent + present + uncertain, pooled over 450 shots; 155 shots are more than half uncertain). This one statistic is the uncertain share quoted in the benchmark caption.

### Criterion pass rates

Each criterion is tested separately, by interval toroidal number, on **absent** time and on **present** time inside that number's own intervals. The absent rates are measured after labelling: they are the share of time the rule called absent that nonetheless passes the criterion, so they are not independent of the rule. Missing criterion inputs do not count as failures; the denominator is the time the input was measured. These are diagnostic associations with the magnetic rule, not independent validation.

| Set | Interval n | Criterion | Absent pass % (post-labelling) | Present pass % (own intervals) |
|---|---|---|---:|---:|
| cohort | n = 1 | rms_above_seed | 0.01 (1487.7 s) | 45.32 (102.7 s) |
| cohort | n = 1 | median5ms_above_seed | 0.00 (1487.7 s) | 45.59 (102.7 s) |
| cohort | n = 1 | rms_above_weak | 1.15 (1487.7 s) | 95.33 (102.7 s) |
| cohort | n = 1 | frequency_in_range_legacy | 79.88 (490.2 s) | 98.71 (102.7 s) |
| cohort | n = 1 | frequency_coherent50ms | 0.23 (490.2 s) | 82.41 (102.7 s) |
| cohort | n = 1 | mirnov_phase_fit | 14.82 (1484.7 s) | 94.68 (102.7 s) |
| cohort | n = 1 | mirnov_prominence | 73.94 (1484.7 s) | 99.32 (102.7 s) |
| cohort | n = 1 | span_coherent_support | 5.13 (1487.7 s) | 96.15 (102.7 s) |
| cohort | n = 1 | seed_coherent_support | 1.21 (1487.7 s) | 82.41 (102.7 s) |
| cohort | n = 1 | weak_mirnov_support | 6.50 (1484.7 s) | 99.02 (102.7 s) |
| cohort | n = 1 | screening_available | 99.89 (1487.7 s) | 100.00 (102.7 s) |
| cohort | n = 2 | rms_above_seed | 0.00 (1487.7 s) | 37.86 (42.0 s) |
| cohort | n = 2 | median5ms_above_seed | 0.00 (1487.7 s) | 38.32 (42.0 s) |
| cohort | n = 2 | rms_above_weak | 0.14 (1487.7 s) | 95.37 (42.0 s) |
| cohort | n = 2 | frequency_in_range_legacy | 17.62 (490.2 s) | 96.93 (42.0 s) |
| cohort | n = 2 | frequency_coherent50ms | 0.69 (490.2 s) | 89.27 (42.0 s) |
| cohort | n = 2 | mirnov_phase_fit | 1.17 (1484.7 s) | 93.64 (42.0 s) |
| cohort | n = 2 | mirnov_prominence | 73.94 (1484.7 s) | 99.88 (42.0 s) |
| cohort | n = 2 | span_coherent_support | 1.17 (1487.7 s) | 99.03 (42.0 s) |
| cohort | n = 2 | seed_coherent_support | 0.77 (1487.7 s) | 91.86 (42.0 s) |
| cohort | n = 2 | weak_mirnov_support | 2.45 (1484.7 s) | 97.65 (42.0 s) |
| cohort | n = 2 | screening_available | 99.89 (1487.7 s) | 100.00 (42.0 s) |
| population | n = 1 | rms_above_seed | 0.01 (16538.8 s) | 48.77 (972.4 s) |
| population | n = 1 | median5ms_above_seed | 0.00 (16538.8 s) | 48.91 (972.4 s) |
| population | n = 1 | rms_above_weak | 1.12 (16538.8 s) | 96.19 (972.4 s) |
| population | n = 1 | frequency_in_range_legacy | 78.75 (3707.1 s) | 98.74 (972.4 s) |
| population | n = 1 | frequency_coherent50ms | 0.28 (3707.1 s) | 81.68 (972.4 s) |
| population | n = 1 | mirnov_phase_fit | 14.00 (16496.5 s) | 94.33 (972.4 s) |
| population | n = 1 | mirnov_prominence | 74.42 (16496.5 s) | 98.97 (972.4 s) |
| population | n = 1 | span_coherent_support | 5.19 (16538.8 s) | 96.43 (972.4 s) |
| population | n = 1 | seed_coherent_support | 1.51 (16538.8 s) | 81.68 (972.4 s) |
| population | n = 1 | weak_mirnov_support | 6.27 (16496.5 s) | 98.54 (972.4 s) |
| population | n = 1 | screening_available | 99.83 (16538.8 s) | 100.00 (972.4 s) |
| population | n = 2 | rms_above_seed | 0.00 (16538.8 s) | 40.23 (387.6 s) |
| population | n = 2 | median5ms_above_seed | 0.00 (16538.8 s) | 40.51 (387.6 s) |
| population | n = 2 | rms_above_weak | 0.14 (16538.8 s) | 93.96 (387.6 s) |
| population | n = 2 | frequency_in_range_legacy | 18.25 (3707.1 s) | 97.90 (387.6 s) |
| population | n = 2 | frequency_coherent50ms | 0.68 (3707.1 s) | 90.75 (387.6 s) |
| population | n = 2 | mirnov_phase_fit | 0.85 (16496.5 s) | 95.14 (386.7 s) |
| population | n = 2 | mirnov_prominence | 74.42 (16496.5 s) | 99.85 (386.7 s) |
| population | n = 2 | span_coherent_support | 0.81 (16538.8 s) | 98.73 (387.6 s) |
| population | n = 2 | seed_coherent_support | 0.54 (16538.8 s) | 92.35 (387.6 s) |
| population | n = 2 | weak_mirnov_support | 2.27 (16496.5 s) | 98.89 (386.7 s) |
| population | n = 2 | screening_available | 99.83 (16538.8 s) | 100.00 (387.6 s) |

Source: [criterion_support_fix4.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/criterion_support_fix4.json).

## Abrupt collapse and locking

An amplitude fall from above seed to below release within ≤ 5 ms is never `ended=decay`. It is `ended=locked` only when independently confirmed, otherwise `ended=unknown`, and the following phase is uncertain until the lock signal falls or the discharge ends. The independent signal is the n = 1 PTDATA radial field `DUSBRADIAL` (native ptdata units, treated as gauss by disruption-py; the unit is not verified here, so no absolute field is claimed). A lock is confirmed by a **step**. At a candidate time t_c (a frequency drop at least 50 ms after the seed starts, an abrupt collapse, or an interval's end) the median of |DUSBRADIAL| over t_c + 20 ms to t_c + 120 ms must exceed its median over t_c − 200 ms to t_c − 20 ms by at least 5 (each window needs 50 ms of measured field, and the earlier window does not reach before the flat-top start). The baseline is local to the candidate, so a field that is already high or ramps slowly shows no step and confirms nothing. The lock is released when the field stays below its baseline plus 5 for 200 ms. Shots 176030–176912 carry a corrupted channel and are never confirmed. `N1FREQ`/`N2FREQ` falling to ≤ 1 kHz alone creates only a candidate. n = 2 has no independent confirmation.

A lock is looked for at **every** interval end, not only after a collapse, and for candidates the rule rejected: a rejected candidate followed by a confirmed lock tail stays uncertain with the lock reason. Across the 12 confirmed cohort locks (n = 1) the step is 5.0 to 12.0 (median 6.7) native units. The false-confirmation rate of the test was measured by drawing a pseudo-onset in time the labels call absent, in a stretch long enough for the 200 ms baseline and the 120 ms step window, and testing for a lock `lag` later. The step test reads only the field around that pseudo-lock; the lags are 300, 1000 and 2000 ms and lags drawn from the cohort's 81 interval durations (median 1608 ms, 90th percentile 3689 ms). The last row keeps only the durations that fit the quiet stretch, so its realized lags are shorter: median 430 ms, 10th percentile 151 ms, 90th percentile 1726 ms.

| Lag | Draws | Shots | False confirmations | Rate |
|---|---:|---:|---:|---:|
| 300 ms | 5893 | 320 | 10 | 0.2% |
| 1000 ms | 4081 | 258 | 10 | 0.2% |
| 2000 ms | 1717 | 105 | 2 | 0.1% |
| drawn from the interval durations (realized median 430 ms) | 6291 | 327 | 27 | 0.4% |

Intervals ending in a confirmed lock: 12 on the development shots. Uncertain `locked_unseeded` time: 11 rows, 2.9 s on 10 shots. The column example's locking shot is **191672**, chosen by a fixed rule: the confirmed cohort n = 1 lock with the largest step (12.0 units: 1.8 before, 13.8 after); its decaying counterpart is 189514.

`DUSBRADIAL` is on file for 327 of the 450 development shots. The shots that lacked one were requested on the login node through `fdp run` with one worker and a stop at the first authentication error (none occurred); the fetch ended after repeated data-server lookup failures (`getservbyname` for PTSERVER, not an authentication error) and 123 shots were not retrieved. Those shots keep an unknown lock status and get no relative baseline or `locked_unseeded` check; unknown is not evidence of no lock. The population has the same limitation on shots with no record.

| Set | Labeled shots | Intervals | Mode shots | Ends | Confirmed locks |
|---|---:|---:|---:|---:|---:|
| cohort | 450 | 81 | 74 | decay 52, locked 12, plasma_end 1, unknown 16 | 12 |
| population | 4798 | 835 | 714 | decay 462, locked 142, plasma_end 17, unknown 214 | 142 |

Source: benchmark `label_counts`, `locking_coverage` and the adjacent label metadata.

The `extend_tm_interval` table requires conversion before promotion to `review/`: TM catalog state 3 is forbidden, n-specific present/uncertain rows overlap, onset rows have zero length, and fractional-ms boundaries must become whole milliseconds. `test_shot_table_validates_extension_intervals_with_onset_points_and_spans` tests geometry, not catalog validity.

## Onsets and historical-onset agreement

The onset is the interval's start, a point event, and the interval is the span. The interval start is the qualified seed grown backwards to max(1 G, 10% of its peak), so it precedes the 50 ms seed crossing; it is not the time the strong rule first holds. Each onset carries `onset_window_ms`, from the start of the preceding same-n weak track to the interval start, the span in which the mode could have begun. 70 of 81 cohort onsets have such a window (the others have no preceding weak track), and 10 of those windows are 5 ms or shorter and say almost nothing about where the mode began: 10 onset rows are flagged with `onset_window_degenerate: true` (an interval with no onset point has no row to flag), and no window is widened.

A reference onset is counted as matched when it is **contained within an interval or within ±100 ms of its edges**. That tests whether the label holds the historical onset, not how accurately it times it; intervals last seconds. Recall on the development shots is **Seo 12/26** and **survival 16/67**. The strong 50 ms seed rule omits short and fast-locking modes: of the missed onsets, 9/14 (Seo) and 30/51 (survival) are short bursts, and 4 and 16 lack a supported coherent line. The Seo onsets fall a median 130 ms after the interval start, inside the interval. A reference onset lies **inside the onset window** when it is between the window's start and the interval's start (no widening: the match already allows 100 ms past the start): 2 of 12 Seo and 8 of 16 survival matched onsets. 8 Seo and 3 survival onsets lie more than 100 ms after the interval start. The label therefore holds the historical onset, it does not time it. The survival archive does not follow a literal continuous-50 ms rule on the 1 kHz `N1RMS`: its onsets classed short bursts have no 50 ms raw-and-median 12 G crossing near them. The rule was not selected to maximise archive agreement.

| Reference | Covered shots | Onsets | Matched | Strictly contained | Onset error, ms (median [q25, q75]) | Within 100 ms | Inside onset window | Intervals without a reference onset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seo | 80 | 26 | 12 | 9 | 130 [49, 262] | 33% | 2 of 12 | 3 of 14 |
| survival | 175 | 67 | 16 | 8 | 0 [-11, 57] | 81% | 8 of 16 | 5 of 21 |

The onset error is the reference onset minus our interval start, over matched onsets only (positive: our interval began first). "Inside onset window" counts matched reference onsets between the window start and the interval start (not widened). "Intervals without a reference onset" counts compared intervals that hold no reference onset of their shot.

Sources: benchmark `agreement`, [agreement_seo_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_seo_cohort_dev.json) and [agreement_survival_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_survival_cohort_dev.json). The frozen-rule sensitivity is generated by `tm_sensitivity.py` and bundled as `sensitivity_survival_dev.json`; it does not tune the rule.

## Development detection benchmark

The outer held-shot assignment is the seed-0 round-robin. Before any fit, `tm_cv_plan.py` freezes inner shot roles in `inner_splits_fix4.json`, stratified by whether a shot has an interval, with within-stratum swaps so every model and legacy target has positive support in its validation shots. No score or held-fold performance chooses roles. Each fit asserts a positive-bearing early-stopping validation set and threshold selection raises on zero positives. The published CNN is shown at its own 0.5 threshold and at the tuned one, in both target groups. The DSM's published survival threshold 0.7 corresponds to risk 0.3.

All learned models are refitted with three seeds on the same recipe as before; hyperparameters stay fixed from earlier development work, so selection is not fully nested. The scalar CNN has t+25 ms input lookahead and is an offline detector. The DSM is a detection head on its original embedding; published DSM rows remain horizon forecasts, with the shots in the published training list excluded from their scores. The published CNN's training overlap with the development shots is unknown (dagger), so its scores are descriptive.

Targets use absolute 10 ms bins; legacy rows use their native 25 ms bins. Each row has its own shot set, listed in the appendix table and its source JSON. In the primary group categories 2 and 3 and unavailable inputs are excluded. AUROC and AUPRC use 1,024 score quantiles and every interval is a 1,000-draw whole-shot bootstrap (seed 0). Segmental F1 uses IoU 0.5, 50 ms minimum segments and ≤ 50 ms negative-gap closing; unavailable bins are hard barriers.

**Two co-primary target definitions.** *Uncertain excluded* removes category 2 and 3 time, so it does not score where an interval starts or ends: the uncertain time borders the intervals. *Uncertain as negative* scores category 2 as negative and still excludes category 3, so it does score the boundaries. Fits, scores and thresholds are the same in both. Finite uncertain features stay in the temporal input context; the target mask controls loss and scoring only. The primary metrics do not score boundary placement in the first group.

| Setting | Model | Shots / bins | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Legacy | tm-onsetcnn-published (thr. 0.5)† | 79 / 5580 | 0.897 [0.832,0.952] | 0.601 [0.385,0.799] | 0.611 [0.437,0.750] | — |
| Legacy | tm-dsm-500ms-published (risk 0.3) | 80 / 8362 | 0.721 [0.614,0.828] | 0.294 [0.179,0.442] | 0.000 [0.000,0.000] | — |
| Tokamak-SI | tm-onsetcnn-published† | 278 / 30977 | 0.974 [0.959,0.987] | 0.903 [0.838,0.948] | 0.779 [0.689,0.848] | 0.615 [0.504,0.713] |
| Tokamak-SI | tm-onsetcnn-published (thr. 0.5)† | 278 / 30977 | 0.974 [0.959,0.987] | 0.903 [0.838,0.948] | 0.708 [0.582,0.808] | 0.588 [0.416,0.746] |
| Tokamak-SI | tm-dsm-500ms-published | 247 / 43680 | 0.779 [0.691,0.852] | 0.405 [0.285,0.528] | 0.442 [0.327,0.534] | 0.279 [0.157,0.403] |
| Tokamak-SI | tm-onsetcnn-retrained | 278 / 31226 | 0.982 [0.968,0.993] | 0.953 [0.916,0.976] | 0.844 [0.758,0.906] | 0.683 [0.538,0.785] |
| Tokamak-SI | tm-dsm-retrained | 336 / 58134 | 0.963 [0.940,0.981] | 0.871 [0.810,0.916] | 0.799 [0.728,0.844] | 0.657 [0.545,0.745] |
| Tokamak-SI | tm-rms-2line | 450 / 162840 | 0.998 [0.998,0.999] | 0.986 [0.980,0.990] | 0.940 [0.926,0.953] | 0.629 [0.538,0.717] |
| Tokamak-SI | tm-rms-2line (seed levels) | 450 / 162840 | 0.998 [0.998,0.999] | 0.986 [0.980,0.990] | 0.645 [0.573,0.717] | 0.475 [0.364,0.600] |
| Tokamak-SI | tm-rms | 450 / 162840 | 0.985 [0.977,0.992] | 0.903 [0.845,0.943] | 0.788 [0.711,0.843] | 0.302 [0.240,0.375] |
| Tokamak-SI | tm-ours | 450 / 162840 | 0.999 [0.997,1.000] | 0.994 [0.989,0.997] | 0.960 [0.930,0.980] | 0.843 [0.778,0.901] |
| Tokamak-SI | tm-ours-rms | 450 / 162840 | 1.000 [0.999,1.000] | 0.997 [0.994,0.999] | 0.967 [0.935,0.986] | 0.863 [0.805,0.917] |
| Tokamak-SI, uncertain = negative | tm-onsetcnn-published† | 286 / 59105 | 0.900 [0.864,0.932] | 0.412 [0.281,0.571] | 0.406 [0.306,0.497] | 0.288 [0.191,0.374] |
| Tokamak-SI, uncertain = negative | tm-onsetcnn-published (thr. 0.5)† | 286 / 59105 | 0.900 [0.864,0.932] | 0.412 [0.281,0.571] | 0.516 [0.396,0.625] | 0.394 [0.240,0.535] |
| Tokamak-SI, uncertain = negative | tm-dsm-500ms-published | 252 / 81920 | 0.712 [0.622,0.795] | 0.157 [0.098,0.228] | 0.252 [0.166,0.330] | 0.167 [0.071,0.271] |
| Tokamak-SI, uncertain = negative | tm-onsetcnn-retrained | 286 / 59293 | 0.903 [0.873,0.932] | 0.444 [0.292,0.599] | 0.435 [0.331,0.528] | 0.313 [0.202,0.417] |
| Tokamak-SI, uncertain = negative | tm-dsm-retrained | 341 / 112199 | 0.839 [0.791,0.880] | 0.297 [0.204,0.411] | 0.356 [0.278,0.427] | 0.262 [0.175,0.347] |
| Tokamak-SI, uncertain = negative | tm-rms-2line | 450 / 246801 | 0.963 [0.954,0.971] | 0.632 [0.559,0.700] | 0.447 [0.372,0.519] | 0.191 [0.152,0.235] |
| Tokamak-SI, uncertain = negative | tm-ours | 450 / 246801 | 0.948 [0.935,0.960] | 0.512 [0.394,0.623] | 0.363 [0.290,0.431] | 0.229 [0.182,0.281] |

The Mirnov-derived uncertainty mask is **34.0%** of observable catalog-window time (the statistic defined above). It shares `tm-ours` inputs, so the primary group emphasises strong modes against quiet magnetic time and can favour the magnetic detector; the uncertain-as-negative group also scores those hard cases. The +RMS row adds circular label inputs. The legacy rows use their published thresholds as primary and tuned thresholds as secondary (appendix); AUPRC is not compared across settings because prevalence differs, and `figure2_tm.json` records it beside each value. Rows without a variant use the threshold tuned on inner validation. The DSM raises no alarm at its published level (risk 0.3) on the legacy bins, so its F1 there is 0 (recall 0, precision undefined).

**Reading the table.** The two-line RMS baseline is the rule restated as a score, with no training. A model that reaches it has recovered the magnetic rule; one that exceeds it uses information the rule does not. On all 450 development shots, which both models score on identical bins, the paired difference `tm-ours` minus the two-line baseline is, with uncertain time excluded (162840 bins), AUROC 0.000 [-0.001, +0.001] (not resolved); AUPRC +0.009 [+0.003, +0.015] (`tm-ours` ahead); F1 +0.020 [-0.011, +0.046] (not resolved); segmental F1 +0.215 [+0.111, +0.309] (`tm-ours` ahead); with uncertain time scored as negative (246801 bins, 450 shots), AUROC -0.015 [-0.029, -0.003] (the two-line baseline ahead); AUPRC -0.120 [-0.222, -0.019] (the two-line baseline ahead); F1 -0.084 [-0.104, -0.065] (the two-line baseline ahead); segmental F1 +0.038 [+0.010, +0.068] (`tm-ours` ahead). The order reverses between the groups on AUPRC (`tm-ours` ahead with uncertain time excluded, the two-line baseline ahead with it scored as negative); `tm-ours` is ahead in both groups on segmental F1; the order is not resolved in at least one group on AUROC and F1. `tm-ours` is therefore reported as recovering the magnetic rule, not as a better detector.

### Paired comparison

Each comparison uses the development shots and available 10 ms bins that both of its models score, within each target group; each model keeps its inner-validation threshold. Paired bootstrap draws resample the same shots, so each difference row carries a paired 95% interval; a difference whose interval spans 0 is not resolved. `tm-ours` and the two-line baseline need only the Mirnov features, so they are compared on all 450 development shots (identical bins) and every statement about the baseline uses that set. The retrained CNN has inputs on fewer shots, so only the comparison with the CNN uses the restricted set; comparing `tm-ours` with the baseline on it would rank them on the shots the CNN happens to cover. Ranking, not a threshold-specific F1 gain, is the primary comparison.

On the 278 shots (31226 bins; 286 shots and 59293 bins with uncertain time scored as negative) where the retrained CNN has inputs, `tm-ours` minus the retrained CNN is AUROC +0.017 [+0.007, +0.031] (`tm-ours` ahead); AUPRC +0.044 [+0.022, +0.078] (`tm-ours` ahead); F1 +0.142 [+0.079, +0.225] (`tm-ours` ahead); segmental F1 +0.274 [+0.176, +0.405] (`tm-ours` ahead); with uncertain time scored as negative, AUROC +0.030 [+0.002, +0.059] (`tm-ours` ahead); AUPRC +0.173 [+0.023, +0.312] (`tm-ours` ahead); F1 -0.055 [-0.100, -0.008] (the retrained CNN ahead); segmental F1 +0.039 [-0.014, +0.108] (not resolved).

| Target | Model | Shots / bins | AUROC | AUPRC | F1 | Segmental F1 |
|---|---|---:|---:|---:|---:|---:|
| uncertain excluded, tm-ours, two-line RMS (all shots) | tm-ours | 450 / 162840 | 0.999 [0.997,1.000] | 0.994 [0.989,0.997] | 0.960 [0.930,0.980] | 0.843 [0.778,0.901] |
| uncertain excluded, tm-ours, two-line RMS (all shots) | tm-rms-2line | 450 / 162840 | 0.998 [0.998,0.999] | 0.986 [0.980,0.990] | 0.940 [0.926,0.953] | 0.629 [0.538,0.717] |
| uncertain excluded, tm-ours, two-line RMS (all shots) | Difference, tm-ours − tm-rms-2line | 450 / 162840 | 0.000 [-0.001,0.001] | 0.009 [0.003,0.015] | 0.020 [-0.011,0.046] | 0.215 [0.111,0.309] |
| uncertain = negative, tm-ours, two-line RMS (all shots) | tm-ours | 450 / 246801 | 0.948 [0.935,0.960] | 0.512 [0.394,0.623] | 0.363 [0.290,0.431] | 0.229 [0.182,0.281] |
| uncertain = negative, tm-ours, two-line RMS (all shots) | tm-rms-2line | 450 / 246801 | 0.963 [0.954,0.971] | 0.632 [0.559,0.700] | 0.447 [0.372,0.519] | 0.191 [0.152,0.235] |
| uncertain = negative, tm-ours, two-line RMS (all shots) | Difference, tm-ours − tm-rms-2line | 450 / 246801 | -0.015 [-0.029,-0.003] | -0.120 [-0.222,-0.019] | -0.084 [-0.104,-0.065] | 0.038 [0.010,0.068] |
| uncertain excluded, tm-ours, retrained CNN (CNN shots) | tm-ours | 278 / 31226 | 0.999 [0.998,1.000] | 0.997 [0.993,1.000] | 0.986 [0.975,0.993] | 0.957 [0.914,0.986] |
| uncertain excluded, tm-ours, retrained CNN (CNN shots) | tm-onsetcnn-retrained | 278 / 31226 | 0.982 [0.968,0.993] | 0.953 [0.916,0.976] | 0.844 [0.758,0.906] | 0.683 [0.538,0.785] |
| uncertain excluded, tm-ours, retrained CNN (CNN shots) | Difference, tm-ours − tm-onsetcnn-retrained | 278 / 31226 | 0.017 [0.007,0.031] | 0.044 [0.022,0.078] | 0.142 [0.079,0.225] | 0.274 [0.176,0.405] |
| uncertain = negative, tm-ours, retrained CNN (CNN shots) | tm-ours | 286 / 59293 | 0.933 [0.906,0.958] | 0.616 [0.444,0.757] | 0.380 [0.279,0.476] | 0.353 [0.242,0.453] |
| uncertain = negative, tm-ours, retrained CNN (CNN shots) | tm-onsetcnn-retrained | 286 / 59293 | 0.903 [0.873,0.932] | 0.444 [0.292,0.599] | 0.435 [0.331,0.528] | 0.313 [0.202,0.417] |
| uncertain = negative, tm-ours, retrained CNN (CNN shots) | Difference, tm-ours − tm-onsetcnn-retrained | 286 / 59293 | 0.030 [0.002,0.059] | 0.173 [0.023,0.312] | -0.055 [-0.100,-0.008] | 0.039 [-0.014,0.108] |

### Published model against its retrained twin

For each architecture the published model and the retrained one are scored on one target, one mask and one set of shots and bins (for the DSM, outside the published training list). The published CNN is shown at its own threshold and at the tuned one; the retrained twin at its tuned threshold.

| Target | Model | Shots / bins | Prevalence | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|---:|---:|
| uncertain excluded | tm-onsetcnn-published (published threshold) | 273 / 27715 | 0.191 | 0.974 [0.958,0.987] | 0.908 [0.846,0.955] | 0.713 [0.597,0.814] |
| uncertain excluded | tm-onsetcnn-published (tuned threshold) | 273 / 27715 | 0.191 | 0.974 [0.958,0.987] | 0.908 [0.846,0.955] | 0.785 [0.686,0.855] |
| uncertain excluded | tm-onsetcnn-retrained | 273 / 27715 | 0.191 | 0.982 [0.968,0.993] | 0.956 [0.921,0.978] | 0.847 [0.756,0.910] |
| uncertain = negative | tm-onsetcnn-published (published threshold) | 281 / 54355 | 0.097 | 0.899 [0.861,0.936] | 0.415 [0.277,0.585] | 0.521 [0.390,0.637] |
| uncertain = negative | tm-onsetcnn-published (tuned threshold) | 281 / 54355 | 0.097 | 0.899 [0.861,0.936] | 0.415 [0.277,0.585] | 0.410 [0.310,0.502] |
| uncertain = negative | tm-onsetcnn-retrained | 281 / 54355 | 0.097 | 0.901 [0.872,0.932] | 0.445 [0.294,0.605] | 0.438 [0.334,0.533] |
| uncertain excluded | tm-dsm-500ms-published (published threshold) | 247 / 43680 | 0.157 | 0.779 [0.691,0.852] | 0.405 [0.285,0.528] | 0.000 [0.000,0.000] |
| uncertain excluded | tm-dsm-500ms-published (tuned threshold) | 247 / 43680 | 0.157 | 0.779 [0.691,0.852] | 0.405 [0.285,0.528] | 0.442 [0.327,0.534] |
| uncertain excluded | tm-dsm-retrained | 247 / 43680 | 0.157 | 0.961 [0.926,0.985] | 0.873 [0.783,0.933] | 0.795 [0.702,0.861] |
| uncertain = negative | tm-dsm-500ms-published (published threshold) | 252 / 81920 | 0.083 | 0.712 [0.622,0.795] | 0.157 [0.098,0.228] | 0.000 [0.000,0.000] |
| uncertain = negative | tm-dsm-500ms-published (tuned threshold) | 252 / 81920 | 0.083 | 0.712 [0.622,0.795] | 0.157 [0.098,0.228] | 0.252 [0.166,0.330] |
| uncertain = negative | tm-dsm-retrained | 252 / 81920 | 0.083 | 0.852 [0.792,0.899] | 0.349 [0.239,0.476] | 0.374 [0.273,0.469] |

**Training-set size.** The published models were trained on thousands of shots (the DSM's list has 8923); each retrained model sees about 324 development shots per outer fold. A retrained model that ranks below its published twin is confounded by that difference, and no population-scale retrain is part of this benchmark. No retraining-generalisation claim follows from published-model comparisons while the original training overlap is unknown.

Sources: benchmark `paired_common_shots` and `like_for_like`, [figure2_tm.json](figure2_tm.json).

## Coverage and publication artifacts

**No TM coverage gain is claimed.** Coverage uses the measured plasma-start to catalog-end domain and the 10 ms grid on each matched shot set. Observable time includes uncertainty; labeled time excludes it. Scoring can include quiet ramp-up, so scoring bins and observable-plasma coverage have different denominators.

| Set | Shots | Observable plasma s | Labeled s | Uncertain s |
|---|---:|---:|---:|---:|
| ours | 450 | 2109.09 | 1281.38 | 827.71 |
| ours_population | 4798 | 22620.97 | 14231.20 | 8389.77 |

| Matched reference | Shots | Legacy labeled s | Interval labeled s | Common labeled s |
|---|---:|---:|---:|---:|
| legacy_seo | 80 | 129.98 | 222.49 | 60.08 |
| legacy_survival | 175 | 799.29 | 464.42 | 411.42 |

Sources: benchmark `coverage`, the adjacent label metadata and [figure2_tm.json](figure2_tm.json). Cohort and population overlap.

`tm_gallery.py --width 3.25 --columns 1` provides column-sized example panels with 7.5 pt text at final width; the example shots are one decaying n = 1 mode (189514) and one locking n = 1 mode (191672, picked by the fixed rule above: the largest radial-field step). In the galleries a present interval is drawn plain and uncertain time is one flat grey without outlines, so uncertain rows that overlap are drawn once and a boxed or darker patch never means more uncertainty. In the spectrogram and RMS panels each interval is shaded in its own colour, and where an n = 1 and an n = 2 interval overlap the two shadings blend into a light grey (legend entry "n = 1 and 2 overlap"); that grey is not uncertain time, which is drawn only in the strip above the spectrogram. Only locked phases are hatched: slashes where a step of the radial field confirms a mode's lock, crosses where the field steps in flat-top time with no mode seen (`locked_unseeded`), dots where a lock is suspected but unconfirmed. A grey spectrogram background is time with no record of that diagnostic. The 7.3-inch MHR and Mirnov galleries are supplementary audit material and must not be shrunk into a paper column. Their JSON sidecars record shots, width, font, diagnostic, label/source hashes and image hashes. `tm_render_tables.py` renders the final TeX at 6.75-inch text width, rejects overfull horizontal boxes and writes PDF and 150-dpi PNG previews with provenance. The appendix explains the dagger and lists the shots behind each row.

## Reproduction

Use the frozen/no-install pixi `labelmaker` environment, the worktree `PYTHONPATH`, a scratch `TMPDIR` and `LABELER_NO_FETCH=1`. Only the `DUSBRADIAL` fetch unsets it and runs through `fdp run` on the login node. Train on `CUDA_VISIBLE_DEVICES=1` with the phase3 CUDA venv.

Sequence: `tm_magfeatures.py`, `tm_harmonic_calibration.py`, `tm_label.py` (`--from cohort`, then `--from population`), `tm_audit_rule.py`, `tm_cv_plan.py`, `tm_rule_diagnostics.py`, `tm_prior_retrain.py --model cnn/dsm`, `tm_ours.py --features magnetics/magnetics+rms` and `--baseline`, `tm_prior_published.py` (legacy and Tokamak-SI), `tm_agreement.py --exclude-test --tag _dev`, `tm_sensitivity.py`, `tm_gallery.py` (the column examples with `--lock-example` on the rule-diagnostics record), `tm_benchmark.py --rescore --gallery-reviewed`, `tm_write_doc.py`, `tm_render_tables.py`, `tm_verify_artifacts.py`. Source JSONs and the shot lists are committed under the benchmark's `sources/`; predictions, weights, signals and figures stay under `$LABELER_ROOT/round4/tm/`.

Remaining limitations: the target and the detector inputs share the magnetic RMS, the n = 2 level and the lock units are local conventions, the harmonic veto is a heuristic that may remove real 3/2 modes, the onset is the interval start and does not time the mode independently, weak and fast-locking modes are omitted, the published CNN's training overlap is unknown, the training sets of the retrained models are far smaller than the published ones, lock status is unknown on development shots without a `DUSBRADIAL` record, and there is no independent ECE island radius or fully nested hyperparameter selection.

Earlier label rules and the numbers each change moved are in the [changelog](tearing_detection_changelog.md).
