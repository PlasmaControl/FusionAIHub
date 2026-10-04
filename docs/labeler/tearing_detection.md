# Whole-interval tearing-mode labels and magnetic-rule recovery

The target is a **strong rotating n=1/n=2 magnetic mode**, a tearing-mode proxy without independent island identification. Because the label is a threshold rule on the same magnetic RMS the detectors read, `tm-ours` recovers that rule from Mirnov spectrogram features, and a two-line RMS baseline with no training (`max(n1 RMS / 12 G, n2 RMS / 6 G)`) is the reference it has to beat. These results do not establish a better TM detector. AUROC and AUPRC are the primary comparisons; F1 depends on calibration.

All labeling, feature screening, fetching, agreement, fitting, scoring, coverage and galleries exclude the 50 cohort blind shots before opening their signals; the blind split carries no tearing-mode labels. The cohort table holds the 450 development shots. The population table covers every non-blind shot with fetched RMS (4798 shots) and **includes** the development shots (450 of them), so cohort and population counts overlap and must not be added. Earlier blind-split score files were moved unread into `results/quarantine_blind_test/`.

## Rule and uncertainty

Both raw RMS and its 5 ms median must exceed 12 G (n1) or 6 G (n2) continuously for 50 ms, before joining runs. The n1 level follows Farre-Kaga et al.; n2 is a local extension. Each qualified seed extends to max(1 G, 10% of its peak). Release gaps up to 50 ms may join; acquisition gaps cannot. An EFIT rational surface alone does not determine m and no ECE island radius is resolved, so m stays unassigned.

Seed and span frequency evidence uses `coherent_frequency` (a 50 ms window with p90−p10 width ≤ max(2 kHz, 25% of its median) and ≥80% coherent support over a span). Where `N1FREQ`/`N2FREQ` is missing the Mirnov fallback requires an n-resolved phase fit ≥ 0.9, prominence ≥ 10 dB, a line frequency between 1 kHz and the cap below, and coherent amplitude above the frozen development quiet-time p95. The frequency cap scales with the toroidal number: 30 kHz for n = 1, 60 kHz for n = 2 (the Mirnov features stop at 30 kHz, so the n = 2 cap acts through `N2FREQ` only). These are magnetic proxies, not proof of an island.

**Harmonic veto.** An n = 2 seed is dropped where n2/n1 is at or below the veto level, so the second harmonic of an n = 1 mode, which is a bounded fraction of it, is not read as a separate n = 2 mode. The level is the 99th percentile of n2/n1 over development bins where the n = 2 line sits at twice the n = 1 frequency **and** the Mirnov best-fit toroidal number at that frequency is 1 (1713 bins on 72 shots), which gives 0.713, set to **0.72**. The earlier level, 0.57, used every frequency-matched bin (30573 bins), a set that can also hold frequency-coupled 3/2 + 2/1 pairs, which are real n = 2 modes. The phase-coherent set is a small minority of those bins, because the harmonic of an n = 1 waveform usually fits n = 2, so the level rests on a small sample; a higher level vetoes more seeds. Of the 86 n = 2 seeds (50 ms above 6 G on the RMS alone) on 27 development shots, the harmonic veto removes 9 at the earlier level and 10 at 0.72; the coherent-frequency cap removes 10 at 30 kHz and 9 at 60 kHz; both together remove 16 before and 15 now. The n = 2 intervals that result: 17 with the earlier settings, 20 with the current ones.

**Weak tracks and uncertainty.** A weak track needs a 100 ms coherent core and is uncertain; it is released at the weak-line amplitude floor itself, and the same weak screen runs over ramp-up and flat-top. Time above the frozen weak RMS thresholds (n1 2.0282 G, n2 1.8280 G) that the screen cannot assess is uncertain rather than absent, and a sustained exceedance with no seed in otherwise-absent flat-top time is uncertain with reason `locked_unseeded`. Quiet time is absent. The label is a strong-mode label, not exhaustive TM truth: weak modes that fail the screen stay absent.

The uncertain share of observable development plasma went from **49.6%** pooled (median shot 58.1%) to **40.3%** pooled (median shot 37.2%) from flat-top start; counted over the whole catalog window it is 42.4% before and 34.9% now (median shot 46.6% before, 31.6% now). 196 of 450 shots are more than half uncertain (before: 247).

### Criterion pass rates

Each criterion is tested separately, by interval toroidal number, on **absent** time and on **present** time inside that number's own intervals. The absent rates are measured after labelling: they are the share of time the rule called absent that nonetheless passes the criterion, so they are not independent of the rule. Missing criterion inputs do not count as failures; the denominator is the time the input was measured. These are diagnostic associations with the magnetic rule, not independent validation.

| Set | Interval n | Criterion | Absent pass % (post-labelling) | Present pass % (own intervals) |
|---|---|---|---:|---:|
| cohort | n = 1 | rms_above_seed | 0.01 (1464.4 s) | 45.32 (102.7 s) |
| cohort | n = 1 | median5ms_above_seed | 0.00 (1464.4 s) | 45.59 (102.7 s) |
| cohort | n = 1 | rms_above_weak | 1.16 (1464.4 s) | 95.33 (102.7 s) |
| cohort | n = 1 | frequency_in_range_legacy | 79.79 (477.2 s) | 98.70 (102.7 s) |
| cohort | n = 1 | frequency_coherent50ms | 0.21 (477.2 s) | 82.42 (102.7 s) |
| cohort | n = 1 | mirnov_phase_fit | 14.98 (1461.5 s) | 94.68 (102.7 s) |
| cohort | n = 1 | mirnov_prominence | 73.88 (1461.5 s) | 99.32 (102.7 s) |
| cohort | n = 1 | span_coherent_support | 5.20 (1464.4 s) | 96.15 (102.7 s) |
| cohort | n = 1 | seed_coherent_support | 1.22 (1464.4 s) | 82.42 (102.7 s) |
| cohort | n = 1 | weak_mirnov_support | 6.59 (1461.5 s) | 99.02 (102.7 s) |
| cohort | n = 1 | screening_available | 99.89 (1464.4 s) | 100.00 (102.7 s) |
| cohort | n = 2 | rms_above_seed | 0.00 (1464.4 s) | 37.64 (41.8 s) |
| cohort | n = 2 | median5ms_above_seed | 0.00 (1464.4 s) | 38.10 (41.8 s) |
| cohort | n = 2 | rms_above_weak | 0.14 (1464.4 s) | 95.36 (41.8 s) |
| cohort | n = 2 | frequency_in_range_legacy | 17.44 (477.2 s) | 96.92 (41.8 s) |
| cohort | n = 2 | frequency_coherent50ms | 0.70 (477.2 s) | 89.33 (41.8 s) |
| cohort | n = 2 | mirnov_phase_fit | 1.18 (1461.5 s) | 93.73 (41.8 s) |
| cohort | n = 2 | mirnov_prominence | 73.88 (1461.5 s) | 99.88 (41.8 s) |
| cohort | n = 2 | span_coherent_support | 1.18 (1464.4 s) | 99.09 (41.8 s) |
| cohort | n = 2 | seed_coherent_support | 0.78 (1464.4 s) | 91.94 (41.8 s) |
| cohort | n = 2 | weak_mirnov_support | 2.47 (1461.5 s) | 97.64 (41.8 s) |
| cohort | n = 2 | screening_available | 99.89 (1464.4 s) | 100.00 (41.8 s) |
| population | n = 1 | rms_above_seed | 0.01 (16483.4 s) | 48.76 (972.1 s) |
| population | n = 1 | median5ms_above_seed | 0.00 (16483.4 s) | 48.91 (972.1 s) |
| population | n = 1 | rms_above_weak | 1.12 (16483.4 s) | 96.18 (972.1 s) |
| population | n = 1 | frequency_in_range_legacy | 78.74 (3662.0 s) | 98.75 (972.1 s) |
| population | n = 1 | frequency_coherent50ms | 0.27 (3662.0 s) | 81.71 (972.1 s) |
| population | n = 1 | mirnov_phase_fit | 14.01 (16441.5 s) | 94.33 (972.1 s) |
| population | n = 1 | mirnov_prominence | 74.41 (16441.5 s) | 98.97 (972.1 s) |
| population | n = 1 | span_coherent_support | 5.20 (16483.4 s) | 96.43 (972.1 s) |
| population | n = 1 | seed_coherent_support | 1.51 (16483.4 s) | 81.71 (972.1 s) |
| population | n = 1 | weak_mirnov_support | 6.28 (16441.5 s) | 98.54 (972.1 s) |
| population | n = 1 | screening_available | 99.83 (16483.4 s) | 100.00 (972.1 s) |
| population | n = 2 | rms_above_seed | 0.00 (16483.4 s) | 40.62 (379.7 s) |
| population | n = 2 | median5ms_above_seed | 0.00 (16483.4 s) | 40.90 (379.7 s) |
| population | n = 2 | rms_above_weak | 0.14 (16483.4 s) | 94.33 (379.7 s) |
| population | n = 2 | frequency_in_range_legacy | 18.14 (3662.0 s) | 97.87 (379.7 s) |
| population | n = 2 | frequency_coherent50ms | 0.69 (3662.0 s) | 90.93 (379.7 s) |
| population | n = 2 | mirnov_phase_fit | 0.85 (16441.5 s) | 95.51 (378.9 s) |
| population | n = 2 | mirnov_prominence | 74.41 (16441.5 s) | 99.85 (378.9 s) |
| population | n = 2 | span_coherent_support | 0.81 (16483.4 s) | 98.98 (379.7 s) |
| population | n = 2 | seed_coherent_support | 0.54 (16483.4 s) | 92.62 (379.7 s) |
| population | n = 2 | weak_mirnov_support | 2.27 (16441.5 s) | 98.80 (378.9 s) |
| population | n = 2 | screening_available | 99.83 (16483.4 s) | 100.00 (379.7 s) |

Source: [criterion_support_fix3.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/criterion_support_fix3.json).

## Abrupt collapse and locking

An amplitude fall from above seed to below release within ≤ 5 ms is never `ended=decay`. It is `ended=locked` only when independently confirmed, otherwise `ended=unknown`, and the following phase is uncertain until the lock signal falls or the discharge ends. The independent signal is the n = 1 PTDATA radial field `DUSBRADIAL` (native ptdata units, treated as gauss by disruption-py; the unit is not verified here, so no absolute field is claimed). A lock is confirmed by a **sustained rise**: |DUSBRADIAL| at least 5 above its median over the 200 ms before the interval's onset, held for 20 ms near the interval end or collapse; it is released when the field stays below that rise for 200 ms. The relative rule replaces an absolute 5-unit level, which fired on shots whose field sits high before any mode. Shots 176030–176912 carry a corrupted channel and are never confirmed. `N1FREQ`/`N2FREQ` falling to ≤ 1 kHz alone creates only a candidate. n = 2 has no independent confirmation.

A lock is looked for at **every** interval end, not only after a collapse, and for candidates the rule rejected: a rejected candidate followed by a confirmed lock tail stays uncertain with the lock reason. Quiet-time false confirmations were measured by placing 6220 pseudo-onsets (20 per shot, seed 0) in stretches labelled absent and testing for a lock 300 ms later: the old absolute rule confirmed **10.4%**, the relative rule **1.2%**; on draws whose pre-onset field was already ≥ 3 the rates are 30.6% and 2.2% (1478 draws). Intervals ending in a confirmed lock: 18 before, 15 now. Uncertain `locked_unseeded` time: 61 rows, 22.8 s on 54 shots.

`DUSBRADIAL` is on file for 327 of the 450 development shots. The shots that lacked one were requested on the login node through `fdp run` with one worker and a stop at the first authentication error (none occurred); the fetch ended after repeated data-server lookup failures (`getservbyname` for PTSERVER, not an authentication error) and 123 shots were not retrieved. Those shots keep an unknown lock status and get no relative baseline or `locked_unseeded` check; unknown is not evidence of no lock. The population has the same limitation on shots with no record.

| Set | Labeled shots | Intervals | Mode shots | Ends | Confirmed locks |
|---|---:|---:|---:|---:|---:|
| cohort | 450 | 81 | 74 | decay 49, locked 15, plasma_end 1, unknown 16 | 15 |
| population | 4798 | 832 | 712 | decay 451, locked 160, plasma_end 17, unknown 204 | 160 |

Source: benchmark `label_counts`, `locking_coverage` and the adjacent label metadata.

The `extend_tm_interval` table requires conversion before promotion to `review/`: TM catalog state 3 is forbidden, n-specific present/uncertain rows overlap, onset rows have zero length, and fractional-ms boundaries must become whole milliseconds. `test_shot_table_validates_extension_intervals_with_onset_points_and_spans` tests geometry, not catalog validity.

## Onsets and historical-onset agreement

The onset is the interval's start, a point event. The interval is the span. Because the interval start is where the strong rule first holds, the true onset lies earlier, in the preceding weak track; each onset carries `onset_window_ms`, the start of the preceding same-n weak track (the interval start where none exists), so the window 71 of 81 cohort onsets have is the span in which the mode could have begun.

A reference onset is counted as matched when it is **contained within an interval or within ±100 ms of its edges**. That tests whether the label holds the historical onset, not how accurately it times it; intervals last seconds. Recall on the development shots is **Seo 12/26** and **survival 16/67**. The strong 50 ms seed rule omits short and fast-locking modes: of the missed onsets, 9/14 (Seo) and 30/51 (survival) are short bursts, and 4 and 16 lack a supported coherent line. The survival archive does not follow a literal continuous-50 ms rule on the 1 kHz `N1RMS`: its onsets classed short bursts have no 50 ms raw-and-median 12 G crossing near them. The rule was not selected to maximise archive agreement.

| Reference | Covered shots | Onsets | Matched | Strictly contained | Onset error, ms (median [q25, q75]) | Within 100 ms | Inside onset window | Intervals without a reference onset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seo | 80 | 26 | 12 | 9 | 130 [49, 262] | 33% | 92% | 3 of 14 |
| survival | 175 | 67 | 16 | 8 | 0 [-11, 57] | 81% | 100% | 5 of 21 |

The onset error is the reference onset minus our interval start, over matched onsets only (positive: our interval began first). "Intervals without a reference onset" counts compared intervals that hold no reference onset of their shot.

Sources: benchmark `agreement`, [agreement_seo_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_seo_cohort_dev.json) and [agreement_survival_cohort_dev.json](../../data/events/neoclassical_tearing_mode/benchmark/sources/agreement_survival_cohort_dev.json). The frozen-rule sensitivity is generated by `tm_sensitivity.py` and bundled as `sensitivity_survival_dev.json`; it does not tune the rule.

## Development detection benchmark

The outer held-shot assignment is the seed-0 round-robin. Before any fit, `tm_cv_plan.py` freezes inner shot roles in `inner_splits_fix3.json`, stratified by whether a shot has an interval, with within-stratum swaps so every model and legacy target has positive support in its validation shots. No score or held-fold performance chooses roles. Each fit asserts a positive-bearing early-stopping validation set and threshold selection raises on zero positives. The published CNN is also shown at its own 0.5 threshold. The DSM's published survival threshold 0.7 corresponds to risk 0.3.

All learned models are refitted with three seeds on the same recipe as before; hyperparameters stay fixed from earlier development work, so selection is not fully nested. The scalar CNN has t+25 ms input lookahead and is an offline detector. The DSM is a detection head on its original embedding; published DSM rows remain horizon forecasts, with the shots in the published training list excluded from their scores. The published CNN's training overlap with the development shots is unknown (dagger), so its scores are descriptive.

Targets use absolute 10 ms bins; legacy rows use their native 25 ms bins. Each row has its own shot set, listed in the appendix table and its source JSON. In the primary group categories 2 and 3 and unavailable inputs are excluded. AUROC and AUPRC use 1,024 score quantiles and every interval is a 1,000-draw whole-shot bootstrap (seed 0). Segmental F1 uses IoU 0.5, 50 ms minimum segments and ≤ 50 ms negative-gap closing; unavailable bins are hard barriers.

**Two co-primary target definitions.** *Uncertain excluded* removes category 2 and 3 time, so it does not score where an interval starts or ends: the uncertain time borders the intervals. *Uncertain as negative* scores category 2 as negative and still excludes category 3, so it does score the boundaries. Fits, scores and thresholds are the same in both. Finite uncertain features stay in the temporal input context; the target mask controls loss and scoring only. The primary metrics do not score boundary placement in the first group.

| Setting | Model | Shots / bins | AUROC [95% CI] | AUPRC [95% CI] | F1 [95% CI] | Segmental F1 [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Legacy | tm-onsetcnn-published (thr. 0.5)† | 79 / 5580 | 0.897 [0.832,0.952] | 0.601 [0.385,0.799] | 0.611 [0.437,0.750] | — |
| Legacy | tm-dsm-500ms-published (risk 0.3) | 80 / 8362 | 0.721 [0.614,0.828] | 0.294 [0.179,0.442] | 0.000 [0.000,0.000] | — |
| Tokamak-SI | tm-onsetcnn-published† | 278 / 30845 | 0.974 [0.959,0.987] | 0.903 [0.838,0.948] | 0.779 [0.690,0.849] | 0.617 [0.506,0.714] |
| Tokamak-SI | tm-onsetcnn-published (thr. 0.5)† | 278 / 30845 | 0.974 [0.959,0.987] | 0.903 [0.838,0.948] | 0.708 [0.582,0.809] | 0.578 [0.397,0.738] |
| Tokamak-SI | tm-dsm-500ms-published | 247 / 43436 | 0.779 [0.689,0.852] | 0.407 [0.286,0.530] | 0.442 [0.328,0.535] | 0.281 [0.160,0.406] |
| Tokamak-SI | tm-onsetcnn-retrained | 278 / 31094 | 0.985 [0.973,0.994] | 0.956 [0.924,0.977] | 0.873 [0.797,0.920] | 0.738 [0.595,0.835] |
| Tokamak-SI | tm-dsm-retrained | 336 / 57800 | 0.962 [0.937,0.980] | 0.867 [0.804,0.913] | 0.803 [0.734,0.848] | 0.647 [0.543,0.725] |
| Tokamak-SI | tm-rms-2line | 450 / 160507 | 0.998 [0.998,0.999] | 0.986 [0.980,0.990] | 0.944 [0.931,0.956] | 0.667 [0.581,0.752] |
| Tokamak-SI | tm-rms-2line (seed levels) | 450 / 160507 | 0.998 [0.998,0.999] | 0.986 [0.980,0.990] | 0.645 [0.572,0.717] | 0.465 [0.353,0.590] |
| Tokamak-SI | tm-rms | 450 / 160507 | 0.985 [0.977,0.992] | 0.904 [0.845,0.944] | 0.787 [0.711,0.843] | 0.306 [0.241,0.379] |
| Tokamak-SI | tm-ours | 450 / 160507 | 0.999 [0.997,1.000] | 0.995 [0.990,0.998] | 0.961 [0.930,0.981] | 0.852 [0.792,0.906] |
| Tokamak-SI | tm-ours-rms | 450 / 160507 | 1.000 [0.999,1.000] | 0.997 [0.994,0.999] | 0.973 [0.955,0.985] | 0.840 [0.747,0.924] |
| Tokamak-SI, uncertain = negative | tm-onsetcnn-published† | 286 / 59105 | 0.900 [0.864,0.932] | 0.412 [0.281,0.571] | 0.405 [0.306,0.497] | 0.288 [0.191,0.374] |
| Tokamak-SI, uncertain = negative | tm-onsetcnn-retrained | 286 / 59293 | 0.907 [0.879,0.933] | 0.429 [0.297,0.578] | 0.462 [0.354,0.557] | 0.330 [0.216,0.436] |
| Tokamak-SI, uncertain = negative | tm-dsm-retrained | 341 / 112199 | 0.834 [0.787,0.874] | 0.297 [0.204,0.411] | 0.346 [0.271,0.415] | 0.258 [0.180,0.338] |
| Tokamak-SI, uncertain = negative | tm-rms-2line | 450 / 246801 | 0.963 [0.954,0.971] | 0.630 [0.556,0.700] | 0.460 [0.381,0.531] | 0.192 [0.151,0.235] |
| Tokamak-SI, uncertain = negative | tm-ours | 450 / 246801 | 0.949 [0.936,0.961] | 0.494 [0.374,0.617] | 0.365 [0.290,0.432] | 0.232 [0.184,0.283] |

The Mirnov-derived uncertainty mask excludes about **40.3%** of observable development plasma. It shares `tm-ours` inputs, so the primary group emphasises strong modes against quiet magnetic time and can favour the magnetic detector; the uncertain-as-negative group also scores those hard cases. The +RMS row adds circular label inputs. The legacy rows use their published thresholds as primary and tuned thresholds as secondary (appendix); AUPRC is not compared across settings because prevalence differs, and `figure2_tm.json` records it beside each value. Rows without a variant use the threshold tuned on inner validation. The DSM raises no alarm at its published level (risk 0.3) on the legacy bins, so its F1 there is 0 (recall 0, precision undefined).

**Reading the table.** The two-line RMS baseline is the rule restated as a score, with no training. A model that reaches it has recovered the magnetic rule; one that exceeds it uses information the rule does not. On the shared shots `tm-ours` exceeds the baseline by +0.001 AUROC [+0.000, +0.003] and +0.005 AUPRC [+0.002, +0.010] and, with uncertain time scored as negative, the order reverses: AUPRC 0.494 for `tm-ours` against 0.630 for the baseline. `tm-ours` is therefore reported as recovering the magnetic rule, not as a better detector.

### Paired comparison

Same 278 development shots and 31094 identical available 10 ms bins for all three models; each keeps its inner-validation threshold. Paired bootstrap draws resample the same shots. Ranking, not a threshold-specific F1 gain, is the primary comparison.

| Model | AUROC | AUPRC | F1 | Segmental F1 |
|---|---:|---:|---:|---:|
| tm-ours | 0.999 [0.999,1.000] | 0.998 [0.996,1.000] | 0.984 [0.972,0.992] | 0.967 [0.931,0.990] |
| tm-onsetcnn-retrained | 0.985 [0.973,0.994] | 0.956 [0.924,0.976] | 0.873 [0.797,0.920] | 0.738 [0.595,0.835] |
| tm-rms-2line | 0.998 [0.997,0.999] | 0.993 [0.986,0.997] | 0.959 [0.934,0.976] | 0.891 [0.809,0.947] |
| Difference, tm-ours − retrained CNN | 0.014 [0.006,0.026] | 0.042 [0.022,0.074] | 0.112 [0.066,0.183] | 0.229 [0.138,0.361] |
| Difference, tm-ours − two-line RMS | 0.001 [0.000,0.003] | 0.005 [0.002,0.010] | 0.025 [0.006,0.048] | 0.076 [0.021,0.154] |

### Published model against its retrained twin

For each architecture the published model and the retrained one are scored on one target, one mask and one set of shots and bins (for the DSM, outside the published training list). The published CNN is shown at its own threshold and at the tuned one; the retrained twin at its tuned threshold.

| Target | Model | Shots / bins | Prevalence | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|---:|---:|
| uncertain excluded | tm-onsetcnn-published at thr. | 273 / 27597 | 0.192 | 0.974 [0.958,0.987] | 0.908 [0.846,0.955] | 0.713 [0.597,0.814] |
| uncertain excluded | tm-onsetcnn-published tuned | 273 / 27597 | 0.192 | 0.974 [0.958,0.987] | 0.908 [0.846,0.955] | 0.786 [0.687,0.855] |
| uncertain excluded | tm-onsetcnn-retrained | 273 / 27597 | 0.192 | 0.985 [0.974,0.994] | 0.959 [0.929,0.979] | 0.876 [0.801,0.926] |
| uncertain = negative | tm-onsetcnn-published at thr. | 281 / 54355 | 0.097 | 0.899 [0.860,0.936] | 0.415 [0.277,0.584] | 0.521 [0.389,0.637] |
| uncertain = negative | tm-onsetcnn-published tuned | 281 / 54355 | 0.097 | 0.899 [0.860,0.936] | 0.415 [0.277,0.584] | 0.410 [0.310,0.502] |
| uncertain = negative | tm-onsetcnn-retrained | 281 / 54355 | 0.097 | 0.905 [0.879,0.932] | 0.431 [0.298,0.599] | 0.465 [0.359,0.563] |
| uncertain excluded | tm-dsm-500ms-published at thr. | 247 / 43436 | 0.157 | 0.779 [0.689,0.852] | 0.407 [0.285,0.530] | 0.000 [0.000,0.000] |
| uncertain excluded | tm-dsm-500ms-published tuned | 247 / 43436 | 0.157 | 0.779 [0.689,0.852] | 0.407 [0.285,0.530] | 0.442 [0.328,0.535] |
| uncertain excluded | tm-dsm-retrained | 247 / 43436 | 0.157 | 0.959 [0.923,0.984] | 0.870 [0.778,0.931] | 0.800 [0.707,0.865] |
| uncertain = negative | tm-dsm-500ms-published at thr. | 252 / 81920 | 0.083 | 0.712 [0.622,0.795] | 0.157 [0.099,0.228] | 0.000 [0.000,0.000] |
| uncertain = negative | tm-dsm-500ms-published tuned | 252 / 81920 | 0.083 | 0.712 [0.622,0.795] | 0.157 [0.099,0.228] | 0.251 [0.166,0.329] |
| uncertain = negative | tm-dsm-retrained | 252 / 81920 | 0.083 | 0.848 [0.787,0.897] | 0.356 [0.247,0.480] | 0.364 [0.267,0.457] |

**Training-set size.** The published models were trained on thousands of shots (the DSM's list has 8923); each retrained model sees about 324 development shots per outer fold. A retrained model that ranks below its published twin is confounded by that difference, and no population-scale retrain is part of this benchmark. No retraining-generalisation claim follows from published-model comparisons while the original training overlap is unknown.

Sources: benchmark `paired_common_shots` and `like_for_like`, [figure2_tm.json](figure2_tm.json).

## Coverage and publication artifacts

**No TM coverage gain is claimed.** Coverage uses the measured plasma-start to catalog-end domain and the 10 ms grid on each matched shot set. Observable time includes uncertainty; labeled time excludes it. Scoring can include quiet ramp-up, so scoring bins and observable-plasma coverage have different denominators.

| Set | Shots | Observable plasma s | Labeled s | Uncertain s |
|---|---:|---:|---:|---:|
| ours | 450 | 2109.09 | 1258.05 | 851.04 |
| ours_population | 4798 | 22620.97 | 14168.72 | 8452.25 |

| Matched reference | Shots | Legacy labeled s | Interval labeled s | Common labeled s |
|---|---:|---:|---:|---:|
| legacy_seo | 80 | 129.98 | 214.11 | 59.99 |
| legacy_survival | 175 | 799.29 | 451.07 | 405.20 |

Sources: benchmark `coverage`, the adjacent label metadata and [figure2_tm.json](figure2_tm.json). Cohort and population overlap.

`tm_gallery.py --width 3.25 --columns 1` provides column-sized example panels with 7.5 pt text at final width; the example shots are one decaying n = 1 mode and one locking n = 1 mode. In the galleries a present interval is drawn plain, uncertain time is flat grey, and only the locked phase is hatched (a rise of the radial field, with or without a preceding mode). A grey spectrogram background is time with no record of that diagnostic. The 7.3-inch MHR and Mirnov galleries are supplementary audit material and must not be shrunk into a paper column. Their JSON sidecars record shots, width, font, diagnostic, label/source hashes and image hashes. `tm_render_tables.py` renders the final TeX at 6.75-inch text width, rejects overfull horizontal boxes and writes PDF and 150-dpi PNG previews with provenance. The appendix explains the dagger and lists the shots behind each row.

## Reproduction

Use the frozen/no-install pixi `labelmaker` environment, the worktree `PYTHONPATH`, a scratch `TMPDIR` and `LABELER_NO_FETCH=1`. Only the `DUSBRADIAL` fetch unsets it and runs through `fdp run` on the login node. Train on `CUDA_VISIBLE_DEVICES=1` with the phase3 CUDA venv.

Sequence: `tm_magfeatures.py`, `tm_harmonic_calibration.py`, `tm_label.py` (`--from cohort`, then `--from population`), `tm_audit_rule.py`, `tm_cv_plan.py`, `tm_fix3_diagnostics.py`, `tm_prior_retrain.py --model cnn/dsm`, `tm_ours.py --features magnetics/magnetics+rms` and `--baseline`, `tm_prior_published.py` (legacy and Tokamak-SI), `tm_agreement.py --exclude-test --tag _dev`, `tm_sensitivity.py`, `tm_gallery.py`, `tm_benchmark.py --rescore --gallery-reviewed`, `tm_write_doc.py`, `tm_render_tables.py`. Source JSONs and the shot lists are committed under the benchmark's `sources/`; predictions, weights, signals and figures stay under `$LABELER_ROOT/round4/tm/`.

Remaining limitations: the target and the detector inputs share the magnetic RMS, the n = 2 level and the lock units are local conventions, weak and fast-locking modes are omitted, the published CNN's training overlap is unknown, the training sets of the retrained models are far smaller than the published ones, lock status is unknown on development shots without a `DUSBRADIAL` record, and there is no independent ECE island radius or fully nested hyperparameter selection.
